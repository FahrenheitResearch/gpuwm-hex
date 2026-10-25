"""first_look.FirstLook: the f000 maps are drawn once the forecast lists its first frame, never before, and a
forecast that ends without a frame ends the wait."""
import json
import threading
import time
from pathlib import Path
from unittest import mock

from hexcore import first_look


def test_draws_the_first_listed_frame_into_the_full_pass_case(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    calls = []
    with mock.patch.object(first_look.subprocess, "run", side_effect=lambda argv, **kw: calls.append(argv) or mock.Mock(returncode=0)):
        look = first_look.FirstLook(forecast, grid="g.nc", init="i.nc", start_time="2026-10-23_12:00:00",
                                    out=tmp_path / "products", case="hex-x1.40962-native-2026102312",
                                    extra=["--region", "event=region:48,10,2500,2500"], poll_seconds=0.01)
        look.start()
        time.sleep(0.05)
        assert calls == []                           # nothing listed yet: nothing drawn
        frame = forecast / "cuda-history.2026-10-23_12.00.00.nc"
        (forecast / first_look.READY).write_text(json.dumps({"step": 0, "path": str(frame)}) + "\n")
        result = look.finish(timeout=5)
    assert result["status"] == "drawn"
    argv = calls[0]
    assert argv[argv.index("--history") + 1] == str(frame)
    assert argv[argv.index("--case") + 1] == "hex-x1.40962-native-2026102312"
    assert argv[argv.index("--simulation-start") + 1] == "2026-10-23_12:00:00"
    assert "--region" in argv


def test_a_forecast_without_frames_ends_the_wait(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    with mock.patch.object(first_look.subprocess, "run") as run:
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path, case="c", poll_seconds=0.01)
        look.start()
        result = look.finish(timeout=5)
    assert result["status"] == "no-frame"
    assert not run.called


def test_a_frame_with_its_durable_sidecar_counts_without_a_ready_list(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    frame = forecast / "cuda-history.2026-10-23_12.00.00.nc"
    frame.write_bytes(b"x" * 10)
    (forecast / (frame.name + ".durable.json")).write_text(json.dumps({"bytes": 9}))   # not yet whole
    calls = []
    with mock.patch.object(first_look.subprocess, "run", side_effect=lambda argv, **kw: calls.append(argv) or mock.Mock(returncode=0)):
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path, case="c", poll_seconds=0.01)
        look.start()
        time.sleep(0.05)
        assert calls == []
        (forecast / (frame.name + ".durable.json")).write_text(json.dumps({"bytes": 10}))
        result = look.finish(timeout=5)
    assert result["status"] == "drawn"
    assert calls[0][calls[0].index("--history") + 1] == str(frame)


def test_every_later_frame_is_drawn_while_the_forecast_runs(tmp_path):
    """Production canary 2026-10-24 (hex global 60 km 24 h): f000 came 62.5 s in, f001-f024 only after the forecast."""
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    frames = [forecast / f"cuda-history.2026-10-23_{h:02d}.00.00.nc" for h in range(12, 16)]
    for frame in frames:
        frame.write_bytes(b"frame")
    ready = forecast / first_look.READY
    ready.write_text(json.dumps({"step": 0, "path": str(frames[0])}) + "\n")
    runs, popens = [], []

    def popen(argv, **kw):
        popens.append(argv)
        return mock.Mock(wait=lambda: 0, poll=lambda: 0)

    with mock.patch.object(first_look.subprocess, "run", side_effect=lambda argv, **kw: runs.append(argv) or mock.Mock(returncode=0)), \
            mock.patch.object(first_look.subprocess, "Popen", side_effect=popen):
        look = first_look.FirstLook(forecast, grid="g.nc", init="i.nc", start_time="2026-10-23_12:00:00",
                                    out=tmp_path / "products", case="c", poll_seconds=0.01)
        look.start()
        deadline = time.monotonic() + 5
        while not runs and time.monotonic() < deadline:
            time.sleep(0.01)
        with ready.open("a") as stream:
            for step, frame in enumerate(frames[1:3], 1):
                stream.write(json.dumps({"step": step, "path": str(frame)}) + "\n")
        while len(popens) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        with ready.open("a") as stream:
            stream.write(json.dumps({"step": 3, "path": str(frames[3])}) + "\n")
        while len(popens) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        result = look.finish(timeout=5)
    assert len(runs) == 1 and runs[0][runs[0].index("--history") + 1] == str(frames[0])   # f000 with its soundings
    drawn = [argv[argv.index("--history") + 1:] for argv in popens]
    assert sum(drawn, []) == [str(f) for f in frames[1:]]                                # each later frame once, in order
    assert all("--no-points" in argv for argv in popens)
    scratches = [argv[argv.index("--scratch") + 1] for argv in popens]
    assert len(set(scratches)) == len(scratches)                                         # no two draws share a scratch
    assert result["live"]["frames"] == 4 and result["live"]["drawn"] == 4
    assert result["live"]["receipt"] == str(tmp_path / first_look.LIVE_RECEIPT)
    receipt = json.loads((tmp_path / first_look.LIVE_RECEIPT).read_text())
    # f000 (the first look) and every later frame, each with the identity the full pass checks before skipping it
    assert receipt["frames"] == receipt["drawn"] == [f.name for f in frames]
    assert receipt["stat"][frames[2].name] == [frames[2].stat().st_size, frames[2].stat().st_mtime_ns]
    assert receipt["out"] == str(tmp_path / "products") and receipt["case"] == "c"


def test_live_frames_off_draws_the_first_frame_only(tmp_path, monkeypatch):
    monkeypatch.setattr(first_look, "LIVE_ON", False)
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    lines = [json.dumps({"step": h, "path": str(forecast / f"cuda-history.2026-10-23_{12 + h:02d}.00.00.nc")}) for h in range(3)]
    (forecast / first_look.READY).write_text("\n".join(lines) + "\n")
    (tmp_path / first_look.LIVE_RECEIPT).write_text("{}")  # an earlier run's: the full pass must not skip by it
    with mock.patch.object(first_look.subprocess, "run", return_value=mock.Mock(returncode=0)) as run, \
            mock.patch.object(first_look.subprocess, "Popen") as popen:
        look = first_look.FirstLook(forecast, grid="g", init="i", start_time="s", out=tmp_path / "p", case="c", poll_seconds=0.01)
        look.start()
        result = look.finish(timeout=5)
    assert result["status"] == "drawn" and run.call_count == 1 and not popen.called
    assert not (tmp_path / first_look.LIVE_RECEIPT).exists()
