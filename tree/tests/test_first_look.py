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
