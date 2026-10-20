"""The cycle driver: --cycle latest, the posting wait, the store, retention, the feed.

CPU only, no network: product addresses are answered by a stand-in for HTTP HEAD
and time by a stand-in clock."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pytest

from hexcore import global_cycle, init_sources
from hexcore.global_cycle import CycleDriverError, Store, apply_retention, resolve_latest

UTC = dt.timezone.utc


class Clock:
    def __init__(self, start: dt.datetime) -> None:
        self.now = start
        self.slept = 0.0

    def __call__(self) -> dt.datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += dt.timedelta(seconds=seconds)


def answers(posted: dict[str, dt.datetime], clock: Clock):
    """HEAD stand-in: a GFS cycle's file answers once the clock passes its posting time."""

    def ask(url: str) -> bool:
        for stamp, when in posted.items():
            day, hour = stamp[:8], stamp[8:]
            if f"gfs.{day}/{hour}/" in url:
                return clock() >= when
        return False

    return ask


def gfs():
    return global_cycle.source_row("gfs")


def test_every_runnable_forecast_source_declares_its_posting():
    table = init_sources.load_table(None)
    rows = {row["id"]: row for row in table["sources"]}
    for name in ("gfs", "gefs", "icon", "aifs", "ifs", "ifs-ens"):
        assert rows[name]["posting"]["cycle_hours"] == [0, 6, 12, 18]
        assert rows[name]["posting"]["lag_minutes"] > 0
    with pytest.raises(CycleDriverError, match="declares no posting block"):
        global_cycle.posting(rows["era5"])


def test_a_bad_posting_block_is_refused_on_load():
    row = json.loads(json.dumps(gfs()))
    row["posting"] = {"cycle_hours": [0, 24], "lag_minutes": 200}
    with pytest.raises(init_sources.InitSourceRefusal, match="cycle_hours"):
        init_sources.validate_row(row)


def test_latest_is_the_newest_due_cycle_when_it_is_posted():
    clock = Clock(dt.datetime(2026, 10, 16, 8, 25, tzinfo=UTC))
    ask = answers({"2026101600": dt.datetime(2026, 10, 16, 3, 30, tzinfo=UTC),
                   "2026101606": dt.datetime(2026, 10, 16, 9, 30, tzinfo=UTC)}, clock)
    got = resolve_latest(gfs(), clock=clock, sleep=clock.sleep, ask=ask)
    assert got["cycle"] == "2026-10-16T00" and got["waited_seconds"] == 0
    assert [c["cycle"] for c in got["checked"]] == ["2026-10-16T06", "2026-10-16T00"]


def test_a_newer_cycle_that_posted_early_is_taken():
    clock = Clock(dt.datetime(2026, 10, 16, 9, 20, tzinfo=UTC))
    ask = answers({"2026101606": dt.datetime(2026, 10, 16, 9, 10, tzinfo=UTC)}, clock)
    got = resolve_latest(gfs(), clock=clock, sleep=clock.sleep, ask=ask)
    assert got["cycle"] == "2026-10-16T06" and got["why"] == "posted before its usual time"


def test_a_due_cycle_not_yet_posted_is_waited_for_not_replaced_by_an_older_one():
    clock = Clock(dt.datetime(2026, 10, 16, 9, 40, tzinfo=UTC))  # 06Z due 09:31, posts 10:05
    ask = answers({"2026101600": dt.datetime(2026, 10, 16, 3, 30, tzinfo=UTC),
                   "2026101606": dt.datetime(2026, 10, 16, 10, 5, tzinfo=UTC)}, clock)
    said = []
    got = resolve_latest(gfs(), clock=clock, sleep=clock.sleep, ask=ask, poll_seconds=60, say=said.append)
    assert got["cycle"] == "2026-10-16T06"
    assert 25 * 60 <= got["waited_seconds"] <= 26 * 60 and said and "not posted yet" in said[0]


def test_a_cycle_that_never_posts_is_refused_by_name_after_the_wait():
    clock = Clock(dt.datetime(2026, 10, 16, 9, 40, tzinfo=UTC))
    ask = answers({"2026101600": dt.datetime(2026, 10, 16, 3, 30, tzinfo=UTC)}, clock)
    with pytest.raises(CycleDriverError, match="an older cycle is not run in its place"):
        resolve_latest(gfs(), clock=clock, sleep=clock.sleep, ask=ask, max_wait_minutes=30)
    assert clock.now == dt.datetime(2026, 10, 16, 10, 10, tzinfo=UTC)


def test_the_schedule_steps_through_the_sources_cycle_hours():
    block = gfs()["posting"]
    assert global_cycle.next_cycle(block, dt.datetime(2026, 10, 16, 0, tzinfo=UTC)) == dt.datetime(
        2026, 10, 16, 6, tzinfo=UTC)
    assert global_cycle.next_cycle(block, dt.datetime(2026, 10, 16, 18, tzinfo=UTC)) == dt.datetime(
        2026, 10, 17, 0, tzinfo=UTC)


def _cycle(store: Store, cycle: str, state: str) -> Path:
    path = store.cycle_dir(cycle)
    path.mkdir(parents=True)
    (path / "cycle.json").write_text(json.dumps({"schema": global_cycle.MANIFEST_SCHEMA, "cycle": cycle,
                                                 "state": state}))
    return path


def test_retention_keeps_the_newest_finished_cycles_and_only_driver_folders(tmp_path):
    store = Store(tmp_path, "global-120km")
    old = _cycle(store, "2026-10-15T00", "done")
    failed = _cycle(store, "2026-10-15T06", "failed")
    kept = [_cycle(store, c, "done") for c in ("2026-10-15T12", "2026-10-15T18")]
    foreign = store.cycles / "notes"
    foreign.mkdir()
    removed = apply_retention(store, keep=2)
    assert sorted(removed) == sorted([str(old), str(failed)])
    assert all(path.is_dir() for path in kept) and foreign.is_dir()
    with pytest.raises(CycleDriverError, match="at least 1"):
        apply_retention(store, keep=0)


def test_latest_names_the_newest_finished_cycle(tmp_path):
    store = Store(tmp_path, "global-120km")
    newer = _cycle(store, "2026-10-16T00", "done")
    _cycle(store, "2026-10-15T18", "done")
    newest = store.done()[0]
    global_cycle.publish_latest(store, Path(newest["run_dir"]), newest)
    assert (store.root / "latest").resolve() == newer.resolve()
    assert json.loads((store.root / "latest.json").read_text())["cycle"] == "2026-10-16T00"


def test_frames_without_a_durability_receipt_are_not_published(tmp_path):
    forecast = tmp_path / "forecast"
    forecast.mkdir()
    (forecast / "cuda-history.2026-10-16_03.00.00.nc").write_bytes(b"x")
    from hexcore.history_writer import mark_closed_frame_durable
    mark_closed_frame_durable(forecast / "cuda-history.2026-10-16_03.00.00.nc")
    (forecast / "cuda-history.2026-10-16_06.00.00.nc").write_bytes(b"partial")
    rows = global_cycle.frames(tmp_path, dt.datetime(2026, 10, 16, 0, tzinfo=UTC))
    assert [r["lead_hours"] for r in rows] == [3.0]


def test_one_driver_per_store_row(tmp_path):
    store = Store(tmp_path, "global-120km")
    with store.lock():
        with pytest.raises(CycleDriverError, match="another cycle driver"):
            with Store(tmp_path, "global-120km").lock():
                pass


def test_the_global_door_resolves_latest_into_its_plan(tmp_path, monkeypatch, capsys):
    from hexcore.cli import main

    clock = Clock(dt.datetime(2026, 10, 16, 8, 25, tzinfo=UTC))
    ask = answers({"2026101600": dt.datetime(2026, 10, 16, 3, 30, tzinfo=UTC)}, clock)
    monkeypatch.setattr(global_cycle, "head", ask)
    monkeypatch.setattr(global_cycle, "utcnow", clock)
    assert main(["run-plan", "--row", "global-120km", "--cycle", "latest", "--work-dir", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cycle"] == "2026-10-16T00" and plan["cycle_resolution"]["source"] == "gfs"
    assert "2026-10-16T00" in " ".join(plan["commands"]["init"])


def test_native_frames_require_finalized_run_and_support_members(tmp_path):
    folders = [tmp_path / "forecast", tmp_path / "ensemble/g00/mem001"]
    for folder in folders:
        folder.mkdir(parents=True)
        (folder / "history.2026-10-16_03.00.00.nc").write_bytes(b"inventory")
        (folder / "diag.2026-10-16_03.00.00.nc").write_bytes(b"diagnostics")
        (folder / "native-physics-receipt.json").write_text('{"completed": false}')
    start = dt.datetime(2026, 10, 16, 0, tzinfo=UTC)
    assert global_cycle.frames(tmp_path, start) == []
    for folder in folders:
        (folder / "native-physics-receipt.json").write_text('{"completed": true}')
    rows = global_cycle.frames(tmp_path, start)
    assert len(rows) == 2 and all(r["lead_hours"] == 3 for r in rows)
    assert rows[1]["member"] == "mem001"
