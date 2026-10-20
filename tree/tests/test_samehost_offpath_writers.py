"""Off-path history and boundary writers: order, ready manifest, failure propagation."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

import numpy as np
import pytest

from hexcore.drivers import samehost_forecast_v841 as samehost

FAKE = '''
import json
from pathlib import Path
N_CELLS = 7
def write_snapshot_netcdf(path, snapshot, static):
    Path(path).write_text(json.dumps({"n": N_CELLS, "keys": sorted(snapshot["arrays"])}))
    return {"path": str(path), "n_cells": N_CELLS}
'''


@pytest.fixture()
def fake_writer(tmp_path, monkeypatch):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "fake_history_writer.py").write_text(FAKE)
    monkeypatch.syspath_prepend(str(package))
    monkeypatch.setenv("PYTHONPATH", f"{package}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    monkeypatch.setenv("GPUWM_HEX_HISTORY_SPOOL", str(tmp_path))
    monkeypatch.setenv("GPUWM_HEX_HISTORY_DEFLATE", "off")
    module = importlib.import_module("fake_history_writer")
    module.N_CELLS = 11  # a run-time rebinding the child must see
    yield module
    sys.modules.pop("fake_history_writer", None)


def _sha(array):
    return str(float(np.asarray(array).sum()))


def test_pool_writes_every_frame_and_orders_receipts_by_step(tmp_path, fake_writer):
    records, hashes, gates = {}, {}, {}
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, records, hashes,
                                          workers=3, ready_path=tmp_path / "ready.jsonl", gates=gates)

    def build(step):
        def make():
            time.sleep(0.05 * (5 - step))  # later steps finish first
            gates[str(step)] = {"ok": step}
            return {"arrays": {"a": np.full(3, step, np.float32)}}, {}
        return make

    for step in range(5):
        writer.submit(step, tmp_path / f"h{step}.nc", build(step))
    writer.close()
    assert list(records) == [str(step) for step in range(5)]
    assert list(hashes) == list(records) and list(gates) == list(records)
    assert all(record["n_cells"] == 11 for record in records.values())
    assert all(record["deflate"]["engaged"] is False for record in records.values())
    assert all(record["deflate"]["why_not"] == "GPUWM_HEX_HISTORY_DEFLATE=off"
               for record in records.values())
    ready = [json.loads(line) for line in (tmp_path / "ready.jsonl").read_text().splitlines()]
    assert sorted(row["step"] for row in ready) == list(range(5))
    assert writer.frames == 5 and writer.workers == 3


def test_a_failed_build_is_raised_by_submit_and_close(tmp_path, fake_writer):
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, {}, {}, workers=2)

    def bad():
        raise FloatingPointError("snapshot rho/theta must be strictly positive")

    writer.submit(0, tmp_path / "h0.nc", bad)
    deadline = time.time() + 10
    while writer._error is None and time.time() < deadline:
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="failed earlier"):
        writer.submit(1, tmp_path / "h1.nc", lambda: ({"arrays": {}}, {}))
    with pytest.raises(RuntimeError, match="strictly positive"):
        writer.close()


def test_boundary_writer_keeps_step_order_and_reraises():
    seen = []
    boundary = samehost._AsyncBoundaryWriter(lambda step, collected: seen.append((step, collected)))
    for step in (0, 30, 60):
        boundary.submit(step, {"step": step})
    boundary.close()
    assert [step for step, _ in seen] == [0, 30, 60]

    def refuse(step, collected):
        raise RuntimeError(f"partition invariance failed at step {step}")

    failing = samehost._AsyncBoundaryWriter(refuse)
    failing.submit(5, {})
    with pytest.raises(RuntimeError, match="partition invariance failed at step 5"):
        failing.close()


def test_writer_count_must_be_positive(monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_HISTORY_WRITERS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        samehost._off_path_workers("GPUWM_HEX_HISTORY_WRITERS", 2)
    monkeypatch.setenv("GPUWM_HEX_HISTORY_WRITERS", "4")
    assert samehost._off_path_workers("GPUWM_HEX_HISTORY_WRITERS", 2) == 4


def test_parallel_assembly_matches_serial(monkeypatch):
    from hexcore.drivers import samehost_identity_v841 as identity
    calls = []

    def square(value):
        calls.append(value)
        return value * value

    monkeypatch.setenv("GPUWM_HEX_GATHER_THREADS", "1")
    serial = identity._parallel_fields(square, range(50))
    monkeypatch.setenv("GPUWM_HEX_GATHER_THREADS", "8")
    parallel = identity._parallel_fields(square, range(50))
    assert serial == parallel == [value * value for value in range(50)]


def test_frames_are_bounded_including_active_builds(tmp_path, fake_writer):
    import threading
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, {}, {}, workers=1)

    def blocked():
        entered.set()
        assert release.wait(10)
        return {"arrays": {}}, {}

    writer.submit(0, tmp_path / "h0.nc", blocked)
    assert entered.wait(10)

    def submit_next():
        writer.submit(1, tmp_path / "h1.nc", lambda: ({"arrays": {}}, {}))
        submitted.set()

    thread = threading.Thread(target=submit_next)
    thread.start()
    try:
        assert not submitted.wait(0.1)
    finally:
        release.set()
        thread.join(10)
        writer.close()
    assert submitted.is_set() and writer.frames == 2
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.submit(2, tmp_path / "h2.nc", blocked)


def test_hashes_overlap_handoff_and_finish_before_publication(tmp_path, fake_writer, monkeypatch):
    import threading
    from hexcore import history_spool
    copying = threading.Event()
    dump = history_spool.dump_payload

    def handoff(*args):
        copying.set()
        return dump(*args)

    def sha(array):
        assert copying.wait(10), "handoff must progress while hashing is active"
        return _sha(array)

    monkeypatch.setattr(history_spool, "dump_payload", handoff)
    records, hashes = {}, {}
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, sha, records, hashes)
    writer.submit(0, tmp_path / "h0.nc", lambda: ({"arrays": {"a": np.ones(3)}}, {}))
    writer.close()
    assert hashes == {"0": {"a": "3.0"}} and records["0"]["n_cells"] == 11


def test_hash_failure_prevents_child_publication(tmp_path, fake_writer):
    def refuse(array):
        raise ValueError("array digest refused")

    records, hashes = {}, {}
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, refuse, records, hashes)
    path = tmp_path / "h0.nc"
    writer.submit(0, path, lambda: ({"arrays": {"a": np.ones(3)}}, {}))
    with pytest.raises(RuntimeError, match="array digest refused"):
        writer.close()
    assert not path.exists() and not records and not hashes


def test_history_capture_bounds_storage_before_gather_and_publishes_closed_future(tmp_path, fake_writer):
    from threading import Event, Thread
    records, hashes = {}, {}
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, records, hashes)
    entered, release, gathered = Event(), Event(), Event()
    def first():
        entered.set()
        assert release.wait(5)
        return {'arrays': {'a': np.arange(3, dtype='f4')}}, {}
    first_done = writer.submit(0, tmp_path / 'h0.nc', first)
    assert entered.wait(5) and not first_done.done()
    second_done = []
    def second():
        second_done.append(writer.capture(1, tmp_path / 'h1.nc',
            lambda: (gathered.set(), np.ones(3, dtype='f4'))[1],
            lambda array: ({'arrays': {'a': array}}, {})))
    thread = Thread(target=second)
    thread.start()
    assert not gathered.wait(0.05)  # bounded before another frame allocation
    release.set()
    thread.join(5)
    assert not thread.is_alive() and gathered.is_set()
    writer.close()
    assert first_done.result() == records['0'] and second_done[0].result() == records['1']
    assert (tmp_path / 'h0.nc').is_file() and (tmp_path / 'h1.nc').is_file()


def test_failed_gather_releases_capture_slot_and_failed_build_completes_future(tmp_path, fake_writer):
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, {}, {})
    def bad_gather():
        raise ValueError('capture refused')
    with pytest.raises(ValueError, match='capture refused'):
        writer.capture(0, tmp_path / 'unused.nc', bad_gather, lambda arrays: arrays)
    assert writer._slots.acquire(blocking=False)
    writer._slots.release()
    def bad_build():
        raise FloatingPointError('gate refused')
    future = writer.submit(1, tmp_path / 'h1.nc', bad_build)
    with pytest.raises(FloatingPointError, match='gate refused'):
        future.result(timeout=5)
    with pytest.raises(RuntimeError, match='gate refused'):
        writer.close()


def test_idle_writer_releases_completed_frame_storage(tmp_path, fake_writer):
    import weakref
    writer = samehost._AsyncHistoryWriter(fake_writer.write_snapshot_netcdf, _sha, {}, {})
    array = np.arange(8, dtype='f4')
    retained = weakref.ref(array)
    future = writer.submit(0, tmp_path / 'h0.nc',
                           lambda array=array: ({'arrays': {'a': array}}, {}))
    del array
    future.result(timeout=5)
    deadline = time.monotonic() + 5
    while retained() is not None and time.monotonic() < deadline:
        time.sleep(0.001)
    try:
        assert retained() is None  # idle get() must not hold the previous frame
    finally:
        writer.close()
