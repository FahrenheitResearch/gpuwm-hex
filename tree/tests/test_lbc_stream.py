from datetime import datetime, timedelta
import json
import threading
import time

import numpy as np
import pytest

from hexcore.lbc import LbcInventory, LbcPool
from hexcore.lbc_stream import atomic_json, initial_paths, wait_file, file_name, SERIES, READY, FORMAT, sha256
from test_lbc import write_lbc


def publish(root, stamp, value):
    path = root / file_name(stamp)
    write_lbc(path, stamp, value)
    atomic_json(path.with_name(path.name + ".ready.json"), {"schema": READY, "valid_time": stamp,
                "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return path


def stamps():
    start = datetime(2026, 8, 12, 6)
    return [(start + timedelta(hours=hour)).strftime(FORMAT) for hour in range(4)]


def test_stream_pool_matches_serial_at_every_interval(tmp_path, monkeypatch):
    serial, stream = tmp_path / "serial", tmp_path / "stream"
    serial.mkdir(); stream.mkdir()
    times = stamps()
    paths = [publish(serial, stamp, value) for stamp, value in zip(times, (1., 4., 2., 8.))]
    atomic_json(stream / "lbc-series.json", {"schema": SERIES, "valid_times": times})
    for stamp, value in zip(times[:2], (1., 4.)):
        publish(stream, stamp, value)
    left, right = LbcPool(LbcInventory(paths)), LbcPool(LbcInventory(initial_paths(stream)))
    start = datetime.strptime(times[0], FORMAT)
    left.start(start); right.start(start)
    for hour in range(3):
        if hour > 0:
            publish(stream, times[hour + 1], (2., 8.)[hour - 1])
        left.advance(); right.advance()
        for field in left._state.fields:
            assert left.tendency(field).tobytes() == right.tendency(field).tobytes()
            moment = start + timedelta(hours=hour, minutes=30)
            assert left.state_at(field, moment).tobytes() == right.state_at(field, moment).tobytes()


def test_late_hour_wait_is_named_and_cannot_skip(tmp_path, monkeypatch):
    from hexcore import time_ledger
    events = []
    monkeypatch.setattr(time_ledger, "emit", lambda stage, phase="mark", **kw: events.append((stage, phase, kw)))
    times = stamps()
    atomic_json(tmp_path / "lbc-series.json", {"schema": SERIES, "valid_times": times})
    for stamp in (times[0], times[1], times[3]):
        publish(tmp_path, stamp, 1.)
    inventory = LbcInventory(initial_paths(tmp_path))
    thread = threading.Thread(target=lambda: (time.sleep(.15), publish(tmp_path, times[2], 2.)))
    thread.start()
    assert inventory.earliest_strictly_after(datetime.strptime(times[1], FORMAT)).name == file_name(times[2])
    thread.join()
    assert [(stage, phase) for stage, phase, _ in events] == [("linger.lbc_hour", "begin"), ("linger.lbc_hour", "end")]
    assert events[0][2]["attrs"]["reason"] == "missing_lbc_hour"


def test_unmarked_or_corrupt_endpoint_refuses(tmp_path):
    stamp = stamps()[0]
    path = publish(tmp_path, stamp, 1.)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        wait_file(tmp_path, stamp)
    marker = path.with_name(path.name + ".ready.json")
    marker.unlink()
    with pytest.raises(ValueError, match="missing_lbc_hour"):
        wait_file(tmp_path, stamp, timeout=.01)


def test_failed_producer_and_wrong_xtime_refuse(tmp_path):
    times = stamps()
    atomic_json(tmp_path / "lbc-failed.json", {"error": "forced producer failure"})
    with pytest.raises(ValueError, match="forced producer failure"):
        wait_file(tmp_path, times[0])
    (tmp_path / "lbc-failed.json").unlink()
    atomic_json(tmp_path / "lbc-series.json", {"schema": SERIES, "valid_times": times})
    publish(tmp_path, times[0], 1.)
    path = publish(tmp_path, times[1], 2.)
    write_lbc(path, times[2], 2.)
    marker = path.with_name(path.name + ".ready.json")
    doc = json.loads(marker.read_text()); doc.update(bytes=path.stat().st_size, sha256=sha256(path))
    marker.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="xtime differs"):
        initial_paths(tmp_path)


def test_stream_start_inside_the_declared_window_uses_latest_endpoint(tmp_path):
    times = stamps()
    atomic_json(tmp_path / "lbc-series.json", {"schema": SERIES, "valid_times": times})
    for hour, stamp in enumerate(times):
        publish(tmp_path, stamp, float(hour))
    pool = LbcPool(LbcInventory(initial_paths(tmp_path)))
    clock = datetime.strptime(times[2], FORMAT) + timedelta(minutes=15)
    assert pool.start(clock).path.name == file_name(times[2])
    assert pool.advance(clock).path.name == file_name(times[3])


def test_quarter_hour_parent_boundary_schedule_and_missing_interval(tmp_path):
    from hexcore.lbc_stream import series_document
    start = datetime(2022, 6, 14)
    times = [(start + timedelta(minutes=15*i)).strftime(FORMAT) for i in range(5)]
    atomic_json(tmp_path / 'lbc-series.json', dict(schema=SERIES, valid_times=times, interval_seconds=900))
    for i, stamp in enumerate(times):
        publish(tmp_path, stamp, float(i))
    pool = LbcPool(LbcInventory(initial_paths(tmp_path)))
    pool.start(start); pool.advance(start)
    np.testing.assert_array_equal(pool.state_at('lbc_theta', start + timedelta(seconds=450)), .5)
    document = json.loads((tmp_path/'lbc-series.json').read_text())
    document['valid_times'].pop(2)
    (tmp_path/'lbc-series.json').write_text(json.dumps(document))
    with pytest.raises(ValueError, match='missing interpolation endpoint'):
        series_document(tmp_path)
