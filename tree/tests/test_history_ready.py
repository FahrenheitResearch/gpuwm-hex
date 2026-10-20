"""Product notifications follow successful writes and never run on the caller."""
from concurrent.futures import Future
import threading

import pytest

from hexcore.history_ready import HistoryReady


def test_blocked_product_callback_does_not_block_writer_completion(tmp_path):
    entered, release = threading.Event(), threading.Event()
    events = []
    caller = threading.get_ident()

    def consume(event):
        assert threading.get_ident() != caller
        assert set(event) == {"schema", "step", "label", "valid_time", "path", "bytes", "sha256"}
        events.append(event)
        entered.set()
        assert release.wait(5)

    ready = HistoryReady(consume)
    first = {"path": str(tmp_path / "first.nc"), "bytes": 3, "sha256": "a" * 64}
    context = {"step": 0, "label": "f000", "valid_time": "2026-10-18_00:00:00"}
    future = Future()
    ready.watch(future, **context)
    try:
        future.set_result(first)
        assert entered.wait(5)
        # The forecast's final result collection must not duplicate the event.
        ready.submit(future.result(), **context)
        second = Future()
        ready.watch(second, step=30, label="f003", valid_time="2026-10-18_03:00:00")
        second.set_result({**first, "path": str(tmp_path / "second.nc")})
        assert second.done()  # callback 1 is still blocked, writer 2 completed
    finally:
        release.set()
        ready.close()
    assert [event["step"] for event in events] == [0, 30]
    assert ready.receipt()["submitted"] == 2


def test_failed_write_and_unresolved_fallback_do_not_notify(tmp_path):
    events = []
    ready = HistoryReady(events.append)
    context = {"step": 0, "label": "f000", "valid_time": "2026-10-18_00:00:00"}
    failed, fallback = Future(), Future()
    ready.watch(failed, **context)
    ready.watch(fallback, **context)
    failed.set_exception(OSError("write failed"))
    fallback.set_result(object())
    assert events == []
    # Notification is allowed only after the existing fallback wrote a file.
    ready.submit({"path": str(tmp_path / "recovered.nc"), "bytes": 3, "sha256": "a" * 64}, **context)
    ready.close()
    assert len(events) == 1


def test_product_error_surfaces_at_drain():
    def fail(event):
        raise ValueError("product failed")
    ready = HistoryReady(fail)
    ready.submit({"path": "closed.nc", "bytes": 3, "sha256": "a" * 64},
                 step=0, label="f000", valid_time="2026-10-18_00:00:00")
    with pytest.raises(ValueError, match="product failed"):
        ready.close()
    assert ready.receipt()["completed"][0]["error"] == "ValueError('product failed')"


def test_full_product_queue_blocks_capture_caller_but_not_writer():
    from concurrent.futures import ThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()
    attempting, submitted = threading.Event(), threading.Event()
    seen = []
    def consume(event):
        entered.set()
        assert release.wait(5)
        seen.append(event['step'])
    ready = HistoryReady(consume, max_pending=2)
    context = dict(label='frame', valid_time='2026-10-18_00:00:00')
    def file(step):
        return dict(path=f'{step}.nc', bytes=3, sha256='a'*64)
    first, second = Future(), Future()
    ready.watch(first, step=0, **context)
    first.set_result(file(0))
    assert entered.wait(5)
    ready.watch(second, step=1, **context)
    def capture():
        attempting.set()
        ready.submit(file(2), step=2, **context)
        submitted.set()
    with ThreadPoolExecutor(max_workers=1) as worker:
        queued = worker.submit(capture)
        try:
            assert attempting.wait(5)
            assert not submitted.wait(0.05)
            # Future completion remains free while the capture caller waits.
            second.set_result(file(1))
            assert second.done()
            assert ready.receipt()['peak_pending'] == 2
        finally:
            release.set()
            queued.result(timeout=5)
            ready.close()
    assert seen == [0, 1, 2]
    assert ready.receipt()['peak_pending'] == 2


def test_one_session_process_survives_frames_and_records_required_readiness():
    from contextlib import contextmanager
    import json
    import os
    import subprocess
    import sys
    import time

    pids, threads, sessions = [], [], []
    first_complete = threading.Event()
    @contextmanager
    def session():
        threads.append(threading.get_ident())
        child = subprocess.Popen([sys.executable, '-u', '-c',
            'import sys,json,os\nfor line in sys.stdin:\n'
            ' e=json.loads(line); print(json.dumps({"pid":os.getpid(),'
            '"products":[{"id":e["label"],"path":e["path"],'
            '"bytes":e["bytes"],"sha256":e["sha256"]}]}),flush=True)'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sessions.append(child)
        def consume(event):
            threads.append(threading.get_ident())
            child.stdin.write(json.dumps(event)+'\n')
            child.stdin.flush()
            result = json.loads(child.stdout.readline())
            pids.append(result['pid'])
            first_complete.set()
            return result
        try:
            yield consume
        finally:
            threads.append(threading.get_ident())
            child.stdin.close()
            assert child.wait(timeout=5) == 0, child.stderr.read()
            child.stdout.close()
            child.stderr.close()
    ready = HistoryReady(session_factory=session, required_product_ids=('f000', 'f003'))
    file = dict(path='first.nc', bytes=3, sha256='a'*64)
    try:
        ready.submit(file, step=0, label='f000', valid_time='2026-10-18_00:00:00')
        assert first_complete.wait(5)
        ready.forecast_ended(time.monotonic(), time.time())
        ready.submit({**file,'path':'last.nc'}, step=30, label='f003', valid_time='2026-10-18_03:00:00')
    finally:
        ready.close()
    record = ready.receipt()
    assert len(sessions) == 1 and sessions[0].poll() == 0
    assert pids == [sessions[0].pid]*2 and pids[0] != os.getpid()
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()
    assert record['required_products_complete']
    assert record['missing_product_ids'] == []
    assert record['last_required_ready_minus_forecast_end_seconds'] >= 0
    assert record['last_required_ready_unix'] >= record['forecast_end_unix']


def test_missing_required_products_cannot_report_last_ready():
    ready = HistoryReady(lambda event: {'products': [
        {'id': 'global/t2/f003', 'status': 'unavailable', 'reason': 'missing input'}]},
        required_product_ids=('global/t2/f003',))
    ready.submit(dict(path='closed.nc', bytes=3, sha256='a'*64), step=30,
                 label='f003', valid_time='2026-10-18_03:00:00')
    ready.forecast_ended(100., 200.)
    ready.close()
    record = ready.receipt()
    assert record['missing_product_ids'] == ['global/t2/f003']
    assert not record['required_products_complete']
    assert record['last_required_ready_unix'] is None
    assert record['last_required_ready_minus_forecast_end_seconds'] is None


def test_product_failure_still_closes_session_on_its_worker():
    from contextlib import contextmanager

    opened, closed = [], []
    @contextmanager
    def session():
        opened.append(threading.get_ident())
        def fail(event):
            raise ValueError('render failed')
        try:
            yield fail
        finally:
            closed.append(threading.get_ident())
    ready = HistoryReady(session_factory=session)
    ready.submit(dict(path='closed.nc', bytes=3, sha256='a'*64), step=0,
                 label='f000', valid_time='2026-10-18_00:00:00')
    with pytest.raises(ValueError, match='render failed'):
        ready.close()
    assert opened == closed and len(opened) == 1
    assert opened[0] != threading.get_ident()
