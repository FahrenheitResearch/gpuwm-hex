"""Primary failure, peer abort and bounded worker-drain checks without CUDA."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.partition_executor_v841 import ExecutorError
from hexcore.partition_samehost_v841 import SameHostRanks


class _Device:
    def __init__(self, device):
        self.id = device

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def use(self):
        pass


class _Stream:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.synchronize_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def synchronize(self):
        self.synchronize_calls += 1
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error


class _FakeGPU:
    def __init__(self, errors_by_rank):
        self.streams = []
        def stream(**_):
            result = _Stream(errors_by_rank[len(self.streams)])
            self.streams.append(result)
            return result
        self.cuda = SimpleNamespace(Device=_Device, Stream=stream)


class _Transport:
    def __init__(self, *, close_error=None, abort_error=None):
        self.close_error = close_error
        self.abort_error = abort_error
        self.aborted = threading.Event()
        self.reasons = []
        self.close_calls = 0
        self.lock = threading.Lock()

    def abort(self, reason):
        with self.lock:
            self.reasons.append(reason)
        self.aborted.set()
        if self.abort_error is not None:
            raise self.abort_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _raise(error):
    raise error


class SameHostFailureTests(unittest.TestCase):
    def test_operation_error_survives_sticky_cuda_drain_and_close(self):
        primary = ValueError("primary kernel failure")
        drain = RuntimeError("cudaErrorIllegalAddress on synchronize")
        close = RuntimeError("cudaErrorIllegalAddress on Event.create")
        cp = _FakeGPU([(drain,)])
        transport = _Transport(close_error=close)
        ranks = SameHostRanks([0], transport, cp=cp)
        with self.assertRaises(ValueError) as caught:
            with ranks:
                ranks.call(lambda _: _raise(primary))
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__notes__, [
            "rank 0 CUDA stream drain also failed: RuntimeError: cudaErrorIllegalAddress on synchronize",
            "same-host rank group close also failed: RuntimeError: cudaErrorIllegalAddress on Event.create",
        ])
        self.assertEqual(cp.streams[0].synchronize_calls, 1)
        self.assertEqual(transport.close_calls, 1)
        self.assertEqual(transport.reasons[0], "rank 0: ValueError: primary kernel failure")
        self.assertTrue(ranks.failed)
        with self.assertRaisesRegex(ExecutorError, "failed rank group cannot resume"):
            ranks.call(lambda _: None)

    def test_first_stream_sync_error_survives_second_sync_error(self):
        primary = RuntimeError("first CUDA synchronize failure")
        drain = RuntimeError("later sticky CUDA synchronize failure")
        cp = _FakeGPU([(primary, drain)])
        transport = _Transport()
        with SameHostRanks([0], transport, cp=cp) as ranks:
            with self.assertRaises(RuntimeError) as caught:
                ranks.call(lambda _: "completed host operation")
            self.assertIs(caught.exception, primary)
            self.assertEqual(cp.streams[0].synchronize_calls, 2)
            self.assertEqual(primary.__notes__, [
                "rank 0 CUDA stream drain also failed: RuntimeError: later sticky CUDA synchronize failure"])
            self.assertTrue(ranks.failed)

    def test_body_error_survives_close_error_without_worker_failure(self):
        primary = RuntimeError("step receipt failure")
        close = RuntimeError("transport Event.create failure")
        transport = _Transport(close_error=close)
        with self.assertRaises(RuntimeError) as caught:
            with SameHostRanks([0], transport):
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__notes__, [
            "same-host rank group close also failed: RuntimeError: transport Event.create failure"])
        self.assertEqual(transport.close_calls, 1)

    def test_close_error_propagates_when_context_body_succeeds(self):
        close = RuntimeError("standalone transport drain failure")
        transport = _Transport(close_error=close)
        with self.assertRaises(RuntimeError) as caught:
            with SameHostRanks([0], transport):
                pass
        self.assertIs(caught.exception, close)
        self.assertEqual(transport.close_calls, 1)

    def test_direct_close_error_propagates(self):
        close = RuntimeError("direct transport drain failure")
        ranks = SameHostRanks([0], _Transport(close_error=close))
        with self.assertRaises(RuntimeError) as caught:
            ranks.close()
        self.assertIs(caught.exception, close)

    def test_close_error_propagates_after_a_body_catches_worker_failure(self):
        primary = ValueError("caught operation failure")
        close = RuntimeError("later standalone transport drain failure")
        transport = _Transport(close_error=close)
        with self.assertRaises(RuntimeError) as caught:
            with SameHostRanks([0], transport) as ranks:
                with self.assertRaises(ValueError) as operation_error:
                    ranks.call(lambda _: _raise(primary))
                self.assertIs(operation_error.exception, primary)
        self.assertIs(caught.exception, close)
        self.assertEqual(transport.close_calls, 1)

    def test_abort_failure_is_not_allowed_to_replace_operation_error(self):
        primary = ValueError("operation failure")
        transport = _Transport(abort_error=RuntimeError("abort hook failure"))
        cp = _FakeGPU([()])
        with SameHostRanks([0], transport, cp=cp) as ranks:
            with self.assertRaises(ValueError) as caught:
                ranks.call(lambda _: _raise(primary))
            self.assertIs(caught.exception, primary)
            self.assertEqual(primary.__notes__, [
                "rank 0 transport abort also failed: RuntimeError: abort hook failure"])
            self.assertEqual(cp.streams[0].synchronize_calls, 1)

    def test_abort_wakes_peers_and_call_waits_for_every_rank_drain(self):
        primary = ValueError("primary operation failure")
        peer_errors = [None, ExecutorError("peer 1 observed abort"), ExecutorError("peer 2 observed abort")]
        started = [threading.Event(), threading.Event()]
        release_peers = threading.Event()
        finished = [False] * 3
        cp = _FakeGPU([(RuntimeError(f"rank {rank} sticky drain failure"),) for rank in range(3)])
        transport = _Transport()
        ranks = SameHostRanks([0, 0, 0], transport, cp=cp)
        def operation(rank):
            if rank == 0:
                if not all(event.wait(2) for event in started):
                    raise AssertionError("peer workers did not enter")
                raise primary
            started[rank - 1].set()
            if not transport.aborted.wait(2):
                raise AssertionError("rank failure did not abort peers")
            if not release_peers.wait(2):
                raise AssertionError("test did not release peer drain")
            finished[rank] = True
            raise peer_errors[rank]
        try:
            with ThreadPoolExecutor(max_workers=1) as coordinator:
                result = coordinator.submit(ranks.call, operation)
                self.assertTrue(transport.aborted.wait(2))
                self.assertFalse(result.done())
                release_peers.set()
                with self.assertRaises(ValueError) as caught:
                    result.result(timeout=2)
                self.assertIs(caught.exception, primary)
            self.assertEqual(finished[1:], [True, True])
            self.assertEqual([stream.synchronize_calls for stream in cp.streams], [1, 1, 1])
            self.assertEqual(transport.reasons[0], "rank 0: ValueError: primary operation failure")
            self.assertEqual(len(transport.reasons), 3)
            for rank, error in enumerate((primary, *peer_errors[1:])):
                self.assertEqual(error.__notes__, [
                    f"rank {rank} CUDA stream drain also failed: RuntimeError: rank {rank} sticky drain failure"])
            self.assertTrue(ranks.failed)
            with self.assertRaisesRegex(ExecutorError, "failed rank group cannot resume"):
                ranks.call(lambda _: None)
        finally:
            release_peers.set()
            ranks.close()

    def test_successful_step_retains_clock_and_single_drain(self):
        cp = _FakeGPU([(), ()])
        transport = _Transport()
        clocks = [0.0, 0.0]
        with SameHostRanks([0, 0], transport, cp=cp) as ranks:
            def operation(rank):
                clocks[rank] += 120.0
                return rank
            result = ranks.step(operation, lambda rank: clocks[rank], dt=120.0)
            self.assertEqual(result, [0, 1])
            self.assertEqual(ranks.time_seconds, 120.0)
            self.assertEqual(clocks, [120.0, 120.0])
            self.assertEqual([stream.synchronize_calls for stream in cp.streams], [1, 1])
            self.assertFalse(ranks.failed)
            self.assertFalse(transport.aborted.is_set())


if __name__ == "__main__":
    unittest.main()
