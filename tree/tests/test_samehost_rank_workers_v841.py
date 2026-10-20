"""Persistent rank worker, device initialization and complete drain checks."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.partition_executor_v841 import ExecutorError
from hexcore.partition_samehost_v841 import SameHostRanks


class _Transport:
    def __init__(self, abort_error=None):
        self.aborted = threading.Event()
        self.reasons = []
        self.close_calls = 0
        self.abort_error = abort_error
        self.lock = threading.Lock()

    def abort(self, reason):
        with self.lock:
            self.reasons.append(reason)
        self.aborted.set()
        if self.abort_error is not None:
            raise self.abort_error

    def close(self):
        self.close_calls += 1


class _FakeCUDA:
    def __init__(self, *, use_errors=None, sync_errors=None):
        self.local = threading.local()
        self.use_errors = use_errors or {}
        self.sync_errors = sync_errors or {}
        self.use_calls, self.contexts, self.streams = [], [], []
        self.lock = threading.Lock()
        owner = self
        class Device:
            def __init__(self, device):
                self.id = int(device)
            def use(self):
                with owner.lock:
                    owner.use_calls.append((threading.get_ident(), self.id))
                if self.id in owner.use_errors:
                    raise owner.use_errors[self.id]
                owner.local.device = self.id
            def __enter__(self):
                self.previous = owner.current()
                with owner.lock:
                    owner.contexts.append((threading.get_ident(), self.previous, self.id))
                owner.local.device = self.id
                return self
            def __exit__(self, *_):
                owner.local.device = self.previous
                return False
        class Stream:
            def __init__(self, **_):
                self.device = owner.current()
                self.errors = list(owner.sync_errors.get(len(owner.streams), ()))
                self.synchronize_calls = 0
                owner.streams.append(self)
            def __enter__(self):
                if owner.current() != self.device:
                    raise AssertionError("rank entered a stream on another device")
                return self
            def __exit__(self, *_):
                return False
            def synchronize(self):
                if owner.current() != self.device:
                    raise AssertionError("rank drained a stream on another device")
                self.synchronize_calls += 1
                if self.errors:
                    raise self.errors.pop(0)
        self.cuda = SimpleNamespace(Device=Device, Stream=Stream)

    def current(self):
        return getattr(self.local, "device", 9)


class RankWorkerTests(unittest.TestCase):
    def test_repeated_gpu_calls_preserve_rank_thread_and_current_device(self):
        cp, transport = _FakeCUDA(), _Transport()
        devices = (2, 4, 6, 8)
        barrier = threading.Barrier(len(devices), timeout=2)
        identities = None
        with SameHostRanks(devices, transport, cp=cp) as ranks:
            for iteration in range(30):
                def operation(rank):
                    barrier.wait()
                    if rank == iteration % len(devices):
                        time.sleep(0.001)
                    return rank, threading.get_ident(), cp.current()
                observed = ranks.call(operation)
                self.assertEqual([row[0] for row in observed], list(range(len(devices))))
                self.assertEqual([row[2] for row in observed], list(devices))
                current_identities = [row[1] for row in observed]
                if identities is None:
                    identities = current_identities
                self.assertEqual(current_identities, identities)
                self.assertEqual(len(set(identities)), len(devices))
                self.assertEqual(cp.current(), 9)
            self.assertEqual(sorted(cp.use_calls), sorted(zip(identities, devices)))
            for worker, previous, selected in cp.contexts:
                if worker in identities:
                    expected = devices[identities.index(worker)]
                    self.assertEqual(previous, expected)
                    self.assertEqual(selected, expected)
            self.assertEqual([stream.synchronize_calls for stream in cp.streams], [30] * len(devices))
        self.assertEqual(transport.close_calls, 1)

    def test_cpu_workers_are_stable_and_execute_barrier_concurrently(self):
        transport = _Transport()
        barrier = threading.Barrier(3, timeout=2)
        with SameHostRanks((0, 0, 0), transport) as ranks:
            def operation(rank):
                barrier.wait()
                return threading.get_ident()
            initial = ranks.call(operation)
            self.assertEqual(len(set(initial)), 3)
            for _ in range(12):
                self.assertEqual(ranks.call(operation), initial)
        self.assertFalse(transport.reasons)

    def test_original_initializer_error_aborts_and_drains_every_rank(self):
        primary = ValueError("worker device initialization failed")
        drain = RuntimeError("sticky initialized-rank stream failure")
        cp = _FakeCUDA(use_errors={4: primary}, sync_errors={1: (drain,)})
        transport = _Transport()
        executed = []
        with SameHostRanks((2, 4, 6), transport, cp=cp) as ranks:
            def operation(rank):
                executed.append(rank)
                if not transport.aborted.wait(2):
                    raise AssertionError("initializer failure did not abort peer operations")
                raise ExecutorError(f"peer {rank} observed initializer abort")
            with self.assertRaises(ValueError) as caught:
                ranks.call(operation)
            self.assertIs(caught.exception, primary)
            self.assertEqual(sorted(executed), [0, 2])
            self.assertEqual([stream.synchronize_calls for stream in cp.streams], [1, 1, 1])
            self.assertEqual(primary.__notes__, [
                "rank 1 CUDA stream drain also failed: RuntimeError: sticky initialized-rank stream failure"])
            self.assertEqual(transport.reasons[0], "rank 1: ValueError: worker device initialization failed")
            self.assertEqual(len(cp.use_calls), 3)
            self.assertTrue(ranks.failed)
            with self.assertRaisesRegex(ExecutorError, "failed rank group cannot resume"):
                ranks.call(operation)

    def test_partial_submit_failure_aborts_and_drains_submitted_gpu_work(self):
        primary = RuntimeError("rank submit failure")
        cp, transport = _FakeCUDA(), _Transport()
        entered, completed = threading.Event(), threading.Event()
        with SameHostRanks((2, 4, 6), transport, cp=cp) as ranks:
            def operation(rank):
                self.assertEqual(rank, 0)
                entered.set()
                if not transport.aborted.wait(2):
                    raise AssertionError("submit failure did not abort the submitted worker")
                completed.set()
                return rank
            def reject(*args, **kwargs):
                if not entered.wait(2):
                    raise AssertionError("submitted worker never entered")
                raise primary
            with patch.object(ranks.pools[1], "submit", side_effect=reject):
                with self.assertRaises(RuntimeError) as caught:
                    ranks.call(operation)
            self.assertIs(caught.exception, primary)
            self.assertTrue(completed.is_set())
            self.assertEqual([stream.synchronize_calls for stream in cp.streams], [1, 0, 0])
            self.assertEqual(len(cp.use_calls), 1)
            self.assertEqual(transport.reasons, ["rank 1 submission failed: RuntimeError: rank submit failure"])
            self.assertTrue(ranks.failed)

    def test_submit_error_survives_abort_hook_failure(self):
        primary = RuntimeError("submit failure")
        transport = _Transport(abort_error=RuntimeError("abort hook failure"))
        with SameHostRanks((0, 0), transport) as ranks:
            with patch.object(ranks.pools[0], "submit", side_effect=primary):
                with self.assertRaises(RuntimeError) as caught:
                    ranks.call(lambda _: None)
            self.assertIs(caught.exception, primary)
            self.assertEqual(primary.__notes__, [
                "rank 0 submission transport abort also failed: RuntimeError: abort hook failure"])

    def test_successful_clock_commit_keeps_rank_result_order(self):
        clocks = [0.0, 0.0, 0.0]
        barrier = threading.Barrier(3, timeout=2)
        with SameHostRanks((0, 0, 0), _Transport()) as ranks:
            def operation(rank):
                barrier.wait()
                clocks[rank] += 5.0
                return 10 - rank
            for step in range(1, 5):
                self.assertEqual(ranks.step(operation, clocks.__getitem__, dt=5.0), [10, 9, 8])
                self.assertEqual(ranks.time_seconds, step * 5.0)
                self.assertEqual(clocks, [step * 5.0] * 3)

    def test_close_shuts_every_dedicated_pool(self):
        transport = _Transport()
        ranks = SameHostRanks((0, 0, 0), transport)
        barrier = threading.Barrier(3, timeout=2)
        ranks.call(lambda _: barrier.wait())
        with ExitStack() as patches:
            spies = [patches.enter_context(patch.object(pool, "shutdown", wraps=pool.shutdown)) for pool in ranks.pools]
            ranks.close()
            for spy in spies:
                spy.assert_called_once_with(wait=True)
        self.assertEqual(transport.close_calls, 1)

    def test_shutdown_failure_still_attempts_every_pool_and_preserves_first(self):
        primary, secondary = RuntimeError("first pool shutdown failure"), RuntimeError("second pool shutdown failure")
        transport = _Transport()
        ranks = SameHostRanks((0, 0, 0), transport)
        originals = [pool.shutdown for pool in ranks.pools]
        def shutdown(index):
            def invoke(*args, **kwargs):
                originals[index](*args, **kwargs)
                if index < 2:
                    raise (primary, secondary)[index]
            return invoke
        with ExitStack() as patches:
            spies = [patches.enter_context(patch.object(pool, "shutdown", side_effect=shutdown(index)))
                     for index, pool in enumerate(ranks.pools)]
            with self.assertRaises(RuntimeError) as caught:
                ranks.close()
            self.assertIs(caught.exception, primary)
            for spy in spies:
                spy.assert_called_once_with(wait=True)
        self.assertEqual(primary.__notes__, [
            "rank 1 worker shutdown also failed: RuntimeError: second pool shutdown failure"])
        self.assertEqual(transport.close_calls, 0)
        self.assertTrue(ranks.failed)


if __name__ == "__main__":
    unittest.main()
