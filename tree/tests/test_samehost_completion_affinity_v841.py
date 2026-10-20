"""Completion affinity and partial-failure ownership checks without CUDA."""

from __future__ import annotations

from collections import deque
import gc
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore import partition_samehost_transport_v841 as transport_module


class _Owner:
    pass


class _Stream:
    def __init__(self, cuda, device, ptr):
        self.cuda, self.device, self.ptr = cuda, device, ptr

    def __enter__(self):
        if self.cuda.current != self.device:
            raise AssertionError("stream entered on the wrong device")
        return self

    def __exit__(self, *_):
        return False

    def wait_event(self, event):
        self.cuda.calls.append(("stream-wait", self.cuda.current, self.ptr, event))


class _FakeCUDA:
    def __init__(self):
        self.local = threading.local()
        self.calls = []
        self.events = []
        self.current_stream = None

    @property
    def current(self):
        return getattr(self.local, "device", -1)

    def Device(self, device):
        cuda = self

        class Device:
            def use(self):
                cuda.local.device = device
                cuda.calls.append(("use", device))

            def __enter__(self):
                self.previous = cuda.current
                cuda.local.device = device
                cuda.calls.append(("enter", device))
                return self

            def __exit__(self, *_):
                cuda.local.device = self.previous
                return False

        return Device()

    def Event(self, **options):
        cuda, device = self, self.current
        cuda.calls.append(("new-event", device, options))

        class Event:
            done = True

            def record(self, stream):
                if cuda.current != device or stream.device != device:
                    raise AssertionError("event recorded on the wrong device")
                cuda.calls.append(("record", device, stream.ptr))

        event = Event()
        self.events.append(event)
        return event

    def Stream(self, **_):
        raise AssertionError("completion and drain must not allocate a stream")

    def get_current_stream(self):
        if self.current != self.current_stream.device:
            raise AssertionError("producer stream requested on the wrong device")
        return self.current_stream


class _Receiver:
    def __init__(self, cuda, rank, device, *, entered=None, release=None, error=None):
        self.cuda, self.rank, self.device = cuda, rank, device
        self.entered, self.release, self.error = entered, release, error

    def synchronize(self):
        if self.cuda.current != self.device:
            raise AssertionError("receiver completion waited on the wrong device")
        self.cuda.calls.append(("receiver-wait", self.rank, self.device))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(2):
            raise AssertionError("test did not release its receiver")
        if self.error is not None:
            raise self.error


def _transport(devices=(7, 2, 5), cuda=None):
    result = transport_module.SameHostTransport.__new__(transport_module.SameHostTransport)
    result.devices = tuple(devices)
    result.n_ranks = len(devices)
    result.xp = SimpleNamespace(cuda=cuda or _FakeCUDA())
    result._cpu = False
    result.timeout = 1
    result._condition = threading.Condition()
    result._stats_lock = threading.Lock()
    result._round = transport_module._Round(0, float("inf"))
    result._pending = deque()
    result._failed_rounds = []
    result._error = None
    result._stopping = result._closed = False
    result._completion_worker = result._watchdog_worker = None
    result._copy_streams = [None] * len(devices)
    result._staging_streams = {}
    result.stats = {"completed_rounds": 0}
    # SameHostTransport.__init__ defaults since a176dc77 (per-rank DMA stall timing, off unless asked)
    result.measure_gpu = False
    result.gpu_step_timings = {}
    return result


def _submission(event=None):
    return transport_module._Submission((), transport_module._Buffers({}, {}), received=event)


class CompletionAffinityTests(unittest.TestCase):
    def test_queue_completion_has_no_cuda_calls_and_owns_every_receiver(self):
        class NoCUDA:
            def __getattr__(self, name):
                raise AssertionError(f"queue completion touched CUDA {name}")

        transport = _transport(cuda=NoCUDA())
        starts = []
        transport._start_completion_workers = lambda: starts.append(True)
        round_ = transport_module._Round(4, float("inf"))
        receivers = [object() for _ in transport.devices]
        for rank in (2, 0, 1):
            round_.submissions[rank] = _submission(receivers[rank])
        with transport._condition:
            transport._queue_completion(round_)
        self.assertEqual(round_.completion, tuple(
            (rank, device, receivers[rank]) for rank, device in enumerate(transport.devices)))
        self.assertIsInstance(round_.completion, tuple)
        self.assertIs(transport._pending[0], round_)
        self.assertEqual(starts, [True])
        self.assertEqual(transport.stats["completed_rounds"], 0)

    def test_waits_every_receiver_on_its_device_and_rotates_first_rank(self):
        transport = _transport()
        cuda = transport.xp.cuda
        for generation in range(5):
            cuda.calls.clear()
            round_ = transport_module._Round(generation, float("inf"))
            round_.completion = tuple((rank, device, _Receiver(cuda, rank, device))
                                      for rank, device in enumerate(transport.devices))
            transport._wait_completion_events(round_)
            order = [(generation + offset) % transport.n_ranks for offset in range(transport.n_ranks)]
            self.assertEqual(cuda.calls, [item for rank in order for item in (
                ("use", transport.devices[rank]), ("receiver-wait", rank, transport.devices[rank]))])
            self.assertEqual(cuda.current, transport.devices[order[-1]])
            self.assertFalse(cuda.events)

    def test_retirement_keeps_all_owners_until_last_receiver_completes(self):
        transport = _transport()
        cuda = transport.xp.cuda
        entered, release = threading.Event(), threading.Event()
        owner = _Owner()
        reference = weakref.ref(owner)
        round_ = transport_module._Round(0, float("inf"), reserved_fields=[owner])
        receivers = [_Receiver(cuda, rank, device,
                               entered=entered if rank == 2 else None,
                               release=release if rank == 2 else None)
                     for rank, device in enumerate(transport.devices)]
        round_.completion = tuple((rank, device, receivers[rank])
                                  for rank, device in enumerate(transport.devices))
        for rank, event in enumerate(receivers):
            submission = _submission(event)
            submission.owners.append(owner)
            round_.submissions[rank] = submission
        transport._pending.append(round_)
        del owner, round_, submission
        worker = threading.Thread(target=transport._retire_completed)
        worker.start()
        try:
            self.assertTrue(entered.wait(2), "worker never reached the last receiver")
            gc.collect()
            self.assertIsNotNone(reference())
            self.assertEqual(len(transport._pending), 1)
            self.assertEqual(transport.stats["completed_rounds"], 0)
            release.set()
            with transport._condition:
                self.assertTrue(transport._condition.wait_for(lambda: not transport._pending, timeout=2))
            self.assertEqual(transport.stats["completed_rounds"], 1)
            self.assertEqual([call[1] for call in cuda.calls if call[0] == "receiver-wait"], [0, 1, 2])
        finally:
            release.set()
            with transport._condition:
                transport._stopping = True
                transport._condition.notify_all()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        gc.collect()
        self.assertIsNone(reference(), "retired round retained its field owners")

    def test_receiver_failure_never_pops_pending_or_failed_owners(self):
        transport = _transport()
        cuda = transport.xp.cuda
        pending_owner, failed_owner = _Owner(), _Owner()
        refs = weakref.ref(pending_owner), weakref.ref(failed_owner)
        round_ = transport_module._Round(0, float("inf"), reserved_fields=[pending_owner])
        round_.completion = tuple((rank, device, _Receiver(
            cuda, rank, device, error=RuntimeError("receiver failed") if rank == 1 else None))
            for rank, device in enumerate(transport.devices))
        transport._pending.append(round_)
        transport._round.reserved_fields.append(failed_owner)
        del pending_owner, failed_owner, round_
        worker = threading.Thread(target=transport._retire_completed)
        worker.start()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in refs))
        self.assertEqual(len(transport._pending), 1)
        self.assertEqual(transport.stats["completed_rounds"], 0)
        self.assertEqual(transport._failed_rounds, [transport._round])
        self.assertIn("round 0 CUDA completion failed: receiver failed", str(transport._error))
        self.assertEqual([call[1] for call in cuda.calls if call[0] == "receiver-wait"], [0, 1])

    def test_receive_event_uses_blocking_host_wait_without_timing(self):
        transport = _transport((5,))
        cuda = transport.xp.cuda
        transport._copy_streams = [_Stream(cuda, 5, 14)]
        transport._copy_packets = lambda *args, **kwargs: None
        submission = _submission()
        submission.produced = object()
        round_ = transport_module._Round(0, float("inf"))
        round_.submissions[0] = submission
        transport._receive(0, round_, submission)
        self.assertIs(submission.received, cuda.events[0])
        self.assertIn(("new-event", 5, {"block": True, "disable_timing": True}), cuda.calls)
        self.assertEqual([call for call in cuda.calls if call[0] == "record"], [("record", 5, 14)])

    def test_drain_includes_current_pending_failed_producers_and_dedups_streams(self):
        transport = _transport((5, 5, 8))
        cuda = transport.xp.cuda
        transport._copy_streams = [_Stream(cuda, 5, 11), _Stream(cuda, 5, 11), _Stream(cuda, 8, 11)]
        transport._staging_streams = {5: _Stream(cuda, 5, 12), 8: _Stream(cuda, 8, 21)}
        transport._round.producer_streams = {0: _Stream(cuda, 5, 13)}
        pending = transport_module._Round(1, float("inf"))
        pending.producer_streams = {1: _Stream(cuda, 5, 14), 2: _Stream(cuda, 8, 22)}
        failed = transport_module._Round(2, float("inf"))
        failed.producer_streams = {0: _Stream(cuda, 5, 13), 2: _Stream(cuda, 8, 23)}
        transport._pending.append(pending)
        transport._failed_rounds.append(failed)
        transport.synchronize()
        self.assertEqual([call for call in cuda.calls if call[0] == "record"], [
            ("record", 5, 11), ("record", 5, 12), ("record", 5, 13), ("record", 5, 14),
            ("record", 8, 11), ("record", 8, 21), ("record", 8, 22), ("record", 8, 23)])
        self.assertEqual(len(cuda.events), 8)
        self.assertFalse(any(call[0] in ("use", "stream-wait") for call in cuda.calls))
        self.assertEqual(transport.stats["completed_rounds"], 1)
        self.assertFalse(transport._pending)
        self.assertIs(transport._failed_rounds[0], failed)

    def test_pack_failure_registers_and_drains_producer_before_submission_publish(self):
        transport = _transport((8,))
        cuda = transport.xp.cuda
        producer = _Stream(cuda, 8, 31)
        cuda.current_stream = producer
        transport._copy_streams = [_Stream(cuda, 8, 32)]
        transport._schema = lambda *_: ()
        buffers = transport_module._Buffers({}, {})
        owner = _Owner()
        reference = weakref.ref(owner)
        fields = (("payload", owner, "cell"),)

        def buffers_before_pack(*_):
            self.assertIs(transport._round.producer_streams[0], producer)
            return buffers

        def failed_pack(*_, **__):
            self.assertIs(transport._round.producer_streams[0], producer)
            raise RuntimeError("pack launch failed after enqueue")

        transport._buffers = buffers_before_pack
        transport._copy_packets = failed_pack
        with self.assertRaisesRegex(transport_module.SameHostTransportError, "pack launch failed"):
            transport.exchange(0, "partial-pack", fields)
        self.assertFalse(transport._round.submissions)
        self.assertIs(transport._failed_rounds[0], transport._round)
        self.assertEqual(transport._round.producer_streams, {0: producer})
        del owner, fields
        gc.collect()
        self.assertIsNotNone(reference())
        transport.synchronize()
        self.assertEqual([call for call in cuda.calls if call[0] == "record"], [
            ("record", 8, 32), ("record", 8, 31)])
        self.assertIsNotNone(reference(), "failed round lost its original field owner")

    def test_drain_failure_keeps_pending_and_original_owners(self):
        transport = _transport((5,))
        cuda = transport.xp.cuda
        transport._copy_streams = [_Stream(cuda, 5, 9)]
        owner = _Owner()
        reference = weakref.ref(owner)
        pending = transport_module._Round(0, float("inf"), reserved_fields=[owner])
        transport._pending.append(pending)
        del pending, owner

        def fail_event(**_):
            raise RuntimeError("event creation failed")

        cuda.Event = fail_event
        with self.assertRaisesRegex(RuntimeError, "event creation failed"):
            transport.synchronize()
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(len(transport._pending), 1)
        self.assertEqual(transport.stats["completed_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
