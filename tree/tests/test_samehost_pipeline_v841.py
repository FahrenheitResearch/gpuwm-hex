"""Schedule-slot lifetime, byte identity, and failure checks on the CPU."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore.partition_samehost_pipeline_v841 import StepSlotTransport
from hexcore.partition_samehost_transport_v841 import SameHostTransport, SameHostTransportError, _Buffers


def tables(ranks):
    result = []
    for rank in range(ranks):
        peers = []
        for slot, peer in enumerate(other for other in range(ranks) if other != rank):
            peers.append(SimpleNamespace(peer=peer, cell_send=np.array([0, 1], dtype=np.int64),
                cell_recv=np.array([2 + 2 * slot, 3 + 2 * slot], dtype=np.int64),
                edge_send=np.array([0], dtype=np.int64), edge_recv=np.array([1 + slot], dtype=np.int64)))
        result.append(SimpleNamespace(rank=rank, peers=peers, n_owned_cells=2,
            n_local_cells=2 * ranks, n_owned_edges=1, n_local_edges=ranks))
    return result


def fields_for(rank_tables):
    fields, backing = [], []
    for table in rank_tables:
        width = table.n_local_cells
        bits = np.resize(np.array([0, 0x80000000, 0x7fc01234, 0xffc05678, 1], np.uint32), (2, 2 * width))
        scalar = bits.view(np.float32)
        edges = np.arange(2 * table.n_local_edges, dtype=np.uint16).reshape(2, -1)
        byte_backing = np.arange(2 * width, dtype=np.uint8)
        fields.append((("scalar", scalar[..., ::-2], "cell"), ("edge", edges, "edge"),
                       ("unaligned", byte_backing[1::2], "cell")))
        backing.append((scalar, byte_backing))
    return fields, backing


def exchange_all(transport, fields, tag):
    with ThreadPoolExecutor(max_workers=len(fields)) as pool:
        return list(pool.map(lambda rank: transport.exchange(rank, tag, fields[rank]), range(len(fields))))


def expected(rank_tables, fields):
    original = [[array.copy() for _, array, _ in rank] for rank in fields]
    result = [[array.copy() for array in rank] for rank in original]
    for rank, table in enumerate(rank_tables):
        for peer in table.peers:
            source_table = next(item for item in rank_tables[peer.peer].peers if item.peer == rank)
            for index, (_, _, kind) in enumerate(fields[rank]):
                result[rank][index][..., getattr(peer, kind + "_recv")] = (
                    original[peer.peer][index][..., getattr(source_table, kind + "_send")])
    return result


class PipelineTests(unittest.TestCase):
    def test_two_and_four_ranks_preserve_bits_and_padding_across_slot_reuse(self):
        for ranks in (2, 4):
            with self.subTest(ranks=ranks):
                rank_tables = tables(ranks)
                fields, backing = fields_for(rank_tables)
                padding = [(scalar[..., ::2].tobytes(), byte[::2].tobytes()) for scalar, byte in backing]
                transport = StepSlotTransport(rank_tables, [0] * ranks, xp=np)
                for step in range(1, 4):
                    transport.begin_step(step, ("C", "B"))
                    for position, label in enumerate(("C", "B")):
                        for rank, rank_fields in enumerate(fields):
                            rank_fields[0][1][..., :2].view(np.uint32)[...] = np.array(
                                [0x80000000 | (step << 8) | rank, 0x7fc01000 | (position << 4) | rank], np.uint32)
                            rank_fields[1][1][..., :1] = rank + 100 * step + 10 * position
                            rank_fields[2][1][..., :2] = rank + step + position
                        truth = expected(rank_tables, fields)
                        exchange_all(transport, fields, f"{step}:{position}:{label}")
                        for rank, rank_fields in enumerate(fields):
                            for index, (_, array, _) in enumerate(rank_fields):
                                self.assertEqual(array.tobytes(), truth[rank][index].tobytes())
                            self.assertEqual(backing[rank][0][..., ::2].tobytes(), padding[rank][0])
                            self.assertEqual(backing[rank][1][::2].tobytes(), padding[rank][1])
                    transport.end_step(step)
                receipt = transport.receipt()
                self.assertEqual(receipt["pipeline"]["steps_completed"], 3)
                self.assertEqual(receipt["pipeline"]["second_barrier_wait_calls"], 0)
                self.assertEqual(receipt["stats"]["rounds"], 6)
                self.assertEqual(receipt["retained_slot_packet_bytes"], 2 * receipt["stats"]["bytes"] // 3)
                self.assertEqual(receipt["prepared_slots"], 2)
                for cache in transport._cache:
                    buffers = list(cache.values())
                    self.assertEqual(len(buffers), 2)
                    self.assertIsNot(buffers[0].send[next(iter(buffers[0].send))].buffer,
                                     buffers[1].send[next(iter(buffers[1].send))].buffer)
                transport.close()

    def test_fast_rank_advances_while_peer_is_still_receiving_previous_slot(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np, timeout=2)
        transport.begin_step(1, 2)
        slow_entered, release_slow, fast_advanced = threading.Event(), threading.Event(), threading.Event()
        receive = transport._receive

        def delayed_receive(rank, round_, submission):
            if rank == 0 and round_.tag == "1:0:C":
                slow_entered.set()
                if not release_slow.wait(1):
                    raise RuntimeError("test did not release the delayed receiver")
            receive(rank, round_, submission)

        transport._receive = delayed_receive

        def run(rank):
            transport.exchange(rank, "1:0:C", fields[rank])
            if rank == 1:
                fast_advanced.set()
            transport.exchange(rank, "1:1:B", fields[rank])

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, rank) for rank in range(2)]
            try:
                self.assertTrue(slow_entered.wait(1))
                self.assertTrue(fast_advanced.wait(1), "fast rank was blocked by a second receive barrier")
                # A free-threaded caller can observe fast_advanced immediately
                # after set(), before that worker enters its next exchange.
                with transport._condition:
                    self.assertTrue(transport._condition.wait_for(
                        lambda: 1 in transport._slot_rounds, timeout=1),
                        "fast rank did not enter the next slot while its peer was receiving")
            finally:
                release_slow.set()
            for future in futures:
                future.result(timeout=2)
        transport.end_step(1)
        transport.close()

    def test_missing_rank_timeout_permanently_stops_the_transport(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np, timeout=0.03)
        transport.begin_step(1, 1)
        with self.assertRaisesRegex(SameHostTransportError, "timed out"):
            transport.exchange(0, "1:0:C", fields[0])
        with self.assertRaises(SameHostTransportError):
            transport.exchange(1, "1:0:C", fields[1])
        self.assertTrue(transport.receipt()["aborted"])
        self.assertTrue(transport._failed_rounds)
        transport.close()

    def test_tag_disagreement_wakes_waiting_peer(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np, timeout=1)
        transport.begin_step(1, 1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(transport.exchange, rank, f"1:0:{'C' if rank == 0 else 'B'}", fields[rank])
                       for rank in range(2)]
            for future in futures:
                with self.assertRaises(SameHostTransportError):
                    future.result(timeout=2)
        self.assertTrue(transport.receipt()["aborted"])
        transport.close()

    def test_duplicate_entry_aborts_both_calls(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np, timeout=1)
        transport.begin_step(1, 1)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(transport.exchange, 0, "1:0:C", fields[0]) for _ in range(2)]
            for future in futures:
                with self.assertRaises(SameHostTransportError):
                    future.result(timeout=2)
        self.assertTrue(transport.receipt()["aborted"])
        transport.close()

    def test_slot_schema_is_immutable_across_steps(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np)
        transport.begin_step(1, 1)
        exchange_all(transport, fields, "1:0:C")
        transport.end_step(1)
        transport.begin_step(2, 1)
        changed = (("scalar", np.zeros((2, 4), dtype=np.float64), "cell"), *fields[0][1:])
        with self.assertRaisesRegex(SameHostTransportError, "schema changed"):
            transport.exchange(0, "2:0:C", changed)
        self.assertTrue(transport.receipt()["aborted"])
        transport.close()

    def test_partial_group_and_unchecked_reuse_are_refused(self):
        for operation in (lambda transport: transport.end_step(1), lambda transport: transport.begin_step(2, 1),
                          lambda transport: transport.synchronize()):
            transport = StepSlotTransport(tables(2), [0, 0], xp=np)
            transport.begin_step(1, 1)
            with self.assertRaises(SameHostTransportError):
                operation(transport)
            self.assertTrue(transport.receipt()["aborted"])
            transport.close()

    def test_end_step_drains_before_marking_packets_reusable(self):
        rank_tables = tables(2)
        fields, _ = fields_for(rank_tables)
        transport = StepSlotTransport(rank_tables, [0, 0], xp=np)
        transport.begin_step(1, 1)
        exchange_all(transport, fields, "1:0:C")
        drained = []
        synchronize = transport.synchronize

        def checked_drain(**kwargs):
            self.assertTrue(transport._ending)
            self.assertEqual(transport._step_id, 1)
            self.assertEqual(transport.pipeline_stats["steps_completed"], 0)
            synchronize(**kwargs)
            drained.append(True)

        transport.synchronize = checked_drain
        transport.end_step(1)
        self.assertEqual(drained, [True])
        self.assertIsNone(transport._step_id)
        transport.synchronize = synchronize
        transport.begin_step(2, 1)
        exchange_all(transport, fields, "2:0:C")
        transport.end_step(2)
        transport.close()

    def test_failure_before_producer_publication_still_drains_queued_stream(self):
        # This fake implements only context selection and event bookkeeping.
        # It never imports CuPy, opens a GPU context, or launches CUDA work.
        class Device:
            def __init__(self, device):
                self.id = device

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        for failure in ("pack", "event_create", "event_record"):
            with self.subTest(failure=failure):
                rank_tables = tables(2)
                fields, _ = fields_for(rank_tables)
                transport = StepSlotTransport(rank_tables, [0, 0], xp=np)
                transport.begin_step(1, 1)
                stream = SimpleNamespace(ptr=1234)
                recorded = []
                event_count = [0]

                class Event:
                    done = True

                    def __init__(self, **_):
                        self.position = event_count[0]
                        event_count[0] += 1
                        if self.position == 0 and failure == "event_create":
                            raise RuntimeError("injected producer event creation failure")

                    def record(self, selected):
                        if self.position == 0 and failure == "event_record":
                            raise RuntimeError("injected producer event record failure")
                        recorded.append(selected)

                transport._cpu = False
                transport.xp = SimpleNamespace(cuda=SimpleNamespace(
                    Device=Device, Event=Event, get_current_stream=lambda: stream))
                transport._schema = lambda rank, payload: (("test", "cell", "<f4", (2,)),)
                transport._slot_buffers = lambda *_: _Buffers({}, {})
                packed = []

                def enqueue_pack(*_, **kwargs):
                    packed.append(kwargs["stream"])
                    if failure == "pack":
                        raise RuntimeError("injected error after queued pack")

                transport._copy_packets = enqueue_pack
                with self.assertRaisesRegex(SameHostTransportError, "injected"):
                    transport.exchange(0, "1:0:C", fields[0])
                self.assertEqual(packed, [stream])
                self.assertEqual(transport._active_exchanges, 0)
                self.assertFalse(transport._slot_rounds[0].submissions)
                self.assertIs(transport._producer_streams[0, 1234], stream)
                self.assertIs(transport._failed_rounds[0], transport._slot_rounds[0])
                self.assertIs(transport._slot_rounds[0].reserved_fields[0], fields[0])
                with patch.object(SameHostTransport, "synchronize", return_value=None) as transfer_drain:
                    transport.synchronize(timeout=1)
                    transfer_drain.assert_called_once()
                self.assertEqual(recorded, [stream], "direct drain failed to fence the queued producer")
                transport._cpu, transport.xp = True, np
                transport.close()


if __name__ == "__main__":
    unittest.main()
