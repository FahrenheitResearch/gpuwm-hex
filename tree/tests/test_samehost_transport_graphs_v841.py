"""Byte-layout and retained descriptor checks without a desktop GPU.

GPU cases and timing run only with the explicit --gpu switch. The benchmark
can load an unmodified transport file with --transport-source for a paired
before/after check under the same rank and packet layout.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import importlib.util
import hashlib
import gc
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hexcore import partition_samehost_transport_v841 as transport_module


def _tables(ranks=4, owned=2):
    result = []
    for rank in range(ranks):
        peers = []
        for slot, peer in enumerate(other for other in range(ranks) if other != rank):
            peers.append(SimpleNamespace(
                peer=peer, cell_send=np.arange(owned, dtype=np.int64),
                cell_recv=np.arange((slot + 1) * owned, (slot + 2) * owned, dtype=np.int64),
                edge_send=np.array([0], dtype=np.int64),
                edge_recv=np.array([slot + 1], dtype=np.int64),
            ))
        result.append(SimpleNamespace(rank=rank, peers=peers, n_owned_cells=owned,
                                     n_local_cells=ranks * owned, n_owned_edges=1,
                                     n_local_edges=ranks))
    return result


def _fields(xp, tables, binding=0):
    fields, storage = [], []
    for rank, table in enumerate(tables):
        width = table.n_local_cells
        bits = np.resize(np.array([0, 0x80000000, 0x7fc01234, 0xffc05678,
                                   0x3f800000, 0x00000001], dtype=np.uint32), (3, 2 * width))
        seed = (rank << 8) ^ (binding << 16)
        bits[:, ::-2][:, :table.n_owned_cells] ^= np.uint32(seed & 0xffffffff)
        backing = xp.asarray(bits.view(np.float32))
        wide_bits = np.resize(np.array([0, 0x8000000000000000, 0x7ff8000000004321], dtype=np.uint64),
                              (2, 2 * width))
        wide_bits[:, ::2][:, :table.n_owned_cells] ^= np.uint64(seed & 0xffffffffffffffff)
        wide_backing = xp.asarray(wide_bits.view(np.float64))
        byte_values = np.arange(2 * width, dtype=np.uint8)
        byte_values[1::2][:table.n_owned_cells] ^= np.uint8(((rank << 4) ^ binding) & 0xff)
        byte_backing = xp.asarray(byte_values)
        edge_values = np.arange(2 * table.n_local_edges, dtype=np.uint16).reshape(2, -1)
        edge_values[:, :table.n_owned_edges] ^= np.uint16(((rank << 8) ^ (binding << 4)) & 0xffff)
        edges = xp.asarray(edge_values)
        fields.append((("payload32", backing[..., ::-2], "cell"),
                       ("payload64", wide_backing[..., ::2], "cell"),
                       ("unaligned", byte_backing[1::2], "cell"),
                       ("edges", edges, "edge")))
        storage.append((backing, wide_backing, byte_backing, edges))
    return fields, storage


def _expected(tables, fields, to_host=np.asarray):
    originals = [[to_host(array).copy() for _, array, _ in rank] for rank in fields]
    result = [[array.copy() for array in rank] for rank in originals]
    for rank, table in enumerate(tables):
        for peer in table.peers:
            reverse = next(item for item in tables[peer.peer].peers if item.peer == rank)
            for slot, (_, _, kind) in enumerate(fields[rank]):
                result[rank][slot][..., getattr(peer, kind + "_recv")] = (
                    originals[peer.peer][slot][..., getattr(reverse, kind + "_send")])
    return result


def _exchange_all(transport, fields, tag, xp=np, streams=None):
    def exchange(rank):
        if xp is np:
            return transport.exchange(rank, tag, fields[rank])
        with xp.cuda.Device(0), streams[rank]:
            return transport.exchange(rank, tag, fields[rank])
    with ThreadPoolExecutor(max_workers=len(fields)) as pool:
        return list(pool.map(exchange, range(len(fields))))


class ByteReferenceTests(unittest.TestCase):
    def test_four_ranks_preserve_payloads_strides_and_padding(self):
        tables = _tables()
        fields, storage = _fields(np, tables)
        expected = _expected(tables, fields)
        untouched = [[array.copy() for array in rank] for rank in storage]
        transport = transport_module.SameHostTransport(tables, [0] * 4, xp=np)
        for step in range(3):
            _exchange_all(transport, fields, step)
            for rank in range(4):
                for slot, (_, array, _) in enumerate(fields[rank]):
                    self.assertEqual(array.tobytes(), expected[rank][slot].tobytes())
                self.assertEqual(storage[rank][0][..., ::2].tobytes(), untouched[rank][0][..., ::2].tobytes())
                self.assertEqual(storage[rank][1][..., 1::2].tobytes(), untouched[rank][1][..., 1::2].tobytes())
                self.assertEqual(storage[rank][2][::2].tobytes(), untouched[rank][2][::2].tobytes())
        self.assertEqual(transport.receipt()["stats"]["rounds"], 3)
        transport.close()

    def test_schema_error_wakes_peers(self):
        tables = _tables(2)
        fields, _ = _fields(np, tables)
        fields[1] = fields[1] + (("payload32", fields[1][0][1], "cell"),)
        transport = transport_module.SameHostTransport(tables, [0, 0], xp=np, timeout=1)
        with self.assertRaises(transport_module.SameHostTransportError):
            _exchange_all(transport, fields, "invalid")
        self.assertTrue(transport.receipt()["aborted"])

    def test_empty_single_rank(self):
        tables = _tables(1)
        fields, _ = _fields(np, tables)
        transport = transport_module.SameHostTransport(tables, [0], xp=np)
        before = [array.tobytes() for _, array, _ in fields[0]]
        moved = _exchange_all(transport, fields, "empty")
        self.assertEqual(moved[0]["bytes_sent"], 0)
        self.assertEqual(before, [array.tobytes() for _, array, _ in fields[0]])
        transport.close()


class _DeviceArray:
    """Address-table test double. It never opens a CUDA context."""

    def __init__(self, array):
        self.array = np.asarray(array)
        self.data = SimpleNamespace(ptr=int(self.array.ctypes.data))
        for name in ("dtype", "shape", "strides", "ndim", "size", "nbytes"):
            setattr(self, name, getattr(self.array, name))


class _AddressXP:
    int64 = np.int64

    @staticmethod
    def asarray(array, dtype=None):
        return _DeviceArray(np.asarray(array, dtype=dtype))


class DescriptorBatchTests(unittest.TestCase):
    def setUp(self):
        self.transport = transport_module.SameHostTransport.__new__(transport_module.SameHostTransport)
        self.transport.xp = _AddressXP
        self.transport._cpu = False
        self.transport.descriptor_cache_size = 2
        # SameHostTransport.__init__ default since a176dc77: no per-field halo plan, the peer's own lists
        self.transport.field_halo_indices = {}
        self.transport._stats_lock = threading.Lock()
        self.transport.stats = dict.fromkeys(("pack_launches", "unpack_launches", "descriptor_uploads",
                                             "dimension_uploads", "descriptor_cache_hits",
                                             "descriptor_cache_misses", "descriptor_cache_evictions",
                                             "serial_unpack_batches", "serial_unpack_launches"), 0)
        self.launches = []
        self.transport._kernel = lambda grid, block, args, **kwargs: self.launches.append((grid, args))
        self.transport._word_kernel = self.transport._kernel

    @staticmethod
    def _packet(count, fields):
        specs, indices, offset = [], [], 0
        for _, array, _ in fields:
            shape = array.shape[:-1] + (count,)
            size = int(np.prod(shape)) * array.dtype.itemsize
            specs.append((array.dtype, shape, offset, size))
            indices.append(_DeviceArray(np.arange(count, dtype=np.int64)))
            offset += size
        return transport_module._Packet(_DeviceArray(np.zeros(offset, dtype=np.uint8)), tuple(specs), tuple(indices))

    def test_unequal_peers_fuse_and_keep_packet_byte_offsets(self):
        fields = (("x", _DeviceArray(np.zeros((3, 8), dtype=np.float32)[:, ::2]), "cell"),
                  ("y", _DeviceArray(np.zeros(4, dtype=np.uint8)), "cell"))
        packets = [self._packet(count, fields) for count in (0, 1, 3)]
        batch = transport_module._Batch()
        self.transport._copy_packets(fields, packets, batch, scatter=False)
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(self.launches[0][0], (32, 4))
        desc = batch.descriptor.array
        self.assertEqual(desc[:, 4].tolist(), [1, 1, 3, 3])
        self.assertEqual(desc[:, 1].tolist(), [packets[1].buffer.data.ptr,
                                             packets[1].buffer.data.ptr + 12,
                                             packets[2].buffer.data.ptr,
                                             packets[2].buffer.data.ptr + 36])
        self.transport._copy_packets(fields, packets, batch, scatter=True)
        self.assertEqual(self.transport.stats["descriptor_uploads"], 1)
        self.assertEqual(self.transport.stats["descriptor_cache_hits"], 1)
        self.assertEqual(self.transport.stats["pack_launches"], 1)
        self.assertEqual(self.transport.stats["unpack_launches"], 1)

    def test_word_kernel_selected_once_for_aligned_level_major_binding(self):
        words = []
        self.transport._word_kernel = lambda *args, **kwargs: words.append(args)
        array = _DeviceArray(np.zeros((55, 8), dtype=np.uint32)[::-1])
        fields = (("x", array, "cell"),)
        packets = [self._packet(3, fields)]
        batch = transport_module._Batch()
        self.transport._copy_packets(fields, packets, batch, scatter=False)
        self.transport._copy_packets(fields, packets, batch, scatter=True)
        self.assertEqual(len(words), 2)
        self.assertEqual(len(self.launches), 0)
        self.assertEqual(self.transport.stats["descriptor_cache_hits"], 1)
        # A changed layout must select the general path even with the same
        # logical schema. It cannot reuse the previous binding's selection.
        fields = (("x", _DeviceArray(array.array[:, ::-1]), "cell"),)
        self.transport._copy_packets(fields, packets, batch, scatter=False)
        self.assertEqual(len(words), 2)
        self.assertEqual(len(self.launches), 1)

    def test_word_kernel_falls_back_before_signed_index_overflow(self):
        words = []
        self.transport._word_kernel = lambda *args, **kwargs: words.append(args)
        for count, eligible in ((2147483647, True), (2147483648, False)):
            with self.subTest(words=count):
                # Virtual metadata exercises the narrowing boundary without
                # allocating an 8 GiB array. Both launch functions are stubs.
                array = _DeviceArray(np.zeros((1, 1), dtype=np.uint32))
                array.shape = (count, 1)
                fields = (("x", array, "cell"),)
                packet = transport_module._Packet(
                    _DeviceArray(np.zeros(4, dtype=np.uint8)),
                    ((array.dtype, (count, 1), 0, 4 * count),),
                    (_DeviceArray(np.array([0], dtype=np.int64)),),
                )
                before_words, before_generic = len(words), len(self.launches)
                self.transport._copy_packets(fields, [packet], transport_module._Batch(), scatter=False)
                self.assertEqual(len(words) - before_words, int(eligible))
                self.assertEqual(len(self.launches) - before_generic, int(not eligible))

    def test_rotation_reuses_tables_and_eviction_retains_submitted_owners(self):
        arrays = [_DeviceArray(np.zeros((2, 4), dtype=np.float32)) for _ in range(4)]
        fields = lambda index: (("x", arrays[index], "cell"),)
        packets = [self._packet(2, fields(0))]
        batch = transport_module._Batch()
        self.transport._copy_packets(fields(0), packets, batch, scatter=False)
        submitted = transport_module._Submission(fields(0), transport_module._Buffers({}, {}))
        submitted.owners.extend((batch.descriptor, *batch.owners))
        old_descriptor = batch.descriptor
        old_dims = batch.owners[0]
        self.transport._copy_packets(fields(1), packets, batch, scatter=False)
        self.transport._copy_packets(fields(0), packets, batch, scatter=False)
        self.assertIs(batch.descriptor, old_descriptor)
        for index in (2, 3):
            self.transport._copy_packets(fields(index), packets, batch, scatter=False)
        self.assertEqual(len(batch.bindings), 2)
        self.assertEqual(self.transport.stats["descriptor_cache_evictions"], 2)
        self.assertIn(old_descriptor, submitted.owners)
        self.assertIn(old_dims, submitted.owners)
        self.assertEqual(old_descriptor.array[0, 0], arrays[0].data.ptr)

    def test_sixty_four_rotating_bindings_need_no_warm_descriptor_uploads(self):
        self.transport.descriptor_cache_size = 64
        arrays = [_DeviceArray(np.zeros((2, 4), dtype=np.float32)) for _ in range(64)]
        fields = lambda index: (("x", arrays[index], "cell"),)
        packets, batch = [self._packet(2, fields(0))], transport_module._Batch()
        for _ in range(2):
            for index in range(64):
                self.transport._copy_packets(fields(index), packets, batch, scatter=False)
        self.assertEqual(self.transport.stats["descriptor_cache_misses"], 64)
        self.assertEqual(self.transport.stats["descriptor_cache_hits"], 64)
        self.assertEqual(self.transport.stats["descriptor_uploads"], 64)
        self.assertEqual(self.transport.stats["dimension_uploads"], 1)
        self.assertEqual(self.transport.stats["descriptor_cache_evictions"], 0)
        self.assertEqual(self.transport.stats["pack_launches"], 128)
        self.assertEqual(len(batch.bindings), 64)

    def test_sixty_four_binding_alias_cache_retains_evicted_launch_owners(self):
        self.transport.descriptor_cache_size = 64
        backing = [np.arange(4, dtype=np.uint32) for _ in range(65)]
        families = [(("cells", _DeviceArray(array[1:4]), "cell"),
                     ("edges", _DeviceArray(array[:3]), "edge")) for array in backing]
        packets = [self._packet(1, families[0]), self._packet(1, families[0])]
        batch = transport_module._Batch()
        self.transport._copy_packets(families[0], packets, batch, scatter=True)
        old_descriptor = self.launches[-1][1][0]
        submitted = transport_module._Submission(families[0], transport_module._Buffers({}, {}))
        submitted.owners.extend((batch.descriptor, *batch.owners))
        for fields in families[1:]:
            self.transport._copy_packets(fields, packets, batch, scatter=True)
        uploads = self.transport.stats["descriptor_uploads"]
        self.transport._copy_packets(families[64], packets, batch, scatter=True)
        self.assertTrue(batch.serial_unpack)
        self.assertEqual(self.transport.stats["descriptor_uploads"], uploads)
        self.assertEqual(self.transport.stats["descriptor_cache_misses"], 65)
        self.assertEqual(self.transport.stats["descriptor_cache_hits"], 1)
        self.assertEqual(self.transport.stats["descriptor_cache_evictions"], 1)
        self.assertEqual(self.transport.stats["serial_unpack_batches"], 66)
        self.assertEqual(self.transport.stats["serial_unpack_launches"], 132)
        self.assertEqual(len(batch.bindings), 64)
        self.assertIn(old_descriptor, submitted.owners)
        self.assertEqual(old_descriptor.array[0, 0], families[0][0][1].data.ptr)

    def test_cached_bindings_do_not_keep_model_arrays_alive(self):
        backing = np.arange(4, dtype=np.uint32)
        fields = (("cells", _DeviceArray(backing[1:4]), "cell"),
                  ("edges", _DeviceArray(backing[:3]), "edge"))
        refs = [weakref.ref(backing), weakref.ref(fields[0][1]), weakref.ref(fields[1][1])]
        packets = [self._packet(1, fields), self._packet(1, fields)]
        batch = transport_module._Batch()
        self.transport._copy_packets(fields, packets, batch, scatter=True)
        del fields, backing
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual(len(batch.bindings), 1)
        self.assertTrue(batch.serial_unpack)

    def test_zero_packets_queue_no_kernel(self):
        fields = (("x", _DeviceArray(np.zeros(4, dtype=np.float32)), "cell"),)
        batch = transport_module._Batch()
        self.transport._copy_packets(fields, [self._packet(0, fields)], batch, scatter=False)
        self.assertFalse(self.launches)
        self.assertEqual(self.transport.stats["descriptor_uploads"], 0)

    def test_crosskind_aliases_keep_serial_peer_scatter(self):
        backing = np.arange(4, dtype=np.uint32)
        fields = (("cells", _DeviceArray(backing[1:4]), "cell"),
                  ("edges", _DeviceArray(backing[:3]), "edge"))
        packets = [self._packet(1, fields), self._packet(1, fields)]
        batch = transport_module._Batch()
        self.transport._copy_packets(fields, packets, batch, scatter=False)
        self.assertEqual(len(self.launches), 1)
        self.transport._copy_packets(fields, packets, batch, scatter=True)
        self.assertEqual(len(self.launches), 3)
        self.assertTrue(batch.serial_unpack)
        self.assertEqual([grid for grid, _ in self.launches], [(32, 4), (32, 2), (32, 2)])
        self.assertEqual(self.transport.stats["unpack_launches"], 2)
        self.assertEqual(self.transport.stats["serial_unpack_batches"], 1)
        self.assertEqual(self.transport.stats["serial_unpack_launches"], 2)
        self.assertEqual(self.transport.stats["descriptor_uploads"], 3)
        self.assertIn(self.launches[-2][1][0], batch.owners)
        self.assertIn(self.launches[-1][1][0], batch.owners)

    def test_byte_spans_include_negative_strides_and_empty_views(self):
        backing = np.arange(12, dtype=np.uint32)
        fields = lambda first, second: (("first", _DeviceArray(first), "cell"),
                                        ("second", _DeviceArray(second), "edge"))
        self.assertTrue(self.transport._overlapping_fields(fields(backing[8:2:-1], backing[1:4])))
        self.assertFalse(self.transport._overlapping_fields(fields(backing[8:4:-1], backing[:4])))
        self.assertFalse(self.transport._overlapping_fields(fields(backing[3:3], backing[:4])))

    def test_parallel_counter_updates_are_exact(self):
        class InterleavedInt(int):
            def __add__(self, other):
                time.sleep(0)
                return InterleavedInt(int(self) + other)

        self.transport.stats = {name: InterleavedInt(value) for name, value in self.transport.stats.items()}
        workers, rounds = 8, 25
        def worker(_):
            fields = (("x", _DeviceArray(np.zeros(4, dtype=np.float32)), "cell"),)
            packets, batch = [self._packet(2, fields)], transport_module._Batch()
            for _ in range(rounds):
                self.transport._copy_packets(fields, packets, batch, scatter=False)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(worker, range(workers)))
        self.assertEqual(self.transport.stats["pack_launches"], workers * rounds)
        self.assertEqual(self.transport.stats["descriptor_cache_misses"], workers)
        self.assertEqual(self.transport.stats["descriptor_cache_hits"], workers * (rounds - 1))
        self.assertEqual(self.transport.stats["descriptor_uploads"], workers)
        self.assertEqual(self.transport.stats["dimension_uploads"], workers)

    def test_only_nonempty_staged_packets_hold_a_source_stream(self):
        stage = object()
        class PacketXP(_AddressXP):
            uint8 = np.uint8
            cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0),
                                   alloc_pinned_memory=lambda size: bytearray(size))
            @staticmethod
            def empty(size, dtype):
                return _DeviceArray(np.empty(size, dtype=dtype))
        self.transport.xp = PacketXP
        self.transport.devices = (0, 1)
        self.transport._staging_streams = {0: stage}
        fields = (("x", _DeviceArray(np.zeros(4, dtype=np.float32)), "cell"),)
        for path, count in (("local", 2), ("peer", 2), ("staged", 0), ("staged", 2)):
            self.transport.paths = {(0, 1): path}
            peer = SimpleNamespace(peer=1, cell_send=np.arange(count, dtype=np.int64))
            packet = self.transport._make_packet(fields, peer, "send")
            if path == "staged" and count:
                self.assertIs(packet.source_stream, stage)
                self.assertEqual(len(packet.host), count * 4)
            else:
                self.assertIsNone(packet.source_stream)
                self.assertIsNone(packet.host)

    def test_tail_drain_records_only_unique_used_streams(self):
        recorded = []
        class Device:
            def __init__(self, _):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        class Event:
            done = True
            def __init__(self, **_):
                pass
            def record(self, stream):
                recorded.append(stream.ptr)
        self.transport.xp = SimpleNamespace(cuda=SimpleNamespace(Device=Device, Event=Event))
        self.transport.timeout = 1
        self.transport._condition = threading.Condition()
        self.transport._round = transport_module._Round(0, float("inf"))
        self.transport._error = None
        self.transport._pending = deque()
        self.transport._failed_rounds = []
        self.transport.devices = (0, 0, 0)
        self.transport._copy_streams = [SimpleNamespace(ptr=value) for value in (7, 7, 8)]
        self.transport._staging_streams = {0: SimpleNamespace(ptr=9)}
        self.transport._round.producer_streams[0] = SimpleNamespace(ptr=10)
        self.transport.stats["completed_rounds"] = 0
        self.transport.synchronize()
        self.assertEqual(recorded, [7, 8, 9, 10])


def _gpu_alias_check(module, cp):
    tables = _tables(3, owned=1)
    reference_storage = [np.arange(4, dtype=np.uint32) + rank * 100 for rank in range(3)]
    reference_fields = [(("cells", array[1:4], "cell"), ("edges", array[:3], "edge"))
                        for array in reference_storage]
    reference = module.SameHostTransport(tables, [0] * 3, xp=np)
    _exchange_all(reference, reference_fields, "alias-reference")
    reference.close()
    storage = [cp.asarray(np.arange(4, dtype=np.uint32) + rank * 100) for rank in range(3)]
    fields = [(("cells", array[1:4], "cell"), ("edges", array[:3], "edge")) for array in storage]
    cp.cuda.get_current_stream().synchronize()
    transport = module.SameHostTransport(tables, [0] * 3, xp=cp)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in tables]
    _exchange_all(transport, fields, "alias-device", cp, streams)
    for stream in streams:
        stream.synchronize()
    transport.synchronize()
    for rank, array in enumerate(storage):
        if cp.asnumpy(array).tobytes() != reference_storage[rank].tobytes():
            raise AssertionError(f"rank {rank} crosskind alias scatter changed serial peer ordering")
    receipt = transport.receipt()
    if "serial_unpack_batches" in receipt["stats"]:
        if receipt["stats"]["serial_unpack_batches"] != 3 or receipt["stats"]["serial_unpack_launches"] != 6:
            raise AssertionError("crosskind alias scatter fallback counters are not exact")
    transport.close()
    return {"byte_identical_to_serial_peer_reference": True, "transport": receipt}


def _gpu_word_check(module, cp):
    """Exercise cached word-kernel dispatch with real CUDA halo exchanges."""
    if not hasattr(module, "_WORD_COPY_2D"):
        return {"applicable": False, "reason": "transport has no word specialization"}
    tables = _tables(2, owned=17)
    fields = []
    for rank, table in enumerate(tables):
        bits = np.resize(np.array([0, 0x80000000, 0x7fa12345, 0x7fc54321,
                                   1, 0xffffffff], dtype=np.uint32), (55, table.n_local_cells))
        bits[:, :table.n_owned_cells] ^= np.uint32(rank << 8)
        fields.append((("level_major", cp.asarray(bits), "cell"),
                       ("negative_rows", cp.asarray(bits)[::-1], "cell"),
                       ("one_dim", cp.asarray(bits[0]), "cell")))
    expected = _expected(tables, fields, cp.asnumpy)
    transport = module.SameHostTransport(tables, [0, 0], xp=cp, timeout=30)
    word_kernel = transport._word_kernel
    word_launches = []
    def checked_word_kernel(*args, **kwargs):
        word_launches.append(int(args[2][1]))
        return word_kernel(*args, **kwargs)
    transport._word_kernel = checked_word_kernel
    streams = [cp.cuda.Stream(non_blocking=True) for _ in tables]
    try:
        for generation in range(2):
            _exchange_all(transport, fields, generation, cp, streams)
            for stream in streams:
                stream.synchronize()
            for rank, row in enumerate(fields):
                for slot, (_, array, _) in enumerate(row):
                    if cp.asnumpy(array).tobytes() != expected[rank][slot].tobytes():
                        raise AssertionError(f"word halo rank {rank} field {slot} changed bytes")
        if word_launches.count(0) != 4 or word_launches.count(1) != 4:
            raise AssertionError(f"word halo did not use four pack and four unpack launches: {word_launches!r}")
        return {"applicable": True, "byte_identical": True, "levels": 55, "ranks": 2, "rounds": 2,
                "word_pack_launches": word_launches.count(0),
                "word_unpack_launches": word_launches.count(1),
                "stats": transport.receipt()["stats"]}
    finally:
        transport.close()


def gpu_check_and_benchmark(args):
    import cupy as cp
    module = transport_module
    if args.transport_source:
        spec = importlib.util.spec_from_file_location("hexcore._transport_before_check", args.transport_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    tables = _tables(args.ranks, args.owned)
    bindings = [_fields(cp, tables, binding) for binding in range(args.bindings)]
    expected = [_expected(tables, fields, cp.asnumpy) for fields, _ in bindings]
    untouched = [[[cp.asnumpy(array) for array in rank] for rank in storage]
                 for _, storage in bindings]
    transport = module.SameHostTransport(tables, [0] * args.ranks, xp=cp, timeout=30)
    streams = [cp.cuda.Stream(non_blocking=True) for _ in tables]
    samples = []
    rank_enqueue = []
    warmed_stats = None
    with ThreadPoolExecutor(max_workers=args.ranks) as pool:
        for step in range(args.rounds):
            fields, _ = bindings[step % args.bindings]
            def exchange(rank):
                with cp.cuda.Device(0), streams[rank]:
                    return transport.exchange(rank, step, fields[rank])
            mark = time.perf_counter()
            receipts = list(pool.map(exchange, range(args.ranks)))
            samples.append(1000 * (time.perf_counter() - mark))
            rank_enqueue.append([1000 * row["seconds"] for row in receipts])
            for stream in streams:
                stream.synchronize()
            if step + 1 == args.bindings:
                warmed_stats = transport.receipt()["stats"]
    transport.synchronize()
    for binding, (fields, storage) in enumerate(bindings):
        for rank in range(args.ranks):
            for slot, (_, array, _) in enumerate(fields[rank]):
                actual = cp.asnumpy(array)
                if actual.tobytes() != expected[binding][rank][slot].tobytes():
                    raise AssertionError(f"binding {binding} rank {rank} field {slot} is not byte-identical")
            for slot, selected in ((0, (..., slice(None, None, 2))),
                                   (1, (..., slice(1, None, 2))), (2, slice(None, None, 2))):
                if cp.asnumpy(storage[rank][slot])[selected].tobytes() != untouched[binding][rank][slot][selected].tobytes():
                    raise AssertionError(f"binding {binding} rank {rank} storage {slot} padding changed")
    final_receipt = transport.receipt()
    counters = final_receipt["stats"]
    checked = {}
    if "pack_launches" in counters:
        expected_launches = args.ranks * args.rounds if args.ranks > 1 else 0
        checked = {"pack_launches_exact": counters["pack_launches"] == expected_launches,
                   "unpack_launches_exact": counters["unpack_launches"] == expected_launches}
        if args.bindings <= final_receipt["descriptor_cache_size"]:
            checked["no_descriptor_uploads_after_warmup"] = (
                counters["descriptor_uploads"] == warmed_stats["descriptor_uploads"])
        if not all(checked.values()):
            raise AssertionError(f"transport counters failed: {checked!r}")
    source_path = Path(module.__file__).resolve()
    receipt = {
        "schema": "samehost-transport-byte-check/v1", "byte_identical": True,
        "padding_unchanged": True, "ranks": args.ranks, "devices": [0] * args.ranks,
        "owned_cells_per_rank": args.owned, "rounds": args.rounds,
        "field_bindings": args.bindings, "transport_source": str(source_path),
        "transport_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "runtime": {"python": sys.version, "cupy": cp.__version__,
                    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion()},
        "host_ms_cold": samples[0], "host_ms_warm_median": float(np.median(samples[args.bindings:])),
        "host_ms_all": samples, "rank_host_ms_all": rank_enqueue,
        "counter_checks": checked, "warmed_stats": warmed_stats, "transport": final_receipt,
    }
    transport.close()
    receipt["crosskind_alias_check"] = _gpu_alias_check(module, cp)
    receipt["word_layout_check"] = _gpu_word_check(module, cp)
    if args.output:
        Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--owned", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--bindings", type=int, default=1)
    parser.add_argument("--transport-source")
    parser.add_argument("--output")
    args, remaining = parser.parse_known_args()
    if args.gpu:
        if remaining or args.ranks < 1 or args.owned < 1 or args.bindings < 1 or args.rounds <= args.bindings:
            parser.error("GPU check requires positive ranks/owned/bindings, more rounds than bindings and no extra arguments")
        gpu_check_and_benchmark(args)
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
