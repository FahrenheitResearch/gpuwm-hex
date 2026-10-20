"""Byte-layout and fail-stop checks for concurrent same-host halo rounds."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import sys
import threading
import time
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexcore.partition_samehost_transport_v841 import (  # noqa: E402
    SameHostTransport, SameHostTransportError, choose_copy_paths,
    _device_numa_nodes,
)


def _tables(ranks=3):
    """Every rank owns two entries; one halo entry comes from each peer."""
    tables = []
    for rank in range(ranks):
        peers = []
        halo_order = [peer for peer in range(ranks) if peer != rank]
        for peer in halo_order:
            peers.append(SimpleNamespace(
                peer=peer,
                cell_send=np.array([(rank + peer) % 2], dtype=np.int64),
                cell_recv=np.array([2 + halo_order.index(peer)], dtype=np.int64),
                edge_send=np.array([1 - (rank + peer) % 2], dtype=np.int64),
                edge_recv=np.array([2 + halo_order.index(peer)], dtype=np.int64),
            ))
        tables.append(SimpleNamespace(
            rank=rank, peers=tuple(peers),
            n_owned_cells=2, n_local_cells=ranks + 1,
            n_owned_edges=2, n_local_edges=ranks + 1,
        ))
    return tables


def _run_round(transport, fields, tags=None):
    tags = ["C"] * len(fields) if tags is None else tags
    with ThreadPoolExecutor(max_workers=len(fields)) as pool:
        jobs = [pool.submit(transport.exchange, rank, tags[rank], row) for rank, row in enumerate(fields)]
        return [job.result(timeout=3) for job in jobs]


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_n_rank_packets_preserve_field_order_receiver_order_and_owned_bytes(ranks):
    tables = _tables(ranks)
    transport = SameHostTransport(tables, [0] * ranks, xp=np)
    fields = []
    originals = []
    for rank in range(ranks):
        cell = (np.arange(3 * (ranks + 1), dtype=np.int32).reshape(3, ranks + 1) + 100 * rank)
        edge = (np.arange(2 * 3 * (ranks + 1), dtype=np.int16).reshape(2, 3, ranks + 1) + 100 * rank)
        fields.append((('cell', cell, 'cell'), ('edge', edge, 'edge')))
        originals.append((cell.copy(), edge.copy()))
    receipts = _run_round(transport, fields)
    for rank, table in enumerate(tables):
        for field_position, kind in enumerate(("cell", "edge")):
            array = fields[rank][field_position][1]
            assert array[..., :2].tobytes() == originals[rank][field_position][..., :2].tobytes()
            for peer in table.peers:
                reverse = next(row for row in tables[peer.peer].peers if row.peer == rank)
                received = array[..., getattr(peer, f"{kind}_recv")]
                expected = originals[peer.peer][field_position][..., getattr(reverse, f"{kind}_send")]
                assert received.tobytes() == expected.tobytes()
        assert receipts[rank]["bytes_sent"] == receipts[rank]["bytes_received"]
    assert transport.stats["rounds"] == 1
    assert transport.stats["bytes"] == sum(row["bytes_sent"] for row in receipts)
    assert transport.stats["local_bytes"] == transport.stats["bytes"]


def test_reused_packets_follow_replaced_live_arrays_and_round_generations():
    tables = _tables(3)
    transport = SameHostTransport(tables, [0] * 3, xp=np)
    for generation in range(5):
        fields = [(('a', np.full((2, 4), rank + 10 * generation, dtype=np.int64), 'cell'),) for rank in range(3)]
        _run_round(transport, fields)
        for rank, table in enumerate(tables):
            for peer in table.peers:
                assert np.all(fields[rank][0][1][..., peer.cell_recv] == peer.peer + 10 * generation)
    assert transport.stats["rounds"] == 5
    assert all(len(cache) == 1 for cache in transport._cache)


def test_float_payloads_and_negative_strides_move_without_changing_bits():
    tables = _tables(2)
    transport = SameHostTransport(tables, [0, 0], xp=np)
    # Signalling/quiet NaN payloads, negative zero and subnormal storage.
    storage = [np.array([
        [0, 0x7FA12345 + rank, 0x80000000],
        [0, 0x7FC54321 + rank, 0x00000001],
    ], dtype=np.uint32) for rank in range(2)]
    views = [array.view(np.float32)[::-1, ::-1] for array in storage]
    originals = [view.copy().view(np.uint32) for view in views]
    fields = [(('payload', view, 'cell'),) for view in views]
    _run_round(transport, fields)
    for rank, view in enumerate(views):
        peer = tables[rank].peers[0]
        reverse = tables[1 - rank].peers[0]
        assert np.array_equal(view.view(np.uint32)[..., :2], originals[rank][..., :2])
        assert np.array_equal(view.view(np.uint32)[..., peer.cell_recv], originals[1 - rank][..., reverse.cell_send])


@pytest.mark.parametrize("mismatch", ["tag", "name", "kind", "dtype", "leading_shape"])
def test_schema_or_tag_fault_stops_every_rank_and_future_rounds(mismatch):
    transport = SameHostTransport(_tables(2), [0, 0], xp=np, timeout=0.3)
    left = ('a', np.zeros((2, 3), dtype=np.float32), 'cell')
    right = ('a', np.zeros((2, 3), dtype=np.float32), 'cell')
    tags = ["C", "C"]
    if mismatch == "tag":
        tags[1] = "B"
    elif mismatch == "name":
        right = ('different', right[1], 'cell')
    elif mismatch == "kind":
        right = ('a', right[1], 'edge')
    elif mismatch == "dtype":
        right = ('a', np.zeros((2, 3), dtype=np.float64), 'cell')
    else:
        right = ('a', np.zeros((3, 3), dtype=np.float32), 'cell')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(transport.exchange, rank, tags[rank], (row,)) for rank, row in enumerate((left, right))]
        for job in jobs:
            with pytest.raises(SameHostTransportError, match="schema|tag"):
                job.result(timeout=1)
    with pytest.raises(SameHostTransportError):
        transport.exchange(0, "C", (left,))
    assert transport.stats["rounds"] == 0


def test_missing_rank_times_out_and_joins_without_hanging():
    transport = SameHostTransport(_tables(3), [0] * 3, xp=np, timeout=0.05)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(transport.exchange, rank, "C", (('a', np.zeros(4), 'cell'),)) for rank in range(2)]
        for job in jobs:
            with pytest.raises(SameHostTransportError, match="timed out"):
                job.result(timeout=1)
    assert transport.stats["rounds"] == 0
    assert len(transport._failed_rounds) == 1


def test_explicit_abort_wakes_a_rank_waiting_for_peers():
    transport = SameHostTransport(_tables(2), [0, 0], xp=np, timeout=10)
    ready = threading.Event()
    original = transport._buffers

    def mark_ready(*args):
        buffers = original(*args)
        ready.set()
        return buffers

    transport._buffers = mark_ready
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(transport.exchange, 0, "C", (('a', np.zeros(3), 'cell'),))
        assert ready.wait(timeout=1)
        transport.abort("another rank failed its step")
        with pytest.raises(SameHostTransportError, match="another rank failed"):
            job.result(timeout=1)


def test_invalid_field_aborts_waiting_peer_before_any_halo_is_written():
    transport = SameHostTransport(_tables(2), [0, 0], xp=np, timeout=0.3)
    left = np.full((2, 3), 17, dtype=np.float32)
    right = np.full((2, 4), 23, dtype=np.float32)
    originals = (left.copy(), right.copy())
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(transport.exchange, rank, "C", (('a', array, 'cell'),)) for rank, array in enumerate((left, right))]
        for job in jobs:
            with pytest.raises(SameHostTransportError, match="last axis"):
                job.result(timeout=1)
    assert np.array_equal(left, originals[0])
    assert np.array_equal(right, originals[1])


def test_duplicate_rank_entry_stops_the_round():
    transport = SameHostTransport(_tables(2), [0, 0], xp=np, timeout=0.3)
    fields = (('a', np.zeros(3), 'cell'),)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(transport.exchange, 0, "C", fields) for _ in range(2)]
        for job in jobs:
            with pytest.raises(SameHostTransportError, match="entered twice"):
                job.result(timeout=1)


def test_receive_failure_wakes_other_ranks_and_does_not_complete_generation():
    transport = SameHostTransport(_tables(3), [0] * 3, xp=np, timeout=0.3)
    original = transport._receive

    def fail_one(rank, round_, submission):
        if rank == 1:
            raise RuntimeError("copy failed")
        original(rank, round_, submission)

    transport._receive = fail_one
    fields = [(('a', np.zeros(4), 'cell'),)] * 3
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(transport.exchange, rank, "C", fields[rank]) for rank in range(3)]
        for job in jobs:
            with pytest.raises(SameHostTransportError, match="copy failed"):
                job.result(timeout=1)
    assert transport.stats["rounds"] == 0
    assert len(transport._failed_rounds[0].submissions) == 3


@pytest.mark.parametrize("defect", ["owned_receive", "halo_send", "missing_halo", "packet_count", "dtype"])
def test_invalid_tables_refuse_structural_miscopies(defect):
    tables = _tables(2)
    peer = tables[0].peers[0]
    if defect == "owned_receive":
        peer.cell_recv[0] = 0
    elif defect == "halo_send":
        peer.cell_send[0] = 2
    elif defect == "missing_halo":
        peer.cell_recv = np.array([], dtype=np.int64)
    elif defect == "packet_count":
        peer.cell_send = np.array([0, 1], dtype=np.int64)
    else:
        peer.cell_send = np.array([0], dtype=np.int32)
    with pytest.raises(SameHostTransportError):
        SameHostTransport(tables, [0, 0], xp=np)


def test_route_selection_distinguishes_same_device_peer_staging_and_numa():
    peers = {(0, 1): True, (1, 0): True}
    assert choose_copy_paths([0, 0], {}) == {(0, 0): "local"}
    assert choose_copy_paths([0, 1], peers)[0, 1] == "peer"
    assert choose_copy_paths([0, 1], {(0, 1): True})[0, 1] == "staged"
    assert choose_copy_paths([0, 1], peers, "staged")[0, 1] == "staged"
    assert choose_copy_paths([0, 1], peers, numa_nodes={0: 0, 1: 1})[0, 1] == "staged"
    assert choose_copy_paths([0, 1], peers, "peer", numa_nodes={0: 0, 1: 1})[0, 1] == "peer"
    with pytest.raises(SameHostTransportError, match="bidirectional"):
        choose_copy_paths([0, 1], {}, "peer")


def test_one_card_guard_explicitly_exercises_host_staging_without_changing_default():
    ordinary = SameHostTransport(_tables(2), [0, 0], xp=np)
    guard = SameHostTransport(_tables(2), [0, 0], xp=np, copy_policy="staged", stage_same_device=True)
    assert ordinary.paths == {(0, 0): "local"}
    assert guard.paths == {(0, 0): "staged"}
    assert guard.receipt()["stage_same_device"] is True
    with pytest.raises(SameHostTransportError, match="explicit staged"):
        SameHostTransport(_tables(2), [0, 0], xp=np, stage_same_device=True)


def test_measured_card_locality_selects_staging_for_mixed_sockets(tmp_path, monkeypatch):
    buses = {0: "0000:01:00.0", 1: "0000:81:00.0"}
    files = {str(tmp_path / bus / "numa_node"): str(device) for device, bus in buses.items()}
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs: files[str(path)])
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: SimpleNamespace(pci_bus_id=buses[device])))
    nodes = _device_numa_nodes(xp, [0, 1, 0], tmp_path)
    peers = {(0, 1): True, (1, 0): True}
    assert nodes == {0: 0, 1: 1}
    assert choose_copy_paths([0, 1], peers, numa_nodes=nodes)[0, 1] == "staged"
    assert choose_copy_paths([0, 1], peers, "peer", numa_nodes=nodes)[0, 1] == "peer"


def test_unknown_card_locality_is_not_reported_as_a_socket(tmp_path, monkeypatch):
    reads = []
    def read(path, *args, **kwargs):
        reads.append(str(path))
        return "-1"
    monkeypatch.setattr(Path, "read_text", read)
    xp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: SimpleNamespace(
        pci_bus_id="0000:01:00.0" if device == 0 else "../../unexpected")))
    assert _device_numa_nodes(xp, [0, 1], tmp_path) == {0: None, 1: None}
    assert reads == [str(tmp_path / "0000:01:00.0" / "numa_node")]


def test_byte_copy_kernel_uses_no_floating_point_storage_or_arithmetic():
    from hexcore.partition_samehost_transport_v841 import _BYTE_COPY

    assert "float" not in _BYTE_COPY
    assert "double" not in _BYTE_COPY
    assert "unsigned char" in _BYTE_COPY
    assert "const long long width" in _BYTE_COPY


def _queued_cuda_transport(path="local", timeout=1):
    """Temporal API double: DMA stays unfinished until its gate is released.

    This checks host ownership and dependency issuance. Real CUDA byte
    identity is checked by the opt-in tests below, not by this double.
    """
    from hexcore.partition_samehost_transport_v841 import _Packet

    gate = threading.Event()
    current = threading.local()
    queries = []
    events = []

    class Device:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            self.previous = getattr(current, "device", 0)
            current.device = self.device

        def __exit__(self, *_):
            current.device = self.previous

        def use(self):
            current.device = self.device

    class Stream:
        def __init__(self):
            self.log = []
            self.ptr = id(self)

        def __enter__(self):
            self.previous = getattr(current, "stream", None)
            current.stream = self

        def __exit__(self, *_):
            current.stream = self.previous

        def wait_event(self, event):
            self.log.append(("wait", event))

    class Event:
        def __init__(self, **kwargs):
            self.recorded = False
            events.append(self)

        def record(self, stream):
            assert not self.recorded, "a generation event must never be re-recorded"
            self.recorded = True
            stream.log.append(("record", self))

        def synchronize(self):
            gate.wait()

        @property
        def done(self):
            queries.append(self)
            return gate.is_set()

    def copy(*args):
        current.stream.log.append(("copy", args[-2]))

    devices = [0, 0] if path == "local" else [0, 1]
    transport = SameHostTransport(_tables(2), devices, xp=np, timeout=timeout)
    transport._cpu = False
    transport.xp = SimpleNamespace(cuda=SimpleNamespace(
        Device=Device, Event=Event,
        runtime=SimpleNamespace(memcpyAsync=copy, memcpyPeerAsync=copy,
            memcpyDeviceToDevice=1, memcpyDeviceToHost=2, memcpyHostToDevice=3),
    ))
    transport.paths = {(src, dst): path for src in devices for dst in devices}
    transport._copy_streams = [Stream(), Stream()]
    streams = [Stream(), Stream()]
    packets = []
    descriptors = []

    class Descriptor:
        pass

    def copy_fields(fields, packet, *, scatter, stream=None):
        packet.descriptor = Descriptor()
        descriptors.append(weakref.ref(packet.descriptor))
        stream.log.append(("scatter" if scatter else "pack", packet.descriptor))
        # The stream log models raw pointers, not allocation ownership.
        stream.log[-1] = (stream.log[-1][0], id(packet.descriptor))

    transport._copy_fields = copy_fields
    def copy_packets(fields, packets, batch, *, scatter, stream=None):
        packets = tuple(packets)
        for packet in packets:
            copy_fields(fields, packet, scatter=scatter, stream=stream)
        batch.owners = tuple(packet.descriptor for packet in packets)
    transport._copy_packets = copy_packets
    for rank in range(2):
        source = _Packet(SimpleNamespace(data=SimpleNamespace(ptr=10 + rank), nbytes=4), (), ())
        source.source_stream = Stream()
        source.host = SimpleNamespace(ptr=30 + rank)
        receive = _Packet(SimpleNamespace(data=SimpleNamespace(ptr=20 + rank), nbytes=4), (), ())
        packets.append((source, receive))
    if path == "staged":
        transport._staging_streams = {device: packets[rank][0].source_stream
                                      for rank, device in enumerate(devices)}
    return SimpleNamespace(transport=transport, gate=gate, queries=queries,
        events=events, streams=streams, packets=packets, descriptors=descriptors)


def _queue_temporal_round(model, generation):
    from hexcore.partition_samehost_transport_v841 import _Buffers, _Round, _Submission

    transport = model.transport
    round_ = _Round(generation, time.monotonic() + transport.timeout, tag=f"C{generation}")
    field_refs = []
    for rank in range(2):
        array = np.zeros(3, dtype=np.float32)
        field_refs.append(weakref.ref(array))
        send, receive = model.packets[rank]
        submission = _Submission((('a', array, 'cell'),), _Buffers({1 - rank: send}, {1 - rank: receive}), model.streams[rank])
        round_.producer_streams[rank] = submission.stream
        transport._copy_fields(submission.fields, send, scatter=False, stream=submission.stream)
        transport._retain_descriptors(submission, (send,))
        submission.produced = transport.xp.cuda.Event()
        submission.produced.record(submission.stream)
        round_.submissions[rank] = submission
    for rank, submission in round_.submissions.items():
        transport._receive(rank, round_, submission)
    for rank, submission in round_.submissions.items():
        transport._fence_consumers(rank, round_, submission)
    with transport._condition:
        transport._queue_completion(round_)
    return field_refs, tuple(submission.received for submission in round_.submissions.values())


@pytest.mark.parametrize("path", ["local", "peer", "staged"])
def test_cuda_rounds_queue_fences_without_host_event_queries_and_retain_owners(path):
    model = _queued_cuda_transport(path)
    try:
        refs1, events1 = _queue_temporal_round(model, 0)
        descriptors1 = tuple(model.descriptors)
        refs2, events2 = _queue_temporal_round(model, 1)
        assert not model.queries
        assert all(event not in events2 for event in events1)
        assert all(reference() is not None for reference in (*refs1, *refs2, *descriptors1))
        for stream in model.streams:
            waited = [entry[1] for entry in stream.log if entry[0] == "wait"]
            assert all(event in waited for event in (*events1, *events2))
        if path == "staged":
            for stream in model.transport._copy_streams:
                h2d = [index for index, entry in enumerate(stream.log) if entry == ("copy", 3)]
                assert h2d
                assert all(stream.log[index - 1][0] == "wait" for index in h2d)
            assert all(("copy", 2) in source.source_stream.log for source, _ in model.packets)
    finally:
        model.gate.set()
        model.transport.close(timeout=1)
    assert model.transport.receipt()["pending_rounds"] == 0
    assert model.transport._completion_worker is None or not model.transport._completion_worker.is_alive()


def test_cuda_completion_watchdog_aborts_and_keeps_owners_until_drain():
    model = _queued_cuda_transport(timeout=0.03)
    try:
        fields, _ = _queue_temporal_round(model, 0)
        with model.transport._condition:
            assert model.transport._condition.wait_for(lambda: model.transport._error is not None, timeout=1)
        assert "CUDA completion" in str(model.transport._error)
        assert not model.queries
        assert all(reference() is not None for reference in fields)
        with pytest.raises(SameHostTransportError, match="drain timed out"):
            model.transport.synchronize(timeout=0.01)
        assert all(reference() is not None for reference in fields)
    finally:
        model.gate.set()
        model.transport.close(timeout=1)


@pytest.mark.skipif(os.environ.get("HEX_SAMEHOST_GPU_TEST") != "1", reason="requires an explicitly assigned CUDA card")
@pytest.mark.parametrize("ranks", [2, 4])
def test_gpu_samecard_rank_packets_fence_producers_and_preserve_strided_payloads(ranks):
    cp = pytest.importorskip("cupy")
    tables = _tables(ranks)
    transport = SameHostTransport(tables, [0] * ranks, xp=cp, timeout=20)
    fields, updates, expected = [], [], []
    for rank in range(ranks):
        bits = np.array([
            [0x80000000, 0x7FA12345 + rank] + [0] * (ranks - 1),
            [0x00000001, 0x7FC54321 + rank] + [0] * (ranks - 1),
        ], dtype=np.uint32)
        # Two noncontiguous axes exercise the descriptor's byte strides.
        backing = np.zeros((4, 2 * (ranks + 1)), dtype=np.uint32)
        backing[::2, ::2] = bits
        cell = cp.asarray(backing).view(cp.float32)[::2, ::2][::-1]
        cell[..., :2] = 0
        edge = cp.asarray(np.arange(2 * (ranks + 1), dtype=np.int64).reshape(2, ranks + 1) + 100 * rank)
        fields.append((('payload', cell, 'cell'), ('edge', edge, 'edge')))
        update = cp.asarray(bits[::-1, :2]).view(cp.float32)
        updates.append(update)
        expected.append((bits[::-1].copy(), cp.asnumpy(edge)))
    cp.cuda.Stream.null.synchronize()

    def worker(rank, generation):
        with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
            # The copy is queued immediately before exchange on the rank's
            # compute stream.  Transfer streams must wait for its producer.
            fields[rank][0][1][..., :2] = updates[rank]
            return transport.exchange(rank, ('C', generation), fields[rank])

    for generation in range(3):
        with ThreadPoolExecutor(max_workers=ranks) as pool:
            jobs = [pool.submit(worker, rank, generation) for rank in range(ranks)]
            for job in jobs:
                job.result(timeout=30)
        transport.synchronize()
        for rank, table in enumerate(tables):
            for position, kind in enumerate(("cell", "edge")):
                actual = cp.asnumpy(fields[rank][position][1])
                if kind == "cell":
                    actual = actual.view(np.uint32)
                assert actual[..., :2].tobytes() == expected[rank][position][..., :2].tobytes()
                for peer in table.peers:
                    reverse = next(row for row in tables[peer.peer].peers if row.peer == rank)
                    assert actual[..., getattr(peer, f"{kind}_recv")].tobytes() == expected[peer.peer][position][..., getattr(reverse, f"{kind}_send")].tobytes()
    transport.close()
    assert transport.receipt()["stats"]["rounds"] == 3
    assert transport.receipt()["stats"]["local_bytes"] > 0


@pytest.mark.skipif(os.environ.get("HEX_SAMEHOST_GPU_TEST") != "1", reason="requires explicitly assigned CUDA cards")
@pytest.mark.parametrize("copy_policy", ["auto", "staged", "peer"])
def test_gpu_crosscard_packets_use_reported_route(copy_policy):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("cross-card transport requires two assigned cards")
    if copy_policy == "peer" and not all(
        cp.cuda.runtime.deviceCanAccessPeer(src, dst) for src, dst in ((0, 1), (1, 0))
    ):
        pytest.skip("cards lack bidirectional peer access")
    tables = _tables(2)
    transport = SameHostTransport(tables, [0, 1], xp=cp, timeout=20, copy_policy=copy_policy)
    fields = []
    for rank in range(2):
        with cp.cuda.Device(rank):
            fields.append((('a', cp.asarray(np.array([10 * rank, 10 * rank + 1, -1], np.int32)), 'cell'),))
            cp.cuda.Stream.null.synchronize()

    def worker(rank):
        with cp.cuda.Device(rank), cp.cuda.Stream(non_blocking=True):
            return transport.exchange(rank, "C", fields[rank])

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, rank) for rank in range(2)]
        for job in jobs:
            job.result(timeout=30)
    transport.synchronize()
    for rank in range(2):
        with cp.cuda.Device(rank):
            assert cp.asnumpy(fields[rank][0][1]).tolist() == [10 * rank, 10 * rank + 1, 10 * (1 - rank) + 1]
    transport.close()
    expected_path = "peer" if copy_policy == "peer" else "staged" if copy_policy == "staged" else transport.paths[0, 1]
    assert transport.stats[f"{expected_path}_bytes"] == 8


# --- Field ownership after the fence and at retirement (io-halo memory lane, 2026-10-15) ---
# Breakage these prevent: the schedule-slot transport kept every exchanged field
# (each acoustic substep's state, each recovery candidate) alive until end_step;
# on x1.40962 a one-rank same-host run held 2.3 GB more than the same forecast
# with the references dropped (io-halo lane, hex mem/io-halo).


def _owned_round(transport, ranks=2):
    from hexcore.partition_samehost_transport_v841 import _Buffers, _Round, _Submission

    round_ = _Round(0, time.monotonic() + 5.0, tag="C")
    refs = []
    for rank in range(ranks):
        array = np.zeros(transport.tables[rank].n_local_cells, dtype=np.float32)
        refs.append(weakref.ref(array))
        fields = (("a", array, "cell"),)
        round_.reserved_fields.append(fields)
        round_.submissions[rank] = _Submission(fields, _Buffers({}, {}))
        del array, fields
    return round_, refs


def test_release_after_fence_forgets_only_that_ranks_fields():
    transport = SameHostTransport(_tables(2), [0, 0], xp=np)
    round_, refs = _owned_round(transport)
    submission = round_.submissions[0]
    transport._release_rank_fields(round_, submission, submission.fields)
    assert submission.fields == ()
    assert refs[0]() is None, "rank 0's field must not outlive its fence in the transport"
    assert refs[1]() is not None, "another rank's fields stay owned until its own fence or retirement"
    assert len(round_.reserved_fields) == 1


@pytest.mark.parametrize("on_retire", [True, False])
def test_retired_round_drops_fields_only_under_the_retire_policy(on_retire):
    transport = SameHostTransport(_tables(2), [0, 0], xp=np, timeout=5)
    transport.release_fields_on_retire = on_retire
    # Completion events are modelled as already complete; the schedule-slot
    # transport keeps the round object itself until end_step, as held here.
    transport._wait_completion_events = lambda round_: None
    round_, refs = _owned_round(transport)
    with transport._condition:
        transport._pending.append(round_)
        transport._start_completion_workers()
        transport._condition.notify_all()
        assert transport._condition.wait_for(lambda: transport.stats["completed_rounds"] == 1, timeout=5)
    alive = [ref() is not None for ref in refs]
    assert alive == [not on_retire] * 2
    with transport._condition:
        transport._stopping = True
        transport._condition.notify_all()
    transport._completion_worker.join(5)


def test_class_defaults_keep_the_documented_ownership_contract():
    # A transport built without __init__ (test doubles, subclasses) keeps the
    # fields through DMA completion and never drops them at a fence.
    assert SameHostTransport.release_fields_after_fence is False
    assert SameHostTransport.release_fields_on_retire is True


@pytest.mark.skipif(os.environ.get("HEX_SAMEHOST_GPU_TEST") != "1", reason="requires an explicitly assigned CUDA card")
@pytest.mark.parametrize("after_fence", [True, False])
def test_gpu_slot_transport_releases_fields_at_the_fence_and_moves_the_same_bytes(after_fence):
    # after_fence=False is the GPUWM_HEX_HALO_FIELD_RELEASE=step policy: fields
    # held to end_step (retirement would otherwise drop them as they complete).
    cp = pytest.importorskip("cupy")
    from hexcore.partition_samehost_pipeline_v841 import StepSlotTransport

    ranks = 2
    tables = _tables(ranks)
    transport = StepSlotTransport(tables, [0] * ranks, xp=cp, timeout=20)
    transport.release_fields_after_fence = after_fence
    transport.release_fields_on_retire = after_fence
    streams = []
    with cp.cuda.Device(0):
        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(ranks)]
    held = []

    def worker(rank, step):
        with cp.cuda.Device(0), streams[rank]:
            # Allocated on the rank's own compute stream, as the same-host
            # driver builds and steps every rank array.
            field = cp.asarray(np.arange(ranks + 1, dtype=np.int32) + 100 * rank + 1000 * step)
            transport.exchange(rank, f"{step}:0:C", (("a", field, "cell"),))
            return field

    for step in range(1, 4):
        transport.begin_step(step, ("C",))
        with ThreadPoolExecutor(max_workers=ranks) as pool:
            fields = list(pool.map(lambda rank: worker(rank, step), range(ranks)))
        round_ = transport._slot_rounds[0]
        owned = [len(submission.fields) for submission in round_.submissions.values()]
        assert owned == ([0] * ranks if after_fence else [1] * ranks)
        assert (len(round_.reserved_fields) == 0) == after_fence
        transport.end_step(step)
        for rank, table in enumerate(tables):
            values = cp.asnumpy(fields[rank]).tolist()
            for peer in table.peers:
                reverse = next(row for row in tables[peer.peer].peers if row.peer == rank)
                sent = (np.arange(ranks + 1, dtype=np.int32) + 100 * peer.peer + 1000 * step)[reverse.cell_send]
                assert [values[i] for i in peer.cell_recv] == sent.tolist()
        held.append(fields)
    transport.close()
