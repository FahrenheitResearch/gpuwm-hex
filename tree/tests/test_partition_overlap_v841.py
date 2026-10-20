"""Interior storage law, real transport order, and compiled damping identity."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import os

import numpy as np
import pytest

from hexcore.partition_overlap_v841 import interior_edges, launch_selected, _CUDA_SOURCE
from hexcore.cuda_horizontal import _CUDA_SOURCE as original_source
from hexcore.partition_samehost_transport_v841 import SameHostTransport
from hexcore.partition_samehost_pipeline_v841 import StepSlotTransport
from test_partition_samehost_transport_v841 import _tables


def test_integer_classification_excludes_all_halo_and_missing_endpoints():
    cells = np.array([[0, 1], [1, 2], [2, 3], [-1, 0], [0, 1]], np.int32)
    assert interior_edges(cells, n_owned_cells=3, n_owned_edges=4).tolist() == [1, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="integer"):
        interior_edges(cells.astype(float), n_owned_cells=3, n_owned_edges=4)
    with pytest.raises(ValueError, match="extents"):
        interior_edges(cells, n_owned_cells=3, n_owned_edges=6)


def test_selected_kernel_preserves_the_original_arithmetic_body_verbatim():
    body = original_source.split('extern "C" __global__ void divergence_damping_f32(', 1)[1]
    arithmetic = body[body.index("    const int cell0 ="):]
    assert _CUDA_SOURCE[_CUDA_SOURCE.index("    const int cell0 ="):] == arithmetic


@pytest.mark.parametrize("transport_class", [SameHostTransport, StepSlotTransport])
def test_transport_packs_before_owned_callback_and_preserves_peer_payload(transport_class):
    transport = transport_class(_tables(2), (0, 0), xp=np)
    arrays = [np.array([[rank + 10, rank + 20, -1]], dtype=np.int32) for rank in range(2)]
    before = [array.copy() for array in arrays]
    if transport_class is StepSlotTransport:
        transport.begin_step(1, ("C",))
    def work(rank):
        transport.exchange(rank, "1:0:C", (("ru_p", arrays[rank], "cell"),),
                           interior_work=lambda: arrays[rank].__setitem__((slice(None), slice(0, 2)), 999))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(work, range(2)))
    if transport_class is StepSlotTransport:
        transport.end_step(1)
    for rank in range(2):
        peer = _tables(2)[rank].peers[0]
        reverse = _tables(2)[1 - rank].peers[0]
        assert np.array_equal(arrays[rank][..., peer.cell_recv], before[1 - rank][..., reverse.cell_send])
        assert arrays[rank][..., :2].tolist() == [[999, 999]]
    transport.close()


@pytest.mark.parametrize("transport_class", [SameHostTransport, StepSlotTransport])
def test_interior_callback_failure_aborts_all_waiters(transport_class):
    transport = transport_class(_tables(2), (0, 0), xp=np, timeout=1)
    if transport_class is StepSlotTransport:
        transport.begin_step(1, ("C",))
    def fail():
        raise RuntimeError("interior launch failed")
    def work(rank):
        transport.exchange(rank, "1:0:C", (("ru_p", np.zeros((1, 3), np.int32), "cell"),),
                           interior_work=fail if rank == 0 else None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, rank) for rank in range(2)]
        for future in futures:
            try:
                future.result(timeout=3)
            except RuntimeError:
                pass
    assert transport.receipt()["aborted"]
    transport.close()


@pytest.mark.gpu
def test_selected_damping_is_byte_identical_to_unsplit_compiled_kernel(tmp_path):
    from hexcore.cuda_backend import KernelCache, require_cuda
    import cupy as cp
    capability = require_cuda(min_compute=(12, 0), cache_dir=tmp_path)
    cache = KernelCache(capability=capability, cache_dir=tmp_path)
    nlev, ncells, nedges = 5, 6, 8
    rng = np.random.default_rng(841)
    cell_fields = [cp.asarray(rng.uniform(1, 3, (nlev, ncells)).astype(np.float32)) for _ in range(3)]
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [1, 5], [2, 4]], np.int32)
    mask = cp.asarray(interior_edges(cells, n_owned_cells=4, n_owned_edges=6))
    initial = rng.normal(size=(nlev, nedges)).astype(np.float32)
    initial[0, :4] = np.array([0x80000000, 0x7fc12345, 1, 0], np.uint32).view(np.float32)
    reference, split = cp.asarray(initial), cp.asarray(initial)
    args = (*cell_fields, cp.asarray(cells), cp.zeros(nedges, cp.float32), np.float32(.125),
            np.int32(ncells), np.int32(nlev), np.int32(ncells), np.int32(nedges))
    kernel = cache.raw_kernel("divergence_damping_f32", original_source, module_key="hexcore.cuda_horizontal")
    kernel((1,), (256,), (*args, reference))
    horizontal = SimpleNamespace(cp=cp, nedges=nedges, kernel_cache=cache, _kernels={})
    launch_selected(horizontal, nedges, (*args, split), (mask, 1))
    launch_selected(horizontal, nedges, (*args, split), (mask, 0))
    cp.cuda.get_current_stream().synchronize()
    assert np.array_equal(cp.asnumpy(reference).view(np.uint32), cp.asnumpy(split).view(np.uint32))


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("HEX_SAMEHOST_GPU_TEST") != "1", reason="requires eight assigned cards")
@pytest.mark.parametrize("transport_class", [SameHostTransport, StepSlotTransport])
def test_eight_card_dma_callback_and_gpu_timing_preserve_payloads(transport_class):
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 8:
        pytest.skip("needs an exclusive eight-card queue grant")
    tables = _tables(8)
    transport = transport_class(tables, range(8), xp=cp, measure_gpu=True, timeout=30)
    arrays, streams, originals = [], [], []
    for rank in range(8):
        with cp.cuda.Device(rank):
            host = np.full((2, 9), 0xfefefefe, np.uint32)
            host[:, 0] = 0x80000000 + rank
            host[:, 1] = 0x7fc12340 + rank
            arrays.append(cp.asarray(host))
            cp.cuda.get_current_stream().synchronize()
            streams.append(cp.cuda.Stream(non_blocking=True))
            originals.append(host.copy())
    if transport_class is StepSlotTransport:
        transport.begin_step(1, ("C",))
    def worker(rank):
        with cp.cuda.Device(rank), streams[rank]:
            transport.exchange(rank, "1:0:C", (("probe", arrays[rank], "cell"),),
                               interior_work=lambda: arrays[rank].__setitem__((slice(None), slice(0, 2)), 0))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    if transport_class is StepSlotTransport:
        transport.end_step(1)
    transport.synchronize()
    for rank in range(8):
        with cp.cuda.Device(rank):
            streams[rank].synchronize()
            result = cp.asnumpy(arrays[rank])
            assert np.all(result[:, :2] == 0)
            for peer in tables[rank].peers:
                reverse = next(row for row in tables[peer.peer].peers if row.peer == rank)
                assert np.array_equal(result[:, peer.cell_recv], originals[peer.peer][:, reverse.cell_send])
    receipt = transport.receipt()
    assert receipt["stats"]["completed_rounds"] == 1
    assert receipt["gpu_step_timings"]["1"]["rounds"] == 1
    assert all(value >= 0 for value in receipt["gpu_step_timings"]["1"]["wait_seconds_by_rank"])
    assert receipt["stats"]["bytes"] == 8 * 7 * 2 * 4
    transport.close()
