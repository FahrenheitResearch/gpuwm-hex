"""Full-buffer acoustic fusion comparison, including padding and specified cells."""
from __future__ import annotations
import os
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore.cuda_acoustic_fusion_v841 import (
    ARGUMENT_NAMES, CUDA_FUSED_SOURCE, KERNEL_NAME, MODULE_KEY, _extract, _name,
    SHARED_SOURCE, CUDA_REGIONAL_SOURCE, launch,
)


def test_fusion_retains_original_arithmetic():
    for name, source in (("acoustic_prepare_v841", SHARED_SOURCE),
                         ("acoustic_rs_ts_regional_v841", CUDA_REGIONAL_SOURCE),
                         ("acoustic_column_solve_regional_v841", CUDA_REGIONAL_SOURCE)):
        _, body = _extract(source, name)
        for line in body.splitlines():
            if any(token in line for token in ("REGIONAL_ELEMENT_LOOP", "element %", "element /")):
                continue
            assert line in CUDA_FUSED_SOURCE
    assert len(ARGUMENT_NAMES) == len(set(ARGUMENT_NAMES))
    assert "element % ncells" not in CUDA_FUSED_SOURCE


def test_reordered_mesh_keeps_the_declared_physical_garbage_donor():
    from hexcore.cuda_regional_forecast_v841 import regional_garbage_donor
    assert regional_garbage_donor({}, 7) == 6
    assert regional_garbage_donor({'regional_garbage_donor_cell': np.int32(2)}, 7) == 2
    for bad in (-1, 7, 1.5, True):
        with pytest.raises(ValueError, match='real cell'):
            regional_garbage_donor({'regional_garbage_donor_cell':bad}, 7)


def test_constructor_and_geometry_pads_use_the_same_physical_donor():
    from hexcore.cuda_regional_forecast_v841 import pad_regional_physics_host, pad_regional_physics_column
    values=np.array([[1.,2.,3.]],dtype=np.float32)
    assert pad_regional_physics_column(values).tobytes()==np.array([[1.,2.,3.,3.]],dtype=np.float32).tobytes()
    padded,gwdo,receipt=pad_regional_physics_host(
        {'n_columns':3,'temperature':values}, {'height':values[0]}, n_cells_solve=3,donor_cell=0)
    assert padded['temperature'].tobytes()==np.array([[1.,2.,3.,1.]],dtype=np.float32).tobytes()
    assert gwdo['height'].tobytes()==np.array([1.,2.,3.,1.],dtype=np.float32).tobytes()
    assert receipt['donor_cell']==0


@pytest.mark.parametrize("small_step", [1, 2, 6])
@pytest.mark.parametrize("nlev,ncells", [(2, 7), (7, 129), (55, 257)])
@pytest.mark.gpu
def test_all_acoustic_buffers_match_on_device(small_step, nlev, ncells):
    if os.environ.get("HEXCORE_FUSION_GPU_TEST") != "1":
        pytest.skip("set HEXCORE_FUSION_GPU_TEST=1 inside a card grant")
    import cupy as cp
    from hexcore.cuda_backend.runtime import KernelCache
    cache = KernelCache()
    rng = np.random.default_rng(726)
    nedges, max_edges = 3 * ncells, 3
    interface = {"rw_p", "ww_avg", "w", "rw", "rw_save", "tend_rw", "coftz"}
    vertical = {"rdzw", "fzm", "fzp", "cofrz", "ewm", "etp", "etm", "ewp"}
    values = dict(nlev=np.int32(nlev), ncells=np.int32(ncells), nedges=np.int32(nedges),
                  max_edges=np.int32(max_edges), small_step=np.int32(small_step), dts=np.float32(0.125))
    values.update(
        n_edges_on_cell=cp.full(ncells, max_edges, dtype=cp.int32),
        edges_on_cell=cp.asarray(rng.integers(0, nedges, (ncells, max_edges), dtype=np.int32)),
        cells_on_edge=cp.asarray(rng.integers(0, ncells, (nedges, 2), dtype=np.int32)),
        spec_zone_mask_cell=cp.asarray((np.arange(ncells) % 7 == 0).astype(np.float32)),
    )
    for name in ARGUMENT_NAMES:
        if name in values:
            continue
        shape = ((nlev + 1, ncells) if name in interface else
                 (nlev + 1,) if name in vertical else
                 (ncells, max_edges) if name == "edge_sign_on_cell" else
                 (nedges,) if name == "dv_edge" else
                 (ncells,) if name == "inv_area_cell" else
                 (nlev, nedges) if name == "ru_p" else (nlev, ncells))
        values[name] = cp.asarray(rng.uniform(0.02, 0.2, shape).astype(np.float32))
    reference = {name: value.copy() if isinstance(value, cp.ndarray) else value
                 for name, value in values.items()}
    for name, source, key, count in (
        ("acoustic_prepare_v841", SHARED_SOURCE, "hexcore.cuda_acoustic_v841", ncells),
        ("acoustic_rs_ts_regional_v841", CUDA_REGIONAL_SOURCE, "hexcore.cuda_regional_v841", nlev * ncells),
        ("acoustic_column_solve_regional_v841", CUDA_REGIONAL_SOURCE, "hexcore.cuda_regional_v841", ncells)):
        declarations, _ = _extract(source, name)
        kernel = cache.raw_kernel(name, source, module_key=key)
        kernel(((count + 127) // 128,), (128,), tuple(reference[_name(d)] for d in declarations))
    launch(SimpleNamespace(kernels=SimpleNamespace(cache=cache)), values, ncells)
    cp.cuda.get_current_stream().synchronize()
    for name, value in values.items():
        if isinstance(value, cp.ndarray):
            assert value.get().tobytes() == reference[name].get().tobytes(), name
