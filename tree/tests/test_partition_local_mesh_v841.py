"""Byte gathers for regional partition meshes and native padded host slots."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hexcore.mesh import Mesh
from hexcore.partition_local_mesh_v841 import build_local_mesh, slice_prepared_host
from hexcore.cuda_regional_forecast_v841 import _PaddedState, _PaddedSaved, _PaddedReference, _PaddedTerrain
from test_partition_assets_v841 import regional_case


def _mesh(coe, cov):
    coc = np.array([[cell - 1 if cell else -1, cell + 1 if cell < 11 else -1] for cell in range(12)], dtype=np.int32)
    arrays = {
        "cellsOnCell": coc, "cellsOnEdge": coe, "cellsOnVertex": cov,
        "nEdgesOnCell": np.full(12, 2, dtype=np.int32),
        "bdyMaskCell": np.array([7] + [0] * 10 + [7], dtype=np.int32),
        "bdyMaskEdge": np.full(14, 7, dtype=np.int32),
        "bdyMaskVertex": np.full(16, 7, dtype=np.int32),
        "areaCell": np.arange(12, dtype=np.float64),
        "dcEdge": np.arange(14, dtype=np.float64),
        "areaTriangle": np.arange(16, dtype=np.float64),
    }
    dims = {name: ("nCells", "width") for name in ("cellsOnCell",)}
    dims.update({"cellsOnEdge": ("nEdges", "two"), "cellsOnVertex": ("nVertices", "three")})
    for name, dim in (("nEdgesOnCell", "nCells"), ("bdyMaskCell", "nCells"), ("bdyMaskEdge", "nEdges"), ("bdyMaskVertex", "nVertices"), ("areaCell", "nCells"), ("dcEdge", "nEdges"), ("areaTriangle", "nVertices")):
        dims[name] = (dim,)
    return Mesh(arrays=arrays, dimensions={"nCells": 12, "nEdges": 14, "nVertices": 16}, variable_dimensions=dims, attrs={"registry_row": "synthetic"})


def test_unpadded_regional_mesh_keeps_sentinels_and_source_provenance(tmp_path):
    layouts, coe, cov, _, _ = regional_case(tmp_path)
    mesh = _mesh(coe, cov)
    for layout in layouts:
        local = build_local_mesh(mesh, layout)
        assert local.partition_source_mesh is mesh
        assert local.partition_layout_v841 is layout
        assert local.is_regional
        assert local.areaCell.tobytes() == mesh.areaCell[layout.cell_l2g].tobytes()
        assert np.any(local.cellsOnEdge < 0) if layout.partition != 1 else True


def test_padded_mesh_gathers_native_tail_and_remaps_garbage_ids(tmp_path):
    layouts, coe, cov, _, _ = regional_case(tmp_path)
    mesh = _mesh(coe, cov)
    arrays = {}
    for name, array in mesh.arrays.items():
        dim = mesh.variable_dimensions[name][0]
        count = mesh.dimensions[dim]
        if name in ("cellsOnCell", "cellsOnEdge", "cellsOnVertex"):
            remapped = np.where(array < 0, 12, array)
            tail = np.full((1, array.shape[1]), 12, dtype=array.dtype)
            arrays[name] = np.concatenate((remapped, tail), axis=0)
        else:
            tail = np.array([0x8000000000000000], dtype=np.uint64).view(np.float64) if array.dtype == np.float64 else np.array([0], dtype=array.dtype)
            arrays[name] = np.concatenate((array, tail), axis=0)
        assert arrays[name].shape[0] == count + 1
    arrays["cellsOnEdgeSentinel"] = np.concatenate((coe, np.full((1, 2), 12, dtype=coe.dtype)))
    padded = SimpleNamespace(base=mesh, arrays=arrays, dimensions={"nCells": 13, "nEdges": 15, "nVertices": 17}, garbage_cell=12, garbage_edge=14, garbage_vertex=16, attrs={})
    for layout in layouts:
        local = build_local_mesh(padded, layout)
        assert local.partition_source_mesh is mesh
        assert local.partition_layout_v841 is layout
        assert local.n_cells == layout.n_local_cells + 1
        assert local.garbage_cell == layout.n_local_cells
        assert local.areaCell[-1:].tobytes() == arrays["areaCell"][-1:].tobytes()
        assert np.all(local.cellsOnEdge[-1] == local.garbage_cell)
        assert np.all(local.cellsOnEdgeSentinel[-1] == local.garbage_cell)
        assert local.cellsOnEdgeSentinel.dtype == coe.dtype


def test_native_slotted_padded_objects_preserve_the_original_garbage_columns(tmp_path):
    layouts, _, _, _, _ = regional_case(tmp_path)
    state = SimpleNamespace(time_seconds=120.0)
    for name, count in (("rho", 12), ("rho_theta", 12), ("rho_u", 14), ("rho_w", 12)):
        setattr(state, name, np.arange(2 * count, dtype=np.float32).reshape(2, count))
    state.scalars = np.arange(6 * 2 * 12, dtype=np.float32).reshape(6, 2, 12)
    saved = SimpleNamespace()
    for name in _PaddedSaved.__slots__:
        count = 14 if name == "normal_velocity" else 12
        setattr(saved, name, np.arange(2 * count, dtype=np.float32).reshape(2, count))
    reference = SimpleNamespace(**{name: np.arange(24, dtype=np.float32).reshape(2, 12) for name in _PaddedReference.__slots__})
    terrain = SimpleNamespace(zb_cell=np.arange(72, dtype=np.float32).reshape(2, 12, 3), zb3_cell=np.arange(72, dtype=np.float32).reshape(2, 12, 3))
    originals = {"state": _PaddedState(state), "saved": _PaddedSaved(saved), "reference": _PaddedReference(reference), "terrain": _PaddedTerrain(terrain)}
    layout = layouts[0]
    local = slice_prepared_host(originals, layout)
    for group in ("state", "saved", "reference"):
        for name in type(originals[group]).__slots__:
            value = getattr(originals[group], name)
            if not isinstance(value, np.ndarray):
                assert getattr(local[group], name) == value
                continue
            index = layout.edge_l2g if name in ("rho_u", "normal_velocity") else layout.cell_l2g
            expected = np.take(value, np.concatenate((index, [value.shape[-1] - 1])), axis=-1)
            assert getattr(local[group], name).tobytes() == expected.tobytes()
            assert getattr(local[group], name)[..., -1:].tobytes() == value[..., -1:].tobytes()
    for name in ("zb_cell", "zb3_cell"):
        source = getattr(originals["terrain"], name)
        expected = source[:, np.concatenate((layout.cell_l2g, [12])), :]
        assert getattr(local["terrain"], name).tobytes() == expected.tobytes()


def test_padded_advection_gathers_aligned_rows_and_global_garbage_cell(tmp_path):
    layouts, _, _, _, _ = regional_case(tmp_path)
    layout = layouts[0]
    source = SimpleNamespace(
        adv_cells_for_edge=np.full((15, 3), 12, dtype=np.int32),
        adv_coefs=np.arange(45, dtype=np.float64).reshape(15, 3),
        n_adv_cells_for_edge=np.zeros(15, dtype=np.int32),
    )
    local = slice_prepared_host(source, layout)
    assert np.all(local.adv_cells_for_edge == layout.n_local_cells)
    assert local.adv_coefs.tobytes() == source.adv_coefs[np.concatenate((layout.edge_l2g, [14]))].tobytes()
    assert local.adv_coefs[-1].tobytes() == source.adv_coefs[-1].tobytes()


def test_parallel_rank_slices_preserve_full_words_source_and_thread_reports(tmp_path):
    from threading import Barrier
    from hexcore.partition_samehost_v841 import SameHostRanks
    from hexcore.partition_local_mesh_v841 import prepared_slice_report
    layouts, coe, cov, _, _ = regional_case(tmp_path)
    mesh = _mesh(coe, cov)
    values = np.arange(5*12, dtype=np.float32).reshape(5,12)
    original = values.tobytes()
    barrier = Barrier(len(layouts))
    def collect(rank, wait=False):
        source = {f'rank{rank}': {'mesh':mesh, 'rho':values}}
        local = slice_prepared_host(source, layouts[rank])
        if wait: barrier.wait(timeout=10)
        report = prepared_slice_report()
        return (local[f'rank{rank}']['rho'].tobytes(),
                local[f'rank{rank}']['mesh'].areaCell.tobytes(), report)
    serial = [collect(rank) for rank in range(len(layouts))]
    with SameHostRanks(tuple(range(len(layouts))),None) as workers:
        parallel = workers.call(lambda rank:collect(rank,True))
    assert parallel == serial
    assert values.tobytes() == original
    for rank, result in enumerate(parallel):
        assert all(row['path'].startswith(f'rank{rank}.') for row in result[2])