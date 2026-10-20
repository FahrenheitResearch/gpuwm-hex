"""Regional sentinel ownership and completeness checks."""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.partition_assets_v841 import assert_exact_cover, build_partition_layouts, load_layouts_npz, save_layouts_npz


def regional_case(tmp_path, n_ranks=3, halo_rings=2):
    n_cells = 12
    graph = tmp_path / "regional.graph.info"
    graph.write_text("\n".join(["12 11"] + [" ".join(str(neighbor + 1) for neighbor in (cell - 1, cell + 1) if 0 <= neighbor < n_cells) for cell in range(n_cells)]) + "\n")
    part = tmp_path / f"regional.graph.info.part.{n_ranks}"
    part.write_text("\n".join(str(cell * n_ranks // n_cells) for cell in range(n_cells)) + "\n")
    coe = np.array([[-1, 0]] + [[cell, cell + 1] for cell in range(11)] + [[11, -1], [-1, 0]], dtype=np.int32)
    cov = np.array([[cell, -1, -1] for cell in range(12)] + [[0, 1, -1], [3, 4, -1], [7, 8, -1], [11, -1, -1]], dtype=np.int32)
    layouts = build_partition_layouts(graph, part, halo_rings=halo_rings, cells_on_edge=coe, cells_on_vertex=cov)
    return layouts, coe, cov, graph, part


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4])
def test_regional_missing_cells_do_not_change_exact_ownership(tmp_path, n_ranks):
    layouts, coe, cov, _, _ = regional_case(tmp_path, n_ranks)
    assert assert_exact_cover(layouts, n_cells=12, n_edges=14, n_vertices=16) == {"cells": 12, "edges": 14, "vertices": 16}
    part = np.arange(12, dtype=np.int32) * n_ranks // 12
    for kind, incidence in (("edge", coe), ("vertex", cov)):
        owner = part[np.where(incidence < 0, 12, incidence).min(axis=1)]
        for layout in layouts:
            owned = getattr(layout, f"{kind}_l2g")[:getattr(layout, f"n_owned_{'vertices' if kind == 'vertex' else 'edges'}")]
            np.testing.assert_array_equal(owned, np.flatnonzero(owner == layout.partition))
    # A missing slot must not index the final real cell's partition.
    assert 0 in layouts[0].edge_l2g[:layouts[0].n_owned_edges]
    assert 13 in layouts[0].edge_l2g[:layouts[0].n_owned_edges]


def test_boundary_missing_incidence_is_not_an_incomplete_reference(tmp_path):
    layouts, _, _, _, _ = regional_case(tmp_path, 3)
    first, last = layouts[0], layouts[-1]
    assert first.edge_row_complete[np.flatnonzero(first.edge_l2g == 0)[0]]
    assert last.edge_row_complete[np.flatnonzero(last.edge_l2g == 12)[0]]
    assert first.vertex_row_complete[np.flatnonzero(first.vertex_l2g == 0)[0]]


@pytest.mark.parametrize("kind, bad", [("edge", -2), ("edge", 12), ("vertex", -2), ("vertex", 12)])
def test_only_minus_one_is_a_missing_cell_sentinel(tmp_path, kind, bad):
    _, coe, cov, graph, part = regional_case(tmp_path)
    (coe if kind == "edge" else cov)[0, 0] = bad
    with pytest.raises(ValueError, match="outside the graph or -1 sentinel"):
        build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov)


@pytest.mark.parametrize("kind", ["edge", "vertex"])
def test_entity_without_any_cell_cannot_have_an_owner(tmp_path, kind):
    _, coe, cov, graph, part = regional_case(tmp_path)
    (coe if kind == "edge" else cov)[0] = -1
    with pytest.raises(ValueError, match="ownership is undefined"):
        build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov)


def test_replicated_seed_cone_preserves_owned_prefix_and_exact_global_cover(tmp_path):
    ordinary, coe, cov, graph, part = regional_case(tmp_path, 3, 2)
    layouts = build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov, replicated_cells=[11, 11])
    assert layouts[0].cell_l2g.tolist() == [0, 1, 2, 3, 11, 4, 10, 5, 9]
    assert layouts[0].cell_ring.tolist() == [0, 0, 0, 0, 0, 1, 1, 2, 2]
    assert layouts[0].n_owned_cells == 4
    for original, seeded in zip(ordinary, layouts):
        assert seeded._replicated_cells == (11,)
        np.testing.assert_array_equal(seeded.cell_l2g[:seeded.n_owned_cells], original.cell_l2g[:original.n_owned_cells])
        assert np.unique(seeded.cell_l2g).size == seeded.n_local_cells
        assert {9, 10, 11} <= set(seeded.cell_l2g)
        donor = int(seeded.cell_g2l()[11])
        assert seeded.cell_row_complete[donor]
        for edge in (11, 12):
            local_edge = int(seeded.edge_g2l()[edge])
            assert local_edge >= 0 and seeded.edge_row_complete[local_edge]
    assert_exact_cover(layouts, n_cells=12, n_edges=14, n_vertices=16)
    from hexcore.partition_samehost_tables_v841 import build_all
    tables = build_all(layouts)
    # The replicated ring-0 cell is still a halo received from its real owner.
    source = next(peer for peer in tables[0].peers if peer.peer == 2)
    assert 4 in source.cell_recv


def test_replicated_seed_metadata_roundtrips_and_binds_cache(tmp_path):
    _, coe, cov, graph, part = regional_case(tmp_path)
    layouts = build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov, replicated_cells=[11])
    target = tmp_path / "seeded.npz"
    save_layouts_npz(target, layouts, mesh_sha256="grid", part_sha256="part", halo_rings=2)
    loaded = load_layouts_npz(target, expect_mesh_sha256="grid", expect_part_sha256="part", expect_halo_rings=2, expect_replicated_cells=[11])
    assert all(layout._replicated_cells == (11,) for layout in loaded)
    assert loaded[0].receipt()["replicated_cells"] == [11]
    with pytest.raises(ValueError, match="cached replicated_cells"):
        load_layouts_npz(target, expect_mesh_sha256="grid", expect_part_sha256="part", expect_halo_rings=2, expect_replicated_cells=[])


@pytest.mark.parametrize("seed", [[-1], [12], [1.5]])
def test_replicated_seed_ids_are_integer_global_cells(tmp_path, seed):
    _, coe, cov, graph, part = regional_case(tmp_path)
    with pytest.raises(ValueError, match="replicated_cells"):
        build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov, replicated_cells=seed)


def test_edge_stencil_naming_a_culled_neighbours_far_edge_is_incomplete(tmp_path):
    """Public conus12km, partition 0 of 2: boundary edge 81697 (cells [24202, absent]) names in its
    edgesOnEdge an edge whose only cell is not a neighbour of 24202. The /3 verdict called that halo
    row complete and the census refused the partition. Here edge 13 (cells [absent, 0]) names edge 8
    (cells 7, 8), which partition 0 does not hold."""
    layouts, coe, cov, graph, part = regional_case(tmp_path, 3, 2)
    n_edges = coe.shape[0]
    eoe = np.full((n_edges, 2), -1, dtype=np.int32)
    eoe[:, 0] = np.arange(n_edges)
    counts = np.ones(n_edges, dtype=np.int32)
    eoe[13] = [0, 8]
    counts[13] = 2
    stencil = build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov,
                                      edges_on_edge=eoe, n_edges_on_edge=counts)
    first_old, first_new = layouts[0], stencil[0]
    row = np.flatnonzero(first_new.edge_l2g == 13)[0]
    assert first_old.edge_row_complete[row] and not first_new.edge_row_complete[row]
    keep = first_new.edge_l2g != 13
    np.testing.assert_array_equal(first_old.edge_row_complete[keep], first_new.edge_row_complete[keep])
    for old, new in zip(layouts[1:], stencil[1:]):
        np.testing.assert_array_equal(old.edge_row_complete, new.edge_row_complete)
    with pytest.raises(ValueError, match="supplied together"):
        build_partition_layouts(graph, part, halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov, edges_on_edge=eoe)


@pytest.mark.parametrize("halo", [0, 1, 4])
@pytest.mark.parametrize("seeds", [[], [0, 11, 11]])
@pytest.mark.parametrize("selection", [None, [0], []])
@pytest.mark.parametrize("layout", ["C", "F", "strided-big-endian"])
def test_single_partition_matches_general_layout_words_and_file(
    tmp_path, monkeypatch, halo, seeds, selection, layout,
):
    """All-owned layouts retain the regional/missing-slot and seed contracts."""
    import hexcore.partition_assets_v841 as assets
    import hexcore.host_parallel as parallel

    _, coe, cov, graph, part = regional_case(tmp_path, 1, halo)
    eoe = np.column_stack((np.arange(len(coe)), np.full(len(coe), -1))).astype(np.int32)
    counts = np.ones(len(coe), dtype=np.int32)
    # Active missing slots and inactive valid slots both remain harmless.
    eoe[0] = [-1, len(coe) - 1]
    counts[0] = 0

    def storage(value):
        if layout == "strided-big-endian":
            backing = np.empty((len(value) * 2,) + value.shape[1:], dtype=">i8")
            result = backing[::2]
            result[...] = value
        else:
            result = np.array(value, order=layout)
        result.flags.writeable = False
        return result

    coe, cov, eoe, counts = map(storage, (coe, cov, eoe, counts))
    inputs = [a.tobytes() for a in (coe, cov, eoe, counts)]
    kwargs = dict(halo_rings=halo, cells_on_edge=coe, cells_on_vertex=cov,
                  edges_on_edge=eoe, n_edges_on_edge=counts,
                  replicated_cells=seeds, partitions=selection)
    shortcut = assets._single_partition_layout
    monkeypatch.setattr(assets, "_single_partition_layout", lambda *a, **kw: None)
    general = assets.build_partition_layouts(graph, part, **kwargs)
    monkeypatch.setattr(assets, "_single_partition_layout", shortcut)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(parallel, "MIN_BLOCK_COLUMNS", 2)
    direct = assets.build_partition_layouts(graph, part, **kwargs)
    assert [x.receipt() for x in general] == [x.receipt() for x in direct]
    for left, right in zip(general, direct):
        for name in ("cell_l2g", "edge_l2g", "vertex_l2g", "cell_ring",
                     "cell_row_complete", "edge_row_complete", "vertex_row_complete"):
            a, b = getattr(left, name), getattr(right, name)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes()
            assert a.flags.c_contiguous == b.flags.c_contiguous
            assert a.flags.writeable == b.flags.writeable
        assert (left._global_cells, left._global_edges, left._global_vertices,
                left._replicated_cells) == (
                    right._global_cells, right._global_edges, right._global_vertices,
                    right._replicated_cells)
    for name, result in (("general", general), ("direct", direct)):
        if result:
            save_layouts_npz(tmp_path / f"{name}.npz", result,
                             mesh_sha256="mesh", part_sha256="part", halo_rings=halo)
        else:
            with pytest.raises(IndexError):
                save_layouts_npz(tmp_path / f"{name}.npz", result,
                                 mesh_sha256="mesh", part_sha256="part", halo_rings=halo)
    if general:
        assert (tmp_path / "general.npz").read_bytes() == (tmp_path / "direct.npz").read_bytes()
    assert [a.tobytes() for a in (coe, cov, eoe, counts)] == inputs


@pytest.mark.parametrize("defect", ["cell-bounds", "cell-missing", "edge-bounds",
                                   "edge-shape", "edge-dtype", "edge-pair", "seed"])
def test_single_partition_keeps_incidence_stencil_and_seed_refusals(tmp_path, monkeypatch, defect):
    import hexcore.partition_assets_v841 as assets

    _, coe, cov, graph, part = regional_case(tmp_path, 1)
    kwargs = dict(halo_rings=2, cells_on_edge=coe, cells_on_vertex=cov,
                  edges_on_edge=np.arange(len(coe), dtype=np.int32)[:, None],
                  n_edges_on_edge=np.ones(len(coe), dtype=np.int32))
    if defect == "cell-bounds":
        coe[0, 0] = 12
    elif defect == "cell-missing":
        cov[0] = -1
    elif defect == "edge-bounds":
        kwargs["edges_on_edge"][0, 0] = len(coe)
    elif defect == "edge-shape":
        kwargs["edges_on_edge"] = kwargs["edges_on_edge"][:-1]
    elif defect == "edge-dtype":
        kwargs["edges_on_edge"] = kwargs["edges_on_edge"].astype(np.float64)
    elif defect == "edge-pair":
        kwargs.pop("n_edges_on_edge")
    else:
        kwargs["replicated_cells"] = [12]
    with pytest.raises(ValueError) as direct:
        assets.build_partition_layouts(graph, part, **kwargs)
    monkeypatch.setattr(assets, "_single_partition_layout", lambda *a, **kw: None)
    with pytest.raises(ValueError) as general:
        assets.build_partition_layouts(graph, part, **kwargs)
    assert str(direct.value) == str(general.value)
