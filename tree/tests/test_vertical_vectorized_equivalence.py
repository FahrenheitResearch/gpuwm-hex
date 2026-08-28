"""The vectorized vertical-coordinate loops are BITWISE the scalar transcription.

THE BREAKAGE THIS PREVENTS.  ``_laplacian_without_area`` and
``build_edge_vertical_metrics`` were cell-major/edge-major Python loops doing
scalar float arithmetic; measured on the 112,676-cell parent of 2026-08-27 the
first was 86.2 % of the init stage and the second 7.2 %.  Both are now
slot-major vector updates.  Reordering floating-point accumulation changes
bits, and an init/vertical artifact whose bits move poisons every registered
digest downstream -- the vertical artifact is the byte-reproducible anchor of
the parent-regeneration chain.  These tests hold the vector form against the
scalar form this file carries verbatim, on a mesh that exercises what the
uniform K4 fixture cannot: ragged ``nEdgesOnCell``, absent (-1) neighbours,
exactly-zero terrain cells (the ``skip_zero_center`` branch), and all three
``theta_adv_order`` branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.vertical import (
    _laplacian_without_area,
    build_edge_vertical_metrics,
    build_vertical_grid,
    edge_dc_squared_over_twelve,
    smooth_terrain,
)


def _scalar_laplacian(mesh, field, *, skip_zero_center):
    """The retired scalar transcription, kept as the referee."""

    n_edges_on_cell = np.asarray(mesh.nEdgesOnCell).astype(np.int64, copy=False)
    edges_on_cell = np.asarray(mesh.edgesOnCell).astype(np.int64, copy=False)
    cells_on_cell = np.asarray(mesh.cellsOnCell).astype(np.int64, copy=False)
    dv_edge = np.asarray(mesh.dvEdge)
    dc_edge = np.asarray(mesh.dcEdge)
    out = np.zeros_like(field)
    for cell in range(field.size):
        if skip_zero_center and field[cell] == field.dtype.type(0.0):
            continue
        for slot in range(int(n_edges_on_cell[cell])):
            edge = int(edges_on_cell[cell, slot])
            neighbor = int(cells_on_cell[cell, slot])
            neighbor_value = field[cell] if neighbor < 0 else field[neighbor]
            out[cell] += (
                dv_edge[edge] / dc_edge[edge] * (neighbor_value - field[cell])
            )
    return out


def _scalar_edge_metrics(mesh, vertical, *, theta_adv_order):
    """The retired scalar zb/zb3 transcription, kept as the referee."""

    cells_on_edge = np.asarray(mesh.cellsOnEdge).astype(np.int64, copy=False)
    n_edges = int(cells_on_edge.shape[0])
    n_cells = int(vertical.zgrid.shape[1])
    dc_edge = np.asarray(mesh.dcEdge)
    dv_edge = np.asarray(mesh.dvEdge)
    area_cell = np.asarray(mesh.areaCell)
    deriv_two = None
    cells_on_cell = None
    n_edges_on_cell = None
    if theta_adv_order != 2:
        raw = np.asarray(mesh.deriv_two)
        deriv_two = np.transpose(raw, (2, 1, 0))
        cells_on_cell = np.asarray(mesh.cellsOnCell).astype(np.int64, copy=False)
        n_edges_on_cell = np.asarray(mesh.nEdgesOnCell).astype(np.int64, copy=False)
    dtype = vertical.zgrid.dtype
    zb = np.zeros((vertical.n_vert_levels + 1, 2, n_edges), dtype=dtype)
    zb3 = np.zeros_like(zb)
    for edge in range(n_edges):
        cell1 = int(cells_on_edge[edge, 0])
        cell2 = int(cells_on_edge[edge, 1])
        if cell1 < 0:
            cell1 = cell2
        if cell2 < 0:
            cell2 = cell1
        assert 0 <= cell1 < n_cells and 0 <= cell2 < n_cells
        scale1 = dtype.type(dv_edge[edge] / area_cell[cell1])
        scale2 = dtype.type(dv_edge[edge] / area_cell[cell2])
        dc2_over_12 = dtype.type(dc_edge[edge] ** 2 / 12.0)
        for level in range(vertical.n_vert_levels):
            z1 = vertical.zgrid[level, cell1]
            z2 = vertical.zgrid[level, cell2]
            z_edge3 = dtype.type(0.0)
            if theta_adv_order == 2:
                z_edge = dtype.type(0.5) * (z1 + z2)
            else:
                d2_1 = deriv_two[0, 0, edge] * z1
                d2_2 = deriv_two[0, 1, edge] * z2
                for slot in range(int(n_edges_on_cell[cell1])):
                    neighbor = int(cells_on_cell[cell1, slot])
                    if neighbor >= 0:
                        d2_1 += (
                            deriv_two[slot + 1, 0, edge]
                            * vertical.zgrid[level, neighbor]
                        )
                for slot in range(int(n_edges_on_cell[cell2])):
                    neighbor = int(cells_on_cell[cell2, slot])
                    if neighbor >= 0:
                        d2_2 += (
                            deriv_two[slot + 1, 1, edge]
                            * vertical.zgrid[level, neighbor]
                        )
                z_edge = dtype.type(0.5) * (z1 + z2) - dc2_over_12 * (d2_1 + d2_2)
                if theta_adv_order == 3:
                    z_edge3 = -dc2_over_12 * (d2_1 - d2_2)
            zb[level, 0, edge] = (z_edge - z1) * scale1
            zb[level, 1, edge] = (z_edge - z2) * scale2
            zb3[level, 0, edge] = z_edge3 * scale1
            zb3[level, 1, edge] = z_edge3 * scale2
    return zb, zb3


class RaggedMesh:
    """A closed ragged mesh: mixed cell degree, absent neighbours, zero terrain.

    Built from an icosahedral-style random adjacency rather than a real
    tessellation -- the loops under test are pure gather/accumulate and do not
    care whether the topology is geometrically realisable, only that it is
    ragged, that some slots hold -1, and that some terrain values are exactly
    zero.
    """

    def __init__(
        self,
        n_cells: int = 97,
        max_edges: int = 10,
        seed: int = 20260827,
        dtype: type = np.float32,
    ):
        rng = np.random.default_rng(seed)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for cell in range(n_cells):
            for _ in range(3):
                other = int(rng.integers(0, n_cells))
                if other == cell:
                    continue
                key = (min(cell, other), max(cell, other))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
        self.cellsOnEdge = np.asarray(pairs, dtype=np.int64)
        n_edges = len(pairs)
        # Real generated statics carry these as float32 -- measured on
        # s01.static.nc, 2026-08-27.  A float64 fixture cannot see the
        # scalar-vs-array power split that moved zb/zb3, because npy_pow
        # short-circuits exponent 2 in double and powf does not.
        self.dcEdge = (1000.0 + rng.random(n_edges) * 500.0).astype(dtype)
        self.dvEdge = (800.0 + rng.random(n_edges) * 500.0).astype(dtype)
        self.areaCell = (1.0e6 + rng.random(n_cells) * 5.0e5).astype(dtype)
        incident: list[list[tuple[int, int]]] = [[] for _ in range(n_cells)]
        for edge, (left, right) in enumerate(pairs):
            incident[left].append((edge, right))
            incident[right].append((edge, left))
        self.nEdgesOnCell = np.asarray(
            [max(len(row), 1) for row in incident], dtype=np.int64
        )
        edges_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)
        cells_on_cell = np.zeros((n_cells, max_edges), dtype=np.int64)
        for cell, row in enumerate(incident):
            for slot, (edge, neighbor) in enumerate(row[:max_edges]):
                edges_on_cell[cell, slot] = edge
                cells_on_cell[cell, slot] = neighbor
            self.nEdgesOnCell[cell] = max(min(len(row), max_edges), 1)
        # Every fourth cell loses its last neighbour to the -1 sentinel: the
        # regional garbage-cell rule the scalar loop carried.
        for cell in range(0, n_cells, 4):
            last = int(self.nEdgesOnCell[cell]) - 1
            if last > 0:
                cells_on_cell[cell, last] = -1
        self.edgesOnCell = edges_on_cell
        self.cellsOnCell = cells_on_cell
        # Native on-disk orientation (nEdges, TWO, nCoeff).
        self.deriv_two = (
            rng.standard_normal((n_edges, 2, max_edges + 1)) * 1.0e-7
        ).astype(dtype)
        terrain = (rng.random(n_cells) * 2500.0).astype(dtype)
        terrain[::5] = 0.0  # exercises the skip_zero_center branch
        self.ter = terrain


@pytest.fixture(params=[np.float32, np.float64], ids=["float32", "float64"])
def ragged(request: pytest.FixtureRequest) -> RaggedMesh:
    return RaggedMesh(dtype=request.param)


def test_dc_squared_over_twelve_reproduces_the_scalar_power() -> None:
    """``dcEdge**2/12`` must be numpy's SCALAR answer, not its array square.

    THE BREAKAGE THIS PREVENTS, MEASURED: writing this term as
    ``float32_array ** 2`` moved 39 ``zb`` and 18,141 ``zb3`` elements of the
    112,676-cell parent on 2026-08-27, because numpy rewrites the array form to
    ``np.square`` while the scalar form calls the libm ``powf``.  Every other
    field stayed bitwise equal, so nothing but this assertion would have caught
    it before the artifact reached a registered digest.
    """

    rng = np.random.default_rng(20260827)
    for dtype in (np.float32, np.float64):
        dc = (rng.random(200_000) * 1.0e5 + 1.0).astype(dtype)
        scalar = np.array(
            [dtype(dtype(x) ** 2 / 12.0) for x in dc], dtype=dtype
        )
        produced = edge_dc_squared_over_twelve(dc, np.dtype(dtype))
        assert produced.dtype == np.dtype(dtype)
        assert produced.tobytes() == scalar.tobytes(), (
            f"{dtype.__name__}: the vectorized dcEdge**2/12 left the scalar "
            f"answer on {int((produced != scalar).sum())} of {dc.size} values"
        )


def test_the_naive_array_square_is_a_real_trap_on_this_build() -> None:
    """Validate the instrument: prove the two forms actually can disagree.

    If a numpy/libm build makes ``arr ** 2`` and ``scalar ** 2`` agree
    everywhere, the test above still states the right contract but proves
    nothing on that box -- so say so out loud rather than read green as
    coverage.
    """

    rng = np.random.default_rng(20260827)
    dc = (rng.random(200_000) * 1.0e5 + 1.0).astype(np.float32)
    scalar = np.array([np.float32(np.float32(x) ** 2) for x in dc], np.float32)
    if np.array_equal(scalar, dc**2):
        pytest.skip(
            "this numpy/libm build returns the same float32 for scalar and "
            "array squaring, so the guard above is untested here; it was "
            "measured to disagree on 207 of 338,022 real dcEdge values "
            "(glibc, numpy 2.5.2) and on 21 of 200,000 here on numpy 2.2.6"
        )
    assert int((scalar != dc**2).sum()) > 0


@pytest.mark.parametrize("skip_zero_center", [True, False])
def test_neighbor_sum_is_bitwise_the_scalar_loop(
    ragged: RaggedMesh, skip_zero_center: bool
) -> None:
    field = np.asarray(ragged.ter)
    reference = _scalar_laplacian(
        ragged, field, skip_zero_center=skip_zero_center
    )
    produced = _laplacian_without_area(
        ragged, field, skip_zero_center=skip_zero_center
    )
    assert produced.dtype == reference.dtype
    assert produced.tobytes() == reference.tobytes()


def test_smoothed_terrain_is_bitwise_the_scalar_loop(ragged: RaggedMesh) -> None:
    result = np.asarray(ragged.ter).copy()
    for _ in range(2):
        first = _scalar_laplacian(ragged, result, skip_zero_center=True)
        hs = result + result.dtype.type(0.216) * first
        second = _scalar_laplacian(ragged, hs, skip_zero_center=True)
        result = hs - hs.dtype.type(0.216) * second
    produced = smooth_terrain(ragged, ragged.ter, passes=2)
    assert produced.tobytes() == result.tobytes()


@pytest.mark.parametrize("theta_adv_order", [2, 3, 4])
def test_edge_vertical_metrics_are_bitwise_the_scalar_loop(
    ragged: RaggedMesh, theta_adv_order: int
) -> None:
    vertical = build_vertical_grid(
        ragged,
        np.asarray(ragged.ter),
        n_vert_levels=9,
        ztop=20_000.0,
        terrain_smoothing_passes=1,
        surface_smoothing_passes=3,
        hybrid_transition_height=16_000.0,
    )
    reference_zb, reference_zb3 = _scalar_edge_metrics(
        ragged, vertical, theta_adv_order=theta_adv_order
    )
    metrics = build_edge_vertical_metrics(
        ragged, vertical, theta_adv_order=theta_adv_order
    )
    assert metrics.zb.tobytes() == reference_zb.tobytes()
    assert metrics.zb3.tobytes() == reference_zb3.tobytes()


@pytest.mark.parametrize("width", [1, 7, 10_000_000])
def test_edge_metric_chunking_does_not_move_a_byte(
    ragged: RaggedMesh, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """The edge loop is chunked for memory; a chunk boundary moves no bit."""

    import hexcore.vertical as module

    vertical = build_vertical_grid(
        ragged,
        np.asarray(ragged.ter),
        n_vert_levels=9,
        ztop=20_000.0,
        smooth_surfaces=False,
    )
    reference = build_edge_vertical_metrics(ragged, vertical, theta_adv_order=3)
    monkeypatch.setattr(module, "EDGE_METRIC_CHUNK_EDGES", width)
    produced = build_edge_vertical_metrics(ragged, vertical, theta_adv_order=3)
    assert produced.zb.tobytes() == reference.zb.tobytes()
    assert produced.zb3.tobytes() == reference.zb3.tobytes()


def test_absent_endpoint_pair_is_refused_by_edge_index(ragged: RaggedMesh) -> None:
    vertical = build_vertical_grid(
        ragged,
        np.asarray(ragged.ter),
        n_vert_levels=5,
        ztop=20_000.0,
        smooth_surfaces=False,
    )
    broken = RaggedMesh()
    broken.cellsOnEdge = np.array(ragged.cellsOnEdge, copy=True)
    broken.cellsOnEdge[7] = (-1, -1)
    with pytest.raises(ValueError, match="edge 7 has no interior cell"):
        build_edge_vertical_metrics(broken, vertical, theta_adv_order=3)
