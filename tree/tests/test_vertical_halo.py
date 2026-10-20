"""The vertical grid on a working halo is the closed mesh's, cell for cell.

THE BREAKAGE THIS PREVENTS.  ``gpuwm-hex mesh-plan --point --generate``
builds the cut's vertical grid on a HALO -- a wider cut of the parent -- and
cuts it down, instead of building it on the whole parent.  That is only
right if every value the cut keeps is the value the whole parent gives, bit
for bit, and if a limited-area mesh handed to the authority WITHOUT being a
halo is still refused (its rim would be smoothed against nothing).

The mesh here is a closed hexagonal lattice on a torus, so the "whole
parent" exists exactly, and the smoothing is cut to a handful of passes so
the dependency cone is small enough to state: a value on a cell depends on
cells at most ``CONE`` rings away.  A halo deeper than the cone must give
the closed lattice's bits on the kept cells; a halo shallower than it must
not, which is what shows the test can fail.
"""

from __future__ import annotations

import numpy as np
import pytest

from _vertical_rust_fixture import delivered_vertical_engine, vertical_rust_engine
from _vertical_rust_fixture import requires_hex_vertical

from hexcore.errors import ConfigurationRefusal
from hexcore.vertical import build_edge_vertical_metrics, build_vertical_grid

#: Axial neighbour offsets of a hexagonal lattice, in slot order.
OFFSETS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))

N = 44
LEVELS = 4
PASSES = 1
#: Rings a kept value can depend on: two terrain Laplacians, then the
#: coordinate-surface passes of every level, then one ring for an edge and
#: one for its third-order stencil.
CONE = 2 + sum(PASSES + level + 1 for level in range(1, LEVELS)) + 2


class Lattice:
    def __init__(self, n_edges_on_cell, edges_on_cell, cells_on_cell, cells_on_edge,
                 dv_edge, dc_edge, area_cell, deriv_two):
        self.nEdgesOnCell = n_edges_on_cell
        self.edgesOnCell = edges_on_cell
        self.cellsOnCell = cells_on_cell
        self.cellsOnEdge = cells_on_edge
        self.dvEdge = dv_edge
        self.dcEdge = dc_edge
        self.areaCell = area_cell
        self.deriv_two = deriv_two


def _torus(seed: int = 7):
    rng = np.random.default_rng(seed)
    cell = lambda i, j: (i % N) * N + (j % N)  # noqa: E731
    n_cells = N * N
    cells_on_cell = np.zeros((n_cells, 6), dtype=np.int64)
    edges_on_cell = np.zeros((n_cells, 6), dtype=np.int64)
    edge_of: dict[tuple[int, int], int] = {}
    cells_on_edge = []
    for i in range(N):
        for j in range(N):
            c = cell(i, j)
            for slot, (di, dj) in enumerate(OFFSETS):
                nb = cell(i + di, j + dj)
                cells_on_cell[c, slot] = nb
                key = (min(c, nb), max(c, nb))
                if key not in edge_of:
                    edge_of[key] = len(cells_on_edge)
                    cells_on_edge.append(key)
                edges_on_cell[c, slot] = edge_of[key]
    cells_on_edge = np.asarray(cells_on_edge, dtype=np.int64)
    n_edges = cells_on_edge.shape[0]
    # Irregular lengths and areas, so a misplaced stencil cannot hide.
    dc = 1000.0 * (1.0 + 0.05 * rng.random(n_edges))
    dv = dc / np.sqrt(3.0) * (1.0 + 0.05 * rng.random(n_edges))
    area = 1000.0 ** 2 * np.sqrt(3.0) / 2.0 * (1.0 + 0.05 * rng.random(n_cells))
    terrain = 800.0 * rng.random(n_cells)
    terrain[rng.random(n_cells) < 0.3] = 0.0  # exactly-zero terrain: the skip branch
    deriv_two = rng.normal(size=(7, 2, n_edges)) * 1e-6
    lattice = Lattice(np.full(n_cells, 6, dtype=np.int64), edges_on_cell, cells_on_cell,
                      cells_on_edge, dv, dc, area, deriv_two)
    axial = np.array([(i, j) for i in range(N) for j in range(N)])
    return lattice, terrain, axial


def _distance(axial: np.ndarray, centre: tuple[int, int]) -> np.ndarray:
    di = (axial[:, 0] - centre[0] + N // 2) % N - N // 2
    dj = (axial[:, 1] - centre[1] + N // 2) % N - N // 2
    return np.maximum.reduce([np.abs(di), np.abs(dj), np.abs(di + dj)])


def _cut(lattice: Lattice, keep: np.ndarray):
    """The lattice restricted to ``keep``, with -1 where a neighbour was cut."""

    cells = np.flatnonzero(keep)
    new_cell = np.full(keep.size, -1, dtype=np.int64)
    new_cell[cells] = np.arange(cells.size)
    edges = np.unique(lattice.edgesOnCell[cells].ravel())
    new_edge = np.full(lattice.cellsOnEdge.shape[0], -1, dtype=np.int64)
    new_edge[edges] = np.arange(edges.size)
    cut = Lattice(
        lattice.nEdgesOnCell[cells],
        new_edge[lattice.edgesOnCell[cells]],
        new_cell[lattice.cellsOnCell[cells]],
        new_cell[lattice.cellsOnEdge[edges]],
        lattice.dvEdge[edges],
        lattice.dcEdge[edges],
        lattice.areaCell[cells],
        lattice.deriv_two[:, :, edges],
    )
    return cut, cells, edges


def _vertical(lattice: Lattice, terrain: np.ndarray, *, regional_halo: bool):
    vertical = build_vertical_grid(
        lattice, terrain, n_vert_levels=LEVELS, ztop=20_000.0, scheme="legacy",
        surface_smoothing_passes=PASSES, hybrid_transition_height=20_000.0,
        regional_halo=regional_halo,
    )
    edge = build_edge_vertical_metrics(lattice, vertical, theta_adv_order=3)
    return {"hx": vertical.hx, "zgrid": vertical.zgrid, "zz": vertical.zz, "dss": vertical.dss,
            "zxu": vertical.zxu, "zb": edge.zb, "zb3": edge.zb3}


def _kept_values(fields, cells_index, edges_index):
    out = {}
    for name, value in fields.items():
        index = edges_index if name in ("zxu", "zb", "zb3") else cells_index
        out[name] = np.ascontiguousarray(np.take(value, index, axis=-1))
    return out


def _compare(depth: int, inner: int = 3):
    lattice, terrain, axial = _torus()
    whole = _vertical(lattice, terrain, regional_halo=False)
    distance = _distance(axial, (N // 2, N // 2))
    halo, cells, edges = _cut(lattice, distance <= inner + depth)
    got = _vertical(halo, terrain[cells], regional_halo=True)
    kept_cells = np.flatnonzero(distance[cells] <= inner)
    # The kept edges: every edge of a kept cell.
    kept_edges = np.unique(halo.edgesOnCell[kept_cells].ravel())
    whole_cells = cells[kept_cells]
    whole_edges = edges[kept_edges]
    a = _kept_values(whole, whole_cells, whole_edges)
    b = _kept_values(got, kept_cells, kept_edges)
    return {name: bool(np.array_equal(a[name].view(np.uint64), b[name].view(np.uint64))) for name in a}


def test_a_limited_area_mesh_that_is_not_a_halo_is_still_refused() -> None:
    lattice, terrain, axial = _torus()
    halo, cells, _ = _cut(lattice, _distance(axial, (N // 2, N // 2)) <= 10)
    with pytest.raises(ConfigurationRefusal) as refusal:
        build_vertical_grid(halo, terrain[cells], n_vert_levels=LEVELS, ztop=20_000.0,
                            scheme="legacy", surface_smoothing_passes=PASSES,
                            hybrid_transition_height=20_000.0)
    assert "regional_boundary_metrics" in str(refusal.value)
    assert "does not invent exterior state" in str(refusal.value)


@requires_hex_vertical
def test_a_halo_deeper_than_the_cone_gives_the_closed_mesh_bit_for_bit() -> None:
    same = _compare(depth=CONE + 1)
    assert all(same.values()), f"fields that moved on the kept cells: {same}"


@requires_hex_vertical
def test_a_halo_inside_the_cone_moves_the_kept_values() -> None:
    same = _compare(depth=2)
    assert not same["hx"], "a 2-ring halo reproduced the closed mesh; the test has no teeth"


def test_the_lattice_is_closed_and_the_cut_carries_sentinels() -> None:
    lattice, terrain, axial = _torus()
    assert lattice.cellsOnCell.min() >= 0 and lattice.cellsOnEdge.min() >= 0
    halo, _, _ = _cut(lattice, _distance(axial, (N // 2, N // 2)) <= 10)
    assert halo.cellsOnCell.min() == -1 and halo.cellsOnEdge.min() == -1
    assert CONE + 1 + 3 < N // 2, "the halo would wrap the torus"


@requires_hex_vertical
def test_two_agreeing_halos_are_checked_against_exterior_terrain(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace
    from hexcore import mesh_point, vertical_spec
    from hexcore.mesh import Mesh

    lattice, _, axial = _torus()
    distance = _distance(axial, (N // 2, N // 2))
    terrain = np.where(distance <= 6, 0.0, 800.0)
    whole = _vertical(lattice, terrain, regional_halo=False)
    cut, cut_cells, cut_edges = _cut(lattice, distance <= 1)
    halo, cells, edges = _cut(lattice, distance <= 6)
    narrow, narrow_cells, narrow_edges = _cut(lattice, distance <= 4)
    wide_fields = _vertical(halo, terrain[cells], regional_halo=True)
    narrow_fields = _vertical(narrow, terrain[narrow_cells], regional_halo=True)
    index = lambda whole_ids, kept: np.array([np.flatnonzero(whole_ids == k)[0] for k in kept])
    wide_indices = (index(cells, cut_cells), index(edges, cut_edges))
    narrow_indices = (index(narrow_cells, cut_cells), index(narrow_edges, cut_edges))
    a = _kept_values(wide_fields, *wide_indices)
    b = _kept_values(narrow_fields, *narrow_indices)
    parent = _kept_values(whole, cut_cells, cut_edges)
    assert all(a[name].tobytes() == b[name].tobytes() for name in a)
    assert a["hx"].tobytes() != parent["hx"].tobytes()

    def built(values):
        return (SimpleNamespace(**{k: values[k] for k in ("hx", "zgrid", "zz", "dss", "zxu")}),
                SimpleNamespace(zb=values["zb"], zb3=values["zb3"]))
    monkeypatch.setattr(Mesh, "from_netcdf", classmethod(lambda cls, grid, static=None: grid.name))
    monkeypatch.setattr(vertical_spec.VerticalSpec, "from_file", classmethod(lambda cls, path: None))
    monkeypatch.setattr(vertical_spec, "build_vertical_from_spec",
                        lambda mesh, spec, **kwargs: built(wide_fields if mesh == "halo" else whole))
    monkeypatch.setattr(mesh_point, "_element_index", lambda sub, sup, kind:
                        wide_indices[0 if kind == "Cell" else 1] if sup.name == "halo"
                        else cut_cells if kind == "Cell" else cut_edges)
    check = mesh_point.check_halo_vertical(
        work_grid=tmp_path / "halo", work_static=tmp_path / "halo-static",
        parent_grid=tmp_path / "parent", parent_static=tmp_path / "parent-terrain",
        cut_grid=tmp_path / "cut", vertical_spec=tmp_path / "spec", log=lambda *args: None)
    assert not check["agreed"]
    assert check["differing"]["hx"] > 0
    assert check["authority"] == "whole-parent"
