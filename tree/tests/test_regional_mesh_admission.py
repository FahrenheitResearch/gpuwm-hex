"""Regional (limited-area) mesh admission in :meth:`Mesh.validate`.

A native-culled regional mesh is a bounded disk cut from a parent sphere.  The
measured conventions (2026-08-25, native MPAS-Limited-Area culls of the
published x1.40962 and the x4.163842 window, real bytes):

* three added ``int32`` variables ``bdyMaskCell/Edge/Vertex``, values 0
  (interior) through 7 (outermost ring), each ring sitting on the ring
  inside it (which is NOT the same as growing populations -- see
  ``test_nest_ratio_ring_shell.py`` for the measured culls that separated
  the two);
* edge/vertex masks equal the MINIMUM of their present cells' masks (held for
  100 percent of elements on both culls) and neighbouring cell masks never
  differ by more than 1;
* absent-element sentinels (file zeros, canonical ``-1``) appear ONLY on
  mask-7 elements and ONLY in ``cellsOnCell``, ``cellsOnEdge``,
  ``edgesOnEdge`` (inside the declared ``nEdgesOnEdge`` row length),
  ``cellsOnVertex`` and ``edgesOnVertex`` -- never in ``edgesOnCell``,
  ``verticesOnCell`` or ``verticesOnEdge``;
* Euler characteristic ``nCells - nEdges + nVertices == 1`` (a disk, not the
  closed sphere's 2).

The tests here run against a synthetic cull of a defect-free synthetic sphere
(battery-runnable everywhere) and, when ``GPUWM_HEX_REGIONAL_REFERENCE_DIR``
points at the native-culled reference set, against the real culled bytes.

The np.add.at wrap this file pins down: with a ``-1`` sentinel present,
``np.add.at(cell_kite_sum, cellsOnVertex.ravel(), kites.ravel())`` folds every
absent-neighbour kite into cell ``nCells-1``.  Measured on the real x1 cull:
relative error 100.46 at that cell, 394.39 on the x4 window -- a fabricated
"kites do not sum to areaCell" report against a defect-free mesh.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mesh import Mesh, MeshValidationError

from test_mesh_validation import graded_sphere_mesh


REFERENCE_DIR_VARIABLE = "GPUWM_HEX_REGIONAL_REFERENCE_DIR"


# ---------------------------------------------------------------------------
# the instrument: a synthetic regional cull following the measured conventions
# ---------------------------------------------------------------------------
def _hop_distances(mesh: Mesh, seeds: np.ndarray) -> np.ndarray:
    """Breadth-first hop distance from ``seeds`` over cellsOnCell."""

    n_cells = int(mesh.dimensions["nCells"])
    counts = np.asarray(mesh.nEdgesOnCell, dtype=np.int64)
    neighbours = np.asarray(mesh.cellsOnCell, dtype=np.int64)
    distance = np.full(n_cells, -1, dtype=np.int64)
    frontier = list(int(seed) for seed in seeds)
    distance[frontier] = 0
    while frontier:
        next_frontier: list[int] = []
        for cell in frontier:
            for neighbour in neighbours[cell, : counts[cell]]:
                if neighbour >= 0 and distance[neighbour] < 0:
                    distance[neighbour] = distance[cell] + 1
                    next_frontier.append(int(neighbour))
        frontier = next_frontier
    return distance


def cull_regional_mesh(parent: Mesh, *, seed_cell: int, disk_hops: int) -> Mesh:
    """Cut a disk from ``parent`` exactly as the native cull stores one.

    Keeps cells within ``disk_hops`` hops of ``seed_cell``, every edge and
    vertex of a kept cell, masks cells by distance to the cut (outermost ring
    7 decreasing inward, interior 0), masks edges/vertices as the minimum of
    their present cells, and writes ``-1`` sentinels in exactly the five
    arrays the native cull zeroes -- with ``nEdgesOnEdge`` NOT shrunk.
    """

    keep_cells = _hop_distances(parent, np.array([seed_cell])) <= disk_hops
    n_parent_cells = int(parent.dimensions["nCells"])
    counts = np.asarray(parent.nEdgesOnCell, dtype=np.int64)
    neighbours = np.asarray(parent.cellsOnCell, dtype=np.int64)

    # Distance to the nearest culled cell: ring 7 touches the cut.
    boundary = [
        cell
        for cell in np.flatnonzero(keep_cells)
        for neighbour in neighbours[cell, : counts[cell]]
        if neighbour >= 0 and not keep_cells[neighbour]
    ]
    cut_distance = _hop_distances(parent, np.unique(np.asarray(boundary)))
    mask_cell_parent = np.where(
        keep_cells, np.maximum(0, 7 - cut_distance), -1
    )

    edges_on_cell = np.asarray(parent.edgesOnCell, dtype=np.int64)
    vertices_on_cell = np.asarray(parent.verticesOnCell, dtype=np.int64)
    keep_edges = np.zeros(int(parent.dimensions["nEdges"]), dtype=bool)
    keep_vertices = np.zeros(int(parent.dimensions["nVertices"]), dtype=bool)
    for cell in np.flatnonzero(keep_cells):
        keep_edges[edges_on_cell[cell, : counts[cell]]] = True
        keep_vertices[vertices_on_cell[cell, : counts[cell]]] = True

    def renumber(keep: np.ndarray) -> np.ndarray:
        new = np.full(keep.size, -1, dtype=np.int64)
        new[keep] = np.arange(int(keep.sum()))
        return new

    cell_new = renumber(keep_cells)
    edge_new = renumber(keep_edges)
    vertex_new = renumber(keep_vertices)

    def take_map(array: np.ndarray, keep: np.ndarray, mapping: np.ndarray) -> np.ndarray:
        taken = np.asarray(array, dtype=np.int64)[keep]
        result = np.where(taken >= 0, mapping[np.clip(taken, 0, None)], -1)
        return result

    arrays: dict[str, np.ndarray] = {}
    arrays["nEdgesOnCell"] = np.asarray(parent.nEdgesOnCell)[keep_cells]
    arrays["nEdgesOnEdge"] = np.asarray(parent.nEdgesOnEdge)[keep_edges]
    arrays["cellsOnCell"] = take_map(parent.cellsOnCell, keep_cells, cell_new)
    arrays["edgesOnCell"] = take_map(parent.edgesOnCell, keep_cells, edge_new)
    arrays["verticesOnCell"] = take_map(parent.verticesOnCell, keep_cells, vertex_new)
    arrays["cellsOnEdge"] = take_map(parent.cellsOnEdge, keep_edges, cell_new)
    arrays["verticesOnEdge"] = take_map(parent.verticesOnEdge, keep_edges, vertex_new)
    arrays["cellsOnVertex"] = take_map(parent.cellsOnVertex, keep_vertices, cell_new)
    arrays["edgesOnVertex"] = take_map(parent.edgesOnVertex, keep_vertices, edge_new)
    arrays["edgesOnEdge"] = take_map(parent.edgesOnEdge, keep_edges, edge_new)

    for name in (
        "weightsOnEdge",
        "dcEdge",
        "dvEdge",
        "areaCell",
        "areaTriangle",
        "kiteAreasOnVertex",
    ):
        keep = {
            "weightsOnEdge": keep_edges,
            "dcEdge": keep_edges,
            "dvEdge": keep_edges,
            "areaCell": keep_cells,
            "areaTriangle": keep_vertices,
            "kiteAreasOnVertex": keep_vertices,
        }[name]
        arrays[name] = np.asarray(parent.arrays[name])[keep]
    for suffix, keep in (("Cell", keep_cells), ("Edge", keep_edges), ("Vertex", keep_vertices)):
        for component in ("x", "y", "z", "lat", "lon"):
            name = f"{component}{suffix}"
            arrays[name] = np.asarray(parent.arrays[name])[keep]

    mask_cell = mask_cell_parent[keep_cells]
    huge = np.iinfo(np.int64).max

    def min_present(cells: np.ndarray) -> np.ndarray:
        masks = np.where(cells >= 0, mask_cell[np.clip(cells, 0, None)], huge)
        return masks.min(axis=1)

    arrays["bdyMaskCell"] = mask_cell.astype(np.int32)
    arrays["bdyMaskEdge"] = min_present(arrays["cellsOnEdge"]).astype(np.int32)
    arrays["bdyMaskVertex"] = min_present(arrays["cellsOnVertex"]).astype(np.int32)

    mesh = Mesh(
        arrays=arrays,
        dimensions={
            "nCells": int(keep_cells.sum()),
            "nEdges": int(keep_edges.sum()),
            "nVertices": int(keep_vertices.sum()),
            "maxEdges": int(parent.dimensions["maxEdges"]),
            "maxEdges2": int(parent.dimensions["maxEdges2"]),
            "vertexDegree": int(parent.dimensions["vertexDegree"]),
        },
        attrs={
            "on_a_sphere": parent.attrs["on_a_sphere"],
            "sphere_radius": parent.attrs["sphere_radius"],
        },
    )
    return mesh


@pytest.fixture(scope="module")
def parent_sphere() -> Mesh:
    # Near-uniform and small: the cull geometry, not the grading, is under test.
    return graded_sphere_mesh(level=4, zoom=1.5)


@pytest.fixture(scope="module")
def regional_mesh(parent_sphere: Mesh) -> Mesh:
    return cull_regional_mesh(parent_sphere, seed_cell=0, disk_hops=12)


# ---------------------------------------------------------------------------
# instrument checks: the synthetic cull reproduces the measured conventions
# ---------------------------------------------------------------------------
def test_fixture_is_a_disk_with_the_measured_sentinel_placement(
    regional_mesh: Mesh,
) -> None:
    dims = regional_mesh.dimensions
    assert dims["nCells"] - dims["nEdges"] + dims["nVertices"] == 1

    mask_cell = np.asarray(regional_mesh.bdyMaskCell)
    assert mask_cell.min() == 0 and mask_cell.max() == 7
    counts = [int((mask_cell == ring).sum()) for ring in range(8)]
    assert all(counts[ring] > 0 for ring in range(8))
    assert all(counts[ring] <= counts[ring + 1] for ring in range(1, 7))

    used = (
        np.arange(dims["maxEdges"])[None, :]
        < np.asarray(regional_mesh.nEdgesOnCell)[:, None]
    )
    for name in ("edgesOnCell", "verticesOnCell"):
        assert not np.any((np.asarray(regional_mesh.arrays[name]) == -1) & used)
    assert not np.any(np.asarray(regional_mesh.verticesOnEdge) == -1)
    # sentinels exist, and only on mask-7 rows
    sentinel_rows = np.flatnonzero(
        ((np.asarray(regional_mesh.cellsOnCell) == -1) & used).any(axis=1)
    )
    assert sentinel_rows.size > 0
    assert np.all(mask_cell[sentinel_rows] == 7)


# ---------------------------------------------------------------------------
# the np.add.at wrap: a defect-free regional mesh must be admitted, and its
# absent-neighbour kites must not be folded into cell nCells-1
# ---------------------------------------------------------------------------
def test_regional_mesh_is_admitted_and_sentinel_kites_are_not_wrapped(
    regional_mesh: Mesh,
) -> None:
    # Red before the fix, in two stages: first the unnamed range/Euler death,
    # then -- with ranges sentinel-tolerant but the summation unmasked -- the
    # fabricated "kites belonging to each cell do not sum to areaCell" at the
    # wrap target nCells-1.
    assert regional_mesh.validate() is regional_mesh


def test_a_real_kite_corruption_at_a_boundary_vertex_is_still_refused(
    regional_mesh: Mesh,
) -> None:
    # The other direction: masking sentinels must not blind the check to a
    # genuine inconsistency at a PRESENT slot of a sentinel-carrying vertex.
    corrupt = regional_mesh.copy()
    cells_on_vertex = np.asarray(corrupt.cellsOnVertex)
    rows = np.flatnonzero((cells_on_vertex == -1).any(axis=1))
    vertex = int(rows[0])
    slot = int(np.flatnonzero(cells_on_vertex[vertex] >= 0)[0])
    corrupt.arrays["kiteAreasOnVertex"] = np.asarray(
        corrupt.arrays["kiteAreasOnVertex"], dtype=np.float64
    ).copy()
    corrupt.arrays["kiteAreasOnVertex"][vertex, slot] *= 1.5
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "kites" in str(refusal.value)


# ---------------------------------------------------------------------------
# named refusals: each corruption is refused by a message naming the breakage
# ---------------------------------------------------------------------------
def test_broken_ring_numbering_is_refused_by_name(regional_mesh: Mesh) -> None:
    corrupt = regional_mesh.copy()
    mask = np.asarray(corrupt.arrays["bdyMaskCell"]).copy()
    cell = int(np.flatnonzero(mask == 3)[0])
    mask[cell] = 5
    corrupt.arrays["bdyMaskCell"] = mask
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "jumps by more than 1 between neighbouring cells" in str(refusal.value)


def test_sentinel_in_a_forbidden_array_is_refused_by_name(
    regional_mesh: Mesh,
) -> None:
    corrupt = regional_mesh.copy()
    vertices_on_edge = np.asarray(corrupt.arrays["verticesOnEdge"]).copy()
    edge = int(np.flatnonzero(np.asarray(corrupt.bdyMaskEdge) == 7)[0])
    vertices_on_edge[edge, 1] = -1
    corrupt.arrays["verticesOnEdge"] = vertices_on_edge
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "verticesOnEdge lost an endpoint" in str(refusal.value)


def test_min_rule_violation_is_refused_by_name(regional_mesh: Mesh) -> None:
    corrupt = regional_mesh.copy()
    mask_edge = np.asarray(corrupt.arrays["bdyMaskEdge"]).copy()
    edge = int(np.flatnonzero(mask_edge == 2)[0])
    mask_edge[edge] = 3
    corrupt.arrays["bdyMaskEdge"] = mask_edge
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "bdyMaskEdge is not the minimum of its present cells" in str(
        refusal.value
    )


def test_sentinel_below_the_outermost_ring_is_refused_by_name(
    regional_mesh: Mesh,
) -> None:
    corrupt = regional_mesh.copy()
    cells_on_edge = np.asarray(corrupt.arrays["cellsOnEdge"]).copy()
    edge = int(np.flatnonzero(np.asarray(corrupt.bdyMaskEdge) == 3)[0])
    cells_on_edge[edge, 1] = -1
    corrupt.arrays["cellsOnEdge"] = cells_on_edge
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "below the outermost ring" in str(refusal.value)


def test_incomplete_mask_triple_is_refused_by_name(regional_mesh: Mesh) -> None:
    corrupt = regional_mesh.copy()
    del corrupt.arrays["bdyMaskVertex"]
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "boundary masks are incomplete" in str(refusal.value)


def test_global_meshes_still_refuse_bare_sentinels(parent_sphere: Mesh) -> None:
    # The regional tolerance must not leak into the closed-sphere contract: a
    # global mesh with a -1 in cellsOnVertex is corruption, not a boundary.
    corrupt = parent_sphere.copy()
    cells_on_vertex = np.asarray(corrupt.arrays["cellsOnVertex"]).copy()
    cells_on_vertex[0, 0] = -1
    corrupt.arrays["cellsOnVertex"] = cells_on_vertex
    with pytest.raises(MeshValidationError):
        corrupt.validate()


# ---------------------------------------------------------------------------
# the real bytes: the native-culled reference pairs load and pass
# ---------------------------------------------------------------------------
def _reference_dir() -> Path:
    root = os.environ.get(REFERENCE_DIR_VARIABLE, "")
    if not root or not Path(root).is_dir():
        pytest.skip(
            f"{REFERENCE_DIR_VARIABLE} does not point at the native-culled "
            "regional reference set"
        )
    return Path(root)


def _single_nc(directory: Path) -> Path:
    candidates = sorted(directory.glob("*.nc"))
    if len(candidates) != 1:
        pytest.skip(f"{directory} does not hold exactly one .nc file")
    return candidates[0]


def test_native_culled_quick_grid_loads_and_passes(monkeypatch) -> None:
    grid = _single_nc(_reference_dir() / "cull-x1")
    mesh = Mesh.from_netcdf(grid)
    assert mesh.dimensions["nCells"] - mesh.dimensions["nEdges"] + mesh.dimensions[
        "nVertices"
    ] == 1
    assert int(np.asarray(mesh.bdyMaskCell).max()) == 7


def test_native_culled_reference_pair_loads_and_passes() -> None:
    root = _reference_dir()
    grid = _single_nc(root / "cull-x4" / "grid")
    static = _single_nc(root / "cull-x4" / "static")
    mesh = Mesh.from_netcdf(grid, static)
    assert mesh.dimensions["nCells"] - mesh.dimensions["nEdges"] + mesh.dimensions[
        "nVertices"
    ] == 1


def test_mesh_check_door_grid_only_reports_the_regional_block(capsys) -> None:
    # The door leg: a culled grid exists before its static does, and
    # `mesh-check --grid-only` is how a user validates it.  The receipt's
    # bdy_mask_sha256 is the digest a registry row pins.
    import json

    from hexcore.cli import build_parser

    grid = _single_nc(_reference_dir() / "cull-x1")
    arguments = build_parser().parse_args(
        ["mesh-check", "--grid-only", "--grid", str(grid)]
    )
    assert arguments.handler(arguments) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["passed"] is True
    assert receipt["static"] is None
    regional = receipt["regional"]
    assert regional["boundary_zone_width"] == 7
    assert regional["euler_characteristic"] == 1
    assert len(regional["bdy_mask_sha256"]) == 64
    assert sum(regional["ring_cell_counts"]) == receipt["dimensions"]["nCells"]


def test_native_culled_grid_with_broken_min_rule_is_refused() -> None:
    grid = _single_nc(_reference_dir() / "cull-x1")
    mesh = Mesh.from_netcdf(grid, validate=False)
    mask_edge = np.asarray(mesh.arrays["bdyMaskEdge"]).copy()
    edge = int(np.flatnonzero(mask_edge == 2)[0])
    mask_edge[edge] = 3
    mesh.arrays["bdyMaskEdge"] = mask_edge
    with pytest.raises(MeshValidationError) as refusal:
        mesh.validate()
    assert "bdyMaskEdge is not the minimum of its present cells" in str(
        refusal.value
    )


# ---------------------------------------------------------------------------
# a GLOBAL mesh carrying the all-zero triple is global (measured 2026-08-26)
# ---------------------------------------------------------------------------
def _with_zero_boundary_masks(mesh: Mesh) -> Mesh:
    """The global sphere as ``rw_mpas_static`` writes it: triple, all zero."""

    arrays = dict(mesh.arrays)
    arrays["bdyMaskCell"] = np.zeros(int(mesh.dimensions["nCells"]), dtype=np.int32)
    arrays["bdyMaskEdge"] = np.zeros(int(mesh.dimensions["nEdges"]), dtype=np.int32)
    arrays["bdyMaskVertex"] = np.zeros(
        int(mesh.dimensions["nVertices"]), dtype=np.int32
    )
    return Mesh(
        arrays=arrays,
        dimensions=dict(mesh.dimensions),
        attrs=dict(mesh.attrs),
    )


def test_a_global_mesh_with_an_all_zero_triple_is_not_a_regional_cull(
    parent_sphere: Mesh,
) -> None:
    """THE BREAKAGE: every generated global mesh refused at load.

    MPAS writes ``bdyMask*`` all-zero on a global mesh -- every element
    interior, no ring at all -- and the unified ``rw_mpas_static`` follows
    that convention, so every static this project GENERATES ships the
    triple.  Classifying on PRESENCE made a closed sphere a bounded disk and
    refused it for being one: "nCells-nEdges+nVertices = 2, not 1" and
    "bdyMask* rings [1..7] are empty".  Measured 2026-08-26 on
    ``v20.80.151649``, the proving RTX 5090: bound clean, then refused at load.
    """

    mesh = _with_zero_boundary_masks(parent_sphere)
    dims = mesh.dimensions
    assert dims["nCells"] - dims["nEdges"] + dims["nVertices"] == 2  # a sphere
    assert mesh.is_regional is False
    mesh.validate()  # the closed sphere admits as the sphere it is


def test_a_populated_triple_still_makes_the_mesh_regional(
    regional_mesh: Mesh,
) -> None:
    """The other direction: a real cull must not be demoted to global.

    Without this arm the fix above could be "never regional" and still pass.
    """

    assert regional_mesh.is_regional is True
    assert int(np.asarray(regional_mesh.bdyMaskCell).max()) == 7
    regional_mesh.validate()


def test_a_global_sphere_with_a_half_written_triple_is_still_refused(
    parent_sphere: Mesh,
) -> None:
    """Incompleteness is refused on PRESENCE, whatever the values.

    A half-written triple cannot say which sentinels are boundary and which
    are corruption, so zero values do not buy it an exemption.
    """

    mesh = _with_zero_boundary_masks(parent_sphere)
    del mesh.arrays["bdyMaskVertex"]
    with pytest.raises(MeshValidationError) as refusal:
        mesh.validate()
    assert "regional boundary masks are incomplete" in str(refusal.value)
