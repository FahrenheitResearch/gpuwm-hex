"""The vectorized host-mesh row laws and deformation-weight donor rows are the scalar code.

``scalar_reference_validation`` below is ``hexcore.cuda_driver._validate_v841_host_mesh``
as it stood before the row loops were vectorized (hex 79225dd), kept verbatim
with its loops run serially.  The vectorized validator must reach the same
verdict with the same message on the whole mesh, on every class-local mesh of
a local-time-stepping plan, and on meshes with one corrupted connectivity entry.
The deformation weights a class-local mesh takes from its source mesh's rows
must be byte-identical to computing every local row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from hexcore import prep_cache
from hexcore.cuda_driver import _authority_member, _refuse_regional_execution, _validate_v841_host_mesh
from hexcore.errors import ConfigurationRefusal
from hexcore.fullstep_lts_classes import build_class_plan, grade_rates
from hexcore.mixing_v841 import initialize_deformation_weights_v841
from hexcore.partition_local_mesh_v841 import build_local_mesh
from test_mesh_validation import graded_sphere_mesh


def _row_chunks_parallel(n_rows, body, label):
    for row in range(n_rows):
        body(row)


def scalar_reference_validation(mesh: Any) -> tuple[int, int, int, int]:
    """hexcore.cuda_driver._validate_v841_host_mesh as of hex 79225dd: the scalar row loops."""

    area_cell = np.asarray(_authority_member(mesh, "areaCell"))
    dc_edge = np.asarray(_authority_member(mesh, "dcEdge"))
    area_triangle = np.asarray(_authority_member(mesh, "areaTriangle"))
    n_cells = int(area_cell.size)
    n_edges = int(dc_edge.size)
    n_vertices = int(area_triangle.size)
    if min(n_cells, n_edges, n_vertices) < 1:
        raise ValueError("the v8.4.1 CUDA mesh dimensions must be non-empty")
    declared = getattr(mesh, "dimensions", {})
    for name, actual in (
        ("nCells", n_cells),
        ("nEdges", n_edges),
        ("nVertices", n_vertices),
    ):
        if name in declared and int(declared[name]) != actual:
            raise ValueError(f"mesh declared {name} differs from array extent")

    def index(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = np.asarray(_authority_member(mesh, name))
        if value.dtype.kind not in "iu":
            raise TypeError(f"{name} must be integer before CUDA upload")
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        return value.astype(np.int64, copy=False)

    # Partition-local meshes (build_local_mesh) carry their layout: rows the
    # brief-1 masks mark INCOMPLETE hold deterministic clamped references
    # (absent neighbour -> local index 0) whose outputs are never consumed as
    # owned truth -- the halo exchange overwrites them.  Topology laws below
    # are therefore asserted on COMPLETE rows only; on a whole mesh (layout
    # None) every row is checked exactly as before.
    _layout = getattr(mesh, "partition_layout", None)
    if _layout is None:
        _cell_ok = _edge_ok = _vertex_ok = None
    else:
        _cell_ok = np.asarray(_layout.cell_row_complete, dtype=bool)
        _edge_ok = np.asarray(_layout.edge_row_complete, dtype=bool)
        _vertex_ok = np.asarray(_layout.vertex_row_complete, dtype=bool)
        for mask, extent, what in (
            (_cell_ok, n_cells, "cell"),
            (_edge_ok, n_edges, "edge"),
            (_vertex_ok, n_vertices, "vertex"),
        ):
            if mask.shape != (extent,):
                raise ValueError(
                    f"partition {what} completeness mask does not match the mesh"
                )

    def _row_ok(mask: np.ndarray | None, row: int) -> bool:
        return mask is None or bool(mask[row])

    cells_on_edge = index("cellsOnEdge", (n_edges, 2))
    if np.any(cells_on_edge >= n_cells):
        raise ConfigurationRefusal(
            "cellsOnEdge",
            "out of range",
            "v8.4.1 CUDA requires every cellsOnEdge entry inside [0,nCells)",
            "a mesh whose connectivity indexes its own cells",
        )
    if np.any(cells_on_edge < 0):
        # A negative entry here is the regional cull's stored-0 sentinel on a
        # ring-7 row: a cull on the global path (see
        # _refuse_regional_execution for why that cannot run).
        _refuse_regional_execution(mesh, n_cells)
    _same_endpoints = cells_on_edge[:, 0] == cells_on_edge[:, 1]
    if _edge_ok is not None:
        _same_endpoints = _same_endpoints & _edge_ok
    if np.any(_same_endpoints):
        raise ValueError("cellsOnEdge endpoints must be distinct")
    raw_edges_on_cell = np.asarray(_authority_member(mesh, "edgesOnCell"))
    if raw_edges_on_cell.ndim != 2:
        raise ValueError("edgesOnCell must have shape (nCells,maxEdges)")
    max_edges = int(raw_edges_on_cell.shape[1])
    if "maxEdges" in declared and int(declared["maxEdges"]) != max_edges:
        raise ValueError("mesh declared maxEdges differs from edgesOnCell")
    edges_on_cell = index("edgesOnCell", (n_cells, max_edges))
    counts_cell = index("nEdgesOnCell", (n_cells,))
    if np.any(counts_cell < 0) or np.any(counts_cell > max_edges):
        raise ValueError("nEdgesOnCell is outside [0,maxEdges]")
    cells_on_cell = index("cellsOnCell", (n_cells, max_edges))
    vertices_on_cell = index("verticesOnCell", (n_cells, max_edges))
    def cell_rows(cell):
        if not _row_ok(_cell_ok, cell):
            return
        count = int(counts_cell[cell])
        if (
            np.any(edges_on_cell[cell, :count] < 0)
            or np.any(edges_on_cell[cell, :count] >= n_edges)
        ):
            raise ValueError("active edgesOnCell entry is outside [0,nEdges)")
        if (
            np.any(cells_on_cell[cell, :count] < 0)
            or np.any(cells_on_cell[cell, :count] >= n_cells)
        ):
            raise ValueError("active cellsOnCell entry is outside [0,nCells)")
        if (
            np.any(vertices_on_cell[cell, :count] < 0)
            or np.any(vertices_on_cell[cell, :count] >= n_vertices)
        ):
            raise ValueError("active verticesOnCell entry is outside [0,nVertices)")
        for name, row in (
            ("edgesOnCell", edges_on_cell[cell, :count]),
            ("cellsOnCell", cells_on_cell[cell, :count]),
            ("verticesOnCell", vertices_on_cell[cell, :count]),
        ):
            if np.unique(row).size != count:
                raise ValueError(f"active {name} entries must be unique per cell")
        for slot in range(count):
            edge = int(edges_on_cell[cell, slot])
            endpoints = cells_on_edge[edge]
            if cell not in endpoints:
                raise ValueError("active edgesOnCell entry is not incident to its cell")
            opposite = int(endpoints[1] if int(endpoints[0]) == cell else endpoints[0])
            if int(cells_on_cell[cell, slot]) != opposite:
                raise ValueError("cellsOnCell is not the opposite cell on edgesOnCell")

    _row_chunks_parallel(n_cells, cell_rows, "mesh-cells")

    raw_edges_on_edge = np.asarray(_authority_member(mesh, "edgesOnEdge"))
    if raw_edges_on_edge.ndim != 2:
        raise ValueError("edgesOnEdge must have shape (nEdges,maxEdges2)")
    max_edges2 = int(raw_edges_on_edge.shape[1])
    if "maxEdges2" in declared and int(declared["maxEdges2"]) != max_edges2:
        raise ValueError("mesh declared maxEdges2 differs from edgesOnEdge")
    edges_on_edge = index("edgesOnEdge", (n_edges, max_edges2))
    counts_edge = index("nEdgesOnEdge", (n_edges,))
    if np.any(counts_edge < 0) or np.any(counts_edge > max_edges2):
        raise ValueError("nEdgesOnEdge is outside [0,maxEdges2]")
    def edge_rows(edge):
        if not _row_ok(_edge_ok, edge):
            return
        count = int(counts_edge[edge])
        if (
            np.any(edges_on_edge[edge, :count] < 0)
            or np.any(edges_on_edge[edge, :count] >= n_edges)
        ):
            raise ValueError("active edgesOnEdge entry is outside [0,nEdges)")
        neighbors = edges_on_edge[edge, :count]
        if np.any(neighbors == edge) or np.unique(neighbors).size != count:
            raise ValueError("active edgesOnEdge neighbors must be unique and non-self")
        endpoints = set(map(int, cells_on_edge[edge]))
        for neighbor in neighbors:
            if not endpoints.intersection(map(int, cells_on_edge[int(neighbor)])):
                raise ValueError("edgesOnEdge neighbors must share a cell endpoint")
        expected_neighbors: set[int] = set()
        for cell in endpoints:
            cell_count = int(counts_cell[cell])
            expected_neighbors.update(
                map(int, edges_on_cell[cell, :cell_count])
            )
        expected_neighbors.discard(edge)
        if set(map(int, neighbors)) != expected_neighbors:
            raise ValueError(
                "edgesOnEdge must contain the complete endpoint-cell edge union"
            )

    _row_chunks_parallel(n_edges, edge_rows, "mesh-edges")

    vertices_on_edge = index("verticesOnEdge", (n_edges, 2))
    if np.any(vertices_on_edge < 0) or np.any(vertices_on_edge >= n_vertices):
        raise ValueError("verticesOnEdge is outside [0,nVertices)")
    _same_vertices = vertices_on_edge[:, 0] == vertices_on_edge[:, 1]
    if _edge_ok is not None:
        _same_vertices = _same_vertices & _edge_ok
    if np.any(_same_vertices):
        raise ValueError("verticesOnEdge endpoints must be distinct")
    edges_on_vertex = index("edgesOnVertex", (n_vertices, 3))
    cells_on_vertex = index("cellsOnVertex", (n_vertices, 3))
    if "vertexDegree" in declared and int(declared["vertexDegree"]) != 3:
        raise ValueError("v8.4.1 CUDA requires mesh vertexDegree=3")
    if np.any(edges_on_vertex < 0) or np.any(edges_on_vertex >= n_edges):
        raise ValueError("v8.4.1 CUDA requires exactly three valid edges per vertex")
    if np.any(cells_on_vertex < 0) or np.any(cells_on_vertex >= n_cells):
        raise ValueError("v8.4.1 CUDA requires exactly three valid cells per vertex")
    def vertex_rows(vertex):
        if not _row_ok(_vertex_ok, vertex):
            return
        if np.unique(edges_on_vertex[vertex]).size != 3:
            raise ValueError("edgesOnVertex entries must be distinct")
        if np.unique(cells_on_vertex[vertex]).size != 3:
            raise ValueError("cellsOnVertex entries must be distinct")
        for edge in edges_on_vertex[vertex]:
            if vertex not in vertices_on_edge[int(edge)]:
                raise ValueError("edgesOnVertex is not incident through verticesOnEdge")

    _row_chunks_parallel(n_vertices, vertex_rows, "mesh-vertices")

    def reciprocal_rows(edge):
        if not _row_ok(_edge_ok, edge):
            return
        for vertex in vertices_on_edge[edge]:
            if not _row_ok(_vertex_ok, int(vertex)):
                continue
            if edge not in edges_on_vertex[int(vertex)]:
                raise ValueError("verticesOnEdge is not reciprocal with edgesOnVertex")

    _row_chunks_parallel(n_edges, reciprocal_rows, "mesh-reciprocal")

    def kite_rows(cell):
        if not _row_ok(_cell_ok, cell):
            return
        for slot in range(int(counts_cell[cell])):
            vertex = int(vertices_on_cell[cell, slot])
            if not np.any(cells_on_vertex[vertex] == cell):
                raise ValueError(
                    "every active cell-vertex pair requires a kite-area association"
                )

    _row_chunks_parallel(n_cells, kite_rows, "mesh-kites")

    float_shapes = {
        "weightsOnEdge": (n_edges, max_edges2),
        "dcEdge": (n_edges,),
        "dvEdge": (n_edges,),
        "areaCell": (n_cells,),
        "areaTriangle": (n_vertices,),
        "kiteAreasOnVertex": (n_vertices, 3),
        "angleEdge": (n_edges,),
        "fVertex": (n_vertices,),
        "fEdge": (n_edges,),
        "latCell": (n_cells,),
        "lonCell": (n_cells,),
        "latEdge": (n_edges,),
        "lonEdge": (n_edges,),
        "meshDensity": (n_cells,),
        "defc_a": (n_cells, max_edges),
        "defc_b": (n_cells, max_edges),
    }
    for name, shape in float_shapes.items():
        value = np.asarray(_authority_member(mesh, name))
        if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError(f"{name} must use float32 or float64 source storage")
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous before CUDA upload")
        if name in ("dcEdge", "dvEdge", "areaCell", "areaTriangle"):
            rounded = value.astype(np.float32, copy=False)
            if np.any(~np.isfinite(rounded)) or np.any(rounded <= 0.0):
                raise ValueError(
                    f"{name} must remain finite and positive after float32 rounding"
                )
    specified = _authority_member(mesh, "spec_zone_mask_edge")
    if specified is not None:
        value = np.asarray(specified)
        if value.shape != (n_edges,) or not np.all(np.isfinite(value)):
            raise ValueError(
                "spec_zone_mask_edge must be finite with shape (nEdges,)"
            )
        if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("spec_zone_mask_edge must use float32 or float64")
        if not value.flags.c_contiguous:
            raise ValueError("spec_zone_mask_edge must be C-contiguous")
        if np.any(value != 0.0):
            _refuse_regional_execution(mesh, n_cells)
    nominal = _authority_member(mesh, "nominalMinDc")
    if nominal is None:
        nominal = _authority_member(mesh, "nominal_min_dc")
    if nominal is None:
        raise ValueError("mesh.nominalMinDc must be present")
    nominal_array = np.asarray(nominal)
    if nominal_array.shape != ():
        raise ValueError("mesh.nominalMinDc must be scalar")
    if nominal_array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("mesh.nominalMinDc must use float32 or float64")
    nominal_rkind = np.float32(nominal_array)
    if not np.isfinite(nominal_rkind) or nominal_rkind <= np.float32(0.0):
        raise ValueError(
            "mesh.nominalMinDc must remain finite and positive in float32"
        )
    return n_cells, n_edges, n_vertices, max_edges


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "prep"
    monkeypatch.setenv(prep_cache.ENVIRONMENT, str(root))
    prep_cache._memory.clear()
    prep_cache._ledger.clear()
    yield root
    prep_cache._memory.clear()
    prep_cache._ledger.clear()


def _whole_mesh():
    mesh = graded_sphere_mesh(level=3)
    n_cells = int(np.asarray(mesh.arrays["nEdgesOnCell"]).size)
    n_edges = int(np.asarray(mesh.arrays["cellsOnEdge"]).shape[0])
    n_vertices = int(np.asarray(mesh.arrays["cellsOnVertex"]).shape[0])
    max_edges = int(np.asarray(mesh.arrays["edgesOnCell"]).shape[1])
    fill = {"fVertex": (n_vertices,), "fEdge": (n_edges,), "meshDensity": (n_cells,), "angleEdge": (n_edges,),
            "defc_a": (n_cells, max_edges), "defc_b": (n_cells, max_edges)}
    for name, shape in fill.items():
        mesh.arrays.setdefault(name, np.ones(shape, dtype=np.float32))
    mesh.attrs.setdefault("nominalMinDc", 1.0)
    mesh.attrs.setdefault("on_a_sphere", "YES")
    return mesh


def _class_meshes(mesh, seed=0):
    coe = np.asarray(mesh.arrays["cellsOnEdge"], dtype=np.int64)
    n_cells = int(np.asarray(mesh.arrays["nEdgesOnCell"]).size)
    rng = np.random.default_rng(seed)
    rates = np.full(n_cells, 8, dtype=np.int32)
    rates[rng.choice(n_cells, n_cells // 3, replace=False)] = 1
    rates = grade_rates(rates, coe)
    plan = build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]), cells_on_edge=coe,
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]),
        area_cell=np.asarray(mesh.arrays["areaCell"], dtype=np.float64),
        dc_edge_m=np.full(coe.shape[0], 1.0), base_dt=1.0,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"))
    return [build_local_mesh(mesh, tables.layout) for tables in plan.classes]


def _verdict(function, mesh):
    try:
        return ("passed", tuple(int(v) for v in function(mesh)))
    except (ValueError, TypeError, ConfigurationRefusal) as error:
        return (type(error).__name__, str(error))


def _vectorized(mesh):
    return _validate_v841_host_mesh.__wrapped_prep_cache__(mesh)


@pytest.fixture
def parallel_rows(monkeypatch):
    from hexcore import host_parallel

    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 32)


def test_same_verdict_on_whole_and_class_local_meshes(parallel_rows):
    mesh = _whole_mesh()
    meshes = [mesh] + _class_meshes(mesh)
    assert len(meshes) >= 3
    for each in meshes:
        assert _verdict(_vectorized, each) == _verdict(scalar_reference_validation, each)
        assert _verdict(_vectorized, each)[0] == "passed"


_TABLES = ("edgesOnCell", "cellsOnCell", "verticesOnCell", "nEdgesOnCell", "edgesOnEdge",
           "nEdgesOnEdge", "edgesOnVertex", "cellsOnVertex", "verticesOnEdge", "cellsOnEdge")


@pytest.mark.parametrize("table", _TABLES)
def test_same_message_on_corrupted_tables(table, parallel_rows):
    mesh = _whole_mesh()
    local = _class_meshes(mesh, seed=1)
    rng = np.random.default_rng(abs(hash(table)) % 2**32)
    for target in [mesh] + local:
        original = np.asarray(target.arrays[table])
        for trial in range(12):
            corrupted = np.array(original, copy=True)
            flat = corrupted.reshape(-1)
            position = int(rng.integers(flat.size))
            choice = trial % 4
            if choice == 0:
                flat[position] = flat[int(rng.integers(flat.size))]
            elif choice == 1:
                flat[position] = flat[position] + 1
            elif choice == 2:
                flat[position] = -1
            else:
                flat[position] = int(original.max()) + 5
            target.arrays[table] = corrupted
            try:
                assert _verdict(_vectorized, target) == _verdict(scalar_reference_validation, target)
            finally:
                target.arrays[table] = original


def _bits(array):
    array = np.asarray(array)
    return array.dtype.str, array.shape, array.tobytes()


def test_donor_rows_are_the_local_computation(cache_root):
    mesh = _whole_mesh()
    for local in _class_meshes(mesh, seed=2):
        donor = initialize_deformation_weights_v841.__wrapped_prep_cache__(local)
        alone = build_local_mesh(mesh, local.partition_layout)
        del alone.partition_source_mesh
        full = initialize_deformation_weights_v841.__wrapped_prep_cache__(alone)
        for name in ("coef_c2", "coef_s2", "coef_cs"):
            assert _bits(getattr(donor, name)) == _bits(getattr(full, name))
