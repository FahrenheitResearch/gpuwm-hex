"""Test-only scalar authorities retained from the replaced host path."""
from __future__ import annotations
import numpy as np
from hexcore.terrain import (TerrainCoupling, ArrayLayout, _layout, _integer_array,
    _float_array, _same_dtype)

def build_terrain_coupling(
    *,
    n_edges_on_cell: object,
    edges_on_cell: object,
    cells_on_edge: object,
    zb: object,
    zb3: object,
    config_coef_3rd_order: float,
    array_layout: ArrayLayout = "logical",
    allow_regional_sentinels: bool = False,
) -> TerrainCoupling:
    """Build the frozen cell-oriented slope metrics.

    With ``array_layout="logical"``, connectivity is canonical zero-based and
    ``zb``/``zb3`` have shape ``(nVertLevels+1, 2, nEdges)``.  With
    ``array_layout="netcdf"``, connectivity is native one-based and the metric
    shape is ``(nEdges, 2, nVertLevels+1)``.
    """

    layout = _layout(array_layout)
    counts_source = _integer_array("n_edges_on_cell", n_edges_on_cell, ndim=1)
    eoc_source = _integer_array("edges_on_cell", edges_on_cell, ndim=2)
    coe_source = _integer_array("cells_on_edge", cells_on_edge, ndim=2)
    raw_zb = _float_array("zb", zb, ndim=3)
    raw_zb3 = _float_array("zb3", zb3, ndim=3)
    _same_dtype("zb", raw_zb, zb3=raw_zb3)

    if not np.isfinite(config_coef_3rd_order):
        raise ValueError("config_coef_3rd_order must be finite")
    if eoc_source.shape[0] != counts_source.size:
        raise ValueError("edges_on_cell and n_edges_on_cell nCells differ")
    if coe_source.shape[1] != 2:
        raise ValueError("cells_on_edge must have shape (nEdges, 2)")

    n_cells, max_edges = eoc_source.shape
    n_edges = coe_source.shape[0]
    counts = np.asarray(counts_source, dtype=np.int64).copy()
    if np.any(counts < 0) or np.any(counts > max_edges):
        raise ValueError("n_edges_on_cell entries must be in [0, maxEdges]")

    if layout == "netcdf":
        if raw_zb.shape[:2] != (n_edges, 2):
            raise ValueError(
                f"zb native shape {raw_zb.shape} must start with ({n_edges}, 2)"
            )
        if raw_zb3.shape != raw_zb.shape:
            raise ValueError("zb3 shape differs from zb")
        logical_zb = np.transpose(raw_zb, (2, 1, 0))
        logical_zb3 = np.transpose(raw_zb3, (2, 1, 0))
        eoc = np.asarray(eoc_source, dtype=np.int64) - 1
        coe = np.asarray(coe_source, dtype=np.int64) - 1
    else:
        if raw_zb.shape[1:] != (2, n_edges):
            raise ValueError(
                f"zb logical shape {raw_zb.shape} must end in (2, {n_edges})"
            )
        if raw_zb3.shape != raw_zb.shape:
            raise ValueError("zb3 shape differs from zb")
        logical_zb = raw_zb
        logical_zb3 = raw_zb3
        eoc = np.asarray(eoc_source, dtype=np.int64).copy()
        coe = np.asarray(coe_source, dtype=np.int64).copy()

    if allow_regional_sentinels:
        # Ring-7 one-cell boundary edges store 0 (loaded as a negative
        # sentinel) in one cellsOnEdge slot; native MPAS maps it to the
        # garbage cell nCells+1 at read time.  The side test below still
        # finds the present cell in the other slot, which is all zb_cell
        # coupling needs.
        coe = np.where(coe < 0, n_cells, coe)
        if np.any(coe > n_cells):
            raise ValueError("cells_on_edge contains an out-of-range cell")
    elif np.any(coe < 0) or np.any(coe >= n_cells):
        raise ValueError("cells_on_edge contains a missing or out-of-range cell")
    if np.any(coe[:, 0] == coe[:, 1]):
        raise ValueError("cells_on_edge contains an edge with the same cell twice")

    dtype = raw_zb.dtype
    n_levels_p1 = logical_zb.shape[0]
    signs = np.zeros((n_cells, max_edges), dtype=dtype)
    zb_cell = np.zeros((n_levels_p1, n_cells, max_edges), dtype=dtype)
    zb3_cell = np.zeros_like(zb_cell)
    canonical_eoc = np.full((n_cells, max_edges), -1, dtype=np.int64)
    coefficient = dtype.type(config_coef_3rd_order)

    for cell in range(n_cells):
        used: set[int] = set()
        for slot in range(int(counts[cell])):
            edge = int(eoc[cell, slot])
            if edge < 0 or edge >= n_edges:
                raise ValueError(
                    f"active edges_on_cell[{cell}, {slot}]={edge} is out of range"
                )
            if edge in used:
                raise ValueError(f"cell {cell} lists edge {edge} more than once")
            used.add(edge)
            canonical_eoc[cell, slot] = edge
            if int(coe[edge, 0]) == cell:
                side = 0
                signs[cell, slot] = dtype.type(1.0)
            elif int(coe[edge, 1]) == cell:
                side = 1
                signs[cell, slot] = dtype.type(-1.0)
            else:
                raise ValueError(
                    f"cell {cell} is not present in cells_on_edge for edge {edge}"
                )
            zb_cell[:, cell, slot] = logical_zb[:, side, edge]
            # Frozen atm_couple_coef_3rd_order scales zb3_cell after copying.
            zb3_cell[:, cell, slot] = coefficient * logical_zb3[:, side, edge]

    return TerrainCoupling(
        n_edges_on_cell=counts,
        edges_on_cell=canonical_eoc,
        cells_on_edge=np.asarray(coe, dtype=np.int64).copy(),
        edges_on_cell_sign=signs,
        zb_cell=zb_cell,
        zb3_cell=zb3_cell,
        config_coef_3rd_order=float(config_coef_3rd_order),
    )

def topology_errors(*, n_vertices, max_edges2, vertex_degree, regional,
                    counts, edges, cells_on_cell, vertices_on_cell, cells_on_edge,
                    vertices_on_edge, cells_on_vertex, edges_on_vertex,
                    edge_counts, edges_on_edge):
    n_cells, max_edges = edges.shape
    n_edges = cells_on_edge.shape[0]
    n_edges_on_cell, edges_on_cell = counts, edges
    n_edges_on_edge = edge_counts
    errors = []
    bad_cell_rows: list[int] = []
    for cell in range(n_cells):
        count = int(n_edges_on_cell[cell])
        row_edges = edges_on_cell[cell, :count]
        row_cells = cells_on_cell[cell, :count]
        row_vertices = vertices_on_cell[cell, :count]
        # On a regional ring-7 cell several neighbour slots hold the
        # -1 sentinel; uniqueness applies to the present ones.  On a
        # global mesh every used slot is present (range-checked
        # above), so this is the original exact check.
        present_cells = row_cells[row_cells >= 0]
        if (
            np.unique(row_edges).size != count
            or np.unique(present_cells).size != present_cells.size
            or np.unique(row_vertices).size != count
        ):
            bad_cell_rows.append(cell)
            continue
        for slot, edge in enumerate(row_edges):
            edge_cells = cells_on_edge[edge]
            if cell not in edge_cells:
                bad_cell_rows.append(cell)
                break
            other = edge_cells[1] if edge_cells[0] == cell else edge_cells[0]
            if row_cells[slot] != other:
                bad_cell_rows.append(cell)
                break
            endpoints = vertices_on_edge[edge]
            next_slot = (slot + 1) % count
            if row_vertices[slot] not in endpoints or row_vertices[next_slot] not in endpoints:
                bad_cell_rows.append(cell)
                break
    if bad_cell_rows:
        errors.append(
            "cell edge/neighbor/vertex slot reciprocity fails at cells "
            + ", ".join(str(cell) for cell in bad_cell_rows[:5])
        )

    bad_edges: list[int] = []
    for edge in range(n_edges):
        edge_cells = cells_on_edge[edge]
        endpoints = vertices_on_edge[edge]
        if edge_cells[0] == edge_cells[1] or endpoints[0] == endpoints[1]:
            bad_edges.append(edge)
            continue
        # A regional ring-7 edge may have one absent cell (-1); the
        # reciprocity a missing cell cannot express is exempt, the
        # present side is still required to list the edge.
        if any(
            edge not in edges_on_cell[cell, : n_edges_on_cell[cell]]
            for cell in edge_cells
            if cell >= 0
        ):
            bad_edges.append(edge)
            continue
        if any(edge not in edges_on_vertex[vertex] for vertex in endpoints):
            bad_edges.append(edge)
    if bad_edges:
        errors.append(
            "edge-to-cell/vertex reciprocity fails at edges "
            + ", ".join(str(edge) for edge in bad_edges[:5])
        )

    bad_vertices: list[int] = []
    for vertex in range(n_vertices):
        row_cells = cells_on_vertex[vertex]
        row_edges = edges_on_vertex[vertex]
        # Regional ring-7 vertices hold -1 for culled cells/edges;
        # uniqueness and reciprocity apply to the present entries.
        # Global rows are fully present, so this is the original
        # exact check there.
        present_cells = row_cells[row_cells >= 0]
        present_edges = row_edges[row_edges >= 0]
        if (
            np.unique(present_cells).size != present_cells.size
            or np.unique(present_edges).size != present_edges.size
        ):
            bad_vertices.append(vertex)
            continue
        if any(
            vertex
            not in vertices_on_cell[cell, : n_edges_on_cell[cell]]
            for cell in present_cells
        ) or any(vertex not in vertices_on_edge[edge] for edge in present_edges):
            bad_vertices.append(vertex)
    if bad_vertices:
        errors.append(
            "vertex-to-cell/edge reciprocity fails at vertices "
            + ", ".join(str(vertex) for vertex in bad_vertices[:5])
        )

    bad_stencils: list[int] = []
    for edge in range(n_edges):
        actual = edges_on_edge[edge, : n_edges_on_edge[edge]]
        if regional and (
            np.any(actual == -1) or np.any(cells_on_edge[edge] == -1)
        ):
            # A ring-7 edge bordering the cut: its parent stencil
            # reached culled edges (now -1 inside the unshrunk row)
            # and possibly retained edges of the culled cell, so the
            # exact-set identity is undefined.  What stays required:
            # no self-reference and no duplicate present entry.
            present = actual[actual >= 0]
            if edge in present or np.unique(present).size != present.size:
                bad_stencils.append(edge)
            continue
        expected: set[int] = set()
        for cell in cells_on_edge[edge]:
            expected.update(
                int(value)
                for value in edges_on_cell[cell, : n_edges_on_cell[cell]]
            )
        expected.discard(edge)
        if (
            edge in actual
            or np.unique(actual).size != actual.size
            or set(int(value) for value in actual) != expected
        ):
            bad_stencils.append(edge)
    if bad_stencils:
        errors.append(
            "edgesOnEdge tangential stencil is invalid at edges "
            + ", ".join(str(edge) for edge in bad_stencils[:5])
        )

    visited = np.zeros(n_cells, dtype=bool)
    stack = [0]
    visited[0] = True
    while stack:
        cell = stack.pop()
        for neighbor in cells_on_cell[cell, : n_edges_on_cell[cell]]:
            # The >= 0 guard is the same wrap family as the kite fix
            # below: visited[-1] silently reads/marks the LAST cell.
            if neighbor >= 0 and not visited[neighbor]:
                visited[neighbor] = True
                stack.append(int(neighbor))
    if not np.all(visited):
        errors.append(
            f"cell graph is disconnected ({np.count_nonzero(~visited)} unvisited cells)"
        )

    return errors


def correct_vertical_momentum(*, counts, edges, fzm, fzp, ru,
        interface_zz, signs, zb_cell, zb3_cell, rw):
    rw = rw.copy()
    for cell in range(rw.shape[1]):
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            for level in range(1, ru.shape[0]):
                flux = (
                    fzm[level] * ru[level, edge]
                    + fzp[level] * ru[level - 1, edge]
                )
                rw[level, cell] -= (
                    signs[cell, slot]
                    * (
                        zb_cell[level, cell, slot]
                        + np.copysign(rw.dtype.type(1.0), flux)
                        * zb3_cell[level, cell, slot]
                    )
                    * flux
                    * interface_zz[level, cell]
                )

    return rw
