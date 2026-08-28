"""Readable NumPy authority for the basic MPAS C-grid operators.

The entity dimension may occur anywhere when ``axis`` is supplied.  Without
it, the authority detects the first or last dimension, so both MPAS/Fortran
``(levels, entity)`` and NumPy-friendly ``(entity, levels)`` layouts work.
Every operator returns the entity dimension in the same position and preserves
float32 or float64 arithmetic.

Frozen source citations are attached to each operator.  Cell-to-vertex
interpolation is a mesh-discretization identity rather than a standalone
atmosphere routine; its citation is labeled accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _mesh_dimension(mesh: object, name: str, fallback_field: str) -> int:
    try:
        return int(getattr(mesh, name))
    except AttributeError:
        dimensions = getattr(mesh, "dimensions", None)
        if dimensions is not None and name in dimensions:
            return int(dimensions[name])
        return int(_mesh_array(mesh, fallback_field).shape[0])


def _looks_like_mesh(value: object, field: str) -> bool:
    if hasattr(value, field):
        return True
    arrays = getattr(value, "arrays", None)
    return arrays is not None and field in arrays


def _allow_mesh_first(
    field: object, mesh: object, required_mesh_field: str
) -> tuple[object, object]:
    """Accept both ``operator(field, mesh)`` and ``operator(mesh, field)``."""

    if _looks_like_mesh(field, required_mesh_field) and not _looks_like_mesh(
        mesh, required_mesh_field
    ):
        return mesh, field
    return field, mesh


def _authority_field(field: object) -> NDArray[np.floating[Any]]:
    array = np.asarray(field)
    if array.dtype == np.dtype(np.float32) or array.dtype == np.dtype(np.float64):
        return array
    if array.dtype.kind not in "biuf":
        raise TypeError("MPAS horizontal fields must be real numeric arrays")
    return np.asarray(array, dtype=np.float64)


def _entity_to_front(
    field: object, entity_count: int, axis: int | None
) -> tuple[NDArray[np.floating[Any]], int]:
    array = _authority_field(field)
    if array.ndim == 0:
        raise ValueError("an MPAS horizontal field must have an entity dimension")

    if axis is None:
        if array.ndim == 1:
            axis = 0
        else:
            first = array.shape[0] == entity_count
            last = array.shape[-1] == entity_count
            if first and not last:
                axis = 0
            elif last and not first:
                axis = array.ndim - 1
            elif first and last:
                raise ValueError(
                    "entity axis is ambiguous because both first and last dimensions "
                    f"have length {entity_count}; pass axis explicitly"
                )
            else:
                raise ValueError(
                    f"neither first nor last field dimension has entity length {entity_count}; "
                    "pass axis explicitly"
                )
    normalized_axis = int(axis)
    if normalized_axis < 0:
        normalized_axis += array.ndim
    if normalized_axis < 0 or normalized_axis >= array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for a {array.ndim}-D field")
    if array.shape[normalized_axis] != entity_count:
        raise ValueError(
            f"field axis {normalized_axis} has length {array.shape[normalized_axis]}, "
            f"expected {entity_count}"
        )
    return np.moveaxis(array, normalized_axis, 0), normalized_axis


def _restore_entity_axis(
    front: NDArray[np.floating[Any]], axis: int
) -> NDArray[np.floating[Any]]:
    return np.moveaxis(front, 0, axis)


def _factor_for_levels(factor: NDArray[Any], field_ndim: int) -> NDArray[Any]:
    return factor.reshape(factor.shape + (1,) * (field_ndim - factor.ndim))


def edge_sign_on_cell(
    mesh: object, *, dtype: np.dtype[Any] | type[Any] = np.int8
) -> NDArray[Any]:
    """Return the signed incidence of every used edge on every cell.

    ``+1`` denotes the first cell in ``cellsOnEdge`` and ``-1`` the second;
    padding is zero.  This is a direct transcription of frozen MPAS-A v8.2.3
    ``src/core_atmosphere/mpas_atm_core.F:1187-1203``.
    """

    cells_on_edge = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    edges_on_cell = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    n_cells = edges_on_cell.shape[0]
    signs = np.zeros(edges_on_cell.shape, dtype=dtype)

    for cell in range(n_cells):
        count = int(counts[cell])
        edges = edges_on_cell[cell, :count]
        if np.any((edges < 0) | (edges >= cells_on_edge.shape[0])):
            raise ValueError(f"edgesOnCell contains an invalid used edge for cell {cell}")
        first = cells_on_edge[edges, 0] == cell
        second = cells_on_edge[edges, 1] == cell
        if not np.all(first | second) or np.any(first & second):
            raise ValueError(f"cellsOnEdge is not reciprocal for cell {cell}")
        signs[cell, :count] = np.where(first, 1, -1).astype(dtype, copy=False)
    return signs


def edge_sign_on_vertex(
    mesh: object, *, dtype: np.dtype[Any] | type[Any] = np.int8
) -> NDArray[Any]:
    """Return the signed incidence of every edge endpoint on every vertex.

    The first endpoint receives ``-1`` and the second ``+1``.  This directly
    transcribes frozen MPAS-A v8.2.3
    ``src/core_atmosphere/mpas_atm_core.F:1173-1185``.
    """

    vertices_on_edge = np.asarray(_mesh_array(mesh, "verticesOnEdge"), dtype=np.int64)
    edges_on_vertex = np.asarray(_mesh_array(mesh, "edgesOnVertex"), dtype=np.int64)
    n_vertices, vertex_degree = edges_on_vertex.shape
    signs = np.zeros(edges_on_vertex.shape, dtype=dtype)

    for vertex in range(n_vertices):
        edges = edges_on_vertex[vertex, :vertex_degree]
        sentinel = edges < 0
        if np.any(edges >= vertices_on_edge.shape[0]):
            raise ValueError(f"edgesOnVertex contains an invalid edge for vertex {vertex}")
        if np.any(sentinel):
            # Regional ring-7 vertex rows store 0 (loaded as a negative
            # sentinel) in absent-edge slots.  Native maps them to the
            # garbage edge, whose verticesOnEdge row holds zeros, so
            # atm_compute_signs (mpas_atm_core.F:1172-1184) lands on the
            # else branch: the sign is -1.  Every consumer multiplies it by
            # the garbage edge's zero dcEdge, so the lane is inert.
            present = edges[~sentinel]
            first = vertices_on_edge[present, 0] == vertex
            second = vertices_on_edge[present, 1] == vertex
            if not np.all(first | second) or np.any(first & second):
                raise ValueError(
                    f"verticesOnEdge is not reciprocal for vertex {vertex}"
                )
            row = np.full(vertex_degree, -1, dtype=np.int64)
            row[~sentinel] = np.where(second, 1, -1)
            signs[vertex] = row.astype(dtype, copy=False)
            continue
        first = vertices_on_edge[edges, 0] == vertex
        second = vertices_on_edge[edges, 1] == vertex
        if not np.all(first | second) or np.any(first & second):
            raise ValueError(f"verticesOnEdge is not reciprocal for vertex {vertex}")
        signs[vertex] = np.where(second, 1, -1).astype(dtype, copy=False)
    return signs


def kite_index_on_cell(mesh: object) -> NDArray[np.int64]:
    """Map each used ``verticesOnCell`` slot to its kite-area slot.

    The result is zero-based with ``-1`` padding.  It transcribes
    ``kiteForCell`` construction in frozen MPAS-A v8.2.3
    ``src/core_atmosphere/mpas_atm_core.F:1205-1222``.
    """

    vertices_on_cell = np.asarray(_mesh_array(mesh, "verticesOnCell"), dtype=np.int64)
    cells_on_vertex = np.asarray(_mesh_array(mesh, "cellsOnVertex"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    result = np.full(vertices_on_cell.shape, -1, dtype=np.int64)
    for cell in range(vertices_on_cell.shape[0]):
        for slot in range(int(counts[cell])):
            vertex = int(vertices_on_cell[cell, slot])
            matches = np.flatnonzero(cells_on_vertex[vertex] == cell)
            if matches.size != 1:
                raise ValueError(
                    f"cellsOnVertex has {matches.size} matches for cell {cell}, vertex {vertex}"
                )
            result[cell, slot] = int(matches[0])
    return result


def edge_scalar_gradient(
    cell_scalar: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Normal gradient of a cell scalar at edges.

    ``(phi[cell2] - phi[cell1]) / dcEdge`` transcribes the normal-gradient
    expression in frozen MPAS-A v8.2.3
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5787-5794``.
    """

    cell_scalar, mesh = _allow_mesh_first(cell_scalar, mesh, "cellsOnEdge")
    n_cells = _mesh_dimension(mesh, "nCells", "cellsOnCell")
    field, original_axis = _entity_to_front(cell_scalar, n_cells, axis)
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if np.any((cells < 0) | (cells >= n_cells)):
        raise ValueError("edge scalar gradient requires two valid cells on every edge")
    dc_edge = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=field.dtype)
    factor = _factor_for_levels(dc_edge, field.ndim)
    result = (field[cells[:, 1]] - field[cells[:, 0]]) / factor
    return _restore_entity_axis(result, original_axis)


def edge_to_cell_divergence(
    edge_normal: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Finite-volume divergence of edge-normal values at cell centers.

    This transcribes frozen MPAS-A v8.2.3
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5602-5617``.
    """

    edge_normal, mesh = _allow_mesh_first(edge_normal, mesh, "edgesOnCell")
    n_edges = _mesh_dimension(mesh, "nEdges", "cellsOnEdge")
    field, original_axis = _entity_to_front(edge_normal, n_edges, axis)
    edges = np.asarray(_mesh_array(mesh, "edgesOnCell"), dtype=np.int64)
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    used = np.arange(edges.shape[1])[None, :] < counts[:, None]
    if np.any((edges[used] < 0) | (edges[used] >= n_edges)):
        raise ValueError("edgesOnCell contains an invalid used edge")
    safe_edges = np.where(used, edges, 0)
    signs = edge_sign_on_cell(mesh, dtype=field.dtype)
    dv_edge = np.asarray(_mesh_array(mesh, "dvEdge"), dtype=field.dtype)
    area_cell = np.asarray(_mesh_array(mesh, "areaCell"), dtype=field.dtype)
    factors = signs * dv_edge[safe_edges]
    gathered = field[safe_edges]
    result = np.sum(
        gathered * _factor_for_levels(factors, gathered.ndim),
        axis=1,
        dtype=field.dtype,
    )
    result /= _factor_for_levels(area_cell, result.ndim)
    return _restore_entity_axis(result, original_axis)


def _edge_to_vertex_circulation(
    edge_normal: object,
    mesh: object,
    *,
    axis: int | None,
    normalize: bool,
) -> NDArray[np.floating[Any]]:
    edge_normal, mesh = _allow_mesh_first(edge_normal, mesh, "edgesOnVertex")
    n_edges = _mesh_dimension(mesh, "nEdges", "cellsOnEdge")
    field, original_axis = _entity_to_front(edge_normal, n_edges, axis)
    edges = np.asarray(_mesh_array(mesh, "edgesOnVertex"), dtype=np.int64)
    if np.any((edges < 0) | (edges >= n_edges)):
        raise ValueError("edgesOnVertex contains an invalid used edge")
    signs = edge_sign_on_vertex(mesh, dtype=field.dtype)
    dc_edge = np.asarray(_mesh_array(mesh, "dcEdge"), dtype=field.dtype)
    factors = signs * dc_edge[edges]
    gathered = field[edges]
    result = np.sum(
        gathered * _factor_for_levels(factors, gathered.ndim),
        axis=1,
        dtype=field.dtype,
    )
    if normalize:
        area = np.asarray(_mesh_array(mesh, "areaTriangle"), dtype=field.dtype)
        result /= _factor_for_levels(area, result.ndim)
    return _restore_entity_axis(result, original_axis)


def edge_circulation_to_vertex(
    edge_normal: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Signed circulation integral around each dual triangle.

    The signed ``dcEdge * u`` accumulation is the numerator transcribed from
    frozen MPAS-A v8.2.3
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5582-5593``.
    """

    return _edge_to_vertex_circulation(
        edge_normal, mesh, axis=axis, normalize=False
    )


def edge_to_vertex_curl(
    edge_normal: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Relative vorticity/curl at vertices from edge-normal values.

    This transcribes the full circulation and ``invAreaTriangle`` operation in
    frozen MPAS-A v8.2.3
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5582-5597``.
    """

    return _edge_to_vertex_circulation(
        edge_normal, mesh, axis=axis, normalize=True
    )


def cell_to_edge(
    cell_field: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Arithmetic interpolation from the two adjacent cells to each edge.

    This transcribes frozen MPAS-A v8.2.3
    ``src/core_atmosphere/dynamics/mpas_atm_time_integration.F:5561-5569``.
    """

    cell_field, mesh = _allow_mesh_first(cell_field, mesh, "cellsOnEdge")
    n_cells = _mesh_dimension(mesh, "nCells", "cellsOnCell")
    field, original_axis = _entity_to_front(cell_field, n_cells, axis)
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if np.any((cells < 0) | (cells >= n_cells)):
        raise ValueError("cell-to-edge interpolation requires two valid cells per edge")
    half = field.dtype.type(0.5)
    result = half * (field[cells[:, 0]] + field[cells[:, 1]])
    return _restore_entity_axis(result, original_axis)


def cell_to_vertex(
    cell_field: object, mesh: object, *, axis: int | None = None
) -> NDArray[np.floating[Any]]:
    """Kite-area finite-volume interpolation from cells to vertices.

    The live atmosphere interpolation is frozen MPAS-A v8.2.3
    ``src/core_atmosphere/diagnostics/mpas_isobaric_diagnostics.F:632-640``:
    it accumulates each cell through ``kiteAreasOnVertex`` and divides by
    ``areaTriangle``.
    """

    cell_field, mesh = _allow_mesh_first(cell_field, mesh, "cellsOnVertex")
    n_cells = _mesh_dimension(mesh, "nCells", "cellsOnCell")
    field, original_axis = _entity_to_front(cell_field, n_cells, axis)
    cells = np.asarray(_mesh_array(mesh, "cellsOnVertex"), dtype=np.int64)
    if np.any((cells < 0) | (cells >= n_cells)):
        raise ValueError("cell-to-vertex interpolation requires valid cellsOnVertex")
    kites = np.asarray(_mesh_array(mesh, "kiteAreasOnVertex"), dtype=field.dtype)
    area = np.asarray(_mesh_array(mesh, "areaTriangle"), dtype=field.dtype)
    gathered = field[cells]
    result = np.sum(
        gathered * _factor_for_levels(kites, gathered.ndim),
        axis=1,
        dtype=field.dtype,
    )
    result /= _factor_for_levels(area, result.ndim)
    return _restore_entity_axis(result, original_axis)


# Descriptive aliases keep call sites readable while the canonical functions
# above retain the exact entity-to-entity direction in their names.
gradient_cell_to_edge = edge_scalar_gradient
cell_to_edge_gradient = edge_scalar_gradient
divergence_edge_to_cell = edge_to_cell_divergence
circulation_edge_to_vertex = edge_circulation_to_vertex
curl_edge_to_vertex = edge_to_vertex_curl
interpolate_cell_to_edge = cell_to_edge
interpolate_cell_to_vertex = cell_to_vertex
edges_on_cell_sign = edge_sign_on_cell
edges_on_vertex_sign = edge_sign_on_vertex
kite_for_cell = kite_index_on_cell


__all__ = [
    "cell_to_edge",
    "cell_to_edge_gradient",
    "cell_to_vertex",
    "circulation_edge_to_vertex",
    "curl_edge_to_vertex",
    "divergence_edge_to_cell",
    "edge_circulation_to_vertex",
    "edge_scalar_gradient",
    "edge_sign_on_cell",
    "edge_sign_on_vertex",
    "edge_to_cell_divergence",
    "edge_to_vertex_curl",
    "edges_on_cell_sign",
    "edges_on_vertex_sign",
    "gradient_cell_to_edge",
    "interpolate_cell_to_edge",
    "interpolate_cell_to_vertex",
    "kite_for_cell",
    "kite_index_on_cell",
]
