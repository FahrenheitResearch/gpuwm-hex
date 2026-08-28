"""Readable CPU authority for MPAS vector operations and reconstruction.

The transcription authority is the frozen MPAS-Model v8.2.3 tree:

* ``src/operators/mpas_vector_operations.F``
* ``src/operators/mpas_vector_reconstruction.F``
* the RBF setup called by reconstruction initialization in
  ``src/operators/mpas_rbf_interpolation.F``

Mesh connectivity is expected in Python/NetCDF order with zero-based indices
and ``-1`` padding.  Public routines accept any duck-typed object exposing the
named mesh arrays as attributes.  Floating point inputs retain float32 or
float64 throughout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class EdgeVectorComponents(NamedTuple):
    """Normal and tangent components of an R3 vector at edges."""

    normal: NDArray[np.floating[Any]]
    tangential: NDArray[np.floating[Any]]


class VectorGeometry(NamedTuple):
    """Geometric vector fields initialized by MPAS."""

    edge_normal_vectors: NDArray[np.floating[Any]]
    local_vertical_unit_vectors: NDArray[np.floating[Any]]
    cell_tangent_plane: NDArray[np.floating[Any]]


class ReconstructedVector(NamedTuple):
    """R3 and local horizontal components reconstructed at cells."""

    x: NDArray[np.floating[Any]]
    y: NDArray[np.floating[Any]]
    z: NDArray[np.floating[Any]]
    zonal: NDArray[np.floating[Any]]
    meridional: NDArray[np.floating[Any]]

    @property
    def r3(self) -> NDArray[np.floating[Any]]:
        """Return X/Y/Z components stacked on the final axis."""

        return np.stack((self.x, self.y, self.z), axis=-1)


__all__ = [
    "EdgeVectorComponents",
    "ReconstructedVector",
    "VectorGeometry",
    "cross_product_in_r3",
    "fix_periodicity",
    "initialize_reconstruction_coefficients",
    "initialize_tangent_vectors",
    "initialize_vector_geometry",
    "reconstruct",
    "reconstruct_1d",
    "reconstruct_2d",
    "tangential_vector_1d",
    "tangential_velocity",
    "unit_vector_in_r3",
    "vec_magnitude_in_r3",
    "vector_lon_lat_r_to_r3",
    "vector_r3_cell_to_2d_edge",
    "vector_r3_cell_to_normal_edge",
    "vector_r3_to_lon_lat_r",
    "zonal_meridional_vectors",
]


def _float_array(value: ArrayLike, name: str) -> NDArray[np.floating[Any]]:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"{name} must be numeric")
        array = array.astype(np.float64)
    return array


def _mesh_value(mesh: Any, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    if isinstance(mesh, Mapping) and name in mesh:
        return mesh[name]
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _mesh_array(mesh: Any, name: str) -> NDArray[Any]:
    value = _mesh_value(mesh, name, None)
    if value is None:
        raise AttributeError(f"mesh does not expose required array {name!r}")
    return np.asarray(value)


def _truth(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("ascii", errors="ignore")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "true", "t", "1"}:
            return True
        if normalized in {"no", "false", "f", "0"}:
            return False
    return bool(value)


def _coordinate_matrix(mesh: Any, suffix: str) -> NDArray[np.floating[Any]]:
    x = _float_array(_mesh_array(mesh, f"x{suffix}"), f"x{suffix}")
    y = _float_array(_mesh_array(mesh, f"y{suffix}"), f"y{suffix}")
    z = _float_array(_mesh_array(mesh, f"z{suffix}"), f"z{suffix}")
    dtype = np.result_type(x.dtype, y.dtype, z.dtype)
    return np.stack(
        (x.astype(dtype, copy=False), y.astype(dtype, copy=False), z.astype(dtype, copy=False)),
        axis=-1,
    )


def _rows(array: ArrayLike, row_count: int, name: str) -> NDArray[Any]:
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if result.shape[0] == row_count:
        return result
    if result.shape[1] == row_count:
        return result.T
    raise ValueError(f"{name} has shape {result.shape}; expected {row_count} rows")


def _edge_vectors(mesh: Any, name: str, n_edges: int) -> NDArray[np.floating[Any]]:
    vectors = _float_array(_mesh_array(mesh, name), name)
    if vectors.shape == (n_edges, 3):
        return vectors
    if vectors.shape == (3, n_edges):
        return vectors.T
    raise ValueError(f"{name} has shape {vectors.shape}; expected ({n_edges}, 3)")


def _cell_coefficients(
    coefficients: ArrayLike, n_cells: int
) -> NDArray[np.floating[Any]]:
    result = _float_array(coefficients, "coeffs_reconstruct")
    if result.ndim != 3:
        raise ValueError("coeffs_reconstruct must be three-dimensional")
    if result.shape[0] == n_cells and result.shape[2] == 3:
        return result
    if result.shape[2] == n_cells and result.shape[0] == 3:
        return result.transpose(2, 1, 0)
    if result.shape[0] == n_cells and result.shape[1] == 3:
        return result.transpose(0, 2, 1)
    raise ValueError(
        f"coeffs_reconstruct has shape {result.shape}; expected (nCells, maxEdges, 3)"
    )


def _compute_count(mesh: Any, entity: str, include_halos: bool, total: int) -> int:
    if include_halos:
        return total
    candidates = (f"n{entity}Solve", f"n_{entity.lower()}_solve")
    for name in candidates:
        value = _mesh_value(mesh, name, None)
        if value is not None:
            count = int(value)
            if count < 0 or count > total:
                raise ValueError(f"{name}={count} is outside [0, {total}]")
            return count
    # Published global meshes have no block/halo split; every item is solved.
    return total


def _dot_last(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """Three-term dot product in the same left-to-right order as Fortran SUM."""

    result = a[..., 0] * b[..., 0]
    result = result + a[..., 1] * b[..., 1]
    result = result + a[..., 2] * b[..., 2]
    return result


def vec_magnitude_in_r3(vector: ArrayLike) -> NDArray[np.floating[Any]]:
    """Magnitude of vectors whose final dimension is three.

    Frozen authority: ``mpas_vector_operations.F:79-83``.
    """

    value = _float_array(vector, "vector")
    if value.shape == () or value.shape[-1] != 3:
        raise ValueError("vector must have final dimension 3")
    squared = value[..., 0] * value[..., 0]
    squared = squared + value[..., 1] * value[..., 1]
    squared = squared + value[..., 2] * value[..., 2]
    return np.sqrt(squared)


def unit_vector_in_r3(vector: ArrayLike) -> NDArray[np.floating[Any]]:
    """Return unit vectors without changing float32/float64 precision.

    Frozen authority: ``mpas_vector_operations.F:95-101``.  Unlike the
    in-place Fortran routine, this Python authority returns a new array.
    """

    value = _float_array(vector, "vector")
    magnitude = vec_magnitude_in_r3(value)
    with np.errstate(divide="ignore", invalid="ignore"):
        return value / magnitude[..., np.newaxis]


def cross_product_in_r3(first: ArrayLike, second: ArrayLike) -> NDArray[np.floating[Any]]:
    """Cross product for broadcast-compatible R3 vectors.

    Frozen authority: ``mpas_vector_operations.F:113-121``.
    """

    left = _float_array(first, "first")
    right = _float_array(second, "second")
    if left.shape == () or right.shape == () or left.shape[-1] != 3 or right.shape[-1] != 3:
        raise ValueError("both vectors must have final dimension 3")
    dtype = np.result_type(left.dtype, right.dtype)
    left = left.astype(dtype, copy=False)
    right = right.astype(dtype, copy=False)
    shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1]) + (3,)
    output = np.empty(shape, dtype=dtype)
    output[..., 0] = left[..., 1] * right[..., 2] - left[..., 2] * right[..., 1]
    output[..., 1] = left[..., 2] * right[..., 0] - left[..., 0] * right[..., 2]
    output[..., 2] = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    return output


def zonal_meridional_vectors(
    lon: ArrayLike, lat: ArrayLike
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute local zonal, meridional, and vertical unit vectors.

    Frozen authority: ``mpas_vector_operations.F:454-503``.
    """

    lon_array = _float_array(lon, "lon")
    lat_array = _float_array(lat, "lat")
    dtype = np.result_type(lon_array.dtype, lat_array.dtype)
    lon_array, lat_array = np.broadcast_arrays(
        lon_array.astype(dtype, copy=False), lat_array.astype(dtype, copy=False)
    )
    sin_lat = np.sin(lat_array)
    cos_lat = np.cos(lat_array)
    sin_lon = np.sin(lon_array)
    cos_lon = np.cos(lon_array)
    zero = np.zeros_like(sin_lon)

    zonal = np.stack((-sin_lon, cos_lon, zero), axis=-1)
    meridional = np.stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat), axis=-1)
    vertical = np.stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat), axis=-1)
    return zonal, meridional, vertical


def vector_r3_to_lon_lat_r(
    vector_r3: ArrayLike, lon: ArrayLike, lat: ArrayLike
) -> NDArray[np.floating[Any]]:
    """Rotate R3 vector components into zonal/meridional/radial components.

    Frozen authority: ``mpas_vector_operations.F:518-568``.
    """

    vector = _float_array(vector_r3, "vector_r3")
    if vector.shape == () or vector.shape[-1] != 3:
        raise ValueError("vector_r3 must have final dimension 3")
    zonal, meridional, vertical = zonal_meridional_vectors(lon, lat)
    dtype = np.result_type(vector.dtype, zonal.dtype)
    vector = vector.astype(dtype, copy=False)
    basis = np.stack((zonal, meridional, vertical), axis=-2).astype(dtype, copy=False)
    vector, basis = np.broadcast_arrays(vector[..., np.newaxis, :], basis)
    return _dot_last(basis, vector)


def vector_lon_lat_r_to_r3(
    vector_lon_lat_r: ArrayLike, lon: ArrayLike, lat: ArrayLike
) -> NDArray[np.floating[Any]]:
    """Rotate zonal/meridional/radial components into R3.

    Frozen authority: ``mpas_vector_operations.F:583-631``.
    """

    vector = _float_array(vector_lon_lat_r, "vector_lon_lat_r")
    if vector.shape == () or vector.shape[-1] != 3:
        raise ValueError("vector_lon_lat_r must have final dimension 3")
    zonal, meridional, vertical = zonal_meridional_vectors(lon, lat)
    dtype = np.result_type(vector.dtype, zonal.dtype)
    vector = vector.astype(dtype, copy=False)
    zonal = zonal.astype(dtype, copy=False)
    meridional = meridional.astype(dtype, copy=False)
    vertical = vertical.astype(dtype, copy=False)
    output = zonal * vector[..., 0, np.newaxis]
    output = output + meridional * vector[..., 1, np.newaxis]
    output = output + vertical * vector[..., 2, np.newaxis]
    return output


def fix_periodicity(pxi: ArrayLike, xci: ArrayLike, period: ArrayLike) -> NDArray[Any]:
    """Move a coordinate at most one period so it is near a reference.

    Frozen authority: ``mpas_vector_operations.F:861-889``.
    """

    point = _float_array(pxi, "pxi")
    center = _float_array(xci, "xci")
    period_array = _float_array(period, "period")
    dtype = np.result_type(point.dtype, center.dtype, period_array.dtype)
    point, center, period_array = np.broadcast_arrays(
        point.astype(dtype, copy=False),
        center.astype(dtype, copy=False),
        period_array.astype(dtype, copy=False),
    )
    distance = point - center
    output = point.copy()
    mask = np.abs(distance) > period_array * dtype.type(0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        output[mask] = point[mask] - (distance[mask] / np.abs(distance[mask])) * period_array[mask]
    return output


def _normalize_geometry(vectors: NDArray[np.floating[Any]], name: str) -> NDArray[np.floating[Any]]:
    magnitude = vec_magnitude_in_r3(vectors)
    if np.any(~np.isfinite(magnitude)) or np.any(magnitude == 0):
        raise ValueError(f"cannot normalize degenerate {name}")
    return vectors / magnitude[:, np.newaxis]


def initialize_vector_geometry(
    mesh: Any,
    *,
    on_a_sphere: bool | None = None,
    is_periodic: bool | None = None,
    x_period: float | None = None,
    y_period: float | None = None,
) -> VectorGeometry:
    """Initialize edge normals and the local cell tangent bases.

    This is a direct port of ``mpas_vector_operations.F:652-771``.  Boundary
    sentinels ``nCells+1`` are represented by ``-1`` in the Python mesh.
    """

    cell = _coordinate_matrix(mesh, "Cell")
    edge = _coordinate_matrix(mesh, "Edge")
    dtype = np.result_type(cell.dtype, edge.dtype)
    cell = cell.astype(dtype, copy=False)
    edge = edge.astype(dtype, copy=False)
    n_cells = cell.shape[0]
    n_edges = edge.shape[0]
    cells_on_edge = _rows(_mesh_array(mesh, "cellsOnEdge"), n_edges, "cellsOnEdge").astype(
        np.int64, copy=False
    )
    edges_on_cell = _rows(_mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell").astype(
        np.int64, copy=False
    )
    if cells_on_edge.shape[1] != 2:
        raise ValueError("cellsOnEdge must have exactly two columns")
    if np.any(cells_on_edge < -1) or np.any(cells_on_edge >= n_cells):
        raise ValueError("cellsOnEdge contains an out-of-range index")

    sphere = _truth(_mesh_value(mesh, "on_a_sphere", True), True) if on_a_sphere is None else on_a_sphere
    periodic = _truth(_mesh_value(mesh, "is_periodic", False), False) if is_periodic is None else is_periodic
    xp = dtype.type(_mesh_value(mesh, "x_period", 0.0) if x_period is None else x_period)
    yp = dtype.type(_mesh_value(mesh, "y_period", 0.0) if y_period is None else y_period)

    local_vertical = np.empty((n_cells, 3), dtype=dtype)
    if sphere:
        local_vertical[:] = _normalize_geometry(cell, "cell position")
    else:
        local_vertical[:] = dtype.type(0)
        local_vertical[:, 2] = dtype.type(1)

    normal = np.empty((n_edges, 3), dtype=dtype)
    cell1 = cells_on_edge[:, 0]
    cell2 = cells_on_edge[:, 1]
    if np.any((cell1 == -1) & (cell2 == -1)):
        raise ValueError("an edge cannot have two missing cells")

    interior = (cell1 >= 0) & (cell2 >= 0)
    first_missing = cell1 == -1
    second_missing = cell2 == -1

    first_position = cell[np.maximum(cell1, 0)]
    second_position = cell[np.maximum(cell2, 0)]
    if periodic:
        second_x = fix_periodicity(second_position[:, 0], first_position[:, 0], xp)
        second_y = fix_periodicity(second_position[:, 1], first_position[:, 1], yp)
        normal[interior, 0] = second_x[interior] - first_position[interior, 0]
        normal[interior, 1] = second_y[interior] - first_position[interior, 1]
        edge_x_near_second = fix_periodicity(edge[:, 0], second_position[:, 0], xp)
        edge_y_near_second = fix_periodicity(edge[:, 1], second_position[:, 1], yp)
        edge_x_near_first = fix_periodicity(edge[:, 0], first_position[:, 0], xp)
        edge_y_near_first = fix_periodicity(edge[:, 1], first_position[:, 1], yp)
        normal[first_missing, 0] = second_position[first_missing, 0] - edge_x_near_second[first_missing]
        normal[first_missing, 1] = second_position[first_missing, 1] - edge_y_near_second[first_missing]
        normal[second_missing, 0] = edge_x_near_first[second_missing] - first_position[second_missing, 0]
        normal[second_missing, 1] = edge_y_near_first[second_missing] - first_position[second_missing, 1]
    else:
        normal[interior] = second_position[interior] - first_position[interior]
        normal[first_missing] = second_position[first_missing] - edge[first_missing]
        normal[second_missing] = edge[second_missing] - first_position[second_missing]
    if periodic:
        normal[interior, 2] = second_position[interior, 2] - first_position[interior, 2]
        normal[first_missing, 2] = second_position[first_missing, 2] - edge[first_missing, 2]
        normal[second_missing, 2] = edge[second_missing, 2] - first_position[second_missing, 2]
    normal = _normalize_geometry(normal, "edge normal")

    n_edges_on_cell = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    if n_edges_on_cell.shape != (n_cells,) or np.any(n_edges_on_cell < 1):
        raise ValueError("every cell must expose at least one edge")
    first_edge = edges_on_cell[:, 0]
    if np.any(first_edge < 0) or np.any(first_edge >= n_edges):
        raise ValueError("the first edge on every cell must be valid")
    radial_dot = _dot_last(normal[first_edge], local_vertical)
    x_plane = normal[first_edge] - radial_dot[:, np.newaxis] * local_vertical
    x_plane = _normalize_geometry(x_plane, "cell tangent x vector")
    y_plane = _normalize_geometry(
        cross_product_in_r3(local_vertical, x_plane), "cell tangent y vector"
    )
    tangent_plane = np.stack((x_plane, y_plane), axis=1)
    return VectorGeometry(normal, local_vertical, tangent_plane)


def initialize_tangent_vectors(
    mesh: Any,
    *,
    is_periodic: bool | None = None,
    x_period: float | None = None,
    y_period: float | None = None,
) -> NDArray[np.floating[Any]]:
    """Initialize edge tangent vectors directed from vertex 1 to vertex 2.

    Frozen authority: ``mpas_vector_operations.F:788-842``.
    """

    vertex = _coordinate_matrix(mesh, "Vertex")
    n_edges = _coordinate_matrix(mesh, "Edge").shape[0]
    vertices_on_edge = _rows(
        _mesh_array(mesh, "verticesOnEdge"), n_edges, "verticesOnEdge"
    ).astype(np.int64, copy=False)
    if vertices_on_edge.shape[1] != 2:
        raise ValueError("verticesOnEdge must have exactly two columns")
    if np.any(vertices_on_edge < 0) or np.any(vertices_on_edge >= vertex.shape[0]):
        raise ValueError("verticesOnEdge contains an out-of-range index")
    dtype = vertex.dtype
    periodic = _truth(_mesh_value(mesh, "is_periodic", False), False) if is_periodic is None else is_periodic
    xp = dtype.type(_mesh_value(mesh, "x_period", 0.0) if x_period is None else x_period)
    yp = dtype.type(_mesh_value(mesh, "y_period", 0.0) if y_period is None else y_period)
    first = vertex[vertices_on_edge[:, 0]]
    second = vertex[vertices_on_edge[:, 1]]
    tangent = second - first
    if periodic:
        tangent[:, 0] = fix_periodicity(second[:, 0], first[:, 0], xp) - first[:, 0]
        tangent[:, 1] = fix_periodicity(second[:, 1], first[:, 1], yp) - first[:, 1]
    return _normalize_geometry(tangent, "edge tangent")


def vector_r3_cell_to_2d_edge(
    vector_r3_cell: ArrayLike,
    mesh: Any,
    edge_tangent_vectors: ArrayLike | None = None,
    *,
    include_halos: bool = True,
) -> EdgeVectorComponents:
    """Average cell R3 vectors to edges and project onto normal/tangent axes.

    Input shape is ``(..., nCells, 3)`` and output shape is ``(..., nEdges)``.
    Frozen authority: ``mpas_vector_operations.F:136-209``.
    """

    vectors = _float_array(vector_r3_cell, "vector_r3_cell")
    n_cells = _coordinate_matrix(mesh, "Cell").shape[0]
    n_edges = _coordinate_matrix(mesh, "Edge").shape[0]
    if vectors.ndim < 2 or vectors.shape[-2:] != (n_cells, 3):
        raise ValueError(f"vector_r3_cell must have trailing shape ({n_cells}, 3)")
    cells_on_edge = _rows(_mesh_array(mesh, "cellsOnEdge"), n_edges, "cellsOnEdge").astype(
        np.int64, copy=False
    )
    count = _compute_count(mesh, "Edges", include_halos, n_edges)
    active = cells_on_edge[:count]
    if np.any(active < 0):
        raise ValueError("R3 cell-to-edge interpolation needs two valid cells per active edge")
    geometry = initialize_vector_geometry(mesh)
    normal_vectors = geometry.edge_normal_vectors.astype(vectors.dtype, copy=False)
    tangent_vectors = (
        initialize_tangent_vectors(mesh)
        if edge_tangent_vectors is None
        else _edge_vectors(
            type("_EdgeVectors", (), {"value": edge_tangent_vectors})(), "value", n_edges
        )
    ).astype(vectors.dtype, copy=False)
    output_shape = vectors.shape[:-2] + (n_edges,)
    normal = np.zeros(output_shape, dtype=vectors.dtype)
    tangential = np.zeros(output_shape, dtype=vectors.dtype)
    half = vectors.dtype.type(0.5)
    average = half * (vectors[..., active[:, 0], :] + vectors[..., active[:, 1], :])
    normal[..., :count] = _dot_last(average, normal_vectors[:count])
    tangential[..., :count] = _dot_last(average, tangent_vectors[:count])
    return EdgeVectorComponents(normal, tangential)


def vector_r3_cell_to_normal_edge(
    vector_r3_cell: ArrayLike, mesh: Any, *, include_halos: bool = True
) -> NDArray[np.floating[Any]]:
    """Normal-only form of cell R3 to edge projection.

    Frozen authority: ``mpas_vector_operations.F:223-289``.
    """

    return vector_r3_cell_to_2d_edge(
        vector_r3_cell, mesh, include_halos=include_halos
    ).normal


def tangential_velocity(
    normal_velocity: ArrayLike, mesh: Any, *, include_halos: bool = True
) -> NDArray[np.floating[Any]]:
    """Reconstruct edge-tangent velocity from neighboring edge normals.

    Input and output use trailing ``nEdges``.  The edge-neighbor accumulation
    order exactly follows ``mpas_vector_operations.F:304-363``.  The frozen 2-D
    routine accumulates into an undefined ``intent(out)`` value at lines
    354-358; this authority initializes to zero, matching the defined 1-D
    routine at lines 430-435 rather than reproducing undefined memory.
    """

    values = _float_array(normal_velocity, "normal_velocity")
    n_edges = _coordinate_matrix(mesh, "Edge").shape[0]
    if values.ndim < 1 or values.shape[-1] != n_edges:
        raise ValueError(f"normal_velocity must have trailing dimension {n_edges}")
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnEdge"), dtype=np.int64)
    neighbors = _rows(_mesh_array(mesh, "edgesOnEdge"), n_edges, "edgesOnEdge").astype(
        np.int64, copy=False
    )
    weights = _rows(_mesh_array(mesh, "weightsOnEdge"), n_edges, "weightsOnEdge").astype(
        values.dtype, copy=False
    )
    if counts.shape != (n_edges,) or np.any(counts < 0) or np.any(counts > neighbors.shape[1]):
        raise ValueError("nEdgesOnEdge is inconsistent with edgesOnEdge")
    count = _compute_count(mesh, "Edges", include_halos, n_edges)
    output = np.zeros_like(values)
    for edge_index in range(count):
        for neighbor_slot in range(int(counts[edge_index])):
            neighbor = int(neighbors[edge_index, neighbor_slot])
            if neighbor < 0 or neighbor >= n_edges:
                raise ValueError("active edgesOnEdge entry is out of range")
            output[..., edge_index] = (
                output[..., edge_index]
                + weights[edge_index, neighbor_slot] * values[..., neighbor]
            )
    return output


def tangential_vector_1d(
    normal_vector: ArrayLike, mesh: Any, *, include_halos: bool = True
) -> NDArray[np.floating[Any]]:
    """One-dimensional tangential reconstruction.

    Frozen authority: ``mpas_vector_operations.F:381-438``.
    """

    values = _float_array(normal_vector, "normal_vector")
    if values.ndim != 1:
        raise ValueError("normal_vector must be one-dimensional")
    return tangential_velocity(values, mesh, include_halos=include_halos)


def _solve_legs(
    matrix: NDArray[np.floating[Any]], rhs: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Scaled-pivot Gaussian solve from ``mpas_rbf_interpolation.F:1670-1846``."""

    a = matrix.copy()
    b = rhs.copy()
    n = a.shape[0]
    dtype = a.dtype
    indices = np.arange(n, dtype=np.int64)
    scales = np.empty(n, dtype=dtype)
    for i in range(n):
        maximum = dtype.type(0)
        for j in range(n):
            maximum = max(maximum, abs(a[i, j]))
        if maximum == 0:
            raise np.linalg.LinAlgError("singular RBF reconstruction matrix")
        scales[i] = maximum

    for j in range(n - 1):
        pivot_value = dtype.type(0)
        pivot_slot = j
        for i in range(j, n):
            candidate = abs(a[indices[i], j]) / scales[indices[i]]
            if candidate > pivot_value:
                pivot_value = candidate
                pivot_slot = i
        indices[j], indices[pivot_slot] = indices[pivot_slot], indices[j]
        pivot_row = indices[j]
        if a[pivot_row, j] == 0:
            raise np.linalg.LinAlgError("singular RBF reconstruction matrix")
        for i in range(j + 1, n):
            row = indices[i]
            ratio = a[row, j] / a[pivot_row, j]
            a[row, j] = ratio
            for k in range(j + 1, n):
                a[row, k] = a[row, k] - ratio * a[pivot_row, k]

    for i in range(n - 1):
        for j in range(i + 1, n):
            b[indices[j]] = b[indices[j]] - a[indices[j], i] * b[indices[i]]
    result = np.empty(n, dtype=dtype)
    if a[indices[n - 1], n - 1] == 0:
        raise np.linalg.LinAlgError("singular RBF reconstruction matrix")
    result[n - 1] = b[indices[n - 1]] / a[indices[n - 1], n - 1]
    for i in range(n - 2, -1, -1):
        result[i] = b[indices[i]]
        for j in range(i + 1, n):
            result[i] = result[i] - a[indices[i], j] * result[j]
        result[i] = result[i] / a[indices[i], i]
    return result


def _rbf_coefficients_for_cell(
    source_points: NDArray[np.floating[Any]],
    unit_vectors: NDArray[np.floating[Any]],
    destination: NDArray[np.floating[Any]],
    alpha: np.floating[Any],
    plane_basis: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Port of ``mpas_rbf_interpolation.F:1079-1145,1527-1559``."""

    point_count = source_points.shape[0]
    dtype = source_points.dtype
    planar_source = np.empty((point_count, 2), dtype=dtype)
    planar_unit = np.empty((point_count, 2), dtype=dtype)
    for i in range(point_count):
        for component in range(2):
            planar_source[i, component] = _dot_last(source_points[i], plane_basis[component])
            planar_unit[i, component] = _dot_last(unit_vectors[i], plane_basis[component])
    planar_destination = np.empty(2, dtype=dtype)
    planar_destination[0] = _dot_last(destination, plane_basis[0])
    planar_destination[1] = _dot_last(destination, plane_basis[1])

    matrix_size = point_count + 2
    matrix = np.zeros((matrix_size, matrix_size), dtype=dtype)
    rhs = np.zeros((matrix_size, 2), dtype=dtype)
    one = dtype.type(1)
    alpha_squared = alpha * alpha
    for j in range(point_count):
        for i in range(j, point_count):
            difference = planar_source[i] - planar_source[j]
            r_squared = difference[0] * difference[0]
            r_squared = r_squared + difference[1] * difference[1]
            r_squared = r_squared / alpha_squared
            rbf = one / np.sqrt(one + r_squared)
            normal_dot = planar_unit[i, 0] * planar_unit[j, 0]
            normal_dot = normal_dot + planar_unit[i, 1] * planar_unit[j, 1]
            matrix[i, j] = rbf * normal_dot
            matrix[j, i] = matrix[i, j]
    for j in range(point_count):
        difference = planar_destination - planar_source[j]
        r_squared = difference[0] * difference[0]
        r_squared = r_squared + difference[1] * difference[1]
        r_squared = r_squared / alpha_squared
        rbf = one / np.sqrt(one + r_squared)
        rhs[j, 0] = rbf * planar_unit[j, 0]
        rhs[j, 1] = rbf * planar_unit[j, 1]
        matrix[j, point_count : point_count + 2] = planar_unit[j]
        matrix[point_count : point_count + 2, j] = planar_unit[j]
    rhs[point_count, 0] = one
    rhs[point_count + 1, 1] = one

    planar_coefficients = np.empty((matrix_size, 2), dtype=dtype)
    planar_coefficients[:, 0] = _solve_legs(matrix, rhs[:, 0])
    planar_coefficients[:, 1] = _solve_legs(matrix, rhs[:, 1])
    coefficients = np.empty((point_count, 3), dtype=dtype)
    for component in range(3):
        coefficients[:, component] = (
            plane_basis[0, component] * planar_coefficients[:point_count, 0]
            + plane_basis[1, component] * planar_coefficients[:point_count, 1]
        )
    return coefficients


def initialize_reconstruction_coefficients(
    mesh: Any, *, include_halos: bool = True
) -> NDArray[np.floating[Any]]:
    """Compute RBF edge-to-cell reconstruction coefficients.

    This transcribes ``mpas_vector_reconstruction.F:51-181`` and its called
    RBF path ``mpas_rbf_interpolation.F:1079-1145,1527-1559,1670-1846``.
    The published x1.2562 static file declares this field but contains zeros;
    MPAS fills it during initialization, so callers must do the same.
    """

    cell = _coordinate_matrix(mesh, "Cell")
    edge = _coordinate_matrix(mesh, "Edge").astype(cell.dtype, copy=False)
    n_cells = cell.shape[0]
    n_edges = edge.shape[0]
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    edges_on_cell = _rows(_mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell").astype(
        np.int64, copy=False
    )
    if counts.shape != (n_cells,) or np.any(counts < 1) or np.any(counts > edges_on_cell.shape[1]):
        raise ValueError("nEdgesOnCell is inconsistent with edgesOnCell")
    geometry = initialize_vector_geometry(mesh)
    normals = geometry.edge_normal_vectors.astype(cell.dtype, copy=False)
    tangent_plane = geometry.cell_tangent_plane.astype(cell.dtype, copy=False)
    periodic = _truth(_mesh_value(mesh, "is_periodic", False), False)
    xp = cell.dtype.type(_mesh_value(mesh, "x_period", 0.0))
    yp = cell.dtype.type(_mesh_value(mesh, "y_period", 0.0))
    cell_count = _compute_count(mesh, "Cells", include_halos, n_cells)
    result = np.zeros((n_cells, edges_on_cell.shape[1], 3), dtype=cell.dtype)

    for cell_index in range(cell_count):
        point_count = int(counts[cell_index])
        edge_indices = edges_on_cell[cell_index, :point_count]
        if np.any(edge_indices < 0) or np.any(edge_indices >= n_edges):
            raise ValueError("active edgesOnCell entry is out of range")
        source = edge[edge_indices].copy()
        if periodic:
            source[:, 0] = fix_periodicity(source[:, 0], cell[cell_index, 0], xp)
            source[:, 1] = fix_periodicity(source[:, 1], cell[cell_index, 1], yp)
        alpha = cell.dtype.type(0)
        for point_index in range(point_count):
            difference = cell[cell_index] - source[point_index]
            radius_squared = difference[0] * difference[0]
            radius_squared = radius_squared + difference[1] * difference[1]
            radius_squared = radius_squared + difference[2] * difference[2]
            alpha = alpha + np.sqrt(radius_squared)
        alpha = alpha / cell.dtype.type(point_count)
        if alpha == 0 or not np.isfinite(alpha):
            raise ValueError("invalid RBF length scale")
        result[cell_index, :point_count] = _rbf_coefficients_for_cell(
            source,
            normals[edge_indices],
            cell[cell_index],
            alpha,
            tangent_plane[cell_index],
        )
    return result


def _reconstruct_common(
    mesh: Any,
    edge_values: NDArray[np.floating[Any]],
    *,
    include_halos: bool,
    coefficients: ArrayLike | None,
) -> ReconstructedVector:
    n_cells = _coordinate_matrix(mesh, "Cell").shape[0]
    n_edges = _coordinate_matrix(mesh, "Edge").shape[0]
    if edge_values.shape[-1] != n_edges:
        raise ValueError(f"u must have trailing dimension {n_edges}")
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    edges_on_cell = _rows(_mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell").astype(
        np.int64, copy=False
    )
    raw_coefficients = (
        _mesh_array(mesh, "coeffs_reconstruct") if coefficients is None else coefficients
    )
    coeffs = _cell_coefficients(raw_coefficients, n_cells).astype(edge_values.dtype, copy=False)
    if not np.any(coeffs):
        raise ValueError(
            "coeffs_reconstruct is all zero; call initialize_reconstruction_coefficients(mesh)"
        )
    if coeffs.shape[1] < edges_on_cell.shape[1]:
        raise ValueError("coeffs_reconstruct has fewer edge slots than edgesOnCell")
    cell_count = _compute_count(mesh, "Cells", include_halos, n_cells)
    output_shape = edge_values.shape[:-1] + (n_cells,)
    x = np.zeros(output_shape, dtype=edge_values.dtype)
    y = np.zeros(output_shape, dtype=edge_values.dtype)
    z = np.zeros(output_shape, dtype=edge_values.dtype)

    # Preserve the edge-slot accumulation order at reconstruction.F:256-265
    # and :370-379; only independent leading dimensions are vectorized.
    for cell_index in range(cell_count):
        for edge_slot in range(int(counts[cell_index])):
            edge_index = int(edges_on_cell[cell_index, edge_slot])
            if edge_index < 0 or edge_index >= n_edges:
                raise ValueError("active edgesOnCell entry is out of range")
            value = edge_values[..., edge_index]
            x[..., cell_index] = x[..., cell_index] + coeffs[cell_index, edge_slot, 0] * value
            y[..., cell_index] = y[..., cell_index] + coeffs[cell_index, edge_slot, 1] * value
            z[..., cell_index] = z[..., cell_index] + coeffs[cell_index, edge_slot, 2] * value

    sphere = _truth(_mesh_value(mesh, "on_a_sphere", True), True)
    if sphere:
        lat = _float_array(_mesh_array(mesh, "latCell"), "latCell").astype(
            edge_values.dtype, copy=False
        )
        lon = _float_array(_mesh_array(mesh, "lonCell"), "lonCell").astype(
            edge_values.dtype, copy=False
        )
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        zonal = -x * sin_lon + y * cos_lon
        meridional = -(x * cos_lon + y * sin_lon) * sin_lat + z * cos_lat
    else:
        zonal = x.copy()
        meridional = y.copy()
    return ReconstructedVector(x, y, z, zonal, meridional)


def reconstruct_1d(
    mesh: Any,
    u: ArrayLike,
    *,
    include_halos: bool = False,
    coefficients: ArrayLike | None = None,
) -> ReconstructedVector:
    """Reconstruct one edge scalar per edge to cell-centered vectors.

    Frozen authority: ``mpas_vector_reconstruction.F:309-408``.
    """

    values = _float_array(u, "u")
    if values.ndim != 1:
        raise ValueError("reconstruct_1d expects a one-dimensional edge field")
    return _reconstruct_common(
        mesh, values, include_halos=include_halos, coefficients=coefficients
    )


def reconstruct_2d(
    mesh: Any,
    u: ArrayLike,
    *,
    include_halos: bool = False,
    coefficients: ArrayLike | None = None,
) -> ReconstructedVector:
    """Reconstruct a level-by-edge field to level-by-cell vectors.

    Frozen authority: ``mpas_vector_reconstruction.F:195-294``.
    """

    values = _float_array(u, "u")
    if values.ndim != 2:
        raise ValueError("reconstruct_2d expects a two-dimensional level-by-edge field")
    return _reconstruct_common(
        mesh, values, include_halos=include_halos, coefficients=coefficients
    )


def reconstruct(
    mesh: Any,
    u: ArrayLike,
    *,
    include_halos: bool = False,
    coefficients: ArrayLike | None = None,
) -> ReconstructedVector:
    """Dispatch to the frozen 1-D or 2-D MPAS reconstruction overload."""

    values = _float_array(u, "u")
    if values.ndim == 1:
        return reconstruct_1d(
            mesh, values, include_halos=include_halos, coefficients=coefficients
        )
    if values.ndim == 2:
        return reconstruct_2d(
            mesh, values, include_halos=include_halos, coefficients=coefficients
        )
    raise ValueError("MPAS reconstruction is defined only for 1-D or 2-D edge fields")
