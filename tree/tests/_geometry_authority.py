"""Retired scalar geometry implementation, retained only as the bitwise referee."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any, NamedTuple
import numpy as np
from numpy.typing import ArrayLike, NDArray

class VectorGeometry(NamedTuple):
    """Geometric vector fields initialized by MPAS."""

    edge_normal_vectors: NDArray[np.floating[Any]]
    local_vertical_unit_vectors: NDArray[np.floating[Any]]
    cell_tangent_plane: NDArray[np.floating[Any]]


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


