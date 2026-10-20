"""Array marshaling for exact MPAS geometry computed by the Rust bridge."""

from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
import struct
import subprocess
from typing import Any

import numpy as np

from .engines import GEOMETRY, EngineRefusal, resolve

ABI_MARKER = b"rw_mpas_geometry --protocol hex-geometry-v1"


@lru_cache(maxsize=8)
def _check_binary(path: str, size: int, modified_ns: int) -> str:
    """Reject a stale bridge before it can consume the versioned array stream."""
    digest = hashlib.sha256()
    marker_found = False
    tail = b""
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            marker_found = marker_found or ABI_MARKER in tail + block
            tail = block[-len(ABI_MARKER):]
    if not marker_found:
        raise EngineRefusal(
            f"{GEOMETRY.name} at {path} carries no hex-geometry-v1 marker; "
            "a stale executable would parse coordinate bytes with the wrong ABI. "
            "Build the geometry bridge from the current gpuwm source"
        )
    return digest.hexdigest()


def geometry_engine() -> tuple[Path, str]:
    path = resolve(GEOMETRY)
    stat = path.stat()
    return path, _check_binary(str(path), stat.st_size, stat.st_mtime_ns)


def _arrays(mesh: Any, dtype: np.dtype | None = None):
    from .vector import _coordinate_matrix, _mesh_array, _rows

    cell = _coordinate_matrix(mesh, "Cell")
    edge = _coordinate_matrix(mesh, "Edge")
    if dtype is not None:
        cell = np.ascontiguousarray(cell, dtype=dtype)
        edge = np.ascontiguousarray(edge, dtype=dtype)
    n_cells, n_edges = cell.shape[0], edge.shape[0]
    coe = _rows(_mesh_array(mesh, "cellsOnEdge"), n_edges, "cellsOnEdge")
    eoc = _rows(_mesh_array(mesh, "edgesOnCell"), n_cells, "edgesOnCell")
    counts = np.asarray(_mesh_array(mesh, "nEdgesOnCell"), dtype=np.int64)
    if coe.shape != (n_edges, 2):
        raise ValueError("cellsOnEdge must have exactly two columns")
    if counts.shape != (n_cells,):
        raise ValueError("nEdgesOnCell is inconsistent with edgesOnCell")
    return cell, edge, coe, eoc, counts


def _run(mesh: Any, *, mode: int, dtype: np.dtype, include_halos: bool = True,
         geometry=None, on_a_sphere=None, is_periodic=None,
         x_period=None, y_period=None):
    from .vector import _compute_count, _mesh_value, _truth

    cell, edge, coe, eoc, counts = _arrays(mesh, dtype)
    n_cells, n_edges, max_edges = cell.shape[0], edge.shape[0], eoc.shape[1]
    solved = _compute_count(mesh, "Cells", include_halos, n_cells)
    sphere = (_truth(_mesh_value(mesh, "on_a_sphere", True), True)
              if on_a_sphere is None else bool(on_a_sphere))
    periodic = (_truth(_mesh_value(mesh, "is_periodic", False), False)
                if is_periodic is None else bool(is_periodic))
    xp = _mesh_value(mesh, "x_period", 0.0) if x_period is None else x_period
    yp = _mesh_value(mesh, "y_period", 0.0) if y_period is None else y_period
    wire_dtype = dtype.newbyteorder("<")
    parts = [struct.pack("<8sQQQQBBBB", b"HEXGEO1\0", n_cells, n_edges,
                         max_edges, solved, dtype.itemsize, sphere, periodic, mode),
             np.asarray([xp, yp], dtype=wire_dtype).tobytes()]
    parts.extend(np.asarray(a, dtype=wire_dtype).tobytes(order="C") for a in (cell, edge))
    parts.extend(np.asarray(a, dtype="<i8").tobytes(order="C") for a in (coe, eoc, counts))
    if mode == 2:
        parts.extend(np.asarray(a, dtype=wire_dtype).tobytes(order="C") for a in
                     (geometry.edge_normal_vectors, geometry.cell_tangent_plane))
    engine, _ = geometry_engine()
    result = subprocess.run([str(engine), "--protocol", "hex-geometry-v1"],
                            input=b"".join(parts), capture_output=True, check=False)
    if result.returncode:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        if "singular RBF reconstruction matrix" in message:
            raise np.linalg.LinAlgError(message)
        raise ValueError(message or f"geometry bridge exited {result.returncode}")
    shapes = ([(n_cells, max_edges, 3)] if mode == 2 else
              [(n_edges, 3), (n_cells, 3), (n_cells, 2, 3)] +
              ([(n_cells, max_edges, 3)] if mode == 0 else []))
    expected = sum(int(np.prod(shape)) for shape in shapes) * dtype.itemsize
    if len(result.stdout) != expected:
        raise EngineRefusal(f"geometry bridge returned {len(result.stdout)} bytes, "
                            f"expected {expected}; incomplete geometry cannot initialize a mesh")
    values = np.frombuffer(result.stdout, dtype=wire_dtype)
    arrays = []
    offset = 0
    for shape in shapes:
        count = int(np.prod(shape))
        arrays.append(values[offset:offset + count].reshape(shape).copy())
        offset += count
    return arrays


def vector_geometry(mesh: Any, **options):
    from .vector import VectorGeometry

    cell, edge, *_ = _arrays(mesh)
    dtype = np.result_type(cell.dtype, edge.dtype)
    return VectorGeometry(*_run(mesh, mode=1, dtype=dtype, **options))


def reconstruction_coefficients(mesh: Any, *, include_halos: bool = True):
    cell, edge, *_ = _arrays(mesh)
    if cell.dtype == np.result_type(cell.dtype, edge.dtype):
        return _run(mesh, mode=0, dtype=cell.dtype, include_halos=include_halos)[3]
    geometry = vector_geometry(mesh)
    return _run(mesh, mode=2, dtype=cell.dtype, include_halos=include_halos,
                geometry=geometry)[0]


def derived_geometry(mesh: Any):
    """Compute the four init fields with one bridge call on ordinary meshes."""
    cell, edge, *_ = _arrays(mesh)
    if cell.dtype == np.result_type(cell.dtype, edge.dtype):
        arrays = _run(mesh, mode=0, dtype=cell.dtype)
    else:
        geometry = vector_geometry(mesh)
        coefficients = _run(mesh, mode=2, dtype=cell.dtype, geometry=geometry)[0]
        arrays = [*geometry, coefficients]
    return dict(zip(("edgeNormalVectors", "localVerticalUnitVectors",
                     "cellTangentPlane", "coeffs_reconstruct"), arrays))
