"""Stream exact initialized terrain and momentum arrays through Rust."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

from .engines import HOSTPREP, EngineRefusal, resolve
from .host_parallel import contiguous_copy
from .prep_cache import cached

ABI_MARKER = b"rw_mpas_hostprep --protocol hex-hostprep-v1"


@lru_cache(maxsize=8)
def _check_binary(path: str, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    found = False
    tail = b""
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            found = found or ABI_MARKER in tail + block
            tail = block[-len(ABI_MARKER):]
    if not found:
        raise EngineRefusal(
            "the host preparation bridge has no hex-hostprep-v1 marker; "
            "a stale executable would interpret initialized arrays with the wrong ABI. "
            "Build rw_mpas_hostprep from the current gpuwm source"
        )
    return digest.hexdigest()


def hostprep_engine() -> tuple[Path, str]:
    path = resolve(HOSTPREP)
    stat = path.stat()
    return path, _check_binary(str(path), stat.st_size, stat.st_mtime_ns)


def _run(*, mode: int, dtype: np.dtype, counts, edges, n_edges: int,
         n_levels: int, arrays, outputs):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("host preparation requires float32 or float64 arrays")
    n_cells, max_edges = edges.shape
    engine, _ = hostprep_engine()
    process = subprocess.Popen([str(engine), "--protocol", "hex-hostprep-v1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    values = []
    try:
        process.stdin.write(struct.pack("<8sQQQQBB", b"HEXHOST1", n_cells,
            n_edges, max_edges, n_levels, dtype.itemsize, mode))
        for array, wire_dtype in [(counts, "<i8"), (edges, "<i8"), *arrays]:
            contiguous = contiguous_copy(array, dtype=wire_dtype)
            data = memoryview(contiguous).cast("B")
            for offset in range(0, len(data), 1024 * 1024):
                process.stdin.write(data[offset:offset + 1024 * 1024])
        process.stdin.close()
        if outputs is None:
            values = json.loads(process.stdout.read())
        for shape, wire_dtype in outputs or ():
            value = np.empty(shape, dtype=wire_dtype)
            data = memoryview(value).cast("B")
            offset = 0
            while offset < len(data):
                read = process.stdout.readinto(data[offset:offset + 1024 * 1024])
                if not read:
                    break
                offset += read
            if offset != len(data):
                message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
                raise ValueError(message or "host preparation returned incomplete arrays")
            values.append(value)
        if process.stdout.read(1):
            raise EngineRefusal("host preparation returned excess bytes; the array ABI is inconsistent")
        message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
        if process.wait():
            raise ValueError(message or "host preparation bridge failed")
    except BrokenPipeError as error:
        message = process.stderr.read().decode("utf-8", errors="replace").strip().removeprefix("host preparation: ")
        raise ValueError(message or "host preparation input was rejected") from error
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BrokenPipeError:
                    pass
        if process.poll() is None:
            process.terminate()
        process.wait()
    return values


def _engine_digest() -> str:
    return hostprep_engine()[1]


@lru_cache(maxsize=8)
def _deformation_capable(path: str, size: int, modified_ns: int) -> bool:
    return b"hex-hostprep-gnu-deformation-v1" in Path(path).read_bytes()


def deformation_engine_digest() -> str:
    """Bind cached weights to the optional bridge, including older installations."""
    from .native_geometry_math import strict_binding
    strict = strict_binding()
    if strict is not None:
        return strict
    try:
        path, digest = hostprep_engine()
    except EngineRefusal:
        return "python"
    stat = path.stat()
    return digest if _deformation_capable(str(path), stat.st_size, stat.st_mtime_ns) else "python"


def deformation_weights(*, counts, edges, cells_on_cell, vertices_on_cell,
                        cells_on_edge, cell_xyz, vertex_xyz, radius, pi):
    """All five GNU binary32 deformation tables, parallel across cell rows.

    An older engine returns None so its Python authority remains available.
    Mode 10 is deliberately unused: it has different intrinsics and only
    three outputs, which cannot supply the native LES gradient coefficients.
    """
    from .native_geometry_math import strict_binding
    if strict_binding() is not None or deformation_engine_digest() == "python":
        return None
    dtype = np.dtype(np.float32)
    wire = dtype.newbyteorder("<")
    arrays = [(np.asarray([vertex_xyz[0].size]), "<i8"),
              (cells_on_cell, "<i8"), (vertices_on_cell, "<i8"),
              (cells_on_edge, "<i8"), (np.asarray([radius, pi]), wire)]
    arrays.extend((a, wire) for a in (*cell_xyz, *vertex_xyz))
    return _run(mode=15, dtype=dtype, counts=counts, edges=edges,
                n_edges=cells_on_edge.shape[0], n_levels=1, arrays=arrays,
                outputs=[(edges.shape, wire)] * 5)


# Mesh-and-terrain only: 10.7 s per run on the published 30 km mesh, the same
# arrays every cycle.  Served by hexcore.prep_cache, keyed on the bridge digest.
@cached("rust-terrain-metrics", salt=_engine_digest)
def terrain_metrics(*, counts, edges, cells_on_edge, zb, zb3, coefficient):
    dtype = np.dtype(zb.dtype)
    nc, me = edges.shape
    nl, _, ne = zb.shape
    wire = dtype.newbyteorder("<")
    return _run(mode=0, dtype=dtype, counts=counts, edges=edges, n_edges=ne,
        n_levels=nl - 1, arrays=[(cells_on_edge, "<i8"),
            (np.asarray([coefficient], dtype=dtype), wire), (zb, wire), (zb3, wire)],
        outputs=[((nc, me), "<i8"), ((nc, me), wire),
                 ((nl, nc, me), wire), ((nl, nc, me), wire)])


def correct_vertical_momentum(*, counts, edges, fzm, fzp, ru, interface_zz,
                              signs, zb_cell, zb3_cell, rw):
    dtype = np.dtype(rw.dtype)
    wire = dtype.newbyteorder("<")
    nl, ne = ru.shape
    values = [np.asarray(value) for value in
              (fzm, fzp, ru, interface_zz, signs, zb_cell, zb3_cell, rw)]
    if any(value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) for value in values):
        raise TypeError("momentum preparation requires float32 or float64 fields")
    mixed = any(value.dtype != dtype for value in values)
    arrays = ([(np.asarray([value.dtype.itemsize for value in values], dtype=np.uint8), "u1")]
              if mixed else [])
    arrays.extend((value, value.dtype.newbyteorder("<")) for value in values)
    return _run(mode=3 if mixed else 1, dtype=dtype, counts=counts, edges=edges, n_edges=ne,
        n_levels=nl, arrays=arrays,
        outputs=[(rw.shape, wire)])[0]


# Called four times per run on the same topology (0.75 s each at 30 km).
@cached("rust-topology-errors", salt=_engine_digest)
def topology_errors(*, n_vertices: int, max_edges2: int, vertex_degree: int,
                    regional: bool, counts, edges, cells_on_cell,
                    vertices_on_cell, cells_on_edge, vertices_on_edge,
                    cells_on_vertex, edges_on_vertex, edge_counts, edges_on_edge):
    """Return the original ordered topology refusals, including graph closure."""
    arrays = [np.asarray([max_edges2, vertex_degree, regional], dtype=np.int64),
        cells_on_cell, vertices_on_cell, cells_on_edge, vertices_on_edge,
        cells_on_vertex, edges_on_vertex, edge_counts, edges_on_edge]
    return _run(mode=2, dtype=np.dtype(np.float64), counts=counts, edges=edges,
        n_edges=cells_on_edge.shape[0], n_levels=n_vertices,
        arrays=[(value, "<i8") for value in arrays], outputs=None)


def compress_advection(*, counts, edges, cells_on_cell, cells_on_edge, ids, deriv, dc, dv, source_order, garbage):
    """Use the shipped parallel Rust compression, preserving each edge's operation order."""
    dtype=deriv.dtype; wire=dtype.newbyteorder('<'); ne=dc.size; nc,me=edges.shape; width=deriv.shape[2]
    return _run(mode=11,dtype=dtype,counts=counts,edges=edges,n_edges=ne,n_levels=1,
        arrays=[(np.asarray([width,source_order,garbage],np.int64),'<i8'),
                (cells_on_cell,'<i8'),(cells_on_edge,'<i8'),(ids,'<i8'),
                (deriv,wire),(dc,wire),(dv,wire)],
        outputs=[((ne,width),wire),((ne,width),wire),((ne,),'<i8'),((ne,width),'<i8')])


def vertical_velocity_damping(heights, *, xnutr, start, pi):
    """Use the shipped RKIND sine and source-order Rust damping loop."""
    dtype=heights.dtype; wire=dtype.newbyteorder('<'); nl=heights.shape[0]-1; nc=heights.shape[1]
    return _run(mode=12,dtype=dtype,counts=np.zeros(nc,np.int64),edges=np.zeros((nc,1),np.int64),
        n_edges=1,n_levels=nl,arrays=[(np.asarray([xnutr,start,pi],dtype=dtype),wire),(heights,wire)],
        outputs=[((nl,nc),wire)])[0]
