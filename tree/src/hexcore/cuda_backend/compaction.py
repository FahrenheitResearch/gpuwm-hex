"""Compact slot layouts for padded device mesh tables.

MPAS stores its neighbour tables padded to a fixed slot count: ``maxEdges``
(10 in the standard grid files) per cell, ``maxEdges2`` (20) per edge, and
FIFTEEN advection cells per edge.  A quasi-uniform mesh only uses 6, 10 and
10 of those slots.  Every CUDA reader of these tables loops
``slot < count[entity]`` and takes the slot stride as a launch argument, so
the padding slots are never read.  Storing a table with the stride set to the
largest count the mesh actually has reads the same words in the same order:
the arithmetic, and so every output bit, is unchanged.

A table is compacted only when every dropped slot holds one constant bit
pattern (zeros, or the -1 index sentinel).  Then the full padded layout can be
rebuilt exactly, which is what the dual-run execution fingerprints hash.

Measured cost these tables carried on x1.40962 (CENSUS.md, mem lanes,
2026-10-14): zb_cell + zb3_cell 175 MiB, edgesOnEdge + weightsOnEdge 18.8 MiB,
the advection stencil 21.5 MiB, the physics-prep reconstruction geometry
13.4 MiB.

The same module also gates ``VERTICAL_HX``: ``hx`` (terrain influence on the
coordinate surfaces, 224 B per column) is an init-time field no forecast kernel
reads, so it stays on the host until something asks for it.

Knobs (A/B and fallback): ``GPUWM_HEX_MESH_COMPACT=0`` stores every table at
its file stride and uploads ``hx`` at once.  ``GPUWM_HEX_MESH_COMPACT_<TABLE>=0``
does that for one table, where ``<TABLE>`` is one of ``TERRAIN``,
``EDGE_STENCIL``, ``ADVECTION``, ``PREP_GEOMETRY`` or ``VERTICAL_HX``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..host_parallel import contiguous_copy, run_blocks

TABLES = ("TERRAIN", "EDGE_STENCIL", "ADVECTION", "PREP_GEOMETRY", "VERTICAL_HX")


def compaction_enabled(table: str) -> bool:
    """True unless the master or the per-table knob is set to 0."""

    if table not in TABLES:
        raise ValueError(f"unknown compaction table {table!r}")
    if os.environ.get("GPUWM_HEX_MESH_COMPACT", "1").strip() == "0":
        return False
    return os.environ.get(f"GPUWM_HEX_MESH_COMPACT_{table}", "1").strip() != "0"


def used_width(counts: Any, width: int) -> int:
    """The largest slot count present, clipped to ``[1, width]``."""

    array = np.asarray(counts)
    if array.size == 0:
        return max(1, int(width))
    return int(min(max(int(array.max()), 1), int(width)))


def uniform_tail(array: Any, keep: int, *, axis: int = -1) -> Any | None:
    """The single value every slot at or past ``keep`` holds, else ``None``.

    The comparison is on bit patterns, so a tail of ``-0.0`` does not pass for
    zeros.  An empty tail returns the dtype's zero.
    """

    host = np.asarray(array)
    # Negative starts historically use np.take's wrapped indices. Keep that
    # uncommon behavior, but never materialize the large normal padding tail.
    if int(keep) < 0:
        tail = np.take(host, np.arange(int(keep), host.shape[axis]), axis=axis)
    else:
        index = [slice(None)] * host.ndim
        index[axis] = slice(int(keep), None)
        tail = host[tuple(index)]
    if tail.size == 0:
        return host.dtype.type(0)
    # Same-sized integer views preserve signed zeros and NaN payloads even on
    # strided inputs. Only small bounded boolean blocks are allocated.
    bits = (tail.view(np.dtype(f"u{tail.dtype.itemsize}"))
            if tail.dtype.kind == "f" else tail)
    first = bits.flat[0]
    split_axis = int(np.argmax(bits.shape))
    failures = []

    def check(block: slice) -> None:
        index = [slice(None)] * bits.ndim
        index[split_axis] = block
        if not bool(np.all(bits[tuple(index)] == first)):
            failures.append(True)

    run_blocks(check, bits.shape[split_axis], elements=bits.size, worker_limit=4)
    if failures:
        return None
    return tail.flat[0]


def compact_host(array: Any, keep: int, *, axis: int = -1) -> np.ndarray:
    """A C-contiguous host copy holding the first ``keep`` slots of ``axis``."""

    host = np.asarray(array)
    index = [slice(None)] * host.ndim
    index[axis] = slice(0, int(keep))
    return contiguous_copy(host[tuple(index)], dtype=host.dtype)


def restore_device(array: Any, width: int, fill: Any, *, axis: int = -1) -> Any:
    """Rebuild the padded layout of a compacted device table (transient)."""

    import cupy as cp

    if int(array.shape[axis]) == int(width):
        return array
    shape = list(array.shape)
    shape[axis] = int(width)
    full = cp.full(tuple(shape), fill, dtype=array.dtype)
    index = [slice(None)] * array.ndim
    index[axis] = slice(0, int(array.shape[axis]))
    full[tuple(index)] = array
    return full


__all__ = [
    "TABLES",
    "compact_host",
    "compaction_enabled",
    "restore_device",
    "uniform_tail",
    "used_width",
]
