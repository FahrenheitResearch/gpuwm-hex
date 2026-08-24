"""Gather/scatter between global and partition-local arrays (brief-1, D3).

Array layout law of the existing port: 3D fields are
``(nVertLevels[, +1], nCells|nEdges)`` -- the entity index is the LAST axis.
Gather/scatter therefore always fancy-index the last axis.

HOST MOVES BYTES ONLY.  Nothing in this module performs arithmetic on field
values: no adds, no divides, no dtype conversion.  The only permitted
transformation is an exact-value ``ascontiguousarray``.  Any accumulation
belongs on the device, in the existing kernels.

Scatter is owner-computes: only ``local[..., :n_owned]`` is written back, into
``global_out[..., l2g[:n_owned]]``.  Because owned locals are a disjoint exact
cover of the global index space (asserted at build time by
``partition_assets_v841.assert_exact_cover``), scattering every partition in
any order writes every global element exactly once.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .partition_assets_v841 import PartitionLayout

__all__ = [
    "gather_cells",
    "gather_edges",
    "gather_vertices",
    "scatter_owned_cells",
    "scatter_owned_edges",
    "scatter_owned_vertices",
    "gather_scalars",
    "scatter_owned_scalars",
]


def _gather(global_arr: Any, index: np.ndarray, expect: int, what: str) -> np.ndarray:
    array = np.asarray(global_arr)
    if array.ndim == 0:
        raise ValueError(f"{what}: a scalar has no entity axis to gather")
    if array.shape[-1] != expect:
        raise ValueError(
            f"{what}: last axis {array.shape[-1]} != global {what} count {expect}"
        )
    return np.ascontiguousarray(array[..., index])


def _scatter(
    global_out: np.ndarray,
    local_arr: Any,
    index: np.ndarray,
    n_owned: int,
    expect: int,
    what: str,
) -> None:
    local = np.asarray(local_arr)
    if global_out.shape[-1] != expect:
        raise ValueError(
            f"{what}: destination last axis {global_out.shape[-1]} != {expect}"
        )
    if local.shape[:-1] != global_out.shape[:-1]:
        raise ValueError(
            f"{what}: leading shape {local.shape[:-1]} != {global_out.shape[:-1]}"
        )
    if local.shape[-1] < n_owned:
        raise ValueError(
            f"{what}: local last axis {local.shape[-1]} < owned count {n_owned}"
        )
    if local.dtype != global_out.dtype:
        raise TypeError(
            f"{what}: host scatter never converts dtype "
            f"({local.dtype} into {global_out.dtype})"
        )
    global_out[..., index[:n_owned]] = local[..., :n_owned]


def gather_cells(global_arr: Any, layout: PartitionLayout) -> np.ndarray:
    return _gather(global_arr, layout.cell_l2g, layout._global_cells, "cells")


def gather_edges(global_arr: Any, layout: PartitionLayout) -> np.ndarray:
    return _gather(global_arr, layout.edge_l2g, layout._global_edges, "edges")


def gather_vertices(global_arr: Any, layout: PartitionLayout) -> np.ndarray:
    return _gather(global_arr, layout.vertex_l2g, layout._global_vertices, "vertices")


def scatter_owned_cells(
    global_out: np.ndarray, local_arr: Any, layout: PartitionLayout
) -> None:
    _scatter(
        global_out,
        local_arr,
        layout.cell_l2g,
        layout.n_owned_cells,
        layout._global_cells,
        "cells",
    )


def scatter_owned_edges(
    global_out: np.ndarray, local_arr: Any, layout: PartitionLayout
) -> None:
    _scatter(
        global_out,
        local_arr,
        layout.edge_l2g,
        layout.n_owned_edges,
        layout._global_edges,
        "edges",
    )


def scatter_owned_vertices(
    global_out: np.ndarray, local_arr: Any, layout: PartitionLayout
) -> None:
    _scatter(
        global_out,
        local_arr,
        layout.vertex_l2g,
        layout.n_owned_vertices,
        layout._global_vertices,
        "vertices",
    )


def gather_scalars(global_scalars: Any, layout: PartitionLayout) -> np.ndarray:
    """Gather the ``(6, nlev, nCells)`` scalar block.

    Species order is fixed by the port: ``("qv","qc","qr","qi","qs","qg")``.
    The species axis is untouched; only the trailing cell axis is gathered.
    """

    array = np.asarray(global_scalars)
    if array.ndim != 3 or array.shape[0] != 6:
        raise ValueError("scalars must have shape (6, nVertLevels, nCells)")
    return gather_cells(array, layout)


def scatter_owned_scalars(
    global_out: np.ndarray, local_arr: Any, layout: PartitionLayout
) -> None:
    if global_out.ndim != 3 or global_out.shape[0] != 6:
        raise ValueError("scalars must have shape (6, nVertLevels, nCells)")
    scatter_owned_cells(global_out, local_arr, layout)
