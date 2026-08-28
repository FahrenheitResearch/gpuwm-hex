"""Local time stepping: cell/edge rate classing for the v8.4.1 acoustic solver.

OPT-IN.  Native MPAS-A v8.4.1 has no local time stepping -- ``Registry.xml``
offers SRK3 only and ``dts`` is one scalar in every acoustic routine
(``mpas_atm_time_integration.F:3305,3382,3665,3864,4122``).  There is therefore
no byte-identical implementation, and the divergence is a declared choice the
user makes by turning the option on.  Everything here is inert unless
``config_local_timestep`` is set.

What the classing does
----------------------
The acoustic sub-step size is limited by the sound-wave Courant number, which
scales with the smallest cell-to-cell distance a column touches.  A cell whose
own spacing is ``r`` times the mesh minimum can legally take ``r`` times the
sub-step.  This module assigns each cell an integer *rate* ``r`` drawn from a
declared ladder, and each edge the finer of its two cells' rates.

``h_c`` is ``min(dcEdge)`` over the cell's own edges, read from the grid file.
``nominalMinDc`` is NOT used: it was measured 10.5% / 23.6% / 39.7% high on
three meshes (uniform x1.40962, published x4.163842, a 15 km box refinement),
i.e. wrong in the unsafe direction by up to 40%.

Admissible rates
----------------
A rate ``r`` skips ``r-1`` of every ``r`` acoustic sub-steps, so ``r`` must
divide every stage's sub-step count.  For the pinned v8.4.1 schedule
``(1, 3, 6)`` (``mpas_atm_time_integration.F:638-686`` transcribed in
``integration.RKSchedule.from_mpas``) the admissible ladder is ``{1, 3}``:
2 and 4 do not divide 3, so a rate-2 or rate-4 class would have to change the
RK2 stage's sub-step count, which is a different scheme rather than a different
rate.  ``admissible_rates`` derives this from the schedule instead of assuming
it, so a mesh run with a different ``config_number_of_sub_steps`` gets its own
ladder.

Uniform meshes
--------------
On a quasi-uniform mesh every ratio is below the smallest non-unit rate, so
every cell lands in class 0, ``interface_edges`` is empty and every index list
is the identity permutation.  The solver then executes one launch per kernel
over ``arange(n)`` with the schedule's own ``dts`` -- the same arithmetic on the
same cells in the same order as the option-off path.  That is the proof the
dycore pin survives the option existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]


def admissible_rates(stage_acoustic_steps: Sequence[int]) -> tuple[int, ...]:
    """Return every rate that divides all stages' sub-step counts.

    A stage with a single sub-step (RK1 in the released schedule) cannot be
    sub-sampled by any rate above one, so it is exempt: every class takes that
    one sub-step.  Requiring ``r | 1`` would collapse the ladder to ``{1}`` and
    silently disable the feature, which is exactly the kind of quiet no-op the
    A/B ladder screens are meant to catch.
    """

    counts = [int(value) for value in stage_acoustic_steps]
    if not counts or any(value < 1 for value in counts):
        raise ConfigurationRefusal(
            "stage_acoustic_steps",
            tuple(counts),
            "every RK stage must declare at least one acoustic sub-step",
            "a positive sub-step count per stage",
        )
    divisible = [value for value in counts if value > 1]
    ceiling = max(divisible) if divisible else 1
    rates = [
        rate
        for rate in range(1, ceiling + 1)
        if all(value % rate == 0 for value in divisible)
    ]
    return tuple(rates)


def cell_min_spacing(
    dc_edge: NDArray[Any],
    edges_on_cell: NDArray[Any],
    n_edges_on_cell: NDArray[Any],
    *,
    one_based: bool = True,
) -> FloatArray:
    """``h_c`` = the smallest ``dcEdge`` on each cell, in the file's units."""

    dc = np.asarray(dc_edge, dtype=np.float64)
    eoc = np.asarray(edges_on_cell)
    counts = np.asarray(n_edges_on_cell).astype(np.int64, copy=False)
    if dc.ndim != 1 or eoc.ndim != 2 or counts.ndim != 1:
        raise ValueError("dcEdge(nEdges), edgesOnCell(nCells,maxEdges), nEdgesOnCell(nCells)")
    if eoc.shape[0] != counts.shape[0]:
        raise ValueError("edgesOnCell and nEdgesOnCell disagree on nCells")
    if not np.all(np.isfinite(dc)) or np.any(dc <= 0.0):
        raise ValueError("dcEdge must be finite and positive")
    n_cells, max_edges = eoc.shape
    slots = np.arange(max_edges, dtype=np.int64)[None, :]
    live = slots < counts[:, None]
    index = eoc.astype(np.int64, copy=False) - (1 if one_based else 0)
    safe = np.where(live & (index >= 0), index, 0)
    values = np.where(live & (index >= 0), dc[safe], np.inf)
    result = values.min(axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("every cell must own at least one edge")
    return result


@dataclass(frozen=True, slots=True)
class LocalTimestepClassing:
    """Per-cell and per-edge acoustic rates plus their launch index lists."""

    rates: tuple[int, ...]
    cell_rate: IntArray
    edge_rate: IntArray
    cell_lists: tuple[IntArray, ...]
    edge_lists: tuple[IntArray, ...]
    interface_edges: IntArray
    buffer_rings: int
    h_min: float
    h_cell: FloatArray

    @property
    def n_cells(self) -> int:
        return int(self.cell_rate.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_rate.size)

    @property
    def is_single_class(self) -> bool:
        """True when the mesh admits no coarsening: the no-op configuration."""

        return len(self.rates) == 1 and int(self.rates[0]) == 1

    @property
    def identity_permutation(self) -> bool:
        """True when every launch list is ``arange`` over the whole domain."""

        if len(self.cell_lists) != 1 or len(self.edge_lists) != 1:
            return False
        cells = self.cell_lists[0]
        edges = self.edge_lists[0]
        return (
            cells.size == self.n_cells
            and edges.size == self.n_edges
            and bool(np.array_equal(cells, np.arange(self.n_cells, dtype=np.int32)))
            and bool(np.array_equal(edges, np.arange(self.n_edges, dtype=np.int32)))
        )

    def cell_steps(self, stage_acoustic_steps: int) -> IntArray:
        """Sub-step count each cell actually executes in a stage."""

        total = int(stage_acoustic_steps)
        rate = np.where(self.cell_rate > total, total, self.cell_rate)
        return np.maximum(total // rate, 1).astype(np.int32)

    def edge_steps(self, stage_acoustic_steps: int) -> IntArray:
        total = int(stage_acoustic_steps)
        rate = np.where(self.edge_rate > total, total, self.edge_rate)
        return np.maximum(total // rate, 1).astype(np.int32)

    def arithmetic_acoustic_saving(self, stage_acoustic_steps: Sequence[int]) -> float:
        """Fraction of per-cell acoustic sub-step work the classing removes.

        Counting only cell work, and only the acoustic sub-steps: this is an
        arithmetic bound, not a measurement.  A stage with one sub-step
        contributes its sub-step to every class, which is why the bound is
        below ``1 - mean(h_min/h_c)``.
        """

        full = 0
        actual = 0
        for count in stage_acoustic_steps:
            total = int(count)
            full += total * self.n_cells
            actual += int(self.cell_steps(total).sum())
        if full == 0:
            return 0.0
        return 1.0 - (actual / full)

    def summary(self) -> dict[str, Any]:
        return {
            "rates": list(self.rates),
            "buffer_rings": int(self.buffer_rings),
            "n_cells": self.n_cells,
            "n_edges": self.n_edges,
            "h_min": float(self.h_min),
            "h_cell_max_over_min": float(self.h_cell.max() / self.h_cell.min()),
            "cells_per_rate": {
                int(rate): int(self.cell_lists[index].size)
                for index, rate in enumerate(self.rates)
            },
            "edges_per_rate": {
                int(rate): int(self.edge_lists[index].size)
                for index, rate in enumerate(self.rates)
            },
            "interface_edges": int(self.interface_edges.size),
            "interface_edge_fraction": (
                float(self.interface_edges.size) / float(self.n_edges)
                if self.n_edges
                else 0.0
            ),
            "single_class": bool(self.is_single_class),
            "identity_permutation": bool(self.identity_permutation),
        }


def classify_local_timestep(
    *,
    dc_edge: NDArray[Any],
    edges_on_cell: NDArray[Any],
    n_edges_on_cell: NDArray[Any],
    cells_on_edge: NDArray[Any],
    rates: Sequence[int],
    buffer_rings: int = 1,
    one_based: bool = True,
    safety_factor: float = 1.0,
) -> LocalTimestepClassing:
    """Assign acoustic rates from the grid file's own ``dcEdge`` array.

    ``safety_factor`` multiplies the measured ratio before the ladder lookup;
    values below one make the classing more conservative.  It exists so a mesh
    with a marginal ratio can be pushed to the safe side without editing the
    ladder, and defaults to no adjustment.
    """

    ladder = tuple(sorted({int(value) for value in rates}))
    if not ladder or ladder[0] != 1 or any(value < 1 for value in ladder):
        raise ConfigurationRefusal(
            "config_local_timestep_rates",
            tuple(rates),
            "the rate ladder must start at 1 so the finest cells keep the "
            "schedule's own sub-step",
            "a ladder such as (1, 3)",
        )
    rings = int(buffer_rings)
    if rings < 1:
        raise ConfigurationRefusal(
            "config_local_timestep_buffer_rings",
            buffer_rings,
            "a class boundary with no buffer puts the rate jump directly on "
            "the cells whose fluxes are already mismatched in time",
            "config_local_timestep_buffer_rings>=1",
        )
    if not np.isfinite(safety_factor) or safety_factor <= 0.0:
        raise ValueError("safety_factor must be finite and positive")

    h_cell = cell_min_spacing(
        dc_edge, edges_on_cell, n_edges_on_cell, one_based=one_based
    )
    coe = np.asarray(cells_on_edge)
    if coe.ndim != 2 or coe.shape[1] != 2:
        raise ValueError("cellsOnEdge must have shape (nEdges, 2)")
    n_cells = int(h_cell.size)
    n_edges = int(coe.shape[0])
    c0 = coe[:, 0].astype(np.int64, copy=False) - (1 if one_based else 0)
    c1 = coe[:, 1].astype(np.int64, copy=False) - (1 if one_based else 0)
    if np.any(c0 < 0) or np.any(c1 < 0) or np.any(c0 >= n_cells) or np.any(c1 >= n_cells):
        raise ValueError("cellsOnEdge references a cell outside the mesh")

    h_min = float(h_cell.min())
    ratio = (h_cell / h_min) * float(safety_factor)
    ladder_array = np.asarray(ladder, dtype=np.int64)
    # Largest admissible rate not exceeding the cell's own ratio.
    slot = np.searchsorted(ladder_array, ratio, side="right") - 1
    cell_rate = ladder_array[np.clip(slot, 0, ladder_array.size - 1)].astype(np.int32)

    # Buffer: a cell adjacent to a finer cell is demoted to that finer rate.
    # One pass widens the fine region by one ring of cells, so the rate jump
    # sits one cell away from the cells the refinement was placed for.
    for _ring in range(rings):
        neighbour = cell_rate.copy()
        np.minimum.at(neighbour, c0, cell_rate[c1])
        np.minimum.at(neighbour, c1, cell_rate[c0])
        if np.array_equal(neighbour, cell_rate):
            break
        cell_rate = neighbour

    edge_rate = np.minimum(cell_rate[c0], cell_rate[c1]).astype(np.int32)
    interface = np.flatnonzero(cell_rate[c0] != cell_rate[c1]).astype(np.int32)

    present = tuple(int(value) for value in ladder if np.any(cell_rate == value))
    if not present:
        raise RuntimeError("rate classing produced no populated class")
    cell_lists = tuple(
        np.flatnonzero(cell_rate == rate).astype(np.int32) for rate in present
    )
    edge_lists = tuple(
        np.flatnonzero(edge_rate == rate).astype(np.int32) for rate in present
    )
    return LocalTimestepClassing(
        rates=present,
        cell_rate=cell_rate,
        edge_rate=edge_rate,
        cell_lists=cell_lists,
        edge_lists=edge_lists,
        interface_edges=interface,
        buffer_rings=rings,
        h_min=h_min,
        h_cell=h_cell,
    )


def classify_from_grid_file(
    path: str,
    *,
    rates: Sequence[int],
    buffer_rings: int = 1,
    safety_factor: float = 1.0,
) -> LocalTimestepClassing:
    """Read ``dcEdge`` and the connectivity straight out of an MPAS grid file."""

    from netCDF4 import Dataset

    with Dataset(str(path), "r") as dataset:
        missing = [
            name
            for name in ("dcEdge", "edgesOnCell", "nEdgesOnCell", "cellsOnEdge")
            if name not in dataset.variables
        ]
        if missing:
            raise ConfigurationRefusal(
                missing[0],
                None,
                "local time stepping classes cells from the grid file's own "
                "dcEdge, never from nominalMinDc",
                f"an MPAS grid file containing {missing[0]}",
            )
        dc_edge = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges_on_cell = np.asarray(dataset.variables["edgesOnCell"][:])
        n_edges_on_cell = np.asarray(dataset.variables["nEdgesOnCell"][:])
        cells_on_edge = np.asarray(dataset.variables["cellsOnEdge"][:])
    return classify_local_timestep(
        dc_edge=dc_edge,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        cells_on_edge=cells_on_edge,
        rates=rates,
        buffer_rings=buffer_rings,
        safety_factor=safety_factor,
    )


__all__ = [
    "LocalTimestepClassing",
    "admissible_rates",
    "cell_min_spacing",
    "classify_from_grid_file",
    "classify_local_timestep",
]
