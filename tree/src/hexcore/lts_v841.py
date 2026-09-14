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

Limited-area culls
------------------
A regional cull carries a driven boundary zone: ``bdyMaskCell`` 1..7 marks the
relaxation and specified rings that the lateral-boundary series overwrites and
relaxes every RK stage, on a schedule built for one global acoustic rate, and
whose specified cells are advanced by the boundary tendency inside the acoustic
loop rather than solved.  A class interface there would reflux a residual into
a cell whose acoustic state is not the solver's to correct, so every driven
cell is held at rate 1 and the rate jump can only sit in the interior.  The
demotion is recorded per ring in the summary, because on the culls the doors
make (a fine core with a 1.35x cut) it is the whole coarse class: the cells
whose spacing would earn rate 3 all sit in the driven rings, and the option is
inert by construction there -- measured, not assumed, on the 2026-09-13 point
and corridor culls.  An edge with one cell (a ring-7 edge) takes its present
cell's rate and is never an interface.

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
    #: Rate each cell earned from its own spacing alone, before the driven-zone
    #: hold and the buffer demotion.  Equal to ``cell_rate`` on a global mesh
    #: with no buffer demotions.  For an explicit classing
    #: (``rate_source == "explicit"``) this holds the rate the instrument
    #: REQUESTED per cell, not anything the spacing said, and the summary
    #: reports it under ``cells_requested`` rather than
    #: ``cells_qualified_by_spacing`` so a receipt cannot present an
    #: instrument's request as a property of the mesh.
    spacing_rate: IntArray | None = None
    #: Indices of the cells held at rate 1 because they sit in the driven
    #: boundary zone, and the ``bdyMaskCell`` ring of each; both empty on a
    #: global mesh and ``None`` when no mask was supplied.
    driven_cells: IntArray | None = None
    driven_rings: IntArray | None = None
    #: How the driven zone was known: ``"bdyMaskCell"``, ``"explicit"`` or
    #: ``None`` when no mask was supplied.
    driven_zone_source: str | None = None
    #: How the per-cell rates were assigned: ``"spacing"`` (the grid file's
    #: dcEdge) or ``"explicit"`` (an instrument handed the rates in).
    rate_source: str = "spacing"
    #: The two cells of every interface edge, ``(n_interface, 2)``, so a
    #: driver can check the rate jump against its own zone masks.
    interface_cells: IntArray | None = None

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
            "rate_source": self.rate_source,
            "cells_qualified_by_spacing": (
                None
                if self.spacing_rate is None or self.rate_source != "spacing"
                else self._count_per_rate(self.spacing_rate)
            ),
            "cells_requested": (
                None
                if self.spacing_rate is None or self.rate_source != "explicit"
                else self._count_per_rate(self.spacing_rate)
            ),
            "driven_zone_source": self.driven_zone_source,
            "cells_held_in_driven_zone": (
                None if self.driven_cells is None else int(self.driven_cells.size)
            ),
            "cells_held_in_driven_zone_by_ring": (
                None
                if self.driven_rings is None
                else {
                    int(ring): int(count)
                    for ring, count in zip(*np.unique(self.driven_rings, return_counts=True))
                }
            ),
            "cells_demoted_by_buffer": (
                None
                if self.spacing_rate is None
                else int(((self.cell_rate < self.spacing_rate) & ~self.driven_mask()).sum())
            ),
        }

    @staticmethod
    def _count_per_rate(rate: IntArray) -> dict[int, int]:
        return {
            int(value): int((rate == value).sum())
            for value in sorted({int(v) for v in np.unique(rate)})
        }

    def driven_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of the cells held at rate 1 by the driven-zone guard."""

        mask = np.zeros(self.n_cells, dtype=bool)
        if self.driven_cells is not None and self.driven_cells.size:
            mask[np.asarray(self.driven_cells, dtype=np.int64)] = True
        return mask


def _finish_classing(
    *,
    cell_rate: IntArray,
    spacing_rate: IntArray,
    c0: NDArray[np.int64],
    c1: NDArray[np.int64],
    ladder: tuple[int, ...],
    rings: int,
    h_min: float,
    h_cell: FloatArray,
    driven_cells: IntArray | None,
    driven_rings: IntArray | None,
    driven_zone_source: str | None,
    rate_source: str,
) -> LocalTimestepClassing:
    """Buffer demotion, edge rates, interfaces and launch lists.

    ``c0``/``c1`` may hold ``-1`` for an edge with one cell (a limited-area
    ring-7 edge): such an edge takes its present cell's rate, contributes no
    buffer demotion and is never an interface.
    """

    two_sided = (c0 >= 0) & (c1 >= 0)
    a = c0[two_sided]
    b = c1[two_sided]
    # Buffer: a cell adjacent to a finer cell is demoted to that finer rate.
    # One pass widens the fine region by one ring of cells, so the rate jump
    # sits one cell away from the cells the refinement was placed for.
    for _ring in range(rings):
        neighbour = cell_rate.copy()
        np.minimum.at(neighbour, a, cell_rate[b])
        np.minimum.at(neighbour, b, cell_rate[a])
        if np.array_equal(neighbour, cell_rate):
            break
        cell_rate = neighbour

    present_c0 = np.where(c0 >= 0, c0, c1)
    present_c1 = np.where(c1 >= 0, c1, c0)
    edge_rate = np.minimum(
        cell_rate[present_c0], cell_rate[present_c1]
    ).astype(np.int32)
    interface = np.flatnonzero(
        two_sided & (cell_rate[present_c0] != cell_rate[present_c1])
    ).astype(np.int32)
    interface_cells = np.stack(
        [c0[interface], c1[interface]], axis=1
    ).astype(np.int32) if interface.size else np.zeros((0, 2), dtype=np.int32)

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
        cell_rate=cell_rate.astype(np.int32),
        edge_rate=edge_rate,
        cell_lists=cell_lists,
        edge_lists=edge_lists,
        interface_edges=interface,
        buffer_rings=rings,
        h_min=h_min,
        h_cell=h_cell,
        spacing_rate=spacing_rate.astype(np.int32),
        driven_cells=driven_cells,
        driven_rings=driven_rings,
        driven_zone_source=driven_zone_source,
        rate_source=rate_source,
        interface_cells=interface_cells,
    )


def _driven_hold(
    cell_rate: IntArray,
    driven_ring: NDArray[Any] | None,
) -> tuple[IntArray, IntArray | None, IntArray | None]:
    """Hold every driven-zone cell at rate 1; return the held cells and rings."""

    if driven_ring is None:
        return cell_rate, None, None
    ring = np.asarray(driven_ring).astype(np.int64, copy=False)
    if ring.shape != cell_rate.shape:
        raise ValueError("the driven-zone ring array must be one value per cell")
    held = np.flatnonzero((ring > 0) & (cell_rate > 1)).astype(np.int32)
    out = cell_rate.copy()
    out[held] = 1
    return out, held, ring[held].astype(np.int32)


def _edge_cells(
    cells_on_edge: NDArray[Any], n_cells: int, *, one_based: bool
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    coe = np.asarray(cells_on_edge)
    if coe.ndim != 2 or coe.shape[1] != 2:
        raise ValueError("cellsOnEdge must have shape (nEdges, 2)")
    c0 = coe[:, 0].astype(np.int64, copy=False) - (1 if one_based else 0)
    c1 = coe[:, 1].astype(np.int64, copy=False) - (1 if one_based else 0)
    # -1 (a 0 in the file's 1-based convention) is a missing cell: a
    # limited-area edge on the outermost ring.  Any other out-of-range index
    # is a broken file.
    for side in (c0, c1):
        if np.any(side < -1) or np.any(side >= n_cells):
            raise ValueError("cellsOnEdge references a cell outside the mesh")
    if np.any((c0 < 0) & (c1 < 0)):
        raise ValueError("an edge must touch at least one cell")
    return c0, c1


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
    driven_ring: NDArray[Any] | None = None,
    driven_zone_source: str | None = None,
) -> LocalTimestepClassing:
    """Assign acoustic rates from the grid file's own ``dcEdge`` array.

    ``safety_factor`` multiplies the measured ratio before the ladder lookup;
    values below one make the classing more conservative.  It exists so a mesh
    with a marginal ratio can be pushed to the safe side without editing the
    ladder, and defaults to no adjustment.

    ``driven_ring`` is a regional cull's ``bdyMaskCell`` (0 interior, 1..7 the
    driven rings).  Every cell with a nonzero ring is held at rate 1 before
    the buffer pass, for the reason in the module docstring; the held cells
    are recorded on the result so a receipt can say how much of the coarse
    class the guard removed.
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
    n_cells = int(h_cell.size)
    c0, c1 = _edge_cells(cells_on_edge, n_cells, one_based=one_based)

    h_min = float(h_cell.min())
    ratio = (h_cell / h_min) * float(safety_factor)
    ladder_array = np.asarray(ladder, dtype=np.int64)
    # Largest admissible rate not exceeding the cell's own ratio.
    slot = np.searchsorted(ladder_array, ratio, side="right") - 1
    spacing_rate = ladder_array[np.clip(slot, 0, ladder_array.size - 1)].astype(np.int32)
    source = driven_zone_source
    if driven_ring is not None and source is None:
        source = "explicit"
    cell_rate, held, held_rings = _driven_hold(spacing_rate, driven_ring)
    return _finish_classing(
        cell_rate=cell_rate,
        spacing_rate=spacing_rate,
        c0=c0,
        c1=c1,
        ladder=ladder,
        rings=rings,
        h_min=h_min,
        h_cell=h_cell,
        driven_cells=held,
        driven_rings=held_rings,
        driven_zone_source=source,
        rate_source="spacing",
    )


def classify_from_cell_rates(
    *,
    cell_rate: NDArray[Any],
    dc_edge: NDArray[Any],
    edges_on_cell: NDArray[Any],
    n_edges_on_cell: NDArray[Any],
    cells_on_edge: NDArray[Any],
    rates: Sequence[int],
    buffer_rings: int = 1,
    one_based: bool = True,
    driven_ring: NDArray[Any] | None = None,
    driven_zone_source: str | None = None,
) -> LocalTimestepClassing:
    """Build a classing from rates an INSTRUMENT assigned, not from spacing.

    The shipped option classes from ``dcEdge``.  This constructor exists so an
    A/B arm can place a class interface where the spacing would not -- inside
    the fine interior of a cull -- and measure what the interface itself does
    to the fields.  The driven-zone hold and the buffer demotion still apply
    and every rate must be on the ladder.  The result records
    ``rate_source="explicit"`` so a receipt cannot present it as the shipped
    classing.  A rate above what the cell's own spacing admits is allowed
    here on purpose: the acoustic Courant number of the released schedule is
    about 0.11 at the finest edge (dts 0.278 s, 841 m), so a rate-3 class in
    the fine interior runs at about 0.34 and the instrument stays inside the
    split-explicit stability region; the receipt carries the ratio so the
    reader can check.
    """

    ladder = tuple(sorted({int(value) for value in rates}))
    if not ladder or ladder[0] != 1:
        raise ConfigurationRefusal(
            "config_local_timestep_rates",
            tuple(rates),
            "the rate ladder must start at 1",
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
    h_cell = cell_min_spacing(
        dc_edge, edges_on_cell, n_edges_on_cell, one_based=one_based
    )
    n_cells = int(h_cell.size)
    requested = np.asarray(cell_rate).astype(np.int32, copy=False)
    if requested.shape != (n_cells,):
        raise ValueError("cell_rate must hold one rate per cell")
    off_ladder = sorted({int(v) for v in np.unique(requested)} - set(ladder))
    if off_ladder:
        raise ConfigurationRefusal(
            "config_local_timestep_rates",
            tuple(off_ladder),
            "an explicit classing names a rate that is not on the ladder",
            f"rates drawn from {ladder}",
        )
    c0, c1 = _edge_cells(cells_on_edge, n_cells, one_based=one_based)
    h_min = float(h_cell.min())
    spacing_rate = requested.copy()
    source = driven_zone_source
    if driven_ring is not None and source is None:
        source = "explicit"
    held_rate, held, held_rings = _driven_hold(requested, driven_ring)
    return _finish_classing(
        cell_rate=held_rate,
        spacing_rate=spacing_rate,
        c0=c0,
        c1=c1,
        ladder=ladder,
        rings=rings,
        h_min=h_min,
        h_cell=h_cell,
        driven_cells=held,
        driven_rings=held_rings,
        driven_zone_source=source,
        rate_source="explicit",
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
        driven_ring = (
            np.asarray(dataset.variables["bdyMaskCell"][:])
            if "bdyMaskCell" in dataset.variables
            else None
        )
    return classify_local_timestep(
        dc_edge=dc_edge,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        cells_on_edge=cells_on_edge,
        rates=rates,
        buffer_rings=buffer_rings,
        safety_factor=safety_factor,
        driven_ring=driven_ring,
        driven_zone_source=None if driven_ring is None else "bdyMaskCell",
    )


def load_cell_rates(path: str) -> IntArray:
    """Read an instrument's per-cell rate array (``.npz`` key ``cell_rate``)."""

    with np.load(str(path)) as archive:
        if "cell_rate" not in archive.files:
            raise ValueError(f"{path} carries no 'cell_rate' array")
        return np.asarray(archive["cell_rate"]).astype(np.int32)


def classify_from_grid_file_with_rates(
    path: str,
    rates_path: str,
    *,
    rates: Sequence[int],
    buffer_rings: int = 1,
) -> LocalTimestepClassing:
    """The instrument route: connectivity from the grid file, rates from a file."""

    from netCDF4 import Dataset

    cell_rate = load_cell_rates(rates_path)
    with Dataset(str(path), "r") as dataset:
        dc_edge = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges_on_cell = np.asarray(dataset.variables["edgesOnCell"][:])
        n_edges_on_cell = np.asarray(dataset.variables["nEdgesOnCell"][:])
        cells_on_edge = np.asarray(dataset.variables["cellsOnEdge"][:])
        driven_ring = (
            np.asarray(dataset.variables["bdyMaskCell"][:])
            if "bdyMaskCell" in dataset.variables
            else None
        )
    return classify_from_cell_rates(
        cell_rate=cell_rate,
        dc_edge=dc_edge,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        cells_on_edge=cells_on_edge,
        rates=rates,
        buffer_rings=buffer_rings,
        driven_ring=driven_ring,
        driven_zone_source=None if driven_ring is None else "bdyMaskCell",
    )


__all__ = [
    "LocalTimestepClassing",
    "admissible_rates",
    "cell_min_spacing",
    "classify_from_cell_rates",
    "classify_from_grid_file",
    "classify_from_grid_file_with_rates",
    "classify_local_timestep",
    "load_cell_rates",
]
