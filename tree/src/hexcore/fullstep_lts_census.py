"""How many cells each full-step LTS rate class gets from the mesh alone, before any device object exists.

CPU only, mesh geometry only (no state).  The forecast door's class clock reads the top class from here, so a
default step is moved for the coarsest class the plan can really have, not for the ladder's ceiling.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: The top rate both executors use when ``--fullstep-lts-max-rate`` is not given: ``ForecastRequest`` (ArWen route,
#: drivers/fullstep_lts_forecast_v841) and ``mpasphys.lts_v842.run_lts_classes`` (MPAS suite route).  THE BREAKAGE
#: A SECOND DEFAULT CAUSED (HEX-LAM-VS-ZOOM-2026-10-27 defect 1): the door's class clock took the config dataclass's
#: ladder ceiling (16) while both executors plan at most 8, so a bare LTS run on the 96 -> 3 km zoom globe moved its
#: step from 18 s to 12.5 s (16 x 12.5 = 200 s) for a plan whose coarsest class is 8 (8 x 12.5 = 100 s): 20 % more
#: steps than the 15 s the plan needed (135.1 vs 105.9 s per forecast hour on an RTX 5090).
DOOR_DEFAULT_MAX_RATE = 8


def read_mesh_edges(path: str | os.PathLike | None) -> tuple[np.ndarray, np.ndarray, int] | None:
    """``(dcEdge in metres, 0-based cellsOnEdge, nCells)`` from an MPAS grid or init file, or None if unreadable."""

    if path is None or not Path(path).is_file():
        return None
    try:
        from netCDF4 import Dataset
        with Dataset(str(path)) as dataset:
            if not {"dcEdge", "cellsOnEdge"} <= set(dataset.variables) or "nCells" not in dataset.dimensions:
                return None
            dc = np.asarray(dataset["dcEdge"][:], dtype=np.float64)
            coe = np.asarray(dataset["cellsOnEdge"][:], dtype=np.int64) - 1
            n_cells = int(dataset.dimensions["nCells"].size)
    except (OSError, ImportError, KeyError, ValueError):
        return None
    if dc.size and dc.max() < 10.0:
        # a unit-sphere grid file, scaled as fullstep_lts_admission.metric_lengths scales it
        from .fullstep_lts_admission import EARTH_RADIUS_M
        dc = dc * EARTH_RADIUS_M
    return dc, coe, n_cells


def mesh_class_census(dc_edge_m: np.ndarray, cells_on_edge: np.ndarray, n_cells: int, *,
                      max_rate: int) -> dict[int, int]:
    """Cells per rate class from the mesh alone, the admission's own steps in its own order.

    ``fullstep_lts_admission.admit_mesh_rates`` without the initial-state bounds: the geometric dyadic ladder
    capped at ``max_rate``, the convective hold (cells finer than ``GPUWM_HEX_LTS_CONVECTIVE_SPACING_M``, default
    8 km, stay at rate 1), the two-ring buffer and the factor-2 grading.  The state bounds, the buffer and the
    grading only lower rates, so ``max(census)`` bounds the plan's top class from above, and both are powers of two.
    """

    from .fullstep_lts_admission import MOVING_CONVECTIVE_SPACING_M
    from .fullstep_lts_classes import DYADIC, cell_spacing, grade_rates, inner_edges
    from .fullstep_lts import buffer_rates

    coe = np.asarray(cells_on_edge, dtype=np.int64)
    spacing = cell_spacing(np.asarray(dc_edge_m, dtype=np.float64), coe, int(n_cells))
    rates = np.ones(int(n_cells), dtype=np.int32)
    for rate in DYADIC[1:]:
        if rate <= int(max_rate):
            rates[spacing >= spacing.min() * rate] = rate
    convective = float(os.environ.get("GPUWM_HEX_LTS_CONVECTIVE_SPACING_M", MOVING_CONVECTIVE_SPACING_M))
    if convective > 0.0:
        rates[(spacing < convective) & (rates > 1)] = 1
    if int(rates.max()) > 1:
        rates = grade_rates(buffer_rates(rates, inner_edges(coe)), coe)
    values, counts = np.unique(rates, return_counts=True)
    return {int(rate): int(count) for rate, count in zip(values, counts)}
