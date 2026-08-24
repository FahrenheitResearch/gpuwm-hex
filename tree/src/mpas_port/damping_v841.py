"""MPAS-A v8.4.1 vertical-velocity damping profile."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]

V841_DSS_SOURCE = (
    "MPAS-A v8.4.1 mpas_atm_core.F::atm_compute_damping_coefs; "
    "commit=91c5eac175eebeaf4206bacd5cb50c39dff3c152"
)


def build_v841_vertical_velocity_damping(
    zgrid: FloatArray,
    *,
    xnutr: float,
    damping_start_z: float,
) -> FloatArray:
    """Transcribe the v8.4.1 ``dss`` loop without the removed mesh scaling.

    The v8.2.3 source divided active coefficients by
    ``meshDensity(cell)**0.25``.  v8.4.1 intentionally removed that operation;
    this function does not accept a mesh-density field so the old behavior
    cannot silently return.
    """

    heights = np.asarray(zgrid)
    if heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] == 0:
        raise ConfigurationRefusal(
            "zgrid",
            f"shape={heights.shape}",
            "v8.4.1 requires nVertLevels+1 interfaces for every cell",
            "zgrid with shape (nVertLevels+1, nCells)",
        )
    if heights.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("zgrid must use float32 or float64 RKIND storage")
    if np.any(~np.isfinite(heights)) or np.any(np.diff(heights, axis=0) <= 0.0):
        raise ConfigurationRefusal(
            "zgrid",
            "non-finite or non-monotonic",
            "the v8.4.1 damping phase requires finite increasing columns",
            "finite strictly increasing zgrid columns",
        )

    dtype = heights.dtype
    xnutr_r = dtype.type(xnutr)
    start = dtype.type(damping_start_z)
    if (
        not np.isfinite(xnutr_r)
        or xnutr_r < dtype.type(0.0)
        or xnutr_r > dtype.type(1.0)
    ):
        raise ConfigurationRefusal(
            "config_xnutr",
            xnutr,
            "v8.4.1 requires a finite bounded damping coefficient",
            "0 <= config_xnutr <= 1",
        )
    if not np.isfinite(start) or start <= dtype.type(0.0):
        raise ConfigurationRefusal(
            "config_zd",
            damping_start_z,
            "v8.4.1 requires a finite positive damping start height",
            "config_zd > 0",
        )

    nlev = heights.shape[0] - 1
    ncells = heights.shape[1]
    dss = np.zeros((nlev, ncells), dtype=dtype)
    if xnutr_r == dtype.type(0.0):
        return dss

    half = dtype.type(0.5)
    pi = np.arccos(dtype.type(-1.0))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for cell in range(ncells):
            top = heights[nlev, cell]
            for level in range(nlev):
                height = half * (heights[level, cell] + heights[level + 1, cell])
                if not np.isfinite(height):
                    raise ConfigurationRefusal(
                        "zgrid",
                        f"midpoint overflow at level {level}, cell {cell}",
                        "the v8.4.1 damping phase requires finite layer midpoints",
                        "zgrid whose source-order midpoint operations remain finite",
                    )
                if height > start:
                    if top == start:
                        raise ConfigurationRefusal(
                            "config_zd",
                            damping_start_z,
                            "the active v8.4.1 damping phase would divide by zero",
                            "config_zd different from each active column top",
                        )
                    phase = half * pi * (height - start) / (top - start)
                    value = xnutr_r * np.sin(phase) ** dtype.type(2.0)
                    if not np.isfinite(phase) or not np.isfinite(value):
                        raise ConfigurationRefusal(
                            "zgrid",
                            f"non-finite damping phase at level {level}, cell {cell}",
                            "the v8.4.1 damping arithmetic must remain finite",
                            "zgrid and config_zd producing a finite damping phase",
                        )
                    dss[level, cell] = value
    return dss
