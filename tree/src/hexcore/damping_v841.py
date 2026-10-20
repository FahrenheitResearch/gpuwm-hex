"""MPAS-A v8.4.1 vertical-velocity damping profile."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import rkind_libm
from .errors import ConfigurationRefusal
from .prep_cache import cached

FloatArray = NDArray[np.floating[Any]]

V841_DSS_SOURCE = (
    "MPAS-A v8.4.2 mpas_atm_core.F::atm_compute_damping_coefs; "
    "commit=f34984b2c89353cf31c235b08d3b4acd96b37bb3"
)


@cached("v841-vertical-velocity-damping", salt=lambda: __import__("hexcore.rust_hostprep",fromlist=["_engine_digest"])._engine_digest())
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
    if dtype == np.dtype(np.float32):
        # GNU strict core F:1310 uses host sinf, then one RKIND product
        # for the literal square. The Rust correctly-rounded sine can
        # differ from this native library evaluation.
        from .native_geometry_math import sin
        midpoint = half * (heights[:-1] + heights[1:])
        if not np.all(np.isfinite(midpoint)):
            raise ConfigurationRefusal("zgrid", "midpoint overflow",
                "the v8.4.1 damping phase requires finite layer midpoints",
                "zgrid whose source-order midpoint operations remain finite")
        active = midpoint > start
        denominator = np.broadcast_to(heights[-1] - start, midpoint.shape)
        phase = (half * pi) * (midpoint[active] - start) / denominator[active]
        sine = sin(phase)
        values = xnutr_r * (sine * sine)
        if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(values)):
            raise ConfigurationRefusal("zgrid", "non-finite damping phase",
                "the v8.4.1 damping arithmetic must remain finite",
                "zgrid and config_zd producing a finite damping phase")
        dss[active] = values
        return dss
    from .rust_hostprep import vertical_velocity_damping
    try:
        dss=vertical_velocity_damping(heights,xnutr=xnutr_r,start=start,pi=pi)
    except ValueError as error:
        raise ConfigurationRefusal("zgrid",str(error),"the v8.4.1 damping arithmetic must remain finite","zgrid and config_zd producing a finite damping phase") from error
    return dss


def vertical_velocity_damping_reference(heights, *, xnutr, start, pi):
    """The independent scalar authority below, with the Rust table's call signature.

    ``hexcore.rust_hostprep.vertical_velocity_damping`` and this function take
    the same arguments, so either can stand where the other is bound (the
    prep-cache hook tests rebind one to the other).  ``pi`` must be the value
    the cached builder passes, ``arccos(-1)`` in the heights' dtype; the
    authority computes its own and a different value is refused rather than
    ignored.
    """

    heights = np.asarray(heights)
    if pi != np.arccos(heights.dtype.type(-1.0)):
        raise ValueError(f"pi={pi!r} is not arccos(-1) in {heights.dtype}; the scalar authority computes its own")
    return _vertical_velocity_damping_serial(heights, xnutr_r=heights.dtype.type(xnutr),
                                             start=heights.dtype.type(start))


def _vertical_velocity_damping_serial(heights, *, xnutr_r, start):
    """Unchanged independent scalar authority for exactness checks."""
    dtype=heights.dtype; nlev=heights.shape[0]-1; ncells=heights.shape[1]
    damping_start_z=float(start); dss=np.zeros((nlev,ncells),dtype=dtype)
    half=dtype.type(0.5); pi=np.arccos(dtype.type(-1.0))
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
                    # Native ``sin(...)**2``: the integer power compiles to
                    # the exact product; sin follows the pinned
                    # correctly-rounded convention (hexcore.rkind_libm).
                    sin_phase = rkind_libm.sinf_rkind(phase)
                    value = xnutr_r * (sin_phase * sin_phase)
                    if not np.isfinite(phase) or not np.isfinite(value):
                        raise ConfigurationRefusal(
                            "zgrid",
                            f"non-finite damping phase at level {level}, cell {cell}",
                            "the v8.4.1 damping arithmetic must remain finite",
                            "zgrid and config_zd producing a finite damping phase",
                        )
                    dss[level, cell] = value
    return dss
