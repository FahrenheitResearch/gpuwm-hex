"""MPAS-A v8.4.1 height-dependent acoustic off-centering coefficients.

This module is deliberately additive while the v8.2.3 whole-step authority
remains available as a regression baseline.  The arithmetic is transcribed
from ``MPAS-Model v8.4.1``
``src/core_atmosphere/mpas_atm_core.F::atm_compute_damping_coefs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import ConfigurationRefusal

FloatArray = NDArray[np.floating[Any]]

V841_OFFCENTERING_SOURCE = (
    "MPAS-A v8.4.1 mpas_atm_core.F::atm_compute_damping_coefs; "
    "tag-object=2a934b5008a7446df96d550bf2e21466feaec686; "
    "commit=91c5eac175eebeaf4206bacd5cb50c39dff3c152"
)


@dataclass(frozen=True, slots=True)
class AcousticOffcenteringV841:
    """The four level arrays consumed by the v8.4.1 acoustic solve."""

    etp: FloatArray
    etm: FloatArray
    ewp: FloatArray
    ewm: FloatArray
    height_u_levels: FloatArray
    height_w_levels: FloatArray


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def build_v841_acoustic_offcentering(
    rdzw: FloatArray,
    *,
    minimum: float = 0.1,
    maximum: float = 0.5,
    transition_bottom_z: float = 30_000.0,
    transition_top_z: float = 50_000.0,
) -> AcousticOffcenteringV841:
    """Build v8.4.1 ``etp/etm/ewp/ewm`` in source loop order.

    ``rdzw`` is the one-dimensional reciprocal computational-layer thickness
    stored on the MPAS mesh.  Heights are reconstructed cumulatively exactly
    as the atmosphere core does; terrain-following physical ``zgrid`` is not
    used for this profile.
    """

    reciprocal = np.asarray(rdzw)
    if reciprocal.ndim != 1 or reciprocal.size == 0:
        _refuse(
            "rdzw",
            f"shape={reciprocal.shape}",
            "v8.4.1 constructs one coefficient per vertical level",
            "a non-empty one-dimensional rdzw array",
        )
    if reciprocal.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("rdzw must use float32 or float64 RKIND storage")
    if np.any(~np.isfinite(reciprocal)) or np.any(reciprocal <= 0.0):
        _refuse(
            "rdzw",
            "non-finite or non-positive",
            "v8.4.1 reconstructs heights with 1/rdzw",
            "finite positive reciprocal layer thicknesses",
        )

    dtype = reciprocal.dtype
    zero = dtype.type(0.0)
    one = dtype.type(1.0)
    half = dtype.type(0.5)
    minimum_r = dtype.type(minimum)
    maximum_r = dtype.type(maximum)
    lower = dtype.type(transition_bottom_z)
    upper = dtype.type(transition_top_z)

    for knob, original, converted in (
        ("config_epssm_minimum", minimum, minimum_r),
        ("config_epssm_maximum", maximum, maximum_r),
        ("config_epssm_transition_bottom_z", transition_bottom_z, lower),
        ("config_epssm_transition_top_z", transition_top_z, upper),
    ):
        if not np.isfinite(converted):
            _refuse(knob, original, "the v8.4.1 profile requires a finite value", "a finite real")
    if minimum_r < zero or minimum_r > one:
        _refuse(
            "config_epssm_minimum",
            minimum,
            "v8.4.1 off-centering must remain between zero and one",
            "0 <= config_epssm_minimum <= 1",
        )
    if maximum_r < minimum_r or maximum_r > one:
        _refuse(
            "config_epssm_maximum",
            maximum,
            "v8.4.1 requires an ordered bounded off-centering profile",
            "config_epssm_minimum <= config_epssm_maximum <= 1",
        )
    if lower < zero:
        _refuse(
            "config_epssm_transition_bottom_z",
            transition_bottom_z,
            "the transition must begin at or above ground",
            "config_epssm_transition_bottom_z >= 0",
        )
    if upper <= lower:
        _refuse(
            "config_epssm_transition_top_z",
            transition_top_z,
            "the transition interval must have positive width above ground",
            "0 <= bottom < top",
        )

    nlev = int(reciprocal.size)
    height_w = np.zeros(nlev + 1, dtype=dtype)
    height_u = np.empty(nlev, dtype=dtype)
    with np.errstate(over="ignore"):
        for level in range(nlev):
            height_w[level + 1] = height_w[level] + one / reciprocal[level]
            height_u[level] = half * (height_w[level + 1] + height_w[level])
            if not np.isfinite(height_w[level + 1]) or not np.isfinite(height_u[level]):
                _refuse(
                    "rdzw",
                    f"overflow at level {level}",
                    "the v8.4.1 cumulative computational height must remain finite",
                    "rdzw values whose source-order reciprocal sum is finite",
                )

    transition_width = upper - lower
    pi = np.arccos(dtype.type(-1.0))

    def coefficient(height: np.floating[Any]) -> np.floating[Any]:
        if height <= lower:
            return minimum_r
        if height >= upper:
            return maximum_r
        z = (height - lower) / transition_width
        return minimum_r + np.sin(half * pi * z) ** 2 * (
            maximum_r - minimum_r
        )

    etp = np.empty(nlev, dtype=dtype)
    etm = np.empty(nlev, dtype=dtype)
    for level in range(nlev):
        epssm = dtype.type(coefficient(height_u[level]))
        etp[level] = half * (one + epssm)
        etm[level] = half * (one - epssm)

    ewp = np.empty(nlev + 1, dtype=dtype)
    ewm = np.empty(nlev + 1, dtype=dtype)
    for level in range(nlev + 1):
        epssm = dtype.type(coefficient(height_w[level]))
        ewp[level] = half * (one + epssm)
        ewm[level] = half * (one - epssm)

    return AcousticOffcenteringV841(
        etp=etp,
        etm=etm,
        ewp=ewp,
        ewm=ewm,
        height_u_levels=height_u,
        height_w_levels=height_w,
    )
