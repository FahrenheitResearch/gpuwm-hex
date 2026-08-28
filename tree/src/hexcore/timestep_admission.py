"""Geometry-bound timestep admission for registered MPAS meshes.

The nominal mesh resolution is descriptive metadata.  Stability admission uses
the real finite positive ``dcEdge`` array from the Earth-scaled static/mesh
artifact.  No function in this module silently changes a requested timestep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


class TimestepAdmissionError(RuntimeError):
    """A mesh/timestep pair is malformed or fails the declared Courant rule."""


@dataclass(frozen=True, slots=True)
class EdgeLengthAuthority:
    source: str
    count: int
    dtype: str
    raw_sha256: str
    minimum_m: float
    percentile_0_1_m: float
    percentile_1_m: float
    percentile_5_m: float
    median_m: float
    maximum_m: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CourantPolicy:
    schema: str = "gpuwm-hex.outer-step-courant/v1"
    max_characteristic_speed_m_s: float = 125.0
    safety_factor: float = 0.90
    description: str = (
        "outer RK horizontal transport/gravity-wave preflight; acoustic modes "
        "remain governed by the split-explicit substep configuration"
    )

    def validate(self) -> None:
        if not math.isfinite(self.max_characteristic_speed_m_s) or self.max_characteristic_speed_m_s <= 0.0:
            raise TimestepAdmissionError(
                "Courant policy max_characteristic_speed_m_s must be finite and positive"
            )
        if not math.isfinite(self.safety_factor) or not 0.0 < self.safety_factor <= 1.0:
            raise TimestepAdmissionError(
                "Courant policy safety_factor must lie in (0, 1]"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TimestepAdmission:
    requested_dt_seconds: float
    resolved_dt_seconds: float
    maximum_admitted_dt_seconds: float
    estimated_outer_courant: float
    admitted_courant: float
    recommended_dt_seconds: float
    auto_shrunk: bool
    authority: EdgeLengthAuthority
    policy: CourantPolicy

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_dt_seconds": self.requested_dt_seconds,
            "resolved_dt_seconds": self.resolved_dt_seconds,
            "maximum_admitted_dt_seconds": self.maximum_admitted_dt_seconds,
            "estimated_outer_courant": self.estimated_outer_courant,
            "admitted_courant": self.admitted_courant,
            "recommended_dt_seconds": self.recommended_dt_seconds,
            "auto_shrunk": self.auto_shrunk,
            "edge_length_authority": self.authority.as_dict(),
            "courant_policy": self.policy.as_dict(),
        }


def _raw_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def edge_length_authority(
    dc_edge: object,
    *,
    source: str = "static.dcEdge",
) -> EdgeLengthAuthority:
    """Validate and fingerprint the physical edge-length authority.

    Invalid values are refused rather than filtered: filtering a zero, negative,
    NaN, or infinity could make an unsafe mesh appear to have a larger minimum.
    """

    raw = np.asarray(dc_edge)
    if raw.ndim != 1 or raw.size == 0:
        raise TimestepAdmissionError(
            f"{source} must be a non-empty one-dimensional nEdges array, got {raw.shape}"
        )
    if raw.dtype.kind != "f":
        raise TimestepAdmissionError(
            f"{source} must be floating point physical metres, got {raw.dtype}"
        )
    finite = np.isfinite(raw)
    positive = raw > 0.0
    if not np.all(finite) or not np.all(positive):
        bad = np.flatnonzero(~finite | ~positive)
        sample = [
            {"edge": int(index), "value": float(raw[index])}
            for index in bad[:8]
        ]
        raise TimestepAdmissionError(
            f"{source} contains {bad.size} non-finite or non-positive lengths; "
            f"first bad entries={sample}. A timestep cannot be admitted from corrupted geometry. "
            "Regenerate or repair the grid/static pair."
        )
    values = np.asarray(raw, dtype=np.float64)
    p = np.percentile(values, [0.0, 0.1, 1.0, 5.0, 50.0, 100.0], method="linear")
    return EdgeLengthAuthority(
        source=source,
        count=int(raw.size),
        dtype=str(raw.dtype),
        raw_sha256=_raw_sha256(raw),
        minimum_m=float(p[0]),
        percentile_0_1_m=float(p[1]),
        percentile_1_m=float(p[2]),
        percentile_5_m=float(p[3]),
        median_m=float(p[4]),
        maximum_m=float(p[5]),
    )


def admit_timestep(
    requested_dt_seconds: float,
    authority: EdgeLengthAuthority,
    *,
    policy: CourantPolicy | None = None,
) -> TimestepAdmission:
    """Admit or refuse a declared timestep; never mutate it automatically."""

    selected = policy or CourantPolicy()
    selected.validate()
    dt = float(requested_dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise TimestepAdmissionError(
            f"requested dt={requested_dt_seconds!r} s is not finite and positive; declare dt_seconds>0"
        )
    maximum = (
        selected.safety_factor
        * authority.minimum_m
        / selected.max_characteristic_speed_m_s
    )
    courant = selected.max_characteristic_speed_m_s * dt / authority.minimum_m
    admitted_courant = selected.safety_factor
    # The recommendation is informational and deliberately rounded down.  It
    # is never substituted for the user's/registry's declared value.
    recommended = math.floor(maximum * 10.0) / 10.0
    tolerance = max(1.0e-12, 8.0 * math.ulp(maximum))
    if dt > maximum + tolerance:
        raise TimestepAdmissionError(
            "unsafe mesh/timestep combination refused before CUDA allocation: "
            f"min(dcEdge)={authority.minimum_m:.9g} m from {authority.source}, "
            f"requested dt={dt:.9g} s, policy speed={selected.max_characteristic_speed_m_s:.9g} m/s, "
            f"safety factor={selected.safety_factor:.9g}, computed maximum dt={maximum:.9g} s "
            f"(estimated Courant={courant:.9g}, admitted <= {admitted_courant:.9g}). "
            f"Declare dt_seconds <= {recommended:.1f} s or use a mesh with a larger real minimum dcEdge. "
            "The runtime will not auto-shrink the timestep."
        )
    return TimestepAdmission(
        requested_dt_seconds=dt,
        resolved_dt_seconds=dt,
        maximum_admitted_dt_seconds=float(maximum),
        estimated_outer_courant=float(courant),
        admitted_courant=float(admitted_courant),
        recommended_dt_seconds=float(recommended),
        auto_shrunk=False,
        authority=authority,
        policy=selected,
    )
