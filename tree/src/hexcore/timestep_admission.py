"""Split-explicit stability admission for a mesh and timestep.

The nominal mesh resolution is descriptive metadata.  Stability admission uses
the real finite positive ``dcEdge`` array from the Earth-scaled static/mesh
artifact.  No function in this module silently changes a requested timestep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import hashlib
import json
import math
from typing import Any

import numpy as np


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


#: MPAS's own gas constants (mpas_constants.F: rgas = 287, cp = 7 rgas / 2,
#: cv = cp - rgas), so gamma = cp / cv = 1.4.
R_DRY = 287.0
CP_DRY = 7.0 * R_DRY / 2.0
CV_DRY = CP_DRY - R_DRY

#: Forward-backward acoustic stepping on the hexagonal C-grid is stable for
#: omega * dtau <= 2.  The largest discrete gradient-divergence frequency on
#: a regular hexagon mesh of centre spacing d is omega = c * sqrt(6) / d
#: (graph-Laplacian eigenvalue 9 on the triangular lattice of centres, times
#: the TRiSK weight dvEdge / (dcEdge * areaCell) = 2 / (3 d^2)), so the
#: acoustic Courant number c * dtau / d must stay at or below 2 / sqrt(6).
ACOUSTIC_COURANT_LIMIT = 2.0 / math.sqrt(6.0)

#: A regular hexagon whose centre spacing is d has area sqrt(3) d^2 / 2 and a
#: largest width 2 d / sqrt(3) across the flow, so the fraction of a cell a
#: wind U sweeps out in dt is at most (4/3) U dt / d.  A one-dimensional
#: Courant limit L therefore becomes (3/4) L in U dt / dcEdge.
#:
#: THIS IS AN ESTIMATE.  It is a swept-area argument that carries a 1-D limit
#: to a hexagon; it is not a von Neumann analysis of MPAS's two-dimensional
#: transport operator, and no measurement in this tree locates the true 2-D
#: limit.  The advective refusal says so and is adjustable (``--max-wind``);
#: the acoustic limit above IS the operator's own eigenvalue bound.
HEXAGON_DIRECTION_FACTOR = 0.75


class TimestepAdmissionError(RuntimeError):
    """A mesh/timestep pair is malformed or fails the split-explicit stability rule."""


@functools.lru_cache(maxsize=64)
def one_dimensional_advective_limit(order: int, coef_3rd_order: float) -> float:
    """The von Neumann Courant limit of MPAS's transport for one RK order.

    The spatial operator is MPAS's own flux3 (mpas_atm_time_integration.F:
    flux4 + coef_3rd_order * |u| * ((q_ip2 - q_im1) - 3 (q_ip1 - q_i)) / 12,
    flux4 = u (7 (q_i + q_ip1) - (q_ip2 + q_im1)) / 12).  The stage weights
    are the source's (mpas_atm_time_integration.F:2066-2092): order 3 is
    (dt/3, dt/2, dt), whose amplification is 1 + z + z^2/2 + z^3/6; order 2
    is (dt/2, dt/2, dt), whose amplification is 1 + z + z^2/2 + z^3/4.  The
    limit is the largest Courant number at which |G| <= 1 at every resolved
    wavenumber, found by bisection.  Measured: coef 0.25 gives 1.520 at
    order 3 and 1.547 at order 2.
    """

    import numpy as np

    if order not in (2, 3):
        raise TimestepAdmissionError(
            f"config_time_integration_order={order!r}: MPAS has order-2 and "
            "order-3 schedules only"
        )
    beta = float(coef_3rd_order)
    theta = np.linspace(1.0e-6, math.pi, 4001)

    def shift(m: int):
        return np.exp(1j * m * theta)

    face = (7.0 * (1.0 + shift(1)) - (shift(-1) + shift(2))) / 12.0 + beta * (
        (shift(2) - shift(-1)) - 3.0 * (shift(1) - 1.0)
    ) / 12.0
    symbol = face * (1.0 - shift(-1))
    third = 1.0 / 6.0 if order == 3 else 0.25

    def stable(courant: float) -> bool:
        z = -courant * symbol
        growth = np.abs(1.0 + z + z * z / 2.0 + third * z ** 3)
        return bool(np.max(growth) <= 1.0 + 1.0e-12)

    low, high = 0.0, 4.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if stable(middle):
            low = middle
        else:
            high = middle
    return low


@dataclass(frozen=True, slots=True)
class SplitExplicitSchedule:
    """The schedule knobs the stability bound depends on (MPAS defaults)."""

    time_integration_order: int = 2
    acoustic_substeps: int = 4
    dynamics_splits: int = 3
    split_dynamics_transport: bool = True
    coef_3rd_order: float = 0.25

    def largest_acoustic_fraction(self) -> float:
        """The largest acoustic sub-step as a fraction of the outer dt.

        mpas_atm_time_integration.F:2066-2092: dt_dynamics = dt / splits
        when transport is split; order 3 steps dt_dynamics/3 in stage 1 and
        dt_dynamics/substeps after it, order 2 steps dt_dynamics/substeps in
        every stage.
        """

        substeps = int(self.acoustic_substeps)
        if substeps < 1:
            raise TimestepAdmissionError(
                f"config_number_of_sub_steps={substeps!r} must be a positive integer")
        splits = int(self.dynamics_splits) if self.split_dynamics_transport else 1
        if splits < 1:
            raise TimestepAdmissionError(
                f"config_dynamics_split_steps={splits!r} must be a positive integer")
        per_stage = 1.0 / substeps
        if int(self.time_integration_order) == 3:
            per_stage = max(1.0 / 3.0, per_stage)
        return per_stage / splits

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CourantPolicy:
    """The physical bounds the stability check measures a timestep against.

    ``max_characteristic_speed_m_s`` is the largest horizontal wind the run
    may meet: 125 m/s sits above the strongest jet-stream winds observed
    (about 115 m/s).  ``max_temperature_k`` bounds the sound speed: 330 K is
    above the hottest near-surface air ever measured (329.9 K), so
    c_max = sqrt(gamma R T_max) = 364 m/s.
    """

    schema: str = "gpuwm-hex.split-explicit-stability/v2"
    max_characteristic_speed_m_s: float = 125.0
    max_temperature_k: float = 330.0
    #: Where the wind bound came from: the default, a mesh row's
    #: ``courant_wave_speed_m_s``, or the run's ``--max-wind``.  It is an
    #: ASSUMED bound either way; nothing here reads the case's winds.
    max_characteristic_speed_source: str = "default"
    description: str = (
        "advective Courant number of the outer step against the von Neumann "
        "limit of MPAS's RK stages and flux3 operator on a hexagon, and the "
        "acoustic Courant number of the largest acoustic sub-step against the "
        "forward-backward limit on the hexagonal C-grid; both from the mesh's "
        "real minimum dcEdge"
    )

    def validate(self) -> None:
        if not math.isfinite(self.max_characteristic_speed_m_s) or self.max_characteristic_speed_m_s <= 0.0:
            raise TimestepAdmissionError(
                "stability policy max_characteristic_speed_m_s must be finite and positive"
            )
        if not math.isfinite(self.max_temperature_k) or self.max_temperature_k <= 0.0:
            raise TimestepAdmissionError(
                "stability policy max_temperature_k must be finite and positive"
            )

    @property
    def max_sound_speed_m_s(self) -> float:
        return math.sqrt(CP_DRY / CV_DRY * R_DRY * float(self.max_temperature_k))

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        record = asdict(self)
        record["max_sound_speed_m_s"] = self.max_sound_speed_m_s
        return record


@dataclass(frozen=True, slots=True)
class TimestepAdmission:
    requested_dt_seconds: float
    resolved_dt_seconds: float
    maximum_admitted_dt_seconds: float
    estimated_outer_courant: float
    admitted_courant: float
    acoustic_courant: float
    acoustic_courant_limit: float
    largest_acoustic_substep_seconds: float
    recommended_dt_seconds: float
    auto_shrunk: bool
    authority: EdgeLengthAuthority
    policy: CourantPolicy
    schedule: SplitExplicitSchedule

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_dt_seconds": self.requested_dt_seconds,
            "resolved_dt_seconds": self.resolved_dt_seconds,
            "maximum_admitted_dt_seconds": self.maximum_admitted_dt_seconds,
            "estimated_outer_courant": self.estimated_outer_courant,
            "admitted_courant": self.admitted_courant,
            "acoustic_courant": self.acoustic_courant,
            "acoustic_courant_limit": self.acoustic_courant_limit,
            "largest_acoustic_substep_seconds": self.largest_acoustic_substep_seconds,
            "recommended_dt_seconds": self.recommended_dt_seconds,
            "auto_shrunk": self.auto_shrunk,
            "edge_length_authority": self.authority.as_dict(),
            "courant_policy": self.policy.as_dict(),
            "schedule": self.schedule.as_dict(),
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


def largest_schedulable_dt(maximum_seconds: float, cadence_seconds: float | None) -> float:
    """The largest timestep at or below ``maximum_seconds`` a run can take.

    With a cadence (the radiation interval) it is the largest WHOLE-SECOND
    divisor of that cadence, because a step that does not divide it is
    refused by the cadence check: recommending 102.4 s on an 1800 s
    radiation interval sent the user from one refusal into the next
    (2026-10-10 audit).  Without one it is ``maximum_seconds`` rounded down
    to a tenth of a second.  0.0 when no whole-second divisor fits.
    """

    maximum = float(maximum_seconds)
    if cadence_seconds is None:
        return math.floor(maximum * 10.0) / 10.0
    cadence = float(cadence_seconds)
    if cadence != round(cadence) or cadence <= 0.0:
        return math.floor(maximum * 10.0) / 10.0
    whole = int(round(cadence))
    for candidate in range(min(whole, int(math.floor(maximum + 1.0e-9))), 0, -1):
        if whole % candidate == 0:
            return float(candidate)
    return 0.0


def admit_timestep(
    requested_dt_seconds: float,
    authority: EdgeLengthAuthority,
    *,
    policy: CourantPolicy | None = None,
    schedule: SplitExplicitSchedule | None = None,
    cadence_seconds: float | None = None,
) -> TimestepAdmission:
    """Admit or refuse a timestep on the mesh's own geometry; never change it.

    Two numbers are measured from the real minimum ``dcEdge`` and refused
    only above the limit the numerics themselves set:

    * advective: ``U_max * dt / min(dcEdge)`` against
      ``(3/4) * L1D(order, coef_3rd_order)``, the von Neumann limit of the
      source's RK stages with its flux3 operator, carried to a hexagon by the
      worst-direction factor (:data:`HEXAGON_DIRECTION_FACTOR`);
    * acoustic: ``c_max * dtau / min(dcEdge)`` against ``2 / sqrt(6)``
      (:data:`ACOUSTIC_COURANT_LIMIT`), with ``dtau`` the largest acoustic
      sub-step the schedule takes.

    Above the acoustic limit a linear mode grows every step.  The advective
    bound is an ESTIMATE on two counts and says so when it refuses: ``U_max``
    is an assumed largest wind (125 m/s by default, a mesh row's
    ``courant_wave_speed_m_s``, or the run's ``--max-wind``), not this
    case's, and the hexagon factor is a swept-area argument, not an analysis
    of the 2-D operator.  MPAS-A's own standard timesteps (6 s per km of
    nominal spacing) pass on the quasi-uniform meshes: 15 km at 90 s
    measures 1.02 against 1.16 and 0.49 against 0.82.

    ``cadence_seconds`` (the radiation interval) makes the recommended step
    one the run can schedule (:func:`largest_schedulable_dt`).
    """

    selected = policy or CourantPolicy()
    selected.validate()
    plan = schedule or SplitExplicitSchedule()
    dt = float(requested_dt_seconds)
    if not math.isfinite(dt) or dt <= 0.0:
        raise TimestepAdmissionError(
            f"requested dt={requested_dt_seconds!r} s is not finite and positive; declare dt_seconds>0"
        )
    minimum = float(authority.minimum_m)
    speed = float(selected.max_characteristic_speed_m_s)
    sound = selected.max_sound_speed_m_s
    one_d = one_dimensional_advective_limit(
        int(plan.time_integration_order), float(plan.coef_3rd_order)
    )
    advective_limit = HEXAGON_DIRECTION_FACTOR * one_d
    fraction = plan.largest_acoustic_fraction()
    advective = speed * dt / minimum
    acoustic = sound * fraction * dt / minimum
    maximum = min(
        advective_limit * minimum / speed,
        ACOUSTIC_COURANT_LIMIT * minimum / (sound * fraction),
    )
    recommended = largest_schedulable_dt(maximum, cadence_seconds)
    problems = []
    remedies = []
    if advective > advective_limit * (1.0 + 1.0e-12):
        wind_at_limit = advective_limit * minimum / dt
        problems.append(
            f"advective Courant estimate U_max*dt/min(dcEdge) = {speed:g}*"
            f"{dt:.9g}/{minimum:.9g} = {advective:.4f} exceeds "
            f"(3/4)*L1D = 0.75*{one_d:.4f} = {advective_limit:.4f} (L1D: the "
            f"von Neumann limit of the order-{plan.time_integration_order} "
            f"RK stages with flux3 at coef_3rd_order={plan.coef_3rd_order:g}"
            f"; 3/4: a swept-area estimate for a hexagon, not an analysis of "
            f"the 2-D operator).  U_max = {speed:g} m/s is an assumed largest "
            f"wind ({selected.max_characteristic_speed_source}), not this "
            f"case's: at dt={dt:.9g} s the estimate holds for winds up to "
            f"{wind_at_limit:.1f} m/s, and where the wind is stronger the "
            f"transport step amplifies the shortest resolved waves"
        )
        remedies.append(
            f"--max-wind {math.floor(wind_at_limit):d} or lower if this "
            f"case's winds stay below that"
        )
    if acoustic > ACOUSTIC_COURANT_LIMIT * (1.0 + 1.0e-12):
        problems.append(
            f"acoustic Courant c_max*dtau/min(dcEdge) = {sound:.1f}*"
            f"{fraction * dt:.6g}/{minimum:.9g} = {acoustic:.4f} exceeds "
            f"2/sqrt(6) = {ACOUSTIC_COURANT_LIMIT:.4f}, the forward-backward "
            f"limit on the hexagonal C-grid (dtau = dt*{fraction:.6g}, the "
            f"largest acoustic sub-step at order {plan.time_integration_order}, "
            f"{plan.acoustic_substeps} sub-steps, {plan.dynamics_splits} splits)"
            ".  Above this limit a sound wave grows every sub-step"
        )
        needed = math.ceil(
            int(plan.acoustic_substeps) * acoustic / ACOUSTIC_COURANT_LIMIT - 1.0e-9)
        if int(plan.time_integration_order) == 2 or needed <= 3:
            remedies.append(f"--acoustic-substeps {max(needed, int(plan.acoustic_substeps) + 1)} or more")
    if problems:
        if recommended > 0.0:
            step = (
                f"--dt {recommended:g} or smaller (the largest whole-second "
                f"step under both limits that divides the {cadence_seconds:g} s "
                "radiation interval)"
                if cadence_seconds is not None and recommended == round(recommended)
                else f"dt <= {recommended:.1f} s"
            )
            remedies.insert(0, step)
        remedies.append("a mesh with a larger real minimum dcEdge")
        raise TimestepAdmissionError(
            "unstable mesh/timestep combination refused before CUDA allocation: "
            f"min(dcEdge)={minimum:.9g} m from {authority.source}, requested "
            f"dt={dt:.9g} s; " + "; ".join(problems) + ".  Computed maximum "
            f"dt={maximum:.9g} s for this schedule.  Any one of: "
            + "; ".join(remedies) + ".  The runtime will not auto-shrink the "
            "timestep."
        )
    return TimestepAdmission(
        requested_dt_seconds=dt,
        resolved_dt_seconds=dt,
        maximum_admitted_dt_seconds=float(maximum),
        estimated_outer_courant=float(advective),
        admitted_courant=float(advective_limit),
        acoustic_courant=float(acoustic),
        acoustic_courant_limit=float(ACOUSTIC_COURANT_LIMIT),
        largest_acoustic_substep_seconds=float(fraction * dt),
        recommended_dt_seconds=float(recommended),
        auto_shrunk=False,
        authority=authority,
        policy=selected,
        schedule=plan,
    )


__all__ = [
    "ACOUSTIC_COURANT_LIMIT",
    "CourantPolicy",
    "EdgeLengthAuthority",
    "HEXAGON_DIRECTION_FACTOR",
    "SplitExplicitSchedule",
    "TimestepAdmission",
    "TimestepAdmissionError",
    "admit_timestep",
    "edge_length_authority",
    "largest_schedulable_dt",
    "one_dimensional_advective_limit",
]
