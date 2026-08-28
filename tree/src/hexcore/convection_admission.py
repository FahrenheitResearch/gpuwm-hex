"""Convection admission: the cumulus scheme is OFF below 3 km, by default.

It was ruled on 2026-08-26 that convection is switched off below 3 km,
answering directly whether a fine swath runs the cumulus scheme at all.
This module is that ruling made executable, and it is deliberately the only
place the threshold exists.

THE BREAKAGE THIS GATE PREVENTS, MEASURED (2026-08-26,
``evidence/dt-anchors-20260826/RECEIPT.md``).  WRF pins ``cudt = 0`` for
``cu_physics = 3``, so Grell-Freitas recomputes on every model step and no
configuration relaxes it -- the cumulus cadence IS the timestep.  A finer
mesh declares a smaller timestep, so it calls the closure proportionally
more often: 30 times an hour at the proven 120 s, 180 at 20 s (a 3 km
swath), 720 at 5 s (a 750 m swath).  At 5 s on ``x1.40962`` the measured
|w| mean over four half-hour windows ran 12.0 / 49.6 / 73.5 / 87.5 m/s
against a same-card 120 s control's 1.15 / 1.17 / 1.20 / 1.48, still
climbing at 2 h, with |w| max 102.67 m/s against 1.680.  Every step was
finite and two arms were byte-identical, so it is a different SOLUTION and
not an unstable one -- which is worse, because nothing crashes.

Every mesh the ruling covers sits between six and twenty-four times the
proven call rate.  So the ruling and the measurement point the same way, and
this gate is where they meet.

**Fixed means default.**  A bare run on a sub-3-km mesh selects no cumulus
scheme, with nobody passing a flag.  The explicit switch is an A/B
instrument -- it is how the attribution above is measured -- and a decision
taken through it says ``source: "explicit"`` in its own receipt so it can
never be read as the remedy.

**The finest spacing anywhere on the mesh decides.**  The question the
ruling answered was about a fine SWATH inside a coarser mesh, so a mesh that
is 3 km anywhere is a mesh the ruling covers.  The decision takes the smaller of the row's
declared nominal spacing and the mesh's own measured ``min(dcEdge)``, so a
row that declares one resolution and carries another cannot smuggle the
closure back on.

**Convection-off is a NEW admitted configuration, not a mutation of the
proven one.**  How often a scheme is called is part of what a timestep
anchor certifies (:mod:`hexcore.dt_admission`), so a convection-off run
earns its own anchors and cannot borrow the Grell-Freitas rows'.  The
frozen v8.4.1 column-physics lane's proven configuration stays exactly
reachable and byte-unchanged.

**The arbitrary acceptance test.**  Selecting the scheme is table work in
one place: a threshold, a named set, and one translation between the config
vocabulary and the sealed constructor's.  Adding a scheme is a row here, not
a new configuration subclass per combination.
"""

from __future__ import annotations

import math
from typing import Any


class ConvectionAdmissionError(RuntimeError):
    """A convection selection is refused, by name."""


#: The config-level name of the proven cumulus scheme.
SCHEME_GRELL_FREITAS: str = "cu_grell_freitas"

#: The config-level name for "no cumulus scheme is called at all".
SCHEME_OFF: str = "off"

#: Every cumulus selection the frozen column-physics lane admits.  A value
#: outside this set is refused by ``V841MpasColumnPhysicsConfig.validate``.
ADMITTED_CONVECTION_SCHEMES: tuple[str, ...] = (SCHEME_GRELL_FREITAS, SCHEME_OFF)

#: The ruling of 2026-08-26: convection is switched off BELOW this spacing.
#: At or above it the proven Grell-Freitas configuration is unchanged.
CONVECTION_OFF_BELOW_M: float = 3_000.0

#: The ruling, quoted where the threshold lives so the number never travels
#: without the authority that set it.
RULING: str = (
    "2026-08-26: convection is switched off below 3 km, answering directly "
    "whether a fine swath runs the convection scheme at all.  Changing this "
    "threshold is a decision of record and not a tuning knob: the number is "
    "the boundary a measured breakage sits on, so moving it needs the "
    "measurement redone, not an edit"
)

#: The measured breakage, kept beside the threshold because a gate that
#: cannot name what it prevents does not exist (gate law, 2026-08-16).
BREAKAGE: str = (
    "WRF pins cudt=0 for cu_physics=3, so Grell-Freitas is called every "
    "model step and a finer mesh calls it proportionally more often -- 30 "
    "times an hour at the proven 120 s, 180 at the 20 s a 3 km swath "
    "declares, 720 at the 5 s a 750 m swath declares.  MEASURED "
    "(2026-08-26, evidence/dt-anchors-20260826/RECEIPT.md, x1.40962): at "
    "5 s the |w| mean over four half-hour windows ran 12.0/49.6/73.5/87.5 "
    "m/s against a same-card 120 s control's 1.15/1.17/1.20/1.48, still "
    "climbing at 2 h, with |w| max 102.67 m/s against 1.680.  Finite at "
    "every one of 1,440 steps and byte-identical across two arms, so it is "
    "a different solution rather than an unstable one and nothing crashes "
    "to announce it"
)

#: What ``--convection`` accepts.  ``auto`` is the ruling; the other two are
#: A/B arms.
REQUESTS: tuple[str, ...] = ("auto", "off", "gf")


def _positive_length(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConvectionAdmissionError(
            f"{name}={value!r} m must be finite and positive: the convection "
            f"ruling is taken on a real grid spacing, and a spacing that is "
            f"not one cannot be compared to the {CONVECTION_OFF_BELOW_M:g} m "
            f"threshold"
        )
    return result


def convection_for_spacing(finest_spacing_m: float) -> str:
    """The cumulus scheme the ruling selects at this grid spacing."""

    spacing = _positive_length("finest_spacing_m", finest_spacing_m)
    return SCHEME_OFF if spacing < CONVECTION_OFF_BELOW_M else SCHEME_GRELL_FREITAS


def default_convection_scheme(
    *, nominal_dx_m: float, minimum_dc_edge_m: float | None = None
) -> str:
    """The scheme a BARE run gets -- no flag, no argument, no opt-in."""

    return convection_decision(
        nominal_dx_m=nominal_dx_m, minimum_dc_edge_m=minimum_dc_edge_m
    )["scheme"]


def finest_spacing(
    *, nominal_dx_m: float, minimum_dc_edge_m: float | None = None
) -> float:
    """The smallest cell spacing the mesh carries, declared or measured."""

    nominal = _positive_length("nominal_dx_m", nominal_dx_m)
    if minimum_dc_edge_m is None:
        return nominal
    measured = _positive_length("minimum_dc_edge_m", minimum_dc_edge_m)
    return min(nominal, measured)


def constructor_scheme(config_scheme: str) -> str | None:
    """Translate the config name into the sealed constructor's name.

    ``SealedArwenConstructorV841`` takes ``'gf'``, ``'kf'`` or ``None``.  One
    translation, in one place, so the two vocabularies cannot drift.
    """

    if config_scheme == SCHEME_GRELL_FREITAS:
        return "gf"
    if config_scheme == SCHEME_OFF:
        return None
    raise ConvectionAdmissionError(
        f"config_convection_scheme={config_scheme!r} is not a convection "
        f"selection the frozen column-physics lane admits "
        f"({', '.join(ADMITTED_CONVECTION_SCHEMES)}).  A third scheme is a "
        f"different, unproven physics lane and earns its own evidence"
    )


def gf_ishallow(config_scheme: str) -> int:
    """GF's shallow branch: on with GF, and meaningless without it.

    Native MPAS v8.4.1 hardwires ``ishallow = 1``
    (``mpas_atmphys_vars.F:340``), and the sealed constructor refuses
    ``gf_ishallow=1`` when no GF is selected.  Derived here rather than
    written as a literal beside every constructor mapping, so switching the
    scheme off cannot leave a shallow flag behind to refuse the run.
    """

    return 1 if constructor_scheme(config_scheme) == "gf" else 0


def label(config_scheme: str) -> str:
    """The human name a receipt, roster or refusal uses for one selection."""

    if config_scheme == SCHEME_OFF:
        return "convection off"
    if config_scheme == SCHEME_GRELL_FREITAS:
        return "GF"
    raise ConvectionAdmissionError(
        f"config_convection_scheme={config_scheme!r} has no admitted label"
    )


def label_for_constructor_scheme(scheme: str | None) -> str:
    """The same label, from the sealed constructor's vocabulary."""

    return "convection off" if scheme is None else str(scheme).upper()


def convection_decision(
    *,
    nominal_dx_m: float,
    minimum_dc_edge_m: float | None = None,
    requested: str = "auto",
) -> dict[str, Any]:
    """Decide the run's cumulus selection, and record why.

    Returns a JSON-ready mapping that rides into the run's own receipt: the
    spacing the decision was taken on, the threshold, the ruling, the
    measured breakage, and whether the decision came from the ruling
    (``source: "resolution"``) or from an explicit A/B arm
    (``source: "explicit"``).
    """

    if requested not in REQUESTS:
        raise ConvectionAdmissionError(
            f"convection request {requested!r} is not one of "
            f"{list(REQUESTS)}: 'auto' applies the ruling (off below "
            f"{CONVECTION_OFF_BELOW_M:g} m), and 'off'/'gf' are explicit A/B "
            f"arms that record themselves as such"
        )

    spacing = finest_spacing(
        nominal_dx_m=nominal_dx_m, minimum_dc_edge_m=minimum_dc_edge_m
    )
    ruled = convection_for_spacing(spacing)
    below = spacing < CONVECTION_OFF_BELOW_M

    if requested == "auto":
        scheme = ruled
        source = "resolution"
        note = (
            f"the ruling decided this: the finest spacing on the mesh is "
            f"{spacing:g} m, which is "
            f"{'below' if below else 'at or above'} the "
            f"{CONVECTION_OFF_BELOW_M:g} m threshold, so the run selects "
            f"{label(scheme)} with no flag passed"
        )
    else:
        scheme = SCHEME_OFF if requested == "off" else SCHEME_GRELL_FREITAS
        source = "explicit"
        if scheme == ruled:
            note = (
                f"explicit A/B arm; it happens to agree with the ruling at "
                f"{spacing:g} m, which would have selected {label(ruled)} "
                f"anyway"
            )
        else:
            note = (
                f"explicit A/B arm OVERRIDING the ruling: at {spacing:g} m "
                f"the ruling selects {label(ruled)} and this run selects "
                f"{label(scheme)}.  This is an experiment arm, never a "
                f"remedy -- the remedy is the default"
            )

    return {
        "schema": "gpuwm-hex.convection-decision/v1",
        "scheme": scheme,
        "constructor_scheme": constructor_scheme(scheme),
        "gf_ishallow": gf_ishallow(scheme),
        "source": source,
        "requested": requested,
        "ruled_scheme": ruled,
        "finest_spacing_m": spacing,
        "nominal_dx_m": float(nominal_dx_m),
        "minimum_dc_edge_m": (
            None if minimum_dc_edge_m is None else float(minimum_dc_edge_m)
        ),
        "threshold_m": CONVECTION_OFF_BELOW_M,
        "below_threshold": below,
        "ruling": RULING,
        "breakage": BREAKAGE,
        "note": note,
    }


__all__ = [
    "ADMITTED_CONVECTION_SCHEMES",
    "BREAKAGE",
    "CONVECTION_OFF_BELOW_M",
    "ConvectionAdmissionError",
    "REQUESTS",
    "RULING",
    "SCHEME_GRELL_FREITAS",
    "SCHEME_OFF",
    "constructor_scheme",
    "convection_decision",
    "convection_for_spacing",
    "default_convection_scheme",
    "finest_spacing",
    "gf_ishallow",
    "label",
    "label_for_constructor_scheme",
]
