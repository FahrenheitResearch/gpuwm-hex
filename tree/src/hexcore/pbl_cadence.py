"""The surface/PBL cadence: welded to the timestep by default, holdable for an A/B.

``config_bldt_seconds`` is welded to ``config_dt`` in
:func:`tools.run_cuda_v841_forecast.build_forecast_config` -- the native x4
v8.4.1 reference ran ``bldt = dt``, i.e. the surface layer, the land-surface
model and the PBL are called on every model step.  That is the proven
configuration's own semantics and this module does not change it.  What this
module adds is the ability to HOLD that cadence while ``dt`` shrinks, which
is an instrument and never a remedy.

WHY THE INSTRUMENT EXISTS, MEASURED.  Two campaigns landed on 2026-08-26 and
between them they define one open question.

``evidence/dt-anchors-20260826/RECEIPT.md`` measured, on the 120 km global
mesh ``x1.40962`` from one native init, that shrinking the timestep produces
vertical velocities that cannot be real: |w| max 1.680 m/s at the proven
120 s, 7.511 at 20 s, and **102.670 at 5 s**, with the |w| mean climbing
monotonically through 12.0 / 49.6 / 73.5 / 87.5 m/s over four half-hour
windows and still climbing at the end of the arm.  Every step was finite and
the two arms were byte-identical, so it is a different SOLUTION rather than
an unstable one -- which is worse, because nothing crashes to announce it.
That campaign reasoned the growth looked like *a forcing that scales with
call count rather than with elapsed time*, named Grell-Freitas as the
candidate because WRF's ``cudt = 0`` pin is written down, and declined the
attribution as NOT MEASURED.

``evidence/convection-off-20260826/RECEIPT.md`` then measured it directly and
**eliminated the closure**: with convection switched off entirely, |w| max
at 5 s is still 93.957 m/s and the mean still climbs 8.19 / 49.05 / 65.83 /
81.07.  **91.4 % of the 5 s excess over the control survives the scheme
never being called at all**, and ``theta_m`` max is 887.3353 K with the
closure and 887.3353 K without it, to every printed digit.

**The call-rate SHAPE of the hypothesis survived; only its subject was
wrong.**  ``config_bldt_seconds`` is welded to ``dt`` in exactly the same way
``cudt`` is, and nobody had named it: at 5 s the surface/PBL stack runs
**720 times an hour against the proven 30**, the identical 24x.  Radiation
sits on a fixed 600 s cadence and does NOT scale with ``dt``, which is why
it is not a candidate and why it makes a useful control.  This module is how
that hypothesis gets tested -- hold the surface/PBL cadence at 120 s while
``dt`` shrinks, and read the band against the arms that welded it.

**Welded is the default, and stays the default.**  ``auto`` is ``bldt = dt``,
the proven semantics, with nobody passing a flag.  It changes no existing
run.  An explicit cadence is an A/B arm: it stamps ``source: "explicit"``
into its own receipt and its note says it is holding the cadence, so it can
never be read as a remedy.  Nothing here is a correctness fix, so "fixed
means default" has nothing to fire on -- there is no fix, there is a
measurement.

THE BREAKAGE THE REGISTRY GATE PREVENTS (gate law, 2026-08-16).  A
timestep anchor certifies a CONFIGURATION, and how often a scheme is CALLED
is part of what its forecasts measured -- the same argument
:mod:`hexcore.convection_admission` made for the cumulus selection on
2026-08-26, now applied to the cadence that argument's own receipt named as
the surviving hypothesis.  So :func:`hexcore.dt_admission.dt_key` keys on
the surface/PBL cadence too.  Without that key fragment a held-cadence
anchor would occupy the SAME registry slot as the welded row at the same
``(dt, cumulus)`` and silently replace it, after which every ordinary welded
run at that timestep would be admitted against a band measured at one
twenty-fourth of its own surface/PBL call rate.  Concretely: the registered
5 s convection-off row certifies |w| mean 8.19/49.05/65.83/81.07 m/s with
the surface/PBL stack called 720 times an hour, and the arm this module
makes possible calls it 24 times an hour at the same timestep.  Those are
different configurations and each earns its own evidence.

WHAT THIS MODULE CANNOT SETTLE, AND MUST NOT BE READ AS SETTLING.  Every arm
it enables runs on ``x1.40962``, a 120 km global mesh whose own Courant
limit measures 698.95 s.  5 s is 140x below that limit and 20 s is 35x below
it; nobody would run a 120 km mesh at either for weather.  No arm on this
mesh can separate "a genuine per-call defect" from "the resolved dynamics of
a configuration nobody would choose".  What settles that is the same trend
on a mesh whose Courant limit is near the timestep, refereed by obs-skill
(MRMS, ASOS) -- the standard [[ArWen goes its own way]] set on 2026-08-03.
No such mesh exists yet.

THE ARBITRARY ACCEPTANCE TEST.  Holding a cadence is table work in one
place: a request vocabulary, one resolution against ``dt``, and one decision
record.  A second cadence becoming holdable is a row here, not a new
configuration subclass per combination.
"""

from __future__ import annotations

import math
from typing import Any

from . import dt_admission


class PblCadenceError(RuntimeError):
    """A surface/PBL cadence request is refused, by name."""


#: The request meaning "the proven semantics": the surface/PBL stack is
#: called every model step, i.e. ``config_bldt_seconds == config_dt``.
WELDED: str = "auto"

#: The surface/PBL cadence of the proven configuration, in seconds.  It is
#: the proven timestep because the two are welded there.
PROVEN_SURFACE_PBL_SECONDS: float = dt_admission.PROVEN_DT_SECONDS

#: The measured breakage kept beside the gate, because a gate that cannot
#: name what it prevents does not exist (gate law, 2026-08-16).
BREAKAGE: str = (
    "config_bldt_seconds is welded to config_dt exactly as cudt is, so a "
    "smaller timestep calls the surface layer, the land-surface model and "
    "the PBL proportionally more often -- 30 times an hour at the proven "
    "120 s, 180 at 20 s, 720 at 5 s.  MEASURED (2026-08-26, x1.40962, "
    "evidence/convection-off-20260826/RECEIPT.md): at 5 s with the cumulus "
    "closure switched off entirely the |w| mean over four half-hour windows "
    "ran 8.19/49.05/65.83/81.07 m/s against a same-card 120 s control's "
    "1.15/1.17/1.20/1.48, |w| max 93.957 against 1.680, still climbing at "
    "2 h -- 91.4 % of the excess surviving the closure never being called.  "
    "Grell-Freitas is eliminated and the call-rate shape of the hypothesis "
    "is not: the surface/PBL stack runs the identical 24x more often and "
    "had never been named.  A held-cadence run sharing the welded run's "
    "registry slot would quote a band measured at 24x its own call rate"
)


def _positive_seconds(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise PblCadenceError(
            f"{name}={value!r} s must be finite and positive: a physics "
            f"cadence is a real number of seconds between calls"
        )
    return result


def parse_request(requested: Any) -> str | float:
    """Normalise a ``--pbl-cadence`` request into ``'auto'`` or seconds."""

    if requested is None:
        return WELDED
    if isinstance(requested, str):
        text = requested.strip()
        if text.lower() == WELDED:
            return WELDED
        try:
            return _positive_seconds("pbl_cadence", text)
        except ValueError as error:
            raise PblCadenceError(
                f"pbl_cadence={requested!r} is neither {WELDED!r} nor a "
                f"number of seconds.  {WELDED!r} welds the surface/PBL "
                f"cadence to dt, which is the proven configuration; an "
                f"explicit number holds it there while dt moves, which is an "
                f"A/B arm and records itself as one"
            ) from error
    return _positive_seconds("pbl_cadence", requested)


def resolve_seconds(*, dt_seconds: float, requested: Any = WELDED) -> float:
    """The surface/PBL cadence in seconds this request selects at this dt."""

    dt = _positive_seconds("dt_seconds", dt_seconds)
    parsed = parse_request(requested)
    return dt if parsed == WELDED else float(parsed)


def calls_per_hour(seconds: float) -> float:
    """How often the surface/PBL stack is called in one forecast hour."""

    return 3600.0 / _positive_seconds("surface_pbl_seconds", seconds)


def label(*, dt_seconds: float, surface_pbl_seconds: float) -> str:
    """The human name a receipt, roster or refusal uses for one cadence."""

    dt = _positive_seconds("dt_seconds", dt_seconds)
    seconds = _positive_seconds("surface_pbl_seconds", surface_pbl_seconds)
    if seconds == dt:
        return "surface/PBL every step"
    return f"surface/PBL held at {seconds:g} s"


def pbl_cadence_decision(
    *, dt_seconds: float, requested: Any = WELDED
) -> dict[str, Any]:
    """Decide the run's surface/PBL cadence, and record why.

    Returns a JSON-ready mapping that rides into the run's own receipt: the
    timestep it was taken at, the cadence chosen, the resulting step count
    and call rate, the call rate the proven configuration runs, and whether
    the decision came from the proven weld (``source: "welded"``) or from an
    explicit A/B arm (``source: "explicit"``).
    """

    dt = _positive_seconds("dt_seconds", dt_seconds)
    parsed = parse_request(requested)
    welded = parsed == WELDED
    seconds = dt if welded else float(parsed)

    # Refuse an incommensurate cadence HERE, on the host, naming both
    # numbers -- the sealed constructor asks the same question after the
    # card is reserved, which is the wrong place to learn it.
    try:
        steps = dt_admission.cadence_steps("surface_pbl_seconds", seconds, dt)
    except dt_admission.DtAdmissionError as error:
        raise PblCadenceError(
            f"a surface/PBL cadence of {seconds:g} s is not a whole number "
            f"of {dt:g} s steps, so there is no step on which it would be "
            f"called: {error}"
        ) from error

    rate = calls_per_hour(seconds)
    proven_rate = calls_per_hour(PROVEN_SURFACE_PBL_SECONDS)

    if welded:
        source = "welded"
        note = (
            f"the proven configuration's own semantics: config_bldt_seconds "
            f"= config_dt = {dt:g} s, so the surface layer, the land-surface "
            f"model and the PBL are called on every model step "
            f"({rate:g} times an hour against the proven {proven_rate:g}).  "
            f"The native x4 v8.4.1 reference ran this, and no flag was passed"
        )
    else:
        source = "explicit"
        note = (
            f"explicit A/B arm HOLDING the surface/PBL cadence at "
            f"{seconds:g} s while config_dt is {dt:g} s, so the stack is "
            f"called once every {steps} steps -- {rate:g} times an hour "
            f"against the {calls_per_hour(dt):g} the weld would give and the "
            f"proven {proven_rate:g}.  This is an instrument for measuring "
            f"whether a forcing scales with call count rather than with "
            f"elapsed time, never a remedy -- the proven configuration is "
            f"the weld and it is the default"
        )

    return {
        "schema": "gpuwm-hex.pbl-cadence-decision/v1",
        "dt_seconds": dt,
        "surface_pbl_seconds": seconds,
        "steps_between_calls": steps,
        "calls_per_hour": rate,
        "calls_per_hour_welded": calls_per_hour(dt),
        "calls_per_hour_at_proven_dt": proven_rate,
        "held": not welded,
        "source": source,
        "requested": WELDED if welded else f"{seconds:g}",
        "label": label(dt_seconds=dt, surface_pbl_seconds=seconds),
        "breakage": BREAKAGE,
        "note": note,
    }


__all__ = [
    "BREAKAGE",
    "PROVEN_SURFACE_PBL_SECONDS",
    "PblCadenceError",
    "WELDED",
    "calls_per_hour",
    "label",
    "parse_request",
    "pbl_cadence_decision",
    "resolve_seconds",
]
