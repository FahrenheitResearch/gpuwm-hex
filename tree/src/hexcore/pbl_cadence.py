"""The surface/PBL cadence: a selectable knob, welded to the timestep by default.

``config_bldt_seconds`` is welded to ``config_dt`` by default in
:func:`tools.run_cuda_v841_forecast.build_forecast_config` -- the native x4
v8.4.1 reference ran ``bldt = dt``, i.e. the surface layer, the land-surface
model and the PBL are called on every model step.  That is the proven
configuration's own semantics and it stays the default: ``auto`` is
``bldt = dt`` with nobody passing a flag, and it changes no existing run.

**An explicit cadence is a selectable configuration.**  ``--pbl-cadence
SECONDS`` calls the surface/PBL stack once every ``SECONDS / dt`` steps and
HOLDS the tendency it produced on the steps between -- the engine's own
positive-``bldt`` path, which is ARW's ``stepbl``: the held rate is applied
unchanged on every non-due step, and the surface layer and land-surface
model integrate their own state with the cadence as their step
(``dtturbl = bldt``).  Nothing about the hold is implemented here; the seam
passes the cadence to the engine's sealed constructor and the engine's
``PhysicsDriver`` does what it does for every ARW run with a positive
``bldt``.  Radiation is on its own fixed cadence
(:data:`hexcore.dt_admission.RADIATION_CADENCE_SECONDS`) and was never
welded to ``dt``; the decision records it beside the surface/PBL cadence so
a receipt names every physics call rate the run uses.

WHY IT IS A KNOB AT ALL.  At the 5 s a sub-kilometre mesh declares the weld
calls the stack 720 times an hour against the 30 the proven 120 s
configuration runs, and the per-step profile of 2026-09-13 (RTX 5090, the
43,884-cell point mesh) put the seam's surface/PBL phase at a fifth of every
step.  Holding the cadence is the direct way to buy that back, and it
changes the answer, so per the project's rule for modes that change
results it is SELECTABLE and never silently the default: the welded default
moves only on a measured recommendation, and the measurement is recorded
in ``docs/hex-point-hrrr.md``.

THE REFUSAL.  A cadence that is not a whole number of steps is refused HERE,
on the host, before a mesh is bound or a card is reserved, naming both
numbers and the multiples of ``dt`` on either side of the request.  The
sealed constructor asks the same question after the card is reserved,
which is the wrong place to learn it.  Nothing is rounded silently: a 25 s
request at 10 s steps could mean 20 s or 30 s and the run would call a
cadence the receipt did not name.

THE BREAKAGE THE REGISTRY KEY PREVENTS (gate law, 2026-08-16).  A timestep
anchor certifies a CONFIGURATION, and how often a scheme is CALLED is part
of what its forecasts measured, so :func:`hexcore.dt_admission.dt_key` keys
on the surface/PBL cadence.  Without that key fragment a held-cadence
anchor would occupy the SAME registry slot as the welded row at the same
``(dt, cumulus)`` and silently replace it, after which every ordinary
welded run at that timestep would be admitted against a band measured at
one twenty-fourth of its own surface/PBL call rate.  A held cadence at an
anchored timestep is admitted through a DERIVED row
(:func:`hexcore.dt_admission.derived_held_cadence_anchor`): the dycore
half of the anchor -- the step, its RK schedule, its clock, its
byte-identical dual run -- is the welded row's own and is unchanged by how
often the seam calls the stack; the host half is re-minted for the held
cadence at admission; the physics band is stamped NOT MEASURED at this
cadence rather than borrowed.

THE MEASUREMENT OF RECORD (2026-08-26, x1.40962 at 5 s, RTX 5070 Ti,
``evidence/pbl-cadence-20260826/``): holding the cadence at 120 s while
``dt`` shrank to 5 s on a 120 km mesh 140x below its own Courant limit did
NOT flatten that configuration's |w| runaway -- it reached |w| max 134.55
against 81.77 over the same first 964 steps and then stopped integrating.
That arm answered an attribution question (the surface/PBL call rate is
not the cause of the runaway) on a configuration nobody would run for
weather; it says nothing about a held cadence on a mesh whose natural step
IS 5 s, which is what the 2026-09-13 campaign measures.

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

#: The radiation cadence every configuration runs, independent of ``dt``.
RADIATION_SECONDS: float = dt_admission.RADIATION_CADENCE_SECONDS

#: The receipt schema.  v2 (2026-09-13): the decision records the held
#: steps, the first-hour call count, the radiation cadence beside the
#: surface/PBL one, and names itself a selectable configuration rather than
#: an instrument.
SCHEMA: str = "gpuwm-hex.pbl-cadence-decision/v2"

#: How the engine applies a held cadence, stated once so every receipt
#: carries the same sentence.
HOLD_SEMANTICS: str = (
    "the surface layer, the land-surface model and the PBL are called on "
    "the first step and then once every steps_between_calls steps; on the "
    "steps between, the tendency the last call produced is applied "
    "unchanged (the engine's positive-bldt path, ARW's stepbl), and the "
    "surface layer and land-surface model integrate their own state with "
    "the cadence as their step (dtturbl = bldt).  Radiation keeps its own "
    "cadence and is not affected"
)

#: The measured breakage kept beside the registry-key gate, because a gate
#: that cannot name what it prevents does not exist (gate law, 2026-08-16).
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
                f"cadence to dt, which is the proven configuration and the "
                f"default; a number of seconds holds the stack's tendency "
                f"between calls at that cadence, which is a selectable "
                f"configuration and records itself as one"
            ) from error
    return _positive_seconds("pbl_cadence", requested)


def resolve_seconds(*, dt_seconds: float, requested: Any = WELDED) -> float:
    """The surface/PBL cadence in seconds this request selects at this dt."""

    dt = _positive_seconds("dt_seconds", dt_seconds)
    parsed = parse_request(requested)
    return dt if parsed == WELDED else float(parsed)


def calls_per_hour(seconds: float) -> float:
    """How often a stack on this cadence is called in one forecast hour."""

    return 3600.0 / _positive_seconds("cadence_seconds", seconds)


def calls_in_steps(*, steps_between_calls: int, executed_steps: int) -> int:
    """The engine's own surface/PBL call count over ``executed_steps`` steps.

    The engine's ``_surface_pbl_step_due`` fires on step 1 and on every step
    whose one-based index is a multiple of ``stepbl``, so a held cadence
    makes one more call than the steady rate says (step 1 and step
    ``stepbl`` are both due).  Welded, every step is due and the two agree.
    This is the number a run's receipt is checked against: a run that made
    any other number of calls did not hold the cadence it declared.
    """

    steps = int(steps_between_calls)
    if steps < 1:
        raise PblCadenceError("steps_between_calls must be a positive integer")
    executed = int(executed_steps)
    if executed < 0:
        raise PblCadenceError("executed_steps must be a non-negative integer")
    if executed == 0:
        return 0
    if steps == 1:
        return executed
    return 1 + executed // steps


def calls_in_first_hour(*, dt_seconds: float, steps_between_calls: int) -> int:
    """:func:`calls_in_steps` over the first forecast hour at this ``dt``."""

    dt = _positive_seconds("dt_seconds", dt_seconds)
    return calls_in_steps(
        steps_between_calls=steps_between_calls,
        executed_steps=int(round(3600.0 / dt)),
    )


def label(*, dt_seconds: float, surface_pbl_seconds: float) -> str:
    """The human name a receipt, roster or refusal uses for one cadence."""

    dt = _positive_seconds("dt_seconds", dt_seconds)
    seconds = _positive_seconds("surface_pbl_seconds", surface_pbl_seconds)
    if seconds == dt:
        return "surface/PBL every step"
    return f"surface/PBL held at {seconds:g} s"


def _nearest_multiples(seconds: float, dt: float) -> tuple[float, float]:
    """The multiples of ``dt`` on either side of an incommensurate request."""

    below = math.floor(seconds / dt) * dt
    above = math.ceil(seconds / dt) * dt
    if below <= 0.0:
        below = dt
    return float(below), float(above)


def pbl_cadence_decision(
    *,
    dt_seconds: float,
    requested: Any = WELDED,
    radiation_seconds: float = RADIATION_SECONDS,
) -> dict[str, Any]:
    """Decide the run's surface/PBL cadence, and record why.

    Returns a JSON-ready mapping that rides into the run's own receipt: the
    timestep it was taken at, the cadence chosen, the step count between
    calls and the steps held between them, the call rates (steady, in the
    first hour, welded, and at the proven timestep), the radiation cadence
    beside it, the hold semantics, and whether the decision came from the
    proven weld (``source: "welded"``) or from an explicit selection
    (``source: "explicit"``).
    """

    dt = _positive_seconds("dt_seconds", dt_seconds)
    radiation = _positive_seconds("radiation_seconds", radiation_seconds)
    parsed = parse_request(requested)
    welded = parsed == WELDED
    seconds = dt if welded else float(parsed)

    # Refuse an incommensurate cadence HERE, on the host, naming both
    # numbers and the way out -- the sealed constructor asks the same
    # question after the card is reserved, which is the wrong place to
    # learn it, and rounding silently would run a cadence the receipt did
    # not name.
    try:
        steps = dt_admission.cadence_steps("surface_pbl_seconds", seconds, dt)
    except dt_admission.DtAdmissionError as error:
        below, above = _nearest_multiples(seconds, dt)
        raise PblCadenceError(
            f"--pbl-cadence {seconds:g} is not a whole number of {dt:g} s "
            f"steps ({seconds / dt:.6f} steps), so there is no step on which "
            f"the surface/PBL stack would be called and nothing is rounded "
            f"for you.  The multiples of dt on either side are {below:g} s "
            f"and {above:g} s; 'auto' is the welded default.  {error}"
        ) from error
    try:
        radiation_steps = dt_admission.cadence_steps(
            "radiation_seconds", radiation, dt
        )
    except dt_admission.DtAdmissionError as error:
        raise PblCadenceError(str(error)) from error

    rate = calls_per_hour(seconds)
    proven_rate = calls_per_hour(PROVEN_SURFACE_PBL_SECONDS)
    welded_rate = calls_per_hour(dt)
    first_hour = calls_in_first_hour(dt_seconds=dt, steps_between_calls=steps)

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
            f"selectable configuration HOLDING the surface/PBL cadence at "
            f"{seconds:g} s while config_dt is {dt:g} s: the stack is called "
            f"on the first step and then once every {steps} steps, and its "
            f"tendency is held on the {steps - 1} steps between -- "
            f"{rate:g} calls an hour ({first_hour} in the first hour) against "
            f"the {welded_rate:g} the weld would make and the proven "
            f"{proven_rate:g}.  This changes the forecast and records itself "
            f"as a selection; the default stays the weld until a measured "
            f"recommendation moves it"
        )

    return {
        "schema": SCHEMA,
        "dt_seconds": dt,
        "surface_pbl_seconds": seconds,
        "steps_between_calls": steps,
        "steps_held_between_calls": steps - 1,
        "calls_per_hour": rate,
        "calls_in_first_hour": first_hour,
        "calls_per_hour_welded": welded_rate,
        "calls_per_hour_at_proven_dt": proven_rate,
        "radiation_seconds": radiation,
        "radiation_steps_between_calls": radiation_steps,
        "radiation_calls_per_hour": calls_per_hour(radiation),
        "held": not welded,
        "selectable": True,
        "default": WELDED,
        "source": source,
        "requested": WELDED if welded else f"{seconds:g}",
        "label": label(dt_seconds=dt, surface_pbl_seconds=seconds),
        "hold_semantics": HOLD_SEMANTICS,
        "breakage": BREAKAGE,
        "note": note,
    }


__all__ = [
    "BREAKAGE",
    "HOLD_SEMANTICS",
    "PROVEN_SURFACE_PBL_SECONDS",
    "RADIATION_SECONDS",
    "SCHEMA",
    "PblCadenceError",
    "WELDED",
    "calls_in_first_hour",
    "calls_in_steps",
    "calls_per_hour",
    "label",
    "parse_request",
    "pbl_cadence_decision",
    "resolve_seconds",
]
