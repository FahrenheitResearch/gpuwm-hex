"""The surface/PBL cadence: welded by default, holdable as an A/B instrument.

Two campaigns landed on 2026-08-26 and between them they define one open
question.  ``evidence/dt-anchors-20260826/RECEIPT.md`` measured |w| max
running 1.680 m/s at the proven 120 s, 7.511 at 20 s and 102.670 at 5 s on
the 120 km global mesh ``x1.40962``, every step finite and the arms
byte-identical -- a different SOLUTION rather than an unstable one.  It
named Grell-Freitas' call rate as the candidate and declined the
attribution.  ``evidence/convection-off-20260826/RECEIPT.md`` then
eliminated the closure by measurement: with convection off entirely, 5 s
still reaches |w| max 93.957 and ``theta_m`` max is 887.3353 K with the
scheme and 887.3353 K without it, to every printed digit.

**The call-rate shape of the hypothesis survived; only its subject was
wrong.**  ``config_bldt_seconds`` is welded to ``config_dt`` exactly as
``cudt`` is, and nobody had named it: at 5 s the surface/PBL stack runs 720
times an hour against the proven 30, the identical 24x.  This module is the
instrument that tests it -- hold the cadence at 120 s while ``dt`` shrinks.

**The weld is the default and stays the default.**  There is no fix here, so
"fixed means default" has nothing to fire on.  ``auto`` is the proven
configuration and changes no existing run; an explicit cadence records
itself as an A/B arm and never as a remedy.

**A held cadence is a NEW configuration, not a mutation of the proven one.**
How often a scheme is CALLED is part of what an anchor's forecasts measured,
so the registry keys on the cadence and a held-cadence run earns its own row
rather than borrowing the welded one's evidence.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import pytest

from hexcore import dt_admission, mesh_row_candidate, pbl_cadence
from hexcore.pbl_cadence import PblCadenceError


# ---------------------------------------------------------------------------
# the default is the proven weld, and it moves nothing
# ---------------------------------------------------------------------------
def test_a_bare_decision_is_the_weld_and_needs_no_flag():
    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=120.0)
    assert decision["source"] == "welded"
    assert decision["held"] is False
    assert decision["surface_pbl_seconds"] == 120.0
    assert decision["steps_between_calls"] == 1
    assert decision["calls_per_hour"] == 30.0


def test_the_weld_follows_dt_down_which_is_the_whole_problem():
    """The proven semantics call the stack 24x more often at 5 s."""

    for dt, rate in ((120.0, 30.0), (20.0, 180.0), (5.0, 720.0)):
        decision = pbl_cadence.pbl_cadence_decision(dt_seconds=dt)
        assert decision["surface_pbl_seconds"] == dt
        assert decision["calls_per_hour"] == rate
        assert decision["steps_between_calls"] == 1
    assert 720.0 / 30.0 == 24.0


def test_holding_the_cadence_restores_the_proven_call_rate():
    """The instrument's whole point: the same 30 calls an hour at any dt."""

    for dt, steps in ((20.0, 6), (5.0, 24)):
        decision = pbl_cadence.pbl_cadence_decision(dt_seconds=dt, requested="120")
        assert decision["source"] == "explicit"
        assert decision["held"] is True
        assert decision["surface_pbl_seconds"] == 120.0
        assert decision["steps_between_calls"] == steps
        assert decision["calls_per_hour"] == 30.0
        assert decision["calls_per_hour_welded"] == 3600.0 / dt


def test_an_explicit_arm_says_so_and_is_never_the_remedy():
    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=5.0, requested="120")
    assert decision["source"] == "explicit"
    assert "never a remedy" in decision["note"]
    assert "A/B arm" in decision["note"]


def test_the_gate_carries_the_measurement_that_produced_it():
    """Gate law: a gate that cannot name what it prevents does not exist."""

    breakage = pbl_cadence.BREAKAGE
    assert "93.957" in breakage
    assert "91.4" in breakage
    assert "720" in breakage and "30" in breakage


# ---------------------------------------------------------------------------
# an incommensurate cadence is refused on the host, naming both numbers
# ---------------------------------------------------------------------------
def test_a_cadence_that_is_not_a_whole_number_of_steps_is_refused():
    """THE BREAKAGE: the sealed constructor asks this AFTER the card is
    reserved, which is the wrong place to learn it."""

    with pytest.raises(PblCadenceError) as caught:
        pbl_cadence.pbl_cadence_decision(dt_seconds=7.0, requested="120")
    message = str(caught.value)
    assert "120" in message and "7" in message


@pytest.mark.parametrize("bad", ["0", "-5", "nonsense", "", "1e400"])
def test_a_request_that_is_neither_auto_nor_seconds_is_refused(bad):
    with pytest.raises(PblCadenceError):
        pbl_cadence.pbl_cadence_decision(dt_seconds=20.0, requested=bad)


# ---------------------------------------------------------------------------
# the registry keys on the configuration, cadence included
# ---------------------------------------------------------------------------
def test_every_registered_anchor_is_a_welded_row():
    """The change moved no existing row: all seven were earned welded."""

    for key, anchor in dt_admission.ADMITTED_TIMESTEPS.items():
        assert anchor.surface_pbl_seconds == anchor.dt_seconds
        assert key.endswith("|surface_pbl=dt")
        assert key == dt_admission.dt_key(
            anchor.dt_seconds, anchor.cumulus_scheme, anchor.surface_pbl_seconds
        )


def test_a_held_cadence_is_a_different_key_from_the_welded_one():
    """THE BREAKAGE THIS PREVENTS: sharing a slot would let a welded run be
    admitted against a band measured at 24x its own call rate."""

    welded = dt_admission.dt_key(5.0, None)
    held = dt_admission.dt_key(5.0, None, 120.0)
    assert welded != held
    assert dt_admission.surface_pbl_key(5.0, 120.0) == "120.0"


def test_spelling_the_weld_explicitly_is_the_same_configuration():
    """bldt == dt IS the weld, however it was written."""

    assert dt_admission.dt_key(5.0, None, 5.0) == dt_admission.dt_key(5.0, None)
    assert dt_admission.surface_pbl_key(5.0, 5.0) == "dt"
    assert dt_admission.surface_pbl_key(5.0, None) == "dt"


def test_a_held_cadence_at_an_anchored_timestep_holds_no_anchor():
    assert dt_admission.admitted_timestep(5.0, None) is not None
    assert dt_admission.admitted_timestep(5.0, None, 120.0) is None


def test_the_roster_names_a_held_row_rather_than_hiding_it():
    """A roster reading "5 s (convection off)" twice would be a trap."""

    anchor = dt_admission.admitted_timestep(5.0, None)
    held = dataclasses_replace(anchor, surface_pbl_seconds=120.0)
    assert "surface/PBL held at 120 s" in dt_admission.anchor_label(held)
    assert "surface/PBL held" not in dt_admission.anchor_label(anchor)


def dataclasses_replace(anchor, **changes):
    import dataclasses

    return dataclasses.replace(anchor, **changes)


# ---------------------------------------------------------------------------
# the refusals, and the near-miss diagnostic the widening had to keep
# ---------------------------------------------------------------------------
def test_the_refusal_keeps_the_near_miss_sentence():
    """Widening the key must not turn "wrong cadence" into "unknown dt".

    A caller who typo'd a cadence wants the row it nearly matched, not a
    refusal reading as though the timestep itself were unrecognised.
    """

    with pytest.raises(dt_admission.DtAdmissionError) as caught:
        dt_admission.require_dt_anchor(
            120.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=600.0,
            cumulus_seconds=120.0,
        )
    message = str(caught.value)
    assert "at physics cadences this run does not use" in message
    assert "--pbl-cadence 600" in message


def test_the_refusal_names_the_remedy_because_the_cadence_is_holdable():
    with pytest.raises(dt_admission.DtAdmissionError) as caught:
        dt_admission.require_dt_anchor(
            5.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=120.0,
            cumulus_seconds=None,
            cumulus_scheme=None,
        )
    assert "--pbl-cadence 120" in str(caught.value)


def test_the_unanchored_refusal_names_the_held_cadence():
    message = dt_admission.unanchored_refusal(3.0, None, 120.0)
    assert "surface/PBL cadence held at 120 s" in message


# ---------------------------------------------------------------------------
# the candidate mint can earn the held row -- the guard that used to block it
# ---------------------------------------------------------------------------
def test_a_candidate_mint_admits_a_held_cadence_at_an_anchored_timestep():
    """5 s convection-off is anchored WELDED; held at 120 s it is not.

    RETIRED GUARD (fix-retires-guards, 2026-08-25): before the key
    carried the cadence, this mint refused with "already holds an anchor" --
    the identical failure the convection lane hit at 20 s, one knob over.
    """

    assert dt_admission.admitted_timestep(5.0, None) is not None
    with dt_admission.candidate_mint(
        5.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
        surface_pbl_seconds=120.0,
    ) as candidate:
        assert candidate.surface_pbl_seconds == 120.0
        assert candidate.admitted_on == "CANDIDATE-UNANCHORED"
        # The held configuration is admitted; the welded row is untouched.
        assert dt_admission.admitted_timestep(5.0, None, 120.0) is not None
        welded = dt_admission.admitted_timestep(5.0, None)
        assert welded.admitted_on == "2026-08-26"
    assert dt_admission.admitted_timestep(5.0, None, 120.0) is None


def test_a_candidate_mint_still_refuses_a_configuration_that_is_anchored():
    with pytest.raises(dt_admission.DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            5.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="test, no card",
            cumulus_scheme=None,
            cumulus_seconds=None,
        )
    assert "nothing for a candidate mint to earn" in str(caught.value)


def test_the_mesh_row_guard_admits_the_held_configuration_too():
    """The second half of the same chicken and egg.

    RETIRED GUARD: this one read the WELDED default and would have refused
    every held-cadence mint with "holds a real anchor", which is the exact
    shape that killed the first convection-off arm on the proving RTX 5070 Ti in 0.1 s.
    """

    with dt_admission.candidate_mint(
        20.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
        surface_pbl_seconds=120.0,
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            20.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            cumulus_scheme=None,
            surface_pbl_seconds=120.0,
        ):
            pass


# ---------------------------------------------------------------------------
# the schedule receipt reports the rate, because the rate is the subject
# ---------------------------------------------------------------------------
def test_the_schedule_receipt_reports_the_surface_pbl_call_rate():
    receipt = dt_admission.schedule_receipt(
        5.0, cumulus_scheme=None, surface_pbl_seconds=120.0, run_steps=1440
    )
    cadences = receipt["cadences"]
    assert cadences["stepbl"] == 24
    assert cadences["surface_pbl_calls_per_hour"] == 30.0
    assert cadences["surface_pbl_calls_per_hour_welded"] == 720.0
    assert cadences["surface_pbl_held"] is True


def test_a_welded_receipt_reports_itself_as_welded():
    receipt = dt_admission.schedule_receipt(
        5.0, cumulus_scheme=None, run_steps=1440
    )
    cadences = receipt["cadences"]
    assert cadences["stepbl"] == 1
    assert cadences["surface_pbl_calls_per_hour"] == 720.0
    assert cadences["surface_pbl_held"] is False


# ---------------------------------------------------------------------------
# the config carries the cadence, and reads the row that measured it
# ---------------------------------------------------------------------------
def test_the_frozen_config_admits_a_held_cadence_only_under_its_own_row():
    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig
    from hexcore.errors import ConfigurationRefusal

    def build():
        return V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=5.0,
            config_bldt_seconds=120.0,
            config_cudt_seconds=None,
            config_convection_scheme="off",
        )

    with pytest.raises(ConfigurationRefusal):
        build().validate()

    with dt_admission.candidate_mint(
        5.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
        surface_pbl_seconds=120.0,
    ):
        build().validate()


def test_the_proven_configuration_still_validates_unchanged():
    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    V841MpasColumnPhysicsSmagorinskyGwdoConfig().validate()


# ---------------------------------------------------------------------------
# the door's row-alone answer, which must be the run's answer
# ---------------------------------------------------------------------------
def test_the_door_answers_the_cadence_question_from_the_row_alone():
    """No card, no file: preflight and the run must not disagree."""

    from hexcore import forecast_door

    admitted = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0
    )
    assert admitted["surface_pbl_seconds"] == 120.0
    assert admitted["pbl_cadence"]["source"] == "welded"
    assert admitted["pbl_cadence"]["held"] is False


def test_the_door_refuses_a_held_cadence_by_name_and_names_the_remedy():
    from hexcore import forecast_door

    with pytest.raises(forecast_door.ForecastDoorRefusal) as caught:
        forecast_door.admit_timestep(
            "x1.40962", 120.0, nominal_dx_m=120_000.0, pbl_cadence="600"
        )
    message = str(caught.value)
    assert "surface/PBL held at 600 s" in message
    assert "--pbl-cadence 600" in message


# ---------------------------------------------------------------------------
# one decision, one source -- the shape the 2026-08-26 clock fix set
# ---------------------------------------------------------------------------
def test_the_door_threads_one_cadence_decision_into_the_driver():
    """Two sources of one decision is how config_dt and the seam's dt came
    to disagree; the cadence takes the same road and the same refusal."""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    door = (root / "src" / "hexcore" / "forecast_door.py").read_text(
        encoding="utf-8"
    )
    assert '"--pbl-cadence", request.pbl_cadence,' in door
    assert "pbl_cadence=request.pbl_cadence," in door

    binding = (root / "tools" / "mpas_mesh_binding.py").read_text(
        encoding="utf-8"
    )
    assert "forecast.PBL_CADENCE_DECISION = dict(pbl_decision)" in binding
    assert '"PBL_CADENCE_DECISION",' in binding

    runner = (root / "tools" / "run_cuda_v841_forecast.py").read_text(
        encoding="utf-8"
    )
    # The driver REFUSES when its own request disagrees with the bind's.
    assert "One decision, " in runner
    assert 'pbl_decision.get("requested") != pbl_cadence' in runner
    # The cadence reaches the config from the decision, not from a default.
    assert 'surface_pbl_seconds=pbl_decision["surface_pbl_seconds"],' in runner


def test_the_campaign_runner_carries_the_cadence_into_every_arm():
    from pathlib import Path

    campaign = (
        Path(__file__).resolve().parents[1] / "tools" / "run_dt_anchor_campaign.py"
    ).read_text(encoding="utf-8")
    assert '"--pbl-cadence", pbl_cadence,' in campaign
    assert "pbl_cadence=arguments.pbl_cadence," in campaign
    assert "surface_pbl_seconds=surface_pbl_seconds," in campaign
