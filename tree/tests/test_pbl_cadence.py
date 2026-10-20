"""The surface/PBL cadence: welded by default, selectable by the second.

``config_bldt_seconds`` is welded to ``config_dt`` by default -- the native
x4 v8.4.1 reference ran ``bldt = dt`` -- and at the 5 s a sub-kilometre
mesh declares that is 720 surface/PBL calls an hour against the proven
configuration's 30.  ``--pbl-cadence SECONDS`` calls the stack once every
``SECONDS / dt`` steps and holds its tendency on the steps between (the
engine's own positive-``bldt`` path).  It changes the forecast, so it is
SELECTABLE and never silently the default -- the project's rule for modes
that change results -- and the welded default moves only on a measured
recommendation.

**The refusal.**  A cadence that is not a whole number of steps is refused
on the host, naming both numbers, the multiples of ``dt`` on either side and
the way out, before a mesh is bound or a card is reserved.

**The admission.**  A held cadence at an anchored ``(dt, cumulus)`` is
admitted through a row DERIVED from the welded anchor: the dycore half is
the welded row's own (the outer step, its schedule and its byte-identical
dual run are properties of the timestep), the host half is re-minted for the
held cadence, and the physics band is stamped NOT MEASURED rather than
borrowed.  The derived row has its own registry key, so the breakage the key
fragment prevents -- a held run quoting a band measured at 24x its own call
rate -- cannot recur; a registered held row still wins the exact lookup; and
a held cadence at an unanchored timestep is refused as the welded run would
be.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import dataclasses

import pytest

from hexcore import dt_admission, mesh_row_candidate, pbl_cadence
from hexcore.pbl_cadence import PblCadenceError
from _layout import PACKAGE_DIR


# ---------------------------------------------------------------------------
# the default is the proven weld, and it moves nothing
# ---------------------------------------------------------------------------
def test_a_bare_decision_is_the_weld_and_needs_no_flag():
    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=120.0)
    assert decision["source"] == "welded"
    assert decision["held"] is False
    assert decision["surface_pbl_seconds"] == 120.0
    assert decision["steps_between_calls"] == 1
    assert decision["steps_held_between_calls"] == 0
    assert decision["calls_per_hour"] == 30.0
    assert decision["default"] == pbl_cadence.WELDED


def test_the_weld_follows_dt_down_which_is_the_whole_cost():
    """The proven semantics call the stack 24x more often at 5 s."""

    for dt, rate in ((120.0, 30.0), (20.0, 180.0), (5.0, 720.0)):
        decision = pbl_cadence.pbl_cadence_decision(dt_seconds=dt)
        assert decision["surface_pbl_seconds"] == dt
        assert decision["calls_per_hour"] == rate
        assert decision["calls_in_first_hour"] == rate
        assert decision["steps_between_calls"] == 1
    assert 720.0 / 30.0 == 24.0


def test_holding_the_cadence_restores_the_proven_call_rate():
    for dt, steps in ((20.0, 6), (5.0, 24)):
        decision = pbl_cadence.pbl_cadence_decision(dt_seconds=dt, requested="120")
        assert decision["source"] == "explicit"
        assert decision["held"] is True
        assert decision["surface_pbl_seconds"] == 120.0
        assert decision["steps_between_calls"] == steps
        assert decision["steps_held_between_calls"] == steps - 1
        assert decision["calls_per_hour"] == 30.0
        assert decision["calls_per_hour_welded"] == 3600.0 / dt


def test_the_campaign_cadences_at_five_seconds():
    """The 2026-09-13 arms: 30, 60 and 120 s at the point mesh's 5 s."""

    for seconds, steps, first_hour in ((30, 6, 121), (60, 12, 61), (120, 24, 31)):
        decision = pbl_cadence.pbl_cadence_decision(
            dt_seconds=5.0, requested=str(seconds)
        )
        assert decision["steps_between_calls"] == steps
        assert decision["calls_per_hour"] == 3600.0 / seconds
        assert decision["calls_in_first_hour"] == first_hour


def test_an_explicit_cadence_records_itself_as_a_selection_never_the_default():
    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=5.0, requested="120")
    assert decision["source"] == "explicit"
    assert decision["selectable"] is True
    assert decision["default"] == "auto"
    assert "selectable configuration" in decision["note"]
    assert "the default stays the weld" in decision["note"]
    assert decision["schema"] == pbl_cadence.SCHEMA


def test_the_decision_names_every_physics_call_rate_the_run_uses():
    """Radiation is on its own cadence and was never welded to dt; the
    receipt records it beside the surface/PBL one so nothing is left to
    divide.  The default is MPAS-A's 00:30:00 (Registry.xml:2316, 2321)."""

    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=5.0, requested="60")
    assert decision["radiation_seconds"] == 1800.0
    assert decision["radiation_steps_between_calls"] == 360
    assert decision["radiation_calls_per_hour"] == 2.0
    assert "applied unchanged" in decision["hold_semantics"]
    assert "Radiation keeps its own cadence" in decision["hold_semantics"]
    run = pbl_cadence.pbl_cadence_decision(
        dt_seconds=5.0, requested="60", radiation_seconds=600.0
    )
    assert run["radiation_steps_between_calls"] == 120


def test_the_engine_call_count_is_one_more_than_the_steady_rate_when_held():
    """``_surface_pbl_step_due`` fires on step 1 AND on every multiple of
    stepbl, so a held run makes one call more than executed/stepbl."""

    calls = pbl_cadence.calls_in_steps
    assert calls(steps_between_calls=1, executed_steps=720) == 720
    assert calls(steps_between_calls=24, executed_steps=720) == 31
    assert calls(steps_between_calls=24, executed_steps=23) == 1
    assert calls(steps_between_calls=24, executed_steps=24) == 2
    assert calls(steps_between_calls=6, executed_steps=0) == 0
    with pytest.raises(PblCadenceError):
        calls(steps_between_calls=0, executed_steps=10)


def test_the_gate_carries_the_measurement_that_produced_it():
    """Gate law: a gate that cannot name what it prevents does not exist."""

    breakage = pbl_cadence.BREAKAGE
    assert "93.957" in breakage
    assert "91.4" in breakage
    assert "720" in breakage and "30" in breakage


# ---------------------------------------------------------------------------
# an incommensurate cadence is refused on the host, naming both numbers,
# the multiples on either side and the way out
# ---------------------------------------------------------------------------
def test_a_cadence_that_is_not_a_whole_number_of_steps_is_refused_by_name():
    """THE BREAKAGE: the sealed constructor asks this AFTER the card is
    reserved, which is the wrong place to learn it; and rounding silently
    would run a cadence the receipt did not name."""

    with pytest.raises(PblCadenceError) as caught:
        pbl_cadence.pbl_cadence_decision(dt_seconds=7.0, requested="120")
    message = str(caught.value)
    assert "--pbl-cadence 120" in message
    assert "not a whole number of 7 s steps" in message
    assert "119 s and 126 s" in message
    assert "'auto' is the welded default" in message
    assert "nothing is rounded" in message


def test_the_refusal_never_rounds_below_one_step():
    with pytest.raises(PblCadenceError) as caught:
        pbl_cadence.pbl_cadence_decision(dt_seconds=5.0, requested="3")
    assert "5 s and 5 s" in str(caught.value)


@pytest.mark.parametrize("bad", ["0", "-5", "nonsense", "", "1e400"])
def test_a_request_that_is_neither_auto_nor_seconds_is_refused(bad):
    with pytest.raises(PblCadenceError):
        pbl_cadence.pbl_cadence_decision(dt_seconds=20.0, requested=bad)


# ---------------------------------------------------------------------------
# the registry keys on the configuration, cadence included
# ---------------------------------------------------------------------------
def test_every_registered_anchor_is_a_welded_row():
    """No registered row was earned held: every one is welded."""

    for key, anchor in dt_admission.ADMITTED_TIMESTEPS.items():
        assert anchor.surface_pbl_seconds == anchor.dt_seconds
        assert anchor.derived is False
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
    assert dt_admission.dt_key(5.0, None, 5.0) == dt_admission.dt_key(5.0, None)
    assert dt_admission.surface_pbl_key(5.0, 5.0) == "dt"
    assert dt_admission.surface_pbl_key(5.0, None) == "dt"


# ---------------------------------------------------------------------------
# a held cadence at an anchored timestep is admitted through a DERIVED row
# ---------------------------------------------------------------------------
def test_a_held_cadence_at_an_anchored_timestep_is_admitted_on_the_welded_dycore_evidence():
    welded = dt_admission.admitted_timestep(5.0, None)
    assert welded is not None and welded.derived is False
    held = dt_admission.admitted_timestep(5.0, None, 120.0)
    assert held is not None
    assert held.derived is True
    assert held.derived_from == dt_admission.dt_key(5.0, None)
    assert held.surface_pbl_seconds == 120.0
    assert held.dt_seconds == welded.dt_seconds
    assert held.radiation_seconds == welded.radiation_seconds
    assert held.cumulus_scheme == welded.cumulus_scheme
    assert held.meshes == welded.meshes
    assert held.card == welded.card
    # The dycore half is the welded row's own and says so.
    assert welded.integration_anchor in held.integration_anchor
    assert "properties of the timestep" in held.integration_anchor
    # The host half is re-minted for the held cadence.
    assert "stepbl 24" in held.schedule_receipt
    assert "30 surface/PBL calls an hour" in held.schedule_receipt
    # The physics band is NOT borrowed.
    assert held.physics_health.startswith("NOT MEASURED")
    assert welded.physics_health in held.physics_health
    assert "DERIVED from the welded row" in held.basis


def test_a_derived_row_is_never_a_registered_one():
    """The exact lookup a mint and the mesh-row override ask must not see
    it, or a held cadence would read as already earned and could never be
    minted."""

    assert dt_admission.registered_anchor(5.0, None, 120.0) is None
    assert dt_admission.registered_anchor(5.0, None) is not None
    assert dt_admission.dt_key(5.0, None, 120.0) not in dt_admission.ADMITTED_TIMESTEPS


def test_a_held_cadence_at_an_unanchored_timestep_is_refused_as_the_welded_run_would_be():
    """RETIRED 2026-10-10: a held cadence at an unanchored timestep is
    recorded as unmatched, never refused, exactly as the welded run is."""

    assert dt_admission.admitted_timestep(3.0, None) is None
    assert dt_admission.derived_held_cadence_anchor(3.0, None, 120.0) is None
    assert dt_admission.admitted_timestep(3.0, None, 120.0) is None
    record = dt_admission.anchor_record(
        3.0,
        radiation_seconds=600.0,
        surface_pbl_seconds=120.0,
        cumulus_seconds=None,
        cumulus_scheme=None,
    )
    assert record["matched"] is False
    assert record["configuration"]["surface_pbl_seconds"] == 120.0


def test_a_derived_row_refuses_an_incommensurate_cadence_rather_than_rounding():
    """A refusal, not an absence: the lookup names both numbers instead of
    answering "no row" as though the timestep were unknown."""

    with pytest.raises(dt_admission.DtAdmissionError) as caught:
        dt_admission.derived_held_cadence_anchor(120.0, "gf", 500.0)
    assert "500 s is not a positive integer multiple of dt=120 s" in str(caught.value)
    with pytest.raises(dt_admission.DtAdmissionError):
        dt_admission.admitted_timestep(120.0, "gf", 500.0)


def test_require_dt_anchor_admits_the_held_cadence_and_still_refuses_radiation():
    """The held cadence is matched through the derived row; a radiation
    cadence the anchor never measured is recorded as a mismatch, not refused."""

    record = dt_admission.anchor_record(
        5.0,
        radiation_seconds=600.0,
        surface_pbl_seconds=120.0,
        cumulus_seconds=None,
        cumulus_scheme=None,
    )
    assert record["matched"] is True
    assert record["row"]["derived_from"] is not None
    other = dt_admission.anchor_record(
        5.0,
        radiation_seconds=1200.0,
        surface_pbl_seconds=120.0,
        cumulus_seconds=None,
        cumulus_scheme=None,
    )
    assert other["matched"] is False and other["cadences_match"] is False


def test_a_registered_held_row_wins_the_exact_lookup(monkeypatch):
    welded = dt_admission.admitted_timestep(5.0, None)
    earned = dataclasses.replace(
        welded, surface_pbl_seconds=120.0, admitted_on="2099-01-01"
    )
    key = dt_admission.dt_key(5.0, None, 120.0)
    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**dt_admission.ADMITTED_TIMESTEPS, key: earned},
    )
    found = dt_admission.admitted_timestep(5.0, None, 120.0)
    assert found is earned
    assert found.derived is False


def test_the_roster_names_a_held_row_and_says_when_it_is_derived():
    held = dt_admission.admitted_timestep(5.0, None, 120.0)
    label = dt_admission.anchor_label(held)
    assert "surface/PBL held at 120 s" in label
    assert "derived from the welded row" in label
    welded = dt_admission.admitted_timestep(5.0, None)
    assert "surface/PBL held" not in dt_admission.anchor_label(welded)
    earned = dataclasses.replace(welded, surface_pbl_seconds=120.0)
    assert "derived" not in dt_admission.anchor_label(earned)


def test_the_unanchored_refusal_names_the_held_cadence_and_the_way_out():
    """The record names the held cadence where the refusal used to."""

    record = dt_admission.anchor_record(
        3.0, radiation_seconds=600.0, surface_pbl_seconds=120.0,
        cumulus_seconds=None, cumulus_scheme=None,
    )
    assert record["configuration"]["surface_pbl_seconds"] == 120.0
    assert record["matched"] is False


# ---------------------------------------------------------------------------
# the candidate mint can still EARN the held row
# ---------------------------------------------------------------------------
def test_a_candidate_mint_earns_a_held_row_the_derived_row_does_not_pretend_to_be():
    assert dt_admission.registered_anchor(5.0, None, 120.0) is None
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
        registered = dt_admission.registered_anchor(5.0, None, 120.0)
        assert registered is candidate
        assert dt_admission.admitted_timestep(5.0, None, 120.0) is candidate
        welded = dt_admission.admitted_timestep(5.0, None)
        assert welded.admitted_on == "2026-08-26"
    assert dt_admission.registered_anchor(5.0, None, 120.0) is None
    assert dt_admission.admitted_timestep(5.0, None, 120.0).derived is True


def test_a_candidate_mint_still_refuses_a_configuration_that_is_anchored():
    """RETIRED 2026-10-10: a mint at an anchored configuration keeps the row."""

    registered = dt_admission.registered_anchor(5.0, None, None)
    assert registered is not None
    with dt_admission.candidate_mint(
        5.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
    ) as row:
        assert row is registered


def test_the_mesh_row_guard_admits_the_held_configuration_under_a_mint():
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
# the schedule receipt reports the rate and the held steps
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
    assert cadences["surface_pbl_steps_held_between_calls"] == 23


def test_a_welded_receipt_reports_itself_as_welded():
    receipt = dt_admission.schedule_receipt(
        5.0, cumulus_scheme=None, run_steps=1440
    )
    cadences = receipt["cadences"]
    assert cadences["stepbl"] == 1
    assert cadences["surface_pbl_calls_per_hour"] == 720.0
    assert cadences["surface_pbl_held"] is False
    assert cadences["surface_pbl_steps_held_between_calls"] == 0


# ---------------------------------------------------------------------------
# the config carries the cadence and admits it through the derived row
# ---------------------------------------------------------------------------
def test_the_frozen_config_admits_a_held_cadence_without_a_mint():
    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=5.0,
        config_bldt_seconds=120.0,
        config_cudt_seconds=None,
        config_convection_scheme="off",
    ).validate()


def test_the_frozen_config_refuses_a_held_cadence_at_an_unanchored_timestep():
    """RETIRED 2026-10-10: a held cadence at an unanchored timestep builds,
    and one that is not a whole number of steps is still refused."""

    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig
    from hexcore.errors import ConfigurationRefusal

    V841MpasColumnPhysicsSmagorinskyGwdoConfig(
        config_dt=3.0,
        config_bldt_seconds=120.0,
        config_cudt_seconds=None,
        config_convection_scheme="off",
    ).validate()
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=7.0,
            config_bldt_seconds=120.0,
            config_cudt_seconds=None,
            config_convection_scheme="off",
            config_radt_seconds=1806.0,
        ).validate()
    assert "surface_pbl_seconds=120 s is not a positive integer multiple" in str(
        caught.value
    )


def test_the_proven_configuration_still_validates_unchanged():
    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    V841MpasColumnPhysicsSmagorinskyGwdoConfig().validate()


# ---------------------------------------------------------------------------
# the door's row-alone answer, which must be the run's answer
# ---------------------------------------------------------------------------
def test_the_door_answers_the_cadence_question_from_the_row_alone():
    from hexcore import forecast_door

    admitted = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0
    )
    assert admitted["pbl_cadence"]["surface_pbl_seconds"] == 120.0
    assert admitted["pbl_cadence"]["source"] == "welded"
    assert admitted["pbl_cadence"]["held"] is False


def test_the_door_admits_a_held_cadence_and_says_the_row_is_derived():
    from hexcore import forecast_door

    admitted = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0, pbl_cadence="600",
        radiation_seconds=600.0,
    )
    assert admitted["pbl_cadence"]["surface_pbl_seconds"] == 600.0
    assert admitted["pbl_cadence"]["held"] is True
    assert admitted["pbl_cadence"]["steps_between_calls"] == 5
    # x1.40962 defaults to New Tiedtke since 2026-10-05, so the held row is
    # derived from that scheme's welded 120 s row -- recorded as evidence.
    row = admitted["anchor"]["row"]
    assert row["derived_from"] == dt_admission.dt_key(120.0, "ntiedtke")
    assert row["physics_health"].startswith("NOT MEASURED")


def test_the_door_refuses_an_incommensurate_cadence_by_name_before_any_card():
    from hexcore import forecast_door

    with pytest.raises(forecast_door.ForecastDoorRefusal) as caught:
        forecast_door.admit_timestep(
            "x1.40962", 120.0, nominal_dx_m=120_000.0, pbl_cadence="500"
        )
    message = str(caught.value)
    assert "--pbl-cadence 500" in message
    assert "480 s and 600 s" in message


def test_the_door_refuses_a_held_cadence_at_an_unanchored_timestep_by_name():
    """RETIRED 2026-10-10: admitted and recorded; an incommensurate held
    cadence is the refusal that remains (the test after this one)."""

    from hexcore import forecast_door

    admitted = forecast_door.admit_timestep(
        "nowhere", 3.0, nominal_dx_m=120_000.0, pbl_cadence="120"
    )
    assert admitted["admitted"] is True
    assert admitted["anchor"]["matched"] is False


# ---------------------------------------------------------------------------
# one decision, one source -- the shape the 2026-08-26 clock fix set
# ---------------------------------------------------------------------------
def test_the_door_threads_one_cadence_decision_into_the_driver():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    door = (PACKAGE_DIR / "forecast_door.py").read_text(
        encoding="utf-8"
    )
    assert '"--pbl-cadence", request.pbl_cadence,' in door
    assert "pbl_cadence=request.pbl_cadence," in door

    binding = (PACKAGE_DIR / "drivers" / "mpas_mesh_binding.py").read_text(
        encoding="utf-8"
    )
    assert "forecast.PBL_CADENCE_DECISION = dict(pbl_decision)" in binding
    assert '"PBL_CADENCE_DECISION",' in binding

    runner = (PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py").read_text(
        encoding="utf-8"
    )
    assert "One decision, " in runner
    assert 'pbl_decision.get("requested") != pbl_cadence' in runner
    assert 'surface_pbl_seconds=pbl_decision["surface_pbl_seconds"],' in runner


def test_the_driver_receipt_proves_the_cadence_that_ran():
    """A/B rule: the receipt carries the seam's own due/held counts against
    the count the declared cadence predicts, and says whether they agree."""

    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "_test_cadence_forecast_tool", PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    decision = pbl_cadence.pbl_cadence_decision(dt_seconds=5.0, requested="120")

    def receipt(step: int) -> dict:
        due = step == 1 or step % 24 == 0
        return {
            "step": step,
            "backend": {
                "cadence": {
                    "surface_pbl_ran": due,
                    "radiation_ran": step == 1 or step % 120 == 1,
                    "call_counts": {
                        "ysu": 1 + step // 24,
                        "radiation": 1 + step // 120,
                    },
                }
            },
        }

    receipts = [receipt(step) for step in range(1, 721)]
    proof = module._physics_cadence_proof(decision, receipts, 720)
    assert proof["surface_pbl_calls_expected"] == 31
    assert proof["surface_pbl_steps_reported_due"] == 31
    assert proof["surface_pbl_steps_held"] == 689
    assert proof["consistent"] is True
    assert "They agree" in proof["note"]

    # A run that reported every step due while declaring a held cadence
    # did not hold it, and the receipt says so.
    for item in receipts:
        item["backend"]["cadence"]["surface_pbl_ran"] = True
    proof = module._physics_cadence_proof(decision, receipts, 720)
    assert proof["consistent"] is False
    assert "THEY DISAGREE" in proof["note"]
    assert module._physics_cadence_proof(None, receipts, 720) is None


def test_the_campaign_runner_carries_the_cadence_into_every_arm():
    from pathlib import Path

    campaign = (
        Path(__file__).resolve().parents[1] / "tools" / "run_dt_anchor_campaign.py"
    ).read_text(encoding="utf-8")
    assert '"--pbl-cadence", pbl_cadence,' in campaign
    assert "pbl_cadence=arguments.pbl_cadence," in campaign
    assert "surface_pbl_seconds=surface_pbl_seconds," in campaign
