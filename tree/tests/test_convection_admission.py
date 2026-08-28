"""Convection admission: the scheme is OFF below 3 km, by default.

It was ruled on 2026-08-26 that convection is switched off below 3 km,
answering directly whether a fine swath runs the cumulus scheme at all.
This is that ruling, made executable.

**Why it is a gate and not a preference.**  THE BREAKAGE THIS PREVENTS,
MEASURED (2026-08-26, `evidence/dt-anchors-20260826/RECEIPT.md`): WRF pins
``cudt = 0`` for ``cu_physics = 3``, so Grell-Freitas recomputes every
model step and no configuration relaxes it.  A 5 s timestep therefore calls
the closure 720 times an hour against the 30 the lane was proven at, and the
5 s anchor measured a |w| mean climbing 12.0 / 49.6 / 73.5 / 87.5 m/s over
four half-hour windows against a 120 s control's 1.15 / 1.17 / 1.20 / 1.48,
with |w| max 102.67 against 1.680.  A 3 km swath declares 20 s and a 750 m
swath declares 5 s, so every mesh the ruling covers runs the closure at
between six and twenty-four times its proven call rate.

**Fixed means default.**  A bare run on a sub-3-km mesh must stop calling
the scheme with nobody passing a flag; the explicit ``--convection`` switch
exists to run A/B arms (that is how the attribution above was measured),
never as the remedy.

**Convection-off is a NEW admitted configuration, not a mutation of the
proven one.**  The frozen v8.4.1 column-physics lane's proven configuration
stays exactly reachable and exactly refusing: how often a scheme is called
is part of what an anchor certifies, so a convection-off run earns its own
anchors and cannot borrow the GF rows'.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import dataclasses

import pytest

from hexcore import convection_admission, dt_admission
from hexcore.convection_admission import (
    CONVECTION_OFF_BELOW_M,
    ConvectionAdmissionError,
    SCHEME_GRELL_FREITAS,
    SCHEME_OFF,
)
from hexcore.config_v841 import V841MpasColumnPhysicsConfig
from hexcore.errors import ConfigurationRefusal


# ---------------------------------------------------------------------------
# the ruling itself
# ---------------------------------------------------------------------------
def test_the_ruling_switches_convection_off_below_three_kilometres():
    assert CONVECTION_OFF_BELOW_M == 3_000.0
    assert convection_admission.convection_for_spacing(2_999.0) == SCHEME_OFF
    assert convection_admission.convection_for_spacing(750.0) == SCHEME_OFF
    assert convection_admission.convection_for_spacing(1_000.0) == SCHEME_OFF
    # At and above the threshold the proven configuration is unchanged.
    assert (
        convection_admission.convection_for_spacing(3_000.0) == SCHEME_GRELL_FREITAS
    )
    assert (
        convection_admission.convection_for_spacing(120_000.0) == SCHEME_GRELL_FREITAS
    )


def test_the_finest_spacing_anywhere_on_the_mesh_decides():
    """A swath is fine somewhere and coarse elsewhere; the fine part rules.

    the question put was about a fine SWATH inside a coarser mesh, so the
    decision is taken on the finest spacing the mesh carries, not on its
    average or its declared nominal alone.
    """

    decision = convection_admission.convection_decision(
        nominal_dx_m=15_000.0, minimum_dc_edge_m=2_800.0
    )
    assert decision["scheme"] == SCHEME_OFF
    assert decision["finest_spacing_m"] == 2_800.0
    assert decision["below_threshold"] is True

    coarse = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, minimum_dc_edge_m=97_076.0
    )
    assert coarse["scheme"] == SCHEME_GRELL_FREITAS
    assert coarse["finest_spacing_m"] == 97_076.0
    assert coarse["below_threshold"] is False


def test_the_decision_rides_into_a_receipt_naming_the_ruling_and_the_measurement():
    decision = convection_admission.convection_decision(nominal_dx_m=750.0)
    assert decision["threshold_m"] == CONVECTION_OFF_BELOW_M
    assert "2026-08-26" in decision["ruling"]
    # A gate names the concrete breakage it prevents, with the measurement.
    assert "102.67" in decision["breakage"]
    assert "720" in decision["breakage"]
    assert decision["source"] == "resolution"


def test_a_spacing_that_is_not_a_positive_length_is_refused_by_name():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ConvectionAdmissionError) as error:
            convection_admission.convection_for_spacing(bad)
        assert "finite and positive" in str(error.value)


# ---------------------------------------------------------------------------
# the two vocabularies, and the one translation between them
# ---------------------------------------------------------------------------
def test_the_config_name_and_the_sealed_constructor_name_agree():
    assert convection_admission.constructor_scheme(SCHEME_GRELL_FREITAS) == "gf"
    assert convection_admission.constructor_scheme(SCHEME_OFF) is None
    with pytest.raises(ConvectionAdmissionError):
        convection_admission.constructor_scheme("cu_ntiedtke")


def test_gf_shallow_is_off_when_there_is_no_gf_to_be_shallow_in():
    """``gf_ishallow=1`` with no GF is refused by the sealed constructor."""

    assert convection_admission.gf_ishallow(SCHEME_GRELL_FREITAS) == 1
    assert convection_admission.gf_ishallow(SCHEME_OFF) == 0


# ---------------------------------------------------------------------------
# fixed means default
# ---------------------------------------------------------------------------
def test_a_bare_run_below_three_kilometres_selects_no_cumulus_scheme():
    """No flag, no argument: the scheme is simply not selected."""

    assert convection_admission.default_convection_scheme(nominal_dx_m=750.0) == (
        SCHEME_OFF
    )
    assert convection_admission.default_convection_scheme(nominal_dx_m=2_500.0) == (
        SCHEME_OFF
    )
    assert convection_admission.default_convection_scheme(
        nominal_dx_m=120_000.0
    ) == SCHEME_GRELL_FREITAS


def test_an_explicit_switch_is_an_ab_arm_and_says_so():
    forced_off = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, requested="off"
    )
    assert forced_off["scheme"] == SCHEME_OFF
    assert forced_off["source"] == "explicit"
    assert "A/B" in forced_off["note"]

    forced_on = convection_admission.convection_decision(
        nominal_dx_m=750.0, requested="gf"
    )
    assert forced_on["scheme"] == SCHEME_GRELL_FREITAS
    assert forced_on["source"] == "explicit"
    # Overriding the ruling on a mesh the ruling covers must say what it is.
    assert "ruling" in forced_on["note"]

    auto = convection_admission.convection_decision(
        nominal_dx_m=120_000.0, requested="auto"
    )
    assert auto["source"] == "resolution"

    with pytest.raises(ConvectionAdmissionError):
        convection_admission.convection_decision(
            nominal_dx_m=120_000.0, requested="sometimes"
        )


# ---------------------------------------------------------------------------
# the frozen configuration: still reachable, still exact
# ---------------------------------------------------------------------------
def test_the_proven_grell_freitas_configuration_is_unchanged():
    config = V841MpasColumnPhysicsConfig()
    config.validate()
    assert config.config_convection_scheme == SCHEME_GRELL_FREITAS
    assert config.config_cudt_seconds == 120.0


def test_convection_off_is_admitted_as_a_configuration_of_its_own():
    config = V841MpasColumnPhysicsConfig(
        config_dt=120.0,
        config_bldt_seconds=120.0,
        config_convection_scheme=SCHEME_OFF,
        config_cudt_seconds=None,
    )
    # No anchor exists for convection-off at 120 s, so the frozen lane still
    # refuses it -- by name, and for the RIGHT reason.
    with pytest.raises(ConfigurationRefusal) as error:
        config.validate()
    message = str(error.value)
    assert "convection off" in message
    assert "anchor" in message


def test_a_third_convection_scheme_is_still_refused():
    config = V841MpasColumnPhysicsConfig(config_convection_scheme="cu_ntiedtke")
    with pytest.raises(ConfigurationRefusal):
        config.validate()


def test_the_cumulus_cadence_and_the_scheme_must_agree():
    """``off`` with a cumulus cadence is a configuration that means nothing."""

    with pytest.raises(ConfigurationRefusal) as error:
        V841MpasColumnPhysicsConfig(
            config_convection_scheme=SCHEME_OFF, config_cudt_seconds=120.0
        ).validate()
    assert "config_cudt_seconds" in str(error.value)

    with pytest.raises(ConfigurationRefusal) as error:
        V841MpasColumnPhysicsConfig(
            config_convection_scheme=SCHEME_GRELL_FREITAS, config_cudt_seconds=None
        ).validate()
    assert "config_cudt_seconds" in str(error.value)


# ---------------------------------------------------------------------------
# an anchor certifies a CONFIGURATION at a timestep, not a timestep alone
# ---------------------------------------------------------------------------
def test_the_registry_is_keyed_by_configuration_not_by_timestep_alone():
    gf_key = dt_admission.dt_key(20.0, "gf")
    off_key = dt_admission.dt_key(20.0, None)
    assert gf_key != off_key
    assert "gf" in gf_key
    assert "off" in off_key


def test_a_convection_off_run_cannot_borrow_the_grell_freitas_anchor():
    assert dt_admission.admitted_timestep(120.0, "gf") is not None
    with pytest.raises(dt_admission.DtAdmissionError) as error:
        dt_admission.require_dt_anchor(
            120.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=120.0,
            cumulus_seconds=None,
            cumulus_scheme=None,
        )
    message = str(error.value)
    assert "convection off" in message
    assert "earns its own" in message or "anchor" in message


def test_every_registered_anchor_keys_on_its_own_configuration():
    for key, anchor in dt_admission.ADMITTED_TIMESTEPS.items():
        assert key == dt_admission.dt_key(anchor.dt_seconds, anchor.cumulus_scheme)


def test_a_candidate_mint_separates_the_two_configurations():
    """20 s holds a GF anchor; convection-off at 20 s is still unearned."""

    assert dt_admission.admitted_timestep(20.0, "gf") is not None
    with pytest.raises(dt_admission.DtAdmissionError):
        # Nothing for a candidate mint to earn: this configuration is anchored.
        dt_admission.candidate_mint(
            20.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="test, no card",
            cumulus_scheme="gf",
        )


def test_the_unanchored_refusal_names_the_configuration():
    message = dt_admission.unanchored_refusal(20.0, cumulus_scheme=None)
    assert "convection off" in message
    assert "20 s" in message


def test_the_admitted_summary_distinguishes_the_configurations():
    roster = dt_admission.admitted_summary()
    assert "120 s" in roster
    for anchor in dt_admission.ADMITTED_TIMESTEPS.values():
        label = "convection off" if anchor.cumulus_scheme is None else "GF"
        assert label in roster


# ---------------------------------------------------------------------------
# the schedule receipt already knows how to describe convection-off
# ---------------------------------------------------------------------------
def test_a_convection_off_schedule_receipt_reports_no_cumulus_calls():
    receipt = dt_admission.schedule_receipt(5.0, cumulus_scheme=None)
    cadences = receipt["cadences"]
    assert cadences["cumulus_scheme"] is None
    assert cadences["cumulus_seconds"] is None
    assert cadences["stepcu"] == 0
    assert cadences["cumulus_calls_per_hour"] is None


def test_an_anchor_row_is_immutable_and_serialises_with_its_scheme():
    anchor = dt_admission.admitted_timestep(120.0, "gf")
    assert anchor is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        anchor.cumulus_scheme = None  # type: ignore[misc]
    assert anchor.as_dict()["cumulus_scheme"] == "gf"


# ---------------------------------------------------------------------------
# guards that were keyed on the timestep alone, and are not any more
# ---------------------------------------------------------------------------
def test_the_row_override_admits_a_convection_off_mint_at_an_anchored_timestep():
    """RETIRED GUARD, MEASURED (2026-08-26, the proving RTX 5070 Ti).

    ``mesh_row_candidate`` refuses to override a registry row's timestep when
    that timestep already holds a real anchor -- correctly, because a proven
    timestep needs a registered row.  It asked the question of the TIMESTEP
    though, with the Grell-Freitas default, so once the registry was keyed by
    configuration it refused every convection-off mint at 20 s and 5 s with
    "20 s holds a real anchor" when the configuration being earned held none.
    It killed the first convection-off arm in 0.1 s.
    """

    from hexcore import mesh_row_candidate

    # DERIVED, never hardcoded.  This control named 20.0 while 20 s held a
    # Grell-Freitas anchor and no convection-off one -- and then the
    # convection-off anchors were earned and the control broke on its own
    # success, which is exactly the trap the dt-anchors mutation control fell
    # into on the same day.  So it searches: a timestep that IS anchored with
    # the closure on and is NOT anchored with it off.
    probe = next(
        (
            anchor.dt_seconds
            for anchor in sorted(
                dt_admission.ADMITTED_TIMESTEPS.values(),
                key=lambda a: a.dt_seconds,
            )
            if anchor.cumulus_scheme == "gf"
            and dt_admission.admitted_timestep(anchor.dt_seconds, None) is None
        ),
        None,
    )
    assert probe is not None, (
        "every Grell-Freitas timestep now also holds a convection-off anchor; "
        "that is a finding, not a test failure -- re-read this control"
    )
    assert dt_admission.admitted_timestep(probe, "gf") is not None

    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="test, no card",
        cumulus_scheme=None,
        cumulus_seconds=None,
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            cumulus_scheme=None,
        ) as override:
            assert override.dt_seconds == probe
            assert "x1.40962" in mesh_row_candidate.active_overrides()
    assert "x1.40962" not in mesh_row_candidate.active_overrides()


def test_the_row_override_still_refuses_an_anchored_configuration():
    """The guard keeps its teeth for the configuration that IS anchored."""

    from hexcore import mesh_row_candidate

    with pytest.raises(dt_admission.DtAdmissionError) as error:
        mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            20.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            cumulus_scheme="gf",
        )
    message = str(error.value)
    assert "holds a real" in message
    assert "GF" in message


def test_the_physics_cadence_table_reports_no_convection_rather_than_a_zero():
    """RETIRED GUARD, MEASURED (2026-08-26, the proving RTX 5070 Ti).

    ``_v841_physics_cadences`` called ``float(config.config_cudt_seconds)``
    unconditionally, so the first convection-off arm bound clean, admitted
    its timestep, built its sealed constructor and died inside composite
    step 0 with ``float() argument must be a string or a real number, not
    'NoneType'`` -- surfacing as "composite step at 0.0 s was aborted
    without publication", which names neither the field nor the reason.

    ``None`` and not ``0.0``: every other entry in that table is a call
    interval, and a zero there reads as "called every instant".
    """

    from hexcore.cuda_driver import _v841_physics_cadences

    off = V841MpasColumnPhysicsConfig(
        config_convection_scheme=SCHEME_OFF, config_cudt_seconds=None
    )
    cadences = _v841_physics_cadences(off)
    assert cadences["convection"] is None
    assert cadences["microphysics"] == 120.0
    assert cadences["radiation_lw"] == 600.0


def test_the_run_refuses_two_sources_for_one_convection_decision():
    """The bind's request and the driver's must be the same string.

    The 2026-08-26 clock defect was two independent sources for one
    timestep, discovered 285 s and 18,820 MiB into a run.  The convection
    selection travels the same road and gets the same check.
    """

    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    source = (tools / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    assert 'decision.get("requested") != convection' in source
    assert "One decision, one " in source
    # And the door hands the SAME string to the bind and to the driver.
    door = (
        Path(__file__).resolve().parents[1]
        / "src" / "hexcore" / "forecast_door.py"
    ).read_text(encoding="utf-8")
    assert '"--convection", request.convection,' in door
    assert "convection=request.convection," in door


# ---------------------------------------------------------------------------
# the door's row-alone answer, which must be the run's answer
# ---------------------------------------------------------------------------
def test_the_door_answers_the_convection_question_from_the_row_alone():
    """No card, no file: preflight and the run must not disagree."""

    from hexcore import forecast_door

    admitted = forecast_door.admit_timestep(
        "x1.40962", 120.0, nominal_dx_m=120_000.0, convection="auto"
    )
    assert admitted["cumulus_scheme"] == "gf"
    assert admitted["convection"]["source"] == "resolution"
    assert admitted["convection"]["scheme"] == SCHEME_GRELL_FREITAS
    # The anchor's verdict rides along, so a caller sees what the row's
    # physics did before spending a card on it.
    assert admitted["physics_health"].startswith("REFERENCE")


def test_a_registered_row_keeps_the_configuration_it_has_always_run():
    """Nothing already registered moves under the ruling.

    MEASURED: the finest registered mesh is 15,000 m, so every row selects
    Grell-Freitas under the 3,000 m threshold and the frozen lane's proven
    configuration is byte-unchanged for all of them.  The ruling fires the
    moment a swath row is registered, and not before.
    """

    import importlib.util
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "_test_mesh_binding_convection", tools / "mpas_mesh_binding.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_mesh_binding_convection"] = module
    spec.loader.exec_module(module)

    finest = min(
        float(row.nominal_dx_m) for row in module.MESH_BINDINGS.values()
    )
    assert finest >= CONVECTION_OFF_BELOW_M, (
        "a row finer than the threshold is now registered; the ruling fires "
        "on it, and this test should be re-read rather than relaxed"
    )
    for name, row in module.MESH_BINDINGS.items():
        decision = convection_admission.convection_decision(
            nominal_dx_m=float(row.nominal_dx_m)
        )
        assert decision["scheme"] == SCHEME_GRELL_FREITAS, name
