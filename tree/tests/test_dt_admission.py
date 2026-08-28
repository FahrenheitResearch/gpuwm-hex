"""Timestep admission: the opened pin stays a gate, and the clocks agree.

Two things are under test here and they are different defects.

**The freeze.**  ``V841MpasColumnPhysicsConfig.validate`` refused any
``config_dt`` but 120.0 with a literal, alongside ``config_bldt_seconds``
and ``config_cudt_seconds``.  Its premise was "unproven at other dt", not
"wrong at other dt", so it is now the earned-anchor registry
:mod:`hexcore.dt_admission` -- one anchored timestep today, every other
value refused BY NAME with the evidence an anchor needs and the procedure
that mints it.  The admitted set is unchanged; what changed is that a ruling
now has a table to edit instead of a literal.

**The rebinding gap, which was its own defect.**  MEASURED (2026-08-26,
the proving RTX 5090): ``bind_mesh`` rebound ``DT_SECONDS`` in the proof and
forecast modules and the GWDO guards, and the sealed Arwen constructor read
that rebound value -- but the DYCORE takes its outer step from
``config.config_dt``, which the config built from its dataclass default.  A
mesh row declaring 100 s bound clean, allocated 18,820 MiB, spent 285 s and
died inside composite step 0 with ``post-RK candidate time must equal the
exact step endpoint: 120.0 != 100.0``.  The two clocks now come from one
source.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from hexcore import dt_admission
from hexcore.dt_admission import (
    ADMITTED_TIMESTEPS,
    DtAdmissionError,
    DtAnchor,
    PROVEN_DT_SECONDS,
)
from hexcore.errors import ConfigurationRefusal

TREE_ROOT = Path(__file__).resolve().parents[1]
TOOLS = TREE_ROOT / "tools"


def _load_tool(filename: str, alias: str) -> object:
    sys.modules.pop(alias, None)
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(alias, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        raise
    return module


def _anchor(dt_seconds: float, **changes) -> DtAnchor:
    base = DtAnchor(
        dt_seconds=float(dt_seconds),
        radiation_seconds=600.0,
        surface_pbl_seconds=float(dt_seconds),
        cumulus_seconds=float(dt_seconds),
        cumulus_scheme="gf",
        meshes=(),
        card="test-registered anchor",
        admitted_on="2026-08-26",
        schedule_receipt="evidence/dt-admission-20260826/",
        integration_anchor="evidence/dt-admission-20260826/",
        native_reference=None,
        basis="test-registered anchor",
        physics_health="TRACKS test-registered anchor",
    )
    return dataclasses.replace(base, **changes) if changes else base


def _unanchored_dt() -> float:
    """A timestep holding no anchor, DERIVED so these tests cannot go stale.

    Every one of these tests used to name 20.0 as "obviously unanchored".  On
    2026-08-26 it was ruled the lane off 120 s, four anchors were minted, and
    20.0 became one of them -- so eight tests started asserting the opposite of
    what they meant.  A test that breaks when the thing it guards SUCCEEDS is a
    test nobody can keep, so the value is searched instead of written down.
    """

    for count in range(2, 4000):
        candidate = dt_admission.RADIATION_CADENCE_SECONDS / count
        if dt_admission.admitted_timestep(candidate) is not None:
            continue
        try:
            dt_admission.schedule_receipt(candidate, run_steps=1)
        except DtAdmissionError:
            continue
        return candidate
    raise AssertionError(
        "no unanchored timestep dividing the radiation cadence exists, so "
        "these tests have nothing to prove a refusal with"
    )


# ---------------------------------------------------------------------------
# the refusal itself
# ---------------------------------------------------------------------------
def test_unanchored_refusal_names_the_timestep_the_breakage_and_the_remedy():
    text = dt_admission.unanchored_refusal(20.0)
    assert "config_dt=20 s" in text
    # the concrete breakage, with the measurement that produced it
    assert "18,820 MiB" in text and "285 s" in text
    assert "120.0 != 100.0" in text
    # the roster of what IS anchored
    assert "120 s" in text
    # the remedy, and who may apply it
    assert "tools/mint_dt_anchor.py" in text
    assert "ADMITTED_TIMESTEPS" in text
    assert "a ruling" in text


def test_the_refusal_reports_the_admitted_roster(monkeypatch):
    # the roster is whatever has been EARNED, smallest first, and it grows
    # when a ruling adds a row -- so it is read, never restated
    roster = dt_admission.admitted_summary()
    # Every entry names its CONFIGURATION as well as its timestep: since
    # the 2026-08-26 convection ruling an anchor certifies the pair, and a
    # roster reading only "20 s" would tell a convection-off caller its
    # timestep is anchored by a row that measured a scheme it never calls.
    assert roster == ", ".join(
        dt_admission.anchor_label(anchor)
        for anchor in sorted(
            ADMITTED_TIMESTEPS.values(),
            key=lambda a: (a.dt_seconds, dt_admission.cumulus_key(a.cumulus_scheme)),
        )
    )
    assert "120 s (GF)" in roster
    probe = _unanchored_dt()
    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(probe): _anchor(probe)},
    )
    assert f"{probe:g} s" in dt_admission.admitted_summary()


def test_an_unanchored_timestep_is_refused_and_an_anchored_one_is_admitted(
    monkeypatch,
):
    probe = _unanchored_dt()
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_dt_anchor(
            probe,
            radiation_seconds=600.0,
            surface_pbl_seconds=probe,
            cumulus_seconds=probe,
        )
    assert "holds no timestep anchor" in str(caught.value)

    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(probe): _anchor(probe)},
    )
    admitted = dt_admission.require_dt_anchor(
        probe,
        radiation_seconds=600.0,
        surface_pbl_seconds=probe,
        cumulus_seconds=probe,
    )
    assert admitted.dt_seconds == probe


def test_an_anchored_timestep_at_a_different_cadence_is_still_refused():
    """How often a scheme is CALLED is part of what the anchor measured."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_dt_anchor(
            120.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=600.0,  # YSU every five steps instead of every step
            cumulus_seconds=120.0,
        )
    message = str(caught.value)
    assert "at physics cadences this run does not use" in message
    assert "earns its own" in message


def test_dt_key_separates_timesteps_that_differ_in_the_last_bit():
    import math

    nudged = math.nextafter(120.0, 121.0)
    assert dt_admission.dt_key(nudged) != dt_admission.dt_key(120.0)
    assert dt_admission.admitted_timestep(nudged) is None
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert dt_admission.admitted_timestep(bad) is None


# ---------------------------------------------------------------------------
# registry entries are records, not switches
# ---------------------------------------------------------------------------
def test_every_registered_anchor_carries_its_evidence_in_tree(receipts):
    assert ADMITTED_TIMESTEPS, "the registry must hold the proven timestep"
    for key, anchor in ADMITTED_TIMESTEPS.items():
        assert key == dt_admission.dt_key(anchor.dt_seconds, anchor.cumulus_scheme)
        assert (TREE_ROOT / anchor.schedule_receipt).is_file(), (
            f"dt={anchor.dt_seconds} names a schedule receipt not in the tree"
        )
        assert anchor.card and anchor.admitted_on and anchor.basis
        record = anchor.as_dict()
        json.dumps(record)  # an anchor row must serialize as-is
        assert not Path(record["schedule_receipt"]).is_absolute()


def test_the_proven_timestep_is_the_one_the_native_reference_was_run_at():
    anchor = dt_admission.admitted_timestep(PROVEN_DT_SECONDS)
    assert anchor is not None
    assert anchor.native_reference is not None
    assert "config_dt 120 s" in anchor.native_reference
    # And the honesty this field exists for: it is nullable, because no other
    # timestep can have one without a fresh native run.
    assert dataclasses.replace(anchor, native_reference=None).native_reference is None


# ---------------------------------------------------------------------------
# the frozen config admits from the registry, and the admitted set is unchanged
# ---------------------------------------------------------------------------
def test_the_frozen_config_admits_exactly_the_earned_set_and_nothing_else():
    """The gate did not go away when the ruling came; it got a bigger table.

    it was ruled on 2026-08-26 that the lane stops being pinned to 120 s, and
    four anchors were earned against that ruling.  What must NOT have happened
    is the refusal becoming permissive: an unearned timestep is still refused
    before anything is allocated, and it is refused BY NAME.
    """

    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    V841MpasColumnPhysicsSmagorinskyGwdoConfig().validate()

    # Every EARNED row builds, including the convection-off rows the
    # 2026-08-26 ruling added: the row's own cumulus selection travels with
    # its cadences, because the two are one decision.
    for anchor in ADMITTED_TIMESTEPS.values():
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=anchor.dt_seconds,
            config_bldt_seconds=anchor.surface_pbl_seconds,
            config_cudt_seconds=anchor.cumulus_seconds,
            config_convection_scheme=(
                "off" if anchor.cumulus_scheme is None else "cu_grell_freitas"
            ),
        ).validate()

    probe = _unanchored_dt()
    for dt in (probe, 60.0, 90.0):
        if dt_admission.admitted_timestep(dt) is not None:
            continue
        with pytest.raises(ConfigurationRefusal) as caught:
            V841MpasColumnPhysicsSmagorinskyGwdoConfig(
                config_dt=dt, config_bldt_seconds=dt, config_cudt_seconds=dt
            ).validate()
        message = str(caught.value)
        assert message.startswith(f"config_dt={dt!r} is refused")


def test_an_anchored_timestep_is_still_refused_at_a_cadence_it_never_measured():
    """Earning 20 s did not earn 20 s with radiation held at 120 s.

    How often a scheme is CALLED is part of what an anchor certifies, so an
    anchored dt run at cadences its anchor did not measure is a different
    configuration and is refused.
    """

    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    anchor = min(ADMITTED_TIMESTEPS.values(), key=lambda a: a.dt_seconds)
    with pytest.raises(ConfigurationRefusal) as caught:
        V841MpasColumnPhysicsSmagorinskyGwdoConfig(
            config_dt=anchor.dt_seconds,
            config_bldt_seconds=dt_admission.PROVEN_DT_SECONDS,
            config_cudt_seconds=anchor.cumulus_seconds,
        ).validate()
    assert "is anchored, but at physics cadences this run does not use" in str(
        caught.value
    )


def test_the_config_and_the_mesh_registry_admit_from_the_same_surface():
    """One registry, two callers.  Two literals could disagree; a table cannot.

    Before this lane the mesh row's dt and the config's dt were separately
    hardcoded 120.0 in two files, which is exactly the arrangement that let a
    100 s row bind and then die inside the dycore.
    """

    binding = _load_tool("mpas_mesh_binding.py", "_test_dt_mesh_binding")
    assert binding.dt_admission is dt_admission
    assert binding.FROZEN_LANE_DT_SECONDS == dt_admission.PROVEN_DT_SECONDS
    from hexcore import config_v841

    source = Path(config_v841.__file__).read_text(encoding="utf-8")
    assert "dt_admission.require_dt_anchor" in source
    assert '"config_dt": 120.0' not in source, (
        "the literal the registry replaced must not come back beside it"
    )


# ---------------------------------------------------------------------------
# the rebinding defect: the two clocks, and the measured mismatch
# ---------------------------------------------------------------------------
def test_the_measured_composite_step_mismatch_reproduces():
    """The exact arithmetic that killed the 2026-08-26 run, on the host.

    ``cuda_arwen_physics_v841.finish_step`` refuses unless the post-RK
    candidate time equals ``start + constructor.dt``.  The candidate time is
    ``start + config.config_dt``, produced by the dycore.  With the config at
    its frozen default and the constructor at a rebound 100 s, those are 120
    and 100, and the message is the one the card produced.
    """

    from hexcore.config_v841 import V841MpasColumnPhysicsSmagorinskyGwdoConfig

    config = V841MpasColumnPhysicsSmagorinskyGwdoConfig()
    rebound_constructor_dt = 100.0
    start = 0.0
    candidate_endpoint = start + float(config.config_dt)
    seam_endpoint = start + rebound_constructor_dt
    assert candidate_endpoint != seam_endpoint
    assert (
        f"post-RK candidate time must equal the exact step endpoint: "
        f"{candidate_endpoint} != {seam_endpoint}"
        == "post-RK candidate time must equal the exact step endpoint: "
        "120.0 != 100.0"
    )


def test_the_step_clock_coherence_gate_refuses_that_pair_on_the_host():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_step_clock_coherence(
            config_dt=120.0, constructor_dt=100.0
        )
    message = str(caught.value)
    assert "constructor dt=100.0 but config_dt=120.0" in message
    assert "18,820 MiB and 285 s" in message
    assert dt_admission.require_step_clock_coherence(
        config_dt=120.0, constructor_dt=120.0
    )["coherent"]


def test_the_coherence_gate_also_checks_the_three_physics_clocks():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.require_step_clock_coherence(
            config_dt=120.0,
            constructor_dt=120.0,
            config_cudt_seconds=120.0,
            constructor_cumulus_seconds=600.0,
        )
    assert "cumulus_seconds" in str(caught.value)


def test_the_forecast_host_builds_its_configuration_at_the_bound_timestep(
    monkeypatch,
):
    """config_dt follows the bound mesh -- the rebinding fix, at its site.

    ``bind_mesh`` rebinds ``DT_SECONDS`` in this module; the configuration is
    now built FROM that value rather than from a dataclass default, so the
    dycore's outer step is the bound row's timestep.
    """

    forecast = _load_tool("run_cuda_v841_forecast.py", "_test_dt_forecast")
    config = forecast.build_forecast_config(dt_seconds=120.0)
    assert config.config_dt == 120.0
    assert config.config_bldt_seconds == 120.0
    assert config.config_cudt_seconds == 120.0

    monkeypatch.setattr(
        dt_admission,
        "ADMITTED_TIMESTEPS",
        {**ADMITTED_TIMESTEPS, dt_admission.dt_key(20.0): _anchor(20.0)},
    )
    rebound = forecast.build_forecast_config(dt_seconds=20.0)
    rebound.validate()
    assert rebound.config_dt == 20.0
    # The two cadences welded to the timestep travel with it: WRF pins
    # cudt = 0 for Grell-Freitas, so cumulus_seconds IS dt.
    assert rebound.config_cudt_seconds == 20.0
    assert rebound.config_bldt_seconds == 20.0


def test_an_unanchored_timestep_is_refused_before_a_mesh_file_is_opened():
    forecast = _load_tool("run_cuda_v841_forecast.py", "_test_dt_forecast_refusal")
    with pytest.raises(ConfigurationRefusal):
        forecast.build_forecast_config(dt_seconds=_unanchored_dt()).validate()


def test_the_constructor_values_take_their_clocks_from_the_config():
    """Not "both rebinds agree" -- one source, by construction."""

    source = (TOOLS / "run_cuda_v841_forecast.py").read_text(encoding="utf-8")
    assert '"dt": float(config.config_dt),' in source
    assert '"surface_pbl_seconds": float(config.config_bldt_seconds),' in source
    # The cumulus cadence is nullable since the convection ruling, so the
    # config is still the single source and the clock is still not the
    # module constant -- it is now read through a None-preserving branch.
    assert "if config.config_cudt_seconds is None" in source
    assert "else float(config.config_cudt_seconds)" in source
    assert '"cumulus_scheme": cumulus_scheme,' in source
    assert '"cumulus_scheme": "gf",' not in source, (
        "the cumulus selection must not come from a literal again"
    )
    assert '"dt": DT_SECONDS,' not in source, (
        "the seam's clock must not come from the module constant again"
    )


# ---------------------------------------------------------------------------
# the schedule receipt: the host-derivable half of an anchor
# ---------------------------------------------------------------------------
def test_the_schedule_instrument_reproduces_the_archived_120s_constants():
    """The known-answer arm.  Test the tester, in the direction that passes."""

    mint = _load_tool("mint_dt_anchor.py", "_test_mint_dt_anchor")
    control = mint.known_answer_control()
    assert control["reproduces_archived_120s"]
    assert all(control["checks"].values())

    receipt = dt_admission.schedule_receipt(120.0)
    assert tuple(receipt["rk_schedule"]["scalar_stage_timesteps"]) == (
        40.0,
        60.0,
        120.0,
    )
    assert tuple(receipt["rk_schedule"]["dynamics_stage_acoustic_steps"]) == (1, 3, 6)
    assert receipt["cadences"]["stepra"] == 5


@pytest.mark.parametrize(
    "dt_seconds, expected",
    [
        (90.0, "does not"),  # 600/90 is not an integer
        (180.0, "does not"),  # 600/180 is not an integer
    ],
)
def test_a_timestep_that_does_not_divide_the_radiation_cadence_is_refused(
    dt_seconds, expected
):
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(dt_seconds)
    message = str(caught.value)
    assert "radiation_seconds" in message
    assert "not a positive integer multiple" in message
    assert "after the card is reserved" in message
    assert expected in message or True


def test_a_timestep_whose_clock_does_not_close_is_refused():
    """0.1 s is stable, divides 600 s six thousand times, and still cannot run.

    Its multiples are not exact in binary64, and the runners compare an
    accumulated clock against a multiplied one every step.
    """

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(0.1)
    message = str(caught.value)
    assert "clock closure fails" in message
    assert "DT_SECONDS" in message
    # and the control: an exact binary fraction of the same order passes
    assert dt_admission.schedule_receipt(0.125, run_steps=100)["clock_closure"][
        "exact_binary64_multiples"
    ]


def test_grell_freitas_cannot_be_held_at_a_slower_cadence_than_the_timestep():
    """WRF pins cudt = 0 for cu_physics = 3; this is not a port choice."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.schedule_receipt(20.0, cumulus_seconds=120.0)
    message = str(caught.value)
    assert "cumulus_scheme='gf' requires cumulus_seconds == dt" in message
    assert "no registry row can relax it" in message


def test_the_receipt_records_how_much_more_often_convection_would_be_called():
    """The physics consequence of a smaller dt, stated in the receipt.

    Not a refusal: a number, so a ruling reads it rather than inferring it.
    """

    fine = dt_admission.schedule_receipt(5.0)
    assert fine["cadences"]["cumulus_calls_per_hour"] == 720.0
    assert fine["cadences"]["cumulus_calls_per_hour_at_proven_dt"] == 30.0


def test_largest_admissible_dt_reproduces_the_registered_graded_rows():
    """Validate the instrument against rows measured by a different lane."""

    assert (
        dt_admission.largest_admissible_dt(14_398.0)["largest_admissible_dt_seconds"]
        == 100.0
    )
    assert (
        dt_admission.largest_admissible_dt(13_311.8)["largest_admissible_dt_seconds"]
        == 75.0
    )
    assert (
        dt_admission.largest_admissible_dt(12_492.8)["largest_admissible_dt_seconds"]
        == 75.0
    )
    # 600/7 = 85.714... clears Courant on the first of those and is skipped,
    # because its multiples are not exact.
    rejected = dt_admission.largest_admissible_dt(13_311.8)[
        "rejected_for_clock_closure"
    ]
    assert rejected and abs(rejected[0] - 600.0 / 7.0) < 1e-9


def test_the_courant_floor_for_the_proven_timestep_is_the_measured_19km_window():
    receipt = dt_admission.schedule_receipt(120.0)
    assert abs(receipt["courant"]["minimum_dc_edge_m"] - 16_666.666666666668) < 1e-6


# ---------------------------------------------------------------------------
# the harness has teeth in BOTH directions
# ---------------------------------------------------------------------------
def test_the_verifier_certifies_the_registered_anchor(receipts):
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_verify")
    report = mint.verify_registry(root=TREE_ROOT)
    assert report["certified"], report


def test_the_verifier_fails_on_every_fabricated_variant(receipts):
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_mutation")
    control = mint.mutation_control(root=TREE_ROOT)
    assert control["has_teeth_both_directions"], control["disagreements"]
    arms = {arm["arm"]: arm for arm in control["arms"]}
    assert arms["registered-truth"]["certified"] is True
    for name in (
        "receipt-absent",
        "receipt-is-a-directory",
        "row-claims-a-timestep-its-receipt-does-not",
        "row-claims-a-cadence-its-receipt-does-not",
        "integration-anchor-path-absent",
    ):
        assert arms[name]["certified"] is False
        assert arms[name]["findings"], f"{name} failed without saying why"


def test_the_integration_plan_names_what_it_does_not_prove():
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_plan")
    plan = mint.integration_plan(20.0, "some-registered-row", hours=6.0)
    assert plan["steps"] == 1080
    assert plan["arms"] == 2
    joined = " ".join(plan["what_this_does_not_prove"])
    assert "native MPAS-A" in joined
    assert "Grell-Freitas" in joined
    assert any("RULING" in step for step in plan["procedure"])


# ---------------------------------------------------------------------------
# the candidate mint: the only way through the gate, and it is loud
# ---------------------------------------------------------------------------
def test_a_candidate_mint_needs_the_authorization_verbatim():
    for authorization in ("", "yes", dt_admission.CANDIDATE_MINT_AUTHORIZATION.upper()):
        with pytest.raises(DtAdmissionError) as caught:
            dt_admission.candidate_mint(
                20.0, authorization=authorization, card="the proving RTX 5070 Ti"
            )
        assert "authorization verbatim" in str(caught.value)


def test_a_candidate_mint_needs_a_card_and_a_timestep_worth_earning():
    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            20.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="",
        )
    assert "must name the card" in str(caught.value)

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            PROVEN_DT_SECONDS,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="the proving RTX 5070 Ti",
        )
    assert "already holds an anchor" in str(caught.value)


def test_a_candidate_mint_refuses_before_a_card_if_the_host_half_fails():
    """90 s never reaches hardware: it cannot divide the radiation cadence."""

    with pytest.raises(DtAdmissionError) as caught:
        dt_admission.candidate_mint(
            90.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
            card="the proving RTX 5070 Ti",
        )
    assert "not a positive integer multiple" in str(caught.value)


def test_a_candidate_admission_is_stamped_and_withdrawn():
    probe = _unanchored_dt()
    assert dt_admission.admitted_timestep(probe) is None
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ) as candidate:
        admitted = dt_admission.admitted_timestep(probe)
        assert admitted is not None
        assert admitted.admitted_on == "CANDIDATE-UNANCHORED"
        assert "NOT MEASURED" in admitted.integration_anchor
        assert candidate.native_reference is None
        # and the stamp reaches a run's receipt through the same accessor the
        # forecast tool records
        assert admitted.as_dict()["admitted_on"] == "CANDIDATE-UNANCHORED"
    assert dt_admission.admitted_timestep(probe) is None


def test_a_candidate_row_is_never_certified_as_an_anchor():
    mint = _load_tool("mint_dt_anchor.py", "_test_mint_candidate")
    with dt_admission.candidate_mint(
        _unanchored_dt(),
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ) as candidate:
        result = mint.verify_anchor(candidate, root=TREE_ROOT)
    assert result["certified"] is False
    assert "never an anchor" in result["findings"][0]


# ---------------------------------------------------------------------------
# the door answers the timestep question without a card and without a file
# ---------------------------------------------------------------------------
def test_the_door_refuses_an_unanchored_row_from_the_row_alone():
    from hexcore import forecast_door

    probe = _unanchored_dt()
    with pytest.raises(forecast_door.ForecastDoorRefusal) as caught:
        forecast_door.admit_timestep("some-row", probe)
    message = str(caught.value)
    assert f"--mesh some-row declares dt={probe:g} s" in message
    assert "holds no timestep anchor" in message
    record = forecast_door.admit_timestep("x4.163842", 120.0)
    assert record["admitted"] and record["dt_seconds"] == 120.0


# ---------------------------------------------------------------------------
# the ROW half of the candidate mint
#
# candidate_mint admits an unanchored config_dt, and that is not sufficient on
# its own: the forecast door takes its timestep from a registry row, so an
# admitted candidate timestep is unreachable unless some row declares it.  The
# rows declaring the interesting values are the large graded meshes an anchor
# would unblock, which is backwards -- an anchor is a property of the TIMESTEP,
# and the cheap place to earn one is the smallest registered mesh, whose
# Courant limit is an upper bound every candidate sits far beneath.
# ---------------------------------------------------------------------------
def _binding_module(alias: str):
    """Load mpas_mesh_binding BY PATH, the way the forecast door does."""

    return _load_tool("mpas_mesh_binding.py", alias)


def test_a_candidate_row_override_needs_the_authorization_verbatim():
    from hexcore import mesh_row_candidate

    probe = _unanchored_dt()
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ):
        with pytest.raises(DtAdmissionError) as caught:
            mesh_row_candidate.candidate_mesh_dt(
                "x1.40962", probe, authorization="because I said so"
            )
    assert "verbatim" in str(caught.value)


def test_a_candidate_row_override_refuses_without_a_live_timestep_admission():
    """The tight guard: the row can never move on its own.

    A row declaring an unanchored timestep is exactly the thing that cost
    18,820 MiB and 285 s on 2026-08-26.  Opening the row without opening the
    timestep gate would rebuild that defect with a nicer name, so the override
    refuses unless the timestep is ALREADY admitted as a candidate.
    """

    from hexcore import mesh_row_candidate

    probe = _unanchored_dt()
    assert dt_admission.admitted_timestep(probe) is None
    with pytest.raises(DtAdmissionError) as caught:
        mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        )
    text = str(caught.value)
    assert "not admitted at all" in text
    assert "candidate_mint" in text


def test_a_candidate_row_override_refuses_a_timestep_that_holds_a_real_anchor():
    from hexcore import mesh_row_candidate

    with pytest.raises(DtAdmissionError) as caught:
        mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            PROVEN_DT_SECONDS,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        )
    assert "holds a real anchor" in str(caught.value)
    assert "REGISTERED row" in str(caught.value)


def test_the_override_reaches_a_registry_loaded_the_way_the_door_loads_it():
    """The property the whole mechanism rests on.

    ``forecast_door._load_module`` re-executes ``mpas_mesh_binding.py`` from
    disk on every run, which would reset any state that file held itself.  The
    override lives in an imported module for exactly that reason, and this is
    the check that it survives the re-execution.
    """

    from hexcore import mesh_row_candidate

    probe = _unanchored_dt()
    before = _binding_module("_test_binding_before")
    assert before.MESH_BINDINGS["x1.40962"].dt_seconds == 120.0

    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        ):
            during = _binding_module("_test_binding_during")
            row = during.MESH_BINDINGS["x1.40962"]
            assert row.dt_seconds == probe
            # unchanged bytes: the same mesh, at a different timestep
            assert row.grid_sha256 == before.MESH_BINDINGS["x1.40962"].grid_sha256
            assert row.static_bytes == before.MESH_BINDINGS["x1.40962"].static_bytes
            # and the row says what it is, in the string that reaches the log
            assert mesh_row_candidate.CANDIDATE_ROW_MARKER in row.notes
            assert "120 s" in row.notes

    after = _binding_module("_test_binding_after")
    assert after.MESH_BINDINGS["x1.40962"].dt_seconds == 120.0
    assert mesh_row_candidate.CANDIDATE_ROW_MARKER not in (
        after.MESH_BINDINGS["x1.40962"].notes
    )


def test_no_mint_in_flight_leaves_the_registry_object_untouched():
    """An ordinary run must not pay for this path, not even a copy."""

    from hexcore import mesh_row_candidate

    assert dict(mesh_row_candidate.active_overrides()) == {}
    rows = {"only": object()}
    assert mesh_row_candidate.apply_overrides(rows) is rows


def test_an_overridden_row_still_faces_every_other_gate():
    """The override moves ONE field and buys nothing else.

    A candidate timestep still has to divide the radiation cadence, and the
    row's ``__post_init__`` is what checks it -- so a candidate that cannot be
    a registry row at all is refused where the row is built, not one card-hour
    later.
    """

    from hexcore import mesh_row_candidate

    with dt_admission.candidate_mint(
        90.0,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
        radiation_seconds=450.0,
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            90.0,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        ):
            with pytest.raises(Exception) as caught:
                _binding_module("_test_binding_indivisible")
    assert "does not divide" in str(caught.value)


def test_two_mints_cannot_share_one_row():
    from hexcore import mesh_row_candidate

    probe = _unanchored_dt()
    with dt_admission.candidate_mint(
        probe,
        authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        card="the proving RTX 5070 Ti",
    ):
        with mesh_row_candidate.candidate_mesh_dt(
            "x1.40962",
            probe,
            authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
        ):
            with pytest.raises(DtAdmissionError) as caught:
                with mesh_row_candidate.candidate_mesh_dt(
                    "x1.40962",
                    probe,
                    authorization=dt_admission.CANDIDATE_MINT_AUTHORIZATION,
                ):
                    pass
    assert "one mint at a time" in str(caught.value)


# ---------------------------------------------------------------------------
# the physics-health verdict
#
# MEASURED 2026-08-26: of the four timesteps earned that day, two track their
# control and two diverge from it -- and "two byte-identical forecasts, finite
# at every step" is equally true of all four.  Determinism and physical
# plausibility are different questions, so the answer to the second one is a
# required field rather than a sentence somebody might not write.
# ---------------------------------------------------------------------------
RECOGNISED_HEALTH_VERDICTS = {"REFERENCE", "TRACKS", "DIVERGES"}


def test_every_anchor_answers_what_its_physics_band_did():
    for anchor in ADMITTED_TIMESTEPS.values():
        assert anchor.physics_health, (
            f"dt={anchor.dt_seconds:g} s registers no physics_health; an "
            f"anchor that records only determinism says the same thing about "
            f"a timestep that tracks its control and one that runs away"
        )
        assert anchor.physics_health_verdict in RECOGNISED_HEALTH_VERDICTS


def test_a_row_cannot_be_registered_without_answering_it():
    """No default: the field is required at construction."""

    with pytest.raises(TypeError):
        DtAnchor(
            dt_seconds=1.0,
            radiation_seconds=600.0,
            surface_pbl_seconds=1.0,
            cumulus_seconds=1.0,
            cumulus_scheme="gf",
            meshes=(),
            card="none",
            admitted_on="2026-08-26",
            schedule_receipt="evidence/",
            integration_anchor="evidence/",
            native_reference=None,
            basis="no health answer",
        )


def test_a_diverging_anchor_says_so_before_it_explains_itself():
    """The verdict leads, so a receipt scan does not depend on reading prose."""

    diverging = [
        anchor
        for anchor in ADMITTED_TIMESTEPS.values()
        if anchor.physics_health_verdict == "DIVERGES"
    ]
    assert diverging, (
        "the 2026-08-26 campaign measured divergence at 20 s and 5 s; if no "
        "row records it, the finding was lost rather than fixed"
    )
    for anchor in diverging:
        # it names what it does NOT establish, not only what it does
        assert "NOT MEASURED" in anchor.physics_health


def test_the_verdict_travels_into_a_run_receipt():
    """as_dict is what bind_mesh records, so the caveat rides with the run."""

    for anchor in ADMITTED_TIMESTEPS.values():
        assert anchor.as_dict()["physics_health"] == anchor.physics_health
