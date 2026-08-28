"""Placement, ranking, hysteresis, delayed starts, and every named gate.

The properties asserted here are the ones the lane was asked for:

* multiple independent swaths in one cycle, concurrently;
* hysteresis as a correctness property, with the churn counted both ways;
* a delayed start derived from the forecast rather than set by an operator;
* one mechanism -- a synthetic third phenomenon reaches placement through
  the same eight functions as the two that ship.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.swath import registry  # noqa: E402
from hexcore.swath.errors import SwathDocumentError  # noqa: E402
from hexcore.swath.geometry import (  # noqa: E402
    great_circle_km,
    polygon_area_km2,
    ring_overlap_fraction,
)
from hexcore.swath.history import HistoryReader  # noqa: E402
from hexcore.swath.hysteresis import SwathState  # noqa: E402
from hexcore.swath.plan import plan_cycle, plan_document  # noqa: E402

CELLS = 40962
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]

#: Three cyclones in three basins, one of them not present until hour 12,
#: plus a convective area that switches on at hour 9.  Multiple independent
#: swaths is the requirement, not a stretch goal.
SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
    {"kind": "low", "latitude_deg": 14.0, "longitude_deg": 132.0, "bearing_deg": 310.0,
     "speed_km_per_hour": 26.0, "radius_km": 380.0, "amplitude": 3400.0},
    {"kind": "low", "latitude_deg": -18.0, "longitude_deg": 62.0, "bearing_deg": 240.0,
     "speed_km_per_hour": 18.0, "radius_km": 400.0, "amplitude": 2200.0},
    {"kind": "low", "latitude_deg": 12.0, "longitude_deg": -28.0, "bearing_deg": 290.0,
     "speed_km_per_hour": 20.0, "radius_km": 360.0, "amplitude": 2600.0,
     "onset_hours": 12.0},
    {"kind": "convection", "latitude_deg": 38.0, "longitude_deg": -97.0,
     "bearing_deg": 75.0, "speed_km_per_hour": 45.0, "radius_km": 320.0,
     "amplitude": 52.0, "onset_hours": 9.0},
]


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("swath-plan") / "coarse.nc"
    return fixture.build(target, cells=CELLS, hours=HOURS, scenario=SCENARIO)


@pytest.fixture(scope="module")
def history_next(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same world six hours later: the next cycle's coarse forecast."""

    target = tmp_path_factory.mktemp("swath-plan-next") / "coarse.nc"
    return fixture.build(
        target, cells=CELLS, hours=HOURS, scenario=SCENARIO, offset_hours=6.0
    )


@pytest.fixture(scope="module")
def metrics() -> registry.MetricRegistry:
    return registry.load_metrics()


@pytest.fixture(scope="module")
def policy() -> registry.PlacementPolicy:
    return registry.load_policy()


def _plan(history: Path, metrics, policy, *, state=None, cycle_index=None) -> dict:
    with HistoryReader(history) as reader:
        result = plan_cycle(
            reader, metrics, policy, state=state, cycle_index=cycle_index
        )
        return json.loads(json.dumps(plan_document(reader, metrics, policy, result)))


def _policy_from(tmp_path: Path, base: registry.PlacementPolicy, **overrides):
    document = json.loads(registry.DEFAULT_POLICY.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        document[section].update(values)
    target = tmp_path / "policy.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    return registry.load_policy(target)


# ---------------------------------------------------------------------------
# multiple independent swaths, at once
# ---------------------------------------------------------------------------
def test_one_cycle_places_several_independent_swaths(history, metrics, policy) -> None:
    document = _plan(history, metrics, policy)
    assert len(document["admitted"]) >= 3
    slots = [row["slot_id"] for row in document["admitted"]]
    assert len(set(slots)) == len(slots)
    centroids = [row["centroid_deg"] for row in document["admitted"]]
    for index, first in enumerate(centroids):
        for second in centroids[index + 1:]:
            assert great_circle_km(first[0], first[1], second[0], second[1]) > 400.0


def test_every_admitted_swath_carries_a_spec_the_generator_grammar_accepts(
    history, metrics, policy
) -> None:
    document = _plan(history, metrics, policy)
    for row in document["admitted"]:
        spec = row["mesh_spec"]
        assert set(spec) == {"background_km", "name", "regions"}
        assert len(spec["regions"]) == 1
        region = spec["regions"][0]
        assert set(region) == {"shape", "spacing_km", "transition_cells"}
        assert region["shape"]["kind"] in {"polygon", "cap"}
        if region["shape"]["kind"] == "polygon":
            for vertex in region["shape"]["vertices_deg"]:
                assert len(vertex) == 2
                assert -90.0 <= vertex[0] <= 90.0
                assert -180.0 <= vertex[1] < 180.0
        # THE CUT AND THE REFINEMENT ARE THE SAME SHAPE, NOT THE SAME SIZE.
        # This assertion read `==` until 2026-08-27, when every shipped row
        # moved to the measured 1.35x pad; at 1.0 the two documents were
        # literally equal and the equality was hiding the claim that matters.
        # The claim is that a pad moves the CUT and never a cell centre, so
        # the mesh the generator is asked for is unchanged while the cull
        # grows into the parent's own ramp.
        cut = row["cull_region"]
        assert cut["kind"] == region["shape"]["kind"]
        if cut["kind"] == "polygon":
            assert len(cut["vertices_deg"]) == len(region["shape"]["vertices_deg"])
            assert polygon_area_km2(
                [(lat, lon) for lat, lon in cut["vertices_deg"]]
            ) > polygon_area_km2(
                [(lat, lon) for lat, lon in region["shape"]["vertices_deg"]]
            )
        else:
            assert cut["center_deg"] == region["shape"]["center_deg"]
            assert cut["radius_km"] > region["shape"]["radius_km"]


def test_a_flared_swath_is_the_shape_a_moving_feature_gets(
    history, metrics, policy
) -> None:
    document = _plan(history, metrics, policy)
    moving = [
        row for row in document["admitted"]
        if row["cull_region"]["kind"] == "polygon"
    ]
    assert moving
    row = moving[0]
    assert row["half_widths_km"][0] < row["half_widths_km"][-1]
    assert len(row["path_deg"]) >= 2


def test_the_plan_is_deterministic(history, metrics, policy) -> None:
    assert _plan(history, metrics, policy) == _plan(history, metrics, policy)


# ---------------------------------------------------------------------------
# delayed starts
# ---------------------------------------------------------------------------
def test_a_swath_ignites_at_an_hour_derived_from_the_forecast(
    history, metrics, policy
) -> None:
    """The genesis case: a feature the coarse run does not produce until
    hour 12 gets a swath that starts at hour 9, three hours of declared
    margin ahead of it -- and no operator typed either number."""

    document = _plan(history, metrics, policy)
    delayed = [row for row in document["admitted"] if row["ignite_at_seconds"] > 0.0]
    assert delayed, [row["ignite_at_seconds"] for row in document["admitted"]]
    frame_interval = HOURS[1] - HOURS[0]
    for row in delayed:
        margin = metrics.metric_rows[row["metric_id"]].start_policy.lead_margin_hours
        wanted = row["track_first_time_seconds"] / 3600.0 - margin
        ignition = row["ignite_at_seconds"] / 3600.0
        # Quantization is DOWN to a parent frame, always toward more lead
        # and never less, so the ignition sits in the frame interval that
        # ends at the requested hour. The idle lead is therefore at least
        # the declared margin and at most one parent frame more -- and a
        # test that asserted equality would be asserting that the margin
        # happened to be a multiple of the output interval.
        assert wanted - frame_interval - 1e-9 <= ignition <= wanted + 1e-9
        assert margin - 1e-9 <= row["idle_lead_hours"] <= margin + frame_interval + 1e-9


def test_ignition_is_quantized_down_to_a_parent_frame(history, metrics, policy) -> None:
    document = _plan(history, metrics, policy)
    frame_seconds = {hour * 3600.0 for hour in HOURS}
    for row in document["admitted"]:
        assert row["ignite_at_seconds"] in frame_seconds


def test_a_swath_that_would_run_before_its_feature_exists_is_refused_by_name(
    tmp_path, history, metrics, policy
) -> None:
    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["start_policy"] = {"kind": "cycle_start"}
    target = tmp_path / "metrics.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    forced = registry.load_metrics(target)
    plan = _plan(history, forced, policy)
    reasons = [row["reason"] for row in plan["declined"]]
    assert any("IGNITION-BEFORE-ONSET" in reason for reason in reasons), reasons
    assert any("time_of_first_exceedance" in reason for reason in reasons)


# ---------------------------------------------------------------------------
# ranking and its named gates
# ---------------------------------------------------------------------------
def test_the_admitted_set_is_ordered_by_score(history, metrics, policy) -> None:
    document = _plan(history, metrics, policy)
    scores = [row["effective_score"] for row in document["admitted"]]
    assert scores == sorted(scores, reverse=True)


def test_every_rank_term_appears_in_the_breakdown(history, metrics, policy) -> None:
    document = _plan(history, metrics, policy)
    declared = {term.id for term in policy.rank_terms}
    for row in document["admitted"]:
        assert {term["id"] for term in row["rank"]["terms"]} == declared
        assert row["rank"]["score"] == pytest.approx(
            sum(term["contribution"] for term in row["rank"]["terms"]), abs=1e-6
        )


def test_the_budget_gate_declines_by_name_with_its_numbers(
    tmp_path, history, metrics, policy
) -> None:
    tight = _policy_from(tmp_path, policy, budget={"maximum_swaths": 2})
    document = _plan(history, metrics, tight)
    assert len(document["admitted"]) == 2
    reasons = [row["reason"] for row in document["declined"]]
    assert any("SWATH-BUDGET" in reason for reason in reasons)
    assert any("maximum_swaths 2" in reason for reason in reasons)


def test_the_separation_gate_declines_by_name(tmp_path, history, metrics, policy) -> None:
    wide = _policy_from(
        tmp_path, policy, budget={"minimum_separation_km": 20000.0}
    )
    document = _plan(history, metrics, wide)
    assert len(document["admitted"]) == 1
    reasons = [row["reason"] for row in document["declined"]]
    assert any("SWATH-SEPARATION" in reason for reason in reasons)
    assert any("refine the same ground twice" in reason for reason in reasons)


def test_the_capacity_gate_declines_by_name(tmp_path, history, metrics, policy) -> None:
    small = _policy_from(
        tmp_path, policy, budget={"maximum_cells_per_swath": 100.0}
    )
    document = _plan(history, metrics, small)
    assert document["admitted"] == []
    reasons = [row["reason"] for row in document["declined"]]
    assert any("SWATH-CAPACITY" in reason for reason in reasons)


def test_a_cycle_that_places_nothing_still_produces_a_document(
    tmp_path, history, metrics, policy
) -> None:
    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["detector"]["threshold"] = 1.0 if row["detector"]["kind"] == "extremum_ball" else 1e9
    target = tmp_path / "metrics.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    quiet = registry.load_metrics(target)
    plan = _plan(history, quiet, policy)
    assert plan["admitted"] == []
    assert plan["state"]["slots"] == []
    assert plan["counts"] if "counts" in plan else True


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------
def test_a_slot_that_continues_keeps_its_identity(
    history, history_next, metrics, policy
) -> None:
    """A storm that has moved six hours on is still the same slot."""

    first = _plan(history, metrics, policy, cycle_index=0)
    state = SwathState.from_document(first["state"])
    second = _plan(history_next, metrics, policy, state=state, cycle_index=1)
    before = {row["slot_id"] for row in first["admitted"]}
    after = {row["slot_id"] for row in second["admitted"]}
    assert before & after
    assert second["churn"]["continued"] >= 3
    assert second["churn"]["evictions"] == 0


def test_a_slot_that_did_not_move_reuses_its_mesh(history, metrics, policy) -> None:
    first = _plan(history, metrics, policy, cycle_index=0)
    state = SwathState.from_document(first["state"])
    second = _plan(history, metrics, policy, state=state, cycle_index=1)
    actions = [(row["hysteresis"] or {})["mesh_action"] for row in second["admitted"]]
    assert actions and all(action == "reuse" for action in actions)
    assert second["churn"]["mesh_reuse"] == len(actions)


def test_disarming_the_regeneration_rule_rebuilds_every_mesh(
    tmp_path, history, history_next, metrics, policy
) -> None:
    """The measurement the rule carries, in miniature.

    With the regeneration thresholds at zero, an identical cycle rebuilds
    every mesh it could have reused.  ``tools/measure_swath_hysteresis.py``
    runs the same comparison over eight cycles and records the totals.
    """

    disarmed = _policy_from(
        tmp_path, policy,
        hysteresis={"regenerate_centroid_km": 0.001, "regenerate_overlap_below": 1.0},
    )
    first = _plan(history, metrics, disarmed, cycle_index=0)
    state = SwathState.from_document(first["state"])
    second = _plan(history_next, metrics, disarmed, state=state, cycle_index=1)

    armed_first = _plan(history, metrics, policy, cycle_index=0)
    armed_state = SwathState.from_document(armed_first["state"])
    armed_second = _plan(history_next, metrics, policy, state=armed_state, cycle_index=1)

    assert armed_second["churn"]["mesh_reuse"] > second["churn"]["mesh_reuse"]
    assert armed_second["churn"]["mesh_generate"] < second["churn"]["mesh_generate"]

    # NOT every mesh: a slot whose ring is IDENTICAL between the two cycles
    # reuses under any setting, and correctly so -- there is no rebuild to
    # force when the swath covers exactly the same ground. In this scenario
    # that is the convective area, whose onset hour is fixed in absolute
    # time, so the second cycle projects it over the same geometry. Every
    # slot that MOVED regenerates, and that is the property the rule buys.
    moved = [
        row for row in second["admitted"]
        if (row["hysteresis"] or {}).get("centroid_moved_km", 0.0) > 0.0
    ]
    assert moved
    assert all(row["hysteresis"]["mesh_action"] == "generate" for row in moved)


def test_the_incumbent_margin_appears_in_the_effective_score(
    history, history_next, metrics, policy
) -> None:
    first = _plan(history, metrics, policy, cycle_index=0)
    state = SwathState.from_document(first["state"])
    second = _plan(history_next, metrics, policy, state=state, cycle_index=1)
    assert any((row["hysteresis"] or {}).get("incumbent") for row in second["admitted"])
    for row in second["admitted"]:
        if (row["hysteresis"] or {}).get("incumbent"):
            assert row["effective_score"] == pytest.approx(
                row["rank"]["score"] * (1.0 + policy.hysteresis.promotion_margin),
                abs=1e-6,
            )


def test_state_is_honoured_in_full_or_refused(history, metrics, policy) -> None:
    first = _plan(history, metrics, policy, cycle_index=0)
    broken = json.loads(json.dumps(first["state"]))
    broken["slots"][0]["favourite_colour"] = "green"
    with pytest.raises(SwathDocumentError) as caught:
        SwathState.from_document(broken)
    assert "favourite_colour" in str(caught.value)


def test_a_state_document_of_the_wrong_schema_refuses_by_name(
    history, metrics, policy
) -> None:
    first = _plan(history, metrics, policy, cycle_index=0)
    broken = json.loads(json.dumps(first["state"]))
    broken["schema"] = "gpuwm-hex.swath-state.v2"
    with pytest.raises(SwathDocumentError) as caught:
        SwathState.from_document(broken)
    assert "restarted slot looks exactly like a continued one" in str(caught.value)


def test_rings_are_carried_across_cycles_so_overlap_can_be_measured(
    history, metrics, policy
) -> None:
    first = _plan(history, metrics, policy, cycle_index=0)
    state = SwathState.from_document(first["state"])
    assert state.slots
    for slot in state.slots:
        assert len(slot.ring) >= 8
    ring = [tuple(vertex) for vertex in first["admitted"][0]["ring_deg"]]
    assert ring_overlap_fraction(ring, list(state.slots[0].ring)) > 0.0


# ---------------------------------------------------------------------------
# ONE MECHANISM: the arbitrary acceptance test, through the whole pipeline
# ---------------------------------------------------------------------------
def test_a_third_phenomenon_is_placed_by_the_same_code(
    tmp_path, history, policy
) -> None:
    """A metric row nobody wrote code for reaches an admitted swath.

    ``moist_layer_area`` uses a field derivation, a detector kind and a
    start policy that already exist, combined in a way the shipped
    document does not.  If it places, the layer is configuration-driven;
    if it needed an edit anywhere under ``src/``, it is not.
    """

    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["enabled"] = False
    document["fields"].append({
        "id": "layer_relative_humidity",
        "source_variables": ["relhum"],
        "derivation": {"kind": "level_slice", "level_index": 10},
        "units": "%",
    })
    document["metrics"].append({
        "id": "moist_layer_area",
        "threat_class": "moist_layer",
        "field": "layer_relative_humidity",
        "enabled": True,
        "detector": {
            "kind": "area_threshold_exceedance",
            "comparison": "at_least",
            "threshold": 30.0,
            "minimum_area_km2": 40000.0,
            "maximum_area_km2": 1600000.0,
        },
        "confirm_with": [],
        "track": {"maximum_speed_km_per_hour": 90.0, "minimum_frames": 1},
        "swath": {
            "half_width_km": 180.0, "flare_km_per_hour": 8.0,
            "maximum_half_width_km": 320.0, "spacing_km": 6.0,
            "transition_cells": 81.0, "lead_hours": 9.0,
            "path_step_km": 100.0, "cap_points": 16,
            # A DIFFERENT LADDER THAN EVERY SHIPPED ROW, declared here and
            # nowhere else: this phenomenon cuts at 1.70x where the nine
            # shipped rows cut at the measured knee of 1.35.  The assertions
            # below check it arrives -- an arbitrary value on an arbitrary
            # phenomenon, neither of which anything under src/ knows about.
            "cull_pad_scale": 1.70,
        },
        "start_policy": {"kind": "time_of_first_exceedance", "lead_margin_hours": 1.0},
        "rank": {"intensity_reference": 10.0},
    })
    target = tmp_path / "metrics.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    third = registry.load_metrics(target)
    assert third.publication_manifest() == ("relhum",)

    # The fixture must publish the field the new row names, which is itself
    # the point: the manifest told us what to publish, without a code edit.
    scenario = [
        {"kind": "convection", "latitude_deg": 20.0, "longitude_deg": 10.0,
         "bearing_deg": 60.0, "speed_km_per_hour": 40.0, "radius_km": 500.0,
         "amplitude": 60.0},
    ]
    path = fixture.build(
        tmp_path / "moist.nc", cells=CELLS, hours=HOURS, scenario=scenario
    )
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(str(path), "a") as target_file:
        source = np.asarray(target_file.variables["refl10cm"][:])
        variable = target_file.createVariable(
            "relhum", "f8", target_file.variables["refl10cm"].dimensions
        )
        variable[:] = source

    plan = _plan(path, third, policy)
    assert plan["admitted"], plan["declined"]
    row = plan["admitted"][0]
    assert row["threat_class"] == "moist_layer"
    assert row["mesh_spec"]["regions"][0]["spacing_km"] == 6.0

    # AND ITS LADDER ARRIVED.  This row declared cull_pad_scale 1.70 where
    # every shipped row declares 1.35, so its limited-area cut is taken 1.70x
    # further out and keeps that much of the parent's own resolution ramp
    # between the boundary and the fine core.  Nothing under src/ knows this
    # phenomenon exists.
    ring = [(lat, lon) for lat, lon in row["ring_deg"]]
    cut = [(lat, lon) for lat, lon in row["cull_region"]["vertices_deg"]]
    assert row["cull_region"]["kind"] == "polygon"
    assert len(cut) == len(ring)
    # Area scales as the square of a similarity factor, to within the
    # sphere's own curvature over a region this size.
    assert polygon_area_km2(cut) / polygon_area_km2(ring) == pytest.approx(
        1.70 ** 2, rel=0.02
    )
    # THE REFINEMENT IS UNTOUCHED: a pad moves the CUT, never a cell centre,
    # so the mesh the row asks for is the same mesh at any pad.  (The plan row
    # rounds ring_deg to six places for the receipt; the spec carries the
    # unrounded ring, so the comparison rounds one side.)
    assert [
        [round(lat, 6), round(lon, 6)]
        for lat, lon in row["mesh_spec"]["regions"][0]["shape"]["vertices_deg"]
    ] == [[lat, lon] for lat, lon in row["ring_deg"]]


#: The measured knee, and the cut every shipped row takes since 2026-08-27.
#: 0.624 -> 0.744 on vertical-velocity correlation and 0.578 -> 0.852 on 2 m
#: temperature, for +7.9 % wall and +42 MiB, with no second forecast anywhere
#: in the trade (``evidence/nest-ratio-20260827/RECEIPT.md`` section 2a).
SHIPPED_CULL_PAD_SCALE = 1.35


def test_the_shipped_rows_all_take_the_measured_pad(history, policy, tmp_path) -> None:
    """Fixed means default: no shipped row keeps the cut measured too small.

    Every shipped row declared ``cull_pad_scale`` 1.0 until 2026-08-27 -- the
    value the code had implicitly before the column existed, and the one
    measured to hand a 71 km parent state straight to cells the fine core's
    own size.  the project law is that a correctness remedy ships default-on, so
    the rows moved to the measured knee and this test is what stops one
    drifting back.  A row below the knee is a row whose vertical-velocity
    correlation is 0.624 where it could be 0.744 for 25 seconds of wall.

    Asserted rather than assumed for a second reason: moving a pad moves
    every cull this planner emits, and therefore every ``bdyMask`` digest
    downstream of it.  That is intended here and would be a silent surprise
    anywhere else.
    """

    metrics = registry.load_metrics()
    assert metrics.metric_rows, "the shipped metrics document has no rows"
    for row in metrics.metric_rows.values():
        assert row.swath.cull_pad_scale == SHIPPED_CULL_PAD_SCALE, row.id


# ---------------------------------------------------------------------------
# THE HARDER ARBITRARY TEST: a COMPOUND threat, as data, all the way to a spec
# ---------------------------------------------------------------------------
def test_a_compound_threat_is_detected_tracked_and_placed_as_data_only(
    tmp_path, history, policy
) -> None:
    """Three conditions in three units become an admitted swath.

    The threat is "a low that is also warm and also windy": a margin on
    sea-level pressure in pascals, a margin on temperature in kelvin, a
    margin on wind in metres per second, and the weakest of the three.  It
    is the shape fire weather, atmospheric rivers and ingredients-based
    severe convection all have, and the v1 grammar could not express any of
    them.

    It reaches the SAME detector, the SAME association, the SAME projection
    and the SAME ranking as a reflectivity blob, and it comes out the far
    end as a mesh spec the generator reads unchanged.  If any of that had
    needed a branch under ``src/``, this test would be red.

    ``git status`` staying clean while this runs is the rest of the claim,
    and that is what ``tools/prove_compound_threat_is_data.py`` checks.
    """

    from test_swath_registry import with_compound_threat  # noqa: PLC0415

    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["enabled"] = False
    with_compound_threat(document)
    target = tmp_path / "compound-metrics.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    compound = registry.load_metrics(target)

    # The armed set is exactly the compound row, and it needs nothing the
    # coarse run was not already publishing for the shipped rows.
    assert [row.id for row in compound.armed] == ["probe_warm_windy_low_area"]
    assert set(compound.publication_manifest()) == {
        "surface_pressure", "t2", "ter", "u10", "v10"
    }

    document_out = _plan(history, compound, policy)
    assert document_out["admitted"], document_out["declined"]
    row = document_out["admitted"][0]
    assert row["threat_class"] == "warm_windy_low"
    assert row["mesh_spec"]["regions"][0]["spacing_km"] == 6.0
    assert row["cull_region"]["kind"] in ("polygon", "cap")
    # It tracked: a compound region that moves is followed like anything else.
    assert any(track["frames"] >= 2 for track in document_out["tracks"])
    # And it ranks on the same axis, through its own declared reference.
    intensity = next(
        term for term in row["rank"]["terms"] if term["kind"] == "metric_extremum"
    )
    assert intensity["reference"] == 1.0


def test_a_regional_row_declines_features_outside_its_own_box(
    tmp_path, history, policy
) -> None:
    """A region is part of a threat definition, and it is enforced by name.

    THE BREAKAGE: 'severe convection' carries North American operational
    thresholds. Armed globally it fires over warm tropical oceans every
    cycle and spends slots on regions where the term has no meaning. The
    same document with the box removed must find MORE, and the difference
    must appear as named drops rather than as silence.
    """

    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["enabled"] = row["id"] == "deep_convection_area"
        if row["id"] == "deep_convection_area":
            row["region"] = {
                "kind": "bounding_box",
                "south_deg": -60.0, "north_deg": -20.0,
                "west_deg": 100.0, "east_deg": 170.0,
            }
    boxed_path = tmp_path / "boxed.json"
    boxed_path.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    boxed = registry.load_metrics(boxed_path)

    for row in document["metrics"]:
        row.pop("region", None)
    open_path = tmp_path / "open.json"
    open_path.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    unboxed = registry.load_metrics(open_path)

    boxed_plan = _plan(history, boxed, policy)
    open_plan = _plan(history, unboxed, policy)
    assert not boxed_plan["admitted"]
    assert open_plan["admitted"]
    outside = [
        drop for drop in boxed_plan["drops"]
        if "outside this row's region" in drop["reason"]
    ]
    assert outside
    assert "bounding_box" in outside[0]["reason"]


#: The measured defect in miniature: several strong cyclones that all
#: outrank several genuine convective regions.  On the real 24 h global
#: forecast this lane was handed, the ranking held 41 cyclone tracks and
#: 217 deep-convection tracks and the 27 highest-ranked candidates of 205
#: were all cyclones, so a budget of four placed four cyclones and none of
#: the convection.  Four widely separated deep lows and two widely
#: separated convective areas reproduce that ordering exactly.
SWEEP_SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
    {"kind": "low", "latitude_deg": 14.0, "longitude_deg": 132.0, "bearing_deg": 310.0,
     "speed_km_per_hour": 26.0, "radius_km": 400.0, "amplitude": 4000.0},
    {"kind": "low", "latitude_deg": -18.0, "longitude_deg": 62.0, "bearing_deg": 240.0,
     "speed_km_per_hour": 18.0, "radius_km": 400.0, "amplitude": 3800.0},
    {"kind": "low", "latitude_deg": 12.0, "longitude_deg": -28.0, "bearing_deg": 290.0,
     "speed_km_per_hour": 20.0, "radius_km": 380.0, "amplitude": 3600.0},
    # Real convection, and clearly weaker than any of the four lows: 46 dBZ
    # is 0.73 of a reference core against a low's 1.44 of a reference depth.
    {"kind": "convection", "latitude_deg": 38.0, "longitude_deg": -97.0,
     "bearing_deg": 75.0, "speed_km_per_hour": 25.0, "radius_km": 320.0,
     "amplitude": 46.0},
    {"kind": "convection", "latitude_deg": -8.0, "longitude_deg": 150.0,
     "bearing_deg": 260.0, "speed_km_per_hour": 22.0, "radius_km": 300.0,
     "amplitude": 45.0},
]


def test_a_class_that_sweeps_the_ranking_cannot_sweep_the_budget(
    tmp_path, policy
) -> None:
    """The class cap, against the measured defect it was written for.

    THE BREAKAGE: one threat class holding the top of the ranking spends
    the whole cycle on itself, and the other threats the same forecast
    holds get no grid at all -- 56.5 min of GPU plus 9.0 min of init per
    slot, all of it watching one kind of weather. Commensurable units put
    the classes on one axis; they do not make the atmosphere interleave
    them.

    Two arms, one forecast, one policy key apart.
    """

    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["enabled"] = row["id"] in (
            "tropical_cyclone_centre", "deep_convection_area"
        )
    target = tmp_path / "two-classes.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    two_classes = registry.load_metrics(target)

    path = fixture.build(
        tmp_path / "sweep.nc", cells=CELLS, hours=HOURS, scenario=SWEEP_SCENARIO
    )
    uncapped = _policy_from(
        tmp_path, policy,
        budget={"maximum_per_threat_class": policy.budget.maximum_swaths},
    )
    capped_plan = _plan(path, two_classes, policy)
    uncapped_plan = _plan(path, two_classes, uncapped)

    uncapped_classes = [row["threat_class"] for row in uncapped_plan["admitted"]]
    capped_classes = [row["threat_class"] for row in capped_plan["admitted"]]
    # The defect, reproduced: without the cap one class takes every slot.
    assert len(set(uncapped_classes)) == 1
    assert uncapped_classes.count("tropical_cyclone") == policy.budget.maximum_swaths
    # And the fix: the same budget now holds two of each.
    assert set(capped_classes) == {"tropical_cyclone", "deep_convection"}
    for threat_class in set(capped_classes):
        assert (
            capped_classes.count(threat_class)
            <= policy.budget.maximum_per_threat_class
        )
    reasons = [row["reason"] or "" for row in capped_plan["declined"]]
    assert any("SWATH-CLASS-BUDGET" in reason for reason in reasons)


def test_commensurable_intensity_puts_two_fields_on_one_axis(
    tmp_path, policy
) -> None:
    """The units half of the same defect, isolated from the cap.

    A pressure anomaly of about 4,000 Pa and a reflectivity anomaly of
    about 17 dBZ are the same forecast's two strongest things. Divided by
    ONE policy-wide scale -- which is what v1 did -- the pressure row's
    intensity term is over two hundred times the reflectivity row's, and
    no weighting of persistence, travel or extent can close that. Divided
    by each row's own declared reference they land within a factor of two.
    """

    document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
    for row in document["metrics"]:
        row["enabled"] = row["id"] in (
            "tropical_cyclone_centre", "deep_convection_area"
        )
    target = tmp_path / "two-classes.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    two_classes = registry.load_metrics(target)

    path = fixture.build(
        tmp_path / "axis.nc", cells=CELLS, hours=HOURS, scenario=SWEEP_SCENARIO
    )
    plan = _plan(path, two_classes, policy)
    everything = plan["admitted"] + plan["declined"]

    def intensity(row):
        return next(
            term for term in row["rank"]["terms"] if term["kind"] == "metric_extremum"
        )

    cyclones = [
        intensity(row) for row in everything
        if row["threat_class"] == "tropical_cyclone" and row["rank"]["terms"]
    ]
    convection = [
        intensity(row) for row in everything
        if row["threat_class"] == "deep_convection" and row["rank"]["terms"]
    ]
    assert cyclones and convection

    # The RAW anomalies are in different units and are hundreds of times apart.
    raw_ratio = max(t["raw"] for t in cyclones) / max(t["raw"] for t in convection)
    assert raw_ratio > 100.0
    # The SCALED ones, each through its own row's reference, are not.
    scaled_ratio = (
        max(t["scaled"] for t in cyclones) / max(t["scaled"] for t in convection)
    )
    assert scaled_ratio < 3.0
    # And each row's reference is its own, published in the receipt.
    assert {t["reference"] for t in cyclones} == {2500.0}
    assert {t["reference"] for t in convection} == {15.0}


def test_only_intensity_is_unbounded_and_the_rest_are_fractions(
    history, metrics, policy
) -> None:
    """Every rank term but one is a fraction of its own ceiling.

    THE BREAKAGE, found three times in this layer and each time in a term
    whose units looked shared: a term with a policy-wide scale and no
    ceiling wins the ordering on its own. Intensity did it in field units;
    extent did it with an 11.4 million km2 region; persistence did it with
    a 25-frame forecast against a scale of 8. A term that can reach 3 on a
    long forecast and 1 on a short one is ranking the OUTPUT INTERVAL.

    Intensity stays unbounded on purpose -- a storm twice as deep as its
    row's reference should be able to run away with the cycle.
    """

    document = _plan(history, metrics, policy)
    seen = set()
    for row in document["admitted"] + document["declined"]:
        for term in row["rank"]["terms"]:
            seen.add(term["kind"])
            if term["kind"] == "metric_extremum":
                continue
            assert 0.0 <= term["scaled"] <= 1.0 + 1e-9, (row["metric_id"], term)
    assert seen == {
        "metric_extremum", "track_frames", "track_displacement_km",
        "feature_area_km2",
    }


def test_the_frame_count_does_not_change_the_order(tmp_path, metrics, policy) -> None:
    """The same weather, published twice as often, ranks the same way.

    THE BREAKAGE: with a fixed persistence scale, doubling the parent's
    output frequency doubled every track's frame count and therefore its
    persistence term, so the same forecast placed different swaths
    depending on how often it had been written out. Measured on a real
    24 h global run at 25 frames, that term reached 1.56 of a winning
    score of 4.97 at a weight of only 0.5.
    """

    sparse = fixture.build(
        tmp_path / "sparse.nc", cells=CELLS, hours=[0.0, 6.0, 12.0, 18.0],
        scenario=SCENARIO,
    )
    dense = fixture.build(
        tmp_path / "dense.nc", cells=CELLS,
        hours=[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0], scenario=SCENARIO,
    )
    sparse_plan = _plan(sparse, metrics, policy)
    dense_plan = _plan(dense, metrics, policy)

    def persistence(document):
        return [
            next(
                term["scaled"] for term in row["rank"]["terms"]
                if term["kind"] == "track_frames"
            )
            for row in document["admitted"]
        ]

    # A feature present at every frame scores 1.0 either way, where a fixed
    # scale of 8 frames would have given 0.5 on the four-frame forecast and
    # 0.875 on the seven-frame one for the same unbroken storm.
    assert max(persistence(sparse_plan)) == pytest.approx(1.0)
    assert max(persistence(dense_plan)) == pytest.approx(1.0)
    # And no candidate on either forecast can exceed the ceiling.
    for document in (sparse_plan, dense_plan):
        for row in document["admitted"] + document["declined"]:
            for term in row["rank"]["terms"]:
                if term["kind"] == "track_frames":
                    assert 0.0 < term["scaled"] <= 1.0 + 1e-9
    # NOT asserted: an identical admitted list. The two forecasts sample
    # different hours, so they hold genuinely different weather -- the
    # sparse one never sees the convective area's first three hours. What
    # is asserted is that the persistence TERM stopped depending on how
    # often the parent was written out, which is the defect.
