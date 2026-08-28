"""The two shipped documents load, and every closed vocabulary refuses.

The point of this file is the arbitrary acceptance test: adding a
phenomenon must be a ROW.  ``test_a_new_threat_row_needs_no_source_edit``
is that test, executable -- it adds a row to the shipped document in a
temporary copy and asserts the publication manifest and the armed set both
grow with no source file touched.

``test_a_compound_threat_needs_no_source_edit`` is the HARDER form, and it
is the one this schema version exists for.  A single threshold on a single
field was always expressible; a threat whose definition is "all of these
conditions, in units that do not match, hold in the same place" was not,
and that is most of the threats anybody actually chases.  The document it
builds is shared with ``tests/test_swath_plan.py``, which runs it against
a real fixture history and asserts that it detects, tracks and places.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore.swath import registry
from hexcore.swath.errors import SwathDocumentError


@pytest.fixture()
def metrics_document() -> dict:
    return json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))


@pytest.fixture()
def policy_document() -> dict:
    return json.loads(registry.DEFAULT_POLICY.read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, document: dict) -> Path:
    target = tmp_path / name
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    return target


# ---------------------------------------------------------------------------
# the shipped documents
# ---------------------------------------------------------------------------
def test_the_shipped_metrics_document_loads() -> None:
    metrics = registry.load_metrics()
    assert metrics.schema == registry.METRICS_SCHEMA
    assert len(metrics.armed) >= 5
    classes = {row.threat_class for row in metrics.armed}
    # The five named by hand, two of them by name in the brief.
    assert {
        "tropical_cyclone",
        "extratropical_cyclone",
        "deep_convection",
        "severe_convection",
        "fire_weather",
    } <= classes


def test_every_shipped_row_declares_its_own_intensity_reference() -> None:
    """The commensurability fix, asserted on the document that ships.

    Without a per-row reference the intensity term is in the units of
    whatever field the row detected on, and a policy-wide scale adds
    pascals to decibels.  A row that forgot to declare one is refused at
    load, so this test is really asserting that the shipped library got
    the refusal right for all of them, and that no two rows on different
    fields quietly share a number.
    """

    metrics = registry.load_metrics()
    for row in metrics.armed:
        assert row.rank.intensity_reference > 0.0, row.id
    by_field: dict[str, set[float]] = {}
    for row in metrics.armed:
        by_field.setdefault(metrics.field_rows[row.field].units, set()).add(
            row.rank.intensity_reference
        )
    # Rows detecting in pascals and rows detecting in dBZ must not have
    # landed on the same reference by accident.
    assert by_field["Pa"] != by_field["dBZ"]


def test_the_shipped_policy_caps_any_one_class_below_the_whole_budget() -> None:
    """Fixed means default: a bare run cannot let one class sweep the budget."""

    policy = registry.load_policy()
    assert policy.budget.maximum_per_threat_class < policy.budget.maximum_swaths


def test_the_shipped_intensity_term_carries_no_field_units() -> None:
    """The policy's own scale is 1.0 now; the units come off the row."""

    policy = registry.load_policy()
    intensity = next(
        term for term in policy.rank_terms if term.kind == "metric_extremum"
    )
    assert intensity.scale == 1.0


def test_the_shipped_policy_document_loads() -> None:
    policy = registry.load_policy()
    assert policy.schema == registry.POLICY_SCHEMA
    assert policy.budget.maximum_swaths >= 1
    assert policy.hysteresis.promotion_margin > 0.0


def test_both_shipped_documents_carry_their_own_digest() -> None:
    metrics = registry.load_metrics()
    policy = registry.load_policy()
    assert len(metrics.sha256) == 64
    assert len(policy.sha256) == 64
    assert metrics.sha256 != policy.sha256


def test_the_publication_manifest_is_derived_from_the_armed_rows() -> None:
    manifest = registry.load_metrics().publication_manifest()
    assert "surface_pressure" in manifest
    assert "refl10cm" in manifest
    assert "u10" in manifest and "v10" in manifest
    assert manifest == tuple(sorted(manifest))


def test_disabling_a_row_shrinks_the_publication_manifest(
    tmp_path: Path, metrics_document: dict
) -> None:
    """And it shrinks it THROUGH a chain of composed rows.

    ``qv`` is not named by the atmospheric-river metric row: it is named
    by a level slice, which is named by a wind/moisture product, which is
    the field the metric detects on.  Disarming the one row at the top has
    to reach several levels down, or the manifest over-states what a
    coarse run must publish for ever.
    """

    before = registry.load_metrics().publication_manifest()
    for row in metrics_document["metrics"]:
        if row["id"] == "atmospheric_river_corridor":
            row["enabled"] = False
    after = registry.load_metrics(
        _write(tmp_path, "metrics.json", metrics_document)
    ).publication_manifest()
    assert "qv" in before
    assert "qv" not in after
    assert "surface_pressure" in after


# ---------------------------------------------------------------------------
# THE ARBITRARY ACCEPTANCE TEST, executable
# ---------------------------------------------------------------------------
def test_a_new_threat_row_needs_no_source_edit(
    tmp_path: Path, metrics_document: dict
) -> None:
    """A synthetic phenomenon, added as data, reaches the whole layer."""

    metrics_document["fields"].append(
        {
            "id": "layer_relative_humidity",
            "source_variables": ["relhum"],
            "derivation": {"kind": "level_slice", "level_index": 20},
            "units": "%",
        }
    )
    metrics_document["metrics"].append(
        {
            "id": "moist_layer_area",
            "threat_class": "moist_layer",
            "field": "layer_relative_humidity",
            "enabled": True,
            "detector": {
                "kind": "area_threshold_exceedance",
                "comparison": "at_least",
                "threshold": 90.0,
                "minimum_area_km2": 50000.0,
                "maximum_area_km2": 1600000.0,
            },
            "confirm_with": [],
            "track": {"maximum_speed_km_per_hour": 70.0, "minimum_frames": 1},
            "swath": {
                "half_width_km": 180.0,
                "flare_km_per_hour": 8.0,
                "maximum_half_width_km": 320.0,
                "spacing_km": 6.0,
        "cull_pad_scale": 1.0,
            "cull_pad_scale": 1.0,
                "transition_cells": 81.0,
                "lead_hours": 9.0,
                "path_step_km": 100.0,
                "cap_points": 16,
            },
            "start_policy": {"kind": "cycle_start"},
            "rank": {"intensity_reference": 10.0},
        }
    )
    extended = registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert len(extended.armed) == len(registry.load_metrics().armed) + 1
    assert "relhum" in extended.publication_manifest()
    assert extended.metric_rows["moist_layer_area"].threat_class == "moist_layer"


#: A COMPOUND threat, expressed entirely as data: a low that is also warm
#: and also windy, which is three conditions in three different units --
#: pascals, kelvin and metres per second.  Nothing about it is a special
#: case anywhere in the package: four margin rows, one combining row, one
#: metric row, and the same ``area_threshold_exceedance`` detector that
#: finds a blob of reflectivity.  ``tests/test_swath_plan.py`` runs this
#: same document against a real fixture history and asserts it places.
COMPOUND_FIELDS = [
    {
        "id": "probe_low_margin",
        "inputs": ["surface_low"],
        "derivation": {
            "kind": "threshold_margin", "comparison": "at_most",
            "threshold": 100600.0, "scale": 1000.0,
        },
        "units": "dimensionless",
    },
    {
        "id": "probe_warm_margin",
        "inputs": ["two_metre_temperature"],
        "derivation": {
            "kind": "threshold_margin", "comparison": "at_least",
            "threshold": 295.0, "scale": 5.0,
        },
        "units": "dimensionless",
    },
    {
        "id": "probe_windy_margin",
        "inputs": ["surface_wind_speed"],
        "derivation": {
            "kind": "threshold_margin", "comparison": "at_least",
            "threshold": 12.0, "scale": 6.0,
        },
        "units": "dimensionless",
    },
    {
        "id": "probe_compound_margin",
        "inputs": ["probe_low_margin", "probe_warm_margin", "probe_windy_margin"],
        "derivation": {"kind": "extremum_of", "extremum": "minimum"},
        "units": "dimensionless",
    },
]

COMPOUND_METRIC = {
    "id": "probe_warm_windy_low_area",
    "threat_class": "warm_windy_low",
    "field": "probe_compound_margin",
    "enabled": True,
    "detector": {
        "kind": "area_threshold_exceedance",
        "comparison": "at_least",
        "threshold": 0.0,
        "minimum_area_km2": 20000.0,
        "maximum_area_km2": 1600000.0,
    },
    "confirm_with": [],
    "track": {"maximum_speed_km_per_hour": 90.0, "minimum_frames": 2},
    "swath": {
        "half_width_km": 150.0,
        "flare_km_per_hour": 10.0,
        "maximum_half_width_km": 250.0,
        "spacing_km": 6.0,
        "cull_pad_scale": 1.0,
        "transition_cells": 81.0,
        "lead_hours": 9.0,
        "path_step_km": 100.0,
        "cap_points": 16,
    },
    "start_policy": {"kind": "cycle_start"},
    "rank": {"intensity_reference": 1.0},
}


def with_compound_threat(document: dict) -> dict:
    """The shipped document plus one compound threat, as data only."""

    document["fields"].extend(copy.deepcopy(COMPOUND_FIELDS))
    document["metrics"].append(copy.deepcopy(COMPOUND_METRIC))
    return document


def test_a_compound_threat_needs_no_source_edit(
    tmp_path: Path, metrics_document: dict
) -> None:
    """THE HARDER ARBITRARY TEST: three conditions in three units, as rows.

    The earlier test adds a phenomenon that is one threshold on one field,
    which the v1 grammar could already express.  This one adds a threat
    whose definition is a CONJUNCTION over quantities that do not share a
    unit -- the thing that made fire weather, atmospheric rivers and
    ingredients-based severe convection inexpressible.  It must cost rows
    and nothing else.
    """

    extended = registry.load_metrics(
        _write(tmp_path, "metrics.json", with_compound_threat(metrics_document))
    )
    row = extended.metric_rows["probe_warm_windy_low_area"]
    assert row.threat_class == "warm_windy_low"
    assert row.detector.kind == "area_threshold_exceedance"
    # The manifest reaches the leaves through four rows of composition and
    # adds nothing the shipped rows did not already need, because every
    # ingredient is built from fields that already exist.
    assert set(extended.publication_manifest()) == set(
        registry.load_metrics().publication_manifest()
    )
    assert set(
        extended.leaf_variables("probe_compound_margin", whose="the compound probe")
    ) == {"surface_pressure", "t2", "ter", "u10", "v10"}


def test_a_field_row_that_depends_on_itself_refuses_with_the_ring(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["fields"].append(
        {
            "id": "ring_a",
            "inputs": ["ring_b"],
            "derivation": {"kind": "linear_combination", "coefficients": [1.0]},
        }
    )
    metrics_document["fields"].append(
        {
            "id": "ring_b",
            "inputs": ["ring_a"],
            "derivation": {"kind": "linear_combination", "coefficients": [1.0]},
        }
    )
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "ring_a" in message and "ring_b" in message
    assert "depends on itself" in message


def test_a_row_naming_both_operand_sides_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["fields"].append(
        {
            "id": "both_sides",
            "source_variables": ["t2"],
            "inputs": ["two_metre_temperature"],
            "derivation": {"kind": "linear_combination", "coefficients": [1.0]},
        }
    )
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "BOTH" in str(caught.value)


def test_a_leaf_kind_fed_from_other_rows_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    """``level_slice`` reads the file's raw shapes; a scalar row cannot feed it."""

    metrics_document["fields"].append(
        {
            "id": "sliced_scalar",
            "inputs": ["two_metre_temperature"],
            "derivation": {"kind": "level_slice", "level_index": 3},
        }
    )
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "level_slice" in message
    assert "source_variables" in message


def test_a_coefficient_list_that_does_not_match_its_operands_refuses(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["fields"].append(
        {
            "id": "mismatched",
            "inputs": ["two_metre_temperature", "surface_wind_speed"],
            "derivation": {"kind": "linear_combination", "coefficients": [1.0]},
        }
    )
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "BY POSITION" in str(caught.value)


def test_a_metric_row_without_an_intensity_reference_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    del metrics_document["metrics"][0]["rank"]
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "intensity_reference" in message
    assert "pascals against decibels" in message


def test_a_v1_document_refuses_naming_what_moved(
    tmp_path: Path, metrics_document: dict
) -> None:
    """The previous schema is refused with the MOVE named, not a bare mismatch."""

    metrics_document["schema"] = "gpuwm-hex.threat-metrics.v1"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "intensity_reference" in message
    assert registry.METRICS_SCHEMA in message


def test_an_unknown_region_kind_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["region"] = {"kind": "conus"}
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "conus" in str(caught.value)
    assert "region.kind" in str(caught.value)


def test_a_bounding_box_region_wraps_the_antimeridian() -> None:
    """West of east is a Pacific box, not an empty one."""

    box = registry.Region.from_mapping(
        {
            "kind": "bounding_box", "south_deg": -10.0, "north_deg": 10.0,
            "west_deg": 150.0, "east_deg": -150.0,
        },
        "probe",
    )
    assert box.contains(0.0, 179.0)
    assert box.contains(0.0, -179.0)
    assert not box.contains(0.0, 0.0)
    assert not box.contains(40.0, 179.0)


# ---------------------------------------------------------------------------
# closed vocabularies, refused at load
# ---------------------------------------------------------------------------
def test_an_unknown_derivation_kind_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["fields"][0]["derivation"] = {"kind": "spectral_filter"}
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "spectral_filter" in message
    assert "derivation.kind" in message
    assert "closed" in message


def test_an_unknown_detector_kind_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["detector"]["kind"] = "eyewall_finder"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "eyewall_finder" in str(caught.value)


def test_an_unknown_aggregation_kind_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["confirm_with"][0]["aggregation"]["kind"] = "ball_median"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "ball_median" in str(caught.value)


def test_an_unknown_start_policy_kind_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["start_policy"] = {"kind": "operator_choice"}
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "operator_choice" in str(caught.value)


def test_an_unknown_rank_term_kind_refuses_by_name(
    tmp_path: Path, policy_document: dict
) -> None:
    policy_document["rank_terms"][0]["kind"] = "operator_priority"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_policy(_write(tmp_path, "policy.json", policy_document))
    assert "operator_priority" in str(caught.value)


def test_an_unknown_key_anywhere_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["swath"]["half_width_kilometres"] = 150.0
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    message = str(caught.value)
    assert "half_width_kilometres" in message
    assert "unknown key" in message


def test_a_metric_naming_an_undefined_field_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["field"] = "sea_surface_temperature"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "sea_surface_temperature" in str(caught.value)


def test_a_wrong_schema_string_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    # A string that is neither this build's schema nor a superseded one, so
    # the refusal under test is the bare mismatch rather than a named move.
    metrics_document["schema"] = "gpuwm-hex.threat-metrics.v99"
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "v99" in str(caught.value)
    assert registry.METRICS_SCHEMA in str(caught.value)


# ---------------------------------------------------------------------------
# the total-order gate
# ---------------------------------------------------------------------------
def test_a_tiebreak_that_is_not_total_refuses_by_name(
    tmp_path: Path, policy_document: dict
) -> None:
    policy_document["tiebreak"] = ["rank_score", "threat_class"]
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_policy(_write(tmp_path, "policy.json", policy_document))
    message = str(caught.value)
    assert "TOTAL" in message
    assert "two different plans" in message


def test_an_empty_tiebreak_refuses(tmp_path: Path, policy_document: dict) -> None:
    policy_document["tiebreak"] = []
    with pytest.raises(SwathDocumentError):
        registry.load_policy(_write(tmp_path, "policy.json", policy_document))


# ---------------------------------------------------------------------------
# numbers that cannot describe a placement
# ---------------------------------------------------------------------------
def test_a_maximum_half_width_below_the_base_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["swath"]["maximum_half_width_km"] = 100.0
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "flare would run backwards" in str(caught.value)


def test_a_negative_flare_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    metrics_document["metrics"][0]["swath"]["flare_km_per_hour"] = -5.0
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "narrower than the error" in str(caught.value)


def test_a_budget_of_zero_swaths_refuses_by_name(
    tmp_path: Path, policy_document: dict
) -> None:
    policy_document["budget"]["maximum_swaths"] = 0
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_policy(_write(tmp_path, "policy.json", policy_document))
    assert "coarse-only" in str(caught.value)


def test_an_empty_metrics_document_refuses_by_name(tmp_path: Path) -> None:
    document = {"schema": registry.METRICS_SCHEMA, "fields": [], "metrics": []}
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", document))
    assert "quiet day" in str(caught.value)


def test_a_detector_without_a_threshold_refuses_by_name(
    tmp_path: Path, metrics_document: dict
) -> None:
    detector = metrics_document["metrics"][0]["detector"]
    del detector["threshold"]
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(_write(tmp_path, "metrics.json", metrics_document))
    assert "threshold" in str(caught.value)


def test_a_missing_document_refuses_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(SwathDocumentError) as caught:
        registry.load_metrics(tmp_path / "absent.json")
    assert "absent.json" in str(caught.value)


def test_loading_twice_gives_the_same_digest(metrics_document: dict) -> None:
    assert registry.load_metrics().sha256 == registry.load_metrics().sha256
    assert copy.deepcopy(metrics_document) == metrics_document
