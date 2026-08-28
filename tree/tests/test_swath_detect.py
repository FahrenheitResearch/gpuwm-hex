"""The detector and the tracker, read against the port's own history writer.

CROSS-LANE ON PURPOSE.  The history these tests read is written by
``hexcore.output.write_history`` -- the shipped writer, with its
one-based ``cellsOnCell``, its zero padding, its radians and its MPAS time
strings.  A reader tested only against a file its own test wrote agrees
with itself and with nothing else, and this project has paid for that
before.

The FIELDS are analytic and the tests say what that does and does not
buy: they prove the mechanism finds and follows a moving minimum on an
unstructured mesh, and they prove nothing about whether the shipped
thresholds find real cyclones in a real forecast.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.swath import detect as detect_module  # noqa: E402
from hexcore.swath import registry, track  # noqa: E402
from hexcore.swath.errors import SwathRefusal  # noqa: E402
from hexcore.swath.geometry import great_circle_km  # noqa: E402
from hexcore.swath.history import HistoryReader  # noqa: E402

#: The size of the coarse global mesh this layer is designed against: the
#: shipped ladder's L1 is a graded mesh of this order, and a search ball of
#: 300 km spans about nineteen cells here.  At 10,242 cells the same ball is
#: seven irregular cells and the weighted centre quantizes hard enough to
#: break the association's own speed gate -- measured, and the reason this
#: number is what it is.
CELLS = 40962

#: One eastbound low, deep and wide enough to be resolved on a 10,242-cell
#: mesh, plus one convective area that switches on
#: at hour 6 so the delayed-start path has something to derive an hour from.
SCENARIO = [
    {"kind": "low", "latitude_deg": 15.0, "longitude_deg": -50.0, "bearing_deg": 90.0,
     "speed_km_per_hour": 40.0, "radius_km": 700.0, "amplitude": 4500.0},
    {"kind": "convection", "latitude_deg": 38.0, "longitude_deg": -97.0,
     "bearing_deg": 90.0, "speed_km_per_hour": 50.0, "radius_km": 600.0,
     "amplitude": 55.0, "onset_hours": 6.0},
]
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0]


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("swath-detect") / "coarse.nc"
    return fixture.build(target, cells=CELLS, hours=HOURS, scenario=SCENARIO)


@pytest.fixture(scope="module")
def metrics() -> registry.MetricRegistry:
    return registry.load_metrics()


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
def test_the_reader_takes_the_mesh_out_of_the_history(history: Path) -> None:
    with HistoryReader(history) as reader:
        assert reader.cell_count == CELLS
        assert reader.latitudes_deg.shape == (CELLS,)
        assert reader.longitudes_deg.min() >= -180.0
        assert reader.longitudes_deg.max() < 180.0
        assert reader.areas_km2.sum() == pytest.approx(5.1e8, rel=0.01)


def test_connectivity_round_trips_through_the_shipped_writer(history: Path) -> None:
    """Zero-based in memory, one-based on disk, zero-based again on read.

    The writer converts on the way out; a reader that forgot to convert
    back would have every neighbour index one too high and the detector's
    balls would walk the wrong cells while still looking plausible.  So the
    assertion is against the FILE's own bytes, not against a second copy of
    the same belief.
    """

    import numpy as np
    from netCDF4 import Dataset

    with HistoryReader(history) as reader:
        neighbours = reader.neighbours()
    assert neighbours.shape[0] == CELLS
    assert neighbours.max() < CELLS
    assert neighbours.min() >= -1
    with Dataset(str(history), "r") as source:
        disk = np.asarray(source.variables["cellsOnCell"][:], dtype=np.int64)
    assert disk.min() >= 0
    assert disk.max() == CELLS
    present = disk > 0
    assert np.array_equal(neighbours[present], disk[present] - 1)


def test_frames_carry_seconds_not_frame_indices(history: Path) -> None:
    with HistoryReader(history) as reader:
        frames = reader.frames()
    assert [frame.time_seconds for frame in frames] == [
        hour * 3600.0 for hour in HOURS
    ]
    assert frames[0].valid_time.startswith("2026-08-26")


def test_an_unpublished_field_refuses_naming_the_manifest(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    row = registry.FieldRow(
        id="absent", source_variables=("cape",), derivation_kind="direct"
    )
    with HistoryReader(history) as reader:
        with pytest.raises(SwathRefusal) as caught:
            reader.derive(row, 0)
    message = str(caught.value)
    assert "cape" in message
    assert "publication-manifest" in message


def test_vertical_extremum_reduces_a_three_dimensional_field(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    row = metrics.field_rows["column_max_reflectivity"]
    with HistoryReader(history) as reader:
        values = reader.derive(row, 3)
    assert values.shape == (CELLS,)
    assert values.max() > 35.0


def test_vector_magnitude_combines_two_source_variables(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    row = metrics.field_rows["surface_wind_speed"]
    with HistoryReader(history) as reader:
        values = reader.derive(row, 0)
    assert values.shape == (CELLS,)
    assert values.min() >= 0.0
    assert values.max() > 17.0


# ---------------------------------------------------------------------------
# the detector
# ---------------------------------------------------------------------------
def test_the_extremum_search_finds_the_low_where_it_was_put(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    first = [
        feature for feature in result.features
        if feature.metric_id == "tropical_cyclone_centre" and feature.frame_index == 0
    ]
    assert len(first) == 1
    assert great_circle_km(first[0].latitude_deg, first[0].longitude_deg, 15.0, -50.0) < 250.0


def test_the_centre_is_sub_cell_and_moves_every_frame(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    """A centre quantized to a cell centre would stand still for hours.

    On this mesh a cell is about 220 km across and the low moves 120 km
    between frames, so an extremum-cell centre would jump in steps.  The
    anomaly-weighted centroid must move every frame.
    """

    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    lows = sorted(
        (f for f in result.features if f.metric_id == "tropical_cyclone_centre"),
        key=lambda f: f.frame_index,
    )
    steps = [
        great_circle_km(a.latitude_deg, a.longitude_deg, b.latitude_deg, b.longitude_deg)
        for a, b in zip(lows, lows[1:])
    ]
    assert len(steps) >= 4
    assert all(step > 0.0 for step in steps), steps
    net = great_circle_km(
        lows[0].latitude_deg, lows[0].longitude_deg,
        lows[-1].latitude_deg, lows[-1].longitude_deg,
    )
    assert net == pytest.approx(40.0 * 12.0, rel=0.25), net


def test_an_area_detector_finds_the_convective_region_only_after_onset(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    frames = sorted(
        feature.frame_index for feature in result.features
        if feature.metric_id == "deep_convection_area"
    )
    assert frames
    assert min(frames) == HOURS.index(6.0)


def test_a_failed_confirmation_is_recorded_as_a_drop_with_its_numbers(
    tmp_path: Path, metrics: registry.MetricRegistry
) -> None:
    """A pressure minimum with no circulation must be dropped, not placed."""

    calm = [
        {"kind": "low", "latitude_deg": 15.0, "longitude_deg": -50.0,
         "bearing_deg": 90.0, "speed_km_per_hour": 40.0, "radius_km": 700.0,
         "amplitude": 4500.0, "wind_factor": 0.05},
    ]
    path = fixture.build(tmp_path / "calm.nc", cells=CELLS, hours=HOURS, scenario=calm)
    with HistoryReader(path) as reader:
        result = detect_module.detect(reader, metrics)
    assert not [f for f in result.features if f.metric_id == "tropical_cyclone_centre"]
    reasons = {drop.reason for drop in result.drops}
    assert any("confirm_with surface_wind_speed" in reason for reason in reasons)
    drop = next(d for d in result.drops if "confirm_with" in d.reason)
    assert drop.required == 17.0
    assert drop.measured < 17.0


def test_the_decision_receipt_carries_the_history_digest(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
        receipt = detect_module.detection_receipt(reader, metrics, result)
    assert receipt["schema"] == "gpuwm-hex.threat-decision.v1"
    assert len(receipt["history"]["sha256"]) == 64
    assert receipt["metrics_document"]["sha256"] == metrics.sha256
    assert receipt["counts"]["features"] == len(result.features)


def test_detection_is_deterministic(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        first = detect_module.detect(reader, metrics)
    with HistoryReader(history) as reader:
        second = detect_module.detect(reader, metrics)
    assert [f.as_row() for f in first.features] == [f.as_row() for f in second.features]
    assert [d.as_row() for d in first.drops] == [d.as_row() for d in second.drops]


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------
def test_one_moving_low_becomes_one_track(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    tracks = track.associate(result.features, metrics.metric_rows)
    lows = [item for item in tracks if item.metric_id == "tropical_cyclone_centre"]
    assert len(lows) == 1
    assert lows[0].frames == len(HOURS)
    assert lows[0].displacement_km() > 300.0


def test_the_speed_gate_refuses_to_join_two_distant_features(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    """Halve the gate below the storm's own speed and the track shatters."""

    import dataclasses

    row = metrics.metric_rows["tropical_cyclone_centre"]
    slow = dataclasses.replace(
        row, track=dataclasses.replace(row.track, maximum_speed_km_per_hour=1.0)
    )
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    tracks = track.associate(result.features, {**metrics.metric_rows, row.id: slow})
    lows = [item for item in tracks if item.metric_id == "tropical_cyclone_centre"]
    # Every frame becomes its own single-frame track, and the row's own
    # minimum_frames=2 then keeps none of them: a gate that cannot join is
    # a gate that reports nothing, not one that reports a wrong join.
    assert lows == []


def test_the_projection_uses_the_tracks_own_frames_not_extrapolation(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    tracks = track.associate(result.features, metrics.metric_rows)
    low = next(item for item in tracks if item.metric_id == "tropical_cyclone_centre")
    path = track.project(low, start_seconds=0.0, lead_hours=12.0)
    assert path.extrapolated_hours == 0.0
    assert len(path.points) == len(HOURS)


def test_a_window_past_the_last_frame_is_labelled_extrapolated(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    tracks = track.associate(result.features, metrics.metric_rows)
    low = next(item for item in tracks if item.metric_id == "tropical_cyclone_centre")
    path = track.project(low, start_seconds=0.0, lead_hours=18.0)
    assert path.extrapolated_hours == pytest.approx(6.0)


def test_the_flare_widens_with_lead(
    history: Path, metrics: registry.MetricRegistry
) -> None:
    with HistoryReader(history) as reader:
        result = detect_module.detect(reader, metrics)
    tracks = track.associate(result.features, metrics.metric_rows)
    low = next(item for item in tracks if item.metric_id == "tropical_cyclone_centre")
    path = track.project(low, start_seconds=0.0, lead_hours=12.0)
    widths = track.half_width_profile(
        path, base_km=150.0, flare_km_per_hour=12.5, maximum_km=300.0
    )
    assert widths[0] == pytest.approx(150.0)
    assert widths[-1] > widths[0]
    assert max(widths) <= 300.0
