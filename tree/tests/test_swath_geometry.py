"""Spherical geometry for a flared swath, and the refusals that guard it.

Every assertion here is either a closed-form spherical identity or a
property the emitted polygon must have for the SHIPPED generator to
accept it: ``rw_mpas_mesh`` reads a ``polygon`` region as a ring of
``[lat, lon]`` vertices and computes exact great-circle distances to its
segments, so a ring that crosses itself is not a swath, it is two
overlapping lobes with an undefined interior.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hexcore.swath import geometry as geo
from hexcore.swath.errors import SwathRefusal


def test_great_circle_matches_a_known_quarter_of_the_equator() -> None:
    km = geo.great_circle_km(0.0, 0.0, 0.0, 90.0)
    assert km == pytest.approx(0.25 * 2.0 * math.pi * geo.EARTH_RADIUS_KM, rel=1e-12)


def test_initial_bearing_due_east_on_the_equator_is_ninety() -> None:
    assert geo.initial_bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0, abs=1e-9)


def test_initial_bearing_due_north_is_zero() -> None:
    assert geo.initial_bearing_deg(10.0, 20.0, 30.0, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_destination_round_trips_through_bearing_and_distance() -> None:
    lat, lon = geo.destination(17.0, -58.0, 47.5, 640.0)
    assert geo.great_circle_km(17.0, -58.0, lat, lon) == pytest.approx(640.0, rel=1e-9)
    assert geo.initial_bearing_deg(17.0, -58.0, lat, lon) == pytest.approx(47.5, abs=1e-6)


def test_destination_crosses_the_antimeridian_without_a_seam() -> None:
    lat, lon = geo.destination(20.0, 179.0, 90.0, 400.0)
    assert -180.0 <= lon < 180.0
    assert lon < 0.0
    assert geo.great_circle_km(20.0, 179.0, lat, lon) == pytest.approx(400.0, rel=1e-9)


def test_a_straight_track_makes_a_polygon_of_the_expected_area() -> None:
    # A due-east track of 1,000 km with a constant 150 km half-width is a
    # 1,000 x 300 km stadium: two half-discs of radius 150 plus a rectangle.
    path = [geo.destination(0.0, 0.0, 90.0, step) for step in (0.0, 500.0, 1000.0)]
    ring = geo.swath_ring(path, [150.0, 150.0, 150.0], cap_points=48)
    expected = 1000.0 * 300.0 + math.pi * 150.0 * 150.0
    assert geo.polygon_area_km2(ring) == pytest.approx(expected, rel=0.01)


def test_a_flared_swath_is_wider_at_the_end_than_at_the_start() -> None:
    path = [geo.destination(0.0, 0.0, 90.0, step) for step in (0.0, 500.0, 1000.0)]
    ring = geo.swath_ring(path, [150.0, 225.0, 300.0], cap_points=8)
    start_width = min(geo.great_circle_km(path[0][0], path[0][1], lat, lon) for lat, lon in ring)
    end_width = min(geo.great_circle_km(path[2][0], path[2][1], lat, lon) for lat, lon in ring)
    assert start_width == pytest.approx(150.0, rel=0.02)
    assert end_width == pytest.approx(300.0, rel=0.02)


def test_the_ring_is_closed_without_repeating_the_first_vertex() -> None:
    path = [geo.destination(0.0, 0.0, 90.0, step) for step in (0.0, 400.0, 800.0)]
    ring = geo.swath_ring(path, [120.0, 120.0, 120.0], cap_points=6)
    assert ring[0] != ring[-1]
    assert len(ring) >= 8


def test_a_hairpin_track_is_refused_by_name_not_silently_emitted() -> None:
    # The track doubles back on itself inside its own half-width: the left
    # offset of the outbound leg crosses the left offset of the return leg.
    path = [(0.0, 0.0), (0.0, 3.0), (0.06, 0.0)]
    with pytest.raises(SwathRefusal) as caught:
        geo.swath_ring(path, [300.0, 300.0, 300.0], cap_points=8)
    message = str(caught.value)
    assert "self-intersect" in message
    assert "half-width" in message


def test_a_track_with_one_point_is_refused_by_name() -> None:
    with pytest.raises(SwathRefusal) as caught:
        geo.swath_ring([(10.0, 20.0)], [150.0], cap_points=8)
    assert "at least two" in str(caught.value)


def test_a_nonpositive_half_width_is_refused_by_name() -> None:
    path = [(0.0, 0.0), (0.0, 5.0)]
    with pytest.raises(SwathRefusal) as caught:
        geo.swath_ring(path, [150.0, 0.0], cap_points=8)
    assert "half-width" in str(caught.value)


def test_polygon_area_of_a_small_cap_matches_the_planar_disc() -> None:
    ring = [geo.destination(30.0, -95.0, bearing, 200.0) for bearing in range(0, 360, 5)]
    assert geo.polygon_area_km2(ring) == pytest.approx(math.pi * 200.0 * 200.0, rel=0.01)


def test_resample_track_puts_points_at_the_requested_spacing() -> None:
    path = [(0.0, 0.0), (0.0, 9.0)]
    dense = geo.resample_track(path, step_km=100.0)
    assert len(dense) >= 10
    gaps = [
        geo.great_circle_km(dense[i][0], dense[i][1], dense[i + 1][0], dense[i + 1][1])
        for i in range(len(dense) - 1)
    ]
    assert max(gaps) <= 105.0
    assert geo.great_circle_km(*dense[0], *path[0]) == pytest.approx(0.0, abs=1e-6)
    assert geo.great_circle_km(*dense[-1], *path[-1]) == pytest.approx(0.0, abs=1e-6)


def test_predicted_cells_in_scales_with_area_over_hexagon_area() -> None:
    ring = [geo.destination(0.0, 0.0, bearing, 300.0) for bearing in range(0, 360, 5)]
    area = geo.polygon_area_km2(ring)
    cells = geo.predicted_cells_in(ring, spacing_km=30.0, boundary_rings=0)
    assert cells == pytest.approx(area / (0.5 * math.sqrt(3.0) * 30.0 * 30.0), rel=1e-9)


def test_predicted_cells_in_counts_the_boundary_zone_by_default() -> None:
    """A limited-area cull emits the interior PLUS seven rings, and every
    one of those cells is integrated on the card.

    Measured on the real published x1.40962 parent with the shipped culler
    (evidence/swath-following-20260826/): 39 interior cells against 354
    total for the same region, so an estimate that stopped at the polygon
    would be 89 % low.
    """

    ring = [geo.destination(0.0, 0.0, bearing, 300.0) for bearing in range(0, 360, 5)]
    bare = geo.predicted_cells_in(ring, spacing_km=30.0, boundary_rings=0)
    with_zone = geo.predicted_cells_in(ring, spacing_km=30.0)
    assert geo.BOUNDARY_RINGS == 7
    assert with_zone > bare
    halo = geo.BOUNDARY_RINGS * 30.0
    expected = (
        geo.polygon_area_km2(ring)
        + geo.polygon_perimeter_km(ring) * halo
        + math.pi * halo * halo
    ) / (0.5 * math.sqrt(3.0) * 30.0 * 30.0)
    assert with_zone == pytest.approx(expected, rel=1e-9)


def test_polygon_perimeter_of_a_small_cap_matches_the_planar_circle() -> None:
    ring = [geo.destination(30.0, -95.0, bearing, 200.0) for bearing in range(0, 360, 2)]
    assert geo.polygon_perimeter_km(ring) == pytest.approx(
        2.0 * math.pi * 200.0, rel=0.01
    )


def test_ring_vertices_are_ordered_lat_then_lon_in_degrees() -> None:
    path = [(12.0, -60.0), (14.0, -64.0)]
    ring = geo.swath_ring(path, [150.0, 200.0], cap_points=8)
    for lat, lon in ring:
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon < 180.0


def test_containment_does_not_call_the_complement_interior() -> None:
    # The measured engine defect, asserted against OUR test so it cannot
    # recur here: a point at the antipode of a small ring is OUTSIDE it.
    ring = [geo.destination(12.0, -58.0, bearing, 250.0) for bearing in range(0, 360, 10)]
    inside = geo.ring_containment(ring)
    assert inside((12.0, -58.0))
    assert not inside((-12.0, 122.0))
    assert not inside((12.0, -40.0))


def test_overlap_of_a_ring_with_itself_is_one() -> None:
    path = [(12.0, -58.0), (14.0, -62.0), (16.0, -66.0)]
    ring = geo.swath_ring(path, [150.0, 200.0, 250.0], cap_points=12)
    assert geo.ring_overlap_fraction(ring, ring) == pytest.approx(1.0, abs=1e-12)


def test_overlap_of_two_far_apart_rings_is_zero() -> None:
    a = geo.swath_ring([(12.0, -58.0), (14.0, -62.0)], [150.0, 200.0], cap_points=12)
    b = geo.swath_ring([(12.0, 58.0), (14.0, 62.0)], [150.0, 200.0], cap_points=12)
    assert geo.ring_overlap_fraction(a, b) == 0.0


def test_overlap_falls_as_a_swath_walks_off_its_predecessor() -> None:
    base = geo.swath_ring([(12.0, -58.0), (14.0, -62.0)], [200.0, 200.0], cap_points=12)
    near = geo.swath_ring([(12.2, -58.4), (14.2, -62.4)], [200.0, 200.0], cap_points=12)
    far = geo.swath_ring([(15.0, -62.0), (17.0, -66.0)], [200.0, 200.0], cap_points=12)
    assert geo.ring_overlap_fraction(base, near) > 0.8
    assert geo.ring_overlap_fraction(base, far) < 0.5


def _worst_turn_deg(ring):
    worst = 0.0
    count = len(ring)
    for index in range(count):
        before, here, after = ring[index - 1], ring[index], ring[(index + 1) % count]
        first = geo.initial_bearing_deg(before[0], before[1], here[0], here[1])
        second = geo.initial_bearing_deg(here[0], here[1], after[0], after[1])
        worst = max(worst, abs(((second - first + 180.0) % 360.0) - 180.0))
    return worst


def test_a_folded_offset_chain_is_forced_forward_along_the_axis() -> None:
    """The measured defect, as a unit.

    Where the axis turns tighter than the local half-width, the offset on
    the inside of the bend overshoots its neighbours and the side chain
    reverses.  A fold is a COLLINEAR overlap, so the transversal
    self-intersection test passes it and the culler makes a perfectly
    valid bounded disk out of a ring carrying a zero-area sliver -- a
    defect nothing downstream can see.
    """

    # Three points heading due east, with the middle one displaced far
    # enough north that the southward offsets fold back past each other.
    axis = [(0.0, 0.0), (2.4, 0.6), (0.0, 1.2)]
    bearings = geo._axis_bearings(axis)  # noqa: SLF001
    raw = [
        geo.destination(point[0], point[1], bearing + 90.0, 400.0)
        for point, bearing in zip(axis, bearings)
    ]
    cleaned = geo._forward_only(raw, bearings)  # noqa: SLF001
    assert len(cleaned) < len(raw)
    assert cleaned[0] == raw[0]
    assert cleaned[-1] == raw[-1]


def test_a_real_bend_produces_a_ring_with_no_reversal() -> None:
    path = [(0.0, 0.0)]
    path.append(geo.destination(*path[-1], 90.0, 260.0))
    path.append(geo.destination(*path[-1], 55.0, 260.0))
    path.append(geo.destination(*path[-1], 20.0, 260.0))
    ring = geo.swath_ring(path, [180.0, 210.0, 240.0, 270.0], cap_points=16)
    assert _worst_turn_deg(ring) < 100.0


def test_the_offset_chain_keeps_both_of_its_ends() -> None:
    """The ends carry the half-width the receipt quotes, so they are never
    the vertices a cleanup drops."""

    path = [(0.0, 0.0)]
    path.append(geo.destination(*path[-1], 90.0, 150.0))
    path.append(geo.destination(*path[-1], 40.0, 150.0))
    widths = [200.0, 250.0, 300.0]
    bearings = geo._axis_bearings(path)  # noqa: SLF001
    for sign in (-90.0, 90.0):
        raw = [
            geo.destination(point[0], point[1], bearing + sign, width)
            for point, bearing, width in zip(path, bearings, widths)
        ]
        cleaned = geo._forward_only(raw, bearings)  # noqa: SLF001
        assert cleaned[0] == raw[0]
        assert cleaned[-1] == raw[-1]


def test_fit_swath_axis_reports_its_own_smoothing_and_drift() -> None:
    jittery = [
        (0.0, 0.0), (0.35, 1.0), (0.0, 2.0), (0.35, 3.0), (0.0, 4.0), (0.35, 5.0),
    ]
    axis, ring, passes, drift = geo.fit_swath_axis(
        jittery, [200.0] * len(jittery), cap_points=12
    )
    assert len(axis) == len(jittery)
    assert passes >= 0
    assert drift >= 0.0
    assert axis[0] == jittery[0]
    assert axis[-1] == jittery[-1]
    assert _worst_turn_deg(ring) < 100.0
