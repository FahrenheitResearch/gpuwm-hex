"""The instrument that settles the nest-ratio question, tested both directions.

THE BREAKAGE THESE PREVENT.  The whole nest-ratio decision rests on one
picture: interior agreement against distance from the boundary, for domains of
different size at one resolution.  A flat picture says the boundary starves
nothing and no intermediate resolution level gets built; a picture that climbs
toward the edge says the opposite and buys several extra forecasts per
cycle.  An instrument that cannot draw a real gradient would answer "flat" for
free, and an instrument that manufactures one out of geometry would answer
"climbing" for free.  Both failures are silent and both decide a programme.

So each property is asserted in the direction that would hide a wrong answer
AND in the direction that would invent one:

* identical arms measure exactly zero, at every distance -- so "flat" is not
  what this returns when nothing is happening;
* a boundary-localised perturbation IS recovered, with the decay in the right
  direction -- so "flat" is not what it returns when something is;
* a uniform perturbation reads flat -- so a gradient is never manufactured
  out of the cell distribution or the binning;
* the distance axis itself is checked against a hand-computed great circle,
  because every profile in the receipt is indexed by it.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import measure_domain_size_agreement as instrument  # noqa: E402
import scale_region_about_centroid as scaler  # noqa: E402


def _patch_geometry(cells: int = 400, span_deg: float = 4.0):
    """A strip of cells with a driven zone along one edge.

    Deliberately one-dimensional: the property under test is "does a
    difference that lives near the boundary show up near the boundary", and a
    strip makes the expected answer computable by hand.
    """

    lat = np.full(cells, -60.0)
    lon = np.linspace(0.0, span_deg, cells)
    driven_lat = np.array([-60.0])
    driven_lon = np.array([-0.05])
    patch = instrument.unit_vectors(np.radians(lat), np.radians(lon))
    driven = instrument.unit_vectors(np.radians(driven_lat), np.radians(driven_lon))
    return patch, driven


def test_the_distance_axis_is_a_real_great_circle():
    """Every profile is indexed by this number, so it is checked by hand.

    One degree of longitude at 60 S is a small circle, not a great one: the
    great-circle distance between two points a degree apart there is
    ``2*asin(cos(60 deg)*sin(0.5 deg))`` on the sphere, which is about
    55.6 km rather than 111.2.  An instrument that used a flat degree scale
    would put every curve at twice its true distance from the boundary and
    the verdict would be read off the wrong axis.
    """

    patch = instrument.unit_vectors(np.radians([-60.0]), np.radians([1.0]))
    driven = instrument.unit_vectors(np.radians([-60.0]), np.radians([0.0]))
    measured = instrument.boundary_distance_km(patch, driven)[0]
    expected = (
        2.0
        * np.arcsin(np.cos(np.radians(60.0)) * np.sin(np.radians(0.5)))
        * instrument.SPHERE_RADIUS_KM
    )
    assert measured == pytest.approx(expected, rel=1e-9)
    assert 55.0 < measured < 56.0


def test_an_arms_mean_radius_is_a_real_great_circle():
    """The headline chart's x axis, checked against a hand computation.

    THE BUG THIS CAUGHT, on its first run: ``latCell``/``lonCell`` are RADIANS
    on disk -- the files declare ``units = "rad"`` -- and the helper converted
    them again.  Every angle came out 57.3 times too small, which put a domain
    whose boundary sits 135 km out at a mean radius of 4.2 km: small enough to
    read as a plausible cell width rather than as an obvious error, on the one
    axis the whole nest-ratio decision is read along.
    """

    lat = np.radians([-60.0, -60.0, -59.0, -61.0])
    lon = np.radians([139.0, 139.0, 139.0, 139.0])
    arm = {
        "lat": lat,
        "lon": lon,
        "bdy": np.array([0, 0, 1, 1]),
        "spacing_km": np.array([5.0, 5.0, 9.0, 9.0]),
    }
    geometry = instrument.arm_geometry(arm)
    one_degree_km = np.radians(1.0) * instrument.SPHERE_RADIUS_KM
    assert geometry["mean_radius_km"] == pytest.approx(one_degree_km, rel=1e-9)
    assert 111.0 < geometry["mean_radius_km"] < 112.0
    assert geometry["driven_ring_mean_width_km"] == pytest.approx(9.0)


def test_an_arm_with_no_boundary_reports_no_geometry():
    """A global run has no radius to report, and must say so rather than guess.

    Reported as a number, it would land on the headline chart as a domain of
    some size, which is exactly what a run with no lateral boundary is not.
    """

    arm = {
        "lat": np.radians([-60.0, -59.0]),
        "lon": np.radians([139.0, 139.0]),
        "bdy": np.array([0, 0]),
        "spacing_km": np.array([5.0, 5.0]),
    }
    geometry = instrument.arm_geometry(arm)
    assert geometry["mean_radius_km"] is None
    assert geometry["driven_ring_mean_width_km"] is None


def test_identical_arms_measure_exactly_zero_at_every_distance():
    """The false-negative direction: agreement must not be assumed.

    If this instrument returned a small non-zero difference for two identical
    arms, every verdict would be read against a floor nobody measured, and
    "the interiors agree" would be a statement about the instrument.
    """

    patch, driven = _patch_geometry()
    distance = instrument.boundary_distance_km(patch, driven)
    edges = instrument.bin_edges_for(distance, 25.0)
    values = np.random.default_rng(0).normal(size=(patch.shape[0], 3))
    rows = instrument.profile_by_distance(values, values.copy(), distance, edges)
    assert rows, "the profile produced no bins at all"
    assert all(row["rms"] == 0.0 for row in rows)
    assert all(row["max_abs"] == 0.0 for row in rows)
    assert all(row["correlation"] == pytest.approx(1.0) for row in rows)


def test_a_boundary_localised_difference_is_recovered_with_its_decay():
    """The false-negative direction that decides the programme.

    A perturbation that decays away from the boundary is exactly the signature
    of boundary contamination.  If the instrument reported it flat, the lane
    would say no ladder is needed while the measurement said one was.
    """

    patch, driven = _patch_geometry()
    distance = instrument.boundary_distance_km(patch, driven)
    edges = instrument.bin_edges_for(distance, 25.0)
    reference = np.zeros(patch.shape[0])
    contaminated = np.exp(-distance / 60.0)
    rows = instrument.profile_by_distance(contaminated, reference, distance, edges)
    assert len(rows) >= 3
    values = [row["rms"] for row in rows]
    assert values[0] > values[-1] * 5.0, values
    assert all(a >= b for a, b in zip(values, values[1:])), values


def test_a_uniform_difference_reads_flat():
    """The false-positive direction: a gradient must not be manufactured.

    The bins hold different numbers of cells and cover different areas.  If
    the reported statistic were sensitive to that -- a sum rather than a mean,
    say -- a perfectly uniform disagreement would be reported as a boundary
    effect and a user would be sold a ladder that fixes nothing.
    """

    patch, driven = _patch_geometry()
    distance = instrument.boundary_distance_km(patch, driven)
    edges = instrument.bin_edges_for(distance, 25.0)
    reference = np.zeros(patch.shape[0])
    offset = np.full(patch.shape[0], 0.25)
    rows = instrument.profile_by_distance(offset, reference, distance, edges)
    assert len(rows) >= 3
    assert all(row["rms"] == pytest.approx(0.25) for row in rows), rows


def test_the_bins_partition_the_patch_exactly_once():
    """No cell counted twice and none dropped, including the furthest one.

    The last bin is closed at its upper edge and every other is half open.
    Getting that wrong loses the single furthest cell -- the one deepest in the
    interior, and so the most load-bearing point on the whole chart.
    """

    patch, driven = _patch_geometry(cells=311, span_deg=3.3)
    distance = instrument.boundary_distance_km(patch, driven)
    edges = instrument.bin_edges_for(distance, 25.0)
    values = np.zeros(patch.shape[0])
    rows = instrument.profile_by_distance(values, values, distance, edges)
    assert sum(row["cells"] for row in rows) == patch.shape[0]


def test_a_shape_mismatch_is_refused_rather_than_broadcast():
    """Numpy would broadcast a (cells,) against a (cells, 1) silently.

    That comparison is meaningless and it does not raise: it returns a number,
    and the receipt would carry it as a measurement.
    """

    patch, driven = _patch_geometry(cells=32, span_deg=0.4)
    distance = instrument.boundary_distance_km(patch, driven)
    edges = instrument.bin_edges_for(distance, 25.0)
    with pytest.raises(SystemExit):
        instrument.profile_by_distance(
            np.zeros((32, 1)), np.zeros(32), distance, edges
        )


def test_an_arm_with_no_driven_zone_is_refused_by_name():
    """A global run cannot be the patch arm, and saying so is the whole point.

    The patch arm's boundary IS the distance axis.  Handed a global run, a
    silent fallback would produce an axis of zeros and every curve would land
    in one bin, which reads as perfect agreement everywhere.
    """

    patch, _ = _patch_geometry(cells=16, span_deg=0.2)
    with pytest.raises(SystemExit) as failure:
        instrument.boundary_distance_km(patch, np.empty((0, 3)))
    assert "bdyMaskCell" in str(failure.value)


# --------------------------------------------------------------------------
# The other half of the instrument: the arms themselves have to be nested
# copies of one shape at different sizes, or the comparison is between two
# different pieces of ground.
# --------------------------------------------------------------------------


def _ring(centre_lat: float, centre_lon: float, radius_deg: float, points: int = 24):
    angles = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)
    return {
        "kind": "polygon",
        "vertices_deg": [
            [
                centre_lat + radius_deg * float(np.cos(a)),
                centre_lon
                + radius_deg * float(np.sin(a)) / float(np.cos(np.radians(centre_lat))),
            ]
            for a in angles
        ],
    }


def test_scaling_a_region_scales_every_arc_by_the_factor():
    """The arms must be similar shapes, not merely smaller ones.

    Scaling degrees instead of arcs would squash the polygon in longitude by
    ``cos(lat)`` -- at 60 S, by half -- so the small arm would be a different
    shape over different ground and any disagreement between arms could be
    that instead of the boundary.
    """

    region = _ring(-60.0, 139.5, 5.0)
    centre = scaler.centroid_of(region)
    before = scaler.region_metrics(region, centre)
    for factor in (0.45, 0.7):
        after = scaler.region_metrics(scaler.scale_region(region, factor), centre)
        ratio = (
            after["centroid_to_vertex_km"]["mean"]
            / before["centroid_to_vertex_km"]["mean"]
        )
        assert ratio == pytest.approx(factor, rel=1e-12)
        # EVERY arc, not just the mean: a transform that scaled the mean while
        # redistributing the vertices would pass a mean test and hand the
        # comparison two different shapes.
        for a, b in zip(
            (after["centroid_to_vertex_km"][k] for k in ("min", "max")),
            (before["centroid_to_vertex_km"][k] for k in ("min", "max")),
        ):
            assert a == pytest.approx(b * factor, rel=1e-12)


def test_the_scaled_ring_stays_concentric_to_within_metres():
    """The arms have to be concentric or they do not share a patch.

    Exactly about the centre it was given, and that is asserted above.  Its
    OWN recomputed centroid moves a little, because a spherical centroid is
    the normalised mean of vertex directions and that mean is preserved under
    contraction only for a shape symmetric about it -- which a track's flared
    swath ring is not.  What matters for the comparison is that the drift is
    far smaller than a cell, so the arms cover the same ground: this bounds it
    at 100 m against a 4.6 km cell.
    """

    region = _ring(-60.0, 139.5, 5.0)
    before = scaler.region_metrics(region)["centroid_deg"]
    after = scaler.region_metrics(scaler.scale_region(region, 0.45))["centroid_deg"]
    drift_km = scaler.region_metrics(
        {"kind": "polygon", "vertices_deg": [after]},
        scaler._unit(before[0], before[1]),
    )["centroid_to_vertex_km"]["mean"]
    assert drift_km < 0.1, f"centroid moved {drift_km * 1000:.1f} m"


def test_a_factor_of_one_is_the_identity():
    """The unscaled arm goes through the same command as the scaled ones.

    If it did not, one arm of the comparison would be produced by a different
    route from the others, and any difference between it and them could be the
    route.
    """

    region = _ring(-60.0, 139.5, 5.0)
    scaled = scaler.scale_region(region, 1.0)
    for original, produced in zip(region["vertices_deg"], scaled["vertices_deg"]):
        assert produced[0] == pytest.approx(original[0], abs=1e-9)
        assert produced[1] == pytest.approx(original[1], abs=1e-9)


def test_an_unscalable_region_kind_is_refused_rather_than_passed_through():
    """Returning it unchanged would make two arms the same domain.

    Two identical domains agree perfectly, which is the verdict this lane is
    looking for -- so the failure mode is a false PASS on the exact question
    being asked.
    """

    with pytest.raises(SystemExit) as failure:
        scaler.scale_region({"kind": "global"}, 0.5)
    assert "proves nothing" in str(failure.value)
