"""Spherical geometry for a track-shaped, flared swath.

WHY THIS EMITS A POLYGON AND NOT A NEW SHAPE KIND.  ``rw_mpas_mesh``
already reads three region shapes as DATA -- ``cap``, ``lat_lon_box`` and
``polygon`` -- and the same three rows are what ``rw_mpas_mesh
--cull-parent --region`` accepts for a limited-area cull
(``rw-mpas/src/mesh/cull.rs::boundary_from_shape``).  A ring of
``[lat, lon]`` vertices is therefore ALREADY a swath: the generator
computes exact great-circle distances to its segments and the culler walks
its boundary.  Adding a fourth shape kind for "swath" would put a code
path where a row does, which is the thing the arbitrary acceptance test
exists to stop.  So this module's whole job is to turn (a track polyline,
a half-width profile) into that ring, and to refuse the rings that are not
swaths.

The perpendicular offset is exact for a polygon and needs no Lipschitz
correction.  A correction is needed when a flared swath is expressed as a
signed-distance FIELD -- the boundary of ``distance_to_axis < w(s)`` is
not perpendicular to the axis where ``w`` grows -- but the ring emitted
here IS the boundary, so the generator's own segment distance is the
right answer by construction.

Angles are DEGREES at this module's surface, because that is the unit the
mesh spec's ``center_deg`` / ``vertices_deg`` rows are written in and a
conversion at the boundary is one place to be wrong instead of many.  The
port's mesh arrays are radians (MPAS Registry.xml 1225-1228) and the
detector converts once, on read.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .errors import SwathRefusal

#: The sphere every figure here is measured on.  Same value the observation
#: referee aligns on (``hexcore.obs_referee.align.EARTH_RADIUS_KM``); it is
#: restated rather than imported so this module carries no dependency on the
#: referee's numpy/scipy stack, and a test asserts the two agree.
EARTH_RADIUS_KM = 6371.0

#: Hexagon area as a multiple of the across-flats spacing squared.  A Voronoi
#: cell of a Goldberg mesh at spacing ``h`` covers ``sqrt(3)/2 * h**2``; this
#: is the same relation the generator's own sizing integral uses, and it is
#: the ONLY cell-count arithmetic in this package.  Every cell figure derived
#: from it is labelled ``area_integral`` and never ``measured``.
HEXAGON_AREA_FACTOR = 0.5 * math.sqrt(3.0)

LatLon = tuple[float, float]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _unit(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    cos_lat = math.cos(lat)
    return (cos_lat * math.cos(lon), cos_lat * math.sin(lon), math.sin(lat))


def _normalize_longitude(lon_deg: float) -> float:
    """Fold into ``[-180, 180)`` exactly, the same half-open convention
    :func:`hexcore.mesh.normalize_longitudes` uses for the mesh arrays."""

    folded = math.fmod(lon_deg + 180.0, 360.0)
    if folded < 0.0:
        folded += 360.0
    return folded - 180.0


def great_circle_km(
    lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float
) -> float:
    """Great-circle distance in km, by the haversine form.

    The haversine form rather than the spherical law of cosines because
    the swath widths this layer works in (100-400 km) are small angles
    where the cosine form loses digits, and a half-width that is wrong by
    a kilometre puts the ring's vertex in the wrong cell.
    """

    phi_a = math.radians(lat_a_deg)
    phi_b = math.radians(lat_b_deg)
    d_phi = phi_b - phi_a
    d_lam = math.radians(lon_b_deg - lon_a_deg)
    h = (
        math.sin(0.5 * d_phi) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(0.5 * d_lam) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def initial_bearing_deg(
    lat_a_deg: float, lon_a_deg: float, lat_b_deg: float, lon_b_deg: float
) -> float:
    """Forward azimuth at ``a`` along the great circle to ``b``, in
    ``[0, 360)`` degrees clockwise from north."""

    phi_a = math.radians(lat_a_deg)
    phi_b = math.radians(lat_b_deg)
    d_lam = math.radians(lon_b_deg - lon_a_deg)
    y = math.sin(d_lam) * math.cos(phi_b)
    x = math.cos(phi_a) * math.sin(phi_b) - math.sin(phi_a) * math.cos(phi_b) * math.cos(d_lam)
    return math.degrees(math.atan2(y, x)) % 360.0


def destination(
    lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float
) -> LatLon:
    """The point ``distance_km`` from here along ``bearing_deg``.

    Latitude comes back through ``atan2(z, hypot(x, y))`` rather than
    ``asin(z)``: the generator's own writer was moved off ``asin`` on
    2026-08-26 after a mesh emitted a latitude near the south pole that
    reconstructed its own Cartesian point only to 1.27e-9, and a swath
    vertex placed by this function feeds the same pipeline.
    """

    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    theta = math.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM
    sin_phi = math.sin(phi) * math.cos(delta) + math.cos(phi) * math.sin(delta) * math.cos(theta)
    cos_phi_x = math.cos(phi) * math.cos(delta) - math.sin(phi) * math.sin(delta) * math.cos(theta)
    cos_phi_y = math.sin(delta) * math.sin(theta)
    out_lat = math.degrees(math.atan2(sin_phi, math.hypot(cos_phi_x, cos_phi_y)))
    out_lon = math.degrees(lam + math.atan2(cos_phi_y, cos_phi_x))
    return (out_lat, _normalize_longitude(out_lon))


def midpoint(a: LatLon, b: LatLon) -> LatLon:
    """The great-circle midpoint, used to densify a long track leg."""

    ax, ay, az = _unit(*a)
    bx, by, bz = _unit(*b)
    x, y, z = ax + bx, ay + by, az + bz
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        raise SwathRefusal(
            f"track points {a} and {b} are antipodal, so the leg between them has no "
            "unique great circle and no unique perpendicular; a swath cannot be "
            "offset from an undefined axis"
        )
    return (
        math.degrees(math.atan2(z / norm, math.hypot(x / norm, y / norm))),
        _normalize_longitude(math.degrees(math.atan2(y / norm, x / norm))),
    )


# ---------------------------------------------------------------------------
# the track
# ---------------------------------------------------------------------------
def resample_track(points: Sequence[LatLon], *, step_km: float) -> list[LatLon]:
    """Densify a polyline so no leg exceeds ``step_km``, keeping every
    original vertex.

    A swath ring built from two far-apart track points would have its
    sides follow ONE great circle while the axis follows another, so the
    reported half-width would be right only at the ends.  Densifying first
    keeps the offset perpendicular everywhere it matters.
    """

    if step_km <= 0.0:
        raise SwathRefusal(f"resample step must be positive, not {step_km}")
    if len(points) < 2:
        return [tuple(point) for point in points]
    out: list[LatLon] = [tuple(points[0])]
    for start, end in zip(points, points[1:]):
        leg = great_circle_km(start[0], start[1], end[0], end[1])
        divisions = max(1, math.ceil(leg / step_km))
        current = tuple(start)
        for index in range(1, divisions + 1):
            if index == divisions:
                nxt = tuple(end)
            else:
                fraction = index / divisions
                nxt = _interpolate(tuple(start), tuple(end), fraction)
            out.append(nxt)
            current = nxt
        del current
    return out


def _interpolate(a: LatLon, b: LatLon, fraction: float) -> LatLon:
    ax, ay, az = _unit(*a)
    bx, by, bz = _unit(*b)
    dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
    omega = math.acos(dot)
    if omega < 1e-12:
        return a
    sin_omega = math.sin(omega)
    wa = math.sin((1.0 - fraction) * omega) / sin_omega
    wb = math.sin(fraction * omega) / sin_omega
    x, y, z = wa * ax + wb * bx, wa * ay + wb * by, wa * az + wb * bz
    norm = math.sqrt(x * x + y * y + z * z)
    return (
        math.degrees(math.atan2(z / norm, math.hypot(x / norm, y / norm))),
        _normalize_longitude(math.degrees(math.atan2(y / norm, x / norm))),
    )


def _axis_bearings(path: Sequence[LatLon]) -> list[float]:
    """One bearing per track point: the segment bearing at the ends, the
    angular bisector of the two adjacent segments inside."""

    segment = [
        initial_bearing_deg(a[0], a[1], b[0], b[1]) for a, b in zip(path, path[1:])
    ]
    bearings = [segment[0]]
    for before, after in zip(segment, segment[1:]):
        delta = ((after - before + 180.0) % 360.0) - 180.0
        bearings.append((before + 0.5 * delta) % 360.0)
    bearings.append(segment[-1])
    return bearings


# ---------------------------------------------------------------------------
# the ring
# ---------------------------------------------------------------------------
def swath_ring(
    path: Sequence[LatLon],
    half_widths_km: Sequence[float],
    *,
    cap_points: int = 16,
) -> list[LatLon]:
    """The closed boundary ring of a flared swath, as ``[lat, lon]`` pairs.

    Wound so the swath interior is enclosed once.  The first vertex is NOT
    repeated at the end: ``rw_mpas_mesh`` closes the ring itself
    (``density.rs`` indexes ``(k + 1) % ring.len()``), and a duplicated
    vertex makes a zero-length segment whose great circle is undefined.
    """

    if len(path) < 2:
        raise SwathRefusal(
            "a swath needs at least two track points: one point has no direction, "
            "so there is no perpendicular to offset the half-width along. A "
            "stationary feature is placed with a cap region, not a swath"
        )
    if len(half_widths_km) != len(path):
        raise SwathRefusal(
            f"the half-width profile has {len(half_widths_km)} entries for "
            f"{len(path)} track points; every track point carries its own "
            "half-width because that is what makes the swath flare"
        )
    for index, width in enumerate(half_widths_km):
        if not math.isfinite(width) or width <= 0.0:
            raise SwathRefusal(
                f"half-width {width} km at track point {index} is not positive; a "
                "zero or negative half-width collapses the ring onto its own axis "
                "and the generator would refine a line of zero area"
            )
    if cap_points < 2:
        raise SwathRefusal(
            f"cap_points={cap_points} cannot describe an end cap; the cap is what "
            "gives the swath its rounded nose, and 2 points make it a flat chord"
        )

    bearings = _axis_bearings(path)
    left = _forward_only(
        [
            destination(point[0], point[1], bearing - 90.0, width)
            for point, bearing, width in zip(path, bearings, half_widths_km)
        ],
        bearings,
    )
    right = _forward_only(
        [
            destination(point[0], point[1], bearing + 90.0, width)
            for point, bearing, width in zip(path, bearings, half_widths_km)
        ],
        bearings,
    )

    ring: list[LatLon] = list(left)
    ring.extend(
        _cap(path[-1], half_widths_km[-1], bearings[-1] - 90.0, bearings[-1] + 90.0, cap_points)
    )
    ring.extend(reversed(right))
    ring.extend(
        _cap(path[0], half_widths_km[0], bearings[0] + 90.0, bearings[0] + 270.0, cap_points)
    )

    if _self_intersects(ring):
        raise SwathRefusal(
            "the swath ring would self-intersect: the track turns inside its own "
            f"half-width (max half-width {max(half_widths_km):.1f} km over a track "
            f"of {track_length_km(path):.1f} km). A self-crossing ring has no "
            "single interior, so the generator's winding test and the culler's "
            "flood fill would disagree about which cells are in the region. Widen "
            "the track's turn, shorten the lead, or reduce the half-width"
        )
    return ring


def _forward_only(offsets: Sequence[LatLon], bearings: Sequence[float]) -> list[LatLon]:
    """Drop offset vertices that fold back on the axis.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, the placement chain
    on the real x1.40962 parent): where the axis turns tighter than the
    local half-width, the offset on the INSIDE of the bend overshoots its
    neighbours and the side chain reverses.  On the first emitted swath
    the left chain turned -171.8 degrees at one vertex -- a near-complete
    fold -- and the ring carried a zero-area sliver.  ``_self_intersects``
    does not catch it because a fold is a COLLINEAR overlap, not a
    transversal crossing, so the orientation test reads the same sign on
    both sides and passes.  The culler still produced a valid bounded disk
    from that ring, which is worse than a refusal: the defect is invisible
    downstream and shows up only as a strange cell pattern inside the
    finest grid in the cascade.

    The rule is local and cheap: a step along a side chain must have a
    forward component along the axis.  The two ENDS are always kept --
    they are what carries the half-width the receipt quotes -- and
    interior vertices are popped until the step to the end is forward
    again, which is the standard monotone-chain cleanup for an offset
    curve.
    """

    if len(offsets) < 3:
        return [tuple(point) for point in offsets]
    kept: list[LatLon] = [tuple(offsets[0])]
    kept_at: list[int] = [0]
    for index in range(1, len(offsets)):
        candidate = tuple(offsets[index])
        while kept:
            step = great_circle_km(kept[-1][0], kept[-1][1], candidate[0], candidate[1])
            if step <= 1e-9:
                break
            heading = initial_bearing_deg(
                kept[-1][0], kept[-1][1], candidate[0], candidate[1]
            )
            reference = bearings[kept_at[-1]]
            delta = abs(((heading - reference + 180.0) % 360.0) - 180.0)
            if delta < 90.0:
                break
            if len(kept) == 1:
                # The first vertex is an end and is never dropped; the
                # candidate is the one that folded, so skip it instead.
                candidate = None  # type: ignore[assignment]
                break
            kept.pop()
            kept_at.pop()
        if candidate is None:  # type: ignore[comparison-overlap]
            if index == len(offsets) - 1:
                kept.append(tuple(offsets[index]))
                kept_at.append(index)
            continue
        if not kept or great_circle_km(kept[-1][0], kept[-1][1], candidate[0], candidate[1]) > 1e-9:
            kept.append(candidate)
            kept_at.append(index)
    if kept[-1] != tuple(offsets[-1]):
        kept.append(tuple(offsets[-1]))
    return kept


def _cap(
    centre: LatLon, radius_km: float, from_bearing: float, to_bearing: float, points: int
) -> list[LatLon]:
    """Arc of ``points`` samples sweeping clockwise from one bearing to the
    other, excluding both endpoints (the sides already carry them)."""

    span = (to_bearing - from_bearing) % 360.0
    return [
        destination(centre[0], centre[1], from_bearing + span * index / points, radius_km)
        for index in range(1, points)
    ]


def smooth_axis(path: Sequence[LatLon], *, passes: int, weight: float = 0.5) -> list[LatLon]:
    """Laplacian-smooth the interior of a track axis, ends pinned.

    Averaged as unit vectors, so a track crossing the antimeridian is
    smoothed rather than folded through the middle of the planet.
    """

    points = [tuple(point) for point in path]
    if len(points) < 3 or passes <= 0:
        return points
    for _ in range(passes):
        vectors = [_unit(*point) for point in points]
        updated = [points[0]]
        for index in range(1, len(points) - 1):
            before, here, after = vectors[index - 1], vectors[index], vectors[index + 1]
            x = (1.0 - weight) * here[0] + 0.5 * weight * (before[0] + after[0])
            y = (1.0 - weight) * here[1] + 0.5 * weight * (before[1] + after[1])
            z = (1.0 - weight) * here[2] + 0.5 * weight * (before[2] + after[2])
            norm = math.sqrt(x * x + y * y + z * z)
            if norm <= 0.0:
                updated.append(points[index])
                continue
            updated.append(
                (
                    math.degrees(math.atan2(z / norm, math.hypot(x / norm, y / norm))),
                    _normalize_longitude(math.degrees(math.atan2(y / norm, x / norm))),
                )
            )
        updated.append(points[-1])
        points = updated
    return points


def fit_swath_axis(
    path: Sequence[LatLon],
    half_widths_km: Sequence[float],
    *,
    cap_points: int,
    maximum_passes: int = 12,
) -> tuple[list[LatLon], list[LatLon], int, float]:
    """Smooth the axis just enough to make a swath of it, and say how much.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, the placement fixture
    at 40,962 cells): a detected centre is quantized by the parent mesh and
    by its own anomaly weighting, so a real track wanders about a smooth
    path by tens of kilometres frame to frame.  Offsetting a half-width of
    150-300 km perpendicular to a path that reverses direction by 39 km
    makes the two sides cross, and ``swath_ring`` correctly refuses it --
    which would decline the strongest storm in the cycle for being detected
    too precisely.  The jitter is far smaller than the half-width that has
    to contain it, so smoothing the AXIS changes which cells are refined by
    less than the flare already allows, while turning a refused candidate
    into a placed one.

    Returns ``(axis, ring, passes, drift_km)``.  ``drift_km`` is how far the
    smoothing moved the axis at its worst point, and it is published: a
    reader must be able to see that the swath is still on the storm.  The
    pass count is bounded, and exhausting it re-raises the ring's own
    refusal rather than smoothing a genuinely hairpin track into a lie.
    """

    original = [tuple(point) for point in path]
    for passes in range(maximum_passes + 1):
        axis = smooth_axis(original, passes=passes)
        try:
            ring = swath_ring(axis, half_widths_km, cap_points=cap_points)
        except SwathRefusal:
            if passes == maximum_passes:
                raise
            continue
        drift = max(
            (
                great_circle_km(a[0], a[1], b[0], b[1])
                for a, b in zip(original, axis)
            ),
            default=0.0,
        )
        return axis, ring, passes, drift
    raise AssertionError("unreachable: the loop either returns or re-raises")


def track_length_km(path: Sequence[LatLon]) -> float:
    return sum(
        great_circle_km(a[0], a[1], b[0], b[1]) for a, b in zip(path, path[1:])
    )


def _self_intersects(ring: Sequence[LatLon]) -> bool:
    """Does any pair of non-adjacent ring segments cross?

    Tested in the local tangent plane of the ring's own centroid.  The
    rings this layer emits span at most a few thousand kilometres, where a
    gnomonic projection about their centroid maps great-circle segments to
    straight lines exactly -- so this is not an approximation of the
    spherical test, it is the spherical test in coordinates that make it
    cheap.  Points more than 90 degrees from the centroid have no gnomonic
    image; a ring that wide is reported as intersecting rather than
    silently passed, because it is not a swath either.
    """

    count = len(ring)
    if count < 4:
        return False
    centre = _centroid(ring)
    projected: list[tuple[float, float]] = []
    for lat, lon in ring:
        image = _gnomonic(centre, (lat, lon))
        if image is None:
            return True
        projected.append(image)
    for i in range(count):
        a1 = projected[i]
        a2 = projected[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i or (j + 1) % count == i or j == (i + 1) % count:
                continue
            b1 = projected[j]
            b2 = projected[(j + 1) % count]
            if _segments_cross(a1, a2, b1, b2):
                return True
    return False


def _centroid(ring: Sequence[LatLon]) -> tuple[float, float, float]:
    x = y = z = 0.0
    for lat, lon in ring:
        px, py, pz = _unit(lat, lon)
        x += px
        y += py
        z += pz
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        return (0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm)


def dilate_ring(ring: Sequence[LatLon], scale: float) -> list[LatLon]:
    """Scale a ring about its own centroid, on the sphere, by ``scale``.

    WHAT THIS IS FOR.  A limited-area cull of a VARIABLE-RESOLUTION parent
    inherits whatever the parent's mesh does between the cut and the fine
    core.  Cutting at the swath ring itself throws the parent's coarsening
    ramp away, so the boundary stream lands a 71 km parent state on cells the
    fine core's own size.  Cutting WIDER keeps some of that ramp, and the ramp
    is model-solved atmosphere at intermediate resolution -- an intermediate
    level, in the form MPAS has rather than the form WRF has.  How much ramp
    to keep is what this scales, and it is a number in a row.

    THE SCALING IS GEODESIC, NOT IN DEGREES.  Scaling latitude and longitude
    by a factor is a different shape at every latitude; at 60 S it would
    stretch the ring by two in longitude relative to latitude.  Each vertex is
    carried to a unit vector and moved along its great circle to the centroid
    direction, so every centroid-to-vertex arc scales by exactly ``scale`` and
    the ring stays a similar figure over concentric ground.

    ``scale == 1.0`` returns the ring unchanged, so the unscaled case goes
    through the same code as every other and no configuration is special.
    """

    if not (scale > 0.0) or scale != scale or scale in (float("inf"),):
        raise SwathRefusal(
            f"SWATH-DILATION: scale is {scale!r}; it must be a finite positive "
            "number. A cull region is a piece of the sphere and there is no "
            "meaning to scaling it by zero or less"
        )
    if scale == 1.0:
        return [(float(lat), float(lon)) for lat, lon in ring]
    centre = _centroid(ring)
    out: list[LatLon] = []
    for lat, lon in ring:
        vertex = _unit(lat, lon)
        dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(vertex, centre))))
        omega = math.acos(dot)
        if omega < 1e-12:
            out.append((float(lat), float(lon)))
            continue
        if scale * omega >= math.pi:
            raise SwathRefusal(
                f"SWATH-DILATION: scaling by {scale} carries a vertex "
                f"{math.degrees(scale * omega):.1f} degrees from the region's "
                "centroid, at or past the antipode. A ring on a sphere bounds "
                "two discs and the region is the smaller one, so a ring this "
                "wide names the complement of what was asked for"
            )
        sin_omega = math.sin(omega)
        near = math.sin(scale * omega) / sin_omega
        far = math.sin((1.0 - scale) * omega) / sin_omega
        x = near * vertex[0] + far * centre[0]
        y = near * vertex[1] + far * centre[1]
        z = near * vertex[2] + far * centre[2]
        norm = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / norm, y / norm, z / norm
        out.append(
            (
                math.degrees(math.asin(max(-1.0, min(1.0, z)))),
                math.degrees(math.atan2(y, x)),
            )
        )
    return out


def dilate_shape(shape: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """Scale a Shape row about its own centre, keeping its kind.

    The three kinds ``rw_mpas_mesh --region`` reads are all scalable and all
    are scaled here, so a threat row that emits a cap because its track stood
    still gets the same treatment as one that emits a swath polygon.  An
    unknown kind is REFUSED rather than passed through unscaled: an unscaled
    cull region silently ignores the row's declared pad, and the run that
    results looks exactly like the run that was asked for.
    """

    kind = shape.get("kind")
    if kind == "polygon":
        out = dict(shape)
        out["vertices_deg"] = [
            [lat, lon]
            for lat, lon in dilate_ring(
                [(float(a), float(b)) for a, b in shape["vertices_deg"]], scale
            )
        ]
        return out
    if kind == "cap":
        out = dict(shape)
        out["radius_km"] = float(shape["radius_km"]) * scale
        return out
    if kind == "lat_lon_box":
        out = dict(shape)
        for axis in ("lat_deg", "lon_deg"):
            low, high = (float(v) for v in shape[axis])
            middle = 0.5 * (low + high)
            half = 0.5 * (high - low) * scale
            out[axis] = [middle - half, middle + half]
        return out
    raise SwathRefusal(
        f"SWATH-DILATION: shape kind {kind!r} cannot be scaled. Returning it "
        "unscaled would silently ignore the row's declared cull pad, and the "
        "resulting run would be indistinguishable from the one that was asked "
        "for"
    )


def _gnomonic(
    centre: tuple[float, float, float], point: LatLon
) -> tuple[float, float] | None:
    px, py, pz = _unit(*point)
    cx, cy, cz = centre
    cos_c = px * cx + py * cy + pz * cz
    if cos_c <= 1e-9:
        return None
    # Local east/north basis at the centre.
    lat_c = math.asin(max(-1.0, min(1.0, cz)))
    lon_c = math.atan2(cy, cx)
    east = (-math.sin(lon_c), math.cos(lon_c), 0.0)
    north = (
        -math.sin(lat_c) * math.cos(lon_c),
        -math.sin(lat_c) * math.sin(lon_c),
        math.cos(lat_c),
    )
    return (
        (px * east[0] + py * east[1] + pz * east[2]) / cos_c,
        (px * north[0] + py * north[1] + pz * north[2]) / cos_c,
    )


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    d1 = _orientation(a1, a2, b1)
    d2 = _orientation(a1, a2, b2)
    d3 = _orientation(b1, b2, a1)
    d4 = _orientation(b1, b2, a2)
    return ((d1 > 0.0) != (d2 > 0.0)) and ((d3 > 0.0) != (d4 > 0.0))


# ---------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------
def polygon_area_km2(ring: Sequence[LatLon]) -> float:
    """Spherical polygon area by the signed spherical excess.

    Absolute value, because the mesh spec accepts a ring wound either way
    and an area that flips sign with the winding would make the cell
    prediction negative for half the legal documents.
    """

    if len(ring) < 3:
        return 0.0
    total = 0.0
    for a, b in zip(ring, list(ring[1:]) + [ring[0]]):
        lam_a = math.radians(a[1])
        lam_b = math.radians(b[1])
        phi_a = math.radians(a[0])
        phi_b = math.radians(b[0])
        d_lam = ((lam_b - lam_a + math.pi) % (2.0 * math.pi)) - math.pi
        total += d_lam * (2.0 + math.sin(phi_a) + math.sin(phi_b))
    return abs(0.5 * total) * EARTH_RADIUS_KM * EARTH_RADIUS_KM


def polygon_perimeter_km(ring: Sequence[LatLon]) -> float:
    if len(ring) < 2:
        return 0.0
    return sum(
        great_circle_km(a[0], a[1], b[0], b[1])
        for a, b in zip(ring, list(ring[1:]) + [ring[0]])
    )


#: How many boundary rings a limited-area cull carries outside the region
#: it was asked for.  MPAS's ``bdyMask`` runs 1..7 and the culler reproduces
#: it exactly (measured: ``ring_cell_counts`` has eight entries, index 0
#: being the free interior).  The halo is real cells that a card must hold,
#: so a cell estimate that ignores it under-counts a limited-area mesh by
#: the whole zone -- measured at 39 interior cells against 354 total on a
#: 120 km parent, an 89 % shortfall.
BOUNDARY_RINGS = 7


def predicted_cells_in(
    ring: Sequence[LatLon], *, spacing_km: float, boundary_rings: int = BOUNDARY_RINGS
) -> float:
    """Cells a limited-area mesh at ``spacing_km`` holds for this ring.

    ``basis: area_integral``.  Arithmetic, not a measurement, and every
    receipt that carries it says so -- but arithmetic with a MEASURED error
    bound: ``evidence/swath-following-20260826/`` culls a family of regions
    out of the real published ``x1.40962`` parent with the shipped culler
    and records this function's error against the culler's own
    ``region_cells`` as a function of how many cells span the region.

    The area is the ring DILATED by the boundary zone -- a Minkowski sum
    with a disc of ``boundary_rings * spacing`` -- because the cull emits
    the interior plus seven rings and every one of those cells is
    integrated.  The dilation term is exact for a convex ring while the
    disc is small against the ring's own half-width, and it over-counts
    once the disc is comparable to it (opposite sides' halos overlap).  At
    the shipped operating point -- a 150-300 km half-width at a few
    kilometres' spacing -- the disc is under 30 km and the regime is the
    good one; the evidence directory publishes the curve rather than a
    single number so the regime is visible.
    """

    if spacing_km <= 0.0:
        raise SwathRefusal(f"spacing must be positive, not {spacing_km} km")
    area = polygon_area_km2(ring)
    if boundary_rings > 0:
        halo = boundary_rings * spacing_km
        area += polygon_perimeter_km(ring) * halo + math.pi * halo * halo
    return area / (HEXAGON_AREA_FACTOR * spacing_km * spacing_km)


def ring_overlap_fraction(a: Sequence[LatLon], b: Sequence[LatLon], *, samples: int = 4096) -> float:
    """Fraction of ``a``'s area also inside ``b``, by a deterministic
    lattice over ``a``'s bounding cap.

    Used by the hysteresis rule to decide whether a moved swath has moved
    ENOUGH to be worth regenerating a mesh for.  A lattice rather than an
    exact spherical clip because the answer feeds a threshold comparison
    at 0.70, where a 1/4096 quadrature error cannot change the decision,
    and an exact spherical polygon clipper is a large surface to be wrong
    in for a number used only against a coarse threshold.
    """

    if not a or not b:
        return 0.0
    centre = _centroid(a)
    radius = max(
        _arc_from_unit(centre, _unit(lat, lon)) for lat, lon in a
    )
    if radius <= 0.0:
        return 0.0
    inside_a_test = ring_containment(a)
    inside_b_test = ring_containment(b)
    inside_a = 0
    inside_both = 0
    for point in _cap_lattice(centre, radius, samples):
        if inside_a_test(point):
            inside_a += 1
            if inside_b_test(point):
                inside_both += 1
    if inside_a == 0:
        return 0.0
    return inside_both / inside_a


def _arc_from_unit(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.acos(dot)


def _cap_lattice(
    centre: tuple[float, float, float], radius_rad: float, samples: int
) -> Iterable[LatLon]:
    """A golden-angle spiral inside a cap: equal-area, deterministic, and
    the same construction the generator uses for its own lattice search."""

    golden = math.pi * (3.0 - math.sqrt(5.0))
    lat_c = math.asin(max(-1.0, min(1.0, centre[2])))
    lon_c = math.atan2(centre[1], centre[0])
    east = (-math.sin(lon_c), math.cos(lon_c), 0.0)
    north = (
        -math.sin(lat_c) * math.cos(lon_c),
        -math.sin(lat_c) * math.sin(lon_c),
        math.cos(lat_c),
    )
    cos_r = math.cos(radius_rad)
    for index in range(samples):
        z = 1.0 - (index + 0.5) / samples * (1.0 - cos_r)
        rho = math.sqrt(max(0.0, 1.0 - z * z))
        angle = index * golden
        dx = rho * math.cos(angle)
        dy = rho * math.sin(angle)
        x = z * centre[0] + dx * east[0] + dy * north[0]
        y = z * centre[1] + dx * east[1] + dy * north[1]
        w = z * centre[2] + dx * east[2] + dy * north[2]
        norm = math.sqrt(x * x + y * y + w * w)
        yield (
            math.degrees(math.atan2(w / norm, math.hypot(x / norm, y / norm))),
            _normalize_longitude(math.degrees(math.atan2(y / norm, x / norm))),
        )


def ring_containment(ring: Sequence[LatLon]):
    """A point-in-ring test for the SMALLER of the two regions the ring
    bounds, as a callable that pays the ring's setup once.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, ``rw_mpas_mesh``
    0.1.0 staged in ``~/.gpuwm/bridges``, receipt in
    ``evidence/swath-following-20260826/``): a closed ring divides a SPHERE
    into two discs and the winding number is +/-2*pi in BOTH -- it is
    -2*pi in the complement, not 0.  A containment test that accepts on
    ``abs(winding) > pi`` therefore calls the complement interior as well.
    The generator's own ``polygon_contains``
    (``rw-mpas/src/mesh/density.rs``) does exactly that, and the visible
    consequence is measured: its ``region_attainment`` lattice search
    finds a "deepest interior point" 19,683 km inside a 4-degree box, so
    every polygon region reports its request exactly met -- a 44 km
    polygon reads ``attained_spacing_km 4.00`` where a 22 km cap of the
    same request correctly reads 6.36.  the ruling is to quote ATTAINED
    spacing and never requested; a number that is structurally always the
    request cannot satisfy it.  The generator's density FIELD is
    unaffected (a 0.4-degree polygon adds 607 cells over uniform, not a
    hemisphere's worth), so this is a reporting defect and the emitted
    meshes are correct.  This layer therefore emits polygons freely and
    never consumes the generator's polygon attainment.

    The fix here is the sign: the winding at the ring's own vertex
    centroid fixes which sign means "the small side", and every other
    point is compared against it.
    """

    units = [_unit(lat, lon) for lat, lon in ring]
    centre = _centroid(ring)
    reference = _winding(centre, units)
    sign = 1.0 if reference >= 0.0 else -1.0
    if abs(reference) <= math.pi:
        # The vertex centroid is not inside the ring -- a strongly
        # non-convex ring can do this.  Fall back to the winding magnitude
        # alone, which is the generator's own convention, and say so.
        def _magnitude_only(point: LatLon) -> bool:
            return abs(_winding(_unit(*point), units)) > math.pi

        return _magnitude_only

    def _signed(point: LatLon) -> bool:
        total = _winding(_unit(*point), units)
        return abs(total) > math.pi and (total > 0.0) == (sign > 0.0)

    return _signed


def _winding(
    point: tuple[float, float, float], units: Sequence[tuple[float, float, float]]
) -> float:
    px, py, pz = point
    total = 0.0
    count = len(units)
    for index in range(count):
        a = _tangent(px, py, pz, units[index])
        b = _tangent(px, py, pz, units[(index + 1) % count])
        if a is None or b is None:
            return 2.0 * math.pi
        cross = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        sin_term = cross[0] * px + cross[1] * py + cross[2] * pz
        cos_term = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        total += math.atan2(sin_term, cos_term)
    return total


def _tangent(
    px: float, py: float, pz: float, q: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    dot = q[0] * px + q[1] * py + q[2] * pz
    vx, vy, vz = q[0] - dot * px, q[1] - dot * py, q[2] - dot * pz
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm <= 1e-12:
        return None
    return (vx / norm, vy / norm, vz / norm)


__all__ = [
    "EARTH_RADIUS_KM",
    "HEXAGON_AREA_FACTOR",
    "LatLon",
    "destination",
    "dilate_ring",
    "dilate_shape",
    "great_circle_km",
    "initial_bearing_deg",
    "midpoint",
    "BOUNDARY_RINGS",
    "polygon_area_km2",
    "polygon_perimeter_km",
    "predicted_cells_in",
    "resample_track",
    "ring_containment",
    "ring_overlap_fraction",
    "swath_ring",
    "track_length_km",
]
