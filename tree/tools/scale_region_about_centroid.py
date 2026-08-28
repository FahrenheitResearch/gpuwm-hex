#!/usr/bin/env python3
"""Shrink or grow a Shape row about its own centroid, on the sphere.

WHAT THIS IS FOR, AND IT IS ONE QUESTION.  A limited-area forecast takes its
lateral boundary from a parent whose cells are about fifteen times wider than
its own.  Nobody nests that hard in WRF.  The suspicion is that such a boundary
cannot hand the child the scales it needs and so starves the interior; the
competing explanation is that the child is correctly resolving structure the
parent cannot represent at all, which would make a low correlation the RIGHT
answer rather than a defect.  Comparing child against parent cannot separate
those two, in either direction, because both predict exactly the same
disagreement.

What separates them is running the SAME resolution over the SAME ground at two
domain sizes and comparing the two children over the smaller one's interior.
If the boundary is starving the interior, the smaller domain -- whose interior
sits closer to its own boundary -- disagrees with the larger one, and the
disagreement grows toward its edge.  If the interiors agree, the boundary is
not the mechanism and an intermediate resolution level buys nothing.

This emits the second and third domains.  A cull moves no cell centre, so
three nested culls of ONE parent share their cells' coordinate bits exactly and
the comparison needs no interpolation anywhere.

THE SCALING IS GEODESIC, NOT IN DEGREES.  Scaling latitude and longitude by a
factor is a different shape at every latitude and at 60 S -- where the measured
case sits -- it would squash the polygon by half in one axis.  Each vertex is
carried to a 3-D unit vector, the centroid is the normalised mean of those
vectors, and the vertex is moved along the great circle toward the centroid by
``1 - factor`` of its own angular distance.  That is a similarity transform on
the sphere: it preserves the shape's angular proportions about its centre and
scales every centroid-to-vertex arc by exactly ``factor``.

The output is a Shape row exactly as ``rw_mpas_mesh --region`` and
``gpuwm-hex cull --region`` read one, so nothing downstream learns that a
region was scaled.

    python tools/scale_region_about_centroid.py \\
        --region s01.cull-region.json --factor 0.45 --out d1.region.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The Shape kinds the mesh layer accepts.  Closed on purpose: an unknown kind
#: is refused by name rather than passed through unscaled, because a region
#: that silently ignores --factor would produce two arms of a domain-size
#: comparison that are the same domain, and the comparison would report perfect
#: agreement for the one reason that proves nothing.
SCALABLE_KINDS = ("polygon", "cap", "lat_lon_box")


def _unit(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )


def _degrees(vector: Sequence[float]) -> list[float]:
    x, y, z = vector
    norm = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm
    return [math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))]


def _normalise(vector: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = vector
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise SystemExit(
            "the vertices average to the centre of the sphere, so this ring has "
            "no centroid direction to scale about; it covers a hemisphere or "
            "more and is not a swath"
        )
    return (x / norm, y / norm, z / norm)


def _slerp_toward(
    vertex: Sequence[float], centre: Sequence[float], factor: float
) -> tuple[float, float, float]:
    """Move ``vertex`` along its great circle to ``centre``, keeping ``factor``.

    Great-circle interpolation rather than a chord: at the 5-degree separations
    a swath ring spans, a straight line through the sphere and then a
    re-normalise differs from the arc by parts in ten thousand -- small, but it
    is a systematic inward bias that grows with the arc, so the scaled ring
    would not be a similarity of the original.  There is no reason to accept
    that when the exact form is three lines.
    """

    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(vertex, centre))))
    omega = math.acos(dot)
    if omega < 1e-12:
        return tuple(vertex)  # type: ignore[return-value]
    sin_omega = math.sin(omega)
    # Fraction of the arc kept, measured from the centre outward.
    a = math.sin(factor * omega) / sin_omega
    b = math.sin((1.0 - factor) * omega) / sin_omega
    return (
        a * vertex[0] + b * centre[0],
        a * vertex[1] + b * centre[1],
        a * vertex[2] + b * centre[2],
    )


def scale_region(region: dict[str, Any], factor: float) -> dict[str, Any]:
    """Return ``region`` scaled by ``factor`` about its own centroid.

    ``factor == 1.0`` returns the row unchanged, byte for byte after a
    round trip, so the unscaled arm of a comparison can go through the same
    command as the scaled ones and no arm is special.
    """

    if factor <= 0.0:
        raise SystemExit(f"--factor must be positive, got {factor}")
    kind = region.get("kind")
    if kind not in SCALABLE_KINDS:
        raise SystemExit(
            f"region kind {kind!r} is not one of {SCALABLE_KINDS}; refusing "
            "rather than returning it unscaled, because an unscaled arm of a "
            "domain-size comparison is the same domain twice and would report "
            "perfect agreement for a reason that proves nothing"
        )
    scaled = dict(region)
    if kind == "cap":
        scaled["radius_km"] = float(region["radius_km"]) * factor
        return scaled
    if kind == "lat_lon_box":
        # A box is scaled about the midpoint of each of its own spans.  This
        # is the one kind where the degree form IS the definition -- the row
        # names two intervals, not a ring -- so scaling the intervals is the
        # faithful reading and no spherical construction applies.
        out = {}
        for axis in ("lat_deg", "lon_deg"):
            low, high = (float(v) for v in region[axis])
            mid = 0.5 * (low + high)
            half = 0.5 * (high - low) * factor
            out[axis] = [mid - half, mid + half]
        scaled.update(out)
        return scaled

    vertices = [tuple(float(v) for v in pair) for pair in region["vertices_deg"]]
    if len(vertices) < 3:
        raise SystemExit(
            f"a polygon region needs at least three vertices, got {len(vertices)}"
        )
    units = [_unit(lat, lon) for lat, lon in vertices]
    centre = _normalise(
        (
            sum(v[0] for v in units) / len(units),
            sum(v[1] for v in units) / len(units),
            sum(v[2] for v in units) / len(units),
        )
    )
    scaled["vertices_deg"] = [
        _degrees(_slerp_toward(v, centre, factor)) for v in units
    ]
    return scaled


def centroid_of(region: Mapping[str, Any]) -> tuple[float, float, float]:
    """The direction a polygon region is scaled about, as a unit vector."""

    units = [_unit(float(a), float(b)) for a, b in region["vertices_deg"]]
    return _normalise(
        (
            sum(v[0] for v in units) / len(units),
            sum(v[1] for v in units) / len(units),
            sum(v[2] for v in units) / len(units),
        )
    )


def region_metrics(
    region: Mapping[str, Any], centre: Sequence[float] | None = None
) -> dict[str, Any]:
    """Angular extent of a region, so a receipt can say what changed.

    Reported as the mean and maximum centroid-to-vertex arc in kilometres on a
    sphere of MPAS's own radius.  These are the numbers that make "0.45" mean
    something to a reader: a factor is not a domain size, an arc is.

    ``centre`` DEFAULTS TO THE REGION'S OWN CENTROID AND THAT IS NOT THE SAME
    THING as the centroid it was scaled about.  A ring's spherical centroid is
    the normalised mean of its vertex directions, and that mean is preserved
    under contraction only for a shape that is symmetric about it -- a true
    small circle.  A real swath ring is not: it is a track's flare, longer at
    one end.  Contracting it therefore moves its own computed centroid by a
    few tens of metres, which is a property of the sphere and not a defect in
    the transform.  Pass the ORIGINAL centre to measure the transform itself,
    where the scaling is exact to rounding; leave it out to describe the shape
    that actually came out.
    """

    if region.get("kind") != "polygon":
        return {"kind": region.get("kind")}
    units = [_unit(float(a), float(b)) for a, b in region["vertices_deg"]]
    if centre is None:
        centre = centroid_of(region)
    radius_m = 6_371_229.0  # MPAS a_ = the sphere every grid file declares
    arcs = [
        math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(v, centre)))))
        * radius_m
        / 1000.0
        for v in units
    ]
    return {
        "kind": "polygon",
        "vertices": len(units),
        "centroid_deg": _degrees(centre),
        "centroid_to_vertex_km": {
            "mean": sum(arcs) / len(arcs),
            "min": min(arcs),
            "max": max(arcs),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=Path, required=True)
    parser.add_argument(
        "--factor",
        type=float,
        required=True,
        help=(
            "geodesic similarity factor about the region's own centroid; 0.5 "
            "halves every centroid-to-vertex arc and so quarters the area"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None)
    arguments = parser.parse_args()

    region = json.loads(arguments.region.read_text(encoding="utf-8"))
    scaled = scale_region(region, arguments.factor)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(scaled, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    centre = centroid_of(region) if region.get("kind") == "polygon" else None
    before = region_metrics(region, centre)
    # Measured about the ORIGINAL centre, so the receipt's ratio is the factor
    # that was applied rather than the factor plus the centroid's own drift.
    after = region_metrics(scaled, centre)
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(
            json.dumps(
                {
                    "schema": "gpuwm-hex.scaled-region/v1",
                    "source": str(arguments.region),
                    "out": str(arguments.out),
                    "factor": arguments.factor,
                    "before": before,
                    "after": after,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if before.get("kind") == "polygon":
        print(
            f"factor {arguments.factor}: mean centroid-to-vertex "
            f"{before['centroid_to_vertex_km']['mean']:.1f} km -> "
            f"{after['centroid_to_vertex_km']['mean']:.1f} km"
        )
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
