#!/usr/bin/env python3
"""What the domains ARE, before any number is read off them.

The nest-ratio verdict rests on concentric limited-area domains cut at one
resolution from one parent, and on the innermost one's free interior being a
patch all of them hold.  That setup decides how every later chart is read, and
it is a geometry -- so it is drawn rather than described.

Cell centres, their boundary rings, and the distance axis every later panel is
binned on.  This is an analysis chart of the EXPERIMENT, not a weather-field
product: no model field is drawn on it and no projection library is used.

    python tools/plot_domain_size_geometry.py \\
        --arm d045=D045/grid.nc --arm d070=D070/grid.nc --arm d100=D100/grid.nc \\
        --patch d045 --out figure.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPHERE_RADIUS_KM = 6_371.229
STYLE = {
    "d045": ("#08306b", "0.45x of the placed swath"),
    "d070": ("#2171b5", "0.70x"),
    "d100": ("#6baed6", "1.00x, the swath as placed"),
    "d135": ("#d94801", "1.35x"),
    "d170": ("#7f2704", "1.70x"),
}
#: Outermost first, so the smaller domains draw on top of the larger ones.
DRAW_ORDER = ("d170", "d135", "d100", "d070", "d045")


def _read(path: Path):
    from netCDF4 import Dataset

    with Dataset(str(path)) as grid:
        lat = np.degrees(np.asarray(grid.variables["latCell"][:], np.float64))
        lon = np.degrees(np.asarray(grid.variables["lonCell"][:], np.float64))
        bdy = np.asarray(grid.variables["bdyMaskCell"][:], np.int64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lat, lon, bdy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=GRID")
    parser.add_argument("--patch", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    arms = {}
    for text in arguments.arm:
        name, _, path = text.partition("=")
        arms[name] = _read(Path(path))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 6.0))

    # Left: the domains, outermost first so the smaller ones sit on top.
    for name in DRAW_ORDER:
        if name not in arms:
            continue
        lat, lon, bdy = arms[name]
        colour, label = STYLE[name]
        left.scatter(
            lon[bdy == 0], lat[bdy == 0], s=1.0, color=colour, alpha=0.35,
            label=f"{label} free interior ({int((bdy == 0).sum()):,} cells)",
        )
        left.scatter(lon[bdy > 0], lat[bdy > 0], s=2.0, color=colour, alpha=0.95)
    left.set_title(
        "Three domains, one resolution, concentric ground\n"
        "solid rims are the seven driven boundary rings", fontsize=11
    )
    left.set_xlabel("longitude (deg)")
    left.set_ylabel("latitude (deg)")
    left.legend(fontsize=8, loc="lower left", markerscale=6)
    left.grid(alpha=0.2, linewidth=0.6)

    # Right: the distance axis every later panel bins on.
    lat, lon, bdy = arms[arguments.patch]
    interior = bdy == 0
    driven = bdy > 0

    def vectors(latitude, longitude):
        a, b = np.radians(latitude), np.radians(longitude)
        return np.stack((np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)), 1)

    dots = np.clip(vectors(lat[interior], lon[interior]) @ vectors(lat[driven], lon[driven]).T, -1.0, 1.0)
    distance = np.arccos(dots.max(axis=1)) * SPHERE_RADIUS_KM
    scatter = right.scatter(
        lon[interior], lat[interior], c=distance, s=3.0, cmap="viridis"
    )
    right.scatter(lon[driven], lat[driven], s=3.0, color="#c1440e")
    figure.colorbar(scatter, ax=right, label="km to the nearest driven cell")
    right.set_title(
        f"The comparison patch: {int(interior.sum()):,} free cells of "
        f"{arguments.patch}\nevery later chart is binned on this distance",
        fontsize=11,
    )
    right.set_xlabel("longitude (deg)")
    right.set_ylabel("latitude (deg)")
    right.grid(alpha=0.2, linewidth=0.6)

    figure.tight_layout()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out, dpi=130)
    print(
        f"wrote {arguments.out}: patch {int(interior.sum())} cells, "
        f"distance {distance.min():.1f}-{distance.max():.1f} km"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
