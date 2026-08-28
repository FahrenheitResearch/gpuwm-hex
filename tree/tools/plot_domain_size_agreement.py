#!/usr/bin/env python3
"""The one picture the nest-ratio decision turns on.

Interior agreement against distance from the boundary, one curve per domain
size, all at the same resolution over the same ground.

HOW TO READ IT, because the two answers look different and mean opposite
things:

* curves lying on top of each other and flat -- the boundary reaches nothing.
  Whatever the fine grid is doing differently from its parent, it does the
  same thing whether its edge is 135 km away or 300 km away, so an
  intermediate resolution level has nothing to fix.
* curves separating near zero, the smallest domain highest -- the boundary is
  contaminating the interior, the contamination reaches about as far as the
  separation persists, and a ladder is worth its forecasts.

These are analysis charts, not weather-field products: no map, no projection,
nothing rendered from a model field onto geography.  Weather fields stay under
the render law and come from the Rust renderer.

    python tools/plot_domain_size_agreement.py \\
        --agreement agreement.json --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Four fields spanning the whole disagreement gradient the receipts measured:
#: temperature agrees to five nines, vertical velocity is the worst of the
#: prognostic fields, and precipitation and reflectivity sit between them.  If
#: the boundary were the mechanism it would show on all four.
PANELS = (
    ("theta", "potential temperature", "K"),
    ("w", "vertical velocity", "m s$^{-1}$"),
    ("refl10cm", "reflectivity", "dBZ"),
    ("rainnc", "grid-scale precipitation", "mm"),
)

#: The domain ladder, smallest first.  The label carries the interface ratio
#: as well as the size, because those are the two things that change together
#: as a cull is widened and a reader has to see both to read the curve.
STYLE = {
    "d045": ("#08306b", "o", "0.45x"),
    "d070": ("#2171b5", "s", "0.70x"),
    "d100": ("#6baed6", "^", "1.00x, as placed"),
    "d135": ("#d94801", "D", "1.35x"),
    "d170": ("#7f2704", "v", "1.70x"),
}
ARMS = tuple(STYLE)


def _label(name: str, geometry: dict[str, tuple[float, float]]) -> str:
    """Legend text carrying the two numbers that place an arm on the ladder.

    Both come from the measurement's own receipt, so a legend cannot disagree
    with the run it describes.
    """

    base = STYLE[name][2]
    if name not in geometry:
        return base
    radius, ratio = geometry[name]
    return f"{base}   {radius:.0f} km   ~{ratio:.1f}:1"


#: The coarse parent's finest spacing, which is the numerator of every
#: interface ratio on the headline chart.  MEASURED on `g96.grid.nc`, the
#: mesh the boundary stream is built from; it is not the mesh's nominal 96 km.
COARSE_PARENT_FINEST_KM = 71.03


def _geometry(report: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Each arm's mean radius and interface ratio, off the arms' own grids.

    Read from the measurement's receipt rather than transcribed here.  The
    headline chart's x axis IS the decision's axis, and a hand-copied radius
    is a number that can quietly disagree with the run it labels.
    """

    out: dict[str, tuple[float, float]] = {}
    for name, arm in report.get("arms", {}).items():
        radius = arm.get("mean_radius_km")
        width = arm.get("driven_ring_mean_width_km")
        if radius is None or not width:
            continue
        out[name] = (radius, COARSE_PARENT_FINEST_KM / width)
    return out


def _series(entry: dict[str, Any], key: str) -> tuple[list[float], list[float]]:
    x: list[float] = []
    y: list[float] = []
    for row in entry["by_distance"]:
        x.append(0.5 * (row["from_km"] + row["to_km"]))
        y.append(row[key])
    return x, y


def _draw(report: dict[str, Any], key: str, ylabel: str, out: Path, title: str) -> None:
    frame = report["frames"][-1]
    reference = report["reference_arm"]
    geometry = _geometry(report)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, (field, pretty, units) in zip(axes.ravel(), PANELS):
        entry = frame["fields"].get(field)
        if entry is None:
            axis.text(0.5, 0.5, f"{field} absent", ha="center", va="center")
            axis.set_title(pretty)
            continue
        for arm in ARMS:
            if arm not in entry or arm == reference:
                continue
            colour, marker, _ = STYLE[arm]
            label = _label(arm, geometry)
            x, y = _series(entry[arm], key)
            axis.plot(
                x, y, marker=marker, color=colour, label=label, linewidth=1.8,
                markersize=5.0,
            )
        axis.set_title(f"{pretty}", fontsize=11)
        axis.set_xlabel("distance from the smallest domain's boundary (km)")
        axis.set_ylabel(f"{ylabel} ({units})" if key == "rms" else ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
        if key == "rms":
            axis.set_ylim(bottom=0.0)
    axes.ravel()[0].legend(fontsize=9, loc="best")
    figure.suptitle(
        f"{title}\nall arms 4.6 km, same ground, same card, differenced against "
        f"the {reference} run (no lateral boundary at all) at "
        f"{frame['xtime']}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    plt.close(figure)


def _draw_summary(report: dict[str, Any], out: Path) -> None:
    """Whole-patch RMS through the forecast, per arm, per field.

    The distance panels answer "where"; this one answers "does it grow".
    Boundary contamination that starts small and accumulates over six hours
    would be flat in space early and visible here.
    """

    reference = report["reference_arm"]
    geometry = _geometry(report)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for axis, (field, pretty, units) in zip(axes.ravel(), PANELS):
        for arm in ARMS:
            hours: list[float] = []
            values: list[float] = []
            for index, frame in enumerate(report["frames"]):
                entry = frame["fields"].get(field)
                if entry is None or arm not in entry or arm == reference:
                    continue
                hours.append(index * 0.5)
                values.append(entry[arm]["patch_rms"])
            if not values:
                continue
            colour, marker, _ = STYLE[arm]
            axis.plot(hours, values, marker=marker, color=colour,
                      label=_label(arm, geometry),
                      linewidth=1.8, markersize=4.0)
        axis.set_title(pretty, fontsize=11)
        axis.set_xlabel("forecast hour")
        axis.set_ylabel(f"patch RMS ({units})")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_ylim(bottom=0.0)
    axes.ravel()[0].legend(fontsize=9, loc="best")
    figure.suptitle(
        "Does the disagreement grow with lead time?\n"
        f"RMS over the smallest domain's whole interior against the {reference} "
        "run, every published frame",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    plt.close(figure)


def _draw_headline(report: dict[str, Any], out: Path) -> None:
    """Agreement against domain size -- the curve the decision is read off.

    One point per domain, one line per field.  A curve that FLATTENS says the
    domains past the knee are big enough and nothing further is bought; a
    curve still climbing at the widest arm says the widest arm is still not
    enough, which is a finding about the cascade rather than about the chart.
    """

    frame = report["frames"][-1]
    reference = report["reference_arm"]
    geometry = _geometry(report)
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.6))
    arms = [a for a in ARMS if a != reference and a in geometry]
    if not arms:
        plt.close(figure)
        print(f"skipped {out.name}: no arm carries a measured geometry")
        return
    # PLOTTED AS 1 - r ON A LOG AXIS, and the reason is legibility rather than
    # taste: temperature correlates at 0.99997 and vertical velocity at 0.62,
    # so on a shared 0-to-1 correlation axis the temperature curve is a
    # straight line at the top and its shape -- which is the thing under test
    # -- cannot be seen at all.  Lower is better on this axis, and a field
    # that improves by the same FACTOR across the ladder has the same slope
    # whatever its absolute agreement.
    for field, pretty, _units in PANELS:
        entry = frame["fields"].get(field)
        if entry is None:
            continue
        points = [
            (geometry[a][0], geometry[a][1], 1.0 - entry[a]["patch_correlation"])
            for a in arms
            if a in entry and entry[a]["patch_correlation"] is not None
        ]
        if not points:
            continue
        radius = [p[0] for p in points]
        ratio = [p[1] for p in points]
        value = [max(p[2], 1e-9) for p in points]
        left.plot(radius, value, marker="o", linewidth=1.9, label=pretty)
        right.plot(ratio, value, marker="o", linewidth=1.9, label=pretty)
    left.set_xlabel("domain mean radius (km)")
    right.set_xlabel("interface ratio (coarse parent : driven ring cell)")
    right.invert_xaxis()
    for axis in (left, right):
        axis.set_yscale("log")
        axis.set_ylabel(f"disagreement with the {reference} run  (1 - r, lower is better)")
        axis.grid(alpha=0.25, linewidth=0.6, which="both")
        axis.legend(fontsize=9, loc="best")
    top = left.get_ylim()[1]
    placed = geometry.get("d100")
    if placed is not None:
        left.axvline(placed[0], color="#888888", linestyle="--", linewidth=1.2)
        left.annotate(
            "the swath as placed today", xy=(placed[0], top),
            xytext=(placed[0] - 8, top * 0.62), fontsize=8, color="#555555",
            rotation=90, va="top", ha="right",
        )
    figure.suptitle(
        "Does a wider cut agree better with the run that has no boundary at all?\n"
        f"same 4.6 km core, same ground, same card, at {frame['xtime']}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    arguments = parser.parse_args()

    report = json.loads(arguments.agreement.read_text(encoding="utf-8"))
    _draw(
        report,
        "rms",
        "RMS difference",
        arguments.out_dir / "01-agreement-by-distance-from-boundary.png",
        "Does moving the boundary further away change the interior?",
    )
    _draw(
        report,
        "correlation",
        "correlation with the no-boundary run",
        arguments.out_dir / "02-correlation-by-distance-from-boundary.png",
        "The same question read as correlation rather than RMS",
    )
    _draw_summary(report, arguments.out_dir / "03-agreement-through-the-forecast.png")
    _draw_headline(report, arguments.out_dir / "04-agreement-against-domain-size.png")
    print(f"wrote four figures to {arguments.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
