"""The figures for the swath layer's first contact with a real forecast.

ANALYSIS CHARTS, NOT WEATHER FIELDS.  Everything drawn here is either a
DECISION the placement layer made (where it put a swath, how it ranked a
candidate, whether a slot survived a cycle) or the OROGRAPHY that explains
the decision.  No modelled weather field is rendered by matplotlib; the
render law puts those through ``rw_wrfbatch``, and nothing here needs one
to make its point.

Run:
    python tools/plot_swath_first_real.py --plans <dir> --basemap <npz> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

INK = "#1b1b1f"
MUTED = "#6b6b76"
RAW = "#c1272d"       # the field as shipped: raw surface pressure
MSLP = "#1f5c99"      # the field as shipped now: sea-level reduction
CONV = "#c77f1a"      # deep convection
GRID = "#dcdce2"


def _style(ax: Any, title: str, subtitle: str = "") -> None:
    if subtitle:
        ax.set_title(title, fontsize=13.5, color=INK, loc="left", pad=22)
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=MUTED, va="bottom")
    else:
        ax.set_title(title, fontsize=13.5, color=INK, loc="left", pad=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _basemap(ax: Any, base: Any) -> None:
    """Land and orography as context: the explanatory variable of this lane.

    Two tones only.  Land is a flat light grey so the coastlines read as
    coastlines; ground above 1,500 m is a darker grey because it is the
    thing that decided the top panel of figure 1.
    """

    lat, lon, ter = base["lat"], base["lon"], base["ter"]
    land = ter > 0.5
    high = ter > 1500.0
    ax.scatter(lon[land & ~high], lat[land & ~high], s=2.4, c="#d8dae0",
               linewidths=0, zorder=1, rasterized=True)
    ax.scatter(lon[high], lat[high], s=3.0, c="#8b8f9a", linewidths=0,
               zorder=2, rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.set_xlabel("longitude", fontsize=9, color=MUTED)
    ax.set_ylabel("latitude", fontsize=9, color=MUTED)
    ax.set_facecolor("#fbfcfd")
    ax.set_aspect("equal", adjustable="box")


def _label(ax: Any, x: float, y: float, text: str, colour: str, dy: float) -> None:
    """A label parked clear of the mark, with a leader back to it."""

    ax.annotate(
        text, xy=(x, y), xytext=(x, y + dy),
        fontsize=9, color=colour, weight="bold",
        ha="center", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor=colour, linewidth=0.9, alpha=0.94),
        arrowprops=dict(arrowstyle="-", color=colour, linewidth=0.9,
                        shrinkA=1.0, shrinkB=6.0),
    )


def _rings(plan: dict[str, Any]) -> list[tuple[np.ndarray, dict[str, Any]]]:
    out = []
    for row in plan.get("admitted", []):
        ring = row.get("ring_deg")
        if not ring:
            continue
        arr = np.asarray(ring, dtype=float)
        out.append((arr, row))
    return out


def _draw_ring(ax: Any, arr: np.ndarray, colour: str, *, label: str | None = None) -> None:
    """A ring in lat/lon, split where it crosses the antimeridian."""

    lat, lon = arr[:, 0], arr[:, 1]
    if lon.max() - lon.min() > 180.0:
        lon = np.where(lon < 0.0, lon + 360.0, lon)
    pts = np.column_stack([lon, lat])
    ax.add_patch(MplPolygon(pts, closed=True, facecolor=colour, alpha=0.20,
                            edgecolor=colour, linewidth=1.4, zorder=5, label=label))
    if lon.max() > 180.0:
        ax.add_patch(MplPolygon(np.column_stack([lon - 360.0, lat]), closed=True,
                                facecolor=colour, alpha=0.20, edgecolor=colour,
                                linewidth=1.4, zorder=5))


def figure_where_they_landed(plans: Path, base: Any, out: Path) -> Path:
    """THE figure: the same forecast, the same code, two fields, two worlds."""

    raw = json.loads((plans / "u96-A-raw" / "swath-plan.json").read_text())
    mslp = json.loads((plans / "u96-B-mslp" / "swath-plan.json").read_text())

    named = {
        # what the placement actually sits on, read off the map
        "raw": ["Tibetan Plateau", "Altiplano / Andes", "Antarctic Peninsula",
                "Ethiopian Highlands"],
        "mslp": ["Southern Ocean", "Southern Ocean", "Southern Ocean",
                 "Southern Ocean"],
    }
    fig = plt.figure(figsize=(15.4, 12.4))
    spec = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.30], hspace=0.30, wspace=0.04)
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[1, 0])]
    tables = [fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[1, 1])]
    for panel in tables:
        panel.axis("off")
    for ax, panel, plan, colour, key, head, sub in (
        (axes[0], tables[0], raw, RAW, "raw",
         "As shipped: a minimum in RAW surface pressure",
         "All four fine grids land on high ground. The top-ranked 'tropical cyclone' "
         "is the Tibetan Plateau, 477 hPa 'below' a 1004 hPa threshold."),
        (axes[1], tables[1], mslp, MSLP, "mslp",
         "After the fix: a minimum in pressure reduced to SEA LEVEL",
         "No mountain survives. All four are Southern Ocean storms of 945-953 hPa, "
         "which are the deepest lows this forecast actually contains."),
    ):
        _basemap(ax, base)
        rows = _rings(plan)
        lines: list[tuple[str, str, str]] = []
        for index, (arr, row) in enumerate(rows):
            _draw_ring(ax, arr, colour, label="placed swath" if index == 0 else None)
            clat, clon = row["centroid_deg"]
            terms = {t["id"]: t for t in row["rank"]["terms"]}
            depth = terms.get("intensity", {}).get("raw", float("nan")) / 100.0
            ax.plot([clon], [clat], marker="x", color=colour, markersize=12,
                    markeredgewidth=2.6, zorder=8)
            ax.annotate(row["slot_id"], (clon, clat), textcoords="offset points",
                        xytext=(0, 13), ha="center", fontsize=9.5, color=colour,
                        weight="bold", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=colour, linewidth=0.8, alpha=0.95))
            place = named[key][index] if index < len(named[key]) else ""
            lines.append((row["slot_id"], place, f"{depth:,.0f} hPa below 1004"))
        _style(ax, head, sub)
        ax.legend(loc="lower left", fontsize=8.5, frameon=False)

        panel.text(0.0, 0.97, "what it landed on", fontsize=10, color=INK,
                   weight="bold", va="top", transform=panel.transAxes)
        y = 0.86
        for slot, place, depth in lines:
            panel.text(0.0, y, slot, fontsize=10, color=colour, weight="bold",
                       va="top", transform=panel.transAxes)
            panel.text(0.20, y, place, fontsize=10, color=INK, va="top",
                       transform=panel.transAxes)
            panel.text(0.20, y - 0.055, depth, fontsize=9, color=MUTED, va="top",
                       transform=panel.transAxes)
            y -= 0.155

    fig.suptitle(
        "Where the machine put the fine grid, on a real 24 h global forecast\n"
        "u96.64002, 96 km uniform, 64,002 cells, GFS 2026-08-12 06Z, the proving RTX 5070 Ti",
        fontsize=14.5, color=INK, x=0.012, ha="left", y=0.975,
    )
    fig.text(0.012, 0.014,
             "Light grey is model land; darker grey is ground above 1,500 m -- the "
             "orography that decided the top panel. Nothing here is a weather field.",
             fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.875, bottom=0.055)
    target = out / "01-where-the-swaths-landed.png"
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    return target


def figure_elevation_is_the_signal(base: Any, out: Path) -> Path:
    """Why it happened, in one scatter: depth against height."""

    lat, ter, psfc, t2 = base["lat"], base["ter"], base["psfc"], base["t2"]
    mslp = psfc * np.exp(9.80665 * ter / (287.058 * (t2 + 0.5 * 0.0065 * ter)))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=False)
    for ax, field, colour, head in (
        (axes[0], psfc, RAW, "Raw surface pressure"),
        (axes[1], mslp, MSLP, "Reduced to sea level"),
    ):
        ax.scatter(ter, field / 100.0, s=2.0, c=colour, alpha=0.20, linewidths=0)
        ax.axhline(1004.0, color=INK, linewidth=1.1, linestyle="--")
        ax.text(5300, 1004, " 1004 hPa threshold", fontsize=8.5, color=INK,
                va="bottom", ha="right")
        ax.set_xlabel("model terrain height (m)", fontsize=9, color=MUTED)
        ax.set_ylabel("pressure (hPa)", fontsize=9, color=MUTED)
        _style(ax, head)
    axes[0].text(
        0.03, 0.06,
        "Every cell above ~1,500 m is below the threshold.\n"
        "43% of the globe qualifies as a 'closed low'.",
        transform=axes[0].transAxes, fontsize=9, color=RAW, weight="bold",
    )
    axes[1].text(
        0.03, 0.06,
        "Height no longer sets the ordering.\n"
        "What is left below the line is weather.",
        transform=axes[1].transAxes, fontsize=9, color=MSLP, weight="bold",
    )
    fig.suptitle(
        "Why the detector chose mountains: the field it searched was a function of height",
        fontsize=14, color=INK, x=0.012, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    target = out / "02-elevation-was-the-signal.png"
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    return target


def figure_rank_scales(plans: Path, out: Path) -> Path:
    """The two armed rows do not share a score scale, so one can never place."""

    plan = json.loads((plans / "g12-C-tropical-band" / "swath-plan.json").read_text())
    rows: list[tuple[str, float]] = []
    for row in plan.get("admitted", []):
        rows.append((row["threat_class"], row["rank"]["score"]))
    for row in plan.get("declined", []):
        rank = row.get("rank") or {}
        if rank.get("score") is not None:
            rows.append((row.get("threat_class", "?"), rank["score"]))
    cyc = sorted([s for c, s in rows if c == "tropical_cyclone"], reverse=True)
    con = sorted([s for c, s in rows if c == "deep_convection"], reverse=True)

    banded = json.loads((plans / "g12-B2-mslp" / "swath-plan.json").read_text())
    allcyc = []
    for row in banded.get("admitted", []) + banded.get("declined", []):
        rank = row.get("rank") or {}
        if rank.get("score") is not None and row.get("threat_class") == "tropical_cyclone":
            allcyc.append(rank["score"])
    allcyc.sort(reverse=True)

    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.plot(range(1, len(allcyc) + 1), allcyc, marker="o", markersize=5,
            color=MSLP, linewidth=1.6, label=f"low_pressure_centre ({len(allcyc)} tracks)")
    ax.plot(range(1, len(con) + 1), con, marker="s", markersize=4,
            color=CONV, linewidth=1.6,
            label=f"deep_moist_convection_area ({len(con)} tracks)")
    ax.axhline(min(allcyc), color=MSLP, linestyle=":", linewidth=1.1)
    ax.axhline(max(con), color=CONV, linestyle=":", linewidth=1.1)
    ax.annotate(
        f"the WORST cyclone ({min(allcyc):.3f}) still beats\n"
        f"the BEST convective region ({max(con):.3f})",
        xy=(len(allcyc) * 0.55, (min(allcyc) + max(con)) / 2.0),
        fontsize=10, color=INK, weight="bold",
    )
    ax.axvspan(0.5, 4.5, color="#f3f4f7", zorder=0)
    ax.text(2.4, ax.get_ylim()[1] * 0.96, "budget: 4 swaths", fontsize=9,
            color=MUTED, ha="center", va="top")
    ax.set_xlabel("rank within its own metric row", fontsize=9, color=MUTED)
    ax.set_ylabel("rank score", fontsize=9, color=MUTED)
    ax.legend(fontsize=9, frameon=False)
    _style(ax, "One 'intensity' scale divides two incompatible units",
           "The policy divides every metric_extremum by 3000. A pressure anomaly is "
           "in pascals (about 5,000); a reflectivity anomaly is in dBZ (about 11).")
    fig.tight_layout()
    target = out / "03-rank-scales-do-not-meet.png"
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    return target


def figure_cycles(plans: Path, base: Any, out: Path) -> Path:
    """Hysteresis on a real atmosphere: one slot held, three turned over."""

    c1 = json.loads((plans / "cyc-c1" / "swath-plan.json").read_text())
    c2 = json.loads((plans / "cyc-c2" / "swath-plan.json").read_text())

    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    _basemap(ax, base)
    ax.set_ylim(-80, -35)
    ax.set_xlim(-180, 180)
    ax.set_aspect("auto")
    for arr, row in _rings(c1):
        _draw_ring(ax, arr, MUTED)
        clat, clon = row["centroid_deg"]
        ax.plot([clon], [clat], marker="o", color=MUTED, markersize=8, zorder=6)
        _label(ax, clon, clat, row["slot_id"] + "  cycle 1", MUTED, -9.0)
    for arr, row in _rings(c2):
        held = (row.get("hysteresis") or {}).get("incumbent")
        colour = MSLP if held else CONV
        _draw_ring(ax, arr, colour)
        clat, clon = row["centroid_deg"]
        ax.plot([clon], [clat], marker="x", color=colour, markersize=10,
                markeredgewidth=2.2, zorder=6)
        moved = (row.get("hysteresis") or {}).get("centroid_moved_km")
        tag = row["slot_id"] + (f"  HELD, moved {moved:.0f} km"
                                if held and moved else "  new slot")
        _label(ax, clon, clat, tag, colour, 9.0)
    ax.plot([], [], color=MUTED, linewidth=2, label="cycle 1 (t+0 to t+12 h)")
    ax.plot([], [], color=MSLP, linewidth=2, label="cycle 2: slot CONTINUED")
    ax.plot([], [], color=CONV, linewidth=2, label="cycle 2: new slot")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    _style(ax, "Two cycles six hours apart, on a real atmosphere",
           "1 of 4 slots continued with its identity intact. The other three were not "
           "evicted by rank oscillation -- their storms simply left the top four.")
    fig.tight_layout()
    target = out / "04-two-cycles-six-hours-apart.png"
    fig.savefig(target, dpi=170, facecolor="white")
    plt.close(fig)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--basemap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)
    base = np.load(arguments.basemap)
    made = [
        figure_where_they_landed(arguments.plans, base, arguments.out),
        figure_elevation_is_the_signal(base, arguments.out),
        figure_rank_scales(arguments.plans, arguments.out),
        figure_cycles(arguments.plans, base, arguments.out),
    ]
    for path in made:
        print(path, path.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
