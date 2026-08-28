"""The evidence charts for the threat-metric library, from the plan itself.

ANALYSIS CHARTS, NOT WEATHER FIELDS.  Nothing here draws a meteorological
field: these are a detection map, a units comparison, a score decomposition
and an inventory, all read off ``swath-plan.json``.  The render law's
matplotlib carve-out is for exactly this and the weather-field renderer is
not involved.

THE BACKGROUND OF THE MAP IS THE MODEL'S OWN GEOGRAPHY.  There is no
coastline shapefile here and no cartopy.  The land is the set of cells
whose top soil layer is not the 1.0 the land surface writes over water --
the same field the fire-weather row uses as its fuel condition -- so the
map is drawn out of the same file the detection ran on, and a coastline
that disagreed with the model's own would be impossible.

Every number plotted comes from ``plan/RANKED-CANDIDATES.json`` and
``plan/PLAN-SUMMARY.json``, which carry the SHA-256 of the forecast and of
both documents they were produced from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

#: Okabe-Ito, the eight-hue set built for colour-vision deficiency, in a
#: fixed order.  Assigned to rows by NAME so a row that finds nothing keeps
#: its colour and the ninth row does not repaint the other eight.
ROW_COLOURS = {
    "extratropical_cyclone_centre": "#0072B2",
    "tropical_cyclone_centre": "#56B4E9",
    "deep_convection_area": "#D55E00",
    "severe_convection_area": "#7F7F7F",
    "fire_weather_area": "#E69F00",
    "heavy_rainfall_area": "#009E73",
    "winter_storm_area": "#CC79A7",
    "damaging_wind_area": "#8C564B",
    "atmospheric_river_corridor": "#4B3F9E",
}

ROW_LABELS = {
    "extratropical_cyclone_centre": "extratropical cyclone",
    "tropical_cyclone_centre": "tropical cyclone",
    "deep_convection_area": "deep convection",
    "severe_convection_area": "severe convection (CONUS)",
    "fire_weather_area": "fire weather",
    "heavy_rainfall_area": "heavy rainfall",
    "winter_storm_area": "winter storm",
    "damaging_wind_area": "damaging wind",
    "atmospheric_river_corridor": "atmospheric river",
}

#: The scale the previous schema divided EVERY row's intensity by, in the
#: units of one field.  Kept here only so the comparison figure can show
#: what it did.
SUPERSEDED_SHARED_INTENSITY_SCALE = 3000.0
SUPERSEDED_SHARED_EXTENT_SCALE = 200000.0

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"
LAND = "#cfcbc3"


def _style(matplotlib) -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "font.size": 9,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def _rows_in_order(candidates: list[dict]) -> list[str]:
    present = {row["metric_id"] for row in candidates}
    return [key for key in ROW_COLOURS if key in present]


# ---------------------------------------------------------------------------
# 1. what each threat row selects on ONE forecast
# ---------------------------------------------------------------------------
def figure_threat_map(data: dict, summary: dict, land: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    candidates = data["candidates"]
    figure, axes = plt.subplots(figsize=(13.5, 7.0))
    axes.scatter(
        land["lon"], land["lat"], s=2.2, c=LAND, marker="s", linewidths=0,
        zorder=0, rasterized=True,
    )
    order = _rows_in_order(candidates)
    for metric_id in order:
        points = [row for row in candidates if row["metric_id"] == metric_id]
        axes.scatter(
            [row["lon"] for row in points], [row["lat"] for row in points],
            s=[8.0 + 26.0 * min(2.5, row["score"]) for row in points],
            facecolors="none", edgecolors=ROW_COLOURS[metric_id],
            linewidths=1.1, zorder=3,
            label=f"{ROW_LABELS[metric_id]}  ({len(points)})",
        )
    for admitted in summary["admitted"]:
        ring = np.asarray(admitted["ring_deg"], dtype=float)
        colour = ROW_COLOURS[admitted["metric_id"]]
        # A ring that crosses the antimeridian would draw a band across the
        # whole map; split it where the longitude jumps instead.
        for segment in _split_at_antimeridian(ring):
            axes.plot(
                segment[:, 1], segment[:, 0], color=colour, linewidth=2.0, zorder=5,
            )
            axes.fill(
                segment[:, 1], segment[:, 0], color=colour, alpha=0.14, zorder=4,
            )
        # Offset the label away from the ring's own centre so it never sits
        # on top of the candidates it was chosen from.
        axes.annotate(
            admitted["slot_id"],
            (admitted["centroid_deg"][1], admitted["centroid_deg"][0]),
            textcoords="offset points", xytext=(0, -26), ha="center",
            fontsize=11, fontweight="bold", color=colour, zorder=7,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white",
                  "edgecolor": colour, "linewidth": 1.0},
        )
    axes.set_xlim(-180, 180)
    axes.set_ylim(-90, 90)
    axes.set_xticks(range(-180, 181, 30))
    axes.set_yticks(range(-90, 91, 30))
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(
        "What every armed threat row found in ONE 24 h global forecast, and "
        "which four got a grid\n"
        f"{len(candidates)} ranked candidates from "
        f"{len(order)} rows; circle size is rank score; filled polygons are "
        "the four admitted swaths",
        loc="left", pad=12,
    )
    axes.legend(
        loc="lower left", bbox_to_anchor=(0.0, -0.30), ncol=5, fontsize=8.5,
        markerscale=1.4, handletextpad=0.4, columnspacing=1.4,
    )
    figure.savefig(out)
    plt.close(figure)
    return out


def _split_at_antimeridian(ring: np.ndarray) -> list[np.ndarray]:
    breaks = np.flatnonzero(np.abs(np.diff(ring[:, 1])) > 180.0)
    if breaks.size == 0:
        return [ring]
    pieces = []
    start = 0
    for index in breaks.tolist():
        pieces.append(ring[start:index + 1])
        start = index + 1
    pieces.append(ring[start:])
    return [piece for piece in pieces if len(piece) > 1]


# ---------------------------------------------------------------------------
# 2. the defect this lane was sent to fix, in one picture
# ---------------------------------------------------------------------------
def figure_commensurability(data: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    candidates = data["candidates"]
    order = _rows_in_order(candidates)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), sharey=True,
    )
    positions = np.arange(len(order))
    rng = np.random.default_rng(20260826)
    for axis, mode in ((left, "shared"), (right, "row")):
        for index, metric_id in enumerate(order):
            values = [
                row["intensity_raw"] / SUPERSEDED_SHARED_INTENSITY_SCALE
                if mode == "shared" else row["intensity_scaled"]
                for row in candidates if row["metric_id"] == metric_id
            ]
            jitter = rng.uniform(-0.18, 0.18, len(values))
            axis.scatter(
                np.asarray(values), index + jitter, s=11,
                facecolors="none", edgecolors=ROW_COLOURS[metric_id],
                linewidths=0.9,
            )
            axis.plot(
                [float(np.median(values))], [index], marker="|", markersize=16,
                color=ROW_COLOURS[metric_id], markeredgewidth=2.4,
            )
        axis.set_xscale("log")
        axis.set_yticks(positions)
        axis.set_ylim(-0.7, len(order) - 0.3)
        axis.set_xlabel("intensity term, after scaling (dimensionless)")
    left.set_yticklabels([ROW_LABELS[key] for key in order])
    left.set_title(
        "BEFORE: one shared scale of 3,000, in the units of one field\n"
        "pascals and decibels on the same axis",
        loc="left",
    )
    right.set_title(
        "AFTER: each row divides by its own declared intensity_reference\n"
        "every class lands in the same decade",
        loc="left",
    )
    figure.suptitle(
        "The commensurability fix, on 541 real candidates from one 24 h global "
        "forecast",
        x=0.005, ha="left", fontsize=12.5, y=1.02,
    )
    figure.savefig(out)
    plt.close(figure)
    return out


# ---------------------------------------------------------------------------
# 3. what the score is actually made of
# ---------------------------------------------------------------------------
def figure_rank_composition(data: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    candidates = sorted(data["candidates"], key=lambda row: -row["score"])[:14]
    weights = {"intensity": 1.0, "persistence": 0.5, "travel": 0.25, "extent": 0.25}
    term_colours = {
        "intensity": "#0072B2", "persistence": "#009E73",
        "travel": "#E69F00", "extent": "#D55E00",
    }

    def shipped(row: dict) -> dict:
        return {
            "intensity": weights["intensity"] * row["intensity_scaled"],
            "persistence": weights["persistence"] * min(
                1.0, row["frames_raw"] / row["frames_ref"]
            ),
            "travel": weights["travel"] * min(
                1.0, row["travel_raw"] / row["travel_ref"]
            ),
            "extent": weights["extent"] * min(
                1.0, row["area_raw"] / row["area_ref"]
            ),
        }

    def superseded(row: dict) -> dict:
        return {
            "intensity": weights["intensity"]
            * row["intensity_raw"] / SUPERSEDED_SHARED_INTENSITY_SCALE,
            "persistence": weights["persistence"] * row["frames_raw"] / 8.0,
            "travel": weights["travel"] * row["travel_raw"] / 1200.0,
            "extent": weights["extent"]
            * row["area_raw"] / SUPERSEDED_SHARED_EXTENT_SCALE,
        }

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(12.5, 8.4), gridspec_kw={"hspace": 0.42},
    )
    labels = [
        f"{ROW_LABELS[row['metric_id']]}"
        + (f"  [{row['admitted']}]" if row["admitted"] else "")
        for row in candidates
    ]
    for axis, builder, title in (
        (bottom, shipped, "AFTER: intensity is the only unbounded term"),
        (top, superseded,
         "BEFORE: one policy scale per term, none of them bounded"),
    ):
        base = np.zeros(len(candidates))
        y = np.arange(len(candidates))
        for name in ("intensity", "persistence", "travel", "extent"):
            values = np.asarray([builder(row)[name] for row in candidates])
            axis.barh(
                y, values, left=base, height=0.62, color=term_colours[name],
                edgecolor="white", linewidth=1.2, label=name,
            )
            base = base + values
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=8.5)
        axis.invert_yaxis()
        axis.set_xlabel("contribution to rank score")
        axis.set_title(title, loc="left")
        axis.grid(axis="y", visible=False)
    top.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9.5,
    )
    figure.suptitle(
        "The same fourteen candidates, decomposed both ways\n"
        "one 11,433,855 km2 moisture corridor scored 14.29 on extent alone "
        "before the reference became the area a swath can resolve",
        x=0.005, ha="left", fontsize=12.5, y=0.99,
    )
    figure.savefig(out)
    plt.close(figure)
    return out


# ---------------------------------------------------------------------------
# 4. which shipped rows actually fired
# ---------------------------------------------------------------------------
def figure_row_inventory(summary: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    per_row = summary["per_row"]
    order = [key for key in ROW_COLOURS if key in per_row or key in ROW_LABELS]
    order = [key for key in order]
    tracks = [per_row.get(key, {}).get("tracks", 0) for key in order]
    ranked = [per_row.get(key, {}).get("ranked", 0) for key in order]
    placed = [per_row.get(key, {}).get("placed", 0) for key in order]
    best = [per_row.get(key, {}).get("best_rank") for key in order]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    y = np.arange(len(order))
    left.barh(
        y, tracks, height=0.6,
        color=[ROW_COLOURS[key] for key in order], edgecolor="white",
        linewidth=1.2,
    )
    for index, (count, place) in enumerate(zip(tracks, placed)):
        note = f"{count} tracks"
        if place:
            note += f"   {place} PLACED"
        left.annotate(
            note, (count, index), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8.5,
            color=INK if place else MUTED,
            fontweight="bold" if place else "normal",
        )
    left.set_yticks(y)
    left.set_yticklabels([ROW_LABELS[key] for key in order])
    left.invert_yaxis()
    left.set_xlabel("tracks formed on the 24 h forecast")
    left.set_xlim(0, max(tracks) * 1.42 if max(tracks) else 1)
    left.grid(axis="y", visible=False)
    left.set_title(
        "Every shipped row, against one real forecast", loc="left",
    )

    ranks = [value if value else np.nan for value in best]
    right.barh(
        y, [0 if value is None else value for value in ranks], height=0.6,
        color=[ROW_COLOURS[key] for key in order], edgecolor="white",
        linewidth=1.2,
    )
    for index, value in enumerate(best):
        right.annotate(
            "no candidate reached the ranking" if not value else f"#{value}",
            (0 if not value else value, index), xytext=(6, 0),
            textcoords="offset points", va="center", fontsize=8.5,
            color=MUTED if not value else INK,
        )
    right.set_yticks(y)
    right.set_yticklabels([])
    right.invert_yaxis()
    right.set_xscale("symlog")
    right.set_xlabel("rank of this row's best candidate (1 is first)")
    right.grid(axis="y", visible=False)
    right.set_title(
        "Where its best candidate ranked among all 541", loc="left",
    )
    figure.savefig(out)
    plt.close(figure)
    return out


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="plot_swath_metrics_evidence",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = Path(__file__).resolve().parents[1] / "evidence" / "swath-metrics-20260826"
    parser.add_argument("--evidence", type=Path, default=here)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    _style(matplotlib)

    root = arguments.evidence
    out = arguments.out or (root / "figures")
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "plan" / "RANKED-CANDIDATES.json").read_text("utf-8"))
    summary = json.loads((root / "plan" / "PLAN-SUMMARY.json").read_text("utf-8"))
    with np.load(root / "plan" / "LAND-MASK.npz") as handle:
        land = {"lat": handle["lat"], "lon": handle["lon"]}

    written = [
        figure_threat_map(data, summary, land, out / "threat-map.png"),
        figure_commensurability(data, out / "commensurability.png"),
        figure_rank_composition(data, out / "rank-composition.png"),
        figure_row_inventory(summary, out / "row-inventory.png"),
    ]
    print(json.dumps(
        {"figures": [str(path) for path in written],
         "history_sha256": data["history_sha256"],
         "metrics_sha256": data["metrics_sha256"],
         "policy_sha256": data["policy_sha256"]},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
