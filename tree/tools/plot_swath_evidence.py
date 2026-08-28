"""Analysis charts for the swath placement lane.

RENDER LAW.  Not one of these is a weather-field product plot.  Chart 1
draws swath OUTLINES and track AXES as line geometry in plain
latitude/longitude with no basemap, no field and no colour scale; charts
2-4 are numbers on axes.  Weather-field frames for this capability come
from ``rw_mpas_convert`` + ``rw_wrfbatch`` and are not produced here,
because this lane runs no forecast.

Every chart states on its own face what it does NOT show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

EVIDENCE = ROOT / "evidence" / "swath-following-20260826"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_antimeridian(points: Sequence[Sequence[float]]) -> list[list[list[float]]]:
    """Break a lon/lat polyline where it crosses +/-180 so a flat plot does
    not draw a line all the way across the frame."""

    runs: list[list[list[float]]] = [[]]
    for index, point in enumerate(points):
        if index and abs(point[1] - points[index - 1][1]) > 180.0:
            runs.append([])
        runs[-1].append([point[1], point[0]])
    return [run for run in runs if len(run) > 1]


def chart_geometry(plan: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    admitted = plan["admitted"]
    figure = plt.figure(figsize=(13.5, 9.6))
    grid = figure.add_gridspec(
        2, max(1, len(admitted)), height_ratios=[2.1, 1.0], hspace=0.42, wspace=0.28
    )
    axes = figure.add_subplot(grid[0, :])
    colours = plt.cm.tab10.colors
    for index, row in enumerate(admitted):
        colour = colours[index % len(colours)]
        ring = row["ring_deg"] + [row["ring_deg"][0]]
        for run in _split_antimeridian(ring):
            axes.plot(
                [p[0] for p in run], [p[1] for p in run],
                color=colour, linewidth=2.0, zorder=3,
            )
        for run in _split_antimeridian(row["path_deg"]):
            axes.plot(
                [p[0] for p in run], [p[1] for p in run],
                color=colour, linewidth=1.0, linestyle="--", zorder=4,
            )
        axes.plot(
            [row["path_deg"][0][1]], [row["path_deg"][0][0]],
            marker="o", markersize=6, color=colour, zorder=5,
        )
        axes.annotate(
            f"{row['slot_id']}  {row['threat_class']}\n"
            f"ignite {row['ignite_at_seconds'] / 3600.0:.0f} h, lead "
            f"{row['lead_hours']:.0f} h",
            xy=(row["centroid_deg"][1], row["centroid_deg"][0]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=8, color=colour, zorder=6,
        )
    axes.set_xlim(-180, 180)
    axes.set_ylim(-90, 90)
    axes.set_xticks(range(-180, 181, 30))
    axes.set_yticks(range(-90, 91, 30))
    axes.grid(alpha=0.25, linewidth=0.5)
    axes.set_xlabel("longitude (degrees east)")
    axes.set_ylabel("latitude (degrees north)")
    axes.set_title(
        f"Where the machine put the fine grid: {len(admitted)} swaths, one cycle, "
        "one decision",
        fontsize=12,
    )

    # The flare is the signature and it is invisible at global scale, so
    # every swath also gets its own frame at its own extent.
    for index, row in enumerate(admitted):
        colour = colours[index % len(colours)]
        panel = figure.add_subplot(grid[1, index])
        ring = row["ring_deg"] + [row["ring_deg"][0]]
        panel.plot(
            [p[1] for p in ring], [p[0] for p in ring],
            color=colour, linewidth=1.8,
        )
        panel.plot(
            [p[1] for p in row["path_deg"]], [p[0] for p in row["path_deg"]],
            color=colour, linewidth=1.0, linestyle="--", marker="o", markersize=3,
        )
        widths = row["half_widths_km"]
        panel.set_title(
            f"{row['slot_id']}   half-width {widths[0]:.0f} -> {widths[-1]:.0f} km",
            fontsize=9,
        )
        panel.tick_params(labelsize=7)
        panel.grid(alpha=0.25, linewidth=0.5)
        panel.set_aspect("equal", adjustable="datalim")
    figure.text(
        0.5, 0.055,
        "Solid outline: the region row handed to the mesh generator and the region culler. "
        "Dashed: the projected track it was built on. Dot: where that track starts.\n"
        "Lower row is the same four outlines at their own extent, where the flare is visible: "
        "narrow on the storm, widening along the forecast track as the track error grows.\n"
        "GEOMETRY ONLY. No meteorological field is drawn here and none is claimed; "
        "weather-field frames come from rw_wrfbatch and this lane runs no forecast.",
        ha="center", va="top", fontsize=8.5,
    )
    figure.subplots_adjust(bottom=0.17, top=0.94)
    target = out / "01-swath-geometry.png"
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


def chart_calibration(chain: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = chain["estimator_error_curve"]
    swaths = chain["calibration"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    left.plot(
        [row["measured_region_cells"] for row in curve],
        [row["predicted_cells_in"] for row in curve],
        marker="o", label="calibration caps",
    )
    left.plot(
        [row["measured_region_cells"] for row in swaths],
        [row["predicted_cells_in"] for row in swaths],
        marker="s", linestyle="none", label="emitted swath polygons",
    )
    limit = max(row["measured_region_cells"] for row in curve) * 1.1
    left.plot([0, limit], [0, limit], color="0.6", linewidth=1.0, label="exact")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("cells the shipped culler actually emitted")
    left.set_ylabel("cells predicted_cells_in estimated")
    left.set_title("The estimator against the culler")
    left.legend(fontsize=8)
    left.grid(alpha=0.3, which="both", linewidth=0.5)

    right.axhline(0.0, color="0.6", linewidth=1.0)
    right.axhline(5.0, color="0.8", linewidth=1.0, linestyle="--")
    right.axhline(-5.0, color="0.8", linewidth=1.0, linestyle="--")
    right.plot(
        [row["cells_across_radius"] for row in curve],
        [row["relative_error"] * 100.0 for row in curve],
        marker="o", label="calibration caps",
    )
    for row in swaths:
        right.plot(
            [2.5], [row["relative_error"] * 100.0],
            marker="s", linestyle="none", color="tab:orange",
        )
    right.set_xscale("log")
    right.set_xlabel("region radius, in cells of the parent mesh")
    right.set_ylabel("estimator error, per cent")
    right.set_title("Error, and where it comes from")
    right.grid(alpha=0.3, which="both", linewidth=0.5)
    figure.text(
        0.5, -0.02,
        "Culled from the real published x1.40962 parent (40,962 cells, 119.9 km) by the shipped "
        "rw_mpas_mesh --cull-parent. Orange squares are the four swath polygons this lane emitted, "
        "at 2.5 cells across -- the worst regime there is.\n"
        "This measures the CELL ESTIMATOR. It does not measure a forecast, and no swath in it has run.",
        ha="center", fontsize=8.5,
    )
    target = out / "02-estimator-calibration.png"
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return target


def chart_hysteresis(measurement: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("evictions", "mesh_generate", "mesh_reuse", "new_slots")
    labels = (
        "slots evicted",
        "meshes rebuilt",
        "meshes reused",
        "cold-start slots",
    )
    armed = [measurement["armed"][key] for key in keys]
    disarmed = [measurement["disarmed"][key] for key in keys]

    figure, axes = plt.subplots(figsize=(9.5, 5.4))
    positions = range(len(keys))
    axes.bar([p - 0.2 for p in positions], disarmed, width=0.4, label="rule disarmed")
    axes.bar([p + 0.2 for p in positions], armed, width=0.4, label="rule armed (shipped)")
    for position, (a, d) in enumerate(zip(armed, disarmed)):
        axes.text(position - 0.2, d + 0.2, str(d), ha="center", fontsize=9)
        axes.text(position + 0.2, a + 0.2, str(a), ha="center", fontsize=9)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels)
    axes.set_ylabel(f"count over {measurement['cycles']} cycles")
    axes.set_title("What the hysteresis rule prevents, counted both ways")
    axes.legend()
    axes.grid(axis="y", alpha=0.3, linewidth=0.5)
    figure.text(
        0.5, -0.04,
        "Same placement sequence, same forecasts, same ranking: only the hysteresis knobs differ. "
        f"Disarmed, the second slot trades every cycle -- {measurement['disarmed']['evictions']} evictions "
        f"in {measurement['cycles']} cycles. Armed, {measurement['armed']['evictions']}.\n"
        f"At the program plan's QUOTED per-slot rate that is "
        f"{measurement['abandoned_gpu_minutes_avoided']:.0f} GPU-minutes of fine forecast whose product "
        "nothing continues. The rate is quoted, not measured here.",
        ha="center", fontsize=8.5,
    )
    target = out / "03-hysteresis-churn.png"
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return target


def chart_attainment(probe: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = probe["rows"]
    figure, axes = plt.subplots(figsize=(9.5, 5.4))
    widths = [row["half_width_km"] for row in rows]
    axes.plot(
        widths, [row["cap"]["attained_spacing_km"] for row in rows],
        marker="o", label="cap region (the control): honest",
    )
    axes.plot(
        widths, [row["polygon"]["attained_spacing_km"] for row in rows],
        marker="s", label="polygon region: reports the request met at every size",
    )
    axes.axhline(
        probe["requested_spacing_km"], color="0.6", linestyle="--", linewidth=1.0,
        label=f"requested {probe['requested_spacing_km']:.1f} km",
    )
    axes.set_xscale("log")
    axes.set_xlabel("region half-width, km")
    axes.set_ylabel("attained spacing the generator reports, km")
    axes.set_title("Why this lane does not quote the generator's polygon attainment")
    axes.legend(fontsize=8.5)
    axes.grid(alpha=0.3, which="both", linewidth=0.5)
    figure.text(
        0.5, -0.06,
        "Same binary, same request, same ramp. A 22 km cap correctly says it reaches only 6.36 km; "
        "a polygon covering comparable ground says 4.00 km at every size, with a reported interior "
        "depth near 19,900 km.\n"
        "The emitted MESHES are fine -- a polygon's cell count tracks the equivalent cap's -- so this "
        "is a reporting defect, and the swath layer measures attainment with an inscribed cap probe instead.",
        ha="center", fontsize=8.5,
    )
    target = out / "04-attained-vs-requested.png"
    figure.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plot_swath_evidence", description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    out = arguments.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    plan = _load(arguments.evidence / "chain" / "swath-plan.json")
    chain = _load(arguments.evidence / "chain" / "CHAIN.json")
    hysteresis = _load(arguments.evidence / "hysteresis" / "HYSTERESIS.json")
    probe = _load(arguments.evidence / "polygon-attainment-probe.json")

    written = [
        chart_geometry(plan, out),
        chart_calibration(chain, out),
        chart_hysteresis(hysteresis, out),
        chart_attainment(probe, out),
    ]
    for path in written:
        print(f"{path}  {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
