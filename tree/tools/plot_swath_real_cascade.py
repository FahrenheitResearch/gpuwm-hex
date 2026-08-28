"""Evidence figures for the cascade loop closing on a REAL detected storm.

WHAT IS AND IS NOT DRAWN HERE.  Every figure this module draws is an ANALYSIS
chart over grid geometry, over the detector's own decision document, or over
scalars the model reported about itself.  **No weather field is drawn here.**
The run's weather imagery comes from ``rw_mpas_convert`` + ``rw_wrfbatch``
through ``gpuwm-hex render`` and is delivered unmodified beside these.

WHY THERE IS NO COMPOSITE OF THE RING ONTO A RENDERED WEATHER RASTER, MEASURED.
Drawing the placed ring on top of the Rust renderer's own global PNG needs that
PNG's geographic transform.  The renderer publishes none -- neither the render
manifest nor the PNG carries the plot rectangle or the projection it drew -- so
the transform can only be recovered by registration.  It was attempted, against
the converter's own 0.25 degree frame for the same valid time: the best linear
plate-carree fit put 4.175 px/deg across longitude against 4.38 px/deg across
latitude, which no equirectangular map has, and only 25 per cent of the
computed 10 dBZ mask landed on coloured pixels.  A mis-registered composite
draws the placed grid over the wrong ocean, so no composite is published and
the fix is named as engine work: the render manifest should carry each PNG's
geographic transform, and then an annotation layer is data rather than
reverse engineering.

Figures:

1. ``02-the-grid-on-the-globe.png`` -- cell spacing over the whole sphere with
   the emitted swath ring and the detected storm's track: one small fine
   region of one global mesh, where the detector put it.
2. ``03-the-grid-where-it-was-placed.png`` -- the same spacing field zoomed to
   the placed region, with the ring and the track.
3. ``04-what-the-detector-ranked.png`` -- every admitted swath's score
   decomposed into its four terms, and the class split that decided which
   metric rows could place at all.
4. ``05-the-w-band.png`` -- this run's half-hour |w| means against the two
   x1.40962 bands the dt anchors quote and against the previous placed mesh.
5. ``06-resolution-histogram.png`` -- cells per spacing bin, against the
   coarse forecast that found the storm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EARTH_RADIUS_M = 6_371_229.0


# --------------------------------------------------------------------------
# grid geometry
# --------------------------------------------------------------------------
def cell_spacing_km(grid: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell mean ``dcEdge`` in km, with cell centre lat/lon in degrees."""

    import netCDF4

    with netCDF4.Dataset(str(grid)) as dataset:
        lat = np.degrees(np.asarray(dataset.variables["latCell"][:], dtype=np.float64))
        lon = np.degrees(np.asarray(dataset.variables["lonCell"][:], dtype=np.float64))
        dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges = np.asarray(dataset.variables["edgesOnCell"][:])
        counts = np.asarray(dataset.variables["nEdgesOnCell"][:])
        radius = float(getattr(dataset, "sphere_radius", 1.0) or 1.0)
    if radius < 1000.0:  # a grid file is on the unit sphere
        dc = dc * EARTH_RADIUS_M
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    one_based = int(edges.max()) >= dc.size
    spacing = np.zeros(lat.size, dtype=np.float64)
    for cell in range(lat.size):
        n = int(counts[cell])
        idx = np.asarray(edges[cell, :n], dtype=np.int64)
        idx = idx - 1 if one_based else idx
        idx = idx[(idx >= 0) & (idx < dc.size)]
        spacing[cell] = dc[idx].mean() / 1000.0 if idx.size else np.nan
    return lat, lon, spacing


def _unwrap(lons: np.ndarray, reference: float) -> np.ndarray:
    out = np.asarray(lons, dtype=np.float64).copy()
    out = np.where(out - reference > 180.0, out - 360.0, out)
    out = np.where(reference - out > 180.0, out + 360.0, out)
    return out


# --------------------------------------------------------------------------
# 2/3. mesh geometry
# --------------------------------------------------------------------------
def figure_region(
    lat: np.ndarray,
    lon: np.ndarray,
    spacing: np.ndarray,
    ring: np.ndarray | None,
    track_points: Sequence[tuple[float, float, float]],
    out: Path,
    *,
    title: str,
) -> None:
    finest = float(np.nanmin(spacing))
    core = spacing <= 2.0 * finest
    centre_lat = float(np.mean(lat[core]))
    centre_lon = float(np.mean(_unwrap(lon[core], float(np.median(lon[core])))))
    span = 14.0
    window = (
        (lat > centre_lat - span)
        & (lat < centre_lat + span)
        & (np.abs(_unwrap(lon, centre_lon) - centre_lon) < span / max(0.2, math.cos(math.radians(centre_lat))))
    )
    fig, axes = plt.subplots(figsize=(11.5, 8.6))
    scatter = axes.scatter(
        _unwrap(lon[window], centre_lon), lat[window], c=spacing[window],
        s=1.4, cmap="viridis_r", vmin=0.0, vmax=float(np.nanpercentile(spacing, 99)),
    )
    if ring is not None:
        closed = np.vstack([ring, ring[:1]])
        axes.plot(_unwrap(closed[:, 1], centre_lon), closed[:, 0],
                  color="#d81b3c", lw=2.2, label="the swath the detector asked for")
    if track_points:
        tlat = np.array([p[1] for p in track_points])
        tlon = _unwrap(np.array([p[2] for p in track_points]), centre_lon)
        axes.plot(tlon, tlat, color="#1b3cd8", lw=1.6, marker="o", ms=3.5,
                  label="the storm's track in the coarse forecast")
        axes.plot(tlon[0], tlat[0], marker="*", ms=16, color="#1b3cd8",
                  mec="white", mew=1.0)
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(title)
    axes.grid(alpha=0.25)
    axes.legend(loc="upper right", fontsize=9)
    fig.colorbar(scatter, ax=axes, label="cell spacing (km)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_globe(
    lat: np.ndarray, lon: np.ndarray, spacing: np.ndarray,
    ring: np.ndarray | None, track_points: Sequence[tuple[float, float, float]],
    out: Path, *, title: str,
) -> None:
    fig, axes = plt.subplots(figsize=(13.0, 6.8))
    step = max(1, lat.size // 90_000)
    scatter = axes.scatter(lon[::step], lat[::step], c=spacing[::step], s=0.7,
                           cmap="viridis_r", vmin=0.0,
                           vmax=float(np.nanpercentile(spacing, 99)))
    if ring is not None:
        closed = np.vstack([ring, ring[:1]])
        axes.plot(closed[:, 1], closed[:, 0], color="#d81b3c", lw=2.0,
                  label="the grid the detector placed")
    if track_points:
        tlat = np.array([p[1] for p in track_points])
        tlon = np.array([p[2] for p in track_points])
        axes.plot(tlon, tlat, color="#1b3cd8", lw=1.6, marker="o", ms=3.0,
                  label="the storm's track in the coarse forecast")
        axes.plot(tlon[0], tlat[0], marker="*", ms=16, color="#1b3cd8",
                  mec="white", mew=1.0)
    axes.legend(loc="upper right", fontsize=9)
    axes.set_xlim(-180, 180)
    axes.set_ylim(-90, 90)
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(title)
    axes.grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes, label="cell spacing (km)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4/5/6. scalars
# --------------------------------------------------------------------------
def figure_w_band(band: dict[str, Any], previous: dict[str, Any] | None, out: Path) -> None:
    windows = [float(w["to_hours"]) for w in band["windows"]]
    means = [float(w["w_abs_max_mean"]) for w in band["windows"]]
    fig, axes = plt.subplots(figsize=(10.5, 6.0))
    axes.plot(windows, means, marker="o", lw=2.2, color="#1b7f3c",
              label=f"this run, {band['dt_seconds']:.0f} s on a real detected storm")
    if previous:
        pw = [float(w["to_hours"]) for w in previous["windows"]]
        pm = [float(w["w_abs_max_mean"]) for w in previous["windows"]]
        axes.plot(pw, pm, marker="s", lw=1.6, ls="--", color="#7f7f7f",
                  label=f"the previous placed mesh ({previous.get('mesh','?')}), "
                        f"same dt, a FIXTURE placement over the tropical Atlantic")
    control = band.get("reference_120s_x1") or {}
    if control.get("window_means"):
        axes.plot([0.5 * (i + 1) for i in range(len(control["window_means"]))],
                  control["window_means"], marker="^", lw=1.4, ls=":",
                  color="#2b5fb5", label="120 s reference band (x1.40962)")
    anchor = band.get("anchor_20s_gf_x1") or {}
    if anchor.get("window_means"):
        axes.plot([0.5 * (i + 1) for i in range(len(anchor["window_means"]))],
                  anchor["window_means"], marker="v", lw=1.4, ls=":",
                  color="#c0392b", label="the 20 s anchor's own band (x1.40962, 35x below limit)")
    axes.set_xlabel("forecast hour (end of half-hour window)")
    axes.set_ylabel("mean of per-step |w| max (m/s)")
    axes.set_title("The vertical-velocity band, half-hour means")
    axes.grid(alpha=0.3)
    axes.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_w_by_region(document: dict[str, Any], out: Path) -> None:
    """What the refined core resolves against what the coarse grid did."""

    frames = [f for f in document["frames"] if "coarse_same_footprint" in f]
    if not frames:
        return
    labels = [f["frame"].split("cuda-history.")[-1].replace(".nc", "").replace("_", " ")
              for f in frames]
    fine_max = [f["refined_core"]["max"] for f in frames]
    fine_p = [f["refined_core"]["p99_9"] for f in frames]
    coarse_max = [f["coarse_same_footprint"]["max"] for f in frames]
    coarse_p = [f["coarse_same_footprint"]["p99_9"] for f in frames]
    x = np.arange(len(frames))
    fig, axes = plt.subplots(figsize=(10.5, 6.0))
    axes.bar(x - 0.19, fine_max, width=0.36, color="#c0392b",
             label=f"fine core, {document['refined_cells']:,} cells at "
                   f"{document['finest_spacing_m']/1000.0:.2f}-"
                   f"{2*document['finest_spacing_m']/1000.0:.2f} km")
    axes.bar(x + 0.19, coarse_max, width=0.36, color="#7f8fa6",
             label=f"the coarse forecast, {document['coarse_cells_in_footprint']} cells "
                   f"over the same ground")
    axes.plot(x - 0.19, fine_p, "o", color="black", ms=5, label="99.9th percentile")
    axes.plot(x + 0.19, coarse_p, "o", color="black", ms=5)
    for i, (a, b) in enumerate(zip(fine_max, coarse_max)):
        if b > 0:
            axes.text(i, max(a, b) * 1.04, f"x{a/b:.2f}", ha="center", fontsize=10)
    axes.set_xticks(x)
    axes.set_xticklabels(labels, fontsize=9)
    axes.set_ylim(0.0, max(max(fine_max), max(coarse_max)) * 1.18)
    axes.set_ylabel("|w| max over the region (m/s)")
    axes.set_title("Vertical motion the fine grid resolves, against the grid that found the storm")
    axes.grid(alpha=0.3, axis="y")
    axes.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_histogram(spacing: np.ndarray, coarse_km: float, out: Path) -> None:
    fig, axes = plt.subplots(figsize=(10.0, 5.8))
    axes.hist(spacing, bins=90, color="#2b5fb5", log=True)
    axes.axvline(float(np.nanmin(spacing)), color="#d81b3c", lw=1.8,
                 label=f"finest {np.nanmin(spacing):.2f} km")
    axes.axvline(coarse_km, color="#7f7f7f", lw=1.8, ls="--",
                 label=f"the coarse forecast that found the storm: {coarse_km:.0f} km")
    axes.set_xlabel("cell spacing (km)")
    axes.set_ylabel("cells (log)")
    axes.set_title("Where the mesh spends its resolution")
    axes.grid(alpha=0.3)
    axes.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_ranking(plan: dict[str, Any], out: Path) -> None:
    admitted = plan["admitted"]
    labels = []
    terms: dict[str, list[float]] = {}
    for row in admitted:
        centre = row["centroid_deg"]
        labels.append(f"{row['slot_id']}\n{centre[0]:.1f}, {centre[1]:.1f}")
        for term in row["rank"]["terms"]:
            terms.setdefault(term["id"], []).append(float(term["contribution"]))
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.8),
                                      gridspec_kw={"width_ratios": [1.5, 1.0]})
    bottom = np.zeros(len(labels))
    colours = {"intensity": "#c0392b", "persistence": "#2b5fb5",
               "travel": "#1b7f3c", "extent": "#e08e0b"}
    for name, values in terms.items():
        left.bar(labels, values, bottom=bottom, label=name,
                 color=colours.get(name, "#888888"))
        bottom = bottom + np.asarray(values)
    left.set_ylabel("score contribution")
    left.set_title("What the detector ranked, and why")
    left.legend(fontsize=9)
    left.grid(alpha=0.25, axis="y")

    from collections import Counter

    counts = Counter(t["metric_id"] for t in plan["tracks"])
    placed = Counter(r["metric_id"] for r in admitted)
    names = sorted(counts)
    right.bar(names, [counts[n] for n in names], color="#9fb8dd", label="tracks formed")
    right.bar(names, [placed.get(n, 0) for n in names], color="#c0392b", label="grids placed")
    right.set_title("Tracks formed against grids placed")
    right.set_ylabel("count")
    right.tick_params(axis="x", labelrotation=12)
    right.legend(fontsize=9)
    right.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--slot", default="s01")
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--coarse-km", type=float, default=96.0)
    parser.add_argument("--w-band", type=Path, default=None)
    parser.add_argument("--previous-w-band", type=Path, default=None)
    parser.add_argument("--w-by-region", type=Path, default=None)
    parser.add_argument("--region-title", default="The grid, where it was placed")
    parser.add_argument("--globe-title", default="One global mesh, one fine region")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    rows = [r for r in plan["admitted"] if r["slot_id"] == arguments.slot]
    if not rows:
        raise SystemExit(f"plan holds no slot {arguments.slot!r}")
    row = rows[0]
    ring = np.asarray(row["ring_deg"], dtype=np.float64)
    track_points: list[tuple[float, float, float]] = []
    for track in plan["tracks"]:
        if track["track_id"] == row["track_id"]:
            track_points = [
                (float(p["time_seconds"]), float(p["latitude_deg"]), float(p["longitude_deg"]))
                for p in track["points"]
            ]
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if arguments.grid is not None:
        lat, lon, spacing = cell_spacing_km(arguments.grid)
        figure_globe(lat, lon, spacing, ring, track_points,
                     out / "02-the-grid-on-the-globe.png",
                     title=arguments.globe_title)
        figure_region(lat, lon, spacing, ring, track_points,
                      out / "03-the-grid-where-it-was-placed.png",
                      title=arguments.region_title)
        figure_histogram(spacing, arguments.coarse_km,
                         out / "06-resolution-histogram.png")
        written += ["02-the-grid-on-the-globe.png",
                    "03-the-grid-where-it-was-placed.png",
                    "06-resolution-histogram.png"]

    if arguments.w_band is not None:
        band = json.loads(arguments.w_band.read_text(encoding="utf-8"))
        previous = (
            json.loads(arguments.previous_w_band.read_text(encoding="utf-8"))
            if arguments.previous_w_band is not None else None
        )
        figure_w_band(band, previous, out / "05-the-w-band.png")
        written.append("05-the-w-band.png")

    if arguments.w_by_region is not None:
        figure_w_by_region(
            json.loads(arguments.w_by_region.read_text(encoding="utf-8")),
            out / "07-what-the-fine-grid-resolved.png")
        written.append("07-what-the-fine-grid-resolved.png")

    figure_ranking(plan, out / "04-what-the-detector-ranked.png")
    written.append("04-what-the-detector-ranked.png")
    print(json.dumps({"out": str(out), "figures": sorted(written)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
