"""Evidence figures for the first forecast on a PLACED mesh.

These are ANALYSIS charts over grid geometry and over scalars the model
reported about itself.  No weather field is drawn here -- weather-field
product plots come from ``rw_mpas_convert`` + ``rw_wrfbatch`` through the
render door, which is where the run's own imagery comes from.

Figures:

1. ``01-the-grid-where-it-was-placed.png`` -- cell spacing in kilometres
   over the placed region, with the swath polygon the placement machinery
   emitted drawn on top.  Shows the 4.5 km core sitting inside the 75 km
   background exactly where the spec asked for it.
2. ``02-the-mesh-on-the-globe.png`` -- the same spacing field over the whole
   sphere, so the fine patch can be seen for what it is: one small region
   of a global mesh.
3. ``03-the-w-band.png`` -- the half-hour |w| means this run measured,
   against the 120 s reference band and the 20 s anchor's band, both from
   ``x1.40962``.  The caption says plainly that those are an orientation
   and not a like-for-like control.
4. ``04-resolution-histogram.png`` -- how many cells sit at each spacing.
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
from matplotlib.collections import PolyCollection

EARTH_RADIUS_M = 6_371_229.0


def _cell_spacing_km(grid: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell mean dcEdge in km, with cell centre lat/lon in degrees."""

    import netCDF4

    with netCDF4.Dataset(str(grid)) as dataset:
        lat = np.degrees(np.asarray(dataset.variables["latCell"][:], dtype=np.float64))
        lon = np.degrees(np.asarray(dataset.variables["lonCell"][:], dtype=np.float64))
        dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges = np.asarray(dataset.variables["edgesOnCell"][:])
        counts = np.asarray(dataset.variables["nEdgesOnCell"][:])
        radius = float(getattr(dataset, "sphere_radius", 1.0) or 1.0)
    if radius < 1000.0:  # unit sphere -> metres
        dc = dc * EARTH_RADIUS_M
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    # The generator and the static writer disagree with native MPAS about
    # padding: native writes 0 into unused edgesOnCell slots (one-based, so
    # 0 means "none"), while a verbatim copy can carry a real one-based index
    # there.  Decide from the data rather than assuming, because reading a
    # one-based table as zero-based silently shifts every cell's spacing by
    # one edge instead of failing.
    one_based = int(edges.max()) >= dc.size
    spacing = np.zeros(lat.size, dtype=np.float64)
    for cell in range(lat.size):
        n = int(counts[cell])
        idx = np.asarray(edges[cell, :n], dtype=np.int64)
        idx = idx - 1 if one_based else idx
        idx = idx[(idx >= 0) & (idx < dc.size)]
        spacing[cell] = dc[idx].mean() / 1000.0 if idx.size else np.nan
    return lat, lon, spacing


def _polygon(spec_regions: Sequence[dict[str, Any]]) -> np.ndarray | None:
    for region in spec_regions:
        shape = region.get("shape", {})
        if shape.get("kind") == "polygon":
            ring = np.asarray(shape["vertices_deg"], dtype=np.float64)
            lon = np.where(ring[:, 1] > 180.0, ring[:, 1] - 360.0, ring[:, 1])
            return np.column_stack([ring[:, 0], lon])
    return None


def figure_region(
    lat: np.ndarray,
    lon: np.ndarray,
    spacing: np.ndarray,
    ring: np.ndarray | None,
    out: Path,
) -> None:
    if ring is not None:
        pad = 9.0
        lat0, lat1 = ring[:, 0].min() - pad, ring[:, 0].max() + pad
        lon0, lon1 = ring[:, 1].min() - pad, ring[:, 1].max() + pad
    else:
        lat0, lat1, lon0, lon1 = 5.0, 30.0, -70.0, -40.0
    keep = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1)
    figure, axes = plt.subplots(figsize=(11.0, 8.5), dpi=150)
    scatter = axes.scatter(
        lon[keep], lat[keep], c=spacing[keep], s=2.2, cmap="viridis_r",
        vmin=float(np.nanmin(spacing)), vmax=float(np.nanpercentile(spacing[keep], 99.5)),
        linewidths=0.0,
    )
    if ring is not None:
        closed = np.vstack([ring, ring[:1]])
        axes.plot(closed[:, 1], closed[:, 0], color="crimson", linewidth=2.2,
                  label="the swath the placement machinery asked for")
        axes.legend(loc="upper right", framealpha=0.92, fontsize=9)
    bar = figure.colorbar(scatter, ax=axes, pad=0.02)
    bar.set_label("cell spacing (km)")
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(
        "The grid, where it was placed\n"
        f"{spacing.size:,} cells; finest {np.nanmin(spacing):.2f} km inside a "
        f"{np.nanmax(spacing):.0f} km background"
    )
    axes.set_xlim(lon0, lon1)
    axes.set_ylim(lat0, lat1)
    axes.grid(alpha=0.25, linewidth=0.5)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)


def figure_globe(
    lat: np.ndarray, lon: np.ndarray, spacing: np.ndarray, ring: np.ndarray | None, out: Path
) -> None:
    figure, axes = plt.subplots(figsize=(13.0, 6.6), dpi=150)
    order = np.argsort(-spacing)  # coarse first, so the fine patch draws on top
    scatter = axes.scatter(
        lon[order], lat[order], c=spacing[order], s=0.7, cmap="viridis_r", linewidths=0.0
    )
    if ring is not None:
        closed = np.vstack([ring, ring[:1]])
        axes.plot(closed[:, 1], closed[:, 0], color="crimson", linewidth=1.6)
    bar = figure.colorbar(scatter, ax=axes, pad=0.02)
    bar.set_label("cell spacing (km)")
    axes.set_xlim(-180, 180)
    axes.set_ylim(-90, 90)
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(
        "One global mesh, refined in one place\n"
        f"spacing ratio {np.nanmax(spacing) / np.nanmin(spacing):.1f}x across the sphere"
    )
    axes.grid(alpha=0.2, linewidth=0.4)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)


def figure_w_band(band: dict[str, Any], out: Path) -> None:
    windows = [w for w in band["windows"] if w["complete"]]
    mine = [w["w_abs_max_mean"] for w in windows]
    hours = [w["to_hours"] for w in windows]
    control = band["reference_120s_x1"]
    anchor = band["anchor_20s_gf_x1"]
    figure, axes = plt.subplots(figsize=(10.5, 6.6), dpi=150)
    axes.plot(hours, mine, marker="o", linewidth=2.4, color="#1b6ca8",
              label=f"this run: {band['mesh']}, dt {band['dt_seconds']:g} s "
                    f"(Courant limit {band['courant_limit_seconds']:.1f} s)")
    ref_hours = [0.5, 1.0, 1.5, 2.0]
    axes.plot(ref_hours, list(anchor["window_means"]), marker="s", linestyle="--",
              linewidth=2.0, color="#c1443c",
              label="anchor: x1.40962, dt 20 s (limit 698.95 s) -- 35x below")
    axes.plot(ref_hours, list(control["window_means"]), marker="^", linestyle=":",
              linewidth=2.0, color="#3f7d3f",
              label="reference: x1.40962, dt 120 s")
    axes.set_xlabel("forecast hour (window end)")
    axes.set_ylabel("mean of per-step |w| max over the window (m/s)")
    axes.set_title(
        "Vertical velocity, half-hour by half-hour\n"
        "the two dashed bands are a DIFFERENT mesh -- an orientation, not a control"
    )
    axes.grid(alpha=0.3, linewidth=0.5)
    axes.legend(fontsize=8.5, loc="upper left")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)


def figure_histogram(spacing: np.ndarray, out: Path) -> None:
    figure, axes = plt.subplots(figsize=(9.5, 5.8), dpi=150)
    axes.hist(spacing, bins=90, color="#3b6ea5", edgecolor="none")
    axes.set_yscale("log")
    axes.set_xlabel("cell spacing (km)")
    axes.set_ylabel("cells (log scale)")
    axes.set_title(
        "How the resolution is spent\n"
        f"{int((spacing < 10.0).sum()):,} cells finer than 10 km out of "
        f"{spacing.size:,}"
    )
    axes.grid(alpha=0.3, linewidth=0.5, axis="y")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--mesh-receipt", type=Path, default=None,
                        help="the generator receipt carrying the swath spec")
    parser.add_argument("--w-band", type=Path, default=None,
                        help="measure_w_band.py output")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arguments.out.mkdir(parents=True, exist_ok=True)

    lat, lon, spacing = _cell_spacing_km(arguments.grid)
    ring = None
    if arguments.mesh_receipt is not None:
        document = json.loads(arguments.mesh_receipt.read_text(encoding="utf-8"))
        regions = document.get("receipt", {}).get("spec", {}).get("regions", [])
        ring = _polygon(regions)

    figure_region(lat, lon, spacing, ring, arguments.out / "01-the-grid-where-it-was-placed.png")
    figure_globe(lat, lon, spacing, ring, arguments.out / "02-the-mesh-on-the-globe.png")
    figure_histogram(spacing, arguments.out / "04-resolution-histogram.png")
    if arguments.w_band is not None and arguments.w_band.is_file():
        band = json.loads(arguments.w_band.read_text(encoding="utf-8"))
        figure_w_band(band, arguments.out / "03-the-w-band.png")

    print(json.dumps({
        "cells": int(spacing.size),
        "finest_km": float(np.nanmin(spacing)),
        "coarsest_km": float(np.nanmax(spacing)),
        "cells_below_10km": int((spacing < 10.0).sum()),
        "out": str(arguments.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
