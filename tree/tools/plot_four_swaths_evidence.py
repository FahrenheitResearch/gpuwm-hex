"""The evidence charts for a cycle of several placed grids, from its own files.

ANALYSIS CHARTS, NOT WEATHER FIELDS.  Nothing here draws a meteorological
field.  These are a placement map, a sizing comparison, a resolution-gain bar
chart, the |w| bands and the memory row -- every one of them read off JSON
this cycle wrote.  The render law's matplotlib carve-out is for exactly this;
every weather field in this lane's gallery came through ``gpuwm-hex render``
and the Rust pair.

THE MAP'S BACKGROUND IS THE MODEL'S OWN GEOGRAPHY.  No coastline dataset and
no cartopy: land is the set of coarse-parent cells whose top soil layer is not
the 1.0 the land surface writes over water, so the map is drawn out of the
same forecast the detection ran on and a coastline that disagreed with the
model's own would be impossible.

Usage::

    python tools/plot_four_swaths_evidence.py --evidence DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

#: Okabe-Ito, assigned to threat CLASS by name so a class keeps its colour
#: whichever slot it lands in.
CLASS_COLOURS = {
    "atmospheric_river": "#4B3F9E",
    "winter_storm": "#CC79A7",
    "extratropical_cyclone": "#0072B2",
    "tropical_cyclone": "#56B4E9",
    "deep_convection": "#D55E00",
    "heavy_rainfall": "#009E73",
    "fire_weather": "#E69F00",
    "damaging_wind": "#8C564B",
}

CLASS_LABELS = {
    "atmospheric_river": "atmospheric river",
    "winter_storm": "winter storm",
    "extratropical_cyclone": "extratropical cyclone",
}

FIELD_LABELS = {
    "w": "|w| (m/s)",
    "u10-v10": "10 m wind (m/s)",
    "rainnc": "grid-scale rain (mm)",
    "snownc": "snowfall (mm)",
}

#: The 120 s reference band on x1.40962, the band every dt anchor is read
#: against.  Quoted from hexcore.dt_admission rather than retyped.
CONTROL_LABEL = "x1.40962 at 120 s (reference)"


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _slots(evidence: Path) -> list[dict]:
    plan = _load(evidence / "plan" / "swath-plan.json")
    rows = []
    for swath in plan["admitted"]:
        slot = swath["slot_id"]
        row = {
            "slot": slot,
            "threat": swath["threat_class"],
            "metric": swath["metric_id"],
            "centroid": swath["centroid_deg"],
            "ring": swath.get("ring_deg") or [],
            "score": swath["effective_score"],
            "sizing": swath.get("sizing", {}),
        }
        facts = evidence / "meshes" / f"{slot}.facts.json"
        if facts.exists():
            row["facts"] = _load(facts)
        run = evidence / "runs" / f"{slot}.run.json"
        if run.exists():
            row["run"] = _load(run)
        band = evidence / "measure" / f"{slot}.w-band.json"
        if band.exists():
            row["band"] = _load(band)
        spec = evidence / "plan" / "specs" / f"{slot}.mesh-spec.json"
        if spec.exists():
            document = _load(spec)
            row["requested_km"] = float(document["regions"][0]["spacing_km"])
            row["background_km"] = float(document["background_km"])
        row["by_region"] = {}
        for item in sorted((evidence / "measure").glob(f"{slot}.by-region.*.json")):
            field = item.name.split(".by-region.")[1][: -len(".json")]
            row["by_region"][field] = _load(item)
        rows.append(row)
    return rows


def _wrap(lon: float) -> float:
    return (float(lon) + 180.0) % 360.0 - 180.0


def placement_map(rows, coarse_mask, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(15.0, 8.6))
    if coarse_mask is not None:
        lat, lon, land = coarse_mask
        axes.scatter(lon[land], lat[land], s=6.0, c="#C9C2B4", linewidths=0,
                     rasterized=True, zorder=1)
    handles = []
    for index, row in enumerate(rows, start=1):
        colour = CLASS_COLOURS.get(row["threat"], "#333333")
        ring = np.array([[p[0], _wrap(p[1])] for p in row["ring"]], dtype=float)
        if ring.size:
            # A ring that crosses the dateline is drawn as the two arcs it
            # really is, never as a line back across the whole map.
            closed = np.vstack([ring, ring[:1]])
            breaks = np.nonzero(np.abs(np.diff(closed[:, 1])) > 180.0)[0]
            pieces = np.split(closed, breaks + 1) if breaks.size else [closed]
            for piece in pieces:
                if len(piece) > 1:
                    axes.plot(piece[:, 1], piece[:, 0], color=colour, lw=2.4,
                              zorder=3)
        lat_c, lon_c = row["centroid"][0], _wrap(row["centroid"][1])
        axes.plot([lon_c], [lat_c], marker="o", ms=13, color=colour,
                  markeredgecolor="white", markeredgewidth=1.4, zorder=4)
        axes.annotate(str(index), (lon_c, lat_c), ha="center", va="center",
                      fontsize=8, color="white", weight="bold", zorder=5)
        facts = row.get("facts", {})
        cells = facts.get("n_cells")
        spacing = facts.get("attained_finest_km")
        detail = (f"{cells:,} cells at {spacing:.2f} km"
                  if cells and spacing else "not built")
        run = row.get("run") or {}
        state = "ran" if str(run.get("rc")) == "0" else "did not run"
        handles.append(plt.Line2D(
            [], [], color=colour, lw=2.4, marker="o", ms=8,
            label=(f"{index}  {row['slot']}  "
                   f"{CLASS_LABELS.get(row['threat'], row['threat'])} -- "
                   f"{detail}, {state}")))
    axes.set_xlim(-180, 180)
    axes.set_ylim(-90, 90)
    axes.set_xticks(range(-180, 181, 30))
    axes.set_yticks(range(-90, 91, 30))
    axes.grid(color="#EEEEEE", lw=0.5)
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.set_title(
        "Four independent grids placed in one cycle, on four kinds of weather.\n"
        "Each ring is the swath one grid refines. The background is the model's "
        "own land mask, not a coastline dataset.")
    axes.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                ncol=2, frameon=False, fontsize=11)
    figure.tight_layout()
    figure.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(figure)


def sizing_chart(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    labels = [r["slot"] for r in rows]
    x = np.arange(len(rows))

    predicted = [r["sizing"].get("parent_cells", np.nan) for r in rows]
    delivered = [r.get("facts", {}).get("n_cells", np.nan) for r in rows]
    left.bar(x - 0.19, predicted, 0.38, label="predicted (generator dry run)",
             color="#9AA0A6")
    left.bar(x + 0.19, delivered, 0.38, label="delivered", color="#0072B2")
    for i, (p, d) in enumerate(zip(predicted, delivered)):
        if np.isfinite(p) and np.isfinite(d) and p:
            left.annotate(f"{abs(d - p) / p * 100:.3f}%", (i, max(p, d)),
                          ha="center", va="bottom", fontsize=9)
    left.set_ylim(0, max(max(predicted), max(delivered)) * 1.22)
    left.set_xticks(x, labels)
    left.set_ylabel("cells")
    left.set_title("Cell count: the dry run is exact")
    left.legend(fontsize=9)

    est = [r["sizing"].get("attained_spacing_km", np.nan) for r in rows]
    got = [r.get("facts", {}).get("attained_finest_km", np.nan) for r in rows]
    asked = [r.get("requested_km", np.nan) for r in rows]
    right.bar(x - 0.26, asked, 0.25, label="requested by the threat row",
              color="#D9D2C5")
    right.bar(x, est, 0.25, label="estimated (inscribed cap probe)", color="#9AA0A6")
    right.bar(x + 0.26, got, 0.25, label="delivered", color="#CC79A7")
    for i, (e, g) in enumerate(zip(est, got)):
        if np.isfinite(e) and np.isfinite(g) and g:
            right.annotate(f"{(e - g) / g * 100:+.1f}%", (i, max(e, g)),
                           ha="center", va="bottom", fontsize=9)
    right.set_ylim(0, max([v for v in est + got + asked if np.isfinite(v)]) * 1.3)
    right.set_xticks(x, labels)
    right.set_ylabel("finest spacing (km)")
    right.set_title("Finest spacing: the cap probe misses, both ways")
    right.legend(fontsize=9)
    figure.suptitle("What the sizing pass predicted, against what the "
                    "generator delivered")
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    plt.close(figure)


def _last_frame(document: dict) -> dict | None:
    frames = document.get("frames") or []
    return frames[-1] if frames else None


def resolution_gain(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields: list[str] = []
    for row in rows:
        for field in row["by_region"]:
            if field not in fields:
                fields.append(field)
    if not fields:
        return
    order = ["w", "u10-v10", "rainnc", "snownc"]
    fields = ([f for f in order if f in fields]
              + [f for f in fields if f not in order])
    figure, axes = plt.subplots(1, len(fields),
                                figsize=(4.6 * len(fields), 5.6), squeeze=False)
    for column, field in enumerate(fields):
        ax = axes[0][column]
        labels, fine, coarse = [], [], []
        for row in rows:
            document = row["by_region"].get(field)
            if document is None:
                continue
            frame = _last_frame(document)
            if frame is None:
                continue
            same = frame.get("coarse_same_footprint")
            if same is None:
                continue
            # A field that is identically zero on BOTH grids is not a
            # comparison: snowfall in a tropical corridor in August is absent
            # from the atmosphere, not unresolved by the parent.
            if frame["refined_core"]["max"] == 0.0 and same["max"] == 0.0:
                continue
            labels.append(row["slot"])
            fine.append(frame["refined_core"]["max"])
            coarse.append(same["max"])
        if not labels:
            ax.set_axis_off()
            continue
        x = np.arange(len(labels))
        ax.bar(x - 0.19, coarse, 0.38, label="96 km parent, same ground",
               color="#9AA0A6")
        ax.bar(x + 0.19, fine, 0.38, label="placed grid, refined core",
               color="#0072B2")
        for i, (c, f) in enumerate(zip(coarse, fine)):
            if c > 0:
                ax.annotate(f"{f / c:.2f}x", (i, max(c, f)), ha="center",
                            va="bottom", fontsize=10, weight="bold")
            else:
                ax.annotate("coarse = 0", (i, max(c, f)), ha="center",
                            va="bottom", fontsize=9)
        ax.set_ylim(0, max(max(coarse), max(fine)) * 1.22)
        ax.set_xticks(x, labels)
        ax.set_title(FIELD_LABELS.get(field, field))
        if column == 0:
            ax.set_ylabel("maximum over the refined footprint")
            ax.legend(fontsize=9)
    figure.suptitle("What each placed grid resolved that 96 km could not, "
                    "at the last forecast hour")
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    plt.close(figure)


def w_bands(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(11.5, 5.6))
    drew = False
    # Two slots can share a threat class, and therefore a colour. The second
    # one gets a dashed line so the pair stays distinguishable.
    seen: dict[str, int] = {}
    for row in rows:
        seen[row["threat"]] = seen.get(row["threat"], 0) + 1
        style = "-" if seen[row["threat"]] == 1 else "--"
        band = row.get("band")
        if not band:
            continue
        windows = band.get("windows") or []
        if not windows:
            continue
        hours = [w["to_hours"] for w in windows]
        means = [w["w_abs_max_mean"] for w in windows]
        axes.plot(hours, means, marker="o", ms=4, ls=style,
                  color=CLASS_COLOURS.get(row["threat"], "#333333"),
                  label=f"{row['slot']} {CLASS_LABELS.get(row['threat'], row['threat'])}")
        drew = True
    if not drew:
        plt.close(figure)
        return
    control = None
    for row in rows:
        band = row.get("band")
        if band and band.get("reference_120s_x1"):
            control = band["reference_120s_x1"]
            break
    if control:
        axes.axhline(control["abs_max"], color="#888888", ls="--", lw=1.2,
                     label=f"{CONTROL_LABEL}: |w| max {control['abs_max']:.3f}")
    axes.set_xlabel("forecast hour")
    axes.set_ylabel("mean of per-step |w| max over each half hour (m/s)")
    axes.set_title(
        "The |w| band on four independent runs at dt 20 s.\n"
        "This statistic is a maximum over EVERY cell on the globe, so it "
        "reports the shared 75 km background, not the refined core.")
    axes.grid(color="#EEEEEE", lw=0.6)
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    plt.close(figure)


def memory_row(rows, extra, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = []
    for row in rows:
        run = row.get("run")
        facts = row.get("facts")
        if not run or not facts or run.get("vram_peak_mib") is None:
            continue
        points.append((f"{row['slot']} {facts['n_cells']:,}",
                       facts["predicted_peak_mib"], float(run["vram_peak_mib"])))
    for label, predicted, measured in extra:
        points.append((label, predicted, measured))
    if not points:
        return
    figure, axes = plt.subplots(figsize=(11.0, 5.4))
    x = np.arange(len(points))
    predicted = [p[1] for p in points]
    measured = [p[2] for p in points]
    axes.bar(x - 0.19, predicted, 0.38, label="row prediction", color="#9AA0A6")
    axes.bar(x + 0.19, measured, 0.38, label="measured peak", color="#009E73")
    for i, (p, m) in enumerate(zip(predicted, measured)):
        axes.annotate(f"{(m - p) / p * 100:+.2f}%", (i, max(p, m)),
                      ha="center", va="bottom", fontsize=9)
    axes.axhline(32607.0, color="#CC0000", ls="--", lw=1.2,
                 label="RTX 5090, 32,607 MiB")
    axes.set_xticks(x, [p[0] for p in points], fontsize=9)
    axes.set_ylabel("device peak (MiB)")
    # The predictions plotted here are the affine row that was of record when
    # these runs were admitted; it was retired for shape on 2026-08-27.  Its
    # coefficients are deliberately NOT restated in the title: they are
    # computable from the retired arm in hexcore.device_admission, and a
    # hand-typed copy in a chart label is how a retired number outlives the
    # constant it came from.
    axes.set_title("The RETIRED affine device-memory row against every point "
                   "this lane measured\n(a line in cell count, retired "
                   "2026-08-27 for shape; coefficients from the retired arm "
                   "in hexcore.device_admission)")
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    plt.close(figure)


def coarse_land(path: Path | None):
    if path is None or not Path(path).exists():
        return None
    return tuple(np.load(path)[k] for k in ("lat", "lon", "land"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = Path(__file__).resolve().parents[1] / "evidence" / "four-swaths-20260827"
    parser.add_argument("--evidence", type=Path, default=here)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--land-mask", type=Path, default=None)
    arguments = parser.parse_args(argv)

    evidence = arguments.evidence
    out = arguments.out or (evidence / "figures")
    out.mkdir(parents=True, exist_ok=True)
    rows = _slots(evidence)
    land = coarse_land(arguments.land_mask or (evidence / "plan" / "LAND-MASK.npz"))

    extra = []
    ledger = evidence / "ledger-365.json"
    if ledger.exists():
        document = _load(ledger)
        extra.append((document["label"], document["predicted_mib"],
                      document["measured_peak_mib"]))

    placement_map(rows, land, out / "placement-map.png")
    sizing_chart(rows, out / "sizing-predicted-vs-delivered.png")
    resolution_gain(rows, out / "resolution-gain.png")
    w_bands(rows, out / "w-bands.png")
    memory_row(rows, extra, out / "memory-row.png")
    written = sorted(p.name for p in out.glob("*.png"))
    print(json.dumps({"out": str(out), "figures": written}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
