#!/usr/bin/env python
"""Evidence charts for the generated-mesh dual-edge defect.

These are ANALYSIS charts over mesh geometry and a kernel-launch trace -- not
weather fields -- so matplotlib is the right tool here; product plots stay on
the Rust renderers.

Three figures, each answering one question:

  1. collapsed dual edges  -- how far the failing mesh's worst edges fall
     below every mesh that integrates, against the admitted floor
  2. runaway growth        -- where the amplitude climbs inside the first
     outer step, as a dimensionless growth factor so one axis carries
     quantities with different units honestly
  3. dislocations          -- the topological cause: cells that are not
     hexagons, per mesh, against the twelve pentagons a sphere requires
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

# Okabe-Ito, assigned by ENTITY and never cycled: each mesh keeps its hue in
# every figure, so a reader who learns the legend once keeps it.
COLOR = {
    "x1.40962": "#0072B2",
    "x4.163842": "#009E73",
    "v15.150.38857": "#D55E00",
    "u96.64002": "#CC79A7",
}
# Secondary encoding, so identity never rests on colour alone.
DASH = {
    "x1.40962": (0, (1, 0)),
    "x4.163842": (0, (6, 2)),
    "v15.150.38857": (0, (1, 0)),
    "u96.64002": (0, (2, 2)),
}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e2e2e2"
SURFACE = "#ffffff"
FLOOR = 0.02


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_color(MUTED)


def _titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.955, title, ha="left", va="top", fontsize=15,
             color=INK, fontweight="600")
    fig.text(0.055, 0.905, subtitle, ha="left", va="top", fontsize=10,
             color=MUTED)


def read_mesh(path: Path) -> dict[str, np.ndarray]:
    with Dataset(str(path)) as data:
        dv = np.asarray(data.variables["dvEdge"][:], dtype=np.float64)
        dc = np.asarray(data.variables["dcEdge"][:], dtype=np.float64)
        degree = np.asarray(data.variables["nEdgesOnCell"][:], dtype=np.int64)
    return {"ratio": np.sort(dv / dc), "degree": degree}


def figure_tail(meshes: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.10, right=0.96)
    worst: dict[str, float] = {}
    for name, mesh in meshes.items():
        ratio = mesh["ratio"]
        worst[name] = float(ratio[0])
        rank = np.arange(1, ratio.size + 1)
        ax.plot(rank, ratio, color=COLOR[name], linewidth=2.0,
                linestyle=DASH[name], zorder=3,
                label=f"{name}   worst {ratio[0]:.4g}")
    ax.axhline(FLOOR, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(f"admitted floor  dvEdge/dcEdge = {FLOOR}",
                xy=(1.2, FLOOR), xytext=(0, 7), textcoords="offset points",
                fontsize=9, color=MUTED)
    defect = "v15.150.38857"
    if defect in worst:
        ax.annotate("edge 19786:  dvEdge 6.5 m against dcEdge 38,657 m",
                    xy=(1, worst[defect]), xytext=(14, 4),
                    textcoords="offset points", fontsize=9, color=COLOR[defect])
    if "x1.40962" in worst and "u96.64002" in worst:
        ax.annotate("the published x1.40962 and the generated u96.64002\n"
                    "lie on top of each other, worst edge to worst edge",
                    xy=(30, worst["x1.40962"]), xytext=(0, -34),
                    textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("edge rank, shortest dual edge first", color=MUTED, fontsize=10)
    ax.set_ylabel("dvEdge / dcEdge", color=MUTED, fontsize=10)
    _style(ax)
    legend = ax.legend(frameon=False, fontsize=9.5, loc="lower right",
                       bbox_to_anchor=(1.0, 0.02))
    for text in legend.get_texts():
        text.set_color(INK)
    _titles(
        fig,
        "One mesh's dual edges collapse; the others' do not",
        "Every edge of four meshes, sorted. Below the floor a Voronoi edge is so "
        "short that the TRiSK\ntangential terms, which divide by it, amplify a "
        "gradient beyond what any timestep can hold.",
    )
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figure_growth(trace: Path, out: Path, n_edges: int, worst_edge: int) -> None:
    rows = [json.loads(line) for line in trace.open(encoding="utf-8")]
    wanted = {
        "pv_apvm_v841_f32": (15, "grad_pv_tangential  (divides by dvEdge)"),
        "vector_momentum_v841_f32": (18, "momentum tendency at the edge"),
        "acoustic_ru_v841": (18, "acoustic normal mass flux"),
    }
    order = list(wanted)
    hues = ["#D55E00", "#0072B2", "#009E73"]
    dashes = [(0, (1, 0)), (0, (5, 2)), (0, (2, 2))]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.10, right=0.96)
    last = rows[-1]["n"]
    for index, kernel in enumerate(order):
        slot, label = wanted[kernel]
        points = [(r["n"], r["max_abs"], r["at"]) for r in rows
                  if r["k"] == kernel and r["a"] == slot]
        if not points:
            continue
        base = next((v for _, v, _ in points if v > 0.0), 1.0)
        x = [p[0] for p in points]
        y = [p[1] / base for p in points]
        on_worst = sum(1 for p in points if p[2] % n_edges == worst_edge)
        ax.plot(x, y, color=hues[index], linewidth=2.0, linestyle=dashes[index],
                zorder=3,
                label=f"{label}   peak on edge {worst_edge} at "
                      f"{on_worst} of {len(points)} launches")
    ax.axvline(last, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("first non-finite value:\nexner at cell 6461,\nlevels 33-37",
                xy=(last, 1.0), xytext=(-108, 2), textcoords="offset points",
                fontsize=9, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("kernel launch, within composite step 0", color=MUTED, fontsize=10)
    ax.set_ylabel("peak magnitude / its own first value", color=MUTED, fontsize=10)
    _style(ax)
    legend = ax.legend(frameon=False, fontsize=9.0, loc="upper left",
                       bbox_to_anchor=(0.02, 0.99))
    for text in legend.get_texts():
        text.set_color(INK)
    _titles(
        fig,
        "The runaway is one edge, from the first operator that divides by dvEdge",
        "Growth relative to each array's own opening magnitude, so one axis carries "
        "three different units\nhonestly. The peak sits on the mesh's worst dual "
        "edge at every launch shown.",
    )
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def figure_dislocations(meshes: dict[str, dict], out: Path) -> None:
    names = list(meshes)
    pent = [int(np.count_nonzero(meshes[n]["degree"] == 5)) for n in names]
    hept = [int(np.count_nonzero(meshes[n]["degree"] >= 7)) for n in names]
    x = np.arange(len(names), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.10, right=0.95)
    # A zero is drawn as no bar and labelled at the baseline: a visible bar for
    # a count of zero is a lie about the geometry.
    drawn_p = np.where(np.asarray(pent) > 0, np.asarray(pent, dtype=float), np.nan)
    drawn_h = np.where(np.asarray(hept) > 0, np.asarray(hept, dtype=float), np.nan)
    bars_p = ax.bar(x - width / 2 - 0.01, drawn_p, width,
                    color="#56B4E9", edgecolor=SURFACE, linewidth=2.0,
                    zorder=3, label="pentagons (5 neighbours)")
    bars_h = ax.bar(x + width / 2 + 0.01, drawn_h, width,
                    color="#D55E00", edgecolor=SURFACE, linewidth=2.0,
                    zorder=3, label="heptagons and above (dislocations)")
    bottom = 0.55
    ax.set_ylim(bottom, max(max(pent), max(hept)) * 6.0)
    for bars, values in ((bars_p, pent), (bars_h, hept)):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            top = bottom if not np.isfinite(height) else height
            ax.annotate(f"{value:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=9, color=INK)
    ax.axhline(12, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("12 pentagons: what a closed sphere requires",
                xy=(-0.45, 12), xytext=(0, -14),
                textcoords="offset points", ha="left", va="top",
                fontsize=9, color=MUTED)
    ax.set_xlim(-0.55, len(names) - 0.30)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("cells", color=MUTED, fontsize=10)
    ax.grid(axis="x", visible=False)
    _style(ax)
    legend = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                       bbox_to_anchor=(0.0, 1.02))
    for text in legend.get_texts():
        text.set_color(INK)
    _titles(
        fig,
        "Dislocations are not the defect; near-cocircular ones are",
        "A pentagon-heptagon pair is a dislocation. A graded mesh cannot avoid them "
        "-- the published\nvariable-resolution x4.163842 carries 32 and integrates. "
        "The Fibonacci seed carries 3,447, and\nrelaxation leaves their quads close "
        "enough to cocircular that the dual edge between them collapses.",
    )
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="append", required=True,
                        metavar="NAME=PATH")
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--trace-n-edges", type=int, default=116_565)
    parser.add_argument("--trace-worst-edge", type=int, default=19_786)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, dict] = {}
    for entry in args.static:
        name, path = entry.split("=", 1)
        meshes[name] = read_mesh(Path(path))
    figure_tail(meshes, args.out / "01-dual-edge-tail.png")
    figure_dislocations(meshes, args.out / "03-dislocations.png")
    if args.trace is not None:
        figure_growth(args.trace, args.out / "02-runaway-growth.png",
                      args.trace_n_edges, args.trace_worst_edge)
    for path in sorted(args.out.glob("*.png")):
        print(f"wrote {path} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
