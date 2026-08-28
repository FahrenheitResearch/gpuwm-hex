#!/usr/bin/env python3
"""The cost of a swath as a global mesh against the same swath as a LAM.

ANALYSIS CHARTS ONLY.  Nothing here draws a weather field: the render law
sends those through rw_mpas_convert + rw_wrfbatch.  What these figures carry
is cells, memory, wall time and where a difference between two model runs
lives -- none of which is a weather field, and all of which matplotlib is the
right tool for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#101418"
GLOBAL_C = "#B4472E"
LAM_C = "#2E6F9E"
GRID_C = "#D8DDE2"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_C)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.8)
    ax.set_axisbelow(True)


def figure_cost(out: Path, m: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    _style(ax)
    cells = [m["global_cells"], m["lam_cells"]]
    bars = ax.bar(["global mesh\n(what ran)", "limited area\n(the same swath)"],
                  cells, color=[GLOBAL_C, LAM_C], width=0.55)
    ax.set_ylabel("cells the card holds")
    ax.set_title("One swath, two domains", color=INK, fontsize=12, loc="left")
    for bar, value in zip(bars, cells):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:,}",
                ha="center", va="bottom", fontsize=10, color=INK)
    ax.text(0.5, 0.55,
            f"{m['global_cells'] / m['lam_cells']:.1f}x fewer",
            transform=ax.transAxes, ha="center", fontsize=13, color=LAM_C)
    ax.set_ylim(0, max(cells) * 1.18)

    ax = axes[1]
    _style(ax)
    mem = [m["global_predicted_mib"], m["lam_predicted_mib"]]
    bars = ax.bar(["global mesh", "limited area"], mem,
                  color=[GLOBAL_C, LAM_C], width=0.55)
    ax.set_ylabel("predicted device footprint (MiB)")
    ax.set_title("What the shipped admission row asks the card for",
                 color=INK, fontsize=12, loc="left")
    for bar, value in zip(bars, mem):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:,.0f}",
                ha="center", va="bottom", fontsize=10, color=INK)
    styles = ["-", "--", ":"]
    for (name, free), style in zip(m["cards"].items(), styles):
        ax.axhline(free, color=INK, linewidth=1.1, linestyle=style, alpha=0.75)
        ax.text(-0.55, free + 250, f"{name}: {free:,.0f} MiB free",
                fontsize=8.5, color=INK, va="bottom")
    ax.set_ylim(0, max(mem + list(m["cards"].values())) * 1.12)
    ax.set_xlim(-0.6, 1.6)

    fig.suptitle(
        "The same storm, refined the same way: the background is what cost the money",
        fontsize=13, color=INK, x=0.02, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def figure_rings(out: Path, rings: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("white")
    _style(ax)
    labels = ["interior\n(free)"] + [f"ring {i}" for i in range(1, len(rings))]
    colors = [LAM_C] + ["#8FB8D4"] * (len(rings) - 1)
    bars = ax.bar(labels, rings, color=colors, width=0.6)
    for bar, value in zip(bars, rings):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02, f"{value:,}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("cells")
    ax.set_yscale("log")
    total = sum(rings)
    ax.set_title(
        f"What the cull kept: {total:,} cells, {rings[0]:,} of them free interior "
        f"({100 * rings[0] / total:.1f}%)",
        color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def figure_wall(out: Path, m: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    _style(ax)
    lam = np.asarray(m["lam_step_wall"], float)
    glo = np.asarray(m["global_step_wall"], float)
    ax.plot(np.arange(1, lam.size + 1), lam, color=LAM_C, linewidth=0.8,
            label=f"limited area, {m['lam_cells']:,} cells")
    ax.plot(np.arange(1, glo.size + 1), glo, color=GLOBAL_C, linewidth=0.8,
            label=f"global, {m['global_cells']:,} cells")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Wall per step, same dycore, same timestep",
                 color=INK, fontsize=12, loc="left")

    ax = axes[1]
    _style(ax)
    clean = float(m["lam_step_wall_uninstrumented_median"])
    med = [float(np.median(glo)), float(np.median(lam)), clean]
    labels = [
        "global",
        "limited area" + chr(10) + "(instrumented)",
        "limited area" + chr(10) + "(as it runs)",
    ]
    bars = ax.bar(labels, med, color=[GLOBAL_C, "#8FB8D4", LAM_C], width=0.6)
    for bar, value in zip(bars, med):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                f"{value:.3f} s", ha="center", va="bottom", fontsize=10,
                color=INK)
    ax.set_ylabel("median seconds per step")
    ax.set_title(
        f"{med[0] / clean:.2f}x faster on {m['global_cells'] / m['lam_cells']:.1f}x "
        f"fewer cells",
        color=INK, fontsize=12, loc="left")
    ax.set_ylim(0, max(med) * 1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def figure_concurrency(out: Path, conc: dict) -> None:
    """N swaths on one card: what each costs and what the card delivers."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    counts = sorted(int(k) for k in conc)

    ax = axes[0]
    _style(ax)
    totals = [conc[str(n)]["card_total_mib"] for n in counts]
    bars = ax.bar([str(n) for n in counts], totals, color=LAM_C, width=0.55)
    for bar, n in zip(bars, counts):
        value = conc[str(n)]["card_total_mib"]
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                f"{value:,}" + chr(10)
                + f"({conc[str(n)]['per_process_peak_mib']:,} each)",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlabel("concurrent swaths on one card")
    ax.set_ylabel("card memory (MiB)")
    ax.set_ylim(0, max(totals) * 1.28)
    ax.set_title("Each swath pays in full; nothing is shared",
                 color=INK, fontsize=12, loc="left")

    ax = axes[1]
    _style(ax)
    rate = [conc[str(n)]["aggregate_steps_per_second"] for n in counts]
    bars = ax.bar([str(n) for n in counts], rate, color=GLOBAL_C, width=0.55)
    for bar, value in zip(bars, rate):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                f"{value:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xlabel("concurrent swaths on one card")
    ax.set_ylabel("aggregate steps per second")
    ax.set_ylim(0, max(rate) * 1.25)
    ax.set_title("...and the card delivers no more work for it",
                 color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def figure_agreement(out: Path, agreement: dict) -> None:
    """How far the limited-area interior drifts from the uncut parent."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    hours = np.arange(len(agreement["frames"])) * 0.5

    ax = axes[0]
    _style(ax)
    rms = [f["fields"]["theta"]["interior_rms"] for f in agreement["frames"]]
    ax.plot(hours, rms, color=LAM_C, linewidth=2, marker="o", markersize=3.5)
    ax.set_xlabel("forecast hour")
    ax.set_ylabel("interior RMS difference (K)")
    ax.set_title(
        "Potential temperature, limited area against the uncut parent",
        color=INK, fontsize=12, loc="left")
    ax.text(0.05, max(rms) * 0.88,
            "the two runs start bit-identical" + chr(10)
            + "on every cell they share",
            fontsize=9, color=INK)

    ax = axes[1]
    _style(ax)
    last = agreement["frames"][-1]["fields"]["theta"]["by_ring"]
    rings = sorted(last, key=int)
    values = [last[r]["rms"] for r in rings]
    colors = [LAM_C] + ["#8FB8D4"] * (len(rings) - 1)
    ax.bar(["interior"] + [f"ring {r}" for r in rings[1:]], values,
           color=colors, width=0.6)
    ax.set_ylabel("RMS difference at +6 h (K)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Flat from the interior out: the boundary is not leaking in",
                 color=INK, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    m = json.loads(arguments.measurements.read_text(encoding="utf-8"))
    arguments.out.mkdir(parents=True, exist_ok=True)
    figure_cost(arguments.out / "01-cells-and-memory.png", m)
    figure_rings(arguments.out / "02-what-the-cull-kept.png", m["ring_cell_counts"])
    if m.get("lam_step_wall") and m.get("global_step_wall"):
        figure_wall(arguments.out / "03-wall-per-step.png", m)
    if m.get("concurrency"):
        figure_concurrency(arguments.out / "04-swaths-on-one-card.png",
                           m["concurrency"])
    agreement = arguments.measurements.parent / "interior-agreement.json"
    if agreement.is_file():
        figure_agreement(
            arguments.out / "05-does-it-hold-the-same-weather.png",
            json.loads(agreement.read_text(encoding="utf-8")),
        )
    print(f"wrote figures to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
