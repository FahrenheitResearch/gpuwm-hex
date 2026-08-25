#!/usr/bin/env python3
"""Evidence charts for the forecast door's device-memory admission gate.

Analysis charts, not weather fields, so matplotlib is the right tool here:
the render law reserves the Rust renderer for weather-field products and
allows matplotlib for analysis that is not one.

Every number plotted is either the measured footprint row the door ships
(``mpas_port.forecast_door.FOOTPRINT_MODEL``, read from there rather than
restated) or a value measured on the certifying card and recorded below.
Nothing here is illustrative.

    python tools/plot_forecast_admission_evidence.py <output-directory>
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpas_port.forecast_door import (  # noqa: E402
    DEFAULT_HEADROOM_BYTES,
    FOOTPRINT_MODEL,
    MIB,
)

HEADROOM_MIB = DEFAULT_HEADROOM_BYTES / MIB

# Registered rows, from tools/mpas_mesh_binding.py.  Named here rather than
# imported because importing the registry pulls netCDF4 and numpy for a
# chart that needs two integers per row.
MESHES = [
    ("v15.150.38857", 38_857),
    ("x1.40962", 40_962),
    ("x4.163842", 163_842),
]

# Measured on the certifying RTX 3080, 2026-08-24, by the two instruments
# that disagree about it.  Both are recorded because the disagreement is a
# finding (WDDM reports what the driver believes it could obtain), not a
# rounding difference.  The door decides on the CUDA driver's number.
CARD = "RTX 3080"
CARD_TOTAL_MIB = 10_239.5
CUDA_FREE_MIB = 9_097.0
SMI_FREE_MIB = 6_778.0

BUDGETS = (10, 12, 16, 24, 32)
# The decision chart draws only the budgets that frame these three rows;
# 32 GiB sits off the top of it and would be a floating label.
DECISION_BUDGETS = (10, 12, 16, 24)

INK = "#2563EB"
STATUS = "#B91C1C"
PRIMARY = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
SURFACE = "#FFFFFF"


def predicted_mib(cells: int) -> float:
    return FOOTPRINT_MODEL.predict_bytes(cells) / MIB


def fitted_cells(budget_gib: float) -> int:
    return FOOTPRINT_MODEL.max_cells(
        int(budget_gib * 1024 * MIB), DEFAULT_HEADROOM_BYTES
    )


def _style(axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=MUTED, labelsize=9)


def decision_chart(path: Path) -> None:
    """The decision itself: three registered rows against one real card."""

    figure, axes = plt.subplots(figsize=(11.0, 6.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    _style(axes)

    names = [f"{name}\n{cells:,} cells" for name, cells in MESHES]
    peaks = [predicted_mib(cells) for _name, cells in MESHES]
    needs = [peak + HEADROOM_MIB for peak in peaks]

    axes.bar(
        names, peaks, width=0.46, color=INK, zorder=4,
        edgecolor=SURFACE, linewidth=2.0,
    )
    for index, (peak, need) in enumerate(zip(peaks, needs)):
        # The headroom rides on top of the peak as its own lighter segment,
        # with a surface gap between them, so the bar shows both terms of
        # the decision rather than one summed number.
        axes.bar(
            index, HEADROOM_MIB, width=0.46, bottom=peak + 60,
            color=INK, alpha=0.28, zorder=4,
            edgecolor=SURFACE, linewidth=2.0,
        )
        axes.text(
            index, need + 420,
            f"{peak:,.0f} MiB peak\n+ {HEADROOM_MIB:,.0f} headroom",
            color=PRIMARY, fontsize=9.5, ha="center", va="bottom",
        )

    # Reference lines stop short of the right edge so their labels sit in a
    # clear gutter rather than on top of the line they name.
    for budget in DECISION_BUDGETS:
        y = budget * 1024.0
        axes.plot(
            [-0.6, 2.52], [y, y], color=MUTED, linewidth=1.0,
            linestyle=(0, (4, 4)), zorder=2,
        )
        axes.text(
            2.62, y, f"{budget} GiB card", color=MUTED,
            fontsize=9.5, ha="left", va="center",
        )

    axes.plot(
        [-0.6, 2.52], [CUDA_FREE_MIB, CUDA_FREE_MIB],
        color=STATUS, linewidth=2.0, zorder=6,
    )
    axes.text(
        2.62, CUDA_FREE_MIB - 330,
        f"{CARD}\n{CUDA_FREE_MIB:,.0f} MiB free",
        color=STATUS, fontsize=9.5, ha="left", va="top",
    )

    # The shortfall the door actually refused on, on the row it refused.
    axes.annotate(
        "", xy=(1.30, needs[1]), xytext=(1.30, CUDA_FREE_MIB),
        arrowprops={"arrowstyle": "<->", "color": STATUS, "linewidth": 1.6},
        zorder=7,
    )
    axes.text(
        1.36, CUDA_FREE_MIB - 700,
        f"short by {needs[1] - CUDA_FREE_MIB:,.0f} MiB",
        color=STATUS, fontsize=9.5, ha="left", va="top", zorder=9,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
    )

    axes.set_xlim(-0.6, 3.75)
    axes.set_ylim(0, 25_600)
    axes.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes.set_ylabel("device memory the run needs (MiB)", color=MUTED, fontsize=10)
    axes.tick_params(axis="x", labelsize=10, colors=PRIMARY)
    axes.set_title(
        "Every registered mesh, against the card the door was proved on",
        color=PRIMARY, fontsize=14, pad=44, loc="left",
    )
    axes.text(
        0.0, 1.045,
        f"predicted from the measured row {FOOTPRINT_MODEL.fixed_bytes / MIB:,.1f} "
        f"MiB + {FOOTPRINT_MODEL.bytes_per_cell:,.0f} B per cell; all three\n"
        f"sat above what this 10 GiB card had free, so the door refused all three",
        transform=axes.transAxes, color=MUTED, fontsize=9.5, va="bottom",
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def capacity_chart(path: Path) -> None:
    """The same row read the other way: what each budget buys."""

    figure, axes = plt.subplots(figsize=(11.0, 5.4), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    _style(axes)

    labels = [f"{budget} GiB" for budget in BUDGETS]
    values = [fitted_cells(budget) for budget in BUDGETS]
    axes.barh(
        labels, values, height=0.56, color=INK, zorder=4,
        edgecolor=SURFACE, linewidth=2.0,
    )
    for label, value in zip(labels, values):
        axes.text(
            value + 4_000, label, f"{value:,.0f} cells",
            color=PRIMARY, fontsize=10, va="center",
        )

    # Only the two separable rows are marked: v15.150.38857 sits within 5 %
    # of x1.40962 and a second line there would be a label collision, not
    # information.  The caption names it instead.
    for name, count in (("x1.40962", 40_962), ("x4.163842", 163_842)):
        axes.axvline(
            count, color=MUTED, linewidth=1.2, linestyle=(0, (4, 4)), zorder=5
        )
        axes.text(
            count + 3_000, -0.62, name, color=MUTED, fontsize=9,
            ha="left", va="center",
        )

    axes.set_xlim(0, 330_000)
    axes.set_ylim(4.6, -0.95)
    axes.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes.set_xlabel(
        f"cells the row admits, after {HEADROOM_MIB:,.0f} MiB headroom",
        color=MUTED, fontsize=10,
    )
    axes.tick_params(axis="y", labelsize=10, colors=PRIMARY)
    axes.set_title(
        "How much mesh each card budget holds",
        color=PRIMARY, fontsize=14, pad=52, loc="left",
    )
    axes.text(
        0.0, 1.05,
        "12 GiB is the first budget that holds the published 40,962-cell global mesh;\n"
        "10 GiB holds none of the three registered rows -- v15.150.38857 is 38,857\n"
        "cells, within 5 % of x1.40962, so only the two separable rows are marked",
        transform=axes.transAxes, color=MUTED, fontsize=9.5, va="bottom",
    )
    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE)
    plt.close(figure)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(f"usage: {argv[0]} <output-directory>", file=sys.stderr)
        return 2
    out = Path(argv[1])
    out.mkdir(parents=True, exist_ok=True)
    decision_chart(out / "01-every-registered-mesh-against-the-rtx3080.png")
    capacity_chart(out / "02-cells-each-card-budget-holds.png")
    print(f"model: {FOOTPRINT_MODEL.provenance}")
    for name, cells in MESHES:
        print(f"{name:>16}  {cells:>8,} cells  ->  {predicted_mib(cells):>9,.1f} MiB")
    for budget in BUDGETS:
        print(f"{budget:>2} GiB budget -> {fitted_cells(budget):>9,} cells")
    print(f"{CARD} free at the decision: {CUDA_FREE_MIB:,.1f} MiB")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
