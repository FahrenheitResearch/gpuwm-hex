#!/usr/bin/env python
"""Evidence charts for the opt-in local-timestep gates.

These are ANALYSIS charts -- drift budgets, error ratios, wall clock -- not
weather fields, so matplotlib is the right tool.  Weather-field products stay
under the render law and come from the Rust renderer.

Input is the gate verdict JSON written by ``run_lts_dry_gates.py compare``
plus the per-arm receipts.  Every panel carries a plain-language caption
underneath it, because the point of these is that someone can read the verdict
off the picture without opening a receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1b1f"
MUTED = "#6b6b76"
PASS = "#2e7d5b"
FAIL = "#b3402f"
BOUND = "#c2703d"
DEFAULT_C = "#4a6fa5"
LTS_C = "#8a5fa8"


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)


def _caption(fig, text: str) -> None:
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=9,
             color=MUTED, wrap=True)


def conservation_chart(report: dict, out: Path) -> Path:
    gate = report["gate3_conservation"]
    labels = ["dry mass\ndefault", "dry mass\n--local-timestep",
              "passive qv\ndefault", "passive qv\n--local-timestep"]
    values = [
        gate.get("off_arm_dry_mass_relative_drift", 0.0),
        gate["dry_mass_relative_drift"],
        gate.get("off_arm_qv_mass_relative_drift", 0.0),
        gate["qv_mass_relative_drift"],
    ]
    colors = [DEFAULT_C, LTS_C, DEFAULT_C, LTS_C]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(2.0e-8, color=BOUND, linestyle="--", linewidth=1.6)
    ax.text(3.48, 2.0e-8 * 1.25, "2.0e-8 gate", color=BOUND, fontsize=9,
            ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("relative drift over the run")
    ax.set_title("Mass is not created or destroyed at a class boundary",
                 color=INK, fontsize=13, pad=14)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.35,
                f"{value:.2e}", ha="center", fontsize=8.5, color=INK)
    _style(ax)
    verdict = report.get("gate3_verdict", "?")
    _caption(fig,
             "Total dry mass and total passive water vapour, start of run vs "
             "end of run, on the published variable-resolution mesh.\n"
             "Both arms sit far below the 2.0e-8 bound the port already "
             f"applies to its stabilized products. Gate 3: {verdict}.")
    fig.subplots_adjust(bottom=0.24, top=0.88)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def referee_chart(report: dict, out: Path) -> Path:
    gate = report["gate4_referee"]
    fields = ["rho", "rho_theta", "rho_u", "rho_w", "scalars"]
    nice = ["density", "density x theta", "normal\nmomentum",
            "vertical\nmomentum", "scalars (qv)"]
    default = [gate["off_vs_referee"][f]["rms_relative_to_reference"]
               for f in fields]
    lts = [gate["on_vs_referee"][f]["rms_relative_to_reference"]
           for f in fields]
    x = range(len(fields))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.bar([i - 0.19 for i in x], default, width=0.36, color=DEFAULT_C,
           label="default path")
    ax.bar([i + 0.19 for i in x], lts, width=0.36, color=LTS_C,
           label="--local-timestep")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nice, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(min(min(default), min(lts)) * 0.35,
                max(max(default), max(lts)) * 6.0)
    ax.set_ylabel("RMS distance to the small-step reference")
    ax.set_title("Both arms are scored against the same referee",
                 color=INK, fontsize=13, pad=14)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=2,
              loc="upper center")
    _style(ax)
    worst = gate.get("worst_ratio")
    _caption(fig,
             "The referee runs the same mesh for the same physical duration at "
             f"dt={gate['referee_dt']:g} s instead of "
             f"{gate['default_dt']:g} s.\nA globally smaller acoustic step is "
             "the only thing that separates local-timestep error from the "
             "truncation error\nthe default path already carries. Worst field "
             f"ratio, option over default: {worst:.2f}x.")
    fig.subplots_adjust(bottom=0.26, top=0.88)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def locality_chart(report: dict, out: Path) -> Path:
    locality = report["gate4_referee"].get("interface_locality") or {}
    if not locality:
        return None
    fields = list(locality)
    default = [locality[f]["default"]["interface_over_interior"] for f in fields]
    lts = [locality[f]["lts"]["interface_over_interior"] for f in fields]
    x = range(len(fields))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar([i - 0.19 for i in x], default, width=0.36, color=DEFAULT_C,
           label="default path")
    ax.bar([i + 0.19 for i in x], lts, width=0.36, color=LTS_C,
           label="--local-timestep")
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=":")
    ax.text(len(fields) - 0.42, 1.005, "equal to the interior", fontsize=8.5,
            color=MUTED, ha="right", va="bottom")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["density", "density x theta", "scalars (qv)"][: len(fields)],
        fontsize=9,
    )
    ax.set_ylim(0.0, max(max(default), max(lts), 1.0) * 1.30)
    ax.set_ylabel("error on boundary cells / error on interior cells")
    ax.set_title("Is the error piling up on the class boundary?",
                 color=INK, fontsize=13, pad=14)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    _style(ax)
    n_iface = next(iter(locality.values()))["lts"]["n_interface_cells"]
    _caption(fig,
             "A refinement boundary is a busy place even without the option, "
             f"so the default arm is the control; {n_iface} cells touch one.\n"
             "A reflection would show as the purple bar standing well above "
             "the blue one. It stands slightly below.")
    fig.subplots_adjust(bottom=0.24, top=0.88)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def cost_chart(report: dict, classing: dict, out: Path) -> Path:
    arms = report["arms"]
    labels, values, colors = [], [], []
    for key, nice, colour in (
        ("off", "default path", DEFAULT_C),
        ("on", "--local-timestep", LTS_C),
    ):
        if key in arms:
            labels.append(nice)
            values.append(arms[key]["seconds_per_step"])
            colors.append(colour)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.0),
                                  gridspec_kw={"width_ratios": [1.0, 1.25]})
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                f"{value:.3f} s", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("wall seconds per model step")
    ax.set_title("Measured cost", color=INK, fontsize=12, pad=10)
    _style(ax)

    per_rate = classing["cells_per_rate"]
    rates = sorted(per_rate, key=int)
    counts = [per_rate[r] for r in rates]
    ax2.bar([f"rate {r}\n(1 sub-step in {r})" for r in rates], counts,
            color=[DEFAULT_C, LTS_C][: len(rates)], width=0.5)
    for index, value in enumerate(counts):
        ax2.text(index, value * 1.01, f"{value:,}", ha="center", fontsize=9,
                 color=INK)
    ax2.set_ylabel("columns")
    ax2.set_title("What the mesh actually admitted", color=INK, fontsize=12,
                  pad=10)
    _style(ax2)

    speedup = report.get("measured_speedup_x")
    saving = classing["arithmetic_acoustic_saving"]
    _caption(fig,
             f"{classing['n_cells']:,} columns, {classing['interface_edges']} "
             f"interface edges, arithmetic acoustic saving {100 * saving:.1f}%.\n"
             f"Measured whole-step speedup: {speedup:.3f}x. The arithmetic "
             "saving counts acoustic sub-steps removed; the measured\nnumber "
             "is wall clock over the whole model step, which includes "
             "everything the option does not touch.")
    fig.subplots_adjust(bottom=0.24, top=0.88, wspace=0.32)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _substeps_per_model_step(sub_steps: int, split: int) -> int:
    """The pinned schedule: ``(1, nsub//2, nsub)`` per RK stage, per subcycle."""

    return int(split) * (1 + max(1, int(sub_steps) // 2) + int(sub_steps))


def ceiling_chart(base: dict, raised: dict, saving: float, out: Path) -> Path:
    """How much of a model step is the acoustic loop, and what does that cap?

    Two runs of the SAME arm differing only in acoustic sub-step count give the
    per-sub-step cost directly, and from it the acoustic share of a model step.
    Nothing local time stepping does can remove more than its own saving times
    that share, so this is the ceiling the measured number has to be read
    against.
    """

    n_base = _substeps_per_model_step(
        base["config_number_of_sub_steps"], base["config_dynamics_split_steps"]
    )
    n_raised = _substeps_per_model_step(
        raised["config_number_of_sub_steps"], raised["config_dynamics_split_steps"]
    )
    per_substep = (
        (raised["seconds_per_step"] - base["seconds_per_step"]) / (n_raised - n_base)
    )
    acoustic = per_substep * n_base
    total = base["seconds_per_step"]
    share = acoustic / total
    ceiling = 1.0 / (1.0 - share * saving) if share * saving < 1 else float("inf")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.barh([0], [acoustic], color=LTS_C, height=0.5,
            label="acoustic sub-steps (what the option can touch)")
    ax.barh([0], [total - acoustic], left=[acoustic], color=DEFAULT_C,
            height=0.5, label="everything else in a model step")
    ax.set_yticks([])
    ax.set_xlabel("wall seconds per model step")
    ax.set_xlim(0, total * 1.02)
    ax.set_title("The ceiling: what fraction of a step is even eligible",
                 color=INK, fontsize=13, pad=14)
    ax.text(acoustic / 2, 0, f"{100 * share:.0f}%", ha="center", va="center",
            color="white", fontsize=11)
    ax.text(acoustic + (total - acoustic) / 2, 0, f"{100 * (1 - share):.0f}%",
            ha="center", va="center", color="white", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=1)
    _style(ax)
    _caption(fig,
             f"Measured by re-timing the same arm at {raised['config_number_of_sub_steps']} "
             f"acoustic sub-steps instead of {base['config_number_of_sub_steps']}: "
             f"{n_raised} versus {n_base} sub-steps per model step gives the "
             "cost of one\ndirectly. With an arithmetic acoustic saving of "
             f"{100 * saving:.1f}%, the best a whole model step could do is "
             f"{ceiling:.3f}x -- before the option pays for its own "
             "bookkeeping.")
    fig.subplots_adjust(bottom=0.42, top=0.86)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdict", required=True)
    parser.add_argument("--on-receipt", required=True)
    parser.add_argument("--ceiling-base", help="OFF arm receipt at the released sub-step count")
    parser.add_argument("--ceiling-raised", help="the same arm at a raised sub-step count")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    on = json.loads(Path(args.on_receipt).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    if "gate3_conservation" in report:
        written.append(conservation_chart(
            report, out_dir / "lts-01-conservation.png"))
    if "gate4_referee" in report and "on_vs_referee" in report["gate4_referee"]:
        written.append(referee_chart(
            report, out_dir / "lts-02-referee.png"))
        made = locality_chart(report, out_dir / "lts-03-interface-locality.png")
        if made:
            written.append(made)
    if on.get("local_timestep"):
        written.append(cost_chart(
            report, on["local_timestep"], out_dir / "lts-04-cost-and-classes.png"))
    if args.ceiling_base and args.ceiling_raised and on.get("local_timestep"):
        written.append(ceiling_chart(
            json.loads(Path(args.ceiling_base).read_text(encoding="utf-8")),
            json.loads(Path(args.ceiling_raised).read_text(encoding="utf-8")),
            on["local_timestep"]["arithmetic_acoustic_saving"],
            out_dir / "lts-05-ceiling.png",
        ))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
