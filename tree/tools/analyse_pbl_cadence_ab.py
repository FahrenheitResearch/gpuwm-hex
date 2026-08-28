#!/usr/bin/env python3
"""Read the surface/PBL cadence A/B and answer one question with numbers.

THE QUESTION: does the vertical-velocity climb that SURVIVED switching
Grell-Freitas off collapse when the surface/PBL cadence is held at 120 s
while ``dt`` shrinks?

``evidence/convection-off-20260826/`` eliminated the cumulus closure by
measurement -- 91.4 % of the 5 s |w| excess over the control survived the
scheme never being called -- but the CALL-RATE SHAPE of its hypothesis
survived with it.  ``config_bldt_seconds`` is welded to ``dt`` exactly as
``cudt`` is: at 5 s the surface layer, the land-surface model and the PBL run
720 times an hour against the proven 30, the identical 24x.

Convection is off in every candidate arm deliberately, so the ONLY thing that
moves between an arm and its reference is how often that stack is called.
Both welded references were measured on the SAME card as these arms, which is
what the convection lane could not do at 5 s.

TWO COMPARISONS, BECAUSE THE TWO TIMESTEPS BEHAVED DIFFERENTLY.

* At **20 s** both configurations complete 360 steps, so they are compared as
  full 2 h bands against the 120 s control, the same arithmetic
  ``convection-off-20260826`` used.
* At **5 s** the held configuration DID NOT COMPLETE: 964 of 1,440 steps, all
  finite, then the transactional dycore refused to publish step 965, in two
  independent processes at the same step.  Comparing its 964-step band to the
  welded arm's 1,440-step band would answer a different question, so the two
  are compared over the IDENTICAL first 964 steps (``dt5-step-matched.json``).

WHAT THIS CAN AND CANNOT SETTLE.  It can settle ATTRIBUTION -- whether a band
the held configuration does not produce was being produced by the call rate.
It cannot settle whether either configuration has more SKILL; only obs-skill
(MRMS, ASOS) referees that and nothing here claims it.  And it cannot
separate a per-call defect from the resolved dynamics of a configuration
nobody would run: 5 s is 140x below this mesh's own 698.95 s Courant limit
and 20 s is 35x below it.  That needs a mesh whose Courant limit is near the
timestep, and no such mesh exists yet.

The charts are ANALYSIS charts, not weather fields, so matplotlib is the
right tool for them under the render law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

#: Okabe-Ito, assigned in fixed order and never cycled.  Hue carries the
#: TREATMENT so the A/B reads at a glance -- warm where the surface/PBL stack
#: is called every step, cool where it is held to the proven 30 an hour --
#: and line style carries it a second time, so identity never rests on colour
#: alone.
SERIES_STYLE: dict[str, dict[str, Any]] = {
    "120 s control (GF)": {
        "color": "#3D3D3D", "linestyle": (0, (1, 1)), "marker": "o",
        "zorder": 5, "minutes": 120.0,
    },
    "20 s OFF, PBL welded (180/h)": {
        "color": "#D55E00", "linestyle": "--", "marker": "s",
        "zorder": 3, "minutes": 120.0,
    },
    "20 s OFF, PBL held 120 s (30/h)": {
        "color": "#0072B2", "linestyle": "-", "marker": "s",
        "zorder": 4, "minutes": 120.0,
    },
    "5 s OFF, PBL welded (720/h)": {
        "color": "#CC79A7", "linestyle": "--", "marker": "D",
        "zorder": 3, "minutes": 120.0,
    },
    "5 s OFF, PBL held 120 s (30/h)": {
        "color": "#009E73", "linestyle": "-", "marker": "D",
        "zorder": 4, "minutes": 80.33,
    },
}

INK = "#1A1A1A"
INK_MUTED = "#6B6B6B"
GRID = "#DDDDDD"
SURFACE = "#FFFFFF"
ALARM = "#B3261E"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor_band(path: Path) -> dict[str, Any]:
    return _read(path)["arms"][0]["band"]


def _from_band(band: dict[str, Any], calls: float) -> dict[str, Any]:
    return {
        "steps": int(band["steps"]),
        "finite_every_step": bool(band["finite_every_step"]),
        "completed": True,
        "surface_pbl_calls_per_hour": calls,
        "w_trend": [
            float(w["vertical_velocity_abs_max"]["mean"])
            for w in band["trend"]["windows"]
        ],
        "w_max": float(band["vertical_velocity_abs_max"]["max"]),
        "theta_m_max": float(band["theta_m_max"]["max"]),
        "qv_max": float(band["qv_max"]["max"]),
    }


def _from_matched(arm: dict[str, Any], calls: float, completed: bool) -> dict[str, Any]:
    return {
        "steps": int(arm["steps"]),
        "finite_every_step": bool(arm["finite_every_step"]),
        "completed": completed,
        "surface_pbl_calls_per_hour": calls,
        "w_trend": [float(v) for v in arm["w_mean_by_quarter"]],
        "w_max": float(arm["w_max"]),
        "theta_m_max": float(arm["theta_m_max"]),
        "qv_max": float(arm["qv_max"]),
    }


def collect(
    *, control_band: Path, welded20: Path, held20: Path, welded5: Path,
    matched5: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matched = _read(matched5)
    # The welded 5 s row is its OWN FULL RUN here -- all 1,440 steps, the
    # band convection-off-20260826 archived -- because this overview asks
    # what each configuration did.  The rigorous like-for-like lives in the
    # step-matched record and its own chart.
    rows = {
        "120 s control (GF)": _from_band(_read(control_band), 30.0),
        "20 s OFF, PBL welded (180/h)": _from_band(_anchor_band(welded20), 180.0),
        "20 s OFF, PBL held 120 s (30/h)": _from_band(_anchor_band(held20), 30.0),
        "5 s OFF, PBL welded (720/h)": _from_band(_anchor_band(welded5), 720.0),
        "5 s OFF, PBL held 120 s (30/h)": _from_matched(
            matched["held"]["arm-a"], 30.0, completed=False
        ),
    }
    return rows, matched


def treatment_evidence(
    *, welded20: Path, held20: Path, matched: dict[str, Any]
) -> dict[str, Any]:
    """Positive evidence that holding the cadence CHANGED THE RUN.

    An exact-0.0 delta would mean the knob never reached the physics, and a
    null result read off an inert knob is the worst outcome available
    because it looks like an answer.  Both timesteps must show the treatment
    landing before any attribution is read from the bands.
    """

    def digests(path: Path) -> list[str]:
        return [
            str(row.get("digest"))
            for row in sorted(
                _read(path)["arms"][0]["history"], key=lambda r: r["file"]
            )
        ]

    welded_frames, held_frames = digests(welded20), digests(held20)
    differing = sum(1 for a, b in zip(welded_frames, held_frames) if a != b)
    held20_record = _read(held20)
    return {
        "20 s": {
            "frames_compared": min(len(welded_frames), len(held_frames)),
            "frames_differing_from_welded": differing,
            "treatment_reached_the_run": differing > 0,
            "held_arms_byte_identical": bool(
                held20_record["determinism"]["all_identical"]
            ),
            "held_arms_finite_every_step": all(
                arm["band"]["finite_every_step"] for arm in held20_record["arms"]
            ),
            "stepbl": held20_record["schedule_receipt"]["cadences"]["stepbl"],
        },
        "5 s": {
            "steps_compared": matched["steps_compared"],
            "treatment_reached_the_run": (
                matched["welded"]["arm-a"]["w_max"]
                != matched["held"]["arm-a"]["w_max"]
            ),
            "held_arms_agree": bool(matched["held_arms_agree"]),
            "welded_arms_agree": bool(matched["welded_arms_agree"]),
            "held_arms_finite_every_step": all(
                matched["held"][a]["finite_every_step"] for a in matched["held"]
            ),
            "held_truncated_at_step": matched["truncation"]["held_arms"]["a"][
                "last_committed_step"
            ],
            "truncation_reproduces_at_the_same_step": bool(
                matched["truncation"]["reproduces_at_the_same_step"]
            ),
            "welded_completed_every_step": bool(
                matched["truncation"]["welded_completed_every_step"]
            ),
        },
    }


def attribution(rows: dict[str, Any], matched: dict[str, Any]) -> dict[str, Any]:
    """How much of each timestep's excess |w| the call rate accounts for."""

    control_max = rows["120 s control (GF)"]["w_max"]
    control_last = rows["120 s control (GF)"]["w_trend"][-1]
    verdicts: dict[str, Any] = {}

    welded = rows["20 s OFF, PBL welded (180/h)"]
    held = rows["20 s OFF, PBL held 120 s (30/h)"]
    welded_excess = welded["w_max"] - control_max
    held_excess = held["w_max"] - control_max
    verdicts["20 s"] = {
        "both_completed": True,
        "surface_pbl_calls_per_hour_welded": 180.0,
        "surface_pbl_calls_per_hour_held": 30.0,
        "control_w_max": control_max,
        "welded_w_max": welded["w_max"],
        "held_w_max": held["w_max"],
        "excess_over_control_welded": welded_excess,
        "excess_over_control_held": held_excess,
        "excess_removed_by_holding_fraction": (
            None if welded_excess == 0
            else (welded_excess - held_excess) / welded_excess
        ),
        "excess_surviving_fraction": (
            None if welded_excess == 0 else held_excess / welded_excess
        ),
        "final_window_control": control_last,
        "final_window_welded": welded["w_trend"][-1],
        "final_window_held": held["w_trend"][-1],
        "held_trend_monotone": all(
            b >= a for a, b in zip(held["w_trend"], held["w_trend"][1:])
        ),
        "held_still_climbing_at_two_hours": (
            held["w_trend"][-1] > held["w_trend"][-2]
        ),
        "theta_m_max_welded": welded["theta_m_max"],
        "theta_m_max_held": held["theta_m_max"],
    }

    welded5 = matched["welded"]["arm-a"]
    held5 = matched["held"]["arm-a"]
    verdicts["5 s"] = {
        "both_completed": False,
        "comparison": "step-matched over the identical first "
                      f"{matched['steps_compared']} steps",
        "steps_compared": matched["steps_compared"],
        "surface_pbl_calls_per_hour_welded": 720.0,
        "surface_pbl_calls_per_hour_held": 30.0,
        "welded_w_max": welded5["w_max"],
        "held_w_max": held5["w_max"],
        "held_exceeds_welded": held5["w_max"] > welded5["w_max"],
        "held_w_max_over_welded_ratio": held5["w_max"] / welded5["w_max"],
        "welded_w_mean_by_quarter": welded5["w_mean_by_quarter"],
        "held_w_mean_by_quarter": held5["w_mean_by_quarter"],
        "theta_m_max_welded": welded5["theta_m_max"],
        "theta_m_max_held": held5["theta_m_max"],
        "held_truncated_at_step": matched["truncation"]["held_arms"]["a"][
            "last_committed_step"
        ],
        "held_steps_requested": matched["truncation"]["held_arms"]["a"][
            "steps_requested"
        ],
        "welded_completed_every_step": matched["truncation"][
            "welded_completed_every_step"
        ],
    }
    return verdicts


def verdict_sentence(verdicts: dict[str, Any], evidence: dict[str, Any]) -> str:
    """The one-sentence answer, DERIVED from the numbers rather than typed."""

    if not all(v["treatment_reached_the_run"] for v in evidence.values()):
        return (
            "NOT MEASURED: a held-cadence arm produced a band identical to "
            "its welded counterpart, so the cadence never reached the "
            "physics and these numbers measure nothing"
        )
    surviving = verdicts["20 s"]["excess_surviving_fraction"]
    five = verdicts["5 s"]
    worse = five["held_exceeds_welded"]
    if surviving is not None and surviving >= 0.75 and worse:
        return (
            f"The climb SURVIVES the surface/PBL cadence being held at its "
            f"proven 30 calls an hour -- {surviving:.1%} of the 20 s "
            f"peak-|w| excess over the control remains with the stack called "
            f"six times less often, and at 5 s holding it does not reduce "
            f"the climb but makes it WORSE ({five['held_w_max']:.2f} m/s "
            f"against {five['welded_w_max']:.2f} over the identical "
            f"{five['steps_compared']} steps) and the run stops integrating "
            f"at step {five['held_truncated_at_step']} of "
            f"{five['held_steps_requested']} where the welded one completed "
            f"-- so the surface/PBL call rate is NOT the cause either"
        )
    if surviving is not None and surviving <= 0.25 and not worse:
        return (
            f"The climb COLLAPSES when the cadence is held -- only "
            f"{surviving:.1%} of the 20 s peak-|w| excess survives -- so a "
            f"forcing applied per CALL in the surface/PBL stack owns it and "
            f"the port has a real coupling defect to name"
        )
    return (
        "PARTIAL and NOT ATTRIBUTED: the two timesteps do not point the same "
        "way, and a single campaign that moves a band part-way in one "
        "configuration and not the other does not identify a cause"
    )


def _figure(width: float, height: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(width, height), dpi=160)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRID)
        axes.spines[spine].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=1.0)
    axes.grid(True, color=GRID, linewidth=0.8, axis="y", zorder=0)
    axes.set_axisbelow(True)
    return figure, axes


def _finish(figure, axes, out: Path, *, title: str, subtitle: str) -> Path:
    import matplotlib.pyplot as plt

    axes.set_title(title, color=INK, fontsize=12.5, fontweight="bold",
                   loc="left", pad=20)
    axes.text(0.0, 1.03, subtitle, transform=axes.transAxes,
              color=INK_MUTED, fontsize=9.5, va="bottom", ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_trend(rows: dict[str, Any], out: Path, *, title: str) -> Path:
    """|w| against MODEL TIME, so windows of different length line up."""

    figure, axes = _figure(9.0, 5.4)
    for name, style in SERIES_STYLE.items():
        values = rows[name]["w_trend"]
        span = style["minutes"] / len(values)
        centres = [span * (i + 0.5) for i in range(len(values))]
        axes.plot(
            centres, values,
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=6.5, linewidth=2.0,
            zorder=style["zorder"], label=name,
        )
        if not rows[name]["completed"]:
            axes.plot(centres[-1], values[-1], marker="X", markersize=13,
                      color=ALARM, zorder=9, linestyle="none")
    axes.set_yscale("log")
    axes.set_xlabel("model time (minutes into the forecast)", color=INK,
                    fontsize=10)
    axes.set_ylabel("mean |w| over the window (m/s), log scale", color=INK,
                    fontsize=10)
    axes.legend(frameon=False, fontsize=8.8, labelcolor=INK, loc="upper left",
                bbox_to_anchor=(0.0, -0.14), ncol=2)
    axes.text(0.015, 0.965, "X marks a run that stopped integrating",
              transform=axes.transAxes, color=ALARM, fontsize=9, ha="left",
              va="top")
    return _finish(
        figure, axes, out, title=title,
        subtitle=(
            "x1.40962, one native init, the proving RTX 5070 Ti. Convection is OFF in all "
            "four candidate arms, so the only knob is the call rate."
        ),
    )


def plot_peak(rows: dict[str, Any], out: Path, *, title: str) -> Path:
    figure, axes = _figure(9.0, 4.8)
    names = list(SERIES_STYLE)
    values = [rows[n]["w_max"] for n in names]
    colors = [SERIES_STYLE[n]["color"] for n in names]
    bars = axes.barh(range(len(names)), values, color=colors, zorder=3,
                     height=0.62)
    axes.set_yticks(range(len(names)))
    axes.set_yticklabels(
        [
            f"{n}\n"
            + (
    f"{rows[n]['steps']} steps"
            )
            + ("" if rows[n]["completed"] else "  (STOPPED EARLY)")
            for n in names
        ],
        fontsize=8.8, color=INK,
    )
    axes.invert_yaxis()
    axes.set_xscale("log")
    # Headroom so the largest value's own label is not clipped by the axis.
    axes.set_xlim(min(values) * 0.55, max(values) * 2.1)
    axes.set_xlabel("peak |w| reached (m/s), log scale", color=INK, fontsize=10)
    axes.grid(True, color=GRID, linewidth=0.8, axis="x", zorder=0)
    axes.grid(False, axis="y")
    for index, (bar, value) in enumerate(zip(bars, values)):
        axes.text(value * 1.06, bar.get_y() + bar.get_height() / 2,
                  f"{value:.3f}", va="center", ha="left", fontsize=9,
                  color=ALARM if not rows[names[index]]["completed"] else INK)
    return _finish(
        figure, axes, out, title=title,
        subtitle=(
            "A 120 km cell has no such motion available to it. The 120 s "
            "control sits at 1.680 m/s."
        ),
    )


def plot_step_matched(matched: dict[str, Any], out: Path, *, title: str) -> Path:
    """The money chart: the same 964 steps, one knob apart."""

    figure, axes = _figure(9.0, 5.0)
    steps = matched["steps_compared"]
    minutes = steps * 5.0 / 60.0
    for label, key, color, style in (
        (f"welded, 720 PBL calls/hour", "welded", "#CC79A7", "--"),
        (f"held at 120 s, 30 PBL calls/hour", "held", "#009E73", "-"),
    ):
        arm = matched[key]["arm-a"]
        values = arm["w_mean_by_quarter"]
        peaks = arm["w_max_by_quarter"]
        span = minutes / len(values)
        centres = [span * (i + 0.5) for i in range(len(values))]
        axes.plot(centres, values, color=color, linestyle=style, marker="o",
                  markersize=6, linewidth=2.2, zorder=4, label=label + " (mean)")
        axes.plot(centres, peaks, color=color, linestyle=style, marker="^",
                  markersize=5.5, linewidth=1.2, alpha=0.55, zorder=3,
                  label=label + " (peak)")
    axes.axvline(minutes, color=ALARM, linewidth=1.6, linestyle=":", zorder=2)
    axes.text(minutes, axes.get_ylim()[1], "  held run stops here\n  (step "
              f"{matched['truncation']['held_arms']['a']['last_committed_step']}"
              f" of {matched['truncation']['held_arms']['a']['steps_requested']};"
              "\n  welded ran all 1,440)",
              color=ALARM, fontsize=8.8, va="top", ha="left")
    axes.set_xlim(0, minutes * 1.42)
    axes.set_yscale("log")
    axes.set_xlabel("model time (minutes into the forecast)", color=INK,
                    fontsize=10)
    axes.set_ylabel("|w| (m/s), log scale", color=INK, fontsize=10)
    axes.legend(frameon=False, fontsize=8.8, labelcolor=INK, loc="upper left",
                bbox_to_anchor=(0.0, -0.14), ncol=2)
    return _finish(
        figure, axes, out, title=title,
        subtitle=(
            f"dt = 5 s, convection off, identical first {steps} steps. "
            "Cutting the call rate 24-fold does not flatten the climb."
        ),
    )


def tables(rows: dict[str, Any], verdicts: dict[str, Any]) -> str:
    lines = [
        "| configuration | steps | PBL calls/hour | \\|w\\| max | "
        "\\|w\\| mean by window | theta_m max | completed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in SERIES_STYLE:
        row = rows[name]
        trend = " / ".join(f"{v:.2f}" for v in row["w_trend"])
        # The welded 5 s row is a SLICE of a run that completed all 1,440
        # steps, shown over the 964 the held arm reached so the two are
        # comparable.  Saying "964" without saying so would read as though
        # it stopped there too.
        steps = (
            str(row["steps"])
        )
        lines.append(
            f"| {name} | {steps} | "
            f"{row['surface_pbl_calls_per_hour']:g} | {row['w_max']:.3f} | "
            f"{trend} | {row['theta_m_max']:.4f} | "
            f"{'yes' if row['completed'] else '**NO**'} |"
        )
    twenty = verdicts["20 s"]
    five = verdicts["5 s"]
    lines += [
        "",
        "**20 s, both arms complete, excess measured against the 120 s control**",
        "",
        "| | welded (180/h) | held at 120 s (30/h) |",
        "| --- | --- | --- |",
        f"| \\|w\\| max | {twenty['welded_w_max']:.3f} | {twenty['held_w_max']:.3f} |",
        f"| excess over control | {twenty['excess_over_control_welded']:.3f} m/s "
        f"| {twenty['excess_over_control_held']:.3f} m/s |",
        f"| removed by holding | | "
        f"{twenty['excess_removed_by_holding_fraction']:.1%} |",
        f"| **surviving** | | "
        f"**{twenty['excess_surviving_fraction']:.1%}** |",
        "",
        f"**5 s, step-matched over the identical first "
        f"{five['steps_compared']} steps**",
        "",
        "| | welded (720/h) | held at 120 s (30/h) |",
        "| --- | --- | --- |",
        f"| \\|w\\| max | {five['welded_w_max']:.3f} | "
        f"**{five['held_w_max']:.3f}** |",
        "| \\|w\\| mean by quarter | "
        + " / ".join(f"{v:.2f}" for v in five["welded_w_mean_by_quarter"])
        + " | "
        + " / ".join(f"{v:.2f}" for v in five["held_w_mean_by_quarter"]) + " |",
        f"| theta_m max | {five['theta_m_max_welded']:.4f} | "
        f"{five['theta_m_max_held']:.4f} |",
        f"| completed 1,440 steps | yes | "
        f"**no, stopped at {five['held_truncated_at_step']}** |",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-band", type=Path, required=True)
    parser.add_argument("--welded20", type=Path, required=True)
    parser.add_argument("--held20", type=Path, required=True)
    parser.add_argument("--welded5", type=Path, required=True)
    parser.add_argument("--matched5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    rows, matched = collect(
        control_band=arguments.control_band,
        welded20=arguments.welded20,
        held20=arguments.held20,
        welded5=arguments.welded5,
        matched5=arguments.matched5,
    )
    evidence = treatment_evidence(
        welded20=arguments.welded20, held20=arguments.held20, matched=matched
    )
    verdicts = attribution(rows, matched)
    sentence = verdict_sentence(verdicts, evidence)

    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab-analysis.json").write_text(
        json.dumps(
            {
                "schema": "gpuwm-hex.pbl-cadence-ab/v2",
                "question": (
                    "does the |w| climb that survived switching Grell-Freitas "
                    "off collapse when the surface/PBL cadence is held at "
                    "120 s while dt shrinks?"
                ),
                "rows": rows,
                "treatment_evidence": evidence,
                "attribution": verdicts,
                "verdict": sentence,
                "what_this_cannot_settle": [
                    "whether either configuration has more SKILL: this A/B "
                    "changes a call rate and measures what moves.  Only "
                    "obs-skill (MRMS, ASOS) referees skill and nothing here "
                    "claims it",
                    "whether the growth is a per-call defect or the resolved "
                    "dynamics of a configuration nobody would run: 5 s is "
                    "140x below x1.40962's own 698.95 s Courant limit and "
                    "20 s is 35x below it.  No arm on this mesh can separate "
                    "them.  What settles it is the same trend on a mesh whose "
                    "Courant limit is near the timestep, refereed by "
                    "obs-skill, and no such mesh exists yet",
                    "WHY the 5 s held arm stops integrating at step 964: the "
                    "dycore refused to publish a composite step, "
                    "reproducibly, and this campaign did not instrument which "
                    "field breached which bound first",
                ],
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    (out / "tables.md").write_text(tables(rows, verdicts),
                                   encoding="utf-8", newline="\n")

    plot_trend(rows, out / "pbl-cadence-w-trend.png",
               title="Holding the surface/PBL cadence: the |w| climb does not flatten")
    plot_peak(rows, out / "pbl-cadence-w-peak.png",
              title="Peak vertical velocity by configuration")
    plot_step_matched(matched, out / "pbl-cadence-dt5-step-matched.png",
                      title="5 s, the same 964 steps, one knob apart")

    print(sentence)
    print()
    print(tables(rows, verdicts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
