#!/usr/bin/env python3
"""Read the convection-off A/B and answer one question with numbers.

THE QUESTION: does the vertical-velocity climb the 20 s and 5 s Grell-Freitas
anchors measured survive with the cumulus scheme switched off?

Five configurations, one mesh (``x1.40962``), one init, one card family, one
2 h forecast each:

* the anchored **120 s control** with GF -- the reference band every other
  arm is read against;
* **20 s GF** and **5 s GF** -- the anchors minted 2026-08-26, which DIVERGE;
* **20 s OFF** and **5 s OFF** -- this campaign's arms, identical in every
  input except the cumulus selection.

The A/B is clean because only one thing moves.  What it can settle is
ATTRIBUTION -- whether a band the convection-off configuration does not
produce was being produced by the closure.  What it cannot settle is which
configuration has more skill; only obs-skill referees that, and this tool
does not claim it.

The charts are ANALYSIS charts, not weather fields, so matplotlib is the
right tool for them under the render law.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Okabe-Ito, assigned in fixed order and never cycled.  Hue family carries
#: the TREATMENT so the A/B reads at a glance -- warm for a run that calls
#: Grell-Freitas, cool for a run that does not -- and line style carries it a
#: second time, so identity never rests on colour alone.
SERIES_STYLE: dict[str, dict[str, Any]] = {
    "120 s control (GF)": {
        "color": "#3D3D3D", "linestyle": (0, (1, 1)), "marker": "o", "zorder": 5
    },
    "20 s GF": {"color": "#D55E00", "linestyle": "--", "marker": "s", "zorder": 3},
    "5 s GF": {"color": "#CC79A7", "linestyle": "--", "marker": "D", "zorder": 4},
    "20 s convection OFF": {
        "color": "#0072B2", "linestyle": "-", "marker": "s", "zorder": 3
    },
    "5 s convection OFF": {
        "color": "#009E73", "linestyle": "-", "marker": "D", "zorder": 4
    },
}

INK = "#1A1A1A"
INK_MUTED = "#6B6B6B"
GRID = "#DDDDDD"
SURFACE = "#FFFFFF"


def _band_from_anchor(path: Path) -> dict[str, Any]:
    """One campaign's arm-a band, which is the arm the health is read from."""

    record = json.loads(path.read_text(encoding="utf-8"))
    return record["arms"][0]["band"]


def _trend(band: dict[str, Any], field: str = "vertical_velocity_abs_max") -> list[float]:
    return [
        float(window[field]["mean"]) for window in band["trend"]["windows"]
    ]


def _maximum(band: dict[str, Any], field: str = "vertical_velocity_abs_max") -> float:
    return float(band[field]["max"])


def collect(
    *,
    control_band: Path,
    gf20: Path,
    gf5: Path,
    off20: Path,
    off5: Path,
) -> dict[str, Any]:
    """Every number the verdict is read from, in one mapping."""

    control = json.loads(control_band.read_text(encoding="utf-8"))
    series = {
        "120 s control (GF)": control,
        "20 s GF": _band_from_anchor(gf20),
        "5 s GF": _band_from_anchor(gf5),
        "20 s convection OFF": _band_from_anchor(off20),
        "5 s convection OFF": _band_from_anchor(off5),
    }
    rows = {}
    for name, band in series.items():
        rows[name] = {
            "steps": int(band["steps"]),
            "finite_every_step": bool(band["finite_every_step"]),
            "w_trend": _trend(band),
            "w_max": _maximum(band),
            "theta_m_max": float(band["theta_m_max"]["max"]),
            "theta_m_min": float(band["theta_m_min"]["min"]),
            "qv_max": float(band["qv_max"]["max"]),
            "exner_min": float(band["exner_min"]["min"]),
            "rho_min": float(band["rho_min"]["min"]),
        }
    return rows


def attribution(rows: dict[str, Any]) -> dict[str, Any]:
    """How much of each timestep's excess |w| the closure accounts for.

    The excess is measured against the SAME 120 s control both arms are read
    against, because the question is about the part of the band the control
    does not have.  Reported as a fraction and as the two raw numbers, so a
    reader can see the arithmetic rather than take the fraction.
    """

    control_max = rows["120 s control (GF)"]["w_max"]
    control_last = rows["120 s control (GF)"]["w_trend"][-1]
    verdicts = {}
    for dt in ("20 s", "5 s"):
        gf = rows[f"{dt} GF"]
        off = rows[f"{dt} convection OFF"]
        gf_excess_max = gf["w_max"] - control_max
        off_excess_max = off["w_max"] - control_max
        gf_excess_last = gf["w_trend"][-1] - control_last
        off_excess_last = off["w_trend"][-1] - control_last
        verdicts[dt] = {
            "control_w_max": control_max,
            "gf_w_max": gf["w_max"],
            "off_w_max": off["w_max"],
            "excess_over_control_gf": gf_excess_max,
            "excess_over_control_off": off_excess_max,
            "excess_removed_by_switching_off_fraction": (
                None if gf_excess_max == 0 else
                (gf_excess_max - off_excess_max) / gf_excess_max
            ),
            "excess_surviving_fraction": (
                None if gf_excess_max == 0 else off_excess_max / gf_excess_max
            ),
            "final_window_control": control_last,
            "final_window_gf": gf["w_trend"][-1],
            "final_window_off": off["w_trend"][-1],
            "final_window_excess_surviving_fraction": (
                None if gf_excess_last == 0 else off_excess_last / gf_excess_last
            ),
            "off_trend_monotone": all(
                b >= a for a, b in zip(off["w_trend"], off["w_trend"][1:])
            ),
            "off_still_climbing_at_two_hours": (
                off["w_trend"][-1] > off["w_trend"][-2]
            ),
            "theta_m_max_gf": gf["theta_m_max"],
            "theta_m_max_off": off["theta_m_max"],
            "theta_m_max_control": rows["120 s control (GF)"]["theta_m_max"],
        }
    return verdicts


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


def plot_trend(rows: dict[str, Any], out: Path, *, title: str) -> Path:
    """The half-hour |w| trend, all five configurations, one axis.

    The headline is passed in rather than written here: it states what the
    measurement FOUND, and a title hardcoded before the runs would be a
    conclusion decided in advance.
    """

    import matplotlib.pyplot as plt

    figure, axes = _figure(8.6, 5.0)
    windows = [1, 2, 3, 4]
    endpoints: list[tuple[float, str]] = []
    for name, style in SERIES_STYLE.items():
        values = rows[name]["w_trend"]
        axes.plot(
            windows, values,
            color=style["color"], linestyle=style["linestyle"],
            marker=style["marker"], markersize=6.5, linewidth=2.0,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
            label=name, zorder=style["zorder"],
        )
        endpoints.append((values[-1], name))
    axes.set_yscale("log")
    # Selective direct labels: the end of each line, never every point --
    # and nudged apart when two lines finish close together, because two
    # numbers printed on top of each other are worse than none.
    import math

    endpoints.sort()
    offsets: dict[str, float] = {}
    previous_log = None
    previous_offset = 0.0
    for value, name in endpoints:
        current_log = math.log10(value)
        if previous_log is not None and current_log - previous_log < 0.055:
            previous_offset = previous_offset + 9.0
        else:
            previous_offset = 0.0
        offsets[name] = previous_offset
        previous_log = current_log
    for value, name in endpoints:
        axes.annotate(
            f"{value:.2f}",
            xy=(windows[-1], value),
            xytext=(7, offsets[name]), textcoords="offset points",
            color=INK, fontsize=9, va="center",
        )
    axes.set_xticks(windows)
    axes.set_xticklabels(["0-30 min", "30-60 min", "60-90 min", "90-120 min"])
    axes.set_xlim(0.85, 4.45)
    axes.set_ylabel("mean |w| over the window  (m/s, log scale)", color=INK, fontsize=10)
    axes.set_title(
        title,
        color=INK, fontsize=13, pad=34, loc="left",
    )
    axes.text(
        0.0, 1.015,
        "x1.40962, one init, 2 h forecasts. Warm dashed = Grell-Freitas on; "
        "cool solid = convection off.",
        transform=axes.transAxes, color=INK_MUTED, fontsize=9, va="bottom",
    )
    legend = axes.legend(
        frameon=False, fontsize=9, loc="upper center",
        bbox_to_anchor=(0.5, -0.09), ncol=3,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out


def plot_peak(rows: dict[str, Any], out: Path) -> Path:
    """Peak |w| over the whole arm: magnitude, so bars from a zero baseline."""

    import matplotlib.pyplot as plt

    figure, axes = _figure(8.6, 4.4)
    names = list(SERIES_STYLE)
    values = [rows[name]["w_max"] for name in names]
    positions = range(len(names))
    axes.bar(
        positions, values,
        color=[SERIES_STYLE[name]["color"] for name in names],
        width=0.62, zorder=2, edgecolor=SURFACE, linewidth=2.0,
    )
    for position, value in zip(positions, values):
        axes.annotate(
            f"{value:.2f}", xy=(position, value), xytext=(0, 4),
            textcoords="offset points", ha="center",
            color=INK, fontsize=10,
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(
        [name.replace(" convection OFF", "\nconvection OFF").replace(" (GF)", "\n(GF)")
         for name in names],
        color=INK, fontsize=9,
    )
    axes.set_ylabel("peak |w| over the 2 h arm  (m/s)", color=INK, fontsize=10)
    axes.set_ylim(0, max(values) * 1.18)
    axes.set_title(
        "Peak vertical velocity, same mesh and init, only the scheme moves",
        color=INK, fontsize=13, pad=14, loc="left",
    )
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE)
    plt.close(figure)
    return out


def plot_excess(verdicts: dict[str, Any], out: Path, *, title: str) -> Path:
    """How much of the excess over the control each treatment leaves behind."""

    import matplotlib.pyplot as plt

    figure, axes = _figure(7.6, 4.2)
    labels = ["20 s", "5 s"]
    removed = [
        100.0 * verdicts[dt]["excess_removed_by_switching_off_fraction"]
        for dt in labels
    ]
    surviving = [100.0 - value for value in removed]
    positions = range(len(labels))
    axes.bar(
        positions, removed, width=0.55, color="#0072B2", zorder=2,
        edgecolor=SURFACE, linewidth=2.0,
        label="removed by switching convection off",
    )
    axes.bar(
        positions, surviving, bottom=removed, width=0.55, color="#E69F00",
        zorder=2, edgecolor=SURFACE, linewidth=2.0,
        label="survives with the scheme off",
    )
    for position, value in zip(positions, removed):
        axes.annotate(
            f"{value:.0f}%", xy=(position, value / 2), ha="center", va="center",
            color=SURFACE, fontsize=11,
        )
        axes.annotate(
            f"{100 - value:.0f}%", xy=(position, value + (100 - value) / 2),
            ha="center", va="center", color=INK, fontsize=11,
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(
        [f"{label}\ntimestep" for label in labels], color=INK, fontsize=10
    )
    axes.set_ylim(0, 100)
    axes.set_ylabel(
        "share of the peak-|w| excess over the 120 s control  (%)",
        color=INK, fontsize=10,
    )
    axes.set_title(title, color=INK, fontsize=13, pad=14, loc="left")
    legend = axes.legend(frameon=False, fontsize=9, loc="lower center",
                         bbox_to_anchor=(0.5, -0.34), ncol=2)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return out


def markdown(rows: dict[str, Any], verdicts: dict[str, Any]) -> str:
    """The receipt's tables, generated rather than transcribed.

    Hand-typing a band into prose is how a receipt comes to disagree with its
    own evidence.  These are the same numbers the JSON carries.
    """

    lines = [
        "| configuration | steps | GF calls/hour | |w| max | "
        "|w| mean by half-hour | theta_m max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    calls = {
        "120 s control (GF)": "30",
        "20 s GF": "180",
        "5 s GF": "720",
        "20 s convection OFF": "0",
        "5 s convection OFF": "0",
    }
    for name in SERIES_STYLE:
        row = rows[name]
        trend = " / ".join(f"{value:.2f}" for value in row["w_trend"])
        lines.append(
            f"| {name} | {row['steps']} | {calls[name]} | "
            f"{row['w_max']:.3f} | {trend} | {row['theta_m_max']:.2f} |"
        )
    lines.append("")
    lines.append(
        "| timestep | excess over control, GF | excess over control, OFF | "
        "removed by switching off | surviving |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for dt in ("20 s", "5 s"):
        entry = verdicts[dt]
        lines.append(
            f"| {dt} | {entry['excess_over_control_gf']:.3f} m/s | "
            f"{entry['excess_over_control_off']:.3f} m/s | "
            f"{100.0 * entry['excess_removed_by_switching_off_fraction']:.1f} % | "
            f"{100.0 * entry['excess_surviving_fraction']:.1f} % |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-band", type=Path, required=True)
    parser.add_argument("--gf20", type=Path, required=True)
    parser.add_argument("--gf5", type=Path, required=True)
    parser.add_argument("--off20", type=Path, required=True)
    parser.add_argument("--off5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, default=None)
    parser.add_argument(
        "--trend-title",
        default="The vertical-velocity trend, with and without the closure",
    )
    parser.add_argument(
        "--excess-title",
        default="How much of the excess over the control the closure accounts for",
    )
    arguments = parser.parse_args(argv)

    rows = collect(
        control_band=arguments.control_band,
        gf20=arguments.gf20,
        gf5=arguments.gf5,
        off20=arguments.off20,
        off5=arguments.off5,
    )
    verdicts = attribution(rows)
    payload = {
        "schema": "gpuwm-hex.convection-off-ab/v1",
        "question": (
            "does the |w| climb the 20 s and 5 s Grell-Freitas anchors "
            "measured survive with the cumulus scheme switched off?"
        ),
        "bands": rows,
        "attribution": verdicts,
        "what_this_cannot_settle": [
            "which configuration has more SKILL: this A/B removes a forcing "
            "and measures what changes, and only obs-skill (MRMS, ASOS) "
            "referees whether the result is better weather",
            "whether the surviving climb is resolved dynamics or the "
            "configuration itself: x1.40962 is a 120 km mesh and these "
            "timesteps sit 35x and 140x below its own 698.95 s Courant "
            "limit, which no arm of this campaign changes",
        ],
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )

    targets = [arguments.out.parent]
    if arguments.gallery is not None:
        targets.append(arguments.gallery)
    for target in targets:
        plot_trend(
            rows, target / "convection-off-w-trend.png",
            title=arguments.trend_title,
        )
        plot_peak(rows, target / "convection-off-w-peak.png")
        plot_excess(
            verdicts, target / "convection-off-excess-share.png",
            title=arguments.excess_title,
        )

    table = markdown(rows, verdicts)
    (arguments.out.parent / "tables.md").write_text(
        table, encoding="utf-8", newline="\n"
    )
    print(table)
    print(json.dumps(payload["attribution"], indent=2, sort_keys=True))
    print(f"\nwritten: {arguments.out}")
    for target in targets:
        print(f"charts:  {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
