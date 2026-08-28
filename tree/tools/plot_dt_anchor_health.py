#!/usr/bin/env python3
"""Chart what the timestep anchors measured, from the anchor JSONs.

These are ANALYSIS charts, not weather-field product plots: they show a
diagnostic band against lead time, computed from numbers the driver already
published in its own receipts.  The render law reserves `rw_wrfbatch` through
`gpuwm.rustwx` for weather fields and allows matplotlib for analysis charts,
which is what these are.

Two panels, and the second is the one that matters:

* **the schedule** -- how often each scheme is called at each anchored
  timestep, which is the thing a smaller timestep changes whether anyone
  wants it to or not (WRF pins ``cudt = 0`` for Grell-Freitas);
* **the vertical-velocity trend** -- each anchor's arm against the 120 s
  control measured on the same card, mesh and init, over four equal windows
  of the same 2 h forecast.  A band's min and max cannot separate a spin-up
  transient from a divergence that grows with lead time; this can.

Input is the campaign's own ``anchor-dt*.json`` files, so the chart cannot
drift from the evidence: every number plotted is read, none is restated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

#: Plotted in this order, coarsest first, so the eye reads dt shrinking.
ORDER = (120.0, 100.0, 75.0, 20.0, 5.0)


def load(evidence: Path) -> dict[float, dict[str, Any]]:
    """Every anchor record under an evidence directory, keyed by timestep."""

    records: dict[float, dict[str, Any]] = {}
    for path in sorted(evidence.glob("**/anchor-dt*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not record.get("integration_anchor_earned"):
            continue
        records.setdefault(float(record["dt_seconds"]), record)
    return records


def control_series(records: dict[float, dict[str, Any]]) -> list[float] | None:
    """The 120 s control's window means, from any anchor that measured one."""

    for dt in ORDER:
        record = records.get(dt)
        if record is None:
            continue
        control = (record.get("control") or {}).get("band")
        if not control:
            continue
        windows = (control.get("trend") or {}).get("windows") or []
        if windows:
            return [w["vertical_velocity_abs_max"]["mean"] for w in windows]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2.0)
    arguments = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load(arguments.evidence)
    if not records:
        print(f"no earned anchors under {arguments.evidence}", file=sys.stderr)
        return 1
    control = control_series(records)

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    figure.suptitle(
        "Model timestep anchors — what each earned timestep costs and changes\n"
        "x1.40962 (120 km global), 2 h, native MPAS-A v8.4.1 init, GFS 2026-08-12 06Z",
        fontsize=12,
    )

    # ---- panel 1: how often the schemes are called ----------------------
    timesteps = [dt for dt in ORDER if dt in records or dt == 120.0]
    calls = [3600.0 / dt for dt in timesteps]
    positions = range(len(timesteps))
    bars = left.bar(
        list(positions),
        calls,
        color=["#4c72b0" if dt == 120.0 else "#dd8452" for dt in timesteps],
    )
    left.set_xticks(list(positions))
    left.set_xticklabels([f"{dt:g} s" for dt in timesteps])
    left.set_yscale("log")
    left.set_ylim(20.0, max(calls) * 3.2)  # headroom so a label never meets the title
    left.set_ylabel("Grell-Freitas calls per forecast hour (log scale)")
    left.set_title("Convection is called every step, by WRF's own rule")
    for bar, value in zip(bars, calls):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:g}",
            ha="center",
            fontsize=9,
        )
    left.text(
        0.02,
        0.96,
        "blue = the proven timestep\norange = earned 2026-08-26",
        transform=left.transAxes,
        va="top",
        fontsize=8.5,
    )

    # ---- panel 2: the vertical-velocity trend ---------------------------
    window_hours = [
        arguments.hours * (index + 0.5) / 4 for index in range(4)
    ]
    if control is not None:
        right.plot(
            window_hours,
            control,
            marker="o",
            linewidth=3.0,
            color="#333333",
            linestyle="--",
            label="120 s control",
            zorder=5,
        )
    # one colour per timestep, none of them the control's
    palette = {100.0: "#4c72b0", 75.0: "#dd8452", 20.0: "#55a868", 5.0: "#c44e52"}
    for dt in ORDER:
        record = records.get(dt)
        if record is None or dt == 120.0:
            continue
        windows = (
            (record["arms"][0]["band"].get("trend") or {}).get("windows") or []
        )
        if not windows:
            continue
        right.plot(
            window_hours[: len(windows)],
            [w["vertical_velocity_abs_max"]["mean"] for w in windows],
            marker="s",
            linewidth=1.9,
            color=palette.get(dt),
            label=f"{dt:g} s",
        )
    right.set_xlabel("forecast hour (window centre)")
    # log: the 5 s arm reaches 87 m/s and a linear axis flattens the other
    # four series into the zero line, hiding the 20 s trend entirely
    right.set_yscale("log")
    right.set_ylabel("mean |w| over the window, m/s (log scale)")
    right.set_title("Vertical velocity against the same card's 120 s control")
    right.legend(fontsize=9)
    right.grid(alpha=0.3)

    figure.tight_layout(rect=(0, 0.02, 1, 0.90))
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.out, dpi=150)
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
