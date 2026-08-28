#!/usr/bin/env python3
"""What a cascade cycle costs, before and after the anchor re-keying.

An ANALYSIS chart, not a weather field: matplotlib is allowed here and the
render law is not touched.  Weather fields in this lane's gallery come from
``rw_mpas_convert`` + ``rw_wrfbatch`` through ``gpuwm-hex render``.

Two panels, both from the cascade's own receipt:

1. the card time one cycle spends, leg by leg, with the forecast MINT the
   re-keying retired drawn beside it as the tax that used to be paid;
2. what a delayed start removes -- the fine-forecast hours a corridor no
   longer integrates before the weather it was placed for arrives.

Usage:
    python tools/plot_cascade_cost.py --receipt cascade-receipt.json \\
        --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: Measured by the nest-ratio lane on four new culls: a contract deck PLUS two
#: 1,080-step forecast mints, per re-placed geometry.  The mint half is what
#: the class re-keying retired; the range is 4,440 to 15,755 cells.
RETIRED_MINT_SECONDS = (288.0, 382.0)

LEG_ORDER = (
    "cull",
    "delayed-start",
    "boundaries",
    "contract-deck",
    "fine-forecast",
    "render",
)
LEG_COLOUR = {
    "cull": "#7a8b99",
    "delayed-start": "#4c9f70",
    "boundaries": "#9ab3c5",
    "contract-deck": "#d98b3a",
    "fine-forecast": "#3d5a80",
    "render": "#b8b8d1",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fine-hours", type=float, default=6.0)
    arguments = parser.parse_args()

    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    arguments.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for cycle in receipt["cycles"]:
        for slot in cycle["slots"]:
            if not slot.get("ran"):
                continue
            legs = {item["leg"]: float(item["seconds"]) for item in slot["legs"]}
            rows.append((cycle["cycle_index"], slot, legs))
    if not rows:
        raise SystemExit("the receipt records no cycle that ran a corridor")

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # -- panel 1: where a cycle's card time goes ---------------------------
    left = axes[0]
    labels = [f"cycle {index}" for index, _, _ in rows]
    bottoms = [0.0] * len(rows)
    for leg in LEG_ORDER:
        values = [legs.get(leg, 0.0) for _, _, legs in rows]
        if not any(values):
            continue
        left.bar(labels, values, bottom=bottoms, label=leg,
                 color=LEG_COLOUR.get(leg, "#cccccc"), edgecolor="white")
        bottoms = [a + b for a, b in zip(bottoms, values)]
    for index, total in enumerate(bottoms):
        left.text(index, total + 8, f"{total:,.0f} s", ha="center", fontsize=10)

    top = max(bottoms)
    # The tax, drawn as what it was: a bar beside each cycle, the same height
    # scale, so nobody has to convert a shaded band into seconds by eye.
    for index in range(len(rows)):
        left.bar(
            index + 0.42, RETIRED_MINT_SECONDS[1], width=0.14,
            bottom=0.0, color="#c0392b", alpha=0.35, edgecolor="none",
        )
        left.bar(
            index + 0.42, RETIRED_MINT_SECONDS[0], width=0.14,
            bottom=0.0, color="#c0392b", alpha=0.55, edgecolor="none",
        )
    left.text(
        (len(rows) - 1) / 2.0, max(bottoms) * 1.22,
        "what a re-placed cull used to pay\nBEFORE its forecast could run:\n"
        f"a second forecast mint, {RETIRED_MINT_SECONDS[0]:.0f}-"
        f"{RETIRED_MINT_SECONDS[1]:.0f} s,\nevery cycle. Retired 2026-08-27.",
        ha="center", va="bottom", fontsize=8.5, color="#7b241c",
    )
    left.set_ylim(0, top * 1.75)
    left.set_xlim(-0.6, len(rows) - 0.05)
    left.set_ylabel("seconds of card time")
    left.set_title("What one cycle costs, leg by leg", loc="left", fontsize=12)
    left.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    left.spines[["top", "right"]].set_visible(False)

    # -- panel 2: what the delayed start removes ---------------------------
    right = axes[1]
    positions = range(len(rows))
    without = []
    with_ds = []
    for _, slot, _ in rows:
        gap = float(slot.get("lead_gap_hours") or 0.0)
        with_ds.append(arguments.fine_hours)
        without.append(arguments.fine_hours + max(gap, 0.0))
    right.barh([p + 0.18 for p in positions], without, height=0.34,
               color="#c0392b", alpha=0.55, label="without a delayed start")
    right.barh([p - 0.18 for p in positions], with_ds, height=0.34,
               color="#4c9f70", label="with one")
    for index, ((_, slot, _), a, b) in enumerate(zip(rows, without, with_ds)):
        measured = slot.get("baseline_wall_seconds")
        ran = next(
            (item["seconds"] for item in slot["legs"]
             if item["leg"] == "fine-forecast"), None
        )
        note = f"+{a - b:.0f} h thrown away" if a > b else "nothing to remove"
        if measured and ran:
            note += f"  --  {measured:,.0f} s of card against {ran:,.0f} s"
        right.text(a + 0.15, index + 0.18, note, va="center", fontsize=9,
                   color="#7b241c")
        right.text(b + 0.15, index - 0.18, f"{b:.0f} h", va="center", fontsize=9)
    right.set_yticks(list(positions))
    right.set_yticklabels(labels)
    right.set_xlabel("fine-forecast hours the corridor has to integrate")
    right.set_title(
        "What a delayed start removes: hours before the weather arrives",
        loc="left", fontsize=12,
    )
    right.legend(frameon=False, fontsize=9, loc="lower right")
    right.spines[["top", "right"]].set_visible(False)
    right.set_xlim(0, max(without) * 1.95)

    figure.tight_layout()
    target = arguments.out / "01-what-a-cycle-costs.png"
    figure.savefig(target, dpi=150)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
