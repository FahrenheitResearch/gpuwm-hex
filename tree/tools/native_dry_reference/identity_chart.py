#!/usr/bin/env python3
"""Analysis chart (not a weather field): words differing from native after one moist step.

usage: identity_chart.py OUT.png LABEL=COUNT [LABEL=COUNT ...]
One series, direct labels, log axis with zero drawn as an explicit '0' label.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
pairs = [a.rsplit("=", 1) for a in sys.argv[2:]]
labels = [p[0].replace("\\n", "\n") for p in pairs]
counts = [int(p[1]) for p in pairs]
fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
floor = 0.5
bars = ax.bar(labels, [max(c, floor) for c in counts], width=0.5, color="#2f6fb0")
ax.set_yscale("log")
ax.set_ylim(floor, max(counts) * 6)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, max(count, floor) * 1.25, f"{count:,}",
            ha="center", va="bottom", fontsize=12, color="#222222")
ax.set_ylabel("32-bit words differing from native (log scale)", color="#444444")
ax.set_title("One moist 360 s step, 240 km terrain mesh, physics off:\n"
             "hex vs native MPAS-A v8.4.2 strict, all 12 fields (36,127,824 words)", fontsize=12, loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.tick_params(colors="#444444")
fig.savefig(out, dpi=160)
print(out)
