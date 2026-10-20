#!/usr/bin/env python3
"""Summarise a ``profile_forecast_step.py`` report into per-step cost tables.

Reads ``profile-steps.json`` and prints, for the steps a caller selects
(default: every step outside the two instrument windows and after warm-up),
the wall per step and its split into the composite transaction, the loop's
end-of-step synchronize and bookkeeping, the host health gate and the history
capture; the inclusive and exclusive host seconds of every region with the
device span its events measured; and the ``cupy.asnumpy`` census by caller.
Writes the same numbers as JSON beside the report when ``--out`` is given.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def _steps_in(report: dict[str, Any], first: int, last: int) -> list[dict[str, Any]]:
    return [s for s in report["steps"] if first <= int(s["step"]) <= last]


def region_table(steps: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per label: mean inclusive/exclusive host seconds per step, device span, calls per step."""
    n = max(len(steps), 1)
    inclusive: dict[str, float] = collections.defaultdict(float)
    exclusive: dict[str, float] = collections.defaultdict(float)
    gpu: dict[str, float] = collections.defaultdict(float)
    calls: dict[str, int] = collections.defaultdict(int)
    parents: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for step in steps:
        regions = step["regions"]
        # exclusive = inclusive minus the direct children's inclusive; a
        # child is a region opened while this one was on the stack, which
        # the recorder expressed as depth+1 and the parent label.
        child_seconds: dict[tuple[str, float], float] = collections.defaultdict(float)
        stack: list[dict[str, Any]] = []
        # regions are appended at END time, so re-sort by start to rebuild nesting
        ordered = sorted(regions, key=lambda r: (r["t0"], -r["t1"]))
        for region in ordered:
            while stack and stack[-1]["t1"] <= region["t0"]:
                stack.pop()
            if stack:
                parent = stack[-1]
                child_seconds[(parent["label"], parent["t0"])] += region["host_s"]
            stack.append(region)
        for region in regions:
            label = region["label"]
            inclusive[label] += region["host_s"]
            exclusive[label] += region["host_s"] - child_seconds.get((label, region["t0"]), 0.0)
            gpu_ms = region.get("gpu_ms")
            gpu[label] += 0.0 if gpu_ms is None or gpu_ms != gpu_ms else gpu_ms / 1000.0
            calls[label] += 1
            if region.get("parent"):
                parents[label][region["parent"]] += 1
    table = {}
    for label in inclusive:
        table[label] = {
            "inclusive_s": inclusive[label] / n,
            "exclusive_s": exclusive[label] / n,
            "device_span_s": gpu[label] / n,
            "calls_per_step": calls[label] / n,
            "parent": (parents[label].most_common(1)[0][0] if parents[label] else None),
        }
    return table


def step_walls(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Wall per step and its segments, from the recorder's timestamps."""
    walls = []
    composite = []
    loop_gap = []
    health = []
    post = []
    asnumpy_calls = []
    asnumpy_seconds = []
    for previous, step in zip(steps, steps[1:]):
        if int(step["step"]) != int(previous["step"]) + 1:
            continue  # a gap in the selection (an excluded window) is not a step wall
        wall = step["t_open"] - previous["t_open"]
        walls.append(wall)
        regions = {r["label"]: r for r in previous["regions"] if r["depth"] == 0}
        comp = regions.get("composite")
        gate = regions.get("health_gate")
        if comp is None or gate is None:
            continue
        composite.append(comp["host_s"])
        loop_gap.append(gate["t0"] - comp["t1"])
        health.append(gate["host_s"])
        post.append(step["t_open"] - gate["t1"])
        asnumpy_calls.append(previous["asnumpy_calls"])
        asnumpy_seconds.append(previous["asnumpy_seconds"])

    def stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    return {
        "wall": stat(walls),
        "composite": stat(composite),
        "loop_sync_and_bookkeeping": stat(loop_gap),
        "health_gate": stat(health),
        "after_health_gate": stat(post),
        "asnumpy_calls": stat(asnumpy_calls),
        "asnumpy_seconds": stat(asnumpy_seconds),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--first", type=int, default=None, help="first step of the selection")
    parser.add_argument("--last", type=int, default=None, help="last step of the selection")
    parser.add_argument("--exclude-windows", action="store_true", help="drop the nsys and cProfile windows from the selection")
    parser.add_argument("--warmup", type=int, default=20, help="steps dropped at the start (default 20)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    steps = report["steps"]
    last_step = max(int(s["step"]) for s in steps)
    first = args.first if args.first is not None else args.warmup + 1
    last = args.last if args.last is not None else last_step
    selected = _steps_in(report, first, last)
    windows = [w for w in (report.get("nsys_window"), report.get("cprofile_window")) if w]
    if args.exclude_windows:
        selected = [
            s for s in selected
            if not any(w[0] <= int(s["step"]) <= w[1] for w in windows)
        ]
    # the wall of a step needs its successor, so keep pairs inside the selection
    selection_label = f"steps {first}-{last}" + (" excluding windows" if args.exclude_windows else "")
    walls = step_walls(selected)
    table = region_table(selected)
    print(f"# {selection_label}: {len(selected)} steps; windows {windows}; nvtx={report.get('nvtx')}")
    print("## wall per step (s)")
    for key, value in walls.items():
        if value:
            print(f"  {key:28s} mean {value['mean']:.4f}  median {value['median']:.4f}  min {value['min']:.4f}  max {value['max']:.4f}  n {value['n']}")
    print("## regions (per step, seconds; sorted by exclusive host time)")
    print(f"  {'label':30s} {'incl_host':>10s} {'excl_host':>10s} {'dev_span':>10s} {'calls':>7s}  parent")
    for label, row in sorted(table.items(), key=lambda kv: -kv[1]["exclusive_s"])[: args.top]:
        print(
            f"  {label:30s} {row['inclusive_s']:10.4f} {row['exclusive_s']:10.4f} "
            f"{row['device_span_s']:10.4f} {row['calls_per_step']:7.2f}  {row['parent']}"
        )
    census = report.get("asnumpy_by_site", {})
    print("## cupy.asnumpy census over the whole run (calls, seconds, bytes) by caller")
    for site, row in list(census.items())[:25]:
        print(f"  {row['calls']:7d} {row['seconds']:9.3f}s {row['bytes'] / 1e6:10.1f} MB  {site}")
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "selection": selection_label,
                    "steps": len(selected),
                    "windows": windows,
                    "walls": walls,
                    "regions": table,
                    "asnumpy_by_site": census,
                    "preloop_regions": report.get("preloop", {}).get("regions", []),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
