#!/usr/bin/env python3
"""Reduce a bounded Nsight trace to small C1 evidence and API counts.

The gpuwm timeline CLI calculates both PROXY FF and card-busy fraction.
This reducer only exports recorded intervals and host-loop gap durations.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = {row[0]: row[1] for row in connection.execute("SELECT id,value FROM StringIds")}
    ranges = []
    for row in connection.execute("SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = row["text"] or strings.get(row["textId"], "")
        if name.startswith("hex.step."):
            ranges.append({"step": int(name.rsplit(".", 1)[1]),
                           "start": row["start"], "end": row["end"]})
    ranges.sort(key=lambda row: row["start"])
    if len(ranges) < 2:
        raise ValueError("trace has fewer than two complete hex step ranges")
    start, end = ranges[0]["start"], ranges[-1]["end"]
    events = []

    def event(stage, phase, ns, attrs=None):
        events.append({"v": 1, "rental_id": None,
            "run_id": "nsys-window", "t_utc": datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat().replace("+00:00", "Z"), "t_mono": ns / 1e9,
            "host": "nsys-relative-clock", "product": "hex", "stage": stage,
            "phase": phase, "card": args.card, "pid": os.getpid(),
            "attrs": {"proxy": True, "basis": "bounded Nsight capture window", **(attrs or {})}})

    event("forecast.integrate", "begin", start, {"card_count": 1})
    event("forecast.integrate", "end", end, {"card_count": 1})
    operations = Counter()
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in tables:
            continue
        for row in connection.execute(f"SELECT start,end FROM {table} WHERE end>? AND start<?", (start, end)):
            left, right = max(start, row["start"]), min(end, row["end"])
            span = f"{table}-{operations[table]}"
            event("forecast.integrate", "begin", left, {"card_busy": True, "span_id": span})
            event("forecast.integrate", "end", right, {"card_busy": True, "span_id": span})
            operations[table] += 1
    api = Counter()
    api_seconds = Counter()
    for row in connection.execute("SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=?", (start, end)):
        name = strings[row["nameId"]]
        api[name] += 1
        api_seconds[name] += (row["end"] - row["start"]) / 1e9
    gaps = [(right["start"] - left["end"]) / 1e9 for left, right in zip(ranges, ranges[1:])]
    summary = {"schema": "hex-pipeline-trace-v1", "source_database": str(args.database),
        "clock_basis": "Nsight relative timestamps, not rental wall time",
        "step_ranges": ranges, "window_seconds": (end - start) / 1e9,
        "between_composite_host_gap_seconds": gaps,
        "between_composite_host_gap_total_seconds": sum(gaps),
        "between_composite_host_gap_mean_seconds": sum(gaps) / len(gaps),
        "gap_scope": "loop host work between composite calls, including snapshot enqueues; output cadence retained",
        "api_counts": dict(api), "api_host_seconds": dict(api_seconds), "device_operations": dict(operations)}
    ledger = args.output / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                              for row in sorted(events, key=lambda row: row["t_mono"])))
    (args.output / "trace-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return subprocess.call([sys.executable, "-m", "gpuwm.cli", "timeline", "report", str(ledger),
        "--proxy-start", str(start / 1e9), "--proxy-end", str(end / 1e9), "--run-id", "nsys-window",
        "--product", "hex", "--cards", "1", "--output", str(args.output / "timeline.json"),
        "--png", str(args.output / "timeline.png")])


if __name__ == "__main__":
    raise SystemExit(main())
