#!/usr/bin/env python3
"""Reduce an Nsight Systems sqlite export of a profiled forecast to per-step numbers.

Companion of ``profile_forecast_step.py``: that instrument brackets a step
window with ``cudaProfilerStart``/``Stop`` and names every phase with an NVTX
range, so the export carries, for exactly those steps, every kernel launch,
memcpy and memset with its device timing, every CUDA runtime call with its
host timing, and the ranges.  This reads the export and prints:

* per step: device busy time (union of every GPU op), host wall, launches,
  host time inside blocking runtime calls;
* per NVTX label (innermost range holding the launch): device busy, launch
  count, host time in each blocking runtime call, so a host stall is
  attributed to the phase it sits in;
* the kernels ranked by device time per step, with the launch geometry
  (grid, block, registers, shared memory), a theoretical occupancy on the
  card the export names, how many waves the grid makes, and a classification
  that is honest about what a timeline can and cannot say -- no hardware
  counters were read, so "memory-bound" is never claimed, only bounded.

Usage::

    python tools/analyze_nsys_profile.py hexprof.sqlite --out nsys-analysis.json
"""

from __future__ import annotations

import argparse
import bisect
import collections
import csv
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

BLOCKING_APIS = (
    "cudaMemcpyAsync",
    "cudaMemcpy",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaMemcpyToSymbol",
    "cudaMalloc",
    "cudaFree",
    "cuMemcpyDtoHAsync",
    "cuMemcpyHtoDAsync",
    "cuMemcpyDtoH",
    "cuMemcpyHtoD",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
    "cuMemAlloc",
    "cuMemFree",
)


def _union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The disjoint union of [start, end) intervals."""
    if not intervals:
        return []
    intervals.sort()
    result = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start > current_end:
            result.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    result.append((current_start, current_end))
    return result


def _union_seconds(intervals: list[tuple[int, int]]) -> float:
    """Length of the union of [start, end) intervals, in seconds."""
    return sum(end - start for start, end in _union_intervals(intervals)) / 1e9


def _card_limits(connection: sqlite3.Connection) -> dict[str, Any]:
    columns = [r[1] for r in connection.execute("pragma table_info(TARGET_INFO_GPU)")]
    row = connection.execute("select * from TARGET_INFO_GPU limit 1").fetchone()
    info = dict(zip(columns, row)) if row else {}
    return {
        "name": info.get("name"),
        "sm_count": int(info.get("smCount") or 170),
        "max_threads_per_sm": int(info.get("maxThreadsPerMultiprocessor") or 1536),
        "max_blocks_per_sm": int(info.get("maxBlocksPerMultiprocessor") or 24),
        "regs_per_sm": int(info.get("maxRegistersPerMultiprocessor") or 65536),
        "smem_per_sm": int(info.get("maxSharedMemoryPerMultiprocessor") or 102400),
        "raw": {k: v for k, v in info.items() if isinstance(v, (int, float, str))},
    }


def occupancy(card: dict[str, Any], block: int, regs: int, smem: int) -> dict[str, Any]:
    """Theoretical occupancy the way the occupancy calculator does it."""
    warps_per_block = max(1, math.ceil(block / 32))
    max_warps = card["max_threads_per_sm"] // 32
    by_threads = max_warps // warps_per_block
    by_blocks = card["max_blocks_per_sm"]
    regs_rounded = max(8, math.ceil(max(regs, 1) / 8) * 8)
    regs_per_warp = math.ceil(regs_rounded * 32 / 256) * 256
    warps_by_regs = card["regs_per_sm"] // regs_per_warp
    by_regs = warps_by_regs // warps_per_block if warps_per_block else 0
    by_smem = (card["smem_per_sm"] // max(smem, 1)) if smem > 0 else by_blocks
    blocks_per_sm = max(0, min(by_threads, by_blocks, by_regs, by_smem))
    active_warps = blocks_per_sm * warps_per_block
    return {
        "warps_per_block": warps_per_block,
        "blocks_per_sm": blocks_per_sm,
        "theoretical_occupancy": active_warps / max_warps if max_warps else 0.0,
        "limiter": min(
            (("threads", by_threads), ("blocks", by_blocks), ("registers", by_regs), ("shared_memory", by_smem)),
            key=lambda kv: kv[1],
        )[0],
    }


def require_graph_node_capture(connection: sqlite3.Connection) -> None:
    """Graph-level traces omit the kernels and cannot support this accounting."""
    table = "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
    if connection.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone():
        if connection.execute(f"select count(*) from {table}").fetchone()[0]:
            raise ValueError("CUDA graph-level capture omits graph node activities; rerun nsys profile with --cuda-graph-trace=node before computing per-kernel costs or GPU idle time")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None, help="ranked kernel cost table")
    parser.add_argument("--points", type=int, default=None, help="cells times vertical levels, or the WRF three-dimensional point count")
    parser.add_argument("--first", type=int, default=None, help="first complete NVTX step to include")
    parser.add_argument("--last", type=int, default=None, help="last complete NVTX step to include")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--bandwidth-tb-s", type=float, default=1.792, help="the card's DRAM bandwidth ceiling used for the bytes-at-ceiling bound")
    args = parser.parse_args(argv)

    if args.points is not None and args.points <= 0:
        parser.error("--points must be positive")
    connection = sqlite3.connect(args.sqlite.resolve().as_uri() + "?mode=ro", uri=True)
    require_graph_node_capture(connection)
    strings = {r[0]: r[1] for r in connection.execute("select id, value from StringIds")}
    card = _card_limits(connection)
    api_rows = []
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if connection.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone():
            api_rows.extend(connection.execute(
                f"select start, end, correlationId, nameId, globalTid from {table}"
            ))

    # ---- NVTX ranges (push/pop): steps and labels, on the launching thread
    ranges = []
    for start, end, text, text_id, tid in connection.execute(
        "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"
    ):
        label = text if text is not None else strings.get(text_id)
        if label is None:
            continue
        ranges.append((int(start), int(end), str(label), int(tid)))
    ranges.sort()
    steps = [(s, e, lab) for s, e, lab, _ in ranges if lab.startswith("step ")]
    if args.first is not None or args.last is not None:
        steps = [
            (s, e, lab) for s, e, lab in steps
            if (args.first is None or int(lab.split()[1]) >= args.first)
            and (args.last is None or int(lab.split()[1]) <= args.last)
        ]
    if not steps:
        parser.error("no complete NVTX 'step N' ranges in the requested trace window")
    labels = [(s, e, lab) for s, e, lab, _ in ranges if not lab.startswith("step ")]
    # innermost label per runtime call: one sweep over calls and ranges in
    # time order with a stack of open ranges (push/pop ranges nest properly
    # on the launching thread, so the top of the stack is the innermost).
    call_times = sorted(
        (int(start), int(correlation), int(tid))
        for start, _, correlation, _, tid in api_rows
    )
    label_of_call: dict[int, str | None] = {}
    ranges_by_thread = collections.defaultdict(list)
    for s, e, lab, tid in ranges:
        if not lab.startswith("step "):
            ranges_by_thread[tid].append((s, e, lab))
    stacks = collections.defaultdict(list)
    cursors = collections.defaultdict(int)
    for time_ns, correlation, tid in call_times:
        thread_labels = ranges_by_thread[tid]
        cursor = cursors[tid]
        open_stack = stacks[tid]
        while cursor < len(thread_labels) and thread_labels[cursor][0] <= time_ns:
            open_stack.append(thread_labels[cursor])
            cursor += 1
        cursors[tid] = cursor
        while open_stack and open_stack[-1][1] <= time_ns:
            open_stack.pop()
        label_of_call[correlation] = open_stack[-1][2] if open_stack else None

    def innermost_call(correlation: int) -> str | None:
        return label_of_call.get(correlation)

    def step_of(time_ns: int) -> str | None:
        for start, end, label in steps:
            if start <= time_ns <= end:
                return label
        return None

    step_index = {label: i for i, (_, _, label) in enumerate(steps)}
    step_bounds = [(s, e) for s, e, _ in steps]
    step_starts = [s for s, _ in step_bounds]

    def step_fast(time_ns: int) -> int | None:
        index = bisect.bisect_right(step_starts, time_ns) - 1
        if index >= 0 and step_bounds[index][0] <= time_ns < step_bounds[index][1]:
            return index
        return None

    # ---- runtime calls: launch times by correlation id, blocking API stalls
    launch_time: dict[int, int] = {}
    api_by_step: dict[int, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    api_by_label: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    api_totals: dict[str, list[float]] = collections.defaultdict(list)
    runtime_api_totals: dict[str, list[float]] = collections.defaultdict(list)
    runtime_api_by_label = collections.defaultdict(lambda: collections.defaultdict(list))
    launch_api_by_step = collections.defaultdict(list)
    for start, end, correlation, name_id, _ in api_rows:
        name = strings.get(name_id, str(name_id))
        base = name.split("_v")[0]
        launch_time[int(correlation)] = min(int(start), launch_time.get(int(correlation), int(start)))
        index = step_fast(int(start))
        if index is None:
            continue
        seconds = (int(end) - int(start)) / 1e9
        runtime_api_totals[base].append(seconds)
        label = innermost_call(int(correlation)) or "(outside any range)"
        runtime_api_by_label[label][base].append(seconds)
        if base.startswith(("cudaLaunch", "cuLaunch")):
            launch_api_by_step[index].append((int(start), int(end)))
        if base in BLOCKING_APIS:
            api_totals[base].append(seconds)
            api_by_step[index][base].append(seconds)
            api_by_label[label][base].append(seconds)

    # ---- GPU ops: kernels, memcpy, memset
    ops_by_step: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    ops_by_label: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    launches_by_step: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    launches_by_label: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kernel_rows: dict[str, dict[str, Any]] = {}
    kernel_by_label: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kernel_time_by_label: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    n_steps = len(steps)
    all_gpu_ops: list[tuple[int, int]] = []

    for row in connection.execute(
        "select start, end, correlationId, shortName, registersPerThread, gridX, gridY, gridZ, "
        "blockX, blockY, blockZ, staticSharedMemory, dynamicSharedMemory, localMemoryPerThread "
        "from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        start, end, correlation, short_id, regs, gx, gy, gz, bx, by, bz, ssm, dsm, lmem = row
        name = strings.get(short_id, str(short_id))
        all_gpu_ops.append((int(start), int(end)))
        launched = launch_time.get(int(correlation), int(start))
        index = step_fast(launched)
        if index is None:
            continue
        label = innermost_call(int(correlation)) or "(outside any range)"
        duration = int(end) - int(start)
        if index is not None:
            ops_by_step[index].append((int(start), int(end)))
            launches_by_step[index]["kernel"] += 1
        ops_by_label[label].append((int(start), int(end)))
        launches_by_label[label]["kernel"] += 1
        kernel_by_label[label][name] += 1
        kernel_time_by_label[label][name] += duration
        entry = kernel_rows.setdefault(
            name,
            {
                "instances": 0,
                "total_ns": 0,
                "durations": [],
                "geometry": collections.Counter(),
                "labels": collections.Counter(),
            },
        )
        entry["instances"] += 1
        entry["total_ns"] += duration
        entry["durations"].append(duration)
        entry["geometry"][(int(gx), int(gy), int(gz), int(bx), int(by), int(bz), int(regs or 0), int(ssm or 0), int(dsm or 0), int(lmem or 0))] += 1
        entry["labels"][label] += 1

    memcpy_by_step: dict[int, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    memcpy_by_label: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    memcpy_time_by_label = collections.defaultdict(lambda: collections.defaultdict(float))
    copy_kinds = {r[0]: r[1] for r in connection.execute("select id, name from ENUM_CUDA_MEMCPY_OPER")} if connection.execute(
        "select count(*) from sqlite_master where name='ENUM_CUDA_MEMCPY_OPER'"
    ).fetchone()[0] else {}
    for start, end, correlation, nbytes, kind in connection.execute(
        "select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        all_gpu_ops.append((int(start), int(end)))
        launched = launch_time.get(int(correlation), int(start))
        index = step_fast(launched)
        if index is None:
            continue
        label = innermost_call(int(correlation)) or "(outside any range)"
        kind_name = copy_kinds.get(kind, str(kind))
        if index is not None:
            ops_by_step[index].append((int(start), int(end)))
            launches_by_step[index]["memcpy"] += 1
            memcpy_by_step[index][kind_name].append(int(nbytes))
        ops_by_label[label].append((int(start), int(end)))
        launches_by_label[label]["memcpy"] += 1
        memcpy_by_label[label][kind_name].append(int(nbytes))
        memcpy_time_by_label[label][kind_name] += (int(end) - int(start)) / 1e9
    for start, end, correlation in connection.execute(
        "select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        all_gpu_ops.append((int(start), int(end)))
        launched = launch_time.get(int(correlation), int(start))
        index = step_fast(launched)
        if index is None:
            continue
        label = innermost_call(int(correlation)) or "(outside any range)"
        if index is not None:
            ops_by_step[index].append((int(start), int(end)))
            launches_by_step[index]["memset"] += 1
        ops_by_label[label].append((int(start), int(end)))
        launches_by_label[label]["memset"] += 1

    # ---- per-step table
    busy_intervals = _union_intervals(all_gpu_ops)
    busy_starts = [s for s, _ in busy_intervals]
    busy_ends = [e for _, e in busy_intervals]
    per_step = []
    for index, (start, end, label) in enumerate(steps):
        # Occupancy belongs to the interval where an operation executes,
        # even when its host launch was in an earlier or unselected step.
        first_overlap = bisect.bisect_right(busy_ends, start)
        last_overlap = bisect.bisect_left(busy_starts, end)
        busy = sum(
            min(e, end) - max(s, start)
            for s, e in busy_intervals[first_overlap:last_overlap]
        ) / 1e9
        wall = (end - start) / 1e9
        api = {k: (len(v), sum(v)) for k, v in api_by_step.get(index, {}).items()}
        per_step.append(
            {
                "step": label,
                "wall_s": wall,
                "device_busy_s": busy,
                "device_idle_s": wall - busy,
                "device_busy_fraction": busy / wall if wall else 0.0,
                "launch_api_host_s": _union_seconds(list(launch_api_by_step.get(index, []))),
                "launches": dict(launches_by_step.get(index, {})),
                "blocking_api": {k: {"calls": c, "seconds": s} for k, (c, s) in api.items()},
                "memcpy_bytes": {k: sum(v) for k, v in memcpy_by_step.get(index, {}).items()},
            }
        )
    typical = per_step

    def mean_of(key: str, rows: list[dict[str, Any]]) -> float:
        return statistics.mean(r[key] for r in rows) if rows else 0.0

    step_summary = {
        "steps": n_steps,
        "typical_steps": len(typical),
        "wall_s_mean": mean_of("wall_s", typical),
        "wall_s_median": statistics.median(r["wall_s"] for r in typical) if typical else 0.0,
        "device_busy_s_mean": mean_of("device_busy_s", typical),
        "device_idle_s_mean": mean_of("device_idle_s", typical),
        "device_busy_fraction_mean": mean_of("device_busy_fraction", typical),
        "launch_api_host_s_mean": mean_of("launch_api_host_s", typical),
        "wall_ns_per_point_step": mean_of("wall_s", typical) * 1e9 / args.points if args.points else None,
        "launches_per_step": {
            k: statistics.mean(r["launches"].get(k, 0) for r in typical)
            for k in ("kernel", "memcpy", "memset")
        },
        "blocking_api_per_step": {
            api: {
                "calls": statistics.mean(r["blocking_api"].get(api, {}).get("calls", 0) for r in typical),
                "seconds": statistics.mean(r["blocking_api"].get(api, {}).get("seconds", 0.0) for r in typical),
            }
            for api in BLOCKING_APIS
        },
        "memcpy_bytes_per_step": {
            k: statistics.mean(r["memcpy_bytes"].get(k, 0) for r in typical)
            for k in sorted({k for r in typical for k in r["memcpy_bytes"]})
        },
        "note": "All selected steps are included. GPU busy follows actual execution overlap with each host step, while counts and kernel costs follow the launch step. Device idle and launch API host time overlap and must not be added.",
    }

    # ---- per-label table (innermost attribution), per step
    per_label = {}
    for label in set(list(ops_by_label) + list(api_by_label) + [l for _, _, l in labels]):
        instances = sum(
            1 for s, e, l in labels
            if l == label and any(s < se and e > ss for ss, se, _ in steps)
        )
        host_wall = sum(
            max(0, min(e, se) - max(s, ss))
            for s, e, l in labels if l == label for ss, se, _ in steps
        ) / 1e9
        busy = _union_seconds(list(ops_by_label.get(label, [])))
        per_label[label] = {
            "instances_per_step": instances / n_steps if n_steps else instances,
            "host_wall_s_per_step": host_wall / n_steps if n_steps else host_wall,
            "device_busy_s_per_step": busy / n_steps if n_steps else busy,
            "launches_per_step": {k: v / n_steps for k, v in launches_by_label.get(label, {}).items()},
            "blocking_api_s_per_step": {
                api: {"calls": len(v) / n_steps, "seconds": sum(v) / n_steps}
                for api, v in api_by_label.get(label, {}).items()
            },
            "memcpy_bytes_per_step": {k: sum(v) / n_steps for k, v in memcpy_by_label.get(label, {}).items()},
            "memcpy_device_s_per_step": {k: v / n_steps for k, v in memcpy_time_by_label.get(label, {}).items()},
            "runtime_api_per_step": {k: {"calls": len(v) / n_steps, "host_s": sum(v) / n_steps} for k, v in runtime_api_by_label.get(label, {}).items()},
            "top_kernels": [
                {
                    "name": name,
                    "launches_per_step": count / n_steps,
                    "device_s_per_step": kernel_time_by_label[label][name] / 1e9 / n_steps,
                }
                for name, count in sorted(
                    kernel_by_label.get(label, {}).items(),
                    key=lambda kv: -kernel_time_by_label[label][kv[0]],
                )[:8]
            ],
        }

    # ---- kernel ranking with geometry and occupancy
    kernels = []
    for name, entry in kernel_rows.items():
        geometry, _ = entry["geometry"].most_common(1)[0]
        gx, gy, gz, bx, by, bz, regs, ssm, dsm, lmem = geometry
        block = bx * by * bz
        grid = gx * gy * gz
        occ = occupancy(card, block, regs, ssm + dsm)
        blocks_per_card = occ["blocks_per_sm"] * card["sm_count"]
        waves = grid / blocks_per_card if blocks_per_card else float("inf")
        durations = entry["durations"]
        avg_ns = entry["total_ns"] / entry["instances"]
        med_ns = statistics.median(durations)
        per_step_s = entry["total_ns"] / 1e9 / n_steps if n_steps else entry["total_ns"] / 1e9
        bytes_at_ceiling = avg_ns * 1e-9 * args.bandwidth_tb_s * 1e12
        threads = grid * block
        if med_ns <= 8_000:
            kind = "launch-latency-bound (shorter than a launch)"
        elif grid < card["sm_count"]:
            kind = "cannot fill the card (grid smaller than the SM count); latency-bound"
        elif waves < 1.0:
            kind = "under one wave; tail-and-latency-bound"
        elif occ["theoretical_occupancy"] < 0.25:
            kind = "occupancy-limited (%s)" % occ["limiter"]
        else:
            kind = "resident-state bound: cannot tell memory from latency without counters"
        kernels.append(
            {
                "name": name,
                "instances_per_step": entry["instances"] / n_steps if n_steps else entry["instances"],
                "device_s_per_step": per_step_s,
                "ns_per_point_step": per_step_s * 1e9 / args.points if args.points else None,
                "avg_us": avg_ns / 1e3,
                "median_us": med_ns / 1e3,
                "min_us": min(durations) / 1e3,
                "max_us": max(durations) / 1e3,
                "grid": [gx, gy, gz],
                "block": [bx, by, bz],
                "threads": threads,
                "registers_per_thread": regs,
                "static_smem_bytes": ssm,
                "dynamic_smem_bytes": dsm,
                "local_memory_per_thread": lmem,
                "occupancy": occ,
                "waves": waves,
                "bytes_movable_at_ceiling_MB": bytes_at_ceiling / 1e6,
                "classification": kind,
                "labels": dict(entry["labels"].most_common(3)),
            }
        )
    kernels.sort(key=lambda k: -k["device_s_per_step"])
    total_kernel_s = sum(k["device_s_per_step"] for k in kernels)
    for kernel in kernels:
        kernel["share_of_kernel_time"] = kernel["device_s_per_step"] / total_kernel_s if total_kernel_s else 0.0

    # families: hex dycore vs cupy vs engine, by name
    def family(name: str) -> str:
        if name.startswith("cupy_"):
            return "cupy elementwise/reduction"
        if name.endswith("_v841") or name.endswith("_v841_f32") or name.endswith("_f32") or name.startswith("acoustic_") or name.startswith("regional_") or name.startswith("transport_"):
            return "hex kernels"
        return "engine kernels"

    families: dict[str, dict[str, float]] = collections.defaultdict(lambda: {"launches_per_step": 0.0, "device_s_per_step": 0.0})
    for kernel in kernels:
        row = families[family(kernel["name"])]
        row["launches_per_step"] += kernel["instances_per_step"]
        row["device_s_per_step"] += kernel["device_s_per_step"]

    result = {
        "schema": "gpuwm-hex.nsys-profile-analysis/v1",
        "sqlite": str(args.sqlite),
        "points": args.points,
        "card": card,
        "steps": step_summary,
        "per_step": per_step,
        "per_label": per_label,
        "kernels": kernels,
        "kernel_families": families,
        "total_kernel_s_per_step": total_kernel_s,
        "blocking_api_totals": {k: {"calls": len(v), "seconds": sum(v)} for k, v in api_totals.items()},
        "runtime_api_per_step": {k: {"calls": len(v) / n_steps, "host_s": sum(v) / n_steps} for k, v in runtime_api_totals.items()},
        "cuda_api_tables": [table for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER") if connection.execute("select 1 from sqlite_master where type='table' and name=?", (table,)).fetchone()],
        "limitations": ["NVTX and runtime wrappers perturb the trace; use a separate plain run for throughput.", "Kernel costs and counts are attributed by launch step. GPU busy is the union of every recorded operation that executes within each selected host step, including work launched before that selection.", "Launch API host time overlaps GPU execution and idle time. It is not an additive launch-overhead estimate.", "Occupancy and latency classes are estimates, without hardware-counter measurements."],
    }

    print(f"# card {card['name']}: {card['sm_count']} SMs; {n_steps} steps in the window; typical {len(typical)}")
    print("## per step (typical steps; seconds)")
    for key in ("wall_s_mean", "wall_s_median", "device_busy_s_mean", "device_idle_s_mean", "device_busy_fraction_mean"):
        print(f"  {key:28s} {step_summary[key]:.4f}")
    print("  launches per step", {k: round(v, 1) for k, v in step_summary["launches_per_step"].items()})
    print("  blocking runtime calls per step:")
    for api, row in step_summary["blocking_api_per_step"].items():
        if row["calls"]:
            print(f"    {api:24s} calls {row['calls']:7.1f}  host s {row['seconds']:.4f}")
    print("  memcpy bytes per step", {k: round(v / 1e6, 1) for k, v in step_summary["memcpy_bytes_per_step"].items()}, "(MB)")
    print("## per label (innermost range; per step)")
    print(f"  {'label':26s} {'host_s':>8s} {'dev_busy':>9s} {'kern':>7s} {'memcpy':>7s} {'memset':>7s}  blocking host s")
    for label, row in sorted(per_label.items(), key=lambda kv: -kv[1]["device_busy_s_per_step"])[:45]:
        blocking = ", ".join(f"{k} {v['seconds']:.4f}" for k, v in sorted(row["blocking_api_s_per_step"].items(), key=lambda kv: -kv[1]["seconds"])[:3] if v["seconds"] >= 0.0005)
        launches = row["launches_per_step"]
        print(
            f"  {label:26s} {row['host_wall_s_per_step']:8.4f} {row['device_busy_s_per_step']:9.4f} "
            f"{launches.get('kernel', 0):7.0f} {launches.get('memcpy', 0):7.0f} {launches.get('memset', 0):7.0f}  {blocking}"
        )
    print(f"## kernels by device time per step (top {args.top}); total kernel time/step {total_kernel_s:.4f} s")
    print(f"  {'name':44s} {'s/step':>7s} {'share':>6s} {'n/step':>7s} {'avg_us':>8s} {'grid':>7s} {'blk':>5s} {'regs':>4s} {'occ':>5s} {'waves':>6s}  class")
    for kernel in kernels[: args.top]:
        print(
            f"  {kernel['name'][:44]:44s} {kernel['device_s_per_step']:7.4f} {kernel['share_of_kernel_time']*100:5.1f}% "
            f"{kernel['instances_per_step']:7.1f} {kernel['avg_us']:8.1f} {kernel['grid'][0]:7d} {kernel['block'][0]:5d} "
            f"{kernel['registers_per_thread']:4d} {kernel['occupancy']['theoretical_occupancy']*100:4.0f}% {kernel['waves']:6.2f}  {kernel['classification']}"
        )
    print("## kernel families per step", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in families.items()})
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "device_s_per_step", "ns_per_point_step", "instances_per_step", "avg_us", "share_of_kernel_time"])
            writer.writeheader()
            writer.writerows({k: row[k] for k in writer.fieldnames} for row in kernels)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
