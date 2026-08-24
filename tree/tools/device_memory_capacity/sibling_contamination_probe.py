#!/usr/bin/env python3
"""Positive control proving why device-wide free memory is not a process ledger.

The parent holds a tiny CUDA context.  A sibling process then allocates a known
buffer on the same device.  A valid control shows the parent process's own
``nvidia-smi`` row staying stable while device-wide free memory falls and a new
child row appears.  Run only on an isolated card: this intentionally creates a
second CUDA process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

MIB = 1024**2
SCHEMA = "gpuwm-hex.sibling-contamination-probe.v1"


class ContaminationRefusal(RuntimeError):
    """The sibling-process control could not be interpreted."""


def _rows() -> dict[int, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ContaminationRefusal(f"nvidia-smi query failed: {exc}") from exc
    rows: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[1] in {"N/A", "[N/A]"}:
            continue
        try:
            pid = int(fields[0])
            used = int(float(fields[1])) * MIB
        except ValueError as exc:
            raise ContaminationRefusal(f"unparseable process row: {line!r}") from exc
        rows[pid] = rows.get(pid, 0) + used
    return rows


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def worker(allocation_mib: int, hold_seconds: float) -> int:
    if allocation_mib <= 0 or hold_seconds <= 0.0:
        raise ValueError("allocation_mib and hold_seconds must be positive")
    import cupy as cp

    count = allocation_mib * MIB // 4
    value = cp.empty((count,), dtype=cp.float32)
    value.fill(cp.float32(1.0))
    cp.cuda.Stream.null.synchronize()
    print(
        json.dumps(
            {
                "status": "READY",
                "pid": os.getpid(),
                "allocation_bytes": int(value.nbytes),
                "device": int(cp.cuda.Device().id),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    time.sleep(hold_seconds)
    return 0


def _read_ready(process: subprocess.Popen[str], timeout_seconds: float) -> dict[str, Any]:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        readable, _, _ = select.select([process.stdout], [], [], min(0.25, remaining))
        if not readable:
            if process.poll() is not None:
                stderr = "" if process.stderr is None else process.stderr.read()
                raise ContaminationRefusal(
                    f"sibling exited before READY: {(stderr or '').strip()}"
                )
            continue
        line = process.stdout.readline()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContaminationRefusal(f"sibling emitted non-JSON READY line: {line!r}") from exc
        if payload.get("status") != "READY":
            raise ContaminationRefusal(f"unexpected sibling payload: {payload!r}")
        return payload
    raise ContaminationRefusal("timed out waiting for sibling CUDA allocation")


def run_control(allocation_mib: int, timeout_seconds: float) -> dict[str, Any]:
    if allocation_mib <= 0 or timeout_seconds <= 1.0:
        raise ValueError("allocation_mib must be positive and timeout_seconds > 1")
    import cupy as cp

    parent_marker = cp.zeros((1,), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    parent_pid = os.getpid()
    before_rows = _rows()
    before_free, before_total = cp.cuda.runtime.memGetInfo()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--allocation-mib",
        str(allocation_mib),
        "--hold-seconds",
        str(timeout_seconds),
    ]
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = _read_ready(child, min(timeout_seconds / 2.0, 30.0))
        cp.cuda.Stream.null.synchronize()
        during_rows = _rows()
        during_free, during_total = cp.cuda.runtime.memGetInfo()
    finally:
        child.terminate()
        try:
            child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5.0)

    child_pid = int(ready["pid"])
    allocation_bytes = int(ready["allocation_bytes"])
    parent_before = before_rows.get(parent_pid)
    parent_during = during_rows.get(parent_pid)
    child_during = during_rows.get(child_pid)
    parent_delta = (
        None
        if parent_before is None or parent_during is None
        else parent_during - parent_before
    )
    device_free_drop = int(before_free) - int(during_free)
    checks = {
        "child_process_row_present": child_during is not None,
        "child_row_covers_known_allocation": (
            child_during is not None and child_during >= allocation_bytes
        ),
        "parent_row_stable_within_16_mib": (
            parent_delta is not None and abs(parent_delta) <= 16 * MIB
        ),
        "device_wide_free_memory_moved": device_free_drop >= allocation_bytes,
        "device_total_stable": int(before_total) == int(during_total),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema": SCHEMA,
        "status": status,
        "allocation_mib_requested": allocation_mib,
        "allocation_bytes": allocation_bytes,
        "parent_pid": parent_pid,
        "child_pid": child_pid,
        "parent_process_before_bytes": parent_before,
        "parent_process_during_bytes": parent_during,
        "parent_process_delta_bytes": parent_delta,
        "child_process_during_bytes": child_during,
        "device_wide_free_before_bytes": int(before_free),
        "device_wide_free_during_bytes": int(during_free),
        "device_wide_free_drop_bytes": device_free_drop,
        "checks": checks,
        "interpretation": (
            "PASS proves a sibling allocation moves device-wide free memory without "
            "being charged to the parent's per-process row."
        ),
        "parent_marker_bytes": int(parent_marker.nbytes),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--allocation-mib", type=int, required=True)
    worker_parser.add_argument("--hold-seconds", type=float, required=True)

    control = subparsers.add_parser("control")
    control.add_argument("--allocation-mib", type=int, default=256)
    control.add_argument("--timeout-seconds", type=float, default=60.0)
    control.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "worker":
            return worker(arguments.allocation_mib, arguments.hold_seconds)
        if arguments.command == "control":
            payload = run_control(arguments.allocation_mib, arguments.timeout_seconds)
            _write(arguments.output, payload)
            return 0 if payload["status"] == "PASS" else 1
        raise AssertionError(arguments.command)  # pragma: no cover
    except (ContaminationRefusal, ValueError, RuntimeError) as exc:
        raise SystemExit(f"sibling contamination probe refused: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
