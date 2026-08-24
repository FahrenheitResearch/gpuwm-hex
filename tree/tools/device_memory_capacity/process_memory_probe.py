#!/usr/bin/env python3
"""Run a command while sampling its per-process ``nvidia-smi`` memory rows.

The sampler follows Linux child PIDs and records a sum plus per-PID peaks.  It
never substitutes ``cudaMemGetInfo`` for process memory.  Short phase peaks must
also be captured by in-process CuPy-pool snapshots; this wrapper supplies the
whole-process anchor and contamination-resistant peak required by the capacity
ledger.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

MIB = 1024**2
SCHEMA = "gpuwm-hex.process-memory-probe.v1"


class ProbeRefusal(RuntimeError):
    """The process probe cannot produce a trustworthy receipt."""


def _nvidia_rows() -> dict[int, list[dict[str, Any]]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except FileNotFoundError as exc:
        raise ProbeRefusal("nvidia-smi is absent") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ProbeRefusal(f"nvidia-smi process query failed: {exc}") from exc
    result: dict[int, list[dict[str, Any]]] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[2] in {"N/A", "[N/A]"}:
            continue
        try:
            pid = int(fields[0])
            used = int(float(fields[2])) * MIB
        except ValueError as exc:
            raise ProbeRefusal(f"unparseable nvidia-smi process row: {line!r}") from exc
        result.setdefault(pid, []).append(
            {"gpu_uuid": fields[1], "used_bytes": used}
        )
    return result


def _children(pid: int) -> tuple[int, ...]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = path.read_text(encoding="ascii")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ()
    values: list[int] = []
    for token in text.split():
        try:
            values.append(int(token))
        except ValueError:
            continue
    return tuple(values)


def process_tree(root_pid: int) -> set[int]:
    found = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in _children(parent):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _device_identity() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ProbeRefusal(f"nvidia-smi device query failed: {exc}") from exc
    devices: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ProbeRefusal(f"unparseable nvidia-smi device row: {line!r}")
        devices.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_bytes": int(float(fields[3])) * MIB,
                "driver_version": fields[4],
            }
        )
    return devices


def _parse_env(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ProbeRefusal(f"--env requires NAME=VALUE, got {value!r}")
        name, setting = value.split("=", 1)
        if not name:
            raise ProbeRefusal("--env variable name is empty")
        result[name] = setting
    return result


def run_probe(
    command: Sequence[str],
    *,
    output: Path,
    stdout_path: Path,
    stderr_path: Path,
    interval_ms: float,
    environment_overrides: Mapping[str, str],
    max_samples: int,
) -> int:
    if not command:
        raise ProbeRefusal("a command is required after --")
    if not (10.0 <= interval_ms <= 10_000.0):
        raise ProbeRefusal("interval_ms must be between 10 and 10000")
    if max_samples <= 0:
        raise ProbeRefusal("max_samples must be positive")

    output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(environment_overrides)
    devices = _device_identity()

    start_wall_ns = time.time_ns()
    start_monotonic = time.monotonic()
    samples: list[dict[str, Any]] = []
    per_pid_peak: dict[int, int] = {}
    process_peak = 0
    observed_row = False
    truncated = False

    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(command),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=environment,
        )
        root_pid = process.pid
        known_tree = {root_pid}
        while True:
            now = time.monotonic()
            known_tree |= process_tree(root_pid)
            rows = _nvidia_rows()
            selected: dict[str, list[dict[str, Any]]] = {}
            total = 0
            for pid in sorted(known_tree):
                entries = rows.get(pid, [])
                if not entries:
                    continue
                observed_row = True
                selected[str(pid)] = entries
                used = sum(int(entry["used_bytes"]) for entry in entries)
                total += used
                per_pid_peak[pid] = max(per_pid_peak.get(pid, 0), used)
            process_peak = max(process_peak, total)
            if len(samples) < max_samples:
                samples.append(
                    {
                        "elapsed_seconds": now - start_monotonic,
                        "tracked_pids": sorted(known_tree),
                        "rows": selected,
                        "sum_used_bytes": total,
                    }
                )
            else:
                truncated = True
            return_code = process.poll()
            if return_code is not None:
                break
            time.sleep(interval_ms / 1000.0)
        end_wall_ns = time.time_ns()

    receipt = {
        "schema": SCHEMA,
        "command": list(command),
        "command_shell_escaped": shlex.join(command),
        "root_pid": root_pid,
        "return_code": return_code,
        "success": return_code == 0,
        "start_wall_ns": start_wall_ns,
        "end_wall_ns": end_wall_ns,
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "sampling_interval_ms": interval_ms,
        "sample_count": len(samples),
        "samples_truncated": truncated,
        "observed_nvidia_process_row": observed_row,
        "process_peak_bytes": process_peak if observed_row else None,
        "per_pid_peak_bytes": {str(pid): value for pid, value in sorted(per_pid_peak.items())},
        "device_identity": devices,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "environment_overrides": dict(sorted(environment_overrides.items())),
        "claim_boundary": (
            "This is a sampled per-process anchor. In-process phase/pool snapshots are "
            "required to attribute short transient peaks."
        ),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(return_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--interval-ms", type=float, default=50.0)
    parser.add_argument("--max-samples", type=int, default=20_000)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    try:
        return run_probe(
            command,
            output=arguments.output,
            stdout_path=arguments.stdout,
            stderr_path=arguments.stderr,
            interval_ms=arguments.interval_ms,
            environment_overrides=_parse_env(arguments.env),
            max_samples=arguments.max_samples,
        )
    except ProbeRefusal as exc:
        raise SystemExit(f"process memory probe refused: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
