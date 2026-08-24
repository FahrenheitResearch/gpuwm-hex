#!/usr/bin/env python3
"""Validate CUDA local-frame accounting with fresh-process launch-order controls.

CUDA keeps one per-context local-memory backing store sized for the largest
launched demand.  A later kernel can therefore show a zero increment even when
its own ``local_size_bytes`` is non-zero.  This tool launches zero-local and
nonzero-local controls in fresh worker processes and records both the kernel
attribute and per-process ``nvidia-smi`` increments in both orders.

For real gpuwm kernels, import ``measure_launch`` in the call site that owns the
actual ``RawKernel``.  Do not discover kernels with post-run garbage-collector
scans: rebound RawKernel handles can already be unreachable by then.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

MIB = 1024**2
SCHEMA = "gpuwm-hex.kernel-reservation-probe.v1"
MATRIX_SCHEMA = "gpuwm-hex.kernel-reservation-matrix.v1"


class ReservationRefusal(RuntimeError):
    """A kernel reservation measurement is incomplete or contaminated."""


def _process_bytes() -> int | None:
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
        raise ReservationRefusal(f"nvidia-smi process query failed: {exc}") from exc
    pid = os.getpid()
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or fields[1] in {"N/A", "[N/A]"}:
            continue
        try:
            row_pid = int(fields[0])
            used = int(float(fields[1])) * MIB
        except ValueError as exc:
            raise ReservationRefusal(f"unparseable process row: {line!r}") from exc
        if row_pid == pid:
            total += used
            found = True
    return total if found else None


def _device_wide_free_total(cp: Any) -> tuple[int, int]:
    free, total = cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def _local_size_bytes(kernel: Any) -> int:
    attributes = dict(kernel.attributes)
    if "local_size_bytes" not in attributes:
        raise ReservationRefusal("RawKernel attributes omit local_size_bytes")
    return int(attributes["local_size_bytes"])


def measure_launch(
    kernel: Any,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    arguments: tuple[Any, ...],
    *,
    label: str,
    synchronize: Callable[[], None],
    cp: Any,
) -> dict[str, Any]:
    """Measure one actual launch in the current process.

    The caller controls process freshness and launch order.  The return value
    labels ``cudaMemGetInfo`` as device-wide and uses the process row as the
    primary reservation increment.
    """

    synchronize()
    before_process = _process_bytes()
    before_free, before_total = _device_wide_free_total(cp)
    started = time.perf_counter()
    kernel(grid, block, arguments)
    synchronize()
    elapsed = time.perf_counter() - started
    after_process = _process_bytes()
    after_free, after_total = _device_wide_free_total(cp)
    return {
        "label": label,
        "local_size_bytes": _local_size_bytes(kernel),
        "elapsed_seconds": elapsed,
        "process_before_bytes": before_process,
        "process_after_bytes": after_process,
        "process_increment_bytes": (
            None
            if before_process is None or after_process is None
            else after_process - before_process
        ),
        "device_wide_before_free_bytes": before_free,
        "device_wide_after_free_bytes": after_free,
        "device_wide_total_bytes": after_total,
        "device_wide_consumed_increment_bytes": before_free - after_free,
        "device_total_changed": before_total != after_total,
    }


def _control_source(local_floats: int) -> str:
    if local_floats <= 0:
        raise ValueError("local_floats must be positive")
    return f'''\nextern "C" __global__ void zero_local(float *out) {{\n    const int i = blockDim.x * blockIdx.x + threadIdx.x;\n    if (i == 0) out[0] = 1.0f;\n}}\n\nextern "C" __global__ void nonzero_local(float *out, int salt) {{\n    volatile float scratch[{local_floats}];\n    const int slot = (threadIdx.x + salt) % {local_floats};\n    scratch[slot] = (float)(threadIdx.x + 1);\n    if (threadIdx.x == 0) out[0] = scratch[slot];\n}}\n'''


def run_worker(order: Sequence[str], *, local_floats: int) -> dict[str, Any]:
    if not order or any(name not in {"zero", "nonzero"} for name in order):
        raise ValueError("order must contain zero and/or nonzero")
    import cupy as cp

    module = cp.RawModule(
        code=_control_source(local_floats),
        options=("--std=c++14",),
        name_expressions=("zero_local", "nonzero_local"),
    )
    zero = module.get_function("zero_local")
    nonzero = module.get_function("nonzero_local")
    output = cp.zeros((1,), dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    baseline = {
        "process_bytes": _process_bytes(),
        "device_wide_free_total": _device_wide_free_total(cp),
    }
    launches: list[dict[str, Any]] = []
    for index, name in enumerate(order):
        if name == "zero":
            launch = measure_launch(
                zero,
                (1,),
                (256,),
                (output,),
                label=f"{index + 1}:zero",
                synchronize=cp.cuda.Stream.null.synchronize,
                cp=cp,
            )
        else:
            launch = measure_launch(
                nonzero,
                (1,),
                (256,),
                (output, cp.int32(index + 1)),
                label=f"{index + 1}:nonzero",
                synchronize=cp.cuda.Stream.null.synchronize,
                cp=cp,
            )
        launches.append(launch)
    return {
        "schema": SCHEMA,
        "pid": os.getpid(),
        "order": list(order),
        "local_floats_requested": local_floats,
        "control_attributes": {
            "zero_local_size_bytes": _local_size_bytes(zero),
            "nonzero_local_size_bytes": _local_size_bytes(nonzero),
        },
        "baseline": baseline,
        "launches": launches,
        "status": "MEASURED",
        "claim_boundary": (
            "These are synthetic controls. Real-kernel frame and reservation numbers "
            "require measure_launch at the real call site in a fresh controlled process."
        ),
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_matrix(output_dir: Path, *, local_floats: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = {
        "zero-only": ("zero",),
        "nonzero-only": ("nonzero",),
        "zero-then-nonzero": ("zero", "nonzero"),
        "nonzero-then-zero": ("nonzero", "zero"),
        "nonzero-twice": ("nonzero", "nonzero"),
    }
    receipts: dict[str, Any] = {}
    script = Path(__file__).resolve()
    for name, order in orders.items():
        path = output_dir / f"{name}.json"
        command = [
            sys.executable,
            str(script),
            "worker",
            "--order",
            *order,
            "--local-floats",
            str(local_floats),
            "--output",
            str(path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise ReservationRefusal(
                f"fresh worker {name} failed: {(completed.stderr or completed.stdout).strip()}"
            )
        receipts[name] = json.loads(path.read_text(encoding="utf-8"))
    matrix = {
        "schema": MATRIX_SCHEMA,
        "local_floats_requested": local_floats,
        "fresh_process_per_order": True,
        "receipts": receipts,
        "interpretation": {
            "zero_control": "zero-local launch should reserve no material local backing store",
            "positive_control": "nonzero-local standalone launch should show a positive reservation",
            "repeat_control": "the second identical nonzero launch should show no material increment",
            "order_control": (
                "zero before nonzero should not consume the later reservation; zero after "
                "nonzero should show no new local-store increment"
            ),
        },
        "automatic_verdict": "NOT COMPUTED: inspect card-specific MiB rounding and noise",
    }
    _write(output_dir / "matrix.json", matrix)
    return matrix


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--order", nargs="+", required=True)
    worker.add_argument("--local-floats", type=int, default=512)
    worker.add_argument("--output", type=Path, required=True)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--local-floats", type=int, default=512)
    matrix.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "worker":
            payload = run_worker(arguments.order, local_floats=arguments.local_floats)
            _write(arguments.output, payload)
        elif arguments.command == "matrix":
            run_matrix(arguments.output_dir, local_floats=arguments.local_floats)
        else:  # pragma: no cover
            raise AssertionError(arguments.command)
    except (ReservationRefusal, ValueError, RuntimeError) as exc:
        raise SystemExit(f"kernel reservation probe refused: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
