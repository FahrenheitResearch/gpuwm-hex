#!/usr/bin/env python3
"""Non-perturbing process/pool/array snapshots for gpuwm-hex capacity runs.

Importing this module does not import CuPy or touch a CUDA device.  CUDA access
occurs only inside ``cuda_snapshot``/``selftest`` or when the CLI is invoked.
The primary process number is the current process's ``nvidia-smi`` row; the
``cudaMemGetInfo`` value is retained only as a separately labelled device-wide
cross-check because sibling processes can move it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

MIB = 1024**2
SCHEMA = "gpuwm-hex.runtime-memory-snapshot.v1"
SELFTEST_SCHEMA = "gpuwm-hex.runtime-memory-selftest.v1"


class LedgerRefusal(RuntimeError):
    """The instrument cannot support the requested claim."""


@dataclass(frozen=True, slots=True)
class ProcessRow:
    pid: int
    gpu_uuid: str
    used_bytes: int


@dataclass(frozen=True, slots=True)
class ArrayRecord:
    name: str
    owner: str
    lifetime: str
    access: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    nbytes: int
    data_ptr: int
    allocation_ptr: int
    allocation_bytes: int
    alias_group: str | None = None


@dataclass(frozen=True, slots=True)
class LifetimeContract:
    name: str
    phase_start: int
    phase_end: int
    access: str
    allocation_ptr: int
    allocation_bytes: int
    alias_group: str | None = None
    allow_writable_overlap: bool = False

    def validate(self) -> None:
        if self.phase_start < 0 or self.phase_end < self.phase_start:
            raise ValueError(f"{self.name}: invalid phase interval")
        if self.access not in {"read", "write", "readwrite"}:
            raise ValueError(f"{self.name}: access must be read/write/readwrite")
        if self.allocation_ptr < 0 or self.allocation_bytes <= 0:
            raise ValueError(f"{self.name}: invalid allocation range")


@dataclass(frozen=True, slots=True)
class ArrayContract:
    """Expected non-owning array shape/dtype/layout at one binding boundary."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    c_contiguous: bool = True

    def validate_value(self, value: Any) -> None:
        observed_shape = _shape(value)
        observed_dtype = str(value.dtype)
        if observed_shape != self.shape:
            raise LedgerRefusal(
                f"{self.name}: wrong shape {observed_shape}; expected {self.shape}"
            )
        if observed_dtype != self.dtype:
            raise LedgerRefusal(
                f"{self.name}: wrong dtype {observed_dtype}; expected {self.dtype}"
            )
        flags = getattr(value, "flags", None)
        observed_contiguous = bool(getattr(flags, "c_contiguous", False))
        if self.c_contiguous and not observed_contiguous:
            raise LedgerRefusal(f"{self.name}: expected a C-contiguous array")


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """Generation token used to reject a view after its workspace is rebound."""

    name: str
    generation: int

    def validate_current(self, current_generation: int) -> None:
        if self.generation < 0 or current_generation < 0:
            raise ValueError(f"{self.name}: generations must be non-negative")
        if self.generation != current_generation:
            raise LedgerRefusal(
                f"stale workspace view {self.name}: issued generation "
                f"{self.generation}, current generation {current_generation}"
            )


def _run_text(command: Sequence[str], *, timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise LedgerRefusal(f"required executable is absent: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise LedgerRefusal(
            f"command failed ({' '.join(command)}): {stderr or exc.returncode}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise LedgerRefusal(f"command timed out: {' '.join(command)}") from exc
    return completed.stdout


def nvidia_process_rows() -> tuple[ProcessRow, ...]:
    """Return compute-process rows without using device-wide free memory."""

    output = _run_text(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[ProcessRow] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[2] in {"N/A", "[N/A]"}:
            continue
        try:
            rows.append(
                ProcessRow(
                    pid=int(fields[0]),
                    gpu_uuid=fields[1],
                    used_bytes=int(float(fields[2])) * MIB,
                )
            )
        except ValueError as exc:
            raise LedgerRefusal(f"unparseable nvidia-smi process row: {line!r}") from exc
    return tuple(rows)


def process_memory_bytes(pid: int | None = None) -> int | None:
    selected = os.getpid() if pid is None else int(pid)
    values = [row.used_bytes for row in nvidia_process_rows() if row.pid == selected]
    return None if not values else sum(values)


def process_driver_identity() -> dict[str, Any]:
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    devices: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise LedgerRefusal(f"unparseable nvidia-smi device row: {line!r}")
        devices.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_bytes": int(float(fields[3])) * MIB,
                "driver_version": fields[4],
            }
        )
    return {"devices": devices}


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in tuple(value.shape))


def _strides(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "strides", None)
    return None if raw is None else tuple(int(item) for item in tuple(raw))


def array_record(
    name: str,
    value: Any,
    *,
    owner: str,
    lifetime: str,
    access: str,
    alias_group: str | None = None,
) -> ArrayRecord:
    """Describe one CUDA array without copying it to host."""

    if access not in {"read", "write", "readwrite"}:
        raise ValueError("access must be read, write, or readwrite")
    if not hasattr(value, "data") or not hasattr(value.data, "ptr"):
        raise TypeError(f"{name}: value does not expose a CUDA data pointer")
    data_ptr = int(value.data.ptr)
    nbytes = int(value.nbytes)
    memory = getattr(value.data, "mem", None)
    allocation_ptr = int(getattr(memory, "ptr", data_ptr))
    allocation_bytes = int(getattr(memory, "size", nbytes))
    if nbytes < 0 or allocation_bytes <= 0:
        raise ValueError(f"{name}: invalid byte extent")
    return ArrayRecord(
        name=name,
        owner=owner,
        lifetime=lifetime,
        access=access,
        dtype=str(value.dtype),
        shape=_shape(value),
        strides=_strides(value),
        nbytes=nbytes,
        data_ptr=data_ptr,
        allocation_ptr=allocation_ptr,
        allocation_bytes=allocation_bytes,
        alias_group=alias_group,
    )


def _phase_overlap(left: LifetimeContract, right: LifetimeContract) -> bool:
    return not (left.phase_end < right.phase_start or right.phase_end < left.phase_start)


def _allocation_overlap(left: LifetimeContract, right: LifetimeContract) -> bool:
    left_end = left.allocation_ptr + left.allocation_bytes
    right_end = right.allocation_ptr + right.allocation_bytes
    return left.allocation_ptr < right_end and right.allocation_ptr < left_end


def assert_lifetime_contracts(contracts: Iterable[LifetimeContract]) -> None:
    """Refuse concurrent writable aliases unless explicitly authorized.

    The allocation-range test is conservative: two disjoint strided views of
    one allocation are treated as overlapping.  Capacity work should prefer a
    false refusal that asks for a more precise contract over an undetected
    shared write that corrupts the model.
    """

    selected = tuple(contracts)
    for contract in selected:
        contract.validate()
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if not _phase_overlap(left, right) or not _allocation_overlap(left, right):
                continue
            both_read = left.access == "read" and right.access == "read"
            explicitly_allowed = (
                left.allow_writable_overlap
                and right.allow_writable_overlap
                and left.alias_group is not None
                and left.alias_group == right.alias_group
            )
            if both_read or explicitly_allowed:
                continue
            raise LedgerRefusal(
                "concurrent writable allocation overlap: "
                f"{left.name}[{left.phase_start},{left.phase_end}] {left.access} and "
                f"{right.name}[{right.phase_start},{right.phase_end}] {right.access}; "
                "bind both as read-only, separate their lifetimes, or name an explicit "
                "tested writable alias group"
            )


def assert_parking_safe(
    target: LifetimeContract,
    live_contracts: Iterable[LifetimeContract],
    *,
    phase: int,
) -> None:
    """Refuse parking when any other live view shares the allocation.

    Parking a canonical owner while an alias remains live can leave a device
    view pointing at released/reused memory or can silently make host/device
    copies diverge.  This helper is intentionally stricter than the writable
    overlap check: even a read-only live alias blocks parking.
    """

    target.validate()
    if phase < target.phase_start or phase > target.phase_end:
        raise ValueError(f"{target.name}: parking phase is outside its lifetime")
    for other in live_contracts:
        other.validate()
        if other.name == target.name:
            continue
        if not (other.phase_start <= phase <= other.phase_end):
            continue
        if _allocation_overlap(target, other):
            raise LedgerRefusal(
                "parking a shared allocation is unsafe: "
                f"{target.name} overlaps live alias {other.name} at phase {phase}; "
                "retire/rebind every alias first or keep the allocation resident"
            )


def cuda_snapshot(
    label: str,
    *,
    arrays: Mapping[str, tuple[Any, str, str, str, str | None]] | None = None,
) -> dict[str, Any]:
    """Capture current-process, CuPy-pool, and device-wide cross-check values.

    ``arrays`` maps a name to ``(array, owner, lifetime, access, alias_group)``.
    No array is copied or synchronized by this function.  Callers must place a
    deliberate stream synchronization at phase boundaries before comparing
    snapshots.
    """

    import cupy as cp

    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", b"")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    records = []
    for array_name, specification in (arrays or {}).items():
        value, owner, lifetime, access, alias_group = specification
        records.append(
            asdict(
                array_record(
                    array_name,
                    value,
                    owner=owner,
                    lifetime=lifetime,
                    access=access,
                    alias_group=alias_group,
                )
            )
        )
    return {
        "schema": SCHEMA,
        "label": label,
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "process_nvidia_smi_bytes": process_memory_bytes(),
        "cupy_pool": {
            "used_bytes": int(pool.used_bytes()),
            "total_bytes": int(pool.total_bytes()),
            "free_bytes_inside_pool": int(pool.total_bytes() - pool.used_bytes()),
            "n_free_blocks": int(pool.n_free_blocks()),
        },
        "cuda_mem_get_info_device_wide_crosscheck": {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "warning": "device-wide; sibling processes can move this value",
        },
        "device": {
            "id": int(device.id),
            "name": str(name),
            "compute_capability": str(device.compute_capability),
        },
        "arrays": records,
    }


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def allocation_and_view_selftest(allocation_mib: int) -> dict[str, Any]:
    if allocation_mib <= 0:
        raise ValueError("allocation_mib must be positive")
    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    before = cuda_snapshot("before-known-allocation")
    count = allocation_mib * MIB // 4
    allocation = cp.empty((count,), dtype=cp.float32)
    allocation.fill(cp.float32(1.0))
    view = allocation[::2]
    cp.cuda.Stream.null.synchronize()
    after = cuda_snapshot(
        "after-known-allocation-and-view",
        arrays={
            "known_allocation": (
                allocation,
                "runtime_ledger.selftest",
                "selftest",
                "readwrite",
                "known-allocation",
            ),
            "known_view": (
                view,
                "runtime_ledger.selftest",
                "selftest",
                "read",
                "known-allocation",
            ),
        },
    )
    records = {record["name"]: record for record in after["arrays"]}
    if records["known_allocation"]["allocation_ptr"] != records["known_view"]["allocation_ptr"]:
        raise LedgerRefusal("view selftest failed: view does not report the owner's allocation")
    requested = int(allocation.nbytes)
    pool_delta = (
        after["cupy_pool"]["used_bytes"] - before["cupy_pool"]["used_bytes"]
    )
    if pool_delta < requested:
        raise LedgerRefusal(
            f"known allocation requested {requested} bytes but pool live rose only {pool_delta}"
        )
    return {
        "schema": SELFTEST_SCHEMA,
        "requested_bytes": requested,
        "view_nbytes": int(view.nbytes),
        "pool_used_delta_bytes": int(pool_delta),
        "process_nvidia_smi_delta_bytes": _optional_delta(
            before["process_nvidia_smi_bytes"], after["process_nvidia_smi_bytes"]
        ),
        "view_added_allocation": False,
        "before": before,
        "after": after,
        "status": "PASS",
        "claim_boundary": (
            "This validates known allocation and view accounting only; it does not "
            "validate kernel local-memory reservation or sibling contamination."
        ),
    }


def _optional_delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return int(after - before)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="capture one CUDA process snapshot")
    snapshot.add_argument("--label", required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    selftest = subparsers.add_parser(
        "selftest-allocation", help="validate known allocation and view accounting"
    )
    selftest.add_argument("--allocation-mib", type=int, default=64)
    selftest.add_argument("--output", type=Path, required=True)

    identity = subparsers.add_parser(
        "identity", help="record nvidia-smi device/driver identity without CuPy"
    )
    identity.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "snapshot":
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
            payload = cuda_snapshot(arguments.label)
        elif arguments.command == "selftest-allocation":
            payload = allocation_and_view_selftest(arguments.allocation_mib)
        elif arguments.command == "identity":
            payload = {
                "schema": "gpuwm-hex.device-driver-identity.v1",
                "timestamp_ns": time.time_ns(),
                **process_driver_identity(),
            }
        else:  # pragma: no cover - argparse owns this branch
            raise AssertionError(arguments.command)
    except (LedgerRefusal, ValueError, RuntimeError) as exc:
        raise SystemExit(f"runtime ledger refused: {exc}") from exc
    _write_json(arguments.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
