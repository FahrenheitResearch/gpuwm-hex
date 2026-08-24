#!/usr/bin/env python3
"""Build the sm_120 MPAS/gpuwm FTZ binding from fresh CUDA compiles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mpas_port.cuda_backend import KernelCache, require_cuda  # noqa: E402
from mpas_port.cuda_ftz import (  # noqa: E402
    build_mpas_ftz_binding,
    build_mpas_ftz_binding_v841,
    production_translation_units,
    run_guarded_kernel_subnormal_audit,
    run_normalized_fallback_performance_control,
    run_scalar_transport_subnormal_deck,
    validate_compile_manifest_relation,
    validate_v841_compile_manifest_relation,
    v841_reached_translation_units,
)


DEFAULT_OUTPUT = ROOT / "receipts" / "cuda-ftz-sm120"
DEFAULT_CACHE = ROOT / "work" / "cupy-cache-ftz-contract-fresh"
DEFAULT_V841_OUTPUT = ROOT / "receipts" / "cuda-ftz-sm120-v841"
DEFAULT_V841_CACHE = ROOT / "work" / "cupy-cache-ftz-contract-v841-fresh"
V841_CAPSULE_SCHEMA = "mpas-port.cuda-ftz-v841-execution-capsule/v1"

_GPUWM_SOURCE_INPUTS = (
    "tools/ftz_receipt/probe.py",
    "tools/ftz_receipt/route_inventory.py",
    "gpuwm/core/kernels/ftz_probe.cu",
    "gpuwm/certify/compile_platform.py",
)
_IGNORED_AUTHORITY_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_WDDM_SYSTEM_SENTINEL_PID = 4
_NVIDIA_SMI_INSUFFICIENT_PERMISSIONS = "[Insufficient Permissions]"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    """Hash one stable regular file, refusing links and mid-read mutation."""

    selected = path.expanduser()
    if selected.is_symlink():
        raise RuntimeError(f"authority input must not be a symlink: {selected}")
    if not selected.is_file():
        raise RuntimeError(f"authority input is not a regular file: {selected}")
    before = selected.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    size = 0
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    after = selected.stat(follow_symlinks=False)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or size != after.st_size:
        raise RuntimeError(f"authority input changed while hashing: {selected}")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _tree_inventory(
    root: Path,
    *,
    ignored_parts: frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    selected = root.expanduser().resolve()
    if root.expanduser().is_symlink() or not selected.is_dir():
        raise RuntimeError(f"authority inventory root is not a real directory: {root}")
    result: dict[str, dict[str, Any]] = {}
    for candidate in sorted(selected.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(selected)
        if candidate.is_symlink():
            raise RuntimeError(f"authority inventory contains a symlink: {candidate}")
        if any(part in ignored_parts for part in relative.parts):
            continue
        if candidate.is_file():
            result[relative.as_posix()] = _file_record(candidate)
    return result


def _inventory_record(
    root: Path,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = {str(key): dict(value) for key, value in sorted(files.items())}
    return {
        "root": str(root.expanduser().resolve()),
        "file_count": len(normalized),
        "files": normalized,
        "files_sha256": _canonical_sha256(normalized),
    }


def _mpas_authority_inventory() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative_root in ("src", "tests", "tools"):
        scope = ROOT / relative_root
        for relative, record in _tree_inventory(
            scope, ignored_parts=_IGNORED_AUTHORITY_PARTS
        ).items():
            files[f"{relative_root}/{relative}"] = record
    return _inventory_record(ROOT, files)


def _git_head(root: Path) -> str:
    selected = root.expanduser().resolve()
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={selected.as_posix()}",
                "-C",
                str(selected),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot resolve gpuwm HEAD at {selected}: {error}"
        ) from error
    head = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise RuntimeError(f"gpuwm HEAD is not a full commit id: {head!r}")
    return head


def _gpuwm_source_inventory(root: Path) -> dict[str, Any]:
    if root.expanduser().is_symlink():
        raise RuntimeError(f"gpuwm authority root must not be a symlink: {root}")
    selected = root.expanduser().resolve()
    files = {
        relative: _file_record(selected / relative) for relative in _GPUWM_SOURCE_INPUTS
    }
    result = _inventory_record(selected, files)
    result["git_head"] = _git_head(selected)
    return result


def _input_snapshot(gpuwm_root: Path, gpuwm_receipt: Path) -> dict[str, Any]:
    receipt = gpuwm_receipt.expanduser().resolve()
    return {
        "mpas_authority": _mpas_authority_inventory(),
        "gpuwm_sources": _gpuwm_source_inventory(gpuwm_root),
        "gpuwm_receipt": _inventory_record(receipt, _tree_inventory(receipt)),
    }


def _assert_same_snapshot(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    if dict(before) != dict(after):
        raise RuntimeError(
            "FTZ authority source/input drifted between the pre- and post-run snapshots: "
            f"{_canonical_sha256(before)} != {_canonical_sha256(after)}"
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_v841_paths(
    *,
    output_root: Path,
    cache_root: Path,
    gpuwm_root: Path,
    gpuwm_receipt: Path,
) -> tuple[Path, Path]:
    if output_root.expanduser().is_symlink():
        raise RuntimeError(f"v8.4.1 output root must not be a symlink: {output_root}")
    if cache_root.expanduser().is_symlink():
        raise RuntimeError(f"v8.4.1 compile cache must not be a symlink: {cache_root}")
    output = output_root.expanduser().resolve()
    cache = cache_root.expanduser().resolve()
    protected = (
        ROOT / "src",
        ROOT / "tests",
        ROOT / "tools",
        gpuwm_root.expanduser().resolve(),
        gpuwm_receipt.expanduser().resolve(),
    )
    if _paths_overlap(output, cache):
        raise RuntimeError("v8.4.1 output root and compile cache must not overlap")
    for target_name, target in (("output root", output), ("compile cache", cache)):
        for source in protected:
            resolved_source = source.resolve()
            if _paths_overlap(target, resolved_source):
                raise RuntimeError(
                    f"v8.4.1 {target_name} overlaps protected authority input "
                    f"{resolved_source}"
                )
    return output, cache


def _require_absent_directory(path: Path, *, purpose: str) -> Path:
    selected = path.expanduser().resolve()
    if selected.exists() or selected.is_symlink():
        raise RuntimeError(f"{purpose} must be absent before the run: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    try:
        selected.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise RuntimeError(
            f"{purpose} lost its exclusive-create race: {selected}"
        ) from error
    return selected


def _write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _copy_receipt_exclusive(
    source_root: Path,
    destination_root: Path,
    expected_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    source = source_root.expanduser().resolve()
    destination = destination_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(
            f"refusing to overlay an existing copied gpuwm probe: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in sorted(expected_inventory.items()):
        source_file = source / relative
        target_file = destination / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with (
            source_file.open("rb") as input_stream,
            target_file.open("xb") as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        actual = _file_record(target_file)
        if actual != dict(expected):
            raise RuntimeError(
                f"copied gpuwm receipt artifact changed bytes: {relative}"
            )
    copied = _tree_inventory(destination)
    if copied != {key: dict(value) for key, value in expected_inventory.items()}:
        raise RuntimeError("copied gpuwm receipt tree differs from its authority input")
    return copied


@contextmanager
def _cuda_device_lease(device_id: int) -> Iterator[dict[str, Any]]:
    """Hold a process-lifetime advisory lock for participating MPAS launchers."""

    lock_root = Path(tempfile.gettempdir()) / "mpas-port-cuda-device-leases"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"device-{device_id}.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    mechanism: str
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"CUDA device {device_id} is held by another MPAS launcher"
                ) from error
            mechanism = "windows-msvcrt-nonblocking-byte-lock"
        else:  # pragma: no cover - the certification host is Windows
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(
                    f"CUDA device {device_id} is held by another MPAS launcher"
                ) from error
            mechanism = "posix-flock-exclusive-nonblocking"
        yield {
            "device_id": device_id,
            "lock_path": str(lock_path.resolve()),
            "mechanism": mechanism,
            "held_for_entire_measurement": True,
        }
    finally:
        try:
            if os.name == "nt" and "mechanism" in locals():
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            elif "mechanism" in locals():  # pragma: no cover - Windows authority
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _nvidia_smi_executable() -> Path:
    resolved = shutil.which("nvidia-smi")
    if resolved is None:
        raise RuntimeError("nvidia-smi is required for fail-closed GPU exclusivity")
    executable = Path(resolved).resolve()
    if not executable.is_file():
        raise RuntimeError(f"nvidia-smi is not a regular file: {executable}")
    return executable


def _run_nvidia_smi(executable: Path, fields: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            [
                str(executable),
                f"--query-{'gpu' if fields[0] == 'index' else 'compute-apps'}="
                + ",".join(fields),
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"nvidia-smi GPU authority query failed: {error}") from error
    return completed.stdout


def _csv_rows(payload: str) -> list[list[str]]:
    return [
        [value.strip() for value in row]
        for row in csv.reader(io.StringIO(payload))
        if row and any(value.strip() for value in row)
    ]


def _gpu_inventory(executable: Path) -> list[dict[str, Any]]:
    fields = (
        "index",
        "uuid",
        "name",
        "pci.bus_id",
        "compute_cap",
        "driver_version",
        "compute_mode",
    )
    rows = _csv_rows(_run_nvidia_smi(executable, fields))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU identities")
    result: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(fields):
            raise RuntimeError(f"nvidia-smi GPU identity row is malformed: {row!r}")
        try:
            index = int(row[0])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi GPU index is invalid: {row[0]!r}"
            ) from error
        if _GPU_UUID_RE.fullmatch(row[1]) is None:
            raise RuntimeError(f"nvidia-smi GPU UUID is invalid: {row[1]!r}")
        if re.fullmatch(r"\d+\.\d+", row[4]) is None:
            raise RuntimeError(
                f"nvidia-smi compute capability is unavailable: {row[4]!r}"
            )
        if any(value in {"", "N/A", "[Not Supported]"} for value in row[2:]):
            raise RuntimeError(f"nvidia-smi GPU identity is incomplete: {row!r}")
        result.append(
            {
                "index": index,
                "uuid": row[1],
                "name": row[2],
                "pci_bus_id": row[3].upper(),
                "compute_capability": row[4],
                "nvidia_driver_version": row[5],
                "compute_mode": row[6],
            }
        )
    result.sort(key=lambda item: int(item["index"]))
    if len({item["index"] for item in result}) != len(result):
        raise RuntimeError("nvidia-smi returned duplicate GPU indices")
    return result


def _compute_processes(executable: Path) -> list[dict[str, Any]]:
    fields = ("gpu_uuid", "pid", "process_name")
    rows = _csv_rows(_run_nvidia_smi(executable, fields))
    if len(rows) == 1 and rows[0][0].lower().startswith("no running processes"):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != len(fields):
            raise RuntimeError(
                "nvidia-smi compute-process inventory is unavailable or malformed: "
                f"{row!r}"
            )
        if _GPU_UUID_RE.fullmatch(row[0]) is None:
            raise RuntimeError(f"nvidia-smi process GPU UUID is invalid: {row[0]!r}")
        try:
            pid = int(row[1])
        except ValueError as error:
            raise RuntimeError(
                f"nvidia-smi compute PID is invalid: {row[1]!r}"
            ) from error
        if not row[2]:
            raise RuntimeError(f"nvidia-smi compute process row is incomplete: {row!r}")
        if (
            sys.platform == "win32"
            and pid == _WDDM_SYSTEM_SENTINEL_PID
            and row[2] == _NVIDIA_SMI_INSUFFICIENT_PERMISSIONS
        ):
            # On Windows/WDDM, nvidia-smi can intermittently include or omit
            # this inaccessible System-process sentinel even though PID 4 did
            # not start or exit. Normalize only that exact telemetry row; all
            # other PID/UUID/name additions and mutations remain fail-closed.
            continue
        result.append(
            {
                "gpu_uuid": row[0],
                "pid": pid,
                "process_name": row[2],
            }
        )
    return sorted(result, key=lambda item: (str(item["gpu_uuid"]), int(item["pid"])))


def _capture_gpu_checkpoint(
    executable: Path,
    *,
    device_id: int,
    label: str,
    expected_gpus: list[dict[str, Any]] | None = None,
    expected_external_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gpus = _gpu_inventory(executable)
    if expected_gpus is not None and gpus != expected_gpus:
        raise RuntimeError(f"GPU identity drifted at checkpoint {label!r}")
    selected = next((row for row in gpus if row["index"] == device_id), None)
    if selected is None:
        raise RuntimeError(f"nvidia-smi has no CUDA device index {device_id}")
    processes = _compute_processes(executable)
    known_uuids = {str(row["uuid"]) for row in gpus}
    if any(str(row["gpu_uuid"]) not in known_uuids for row in processes):
        raise RuntimeError("nvidia-smi reported a process on an unknown GPU UUID")
    own_processes = [row for row in processes if row["pid"] == os.getpid()]
    external_processes = [row for row in processes if row["pid"] != os.getpid()]
    if (
        expected_external_processes is not None
        and external_processes != expected_external_processes
    ):
        raise RuntimeError(
            "nvidia-smi process inventory added, removed, or drifted from its "
            f"pre-run baseline at checkpoint {label!r}: "
            f"{external_processes!r} != {expected_external_processes!r}"
        )
    return {
        "label": label,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_inventory": gpus,
        "selected_gpu": selected,
        "all_reported_processes": processes,
        "external_process_baseline": external_processes,
        "own_reported_processes": own_processes,
        "own_pid": os.getpid(),
        "new_or_drifted_external_process_count": 0,
        "physical_gpu_exclusivity_claim": False,
    }


class _GpuExclusivityMonitor:
    """Sample nvidia-smi while CUDA work runs and retain every raw verdict."""

    def __init__(
        self,
        executable: Path,
        *,
        device_id: int,
        expected_gpus: list[dict[str, Any]],
        expected_external_processes: list[dict[str, Any]],
        poll_seconds: float = 1.0,
    ) -> None:
        self.executable = executable
        self.device_id = device_id
        self.expected_gpus = expected_gpus
        self.expected_external_processes = expected_external_processes
        self.poll_seconds = poll_seconds
        self.samples: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mpas-ftz-gpu-exclusivity",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.samples.append(
                    _capture_gpu_checkpoint(
                        self.executable,
                        device_id=self.device_id,
                        label=f"monitor-{len(self.samples) + 1}",
                        expected_gpus=self.expected_gpus,
                        expected_external_processes=self.expected_external_processes,
                    )
                )
            except BaseException as error:  # retained and raised on the main thread
                self.error = error
                self._stop.set()

    def assert_healthy(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"GPU exclusivity monitor failed closed: {self.error}"
            ) from self.error

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.poll_seconds * 4.0))
        if self._thread.is_alive():
            raise RuntimeError("GPU exclusivity monitor did not stop")
        self.assert_healthy()
        return {
            "poll_interval_seconds": self.poll_seconds,
            "sample_count": len(self.samples),
            "samples": self.samples,
            "samples_sha256": _canonical_sha256(self.samples),
            "new_or_drifted_external_process_count": 0,
            "physical_gpu_exclusivity_claim": False,
        }


def _normalize_pci_bus_id(value: str) -> str:
    pieces = value.strip().upper().split(":")
    if len(pieces) != 3 or "." not in pieces[2]:
        raise RuntimeError(f"CUDA PCI bus id is malformed: {value!r}")
    normalized = f"{pieces[0][-4:].zfill(4)}:{pieces[1].zfill(2)}:{pieces[2]}"
    if re.fullmatch(r"[0-9A-F]{4}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]", normalized) is None:
        raise RuntimeError(f"CUDA PCI bus id is malformed: {value!r}")
    return normalized


def _cuda_pci_bus_id(device_id: int) -> str:
    import cupy as cp

    measured = cp.cuda.runtime.deviceGetPCIBusId(device_id)
    if isinstance(measured, bytes):
        measured = measured.decode("ascii", errors="strict")
    return _normalize_pci_bus_id(str(measured))


def _runtime_binding(
    *,
    capability: Any,
    cache: Any,
    cache_dir: Path,
    manifest: Mapping[str, Any],
    selected_gpu: Mapping[str, Any],
    nvidia_smi: Path,
) -> dict[str, Any]:
    exact_cache = str(cache_dir.resolve())
    if str(Path(capability.cache_directory).resolve()) != exact_cache:
        raise RuntimeError("CUDA capability did not bind the exact fresh cache path")
    if str(Path(cache.cache_directory).resolve()) != exact_cache:
        raise RuntimeError("KernelCache did not bind the exact fresh cache path")
    fingerprint = manifest.get("compile_platform", {}).get("fingerprint", {})
    expected = {
        "device_compute_capability": (
            f"{capability.compute_major}{capability.compute_minor}"
        ),
        "cuda_driver_version": str(capability.driver_version),
        "cupy_version": str(capability.cupy_version),
    }
    mismatch = {
        key: (fingerprint.get(key), value)
        for key, value in expected.items()
        if fingerprint.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            f"runtime and compile-platform identity disagree: {mismatch}"
        )
    if int(selected_gpu.get("index", -1)) != int(capability.device_id):
        raise RuntimeError("CUDA runtime and nvidia-smi device indices disagree")
    if selected_gpu.get("name") != capability.name:
        raise RuntimeError("CUDA runtime and nvidia-smi GPU names disagree")
    runtime_compute = f"{capability.compute_major}.{capability.compute_minor}"
    if selected_gpu.get("compute_capability") != runtime_compute:
        raise RuntimeError("CUDA runtime and nvidia-smi compute capability disagree")
    runtime_pci = _cuda_pci_bus_id(int(capability.device_id))
    if _normalize_pci_bus_id(str(selected_gpu.get("pci_bus_id"))) != runtime_pci:
        raise RuntimeError("CUDA runtime and nvidia-smi PCI bus identities disagree")
    executable = Path(sys.executable).resolve()
    return {
        "cuda_capability": capability.as_dict(),
        "kernel_cache_directory": exact_cache,
        "kernel_cache_was_absent_and_exclusively_created": True,
        "cuda_runtime_pci_bus_id": runtime_pci,
        "nvidia_smi": {
            "path": str(nvidia_smi),
            **_file_record(nvidia_smi),
        },
        "selected_gpu": dict(selected_gpu),
        "compile_platform": json.loads(
            json.dumps(manifest["compile_platform"], sort_keys=True)
        ),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(executable),
            **_file_record(executable),
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _require_fresh_directory(path: Path) -> Path:
    selected = path.expanduser().resolve()
    if selected.exists() and any(selected.iterdir()):
        raise RuntimeError(
            f"compile cache must be absent or empty so NVRTC evidence is real: {selected}"
        )
    selected.mkdir(parents=True, exist_ok=True)
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile every exported kernel in all five MPAS CUDA translation "
            "units, verify gpuwm's two-pass FTZ receipt, run the production "
            "scalar adversarial deck twice, and write one fail-closed binding."
        )
    )
    parser.add_argument("--gpuwm-root", type=Path, required=True)
    parser.add_argument("--gpuwm-receipt", type=Path, required=True)
    parser.add_argument(
        "--source-release",
        choices=("v8.2.3", "v8.4.1"),
        default="v8.2.3",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--no-copy-probe",
        action="store_true",
        help="reference the verified input probe without copying its artifacts",
    )
    return parser


def _run_v841(args: argparse.Namespace) -> int:
    output_root = args.output_root or DEFAULT_V841_OUTPUT
    cache_root = args.cache_dir or DEFAULT_V841_CACHE
    output_target, cache_target = _validate_v841_paths(
        output_root=output_root,
        cache_root=cache_root,
        gpuwm_root=args.gpuwm_root,
        gpuwm_receipt=args.gpuwm_receipt,
    )
    if output_target.exists() or output_target.is_symlink():
        raise RuntimeError(
            f"v8.4.1 output root must be absent before the run: {output_target}"
        )
    if cache_target.exists() or cache_target.is_symlink():
        raise RuntimeError(
            f"v8.4.1 compile cache must be absent before the run: {cache_target}"
        )

    nvidia_smi = _nvidia_smi_executable()
    nvidia_smi_pre = _file_record(nvidia_smi)
    checkpoints: list[dict[str, Any]] = []
    with _cuda_device_lease(0) as lease:
        pre_gpu = _capture_gpu_checkpoint(
            nvidia_smi,
            device_id=0,
            label="before-context",
        )
        checkpoints.append(pre_gpu)
        expected_gpus = pre_gpu["gpu_inventory"]
        expected_external_processes = pre_gpu["external_process_baseline"]
        monitor = _GpuExclusivityMonitor(
            nvidia_smi,
            device_id=0,
            expected_gpus=expected_gpus,
            expected_external_processes=expected_external_processes,
        )
        monitor.start()
        try:
            inputs_pre = _input_snapshot(args.gpuwm_root, args.gpuwm_receipt)
            output = _require_absent_directory(
                output_target, purpose="v8.4.1 output root"
            )
            cache_dir = _require_absent_directory(
                cache_target, purpose="v8.4.1 compile cache"
            )
            capability = require_cuda(
                device_id=0,
                min_compute=(12, 0),
                required_compute=(12, 0),
                cache_dir=cache_dir,
            )
            cache = KernelCache(capability=capability, cache_dir=cache_dir)
            exact_cache = str(cache_dir.resolve())
            if str(Path(capability.cache_directory).resolve()) != exact_cache:
                raise RuntimeError(
                    "CUDA capability did not select the exact fresh cache directory"
                )
            if str(Path(cache.cache_directory).resolve()) != exact_cache:
                raise RuntimeError(
                    "KernelCache did not select the exact fresh cache directory"
                )
            if str(Path(os.environ.get("CUPY_CACHE_DIR", "")).resolve()) != exact_cache:
                raise RuntimeError("CUPY_CACHE_DIR is not the exact fresh cache path")

            context_gpu = _capture_gpu_checkpoint(
                nvidia_smi,
                device_id=0,
                label="after-context",
                expected_gpus=expected_gpus,
                expected_external_processes=expected_external_processes,
            )
            checkpoints.append(context_gpu)
            monitor.assert_healthy()

            inventory = v841_reached_translation_units()
            for module_key, (source, names) in inventory.items():
                cache.raw_kernels(names, source, module_key=module_key)
            manifest = cache.compile_manifest()
            relation = validate_v841_compile_manifest_relation(manifest)
            compile_gpu = _capture_gpu_checkpoint(
                nvidia_smi,
                device_id=0,
                label="after-eight-tu-compile",
                expected_gpus=expected_gpus,
                expected_external_processes=expected_external_processes,
            )
            checkpoints.append(compile_gpu)
            monitor.assert_healthy()

            binding = build_mpas_ftz_binding_v841(
                gpuwm_root=args.gpuwm_root,
                gpuwm_receipt_root=args.gpuwm_receipt,
                compile_manifest=manifest,
            )
            kernel_audit = binding["kernel_audit"]
            measurement_gpu = _capture_gpu_checkpoint(
                nvidia_smi,
                device_id=0,
                label="after-four-pass-kernel-measurement",
                expected_gpus=expected_gpus,
                expected_external_processes=expected_external_processes,
            )
            checkpoints.append(measurement_gpu)
            monitor.assert_healthy()
            runtime = _runtime_binding(
                capability=capability,
                cache=cache,
                cache_dir=cache_dir,
                manifest=manifest,
                selected_gpu=measurement_gpu["selected_gpu"],
                nvidia_smi=nvidia_smi,
            )
        finally:
            monitor_record = monitor.stop()

        # The v8.4.1 kernel audit uses separately fresh per-pass caches.  Restore
        # the promoted outer compile-cache route before recording the capsule.
        os.environ["CUPY_CACHE_DIR"] = str(cache_dir.resolve())
        inputs_after_measurement = _input_snapshot(args.gpuwm_root, args.gpuwm_receipt)
        _assert_same_snapshot(inputs_pre, inputs_after_measurement)
        if _file_record(nvidia_smi) != nvidia_smi_pre:
            raise RuntimeError("nvidia-smi executable changed during the run")
        pre_publish_gpu = _capture_gpu_checkpoint(
            nvidia_smi,
            device_id=0,
            label="before-exclusive-publication",
            expected_gpus=expected_gpus,
            expected_external_processes=expected_external_processes,
        )
        checkpoints.append(pre_publish_gpu)

        artifacts: dict[str, object] = {
            "compile-manifest.json": manifest,
            "kernel-audit.json": kernel_audit,
            "binding.json": binding,
        }
        for relative, payload in artifacts.items():
            _write_json_exclusive(output / relative, payload)

        copied_inventory: dict[str, dict[str, Any]] | None = None
        if not args.no_copy_probe:
            copied_inventory = _copy_receipt_exclusive(
                args.gpuwm_receipt,
                output / "gpuwm-probe",
                inputs_pre["gpuwm_receipt"]["files"],
            )

        inputs_post = _input_snapshot(args.gpuwm_root, args.gpuwm_receipt)
        _assert_same_snapshot(inputs_pre, inputs_post)
        if _file_record(nvidia_smi) != nvidia_smi_pre:
            raise RuntimeError("nvidia-smi executable changed during publication")
        post_publish_gpu = _capture_gpu_checkpoint(
            nvidia_smi,
            device_id=0,
            label="after-exclusive-artifact-writes",
            expected_gpus=expected_gpus,
            expected_external_processes=expected_external_processes,
        )
        checkpoints.append(post_publish_gpu)

        output_before_capsule = _tree_inventory(output)
        expected_output_files = set(artifacts)
        if copied_inventory is not None:
            expected_output_files.update(
                f"gpuwm-probe/{relative}" for relative in copied_inventory
            )
        if set(output_before_capsule) != expected_output_files:
            raise RuntimeError(
                "v8.4.1 output root contains an unowned or missing artifact before seal"
            )
        bound_artifacts = _inventory_record(output, output_before_capsule)
        runner_relative = "tools/run_cuda_ftz_contract.py"
        runner_record = inputs_pre["mpas_authority"]["files"].get(runner_relative)
        if runner_record is None:
            raise RuntimeError("runner is absent from the MPAS authority inventory")
        capsule = {
            "schema": V841_CAPSULE_SCHEMA,
            "source_release": "v8.4.1",
            "runner": {
                "path": str((ROOT / runner_relative).resolve()),
                **runner_record,
            },
            "authority_scope": (
                "Every non-generated regular file below MPAS src/, tests/, and "
                "tools/ is hashed before and after execution. The exact four gpuwm "
                "FTZ source inputs, gpuwm HEAD, and every receipt file are also "
                "hashed before and after execution."
            ),
            "authority_inputs": {
                "pre": inputs_pre,
                "post": inputs_post,
                "pre_sha256": _canonical_sha256(inputs_pre),
                "post_sha256": _canonical_sha256(inputs_post),
                "byte_identical": inputs_pre == inputs_post,
            },
            "runtime_platform_binding": runtime,
            "gpu_exclusivity": {
                "lease": lease,
                "nvidia_smi_executable": {
                    "path": str(nvidia_smi),
                    **nvidia_smi_pre,
                },
                "checkpoints": checkpoints,
                "continuous_monitor": monitor_record,
                "own_pid": os.getpid(),
                "participating_launcher_exclusivity": True,
                "stable_external_process_baseline_enforced": True,
                "wddm_system_pid4_insufficient_permissions_sentinel_normalized": True,
                "new_or_drifted_external_process_count": 0,
                "physical_gpu_exclusivity_claim": False,
                "wddm_nonclaim": (
                    "On Windows/WDDM, nvidia-smi can report stable C+G desktop "
                    "clients in the same process inventory and does not expose "
                    "enough evidence to prove physical GPU exclusivity. This "
                    "capsule therefore makes no physical-exclusivity claim. It "
                    "holds an advisory lease for participating MPAS launchers and "
                    "requires the normalized external PID/UUID/name baseline to stay "
                    "byte-stable at one-second samples and every phase boundary. The "
                    "only normalization is Windows System PID 4 reported with the "
                    "literal name '[Insufficient Permissions]', an unstable WDDM "
                    "nvidia-smi telemetry sentinel; every other row remains exact."
                ),
                "sampling_limitation": (
                    "No user-space sampling API can eliminate the interval between "
                    "finite nvidia-smi observations."
                ),
            },
            "publication": {
                "output_root": str(output),
                "output_root_was_absent_and_exclusively_created": True,
                "all_artifact_writes_used_exclusive_create": True,
                "copied_gpuwm_probe": not args.no_copy_probe,
                "capsule_written_last": True,
                "successful_completion_requires_post_capsule_stdout_summary": True,
                "expected_files": sorted(
                    {*expected_output_files, "execution-capsule.json"}
                ),
                "bound_artifacts_before_capsule": bound_artifacts,
            },
        }
        if capsule["authority_inputs"]["byte_identical"] is not True:
            raise RuntimeError("v8.4.1 capsule cannot publish drifted authority inputs")
        _write_json_exclusive(output / "execution-capsule.json", capsule)
        sealed_output = _tree_inventory(output)
        if set(sealed_output) != set(capsule["publication"]["expected_files"]):
            raise RuntimeError("v8.4.1 sealed output root has an unexpected file set")

        # This is deliberately after the last receipt-file write. A cooperating
        # launcher must capture the JSON stdout summary: its digests prove the
        # authority inputs, nvidia-smi binary, GPU identity/process baseline and
        # sealed output tree were still stable after capsule publication.
        inputs_after_capsule = _input_snapshot(args.gpuwm_root, args.gpuwm_receipt)
        _assert_same_snapshot(inputs_pre, inputs_after_capsule)
        nvidia_smi_after_capsule = _file_record(nvidia_smi)
        if nvidia_smi_after_capsule != nvidia_smi_pre:
            raise RuntimeError(
                "nvidia-smi executable changed after capsule publication"
            )
        after_capsule_gpu = _capture_gpu_checkpoint(
            nvidia_smi,
            device_id=0,
            label="after-sealed-tree-verification",
            expected_gpus=expected_gpus,
            expected_external_processes=expected_external_processes,
        )

        summary = {
            "source_release": "v8.4.1",
            "binding": str(output / "binding.json"),
            "execution_capsule": str(output / "execution-capsule.json"),
            "execution_capsule_sha256": sealed_output["execution-capsule.json"][
                "sha256"
            ],
            "compile_manifest_sha256": relation["compile_manifest_sha256"],
            "translation_units": {
                key: len(value["resolved_kernels"])
                for key, value in relation["translation_units"].items()
            },
            "compiled_kernel_audit_count": kernel_audit["kernel_count"],
            "disabled_fallback_red_count": sum(
                bool(row["mutation_red"]) for row in kernel_audit["kernels"].values()
            ),
            "gpuwm_receipt_sha256": binding["gpuwm_ftz_probe"]["receipt_sha256"],
            "post_capsule_authority_inputs_sha256": _canonical_sha256(
                inputs_after_capsule
            ),
            "post_capsule_nvidia_smi_sha256": nvidia_smi_after_capsule["sha256"],
            "post_capsule_gpu_baseline_checkpoint_sha256": _canonical_sha256(
                after_capsule_gpu
            ),
            "sealed_output_tree_sha256": _canonical_sha256(sealed_output),
            "post_capsule_checks_passed": True,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    is_v841 = args.source_release == "v8.4.1"
    if is_v841:
        return _run_v841(args)
    cache_root = args.cache_dir or (DEFAULT_V841_CACHE if is_v841 else DEFAULT_CACHE)
    output_root = args.output_root or (
        DEFAULT_V841_OUTPUT if is_v841 else DEFAULT_OUTPUT
    )
    cache_dir = _require_fresh_directory(cache_root)
    capability = require_cuda(min_compute=(12, 0), cache_dir=cache_dir)
    cache = KernelCache(capability=capability, cache_dir=cache_dir)

    inventory = (
        v841_reached_translation_units() if is_v841 else production_translation_units()
    )
    for module_key, (source, names) in inventory.items():
        cache.raw_kernels(names, source, module_key=module_key)
    manifest = cache.compile_manifest()
    if is_v841:
        relation = validate_v841_compile_manifest_relation(manifest)
        binding = build_mpas_ftz_binding_v841(
            gpuwm_root=args.gpuwm_root,
            gpuwm_receipt_root=args.gpuwm_receipt,
            compile_manifest=manifest,
        )
        kernel_audit = binding["kernel_audit"]
        deck = None
        performance_control = None
    else:
        relation = validate_compile_manifest_relation(manifest)
        deck = run_scalar_transport_subnormal_deck()
        kernel_audit = run_guarded_kernel_subnormal_audit()
        performance_control = run_normalized_fallback_performance_control()
        binding = build_mpas_ftz_binding(
            gpuwm_root=args.gpuwm_root,
            gpuwm_receipt_root=args.gpuwm_receipt,
            compile_manifest=manifest,
            transport_deck=deck,
            kernel_audit=kernel_audit,
            performance_control=performance_control,
        )

    output = output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "compile-manifest.json", manifest)
    _write_json(output / "kernel-audit.json", kernel_audit)
    if deck is not None and performance_control is not None:
        _write_json(output / "transport-deck.json", deck)
        _write_json(output / "normalized-performance-control.json", performance_control)
    _write_json(output / "binding.json", binding)
    if not args.no_copy_probe:
        destination = output / "gpuwm-probe"
        if destination.exists():
            raise RuntimeError(
                f"refusing to overlay an existing copied gpuwm probe: {destination}"
            )
        shutil.copytree(args.gpuwm_receipt.expanduser().resolve(), destination)

    summary = {
        "source_release": args.source_release,
        "binding": str(output / "binding.json"),
        "compile_manifest_sha256": relation["compile_manifest_sha256"],
        "translation_units": {
            key: len(value["resolved_kernels"])
            for key, value in relation["translation_units"].items()
        },
        "compiled_kernel_audit_count": kernel_audit["kernel_count"],
        "disabled_fallback_red_count": sum(
            bool(row["mutation_red"]) for row in kernel_audit["kernels"].values()
        ),
        "gpuwm_receipt_sha256": binding["gpuwm_ftz_probe"]["receipt_sha256"],
    }
    if deck is not None and performance_control is not None:
        summary.update(
            {
                "transport_candidate_max_abs_gap": deck["maximum_candidate_gap"],
                "transport_disabled_fallback_max_abs_gap": deck["mutation_control"][
                    "maximum_gap"
                ],
                "transport_dual_run_byte_identical": deck["dual_run_byte_identical"],
                "normalized_maximum_enabled_over_disabled": performance_control[
                    "maximum_enabled_over_disabled"
                ],
            }
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
