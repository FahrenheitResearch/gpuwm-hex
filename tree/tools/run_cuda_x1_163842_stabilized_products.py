#!/usr/bin/env python3
"""Replay the certified stabilized x1.163842 CUDA arm and make products.

This lane is deliberately narrower than a new forecast certification.  It first
loads the committed hard-gated scout and dual-run evidence, prepares the exact
same pinned GFS/x1.163842 binary32 execution input, compiles into a fresh cache,
and runs one generic 180-step CUDA arm.  History, 0.5-degree NetCDF, renderer
input, and receipts remain forbidden until the new capsule is byte-for-byte and
canonical-JSON equal to the certified arm.

The admitted configuration is a supported 120 s, split=1 stabilization.  It is
not evidence of native 360 s/split=3 equivalence, Fortran equivalence, forecast
skill, or column physics.  Water vapor is a passive transported scalar only.
Rust rendering is a separate ``--render-only`` operation so the CUDA node need
not carry the Windows renderer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for import_root in (SRC, TOOLS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

RUNNER_PATH = TOOLS / "run_real_gfs_cuda_x1_163842.py"
# Both pins are re-derived by tools/repin_source_tables.py.  EXPECTED_RUNNER_SHA256
# was DEAD before the 0.2.0 rename: it arrived with the base import at 8a34759
# and 0911c88 edited the runner without moving it, so this tool would have
# refused its own high-resolution runner at _verify_runner -- before a device is
# touched, and only ever on a machine that has one.
EXPECTED_RUNNER_SHA256 = (
    "5b80b191ad974226e53f9dd9d61c03b82e165ef3eedaa206f2742205cbe5a728"
)
REMOTE_EVIDENCE_VALIDATOR_PATH = TOOLS / "validate_cuda_x1_163842_remote_evidence.py"
EXPECTED_REMOTE_EVIDENCE_VALIDATOR_SHA256 = (
    "800bb6e78f8b9dd3f7266637dfe1cff91e83e6b18da0b77e624ab7a536a29c98"
)


def _sha256_file(path: str | Path) -> str:
    selected = Path(path)
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_runner() -> Any:
    before = _sha256_file(RUNNER_PATH)
    if before != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(
            f"pinned x1.163842 runner changed: {before} != {EXPECTED_RUNNER_SHA256}"
        )
    name = "_x1_163842_stabilized_product_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned runner at {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    after = _sha256_file(RUNNER_PATH)
    if after != EXPECTED_RUNNER_SHA256:
        raise RuntimeError(
            f"pinned x1.163842 runner changed during import: {after} != {before}"
        )
    return module


runner = _load_pinned_runner()
import run_rust_renderer_gate as rust_renderer_gate  # noqa: E402


def _load_remote_evidence_validator() -> Any:
    before = _sha256_file(REMOTE_EVIDENCE_VALIDATOR_PATH)
    if before != EXPECTED_REMOTE_EVIDENCE_VALIDATOR_SHA256:
        raise RuntimeError(
            "pinned remote-evidence validator changed: "
            f"{before} != {EXPECTED_REMOTE_EVIDENCE_VALIDATOR_SHA256}"
        )
    name = "_x1_163842_product_remote_evidence_validator"
    spec = importlib.util.spec_from_file_location(name, REMOTE_EVIDENCE_VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot import remote-evidence validator at {REMOTE_EVIDENCE_VALIDATOR_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if _sha256_file(REMOTE_EVIDENCE_VALIDATOR_PATH) != before:
        raise RuntimeError("remote-evidence validator changed during import")
    return module


remote_evidence_validator = _load_remote_evidence_validator()

from hexcore.cuda_backend import canonical_sha256  # noqa: E402
from hexcore.cuda_driver import (  # noqa: E402
    CUDA_IMPLEMENTED_UNLINKED_EVIDENCE,
)
from hexcore.cuda_dualrun import (  # noqa: E402
    validate_cuda_capsule,
    write_json_atomic,
)
from hexcore.driver import DryDycoreDriver, StabilityBounds  # noqa: E402
from hexcore.output import (  # noqa: E402
    HistoryField,
    HistoryStreamOptions,
    write_history,
)
from hexcore.regrid import (  # noqa: E402
    REGRID_EVIDENCE,
    build_regrid_weights,
    load_regrid_weights,
    save_regrid_weights,
    write_regridded_netcdf,
)
from hexcore.rust_renderer import (  # noqa: E402
    RustWrf2dFields,
    inspect_renderer_products,
    render_catalogued_products,
    sha256_file,
    validate_rust_wrf2d_netcdf,
    write_rust_wrf2d_netcdf,
)
from hexcore.vector import initialize_reconstruction_coefficients  # noqa: E402


TARGET_DT_SECONDS = 120.0
TARGET_DURATION_SECONDS = 21_600.0
TARGET_STEPS = 180
TARGET_ACOUSTIC_SUBSTEPS = 6
TARGET_LATLON_DEGREES = 0.5
TIME_STRING_LENGTH = 64
TARGET_LEVELS = 55
TARGET_ZTOP_M = 30_000.0
SMOOTH_SURFACES = False
EXPECTED_CONFIG_SHA256 = (
    "3780b0718632ea88bd00e073bc374aca2d1f20e842e10bccbd9da35fa5ee4ec0"
)
EXPECTED_CERTIFIED_ARM_SHA256 = (
    "2aa37a43460eff249c26091c1b591d731c00f64ed006e2a63b081ab204397c66"
)
EXPECTED_CERTIFIED_COMPARISON_SHA256 = (
    "f7d2f94f867b95fdb8b951406130cf2d82649f795569827332fa6883ac54d9e9"
)
EXPECTED_CERTIFIED_SUMMARY_SHA256 = (
    "06f1cee5c75111315a02d18a41b226698ad2d6b34fd271b39d3cefd6afe4c3bf"
)
EXPECTED_CERTIFIED_JSONL_SHA256 = (
    "bc2597ecfae1829da51a2ff9c22df997dcf2f41151f7c358b7973d06ba7a9fe7"
)
EXPECTED_CERTIFIED_LOG_SHA256 = (
    "14a5491e2a3dc990b9c576dff6147945731f8d05d2dcb0bc79563771875a1bd0"
)
EXPECTED_CERTIFIED_INVENTORY_SHA256 = (
    "3cc9ee6b9d043dd4a11f8d2abf5b1b9d59945afe11ba775c1e54fd7c1c36aeab"
)
EXPECTED_FINAL_SNAPSHOT_SHA256 = (
    "5abf288bed7b714adaa207dbb4350e21ca9afcce027487fddde5629940f12948"
)

CERTIFIED_CAPSULE_PROFILE = (
    "real-gfs-20260326-x1.163842-stabilized-dt120-split1-work-scout"
)
CERTIFIED_CAPSULE_TARGET = (
    "diagnostic-only real GFS 2026-03-26 00Z x1.163842 dry CUDA 6 h "
    "stabilized scout; supported dt120 split1, not native dt360/split3 "
    "equivalence"
)
CERTIFIED_CAPSULE_PREPARATION_METHOD = (
    "load the three pinned real-GFS/x1.163842 inputs exactly once; build the "
    "55-level vertical grid with smooth_surfaces=False; initialize and "
    "materialize one level-major C-contiguous binary32 host atmosphere; seal "
    "it under canonical stabilized dt120/split1 configuration SHA-256 "
    f"{EXPECTED_CONFIG_SHA256}; reuse that same host preparation for the scout "
    "and two independent CUDA uploads"
)
ACTUAL_FRESH_REPLAY_METHOD = (
    "validate the committed hard-gated scout, dual capsules, gpuwm total "
    "comparison, execution sources, and byte inventories; rebuild one exact "
    "pinned binary32 host preparation; independently upload it once; execute "
    "one generic 180-step CUDA arm; require its complete serialized and "
    "canonical capsule to equal the certified arm; re-hash the downloaded "
    "final state+sidecar; only then write history, 0.5-degree fields, and the "
    "Rust renderer input"
)
PRODUCT_STEM = "GFS-2026-03-26-00.x1.163842.cuda-port-sm120-stabilized-replay-6h"
PRODUCT_CLASSIFICATION = (
    "real-GFS-initialized dry CUDA state evolution replayed exactly from a "
    "previously certified hard-gated trajectory"
)
TIMESTEP_LABEL = "supported dt120 split1; not native dt360/split3 equivalence"
PHYSICS_LABEL = "none; qv is passive scalar transport only"
PARTIAL_HISTORY_SCHEMA = "mpas-port.partial-unstructured-product-history/v1"
LATLON_PRODUCT_SCHEMA = "mpas-port.certified-replay-latlon-product/v1"
LATLON_PRODUCT_SCOPE = (
    "t0 and F006 six-field 0.5-degree diagnostic product; not an MPAS native "
    "or full-state history"
)
PARTIAL_TIMED_FIELDS = (
    "surface_pressure",
    "pressure_lowest_model_level",
    "temperature_lowest_model_level",
    "u_lowest_model_level",
    "v_lowest_model_level",
    "wind_speed_lowest_model_level",
    "qv_lowest_model_level",
)
PARTIAL_STATIC_FIELDS = (
    "indexToCellID",
    "latCell",
    "lonCell",
    "terrain_height",
    "height_lowest_model_level",
)
PRODUCT_CLAIMS = {
    "real_gfs_initialized": True,
    "six_hour_cuda_state_evolution": True,
    "certified_capsule_exact_replay": True,
    "forecast_skill": False,
    "fortran_equivalence": False,
    "native_dt360_split3_equivalence": False,
    "column_or_moist_physics": False,
    "partial_2d_product_history": True,
    "full_3d_state_archived": False,
    "two_meter_or_ten_meter_fields": False,
}


class _NCellsOnlyHistoryMesh:
    """Minimal mesh facade that cannot emit unused topology dimensions."""

    def __init__(self, n_cells: int) -> None:
        if int(n_cells) <= 0:
            raise ValueError("partial history nCells must be positive")
        self.dimensions = {"nCells": int(n_cells)}
        self.arrays: dict[str, np.ndarray] = {}
        self.variable_dimensions: dict[str, tuple[str, ...]] = {}
        self.variable_attrs: dict[str, dict[str, Any]] = {}
        self.attrs: dict[str, Any] = {}


def _n_cells_only_history_mesh(mesh: Any) -> _NCellsOnlyHistoryMesh:
    lat_cell = np.asarray(mesh.latCell)
    lon_cell = np.asarray(mesh.lonCell)
    if lat_cell.shape != (runner.TARGET_CELLS,) or lon_cell.shape != (
        runner.TARGET_CELLS,
    ):
        raise RuntimeError("partial history source mesh cell coordinates changed")
    return _NCellsOnlyHistoryMesh(runner.TARGET_CELLS)


DEFAULT_CERTIFIED_ROOT = (
    ROOT / "receipts" / "cuda-gfs-forecast" / "x1.163842-stabilized-scout-20260810a"
)
CERTIFIED_JSONL_NAME = "x1.163842-stabilized-scout-20260810a.jsonl"
CERTIFIED_LOG_NAME = "x1.163842-stabilized-scout-20260810a.log"
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "cuda-gfs" / "x1.163842-60km-stabilized-replay"
)
DEFAULT_RECEIPT_ROOT = (
    ROOT / "receipts" / "cuda-gfs-forecast" / "x1.163842-60km-stabilized-replay"
)
DEFAULT_CACHE_ROOT = ROOT / "work" / "cuda-gfs-60km-stabilized-replay-cache" / "fresh"
DEFAULT_RENDER_ROOT = (
    ROOT / "artifacts" / "cuda-gfs" / "x1.163842-60km-stabilized-replay-plots"
)
DEFAULT_RENDER_STORE = ROOT / "work" / "cuda-gfs-60km-stabilized-replay-renderer-store"
DEFAULT_RENDER_RECEIPT_ROOT = (
    ROOT / "receipts" / "cuda-gfs-forecast" / "x1.163842-60km-stabilized-replay-render"
)

_CERTIFIED_FILE_PINS = {
    "evidence/arm-a.json": EXPECTED_CERTIFIED_ARM_SHA256,
    "evidence/arm-b.json": EXPECTED_CERTIFIED_ARM_SHA256,
    "evidence/gpuwm-total-comparison.json": EXPECTED_CERTIFIED_COMPARISON_SHA256,
    "evidence/summary.json": EXPECTED_CERTIFIED_SUMMARY_SHA256,
    "execution-sources/diagnose_cuda_x1_163842_stability.py": (
        "c2871670d19715a07ddf30e6cb9a3d9814654b90a85ab2fe8f80ade021041562"
    ),
    "execution-sources/remote_launch_60km_stabilized_scout.sh": (
        "e2d2bc2e2fc84cb40e53c286ff505dd11533e4b838defe4adaf098946bd4da9a"
    ),
    "execution-sources/run_cuda_x1_163842_stabilized_scout.py": (
        "d4f65b3f9b295e611b6dfa1f6317eaa614f91262c33c822a0c138896cc121b50"
    ),
    "execution-sources/run_real_gfs_cuda_x1_163842.py": EXPECTED_RUNNER_SHA256,
    CERTIFIED_JSONL_NAME: EXPECTED_CERTIFIED_JSONL_SHA256,
    CERTIFIED_LOG_NAME: EXPECTED_CERTIFIED_LOG_SHA256,
}

_RUNTIME_SOURCE_PATHS = {
    "product_replay_tool": Path(__file__).resolve(),
    "remote_evidence_validator": REMOTE_EVIDENCE_VALIDATOR_PATH,
    "high_resolution_runner": RUNNER_PATH,
    "coarse_cuda_runner": TOOLS / "run_real_gfs_cuda_forecast.py",
    "cpu_gfs_runner": TOOLS / "run_real_gfs_forecast.py",
    "cuda_dualrun": SRC / "hexcore" / "cuda_dualrun.py",
    "cuda_driver": SRC / "hexcore" / "cuda_driver.py",
    "dry_driver": SRC / "hexcore" / "driver.py",
    "initialization": SRC / "hexcore" / "initialization.py",
    "mesh": SRC / "hexcore" / "mesh.py",
    "vertical": SRC / "hexcore" / "vertical.py",
    "output": SRC / "hexcore" / "output.py",
    "regrid": SRC / "hexcore" / "regrid.py",
    "rust_renderer": SRC / "hexcore" / "rust_renderer.py",
    "vector": SRC / "hexcore" / "vector.py",
    "rust_renderer_gate": TOOLS / "run_rust_renderer_gate.py",
}


@dataclass(frozen=True, slots=True)
class CertifiedEvidence:
    root: Path
    files: dict[str, dict[str, Any]]
    arm_a: dict[str, Any]
    arm_a_bytes: bytes
    summary: dict[str, Any]
    comparison: dict[str, Any]
    jsonl_records: tuple[dict[str, Any], ...]
    authority_verdict: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OutputPlan:
    artifact_root: Path
    receipt_root: Path
    history: Path
    latlon: Path
    regrid_weights: Path
    renderer_input: Path
    replay_capsule: Path
    replay_gate: Path
    receipt: Path
    checksums: Path


@dataclass(frozen=True, slots=True)
class RenderPlan:
    output_root: Path
    store_root: Path
    receipt_root: Path
    receipt: Path
    checksums: Path


def capture_runtime_source_pins() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for label, path in _RUNTIME_SOURCE_PATHS.items():
        selected = path.resolve(strict=True)
        if selected.is_symlink():
            raise RuntimeError(f"runtime source must not be a symlink: {label}")
        records[label] = _record(selected)
    if records["high_resolution_runner"]["sha256"] != EXPECTED_RUNNER_SHA256:
        raise RuntimeError("runtime source inventory binds the wrong high-res runner")
    if records["remote_evidence_validator"]["sha256"] != (
        EXPECTED_REMOTE_EVIDENCE_VALIDATOR_SHA256
    ):
        raise RuntimeError(
            "runtime source inventory binds the wrong evidence validator"
        )
    return records


def assert_runtime_sources_unchanged(
    captured: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(captured) != set(_RUNTIME_SOURCE_PATHS):
        raise RuntimeError("runtime source inventory labels changed")
    for label, path in _RUNTIME_SOURCE_PATHS.items():
        current = _record(path)
        if current != dict(captured[label]):
            raise RuntimeError(f"runtime source changed during product work: {label}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RuntimeError(f"invalid certified JSON in {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"certified JSON root is not an object: {label}")
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _written_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _record(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve(strict=True)
    try:
        logical = selected.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"evidence must remain beneath the repository: {selected}"
        ) from error
    return {
        "path": logical,
        "path_kind": "repo_relative",
        "bytes": selected.stat().st_size,
        "sha256": _sha256_file(selected),
    }


def _require_beneath(path: str | Path, parent: str | Path, label: str) -> Path:
    selected = Path(path).expanduser().resolve()
    boundary = Path(parent).expanduser().resolve()
    if selected == boundary or not selected.is_relative_to(boundary):
        raise ValueError(f"{label} must be a child of {boundary}: {selected}")
    return selected


def _require_disjoint(left: Path, right: Path, label: str) -> None:
    a = left.expanduser().resolve()
    b = right.expanduser().resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError(f"{label} must be path-disjoint: {a} vs {b}")


def _require_file_record(
    record: Any,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> None:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"certified {label} file record is missing")
    if record.get("sha256") != expected_sha256:
        raise RuntimeError(f"certified {label} file record has the wrong SHA-256")
    if int(record.get("bytes", -1)) != expected_bytes:
        raise RuntimeError(f"certified {label} file record has the wrong byte count")


def _validate_certified_capsule(capsule: Mapping[str, Any]) -> None:
    validate_cuda_capsule(capsule)
    if capsule.get("configuration", {}).get("sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("certified arm binds the wrong stabilized configuration")
    trajectory = capsule.get("trajectory", {})
    records = trajectory.get("step_records")
    if not isinstance(records, list) or len(records) != TARGET_STEPS:
        raise RuntimeError("certified arm is not a complete 180-step trajectory")
    if int(trajectory.get("steps", -1)) != TARGET_STEPS:
        raise RuntimeError("certified arm declares the wrong step count")
    if float(trajectory.get("dt_seconds", np.nan)) != TARGET_DT_SECONDS:
        raise RuntimeError("certified arm declares the wrong timestep")
    if trajectory.get("final_snapshot_sha256") != EXPECTED_FINAL_SNAPSHOT_SHA256:
        raise RuntimeError("certified arm final snapshot changed")
    for expected_step, record in enumerate(records, 1):
        contract = record.get("step_contract", {})
        if int(record.get("step", -1)) != expected_step:
            raise RuntimeError("certified arm has noncontiguous step records")
        if contract.get("evidence") != CUDA_IMPLEMENTED_UNLINKED_EVIDENCE:
            raise RuntimeError("certified arm makes a linked-authority claim")
        if contract.get("authority_ruler") is not None:
            raise RuntimeError("certified arm unexpectedly carries an authority ruler")
        if contract.get("authority_ruler_sha256") is not None:
            raise RuntimeError("certified arm carries a ruler digest without a ruler")
        if contract.get("configuration_sha256") != EXPECTED_CONFIG_SHA256:
            raise RuntimeError("certified arm step changes stabilized configuration")
        if int(contract.get("d2h_bytes_inside_step", -1)) != 0:
            raise RuntimeError("certified arm performed an internal step D2H")


def _parse_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise RuntimeError(f"blank certified JSONL line at {line_number}")
            records.append(_load_json_bytes(line, label=f"{path.name}:{line_number}"))
    if not records:
        raise RuntimeError("certified JSONL is empty")
    return tuple(records)


def _validate_inventory(root: Path, files: Mapping[str, Mapping[str, Any]]) -> None:
    inventory_path = root / "inventory.json"
    raw = inventory_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_CERTIFIED_INVENTORY_SHA256:
        raise RuntimeError(
            "certified inventory changed: "
            f"{digest} != {EXPECTED_CERTIFIED_INVENTORY_SHA256}"
        )
    inventory = _load_json_bytes(raw, label="inventory.json")
    if inventory.get("schema") != (
        "mpas-port.x1.163842-stabilized-scout-evidence-inventory/v1"
    ):
        raise RuntimeError("certified inventory schema changed")
    records = inventory.get("files")
    if not isinstance(records, Mapping):
        raise RuntimeError("certified inventory has no files mapping")
    indexed = dict(records)
    if set(indexed) != set(_CERTIFIED_FILE_PINS):
        missing = sorted(set(_CERTIFIED_FILE_PINS) - set(indexed))
        extra = sorted(set(indexed) - set(_CERTIFIED_FILE_PINS))
        raise RuntimeError(
            f"certified inventory coverage changed: missing={missing}, extra={extra}"
        )
    for logical, actual in files.items():
        listed = indexed.get(logical)
        if not isinstance(listed, Mapping):
            raise RuntimeError(f"certified inventory omits required file: {logical}")
        if listed.get("sha256") != actual["sha256"]:
            raise RuntimeError(f"certified inventory SHA mismatch: {logical}")
        if int(listed.get("bytes", -1)) != int(actual["bytes"]):
            raise RuntimeError(f"certified inventory size mismatch: {logical}")


def validate_certified_evidence(root: str | Path) -> CertifiedEvidence:
    """Load and fully bind the committed hard-gated scout evidence."""

    unresolved = Path(root).expanduser()
    if unresolved.is_symlink():
        raise RuntimeError("certified evidence root must not be a symlink")
    selected = _require_beneath(root, ROOT / "receipts", "certified evidence root")
    selected = selected.resolve(strict=True)
    authority_verdict = remote_evidence_validator.validate_scout_evidence(
        selected, remote_evidence_validator.DEFAULT_FTZ_ROOT
    )
    required_authority_verdict = {
        "status": "passed",
        "inventory_sha256": EXPECTED_CERTIFIED_INVENTORY_SHA256,
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "arm_sha256": EXPECTED_CERTIFIED_ARM_SHA256,
        "arms_byte_identical": True,
        "gpuwm_total_comparison": True,
        "scout_steps": TARGET_STEPS,
        "simulated_seconds": TARGET_DURATION_SECONDS,
        "final_snapshot_sha256": EXPECTED_FINAL_SNAPSHOT_SHA256,
        "forecast_products_written": False,
    }
    for key, expected in required_authority_verdict.items():
        if authority_verdict.get(key) != expected:
            raise RuntimeError(f"remote authority validator verdict changed: {key}")
    ftz_verdict = authority_verdict.get("ftz", {})
    if (
        ftz_verdict.get("binding_sha256")
        != "d120faf49894dec04cca97ca5ceecde18c0030bc1930bd326e8580f22102b145"
        or ftz_verdict.get("compile_manifest_sha256")
        != "bfd9ffc1b42af862dac65fa1d713986354db0c1eea2bc15a3e70e9964fbee68b"
        or ftz_verdict.get("gpuwm_probe_files") != 26
        or ftz_verdict.get("gpuwm_static_source_files") != 5
    ):
        raise RuntimeError("remote sm_120 FTZ authority verdict changed")
    expected_tree = {"inventory.json", *_CERTIFIED_FILE_PINS}
    actual_tree: set[str] = set()
    for entry in selected.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(
                "certified evidence tree contains a symlink: "
                f"{entry.relative_to(selected).as_posix()}"
            )
        if entry.is_file():
            actual_tree.add(entry.relative_to(selected).as_posix())
    if actual_tree != expected_tree:
        missing = sorted(expected_tree - actual_tree)
        extra = sorted(actual_tree - expected_tree)
        raise RuntimeError(
            f"certified physical evidence tree changed: missing={missing}, extra={extra}"
        )
    files: dict[str, dict[str, Any]] = {}
    payloads: dict[str, bytes] = {}
    for logical, expected_sha in _CERTIFIED_FILE_PINS.items():
        path = selected / Path(logical)
        if not path.is_file():
            raise RuntimeError(f"certified evidence file is missing: {logical}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_sha:
            raise RuntimeError(
                f"certified evidence changed: {logical} {digest} != {expected_sha}"
            )
        files[logical] = {
            "path": path.relative_to(ROOT).as_posix(),
            "path_kind": "repo_relative",
            "bytes": len(payload),
            "sha256": digest,
        }
        payloads[logical] = payload
    _validate_inventory(selected, files)
    inventory_path = selected / "inventory.json"
    files["inventory.json"] = {
        "path": inventory_path.relative_to(ROOT).as_posix(),
        "path_kind": "repo_relative",
        "bytes": inventory_path.stat().st_size,
        "sha256": EXPECTED_CERTIFIED_INVENTORY_SHA256,
    }
    executed_runner = selected / "execution-sources" / "run_real_gfs_cuda_x1_163842.py"
    if executed_runner.read_bytes() != RUNNER_PATH.read_bytes():
        raise RuntimeError(
            "live production runner is not byte-identical to the certified copy"
        )

    arm_a = _load_json_bytes(payloads["evidence/arm-a.json"], label="arm-a.json")
    arm_b = _load_json_bytes(payloads["evidence/arm-b.json"], label="arm-b.json")
    _validate_certified_capsule(arm_a)
    _validate_certified_capsule(arm_b)
    if payloads["evidence/arm-a.json"] != payloads["evidence/arm-b.json"]:
        raise RuntimeError("certified arm capsules are not byte-identical")
    if arm_a != arm_b:
        raise RuntimeError("certified arm capsules are not structurally identical")

    comparison = _load_json_bytes(
        payloads["evidence/gpuwm-total-comparison.json"],
        label="gpuwm-total-comparison.json",
    )
    gpuwm = comparison.get("gpuwm_comparison", {})
    if comparison.get("schema") != "mpas-port.cuda-dual-run-report/v1":
        raise RuntimeError("certified gpuwm report schema changed")
    if comparison.get("total_comparison") is not True:
        raise RuntimeError("certified gpuwm report is not a total comparison")
    if gpuwm.get("identical") is not True or gpuwm.get("divergence_count") != 0:
        raise RuntimeError("certified gpuwm comparison is not exactly green")
    if comparison.get("capsules") != {
        "a": {"sha256": EXPECTED_CERTIFIED_ARM_SHA256},
        "b": {"sha256": EXPECTED_CERTIFIED_ARM_SHA256},
    }:
        raise RuntimeError("certified gpuwm report capsule bindings changed")
    comparison_authority = arm_a.get("contracts", {}).get("comparison_authority")
    if comparison.get("comparison_authority") != comparison_authority:
        raise RuntimeError("certified gpuwm report comparison authority changed")

    summary = _load_json_bytes(payloads["evidence/summary.json"], label="summary.json")
    if summary.get("schema") != "mpas-port.x1.163842-stabilized-scout-dual/v1":
        raise RuntimeError("certified summary schema changed")
    if summary.get("configuration", {}).get("sha256") != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("certified summary configuration changed")
    if summary.get("scout", {}).get("status") != "passed":
        raise RuntimeError("certified hard-gated scout is not green")
    if summary.get("forecast_products_written") is not False:
        raise RuntimeError("certified scout improperly claims forecast products")
    if summary.get("scout", {}).get("final_snapshot_sha256") != (
        EXPECTED_FINAL_SNAPSHOT_SHA256
    ):
        raise RuntimeError("certified scout summary final snapshot changed")
    if summary.get("configuration", {}).get("value") != arm_a.get(
        "configuration", {}
    ).get("value"):
        raise RuntimeError("certified summary and arm configuration documents differ")
    if canonical_sha256(summary["configuration"]["value"]) != (EXPECTED_CONFIG_SHA256):
        raise RuntimeError("certified summary configuration digest is false")

    source_records = summary.get("gate_implementation_sources", {})
    expected_source_bindings = {
        "hardened_locator": (
            "work/diagnose_cuda_x1_163842_stability.py",
            _CERTIFIED_FILE_PINS[
                "execution-sources/diagnose_cuda_x1_163842_stability.py"
            ],
        ),
        "stabilized_scout": (
            "work/run_cuda_x1_163842_stabilized_scout.py",
            _CERTIFIED_FILE_PINS[
                "execution-sources/run_cuda_x1_163842_stabilized_scout.py"
            ],
        ),
        "transitive_high_resolution_runner": (
            "tools/run_real_gfs_cuda_x1_163842.py",
            EXPECTED_RUNNER_SHA256,
        ),
    }
    if set(source_records) != set(expected_source_bindings):
        raise RuntimeError("certified summary source-binding inventory changed")
    for label, (expected_path, expected_sha) in expected_source_bindings.items():
        record = source_records[label]
        if record.get("path") != expected_path:
            raise RuntimeError(f"certified summary source path changed: {label}")
        if record.get("observed_sha256") != expected_sha:
            raise RuntimeError(f"certified summary source SHA changed: {label}")
        declared = record.get("expected_sha256")
        if declared is not None and declared != expected_sha:
            raise RuntimeError(
                f"certified summary expected source SHA is false: {label}"
            )

    trajectory = arm_a["trajectory"]
    step_hashes = [
        record.get("snapshot", {}).get("sha256")
        for record in trajectory["step_records"]
    ]
    if summary["scout"].get("step_snapshot_sha256") != step_hashes:
        raise RuntimeError("certified scout and arm step trajectories differ")
    if summary["scout"].get("t0_snapshot_sha256") != trajectory.get(
        "initial_snapshot", {}
    ).get("sha256"):
        raise RuntimeError("certified scout and arm initial snapshots differ")
    if summary.get("host_execution_seal_sha256") != arm_a.get("preparation", {}).get(
        "initial_execution_fingerprint_sha256"
    ):
        raise RuntimeError("certified summary and arm execution seals differ")
    if summary.get("host_after_sha256") != summary.get("host_execution_seal_sha256"):
        raise RuntimeError("certified dual run mutated its sealed host preparation")
    _require_file_record(
        summary.get("arms", {}).get("a", {}).get("file"),
        expected_sha256=EXPECTED_CERTIFIED_ARM_SHA256,
        expected_bytes=len(payloads["evidence/arm-a.json"]),
        label="arm A",
    )
    _require_file_record(
        summary.get("arms", {}).get("b", {}).get("file"),
        expected_sha256=EXPECTED_CERTIFIED_ARM_SHA256,
        expected_bytes=len(payloads["evidence/arm-b.json"]),
        label="arm B",
    )
    _require_file_record(
        summary.get("comparison", {}).get("file"),
        expected_sha256=EXPECTED_CERTIFIED_COMPARISON_SHA256,
        expected_bytes=len(payloads["evidence/gpuwm-total-comparison.json"]),
        label="comparison",
    )
    if summary.get("comparison", {}).get("total_comparison") is not True:
        raise RuntimeError("certified summary does not bind total comparison")
    if summary.get("comparison", {}).get("gpuwm_identical") is not True:
        raise RuntimeError("certified summary does not bind gpuwm equality")
    if summary.get("comparison", {}).get("gpuwm_divergence_count") != 0:
        raise RuntimeError("certified summary gpuwm divergence count changed")
    if summary.get("comparison", {}).get("authority") != comparison_authority:
        raise RuntimeError("certified summary comparison authority changed")
    compile_sha = arm_a.get("contracts", {}).get("compile_manifest", {}).get("sha256")
    layout_sha = arm_a.get("contracts", {}).get("layout", {}).get("sha256")
    for arm_label in ("a", "b"):
        arm_summary = summary.get("arms", {}).get(arm_label, {})
        required = {
            "steps": TARGET_STEPS,
            "dt_seconds": TARGET_DT_SECONDS,
            "configuration_sha256": EXPECTED_CONFIG_SHA256,
            "compile_manifest_sha256": compile_sha,
            "layout_contract_sha256": layout_sha,
            "final_snapshot_sha256": EXPECTED_FINAL_SNAPSHOT_SHA256,
            "t0_and_all_step_snapshots_match_hard_gated_scout": True,
            "all_steps_implemented_unlinked": True,
            "all_authority_rulers_null": True,
            "all_internal_d2h_bytes_zero": True,
        }
        for key, expected in required.items():
            if arm_summary.get(key) != expected:
                raise RuntimeError(
                    f"certified summary arm {arm_label} binding changed: {key}"
                )
    if summary["scout"].get("compile_manifest_sha256") != compile_sha:
        raise RuntimeError("certified scout and arm compile manifests differ")
    if summary["scout"].get("layout_contract_sha256") != layout_sha:
        raise RuntimeError("certified scout and arm layout contracts differ")

    jsonl_path = selected / CERTIFIED_JSONL_NAME
    jsonl_records = _parse_jsonl(jsonl_path)
    phases = [record.get("phase") for record in jsonl_records]
    expected_phases = [
        "host-preparation-start",
        "host-preparation-complete",
        "scout-start",
        *(["scout-step"] * TARGET_STEPS),
        "scout-complete",
        "arm-start",
        "arm-complete",
        "arm-start",
        "arm-complete",
        "gpuwm-total-comparison-start",
        "gpuwm-total-comparison-complete",
        "run-complete",
    ]
    if phases != expected_phases:
        raise RuntimeError("certified JSONL phase sequence changed")
    start = jsonl_records[0]
    if (
        start.get("tool_sha256")
        != _CERTIFIED_FILE_PINS[
            "execution-sources/run_cuda_x1_163842_stabilized_scout.py"
        ]
        or start.get("locator_sha256")
        != _CERTIFIED_FILE_PINS[
            "execution-sources/diagnose_cuda_x1_163842_stability.py"
        ]
        or start.get("runner_sha256") != EXPECTED_RUNNER_SHA256
    ):
        raise RuntimeError("certified JSONL execution-source binding changed")
    host_complete = jsonl_records[1]
    if host_complete.get("execution_seal_sha256") != summary.get(
        "host_execution_seal_sha256"
    ):
        raise RuntimeError("certified JSONL host execution seal changed")
    scout_start = jsonl_records[2]
    if scout_start.get("t0_snapshot_sha256") != summary["scout"].get(
        "t0_snapshot_sha256"
    ):
        raise RuntimeError("certified JSONL scout t0 changed")
    scout_steps = jsonl_records[3 : 3 + TARGET_STEPS]
    for expected_step, (record, expected_snapshot) in enumerate(
        zip(scout_steps, step_hashes, strict=True), 1
    ):
        failures = record.get("failures")
        expected_failure_keys = {
            "bound_failures",
            "cfl_failures",
            "hard_domain_failures",
            "nonfinite_fields",
        }
        failures_green = (
            isinstance(failures, Mapping)
            and set(failures) == expected_failure_keys
            and all(failures[key] == [] for key in expected_failure_keys)
        )
        positivity = record.get("thermodynamic_positivity", {})
        positivity_counts = positivity.get("counts", {})
        positivity_green = (
            positivity.get("hard_domain_failures") == []
            and positivity.get("nonfinite_fields") == []
            and isinstance(positivity_counts, Mapping)
            and positivity_counts
            and all(value == 0 for value in positivity_counts.values())
        )
        fields = record.get("fields", {})
        fields_green = (
            isinstance(fields, Mapping)
            and fields
            and all(field.get("all_finite") is True for field in fields.values())
        )
        metrics = record.get("metrics", {})
        hard_cfl_names = (
            "max_horizontal_large_step_advective_cfl",
            "max_horizontal_acoustic_cfl_estimate",
            "max_horizontal_mass_outflow_courant_diagnostic",
            "max_vertical_mass_outflow_courant_diagnostic",
        )
        hard_cfls_green = all(
            np.isfinite(float(metrics.get(name, np.nan)))
            and 0.0 <= float(metrics[name]) <= 1.0
            for name in hard_cfl_names
        )
        declared_bounds_green = (
            abs(float(metrics.get("mass_relative_drift", np.inf))) <= 2.0e-8
            and abs(float(metrics.get("qv_mass_relative_drift", np.inf))) <= 2.0e-8
            and abs(float(metrics.get("energy_proxy_relative_drift", np.inf))) <= 0.5
            and abs(float(metrics.get("max_abs_normal_velocity", np.inf))) <= 500.0
            and abs(float(metrics.get("max_abs_vertical_velocity", np.inf))) <= 500.0
            and float(metrics.get("min_density", -np.inf)) >= 1.0e-7
            and metrics.get("sound_speed", {}).get("all_finite") is True
            and metrics.get("total_pressure", {}).get("all_finite") is True
        )
        if (
            record.get("step") != expected_step
            or record.get("status") != "passed"
            or not failures_green
            or not positivity_green
            or not fields_green
            or not hard_cfls_green
            or not declared_bounds_green
            or record.get("snapshot_sha256") != expected_snapshot
            or record.get("model_time_seconds") != expected_step * TARGET_DT_SECONDS
            or record.get("compile_manifest_sha256") != compile_sha
            or record.get("layout_contract_sha256") != layout_sha
        ):
            raise RuntimeError(
                f"certified JSONL hard-gated scout step changed: {expected_step}"
            )
    scout_complete = [
        record for record in jsonl_records if record.get("phase") == "scout-complete"
    ]
    if len(scout_complete) != 1 or scout_complete[0].get("status") != "passed":
        raise RuntimeError(
            "certified JSONL does not contain one green scout completion"
        )
    arm_complete = [
        record for record in jsonl_records if record.get("phase") == "arm-complete"
    ]
    if [record.get("arm") for record in arm_complete] != ["a", "b"]:
        raise RuntimeError("certified JSONL arm completion order changed")
    if any(
        record.get("capsule", {}).get("sha256") != EXPECTED_CERTIFIED_ARM_SHA256
        for record in arm_complete
    ):
        raise RuntimeError("certified JSONL arm capsule digest changed")
    comparison_complete = [
        record
        for record in jsonl_records
        if record.get("phase") == "gpuwm-total-comparison-complete"
    ]
    if len(comparison_complete) != 1:
        raise RuntimeError("certified JSONL comparison completion changed")
    emitted_summary = dict(comparison_complete[0])
    emitted_summary.pop("phase")
    summary_file_record = emitted_summary.pop("summary_file", None)
    if emitted_summary != summary:
        raise RuntimeError("certified JSONL embedded summary differs from summary file")
    _require_file_record(
        summary_file_record,
        expected_sha256=EXPECTED_CERTIFIED_SUMMARY_SHA256,
        expected_bytes=len(payloads["evidence/summary.json"]),
        label="JSONL summary",
    )
    run_complete = [
        record for record in jsonl_records if record.get("phase") == "run-complete"
    ]
    if len(run_complete) != 1 or run_complete[0].get("status") != "passed":
        raise RuntimeError("certified JSONL does not terminate green")
    if run_complete[0].get("forecast_products_written") is not False:
        raise RuntimeError("certified JSONL improperly claims products")

    return CertifiedEvidence(
        root=selected,
        files=files,
        arm_a=arm_a,
        arm_a_bytes=payloads["evidence/arm-a.json"],
        summary=summary,
        comparison=comparison,
        jsonl_records=jsonl_records,
        authority_verdict=authority_verdict,
    )


def canonical_stabilized_config() -> Any:
    config = replace(
        runner.coarse.cpu_gfs.forecast_config(
            TARGET_DT_SECONDS, TARGET_ACOUSTIC_SUBSTEPS
        ),
        config_horiz_mixing="2d_smagorinsky",
        config_len_disp=0.0,
        config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125,
        config_del4u_div_factor=10.0,
        config_h_ScaleWithMesh=True,
        config_mpas_cam_coef=0.0,
        config_divergence_damping=True,
        config_smdiv=0.1,
        config_xnutr=0.2,
        config_zd=22_000.0,
    )
    config.validate()
    digest = canonical_sha256(asdict(config))
    if digest != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"canonical stabilized config changed: {digest} != {EXPECTED_CONFIG_SHA256}"
        )
    if config.config_dynamics_split_steps != 1:
        raise RuntimeError("stabilized product replay must remain split=1")
    if config.config_physics_suite != "none" or config.config_moist_physics:
        raise RuntimeError("stabilized product replay must remain no-physics")
    return config


def prepare_stabilized_case(
    source_path: str | Path,
    grid_path: str | Path,
    static_path: str | Path,
    config: Any,
) -> Any:
    """Rebuild the exact certified host preparation from the three pinned inputs."""

    if canonical_sha256(asdict(config)) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("host preparation received a noncanonical config")
    source_file = Path(source_path).expanduser().resolve(strict=True)
    grid_file = Path(grid_path).expanduser().resolve(strict=True)
    static_file = Path(static_path).expanduser().resolve(strict=True)
    input_records = {
        "gfs_wps_intermediate": runner.coarse.pinned_input_record(
            source_file, runner.INPUT_PINS["gfs_wps_intermediate"]
        ),
        "x1_163842_grid": runner.coarse.pinned_input_record(
            grid_file, runner.INPUT_PINS["x1_163842_grid"]
        ),
        "x1_163842_static": runner.coarse.pinned_input_record(
            static_file, runner.INPUT_PINS["x1_163842_static"]
        ),
    }
    source = runner.load_structured_atmosphere(source_file)
    mesh, output_mesh, mesh_merge = runner.merge_official_x1_mesh(
        grid_file, static_file
    )
    runner.coarse.cpu_gfs.validate_output_mesh(mesh, output_mesh)
    dimensions = {
        name: int(mesh.dimensions[name]) for name in ("nCells", "nEdges", "nVertices")
    }
    if dimensions != {
        "nCells": runner.TARGET_CELLS,
        "nEdges": runner.TARGET_EDGES,
        "nVertices": runner.TARGET_VERTICES,
    }:
        raise RuntimeError(f"pinned x1.163842 topology changed: {dimensions}")
    if float(np.asarray(mesh.nominalMinDc)) != runner.TARGET_NOMINAL_MIN_DC_M:
        raise RuntimeError("pinned x1.163842 nominalMinDc changed")

    vertical = runner.build_vertical_grid(
        mesh,
        np.asarray(mesh.ter, dtype=np.float64),
        n_vert_levels=TARGET_LEVELS,
        ztop=TARGET_ZTOP_M,
        smooth_surfaces=SMOOTH_SURFACES,
    )
    vertical = runner.coarse.cpu_gfs.normalize_runtime_vertical(vertical)
    vertical = runner.coarse.normalize_cuda_vertical_sentinels(vertical)
    initialized = runner.initialize_from_structured(source, mesh, vertical)
    terrain, coupling = runner.coarse.cpu_gfs.build_order2_terrain_metrics(
        mesh, vertical, config.config_coef_3rd_order
    )
    reference, saved = runner.coarse.cpu_gfs.build_reference_and_sidecar(
        initialized, mesh, vertical, coupling
    )
    state32, saved32, vertical32, reference32, terrain32 = (
        runner.coarse.materialize_binary32_atmosphere(
            initialized.state,
            saved,
            vertical,
            reference,
            terrain,
            n_cells=runner.TARGET_CELLS,
            n_edges=runner.TARGET_EDGES,
        )
    )
    if float(state32.time_seconds) != 0.0:
        raise RuntimeError("pinned GFS state no longer starts at model time zero")
    provenance = dict(source.provenance)
    if str(provenance.get("valid_time")) != "2026-03-26_00:00:00":
        raise RuntimeError("pinned GFS valid time changed")
    if float(provenance.get("forecast_hour", np.nan)) != 0.0:
        raise RuntimeError("pinned GFS input is no longer forecast hour zero")

    prepared = runner.coarse.PreparedCudaInputs.validated(
        config=config,
        profile=CERTIFIED_CAPSULE_PROFILE,
        target=CERTIFIED_CAPSULE_TARGET,
        preparation_method=CERTIFIED_CAPSULE_PREPARATION_METHOD,
        mesh=mesh,
        state=state32,
        vertical=vertical32,
        reference=reference32,
        saved_diagnostics=saved32,
        terrain_metrics=terrain32,
        input_bytes=input_records,
    )
    if prepared.expected_execution_fingerprint["configuration_sha256"] != (
        EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("prepared execution seal binds the wrong config")
    return runner.PreparedGfsCase(
        cuda=prepared,
        output_mesh=output_mesh,
        source_provenance=provenance,
        input_records=input_records,
        grid_path=grid_file,
        static_path=static_file,
        mesh_merge=mesh_merge,
    )


def build_output_plan(
    artifact_root: str | Path, receipt_root: str | Path
) -> OutputPlan:
    artifacts = _require_beneath(
        artifact_root, ROOT / "artifacts" / "cuda-gfs", "artifact root"
    )
    receipts = _require_beneath(
        receipt_root,
        ROOT / "receipts" / "cuda-gfs-forecast",
        "receipt root",
    )
    return OutputPlan(
        artifact_root=artifacts,
        receipt_root=receipts,
        history=artifacts / f"{PRODUCT_STEM}.partial-product-history.nc",
        latlon=artifacts / f"{PRODUCT_STEM}.latlon-0p5deg.nc",
        regrid_weights=artifacts / f"{PRODUCT_STEM}.regrid-weights.npz",
        renderer_input=artifacts / f"{PRODUCT_STEM}.rw-wrf2d.nc",
        replay_capsule=receipts / f"{PRODUCT_STEM}.replay-arm.json",
        replay_gate=receipts / f"{PRODUCT_STEM}.certified-replay-gate.json",
        receipt=receipts / f"{PRODUCT_STEM}.json",
        checksums=receipts / "SHA256SUMS",
    )


def validate_fresh_output_plan(plan: OutputPlan) -> None:
    for root in (plan.artifact_root, plan.receipt_root):
        if root.exists():
            raise FileExistsError(f"product output root must be absent: {root}")


def _validate_cache_preimport(cache_root: str | Path) -> Path:
    selected = _require_beneath(
        cache_root,
        ROOT / "work" / "cuda-gfs-60km-stabilized-replay-cache",
        "CUDA cache root",
    )
    if selected.exists():
        raise FileExistsError(f"CUDA cache root must be absent: {selected}")
    if "CUPY_CACHE_IN_MEMORY" in os.environ:
        raise RuntimeError("CUPY_CACHE_IN_MEMORY must be unset")
    for name in ("CUPY_CACHE_DIR", "MPAS_PORT_CUDA_CACHE_DIR"):
        raw = os.environ.get(name)
        if raw is None:
            raise RuntimeError(f"{name} must be exported before Python starts")
        if Path(raw).expanduser().resolve() != selected:
            raise RuntimeError(f"{name} does not match --cache-root")
    if "cupy" in sys.modules:
        raise RuntimeError("CuPy was imported before the fresh-cache gate")
    return selected


def certify_replay_capsule(
    candidate: Mapping[str, Any], certified: CertifiedEvidence
) -> dict[str, Any]:
    """Require total serialized and canonical equality before product admission."""

    _validate_certified_capsule(candidate)
    candidate_bytes = _written_json_bytes(candidate)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if candidate_sha != EXPECTED_CERTIFIED_ARM_SHA256:
        raise RuntimeError(
            "fresh replay capsule byte SHA differs from certified arm: "
            f"{candidate_sha} != {EXPECTED_CERTIFIED_ARM_SHA256}"
        )
    if candidate_bytes != certified.arm_a_bytes:
        raise RuntimeError("fresh replay capsule is not byte-for-byte certified")
    if dict(candidate) != certified.arm_a:
        raise RuntimeError("fresh replay capsule object differs from certified arm")
    candidate_canonical = hashlib.sha256(_canonical_json_bytes(candidate)).hexdigest()
    certified_canonical = hashlib.sha256(
        _canonical_json_bytes(certified.arm_a)
    ).hexdigest()
    if candidate_canonical != certified_canonical:
        raise RuntimeError("fresh replay capsule canonical JSON differs")
    final_sha = candidate.get("trajectory", {}).get("final_snapshot_sha256")
    if final_sha != EXPECTED_FINAL_SNAPSHOT_SHA256:
        raise RuntimeError("fresh replay final snapshot differs from certification")
    return {
        "schema": "mpas-port.x1.163842-certified-product-replay-gate/v1",
        "status": "passed",
        "admission_policy": (
            "no history, regrid, renderer input, or plots before the complete "
            "fresh capsule is byte-for-byte and canonical-JSON equal to the "
            "certified hard-gated arm"
        ),
        "candidate_capsule_sha256": candidate_sha,
        "certified_capsule_sha256": EXPECTED_CERTIFIED_ARM_SHA256,
        "entire_capsule_byte_equal": True,
        "entire_capsule_object_equal": True,
        "entire_capsule_canonical_json_equal": True,
        "canonical_json_sha256": candidate_canonical,
        "final_snapshot_sha256": final_sha,
        "configuration_sha256": EXPECTED_CONFIG_SHA256,
        "steps": TARGET_STEPS,
        "dt_seconds": TARGET_DT_SECONDS,
        "timestep_interpretation": TIMESTEP_LABEL,
        "physics": PHYSICS_LABEL,
        "forecast_skill_claimed": False,
    }


def _qv_mass(case: Any, state: Any) -> float:
    area = np.asarray(case.cuda.mesh.areaCell, dtype=np.float64)
    dzw = np.asarray(case.cuda.vertical.dzw, dtype=np.float64)
    rho = np.asarray(state.rho, dtype=np.float64)
    qv = np.asarray(state.scalars[0], dtype=np.float64)
    return float(np.sum(rho * qv * dzw[:, None] * area[None, :], dtype=np.float64))


def _array_stats(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    if not np.all(finite):
        raise RuntimeError("product state contains non-finite values")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(array.size),
        "finite_count": int(np.count_nonzero(finite)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean_float64": float(np.mean(array, dtype=np.float64)),
    }


def validate_downloaded_final(
    state: Any, saved: Any, capsule: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-hash the downloaded state+sidecar before any product path exists."""

    class _HostAtmosphere:
        pass

    host = _HostAtmosphere()
    host.state = state
    host.saved = saved
    snapshot = runner.fingerprint_atmosphere(host)
    capsule_final = capsule.get("trajectory", {}).get("final_snapshot_sha256")
    if capsule_final != EXPECTED_FINAL_SNAPSHOT_SHA256:
        raise RuntimeError("admitted replay capsule final digest changed")
    if snapshot.get("sha256") != capsule_final:
        raise RuntimeError(
            "downloaded final state+sidecar differs from the certified capsule: "
            f"{snapshot.get('sha256')} != {capsule_final}"
        )
    return snapshot


def load_partial_product_history(path: str | Path) -> dict[str, np.ndarray]:
    selected = Path(path).expanduser().resolve(strict=True)
    expected_variables = {
        "initial_time",
        "xtime",
        "Time",
        *PARTIAL_TIMED_FIELDS,
        *PARTIAL_STATIC_FIELDS,
    }
    result: dict[str, np.ndarray] = {}
    with Dataset(selected) as dataset:
        dataset.set_auto_mask(False)
        if set(dataset.dimensions) != {"Time", "StrLen", "nCells"}:
            raise RuntimeError("partial product history dimension inventory changed")
        if (
            len(dataset.dimensions["Time"]) != 2
            or len(dataset.dimensions["StrLen"]) != TIME_STRING_LENGTH
            or len(dataset.dimensions["nCells"]) != runner.TARGET_CELLS
        ):
            raise RuntimeError("partial product history dimension sizes changed")
        if set(dataset.variables) != expected_variables:
            raise RuntimeError("partial product history variable inventory changed")
        if getattr(dataset, "history_schema", None) != PARTIAL_HISTORY_SCHEMA:
            raise RuntimeError("partial product history schema changed")
        if getattr(dataset, "history_scope", None) != (
            "t0 and F006 two-dimensional product fields only"
        ):
            raise RuntimeError("partial product history scope changed")
        if getattr(dataset, "full_3d_state_archived", None) != (
            "false; certified by capsule hashes only"
        ):
            raise RuntimeError("partial product history makes a full-state claim")
        if str(getattr(dataset, "exact_partial_field_inventory", "")) != ",".join(
            (*PARTIAL_TIMED_FIELDS, *PARTIAL_STATIC_FIELDS)
        ):
            raise RuntimeError("partial product history field declaration changed")
        if tuple(dataset.variables["Time"].dimensions) != ("Time",):
            raise RuntimeError("partial product history Time dimensions changed")
        if tuple(dataset.variables["initial_time"].dimensions) != ("StrLen",):
            raise RuntimeError(
                "partial product history initial_time dimensions changed"
            )
        if tuple(dataset.variables["xtime"].dimensions) != ("Time", "StrLen"):
            raise RuntimeError("partial product history xtime dimensions changed")
        if str(dataset.variables["Time"].units) != (
            "seconds since 2026-03-26 00:00:00"
        ):
            raise RuntimeError("partial product history time units changed")
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["Time"][:], dtype=np.float64),
            np.asarray((0.0, TARGET_DURATION_SECONDS), dtype=np.float64),
        )
        initial_text = (
            np.asarray(dataset.variables["initial_time"][:])
            .tobytes()
            .decode("ascii")
            .strip()
        )
        if initial_text != "2026-03-26_00:00:00":
            raise RuntimeError("partial product history initial time changed")
        xtime = [
            np.asarray(row).tobytes().decode("ascii").strip()
            for row in np.asarray(dataset.variables["xtime"][:])
        ]
        if xtime != ["2026-03-26_00:00:00", "2026-03-26_06:00:00"]:
            raise RuntimeError("partial product history valid times changed")
        for name in PARTIAL_TIMED_FIELDS:
            variable = dataset.variables[name]
            if tuple(variable.dimensions) != ("Time", "nCells"):
                raise RuntimeError(f"partial history dimensions changed: {name}")
            values = np.asarray(variable[:])
            if values.shape != (2, runner.TARGET_CELLS) or not np.all(
                np.isfinite(values)
            ):
                raise RuntimeError(f"partial history field is invalid: {name}")
            result[name] = values
        for name in PARTIAL_STATIC_FIELDS:
            variable = dataset.variables[name]
            if tuple(variable.dimensions) != ("nCells",):
                raise RuntimeError(f"partial static dimensions changed: {name}")
            values = np.asarray(variable[:])
            if values.shape != (runner.TARGET_CELLS,) or not np.all(
                np.isfinite(values)
            ):
                raise RuntimeError(f"partial static field is invalid: {name}")
            result[name] = values
        for name in (
            "surface_pressure",
            "pressure_lowest_model_level",
            "temperature_lowest_model_level",
        ):
            if np.any(result[name] <= 0.0):
                raise RuntimeError(f"partial history positive field changed: {name}")
        if np.any(result["pressure_lowest_model_level"] > result["surface_pressure"]):
            raise RuntimeError(
                "partial history lowest pressure exceeds surface pressure"
            )
        if np.any(result["wind_speed_lowest_model_level"] < 0.0):
            raise RuntimeError("partial history wind speed became negative")
        if np.any(result["qv_lowest_model_level"] < 0.0):
            raise RuntimeError("partial history passive qv became negative")
        cell_ids = np.asarray(result["indexToCellID"], dtype=np.int64)
        if np.any(cell_ids <= 0) or np.unique(cell_ids).size != runner.TARGET_CELLS:
            raise RuntimeError("partial history cell identity changed")
    if selected.stat().st_size >= 100_000_000:
        raise RuntimeError("partial product history exceeds 100,000,000 bytes")
    return result


def load_exact_latlon_products(
    path: str | Path,
    expected: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    selected = Path(path).expanduser().resolve(strict=True)
    fields = (
        "surface_pressure",
        "pressure_lowest_model_level",
        "temperature_lowest_model_level",
        "u_lowest_model_level",
        "v_lowest_model_level",
        "wind_speed_lowest_model_level",
    )
    result: dict[str, np.ndarray] = {}
    with Dataset(selected) as dataset:
        dataset.set_auto_mask(False)
        if getattr(dataset, "product_schema", None) != LATLON_PRODUCT_SCHEMA:
            raise RuntimeError("0.5-degree product schema changed")
        if getattr(dataset, "product_scope", None) != LATLON_PRODUCT_SCOPE:
            raise RuntimeError("0.5-degree product scope changed")
        forbidden_history_attrs = {
            "history_schema",
            "history_scope",
            "full_3d_state_archived",
        }
        if forbidden_history_attrs.intersection(dataset.ncattrs()):
            raise RuntimeError("0.5-degree product falsely claims history metadata")
        expected_variables = {
            "lat",
            "lon",
            "latitude",
            "longitude",
            "Time",
            "xtime",
            *fields,
        }
        if set(dataset.variables) != expected_variables:
            raise RuntimeError("0.5-degree product variable inventory changed")
        if set(dataset.dimensions) != {"Time", "StrLen", "lat", "lon"}:
            raise RuntimeError("0.5-degree product dimension inventory changed")
        if (
            len(dataset.dimensions["Time"]) != 2
            or len(dataset.dimensions["StrLen"]) != TIME_STRING_LENGTH
            or len(dataset.dimensions["lat"]) != 361
            or len(dataset.dimensions["lon"]) != 720
        ):
            raise RuntimeError("0.5-degree product dimension sizes changed")
        latitude = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
        latitude_alias = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        longitude_alias = np.asarray(
            dataset.variables["longitude"][:], dtype=np.float64
        )
        if tuple(dataset.variables["latitude"].dimensions) != ("lat",) or tuple(
            dataset.variables["longitude"].dimensions
        ) != ("lon",):
            raise RuntimeError("0.5-degree coordinate alias dimensions changed")
        np.testing.assert_array_equal(latitude_alias, latitude)
        np.testing.assert_array_equal(longitude_alias, longitude)
        np.testing.assert_array_equal(
            latitude, np.arange(-90.0, 90.0 + 0.25, TARGET_LATLON_DEGREES)
        )
        if str(dataset.variables["Time"].units) != (
            "seconds since 2026-03-26 00:00:00"
        ):
            raise RuntimeError("0.5-degree product time units changed")
        xtime = [
            np.asarray(row).tobytes().decode("ascii").strip()
            for row in np.asarray(dataset.variables["xtime"][:])
        ]
        if xtime != ["2026-03-26_00:00:00", "2026-03-26_06:00:00"]:
            raise RuntimeError("0.5-degree product valid times changed")
        np.testing.assert_array_equal(
            longitude, np.arange(0.0, 360.0, TARGET_LATLON_DEGREES)
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["Time"][:], dtype=np.float64),
            np.asarray((0.0, TARGET_DURATION_SECONDS), dtype=np.float64),
        )
        result["lat"] = latitude
        result["lon"] = longitude
        for name in fields:
            variable = dataset.variables[name]
            if tuple(variable.dimensions) != ("Time", "lat", "lon"):
                raise RuntimeError(f"0.5-degree product dimensions changed: {name}")
            values = np.asarray(variable[:])
            if values.shape != (2, 361, 720) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"0.5-degree product field is invalid: {name}")
            if expected is not None:
                np.testing.assert_array_equal(values, expected[name])
            result[name] = values
    if selected.stat().st_size >= 100_000_000:
        raise RuntimeError("0.5-degree product exceeds 100,000,000 bytes")
    return result


def validate_partial_renderer_input(
    adapter_path: str | Path,
    *,
    history_path: str | Path,
    latlon_path: str | Path,
    regrid_weights_path: str | Path,
    grid_record: Mapping[str, Any],
    static_record: Mapping[str, Any],
) -> dict[str, Any]:
    adapter = Path(adapter_path).expanduser().resolve(strict=True)
    history = Path(history_path).expanduser().resolve(strict=True)
    latlon = Path(latlon_path).expanduser().resolve(strict=True)
    authority = Path(regrid_weights_path).expanduser().resolve(strict=True)
    partial = load_partial_product_history(history)
    weights = load_regrid_weights(authority)
    weights.validate_source(
        partial["latCell"], partial["lonCell"], source_units="radians"
    )
    if (
        weights.method != "inverse_distance"
        or weights.n_neighbors != 4
        or weights.power != 2.0
        or weights.source_count != runner.TARGET_CELLS
        or weights.evidence != REGRID_EVIDENCE
    ):
        raise RuntimeError("partial adapter regrid authority changed")
    np.testing.assert_array_equal(
        weights.target_latitude,
        np.arange(-90.0, 90.0 + 0.25, TARGET_LATLON_DEGREES),
    )
    np.testing.assert_array_equal(
        weights.target_longitude,
        np.arange(0.0, 360.0, TARGET_LATLON_DEGREES),
    )
    regridded = {
        name: weights.apply(partial[name], cell_axis=1)
        for name in (
            "surface_pressure",
            "pressure_lowest_model_level",
            "temperature_lowest_model_level",
            "u_lowest_model_level",
            "v_lowest_model_level",
            "wind_speed_lowest_model_level",
        )
    }
    latlon_fields = load_exact_latlon_products(latlon, regridded)
    expected_terrain = weights.apply(partial["terrain_height"])
    expected_height = weights.apply(partial["height_lowest_model_level"])
    validate_rust_wrf2d_netcdf(adapter)
    grid_pin = runner.INPUT_PINS["x1_163842_grid"]
    static_pin = runner.INPUT_PINS["x1_163842_static"]
    if dict(grid_record) != {"bytes": grid_pin.bytes, "sha256": grid_pin.sha256}:
        raise RuntimeError("partial adapter grid input record changed")
    if dict(static_record) != {
        "bytes": static_pin.bytes,
        "sha256": static_pin.sha256,
    }:
        raise RuntimeError("partial adapter static input record changed")
    with Dataset(adapter) as dataset:
        dataset.set_auto_mask(False)
        expected_attrs = {
            "partial_materialization_schema": (
                "mpas-port.partial-unstructured-product-to-rust-wrf2d/v1"
            ),
            "partial_history_field_inventory": ",".join(
                (*PARTIAL_TIMED_FIELDS, *PARTIAL_STATIC_FIELDS)
            ),
            "full_3d_state_archived": "false; certified by capsule hashes only",
            "source_history_name": history.name,
            "source_history_sha256": sha256_file(history),
            "source_latlon_name": latlon.name,
            "source_latlon_sha256": sha256_file(latlon),
            "source_grid_name": grid_pin.filename,
            "source_grid_sha256": grid_pin.sha256,
            "source_static_name": static_pin.filename,
            "source_static_sha256": static_pin.sha256,
            "partial_regrid_weights_name": authority.name,
            "partial_regrid_weights_sha256": sha256_file(authority),
            "timestep_interpretation": TIMESTEP_LABEL,
            "physics": PHYSICS_LABEL,
        }
        for name, expected in expected_attrs.items():
            if getattr(dataset, name, None) != expected:
                raise RuntimeError(f"partial adapter provenance changed: {name}")
        latitude_grid, longitude_grid = np.meshgrid(
            latlon_fields["lat"], latlon_fields["lon"], indexing="ij"
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["XLAT"][:]), latitude_grid
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["XLONG"][:]), longitude_grid
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["time"][:], dtype=np.float64),
            np.asarray((0.0, TARGET_DURATION_SECONDS), dtype=np.float64),
        )
        mappings = {
            "surface_pressure": "surface_pressure",
            "temperature_lowest_model_level": "temperature_lowest_model_level",
            "u_lowest_model_level": "u_lowest_model_level",
            "v_lowest_model_level": "v_lowest_model_level",
            "wind_speed_lowest_model_level": "wind_speed_lowest_model_level",
        }
        for adapter_name, latlon_name in mappings.items():
            np.testing.assert_array_equal(
                np.asarray(dataset.variables[adapter_name][:]),
                latlon_fields[latlon_name],
            )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["P"][:]),
            latlon_fields["pressure_lowest_model_level"],
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["PSFC"][:]),
            latlon_fields["surface_pressure"],
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["TK"][:]),
            latlon_fields["temperature_lowest_model_level"],
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["HGT"][:]), expected_terrain
        )
        np.testing.assert_array_equal(
            np.asarray(dataset.variables["Z"][:]), expected_height
        )
    return {
        "schema": "mpas-port.partial-renderer-input-binding/v1",
        "status": "passed",
        "history_sha256": sha256_file(history),
        "latlon_sha256": sha256_file(latlon),
        "regrid_weights_sha256": sha256_file(authority),
        "adapter_sha256": sha256_file(adapter),
        "plotted_latlon_fields_exact": sorted(
            {
                "surface_pressure",
                "temperature_lowest_model_level",
                "u_lowest_model_level",
                "v_lowest_model_level",
                "wind_speed_lowest_model_level",
            }
        ),
        "pressure_lowest_model_level_exact": True,
        "terrain_and_lowest_height_exact": True,
    }


def write_products(
    case: Any,
    arm: Any,
    config: Any,
    bounds: StabilityBounds,
    *,
    plan: OutputPlan,
) -> dict[str, Any]:
    """Write truthful products after :func:`certify_replay_capsule` passes."""

    download = runner.coarse.download_final_atmosphere(arm.final_atmosphere)
    initial_state = case.cuda.state
    initial_saved = case.cuda.saved_diagnostics
    final_state = download.state
    final_saved = download.saved
    downloaded_snapshot = validate_downloaded_final(
        final_state, final_saved, arm.capsule
    )
    if float(final_state.time_seconds) != TARGET_DURATION_SECONDS:
        raise RuntimeError("replayed final model time differs from six hours")

    diagnostics = DryDycoreDriver(
        case.cuda.mesh,
        case.cuda.vertical,
        case.cuda.reference,
        config,
        terrain_metrics=case.cuda.terrain_metrics,
    )
    initial_metrics = asdict(diagnostics.metrics(initial_state))
    final_metrics = asdict(diagnostics.metrics(final_state))
    mass_drift, energy_drift = runner.coarse.cpu_gfs.check_bounds(
        final_metrics,
        initial_metrics,
        bounds,
        np.asarray(final_state.scalars[0]),
    )
    initial_qv_mass = _qv_mass(case, initial_state)
    final_qv_mass = _qv_mass(case, final_state)
    qv_mass_drift = abs(final_qv_mass - initial_qv_mass) / max(
        abs(initial_qv_mass), np.finfo(np.float64).tiny
    )
    if not np.isfinite(qv_mass_drift) or qv_mass_drift > 2.0e-8:
        raise RuntimeError(f"passive-qv mass drift exceeds 2e-8: {qv_mass_drift}")

    coefficients = initialize_reconstruction_coefficients(case.cuda.mesh)
    initial_products = runner.coarse.cpu_gfs.diagnose_products(
        initial_state,
        initial_saved,
        case.cuda.mesh,
        case.cuda.vertical,
        coefficients,
    )
    final_products = runner.coarse.cpu_gfs.diagnose_products(
        final_state,
        final_saved,
        case.cuda.mesh,
        case.cuda.vertical,
        coefficients,
    )
    initial_time = datetime.fromisoformat(
        str(case.source_provenance["valid_time"]).replace("_", "T", 1)
    )
    final_time = initial_time + timedelta(seconds=TARGET_DURATION_SECONDS)

    plan.artifact_root.mkdir(parents=True)
    pressure_lowest = np.stack(
        (initial_products["pressure"][0], final_products["pressure"][0])
    )
    qv_lowest = np.stack((initial_state.scalars[0, 0], final_state.scalars[0, 0]))
    terrain_height = np.asarray(case.cuda.mesh.ter)
    height_lowest = 0.5 * (
        np.asarray(case.cuda.vertical.zgrid[0], dtype=np.float64)
        + np.asarray(case.cuda.vertical.zgrid[1], dtype=np.float64)
    )
    history_fields = {
        "surface_pressure": HistoryField(
            np.stack(
                (
                    initial_products["surface_pressure"],
                    final_products["surface_pressure"],
                )
            ),
            ("Time", "nCells"),
        ),
        "temperature_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["temperature_lowest_model_level"],
                    final_products["temperature_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "K"},
        ),
        "pressure_lowest_model_level": HistoryField(
            pressure_lowest,
            ("Time", "nCells"),
            {"units": "Pa", "long_name": "pressure at lowest model level"},
        ),
        "u_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["u_lowest_model_level"],
                    final_products["u_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "m s-1", "long_name": "zonal wind at lowest model level"},
        ),
        "v_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["v_lowest_model_level"],
                    final_products["v_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "m s-1", "long_name": "meridional wind at lowest model level"},
        ),
        "wind_speed_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["wind_speed_lowest_model_level"],
                    final_products["wind_speed_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "m s-1"},
        ),
        "qv_lowest_model_level": HistoryField(
            qv_lowest,
            ("Time", "nCells"),
            {
                "units": "kg kg-1",
                "long_name": "passively transported qv at lowest model level",
            },
        ),
    }
    base_product_attrs = {
        "model": "MPAS-Atmosphere NumPy/CUDA dry port",
        "initial_condition": "real GFS 2026-03-26 00Z",
        "mesh": "official x1.163842 approximately 60 km variable-resolution-capable topology",
        "physics_suite": "none",
        "water_vapor_treatment": "passive scalar transport only",
        "certification": (
            "fresh arm complete capsule byte-equal to committed hard-gated "
            "dual-run arm before product writing"
        ),
        "timestep": TIMESTEP_LABEL,
        "forecast_skill_claimed": "false",
    }
    write_history(
        plan.history,
        _n_cells_only_history_mesh(case.output_mesh),
        history_fields,
        (initial_time, final_time),
        initial_time=initial_time,
        time_seconds=(0.0, TARGET_DURATION_SECONDS),
        global_attrs={
            **base_product_attrs,
            "title": "Certified-replay x1.163842 partial 2-D product history",
            "history_schema": PARTIAL_HISTORY_SCHEMA,
            "history_scope": "t0 and F006 two-dimensional product fields only",
            "full_3d_state_archived": ("false; certified by capsule hashes only"),
            "surface_height_semantics": "lowest model level; never 2 m or 10 m",
        },
        include_mesh=False,
        stream_options=HistoryStreamOptions(clobber_mode="truncate"),
    )
    with Dataset(plan.history, "a") as dataset:
        static_values = {
            "indexToCellID": np.asarray(case.output_mesh.indexToCellID, dtype=np.int64),
            "latCell": np.asarray(case.output_mesh.latCell, dtype=np.float64),
            "lonCell": np.asarray(case.output_mesh.lonCell, dtype=np.float64),
            "terrain_height": terrain_height,
            "height_lowest_model_level": height_lowest,
        }
        static_attrs = {
            "indexToCellID": {"long_name": "global cell identifier"},
            "latCell": {"units": "radian", "long_name": "cell center latitude"},
            "lonCell": {"units": "radian", "long_name": "cell center longitude"},
            "terrain_height": {
                "units": "m",
                "long_name": "unsmoothed static terrain height above mean sea level",
            },
            "height_lowest_model_level": {
                "units": "m",
                "long_name": (
                    "runtime binary32 zgrid[0:2] midpoint at lowest model level"
                ),
            },
        }
        for name in PARTIAL_STATIC_FIELDS:
            values = static_values[name]
            dtype = "i8" if values.dtype.kind in "iu" else "f8"
            variable = dataset.createVariable(name, dtype, ("nCells",))
            variable.setncatts(static_attrs[name])
            variable[:] = values
        dataset.setncattr(
            "exact_partial_field_inventory",
            ",".join((*PARTIAL_TIMED_FIELDS, *PARTIAL_STATIC_FIELDS)),
        )
        dataset.setncattr("terrain_smoothing_passes", 1)
        dataset.setncattr("vertical_surface_smoothing", 0)
        dataset.setncattr(
            "height_terrain_relation",
            "Z uses the runtime binary32 one-pass-smoothed MPAS terrain coordinate; "
            "terrain_height preserves mesh.ter and Z-HGT is not constrained positive",
        )
    if plan.history.stat().st_size >= 100_000_000:
        raise RuntimeError("partial product history must stay below 100,000,000 bytes")
    partial_history = load_partial_product_history(plan.history)
    for name in PARTIAL_TIMED_FIELDS:
        np.testing.assert_array_equal(
            partial_history[name], np.asarray(history_fields[name].values)
        )
    for name in PARTIAL_STATIC_FIELDS:
        np.testing.assert_array_equal(partial_history[name], static_values[name])

    latitude = np.arange(
        -90.0, 90.0 + 0.5 * TARGET_LATLON_DEGREES, TARGET_LATLON_DEGREES
    )
    longitude = np.arange(0.0, 360.0, TARGET_LATLON_DEGREES)
    weights = build_regrid_weights(
        case.output_mesh,
        target_latitude=latitude,
        target_longitude=longitude,
        method="inverse_distance",
        neighbors=4,
        power=2.0,
    )
    save_regrid_weights(weights, plan.regrid_weights, overwrite=False)
    if plan.regrid_weights.stat().st_size >= 100_000_000:
        raise RuntimeError("regrid weights must stay below 100,000,000 bytes")
    weights = load_regrid_weights(plan.regrid_weights)
    weights.validate_source(case.output_mesh)
    if (
        weights.method != "inverse_distance"
        or weights.n_neighbors != 4
        or weights.power != 2.0
    ):
        raise RuntimeError("reloaded compact regrid authority changed algorithm")
    np.testing.assert_array_equal(weights.target_latitude, latitude)
    np.testing.assert_array_equal(weights.target_longitude, longitude)
    regrid_fields: dict[str, HistoryField] = {}
    for name in (
        "surface_pressure",
        "pressure_lowest_model_level",
        "temperature_lowest_model_level",
        "u_lowest_model_level",
        "v_lowest_model_level",
        "wind_speed_lowest_model_level",
    ):
        units = (
            "Pa"
            if name in {"surface_pressure", "pressure_lowest_model_level"}
            else "K"
            if name.startswith("temperature")
            else "m s-1"
        )
        native_values = partial_history[name]
        regrid_fields[name] = HistoryField(
            native_values,
            ("Time", "nCells"),
            {"units": units},
        )
    write_regridded_netcdf(
        plan.latlon,
        weights,
        regrid_fields,
        cell_axis={name: 1 for name in regrid_fields},
        valid_time=(initial_time, final_time),
        initial_time=initial_time,
        global_attrs={
            **base_product_attrs,
            "title": "Certified-replay x1.163842 CUDA dry products on 0.5 degree grid",
            "product_schema": LATLON_PRODUCT_SCHEMA,
            "product_scope": LATLON_PRODUCT_SCOPE,
            "target_grid_resolution_degrees": TARGET_LATLON_DEGREES,
        },
        clobber=True,
    )
    authority_sha = sha256_file(plan.regrid_weights)
    expected_regridded = {
        name: weights.apply(field.values, cell_axis=1)
        for name, field in regrid_fields.items()
    }
    latlon_products = load_exact_latlon_products(plan.latlon, expected_regridded)
    regridded_terrain = weights.apply(partial_history["terrain_height"])
    regridded_height = weights.apply(partial_history["height_lowest_model_level"])
    adapter_attrs = {
        "partial_materialization_schema": (
            "mpas-port.partial-unstructured-product-to-rust-wrf2d/v1"
        ),
        "partial_history_field_inventory": ",".join(
            (*PARTIAL_TIMED_FIELDS, *PARTIAL_STATIC_FIELDS)
        ),
        "full_3d_state_archived": "false; certified by capsule hashes only",
        "source_history_name": plan.history.name,
        "source_history_sha256": sha256_file(plan.history),
        "source_latlon_name": plan.latlon.name,
        "source_latlon_sha256": sha256_file(plan.latlon),
        "source_grid_name": case.grid_path.name,
        "source_grid_sha256": case.input_records["x1_163842_grid"]["sha256"],
        "source_static_name": case.static_path.name,
        "source_static_sha256": case.input_records["x1_163842_static"]["sha256"],
        "partial_regrid_weights_name": plan.regrid_weights.name,
        "partial_regrid_weights_sha256": authority_sha,
        "vertical_levels": TARGET_LEVELS,
        "vertical_top_m": TARGET_ZTOP_M,
        "terrain_smoothing_passes": 1,
        "vertical_surface_smoothing": 0,
        "height_terrain_relation": (
            "Z uses the one-pass-smoothed MPAS terrain coordinate; HGT preserves "
            "the unsmoothed static-file terrain, so Z-HGT is not constrained positive"
        ),
        "wind_speed_regrid_semantics": (
            "the native-cell scalar wind speed is regridded independently of the "
            "native-cell u/v components"
        ),
        "regrid_algorithm": "spherical inverse-distance, k=4, power=2",
        "regrid_evidence": weights.evidence,
        "regrid_source_fingerprint": weights.source_fingerprint,
        "timestep_interpretation": TIMESTEP_LABEL,
        "physics": PHYSICS_LABEL,
    }
    fields = RustWrf2dFields(
        latitude=latitude,
        longitude=longitude,
        valid_times=(initial_time, final_time),
        initial_time=initial_time,
        temperature_lowest_model_level=latlon_products[
            "temperature_lowest_model_level"
        ],
        pressure_lowest_model_level=latlon_products["pressure_lowest_model_level"],
        height_lowest_model_level=regridded_height,
        surface_pressure=latlon_products["surface_pressure"],
        terrain_height=regridded_terrain,
        u_lowest_model_level=latlon_products["u_lowest_model_level"],
        v_lowest_model_level=latlon_products["v_lowest_model_level"],
        wind_speed_lowest_model_level=latlon_products["wind_speed_lowest_model_level"],
        global_attrs=adapter_attrs,
    )
    write_rust_wrf2d_netcdf(plan.renderer_input, fields, clobber=False)
    validate_rust_wrf2d_netcdf(
        plan.renderer_input,
        materialized_sources={
            "history": plan.history,
            "latlon": plan.latlon,
            "grid": case.grid_path,
            "static": case.static_path,
        },
        require_materialized_provenance=True,
    )
    adapter_binding = validate_partial_renderer_input(
        plan.renderer_input,
        history_path=plan.history,
        latlon_path=plan.latlon,
        regrid_weights_path=plan.regrid_weights,
        grid_record=case.input_records["x1_163842_grid"],
        static_record=case.input_records["x1_163842_static"],
    )
    if plan.renderer_input.stat().st_size >= 100_000_000:
        raise RuntimeError("Rust renderer adapter must stay below 100,000,000 bytes")
    return {
        "download": {"bytes": download.bytes, "seconds": download.seconds},
        "downloaded_final_snapshot": downloaded_snapshot,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mass_relative_drift": mass_drift,
        "energy_proxy_relative_drift": energy_drift,
        "qv_mass_initial": initial_qv_mass,
        "qv_mass_final": final_qv_mass,
        "qv_mass_relative_drift": qv_mass_drift,
        "initial_state": {
            name: _array_stats(getattr(initial_state, name))
            for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        },
        "final_state": {
            name: _array_stats(getattr(final_state, name))
            for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        },
        "surface_pressure_change_pa": _array_stats(
            final_products["surface_pressure"] - initial_products["surface_pressure"]
        ),
        "partial_product_history": {
            "schema": PARTIAL_HISTORY_SCHEMA,
            "field_inventory": [*PARTIAL_TIMED_FIELDS, *PARTIAL_STATIC_FIELDS],
            "records": ["t0", "F006"],
            "full_3d_state_archived": False,
            "full_3d_state_certification": (
                "complete per-step capsule hashes; final state+sidecar re-hashed "
                "after download"
            ),
        },
        "partial_renderer_input_binding": adapter_binding,
        "history": _record(plan.history),
        "latlon": _record(plan.latlon),
        "regrid_weights": _record(plan.regrid_weights),
        "renderer_input": _record(plan.renderer_input),
    }


def _write_checksums(path: Path, targets: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for target in targets:
        record = _record(target)
        lines.append(f"{record['sha256']}  {record['path']}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _assert_exact_file_tree(root: Path, expected_names: set[str], label: str) -> None:
    selected = root.resolve(strict=True)
    actual: set[str] = set()
    for entry in selected.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(f"{label} contains a symlink: {entry}")
        if entry.is_file():
            actual.add(entry.relative_to(selected).as_posix())
    if actual != expected_names:
        raise RuntimeError(
            f"{label} file tree changed: "
            f"missing={sorted(expected_names - actual)}, "
            f"extra={sorted(actual - expected_names)}"
        )


def _validate_checksum_inventory(path: Path, targets: Sequence[Path]) -> None:
    selected = path.resolve(strict=True)
    rows: dict[str, str] = {}
    for line_number, line in enumerate(
        selected.read_text(encoding="ascii").splitlines(), 1
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RuntimeError(f"invalid checksum row {line_number}: {selected}")
        digest, logical = parts
        if logical in rows:
            raise RuntimeError(f"duplicate checksum target: {logical}")
        rows[logical] = digest
    expected: dict[str, str] = {}
    for target in targets:
        record = _record(target)
        expected[record["path"]] = record["sha256"]
    if rows != expected:
        raise RuntimeError("checksum target set or digest changed")


def write_product_receipt(
    case: Any,
    config: Any,
    certified: CertifiedEvidence,
    replay_gate: Mapping[str, Any],
    products: Mapping[str, Any],
    plan: OutputPlan,
    *,
    timing: Mapping[str, float],
    runtime_source_pins: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assert_runtime_sources_unchanged(runtime_source_pins)
    certified_targets = [
        certified.root / "inventory.json",
        *(certified.root / logical for logical in _CERTIFIED_FILE_PINS),
    ]
    checksum_targets = (
        *certified_targets,
        plan.replay_capsule,
        plan.replay_gate,
        plan.history,
        plan.latlon,
        plan.regrid_weights,
        plan.renderer_input,
        plan.receipt,
    )
    artifact_tree = {
        plan.history.name,
        plan.latlon.name,
        plan.regrid_weights.name,
        plan.renderer_input.name,
    }
    receipt_tree = {
        plan.replay_capsule.name,
        plan.replay_gate.name,
        plan.receipt.name,
        plan.checksums.name,
    }
    payload = {
        "schema": "mpas-port.x1.163842-stabilized-product-replay/v1",
        "status": "passed",
        "classification": PRODUCT_CLASSIFICATION,
        "claims": dict(PRODUCT_CLAIMS),
        "configuration": {
            "value": asdict(config),
            "sha256": EXPECTED_CONFIG_SHA256,
            "timestep_interpretation": TIMESTEP_LABEL,
        },
        "physics": {
            "suite": "none",
            "column_backend_executed": False,
            "qv_treatment": "passive scalar transport only",
        },
        "certified_authority": {
            "root": certified.root.relative_to(ROOT).as_posix(),
            "files": certified.files,
            "hard_gated_scout_status": certified.summary["scout"]["status"],
            "gpuwm_total_comparison": True,
            "gpuwm_identical": True,
            "committed_ftz_and_scout_validator": {
                "source": dict(runtime_source_pins["remote_evidence_validator"]),
                "verdict": certified.authority_verdict,
            },
        },
        "replay_gate": dict(replay_gate),
        "inputs": case.input_records,
        "preparation": {
            "capsule_identity_metadata_preserved_for_byte_replay": {
                "profile": CERTIFIED_CAPSULE_PROFILE,
                "target": CERTIFIED_CAPSULE_TARGET,
                "method": CERTIFIED_CAPSULE_PREPARATION_METHOD,
                "interpretation": (
                    "historical certified-capsule identity only; the fresh "
                    "product replay did not rerun the scout or two arms"
                ),
            },
            "actual_fresh_replay_method": ACTUAL_FRESH_REPLAY_METHOD,
            "fresh_independent_cuda_uploads": 1,
            "fresh_scout_executed": False,
            "execution_fingerprint": case.cuda.expected_execution_fingerprint,
            "layout": "[level, entity], C-contiguous, horizontal entity fastest",
            "smooth_surfaces": SMOOTH_SURFACES,
        },
        "observed": {
            key: value
            for key, value in products.items()
            if key
            not in {
                "history",
                "latlon",
                "regrid_weights",
                "renderer_input",
            }
        },
        "artifacts": {
            "replay_capsule": _record(plan.replay_capsule),
            "replay_gate": _record(plan.replay_gate),
            "partial_product_history": products["history"],
            "latlon_0p5_degree": products["latlon"],
            "regrid_weights": products["regrid_weights"],
            "rust_renderer_input": products["renderer_input"],
        },
        "runner": {
            "path": RUNNER_PATH.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_RUNNER_SHA256,
        },
        "runtime_source_pins": {
            label: dict(record) for label, record in sorted(runtime_source_pins.items())
        },
        "publication_contract": {
            "artifact_root": plan.artifact_root.relative_to(ROOT).as_posix(),
            "artifact_tree_exact_files": sorted(artifact_tree),
            "receipt_root": plan.receipt_root.relative_to(ROOT).as_posix(),
            "receipt_tree_exact_files": sorted(receipt_tree),
            "checksum_inventory": plan.checksums.relative_to(ROOT).as_posix(),
            "checksum_exact_targets": [
                Path(target).resolve(strict=True).relative_to(ROOT).as_posix()
                for target in checksum_targets
                if Path(target) != plan.receipt
            ]
            + [plan.receipt.relative_to(ROOT).as_posix()],
        },
        "timing_seconds": dict(timing),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    assert_runtime_sources_unchanged(runtime_source_pins)
    write_json_atomic(plan.receipt, payload)
    assert_runtime_sources_unchanged(runtime_source_pins)
    _write_checksums(
        plan.checksums,
        checksum_targets,
    )
    _assert_exact_file_tree(plan.artifact_root, artifact_tree, "product artifact root")
    _assert_exact_file_tree(plan.receipt_root, receipt_tree, "product receipt root")
    _validate_checksum_inventory(plan.checksums, checksum_targets)
    for path in (
        plan.history,
        plan.latlon,
        plan.regrid_weights,
        plan.renderer_input,
        plan.replay_capsule,
        plan.replay_gate,
        plan.receipt,
        plan.checksums,
    ):
        if path.stat().st_size >= 100_000_000:
            raise RuntimeError(f"published product exceeds 100,000,000 bytes: {path}")
    assert_runtime_sources_unchanged(runtime_source_pins)
    return payload


def build_render_plan(
    output_root: str | Path,
    store_root: str | Path,
    receipt_root: str | Path,
) -> RenderPlan:
    outputs = _require_beneath(
        output_root, ROOT / "artifacts" / "cuda-gfs", "render output root"
    )
    store = _require_beneath(store_root, ROOT / "work", "renderer store root")
    receipts = _require_beneath(
        receipt_root,
        ROOT / "receipts" / "cuda-gfs-forecast",
        "render receipt root",
    )
    _require_disjoint(outputs, store, "render outputs/store")
    _require_disjoint(outputs, receipts, "render outputs/receipts")
    _require_disjoint(store, receipts, "render store/receipts")
    return RenderPlan(
        output_root=outputs,
        store_root=store,
        receipt_root=receipts,
        receipt=receipts / "renderer.json",
        checksums=receipts / "SHA256SUMS",
    )


def _contained_existing_file(path: str | Path, boundary: Path, label: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink")
    selected = unresolved.resolve(strict=True)
    parent = boundary.resolve(strict=True)
    if not selected.is_file() or not selected.is_relative_to(parent):
        raise RuntimeError(f"{label} must be a file beneath {parent}: {selected}")
    return selected


def _validated_record_path(
    record: Any,
    *,
    boundary: Path,
    label: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} record is missing")
    if record.get("path_kind") != "repo_relative":
        raise RuntimeError(f"{label} is not repository-relative")
    logical = str(record.get("path", ""))
    selected = _contained_existing_file(ROOT / logical, boundary, label)
    if _record(selected) != dict(record):
        raise RuntimeError(f"{label} bytes changed after product publication")
    return selected


def validate_product_publication(
    product_receipt_path: str | Path,
    renderer_input_path: str | Path,
) -> dict[str, Any]:
    """Strictly re-adjudicate a product replay before local Rust rendering."""

    receipt_file = _contained_existing_file(
        product_receipt_path,
        ROOT / "receipts" / "cuda-gfs-forecast",
        "product receipt",
    )
    renderer_input = _contained_existing_file(
        renderer_input_path,
        ROOT / "artifacts" / "cuda-gfs",
        "renderer input",
    )
    receipt = _load_json_bytes(receipt_file.read_bytes(), label=receipt_file.name)
    if (
        receipt.get("schema") != ("mpas-port.x1.163842-stabilized-product-replay/v1")
        or receipt.get("status") != "passed"
    ):
        raise RuntimeError("renderer requires a passed exact product-replay receipt")
    if receipt.get("classification") != PRODUCT_CLASSIFICATION:
        raise RuntimeError("product receipt classification changed")
    if receipt.get("runner") != {
        "path": RUNNER_PATH.relative_to(ROOT).as_posix(),
        "sha256": EXPECTED_RUNNER_SHA256,
    }:
        raise RuntimeError("product receipt runner binding changed")
    claims = receipt.get("claims", {})
    if claims != PRODUCT_CLAIMS:
        raise RuntimeError("product receipt claims changed")
    config = canonical_stabilized_config()
    configuration = receipt.get("configuration", {})
    if (
        configuration.get("sha256") != EXPECTED_CONFIG_SHA256
        or configuration.get("value") != asdict(config)
        or configuration.get("timestep_interpretation") != TIMESTEP_LABEL
    ):
        raise RuntimeError("product receipt configuration changed")
    physics = receipt.get("physics", {})
    if physics != {
        "suite": "none",
        "column_backend_executed": False,
        "qv_treatment": "passive scalar transport only",
    }:
        raise RuntimeError("product receipt physics statement changed")

    certified_authority = receipt.get("certified_authority", {})
    certified = validate_certified_evidence(DEFAULT_CERTIFIED_ROOT)
    runtime_pins = receipt.get("runtime_source_pins")
    if not isinstance(runtime_pins, Mapping):
        raise RuntimeError("product receipt runtime source pins are missing")
    assert_runtime_sources_unchanged(runtime_pins)
    expected_certified_authority = {
        "root": DEFAULT_CERTIFIED_ROOT.relative_to(ROOT).as_posix(),
        "files": certified.files,
        "hard_gated_scout_status": certified.summary["scout"]["status"],
        "gpuwm_total_comparison": True,
        "gpuwm_identical": True,
        "committed_ftz_and_scout_validator": {
            "source": dict(runtime_pins["remote_evidence_validator"]),
            "verdict": certified.authority_verdict,
        },
    }
    if certified_authority != expected_certified_authority:
        raise RuntimeError("product receipt certified authority changed")

    publication = receipt.get("publication_contract", {})
    artifact_root = _require_beneath(
        ROOT / str(publication.get("artifact_root", "")),
        ROOT / "artifacts" / "cuda-gfs",
        "published product artifact root",
    ).resolve(strict=True)
    receipt_root = _require_beneath(
        ROOT / str(publication.get("receipt_root", "")),
        ROOT / "receipts" / "cuda-gfs-forecast",
        "published product receipt root",
    ).resolve(strict=True)
    if receipt_file.parent != receipt_root:
        raise RuntimeError("product receipt is outside its declared exact tree")
    artifact_names = set(publication.get("artifact_tree_exact_files", []))
    receipt_names = set(publication.get("receipt_tree_exact_files", []))
    if len(artifact_names) != 4 or len(receipt_names) != 4:
        raise RuntimeError("product publication exact-tree declaration changed")
    _assert_exact_file_tree(artifact_root, artifact_names, "product artifact root")
    _assert_exact_file_tree(receipt_root, receipt_names, "product receipt root")

    artifacts = receipt.get("artifacts", {})
    expected_artifact_keys = {
        "replay_capsule",
        "replay_gate",
        "partial_product_history",
        "latlon_0p5_degree",
        "regrid_weights",
        "rust_renderer_input",
    }
    if set(artifacts) != expected_artifact_keys:
        raise RuntimeError("product receipt artifact inventory changed")
    paths = {
        "replay_capsule": _validated_record_path(
            artifacts["replay_capsule"],
            boundary=receipt_root,
            label="replay capsule",
        ),
        "replay_gate": _validated_record_path(
            artifacts["replay_gate"], boundary=receipt_root, label="replay gate"
        ),
        "history": _validated_record_path(
            artifacts["partial_product_history"],
            boundary=artifact_root,
            label="partial product history",
        ),
        "latlon": _validated_record_path(
            artifacts["latlon_0p5_degree"],
            boundary=artifact_root,
            label="0.5-degree NetCDF",
        ),
        "regrid_weights": _validated_record_path(
            artifacts["regrid_weights"],
            boundary=artifact_root,
            label="compact regrid weights",
        ),
        "renderer_input": _validated_record_path(
            artifacts["rust_renderer_input"],
            boundary=artifact_root,
            label="Rust renderer input",
        ),
    }
    if len(set(paths.values())) != len(paths):
        raise RuntimeError("product artifact records are not distinct")
    expected_artifact_names = {
        paths["history"].name,
        paths["latlon"].name,
        paths["regrid_weights"].name,
        paths["renderer_input"].name,
    }
    expected_receipt_names = {
        paths["replay_capsule"].name,
        paths["replay_gate"].name,
        receipt_file.name,
        "SHA256SUMS",
    }
    if artifact_names != expected_artifact_names:
        raise RuntimeError("product artifact exact-tree names are false")
    if receipt_names != expected_receipt_names:
        raise RuntimeError("product receipt exact-tree names are false")
    if paths["renderer_input"] != renderer_input:
        raise RuntimeError("requested renderer input differs from product receipt")

    capsule = _load_json_bytes(
        paths["replay_capsule"].read_bytes(), label=paths["replay_capsule"].name
    )
    expected_gate = certify_replay_capsule(capsule, certified)
    gate = _load_json_bytes(
        paths["replay_gate"].read_bytes(), label=paths["replay_gate"].name
    )
    if gate != expected_gate or receipt.get("replay_gate") != expected_gate:
        raise RuntimeError("product receipt replay firewall is false")
    trajectory = capsule.get("trajectory", {})
    step_records = trajectory.get("step_records", [])
    if not isinstance(step_records, list) or len(step_records) != TARGET_STEPS:
        raise RuntimeError("product replay capsule step inventory changed")
    expected_final_snapshot = step_records[-1].get("snapshot")
    observed_snapshot = receipt.get("observed", {}).get("downloaded_final_snapshot", {})
    if (
        not isinstance(expected_final_snapshot, Mapping)
        or observed_snapshot != expected_final_snapshot
        or observed_snapshot.get("sha256") != EXPECTED_FINAL_SNAPSHOT_SHA256
    ):
        raise RuntimeError("product receipt downloaded final snapshot changed")
    expected_execution_fingerprint = trajectory.get("initial_execution_fingerprint")
    if not isinstance(expected_execution_fingerprint, Mapping) or (
        expected_execution_fingerprint.get("sha256")
        != certified.summary.get("host_execution_seal_sha256")
    ):
        raise RuntimeError("product receipt host execution seal changed")
    expected_preparation = {
        "capsule_identity_metadata_preserved_for_byte_replay": {
            "profile": CERTIFIED_CAPSULE_PROFILE,
            "target": CERTIFIED_CAPSULE_TARGET,
            "method": CERTIFIED_CAPSULE_PREPARATION_METHOD,
            "interpretation": (
                "historical certified-capsule identity only; the fresh "
                "product replay did not rerun the scout or two arms"
            ),
        },
        "actual_fresh_replay_method": ACTUAL_FRESH_REPLAY_METHOD,
        "fresh_independent_cuda_uploads": 1,
        "fresh_scout_executed": False,
        "execution_fingerprint": expected_execution_fingerprint,
        "layout": "[level, entity], C-contiguous, horizontal entity fastest",
        "smooth_surfaces": SMOOTH_SURFACES,
    }
    if receipt.get("preparation") != expected_preparation:
        raise RuntimeError("product receipt replay preparation changed")
    inputs = receipt.get("inputs", {})
    if inputs != capsule.get("input_bytes"):
        raise RuntimeError("product receipt input byte inventory changed")
    partial_binding = validate_partial_renderer_input(
        paths["renderer_input"],
        history_path=paths["history"],
        latlon_path=paths["latlon"],
        regrid_weights_path=paths["regrid_weights"],
        grid_record=inputs.get("x1_163842_grid", {}),
        static_record=inputs.get("x1_163842_static", {}),
    )
    if receipt.get("observed", {}).get("partial_renderer_input_binding") != (
        partial_binding
    ):
        raise RuntimeError("product receipt partial adapter binding changed")

    checksum_file = _contained_existing_file(
        ROOT / str(publication.get("checksum_inventory", "")),
        receipt_root,
        "product checksum inventory",
    )
    declared_targets = publication.get("checksum_exact_targets")
    expected_target_paths = {
        (DEFAULT_CERTIFIED_ROOT / "inventory.json").relative_to(ROOT).as_posix(),
        *(
            (DEFAULT_CERTIFIED_ROOT / logical).relative_to(ROOT).as_posix()
            for logical in _CERTIFIED_FILE_PINS
        ),
        *(path.relative_to(ROOT).as_posix() for path in paths.values()),
        receipt_file.relative_to(ROOT).as_posix(),
    }
    if (
        not isinstance(declared_targets, list)
        or len(declared_targets) != len(set(declared_targets))
        or set(declared_targets) != expected_target_paths
    ):
        raise RuntimeError("product checksum target declaration changed")
    target_paths = [
        _contained_existing_file(ROOT / str(logical), ROOT, "checksum target")
        for logical in declared_targets
    ]
    _validate_checksum_inventory(checksum_file, target_paths)
    if receipt_file not in target_paths:
        raise RuntimeError("product checksum inventory does not bind its receipt")
    return {
        "receipt": receipt,
        "receipt_file": receipt_file,
        "renderer_input": renderer_input,
        "artifact_root": artifact_root,
        "receipt_root": receipt_root,
        "certified": certified,
        "runtime_source_pins": dict(runtime_pins),
    }


def render_products(
    renderer_input: str | Path,
    product_receipt_path: str | Path,
    renderer: str | Path | None,
    *,
    plan: RenderPlan,
    width: int,
    height: int,
) -> dict[str, Any]:
    if (int(width), int(height)) != (1_600, 1_000):
        raise ValueError("certified Rust products require exactly 1600x1000 pixels")
    _require_disjoint(plan.output_root, plan.store_root, "render outputs/store")
    _require_disjoint(plan.output_root, plan.receipt_root, "render outputs/receipts")
    _require_disjoint(plan.store_root, plan.receipt_root, "render store/receipts")
    source_pins = capture_runtime_source_pins()
    publication = validate_product_publication(product_receipt_path, renderer_input)
    source = publication["renderer_input"]
    product_receipt_file = publication["receipt_file"]
    source_record_before = _record(source)
    product_receipt_record_before = _record(product_receipt_file)
    _require_disjoint(
        plan.output_root, publication["artifact_root"], "render/product artifacts"
    )
    _require_disjoint(
        plan.receipt_root, publication["receipt_root"], "render/product receipts"
    )
    _require_disjoint(
        plan.store_root, publication["artifact_root"], "render store/product artifacts"
    )
    _require_disjoint(
        plan.store_root, publication["receipt_root"], "render store/product receipts"
    )
    for root in (plan.output_root, plan.store_root, plan.receipt_root):
        if root.exists():
            raise FileExistsError(f"Rust render root must be absent: {root}")

    rustwx_before = rust_renderer_gate.rustwx_record()
    probe = runner.coarse.require_current_renderer(renderer)
    executable_before = {
        "bytes": probe.executable.stat().st_size,
        "sha256": _sha256_file(probe.executable),
    }
    if executable_before != {
        "bytes": probe.executable_bytes,
        "sha256": probe.executable_sha256,
    }:
        raise RuntimeError("pinned Rust renderer probe record is false")
    catalog = inspect_renderer_products(source, store_root=plan.store_root, probe=probe)
    run = render_catalogued_products(
        source,
        store_root=plan.store_root,
        out_dir=plan.output_root,
        products=runner.coarse.DEFAULT_PRODUCTS,
        probe=probe,
        catalog=catalog,
        frames="1",
        width=width,
        height=height,
        source_label=(
            "MPAS x1.163842 (~60 km) CUDA dry dt120/split1 stabilized replay; "
            "capsule-exact; no physics, passive qv"
        ),
    )
    if (
        _sha256_file(probe.executable) != executable_before["sha256"]
        or probe.executable.stat().st_size != executable_before["bytes"]
    ):
        raise RuntimeError("pinned Rust renderer changed during rendering")
    if rust_renderer_gate.rustwx_record() != rustwx_before:
        raise RuntimeError("gpuwm.rustwx wrapper changed during rendering")
    assert_runtime_sources_unchanged(source_pins)
    if (
        _record(source) != source_record_before
        or _record(product_receipt_file) != product_receipt_record_before
    ):
        raise RuntimeError("product publication changed during rendering")
    product_records: list[dict[str, Any]] = []
    for output, expected_sha in zip(run.outputs, run.output_sha256, strict=True):
        record = _record(output)
        if record["sha256"] != expected_sha:
            raise RuntimeError("Rust plot changed after renderer return")
        record["product"] = runner.coarse._renderer_product(
            output, runner.coarse.DEFAULT_PRODUCTS
        )
        record["png_integrity"] = rust_renderer_gate._png_metadata(
            output, width, height
        )
        if output.stat().st_size >= 100_000_000:
            raise RuntimeError("Rust PNG exceeds 100,000,000 bytes")
        product_records.append(record)
    if (
        len(product_records) != len(runner.coarse.DEFAULT_PRODUCTS)
        or len({record["product"] for record in product_records})
        != len(runner.coarse.DEFAULT_PRODUCTS)
        or {record["product"] for record in product_records}
        != set(runner.coarse.DEFAULT_PRODUCTS)
        or len({record["path"] for record in product_records})
        != len(runner.coarse.DEFAULT_PRODUCTS)
    ):
        raise RuntimeError("Rust renderer did not create the exact six-product set")

    payload = {
        "schema": "mpas-port.x1.163842-stabilized-product-replay-render/v1",
        "status": "passed",
        "classification": PRODUCT_CLASSIFICATION,
        "source_product_receipt": product_receipt_record_before,
        "renderer_input": source_record_before,
        "renderer": {
            "executable_bytes": probe.executable_bytes,
            "executable_sha256": probe.executable_sha256,
            "catalog_summary": run.catalog.summary,
            "elapsed_seconds": run.elapsed_seconds,
            "width": int(width),
            "height": int(height),
            "source_label": (
                "MPAS x1.163842 (~60 km) CUDA dry dt120/split1 stabilized replay; "
                "capsule-exact; no physics, passive qv"
            ),
        },
        "renderer_executable_start_and_end": {
            **executable_before,
            "unchanged_after_render": True,
        },
        "gpuwm_rustwx_start_and_end": {
            **rustwx_before,
            "unchanged_after_render": True,
        },
        "runtime_source_pins": {
            label: dict(record) for label, record in sorted(source_pins.items())
        },
        "publication_contract": {
            "png_root": plan.output_root.relative_to(ROOT).as_posix(),
            "png_exact_files": sorted(output.name for output in run.outputs),
            "receipt_root": plan.receipt_root.relative_to(ROOT).as_posix(),
            "receipt_exact_files": sorted((plan.receipt.name, plan.checksums.name)),
            "checksum_inventory": plan.checksums.relative_to(ROOT).as_posix(),
            "checksum_exact_targets": [
                product_receipt_file.relative_to(ROOT).as_posix(),
                source.relative_to(ROOT).as_posix(),
                *(output.relative_to(ROOT).as_posix() for output in run.outputs),
                plan.receipt.relative_to(ROOT).as_posix(),
            ],
            "renderer_store_transient_not_published": True,
        },
        "products": sorted(product_records, key=lambda item: str(item["product"])),
        "claims": {
            "forecast_skill": False,
            "native_dt360_split3_equivalence": False,
            "column_or_moist_physics": False,
        },
        "timestep_interpretation": TIMESTEP_LABEL,
        "physics": PHYSICS_LABEL,
    }
    assert_runtime_sources_unchanged(source_pins)
    if rust_renderer_gate.rustwx_record() != rustwx_before:
        raise RuntimeError("gpuwm.rustwx changed before renderer receipt")
    if (
        _record(source) != source_record_before
        or _record(product_receipt_file) != product_receipt_record_before
    ):
        raise RuntimeError("product publication changed before renderer receipt")
    plan.receipt_root.mkdir(parents=True)
    write_json_atomic(plan.receipt, payload)
    render_targets = (product_receipt_file, source, *run.outputs, plan.receipt)
    _write_checksums(
        plan.checksums,
        render_targets,
    )
    expected_output_names = {output.name for output in run.outputs}
    _assert_exact_file_tree(
        plan.output_root, expected_output_names, "Rust PNG output root"
    )
    _assert_exact_file_tree(
        plan.receipt_root, {plan.receipt.name, plan.checksums.name}, "Rust receipt root"
    )
    _validate_checksum_inventory(plan.checksums, render_targets)
    if (
        _sha256_file(probe.executable) != executable_before["sha256"]
        or probe.executable.stat().st_size != executable_before["bytes"]
    ):
        raise RuntimeError("pinned Rust renderer changed after receipt")
    if rust_renderer_gate.rustwx_record() != rustwx_before:
        raise RuntimeError("gpuwm.rustwx changed after renderer receipt")
    if (
        _record(source) != source_record_before
        or _record(product_receipt_file) != product_receipt_record_before
    ):
        raise RuntimeError("product publication changed after renderer receipt")
    assert_runtime_sources_unchanged(source_pins)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=runner.DEFAULT_SOURCE)
    parser.add_argument("--grid", type=Path, default=runner.DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=runner.DEFAULT_STATIC)
    parser.add_argument("--certified-root", type=Path, default=DEFAULT_CERTIFIED_ROOT)
    parser.add_argument("--gpuwm-root", type=Path, default=runner.DEFAULT_GPUWM_ROOT)
    parser.add_argument("--gpuwm-probe", type=Path, default=runner.DEFAULT_GPUWM_PROBE)
    parser.add_argument("--ftz-binding", type=Path, default=runner.DEFAULT_FTZ_BINDING)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--renderer-input", type=Path)
    parser.add_argument("--product-receipt", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--render-store", type=Path, default=DEFAULT_RENDER_STORE)
    parser.add_argument(
        "--render-receipt-root", type=Path, default=DEFAULT_RENDER_RECEIPT_ROOT
    )
    parser.add_argument("--renderer-width", type=int, default=1_600)
    parser.add_argument("--renderer-height", type=int, default=1_000)
    return parser


def execute(args: argparse.Namespace) -> int:
    if args.render_only:
        if args.renderer_input is None or args.product_receipt is None:
            raise ValueError(
                "--render-only requires --renderer-input and --product-receipt"
            )
        render_plan = build_render_plan(
            args.render_root, args.render_store, args.render_receipt_root
        )
        payload = render_products(
            args.renderer_input,
            args.product_receipt,
            args.renderer,
            plan=render_plan,
            width=args.renderer_width,
            height=args.renderer_height,
        )
        print(json.dumps({"status": "passed", "pngs": len(payload["products"])}))
        return 0

    certified_root = Path(args.certified_root).expanduser().resolve(strict=True)
    if certified_root != DEFAULT_CERTIFIED_ROOT.resolve(strict=True):
        raise ValueError(
            "product replay requires the exact committed x1.163842 scout root"
        )
    plan = build_output_plan(args.artifact_root, args.receipt_root)
    _require_disjoint(
        plan.receipt_root,
        certified_root,
        "product receipt/certified evidence roots",
    )
    runtime_source_pins = capture_runtime_source_pins()
    certified = validate_certified_evidence(certified_root)
    config = canonical_stabilized_config()
    validate_fresh_output_plan(plan)
    cache_root = _validate_cache_preimport(args.cache_root)
    bounds = StabilityBounds(
        max_mass_relative_drift=2.0e-8,
        max_energy_relative_drift=0.5,
        max_abs_velocity=500.0,
        min_density=1.0e-7,
    )
    bounds.validate()

    timing: dict[str, float] = {}
    started_total = time.perf_counter()
    started = time.perf_counter()
    case = prepare_stabilized_case(args.source, args.grid, args.static, config)
    timing["host_preparation"] = time.perf_counter() - started
    sealed_before = runner.fingerprint_prepared_execution(case.cuda, config)
    if sealed_before != case.cuda.expected_execution_fingerprint:
        raise RuntimeError("fresh host preparation differs from its execution seal")

    gpuwm_root = args.gpuwm_root.expanduser().resolve(strict=True)
    _, comparison_authority = runner.load_gpuwm_dualrun(gpuwm_root)
    ftz_binding = runner.load_ftz_binding_record(
        args.ftz_binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=args.gpuwm_probe,
    )
    capability = runner.require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    started = time.perf_counter()
    kernel_cache = runner.prepare_cuda_kernel_cache(
        capability, cache_root, ftz_binding=ftz_binding
    )
    timing["compile"] = time.perf_counter() - started
    started = time.perf_counter()
    arm = runner.run_cuda_arm_generic(
        case.cuda,
        config,
        steps=TARGET_STEPS,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    timing["single_replay_arm"] = time.perf_counter() - started

    # This is the product firewall.  No output root has been created above.
    replay_gate = certify_replay_capsule(arm.capsule, certified)
    if runner.fingerprint_prepared_execution(case.cuda, config) != sealed_before:
        raise RuntimeError("single replay arm mutated the sealed host preparation")
    assert_runtime_sources_unchanged(runtime_source_pins)

    plan.receipt_root.mkdir(parents=True)
    write_json_atomic(plan.replay_capsule, arm.capsule)
    write_json_atomic(plan.replay_gate, replay_gate)
    if _sha256_file(plan.replay_capsule) != EXPECTED_CERTIFIED_ARM_SHA256:
        raise RuntimeError("persisted replay capsule changed after admission")

    started = time.perf_counter()
    products = write_products(case, arm, config, bounds, plan=plan)
    timing["history_regrid_renderer_input"] = time.perf_counter() - started
    timing["total_before_receipt"] = time.perf_counter() - started_total
    assert_runtime_sources_unchanged(runtime_source_pins)
    receipt = write_product_receipt(
        case,
        config,
        certified,
        replay_gate,
        products,
        plan,
        timing=timing,
        runtime_source_pins=runtime_source_pins,
    )
    assert_runtime_sources_unchanged(runtime_source_pins)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": _record(plan.receipt)["path"],
                "renderer_input": products["renderer_input"]["path"],
                "capsule_exact_replay": True,
                "physics": PHYSICS_LABEL,
                "timestep_interpretation": TIMESTEP_LABEL,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
