"""Validate the preserved x1.163842 scout and remote sm_120 authority.

This is a read-only clean-checkout gate.  It does not import CuPy, compile a
kernel, contact the remote host, or write forecast products.  The remote P2
binding is reconstructed from the committed 26-file gpuwm probe, five-file
static gpuwm source mirror, and the live MPAS CUDA translation units before
the stabilized scout is admitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import mpas_port.cuda_ftz as cuda_ftz  # noqa: E402
from mpas_port.cuda_dualrun import (  # noqa: E402
    compare_cuda_capsule_files,
    validate_cuda_capsule,
)


DEFAULT_FTZ_ROOT = ROOT / "receipts" / "cuda-ftz-sm120-remote"
DEFAULT_SCOUT_ROOT = (
    ROOT
    / "receipts"
    / "cuda-gfs-forecast"
    / "x1.163842-stabilized-scout-20260810a"
)

FTZ_INVENTORY_SHA256 = (
    "12b56f26d7a42f1ff6b6b2281d317e4ea410f097cec0942bbcc669ba413c792f"
)
SCOUT_INVENTORY_SHA256 = (
    "3cc9ee6b9d043dd4a11f8d2abf5b1b9d59945afe11ba775c1e54fd7c1c36aeab"
)
GPUWM_HEAD = "4152fcb318d7a17ae39967632a788319d64913e3"
P2_BINDING_SHA256 = (
    "d120faf49894dec04cca97ca5ceecde18c0030bc1930bd326e8580f22102b145"
)
PROBE_RECEIPT_SHA256 = (
    "ca84d953aa8a707cd1973bb92e42bce8ca8620f1b8a0b15ccc770cf093082b60"
)
COMPILE_MANIFEST_SHA256 = (
    "bfd9ffc1b42af862dac65fa1d713986354db0c1eea2bc15a3e70e9964fbee68b"
)
CONFIGURATION_SHA256 = (
    "3780b0718632ea88bd00e073bc374aca2d1f20e842e10bccbd9da35fa5ee4ec0"
)
HOST_EXECUTION_SEAL_SHA256 = (
    "86ebb88e52ec15cf5ed9a2edc95b7ae7d8b3bbfea1f203f02f2f943437a3d6fa"
)
FINAL_SNAPSHOT_SHA256 = (
    "5abf288bed7b714adaa207dbb4350e21ca9afcce027487fddde5629940f12948"
)
ARM_SHA256 = (
    "2aa37a43460eff249c26091c1b591d731c00f64ed006e2a63b081ab204397c66"
)
SUMMARY_SHA256 = (
    "06f1cee5c75111315a02d18a41b226698ad2d6b34fd271b39d3cefd6afe4c3bf"
)
COMPARISON_SHA256 = (
    "f7d2f94f867b95fdb8b951406130cf2d82649f795569827332fa6883ac54d9e9"
)
GPUWM_DUALRUN_SHA256 = (
    "e0f70713a92efaf9acc7fcc4142ef52f4e0a1ffad4b76ef36ed46d4d5291cf12"
)


class EvidenceValidationError(RuntimeError):
    """A committed byte, inventory, binding, or scout verdict is invalid."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise EvidenceValidationError(f"required evidence file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot load JSON evidence {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"JSON evidence is not an object: {path}")
    return value


def _safe_file(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in candidate.parents:
        raise EvidenceValidationError(f"inventory path escapes its root: {relative!r}")
    return candidate


def _validate_exact_inventory(
    root: Path,
    inventory_path: Path,
    *,
    expected_schema: str,
    expected_inventory_sha256: str,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    inventory_path = inventory_path.resolve(strict=True)
    if _sha256_file(inventory_path) != expected_inventory_sha256:
        raise EvidenceValidationError(
            f"inventory bytes changed: {inventory_path}"
        )
    inventory = _load_json(inventory_path)
    if inventory.get("schema") != expected_schema:
        raise EvidenceValidationError(
            f"unexpected evidence inventory schema: {inventory.get('schema')!r}"
        )
    files = inventory.get("files")
    if not isinstance(files, Mapping) or not files:
        raise EvidenceValidationError("evidence inventory has no file mapping")
    declared = {str(relative) for relative in files}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != inventory_path
    }
    if actual != declared:
        raise EvidenceValidationError(
            "evidence inventory coverage changed: "
            f"missing={sorted(declared - actual)!r}, extra={sorted(actual - declared)!r}"
        )
    for relative in sorted(declared):
        record = files[relative]
        if not isinstance(record, Mapping) or set(record) != {"bytes", "sha256"}:
            raise EvidenceValidationError(
                f"invalid inventory record for {relative!r}"
            )
        path = _safe_file(root, relative)
        if path.stat().st_size != record["bytes"]:
            raise EvidenceValidationError(f"byte count changed for {relative!r}")
        if _sha256_file(path) != record["sha256"]:
            raise EvidenceValidationError(f"SHA-256 changed for {relative!r}")
    return inventory


def _validate_static_gpuwm_sources(ftz_root: Path) -> dict[str, Any]:
    manifest_path = ftz_root / "metadata" / "gpuwm-static-source-manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "mpas-port.gpuwm-static-source-mirror/v1":
        raise EvidenceValidationError("static gpuwm source manifest schema changed")
    if manifest.get("git_head") != GPUWM_HEAD:
        raise EvidenceValidationError("static gpuwm source HEAD changed")
    source_root = ftz_root / "gpuwm-static-source"
    if any(path.name == ".git" for path in source_root.rglob("*")):
        raise EvidenceValidationError("static gpuwm source mirror must not carry .git")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or len(files) != 5:
        raise EvidenceValidationError("static gpuwm source mirror is not five files")
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual != set(files):
        raise EvidenceValidationError("static gpuwm source inventory changed")
    for relative, record in files.items():
        path = _safe_file(source_root, str(relative))
        if path.stat().st_size != record.get("bytes"):
            raise EvidenceValidationError(
                f"static gpuwm source byte count changed: {relative}"
            )
        if _sha256_file(path) != record.get("sha256"):
            raise EvidenceValidationError(
                f"static gpuwm source SHA-256 changed: {relative}"
            )
    return manifest


def validate_ftz_authority(ftz_root: str | Path = DEFAULT_FTZ_ROOT) -> dict[str, Any]:
    """Validate P2/probe/source bytes and rebuild the MPAS binding statically."""

    root = Path(ftz_root).expanduser().resolve(strict=True)
    _validate_exact_inventory(
        root,
        root / "metadata" / "evidence-inventory.json",
        expected_schema="mpas-port.cuda-ftz-sm120-remote-evidence-inventory/v1",
        expected_inventory_sha256=FTZ_INVENTORY_SHA256,
    )
    root_files = {path.name for path in root.iterdir() if path.is_file()}
    if root_files != {
        "binding.json",
        "compile-manifest.json",
        "kernel-audit.json",
        "normalized-performance-control.json",
        "transport-deck.json",
    }:
        raise EvidenceValidationError("remote P2 root is not the exact five files")
    probe_root = root / "gpuwm-probe"
    probe_files = [path for path in probe_root.rglob("*") if path.is_file()]
    if len(probe_files) != 26:
        raise EvidenceValidationError("gpuwm FTZ probe is not the exact 26 files")

    static_manifest = _validate_static_gpuwm_sources(root)
    binding_path = root / "binding.json"
    if _sha256_file(binding_path) != P2_BINDING_SHA256:
        raise EvidenceValidationError("remote P2 binding bytes changed")
    binding = _load_json(binding_path)
    if binding.get("schema") != cuda_ftz.MPAS_FTZ_SCHEMA:
        raise EvidenceValidationError("remote P2 binding schema changed")

    # The raw probe is validated before it is allowed to participate in the
    # MPAS binding.  This recomputes all 36 cells and both-pass byte identity.
    validated_probe = cuda_ftz.validate_gpuwm_ftz_receipt(probe_root)
    if validated_probe.get("receipt_sha256") != PROBE_RECEIPT_SHA256:
        raise EvidenceValidationError("gpuwm FTZ receipt changed")
    if validated_probe.get("verified_artifact_count") != 25:
        raise EvidenceValidationError("gpuwm FTZ receipt artifact coverage changed")
    if validated_probe != binding.get("gpuwm_ftz_probe"):
        raise EvidenceValidationError("P2 binding does not embed the validated probe")

    source_pins = {
        "git_head": static_manifest["git_head"],
        "sources": {
            label: {
                "path": row["path"],
                "sha256": static_manifest["files"][row["path"]]["sha256"],
            }
            for label, row in binding["gpuwm"]["sources"].items()
        },
    }
    if source_pins != binding.get("gpuwm"):
        raise EvidenceValidationError("P2 binding gpuwm source pins changed")

    documents = {
        "compile_manifest": _load_json(root / "compile-manifest.json"),
        "kernel_audit": _load_json(root / "kernel-audit.json"),
        "normalized_performance_control": _load_json(
            root / "normalized-performance-control.json"
        ),
        "transport_deck": _load_json(root / "transport-deck.json"),
    }
    for key, document in documents.items():
        if document != binding.get(key):
            raise EvidenceValidationError(f"P2 external {key} differs from binding")

    # build_mpas_ftz_binding normally resolves gpuwm's git checkout.  The
    # committed clean-checkout authority intentionally has no .git; substitute
    # only the independently byte-validated static source resolver, then run
    # every production probe/compile/deck/audit check unchanged.
    with mock.patch.object(
        cuda_ftz,
        "measure_gpuwm_source_pins",
        return_value=source_pins,
    ):
        rebuilt = cuda_ftz.build_mpas_ftz_binding(
            gpuwm_root=root / "gpuwm-static-source",
            gpuwm_receipt_root=probe_root,
            compile_manifest=documents["compile_manifest"],
            transport_deck=documents["transport_deck"],
            kernel_audit=documents["kernel_audit"],
            performance_control=documents["normalized_performance_control"],
        )
    if rebuilt != binding:
        raise EvidenceValidationError("P2 binding differs from rebuilt live evidence")
    if binding["compile_relation"]["compile_manifest_sha256"] != COMPILE_MANIFEST_SHA256:
        raise EvidenceValidationError("P2 compile-manifest relation changed")
    return {
        "binding_sha256": P2_BINDING_SHA256,
        "compile_manifest_sha256": COMPILE_MANIFEST_SHA256,
        "gpuwm_git_head": GPUWM_HEAD,
        "gpuwm_probe_files": 26,
        "gpuwm_probe_receipt_sha256": PROBE_RECEIPT_SHA256,
        "gpuwm_static_source_files": 5,
        "inventory_sha256": FTZ_INVENTORY_SHA256,
    }


def _validate_live_capsule_sources(capsule: Mapping[str, Any]) -> None:
    sources = capsule.get("contracts", {}).get("implementation_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise EvidenceValidationError("capsule has no implementation source pins")
    for name, record in sources.items():
        if not isinstance(record, Mapping):
            raise EvidenceValidationError(f"capsule source pin is invalid: {name}")
        path = _safe_file(SOURCE_ROOT / "mpas_port", str(record.get("path")))
        if _sha256_file(path) != record.get("sha256"):
            raise EvidenceValidationError(
                f"capsule implementation source differs from clean checkout: {name}"
            )


def _validate_scout_jsonl(root: Path, summary: Mapping[str, Any]) -> None:
    diagnostic = root / "x1.163842-stabilized-scout-20260810a.jsonl"
    log = root / "x1.163842-stabilized-scout-20260810a.log"
    diagnostic_lines = diagnostic.read_text(encoding="utf-8").splitlines()
    log_lines = log.read_text(encoding="utf-8").splitlines()
    if len(diagnostic_lines) != 191 or len(log_lines) != 192:
        raise EvidenceValidationError("scout JSONL/log line inventory changed")
    if log_lines[:-1] != diagnostic_lines:
        raise EvidenceValidationError("scout log is not canonical JSONL plus wrapper")
    try:
        rows = [json.loads(line) for line in diagnostic_lines]
        wrapper = json.loads(log_lines[-1])
    except json.JSONDecodeError as error:
        raise EvidenceValidationError(f"scout JSONL/log is invalid: {error}") from error
    if not all(isinstance(row, dict) for row in rows) or not isinstance(wrapper, dict):
        raise EvidenceValidationError("scout JSONL/log contains a non-object row")
    phase_counts = Counter(row.get("phase") for row in rows)
    if phase_counts != Counter(
        {
            "host-preparation-start": 1,
            "host-preparation-complete": 1,
            "scout-start": 1,
            "scout-step": 180,
            "scout-complete": 1,
            "arm-start": 2,
            "arm-complete": 2,
            "gpuwm-total-comparison-start": 1,
            "gpuwm-total-comparison-complete": 1,
            "run-complete": 1,
        }
    ):
        raise EvidenceValidationError(f"scout JSONL phase inventory changed: {phase_counts}")
    step_rows = [row for row in rows if row.get("phase") == "scout-step"]
    expected_snapshots = summary["scout"]["step_snapshot_sha256"]
    for expected_step, (row, snapshot) in enumerate(
        zip(step_rows, expected_snapshots, strict=True),
        1,
    ):
        if (
            row.get("step") != expected_step
            or row.get("model_time_seconds") != 120.0 * expected_step
            or row.get("status") != "passed"
            or row.get("snapshot_sha256") != snapshot
            or row.get("compile_manifest_sha256") != COMPILE_MANIFEST_SHA256
            or row.get("failures")
            != {
                "bound_failures": [],
                "cfl_failures": [],
                "hard_domain_failures": [],
                "nonfinite_fields": [],
            }
        ):
            raise EvidenceValidationError(f"scout step {expected_step} is not green")
        positivity = row.get("thermodynamic_positivity", {})
        if positivity.get("hard_domain_failures") or positivity.get("nonfinite_fields"):
            raise EvidenceValidationError(
                f"scout step {expected_step} has a thermodynamic failure"
            )
        fields = row.get("fields", {})
        if not fields or not all(field.get("all_finite") is True for field in fields.values()):
            raise EvidenceValidationError(f"scout step {expected_step} is nonfinite")
    final = rows[-1]
    if final != {
        "configuration_sha256": CONFIGURATION_SHA256,
        "forecast_products_written": False,
        "gpuwm_identical": True,
        "phase": "run-complete",
        "scout_status": "passed",
        "status": "passed",
        "summary_file": {
            "bytes": 25211,
            "sha256": SUMMARY_SHA256,
        },
        "total_comparison": True,
    }:
        raise EvidenceValidationError("scout run-complete row changed")
    if (
        wrapper.get("status") != "passed"
        or wrapper.get("configuration_sha256") != CONFIGURATION_SHA256
        or wrapper.get("forecast_products_written") is not False
    ):
        raise EvidenceValidationError("scout wrapper-final row changed")


def validate_scout_evidence(
    scout_root: str | Path = DEFAULT_SCOUT_ROOT,
    ftz_root: str | Path = DEFAULT_FTZ_ROOT,
) -> dict[str, Any]:
    """Validate the stabilized scout only after its FTZ authority is green."""

    ftz = validate_ftz_authority(ftz_root)
    root = Path(scout_root).expanduser().resolve(strict=True)
    _validate_exact_inventory(
        root,
        root / "inventory.json",
        expected_schema="mpas-port.x1.163842-stabilized-scout-evidence-inventory/v1",
        expected_inventory_sha256=SCOUT_INVENTORY_SHA256,
    )
    evidence_root = root / "evidence"
    summary_path = evidence_root / "summary.json"
    comparison_path = evidence_root / "gpuwm-total-comparison.json"
    arm_a_path = evidence_root / "arm-a.json"
    arm_b_path = evidence_root / "arm-b.json"
    if _sha256_file(summary_path) != SUMMARY_SHA256:
        raise EvidenceValidationError("stabilized scout summary changed")
    if _sha256_file(comparison_path) != COMPARISON_SHA256:
        raise EvidenceValidationError("gpuwm total comparison changed")
    if _sha256_file(arm_a_path) != ARM_SHA256 or _sha256_file(arm_b_path) != ARM_SHA256:
        raise EvidenceValidationError("stabilized scout arm bytes changed")
    if arm_a_path.read_bytes() != arm_b_path.read_bytes():
        raise EvidenceValidationError("stabilized scout arms are not byte-identical")

    summary = _load_json(summary_path)
    comparison = _load_json(comparison_path)
    arm_a = _load_json(arm_a_path)
    arm_b = _load_json(arm_b_path)
    validate_cuda_capsule(arm_a)
    validate_cuda_capsule(arm_b)
    _validate_live_capsule_sources(arm_a)
    _validate_live_capsule_sources(arm_b)
    previous_bytecode_policy = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        regenerated = compare_cuda_capsule_files(
            arm_a_path,
            arm_b_path,
            gpuwm_root=Path(ftz_root) / "gpuwm-static-source",
        )
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    if regenerated != comparison:
        raise EvidenceValidationError("committed gpuwm total comparison is false")
    authority = comparison.get("comparison_authority", {})
    if (
        authority.get("source_sha256") != GPUWM_DUALRUN_SHA256
        or comparison.get("total_comparison") is not True
        or comparison.get("gpuwm_comparison")
        != {
            "schema": "gpuwm.dual-run-comparison/v1",
            "identical": True,
            "first_divergent_field": None,
            "divergence_count": 0,
            "divergences": [],
        }
    ):
        raise EvidenceValidationError("gpuwm total-comparison verdict changed")

    if (
        summary.get("schema") != "mpas-port.x1.163842-stabilized-scout-dual/v1"
        or summary.get("classification")
        != "work-only diagnostic/evidence; no forecast products"
        or summary.get("forecast_products_written") is not False
        or summary.get("host_execution_seal_sha256") != HOST_EXECUTION_SEAL_SHA256
        or summary.get("host_after_sha256") != HOST_EXECUTION_SEAL_SHA256
    ):
        raise EvidenceValidationError("stabilized scout summary identity changed")
    configuration = summary.get("configuration", {})
    value = configuration.get("value", {})
    if (
        configuration.get("sha256") != CONFIGURATION_SHA256
        or value.get("config_dt") != 120.0
        or value.get("config_dynamics_split_steps") != 1
        or value.get("config_horiz_mixing") != "2d_smagorinsky"
        or value.get("config_divergence_damping") is not True
        or value.get("config_physics_suite") != "none"
    ):
        raise EvidenceValidationError("stabilized dt120/split1 configuration changed")
    scout = summary.get("scout", {})
    hard_courants = scout.get("maximum_hard_courants", {})
    if (
        scout.get("status") != "passed"
        or scout.get("completed_steps") != 180
        or scout.get("simulated_seconds") != 21600.0
        or scout.get("final_model_time_seconds") != 21600.0
        or scout.get("final_snapshot_sha256") != FINAL_SNAPSHOT_SHA256
        or scout.get("compile_manifest_sha256") != COMPILE_MANIFEST_SHA256
        or scout.get("host_after_sha256") != HOST_EXECUTION_SEAL_SHA256
        or len(scout.get("step_snapshot_sha256", ())) != 180
        or not hard_courants
        or max(hard_courants.values()) != 0.5900760293006897
        or max(hard_courants.values()) > 1.0
        or scout.get("nonvacuous_qv_tracer", {}).get("nonvacuous") is not True
    ):
        raise EvidenceValidationError("stabilized scout hard-gate verdict changed")
    _validate_scout_jsonl(root, summary)

    expected_steps = scout["step_snapshot_sha256"]
    for label, capsule in (("a", arm_a), ("b", arm_b)):
        trajectory = capsule["trajectory"]
        records = trajectory["step_records"]
        observed_steps = [record["snapshot"]["sha256"] for record in records]
        if (
            trajectory.get("steps") != 180
            or trajectory.get("dt_seconds") != 120.0
            or trajectory["initial_snapshot"].get("sha256")
            != scout.get("t0_snapshot_sha256")
            or trajectory.get("final_snapshot_sha256") != FINAL_SNAPSHOT_SHA256
            or observed_steps != expected_steps
        ):
            raise EvidenceValidationError(f"arm {label} differs from hard-gated scout")
        for record in records:
            contract = record["step_contract"]
            if (
                contract.get("evidence") != "implemented-cuda-dry-rk3-unlinked"
                or contract.get("authority_ruler") is not None
                or contract.get("authority_ruler_sha256") is not None
                or contract.get("d2h_bytes_inside_step") != 0
                or contract.get("compile_manifest_sha256") != COMPILE_MANIFEST_SHA256
                or contract.get("configuration_sha256") != CONFIGURATION_SHA256
            ):
                raise EvidenceValidationError(f"arm {label} step contract changed")

    gate_source_map = {
        "hardened_locator": "diagnose_cuda_x1_163842_stability.py",
        "stabilized_scout": "run_cuda_x1_163842_stabilized_scout.py",
        "transitive_high_resolution_runner": "run_real_gfs_cuda_x1_163842.py",
    }
    for label, filename in gate_source_map.items():
        record = summary["gate_implementation_sources"][label]
        copied = root / "execution-sources" / filename
        if _sha256_file(copied) != record["observed_sha256"]:
            raise EvidenceValidationError(f"executed gate source changed: {label}")

    return {
        "arms_byte_identical": True,
        "arm_sha256": ARM_SHA256,
        "configuration_sha256": CONFIGURATION_SHA256,
        "final_snapshot_sha256": FINAL_SNAPSHOT_SHA256,
        "forecast_products_written": False,
        "ftz": ftz,
        "gpuwm_total_comparison": True,
        "inventory_sha256": SCOUT_INVENTORY_SHA256,
        "maximum_hard_courant": 0.5900760293006897,
        "scout_steps": 180,
        "simulated_seconds": 21600.0,
        "status": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftz-root", type=Path, default=DEFAULT_FTZ_ROOT)
    parser.add_argument("--scout-root", type=Path, default=DEFAULT_SCOUT_ROOT)
    args = parser.parse_args(argv)
    result = validate_scout_evidence(args.scout_root, args.ftz_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
