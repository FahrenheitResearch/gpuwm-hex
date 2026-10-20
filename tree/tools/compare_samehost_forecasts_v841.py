#!/usr/bin/env python3
"""Require complete real-door forecast receipts and byte-identical outputs.

This compares recorded logical array hashes and actual history-file bytes.
It does not run kernels, reconstruct a restart, or evaluate weather skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS
from hexcore.drivers.samehost_checks_v841 import (
    EXPECTED_METADATA_DIVERGENCE_SUBSTRINGS,
    compare_boundary_files,
    compare_boundary_records,
    read_boundary_jsonl,
)
from hexcore.drivers.samehost_identity_v841 import _PADDED_CARRIERS


SCHEMA = "mpas-port.cuda-v841-complete-samehost-identity/v1"
DRIVER_NAME = "cuda-v841-forecast-receipt.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_PATHS = frozenset(path.removeprefix("__backend__/") for path in _PADDED_CARRIERS)
_FORCING_FIELDS = frozenset(("rthdynten", "rqvdynten"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: receipt must be a JSON object")
    return value


def _fingerprint(value: Any, label: str, *, atmosphere: bool = False) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: array fingerprint is absent")
    dtype = np.dtype(value.get("dtype"))
    if not isinstance(value.get("dtype"), str) or dtype.hasobject or dtype.fields is not None:
        raise ValueError(f"{label}: plain fixed-width dtype is absent")
    shape = value.get("shape")
    if not isinstance(shape, list) or any(type(extent) is not int or extent < 0 for extent in shape):
        raise ValueError(f"{label}: array shape is absent or invalid")
    keys = ("sha256", "bytes_sha256") if atmosphere else ("sha256",)
    for key in keys:
        if not isinstance(value.get(key), str) or not _SHA256.fullmatch(value[key]):
            raise ValueError(f"{label}: {key} is absent or invalid")


def _validate_boundaries(records: Mapping[int, Any], arm: str, expected_steps: int) -> bool:
    wanted = set(range(expected_steps + 1))
    if set(records) != wanted:
        raise ValueError(f"{arm}: expected boundary steps 0..{expected_steps}, found {sorted(records)}")
    previous_time = None
    timestep = None
    for step, record in sorted(records.items()):
        atmosphere = record.get("atmosphere")
        if not isinstance(atmosphere, Mapping):
            raise ValueError(f"{arm} step {step}: atmosphere fingerprint is absent")
        clock = atmosphere.get("model_time_seconds")
        if isinstance(clock, bool) or not isinstance(clock, (float, int)) or not math.isfinite(clock):
            raise ValueError(f"{arm} step {step}: shared model clock is absent or non-finite")
        if (step == 0 and clock != 0) or (previous_time is not None and clock <= previous_time):
            raise ValueError(f"{arm} step {step}: shared model clock does not advance from zero")
        if step == 1:
            timestep = clock
        elif step > 1 and clock != step * timestep:
            raise ValueError(f"{arm} step {step}: shared model clock differs from the fixed step schedule")
        previous_time = clock
        for group, names in (("state", STATE_FIELDS), ("saved_diagnostics", SIDECAR_FIELDS)):
            fields = (atmosphere.get(group) or {}).get("fields")
            if not isinstance(fields, Mapping) or set(fields) != set(names):
                raise ValueError(f"{arm} step {step}: complete {group} field fingerprints are absent")
            for name, fingerprint in fields.items():
                _fingerprint(fingerprint, f"{arm} step {step} {group}/{name}", atmosphere=True)
        backend = record.get("backend")
        arrays = backend.get("arrays") if isinstance(backend, Mapping) else None
        scalars = backend.get("scalars") if isinstance(backend, Mapping) else None
        if not isinstance(arrays, Mapping) or not arrays or not isinstance(scalars, Mapping):
            raise ValueError(f"{arm} step {step}: persisted backend array/scalar fingerprints are absent")
        for name, fingerprint in arrays.items():
            _fingerprint(fingerprint, f"{arm} step {step} backend/{name}")
        artifacts = record.get("batch_order_artifacts", {})
        if not isinstance(artifacts, Mapping) or set(artifacts) - _ARTIFACT_PATHS:
            raise ValueError(f"{arm} step {step}: an unnamed backend artifact would be excluded")
        for difference in record.get("rank_metadata_divergences", []):
            if not isinstance(difference, Mapping) or not any(
                marker in str(difference.get("path")) for marker in EXPECTED_METADATA_DIVERGENCE_SUBSTRINGS
            ):
                raise ValueError(f"{arm} step {step}: unexpected metadata differs between ranks")
    return True


def _history_inventory(output: Path, forecast: Mapping[str, Any], arm: str, expected_steps: int) -> dict[str, str]:
    declared = forecast.get("snapshot_files")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError(f"{arm}: recorded history frames are absent")
    if not {"0", str(expected_steps)} <= set(declared):
        raise ValueError(f"{arm}: initial or final history frame is absent")
    actual = {path.name: file_sha256(path) for path in output.glob("cuda-history.*.nc") if path.is_file()}
    names = set()
    for step, descriptor in declared.items():
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
            raise ValueError(f"{arm}: history descriptor at step {step} is invalid")
        name = Path(descriptor["path"]).name
        if name in names:
            raise ValueError(f"{arm}: history file {name} is declared more than once")
        names.add(name)
        if name not in actual:
            raise ValueError(f"{arm}: declared history file {name} is missing")
        if descriptor.get("sha256") != actual[name]:
            raise ValueError(f"{arm}: history file {name} differs from its recorded full SHA-256")
        if "bytes" in descriptor and descriptor["bytes"] != (output / name).stat().st_size:
            raise ValueError(f"{arm}: history file {name} differs from its recorded byte count")
    if names != set(actual):
        raise ValueError(f"{arm}: actual and recorded history filename sets differ")
    return actual


def compare_forecasts(*, reference_output: Path, candidate_output: Path,
                      reference_measurement: Path, candidate_measurement: Path,
                      expected_steps: int) -> dict[str, Any]:
    """Return a fail-closed receipt; unreadable or incomplete evidence is FAIL."""
    failures = []
    evidence: dict[str, Any] = {}

    def attempt(check, operation):
        try:
            return operation()
        except Exception as error:
            failures.append({"check": check, "type": type(error).__name__, "reason": str(error)})
            return None

    def require(check, condition, reason):
        if not condition:
            failures.append({"check": check, "reason": reason})

    require("expected_steps", type(expected_steps) is int and expected_steps > 0,
            "expected steps must be a positive integer")
    arms = {}
    for arm, output, measurement_path in (
        ("reference", Path(reference_output), Path(reference_measurement)),
        ("candidate", Path(candidate_output), Path(candidate_measurement)),
    ):
        driver_path = output / DRIVER_NAME
        driver = attempt(f"{arm}/driver_read", lambda: _json_object(driver_path))
        measurement = attempt(f"{arm}/measurement_read", lambda: _json_object(measurement_path))
        evidence[arm] = {"output": str(output), "driver_receipt": str(driver_path),
                         "measurement_receipt": str(measurement_path)}
        if driver is None or measurement is None:
            continue
        evidence[arm]["driver_receipt_sha256"] = file_sha256(driver_path)
        evidence[arm]["measurement_receipt_sha256"] = file_sha256(measurement_path)
        forecast = driver.get("forecast")
        require(f"{arm}/driver_status", driver.get("status") == "passed", "actual driver status is not passed")
        require(f"{arm}/measurement_status", measurement.get("driver_status") == "passed"
                and measurement.get("driver_completed") is True and measurement.get("cli_exit_code") == 0,
                "measurement does not record a completed successful real-door run")
        evidence[arm]["run_identity_eligible"] = measurement.get("run_identity_eligible")
        require(f"{arm}/runtime_qualification", measurement.get("run_identity_eligible") is True,
                "post-import interpreter/runtime identity qualification is absent or failed")
        interpreter = ((measurement.get("source_runtime_inputs_after") or {}).get("runtime") or {}).get("interpreter") or {}
        if interpreter.get("Py_GIL_DISABLED") == 1 and interpreter.get("gil_enabled") is False:
            serial_io = measurement.get("serial_extension_io") or {}
            require(f"{arm}/serial_extension_io", serial_io.get("enabled") is True
                    and serial_io.get("qualified") is True
                    and serial_io.get("module_entrypoints_restored") is True
                    and type(serial_io.get("worker_refusals")) is int and serial_io["worker_refusals"] == 0,
                    "the no-GIL run lacks a restored coordinator-only legacy I/O policy or records worker access")
            evidence[arm]["serial_extension_io"] = serial_io
        require(f"{arm}/protected_bytes", measurement.get("protected_bytes_unchanged") is True,
                "measurement source/runtime/input bytes changed or were not checked")
        require(f"{arm}/door_restoration", measurement.get("ordinary_door_restored") is True,
                "ordinary forecast door was not restored")
        require(f"{arm}/measurement_exception", "exception" in measurement and measurement["exception"] is None,
                "measurement exception is present or its absence was not recorded")
        if not isinstance(forecast, Mapping):
            require(f"{arm}/forecast", False, "driver forecast receipt is absent")
            continue
        steps = (forecast.get("walls") or {}).get("steps")
        require(f"{arm}/completed_steps", type(steps) is int and steps == expected_steps,
                f"actual forecast wall steps {steps!r} do not equal {expected_steps}")
        require(f"{arm}/refusal", forecast.get("refusal") is None, "forecast records a model refusal")
        boundary_path = output / "boundary-fingerprints.jsonl"
        records = attempt(f"{arm}/boundaries_read", lambda: read_boundary_jsonl(boundary_path))
        boundaries_valid = False
        if records is not None and type(expected_steps) is int and expected_steps > 0:
            boundaries_valid = attempt(f"{arm}/complete_boundaries", lambda: _validate_boundaries(records, arm, expected_steps)) is True
            evidence[arm]["boundary_steps"] = sorted(records)
            evidence[arm]["boundary_receipt_sha256"] = file_sha256(boundary_path)
        history = attempt(f"{arm}/history", lambda: _history_inventory(output, forecast, arm, expected_steps))
        evidence[arm]["history_files"] = history
        forcing = measurement.get("final_gf_tendency_hashes")
        if not isinstance(forcing, Mapping) or set(forcing) != _FORCING_FIELDS:
            require(f"{arm}/final_forcing", False, "both final GF tendency fingerprints are required")
            forcing = None
        else:
            for name, value in forcing.items():
                attempt(f"{arm}/final_forcing/{name}", lambda: _fingerprint(value, name))
            driver_forcing = forecast.get("final_gf_tendency_hashes")
            if driver_forcing is not None:
                require(f"{arm}/forcing_receipt_consistency", driver_forcing == forcing,
                        "driver and measurement final forcing fingerprints differ")
        evidence[arm]["final_gf_tendency_hashes"] = forcing
        arms[arm] = {"output": output, "records": records, "boundaries_valid": boundaries_valid,
                     "history": history, "forcing": forcing}
    if len(arms) == 2:
        reference, candidate = arms["reference"], arms["candidate"]
        if reference["boundaries_valid"] and candidate["boundaries_valid"]:
            comparison = attempt("boundary_comparison", lambda: compare_boundary_files(
                reference["output"] / "boundary-fingerprints.jsonl",
                candidate["output"] / "boundary-fingerprints.jsonl", require_steps=expected_steps + 1))
            evidence["boundary_comparison"] = comparison
            require("boundary_identity", comparison is not None and comparison["status"] == "PASS",
                    "full boundary array, metadata and clock comparison did not pass")
            evidence["declared_boundary_exceptions"] = {
                str(step): {"excluded_batch_order_artifacts": verdict["excluded_batch_order_artifacts"],
                            "metadata_diffs": verdict["metadata_diffs"],
                            "rank_metadata_divergences": candidate["records"][step].get("rank_metadata_divergences", [])}
                for step in sorted(set(reference["records"]) & set(candidate["records"]))
                for verdict in [compare_boundary_records(candidate["records"][step], reference["records"][step])]
                if verdict["excluded_batch_order_artifacts"] or verdict["metadata_diffs"]
                    or candidate["records"][step].get("rank_metadata_divergences")
            }
        require("history_identity", reference["history"] is not None and candidate["history"] is not None
                and reference["history"] == candidate["history"], "history filename sets or full file SHA-256 differ")
        require("final_forcing_identity", reference["forcing"] is not None and candidate["forcing"] is not None
                and reference["forcing"] == candidate["forcing"], "final GF tendency dtype, shape or SHA-256 differ")
    byte_comparison_passed = bool(len(arms) == 2
        and (evidence.get("boundary_comparison") or {}).get("status") == "PASS"
        and arms["reference"]["history"] is not None
        and arms["reference"]["history"] == arms["candidate"]["history"]
        and arms["reference"]["forcing"] is not None
        and arms["reference"]["forcing"] == arms["candidate"]["forcing"])
    return {"schema": SCHEMA, "status": "FAIL" if failures else "PASS", "expected_steps": expected_steps,
            "byte_comparison_passed": byte_comparison_passed,
            "evidence": evidence, "failures": failures,
            "identity_scope": "all 12 atmosphere state/diagnostic arrays, persisted backend arrays except named artifacts, shared clocks, published history file bytes and final next-step GF forcing",
            "allowed_backend_artifact_paths": sorted(_ARTIFACT_PATHS),
            "allowed_metadata_divergence_substrings": list(EXPECTED_METADATA_DIVERGENCE_SUBSTRINGS),
            "nonclaims": ["byte identity of the three named unconsumed staggered-u batch carriers",
                          "complete backend restart identity including the excluded artifacts",
                          "native MPAS whole-model identity or forecast skill"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-output", required=True, type=Path)
    parser.add_argument("--candidate-output", required=True, type=Path)
    parser.add_argument("--reference-measurement", required=True, type=Path)
    parser.add_argument("--candidate-measurement", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_forecasts(reference_output=args.reference_output, candidate_output=args.candidate_output,
                                  reference_measurement=args.reference_measurement,
                                  candidate_measurement=args.candidate_measurement, expected_steps=args.expected_steps)
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except Exception as error:
        result = {"schema": SCHEMA, "status": "FAIL", "expected_steps": args.expected_steps,
                  "failures": [{"check": "receipt_validation", "type": type(error).__name__, "reason": str(error)}]}
        serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    print(f"{result['status']} complete forecast identity: steps={args.expected_steps} failures={len(result['failures'])} receipt={args.receipt}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
