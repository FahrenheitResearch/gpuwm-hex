#!/usr/bin/env python3
"""Compare a constructed vertical artifact with a native MPAS init oracle.

A report is always useful; a green scientific verdict is emitted only when the
supplied policy is explicitly marked ``ADMITTED``.  This prevents a provisional
or guessed tolerance table from becoming evidence merely because the arrays
look close.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FIELDS = (
    "ter",
    "zgrid",
    "zz",
    "zxu",
    "dss",
    "rdzw",
    "dzu",
    "fzm",
    "fzp",
    "zb",
    "zb3",
    "cf1",
    "cf2",
    "cf3",
)
POLICY_SCHEMA = "gpuwm-hex.vertical-comparison-policy/v1"
REPORT_SCHEMA = "gpuwm-hex.vertical-comparison-report/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"policy schema must be {POLICY_SCHEMA}")
    if raw.get("status") not in {"ADMITTED", "NOT_MEASURED"}:
        raise ValueError("policy status must be ADMITTED or NOT_MEASURED")
    rules = raw.get("fields")
    if not isinstance(rules, dict):
        raise ValueError("policy fields must be an object")
    missing = sorted(set(FIELDS) - set(rules))
    unknown = sorted(set(rules) - set(FIELDS))
    if missing or unknown:
        raise ValueError(f"policy field mismatch: missing={missing}, unknown={unknown}")
    for name, rule in rules.items():
        if rule.get("mode") not in {"exact", "tolerance"}:
            raise ValueError(f"{name}: mode must be exact or tolerance")
        if rule["mode"] == "tolerance":
            for key in ("atol", "rtol"):
                value = float(rule.get(key, -1.0))
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name}: {key} must be finite and non-negative")
    return raw


def array_stats(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        return {
            "shape_equal": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "exact": False,
        }
    same_dtype = candidate.dtype == reference.dtype
    exact = bool(same_dtype and np.array_equal(candidate, reference))
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    finite = np.isfinite(c) & np.isfinite(r)
    delta = np.abs(c - r)
    denominator = np.maximum(np.abs(r), np.finfo(np.float64).tiny)
    relative = delta / denominator
    return {
        "shape_equal": True,
        "candidate_shape": list(candidate.shape),
        "reference_shape": list(reference.shape),
        "candidate_dtype": str(candidate.dtype),
        "reference_dtype": str(reference.dtype),
        "dtype_equal": same_dtype,
        "exact": exact,
        "finite_pair_count": int(np.count_nonzero(finite)),
        "nonfinite_pair_count": int(finite.size - np.count_nonzero(finite)),
        "max_abs": float(np.max(delta[finite], initial=0.0)),
        "max_rel": float(np.max(relative[finite], initial=0.0)),
        "rmse": float(np.sqrt(np.mean(np.square(delta[finite]), dtype=np.float64))) if np.any(finite) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("native_reference", type=Path)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        from netCDF4 import Dataset
    except ImportError as error:
        raise SystemExit(f"netCDF4 is required: {error}")

    candidate = args.candidate.expanduser().resolve(strict=True)
    native = args.native_reference.expanduser().resolve(strict=True)
    policy_path = args.policy.expanduser().resolve(strict=True)
    policy = load_policy(policy_path)
    field_reports: dict[str, Any] = {}
    failures: list[str] = []

    with Dataset(candidate) as left, Dataset(native) as right:
        for name in FIELDS:
            if name not in left.variables or name not in right.variables:
                field_reports[name] = {
                    "present_candidate": name in left.variables,
                    "present_reference": name in right.variables,
                    "passed": False,
                }
                failures.append(f"{name}: missing variable")
                continue
            stats = array_stats(np.asarray(left.variables[name][:]), np.asarray(right.variables[name][:]))
            rule = policy["fields"][name]
            if rule["mode"] == "exact":
                passed = bool(stats.get("exact"))
            else:
                passed = bool(
                    stats.get("shape_equal")
                    and stats.get("nonfinite_pair_count") == 0
                    and stats.get("max_abs", float("inf")) <= float(rule["atol"])
                    and stats.get("max_rel", float("inf")) <= float(rule["rtol"])
                )
            stats["rule"] = rule
            stats["passed"] = passed
            field_reports[name] = stats
            if not passed:
                failures.append(f"{name}: comparison rule failed")

    admitted = policy["status"] == "ADMITTED"
    verdict = "PASS" if admitted and not failures else (
        "NOT_MEASURED" if not admitted else "FAIL"
    )
    payload = {
        "schema": REPORT_SCHEMA,
        "candidate": {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)},
        "native_reference": {"path": str(native), "bytes": native.stat().st_size, "sha256": sha256_file(native)},
        "policy": {"path": str(policy_path), "sha256": sha256_file(policy_path), "status": policy["status"]},
        "fields": field_reports,
        "failures": failures,
        "verdict": verdict,
        "nonclaim": (
            "NOT_MEASURED policy cannot support a parity claim; admit tolerances only after "
            "two materially different compiled-native authorities have been measured"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "failures": failures, "report": str(args.output)}, indent=2))
    return 0 if verdict in {"PASS", "NOT_MEASURED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
