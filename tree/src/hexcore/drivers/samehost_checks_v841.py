"""Byte-identity checks for complete same-host forecast boundary receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

__all__ = ["compare_boundary_records", "read_boundary_jsonl", "compare_boundary_files"]

EXPECTED_METADATA_DIVERGENCE_SUBSTRINGS = (
    "identity/constructor_identity_sha256",
    # the seam's identity block records the (sliced) constructor's column
    # extent -- size provenance of an input, not evolved state
    "identity/n_columns",
)


def _group_diffs(mine: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = sorted(set(mine) | set(reference))
    return [
        {"path": key, "candidate": mine.get(key), "reference": reference.get(key)}
        for key in keys
        if mine.get(key) != reference.get(key)
    ]


def compare_boundary_records(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Structured comparison of one boundary record pair."""

    state_diffs: list[dict[str, Any]] = []
    atmosphere_equal = candidate.get("atmosphere") == reference.get("atmosphere")
    if not atmosphere_equal:
        mine = candidate.get("atmosphere") or {}
        theirs = reference.get("atmosphere") or {}
        for group in ("state", "saved_diagnostics"):
            group_mine = (mine.get(group) or {}).get("fields", {})
            group_theirs = (theirs.get(group) or {}).get("fields", {})
            for diff in _group_diffs(group_mine, group_theirs):
                state_diffs.append({"group": f"atmosphere/{group}", **diff})
        if float(mine.get("model_time_seconds", -1)) != float(
            theirs.get("model_time_seconds", -2)
        ):
            state_diffs.append(
                {
                    "group": "atmosphere",
                    "path": "model_time_seconds",
                    "candidate": mine.get("model_time_seconds"),
                    "reference": theirs.get("model_time_seconds"),
                }
            )

    backend_mine = candidate.get("backend") or {}
    backend_theirs = reference.get("backend") or {}
    # Batch-order artifacts: staggered-u carriers the frozen Arwen WRF-grid
    # legacy coupling builds by rolling over the BATCH index (a function of
    # the local ordering by construction, unconsumed by the MPAS seam).  The
    # candidate names them per boundary; they are excluded from the byte gate
    # on BOTH sides, reported, and required to exist in the reference.
    artifacts = dict(candidate.get("batch_order_artifacts") or {})
    reference_arrays = dict(backend_theirs.get("arrays", {}))
    artifact_report: list[dict[str, Any]] = []
    reference_artifacts = dict(reference.get("batch_order_artifacts") or {})
    artifacts_sound = True
    if reference_artifacts:
        # An N-rank reference has already separated these arrays. Require
        # its complete artifact inventory and words instead of looking for
        # them among the consumed backend arrays.
        differences = _group_diffs(artifacts, reference_artifacts)
        artifacts_sound = not differences
        state_diffs.extend({"group":"backend/artifacts", **diff} for diff in differences)
        for path, info in sorted(artifacts.items()):
            present = path in reference_artifacts
            artifact_report.append({"path":path,"in_reference":present, **dict(info)})
        for path in sorted((set(artifacts) & set(backend_mine.get("arrays", {}))) |
                           (set(reference_artifacts) & set(reference_arrays))):
            artifacts_sound = False
            state_diffs.append({"group":"backend/artifacts","path":path,
                                "candidate":"artifact and consumed array overlap","reference":None})
    else:
        for path, info in sorted(artifacts.items()):
            present = path in reference_arrays
            artifacts_sound = artifacts_sound and present
            artifact_report.append({"path": path, "in_reference": present, **dict(info)})
            reference_arrays.pop(path, None)
    arrays_equal = backend_mine.get("arrays") == reference_arrays and artifacts_sound
    if not arrays_equal:
        for diff in _group_diffs(backend_mine.get("arrays", {}), reference_arrays):
            state_diffs.append({"group": "backend/arrays", **diff})
        for entry in artifact_report:
            if not entry["in_reference"]:
                state_diffs.append(
                    {"group": "backend/artifacts", "path": entry["path"],
                     "candidate": "artifact", "reference": None}
                )

    metadata_diffs = [
        {
            **diff,
            "expected_divergence": any(
                marker in str(diff["path"])
                for marker in EXPECTED_METADATA_DIVERGENCE_SUBSTRINGS
            ),
        }
        for diff in _group_diffs(
            backend_mine.get("scalars", {}), backend_theirs.get("scalars", {})
        )
    ]

    return {
        "state_invariant": bool(atmosphere_equal and arrays_equal),
        "atmosphere_equal": bool(atmosphere_equal),
        "backend_arrays_equal": bool(arrays_equal),
        "excluded_batch_order_artifacts": artifact_report,
        "state_diffs": state_diffs,
        "metadata_diffs": metadata_diffs,
        "unexpected_metadata_diffs": [
            diff for diff in metadata_diffs if not diff["expected_divergence"]
        ],
    }


def _read_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            step = int(record.pop("step"))
            if step in records:
                raise ValueError(f"{path}: duplicate boundary for step {step}")
            records[step] = record
    return records


read_boundary_jsonl = _read_jsonl


def compare_boundary_files(reference: Path, candidate: Path, *, require_steps: int | None = None) -> dict[str, Any]:
    """Require the same recorded clock boundaries, including the last one."""
    reference_records = _read_jsonl(reference)
    candidate_records = _read_jsonl(candidate)
    common = sorted(set(reference_records) & set(candidate_records))
    missing_reference = sorted(set(candidate_records) - set(reference_records))
    missing_candidate = sorted(set(reference_records) - set(candidate_records))
    outcomes = [{"step": step, **compare_boundary_records(candidate_records[step], reference_records[step])}
                for step in common]
    invariant = [entry["step"] for entry in outcomes if entry["state_invariant"]]
    unexpected = any(entry["unexpected_metadata_diffs"] for entry in outcomes)
    enough = require_steps is None or len(common) >= require_steps
    if not common:
        status = "NO-OVERLAP"
    elif missing_reference or missing_candidate or not enough:
        status = "INSUFFICIENT"
    elif len(invariant) != len(common):
        status = "FAIL"
    elif unexpected:
        status = "INSPECT"
    else:
        status = "PASS"
    return {"schema": "mpas-port.cuda-v841-samehost-invariance/v1",
            "reference": str(reference), "candidate": str(candidate),
            "status": status, "boundaries_compared": len(common),
            "boundaries_state_invariant": len(invariant),
            "candidate_steps_missing_in_reference": missing_reference,
            "reference_steps_missing_in_candidate": missing_candidate,
            "first_compared_step": common[0] if common else None,
            "last_compared_step": common[-1] if common else None,
            "final_state_identical": bool(common and common[-1] == max(reference_records)
                                           and common[-1] == max(candidate_records)
                                           and outcomes[-1]["state_invariant"]),
            "first_divergence": next((entry for entry in outcomes if not entry["state_invariant"]), None),
            "metadata_divergences": {str(entry["step"]): entry["metadata_diffs"]
                                     for entry in outcomes if entry["metadata_diffs"]},
            "unexpected_metadata_divergence": unexpected}
