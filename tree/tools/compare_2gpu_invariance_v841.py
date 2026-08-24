#!/usr/bin/env python3
"""Partition-invariance comparator (multi-GPU design D4, build item 7).

Compares a 2-GPU run's reconstructed ``boundary-fingerprints.jsonl`` against a
single-GPU reference's, record by record.  The gate law:

* ``state_invariant`` at a boundary demands byte equality of the FULL
  atmosphere record (state + saved diagnostics, every dtype/shape/sha256 and
  the model time) AND of every backend ARRAY leaf.  Owned regions are an
  exact disjoint cover, so the reconstructed union is the whole state and
  equality here is bitwise 2-GPU == 1-GPU.
* Backend NON-ARRAY leaves are provenance metadata, not evolved state.  One
  divergence class is expected by construction on a partitioned run --
  constructor identity digests hash the (sliced) constructor inputs.  Every
  metadata difference is reported verbatim either way; expected ones are
  annotated, unexpected ones make the overall verdict INSPECT, never PASS.

Exit code 0 = PASS (every compared boundary state-invariant, no unexpected
metadata drift), 1 = anything else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

SCHEMA = "mpas-port.cuda-v841-2gpu-invariance/v1"

# Metadata paths that hash constructor INPUTS (sliced on a partitioned run)
# rather than evolved state; substring match against the backend scalar path.
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
    artifacts_sound = True
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--require-steps",
        type=int,
        default=None,
        help="refuse unless the candidate carries at least this many boundaries",
    )
    args = parser.parse_args(argv)

    reference = _read_jsonl(args.reference)
    candidate = _read_jsonl(args.candidate)

    compared: list[int] = []
    invariant: list[int] = []
    first_divergence: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    missing_in_reference = sorted(set(candidate) - set(reference))
    for step in sorted(candidate):
        if step not in reference:
            continue
        outcome = compare_boundary_records(candidate[step], reference[step])
        compared.append(step)
        if outcome["state_invariant"]:
            invariant.append(step)
        elif first_divergence is None:
            first_divergence = {"step": step, **outcome}
        if outcome["metadata_diffs"]:
            metadata[str(step)] = outcome["metadata_diffs"]

    unexpected_metadata = any(
        any(not diff["expected_divergence"] for diff in diffs)
        for diffs in metadata.values()
    )
    enough = (
        args.require_steps is None or len(compared) >= int(args.require_steps)
    )
    all_invariant = bool(compared) and len(invariant) == len(compared)
    if not compared:
        status = "NO-OVERLAP"
    elif not enough:
        status = "INSUFFICIENT"
    elif not all_invariant:
        status = "FAIL"
    elif unexpected_metadata:
        status = "INSPECT"
    else:
        status = "PASS"

    verdict = {
        "schema": SCHEMA,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "boundaries_compared": len(compared),
        "boundaries_state_invariant": len(invariant),
        "first_compared_step": compared[0] if compared else None,
        "last_compared_step": compared[-1] if compared else None,
        "candidate_steps_missing_in_reference": missing_in_reference,
        "first_divergence": first_divergence,
        "metadata_divergences": metadata,
        "unexpected_metadata_divergence": unexpected_metadata,
        "status": status,
        "law": (
            "state_invariant = full atmosphere record byte-equal AND every "
            "backend array leaf byte-equal; metadata reported, expected class "
            "= sliced-constructor identity digests"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                "status": status,
                "compared": len(compared),
                "state_invariant": len(invariant),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
