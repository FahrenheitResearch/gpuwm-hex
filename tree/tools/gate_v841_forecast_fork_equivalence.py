#!/usr/bin/env python3
"""Fork-equivalence gate for the derived MPAS-A v8.4.1 CUDA forecast driver.

QUESTION: did forking the sealed proof harness into
``tools/run_cuda_v841_forecast.py`` change the model?

METHOD: run the AUTHORITY init -- the exact file the release proof pinned --
through the derived driver for 30 steps (1 h) at the proven dt of 120 s, with
history every 30 minutes so the capture points coincide with the proof's
F000/F030/F001, and with a boundary fingerprint at every committed step.  Then
compare, bitwise, against the release proof's own uninterrupted arm as recorded
in its receipt:

  * step 15 boundary: the full MPAS atmosphere and Arwen backend fingerprints
    (``checkpoint_atmosphere`` / ``checkpoint_backend``), leaf by leaf;
  * step 16 boundary: the atmosphere and backend group digests (the release
    proof retained only the digests here, via its first-resumed-step-16
    identity gate);
  * step 30 boundary: the full atmosphere and backend fingerprints
    (``f001_full_state_identity``), leaf by leaf;
  * snapshots at steps 0, 15 and 30: every published array digest of the
    committed diagnostic surface (38 arrays per capture).

``q2`` is excluded from the verdict and reported informationally, following the
restart-proof precedent in ``_snapshot_hash_projection``.

BITWISE-IDENTICAL is required.  Any mismatch means the fork changed the model
and must be fixed before any showcase forecast runs.

If a mismatch appears only from step 16 onward while step 15 and the snapshots
at 0 and 15 are identical, re-run the candidate with ``--fingerprint-every 0``
before concluding: the driver's per-step instrumentation is deliberately
allocation-light, but the frozen Arwen phase-one seam has a documented
device-pool sensitivity, and that discriminates instrumentation from fork.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_cuda_v841_full_physics_x4 as proof  # noqa: E402

AUTHORITY_INIT_SHA256 = proof.AUTHORITY_PINS["init"]["sha256"]
CHECKPOINT_STEP = proof.CHECKPOINT_STEP
FULL_STEPS = proof.FULL_STEPS
SNAPSHOT_LABEL_BY_STEP = {0: "F000", CHECKPOINT_STEP: "F030", FULL_STEPS: "F001"}


def _leaves(value):
    return proof._fingerprint_leaf_projection(value)


def _compare_leaves(label, reference, candidate):
    left = _leaves(reference)
    right = _leaves(candidate)
    missing = object()
    mismatches = sorted(
        path
        for path in set(left) | set(right)
        if left.get(path, missing) != right.get(path, missing)
    )
    return {
        "label": label,
        "kind": "full-leaf",
        "reference_leaves": len(left),
        "candidate_leaves": len(right),
        "mismatches": mismatches[:200],
        "mismatch_count": len(mismatches),
        "identical": not mismatches,
    }


def _compare_digest(label, reference_sha, candidate_fingerprint):
    candidate_sha = candidate_fingerprint.get("sha256")
    return {
        "label": label,
        "kind": "group-digest",
        "reference_sha256": reference_sha,
        "candidate_sha256": candidate_sha,
        "identical": reference_sha == candidate_sha,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proof-receipt",
        type=Path,
        required=True,
        help="release-proof cuda-v841-full-physics-x4-receipt.json",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="output root of the derived driver's 30-step authority-init run",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    proof_payload = json.loads(args.proof_receipt.read_text("utf-8"))
    if proof_payload.get("status") != "passed":
        raise SystemExit("release-proof receipt is not a passed proof")
    reference = proof_payload["proof"]
    checkpoint_restart = reference["checkpoint_restart"]
    full_state = checkpoint_restart["f001_full_state_identity"]

    candidate_receipt = json.loads(
        (args.candidate / "cuda-v841-forecast-receipt.json").read_text("utf-8")
    )
    boundaries = {}
    with (args.candidate / "boundary-fingerprints.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            boundaries[int(record.pop("step"))] = record
    snapshot_hashes = json.loads(
        (args.candidate / "snapshot-hashes.json").read_text("utf-8")
    )

    report = {
        "proof_receipt": str(args.proof_receipt),
        "candidate": str(args.candidate),
        "preconditions": {},
        "boundaries": [],
        "snapshots": [],
    }

    # ---- preconditions: same init bytes, same Arwen, same executing sources.
    candidate_init = candidate_receipt["init"]
    proof_init = reference["authority"]["files"]["init"]
    preconditions = {
        "candidate_init_sha256": candidate_init["sha256"],
        "authority_init_sha256": AUTHORITY_INIT_SHA256,
        "proof_init_sha256": proof_init["sha256"],
        "init_is_the_authority": candidate_init["sha256"] == AUTHORITY_INIT_SHA256
        and proof_init["sha256"] == AUTHORITY_INIT_SHA256,
        "candidate_arwen_head": candidate_receipt["arwen_git"]["before"]["head"],
        "proof_arwen_head": proof_payload["arwen_git"]["before"]["head"],
        "arwen_head_matches": candidate_receipt["arwen_git"]["before"]["head"]
        == proof_payload["arwen_git"]["before"]["head"],
        "candidate_arwen_tree": candidate_receipt["arwen_git"]["before"]["tree"],
        "proof_arwen_tree": proof_payload["arwen_git"]["before"]["tree"],
        "arwen_tree_matches": candidate_receipt["arwen_git"]["before"]["tree"]
        == proof_payload["arwen_git"]["before"]["tree"],
        "candidate_source_pins_sha256": candidate_receipt["forecast"]["source_pins"][
            "sha256"
        ],
        "proof_source_pins_sha256": reference["source_pins"]["sha256"],
        "source_pins_match": candidate_receipt["forecast"]["source_pins"]["sha256"]
        == reference["source_pins"]["sha256"],
        "candidate_steps": candidate_receipt["forecast"]["walls"]["steps"],
        "candidate_dt_seconds": candidate_receipt["forecast"]["schedule"]["dt_seconds"],
    }
    preconditions["all_met"] = all(
        preconditions[name]
        for name in (
            "init_is_the_authority",
            "arwen_head_matches",
            "arwen_tree_matches",
            "source_pins_match",
        )
    ) and preconditions["candidate_steps"] == FULL_STEPS and float(
        preconditions["candidate_dt_seconds"]
    ) == float(proof.DT_SECONDS)
    report["preconditions"] = preconditions
    for name, value in sorted(preconditions.items()):
        print(f"PRECONDITION {name}={value}", flush=True)
    if not preconditions["all_met"]:
        print("VERDICT: GATE VOID (preconditions not met)", flush=True)
        report["verdict"] = "VOID"
        if args.out is not None:
            args.out.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
        return 2

    failures = 0

    # ---- boundary comparisons.
    comparisons = []
    for step, ref_atmosphere, ref_backend in (
        (
            CHECKPOINT_STEP,
            checkpoint_restart["checkpoint_atmosphere"],
            checkpoint_restart["checkpoint_backend"],
        ),
        (FULL_STEPS, full_state["atmosphere"], full_state["backend"]),
    ):
        if step not in boundaries:
            comparisons.append(
                {"label": f"step {step}", "identical": False, "error": "absent"}
            )
            continue
        comparisons.append(
            _compare_leaves(
                f"step {step} atmosphere", ref_atmosphere, boundaries[step]["atmosphere"]
            )
        )
        comparisons.append(
            _compare_leaves(
                f"step {step} backend", ref_backend, boundaries[step]["backend"]
            )
        )
    step16 = full_state["first_resumed_step16"]
    if CHECKPOINT_STEP + 1 in boundaries:
        record = boundaries[CHECKPOINT_STEP + 1]
        comparisons.append(
            _compare_digest(
                f"step {CHECKPOINT_STEP + 1} atmosphere",
                step16["atmosphere"]["sha256"],
                record["atmosphere"],
            )
        )
        comparisons.append(
            _compare_digest(
                f"step {CHECKPOINT_STEP + 1} backend",
                step16["backend"]["sha256"],
                record["backend"],
            )
        )
    for record in comparisons:
        report["boundaries"].append(record)
        if record.get("identical"):
            print(
                f"PASS boundary {record['label']} ({record['kind']})"
                + (
                    f" leaves={record['reference_leaves']}"
                    if "reference_leaves" in record
                    else ""
                ),
                flush=True,
            )
        else:
            failures += 1
            print(
                f"FAIL boundary {record['label']} "
                f"mismatches={record.get('mismatches', record.get('error'))}",
                flush=True,
            )

    # ---- snapshot comparisons.
    for step, label in sorted(SNAPSHOT_LABEL_BY_STEP.items()):
        reference_arrays = reference["uninterrupted"]["snapshot_receipts"][label][
            "arrays"
        ]
        reference_projection = {
            name: value["sha256"]
            for name, value in reference_arrays.items()
            if name != "q2"
        }
        candidate_projection = snapshot_hashes["projection"].get(str(step), {})
        mismatches = sorted(
            name
            for name in set(reference_projection) | set(candidate_projection)
            if reference_projection.get(name) != candidate_projection.get(name)
        )
        q2_reference = reference_arrays.get("q2", {}).get("sha256")
        q2_candidate = snapshot_hashes["q2"].get(str(step))
        record = {
            "step": step,
            "proof_label": label,
            "reference_arrays": len(reference_projection),
            "candidate_arrays": len(candidate_projection),
            "mismatches": mismatches,
            "identical": not mismatches,
            "q2_identical": q2_reference == q2_candidate,
        }
        report["snapshots"].append(record)
        if record["identical"]:
            print(
                f"PASS snapshot step={step:02d} ({label}) "
                f"arrays={len(reference_projection)} "
                f"q2_identical={record['q2_identical']} (q2 informational)",
                flush=True,
            )
        else:
            failures += 1
            print(
                f"FAIL snapshot step={step:02d} ({label}) mismatches={mismatches}",
                flush=True,
            )

    report["compared_boundaries"] = len(report["boundaries"])
    report["compared_snapshots"] = len(report["snapshots"])
    report["failures"] = failures
    if failures == 0:
        report["verdict"] = "BITWISE-IDENTICAL"
        print("VERDICT: BITWISE-IDENTICAL -- the fork did not change the model", flush=True)
    else:
        report["verdict"] = "MISMATCH"
        print(
            f"VERDICT: MISMATCH in {failures} comparison(s) -- the fork changed "
            "the model; fix before any showcase sim",
            flush=True,
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
