#!/usr/bin/env python3
"""Validate measured module receipts and summarize their branch/word counts.

This processes test receipts only, never weather arrays. Native authority is
mpas_atm_dissipation_models.F:579-1606 and the extracted upper damping blocks
mpas_atm_time_integration.F:6390-6408,6610-6629. It makes no forecast claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


FIELDS = frozenset((
    "tend_u_euler", "tend_w_euler", "tend_theta_euler", "tend_scalars",
    "delsq_u", "delsq_divergence", "delsq_vorticity", "delsq_w",
    "delsq_theta", "kdiff",
))


def validate_receipt(path: Path, *, require_direct_nvrtc=False) -> dict:
    """Require every measured scientific branch and every upper-deck word."""
    raw = path.read_bytes()
    receipt = json.loads(raw)
    if require_direct_nvrtc:
        required = {"--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"}
        inventory = receipt.get("compile_inventory", [])
        if (receipt.get("compiler") != "direct NVRTC"
                or receipt.get("native_process_count") != 1 or receipt.get("hex_process_count") != 1
                or receipt.get("mpi") is not False or not inventory
                or not required.issubset(receipt.get("cuda_options", []))
                or any(row["compiler"] != "direct NVRTC"
                       or not required.issubset(row["options"]) for row in inventory)):
            raise ValueError(f"{path}: gradual-underflow compiler/process contract is not witnessed")
    comparisons = receipt["comparisons"]
    if (receipt["status"] != "PASS" or receipt["different_words"] != 0
            or receipt["excluded_words"] != []):
        raise ValueError(f"{path}: receipt does not claim unexcluded identity")
    if not (len(comparisons) == receipt["completed_comparisons"]
            == receipt["expected_comparisons"]):
        raise ValueError(f"{path}: comparison count is incomplete")
    if sum(c["words"] for c in comparisons) != receipt["words"]:
        raise ValueError(f"{path}: aggregate word count is inconsistent")
    branch_fields = defaultdict(set)
    labels = set()
    upper_labels = set()
    for comparison in comparisons:
        label = comparison["branch"]
        if label in labels:
            raise ValueError(f"{path}: duplicate comparison {label}")
        labels.add(label)
        if (comparison["different_words"] != 0 or comparison["words"] <= 0
                or comparison["native_sha256"] != comparison["cuda_sha256"]):
            raise ValueError(f"{path}: word identity failed for {label}")
        branch, field = label.rsplit(":", 1)
        if field in FIELDS:
            branch_fields[branch].add(field)
        else:
            upper_labels.add(label)
    expected_upper = {f"{kind}:{deck}" for kind in ("CAM", "Rayleigh")
                      for deck in range(1, 4)}
    if upper_labels != expected_upper:
        raise ValueError(f"{path}: extracted upper-driver decks are incomplete")
    groups = defaultdict(set)
    for label, fields in branch_fields.items():
        if fields != FIELDS:
            raise ValueError(f"{path}: fields missing from scientific branch {label}")
        prefix, _, number = label.rpartition(":")
        groups[prefix].add(int(number))
    if not groups or any(numbers != set(range(1, 257)) for numbers in groups.values()):
        raise ValueError(f"{path}: crossed native branch matrix is incomplete")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "comparisons": len(comparisons),
        "words": receipt["words"],
        "scientific_branch_evaluations": len(branch_fields),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("receipts", nargs="+")
    parser.add_argument("--require-direct-nvrtc", action="store_true")
    args = parser.parse_args()
    if len(set(args.receipts)) != len(args.receipts):
        raise ValueError("receipt arguments must be unique")
    root = args.receipt_root.resolve()
    summaries = []
    for name in args.receipts:
        path = (root / name).resolve()
        path.relative_to(root)
        summaries.append({"receipt": name, **validate_receipt(path,
            require_direct_nvrtc=args.require_direct_nvrtc)})
    summary = {
        "status": "PASS",
        "scope": "standalone compiled native/CUDA module only; no forecast acceptance",
        "comparisons": sum(s["comparisons"] for s in summaries),
        "words": sum(s["words"] for s in summaries),
        "different_words": 0,
        "excluded_words": [],
        "compiler_contract_required": "direct NVRTC" if args.require_direct_nvrtc else "historical",
        "scientific_branch_evaluations": sum(
            s["scientific_branch_evaluations"] for s in summaries),
        "receipts": summaries,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}))


if __name__ == "__main__":
    main()
