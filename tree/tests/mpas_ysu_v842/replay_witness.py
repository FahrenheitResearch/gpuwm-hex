"""Compare complete raw-scheme outputs with pristine Fortran, word for word."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
from itertools import chain
from oracle import Oracle, OUTPUTS, extended_fixtures, tracer_fixtures, branch_fixtures, provenance
from hexcore.mpasphys import ysu_v842 as ysu
import numpy as np


def replay(work: Path, fixtures: Path):
    oracle = Oracle(work)
    mismatches = []
    rejected = []
    compared = 0
    words = 0
    for tag, arrays, flags in chain(extended_fixtures(fixtures), tracer_fixtures(fixtures), branch_fixtures(fixtures)):
        actual, status = oracle.run(arrays, flags)
        if status:
            rejected.append({"column": tag, "status": status,
                             "defect": f"unexpected scheme status {status}"})
            continue
        expected, _ = oracle.run(arrays, flags, native=True)
        compared += 1
        for name in OUTPUTS:
            if actual[name] is None:
                continue
            a, b = actual[name].view(np.uint32), expected[name].view(np.uint32)
            differing = int((a != b).sum())
            words += a.size
            if differing:
                first = int(np.flatnonzero(a != b)[0])
                mismatches.append({"column": tag, "array": name,
                                   "differing_words": differing, "first_index": first,
                                   "actual_word": int(a[first]), "native_word": int(b[first])})
    return {**provenance(work, fixtures), "schema": "mpas-ysu-v842/host-witness-v1",
            "identity_tier": "HOST-WITNESS-ONLY", "process_count": 1, "host": platform.node(),
            "libc": platform.libc_ver(), "compared_calls": compared,
            "compared_words": words, "differing_words": sum(d["differing_words"] for d in mismatches),
            "mismatches": mismatches, "native_undefined_columns_rejected": rejected,
            "scheme_sha256": ysu.SOURCE_SHA256, "cuda_sha256": ysu.CUDA_SHA256,
            "witness_sha256": hashlib.sha256((work / "witness.cpp").read_bytes()).hexdigest(),
            "not_proven": ["CUDA column identity", "real CLI selection", "whole forecast", "G0-G2"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = replay(args.work, args.fixtures)
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("mismatches", "native_undefined_columns_rejected")}, indent=2))
    print("rejected native-undefined calls", len(receipt["native_undefined_columns_rejected"]))
    raise SystemExit(bool(receipt["differing_words"]))
