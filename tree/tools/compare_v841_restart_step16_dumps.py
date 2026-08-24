#!/usr/bin/env python3
"""Compare the 15+1 baseline/restored step-16 cutpoint dumps bitwise.

Reads the two dump roots produced by ``diagnose_v841_restart_step16_x4.py``,
walks the cutpoints in execution order, and reports the FIRST cutpoint whose
arrays or comparable scalars differ.  For every mismatching array at that
cutpoint it prints the first differing flat index, both values, and both bit
patterns.  Exit code 0 = bitwise identical everywhere; 1 = divergence found;
2 = structural problem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CUTPOINTS = (
    "prestep",
    "phase1_raw",
    "phase1_held",
    "post_dycore",
    "post_wsm6",
    "committed",
)

# Scalar meta keys that are phase-label noise, not trajectory state.
_IGNORED_SCALAR_SUFFIXES = (
    "backend_phase",
    "device_pointers",
    "clamp_d2h",
)


def _bits(value: np.ndarray) -> str:
    kind = value.dtype
    if kind == np.float32:
        return f"0x{value.view(np.uint32):08x}"
    if kind == np.float64:
        return f"0x{value.view(np.uint64):016x}"
    if kind.kind in "iu":
        return hex(int(value))
    return repr(value.tobytes())


def _comparable_scalars(meta: dict) -> dict:
    flat: dict[str, object] = {}

    def walk(prefix: str, item: object) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                walk(f"{prefix}/{key}" if prefix else str(key), item[key])
        else:
            flat[prefix] = item

    walk("", meta)
    return {
        key: value
        for key, value in flat.items()
        if not any(part in key for part in _IGNORED_SCALAR_SUFFIXES)
    }


def compare_cutpoint(base_root: Path, rest_root: Path, cutpoint: str) -> dict:
    base_manifest = json.loads((base_root / f"{cutpoint}.manifest.json").read_text())
    rest_manifest = json.loads((rest_root / f"{cutpoint}.manifest.json").read_text())
    result: dict = {"cutpoint": cutpoint, "scalar_mismatches": [], "array_mismatches": []}

    base_scalars = _comparable_scalars(base_manifest.get("meta", {}))
    rest_scalars = _comparable_scalars(rest_manifest.get("meta", {}))
    for key in sorted(set(base_scalars) | set(rest_scalars)):
        missing = object()
        b = base_scalars.get(key, missing)
        r = rest_scalars.get(key, missing)
        if b != r:
            result["scalar_mismatches"].append(
                {"path": key, "baseline": repr(b), "restored": repr(r)}
            )

    base_arrays = base_manifest["arrays"]
    rest_arrays = rest_manifest["arrays"]
    names = sorted(set(base_arrays) | set(rest_arrays))
    differing = [
        name
        for name in names
        if base_arrays.get(name, {}).get("sha256") != rest_arrays.get(name, {}).get("sha256")
    ]
    if differing:
        base_npz = np.load(base_root / f"{cutpoint}.npz")
        rest_npz = np.load(rest_root / f"{cutpoint}.npz")
        for name in differing:
            if name not in base_npz.files or name not in rest_npz.files:
                result["array_mismatches"].append(
                    {"path": name, "error": "present in only one arm"}
                )
                continue
            b = base_npz[name]
            r = rest_npz[name]
            if b.shape != r.shape or b.dtype != r.dtype:
                result["array_mismatches"].append(
                    {
                        "path": name,
                        "error": f"layout {b.dtype}{b.shape} != {r.dtype}{r.shape}",
                    }
                )
                continue
            bb = np.ascontiguousarray(b).reshape(-1)
            rr = np.ascontiguousarray(r).reshape(-1)
            # Bitwise comparison catches -0.0 vs +0.0 and NaN payload changes.
            if bb.dtype == np.float32:
                mask = bb.view(np.uint32) != rr.view(np.uint32)
            elif bb.dtype == np.float64:
                mask = bb.view(np.uint64) != rr.view(np.uint64)
            else:
                mask = bb != rr
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                result["array_mismatches"].append(
                    {"path": name, "error": "sha mismatch but no element diff (layout/dtype?)"}
                )
                continue
            first = int(indices[0])
            entry = {
                "path": name,
                "dtype": b.dtype.str,
                "shape": list(b.shape),
                "count_differing": int(indices.size),
                "first_flat_index": first,
                "first_multi_index": [int(v) for v in np.unravel_index(first, b.shape)],
                "baseline_value": repr(bb[first]),
                "restored_value": repr(rr[first]),
                "baseline_bits": _bits(bb[first]),
                "restored_bits": _bits(rr[first]),
                "max_abs_diff": float(
                    np.max(np.abs(bb[mask].astype(np.float64) - rr[mask].astype(np.float64)))
                )
                if b.dtype.kind == "f"
                else None,
            }
            result["array_mismatches"].append(entry)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    report: dict = {"cutpoints": [], "first_divergence": None, "identical": True}
    for cutpoint in CUTPOINTS:
        base_path = args.baseline / f"{cutpoint}.manifest.json"
        rest_path = args.restored / f"{cutpoint}.manifest.json"
        if not base_path.is_file() or not rest_path.is_file():
            print(f"[compare] MISSING manifests for cutpoint {cutpoint}")
            report["cutpoints"].append({"cutpoint": cutpoint, "error": "missing"})
            report["identical"] = False
            if report["first_divergence"] is None:
                report["first_divergence"] = {"cutpoint": cutpoint, "error": "missing"}
            break
        result = compare_cutpoint(args.baseline, args.restored, cutpoint)
        clean = not result["scalar_mismatches"] and not result["array_mismatches"]
        print(f"[compare] {cutpoint}: {'IDENTICAL' if clean else 'DIVERGED'}")
        report["cutpoints"].append(result)
        if not clean:
            report["identical"] = False
            report["first_divergence"] = result
            for item in result["scalar_mismatches"]:
                print(f"  scalar {item['path']}: baseline={item['baseline']} restored={item['restored']}")
            for item in result["array_mismatches"]:
                print(f"  array {item['path']}: {json.dumps(item, default=repr)}")
            break

    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=repr) + "\n",
            encoding="utf-8",
        )
        print(f"[compare] report written to {args.report}")
    if report["identical"]:
        print("[compare] ALL CUTPOINTS BITWISE IDENTICAL")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
