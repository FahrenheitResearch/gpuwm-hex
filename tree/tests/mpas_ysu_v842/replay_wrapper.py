"""Check the native WRF wrapper against the exact host scheme body plus Exner division."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import numpy as np
from oracle import Oracle, wrapper_fixtures, provenance
from hexcore.mpasphys import ysu_v842 as ysu


def replay(work: Path, fixtures: Path):
    oracle = Oracle(work)
    mismatches, words, calls = [], 0, 0
    # Inactive cloud species are undefined local outputs in the native wrapper,
    # unlike the raw scheme's defined preservation of caller-owned carriers.
    for tag, arrays, initial in wrapper_fixtures(fixtures):
        for selector in (-1, 0, 1, 2):
            flags = {**initial, "nmix": 0, "ysu_topdown_pblmix": selector}
            arrays["qmix"] = np.empty(0, np.float32)
            arrays["qmixtnp"] = np.empty(0, np.float32)
            actual, status = oracle.run(arrays, {**flags, "ysu_topdown_pblmix": int(selector == 1)})
            if status:
                raise AssertionError(f"{tag}: unexpected scheme status {status}")
            # Test-only host conversion corresponds to the explicit RN device
            # divide in module_bl_ysu.F:448; never cancel the preceding multiply.
            actual["ttnp"] = np.divide(actual["ttnp"], arrays["pi2d"], dtype=np.float32)
            expected, _ = oracle.run(arrays, flags, wrapper=True)
            calls += 1
            for name in ysu._OUTPUT_LEVEL + ysu._OUTPUT_SURFACE + ("u10", "v10"):
                a, b = actual[name].view(np.uint32), expected[name].view(np.uint32)
                differing = int((a != b).sum())
                words += a.size
                if differing:
                    mismatches.append(dict(column=tag, selector=selector, array=name,
                                           differing_words=differing))
    return dict(**provenance(work, fixtures), schema="mpas-ysu-v842/wrapper-witness-v1", host=platform.node(), process_count=1,
                identity_tier="HOST-WITNESS-ONLY", compared_calls=calls,
                compared_words=words, differing_words=sum(m["differing_words"] for m in mismatches),
                mismatches=mismatches, cuda_sha256=ysu.CUDA_SHA256,
                not_proven=["device conversion", "real CLI", "whole forecast", "G0-G2"],
                native_undefined_inputs=["missing ctopo/ctopo2", "inactive qc or qi wrapper outputs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = replay(args.work, args.fixtures)
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    raise SystemExit(bool(receipt["differing_words"]))
