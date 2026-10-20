"""Queued CUDA launcher replay against pristine MPAS, with no tolerances."""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import chain
import json
import os
from pathlib import Path
import platform

import cupy as cp
import numpy as np
from hexcore.mpasphys import ysu_v842 as ysu
from oracle import Oracle, OUTPUTS, constants, extended_fixtures, tracer_fixtures, branch_fixtures, wrapper_fixtures, provenance, cuda_provenance


def pack(columns, flags):
    """Test-only layout construction; production only accepts device carriers."""
    nz, nmix = flags["kte"], flags["nmix"]
    state = {}
    for name in ysu.ARRAY_ARGUMENTS:
        first = columns[0][name]
        if first is None:
            state[name] = None
            continue
        if name in ("qmix", "qmixtnp"):
            data = np.stack([c[name].reshape(nmix, nz) for c in columns], axis=-1)
        elif name in ysu._REQUIRED_LEVEL + ysu._OUTPUT_LEVEL + ysu._OPTIONAL_LEVEL + ("p2di",):
            data = np.stack([c[name] for c in columns], axis=-1)
        else:
            data = np.concatenate([c[name] for c in columns])
        state[name] = cp.asarray(np.ascontiguousarray(data))
    return state


def replay(work, fixtures):
    os.environ.setdefault("GCOV_PREFIX", str((work.parent / "cuda-gcov").resolve()))
    oracle = Oracle(work)
    header = ysu.shared_device_math_header()
    groups = defaultdict(list)
    for tag, arrays, flags in chain(extended_fixtures(fixtures), tracer_fixtures(fixtures), branch_fixtures(fixtures)):
        key = tuple(flags[k] for k in ysu.INTEGER_ARGUMENTS), tuple(k for k, v in arrays.items() if v is None)
        groups[key].append((tag, arrays, flags))
    compared = words = differing = 0
    mismatches = []
    per_array = {}

    def compare(tag, result, expected, mapping):
        nonlocal words, differing
        for name, target in mapping.items():
            if name not in result or expected[target] is None:
                continue
            a = result[name].reshape(-1).view(np.uint32)
            b = expected[target].reshape(-1).view(np.uint32)
            diff = int(np.count_nonzero(a != b))
            words += a.size
            differing += diff
            key = ("wrapper:" if tag.startswith("wrapper:") else "raw:") + name
            totals = per_array.setdefault(key, dict(compared_words=0, differing_words=0))
            totals["compared_words"] += a.size
            totals["differing_words"] += diff
            if diff and len(mismatches) < 100:
                first = int(np.flatnonzero(a != b)[0])
                mismatches.append(dict(column=tag, array=name, differing_words=diff,
                                       first_index=first, actual_word=int(a[first]), native_word=int(b[first])))

    values = constants()
    for group in groups.values():
        flags = group[0][2]
        # A five-column tile tests interleaved workspaces, repeated launches,
        # and the partial final tile. The wide launch checks the other layout.
        for tile in (5, 128):
            state = pack([c[1] for c in group], flags)
            out = ysu.bl_ysu_run(state, values, dt=values["dt"], f_qc=bool(flags["f_qc"]),
                                 f_qi=bool(flags["f_qi"]), flag_bep=bool(flags["flag_bep"]),
                                 ysu_topdown_pblmix=bool(flags["ysu_topdown_pblmix"]),
                                 workspace_columns=tile, device_math_header=header)
            actual = {k: cp.asnumpy(v) for k, v in out.items()}
            for col, (tag, arrays, _) in enumerate(group):
                expected, _ = oracle.run(arrays, flags, native=True)
                compare(f"{tag},tile={tile}", {k: v[..., col] for k, v in actual.items()},
                        expected, {k: k for k in OUTPUTS})
                compared += 1
        print(f"raw group {len(group)} columns complete; differing words {differing}", flush=True)

    wrapper_calls = 0
    wrapper_groups = defaultdict(list)
    for tag, arrays, flags in wrapper_fixtures(fixtures):
        key = tuple(flags[k] for k in ysu.INTEGER_ARGUMENTS), tuple(k for k, v in arrays.items() if v is None)
        wrapper_groups[key].append((tag, arrays, flags))
    for group in wrapper_groups.values():
        flags = group[0][2]
        for selector in (-1, 0, 1, 2):
            state = pack([c[1] for c in group], flags)
            out = ysu.ysu(state, values, dt=values["dt"], flag_qc=True, flag_qi=True,
                          flag_bep=bool(flags["flag_bep"]), ysu_topdown_pblmix=selector,
                          device_math_header=header)
            actual = {k: cp.asnumpy(v) for k, v in out.items()}
            mapping = {**ysu._WRAPPER_OUTPUTS, "rthblten": "ttnp"}
            for col, (tag, arrays, _) in enumerate(group):
                expected, _ = oracle.run(arrays, {**flags, "nmix": 0,
                                                 "ysu_topdown_pblmix": selector}, wrapper=True)
                compare(f"wrapper:{tag},selector={selector}",
                        {k: v[..., col] for k, v in actual.items()}, expected, mapping)
                wrapper_calls += 1
    result = dict(**provenance(work, fixtures), schema="mpas-ysu-v842/cuda-replay-v1", host=platform.node(), process_count=1,
                  libc=platform.libc_ver(), identity_tier="SYNTHETIC-CUDA-COLUMNS",
                  raw_calls=compared, wrapper_calls=wrapper_calls, compared_words=words,
                  differing_words=differing, mismatches=mismatches,
                  per_array=per_array,
                  scheme_sha256=ysu.SOURCE_SHA256, cuda_sha256=ysu.CUDA_SHA256,
                  shared_math_sha256=ysu.sha256(header.encode()).hexdigest(),
                  compiler_options=ysu.COMPILER_OPTIONS, device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                  not_proven=["real columns", "real CLI selection", "whole forecast", "G0-G2"])
    result.update(cuda_provenance())
    assert sum(v["compared_words"] for v in per_array.values()) == words
    assert sum(v["differing_words"] for v in per_array.values()) == differing
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = replay(args.work, args.fixtures)
    args.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "mismatches"}, indent=2), flush=True)
    raise SystemExit(bool(receipt["differing_words"]))
