"""Strict witnesses for dry cloud-base and Berry detrainment branches.

Uses the whole pristine module through its existing auxiliary ABI; synthetic
inputs vary cloud water history and entrainment without altering native code.
"""
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from auxiliary_check import invoke


def check(root, fixture, receipt, gpu, module_source=None):
    if gpu:
        import cupy as cp
        from hexcore.mpasphys import gf_mpas_v842 as gf
        if module_source is not None:
            import importlib.util
            import sys
            spec = importlib.util.spec_from_file_location("gf_frozen_witness", module_source)
            gf = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = gf
            spec.loader.exec_module(gf)
        module = gf._compile_module(gf.cuda_source() + "\n" +
            Path(__file__).with_name("auxiliary_entry.cuh").read_text(), cp.cuda.runtime.getDevice())
        arena = cp.empty(65536, dtype=cp.uint8)
        saved = np.load(fixture)
    else:
        native = ctypes.CDLL(str(root / "gf_native.so"))
        host = ctypes.CDLL(str(root / "gf_host.so"))
        saved = {}
    cases = differences = nonfinite = 0
    mismatch_patterns = {}
    nz = 8
    for index, (qe, entr, detr, dry, shallow, autoconv, excess) in enumerate(itertools.product(
            (.00001, .01), (1., 10., 100., 1000.), (.1, 1.),
            (.002001, .00205, .004), (0, 1), (1, 2), (0, 2))):
        a = np.zeros((17, nz, 4), dtype="f4")
        a[4] = np.arange(nz, dtype="f4")[:, None] * np.float32(500.)
        a[5] = 1000. - np.arange(nz, dtype="f4")[:, None] * np.float32(80.)
        a[6] = .001
        a[7] = 100.
        a[8] = 280.
        a[9] = .04
        a[9, 3:] = dry
        a[10] = 100.
        a[11] = 1.
        a[12] = .002
        a[13] = qe
        a[14] = .02
        a[15] = 1.
        a[15, 3:] = entr
        a[16] = detr
        s = np.array([[0.] * 4, [0.] * 4, [0.] * 4,
                      [0.] * 4, [1., 100., 10000., 100.]], dtype="f4")
        ints = np.array([[0, 0, 0, 7], [3] * 4, [6] * 4, [2] * 4], dtype="i4")
        arrays = [a, s, ints]
        if gpu:
            expected = [saved[f"case{index}_{i}"] for i in range(3)]
            device = [cp.asarray(x) for x in arrays]
            module.get_function("gf_aux_up")((1,), (1,),
                tuple(np.int32(x) for x in (nz, shallow, excess, autoconv)) + (*device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [x.copy() for x in arrays]
            got = [x.copy() for x in arrays]
            invoke(native, "gf_native_up", (nz, shallow, excess, autoconv), expected)
            invoke(host, "gf_host_up", (nz, shallow, excess, autoconv), got)
            for i, x in enumerate(expected):
                saved[f"case{index}_{i}"] = x
        cases += 4
        differences += sum(int(np.count_nonzero(x.view("u4") != y.view("u4")))
                           for x, y in zip(expected, got))
        for x, y in zip(expected, got):
            left, right = x.view("u4").ravel(), y.view("u4").ravel()
            for native_word, cuda_word in zip(left[left != right], right[left != right]):
                pattern = f"{int(native_word):08x}->{int(cuda_word):08x}"
                mismatch_patterns[pattern] = mismatch_patterns.get(pattern, 0) + 1
        nonfinite += sum(int(np.count_nonzero(~np.isfinite(x)))
                         for x in expected if x.dtype == np.float32)
    if not gpu:
        np.savez_compressed(fixture, **saved)
    result = dict(backend="hex-cuda" if gpu else "host-glibc-witness", cases=cases,
                  different_words=differences, native_nonfinite_words=nonfinite,
                  mismatch_patterns=mismatch_patterns,
                  fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  source_lines="module_cu_gf.mpas.F:3928,3997", process_count=1,
                  scope="Direct pristine-module cloud-history witnesses; no forecast proof")
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(receipt.read_text())
    return bool(differences)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--module-source", type=Path)
    a = p.parse_args()
    raise SystemExit(check(a.root.resolve(), a.fixture, a.receipt, a.gpu, a.module_source))
