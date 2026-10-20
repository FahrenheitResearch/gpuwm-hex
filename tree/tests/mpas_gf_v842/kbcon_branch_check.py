"""Replay cap-limited repeated cloud-base searches against pristine GF.

MPAS module_cu_gf.mpas.F:3038-3103, including excess modes and the
iloop=4 error-preservation branch. Eight levels keep every accessed word
inside the native column; the checked oracle verifies that condition.
"""
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from auxiliary_check import invoke


def check(root, fixture, receipt, gpu):
    nz = 8
    if gpu:
        import cupy as cp
        from hexcore.mpasphys import gf_mpas_v842 as gf
        module = gf._compile_module(gf.cuda_source() + "\n" +
            Path(__file__).with_name("auxiliary_entry.cuh").read_text(), cp.cuda.runtime.getDevice())
        arena = cp.empty(65536, dtype=cp.uint8)
        saved = np.load(fixture)
    else:
        native = ctypes.CDLL(str(root / "gf_native.so"))
        host = ctypes.CDLL(str(root / "gf_host.so"))
        saved = {}
    differences = cases = 0
    for index, (iloop, excess) in enumerate(itertools.product(range(1, 6), range(4))):
        a = np.zeros((3, nz, 4), dtype="f4")
        a[0] = 400000. - np.arange(1, nz + 1, dtype="f4")[:, None] * 50000.
        a[1] = a[0] + 75000.
        a[2] = 1000. - np.arange(nz, dtype="f4")[:, None] * 80.
        s = np.array([[300000.] * 4, [50.] * 4, [25.] * 4,
                      [0.] * 4, [0.] * 4], dtype="f4")
        ints = np.array([[2] * 4, [-9] * 4, [5] * 4, [0, 0, 0, 7]], dtype="i4")
        text = np.full((4, 50), ord(" "), dtype="u1")
        arrays = [a, s, ints, text]
        if gpu:
            expected = [saved[f"case{index}_{i}"] for i in range(4)]
            device = [cp.asarray(x) for x in arrays]
            module.get_function("gf_aux_kbcon")((1,), (1,),
                tuple(np.int32(x) for x in (nz, iloop, excess)) + (*device, arena))
            got = [cp.asnumpy(x) for x in device]
        else:
            expected = [x.copy() for x in arrays]
            got = [x.copy() for x in arrays]
            invoke(native, "gf_native_kbcon", (nz, iloop, excess), expected)
            invoke(host, "gf_host_kbcon", (nz, iloop, excess), got)
            for i, x in enumerate(expected):
                saved[f"case{index}_{i}"] = x
        cases += 4
        differences += sum(int(np.count_nonzero(
            x.view("u4") != y.view("u4") if x.dtype.itemsize == 4 else x != y))
            for x, y in zip(expected, got))
    if not gpu:
        np.savez_compressed(fixture, **saved)
    result = dict(backend="hex-cuda" if gpu else "host-glibc-witness", cases=cases,
                  different_words_or_diagnostic_bytes=differences,
                  fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  source_lines="module_cu_gf.mpas.F:3038-3103", process_count=1,
                  scope="Pristine whole-module auxiliary witnesses; no forecast proof")
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(receipt.read_text())
    return bool(differences)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args()
    raise SystemExit(check(a.root.resolve(), a.fixture, a.receipt, a.gpu))
