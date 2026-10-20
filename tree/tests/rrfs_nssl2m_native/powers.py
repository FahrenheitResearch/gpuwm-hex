"""Compare the actual translated integer-power operators with strict GNU.

Raw synthetic words include zeros, subnormals, infinities and NaN payloads.
This diagnostic catches multiply-chain gaps that complete-call fixtures miss;
it is not CUDA or forecast proof.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "host", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    rng = np.random.default_rng(170532)
    inputs = []
    for dtype, words, special in (
        (np.float32, np.uint32, [0, 0x80000000, 1, 0x80000001, 0x007fffff,
                                 0x7f800000, 0xff800000, 0x7fc12345, 0x7f812345]),
        (np.float64, np.uint64, [0, 0x8000000000000000, 1, 0x8000000000000001,
                                 0x000fffffffffffff, 0x7ff0000000000000,
                                 0xfff0000000000000, 0x7ff8123456789abc,
                                 0x7ff0123456789abc]),
    ):
        raw = np.frombuffer(rng.bytes(16384*np.dtype(words).itemsize), dtype=words).copy()
        raw[:len(special)] = special
        inputs.append(raw.view(dtype))
    n = len(inputs[0])
    outputs = []
    for path, symbol in ((a.native, "rrfs_nssl2m_powers"), (a.host, "nssl_host_powers")):
        lib = ctypes.CDLL(str(path.resolve()))
        call = getattr(lib, symbol)
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        call.argtypes = [ctypes.c_int, f32, f64, f32, f64]
        call.restype = None
        result = [np.empty((8,n), np.float32), np.empty((8,n), np.float64)]
        call(n, *inputs, *result)
        outputs.append(result)
    rows = {}
    for index, words in enumerate((np.uint32, np.uint64)):
        reference, candidate = [out[index].view(words) for out in outputs]
        different = np.argwhere(reference != candidate)
        width = np.dtype(words).itemsize*2
        rows[np.dtype(words).name] = dict(words=int(reference.size), differing_words=len(different),
            input_sha256=hashlib.sha256(inputs[index].tobytes()).hexdigest(),
            witnesses=[dict(exponent=int(k)+2, input_word=f"{int(inputs[index].view(words)[i]):0{width}x}",
                            native_word=f"{int(reference[k,i]):0{width}x}",
                            candidate_word=f"{int(candidate[k,i]):0{width}x}") for k,i in different[:8]])
    result = dict(scope="host integer-power lowering versus strict GNU; no CUDA/forecast claim",
                  native_library_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),
                  host_library_sha256=hashlib.sha256(a.host.read_bytes()).hexdigest(), arrays=rows)
    a.out.write_text(json.dumps(result, indent=2)+"\n")
    failures = sum(row["differing_words"] for row in rows.values())
    print(f"Integer-power diagnostic: {failures} differing words")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
