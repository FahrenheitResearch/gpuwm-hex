"""RKIND elementary functions with a pinned, defined rounding convention.

The regional byte ladder measured (2026-08-25, evidence
``regional-cpu-l4-20260825``) that instruction-level identity against a
default ``-O3`` ifx reference executable is not source-defined: ifx
fast-math reassociates and contracts at will, and even under the value-safe
reference flags (``-fp-model=source -no-fma -fimf-arch-consistency=true``)
Intel's ``powf`` is not correctly rounded — 4 of 326,810 real regional
exner arguments land 1 ulp off the correctly-rounded value (adjudicated
with 60-digit arithmetic), and ``-fimf-max-error=0.5`` measurably changes
nothing.  numpy/UCRT ``powf`` disagrees with libimf on 33 of the same
arguments.

House law (never bit-exact to a bug): where the native answer is an
implementation accident, the port pins DEFINED behaviour and documents the
divergence.  The defined behaviour here is CORRECT ROUNDING: evaluate in
float64 — whose own worst-case libm error is far below half a float32 ulp
on these domains — and round once to float32.  Against the value-safe
reference this leaves a measured residue of at most a few 1-ulp cells per
field per snapshot (the sites where libimf itself is off correct rounding).

For the byte-identity authority proof the hooks below can be rebound to a
ctypes bridge over the exact libimf the reference executable linked
(the proving RTX 5090), driving the residue to zero without changing the shipped
default.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


#: Arrays at least this large are evaluated in contiguous chunks on threads.
#: The functions are elementwise, so every element is computed exactly as in
#: one call; only the wall time changes.  MEASURED (8-card RTX PRO 6000 host, 2026-10-10, 15 km
#: global host preparation): the float64 pow over 2.6 million columns x 55
#: levels took 50 s of a 363 s host preparation on one core.
_PARALLEL_ELEMENTS = 1 << 20
_POOL = None


def _chunked(function: Callable[[np.ndarray], np.ndarray], array: np.ndarray, dtype: Any) -> np.ndarray:
    global _POOL
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.size < _PARALLEL_ELEMENTS:
        return function(flat).astype(dtype, copy=False).reshape(array.shape)
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, min(32, os.cpu_count() or 1))
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rkind-libm")
    out = np.empty(flat.size, dtype=dtype)
    bounds = np.linspace(0, flat.size, workers + 1, dtype=np.int64)

    def run(index: int) -> None:
        lo, hi = int(bounds[index]), int(bounds[index + 1])
        out[lo:hi] = function(flat[lo:hi])

    list(_POOL.map(run, range(workers)))
    return out.reshape(array.shape)


def _powf_correctly_rounded(base: FloatArray, exponent: float) -> FloatArray:
    array = np.asarray(base)
    if array.dtype == np.float64:
        power = np.float64(exponent)
        return _chunked(lambda part: part ** power, array, np.float64)
    power = np.float64(np.float32(exponent))
    return _chunked(lambda part: part.astype(np.float64) ** power, array, array.dtype)


def _sinf_correctly_rounded(phase: FloatArray) -> FloatArray:
    array = np.asarray(phase)
    if array.dtype == np.float64:
        return _chunked(np.sin, array, np.float64)
    return _chunked(lambda part: np.sin(part.astype(np.float64)), array, array.dtype)


#: Rebindable hooks.  Default: correctly rounded via float64 (the pinned
#: defined behaviour).  The proving RTX 5090 authority ladder may rebind these to the
#: reference executable's own libimf via ctypes to prove full byte identity.
powf_rkind: Callable[[FloatArray, float], FloatArray] = _powf_correctly_rounded
sinf_rkind: Callable[[FloatArray], FloatArray] = _sinf_correctly_rounded


__all__ = [
    "powf_rkind",
    "sinf_rkind",
]
