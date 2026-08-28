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


def _powf_correctly_rounded(base: FloatArray, exponent: float) -> FloatArray:
    array = np.asarray(base)
    if array.dtype == np.float64:
        return array ** np.float64(exponent)
    result = array.astype(np.float64) ** np.float64(np.float32(exponent))
    return result.astype(array.dtype)


def _sinf_correctly_rounded(phase: FloatArray) -> FloatArray:
    array = np.asarray(phase)
    if array.dtype == np.float64:
        return np.sin(array)
    return np.sin(array.astype(np.float64)).astype(array.dtype)


#: Rebindable hooks.  Default: correctly rounded via float64 (the pinned
#: defined behaviour).  The the proving RTX 5090 authority ladder may rebind these to the
#: reference executable's own libimf via ctypes to prove full byte identity.
powf_rkind: Callable[[FloatArray, float], FloatArray] = _powf_correctly_rounded
sinf_rkind: Callable[[FloatArray], FloatArray] = _sinf_correctly_rounded


__all__ = [
    "powf_rkind",
    "sinf_rkind",
]
