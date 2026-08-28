"""Small v8.4.1 whole-step invariants kept outside the v8.2.3 path."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]


def enforce_recovered_rw_endpoints_v841(rho_w: FloatArray) -> FloatArray:
    """Apply v8.4.1's explicit recovered ``rw`` bottom/top conditions.

    The array is mutated and returned so the driver cannot accidentally apply
    the boundary condition to a detached copy.  Interior bits are untouched.
    """

    value = np.asarray(rho_w)
    if value.ndim != 2 or value.shape[0] < 2 or value.dtype.kind != "f":
        raise ValueError("rho_w must be a floating (nVertLevels+1, nCells) array")
    value[0, :] = value.dtype.type(0.0)
    value[-1, :] = value.dtype.type(0.0)
    return value


__all__ = ["enforce_recovered_rw_endpoints_v841"]
