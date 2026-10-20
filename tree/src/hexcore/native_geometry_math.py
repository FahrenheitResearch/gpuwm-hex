"""Host float intrinsics used by the GNU strict deformation geometry.

Strict mode binds GNU 2.43 / CORE-MATH float32 implementations, from the
wheel or a source-addressed build of the shipped sources, so geometry
matches the native strict reference across host glibc versions. Fast mode
uses the host GNU library when available. Float64 inputs retain NumPy.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import platform
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_LIBM = None
if platform.libc_ver()[0] == "glibc":
    _name = ctypes.util.find_library("m")
    if _name is not None:
        _LIBM = ctypes.CDLL(_name)


@lru_cache(maxsize=1)
def _strict_library():
    path = Path(__file__).with_name("libexec") / "libhex_geometry_math.so"
    if not path.is_file():
        from .geometry_math_build import cached_geometry_math
        path = cached_geometry_math(Path(__file__).parent)
    return ctypes.CDLL(str(path))


@lru_cache(maxsize=4)
def _strict_intrinsic(name: str, nargs: int):
    native = getattr(_strict_library(), "hex_" + name)
    native.argtypes = [ctypes.c_float] * nargs
    native.restype = ctypes.c_float
    return native


def strict_binding() -> str | None:
    from .numerics import is_strict
    if not is_strict():
        return None
    library = _strict_library()
    return "gnu243-geometry:" + hashlib.sha256(Path(library._name).read_bytes()).hexdigest()


def _intrinsic(numpy_name: str, native_name: str, nargs: int):
    fallback = getattr(np, numpy_name)
    native = getattr(_LIBM, native_name) if _LIBM is not None else None
    if native is not None:
        native.argtypes = [ctypes.c_float] * nargs
        native.restype = ctypes.c_float

    def evaluate(*values: Any):
        arrays = np.broadcast_arrays(*[np.asarray(v) for v in values])
        if any(v.dtype != np.float32 for v in arrays):
            return fallback(*values)
        from .numerics import is_strict
        active = _strict_intrinsic(native_name, nargs) if is_strict() else native
        if active is None:
            return fallback(*values)
        if arrays[0].ndim == 0:
            return np.float32(active(*[float(v) for v in arrays]))
        flat = [v.reshape(-1) for v in arrays]
        if flat[0].size >= 4096:
            from .native_libm_batch import intrinsic_batch
            result = intrinsic_batch(active, *flat)
            if result is not None:
                return result.reshape(arrays[0].shape)
        result = np.fromiter(
            (active(*[float(v[i]) for v in flat]) for i in range(flat[0].size)),
            dtype=np.float32,
            count=flat[0].size,
        )
        return result.reshape(arrays[0].shape)

    return evaluate


sin = _intrinsic("sin", "sinf", 1)
cos = _intrinsic("cos", "cosf", 1)
asin = _intrinsic("arcsin", "asinf", 1)
atan2 = _intrinsic("arctan2", "atan2f", 2)

_GNU_BINDINGS = (sin, cos, asin, atan2)


def unmodified_gnu_float32() -> bool:
    """The parallel bridge may only replace the exact GNU float32 authority."""
    from .numerics import is_strict
    return not is_strict() and _LIBM is not None and all(a is b for a, b in
        zip((sin, cos, asin, atan2), _GNU_BINDINGS))


def cache_binding(value: Any) -> str | None:
    """Identify only this module's unmodified GNU float library functions."""
    if _LIBM is None:
        return None
    name = getattr(value, "__name__", None)
    arity = {"sinf": 1, "cosf": 1, "asinf": 1, "atan2f": 2}.get(name)
    if arity is None or getattr(_LIBM, name) is not value:
        return None
    if value.argtypes != [ctypes.c_float] * arity or value.restype is not ctypes.c_float:
        return None
    # The producer digest also binds the interpreter and C library version.
    return f"gnu-libm:{_LIBM._name}:{name}:float32:{arity}"
