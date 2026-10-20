"""Compile-time dimensions for hexcore's native MPAS shortwave source.

Only integer addressing and loop bounds are specialized. Runtime argument order,
field arithmetic, boundary predicates and padded-grid tail guards are retained.
The caller owns compilation, numeric options and the per-device module cache.
"""
from __future__ import annotations
import re
from functools import lru_cache

FUNCTIONS = {"rrtmg_sw": ("rsw_spcvmc_gpt_b",)}
_KEYS = ("GPUWM_SHAPE_NZ", "GPUWM_SHAPE_NY", "GPUWM_SHAPE_NX")


def shape_defines(name, function, shape):
    if not shape or function not in FUNCTIONS.get(name, ()):
        return ()
    if len(shape) != 3 or any(type(x) is not int or not 0 < x <= 2147483647 for x in shape):
        raise ValueError("kernel shape must be three positive int32 dimensions (nz, ny, nx)")
    if shape[1] * shape[2] > 2147483647:
        raise ValueError("kernel column count exceeds int32")
    return tuple(zip(_KEYS, shape))


@lru_cache(maxsize=256)
def specialize_source(name, source, defines):
    values = dict(defines)
    if not any(k in values for k in _KEYS):
        return source
    if not all(k in values for k in _KEYS):
        raise ValueError("incomplete CUDA shape constants")
    nz, ny, nx = (values[k] for k in _KEYS)
    dims = {"nz": nz, "ny": ny, "nx": nx, "ncol": ny * nx, "n": ny * nx,
            "nc": ny * nx, "nl": nz, "nlayers": nz}
    # Mask comments without changing offsets. Conditional signatures are kept
    # verbatim, including both strict and ordinary parameter branches.
    masked = re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: " " * len(m[0]), source, flags=re.S)
    edits = []
    for function in FUNCTIONS.get(name, ()):
        match = re.search(r"\bvoid\s+" + function + r"\s*\(", masked)
        if match is None:
            raise ValueError(f"shape registry symbol missing: {name}:{function}")
        begin = match.end()
        end = masked.index("{", begin)
        header = source[begin:end]
        present = set(re.findall(r"\bint\s+(nz|ny|nx|ncol|n|nc|nl|nlayers)\b", masked[begin:end]))
        if not present:
            raise ValueError(f"shape registry dimensions missing: {name}:{function}")
        for dim in sorted(present):
            header = re.sub(r"\bint\s+" + dim + r"\b", "int gpuwm_runtime_" + dim, header)
        constants = "\n" + "".join(f"    constexpr int {dim} = {dims[dim]};\n" for dim in sorted(present))
        edits.append((begin, end + 1, header + "{" + constants))
    for begin, end, replacement in sorted(edits, reverse=True):
        source = source[:begin] + replacement + source[end:]
    return source
