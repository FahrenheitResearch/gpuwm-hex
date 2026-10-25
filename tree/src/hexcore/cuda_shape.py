"""Dimension constants for selected hot kernels, with the original launch ABI.

Only const-int parameters named by the caller are specialized. Floating point
expressions, mesh predicates and rounded-grid tail checks remain unchanged.
Source and option/SM hashes continue through KernelCache and the existing pack.
A generic request (no dimensions) uses the original source and module identity.
"""
from __future__ import annotations

import hashlib
import operator
import re


def specialize(source: str, name: str, dimensions: tuple[tuple[str, int], ...]) -> tuple[str, str]:
    """Return source and identity suffix; fail closed on a changed signature.

    The original argument positions/types remain, so ordinary cuLaunchKernel
    argument buffers and the native step recorder need no alternate layout.
    """
    if not dimensions:
        return source, ""
    match = re.search(r'extern\s+"C"\s+__global__\s+void\s+' + re.escape(name)
                      + r'\s*\(([^()]*)\)\s*\{', source)
    if match is None:
        raise ValueError(f"shape kernel signature not found: {name}")
    signature = match.group(1)
    declarations = []
    seen = set()
    for dimension, raw in dimensions:
        value = operator.index(raw)
        if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', dimension) or dimension in seen:
            raise ValueError("shape dimensions must be distinct C identifiers")
        if isinstance(raw, bool) or not 0 < value <= 2147483647:
            raise ValueError(f"shape dimension {dimension} must be a positive int32")
        seen.add(dimension)
        pattern = r'\bconst\s+int\s+' + re.escape(dimension) + r'\b(?=\s*(?:,|$))'
        signature, count = re.subn(pattern, f'const int hex_shape_arg_{dimension}', signature)
        if count != 1:
            raise ValueError(f"shape dimension is not one const int parameter: {name}.{dimension}")
        declarations.append(f'    constexpr int {dimension} = {value};')
    replacement = (source[match.start():match.start(1)] + signature
                   + source[match.end(1):match.end()] + '\n' + '\n'.join(declarations))
    result = source[:match.start()] + replacement + source[match.end():]
    suffix = '.shape_' + hashlib.sha256(result.encode()).hexdigest()[:20]
    return result, suffix


def resolve(cache, name, source, module_key, dimensions=()):
    """Resolve once at the caller's existing shape/instance cache miss."""
    specialized, suffix = specialize(source, name, tuple(dimensions))
    kernel = cache.raw_kernel(name, specialized, module_key=module_key + suffix)
    if suffix:
        # Same-host compile checks compare shape modules by this origin: each
        # rank's dimensions are its own, the generic source must be shared.
        recorder = getattr(cache, "record_shape_origin", None)
        if recorder is not None:
            recorder(module_key + suffix, source, tuple(dimensions))
    return kernel
