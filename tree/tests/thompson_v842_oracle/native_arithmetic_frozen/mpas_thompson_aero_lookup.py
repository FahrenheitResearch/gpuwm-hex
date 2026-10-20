"""Test transport of native lookup bytes through p3's shared Rust decoder.

Rust validates record markers, endian order and word extents. Python supplies
metadata and owns copied native buffers for ctypes/CUDA launches only.
"""
import ctypes as C
import os

import numpy as np


def library(path):
    lib = C.CDLL(str(path.resolve()))
    lib.thompson_lookup_open.argtypes = [C.c_char_p, C.POINTER(C.c_uint64),
        C.POINTER(C.c_uint32), C.c_size_t, C.c_bool, C.c_bool]
    lib.thompson_lookup_open.restype = C.c_void_p
    lib.thompson_lookup_data.argtypes = [C.c_void_p, C.c_size_t]
    lib.thompson_lookup_data.restype = C.c_void_p
    lib.thompson_lookup_close.argtypes = [C.c_void_p]
    lib.thompson_lookup_error.restype = C.c_char_p
    return lib


def arrays(lib, path, core, names, *, records, little):
    shapes = core.TABLE_SHAPES
    counts = [int(np.prod(shapes[n])) for n in names]
    widths = [4 if n == 'tnccn_act' else 8 for n in names]
    handle = lib.thompson_lookup_open(os.fsencode(path),
        (C.c_uint64*len(names))(*counts), (C.c_uint32*len(names))(*widths),
        len(names), records, little)
    if not handle:
        raise ValueError(lib.thompson_lookup_error().decode())
    try:
        result = {}
        for i, (name, count, width) in enumerate(zip(names, counts, widths)):
            scalar = C.c_float if width == 4 else C.c_double
            pointer = lib.thompson_lookup_data(handle, i)
            view = np.ctypeslib.as_array(C.cast(pointer, C.POINTER(scalar)), shape=(count,))
            result[name] = view.copy().reshape(shapes[name], order='F')
        return result
    finally:
        lib.thompson_lookup_close(handle)
