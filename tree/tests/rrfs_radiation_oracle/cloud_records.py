"""Rust-backed pristine cloud-driver records; no Python codec."""
import ctypes as ct
import json
import os
import numpy as np

class Records:
    """Rust owns the typed codec, payload boundaries and exact comparator."""
    def __init__(self, library):
        self.lib = ct.CDLL(str(library.resolve()))
        byteptr = ct.POINTER(ct.c_uint8)
        self.lib.r9_record_open.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int32),
            ct.POINTER(byteptr), ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_char_p)]
        self.lib.r9_record_open.restype = ct.c_void_p
        self.lib.r9_record_close.argtypes = [ct.c_void_p]
        self.lib.r9_record_text.argtypes = [ct.c_void_p, ct.POINTER(ct.c_char_p)]
        self.lib.r9_record_post.argtypes = [ct.c_char_p, ct.c_char_p, byteptr, ct.c_size_t]
        self.lib.r9_record_compare.argtypes = [ct.c_char_p, ct.c_char_p]
        self.lib.r9_record_compare.restype = ct.c_void_p
        self.lib.r9_string_free.argtypes = [ct.c_void_p]

    def load(self, path):
        header = (ct.c_int32 * 21)()
        payload = ct.POINTER(ct.c_uint8)()
        length, name = ct.c_size_t(), ct.c_char_p()
        handle = self.lib.r9_record_open(os.fsencode(path), header, ct.byref(payload),
                                         ct.byref(length), ct.byref(name))
        if not handle:
            raise ValueError(f'Rust decode refused {path}')
        try:
            if header[0] == 5:
                text = ct.c_char_p()
                if self.lib.r9_record_text(handle, ct.byref(text)):
                    raise ValueError(f'Rust character decode refused {path}')
                value = text.value.decode()
            else:
                dtype = {1: 'f4', 2: 'f8', 3: 'i4', 4: 'i4', 6: 'i8', 7: 'c8', 8: 'c16'}[header[0]]
                value = np.ctypeslib.as_array(payload, shape=(length.value,)).copy().view(dtype)
                value = value.reshape(tuple(header[3:3+header[2]]), order='F')
            return value, tuple(header)
        finally:
            self.lib.r9_record_close(handle)

    def post(self, source, target, array=None):
        # An untouched record is passed directly through Rust. Replaced words
        # come solely from PRE and actual GPU results, never expected POST.
        words = None if array is None else np.ascontiguousarray(array.ravel(order='F')).view('u1')
        payload = None if words is None else words.ctypes.data_as(ct.POINTER(ct.c_uint8))
        if self.lib.r9_record_post(os.fsencode(source), os.fsencode(target), payload,
                                    0 if words is None else words.size):
            raise ValueError(f'Rust POST emission refused {target}')

    def compare(self, expected, actual):
        pointer = self.lib.r9_record_compare(os.fsencode(expected), os.fsencode(actual))
        if not pointer:
            raise ValueError(f'Rust comparison refused {expected}')
        try:
            return json.loads(ct.string_at(pointer))
        finally:
            self.lib.r9_string_free(pointer)

