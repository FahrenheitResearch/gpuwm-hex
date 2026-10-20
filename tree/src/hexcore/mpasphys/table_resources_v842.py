"""Immutable slot table transport; module_mp_thompson.F:862-976.

Rust decodes encoded records. Python validates provenance, exposes ABI buffers
and copies their words into device arrays; no scheme arithmetic is performed.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import math
import os
from pathlib import Path

import numpy as np

THOMPSON_AUXILIARY_SHA256 = '8d80a02edfddd9ce07a659c2310ca37bbeae6062a04abe47a4ddb664676e79c5'


def certified_path(row):
    """Authenticate native initialized input bytes; module_mp_thompson.F:862-976."""
    path = Path(row['path']).resolve()
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream,'sha256').hexdigest()
    if actual != row['sha256']:
        raise ValueError(f'physics resource SHA256 differs: {path}')
    return path


class RustLookupRecords:
    """Owned Rust-decoded words for native records; module_mp_thompson.F:862-976."""

    def __init__(self, decoder):
        """Explicit pinned Rust ABI; module_mp_thompson.F:862-976."""
        self.lib = ct.CDLL(str(certified_path(decoder)))
        self.lib.thompson_lookup_open.argtypes = [ct.c_char_p,ct.POINTER(ct.c_uint64),
                ct.POINTER(ct.c_uint32),ct.c_size_t,ct.c_bool,ct.c_bool]
        self.lib.thompson_lookup_open.restype = ct.c_void_p
        self.lib.thompson_lookup_data.argtypes = [ct.c_void_p,ct.c_size_t]
        self.lib.thompson_lookup_data.restype = ct.c_void_p
        self.lib.thompson_lookup_close.argtypes = [ct.c_void_p]
        self.lib.thompson_lookup_close.restype = None
        self.lib.thompson_lookup_error.argtypes = []
        self.lib.thompson_lookup_error.restype = ct.c_char_p

    def arrays(self, path, names, shapes, *, records, little):
        """Copy Rust buffers without interpreting encoded files; module_mp_thompson.F:862-976."""
        counts = [math.prod(shapes[name]) for name in names]
        widths = [4 if name == 'tnccn_act' else 8 for name in names]
        handle = self.lib.thompson_lookup_open(os.fsencode(path),
                (ct.c_uint64*len(names))(*counts),(ct.c_uint32*len(names))(*widths),
                len(names),records,little)
        if not handle:
            raise ValueError(self.lib.thompson_lookup_error().decode())
        try:
            result = {}
            for index,(name,count,width) in enumerate(zip(names,counts,widths)):
                pointer = self.lib.thompson_lookup_data(handle,index)
                scalar = ct.c_float if width == 4 else ct.c_double
                host = np.ctypeslib.as_array(ct.cast(pointer,ct.POINTER(scalar)),shape=(count,))
                result[name] = host.copy().reshape(shapes[name],order='F')
            return result
        finally:
            self.lib.thompson_lookup_close(handle)


def load_thompson_tables(resources, device, scheme):
    """Complete strict packet and native record order; module_mp_thompson.F:862-976,4363-4365."""
    decoder = RustLookupRecords(resources['rust_decoder'])
    filenames = {
        'MP_THOMPSON_QRacrQG_DATA.DBL':scheme.TABLE_ARGUMENTS[:6],
        'MP_THOMPSON_QRacrQS_DATA.DBL':scheme.TABLE_ARGUMENTS[6:18],
        'MP_THOMPSON_freezeH2O_DATA.DBL':scheme.TABLE_ARGUMENTS[18:24],
        'MP_THOMPSON_QIautQS_DATA.DBL':scheme.TABLE_ARGUMENTS[24:27],
    }
    host = {}
    certificate = {}
    for filename,names in filenames.items():
        expected = scheme.TABLE_FILE_SHA256[filename]
        path = certified_path(dict(path=Path(resources['table_directory'])/filename,sha256=expected))
        host.update(decoder.arrays(path,names,scheme.TABLE_SHAPES,records=True,little=False))
        certificate[filename] = expected
    auxiliary = certified_path(dict(path=resources['auxiliary'],sha256=THOMPSON_AUXILIARY_SHA256))
    auxiliary_names = ('t_efrw','t_efsw','tnr_rev','tpc_wev','tnc_wev','tnccn_act')
    host.update(decoder.arrays(auxiliary,auxiliary_names,scheme.TABLE_SHAPES,records=False,little=True))
    if set(host) != set(scheme.TABLE_SHAPES):
        raise ValueError('certified Thompson table packet does not cover the complete declared table set')
    tables = {name:device.asarray(array,order='F') for name,array in host.items()}
    tables['_file_sha256'] = certificate
    return tables


def load_thompson_resources(resources, device):
    """Bind tables and explicitly certified shared mathematics; module_mp_thompson.F:1001-1387."""
    from . import thompson_v842
    math_source = certified_path(resources['device_math_source']).read_text()
    return dict(tables=load_thompson_tables(resources,device,thompson_v842),
                device_math_source=math_source)
