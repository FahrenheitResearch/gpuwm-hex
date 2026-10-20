#!/usr/bin/env python3
"""Strict fork wrapper on captured PRE records; every native POST is observed.

Fork 1d6e370e src/core_atmosphere/physics/physics_wrf/module_sf_mynn.F:223-388
and physics_mmm/sf_mynn.F90:81-82 (four SAVE stability tables).
Opaque record decoding/encoding is the existing Rust transport. Expected POST
records are never opened by this executable.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from hexcore.mpasphys import sf_mynn_v842 as scheme


def main():
    support = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('shared_record_transport',
        support.parent / 'mpas_sf_mynn/replay_capture.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    transport = adapter.Transport(Path(os.environ['MYNN_ORACLE_RECORD']))
    source, dest = map(Path, sys.argv[1:3])
    def pre(name):
        return Path(str(source) + '.pre.' + name + '.bin')
    def post(name):
        return Path(str(dest) + '.post.' + name + '.bin')
    def read(name):
        return transport.read(pre(name))[0]
    def scalar(name):
        return read(name).item()
    n = scalar('ime') - scalar('ims') + 1
    levels = scalar('kme') - scalar('kms') + 1
    if (scalar('its'), scalar('ite'), scalar('jts'), scalar('jte')) != (scalar('ims'), scalar('ime'), 1, 1):
        raise ValueError('Native full wrapper replay requires the captured full horizontal extent')
    fields = np.zeros((len(scheme.FIELDS), n), np.float32)
    for index, name in enumerate(scheme.FIELDS):
        if pre(name).exists():
            fields[index] = read(name).reshape(n)
    profiles = np.stack([read(name).reshape((n, levels), order='F').T for name in scheme._PROFILE])
    constants = np.asarray([scalar(name) for name in scheme.CONSTANTS], np.float32)
    tables = ('psim_unstab', 'psih_unstab', 'psim_stab', 'psih_stab')
    state = np.stack([read('state.sf_mynn.' + name).reshape(-1) for name in tables])
    if state.shape != (4, 1001):
        raise ValueError('Captured native stability table bounds changed')
    library = Path(os.environ['MYNN_NATIVE_LIBRARY'])
    authority = json.loads(library.with_name('build.json').read_text())
    if hashlib.sha256(library.read_bytes()).hexdigest() != authority['library_sha256']:
        raise ValueError('Observed native ELF library changed')
    lib = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    def ptr(array):
        return array.ctypes.data_as(pointer)
    lib.native_init()
    # PRIVATE SAVE symbols are local ELF BSS, so use their authenticated offsets
    # from the exported bridge anchor. The native source and storage stay intact.
    symbols = authority['elf_symbols']
    anchor = ctypes.cast(lib.native_init, ctypes.c_void_p).value - symbols['native_init']['offset']
    native_tables = []
    for name in tables:
        symbol = symbols['__sf_mynn_MOD_' + name]
        if symbol['size'] != 4004:
            raise ValueError('Native stability array extent changed')
        native_tables.append(np.ctypeslib.as_array((ctypes.c_float * 1001).from_address(anchor + symbol['offset'])))
    for index, values in enumerate(native_tables):
        values[:] = state[index]
    errmsg = np.empty(512, np.uint8)
    if read('errmsg').size != errmsg.size:
        raise ValueError('Captured native StrKIND differs from the pinned 512-byte error buffer')
    errflg = ctypes.c_int()
    lib.native_wrapper_observed.argtypes = ([pointer] * 3 + [ctypes.c_int] * 7 +
        [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int)])
    flags = [int(scalar(name)) for name in ('itimestep', 'isfflx', 'spp_pbl', 'isftcflx', 'iz0tlnd')]
    lib.native_wrapper_observed(ptr(fields), ptr(constants), ptr(profiles), *flags, n, levels,
        errmsg.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.byref(errflg))
    for index, values in enumerate(native_tables):
        state[index] = values
    for name in scheme.OUTPUTS:
        if pre(name).exists():
            _, header = transport.read(pre(name))
            transport.write(pre(name), post(name), fields[scheme.FIELDS.index(name)].reshape(
                tuple(header[3:3 + header[2]]), order='F'))
    for template in source.parent.glob(source.name + '.pre.*.bin'):
        name = template.name[len(source.name + '.pre.'):-4]
        if (name.endswith('.present') and name[:-8] in scheme.OUTPUTS) or name.startswith(('state.', 'local.')):
            values = read(name)
            if name.startswith('state.sf_mynn.'):
                values = state[tables.index(name.removeprefix('state.sf_mynn.'))].reshape(values.shape)
            transport.write(template, post(name), values)
    transport.write(pre('errmsg'), post('errmsg'), errmsg)
    transport.write(pre('errflg'), post('errflg'), np.asarray(errflg.value, np.int32))


if __name__ == '__main__':
    main()
