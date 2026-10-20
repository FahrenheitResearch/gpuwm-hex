#!/usr/bin/env python3
"""f1 executable adapter: Rust oracle I/O, CUDA/strict-host YSU, no tolerances.

MPAS v8.4.2 physics_wrf/module_bl_ysu.F:16-475. Entry storage supplies
untouched levels/halos; expected post-call outputs never supply input words.
"""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
from hexcore.mpasphys import ysu_v842 as ysu


class Record:
    def __init__(self, library, path):
        self.library = library
        self.handle = library.ysu_load(os.fsencode(path))
        if not self.handle:
            raise ValueError(f"Rust rejected oracle record {path}")
        self.header = list(library.ysu_header(self.handle)[:21])
        self.count = library.ysu_count(self.handle)
        # This is an ABI view of Rust-decoded storage, not a file decoder.
        self.words = np.ctypeslib.as_array(library.ysu_words(self.handle), shape=(self.count,))

    def array(self):
        h = self.header
        if h[0] not in (1, 3, 4) or h[1] != 4:
            raise ValueError("numeric carriers must be real32/int32/logical32")
        dtype = np.float32 if h[0] == 1 else np.int32
        return self.words.view(dtype).reshape(tuple(reversed(h[3:3+h[2]])))

    def scalar(self):
        if self.header[2] != 0:
            raise ValueError("scalar argument has array extent")
        return self.array().item()

    def save(self, path, words):
        words = np.ascontiguousarray(words).reshape(-1).view(np.uint32)
        rc = self.library.ysu_save(self.handle, os.fsencode(path),
                                  words.ctypes.data_as(ct.POINTER(ct.c_uint32)), words.size)
        if rc:
            raise ValueError(f"Rust failed to serialize oracle record {path}: {rc}")

    def close(self):
        if self.handle:
            self.library.ysu_free(self.handle)
            self.handle = None


def io_library(path):
    lib = ct.CDLL(str(path.resolve()))
    lib.ysu_load.argtypes = [ct.c_char_p]
    lib.ysu_load.restype = ct.c_void_p
    lib.ysu_header.argtypes = [ct.c_void_p]
    lib.ysu_header.restype = ct.POINTER(ct.c_int32)
    lib.ysu_count.argtypes = [ct.c_void_p]
    lib.ysu_count.restype = ct.c_size_t
    lib.ysu_words.argtypes = [ct.c_void_p]
    lib.ysu_words.restype = ct.POINTER(ct.c_uint32)
    lib.ysu_save.argtypes = [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_uint32), ct.c_size_t]
    lib.ysu_save.restype = ct.c_int
    lib.ysu_free.argtypes = [ct.c_void_p]
    lib.ysu_success_message.argtypes = [ct.c_void_p, ct.c_char_p]
    lib.ysu_success_message.restype = ct.c_int
    return lib


def replay(prefix, output, backend, library, oracle_dir):
    records = {}
    try:
        for path in prefix.parent.glob(prefix.name+'.pre.*.bin'):
            name = path.name[len(prefix.name+'.pre.'):-4]
            records[name] = Record(library, path)
        scalar = lambda name: records[name].scalar()
        bounds = {name: int(scalar(name)) for name in
                  ('its', 'ite', 'ims', 'ime', 'jts', 'jte', 'jms', 'jme', 'kts', 'kte', 'kms', 'kme')}
        if bounds['kts'] != 1 or bounds['kms'] != 1:
            raise ValueError('MPAS driver vertical lower bound must be one')
        nz = bounds['kte']
        i = slice(bounds['its']-bounds['ims'], bounds['ite']-bounds['ims']+1)
        j = slice(bounds['jts']-bounds['jms'], bounds['jte']-bounds['jms']+1)
        ncol = (bounds['ite']-bounds['its']+1)*(bounds['jte']-bounds['jts']+1)
        if backend == 'cuda':
            import cupy as xp
        else:
            xp = np

        def selected(name, interfaces=False):
            a = xp.asarray(records[name].array())
            if a.ndim == 3:
                # Native storage (i,k,j) is a C view (j,k,i). Rearrange on device.
                return xp.ascontiguousarray(a[j, :nz+int(interfaces), i].transpose(1, 0, 2).reshape(nz+int(interfaces), ncol))
            if a.ndim == 2:
                return xp.ascontiguousarray(a[j, i].reshape(ncol))
            raise ValueError(f'unsupported carrier rank for {name}')

        state = {}
        for name in ysu._WRAPPER_FIELDS:
            if name in records:
                state[ysu._WRAPPER_FIELDS[name]] = selected(name, interfaces=(name == 'p3di'))
        for name in ysu._REQUIRED_SURFACE + ('rthraten', 'ctopo', 'ctopo2'):
            if name in records:
                state[name] = selected(name)
        for name, raw in ysu._WRAPPER_OUTPUTS.items():
            if name in records:
                state[raw] = selected(name)
        values = {name: scalar(name) for name in ysu.FLOAT_ARGUMENTS}
        flags = dict(f_qc=bool(scalar('flag_qc')), f_qi=bool(scalar('flag_qi')),
                     flag_bep=bool(scalar('flag_bep')), ysu_topdown_pblmix=int(scalar('ysu_topdown_pblmix')))
        if backend == 'cuda':
            result = ysu.ysu(state, values, dt=values['dt'], flag_qc=flags['f_qc'],
                             flag_qi=flags['f_qi'], flag_bep=flags['flag_bep'],
                             ysu_topdown_pblmix=flags['ysu_topdown_pblmix'],
                             device_math_header=ysu.shared_device_math_header())
        else:
            from oracle import Oracle
            oracle = Oracle(oracle_dir)
            result = {name: np.empty((nz, ncol), np.float32) if raw in ysu._OUTPUT_LEVEL
                      else np.empty(ncol, np.int32 if name == 'kpbl2d' else np.float32)
                      for name, raw in ysu._WRAPPER_OUTPUTS.items()}
            result['rthblten'] = np.empty((nz, ncol), np.float32)
            for col in range(ncol):
                arrays = {name: None for name in ysu.ARRAY_ARGUMENTS}
                for name, value in state.items():
                    arrays[name] = np.ascontiguousarray(value[:, col] if value.ndim == 2 else value[col:col+1])
                for name in ysu._OUTPUT_LEVEL + ysu._OUTPUT_SURFACE:
                    if arrays[name] is None:
                        arrays[name] = np.zeros(nz if name in ysu._OUTPUT_LEVEL else 1,
                                                np.int32 if name == 'kpbl1d' else np.float32)
                arrays['qmix'] = arrays['qmixtnp'] = np.empty(0, np.float32)
                native_flags = dict(its=1, ite=1, kte=nz, kme=nz+1, nmix=0,
                                    **{**flags, 'ysu_topdown_pblmix': flags['ysu_topdown_pblmix'] if backend == 'native' else int(flags['ysu_topdown_pblmix'] == 1)})
                out, status = oracle.run(arrays, native_flags, wrapper=(backend == 'native'), values=values)
                if status:
                    raise ValueError(f'real column {col}: {f"unexpected scheme status {status}"}')
                for name, raw in ysu._WRAPPER_OUTPUTS.items():
                    result[name][..., col] = out[raw]
                result['rthblten'][:, col] = out['ttnp'] if backend == 'native' else np.divide(out['ttnp'], arrays['pi2d'], dtype=np.float32)
        output.parent.mkdir(parents=True, exist_ok=True)
        for name, value in result.items():
            record = records[name]
            full = xp.asarray(record.array()).copy()
            if full.ndim == 3:
                full[j, :nz, i] = value.reshape(nz, bounds['jte']-bounds['jts']+1, bounds['ite']-bounds['its']+1).transpose(1, 0, 2)
            else:
                full[j, i] = value.reshape(bounds['jte']-bounds['jts']+1, bounds['ite']-bounds['its']+1)
            host = xp.asnumpy(full) if backend == 'cuda' else full
            record.save(Path(str(output)+'.post.'+name+'.bin'), host)
        # physics_mmm/bl_ysu.F90:1417-1418: exact successful control outputs.
        records['errflg'].save(Path(str(output)+'.post.errflg.bin'), np.zeros(records['errflg'].count, np.uint32))
        if library.ysu_success_message(records['errmsg'].handle, os.fsencode(str(output)+'.post.errmsg.bin')):
            raise ValueError('Rust failed to serialize native success message')
        print(f'{prefix.name}: {backend}, {ncol} columns, {nz} levels', flush=True)
        if backend == 'cuda':
            from oracle import cuda_provenance
            print(json.dumps(dict(cuda_sha256=ysu.CUDA_SHA256, capture_process_count=1,
                                  replay_process_count=1, **cuda_provenance())), flush=True)
    finally:
        for record in records.values():
            record.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--backend', choices=('cuda', 'host', 'native'), default=os.environ.get('YSU_REAL_BACKEND', 'cuda'))
    parser.add_argument('--io', type=Path, default=Path(os.environ.get('YSU_ORACLE_IO', 'n1out/oracle_io.so')))
    parser.add_argument('--oracle', type=Path, default=Path(os.environ.get('YSU_ORACLE_DIR', 'n1out/proof-bin')))
    args = parser.parse_args()
    replay(args.prefix, args.output, args.backend, io_library(args.io), args.oracle)
