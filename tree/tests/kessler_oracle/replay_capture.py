#!/usr/bin/env python3
"""f1 typed-record replay adapter: Rust I/O, CUDA or strict native arithmetic.

MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_kessler.F:9-243.
MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:371-385.
"""
from pathlib import Path
import ctypes
import json
import os
import subprocess
import sys
import numpy as np
from hexcore.mpasphys.kessler_v842 import launch_kessler, shared_math_header


def main():
    source, target = map(Path, sys.argv[1:])
    helper = os.environ['KESSLER_RECORDS']
    def path(name):
        return str(source)+'.pre.'+name+'.bin'
    def read(name, dtype):
        return np.frombuffer(subprocess.check_output([helper,'payload',path(name),'-']),dtype=dtype)
    def integer(name):
        return int(read(name,'<i4')[0])
    def scalar(name):
        return read(name,'<f4')[0]
    # Native driver uses one j row and one-based contiguous active extents.
    # Retain all uncalled trailing levels from entry storage when serializing.
    nc, nz = integer('ite'), integer('kte')
    header=json.loads(subprocess.check_output([helper,'header',path('t'),'-']))
    if (header['shape'][0]!=nc or header['shape'][2]!=1
            or any(integer(k)!=1 for k in ('its','kts','jts','jte','ims','kms','jms'))):
        raise ValueError('replay adapter expects native single-row driver storage with contiguous active prefix')
    names=('t','qv','qc','qr','rho','pii','z','dz8w','rainnc','rainncv')
    fields={k:read(k,'<f4')[:nc*nz if k not in ('rainnc','rainncv') else nc].reshape((nz,nc) if k not in ('rainnc','rainncv') else (nc,)).copy() for k in names}
    scalar_names=('xlv','cp','ep2','svp1','svp2','svp3','svpt0','rhowater')
    values={k:scalar(k) for k in scalar_names}
    if os.environ['KESSLER_REPLAY_BACKEND']=='hex-cuda':
        import cupy as cp
        device={k:cp.asarray(v) for k,v in fields.items()}
        work=launch_kessler(**device,dt_in=scalar('dt_in'),**values,math_header=shared_math_header())
        outputs={k:cp.asnumpy(device[k]) for k in ('t','qv','qc','qr','rainnc','rainncv')}
    else:
        lib=ctypes.CDLL(os.environ['KESSLER_ORACLE_LIBRARY'])
        ptr=ctypes.POINTER(ctypes.c_float)
        function=getattr(lib,os.environ.get('KESSLER_ORACLE_SYMBOL','oracle'))
        function.argtypes=[ptr]*10+[ctypes.c_float,ptr,ctypes.c_int,ctypes.c_int]
        function.restype=None
        constants=np.asarray([values[k] for k in scalar_names],dtype='f4')
        function(*(fields[k].ctypes.data_as(ptr) for k in names),scalar('dt_in'),constants.ctypes.data_as(ptr),nz,nc)
        outputs={k:fields[k] for k in ('t','qv','qc','qr','rainnc','rainncv')}
    for k,v in outputs.items():
        subprocess.run([helper,'wrap-prefix',path(k),str(target)+'.post.'+k+'.bin'],input=v.tobytes(),check=True)

if __name__=='__main__':
    main()
