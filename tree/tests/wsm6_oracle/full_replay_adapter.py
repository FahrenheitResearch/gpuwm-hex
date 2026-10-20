#!/usr/bin/env python3
"""Complete f1 WSM6 replay executable: INPUT_PREFIX OUTPUT_PREFIX.

Rust reads only authenticated PRE records and writes all ORCL0001 results.
Python binds raw buffers to production CUDA, or the explicit host witness.
Expected POST words are never inputs to this executable.
"""
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main(input_prefix, output_prefix):
    root, prefix = input_prefix.parent, input_prefix.name
    transport = Path(os.environ['WSM6_REPLAY_TRANSPORT']).resolve()
    fixture = json.loads(subprocess.check_output(
        [str(transport), 'replay-input', str(root), prefix], text=True))
    state_schema = json.loads(subprocess.check_output(
        [str(transport), 'state', str(root), prefix], text=True))
    nz = fixture['nz']
    bind = lambda words, shape: np.asarray(words, dtype=np.uint32).view(np.float32).reshape(shape)
    constants = bind(fixture['constants'], (26,))
    dt = bind([fixture['dt']], (1,))[0]
    backend = os.environ.get('WSM6_REPLAY_BACKEND', 'hex-cuda')
    from hexcore.mpasphys.wsm6_v842 import _state_layout
    nc = len(fixture['rows'])
    packed = fixture['packed']
    v = bind(packed['volume'], (10,nz,nc))
    s = bind(packed['surface'], (7,nc))
    r = bind(packed['radii'], (3,nz,nc))
    pi = bind(packed['pi'], (nz,nc))
    dbz = np.zeros((nz,nc), np.float32)
    prod = np.zeros((2,nz,nc), np.float32)
    states = {}
    if backend == 'hex-cuda':
        import cupy as cp
        from hexcore import rkind_libm
        from hexcore.mpasphys.wsm6_v842 import columns, initialized_state_bytes
        diagnostic = os.environ.get('WSM6_DIAGNOSTIC_SHARED_HEADER')
        if diagnostic:
            path = Path(diagnostic).resolve(strict=True)
            header = path.read_text()
            certificate = json.loads(path.with_suffix('.json').read_text())
            if certificate['acceptance'] or certificate['header_sha256'] != hashlib.sha256(header.encode()).hexdigest():
                raise ValueError('diagnostic shared-maths header certificate does not match')
        else:
            header = rkind_libm.wsm6_device_math_header()
        device_constants = cp.asarray(constants)
        snapshots = initialized_state_bytes(device_constants, hail_opt=0, device_math_header=header)
        for member in state_schema['records']:
            states[member] = cp.asnumpy(snapshots[member.split('.', 1)[1]]).tolist()
        buffers = [cp.asarray(a) for a in (v,s,r,dbz,prod)]
        messages = cp.empty((256,nc), cp.uint8)
        status = columns(*buffers, device_constants, dt=dt, hail_opt=0,
                         device_math_header=header, exner=cp.asarray(pi),
                         radii_on=fixture['radii_on'], radar_on=False,
                         production_on=False, messages=messages)
        if np.any(cp.asnumpy(status)):
            raise ValueError('port wrapper returned a nonzero status')
        v,s,r = [cp.asnumpy(a) for a in buffers[:3]]
        messages = cp.asnumpy(messages)
    elif backend == 'host-witness':
        library = ct.CDLL(str(Path(os.environ['WSM6_REPLAY_HOST_LIBRARY']).resolve()))
        fp = ct.POINTER(ct.c_float)
        run = library.wsm6_wrapper_message_host
        run.argtypes = [ct.c_int, ct.c_float, ct.c_int] + [fp]*7 + [ct.c_int, ct.POINTER(ct.c_int), ct.c_void_p]
        snapshot = library.wsm6_initialized_state_host
        snapshot.argtypes = [fp, ct.c_int, ct.c_int, ct.c_void_p]
        layout = {name:(index,size) for index,(name,kind,size) in enumerate(_state_layout())}
        for member, schema in state_schema['records'].items():
            index,size = layout[member.split('.', 1)[1]]
            assert size == len(schema['bytes']), (member,size,len(schema['bytes']))
            output = (ct.c_ubyte * size)()
            snapshot(constants.ctypes.data_as(fp),0,index,output)
            states[member] = list(output)
        messages = np.empty((256,nc), np.uint8)
        for i in range(nc):
            column = [np.ascontiguousarray(a[...,i:i+1]) for a in (v,s,r,dbz,prod,pi)]
            status = ct.c_int(-1)
            message = ct.create_string_buffer(256)
            run(nz,dt,0,constants.ctypes.data_as(fp),
                *[a.ctypes.data_as(fp) for a in column],int(fixture['radii_on']),
                ct.byref(status),message)
            assert status.value == 0
            for target,source in zip((v,s,r),column):
                target[...,i:i+1] = source
            messages[:,i] = np.frombuffer(message.raw,np.uint8)
    else:
        raise ValueError('select hex-cuda or explicit host-witness')
    # Character output is produced by the actual wrapper, independent of PRE/POST.
    message = messages[:,0].tobytes().decode('ascii').rstrip(' ')
    if np.any(messages != messages[:,0:1]):
        raise ValueError('per-column wrapper messages differ')
    payload = {'dump_root_sha256':fixture['dump_root_sha256'], 'rows':fixture['rows'],
               'packed':{name:a.view(np.uint32).ravel().tolist()
                         for name,a in zip(('volume','surface','radii'),(v,s,r))},
               'state':states, 'errmsg':message, 'errflg':0}
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Output-prefix contract requires every file directly in the results root.
    # The temporary payload stays in the owned results directory and is retired.
    with tempfile.NamedTemporaryFile(mode='w', dir=output_prefix.parent, suffix='.json') as f:
        json.dump(payload, f); f.flush()
        subprocess.run([str(transport),'replay-write',str(root),prefix,
                        str(output_prefix.parent),f.name],check=True)
    print(f'{backend}: {nc} columns; complete outputs produced without POST inputs')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: full_replay_adapter.py INPUT_PREFIX OUTPUT_PREFIX')
    main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
