"""Check the published IO getter against an independent direct Fortran export.

MPAS source: src/core_atmosphere/physics/mpas_atmphys_init.F:83-481.
MPAS source: src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90:18-938.
Rust decodes original inputs; native initializes the real framework and schemes.
This is complete interface evidence, never a whole-forecast identity grade.
"""
import argparse
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import socket

from hexcore.mpasphys.driver_v842 import resolve_namelist
from hexcore.mpasphys.forecast_v842 import read_streams, bind_stream_inputs
from hexcore.mpasphys.native_backend import NativePhysics
from hexcore.mpasphys.pool_v842 import decode_registry_pools
from hexcore.mpasphys.runtime_v842 import _configure_streams
from hexcore.mpasphys.session_v842 import NativePhysicsSession, native_working_directory


def replay(*, library, control, source, reader, inputs, namelist, streams,
           stream_lists, tables, stop, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    text = namelist.read_text()
    options = resolve_namelist(text)
    layout = decode_registry_pools(reader, source, options, text, inputs, output / 'payload')
    ct.CDLL(str(library), mode=ct.RTLD_GLOBAL)
    independent = ct.CDLL(str(control)).native_noahmp_io_copy
    independent.argtypes = [ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                            ct.POINTER(ct.c_int), ct.c_void_p, ct.POINTER(ct.c_int)]
    independent.restype = None
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(schema='mpas-initialized-noahmp-io-v1', complete=False,
                  forecast_proof=False, native_service_proposal_injected=False,
                  process_count=1, host=socket.gethostname(),
                  library_sha256=digest(library), control_sha256=digest(control),
                  abi_sha256=digest(library.with_name('abi.json')),
                  namelist_sha256=digest(namelist), fields=[])
    with NativePhysics(library, test_only=True) as native:
        session = NativePhysicsSession(native, layout, output / 'payload', options,
            dt=layout.configs['config_dt'], start=layout.configs['config_start_time'],
            stop=stop, stream_intervals={},
            post_transport=lambda call: None)
        bound = bind_stream_inputs(read_streams(streams, layout, stream_lists),
                                   {'input': inputs[-1]})
        result['missing_stream_fields'] = _configure_streams(native, bound, output, layout)
        with native_working_directory(tables):
            session.initialize()
        result['native_framework_initialized'] = session._initialized
        for name, field in native.abi['noahmp_io'].items():
            dims, lower = (ct.c_int * 5)(), (ct.c_int * 5)()
            width, status = ct.c_int(), ct.c_int()
            independent(field['index'], dims, lower, ct.byref(width), None, ct.byref(status))
            if status.value not in (0, 1):
                raise ValueError(f'independent IO metadata failed: {name}')
            actual = native.get_noahmp_io(name)
            row = dict(name=name, kind=field['kind'], rank=field['rank'],
                       present=status.value == 0)
            if status.value:
                if actual is not None:
                    raise ValueError(f'absent native IO was synthesized: {name}')
            else:
                value, bounds = actual
                shape = tuple(dims[:field['rank']])
                expected = ct.create_string_buffer(width.value * math.prod(shape))
                independent(field['index'], dims, lower, ct.byref(width), expected, ct.byref(status))
                if (status.value or value.shape != shape or bounds != tuple(lower[:field['rank']])
                        or value.dtype.itemsize != width.value
                        or value.tobytes(order='F') != expected.raw):
                    raise ValueError(f'published native IO differs: {name}')
                second, second_bounds = native.get_noahmp_io(name)
                if second_bounds != bounds or second.tobytes(order='F') != expected.raw:
                    raise ValueError(f'read-only native IO changed: {name}')
                row.update(shape=shape, lower=bounds, width=width.value,
                           bytes=len(expected.raw), sha256=hashlib.sha256(expected.raw).hexdigest(),
                           different_bytes=0)
            result['fields'].append(row)
    result.update(complete=True,
                  allocated_arrays=sum(r['present'] and r['rank'] > 0 for r in result['fields']),
                  absent_arrays=sum(not r['present'] for r in result['fields']),
                  scalars=sum(r['rank'] == 0 for r in result['fields']),
                  bytes=sum(r.get('bytes', 0) for r in result['fields']))
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('fields', 'missing_stream_fields')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('library', 'control', 'source', 'reader', 'namelist', 'streams',
                 'stream-lists', 'tables', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stop', required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    replay(**vars(parser.parse_args()))
