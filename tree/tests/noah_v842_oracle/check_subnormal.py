"""Native EVAPO must preserve subnormal evaporation (MPAS Noah:1330-1465).

This uses the unchanged supplemental entry and production device arithmetic.
Nonzero native subnormal outputs make a reintroduced FTZ flag fail wordwise.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path

import numpy as np
from hexcore.mpasphys.noah_v842 import _CORE_CUDA, _compile_module, _device_math_source

p = argparse.ArgumentParser()
p.add_argument('--gpu', action='store_true')
p.add_argument('--tables-from', type=Path, default=Path('n1out/noah-synthetic.npz'))
a = p.parse_args()
base = Path(__file__).resolve().parent
layout = json.loads((base/'helper_layouts.json').read_text())['noah_evapo']
f = {name: k for k, name in enumerate(layout['float'])}
fixture = Path('n1out/subnormal-native.npz')
if not a.gpu:
    n = 8
    x = np.zeros((n, len(f)), np.float32)
    # SMC == SMCMAX gives FX=1; shading scales demand before the evaporation
    # assignment. The first four demand values are themselves subnormal.
    words = np.array([1, 2, 0x00400000, 0x007fffff,
                      0x00800000, 0x00800000, 0x01000000, 0], np.uint32)
    x[:, f['ETP1']] = words.view(np.float32)
    for name, value in dict(SMCMAX=.45, SMCDRY=.05, FXEXP=1.,
                            DT=60., BEXP=4., SMCREF=.3, SMCWLT=.1).items():
        x[:, f[name]] = value
    x[4:6, f['SHDFAC']] = .5
    for k, depth in enumerate((-.1, -.4, -1., -2.)):
        x[:, f[f'SMC[{k}]']] = .45
        x[:, f[f'SH2O[{k}]']] = .45
        x[:, f[f'ZSOIL[{k}]']] = depth
        x[:, f[f'RTDIS[{k}]']] = .25
    for name, value in dict(CMCMAX=.0005, CFACTR=.5, PC=.5, SFCTMP=280., Q2=.005).items():
        x[:, f[name]] = value
    ix = np.zeros((n, len(layout['int'])), np.int32)
    for name in ('NSOIL', 'NROOT'):
        ix[:, layout['int'].index(name)] = 4
    tables = np.load(a.tables_from)['table_bytes']
    native = x.copy()
    fn = C.CDLL(str(base/'libnoah_oracle.so')).helper_noah_evapo
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
    fn(native.ctypes.data, ix.ctypes.data, n)
    host = x.copy()
    fn = C.CDLL(str(base/'libnoah_host.so')).helper_noah_evapo
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    fn(host.ctypes.data, ix.ctypes.data, tables.ctypes.data, n)
    np.savez(fixture, input=x, input_int=ix, native=native, table_bytes=tables)
else:
    import cupy as cp
    z = np.load(fixture)
    x, ix, tables, native = (z[k] for k in ('input', 'input_int', 'table_bytes', 'native'))
    n = len(x)
    dx, di, dt = map(cp.asarray, (x, ix, tables))
    status = cp.empty(n, cp.int32)
    module = _compile_module(_device_math_source()+_CORE_CUDA+(base/'helper_entry.cu').read_text())
    module.get_function('helper_noah_evapo')((1,), (64,), (dx, di, dt, status, np.int32(n)))
    host = cp.asnumpy(dx)
    assert not cp.asnumpy(status).any()
expected = native[:, f['EDIR']].view(np.uint32)
subnormals = (expected & 0x7fffffff != 0) & (expected & 0x7f800000 == 0)
assert int(subnormals.sum()) >= 6, 'fixture must expose nonzero native subnormal outputs'
diff = int((native.view(np.uint32) != host.view(np.uint32)).sum())
result = {'scope': 'strict standalone MPAS EVAPO; single process, no MPI',
          'process_count': 1, 'backend': 'hex-cuda' if a.gpu else 'host-glibc',
          'columns': n, 'nonzero_subnormal_outputs': int(subnormals.sum()),
          'native_evaporation_words': [f'{int(w):08x}' for w in expected],
          'differing_words': diff,
          'production_source_sha256': hashlib.sha256(Path(__import__('hexcore.mpasphys.noah_v842', fromlist=['']).__file__).read_bytes()).hexdigest()}
out = Path('receipts/subnormal-cuda.json' if a.gpu else 'n1out/subnormal-host.json')
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert diff == 0
