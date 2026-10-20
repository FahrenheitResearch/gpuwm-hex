"""Native whole driver stress replay: real initialized state, synthetic air."""
from pathlib import Path
import argparse
import ctypes as C
import hashlib
import json
import os
import subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--prefix', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--input-file', type=Path, required=True)
p.add_argument('--publication', type=Path, required=True)
p.add_argument('--decoded', type=Path, required=True)
p.add_argument('--initialized-input', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
a = p.parse_args()
base = Path(__file__).resolve().parent
a.output.mkdir(parents=True, exist_ok=True)
names = ('skintemp', 'tmn', 'xland', 'xice', 'vegfra', 'shdmin', 'shdmax', 'snowc', 'sfc_albbck', 'dzs')
metadata = json.loads((a.decoded / 'metadata.json').read_text())
assert tuple(v['name'] for v in metadata['variables']) == names
z = np.load('n1out/driver-synthetic.npz')
z['input'].tofile(str(a.prefix) + '-synthetic.f32')
z['input_int'].tofile(str(a.prefix) + '-synthetic.i32')
real = np.load(a.initialized_input)
real['native'].tofile(str(a.prefix) + '-init.f32')
real['native_int'].tofile(str(a.prefix) + '-init.i32')
subprocess.run([str(base / 'pack_real_driver'), str(a.prefix) + '-synthetic',
                str(a.prefix) + '-init', str(a.decoded), str(a.prefix)], check=True)
x = np.fromfile(str(a.prefix) + '.f32', dtype='<f4').reshape(-1, 79)
ix = np.fromfile(str(a.prefix) + '.i32', dtype='<i4').reshape(-1, 7)
native = C.CDLL(str(base / 'libnoah_oracle.so'))
host = C.CDLL(str(base / 'libnoah_host.so'))
old = Path.cwd()
os.chdir(base)
native.read_tables()
os.chdir(old)
tb = z['table_bytes']
native_float, native_int = x.copy(), ix.copy()
fn = native.replay_driver
fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
fn(native_float.ctypes.data, native_int.ctypes.data, len(x))
host_float, host_int = x.copy(), ix.copy()
fn = host.replay_driver
fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
fn(host_float.ctypes.data, host_int.ctypes.data, tb.ctypes.data, len(x))
float_words = int((native_float.view(np.uint32) != host_float.view(np.uint32)).sum())
int_words = int((native_int != host_int).sum())
result = {'scheme': 'driver', 'physical_cells': len(x) // 4, 'columns': len(x),
          'differing_words': float_words, 'integer_differing_words': int_words,
          'source': 'published real initialized surface/soil states with broad synthetic atmospheric forcing',
          'scope': 'standalone whole-driver stress replay; no native capture or forecast acceptance',
          'input_file_sha256': hashlib.sha256(a.input_file.read_bytes()).hexdigest(),
          'publication_sha256': hashlib.sha256(a.publication.read_bytes()).hexdigest()}
assert result['input_file_sha256'] in json.loads(a.publication.read_text())['files'].values(), 'input is absent from published file hashes'
np.savez(a.output / 'driver-synthetic.npz', input=x, input_int=ix,
         native=native_float, native_int=native_int, table_bytes=tb)
a.receipt.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
if float_words or int_words:
    raise SystemExit(1)
