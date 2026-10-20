"""Compare Rust file decoding with the pristine MPAS table reader, byte by byte."""
from pathlib import Path
import ctypes as C
import json
import os
import subprocess
import sys
import numpy as np
from hexcore.mpasphys.noah_v842 import TABLE_SCHEMA, table_dtype

base = Path(__file__).resolve().parent
if len(sys.argv) == 1:
    results = []
    for classification in ('USGS', 'MODIFIED_IGBP_MODIS_NOAH'):
        # Fresh processes preserve native BSS zeros outside the selected table.
        raw = subprocess.check_output((sys.executable, __file__, classification))
        results.append(json.loads(raw))
    Path('n1out/table-comparison.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    raise SystemExit(int(any(r['differing_bytes'] for r in results)))

classification = sys.argv[1].encode()
native = C.CDLL(str(base / 'libnoah_oracle.so'))
os.chdir(base)
fn = native.read_tables_for
fn.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int]
fn(classification, b'STAS', len(classification), 4)
expected = np.zeros(1, dtype=table_dtype())
for name, spec in TABLE_SCHEMA.items():
    dtype, offset = expected.dtype.fields[name]
    if spec['parameter']:
        expected[name] = int(spec['init']) if spec['type'] == 'int' else float(spec['init'])
    else:
        try:
            source = (C.c_ubyte * dtype.itemsize).in_dll(native, '__module_sf_noahlsm_MOD_' + name.lower())
        except ValueError:
            continue
        C.memmove(expected.ctypes.data + offset, C.addressof(source), dtype.itemsize)
landuse = np.zeros((7,12,100), dtype=np.float32)
native.read_landuse_for.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
native.read_landuse_for(landuse.ctypes.data, classification, len(classification))
actual = subprocess.check_output((str(base / 'table_decode'), str(base), classification.decode(), 'STAS'))
want = expected.tobytes() + landuse.tobytes()
a = np.frombuffer(actual, dtype=np.uint8)
b = np.frombuffer(want, dtype=np.uint8)
diff = a != b
print(json.dumps({'landuse': classification.decode(), 'bytes': len(actual),
                  'differing_bytes': int(diff.sum()),
                  'first_differing_offsets': np.flatnonzero(diff)[:20].tolist()}))
