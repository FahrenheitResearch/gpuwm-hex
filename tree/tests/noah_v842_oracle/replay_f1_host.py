#!/usr/bin/env python
"""Host-glibc witness for real typed captures; numerical packing stays in Rust."""
import ctypes as C
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

base = Path(__file__).resolve().parent
source, output = map(Path, sys.argv[1:])
names = sorted(p.name[len(source.name + '.post.'):-4]
               for p in source.parent.glob(source.name + '.post.*.bin'))
with tempfile.TemporaryDirectory(prefix='noah-f1-host-', dir=output.parent) as tmp:
    packed = Path(tmp) / 'packed'
    subprocess.run([str(base / 'dump_transport'), 'pack', str(source), str(packed),
                    str(base / 'table_decode'), str(base)], check=True)
    x = np.fromfile(str(packed) + '.f32', dtype='<f4').reshape(-1, 79)
    ix = np.fromfile(str(packed) + '.i32', dtype='<i4').reshape(-1, 7)
    tb = np.fromfile(str(packed) + '.table', dtype=np.uint8)
    locations = np.fromfile(str(packed) + '.locations', dtype='<i4')
    state = np.fromfile(str(packed) + '.module_locations', dtype='<i4')
    calls = np.zeros(len(x), dtype='<i4')
    library = C.CDLL(str(base / 'libnoah_host.so'))
    fn = library.replay_driver_with_state
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    fn(x.ctypes.data, ix.ctypes.data, tb.ctypes.data, len(x),
       locations.ctypes.data, calls.ctypes.data)
    fn = library.noah_finish_module_locations
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
    fn(locations.ctypes.data, calls.ctypes.data, state.ctypes.data, len(x))
    x.tofile(str(packed) + '.f32')
    ix.tofile(str(packed) + '.i32')
    state.tofile(str(packed) + '.module_locations')
    listing = Path(tmp) / 'outputs.txt'
    listing.write_text('\n'.join(names) + '\n')
    subprocess.run([str(base / 'dump_transport'), 'unpack', str(source), str(output),
                    str(packed), str(listing)], check=True)
