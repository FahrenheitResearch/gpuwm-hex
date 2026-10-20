"""Observe unmodified strict MPAS cuflx index-zero writes before port changes."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def run(build, receipt):
    native = c.CDLL(str((build / 'libcu6_oracle.so').resolve()))
    native.cu6_unsafe_flux_values.argtypes = [c.c_int, c.c_void_p, c.c_void_p]
    rows = []
    for columns in (1, 2):
        for repeat in (1, 2):
            values = np.empty(columns * 120, np.float32)
            flags = np.empty(columns * 3 + 1, np.int32)
            native.cu6_unsafe_flux_values(columns, values.ctypes.data, flags.ctypes.data)
            rows.append({'columns': columns, 'repeat': repeat,
                         'values_uint32': values.view(np.uint32).tolist(),
                         'flags': flags.tolist()})
    command = 'import ctypes,numpy as np; a=np.empty(120,np.float32); b=np.empty(4,np.int32); ctypes.CDLL(' + repr(str((build/'libcu6_checked.so').resolve())) + ').cu6_unsafe_flux_values(1,ctypes.c_void_p(a.ctypes.data),ctypes.c_void_p(b.ctypes.data))'
    checked = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
    result = {'scope': 'standalone strict pristine MPAS memory-slab witness; no forecast or MPI-rank proof',
              'host': os.uname().nodename, 'process_count': 1,
              'source': 'module_cu_tiedtke.F:2329,2339-2341',
              'oracle_sha256': hashlib.sha256((build/'libcu6_oracle.so').read_bytes()).hexdigest(),
              'stable_repeats': all(rows[i]['values_uint32'] == rows[i+1]['values_uint32'] and
                                    rows[i]['flags'] == rows[i+1]['flags'] for i in (0,2)),
              'checked_exit': checked.returncode, 'checked_stderr': checked.stderr,
              'rows': rows}
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return int(not result['stable_repeats'] or checked.returncode == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.build, args.receipt))
