"""Measure strict native behaviour at every current column-launcher guard.

This is a standalone debug witness, not a whole-forecast exclusion or a
cross-host/process-count witness. Run children separately so native crashes
and stalls remain evidence without preventing the rest of the matrix.
"""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
import sys

import numpy as np


def child(root, index):
    from replay import CONSTANTS
    wrapper = bool(index // 24)
    local = index % 24
    nz = (1, 2)[local % 2]
    dt = (30., 0., -30.)[(local // 2) % 3]
    regime = (local // 6) % 4
    c = CONSTANTS.copy()
    v = np.zeros((10, nz), np.float32)
    v[0] = 280 if regime < 2 else 250
    v[1] = .015
    if regime % 2:
        v[2:7] = .001
    v[7] = 1.1
    v[8] = 90000
    v[9] = 500
    pi = np.full(nz, .97, np.float32)
    if wrapper:
        v[0] /= pi
    s = np.zeros(7, np.float32)
    r = np.zeros((3, nz), np.float32)
    d = np.zeros(nz, np.float32)
    p = np.zeros((2, nz), np.float32)
    arrays = [c, v, s, r, d, p]
    fp = ct.POINTER(ct.c_float)
    library = ct.CDLL(str(root / ('libwrapper.so' if wrapper else 'liboracle.so')))
    fn = library.wsm6_wrapper_oracle if wrapper else library.wsm6_oracle
    fn.argtypes = [ct.c_int, ct.c_float, ct.c_int] + [fp] * (7 if wrapper else 6) + ([ct.c_int] if wrapper else [])
    inputs = {name: a.view(np.uint32).ravel().tolist() for name, a in zip(('constants', 'volume', 'surface', 'radii', 'dbz', 'production'), arrays)}
    arguments = [a.ctypes.data_as(fp) for a in arrays]
    if wrapper:
        arguments.extend((pi.ctypes.data_as(fp), 1))
    fn(nz, dt, 0, *arguments)
    print(json.dumps({'case': index, 'nz': nz, 'dt': dt, 'wrapper': wrapper,
        'regime': regime, 'input_words': inputs,
        'output_words': {name: a.view(np.uint32).ravel().tolist() for name, a in zip(('volume', 'surface', 'radii', 'dbz', 'production'), arrays[1:])}}, sort_keys=True))


def measure(root, out):
    out.mkdir(exist_ok=False, parents=True)
    rows = []
    for index in range(48):
        repeats = []
        for repetition in range(2):
            env = os.environ.copy()
            env['MALLOC_PERTURB_'] = str((repetition + 1) * 85)
            command = [sys.executable, str(Path(__file__).resolve()), str(root), str(out), '--child', str(index)]
            try:
                p = subprocess.run(command, env=env, capture_output=True, text=True, timeout=5)
                item = {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr, 'timed_out': False}
                if p.returncode == 0:
                    item['result'] = json.loads(p.stdout)
            except subprocess.TimeoutExpired as error:
                item = {'exit_code': None, 'timed_out': True,
                    'stdout': str(error.stdout or ''), 'stderr': str(error.stderr or '')}
            repeats.append(item)
        successful = [a['result'] for a in repeats if 'result' in a]
        rows.append({'case': index, 'repeats': repeats, 'both_complete': len(successful) == 2,
            'repeated_output_words_identical': len(successful) == 2 and successful[0]['output_words'] == successful[1]['output_words']})
        print(index, [x['exit_code'] for x in repeats], flush=True)
    value = {'scope': __doc__, 'host': os.uname().nodename, 'process_count': 1,
        'acceptance': False, 'exclusions_proposed': [],
        'native_build_manifest': json.loads((root/'build-manifest.json').read_text()),
        'native_library_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('liboracle.so', 'libwrapper.so')},
        'rows': rows, 'complete': True}
    (out/'matrix.json').write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--child', type=int)
    args = parser.parse_args()
    if args.child is None:
        measure(args.root.resolve(), args.out.resolve())
    else:
        child(args.root.resolve(), args.child)
