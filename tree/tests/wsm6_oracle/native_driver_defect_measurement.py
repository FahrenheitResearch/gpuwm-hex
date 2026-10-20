"""Measure strict native driver boundary behaviour; no exclusions or parity claim.

The full native driver slice runs in isolated children. Low model tops exercise
mpas_atmphys_driver_microphysics.F:711-727's kp+1 read. Zero and negative
timesteps exercise the native scheme and tendency writeback without a repair.
These standalone observations do not replace the policy's whole-forecast,
cross-host and process-count witness.
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
from hexcore.mpasphys.wsm6_v842 import FIELD_NAMES
from replay import CONSTANTS


def child(root, index, levels=(2, 8)):
    count = len(levels)
    nz = levels[index % count]
    dt = (30., 0., -30.)[(index // count) % 3]
    dt_dyn = (30., 0., -30.)[(index // (3*count)) % 3]
    low_top = bool((index // (9*count)) % 2)
    wet = bool((index // (18*count)) % 2)
    nc = ns = 1
    c = CONSTANTS.copy()
    surfaces = set(FIELD_NAMES[30:40]) | set(FIELD_NAMES[41:])
    fields = {n: np.full((nc,) if n in surfaces else
                        (nz+1, nc) if n == 'zgrid' else (nz, nc),
                        .0123, np.float32) for n in FIELD_NAMES}
    ratio = np.float32(c[4] / c[3])
    rcv = np.float32(c[3] / np.float32(c[1]-c[3]))
    p = np.linspace(96000, 22000, nz, dtype=np.float32)[:, None]
    pi = np.power(p/np.float32(100000), np.float32(c[3]/c[1]))
    fields['qv'].fill(.015)
    for n in ('qc', 'qi', 'qr', 'qs', 'qg'):
        fields[n].fill(.001 if wet else 0)
    fields['theta_m'] = np.float32(280)/pi*(np.float32(1)+ratio*fields['qv'])
    fields['rho_zz'] = p/(c[3]*np.float32(280))
    fields['zz'].fill(1)
    fields['exner'] = pi.copy()
    fields['exner_base'] = pi.copy()
    fields['pressure_base'] = p
    fields['pressure_p'].fill(0)
    fields['rtheta_base'] = fields['rho_zz']*fields['theta_m']
    fields['rtheta_p'].fill(0)
    fields['zgrid'] = np.linspace(17, 900 if low_top else 17000,
                                  nz+1, dtype=np.float32)[:, None]
    fields['rainnc'].fill(0)
    fields['surface_pressure'].fill(98000)
    counter = np.zeros(nc, np.int32)
    inputs = {n: a.view(np.uint32).ravel().tolist() for n, a in fields.items()}
    fn = ct.CDLL(str(root/'libdriver.so')).wsm6_driver_oracle
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int)
    fn.argtypes = ([ct.c_int]*4 + [ct.c_float]*2 +
                   [ct.c_int, fp, ct.POINTER(ct.c_void_p), ip] +
                   [ct.c_int]*2 + [ct.c_float, ct.c_int] + [ct.c_float]*3)
    ptrs = (ct.c_void_p*len(FIELD_NAMES))(*[fields[n].ctypes.data for n in FIELD_NAMES])
    fn(nz, nc, ns, nz, dt, dt_dyn, 0, c.ctypes.data_as(fp), ptrs,
       counter.ctypes.data_as(ip), 1, 1, 0., 0, ratio, rcv, np.float32(100000))
    print(json.dumps({'case': index, 'nz': nz, 'dt': dt, 'dt_dyn': dt_dyn,
                     'low_top': low_top, 'wet': wet, 'input_words': inputs,
                     'output_words': {n: a.view(np.uint32).ravel().tolist()
                                      for n, a in fields.items()},
                     'counter_words': counter.view(np.uint32).tolist()}, sort_keys=True))


def measure(root, out, levels=(2, 8)):
    out.mkdir(exist_ok=False, parents=True)
    rows = []
    for index in range(36*len(levels)):
        repeats = []
        for repetition in range(2):
            env = os.environ.copy()
            env['MALLOC_PERTURB_'] = str((repetition+1)*85)
            command = [sys.executable, str(Path(__file__).resolve()), str(root),
                       str(out), '--child', str(index), '--levels',
                       ','.join(str(n) for n in levels)]
            try:
                p = subprocess.run(command, env=env, capture_output=True,
                                   text=True, timeout=5)
                item = dict(exit_code=p.returncode, stdout=p.stdout,
                            stderr=p.stderr, timed_out=False)
                if p.returncode == 0:
                    item['result'] = json.loads(p.stdout)
            except subprocess.TimeoutExpired as error:
                item = dict(exit_code=None, timed_out=True,
                            stdout=str(error.stdout or ''), stderr=str(error.stderr or ''))
            repeats.append(item)
        results = [r['result'] for r in repeats if 'result' in r]
        rows.append(dict(case=index, repeats=repeats, both_complete=len(results)==2,
                         repeated_output_words_identical=len(results)==2 and
                         results[0]['output_words']==results[1]['output_words'] and
                         results[0]['counter_words']==results[1]['counter_words']))
        print(index, [r['exit_code'] for r in repeats], flush=True)
    value = dict(scope=__doc__, host=os.uname().nodename, process_count=1,
                 acceptance=False, exclusions_proposed=[], complete=True, rows=rows,
                 levels=list(levels),
                 native_build_manifest=json.loads((root/'build-manifest.json').read_text()),
                 native_library_sha256=hashlib.sha256((root/'libdriver.so').read_bytes()).hexdigest(),
                 measurement_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'matrix.json').write_text(json.dumps(value, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--child', type=int)
    p.add_argument('--levels', default='2,8',
                   help='explicit logical level counts; default preserves the original 72-case matrix')
    args = p.parse_args()
    levels = tuple(int(n) for n in args.levels.split(','))
    if not levels or len(set(levels)) != len(levels) or any(not 1 <= n <= 256 for n in levels):
        p.error('levels must be unique integers in 1..256')
    if args.child is None:
        measure(args.root.resolve(), args.out.resolve(), levels)
    else:
        child(args.root.resolve(), args.child, levels)
