"""Compare all sealed column and driver boundary words with the host-glibc bodies.

This is a CPU diagnostic using the complete production CUDA source compiled
for host execution; no expected output is fed to those bodies.
"""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import json

import numpy as np
from hexcore.mpasphys.wsm6_v842 import FIELD_NAMES
from replay import CONSTANTS


def run(root, native, driver, receipt):
    lib = ct.CDLL(str(root/'libhost.so'))
    fp, ip = ct.POINTER(ct.c_float), ct.POINTER(ct.c_int)
    rows = []
    for scope, path in [('column', native), ('driver', driver)]:
        data = json.loads(path.read_text())
        assert data['complete'] and all(r['both_complete'] for r in data['rows'])
        for row in data['rows']:
            source = row['repeats'][0]['result']
            nz = source['nz']
            if scope == 'column':
                initial = source['input_words']
                arrays = [np.asarray(initial[n], np.uint32).view(np.float32).copy()
                          for n in ('constants', 'volume', 'surface', 'radii', 'dbz', 'production')]
                fn = lib.wsm6_wrapper_host if source['wrapper'] else lib.wsm6_host
                fn.argtypes = [ct.c_int, ct.c_float, ct.c_int]+[fp]*(7 if source['wrapper'] else 6)+([ct.c_int] if source['wrapper'] else [])
                arguments = [a.ctypes.data_as(fp) for a in arrays]
                if source['wrapper']:
                    pi = np.full(nz, .97, np.float32)
                    arguments += [pi.ctypes.data_as(fp), 1]
                fn(nz, source['dt'], 0, *arguments)
                actual = {n:a.view(np.uint32).tolist() for n,a in zip(
                    ('volume','surface','radii','dbz','production'),arrays[1:])}
            else:
                fields = {n:np.asarray(words, np.uint32).view(np.float32).copy()
                          for n,words in source['input_words'].items()}
                c = CONSTANTS.copy()
                counter = np.zeros(1, np.int32)
                ptrs = (ct.c_void_p*len(FIELD_NAMES))(*[fields[n].ctypes.data for n in FIELD_NAMES])
                fn = lib.wsm6_driver_host
                fn.argtypes = [ct.c_int]*4+[ct.c_float]*2+[ct.c_int,fp,ct.POINTER(ct.c_void_p),ip]+[ct.c_int]*2+[ct.c_float,ct.c_int]+[ct.c_float]*3
                fn(nz,1,1,nz,source['dt'],source['dt_dyn'],0,c.ctypes.data_as(fp),ptrs,
                   counter.ctypes.data_as(ip),1,1,0.,0,np.float32(c[4]/c[3]),
                   np.float32(c[3]/np.float32(c[1]-c[3])),np.float32(100000))
                actual = {n:a.view(np.uint32).tolist() for n,a in fields.items()}
            diff = {n:[{'index':i,'native':x,'candidate':y} for i,(x,y) in enumerate(
                    zip(words,actual[n])) if x!=y] for n,words in source['output_words'].items()}
            rows.append(dict(scope=scope,case=row['case'],parameters={k:v for k,v in source.items()
                             if k not in ('input_words','output_words','counter_words')},
                             actual_words=actual,differences=diff,
                             compared_words=sum(len(a) for a in actual.values())))
    result = dict(scope=__doc__,complete=True,acceptance=False,process_count=1,rows=rows,
                  compared_words=sum(r['compared_words'] for r in rows),
                  differing_words=sum(len(a) for r in rows for a in r['differences'].values()),
                  native_input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (native,driver)},
                  host_library_sha256=hashlib.sha256((root/'libhost.so').read_bytes()).hexdigest(),
                  build_manifest=json.loads((root/'build-manifest.json').read_text()))
    receipt.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('native',type=Path)
    p.add_argument('driver',type=Path)
    p.add_argument('receipt',type=Path)
    a=p.parse_args()
    run(a.root.resolve(),a.native,a.driver,a.receipt)
