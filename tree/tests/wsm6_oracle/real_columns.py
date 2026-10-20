"""Bind Rust-decoded real WSM6 columns to the complete standalone oracles.

This is host witness evidence. It neither decodes weather files in Python nor
asserts f1 CUDA replay, native hidden-state replay or whole-model acceptance.
"""
from pathlib import Path
import argparse
import ctypes as ct
import hashlib
import json
import subprocess
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array(words, shape):
    # Buffer binding only: all file decode, selection and physical preparation
    # occur in rw_mpas_wsm6_oracle. No NumPy weather arithmetic occurs here.
    return np.array(words, dtype=np.uint32).view(np.float32).reshape(shape)


def run(fixture, oracle):
    nz = fixture['nz']; wrapper = 'capture' in fixture
    c = array(fixture['constants'], (26,))
    dt = array([fixture['dt']], (1,))[0] if wrapper else np.float32(30)
    fp = ct.POINTER(ct.c_float)
    native = ct.CDLL(str(oracle.resolve()/('libwrapper.so' if wrapper else 'liboracle.so')))
    host = ct.CDLL(str(oracle.resolve()/'libhost.so'))
    nf = native.wsm6_wrapper_oracle if wrapper else native.wsm6_oracle
    hf = host.wsm6_wrapper_host if wrapper else host.wsm6_host
    for f in (nf, hf):
        f.argtypes = [ct.c_int, ct.c_float, ct.c_int]+[fp]*(7 if wrapper else 6)+([ct.c_int] if wrapper else [])
    rows = []
    for row in fixture['rows']:
        initial = [array(row['volume'], (10,nz)), array(row['surface'], (7,)),
                   array(row['radii'], (3,nz)), np.full(nz,-35,np.float32),
                   np.zeros((2,nz),np.float32)]
        outputs = []
        for f in (nf,hf):
            buffers = [a.copy() for a in initial]
            args = [c,*buffers]
            if wrapper:
                pi = array(row['pi'], (nz,)); args.append(pi)
            f(nz,dt,0,*[a.ctypes.data_as(fp) for a in args],
              *([int(fixture['radii_on'])] if wrapper else []))
            outputs.append(buffers)
        differences = {name:int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
                       for name,a,b in zip(('volume','surface','radii','dbz','production'),*outputs)}
        captured = {}
        if wrapper:
            expected = [array(row['expected_volume'], (7,nz)),
                        array(row['expected_surface'], (7,)),
                        array(row['expected_radii'], (3,nz))]
            captured = {name:int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
                        for name,a,b in zip(('volume','surface','radii'),
                                             (outputs[1][0][:7],outputs[1][1],outputs[1][2]), expected)}
        rows.append(dict(cell_id=row['cell_id'], differences=differences,
                         captured_post_differences=captured))
    metadata = {k:v for k,v in fixture.items() if k!='rows'}
    total = sum(sum(r['differences'].values())+sum(r['captured_post_differences'].values()) for r in rows)
    total += sum(fixture.get('inactive_native_padding_differences', {}).values())
    return dict(scope='strict standalone and actual CUDA bodies on host glibc; not GPU or forecast acceptance',
                metadata=metadata, rows=rows,total_differing_words=total)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--transport',required=True,type=Path)
    p.add_argument('--oracle',required=True,type=Path)
    p.add_argument('--receipt',required=True,type=Path)
    p.add_argument('--count',type=int,default=256)
    p.add_argument('--capture',action='append',nargs=2,default=[],metavar=('ROOT','PREFIX'))
    p.add_argument('--init',action='append',type=Path,default=[])
    p.add_argument('--constants-json',required=True,type=Path)
    a=p.parse_args();results=[]
    receipt=dict(transport_sha256=sha(a.transport),
                 oracle_sha256={n:sha(a.oracle/n) for n in ('liboracle.so','libwrapper.so','libhost.so')},
                 comparisons=results,comparison_complete=False,total_differing_words=0)
    def checkpoint():
        receipt['total_differing_words']=sum(r['total_differing_words'] for r in results)
        a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        for root,prefix in a.capture:
            fixture=json.loads(subprocess.check_output([str(a.transport.resolve()),'capture',root,prefix,str(a.count)],text=True))
            # Only scalar constant metadata is written; column data remains in memory.
            a.constants_json.write_text(json.dumps({'constants':fixture['constants']})+'\n')
            results.append(run(fixture,a.oracle)); checkpoint()
        for path in a.init:
            fixture=json.loads(subprocess.check_output([str(a.transport.resolve()),'init',str(path),str(a.count),str(a.constants_json)],text=True))
            results.append(run(fixture,a.oracle)); checkpoint()
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        receipt['error']=str(e); checkpoint()
        raise
    receipt['comparison_complete']=True; checkpoint()
    print(len(results),'fixtures;',sum(len(r['rows']) for r in results),'columns;',receipt['total_differing_words'],'differing words')
    return bool(receipt['total_differing_words'])


if __name__=='__main__':
    raise SystemExit(main())
