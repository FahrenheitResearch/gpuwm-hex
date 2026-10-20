"""Diagnose the native exact-final-latitude read in isolated child processes.

This is a standalone diagnostic, not the full-executable defect-policy proof.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--strict', type=Path, required=True)
    parser.add_argument('--bounds', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--child', choices=('strict', 'bounds'))
    parser.add_argument('--repeats',type=int,default=8)
    args=parser.parse_args()
    if args.child:
        import numpy as np
        from radshared_replay_v842 import P, ptr
        lib=ct.CDLL(str((args.strict if args.child=='strict' else args.bounds).resolve()))
        lib.ozone_init.argtypes=[ct.c_int]*2+[P]*3
        os.chdir(args.tables.resolve())
        latitudes=np.loadtxt('OZONE_LAT.TBL', dtype='f4')
        levels=np.loadtxt('OZONE_PLEV.TBL', dtype='f4')
        degrad=np.float32(np.float32(3.1415926)/np.float32(180))
        lat=np.array([latitudes[-1]*degrad], dtype='f4')
        assert lat[0]/degrad==latitudes[-1]
        out=np.full((1,levels.size,12), np.float32(-12345), dtype='f4')
        pin=np.empty_like(levels)
        lib.ozone_init(1,levels.size,ptr(lat),ptr(pin),ptr(out))
        print(json.dumps({'monthly_words':out.view('u4').reshape(-1).tolist()}))
        return
    runs=[]
    assert args.repeats>=2
    for kind in ['strict']*args.repeats+['bounds']:
        command=[sys.executable,str(Path(__file__).resolve()),'--strict',str(args.strict.resolve()),
                 '--bounds',str(args.bounds.resolve()),'--tables',str(args.tables.resolve()),
                 '--receipt',str(args.receipt.resolve()),'--child',kind]
        environment=os.environ.copy()
        # Do not count undefined-input diagnostics as valid-column branch coverage.
        environment['GCOV_PREFIX']=str((Path.cwd()/'scratch/native-defect-gcov').resolve())
        environment['GCOV_PREFIX_STRIP']='0'
        result=subprocess.run(command,capture_output=True,text=True,env=environment)
        entry={'kind':kind,'exit_code':result.returncode,'stderr':result.stderr}
        if result.returncode==0:
            entry.update(json.loads(result.stdout))
        else:
            entry['stdout']=result.stdout
        runs.append(entry)
    assert runs[-1]['exit_code']!=0 and '65' in runs[-1]['stderr'],runs[-1]
    strict_words=[x.get('monthly_words') for x in runs[:-1]]
    differing=[] if None in strict_words else [i for i,words in enumerate(zip(*strict_words)) if len(set(words))>1]
    receipt={'purpose':'standalone diagnostic only; full strict executable, hosts/ranks still required',
             'source':'mpas_atmphys_o3climatology.F:164-167',
             'hostname':platform.node(),'libc':platform.libc_ver(),
             'sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.strict,args.bounds)},
             'strict_children_same_words':None if None in strict_words else not differing,
             'runs':runs,'standalone_varying_word_indices':differing,
             'declared_native_undefined':False,'excluded_words':[]}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'bounds_exit':runs[-1]['exit_code'],
                      'strict_children_same_words':receipt['strict_children_same_words']}))


if __name__=='__main__':main()
