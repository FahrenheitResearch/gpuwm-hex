"""Standalone MPAS oracle replay, never a whole-forecast acceptance receipt."""
import argparse
import ctypes as C
from pathlib import Path
import json
import hashlib
import numpy as np
if __package__:
    from .support import PARAMS,cases,load_oracles,native_column,word_diff
else:
    from support import PARAMS,cases,load_oracles,native_column,word_diff
from hexcore.mpasphys import kain_fritsch_v842 as kf


def replay(build_dir, backend):
    fixture_sha=hashlib.sha256(Path(__file__).with_name("support.py").read_bytes()).hexdigest()
    native,host,table=load_oracles(build_dir)
    rows=[]
    if backend=='hex-cuda':
        import cupy as cp
        tables=kf.build_tables(cp.asarray(PARAMS))
    for key,(f,s,v),flags in cases():
        params=PARAMS.copy()
        if key[0]=='schedule':params[1]=key[-1]
        if key[0]=='adaptive':params[0],params[2]=key[1:]
        if key[0] in ('closure','tenth-cycle'):params[0]=key[-1]
        if key[0]=='parameter-gravity':params[9]=key[-1]
        ns,nv,na=native_column(native,f,s,v,flags,params)
        nz=f.shape[1]-1
        if backend=='host-glibc':
            ds,dv=s.copy(),v.copy();a=np.zeros(1,np.uint8)
            workspace=np.empty(kf.WORKSPACE_ARRAYS*nz+1,np.float32)
            host.kf_host_column(nz,1,f,ds,dv,a,params,flags,table,workspace)
            da=int(a[0])
        else:
            ds,dv=cp.asarray(s[:,:,None]),cp.asarray(v[:,None]);a=cp.zeros(1,cp.uint8)
            kf.run_packed(cp.asarray(f[:,:,None]),ds,dv,a,cp.asarray(params),cp.asarray(flags),tables=tables)
            ds,dv=cp.asnumpy(ds)[:,:,0].copy(),cp.asnumpy(dv)[:,0].copy();da=int(a[0])
        rows.append({'case':key,'state_words':word_diff(ns,ds),'surface_words':word_diff(nv,dv),
            'active_equal':na==da,'rain_native':float(nv[2]),'rain_backend':float(dv[2]),'nca':float(nv[4])})
    return {'schema':'mpas-kf-standalone-replay-v1','backend':backend,'source_sha256':kf.SOURCE_SHA256,
        'fixture_source_sha256':fixture_sha,
        'fixture_input_sha256':{'closure_tenth_cycle.npz':hashlib.sha256(Path(__file__).with_name('closure_tenth_cycle.npz').read_bytes()).hexdigest()},
        'translation_unit_sha256':kf.source_digest() if backend=='hex-cuda' else None,
        'cuda_options':list(kf.CUDA_OPTIONS) if backend=='hex-cuda' else None,
        'columns':len(rows),'parameter_stress_columns':sum(r['case'][0]=='parameter-gravity' for r in rows),'different_columns':sum(x['state_words']+x['surface_words']>0 or not x['active_equal'] for x in rows),
        'state_different_words':sum(x['state_words'] for x in rows),'surface_different_words':sum(x['surface_words'] for x in rows),
        'native_negative_rain_columns':sum(x['rain_native']<0 for x in rows),
        'nca_300_columns':sum(x['nca']==300 for x in rows),'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',required=True)
    p.add_argument('--backend',choices=('host-glibc','hex-cuda'),required=True)
    p.add_argument('--receipt',required=True);args=p.parse_args()
    result=replay(args.build_dir,args.backend)
    Path(args.receipt).write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})
    raise SystemExit(bool(result['different_columns']))
