"""Defined direct tests of saturation clipping, error dispatch and ensembles."""
import argparse
import ctypes as C
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def check(root, fixture, receipt, gpu):
    if gpu:
        import cupy as cp
        from hexcore.mpasphys import gf_mpas_v842 as gf
        source=gf.cuda_source()+'\n'+Path(__file__).with_name('extra_auxiliary_entry.cuh').read_text()
        module=gf._compile_module(source,cp.cuda.runtime.getDevice());arena=cp.empty(65536,dtype=cp.uint8)
        saved=np.load(fixture)
    else:
        native=C.CDLL(str(root/'gf_native.so'));host=C.CDLL(str(root/'gf_host.so'));saved={}
    counts={};errors={};nz=8
    controls=[(0,1,1,0,1,v) for v in (0,1)] + [(1,1,1,0,1,v) for v in (0,1)]
    controls += [(2,*x) for x in itertools.product((1,2,3),(1,2,3),(0,1,8,16),(0,1),(0,1,2))]
    for ci,(mode,ens4,maxens,choice,deeps,variant) in enumerate(controls):
        a=np.full((15,nz,4),.25,dtype='f4');s=np.full((14,4),.5,dtype='f4')
        e=np.full((2,48,4),.25,dtype='f4');ints=np.zeros((6,4),dtype='i4')
        ints[0]=[0,0,0,7]
        if mode==0:
            a[0]=280.;a[1]=.01;a[2]=.02;a[3]=450000.;a[4]=350000.
            a[5]=np.arange(nz,dtype='f4')[:,None]*500.;a[6]=900.-np.arange(nz,dtype='f4')[:,None]*50.
            s[0]=1000.;s[1]=100.
            if variant: a[3]=300000.
        elif mode==1:
            a[0]=np.array([3,1,4,4,2,0,2,3],dtype='f4')[:,None];ints[1]=[7,2,5,7]
            if variant:ints[0]=[7,0,7,0]
        else:
            s[0]=16.;s[1]=[1.,0.,1.,0.];s[2]=[100.,1000.,10.,100.];s[3]=[200.,100.,20.,200.]
            s[4]=.2
            for n in range(3):
                s[5+n]=s[3]+np.float32((-100.,10.,-.001)[n]);s[8+n]=[50.+n*100.,500.-n*100.,10.,50.]
                s[11+n]=[100.,1.,1000.,100.][n%4]
                a[3+n]=np.array([-2.,-1.,1.,2.],dtype='f4')*(np.arange(nz,dtype='f4')[:,None]+1.)
            a[0]=1000.-np.arange(nz,dtype='f4')[:,None]*80.;a[1]=.1;a[2]=1.
            ints[1]=[0,1,0,1];ints[2]=[0,0,1,1];ints[3]=6;ints[4]=2;ints[5]=4
            if variant==1:ints[0]=[996,7,20,0]
            if variant==2:ints[1]=0;ints[2]=0
        arrays=[a,s,e,ints];args=(nz,mode,ens4,maxens,choice,deeps)
        if gpu:
            expected=[saved[f'{ci}_{i}'] for i in range(4)];device=[cp.asarray(x) for x in arrays]
            module.get_function('gf_aux_extra')((1,),(1,),tuple(np.int32(x) for x in args)+(*device,arena))
            got=[cp.asnumpy(x) for x in device]
        else:
            expected=[x.copy() for x in arrays];got=[x.copy() for x in arrays]
            for lib,name,values in ((native,'gf_native_extra',expected),(host,'gf_host_extra',got)):
                fn=getattr(lib,name);fn.argtypes=[C.c_int]*6+[C.c_void_p]*4
                fn(*args,*[C.c_void_p(x.ctypes.data) for x in values])
            for i,x in enumerate(expected):saved[f'{ci}_{i}']=x
        key=('cup_env_clev','cup_maximi','cup_forcing_ens_3d')[mode]
        counts[key]=counts.get(key,0)+4
        errors[key]=errors.get(key,0)+sum(int(np.count_nonzero(x.view('u4')!=y.view('u4'))) for x,y in zip(expected,got))
    if not gpu:np.savez_compressed(fixture,**saved)
    result=dict(backend='hex-cuda' if gpu else 'host-glibc',cases=counts,different_words=errors,
                fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),forecast_proven=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return bool(sum(errors.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    raise SystemExit(check(a.root.resolve(),a.fixture,a.receipt,a.gpu))
