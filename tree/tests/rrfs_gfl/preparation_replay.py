"""Native driver dx/radiation replay; not a whole-forecast receipt."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np


def capture(root,fixture):
    rng=np.random.default_rng(81231)
    nc,nz,n=137,55,128
    density=np.exp(rng.uniform(-7,2,nc)).astype('f4')
    lw=np.asfortranarray(rng.normal(0,1e-4,(nz,nc)).astype('f4'))
    sw=np.asfortranarray(rng.normal(0,1e-4,(nz,nc)).astype('f4'))
    lw[0,:4]=np.array([2e-40,-2e-40,0.,-0.],dtype='f4')
    sw[0,:4]=np.array([1e-40,-1e-40,-0.,-0.],dtype='f4')
    dx=np.full(n,73.25,dtype='f4');rad=np.full((n,nz+1),73.25,dtype='f4',order='F')
    lib=ctypes.CDLL(str(root.resolve()/'gfl_native.so'));f=lib.gfl_prepare_native
    f.argtypes=[ctypes.c_int]*3+[ctypes.c_float]+[ctypes.c_void_p]*5
    f(n,nz,nc,240000.,*[v.ctypes.data for v in (density,lw,sw,dx,rad)])
    np.savez_compressed(fixture,n=n,nz=nz,len_disp=np.float32(240000),density=density,
                        lw=lw,sw=sw,dx=dx,rad=rad)


def replay(fixture,receipt):
    import cupy as cp
    from hexcore.mpasphys.rrfs.cu_gfl_bf6a4b0 import _module
    x=np.load(fixture);n,nz=int(x['n']),int(x['nz']);nc=x['density'].size
    density=cp.asarray(x['density']);lw=cp.asarray(np.ascontiguousarray(x['lw']))
    sw=cp.asarray(np.ascontiguousarray(x['sw']));dx=cp.full(n,73.25,dtype='f4')
    rad=cp.full((n,nz+1),73.25,dtype='f4');m=_module(cp.cuda.runtime.getDevice())
    m.get_function('gfl_driver_dx')(((n+127)//128,),(128,),
        (np.int32(n),np.float32(x['len_disp']),density,dx))
    m.get_function('gfl_driver_radiation')(((n*nz+127)//128,),(128,),
        (np.int32(n),np.int32(nz),np.int32(nc),lw,sw,rad))
    counts={name:int(np.count_nonzero(cp.asnumpy(a).view('u4')!=x[name].view('u4')))
            for name,a in (('dx',dx),('rad',rad))}
    out={'scope':'standalone fork driver preparation; not forecast acceptance',
         'fork_pin':'1d6e370e','different_words':counts,'total_different_words':sum(counts.values())}
    receipt.write_text(json.dumps(out,indent=2)+'\n');return bool(sum(counts.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.root:capture(a.root,a.fixture)
    else:raise SystemExit(replay(a.fixture,a.receipt))
