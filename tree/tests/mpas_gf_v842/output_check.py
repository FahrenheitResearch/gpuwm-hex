"""Exercise native mass-flux refusal and precipitation/error dispatch branches."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np


def check(root,fixture,receipt,gpu):
    if gpu:
        import cupy as cp
        from hexcore.mpasphys import gf_mpas_v842 as gf
        module=gf._compile_module(gf.cuda_source()+'\n'+Path(__file__).with_name('output_entry.cuh').read_text(),cp.cuda.runtime.getDevice())
        arena=cp.empty(65536,dtype=cp.uint8);saved=np.load(fixture)
    else:
        native=C.CDLL(str(root/'gf_native.so'));host=C.CDLL(str(root/'gf_host.so'));saved={}
    errors=0;high_flux=0;cases=0;nz=8
    for ci,flux in enumerate((0.,.5,100.,101.,200.)):
        a=np.full((13,nz,4),.25,dtype='f4');s=np.full((18,4),.5,dtype='f4')
        e=np.full((2,16,4),flux,dtype='f4');e[1]=[1.,0.,-1.,1.]
        ints=np.zeros((4,4),dtype='i4');ints[0,3]=7;ints[1]=[nz,nz-1,1,nz]
        s[2]=1.;s[11]=[16.,8.,1.,16.]
        arrays=[a,s,e,ints]
        if gpu:
            expected=[saved[f'{ci}_{i}'] for i in range(4)];device=[cp.asarray(x) for x in arrays]
            module.get_function('gf_aux_output')((1,),(1,),(np.int32(nz),*device,arena))
            got=[cp.asnumpy(x) for x in device]
        else:
            expected=[x.copy() for x in arrays];got=[x.copy() for x in arrays]
            for lib,name,values in ((native,'gf_native_output',expected),(host,'gf_host_output',got)):
                fn=getattr(lib,name);fn.argtypes=[C.c_int]+[C.c_void_p]*4
                fn(nz,*[C.c_void_p(x.ctypes.data) for x in values])
            for i,x in enumerate(expected):saved[f'{ci}_{i}']=x
        errors+=sum(int(np.count_nonzero(x.view('u4')!=y.view('u4'))) for x,y in zip(expected,got));cases+=4
        if flux>100.:high_flux+=int(expected[3][0,0]==19 and expected[1][1,0]>100.)
    if not gpu:np.savez_compressed(fixture,**saved)
    result=dict(backend='hex-cuda' if gpu else 'host-glibc',cases=cases,different_words=errors,
                high_mass_flux_witnesses=high_flux,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),forecast_proven=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return bool(errors or high_flux!=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    raise SystemExit(check(a.root.resolve(),a.fixture,a.receipt,a.gpu))
