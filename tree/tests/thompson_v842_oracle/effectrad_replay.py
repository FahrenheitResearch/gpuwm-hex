"""Direct pristine radius-helper calls exercise prescribed-count limits.

MPAS's classic forecast supplies large prescribed counts. Direct calls also
test valid low/high counts and mixed vertical masks without changing defaults.
"""
import argparse
import ctypes as C
import itertools
import json
from pathlib import Path
import numpy as np
from hexcore.mpasphys import thompson_v842 as port
if __package__:
    from .replay import compare,pointer,engine_math_source
else:
    from replay import compare,pointer,engine_math_source


def effectrad_source():
    return r'''
extern "C" __global__ void mpas_thompson_effectrad_cases(int nz,int ncol,const float* a,const float* prescribed,float* out) {
  int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=ncol)return;
  ThompsonColumn m;m.nt_c=prescribed[c];
  float f[8][THOMPSON_COLUMN_KMAX],o[3][THOMPSON_COLUMN_KMAX];
  for(int k=0;k<nz;k++) {
    for(int n=0;n<8;n++)f[n][k]=a[(n*nz+k)*ncol+c];
    for(int n=0;n<3;n++)o[n][k]=out[(n*nz+k)*ncol+c];
  }
  m.calc_effectrad(f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7],o[0],o[1],o[2],1,nz);
  for(int k=0;k<nz;k++)for(int n=0;n<3;n++)out[(n*nz+k)*ncol+c]=o[n][k];
}
'''


def replay(directory,receipt,backend='host',engine=None,nonfinite=False):
    directory=directory.resolve()
    cases=list(itertools.product((1e-7,20.,100e6,2e10),('clear','cloud','ice','snow','all'),(0.,1e-6,1e5)))
    nz=4;ncol=len(cases);a=np.zeros((8,nz,ncol),np.float32)
    a[0]=250.;a[1]=80000.;a[2]=.001;a[4]=1e8
    prescribed=np.array([row[0] for row in cases],np.float32)
    for c,(_,species,number) in enumerate(cases):
        # Empty levels between particles test native CYCLE/writeback masks.
        for n,label in ((3,'cloud'),(5,'ice'),(7,'snow')):
            if species in (label,'all'):a[n,1::2,c]=1e-4
        a[6,:,c]=number
        # Keep ice mass present at one level while its number is at the native
        # lower bound; another populated level enables has_qi. This reaches
        # the second OR edge of calc_effectrad's CYCLE at MPAS F:4866.
        if species in ('ice','all') and number==1e5:a[6,1,c]=0.
    if nonfinite:
        extra=[]
        for bits in (0x7fc54321,0xffc54321):
            for field in (3,5,6,7):
                column=a[:,:,cases.index((100e6,'all',1e5))].copy()
                column[field,1]=np.array([bits],np.uint32).view(np.float32)[0]
                extra.append(column)
        a=np.concatenate((a,np.stack(extra,axis=2)),axis=2)
        prescribed=np.concatenate((prescribed,np.full(len(extra),100e6,np.float32)))
        ncol=a.shape[2]
    ref=np.full((3,nz,ncol),17e-6,np.float32);actual=ref.copy()
    native=C.CDLL(str(directory.resolve()/'liboracle.so'))
    native.oracle_effectrad.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p]
    # Native initialization is required for ice/snow immutable coefficients.
    import os
    os.chdir(directory.resolve());native.init_oracle()
    native.oracle_effectrad(nz,ncol,pointer(a),pointer(prescribed),pointer(ref))
    if backend=='host':
        host=C.CDLL(str(directory.resolve()/'host_port.so'))
        host.effectrad_host.argtypes=native.oracle_effectrad.argtypes
        host.effectrad_host(nz,ncol,pointer(a),pointer(prescribed),pointer(actual))
    else:
        import cupy as cp
        source=port.cuda_source(device_math_source=engine_math_source(engine.resolve(),backend=='engine-promoted-cuda',backend=='engine-native-log10-cuda'))+effectrad_source()
        kernel=port.compile_module(source).get_function('mpas_thompson_effectrad_cases')
        output=cp.asarray(actual)
        kernel(((ncol+31)//32,),(32,),(np.int32(nz),np.int32(ncol),cp.asarray(a),cp.asarray(prescribed),output))
        actual=cp.asnumpy(output)
    result=compare(ref,actual,('re_cloud','re_ice','re_snow'))
    receipt.write_text(json.dumps(dict(acceptance=False,backend=backend,columns=ncol,levels=nz,nonfinite_stress=nonfinite,arrays=result),indent=2)+'\n')
    print(result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','engine-cuda','engine-promoted-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    p.add_argument('--nonfinite',action='store_true')
    a=p.parse_args();replay(a.directory.resolve(),a.receipt.resolve(),a.backend,a.engine,a.nonfinite)
