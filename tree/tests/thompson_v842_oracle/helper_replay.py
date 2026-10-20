"""Pristine helper checks, including branches inactive in classic columns.

This complements complete driver replays; it never substitutes for them.
Compile (through the build queue on a shared host) before invoking the GPU replay
(through its GPU queue).
"""
import argparse
import ctypes as C
import itertools
import json
import os
from pathlib import Path
import numpy as np
from hexcore.mpasphys import thompson_v842 as port
if __package__:
    from .replay import pointer,compare,tables,engine_math_source
else:
    from replay import pointer,compare,tables,engine_math_source


def helper_source():
    return r'''
extern "C" __global__ void mpas_thompson_helpers(int mode,int n,const float* a,float* out,const float* activation) {
  int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
  ThompsonColumn m;m.tnccn_act=activation;
  switch(mode) {
    case 1:out[k]=m.delta_p(a[k],a[n+k],a[2*n+k],a[3*n+k],a[4*n+k]);break;
    case 2:out[k]=m.icedemott(a[k],a[n+k],a[2*n+k],a[3*n+k],a[4*n+k],a[5*n+k]);break;
    case 3:out[k]=m.icekoop(a[k],a[n+k],a[2*n+k],a[3*n+k],a[4*n+k]);break;
    case 4:out[k]=m.eff_aero(a[k],a[n+k],a[2*n+k],a[3*n+k],a[4*n+k],(int)a[5*n+k]);break;
    case 5:out[k]=m.activ_ncloud(a[k],a[n+k],a[2*n+k]);break;
  }
}
'''


def inputs():
    cases={}
    cases['delta_p']=[(y,lo,hi,a,b) for lo,hi in ((0.,1.),(-40.,-25.),(1.,5.))
        for y in (lo-1,lo,lo+(hi-lo)*.25,lo+(hi-lo)*.75,hi,hi+1)
        for a,b in ((0.,1.),(1.,0.),(-2.,3.))]
    cases['icedemott']=[(t,.001,.001,.0008,rho,nifa)
        for t,rho,nifa in itertools.product((-80.,-40.,-20.,-1.,0.),(.1,1.,2.),(0.,1.,1e4,1e9))]
    cases['icekoop']=[(t,.001*sat,.001,number,dt)
        for t,sat,number,dt in itertools.product((190.,220.,238.,260.),(0.,.5,1.,1.5,5.),(1e5,1e9),(.001,1.,1000.))]
    cases['eff_aero']=[(d,da,2e-5,1.,250.,ord(species))
        for d,da,species in itertools.product((1e-4,1e-3,3e-3),(4e-8,8e-7,1e-5),'rsgx')]
    cases['activ_ncloud']=[(t,w,number)
        for t,w,number in itertools.product((200.,250.,303.,320.),(0.,.01,.5,100.,101.),(0.,10e6,500e6,1e10,1e11))]
    return {name:np.array([tuple(row)+(0.,)*(6-len(row)) for row in rows],np.float32).T.copy()
            for name,rows in cases.items()}


def replay(directory,receipt,backend='host',engine=None):
    directory=directory.resolve();os.chdir(directory)
    native=C.CDLL(str(directory/'liboracle.so'));native.init_oracle();native.export_aux()
    activation=tables(directory)['tnccn_act']
    native.oracle_helpers.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p]
    if backend=='host':
        host=C.CDLL(str(directory/'host_port.so'))
        host.helpers_host.argtypes=native.oracle_helpers.argtypes+[C.c_void_p]
    else:
        import cupy as cp
        source=port.cuda_source(device_math_source=engine_math_source(engine.resolve(),backend=='engine-promoted-cuda',backend=='engine-native-log10-cuda'))+helper_source()
        module=port.compile_module(source)
        kernel=module.get_function('mpas_thompson_helpers');device_activation=cp.asarray(activation)
    result=[]
    for mode,(name,a) in enumerate(inputs().items(),1):
        n=a.shape[1];ref=np.zeros(n,np.float32);actual=ref.copy()
        native.oracle_helpers(mode,n,pointer(a),pointer(ref))
        if backend=='host':host.helpers_host(mode,n,pointer(a),pointer(actual),pointer(activation))
        else:
            output=cp.asarray(np.zeros(n,np.float32))
            kernel(((n+31)//32,),(32,),(np.int32(mode),np.int32(n),cp.asarray(a),output,device_activation))
            actual=cp.asnumpy(output)
        row=dict(routine=name,**compare(ref[None],actual[None],(name,))[name])
        result.append(row);print(name,row,flush=True)
    receipt.write_text(json.dumps(dict(acceptance=False,backend=backend,cases=result),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','engine-cuda','engine-promoted-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    a=p.parse_args();replay(a.directory,a.receipt.resolve(),a.backend,a.engine)
