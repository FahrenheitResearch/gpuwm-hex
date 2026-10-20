"""Search host-debug candidates, then verify the finite witness in pristine MPAS."""
import ctypes
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--kind',type=int,choices=(1,2),default=1)
kind=parser.parse_args().kind
pointer=ctypes.POINTER(ctypes.c_float)
root=Path('oracle');lib=ctypes.CDLL(str((root/'trace.so').resolve()))
lib.trace_secant.argtypes=[pointer,ctypes.c_int,ctypes.c_int]
native=ctypes.CDLL(str((root/'leaf-native.so').resolve()))
native.native_leaves.argtypes=[pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int]
native.native_leaf_init.argtypes=[ctypes.c_int]
rng=np.random.default_rng(4301);n=65536;rows=[]
for psi in (0,1):
    for batch in range(32):
        data=np.zeros((7,n),np.float32)
        data[0]=rng.uniform(-10,10,n)
        data[1]=10**rng.uniform(-2,3,n)
        data[2]=data[1]*10**rng.uniform(-8,-.01,n)
        data[3]=data[1]*10**rng.uniform(-8,-.01,n)
        data[4]=np.copysign(10**rng.uniform(-1,5,n),data[0])
        lib.trace_secant(data.ctypes.data_as(pointer),psi,n)
        matches=np.flatnonzero(data[6]==kind)
        print(psi,batch,len(matches),flush=True)
        if matches.size:
            j=int(matches[0]); x=np.zeros((11,1),np.float32);x[:5,0]=data[:5,j]
            native.native_leaf_init(psi)
            native.native_leaves(x.ctypes.data_as(pointer),18,psi,1)
            assert x[10,0].view(np.uint32)==data[5,j].view(np.uint32)
            cross=[]
            for other in (0,1):
                candidate=np.zeros((7,1),np.float32);candidate[:5,0]=data[:5,j]
                lib.trace_secant(candidate.ctypes.data_as(pointer),other,1)
                assert np.isfinite(candidate[5,0]), 'candidate must be defined in both table modes'
                native.native_leaf_init(other);x[:5,0]=data[:5,j]
                native.native_leaves(x.ctypes.data_as(pointer),18,other,1)
                assert x[10,0].view(np.uint32)==candidate[5,0].view(np.uint32)
                cross.append({'psi':other,'native_word':int(x[10,0].view(np.uint32))})
            rows.append({'psi':psi,'input':dict(zip(('ri','za','z0','zt','zol1'),map(float,data[:5,j]))),'native_word':int(x[10,0].view(np.uint32)),'cross_mode':cross})
            break
Path(f'n1out/secant-traced-witnesses-{kind}.json').write_text(json.dumps(rows,indent=2)+'\n')
