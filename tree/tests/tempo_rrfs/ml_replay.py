"""Diagnose native ML transform/layers through public pristine routines."""
import ctypes as C,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
lib=C.CDLL(str(ROOT/'scratch/tempo-build/libtempo_oracle.so'));host=C.CDLL(str(ROOT/'scratch/tempo-build/libtempo_host.so'))
lib.oracle_ml.argtypes=host.cpp_ml.argtypes=[C.c_int,C.c_void_p,C.c_void_p]
nz=4096
rng=np.random.default_rng(2704)
a=np.empty((7,nz),np.float32)
a[:4]=rng.uniform(0,0.003,(4,nz));a[4]=rng.uniform(20000,99000,nz);a[5]=rng.uniform(220,300,nz);a[6]=rng.uniform(-2,2,nz)
x=np.zeros((33,nz),np.float32);y=x.copy()
# Initialize ML SAVE data through pristine tempo_init, using the same debug tables.
import os
os.chdir(ROOT/'scratch/tables');flags=np.array([1,1,1,0],np.int32)
lib.oracle_init(C.c_void_p(flags.ctypes.data))
# Exercise native MATMUL at the actual per-column extent, rather than a
# 4096-level matrix that takes a different GNU library-dispatch branch.
for first in range(0,nz,24):
 n=min(24,nz-first);inputs=np.ascontiguousarray(a[:,first:first+n]);native=np.zeros((33,n),np.float32);got=native.copy()
 lib.oracle_ml(n,C.c_void_p(inputs.ctypes.data),C.c_void_p(native.ctypes.data));host.cpp_ml(n,C.c_void_p(inputs.ctypes.data),C.c_void_p(got.ctypes.data))
 x[:,first:first+n]=native;y[:,first:first+n]=got
rows=[]
for stage in range(33):
 mask=x[stage].view(np.uint32)!=y[stage].view(np.uint32);idx=np.flatnonzero(mask)
 rows.append({'stage':stage,'words':int(mask.sum()),'examples':[{'column':int(i),'native':float(x[stage,i]),'host':float(y[stage,i])} for i in idx[:3]]})
(ROOT/'n1out/ml-stages.json').write_text(json.dumps({'forecast_proven':False,'rows':rows},indent=2)+'\n')
print(json.dumps([r for r in rows if r['words']]))
