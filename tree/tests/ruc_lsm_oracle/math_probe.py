"""GPU subnormal and GNU glibc probes; never a forecast grade/tolerance."""
import ctypes,json
from pathlib import Path
import cupy as cp
import numpy as np
from hexcore.mpasphys.rrfs import ruc_lsm_8c9f1a8 as ruc
xs=[np.nextafter(np.float32(0),np.float32(1)),np.finfo(np.float32).tiny/2,1e-30,1e-8,.1,.5,.9,1.,1.01,2.,3.,273.15,300.,.25,.75,4.]
# Exact first-divergence inputs from the complete native/CUDA RUC trace.
xs.extend(np.array([1057911453,1058252240,1058003829],np.uint32).view(np.float32))
x=np.array(xs,np.float32);y=np.resize(np.array([.5,1.25,1.5,2.,3.,4.,-.25],np.float32),x.size)
libm=ctypes.CDLL('libm.so.6');expected=np.empty((7,x.size),np.float32);expected[0]=x*y
for k,name in enumerate(('powf','logf','expf','tanhf','cosf','log10f'),1):
 f=getattr(libm,name);f.restype=ctypes.c_float;f.argtypes=[ctypes.c_float]*(2 if name=='powf' else 1)
 expected[k]=[f(float(v),float(w)) if name=='powf' else f(float(v)) for v,w in zip(x,y)]
out=cp.empty_like(cp.asarray(expected));ruc._module(0).get_function('ruc_math_probe')((1,),(64,),(cp.asarray(x),cp.asarray(y),out,np.int32(x.size)));actual=out.get()
mask=actual.view(np.uint32)!=expected.view(np.uint32)
# Every word, including NaN payload/sign and subnormal values, is compared.
rows=[]
for k,name in enumerate(('subnormal-multiply','powf','logf','expf','tanhf','cosf','log10f')):
 rows.append(dict(operation=name,differing_words=int(mask[k].sum()),differences=[dict(x=int(x.view(np.uint32)[i]),y=int(y.view(np.uint32)[i]),native=int(expected[k].view(np.uint32)[i]),cuda=int(actual[k].view(np.uint32)[i])) for i in np.flatnonzero(mask[k])]))
Path('receipts/math-probe.json').write_text(json.dumps(dict(source_sha256=ruc.source_digest(),operations=rows),indent=2)+'\n')
print(rows)
raise SystemExit(bool(mask.any()))
