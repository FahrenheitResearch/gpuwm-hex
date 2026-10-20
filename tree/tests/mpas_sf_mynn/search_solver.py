"""Find reproducible strict-native witnesses for the private secant fallback.

MPAS physics_mmm/sf_mynn.F90:1791-1844; this search changes no native source.
Candidates keep roughness/height ratios positive and initial bounds in the
Richardson-number quadrant, so the native result has been assigned.
"""
import ctypes
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('oracle'); n=65536; rng=np.random.default_rng(4301)
    meta=json.loads((root/'leaf-meta.json').read_text())
    indices={m['name']:i for i,m in enumerate(meta)}
    lib=ctypes.CDLL(str((root/'leaf-native.so').resolve()))
    pointer=ctypes.POINTER(ctypes.c_float)
    lib.native_leaves.argtypes=[pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int]
    lib.native_leaf_init.argtypes=[ctypes.c_int]
    witnesses=[]
    for psi in (0,1):
        lib.native_leaf_init(psi)
        for batch in range(16):
            x=np.zeros((11,n),np.float32)
            x[0]=rng.uniform(-5,5,n)
            x[1]=rng.uniform(10,100,n)
            x[2]=10**rng.uniform(-5,-.5,n)
            x[3]=10**rng.uniform(-7,-3,n)
            x[4]=np.copysign(10**rng.uniform(-2,4,n),x[0])
            print('psi',psi,'batch',batch,flush=True)
            result=x.copy()
            lib.native_leaves(result.ctypes.data_as(pointer),indices['zolri'],psi,n)
            approx=np.zeros_like(x)
            approx[1]=x[0]; approx[2]=x[1]/x[2]; approx[3]=x[2]/x[3]
            lib.native_leaves(approx.ctypes.data_as(pointer),indices['li_etal_2010'],psi,n)
            matches=np.flatnonzero((result[10].view(np.uint32)==approx[0].view(np.uint32)) & np.isfinite(result[10]))
            if matches.size:
                j=int(matches[0]);witnesses.append({'psi':psi,'batch':batch,
                    'input':{a:float(x[i,j]) for i,a in enumerate(meta[indices['zolri']]['float_args'])},
                    'native_result_word':int(result[10,j].view(np.uint32))})
                break
    Path('n1out/secant-witnesses.json').write_text(json.dumps(witnesses,indent=2)+'\n')
    print(witnesses)


if __name__=='__main__':main()
