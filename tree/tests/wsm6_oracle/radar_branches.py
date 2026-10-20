"""Alternate Rayleigh mixing configurations through the public native routine."""
from pathlib import Path
import ctypes as ct
import itertools
import json
import numpy as np


def compare(root: Path, receipt: Path, backend='host-witness', device=None):
    funcs = []
    for lib, name in [('liboracle.so','radar_branch_oracle'),('libhost.so','radar_branch_host')]:
        if backend=='hex-cuda' and lib=='libhost.so':
            if device is None:raise ValueError('hex-cuda requires the shared-math device exports')
            f=getattr(device,name)
        else:f = getattr(ct.CDLL(str(root.resolve()/lib)),name)
        f.argtypes = [ct.POINTER(ct.c_double),ct.POINTER(ct.c_int),ct.POINTER(ct.c_double)]
        funcs.append(f)
    rows = []
    configurations = itertools.product((1,2),(3,4,5,2),(6,7,2),(3,4,5,8,2),(3,4,5,9,10,11,2),(6,7,2))
    for case, config in enumerate(configurations):
        # Vary density, water fraction, coating and tiny-particle exit paths.
        a = np.array([1e-6, .08, 1./3., (.01,.5,.99)[case%3], (0.,.1,.9,1.)[case%4], .1],np.float64)
        if case%19 == 0:a[0]=1e-40
        opts=np.array(config,np.int32);out=[]
        for f in funcs:
            value=ct.c_double()
            f(a.ctypes.data_as(ct.POINTER(ct.c_double)),opts.ctypes.data_as(ct.POINTER(ct.c_int)),ct.byref(value))
            out.append(np.array(value.value,np.float64).view(np.uint64).item())
        if out[0]!=out[1]:
            count=int((out[0]&0xffffffff)!=(out[1]&0xffffffff))+int((out[0]>>32)!=(out[1]>>32))
            rows.append(dict(case=case,config=config,native=hex(out[0]),host=hex(out[1]),differing_words=count))
    total=sum(row['differing_words'] for row in rows)
    receipt.write_text(json.dumps(dict(scope='public Rayleigh routine alternate configurations; not forecast acceptance',backend=backend,process_count=1,cases=case+1,compared_words=2*(case+1),differing_words=total,differences=rows),indent=2)+'\n')
    print('Rayleigh configurations',case+1,'differing words',total)
    return total

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();raise SystemExit(bool(compare(a.root,a.receipt)))
