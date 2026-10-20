"""Replay private paths omitted by the native WSM6 public driver.

Test-only visibility instrumentation is authenticated in build-manifest.json.
This is strict host execution of production CUDA bodies, not device identity.
"""
import argparse
import ctypes as ct
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def compare(root, receipt, constants, backend='host-witness', device=None):
    root=root.resolve();fp=ct.POINTER(ct.c_float);rows=[];words=0
    native=ct.CDLL(str(root/'libprivate.so'));host=ct.CDLL(str(root/'libhost.so'))
    if backend == 'hex-cuda':
        if device is None:raise ValueError('hex-cuda requires the shared-math device exports')
        host=device
    gamma=[native.gamma_oracle,host.gamma_host]
    for f in gamma:f.argtypes=[ct.c_float];f.restype=ct.c_float
    for x in (1.,.25,.5,1.5,2.,3.,4.8,6.):
        out=[np.array([f(x)],np.float32).view(np.uint32) for f in gamma]
        diff=int(np.count_nonzero(out[0]!=out[1]));words+=1
        rows.append(dict(kind='gamma',x=x,differing_words=diff))
    functions=[native.sedimentation_oracle,host.sedimentation_host]
    for f in functions:f.argtypes=[ct.c_int]*3+[ct.c_float]+[fp]*3
    rng=np.random.default_rng(8675309)
    for category,nz,iterations,dt,regime in itertools.product((1,2),(2,3,8,55),(0,1,2,3),(1.5,30,300),range(6)):
        initial=np.zeros((7,nz),np.float32)
        initial[0]=rng.uniform(80,900,nz)
        initial[1]=rng.uniform(.3,1.28,nz)
        initial[2]=np.sqrt(np.float32(1.28)/initial[1])
        initial[3]=rng.uniform(210,290,nz)
        initial[4]=rng.uniform(0,12,nz)
        initial[5:]=rng.uniform(1e-7,.005,(2,nz))
        if regime==0:initial[5:]=0
        elif regime==1:initial[5]=0
        elif regime==2:initial[6]=0
        elif regime==3:initial[5:]*=np.float32(1e-8)
        elif regime==4:initial[4,1:]=0
        else:initial[5:,::2]=0
        outputs=[]
        for f in functions:
            a=initial.copy();precip=np.full(2,42,np.float32)
            f(nz,iterations,category,dt,*[arr.ctypes.data_as(fp) for arr in (constants,a,precip)])
            outputs.append((a,precip))
        diff=sum(int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for a,b in zip(*outputs))
        words+=sum(a.size for a in outputs[0])
        rows.append(dict(kind='sedimentation',category=category,nz=nz,iterations=iterations,dt=dt,regime=regime,differing_words=diff))
    maxwell=[native.maxwell_oracle,host.maxwell_host]
    for f in maxwell:
        f.argtypes=[ct.POINTER(ct.c_double),ct.c_int,ct.POINTER(ct.c_double),ct.POINTER(ct.c_int)]
    for volumes,inclusion in itertools.product(((.3,.2,.5),(0.,0.,1.),(.3,.2,.6),(.3,.2,.499)),(0,1,2)):
        v=np.array(volumes,np.float64);outputs=[]
        for f in maxwell:
            answer=np.full(2,42,np.float64);error=np.full(1,42,np.int32)
            f(v.ctypes.data_as(ct.POINTER(ct.c_double)),inclusion,
              answer.ctypes.data_as(ct.POINTER(ct.c_double)),error.ctypes.data_as(ct.POINTER(ct.c_int)))
            outputs.append((answer.view(np.uint32),error.view(np.uint32)))
        diff=sum(int(np.count_nonzero(a!=b)) for a,b in zip(*outputs))
        words+=5
        row=dict(kind='maxwell',volumes=list(volumes),inclusion=inclusion,differing_words=diff,
                 output_scope='numeric complex result and error flag; private diagnostic string is not asserted')
        if diff:
            row['native_words']=[hex(int(x)) for a in outputs[0] for x in a]
            row['candidate_words']=[hex(int(x)) for a in outputs[1] for x in a]
        rows.append(row)
    total=sum(row['differing_words'] for row in rows)
    receipt.write_text(json.dumps(dict(scope='strict test-interface native private paths versus actual CUDA bodies; not forecast acceptance',
        backend=backend,process_count=1,cases=len(rows),compared_words=words,total_differing_words=total,rows=rows,
        manifest=json.loads((root/'build-manifest.json').read_text()),
        library_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('libprivate.so','libhost.so')}),indent=2)+'\n')
    print(len(rows),'private-path cases;',words,'words;',total,'different')
    return total


if __name__=='__main__':
    from replay import CONSTANTS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('oracle',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();raise SystemExit(bool(compare(a.oracle,a.receipt,CONSTANTS)))
