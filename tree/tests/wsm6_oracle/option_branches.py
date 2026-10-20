"""Strict whole-column replay with independent optional diagnostic arguments.

MPAS mp_wsm6.F90:472-473,751-769 and module_mp_wsm6.F:214-232.
The real MPAS driver supplies all four optional surface arrays. Native line767
reads graupelncv whenever snowncv is present and precipitation falls, so those
four unsafe combinations are tested only with zero precipitation. Nothing in
the production port is guarded, repaired or excluded by this fixture.
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
    c=constants
    libs={name:ct.CDLL(str(root/name)) for name in
          ('liboracle.so','libwrapper.so','libhost.so')}
    if backend == 'hex-cuda':
        if device is None:raise ValueError('hex-cuda requires the shared-math device exports')
        libs['libhost.so']=device
    for dry,mask,hail,dt in itertools.product((True,False),range(16),(0,1,-1),(30,300)):
        if not dry and mask&2 and not mask&8:
            continue
        outputs=[]
        for name,fn in (('liboracle.so','optional_surface_oracle'),
                        ('libhost.so','optional_surface_host')):
            f=getattr(libs[name],fn)
            f.argtypes=[ct.c_int,ct.c_float,ct.c_int,ct.c_int]+[fp]*4
            v=np.zeros((10,8),np.float32)
            v[0]=300 if dry else np.linspace(290,250,8,dtype=np.float32)
            if not dry:v[1]=.005;v[2:7]=.001
            v[7]=1.;v[8]=70000.;v[9]=500.
            s=np.full(7,1.25,np.float32);prod=np.full((2,8),42,np.float32)
            f(8,dt,hail,mask,*[a.ctypes.data_as(fp) for a in (c,v,s,prod)])
            outputs.append((v,s,prod))
        diff={label:int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
              for label,a,b in zip(('volume','surface','production'),*outputs)}
        words+=sum(a.size for a in outputs[0])
        rows.append(dict(kind='surface',dry=dry,mask=mask,hail=hail,dt=dt,differences=diff))
    for flags,hail,dt in itertools.product(itertools.product((0,1,2),repeat=3),
                                         (0,1,-1),(30,300)):
        outputs=[]
        for name,fn in (('libwrapper.so','wsm6_wrapper_flags_oracle'),
                        ('libhost.so','wsm6_wrapper_flags_host')):
            f=getattr(libs[name],fn)
            f.argtypes=[ct.c_int,ct.c_float,ct.c_int]+[fp]*7+[ct.c_int]*3
            pi=np.linspace(.96,.6,8,dtype=np.float32)
            v=np.zeros((10,8),np.float32)
            v[0]=np.linspace(290,250,8,dtype=np.float32)/pi
            v[1]=.005;v[2:7]=.001;v[7]=1.;v[8]=70000.;v[9]=500.
            s=np.full(7,1.25,np.float32);r=np.full((3,8),1.7e-5,np.float32)
            dbz=np.full(8,-35,np.float32);prod=np.full((2,8),.0027,np.float32)
            f(8,dt,hail,*[a.ctypes.data_as(fp) for a in (c,v,s,r,dbz,prod,pi)],*flags)
            outputs.append((v,s,r,dbz,prod))
        diff={label:int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
              for label,a,b in zip(('volume','surface','radii','dbz','production'),*outputs)}
        words+=sum(a.size for a in outputs[0])
        rows.append(dict(kind='radii_flags',flags=list(flags),hail=hail,dt=dt,differences=diff))
    total=sum(sum(row['differences'].values()) for row in rows)
    receipt.write_text(json.dumps(dict(scope='strict standalone whole-column optional diagnostic replay; not forecast acceptance',
        process_count=1,backend=backend,cases=len(rows),compared_words=words,
        native_libraries={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in libs},
        total_differing_words=total,rows=rows,
        native_unsafe_wet_masks=[2,3,6,7],
        unsafe_reason='mp_wsm6.F90:767 dereferences absent graupelncv when snowncv is present and precipitation falls; real driver supplies both'),
        indent=2)+'\n')
    print(len(rows),'optional diagnostic columns;',words,'words;',total,'different')
    return total


if __name__=='__main__':
    from replay import CONSTANTS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('oracle',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();raise SystemExit(bool(compare(a.oracle,a.receipt,CONSTANTS)))
