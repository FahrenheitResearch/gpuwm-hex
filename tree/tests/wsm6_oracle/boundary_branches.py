"""Strict WSM6 initialization/radii/finalization boundary words.

Uses mp_radar.F90:72-190, mp_wsm6_effectRad.F90:22-193 and
mp_wsm6.F90:73-211 through the actual production device routine views.
"""
import ctypes as ct
import json
from pathlib import Path
import numpy as np


def compare(root,receipt,constants,backend='host-witness',device=None):
    fp=ct.POINTER(ct.c_float);ip=ct.POINTER(ct.c_int)
    native=ct.CDLL(str(Path(root).resolve()/'liboracle.so'))
    if backend=='hex-cuda':
        if device is None:raise ValueError('hex-cuda requires shared-math device exports')
        candidate=device
    else:candidate=ct.CDLL(str(Path(root).resolve()/'libhost.so'))
    rows=[];words=0
    outputs=[]
    for library,name in ((native,'radar_constants_oracle'),(candidate,'radar_constants_host')):
        f=getattr(library,name);f.argtypes=[ct.POINTER(ct.c_double)]
        answer=np.zeros(32,np.float64);f(answer.ctypes.data_as(ct.POINTER(ct.c_double)))
        # Count both binary32 words of each binary64 constant consistently
        # with the rest of the complete CUDA replay matrix.
        outputs.append(answer.view(np.uint32))
    rows.append(dict(kind='initialization',different_words=int(np.count_nonzero(outputs[0]!=outputs[1]))));words+=64
    for enabled in (0,1):
        outputs=[]
        for library,name in ((native,'radii_boundary_oracle'),(candidate,'radii_boundary_host')):
            f=getattr(library,name);f.argtypes=[ct.c_int,ct.c_int,fp,fp,fp,ip]
            v=np.zeros((10,8),np.float32);v[0]=[250,280]*4
            v[7]=[1e-10,1,1e-10,1,1,1,1,1];v[2:7]=.2;v[2:7,4:]=0
            radius=np.full((3,8),.00007,np.float32);status=np.full(3,42,np.int32)
            f(8,enabled,constants.ctypes.data_as(fp),v.ctypes.data_as(fp),radius.ctypes.data_as(fp),status.ctypes.data_as(ip))
            outputs.append((radius.view(np.uint32),status.view(np.uint32)))
        count=sum(int(np.count_nonzero(a!=b)) for a,b in zip(*outputs));words+=27
        rows.append(dict(kind='radii-and-finalization',enabled=enabled,different_words=count))
    total=sum(row['different_words'] for row in rows)
    Path(receipt).write_text(json.dumps(dict(scope=__doc__.strip(),backend=backend,process_count=1,cases=len(rows),compared_words=words,total_differing_words=total,rows=rows),indent=2)+'\n')
    return total
