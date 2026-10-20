"""Expected failure: pristine native single-level sedimentation reads ww(2)."""
from pathlib import Path
import ctypes as ct
import sys
import numpy as np
from replay import CONSTANTS

f=ct.CDLL(str(Path(sys.argv[1]).resolve()/'liboracle.so')).wsm6_oracle
fp=ct.POINTER(ct.c_float);f.argtypes=[ct.c_int,ct.c_float,ct.c_int]+[fp]*6
v=np.array([[280],[.015],[.001],[.001],[.001],[.001],[.001],[1.1],[90000],[500]],np.float32)
s=np.zeros(7,np.float32);r=np.zeros((3,1),np.float32);d=np.zeros(1,np.float32);p=np.zeros((2,1),np.float32)
f(1,30,0,*[a.ctypes.data_as(fp) for a in (CONSTANTS,v,s,r,d,p)])
raise AssertionError('bounds-checking native must reject km=1')
