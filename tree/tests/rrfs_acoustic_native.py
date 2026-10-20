"""C ABI adapter to complete, extracted fork acoustic work routines.

Source: RRFSx/MPAS-Model src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2366,2646-2984
at 1d6e370e. Preparation changes the interface and drops timing macros only.
"""
from pathlib import Path
import ctypes as ct
import re
import numpy as np

class NativeAcoustic:
    def __init__(self, path, nlev, nscalars, maxedges):
        self.library = ct.CDLL(str(Path(path).resolve()))
        self.library.set_dimensions(ct.c_int(nlev),ct.c_int(nscalars),ct.c_int(maxedges))
        self.source = Path(path).with_name('oracle.f90').read_text()

    def call(self, name, fields):
        signature = re.search(r'\bsubroutine\s+'+name+r'\((.*?)\)\s*bind',self.source,re.S|re.I)[1]
        names = re.sub(r'[&\s]','',signature).split(',')
        buffers=[]; args=[]
        for key in names:
            value=fields[key.lower()]
            if isinstance(value,np.ndarray):
                if not value.flags.f_contiguous: raise ValueError(key+' must be Fortran contiguous')
                args.append(value.ctypes.data_as(ct.c_void_p))
            else:
                scalar = ct.c_int(value) if isinstance(value,int) else ct.c_float(value)
                buffers.append(scalar); args.append(ct.byref(scalar))
        routine=getattr(self.library,name)
        routine.restype=None
        routine(*args)
