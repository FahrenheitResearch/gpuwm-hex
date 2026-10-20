"""Execute the actual complete CUDA transcription on a glibc CPU witness.

Only CUDA launch indices and explicit arithmetic intrinsics are bridged.
exp/log/pow call the host's scalar glibc. No Python parameterization exists.
This diagnostic is not a forecast proof or native backend implementation.
"""
from __future__ import annotations
import ctypes
from functools import lru_cache
import re
from pathlib import Path
import numpy as np

from hexcore.mpasphys.ntiedtke_v842 import DRIVER_CUDA, NATIVE_ARITHMETIC_CUDA, _preserve_native_float_words, _adapt_scheme_source, LEVEL_INPUTS, LEVEL_OUTPUTS, INTERFACE_INPUTS, SURFACE_INPUTS, SURFACE_OUTPUTS, Constants
from gpuwm.core import ntiedtke as nt

PREFIX = r'''
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
using std::min;
using std::max;
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__ __restrict
struct Dim { int x=1, y=1, z=1; };
static Dim blockDim, gridDim, blockIdx, threadIdx;
inline uint32_t __float_as_uint(float x) {union {float f;uint32_t u;} v={x};return v.u;}
inline float __uint_as_float(uint32_t x) {union {uint32_t u;float f;} v={x};return v.f;}
inline float __fsub_rn(float a,float b) {volatile float c=a-b;return c;}
inline float __fadd_rn(float a,float b) { volatile float c=a+b;return c; }
inline float __fmul_rn(float a,float b) { volatile float c=a*b;return c; }
inline float __fdiv_rn(float a,float b) { volatile float c=a/b;return c; }
inline float __fsqrt_rn(float a) {return sqrtf(a);}
inline int atomicAdd(int *p,int a) {int b=*p; *p+=a;return b;}
inline int atomicMax(int *p,int a) {int b=*p; if(a>*p)*p=a;return b;}
inline float gfk_exp(float x) {return expf(x);}
inline float gfk_log(float x) {return logf(x);}
inline float gfk_pow(float x,float y) {return powf(x,y);}
'''


@lru_cache(maxsize=1)
def host_source():
    path = Path(nt.__file__).parent / "kernels/ntiedtke.cu"
    return _preserve_native_float_words(_adapt_scheme_source(path.read_text()) + DRIVER_CUDA)



@lru_cache(maxsize=1)
def signatures():
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", host_source(), flags=re.S)
    out = {}
    for name, params in re.findall(r'extern "C" __global__ void (\w+)\s*\((.*?)\)\s*\{', source, re.S):
        declarations = [p.strip() for p in params.split(",")]
        out[name] = declarations
    return out


def generate(output: Path):
    wrappers = []
    for name, declarations in signatures().items():
        names = [re.findall(r"\w+", p)[-1] for p in declarations]
        block = "expect_tpb" if "expect_tpb" in names else ("128" if "ncol" in names or "size" in names else "1")
        grid = "expect_nblocks" if "expect_nblocks" in names else (f"(({('ncol' if 'ncol' in names else 'size')}+127)/128)" if "ncol" in names or "size" in names else "1")
        wrappers.append(f'''extern "C" void host_{name}({','.join(declarations)}) {{
            blockDim.x={block}; gridDim.x={grid};
            for(blockIdx.x=0;blockIdx.x<gridDim.x;++blockIdx.x)
              for(threadIdx.x=0;threadIdx.x<blockDim.x;++threadIdx.x)
                {name}({','.join(names)});
        }}''')
    output.write_text(PREFIX + NATIVE_ARITHMETIC_CUDA + host_source() + "\n" + "\n".join(wrappers))


class HostWorkspace(nt.NtWorkspace):
    def __init__(self, ncol, nz):
        self.ncol, self.nz = ncol, nz
        self._level, self._surface = {}, {}
        self._bindings, self._stage_args = {}, {}
        aliases = nt.nt_aliased_names()
        for names, dtype in ((nt.NT_LEVEL_F,np.float32), (nt.NT_LEVEL_I,np.int32)):
            for name in names:
                if name not in aliases:
                    self._level[name] = np.zeros((nz+2,ncol),dtype=dtype)
        for names, dtype in ((nt.NT_SURFACE_F,np.float32), (nt.NT_SURFACE_I,np.int32)):
            for name in names:
                if name not in aliases:
                    self._surface[name] = np.zeros(ncol,dtype=dtype)
        self.scr = np.zeros((nt.NT_CUTYPEN_SCR_SLOTS,nz+2,ncol),dtype=np.float32)
        self.scr_i = np.zeros((nt.NT_CUTYPEN_SCR_I_SLOTS,nz+2,ncol),dtype=np.int32)


class HostPipeline(nt.NtPipeline):
    def __init__(self, library, ncol, nz, dt, stepcu, initflag, constants):
        self.w = HostWorkspace(ncol,nz)
        self._base = dict(nt.NT_LEVEL_BASE)
        self.geometry = nt.NtLaunchGeometry(ncol=ncol,nz=nz)
        delt = np.float32(np.float32(dt)*np.float32(stepcu))
        self.scalars = dict(ncol=np.int32(ncol), nz=np.int32(nz), klev=np.int32(nz),
            stepcu=np.int32(stepcu),itimestep=np.int32(initflag),ktopm2=np.int32(2),
            llo3=np.int32(0),dt=np.float32(dt),delt=delt,ztmst=delt,tiedtke_closure=np.int32(0))
        self.scalars.update({k: np.float32(v) for k,v in vars(constants).items()})
        self._signatures = signatures()
        self._library = ctypes.CDLL(str(library.resolve()))
        self._tail_values = dict(llo3_mask=np.zeros(1,dtype=np.int32),
            expect_tpb=np.int32(self.geometry.tpb),expect_nblocks=np.int32(self.geometry.nblocks),
            geom_report=np.full(nt.NT_STAGE_COUNT,-1,dtype=np.int32),
            order_report=np.full(nt.NT_STAGE_COUNT,-1,dtype=np.int32),ticket=np.zeros(1,dtype=np.int32))
        self._ascent_start = np.zeros(1,dtype=np.int32)

    def _call(self, stage, values):
        fn = getattr(self._library, "host_"+stage)
        types, args = [], []
        for declaration, value in zip(self._signatures[stage], values, strict=True):
            if "*" in declaration:
                types.append(ctypes.c_void_p)
                args.append(ctypes.c_void_p(value.ctypes.data))
            elif "float" in declaration:
                types.append(ctypes.c_float)
                args.append(ctypes.c_float(value))
            else:
                types.append(ctypes.c_int)
                args.append(ctypes.c_int(value))
        fn.argtypes, fn.restype = types, None
        fn(*args)

    def run_stage(self, stage):
        values = []
        for declaration in self._signatures[stage]:
            name = re.findall(r"\w+",declaration)[-1]
            if name == "mpas_llo3_start":
                value = self._ascent_start
            elif name in self._tail_values:
                value = self._tail_values[name]
            elif name in self.scalars:
                value = self.scalars[name]
                if name == "dt" and stage in ("ntiedtke_mfub", "ntiedtke_closure"):
                    value = self.scalars["ztmst"]
            elif name in ("scr","scr_i"):
                value = getattr(self.w,name)
            else:
                value = self.w.bind(nt.NT_STAGE_ALIASES.get((stage,name),name),self.base(stage))
            values.append(value)
        self._call(stage,values)

    def reduce_llo3(self):
        names = ("ldcum", "ktype", "klab", "pqen", "pqsen", "pgeo")
        self._call("mpas_ntiedtke_ascent_start", [self.w.bind(n,1) for n in names] +
            [self._ascent_start,self.w.ncol,self.w.nz,self.scalars["grav"]])


class HostTiedtke:
    def __init__(self, library, *, constants=Constants()):
        self.library, self.constants = library, constants

    def __call__(self, fields, *, dt, itimestep, stepcu=1, restart=False, f_qc=True, f_qi=True, previous=None):
        nz,ncol = fields["t3d"].shape
        p = HostPipeline(self.library,ncol,nz,dt,stepcu,0 if restart or itimestep>1 else 1,self.constants)
        for name in LEVEL_INPUTS:
            p.w.bind(name,0)[:nz] = fields[name]
        for name in INTERFACE_INPUTS:
            p.w.bind(name,0)[:nz+1] = fields[name]
        for name in SURFACE_INPUTS:
            p.w.bind(name,1)[:] = fields[name]
        nt._nt_walk(p)
        out = {n: p.w.bind(n,0)[:nz].copy() for n in LEVEL_OUTPUTS}
        out.update({n: p.w.bind(n,1).copy() for n in SURFACE_OUTPUTS})
        out["cu_act_flag"] = np.ones(ncol,dtype=np.bool_)
        for flag,name in ((f_qc,"rqccuten"),(f_qi,"rqicuten")):
            if not flag:
                out[name] = previous[name].copy()
        return out


if __name__ == "__main__":
    import sys
    generate(Path(sys.argv[1]))
