#!/usr/bin/env python3
"""Bitwise sampled proof of every shared binary32 MYNN binding."""
import argparse,ctypes,hashlib,json,platform
from pathlib import Path
import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import nvrtc
from hexcore.mynn_glibc_math import mynn_glibc_source
p=argparse.ArgumentParser();p.add_argument('--oracle',required=True);p.add_argument('--out',required=True);a=p.parse_args()
s=mynn_glibc_source()+r'''
extern "C" __global__ void run_math(int op,unsigned int n,const unsigned int* input,const unsigned int* other,unsigned int* out) {
unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
if(i>=n)return;float x=__uint_as_float(input[i]),y=__uint_as_float(other[i]),z;
switch(op){case 0:z=mynn_exp(x);break;case 1:z=mynn_log(x);break;case 2:z=mynn_log10(x);break;case 3:z=mynn_atan(x);break;case 4:z=mynn_tanh(x);break;case 5:z=mynn_erf(x);break;default:z=mynn_pow(x,y);break;}
out[i]=__float_as_uint(z);
}
'''
opts=('--std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--gpu-architecture=sm_'+cp.cuda.Device().compute_capability)
program=nvrtc.createProgram(s,'mynn_math.cu',(),())
try:
 try:nvrtc.compileProgram(program,opts)
 except Exception as e:raise RuntimeError(nvrtc.getProgramLog(program)) from e
 cubin=nvrtc.getCUBIN(program)
finally:nvrtc.destroyProgram(program)
m=cp.cuda.function.Module();m.load(cubin);k=m.get_function('run_math')
lib=ctypes.CDLL(a.oracle);lib.oracle_glibc_version.restype=ctypes.c_char_p
if lib.oracle_glibc_version()!=b'2.43':raise RuntimeError('pinned glibc version mismatch')
lib.oracle_math.argtypes=(ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p)
rng=np.random.default_rng(942418)
words=np.concatenate([rng.integers(0,2**32,2**20,dtype=np.uint32),np.linspace(-100,100,2**18,dtype=np.float32).view(np.uint32),np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xff812345,0x7f812345,0x3f800000,0xbf800000],dtype=np.uint32)])
y=rng.choice(np.array([.25,.5,.33333334,1.5,2.,3.,-2.],np.float32),len(words)).view(np.uint32)
dx,dy=cp.asarray(words),cp.asarray(y);out=cp.empty_like(dx);expected=np.empty_like(words);results=[]
for op,name in enumerate(['exp','log','log10','atan','tanh','erf','pow']):
 k(((len(words)+255)//256,),(256,),(np.int32(op),np.uint32(len(words)),dx,dy,out))
 lib.oracle_math(op,len(words),words.ctypes.data,y.ctypes.data,expected.ctypes.data)
 got=out.get();bad=np.flatnonzero(got!=expected);results.append({'binding':name,'words':len(words),'differing_words':len(bad),'samples':[{'input':hex(int(words[i])),'other':hex(int(y[i])),'native':hex(int(expected[i])),'cuda':hex(int(got[i]))} for i in bad[:16]]})
r={'schema':'mynn-shared-math-sample-v1','host':platform.node(),'glibc':'2.43','source_sha256':hashlib.sha256(s.encode()).hexdigest(),'results':results,'cuda_options':opts,'bitwise':all(v['differing_words']==0 for v in results)}
Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
raise SystemExit(0 if r['bitwise'] else 1)
