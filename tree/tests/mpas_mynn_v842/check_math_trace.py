"""Replay Rust-borrowed native math arguments through the shared CUDA resource."""
import argparse
import ctypes as ct
from pathlib import Path
import cupy as cp
import numpy as np
from hexcore.mpasphys.bl_mynn_v842 import _compile_device_module


def replay(a):
    lib=ct.CDLL(str(a.bridge)); ptr=ct.POINTER(ct.c_uint64)
    for name,types,result in [('open',[ct.c_char_p],ct.c_void_p),('data',[ct.c_void_p],ptr),('len',[ct.c_void_p],ct.c_size_t),('grade',[ct.c_void_p,ptr,ct.c_char_p],ct.c_size_t),('close',[ct.c_void_p],None)]:
        f=getattr(lib,'p12_trace_'+name); f.argtypes,f.restype=types,result
    handle=lib.p12_trace_open(str(a.trace).encode())
    try:
        n=lib.p12_trace_len(handle)
        host=np.ctypeslib.as_array(lib.p12_trace_data(handle),shape=(5*n,))
        records=cp.asarray(host); output=cp.empty(n,dtype=cp.uint64)
        source=a.shared_math.read_text()+r'''
extern "C" __global__ void trace_math(const unsigned long long* p,unsigned long long* out,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 p+=5*i; int op=p[1];
 if(op<100) {
  float a=__uint_as_float((unsigned)p[2]),b=__uint_as_float((unsigned)p[3]),r;
  switch(op) {case 1:r=mynn_pow(a,b);break;case 2:r=mynn_exp(a);break;case 3:r=mynn_log(a);break;case 4:r=mynn_log10(a);break;case 5:r=mynn_atan(a);break;case 6:r=mynn_tanh(a);break;case 7:r=mynn_erf(a);break;default:r=__uint_as_float(0xffffffff);}
  out[i]=__float_as_uint(r);
 } else {
  double a=__longlong_as_double(p[2]),b=__longlong_as_double(p[3]),r;
  switch(op) {case 101:r=mynn_pow(a,b);break;case 102:r=mynn_exp(a);break;case 103:r=mynn_log(a);break;default:r=__longlong_as_double(0xffffffffffffffffULL);}
  out[i]=__double_as_longlong(r);
 }
}
'''
        module=_compile_device_module(source)
        module.get_function('trace_math')(((n+127)//128,),(128,),(records,output,np.int32(n)))
        actual=cp.asnumpy(output)
        count=lib.p12_trace_grade(handle,actual.ctypes.data_as(ptr),str(a.receipt).encode())
        print(f'records={n} differing_words={count}')
        return 2*bool(count)
    finally: lib.p12_trace_close(handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','bridge','shared-math','receipt'):
        p.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    raise SystemExit(replay(p.parse_args()))
