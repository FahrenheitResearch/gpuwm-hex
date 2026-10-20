"""Word-scan the unchanged pinned shared glibc 2.43 log10f routine.

Python prepares/loads authenticated images and bounded buffers. The existing
shared header owns all transcendental arithmetic; compiled glibc is the oracle.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
from replay import engine_math_source, shared_native_log10_source, pointer
from hexcore.mpasphys import thompson_v842 as port

KERNEL = r'''
extern "C" __global__ void native_shared_log10(unsigned base,unsigned count,
    const unsigned* expected,unsigned long long* counters,unsigned* witnesses) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  unsigned word=base+i;
  unsigned actual=__float_as_uint(mpas_log10f(__uint_as_float(word)));
  if(actual!=expected[i]) {
    unsigned long long at=atomicAdd(counters,1ULL);
    if(at<16){witnesses[3*at]=word;witnesses[3*at+1]=expected[i];witnesses[3*at+2]=actual;}
  }
}
'''

HOST = r'''
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <gnu/libc-version.h>
extern "C" void native_log10_words(uint32_t start,uint32_t count,uint32_t* words) {
  static void* libm=dlopen("libm.so.6",RTLD_NOW|RTLD_LOCAL);
  auto native=reinterpret_cast<float(*)(float)>(libm?dlsym(libm,"log10f"):nullptr);
  for(uint32_t i=0;i<count;i++)words[i]=__float_as_uint(native(__uint_as_float(start+i)));
}
int main(int argc,char** argv) {
  uint32_t start=argc>1?strtoul(argv[1],nullptr,0):1;
  uint32_t end=argc>2?strtoul(argv[2],nullptr,0):0x7f800000u;
  static void* libm=dlopen("libm.so.6",RTLD_NOW|RTLD_LOCAL);
  auto native=reinterpret_cast<float(*)(float)>(libm?dlsym(libm,"log10f"):nullptr);
  if(!native||!start||start>=end||end>0x7f800000u) {
    std::fprintf(stderr,"Native scan setup failed: libm=%p symbol=%p start=%u end=%u\n",libm,(void*)native,start,end);
    return 2;
  }
  uint64_t differences=0;
  std::printf("{\"acceptance\":false,\"glibc\":\"%s\",\"start_word\":%u,\"exclusive_end_word\":%u,\"witnesses\":[",gnu_get_libc_version(),start,end);
  for(uint32_t word=start;word<end;word++) {
    float x=__uint_as_float(word),ref=native(x),actual=ubm_log10f(x);
    if(__float_as_uint(actual)!=__float_as_uint(ref)) {
      if(differences<16) {
        if(differences)std::printf(",");
        std::printf("{\"input_word\":%u,\"native_word\":%u,\"actual_word\":%u}",word,__float_as_uint(ref),__float_as_uint(actual));
      }
      differences++;
    }
  }
  std::printf("],\"tested\":%u,\"differing_words\":%llu,\"requested_scan_complete\":true}\n",end-start,(unsigned long long)differences);
  return differences?1:0;
}
'''


def prepare_host(engine, output):
    shim=(engine/'tools/thompson_real_column_parity/cuda_host_shim.h').read_text()
    # CUDA integer intrinsic glue, with no floating-point implementation.
    text=shim+'\nstatic inline int __clz(unsigned x){return x?__builtin_clz(x):32;}\n'
    output.write_text(text+shared_native_log10_source(engine)+HOST)


def compile_probe(engine, output, architecture):
    from cupy.cuda import compiler
    output.mkdir(parents=True,exist_ok=True)
    source=engine_math_source(engine,native_log10=True)+KERNEL
    image,_=compiler.compile_using_nvrtc(source,options=port.COMPILE_OPTIONS,
                                        arch=architecture,cache_in_memory=True)
    (output/'probe.cubin').write_bytes(image)
    (output/'probe.json').write_text(json.dumps(dict(acceptance=False,architecture=architecture,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),options=list(port.COMPILE_OPTIONS),
        cubin_sha256=hashlib.sha256(image).hexdigest()),indent=2)+'\n')


def replay(directory, engine, images, receipt):
    import cupy as cp
    from cupy.cuda import function
    source=engine_math_source(engine,native_log10=True)+KERNEL
    certificate=json.loads((images/'probe.json').read_text());image=(images/'probe.cubin').read_bytes()
    assert certificate['source_sha256']==hashlib.sha256(source.encode()).hexdigest()
    assert certificate['options']==list(port.COMPILE_OPTIONS)
    assert certificate['cubin_sha256']==hashlib.sha256(image).hexdigest()
    assert certificate['architecture']==str(cp.cuda.Device().compute_capability)
    native=C.CDLL(str(directory.resolve()/'native_log10_probe.so'))
    native.native_log10_words.argtypes=[C.c_uint32,C.c_uint32,C.c_void_p]
    module=function.Module();module.load(image);kernel=module.get_function('native_shared_log10')
    counts=cp.asarray(np.zeros(1,np.uint64));witness=cp.asarray(np.zeros((16,3),np.uint32))
    chunk=1<<24;reference=np.empty(chunk,np.uint32);device=cp.empty(chunk,cp.uint32)
    for base in range(1,0x7f800000,chunk):
        n=min(chunk,0x7f800000-base)
        native.native_log10_words(base,n,pointer(reference));device[:n].set(reference[:n])
        kernel(((n+255)//256,),(256,),(np.uint32(base),np.uint32(n),device,counts,witness))
        cp.cuda.get_current_stream().synchronize()
    differences=int(cp.asnumpy(counts)[0]);rows=cp.asnumpy(witness)[:min(differences,16)]
    result=dict(acceptance=False,scope='Every positive finite binary32 input, unchanged pinned shared log10f',
        tested=0x7f800000-1,differing_words=differences,requested_scan_complete=True,
        certificate=certificate,device_math_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        witnesses=[dict(input_word=int(a),native_word=int(b),actual_word=int(c)) for a,b,c in rows])
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    assert differences==0,result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--prepare-host',type=Path);p.add_argument('--compile',type=Path);p.add_argument('--architecture')
    p.add_argument('--directory',type=Path);p.add_argument('--load',type=Path);p.add_argument('--receipt',type=Path)
    a=p.parse_args()
    if a.prepare_host:prepare_host(a.engine.resolve(),a.prepare_host)
    elif a.compile:
        if not a.architecture or not a.architecture.isdecimal():p.error('numeric --architecture required')
        compile_probe(a.engine.resolve(),a.compile,a.architecture)
    else:
        if not a.directory or not a.load or not a.receipt:p.error('--directory --load --receipt required')
        replay(a.directory,a.engine.resolve(),a.load,a.receipt)
