"""Build/replay the complete generated arithmetic with host glibc maths.

Development and libm witness only; this does not replace a whole forecast.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np
from hexcore.mpasphys.rrfs import sf_mynnsfc_v600 as s


def generate():
    pre='''#include <cmath>
#include <cstring>
#define __device__
#define __global__
#define __noinline__ __attribute__((noinline))
inline float __fadd_rn(float a,float b) { return a+b; }
inline float __fsub_rn(float a,float b) { return a-b; }
inline float __fmul_rn(float a,float b) { return a*b; }
inline float __fdiv_rn(float a,float b) { return a/b; }
#define SL_LOG logf
#define SL_EXP expf
#define SL_ATAN atanf
#define SL_POW powf
#define SL_SQRT sqrtf
struct Dim { int x=0; } blockIdx,blockDim,threadIdx;
'''
    post='''
extern "C" void host_columns(float *f,const int *o,const int *v,int *e,float threshold,
    unsigned long long present,int n) {
    blockDim.x=1;int error=0;threadIdx.x=0;blockIdx.x=0;mynnsfc_tables(&error);
    for(int i=0;i<n;i++) {threadIdx.x=i;mynnsfc_columns(f,o,v,e,threshold,present,n);}
}
extern "C" void host_tables(float *out) {
    std::memcpy(out,psim_stab,4004);std::memcpy(out+1001,psim_unstab,4004);
    std::memcpy(out+2002,psih_stab,4004);std::memcpy(out+3003,psih_unstab,4004);
}
'''
    Path('oracle/host_witness.cpp').write_text(pre+s.SOURCE+post)


def replay(out,dataset='synthetic'):
    lib=ctypes.CDLL(str((out/'host_witness.so').resolve()));fn=lib.host_columns
    ptr=np.ctypeslib.ndpointer
    fn.argtypes=[ptr(dtype=np.float32,flags='C_CONTIGUOUS'),ptr(dtype=np.int32,flags='C_CONTIGUOUS'),
        ptr(dtype=np.int32,flags='C_CONTIGUOUS'),ptr(dtype=np.int32,flags='C_CONTIGUOUS'),
        ctypes.c_float,ctypes.c_uint64,ctypes.c_int]
    records=json.loads((out/(dataset+'-native.json')).read_text())['cases'];data=np.load(out/(dataset+'.npz'))
    source=data['input'];n=source.shape[1];results=[]
    for k,r in enumerate(records):
        fields=source.copy();errors=np.empty(n,np.int32)
        fn(fields,np.asarray(r['options'],np.int32),data['vegtype'],errors,.02,r['present'],n)
        expected=data[f'fields{k}'];counts={name:int(np.count_nonzero(fields[j].view(np.uint32)!=expected[j].view(np.uint32))) for j,name in enumerate(s.FIELDS)}
        counts={name:v for name,v in counts.items() if v}
        if k==0:np.savez(out/'first-host.npz',fields=fields)
        results.append(dict(label=r['label'],per_array=counts,differing_words=sum(counts.values()),errors_differ=int(np.count_nonzero(errors!=data[f'errors{k}']))))
    tables=np.empty((4,1001),np.float32);lib.host_tables.argtypes=[ptr(dtype=np.float32,flags='C_CONTIGUOUS')];lib.host_tables(tables)
    table_words=int(np.count_nonzero(tables.view(np.uint32)!=np.load(out/'tables.npy').view(np.uint32)))
    result=dict(scope='standalone host-glibc witness; no whole-model proof',cases=results,
        dataset=dataset,
        compared_words=len(results)*source.size,differing_words=sum(r['differing_words'] for r in results),table_differing_words=table_words,
        source_sha256=hashlib.sha256(s.SOURCE.encode()).hexdigest())
    name='host-witness' if dataset=='synthetic' else dataset+'-host-witness'
    (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if (result['differing_words'] or table_words
            or sum(r['errors_differ'] for r in results)):
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['generate','replay']);p.add_argument('--out',type=Path,default=Path('n1out'))
    p.add_argument('--dataset',choices=['synthetic','upstream'],default='synthetic');a=p.parse_args()
    generate() if a.mode=='generate' else replay(a.out,a.dataset)
