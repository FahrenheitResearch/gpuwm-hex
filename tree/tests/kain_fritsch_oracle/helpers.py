"""Exercise scalar boundaries in the complete translated MPAS module.

MPAS-A v8.4.2 module_cu_kfeta.F:2373-2762. These tests supplement the
complete CPS/driver comparisons; they never replace those comparisons.
"""
from pathlib import Path
import ctypes as C
import itertools,json,argparse,os
import numpy as np
if __package__:
    from .support import PARAMS,load_oracles,word_diff
else:
    from support import PARAMS,load_oracles,word_diff
from hexcore.mpasphys import kain_fritsch_v842 as kf

# Test ABI only: each call dispatches directly to the full module's routine.
HELPER_SOURCE=r'''
#ifdef KF_HOST_TEST
extern "C" void kf_host_helper(int kind,float *x,float *table) {
#else
extern "C" __global__ void kf_helper_kernel(int kind,float *x,float *table) {
#endif
    Tables tb(table); Arena arena{nullptr,0,1};
    if (kind==0) condload(tb,arena,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11]);
    if (kind==1) prof5(tb,arena,x[0],x[1],x[2]);
    if (kind==2) envirtht(tb,arena,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]);
    if (kind==3) dtfrznew(tb,arena,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
    if (kind==4) tpmix2(tb,arena,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
    if (kind==5) tpmix2dd(tb,arena,x[0],x[1],x[2],x[3],1,1);
}
'''


def helper_cases(table):
    for wtw,bot,q,rate in itertools.product([1,10,100],[0,-1,-10],[0,.001],[0,.01]):
        yield 'condload',0,np.array([q,q,wtw,250,bot,0,rate,q,q,0,0,PARAMS[9]],np.float32)
    for eq in [0,.01,.25,.499999,.5,.500001,.75,.99,1]:
        yield 'prof5',1,np.array([eq,99,99],np.float32)
    aliq=np.float32(PARAMS[12]*1000);bliq=PARAMS[13];cliq=np.float32(bliq*PARAMS[15]);dliq=PARAMS[14]
    for p,t,q in itertools.product([10000,50000,100000],[240,270,300],[.00001,.001,.01]):
        yield 'envirtht',2,np.array([p,t,q,99,aliq,bliq,cliq,dliq],np.float32)
    for p,t,qfrz in itertools.product([30000,70000,100000],[240,260,273],[0,.00001,.001]):
        es=np.float32(aliq*np.exp(np.float32((bliq*t-cliq)/(t-dliq))))
        qs=np.float32(es*.622/(p-es))
        yield 'dtfrznew',3,np.array([t,p,99,qs,qfrz,.002,aliq,bliq,cliq,dliq],np.float32)
    # First table row/column is legal for bilinear interpolation and reaches
    # the native diagnostic condition without reading outside the table.
    for ip,it,qu,cond in itertools.product([1,2,100,219],[1,2,125,249],[0,.001,.02],[0,.01]):
        p=np.float32(table[-1]+np.float32((ip-1+.25)/table[-3]))
        tp=np.float32(np.float32(p-table[-1])*table[-3]);frac=np.float32(tp-np.trunc(tp))
        i=int(tp)
        base=np.float32(np.float32(table[110000+i+1]-table[110000+i])*frac+table[110000+i])
        thes=np.float32(base+np.float32(it-1+.25)/table[-2])
        yield 'tpmix2',4,np.array([p,thes,99,qu,cond,cond,99,99,PARAMS[4],PARAMS[3]],np.float32)
        if qu==0 and cond==0:
            yield 'tpmix2dd',5,np.array([p,thes,99,99],np.float32)


def replay_helpers(build_dir,backend):
    native,host,table=load_oracles(build_dir)
    ptr=C.POINTER(C.c_float)
    names=['condload','prof5','envirtht','dtfrznew','tpmix2','tpmix2dd']
    routines=[getattr(native,'__module_cu_kfeta_MOD_'+n) for n in names]
    for kind,fn in enumerate(routines):fn.argtypes=[ptr]*[12,3,8,10,10,4][kind]+([C.POINTER(C.c_int)]*2 if kind==5 else [])
    host.kf_host_helper.argtypes=[C.c_int,np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')]
    if backend=='hex-cuda':
        import cupy as cp
        from cupy.cuda import compiler
        source=kf.shared_math_source()+kf.CUDA_SOURCE+HELPER_SOURCE
        architecture=cp.cuda.Device().compute_capability
        image_path=os.environ.get('MPAS_KF_HELPER_CUDA_IMAGE')
        if image_path:
            binary=kf.cuda_image(image_path,source,architecture)
        else:
            binary,_=compiler.compile_using_nvrtc(source,options=kf.CUDA_OPTIONS,arch=architecture)
        module=cp.cuda.function.Module();module.load(binary)
        fn=module.get_function('kf_helper_kernel');dt=kf.build_tables(cp.asarray(PARAMS))
    rows=[]
    for name,kind,x in helper_cases(table):
        expected=x.copy();args=[C.cast(expected.ctypes.data+i*4,ptr) for i in range(len(x))]
        if kind==5:
            i,j=C.c_int(1),C.c_int(1);args += [C.byref(i),C.byref(j)]
        routines[kind](*args)
        if backend=='host-glibc':
            actual=x.copy();host.kf_host_helper(kind,actual,table)
        else:
            actual=cp.asarray(x);fn((1,),(1,),(np.int32(kind),actual,dt));actual=cp.asnumpy(actual)
        rows.append({'routine':name,'input':x.tolist(),'differing_words':word_diff(expected,actual)})
    result={'schema':'mpas-kf-scalar-boundary-v1','backend':backend,'calls':len(rows),
            'differing_words':sum(r['differing_words'] for r in rows),
            'cuda_options':list(kf.CUDA_OPTIONS) if backend=='hex-cuda' else None,'rows':rows}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--backend',choices=['host-glibc','hex-cuda'],required=True);p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();r=replay_helpers(args.build_dir,args.backend)
    args.receipt.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='rows'})
    raise SystemExit(bool(r['differing_words']))
