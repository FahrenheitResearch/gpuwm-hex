"""Test the complete CPS optional-argument paths beyond the driver ABI.

MPAS-A v8.4.2 module_cu_kfeta.F:19-361. Disassociated actual pointers
are absent optional arguments under Fortran 2008. Unsafe native calls that
omit a species flag while requesting its corresponding output are not made.
"""
from pathlib import Path
import argparse, ctypes as C, hashlib, itertools, json, os
import numpy as np
from support import PARAMS, fixture, load_oracles, word_diff
from hexcore.mpasphys import kain_fritsch_v842 as kf


def native_optional_source(wrapper):
    start = wrapper.index('subroutine kf_native_column(')
    end = wrapper.index('end subroutine', start) + len('end subroutine')
    text = wrapper[start:end].replace('kf_native_column(', 'kf_native_optional(')
    text = text.replace('params,flags) bind(c)', 'params,flags,presence) bind(c)')
    text = text.replace('intent(inout) :: state', 'intent(inout),target :: state')
    text = text.replace('integer(c_int),intent(in) :: flags(7)',
                        'integer(c_int),intent(in) :: flags(7),presence(9)')
    text = text.replace('logical :: cu_act_flag', 'logical,target :: cu_act_flag')
    declarations = ['logical,pointer :: pqr,pqi,pqs',
                    'real(c_float),pointer :: pt(:,:,:),pv(:,:,:),pc(:,:,:),pr(:,:,:),pi(:,:,:),ps(:,:,:)']
    text = text.replace('cu_act_flag=active/=0', '\n'.join(declarations)+'\ncu_act_flag=active/=0')
    setup=[]
    for i,(ptr,val) in enumerate([('pqr','f_qr'),('pqi','f_qi'),('pqs','f_qs'),
              ('pt','state(:,:,:,2)'),('pv','state(:,:,:,3)'),('pc','state(:,:,:,4)'),
              ('pr','state(:,:,:,5)'),('pi','state(:,:,:,6)'),('ps','state(:,:,:,7)')],1):
        setup += [f'nullify({ptr})', f'if(presence({i})/=0) {ptr}=>{val}']
    text = text.replace('call kf_eta_cps(', '\n'.join(setup)+'\ncall kf_eta_cps(')
    for arg,ptr in [('f_qr','pqr'),('f_qi','pqi'),('f_qs','pqs')]:
        text=text.replace(f'{arg}={arg}, &',f'{arg}={ptr}, &')
    for arg,ptr,i in [('rthcuten','pt',2),('rqvcuten','pv',3),('rqccuten','pc',4),
                     ('rqrcuten','pr',5),('rqicuten','pi',6),('rqscuten','ps',7)]:
        text=text.replace(f'{arg}=state(:,:,:,{i})',f'{arg}={ptr}')
    return '\n! MPAS-A v8.4.2 module_cu_kfeta.F:19-361.\n'+text+'\n'


def optional_source():
    start=kf.CUDA_SOURCE.index('KFDEV void kf_batch_column(')
    end=kf.CUDA_SOURCE.index('\n#ifndef KF_HOST_TEST',start)
    text=kf.CUDA_SOURCE[start:end].replace('kf_batch_column(', 'kf_optional_column(')
    text=text.replace('float *table, float *work)', 'float *table, float *work, int *presence)')
    old=', true, true, true, true, true, true, true, true, true);'
    assert text.count(old)==1
    text=text.replace(old,', '+', '.join(f'bool(presence[{i}])' for i in range(9))+');')
    return text+'''\n#ifdef KF_HOST_TEST
extern "C" void kf_host_optional(int nz,float *fields,float *state,float *surface,
 unsigned char *active,float *params,int *flags,float *table,float *work,int *presence) {
 kf_optional_column(nz,1,0,fields,state,surface,active,params,flags,table,work,presence);
}
#else
extern "C" __global__ void kf_optional_kernel(int nz,float *fields,float *state,float *surface,
 unsigned char *active,float *params,int *flags,float *table,float *work,int *presence) {
 if (blockIdx.x || threadIdx.x) return;
 kf_optional_column(nz,1,0,fields,state,surface,active,params,flags,table,work,presence);
}
#endif
'''


def replay(build_dir,backend):
    native,host,table=load_oracles(build_dir)
    ptr=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS')
    iptr=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    bptr=np.ctypeslib.ndpointer(np.uint8,flags='C_CONTIGUOUS')
    native.kf_native_optional.argtypes=[C.c_int,ptr,ptr,ptr,C.POINTER(C.c_int),ptr,iptr,iptr]
    if backend=='host-glibc':
        host.kf_host_optional.argtypes=[C.c_int,ptr,ptr,ptr,bptr,ptr,iptr,ptr,ptr,iptr]
    else:
        import cupy as cp
        from cupy.cuda import compiler
        source = kf.shared_math_source()+kf.CUDA_SOURCE+optional_source()
        image_path = os.environ.get('MPAS_KF_OPTIONAL_CUDA_IMAGE')
        if image_path:
            binary = kf.cuda_image(image_path,source,cp.cuda.Device().compute_capability)
        else:
            binary,_=compiler.compile_using_nvrtc(source,
                                  options=kf.CUDA_OPTIONS,arch=cp.cuda.Device().compute_capability)
        module=cp.cuda.function.Module();module.load(binary)
        kernel=module.get_function('kf_optional_kernel')
        dt=cp.asarray(table);dp=cp.asarray(PARAMS)
    rows=[]
    for mask in itertools.product([0,1],repeat=9):
        if ((not mask[0] and mask[5] and mask[6]) or
            (not mask[1] and mask[7]) or (not mask[2] and mask[8])):
            continue # Native F:323,341,349 dereference those omitted flags.
        for warm in (0,1):
            f,s,v=fixture(55,8,.95,300,2);s[1:]=np.float32(.0125)
            sn=s.copy();vn=v.copy();an=C.c_int(0)
            flags=np.array([5,1,0,warm,1,1,1],np.int32);presence=np.array(mask,np.int32)
            native.kf_native_optional(55,f,sn,vn,C.byref(an),PARAMS,flags,presence)
            a=np.zeros(1,np.uint8)
            if backend=='host-glibc':
                host.kf_host_optional(55,f,s,v,a,PARAMS,flags,table,np.empty(112*55+1,np.float32),presence)
            else:
                df,ds,dv,da,dflags,dm=map(cp.asarray,(f,s,v,a,flags,presence))
                kernel((1,),(1,),(np.int32(55),df,ds,dv,da,dp,dflags,dt,cp.empty(112*55+1,cp.float32),dm))
                s,v,a=map(cp.asnumpy,(ds,dv,da))
            counts={'state':word_diff(sn,s),'surface':word_diff(vn,v),'active':int(an.value!=int(a[0]))}
            rows.append({'presence':list(mask),'warm_rain':warm,'differing_words':counts})
    return {'schema':'mpas-kf-optional-columns-v1','backend':backend,'columns':len(rows),
            'differing_words':sum(sum(r['differing_words'].values()) for r in rows),
            'driver_abi':False,'native_module_modified':False,
            'artifacts':{str(Path(build_dir)/name):hashlib.sha256((Path(build_dir)/name).read_bytes()).hexdigest() for name in ('libnative.so','libtranslated.so','source-pins.json','Makefile')},
            'source_sha256':kf.SOURCE_SHA256,'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'rows':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--backend',choices=['host-glibc','hex-cuda'],required=True)
    p.add_argument('--receipt',type=Path,required=True);args=p.parse_args()
    r=replay(args.build_dir,args.backend);args.receipt.write_text(json.dumps(r,indent=2)+'\n')
    print({k:v for k,v in r.items() if k!='rows'});raise SystemExit(bool(r['differing_words']))
