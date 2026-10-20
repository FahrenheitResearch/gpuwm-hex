"""Measure shared device maths against host glibc; no scheme substitution.

Gamma inputs span the positive domain used by GFL beta distributions, including
overflow. Log10 inputs cover normal/subnormal positive binary32 arguments.
This diagnostic cannot establish an inside-native-spread forecast tier.
"""
import argparse
import ctypes
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--state-probe',type=Path)
    args=parser.parse_args()
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc
    from gpuwm.core import kernels
    from hexcore.mpasphys.rrfs import cu_gfl_bf6a4b0 as gfl
    header=Path(kernels.__file__).with_name('glibc_flt32.cuh')
    source=gfl.cuda_source()+"\n"+r"""
extern "C" __global__ void measure_math(int n,int function,const float* input,const float* exponent,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)output[i]=function==0?gf_tgamma(GFReal(input[i])).value:
    function==1?gf_log10(GFReal(input[i])).value:
    function==2?gf_abs(GFReal(input[i])).value:
    function==3?gf_exp(GFReal(input[i])).value:
    function==4?gf_log(GFReal(input[i])).value:
    function==5?gf_sin(GFReal(input[i])).value:
    function==6?gf_atan(GFReal(input[i])).value:
    function==7?gf_sqrt(GFReal(input[i])).value:
                gf_pow(GFReal(input[i]),GFReal(exponent[i])).value;
}
// MPAS-Model-rrfs/src/core_atmosphere/physics/physics_noaa/GFL/module_cu_gfl_deep.F90:2639-2794
extern "C" __global__ void measure_fct_save(float* z,float* tracer,float* mass,float* flux,
 float* output,unsigned int* state,char* scratch){
 GFWorkspace workspace{scratch,65536,0,0,0,GFReal(0),state[0]!=0};
 int n=8,top=3;GFReal dt=30,g=9.80616f;
 fct1d3(workspace,&top,&n,&dt,(GFReal*)z,(GFReal*)tracer,(GFReal*)mass,(GFReal*)flux,(GFReal*)output,&g);
 state[0]=workspace.fct1d3_error;
}
"""
    program=nvrtc.createProgram(source,'gfl_shared_gamma.cu',(),())
    try:
        nvrtc.compileProgram(program,gfl.COMPILER_OPTIONS+(f'--gpu-architecture=sm_{cp.cuda.Device().compute_capability}',))
        binary=nvrtc.getCUBIN(program)
    except Exception as error:raise RuntimeError(nvrtc.getProgramLog(program)) from error
    finally:nvrtc.destroyProgram(program)
    module=cp.cuda.function.Module();module.load(binary)
    gamma=np.unique(np.concatenate((np.linspace(.5,60.,20001,dtype='f4'),
        np.array([1.,1.3,2.2,3.7,4.,6.,7.7],dtype='f4'))))
    log10=np.unique(np.concatenate((np.random.default_rng(1310).integers(1,0x7f800000,20001,dtype='u4').view('f4'),
        np.array([1,0x007fffff,0x00800000,0x7f7fffff],dtype='u4').view('f4'),
        np.array([10.**n for n in range(-44,39)],dtype='f4'))))
    checks=[]
    absolute=np.array([0x80000000,0,1,0x80000001,0x007fffff,0x807fffff,
                       0x00800000,0x80800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000],dtype='u4').view('f4')
    signed=np.concatenate((log10,-log10,np.linspace(-1000.,1000.,10001,dtype='f4')))
    powers=np.concatenate((np.linspace(.0001,30.,10001,dtype='f4'),log10))
    exponents=np.resize(np.array([-5.,-2.5,-1.,0.,.25,.5,1.,1.3,2.,2.2,3.7,4.,6.],dtype='f4'),powers.size)
    for function,(name,inputs) in enumerate((('tgammaf',gamma),('log10f',log10),('fabsf',absolute),
            ('expf',np.linspace(-110.,90.,20001,dtype='f4')),('logf',log10),
            ('sinf',signed),('atanf',signed),('sqrtf',log10),('powf',powers))):
        native=getattr(ctypes.CDLL('libm.so.6'),name)
        native.argtypes=[ctypes.c_float]*(2 if name=='powf' else 1);native.restype=ctypes.c_float
        want=np.array([native(float(x),float(y)) for x,y in zip(inputs,exponents)] if name=='powf'
                      else [native(float(x)) for x in inputs],dtype='f4')
        a=cp.asarray(inputs);b=cp.empty_like(a);e=cp.asarray(exponents) if name=='powf' else a
        module.get_function('measure_math')(((a.size+127)//128,),(128,),(np.int32(a.size),np.int32(function),a,e,b))
        have=cp.asnumpy(b);bad=np.flatnonzero(want.view('u4')!=have.view('u4'))
        checks.append({'function':name,'samples':int(inputs.size),'different_words':int(bad.size),
            'input_sha256':hashlib.sha256(inputs.tobytes()).hexdigest(),
            'examples':[{'input_word':int(inputs.view('u4')[i]),'native_word':int(want.view('u4')[i]),
                         'device_word':int(have.view('u4')[i])} for i in bad[:30]]})
        if name=='powf':
            checks[-1]['exponent_sha256']=hashlib.sha256(exponents.tobytes()).hexdigest()
            for example,i in zip(checks[-1]['examples'],bad):example['exponent_word']=int(exponents.view('u4')[i])
    state_checks=[]
    if args.state_probe:
        transport=Path('native/oracle_transport').resolve()
        scratch_decode=tempfile.TemporaryDirectory(prefix='r13-gfl-state-decode-')
        rawdir=Path(scratch_decode.name)
        decoded={}
        def decode(path):
            if path in decoded:return decoded[path].copy()
            raw=rawdir/(path.name+'.raw')
            info=json.loads(subprocess.check_output([str(transport),'decode',str(path),str(raw)],text=True))
            if info['width']!=4 or info['code'] not in (1,4):raise ValueError(info)
            decoded[path]=np.frombuffer(raw.read_bytes(),dtype='u4' if info['code']==4 else 'f4').copy()
            return decoded[path].copy()
        for record in sorted(args.state_probe.glob('*.pre.z.bin')):
            prefix=record.name.removesuffix('.pre.z.bin')
            path=lambda phase,name:args.state_probe/(prefix+'.'+phase+'.'+name+'.bin')
            inputs=[cp.asarray(decode(path('pre',name))) for name in ('z','tracr','massflx','trflx','out')]
            state_name='local.module_cu_gfl_deep.fct1d3.error'
            state=cp.asarray(decode(path('pre',state_name)))
            scratch=cp.empty(65536,dtype=cp.uint8)
            module.get_function('measure_fct_save')((1,),(1,),(*inputs,state,scratch))
            have=cp.asnumpy(inputs[-1]);want=decode(path('post','out'))
            got_state=cp.asnumpy(state);want_state=decode(path('post',state_name))
            state_checks.append({'capture':prefix,'output_different_words':int(np.count_nonzero(have.view('u4')!=want.view('u4'))),
                'state_different_words':int(np.count_nonzero(got_state!=want_state)),
                'native_pre_state':decode(path('pre',state_name)).tolist(),
                'native_post_state':want_state.tolist(),'cuda_post_state':got_state.tolist(),
                'record_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in args.state_probe.glob(prefix+'.*.bin')}})
        if len(state_checks)!=3:raise ValueError('incomplete native SAVE probe')
        scratch_decode.cleanup()
    result={'scope':'shared maths and retained SAVE diagnostic; not forecast or spread-tier acceptance',
        'shared_header_sha256':hashlib.sha256(header.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'compiler_options':gfl.COMPILER_OPTIONS,'checks':checks,
        'state_checks':state_checks,
        'different_words':sum(c['different_words'] for c in checks)+sum(c['output_different_words']+c['state_different_words'] for c in state_checks)}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return bool(result['different_words'])


if __name__=='__main__':raise SystemExit(main())
