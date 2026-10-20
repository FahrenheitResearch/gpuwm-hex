"""Offline whole-module syntax/PTX check; no GPU load or numerical claim."""
import argparse,hashlib,json
from pathlib import Path
import cupy as cp
from cupy_backends.cuda.libs import nvrtc
from hexcore.cuda_fp32 import CUDA_FTZ_HELPERS
from hexcore.mpasphys.rrfs.bl_mynnedmf_c942418 import ARRAY_SUPPORT,MODULE_CUDA,CUDA_OPTIONS

def check(path):
    declarations='\n'.join(f'extern __device__ {kind} mynn_{name}({kind});' for name in ('exp','log','log10','atan','tanh','erf') for kind in ('float','double'))
    declarations+='\n'+'\n'.join(f'extern __device__ {ret} mynn_pow({a},{b});' for a,b,ret in [('float','float','float'),('double','double','double'),('float','double','double'),('double','float','double')])
    source=CUDA_FTZ_HELPERS+declarations+ARRAY_SUPPORT+MODULE_CUDA
    options=CUDA_OPTIONS+('--device-c','--device-debug','--gpu-architecture=compute_120')
    result={'scope':'CPU-only offline NVRTC compiler check; shared maths unresolved, no numerical load/launch','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'options':options,'cupy':cp.__version__}
    program=nvrtc.createProgram(source,'rrfs_mynnedmf.cu',(),())
    try:
        nvrtc.compileProgram(program,options);ptx=nvrtc.getPTX(program)
        result.update(compiled=True,ptx_bytes=len(ptx),ptx_sha256=hashlib.sha256(ptx).hexdigest())
    except Exception as exc:result.update(compiled=False,error=str(exc))
    finally:
        result['compiler_log']=nvrtc.getProgramLog(program);nvrtc.destroyProgram(program)
    path.write_text(json.dumps(result,indent=2)+'\n');return int(not result['compiled'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();raise SystemExit(check(a.receipt))
