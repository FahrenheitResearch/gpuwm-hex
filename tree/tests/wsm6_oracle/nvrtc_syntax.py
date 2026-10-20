import os
os.environ["CUPY_COMPILE_WITH_PTX"]="1"
import json
import hashlib
import cupy as cp
from pathlib import Path
from hexcore.mpasphys.wsm6_v842 import cuda_source, CUDA_HEIGHT_GUARD
from cupy.cuda.compiler import compile_using_nvrtc
h=''
for n in ('exp','log','sqrt','log10','sin','cos','atan'):
 for t in ('float','double'):h+=f'extern __device__ {t} mp_{n}({t});\n'
for t in ('float','double'):h+=f'extern __device__ {t} mp_pow({t},{t});\n'
h+='extern __device__ double mp_abs(cx);\nextern __device__ cx mp_log(cx);\nextern __device__ cx mp_sqrt(cx);\nextern __device__ cx4 mp_sqrt(cx4);\n'
for t in ('cx','cx4'):h+=f'extern __device__ {t} mp_complex_div({t},{t});\n'
try:
 result=compile_using_nvrtc(cuda_source(h)+CUDA_HEIGHT_GUARD,options=(f'-I{Path(cp.__file__).parent / "_core/include"}', f'-I{Path(cp.__file__).parent / "_core/include/cupy/_cccl/thrust"}', f'-I{Path(cp.__file__).parent / "_core/include/cupy/_cccl/libcudacxx"}', '--std=c++17','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true'),name_expressions=('wsm6_columns','wsm6_from_mpas','wsm6_to_mpas','wsm6_radar_to_mpas','wsm6_height_guard','wsm6_initialized_state'))
 Path('receipts/nvrtc-syntax.json').write_text(json.dumps({'scope':'compile only: external shared math functions are unresolved; no device computation or identity claimed','source_sha256':hashlib.sha256((cuda_source(h)+CUDA_HEIGHT_GUARD).encode()).hexdigest(),'ptx_bytes':len(result[0]),'entrypoints':list(result[1])},indent=2)+'\n')
 print('NVRTC syntax passed',len(result[0]))
except Exception as e:
 Path('receipts/nvrtc-syntax-failure.txt').write_text(str(e))
 raise
