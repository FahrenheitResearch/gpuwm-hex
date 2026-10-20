"""CPU-only NVRTC syntax/code-generation check; never initializes a GPU."""
import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from cupy.cuda import nvrtc
from hexcore.mpasphys import sf_mynn_v842 as scheme

p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
args=p.parse_args()
source=scheme._cuda_source()
program=nvrtc.createProgram(source,'sf_mynn_v842.cu',(),())
options=('--gpu-architecture=compute_120','--std=c++17','--fmad=false',
         '--ftz=false','--prec-div=true','--prec-sqrt=true')
try:
    nvrtc.compileProgram(program,options)
    ptx=nvrtc.getPTX(program)
    receipt={'nvrtc':nvrtc.getVersion(),'options':options,'compiled':True,
             'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
             'ptx_sha256':hashlib.sha256(ptx).hexdigest(),
             'compiler_log':nvrtc.getProgramLog(program),
             'gpu_execution':False}
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
finally:
    nvrtc.destroyProgram(program)
