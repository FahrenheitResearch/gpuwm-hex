"""CPU-only NVRTC build; no CUDA driver/device API or kernel launch."""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import sys

from hexcore.mpasdiag.cloud_v842 import CUDA_SOURCE as CLOUD
from hexcore.mpasdiag.pv_v842 import CUDA_SOURCE as PV
from hexcore.mpasdiag.soundings_v842 import CUDA_SOURCE as SOUND, shared_device_math


def compile_source(name,source):
    nvrtc=ctypes.CDLL("libnvrtc.so.13")
    program=ctypes.c_void_p()
    status=nvrtc.nvrtcCreateProgram(ctypes.byref(program),source.encode(),name.encode(),0,None,None)
    if status: raise RuntimeError(f"nvrtcCreateProgram {status}")
    options=[b"--gpu-architecture=compute_120",b"--std=c++17",b"--fmad=false",b"--ftz=false"]
    pointers=(ctypes.c_char_p*len(options))(*options)
    try:
        status=nvrtc.nvrtcCompileProgram(program,len(options),pointers)
        size=ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program,ctypes.byref(size))
        log=ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program,log)
        result=dict(module=name,status=status,log=log.value.decode(),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            options=[o.decode() for o in options])
        if status==0:
            nvrtc.nvrtcGetPTXSize(program,ctypes.byref(size))
            ptx=ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetPTX(program,ptx)
            result["ptx_sha256"]=hashlib.sha256(ptx.raw).hexdigest()
        return result
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


if __name__=="__main__":
    results=[compile_source(name,source) for name,source in (
        ("cloud_v842",CLOUD),("pv_v842",PV),("soundings_v842",shared_device_math()+SOUND))]
    Path(sys.argv[1]).write_text(json.dumps({"kind":"NVRTC-compile-only-no-device", "rows":results},indent=2)+"\n")
    assert all(r["status"]==0 for r in results),results
