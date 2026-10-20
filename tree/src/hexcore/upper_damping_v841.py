"""Rust CPU bindings for the native upper damping operators."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from functools import lru_cache
import numpy as np
from .errors import ConfigurationRefusal


@lru_cache(maxsize=1)
def _library():
    source=Path(__file__).with_name("native")/"upper_damping.rs"
    cache=Path(os.environ.get("HEXCORE_NATIVE_CPU_CACHE",os.environ.get("TMPDIR",str(Path.home()/".cache"/"woof"/"dycore"))))
    compiler=shutil.which("rustc")
    if compiler is None:
        raise ConfigurationRefusal("rustc",None,"active CPU upper damping requires its Rust numerical library","install rustc and set HEXCORE_NATIVE_CPU_CACHE to a writable build cache")
    version=subprocess.check_output([compiler,"--version"],text=True).strip()
    key=hashlib.sha256(source.read_bytes()+version.encode()).hexdigest()
    cache=cache/key;cache.mkdir(parents=True,exist_ok=True)
    suffix=".dll" if os.name=="nt" else ".so"
    target=cache/("upper_damping"+suffix)
    if not target.is_file():
        temporary=cache/("upper_damping-"+str(os.getpid())+suffix)
        subprocess.run([compiler,"--edition=2021","--crate-type=cdylib","-C","opt-level=2",str(source),"-o",str(temporary)],check=True,capture_output=True)
        os.replace(temporary,target)
    library=ctypes.CDLL(str(target))
    for dtype,scalar in (("f32",ctypes.c_float),("f64",ctypes.c_double)):
        pointer=ctypes.POINTER(scalar)
        cam=getattr(library,"upper_cam_"+dtype);cam.restype=ctypes.c_int
        cam.argtypes=[ctypes.c_size_t]*3+[scalar,scalar,pointer]
        ray=getattr(library,"upper_rayleigh_"+dtype);ray.restype=ctypes.c_int
        ray.argtypes=[ctypes.c_size_t]*4+[scalar,pointer,pointer,pointer]
    return library


def _field(value):
    array=np.asarray(value)
    if array.ndim!=2 or array.dtype not in (np.dtype(np.float32),np.dtype(np.float64)):
        raise TypeError("upper damping requires a float32 or float64 level-major field")
    if not array.flags.c_contiguous:
        raise ValueError("upper damping requires contiguous storage")
    scalar=ctypes.c_float if array.dtype==np.float32 else ctypes.c_double
    return array,array.ctypes.data_as(ctypes.POINTER(scalar)),"f32" if array.dtype==np.float32 else "f64"


def apply_cam_floor(values,*,length,coefficient,levels=4):
    array,pointer,dtype=_field(values)
    if type(levels) is not int:raise TypeError("damping level count must be an integer")
    if getattr(_library(),"upper_cam_"+dtype)(*array.shape,levels,length,coefficient,pointer):
        raise ValueError("CAM damping requires 1..nVertLevels levels and a finite positive filter length")
    return array


def apply_rayleigh_u(tendency,rho,velocity,*,levels=6,days=5.0,solved=None):
    array,output,dtype=_field(tendency)
    density,rho_pointer,rho_type=_field(np.ascontiguousarray(rho))
    wind,wind_pointer,wind_type=_field(np.ascontiguousarray(velocity))
    if density.shape!=array.shape or wind.shape!=array.shape or rho_type!=dtype or wind_type!=dtype:
        raise ValueError("Rayleigh density, wind and tendency must share shape and dtype")
    if type(levels) is not int:raise TypeError("damping level count must be an integer")
    count=array.shape[1] if solved is None else solved
    if getattr(_library(),"upper_rayleigh_"+dtype)(*array.shape,count,levels,days,output,rho_pointer,wind_pointer):
        raise ValueError("Rayleigh damping requires 1..nVertLevels levels and a finite positive timescale")
    return array
