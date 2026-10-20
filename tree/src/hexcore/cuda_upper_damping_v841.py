"""Device upper damping at the native tendency and viscosity sites."""
import numpy as np
from .cuda_fp32 import CUDA_FTZ_HELPERS
_CUDA_SOURCE=CUDA_FTZ_HELPERS+r'''
extern "C" __global__ void upper_cam_f32(const int nl,const int nc,const int solved,const int levels,const float length,const float coefficient,float *values) {
 int index=blockDim.x*blockIdx.x+threadIdx.x;if(index>=nl*nc)return;
 int k=index/nc;if(index%nc>=solved || k<nl-levels)return;
 float base=mpas_mul(mpas_mul(mpas_mul(4.0f,2.0833f),length),coefficient);
 float floor=mpas_mul(base,mpas_sub(1.0f,mpas_div(float(nl-1-k),float(levels))));
 values[index]=mpas_max(values[index],floor);
}
extern "C" __global__ void upper_rayleigh_f32(const int nl,const int ne,const int solved,const int levels,const float days,float *tendency,const float *rho,const float *velocity) {
 int index=blockDim.x*blockIdx.x+threadIdx.x;if(index>=nl*ne)return;
 int k=index/ne;if(index%ne>=solved || k<nl-levels)return;
 float inverse=mpas_div(1.0f,mpas_mul(float(levels),mpas_mul(days,86400.0f)));
 float coefficient=mpas_mul(float(k-(nl-levels)+1),inverse);
 tendency[index]=mpas_sub(tendency[index],mpas_mul(mpas_mul(rho[index],velocity[index]),coefficient));
}
'''


def _launch(name,values,cache,levels,scalar,solved,extra=()):
    from .cuda_backend import require_resident_array
    values=require_resident_array("upper damping field",values,dtype=np.float32)
    if values.ndim!=2:raise ValueError("upper damping requires a level-major matrix")
    nl,nc=values.shape
    if type(levels) is not int or not 1<=levels<=nl:raise ValueError("damping level count must lie in 1..nVertLevels")
    count=nc if solved is None else solved
    if type(count) is not int or not 0<=count<=nc:raise ValueError("upper damping solve count must lie inside the field")
    for value in extra:
        field=require_resident_array("upper damping input",value,dtype=np.float32)
        if field.shape!=values.shape:raise ValueError("upper damping input shape differs from tendency")
    if any(not np.isfinite(value) for value in scalar):raise ValueError("upper damping parameters must be finite")
    kernel=cache.raw_kernel(name,_CUDA_SOURCE,module_key="hexcore.cuda_upper_damping_v841")
    kernel(((values.size+127)//128,),(128,),
        (np.int32(nl),np.int32(nc),np.int32(count),np.int32(levels),*scalar,values,*extra))


def apply_cam_floor_cuda(values,*,length,coefficient,levels=4,solved=None,cache):
    if not np.isfinite(length) or length<=0 or not np.isfinite(coefficient) or coefficient<0:
        raise ValueError("CAM damping requires positive length and nonnegative coefficient")
    _launch("upper_cam_f32",values,cache,levels,(np.float32(length),np.float32(coefficient)),solved)
    return values


def apply_rayleigh_u_cuda(tendency,rho,velocity,*,levels=6,days=5.0,solved=None,cache):
    if not np.isfinite(days) or days<=0:raise ValueError("Rayleigh damping requires a finite positive timescale")
    _launch("upper_rayleigh_f32",tendency,cache,levels,(np.float32(days),),solved,(rho,velocity))
    return tendency
