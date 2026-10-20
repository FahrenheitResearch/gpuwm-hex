"""RRFS specified-zone w boundary subsystem.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:7398-7446
at 1d6e370e. All other boundary work routines share v8.4.2 arithmetic.
"""
import numpy as np
from .cuda_backend.containers import require_resident_array
from .cuda_compile_rrfs import kernel
_CUDA_SOURCE=r'''
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:7398-7446 at 1d6e370e.
extern "C" __global__ void w_boundary_rrfs(int nl,int nc,int first,int end,int relaxation,
 int zero,const int *mask,const int *nearest,const float *input,float *output){
 int c=blockIdx.x*blockDim.x+threadIdx.x;
 if(c<first || c>=end || mask[c]<=relaxation)return;
 for(int k=1;k<nl;++k)output[k*nc+c]=zero?0.0f:input[k*nc+nearest[c]];
}
'''

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:7398-7446 at 1d6e370e.
def apply_w_boundary_cuda_rrfs(w, bdy_mask_cell, nearest_relaxation_cell, *,
    config_lbc_w, n_relax_zone=5, cell_solve_start=0, cell_solve_end=None,kernel_cache=None):
    """Fork mpas_atm_time_integration.F:7398-7446 at 1d6e370e; endpoints retained."""
    import cupy as cp
    require_resident_array('w',w,dtype=np.float32)
    if w.ndim!=2 or not w.flags.c_contiguous: raise ValueError('w must be contiguous (interface,cell)')
    nl=w.shape[0]-1; nc=w.shape[1]
    require_resident_array('bdy_mask_cell',bdy_mask_cell,dtype=np.int32,shape=(nc,))
    require_resident_array('nearest_relaxation_cell',nearest_relaxation_cell,dtype=np.int32,shape=(nc,))
    option=str(config_lbc_w).strip()
    if option not in ('zero','nearest'): raise ValueError('unknown native config_lbc_w option: '+option)
    end=nc if cell_solve_end is None else int(cell_solve_end)
    if not 0<=cell_solve_start<=end<=nc: raise ValueError('w boundary bounds exceed resident cells')
    # Native nearest indices reference relaxation cells, which this routine never writes.
    if end==0:return w
    kernel(_CUDA_SOURCE,'hexcore.cuda_regional_rrfs','w_boundary_rrfs',kernel_cache)(
        ((end+127)//128,),(128,),
        (np.int32(nl),np.int32(nc),np.int32(cell_solve_start),np.int32(end),np.int32(n_relax_zone),
         np.int32(option=='zero'),bdy_mask_cell,nearest_relaxation_cell,w,w))
    return w
