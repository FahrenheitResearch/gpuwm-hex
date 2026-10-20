"""RRFS scalar transport subsystem with GSL horizontal mixing.

Source: RRFSx/MPAS-Model MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023,4180-4902
at 1d6e370e. Non-mixing advection and monotonic limiting retain the shared
v8.4.2 work-routine arithmetic (see lane source-audit.json). The tendency
buffer is the native scalar_tend_save and persists across RK stages.
"""
from typing import Any
import numpy as np
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_backend.containers import require_resident_array
from .cuda_transport_v841 import advance_scalar_transport_cuda_v841
from .cuda_compile_rrfs import kernel

_CUDA_SOURCE = CUDA_FTZ_HELPERS + r'''
#define Q(s,k,c,nl,nc) (((s)*(nl)+(k))*(nc)+(c))
/* Fork mpas_atm_time_integration.F:3737-3795 at 1d6e370e. */
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
extern "C" __global__ void gsl_scalar_laplacian_rrfs(
 int ns,int nl,int nc,int ne,int me,int cells,const int *counts,
 const int *edges,const int *coe,const float *sign,const float *inv_area,
 const float *dv,const float *inv_dc,const float *scale2,const float *km,
 const float *rho,const float *old,float prandtl_inv,float *lap,float *tend)
{
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=cells)return;
 for(int j=0;j<counts[c];++j){
  int e=edges[c*me+j], c0=coe[2*e], c1=coe[2*e+1];
  float edge_sign=mpas_mul(mpas_mul(mpas_mul(inv_area[c],sign[c*me+j]),dv[e]),inv_dc[e]);
  float pr_scale=mpas_mul(prandtl_inv,scale2[e]);
  for(int k=0;k<nl;++k){
   float avg=mpas_mul(mpas_mul(mpas_mul(0.5f,mpas_add(km[k*nc+c0],km[k*nc+c1])),pr_scale),rho[k*ne+e]);
   for(int s=0;s<ns;++s){
    int q=Q(s,k,c,nl,nc);
    float flux=mpas_mul(edge_sign,mpas_sub(old[Q(s,k,c1,nl,nc)],old[Q(s,k,c0,nl,nc)]));
    tend[q]=mpas_add(tend[q],mpas_mul(flux,avg));
    lap[q]=mpas_add(lap[q],flux);
   }
  }
 }
}
/* Fork mpas_atm_time_integration.F:3799-3830 at 1d6e370e. */
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
extern "C" __global__ void gsl_scalar_filter4_rrfs(
 int ns,int nl,int nc,int ne,int me,int solved,const int *counts,
 const int *edges,const int *coe,const float *sign,const float *inv_area,
 const float *dv,const float *inv_dc,const float *scale4,const float *rho,
 const float *lap,float filter4,float *tend)
{
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=solved)return;
 float area=mpas_mul(filter4,inv_area[c]);
 for(int j=0;j<counts[c];++j){
  int e=edges[c*me+j], c0=coe[2*e], c1=coe[2*e+1];
  float edge_sign=mpas_mul(mpas_mul(mpas_mul(mpas_mul(scale4[e],area),dv[e]),sign[c*me+j]),inv_dc[e]);
  for(int k=0;k<nl;++k){
   // Native overwrites area here: its final value affects the next edge.
   area=mpas_mul(mpas_mul(filter4,inv_area[c]),rho[k*ne+e]);
   for(int s=0;s<ns;++s){
    int q=Q(s,k,c,nl,nc);
    tend[q]=mpas_sub(tend[q],mpas_mul(mpas_mul(edge_sign,rho[k*ne+e]),
       mpas_sub(lap[Q(s,k,c1,nl,nc)],lap[Q(s,k,c0,nl,nc)])));
   }
  }
 }
}

/* Complete standard scalar advection work: fork F:3835-4023 at 1d6e370e. */
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
extern "C" __global__ void scalar_edge_rrfs(int ns,int nl,int nc,int ne,int width,int regional,
 const int edge_start,const int edge_end,
 const int *mask,const int *coe,const float *dv,const float *stage,const float *uh,
 const int *counts,const int *cells,const float *coef,const float *coef3,float *edge){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<edge_start || e>=edge_end)return;
 if(regional && mask[e]>5)return;
 for(int k=0;k<nl;++k){float velocity=uh[k*ne+e];float direction=copysignf(1.0f,velocity);
  for(int t=0;t<ns;++t){float value=0.0f;
   if(!regional || mask[e]<4){
    for(int j=0;j<counts[e];++j){int c=cells[e*width+j];
     float weight=mpas_add(coef[e*width+j],mpas_mul(direction,coef3[e*width+j]));
     float term=mpas_mul(weight,stage[Q(t,k,c,nl,nc)]);
     value=(counts[e]==10 && j==0)?term:mpas_add(value,term);}
   }else if(mask[e]<=5){
    float ud=copysignf(.5f,velocity);
    float positive=mpas_mul(dv[e],fabsf(mpas_add(ud,.5f)));
    float negative=mpas_mul(dv[e],fabsf(mpas_sub(ud,.5f)));
    value=mpas_add(mpas_mul(positive,stage[Q(t,k,coe[2*e],nl,nc)]),mpas_mul(negative,stage[Q(t,k,coe[2*e+1],nl,nc)]));
   }
   edge[(t*nl+k)*ne+e]=value;
  }
 }
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
__device__ float scalar_vertical_rrfs(int s,int k,int c,int nl,int nc,float coef,
 const float *q,const float *ww,const float *fnm,const float *fnp){
 if(k==0 || k==nl)return 0.0f;
 float u=ww[k*nc+c];float qi=q[Q(s,k,c,nl,nc)],qm=q[Q(s,k-1,c,nl,nc)];
 if(k==1 || k==nl-1)return mpas_mul(u,mpas_add(mpas_mul(fnm[k],qi),mpas_mul(fnp[k],qm)));
 float qp=q[Q(s,k+1,c,nl,nc)],qmm=q[Q(s,k-2,c,nl,nc)];
 float centered=mpas_div(mpas_mul(u,mpas_sub(mpas_mul(7.0f,mpas_add(qi,qm)),mpas_add(qp,qmm))),12.0f);
 float up=mpas_div(mpas_mul(mpas_mul(coef,fabsf(u)),mpas_sub(mpas_sub(qp,qmm),mpas_mul(3.0f,mpas_sub(qi,qm)))),12.0f);
 return mpas_add(centered,up);
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
extern "C" __global__ void scalar_finish_rrfs(int ns,int nl,int nc,int ne,int me,int solved,
 const int *mask,const int *counts,const int *edges,const float *sign,const float *inv_area,
 const float *uh,const float *edge,const float *ww,const float *fnm,const float *fnp,
 const float *rdzw,const float *old,const float *stage,const float *rho0,const float *rho1,
 const float *source,float dt,float weight,float coef,float *out,float *target){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 float old_weight=mpas_sub(1.0f,weight);
 for(int k=0;k<nl;++k){float density=mpas_add(mpas_mul(old_weight,rho0[k*nc+c]),mpas_mul(weight,rho1[k*nc+c]));
  target[k*nc+c]=density;
  for(int t=0;t<ns;++t){int q=Q(t,k,c,nl,nc);out[q]=stage[q];
   if(c>=solved || mask[c]>5)continue;
   float tend=0.0f;
   for(int j=0;j<counts[c];++j){int e=edges[c*me+j];
    tend=mpas_sub(tend,mpas_mul(mpas_mul(sign[c*me+j],uh[k*ne+e]),edge[(t*nl+k)*ne+e]));}
   tend=mpas_add(mpas_mul(tend,inv_area[c]),source[q]);
   float lower=scalar_vertical_rrfs(t,k,c,nl,nc,coef,stage,ww,fnm,fnp);
   float upper=scalar_vertical_rrfs(t,k+1,c,nl,nc,coef,stage,ww,fnm,fnp);
   float update=mpas_add(mpas_mul(old[q],rho0[k*nc+c]),mpas_mul(dt,mpas_sub(tend,mpas_mul(rdzw[k],mpas_sub(upper,lower)))));
   out[q]=mpas_mul(update,mpas_div(1.0f,density));
  }
 }
}
'''

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
def apply_gsl_scalar_mixing_cuda_rrfs(mesh: Any, context: Any, scalar_old: Any,
    scalar_tendency: Any, *, kdiff: Any, rho_edge: Any,
    mesh_scaling_del2: Any, mesh_scaling_del4: Any,
    h_scalar_filter4: float, cell_count: int | None = None,
    cell_solve_count: int | None = None, prandtl: float = 1.0, kernel_cache=None) -> Any:
    """Fork mpas_atm_time_integration.F:3723-3834 at 1d6e370e; RK-1 saved source."""
    import cupy as cp
    old = require_resident_array('scalar_old',scalar_old,dtype=np.float32)
    squeezed=old.ndim==2
    if squeezed: old=old[None,:,:]
    if old.ndim!=3 or not old.flags.c_contiguous:
        raise ValueError('scalar_old must be contiguous (scalar,level,cell)')
    ns,nl,nc=map(int,old.shape); ne=int(mesh.n_edges); me=int(mesh.max_edges)
    tend=require_resident_array('scalar_tendency',scalar_tendency,dtype=np.float32,shape=scalar_old.shape)
    if squeezed: tend=tend[None,:,:]
    cells=nc if cell_count is None else int(cell_count)
    solved=cells if cell_solve_count is None else int(cell_solve_count)
    if not 0<=solved<=cells<=nc: raise ValueError('mixing cell bounds exceed resident cells')
    arrays=[('kdiff',kdiff,(nl,nc)),('rho_edge',rho_edge,(nl,ne)),
        ('mesh_scaling_del2',mesh_scaling_del2,(ne,)),('mesh_scaling_del4',mesh_scaling_del4,(ne,))]
    for name,value,shape in arrays: require_resident_array(name,value,dtype=np.float32,shape=shape)
    lap=cp.zeros_like(old)
    common=(np.int32(ns),np.int32(nl),np.int32(nc),np.int32(ne),np.int32(me))
    geometry=(mesh.n_edges_on_cell,mesh.edges_on_cell,mesh.cells_on_edge,mesh.edge_sign_on_cell,
              context.inv_area_cell,mesh.dv_edge,context.inv_dc_edge)
    if cells:
        kernel(_CUDA_SOURCE,'hexcore.cuda_transport_rrfs','gsl_scalar_laplacian_rrfs',kernel_cache)(((cells+127)//128,),(128,),
            (*common,np.int32(cells),*geometry,mesh_scaling_del2,kdiff,rho_edge,old,
             np.float32(np.float32(1)/np.float32(prandtl)),lap,tend))
    if h_scalar_filter4>0 and solved:
        kernel(_CUDA_SOURCE,'hexcore.cuda_transport_rrfs','gsl_scalar_filter4_rrfs',kernel_cache)(((solved+127)//128,),(128,),
            (*common,np.int32(solved),*geometry,mesh_scaling_del4,rho_edge,lap,np.float32(h_scalar_filter4),tend))
    return scalar_tendency


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
def advance_scalar_transport_cuda_rrfs(mesh: Any, context: Any, scalar_old: Any,
    scalar_stage: Any, rho_zz_old: Any, rho_zz_new: Any, uh_avg: Any, ww_avg: Any,
    dt: float, *, config_scalar_eddy_mix: bool = False,
    config_h_scalar_filter4: float = 0.0, kdiff: Any = None,
    rho_edge: Any = None, mesh_scaling_del2: Any = None,
    mesh_scaling_del4: Any = None, mixing_cell_count: int | None = None,
    mixing_cell_solve_count: int | None = None, **kwargs: Any) -> Any:
    """Fork mpas_atm_time_integration.F:3615-4023,4180-4902 at 1d6e370e."""
    if kwargs['config_scalar_advection'] and kwargs['rk_step']==1 and config_scalar_eddy_mix:
        apply_gsl_scalar_mixing_cuda_rrfs(mesh,context,scalar_old,kwargs['scalar_tendency'],
            kdiff=kdiff,rho_edge=rho_edge,mesh_scaling_del2=mesh_scaling_del2,
            mesh_scaling_del4=mesh_scaling_del4,h_scalar_filter4=config_h_scalar_filter4,
            cell_count=mixing_cell_count,cell_solve_count=mixing_cell_solve_count,kernel_cache=kwargs.get("kernel_cache"))
    if kwargs['config_scalar_advection'] and not (kwargs['rk_step']==3 and (kwargs['config_monotonic'] or kwargs['config_positive_definite'])):
        if kwargs['config_scalar_advection'] and kwargs['rk_step']==1 and config_scalar_eddy_mix:
            scalar_stage=scalar_old.copy()
        return advance_scalars_standard_cuda_rrfs(mesh,context,scalar_old,scalar_stage,rho_zz_old,rho_zz_new,uh_avg,ww_avg,dt,cell_solve_count=mixing_cell_solve_count,advance_density=kwargs.get("config_split_dynamics_transport",False),**kwargs)
    if kwargs['config_scalar_advection']:
        from .cuda_monotonic_rrfs import advance_scalars_monotonic_cuda_rrfs
        return advance_scalars_monotonic_cuda_rrfs(mesh,context,scalar_old,scalar_stage,
            rho_zz_old,rho_zz_new,uh_avg,ww_avg,dt,
            advance_density=kwargs.get('config_split_dynamics_transport',False),
            cell_solve_count=mixing_cell_solve_count,**kwargs)
    return advance_scalar_transport_cuda_v841(mesh,context,scalar_old,scalar_stage,
        rho_zz_old,rho_zz_new,uh_avg,ww_avg,dt,**kwargs)


# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023 at 1d6e370e.
def advance_scalars_standard_cuda_rrfs(mesh,context,scalar_old,scalar_stage,rho_zz_old,rho_zz_new,uh_avg,ww_avg,dt,*,
    coefficients,fzm,fzp,rdzw,rk_step,config_time_integration_order,config_coef_3rd_order,scalar_tendency=None,
    advance_density=True,config_apply_lbcs=False,bdy_mask_cell=None,bdy_mask_edge=None,cell_solve_count=None,kernel_cache=None,
    horiz_flux_array=None,edge_start=0,edge_end=None,**unused):
    """Complete fork mpas_atm_time_integration.F:3615-4023 at 1d6e370e."""
    import cupy as cp
    from .cuda_transport import CudaScalarTransportResult
    old=require_resident_array('scalar_old',scalar_old,dtype=np.float32)
    stage=require_resident_array('scalar_stage',scalar_stage,dtype=np.float32,shape=old.shape)
    squeezed=old.ndim==2
    if squeezed:old=old[None,:,:];stage=stage[None,:,:]
    ns,nl,nc=map(int,old.shape);ne=int(uh_avg.shape[1]);me=int(mesh.max_edges)
    solved=nc if cell_solve_count is None else int(cell_solve_count)
    source=cp.zeros_like(old) if scalar_tendency is None else scalar_tendency.reshape(old.shape)
    maskc=cp.zeros(nc,dtype=cp.int32) if bdy_mask_cell is None else bdy_mask_cell
    maske=cp.zeros(ne,dtype=cp.int32) if bdy_mask_edge is None else bdy_mask_edge
    width=int(coefficients.adv_coefs.shape[1])
    edge=cp.zeros((ns,nl,ne),dtype=cp.float32) if horiz_flux_array is None else require_resident_array(
        'horiz_flux_array',horiz_flux_array,dtype=np.float32,shape=(ns,nl,ne))
    edge_end=ne if edge_end is None else int(edge_end)
    if not 0<=edge_start<=edge_end<=ne:raise ValueError('scalar edge interval exceeds resident edges')
    kernel(_CUDA_SOURCE,'hexcore.cuda_transport_rrfs','scalar_edge_rrfs',kernel_cache)(((ne+127)//128,),(128,),
        (np.int32(ns),np.int32(nl),np.int32(nc),np.int32(ne),np.int32(width),np.int32(config_apply_lbcs),np.int32(edge_start),np.int32(edge_end),maske,mesh.cells_on_edge,mesh.dv_edge,stage,uh_avg,
         coefficients.n_adv_cells_for_edge,coefficients.adv_cells_for_edge,coefficients.adv_coefs,coefficients.adv_coefs_3rd,edge))
    weight=np.float32(1/3 if advance_density and rk_step==1 and config_time_integration_order==3 else .5 if advance_density and rk_step in (1,2) else 1)
    output=cp.empty_like(old);target=cp.empty_like(rho_zz_old)
    kernel(_CUDA_SOURCE,'hexcore.cuda_transport_rrfs','scalar_finish_rrfs',kernel_cache)(((nc+127)//128,),(128,),
        (np.int32(ns),np.int32(nl),np.int32(nc),np.int32(ne),np.int32(me),np.int32(solved),maskc,mesh.n_edges_on_cell,mesh.edges_on_cell,mesh.edge_sign_on_cell,context.inv_area_cell,
         uh_avg,edge,ww_avg,fzm,fzp,rdzw,old,stage,rho_zz_old,rho_zz_new,source,np.float32(dt),weight,np.float32(config_coef_3rd_order),output,target))
    return CudaScalarTransportResult(output[0] if squeezed else output,target)
