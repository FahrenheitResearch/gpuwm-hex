"""Complete fork monotonic scalar work, preserving native pool entry storage.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902
at 1d6e370e. All arithmetic stays on CUDA; Python only launches native stages.
"""
import numpy as np
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_compile_rrfs import kernel

_CUDA_SOURCE = CUDA_FTZ_HELPERS + r'''
#define C(k,c) ((k)*nc+(c))
#define E(k,e) ((k)*ne+(e))
#define S(k,j,c) (((k)*2+(j))*nc+(c))
#define Q(s,k,c) (((s)*nl+(k))*nc+(c))
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_source_rrfs(int ns,int nl,int nc,int lo,int hi,float dt,
 const float *rho,float *old,float *tend){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<lo||c>=hi)return;
 for(int k=0;k<nl;++k)for(int s=0;s<ns;++s){int q=Q(s,k,c);
 old[q]=mpas_add(old[q],mpas_div(mpas_mul(dt,tend[q]),rho[C(k,c)]));tend[q]=0.0f;}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_density_rrfs(int nl,int nc,int ne,int me,int lo,int hi,float dt,
 const int *counts,const int *edges,const float *sign,const float *dv,const float *area,
 const float *uh,const float *ww,const float *rd,const float *rho,float *out){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<lo||c>=hi)return;
 for(int k=0;k<nl;++k){float v=0.0f;
 for(int j=0;j<counts[c];++j){int e=edges[c*me+j];v=mpas_sub(v,mpas_mul(mpas_mul(mpas_mul(sign[c*me+j],uh[E(k,e)]),dv[e]),area[c]));}
 out[C(k,c)]=mpas_add(rho[C(k,c)],mpas_mul(dt,mpas_sub(v,mpas_mul(rd[k],mpas_sub(ww[C(k+1,c)],ww[C(k,c)])))));}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_carriers_rrfs(int s,int nl,int nc,int lo,int hi,int dummy,
 const float *old,const float *stage,float *o,float *n){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(c==dummy){for(int k=0;k<nl;++k){o[C(k,c)]=0.0f;n[C(k,c)]=0.0f;}}
 else if(c>=lo&&c<hi)for(int k=0;k<nl;++k){o[C(k,c)]=old[Q(s,k,c)];n[C(k,c)]=stage[Q(s,k,c)];}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_vertical_rrfs(int nl,int nc,int me,int lo,int hi,float coef,
 const int *counts,const int *neighbors,const float *o,const float *n,
 const float *ww,const float *fnm,const float *fnp,float *mn,float *mx,float *w){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<lo||c>=hi)return;
 w[C(0,c)]=0.0f;w[C(nl,c)]=0.0f;
 for(int k=0;k<nl;++k){float lower=o[C(k? k-1:1,c)],current=o[C(k,c)];
 float maxv=fmaxf(lower,current),minv=fminf(lower,current);
 if(k>0&&k<nl-1){maxv=fmaxf(maxv,o[C(k+1,c)]);minv=fminf(minv,o[C(k+1,c)]);}
 for(int j=0;j<counts[c];++j){float v=o[C(k,neighbors[c*me+j])];maxv=fmaxf(maxv,v);minv=fminf(minv,v);}
 mx[C(k,c)]=maxv;mn[C(k,c)]=minv;
 if(k==0)continue;
 float qi=n[C(k,c)],qm=n[C(k-1,c)],u=ww[C(k,c)];
 if(k==1||k==nl-1)w[C(k,c)]=mpas_mul(u,mpas_add(mpas_mul(fnm[k],qi),mpas_mul(fnp[k],qm)));
 else {float qp=n[C(k+1,c)],qmm=n[C(k-2,c)];
 float centered=mpas_div(mpas_mul(u,mpas_sub(mpas_mul(7.0f,mpas_add(qi,qm)),mpas_add(qp,qmm))),12.0f);
 float up=mpas_div(mpas_mul(mpas_mul(coef,fabsf(u)),mpas_sub(mpas_sub(qp,qmm),mpas_mul(3.0f,mpas_sub(qi,qm)))),12.0f);
 w[C(k,c)]=mpas_add(centered,up);}}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_high_edge_rrfs(int nl,int nc,int ne,int width,int lo,int hi,int solved,
 const int *coe,const int *counts,const int *cells,const float *coef,const float *coef3,
 const float *uh,const float *q,float *flux){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<lo||e>=hi)return;
 for(int k=0;k<nl;++k){float v=0.0f,u=uh[E(k,e)];
 if(coe[2*e]<solved||coe[2*e+1]<solved){
 for(int j=0;j<counts[e];++j){float weight=mpas_add(coef[e*width+j],mpas_mul(copysignf(1.0f,u),coef3[e*width+j]));
 if(counts[e]==10){float term=mpas_mul(weight,q[C(k,cells[e*width+j])]);v=j?mpas_add(v,term):term;}
 else v=mpas_add(v,mpas_mul(mpas_mul(u,weight),q[C(k,cells[e*width+j])])) ;}
 if(counts[e]==10)v=mpas_mul(u,v);}
 flux[E(k,e)]=v;}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_low_vertical_rrfs(int nl,int nc,int lo,int hi,float dt,
 const float *o,const float *rho,const float *ww,const float *rd,float *n,float *w,float *scale){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<lo||c>=hi)return;
 for(int k=0;k<nl;++k){float mass=mpas_mul(o[C(k,c)],rho[C(k,c)]);
 if(k<nl-1){float u=ww[C(k+1,c)];float f=mpas_mul(dt,mpas_add(mpas_mul(fmaxf(0.0f,u),o[C(k,c)]),mpas_mul(fminf(0.0f,u),o[C(k+1,c)])));mass=mpas_sub(mass,mpas_mul(f,rd[k]));}
 if(k>0){float u=ww[C(k,c)];float f=mpas_mul(dt,mpas_add(mpas_mul(fmaxf(0.0f,u),o[C(k-1,c)]),mpas_mul(fminf(0.0f,u),o[C(k,c)])));mass=mpas_add(mass,mpas_mul(f,rd[k]));w[C(k,c)]=mpas_sub(mpas_mul(dt,w[C(k,c)]),f);}
 n[C(k,c)]=mass;}
 for(int k=0;k<nl;++k){scale[S(k,0,c)]=mpas_mul(-rd[k],mpas_sub(fminf(0.0f,w[C(k+1,c)]),fmaxf(0.0f,w[C(k,c)])));
 scale[S(k,1,c)]=mpas_mul(-rd[k],mpas_sub(fmaxf(0.0f,w[C(k+1,c)]),fminf(0.0f,w[C(k,c)])));}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_low_edge_rrfs(int nl,int nc,int ne,int lo,int hi,int regional,float dt,
 const int *coe,const int *mask,const float *dv,const float *uh,const float *o,
 float *flux,float *upwind,float *residual){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<lo||e>=hi)return;
 for(int k=0;k<nl;++k){float u=uh[E(k,e)];float f=mpas_mul(mpas_mul(dv[e],dt),mpas_add(mpas_mul(fmaxf(0.0f,u),o[C(k,coe[2*e])]),mpas_mul(fminf(0.0f,u),o[C(k,coe[2*e+1])])));
 upwind[E(k,e)]=f;residual[E(k,e)]=mpas_sub(mpas_mul(dt,flux[E(k,e)]),f);
 if((regional&&mask[e]==5)||mask[e]==4){residual[E(k,e)]=0.0f;flux[E(k,e)]=f;}}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_scale_rrfs(int nl,int nc,int ne,int me,int lo,int hi,
 const int *counts,const int *edges,const float *sign,const float *area,const float *upwind,
 const float *residual,const float *mn,const float *mx,const float *rho,float *n,float *scale){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<lo||c>=hi)return;
 for(int k=0;k<nl;++k){float mass=n[C(k,c)],si=scale[S(k,0,c)],so=scale[S(k,1,c)];
 for(int j=0;j<counts[c];++j){int e=edges[c*me+j];float sg=sign[c*me+j];
 mass=mpas_sub(mass,mpas_mul(mpas_mul(sg,upwind[E(k,e)]),area[c]));
 so=mpas_sub(so,mpas_mul(fmaxf(0.0f,mpas_mul(sg,residual[E(k,e)])),area[c]));
 si=mpas_sub(si,mpas_mul(fminf(0.0f,mpas_mul(sg,residual[E(k,e)])),area[c]));}
 n[C(k,c)]=mass;
 scale[S(k,0,c)]=fminf(1.0f,fmaxf(0.0f,mpas_div(mpas_sub(mpas_mul(mx[C(k,c)],rho[C(k,c)]),mass),mpas_add(si,1.e-20f))));
 scale[S(k,1,c)]=fminf(1.0f,fmaxf(0.0f,mpas_div(mpas_sub(mpas_mul(mn[C(k,c)],rho[C(k,c)]),mass),mpas_sub(so,1.e-20f))));}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_limit_edge_rrfs(int nl,int nc,int ne,int lo,int hi,int solved,int regional,float dt,
 const int *coe,const int *mask,const float *dv,const float *uh,const float *o,const float *scale,float *flux){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e<lo||e>=hi)return;int c0=coe[2*e],c1=coe[2*e+1];
 if(c0>=solved&&c1>=solved)return;
 for(int k=0;k<nl;++k){float u=uh[E(k,e)];float f=mpas_mul(mpas_mul(dv[e],dt),mpas_add(mpas_mul(fmaxf(0.0f,u),o[C(k,c0)]),mpas_mul(fminf(0.0f,u),o[C(k,c1)])));
 float v=mpas_sub(mpas_mul(dt,flux[E(k,e)]),f);if((regional&&mask[e]==5)||mask[e]==4)v=0.0f;
 flux[E(k,e)]=mpas_add(mpas_mul(fmaxf(0.0f,v),fminf(scale[S(k,1,c0)],scale[S(k,0,c1)])),mpas_mul(fminf(0.0f,v),fminf(scale[S(k,0,c0)],scale[S(k,1,c1)])));}
}
// MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
extern "C" __global__ void mono_finish_rrfs(int s,int nl,int nc,int ne,int me,int lo,int hi,int copylo,int copyhi,
 const int *counts,const int *edges,const int *mask,const float *sign,const float *area,const float *rd,
 const float *rho,const float *scale,const float *flux,float *w,float *n,float *stage){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
 if(c>=lo&&c<hi){for(int k=1;k<nl;++k){float v=w[C(k,c)];w[C(k,c)]=mpas_add(mpas_mul(fmaxf(0.0f,v),fminf(scale[S(k-1,1,c)],scale[S(k,0,c)])),mpas_mul(fminf(0.0f,v),fminf(scale[S(k,1,c)],scale[S(k-1,0,c)])));}
 for(int k=0;k<nl;++k){float mass=n[C(k,c)];for(int j=0;j<counts[c];++j){int e=edges[c*me+j];mass=mpas_sub(mass,mpas_mul(mpas_mul(sign[c*me+j],flux[E(k,e)]),area[c]));}
 n[C(k,c)]=mpas_div(mpas_add(mass,mpas_mul(-rd[k],mpas_sub(w[C(k+1,c)],w[C(k,c)]))),rho[C(k,c)]);}}
 if(c>=copylo&&c<copyhi&&mask[c]<=2)for(int k=0;k<nl;++k)stage[Q(s,k,c)]=fmaxf(0.0f,n[C(k,c)]);
}
'''

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4180-4902 at 1d6e370e.
def advance_monotonic_work_cuda_rrfs(a, *, config_apply_lbcs, kernel_cache=None, halo_exchange=None):
    """Native numeric argument mapping, zero-based device topology and full entry buffers.

    Native block/state/procedure handles are represented by the supplied halo
    adapter. MPI1 needs no exchange; multi-rank callers must supply the adapter.
    """
    ns,nl,nc = map(int,a['scalars_old'].shape)
    ne=int(a['uhavg'].shape[1]);me=int(a['edgesoncell'].shape[1]);width=int(a['adv_coefs'].shape[1])
    cl,ch,sl,sh,el,eh=(int(a[n]) for n in ('cellstart','cellend','cellsolvestart','cellsolveend','edgestart','edgeend'))
    dummy=int(a['ncells']);solved=int(a['ncellssolve'])
    if not (0<=cl<=ch<=dummy<nc and 0<=sl<=sh<=dummy and 0<=el<=eh<=int(a['nedges'])<ne):
        raise ValueError('monotonic work intervals exceed native physical and allocation bounds')
    def launch(name, count, *args):
        kernel(_CUDA_SOURCE,'hexcore.cuda_monotonic_rrfs',name,kernel_cache)(((count+127)//128,),(128,),args)
    I=np.int32;F=np.float32
    base=(I(nl),I(nc));ebase=(*base,I(ne));cb=(*ebase,I(me));dt=F(a['dt'])
    launch('mono_source_rrfs',nc,I(ns),*base,I(sl),I(sh),dt,a['rho_zz_old'],a['scalars_old'],a['scalar_tend'])
    if halo_exchange is not None:halo_exchange.round_transport(a['scalars_old'][...,:dummy])
    advance=bool(a.get('advance_density',True))
    target=a['rho_zz_new']
    if advance:
        target=a['rho_zz_int']
        launch('mono_density_rrfs',nc,*cb,I(sl),I(sh),dt,a['nedgesoncell'],a['edgesoncell'],a['edgesoncell_sign'],a['dvedge'],a['invareacell'],a['uhavg'],a['wwavg'],a['rdnw'],a['rho_zz_old'],target)
    for s in range(ns):
        launch('mono_carriers_rrfs',nc,I(s),*base,I(cl),I(ch),I(dummy),a['scalars_old'],a['scalars_new'],a['scalar_old'],a['scalar_new'])
        launch('mono_vertical_rrfs',nc,*base,I(me),I(sl),I(sh),F(a['coef_3rd_order']),a['nedgesoncell'],a['cellsoncell'],a['scalar_old'],a['scalar_new'],a['wwavg'],a['fnm'],a['fnp'],a['s_min'],a['s_max'],a['wdtn'])
        launch('mono_high_edge_rrfs',ne,*ebase,I(width),I(el),I(eh),I(solved),a['cellsonedge'],a['nadvcellsforedge'],a['advcellsforedge'],a['adv_coefs'],a['adv_coefs_3rd'],a['uhavg'],a['scalar_new'],a['flux_arr'])
        launch('mono_low_vertical_rrfs',nc,*base,I(sl),I(sh),dt,a['scalar_old'],a['rho_zz_old'],a['wwavg'],a['rdnw'],a['scalar_new'],a['wdtn'],a['scale_arr'])
        launch('mono_low_edge_rrfs',ne,*ebase,I(el),I(eh),I(config_apply_lbcs),dt,a['cellsonedge'],a['bdymaskedge'],a['dvedge'],a['uhavg'],a['scalar_old'],a['flux_arr'],a['flux_upwind_tmp'],a['flux_tmp'])
        launch('mono_scale_rrfs',nc,*cb,I(sl),I(sh),a['nedgesoncell'],a['edgesoncell'],a['edgesoncell_sign'],a['invareacell'],a['flux_upwind_tmp'],a['flux_tmp'],a['s_min'],a['s_max'],target,a['scalar_new'],a['scale_arr'])
        if halo_exchange is not None:halo_exchange.round_fct_scale(a['scale_arr'][:,0,:dummy],a['scale_arr'][:,1,:dummy])
        launch('mono_limit_edge_rrfs',ne,*ebase,I(el),I(eh),I(solved),I(config_apply_lbcs),dt,a['cellsonedge'],a['bdymaskedge'],a['dvedge'],a['uhavg'],a['scalar_old'],a['scale_arr'],a['flux_arr'])
        launch('mono_finish_rrfs',nc,I(s),*cb,I(sl),I(sh),I(cl),I(ch),a['nedgesoncell'],a['edgesoncell'],a['bdymaskcell'],a['edgesoncell_sign'],a['invareacell'],a['rdnw'],target,a['scale_arr'],a['flux_arr'],a['wdtn'],a['scalar_new'],a['scalars_new'])
    return a

# MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:4037-4902 at 1d6e370e.
def advance_scalars_monotonic_cuda_rrfs(mesh, context, scalar_old, scalar_stage,
    rho_zz_old, rho_zz_new, uh_avg, ww_avg, dt, *, coefficients, fzm, fzp, rdzw,
    config_coef_3rd_order, scalar_tendency=None, advance_density=False,
    cell_solve_count=None, config_apply_lbcs=False, bdy_mask_cell=None,
    bdy_mask_edge=None, garbage_cell=None, kernel_cache=None, halo_exchange=None,
    native_work=None, **unused):
    """Production driver adapter to the complete native work boundary.

    native_work supplies original pool buffers when available. Otherwise scratch
    is allocated explicitly. Global carriers gain an inert native dummy column;
    regional carriers already contain it. Untouched entry words are preserved.
    """
    import cupy as cp
    from .cuda_transport import CudaScalarTransportResult
    from .cuda_backend.containers import require_resident_array
    old=require_resident_array('scalar_old',scalar_old,dtype=np.float32)
    stage=require_resident_array('scalar_stage',scalar_stage,dtype=np.float32,shape=old.shape)
    squeezed=old.ndim==2
    if squeezed:old=old[None,:,:];stage=stage[None,:,:]
    ns,nl,original_nc=map(int,old.shape);original_ne=int(uh_avg.shape[1])
    physical_nc=original_nc if garbage_cell is None else int(garbage_cell)
    physical_ne=original_ne if garbage_cell is None else original_ne-1
    if physical_nc not in (original_nc,original_nc-1):
        raise ValueError('native dummy must follow the physical cell allocation')
    nc,ne=physical_nc+1,physical_ne+1
    solved=physical_nc if cell_solve_count is None else int(cell_solve_count)
    # Device copies and layout transport, with no host field arithmetic.
    def padded(value, extent, axis=-1):
        value=cp.ascontiguousarray(value)
        if value.shape[axis]==extent:return value
        if value.shape[axis]!=extent-1:raise ValueError('native pool extent differs from physical plus dummy')
        shape=list(value.shape);shape[axis]=extent
        out=cp.zeros(shape,dtype=value.dtype)
        index=[slice(None)]*value.ndim;index[axis]=slice(0,extent-1)
        out[tuple(index)]=value
        return out
    a={} if native_work is None else dict(native_work)
    a.update(ncells=physical_nc,nedges=physical_ne,ncellssolve=solved,
        cellstart=0,cellend=physical_nc,cellsolvestart=0,cellsolveend=solved,
        edgestart=0,edgeend=physical_ne,dt=dt,coef_3rd_order=config_coef_3rd_order,
        advance_density=advance_density,scalars_old=padded(old,nc),
        scalars_new=padded(stage.copy(),nc),rho_zz_old=padded(rho_zz_old,nc),
        rho_zz_new=padded(rho_zz_new,nc),uhavg=padded(uh_avg,ne),wwavg=padded(ww_avg,nc),
        fnm=fzm,fnp=fzp,rdnw=rdzw,invareacell=padded(context.inv_area_cell,nc),
        dvedge=padded(mesh.dv_edge,ne),cellsonedge=padded(mesh.cells_on_edge,ne,axis=0),
        cellsoncell=padded(mesh.cells_on_cell,nc,axis=0),edgesoncell=padded(mesh.edges_on_cell,nc,axis=0),
        nedgesoncell=padded(mesh.n_edges_on_cell,nc),edgesoncell_sign=padded(mesh.edge_sign_on_cell,nc,axis=0),
        nadvcellsforedge=padded(coefficients.n_adv_cells_for_edge,ne),
        advcellsforedge=padded(coefficients.adv_cells_for_edge,ne,axis=0),
        adv_coefs=padded(coefficients.adv_coefs,ne,axis=0),adv_coefs_3rd=padded(coefficients.adv_coefs_3rd,ne,axis=0),
        scalar_tend=cp.zeros((ns,nl,nc),'f4') if scalar_tendency is None else padded(scalar_tendency.reshape(old.shape),nc),
        bdymaskcell=cp.zeros(nc,'i4') if bdy_mask_cell is None else padded(bdy_mask_cell,nc),
        bdymaskedge=cp.zeros(ne,'i4') if bdy_mask_edge is None else padded(bdy_mask_edge,ne))
    for name,shape in dict(scalar_old=(nl,nc),scalar_new=(nl,nc),s_min=(nl,nc),s_max=(nl,nc),
        wdtn=(nl+1,nc),scale_arr=(nl,2,nc),flux_arr=(nl,ne),flux_upwind_tmp=(nl,ne),flux_tmp=(nl,ne),rho_zz_int=(nl,nc)).items():
        if name not in a:a[name]=cp.zeros(shape,'f4')
        require_resident_array(name,a[name],dtype=np.float32,shape=shape)
    advance_monotonic_work_cuda_rrfs(a,config_apply_lbcs=config_apply_lbcs,kernel_cache=kernel_cache,halo_exchange=halo_exchange)
    old[...] = a['scalars_old'][...,:original_nc]
    if scalar_tendency is not None:scalar_tendency[...] = a['scalar_tend'][...,:original_nc].reshape(scalar_tendency.shape)
    result=a['scalars_new'][...,:original_nc]
    target=a['rho_zz_int'] if advance_density else a['rho_zz_new']
    return CudaScalarTransportResult(cp.ascontiguousarray(result[0] if squeezed else result),
                                     cp.ascontiguousarray(target[...,:original_nc]))
