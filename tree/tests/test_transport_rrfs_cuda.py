"""Full fork scalar-work replay for the added GSL sources and w boundary.

RRFSx/MPAS-Model src/core_atmosphere/dynamics/mpas_atm_time_integration.F:3615-4023,7398-7446
at 1d6e370e. These receipts establish subsystem parity only.
"""
import ctypes as ct
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rrfs_acoustic_native import NativeAcoustic

pytestmark = pytest.mark.cuda

@pytest.mark.parametrize('regional',[False,True])
@pytest.mark.parametrize('filter4',[0.,.037])
@pytest.mark.parametrize('subnormal',[False,True])
@pytest.mark.parametrize('eddy_mix',[False,True])
@pytest.mark.parametrize('advance_density',[False,True])
@pytest.mark.parametrize('order',[2,3])
def test_full_native_scalar_sources(regional,filter4,subnormal,eddy_mix,advance_density,order):
    cp=pytest.importorskip('cupy')
    from hexcore.cuda_compile_rrfs import RRFSKernelCache
    cache=RRFSKernelCache()
    from hexcore.cuda_transport_rrfs import apply_gsl_scalar_mixing_cuda_rrfs, advance_scalars_standard_cuda_rrfs
    path=os.environ.get('RRFS_TRANSPORT_ORACLE')
    if not path: pytest.skip('RRFS_TRANSPORT_ORACLE must name the strict native library')
    nl=7;n=5;nc=ne=n+1;ns=3;me=2
    native=NativeAcoustic(path,nl,ns,me)
    native.library.set_regional(ct.c_int(regional))
    rng=np.random.default_rng(4032)
    def f(shape,low=.2,high=1.3):return np.asfortranarray(rng.uniform(low,high,shape).astype('f4'))
    coe=np.full((2,ne),nc,'i4',order='F');eoc=np.zeros((me,nc),'i4',order='F');sign=np.zeros((me,nc),'f4',order='F')
    for e in range(n):coe[:,e]=e+1,(e+1)%n+1
    for c in range(n):eoc[:,c]=c+1,(c-1)%n+1;sign[:,c]=1,-1
    a=dict(ncells=n,nedges=n,num_scalars=ns,dt=.375,cellstart=1,cellend=n,edgestart=1,edgeend=n,cellsolvestart=1,cellsolveend=n,
        coef_3rd_order=.25,scalar_old=f((ns,nl,nc)),scalar_new=f((ns,nl,nc)),rho_zz_old=f((nl,nc)),rho_zz_new=f((nl,nc)),
        uhavg=np.zeros((nl,ne),'f4',order='F'),wwavg=np.zeros((nl+1,nc),'f4',order='F'),dvedge=f((ne,)),cellsonedge=coe,
        edgesoncell=eoc,nedgesoncell=np.array([2]*n+[0],'i4'),edgesoncell_sign=sign,invareacell=f((nc,),.003,.009),invdcedge=f((ne,),.05,.08),
        meshscalingdel4=f((ne,)),meshscalingdel2=f((ne,)),scalar_tend_save=f((ns,nl,nc),-.02,.02),fnm=f((nl,)),fnp=f((nl,)),rdnw=f((nl,)),
        bdymaskcell=np.array([0,1,5,6,7,7] if regional else [0]*nc,'i4'),bdymaskedge=np.array([0,3,4,5,6,7] if regional else [0]*ne,'i4'),
        nadvcellsforedge=np.zeros(ne,'i4'),advcellsforedge=np.ones((15,ne),'i4',order='F'),adv_coefs=np.zeros((15,ne),'f4',order='F'),adv_coefs_3rd=np.zeros((15,ne),'f4',order='F'),
        kdiff=f((nl,nc),0,.5),rho_edge=f((nl,ne),.3,1.5),horiz_flux_arr=f((ns,nl,ne),-.1,.1),rk_step=1,config_time_integration_order=order,advance_density=int(advance_density),h_scalar_filter4=filter4,scalar_eddy_mix=int(eddy_mix))
    if subnormal:
        a['scalar_old'][:]=np.array(64,dtype='u4').view('f4');a['scalar_tend_save'][:]=np.float32(0)
    def gpu(x):return cp.ascontiguousarray(cp.asarray(x))
    mesh=SimpleNamespace(n_edges=ne,max_edges=me,n_edges_on_cell=gpu(a['nedgesoncell']),edges_on_cell=gpu((eoc-1).T),cells_on_edge=gpu((coe-1).T),edge_sign_on_cell=gpu(sign.T),dv_edge=gpu(a['dvedge']))
    ctx=SimpleNamespace(inv_area_cell=gpu(a['invareacell']),inv_dc_edge=gpu(a['invdcedge']))
    tend=gpu(a['scalar_tend_save']);old=gpu(a['scalar_old'])
    if eddy_mix:
        apply_gsl_scalar_mixing_cuda_rrfs(mesh,ctx,old,tend,kdiff=gpu(a['kdiff']),rho_edge=gpu(a['rho_edge']),mesh_scaling_del2=gpu(a['meshscalingdel2']),mesh_scaling_del4=gpu(a['meshscalingdel4']),h_scalar_filter4=filter4,cell_count=n,cell_solve_count=n,kernel_cache=cache)
    a['uhavg'][:]=f((nl,ne),-.03,.03)
    a['wwavg'][1:nl,:]=f((nl-1,nc),-.003,.003)
    a['nadvcellsforedge'][:n]=[10,2,3,1,10]
    for e in range(n):
        for j in range(a['nadvcellsforedge'][e]):
            a['advcellsforedge'][j,e]=j%n+1
            a['adv_coefs'][j,e]=np.float32(.1+.01*j)
            a['adv_coefs_3rd'][j,e]=np.float32(.004*j)
    coefficients=SimpleNamespace(adv_coefs=gpu(a['adv_coefs'].T),adv_coefs_3rd=gpu(a['adv_coefs_3rd'].T),n_adv_cells_for_edge=gpu(a['nadvcellsforedge']),adv_cells_for_edge=gpu((a['advcellsforedge']-1).T))
    a['scalar_new'][:]=a['scalar_old'];stage_gpu=gpu(a['scalar_new'])
    flux_gpu=gpu(a['horiz_flux_arr'])
    comparisons=[]
    for stage in (1,2,3):
        a['rk_step']=stage
        native.call('atm_advance_scalars_work',a)
        result=advance_scalars_standard_cuda_rrfs(mesh,ctx,old,stage_gpu,gpu(a['rho_zz_old']),gpu(a['rho_zz_new']),gpu(a['uhavg']),gpu(a['wwavg']),a['dt'],
            coefficients=coefficients,fzm=gpu(a['fnm']),fzp=gpu(a['fnp']),rdzw=gpu(a['rdnw']),rk_step=stage,config_time_integration_order=order,config_coef_3rd_order=a['coef_3rd_order'],
            scalar_tendency=tend,advance_density=advance_density,config_apply_lbcs=regional,bdy_mask_cell=gpu(a['bdymaskcell']),bdy_mask_edge=gpu(a['bdymaskedge']),cell_solve_count=n,kernel_cache=cache,horiz_flux_array=flux_gpu,edge_end=n)
        stage_gpu=result.scalars
        actual=cp.asnumpy(stage_gpu);expected=a['scalar_new']
        mismatches=int(np.count_nonzero(actual.view('u4')!=expected.view('u4')))
        if mismatches:
            key=tuple(np.argwhere(actual.view('u4')!=expected.view('u4'))[0]);print('scalar',stage,key,actual[key],expected[key])
        comparisons.append(dict(stage=stage,field='scalars',words=actual.size,differing_words=mismatches))
        assert mismatches==0
        got=cp.asnumpy(tend);ref=a['scalar_tend_save']
        diff=int(np.count_nonzero(got.view('u4')!=ref.view('u4')))
        comparisons.append(dict(stage=stage,field='scalar_tend_save',words=got.size,differing_words=diff))
        if diff:
            idx=tuple(np.argwhere(got.view('u4')!=ref.view('u4'))[0]);print(idx,got[idx],ref[idx])
        assert diff==0
        got=cp.asnumpy(flux_gpu);ref=a['horiz_flux_arr']
        diff=int(np.count_nonzero(got.view('u4')!=ref.view('u4')))
        comparisons.append(dict(stage=stage,field='horiz_flux_arr',words=got.size,differing_words=diff))
        assert diff==0
    out=os.environ.get('RRFS_RECEIPTS')
    if out:Path(out,f'gsl-regional{int(regional)}-filter{filter4}-subnormal{int(subnormal)}-eddy{int(eddy_mix)}-density{int(advance_density)}-order{order}.json').write_text(json.dumps(dict(tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,regional=regional,filter4=filter4,subnormal=subnormal,eddy_mix=eddy_mix,advance_density=advance_density,order=order,comparisons=comparisons,compile=cache.compile_manifest()),indent=2)+'\n')

@pytest.mark.parametrize('option',['zero','nearest'])
def test_full_native_w_boundary(option):
    cp=pytest.importorskip('cupy')
    from hexcore.cuda_regional_rrfs import apply_w_boundary_cuda_rrfs
    path=os.environ.get('RRFS_TRANSPORT_ORACLE')
    if not path:pytest.skip('RRFS_TRANSPORT_ORACLE must name the strict native library')
    nl=7;nc=9;n=nc-1
    native=NativeAcoustic(path,nl,1,2)
    w=np.asfortranarray(np.random.default_rng(532).normal(size=(nl+1,nc)).astype('f4'))
    mask=np.array([0,1,4,5,6,7,8,6,7],'i4');nearest=np.array([1,2,3,4,2,3,4,1,2],'i4')
    device=cp.ascontiguousarray(cp.asarray(w))
    apply_w_boundary_cuda_rrfs(device,cp.asarray(mask),cp.asarray(nearest-1),config_lbc_w=option,cell_solve_start=1,cell_solve_end=n)
    native.call('atm_zero_gradient_w_bdy_work',dict(w=w,bdymaskcell=mask,nearestrelaxationcell=nearest,ncells=n,cellsolvestart=2,cellsolveend=n,option=int(option=='nearest')))
    diff=int(np.count_nonzero(cp.asnumpy(device).view('u4')!=w.view('u4')))
    assert diff==0
    out=os.environ.get('RRFS_RECEIPTS')
    if out:Path(out,f'w-boundary-{option}.json').write_text(json.dumps(dict(tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,option=option,words=w.size,differing_words=diff),indent=2)+'\n')
