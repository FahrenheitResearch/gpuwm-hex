"""Replay complete native fork work routines; GPU only (through HEX_GPU_QUEUE on a shared host).

Source: RRFSx/MPAS-Model src/core_atmosphere/dynamics/mpas_atm_time_integration.F:2225-2366,2646-2984
at 1d6e370e. This is subsystem proof, not a whole-forecast receipt.
"""
from dataclasses import fields
import json, os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from rrfs_acoustic_native import NativeAcoustic

pytestmark = pytest.mark.cuda

@pytest.mark.parametrize('regional', [False,True])
@pytest.mark.parametrize('epssm,dts', [(0.0,0.625),(0.1,3.7),(0.7,0.625)])
@pytest.mark.parametrize('subnormal', [False,True])
def test_native_fork_acoustic(regional,epssm,dts,subnormal):
    cp=pytest.importorskip('cupy')
    from hexcore.cuda_compile_rrfs import RRFSKernelCache
    from hexcore.cuda_acoustic import CudaAcousticForcing,CudaAcousticState
    from hexcore.cuda_acoustic_rrfs import compute_vertical_implicit_coefficients_cuda_rrfs,advance_acoustic_step_cuda_rrfs
    lib=os.environ.get('RRFS_ACOUSTIC_ORACLE')
    if not lib: pytest.skip('RRFS_ACOUSTIC_ORACLE must name the standalone strict native .so')
    cache=RRFSKernelCache()
    nl=7; n=5; nc=n+1; ne=n+1; me=2
    native=NativeAcoustic(lib,nl,2,me)
    rng=np.random.default_rng(243)
    def cell(lo,hi): return np.asfortranarray(rng.uniform(lo,hi,(nl,nc)).astype('f'))
    def iface(lo,hi): return np.asfortranarray(rng.uniform(lo,hi,(nl+1,nc)).astype('f'))
    def vert(lo,hi): return rng.uniform(lo,hi,nl).astype('f')
    a=dict(zz=cell(.7,1.5),cqw=cell(.8,1.0),p=cell(.6,1.0),t=cell(270,310),rb=cell(.4,1.2),rtb=cell(180,350),pb=cell(.6,1),rt=cell(-.2,.2),qtot=cell(0,.02),rdzw=vert(.0004,.002),rdzu=vert(.0004,.002),fzm=vert(.3,.7))
    a['fzp']=np.float32(1)-a['fzm']
    for name in ['cofwr','cofwz','cofwt','a_tri','alpha_tri','gamma_tri']: a[name]=cell(-.2,.2)
    a['coftz']=iface(-.2,.2); a['cofrz']=np.zeros(nl,'f')
    a.update(ncells=n,moist_start=1,moist_end=2,dts=dts,epssm=epssm,cellstart=1,cellend=n,edgestart=1,edgeend=n,cellsolvestart=1,cellsolveend=n,edgesolvestart=1,edgesolveend=n,scalars=np.zeros((2,nl,nc),'f',order='F'))
    from hexcore.cuda_acoustic import CudaVerticalImplicitCoefficients
    entry={name:cp.ascontiguousarray(cp.asarray(a[name])) if name in a else cp.zeros((nl,nc),'f')
           for name in CudaVerticalImplicitCoefficients.__dataclass_fields__}
    native.call('atm_compute_vert_imp_coefs_work',a)
    kwargs={name:cp.asarray(a[alias]) for name,alias in dict(zz='zz',cqw='cqw',pressure='p',theta='t',rho_base='rb',rho_theta_base='rtb',pressure_base='pb',rho_theta_perturbation='rt',qtot='qtot',rdzw='rdzw',rdzu='rdzu',fzm='fzm',fzp='fzp').items()}
    # Device buffers are C order (level,owner), including the dummy halo cell.
    kwargs={k:cp.ascontiguousarray(v) for k,v in kwargs.items()}
    coefs=compute_vertical_implicit_coefficients_cuda_rrfs(dts=dts,epssm=epssm,cell_solve_count=n,kernel_cache=cache,initial_coefficients=CudaVerticalImplicitCoefficients(**entry),**kwargs)
    checks=[]
    def compare(name,actual,want):
        got=cp.asnumpy(actual)
        ref=want
        diff=np.count_nonzero(got.view('u4')!=ref.view('u4'))
        checks.append(dict(field=name,words=got.size,differing_words=int(diff)))
        if diff:
            idx=np.argwhere(got.view('u4')!=ref.view('u4'))[0]
            key=tuple(idx); print(name,key,got[key],ref[key],hex(int(got.view('u4')[key])),hex(int(ref.view('u4')[key])))
        assert diff==0,name
    for name in ['cofwr','cofwz','coftz','cofwt','cofrz','a_tri','alpha_tri','gamma_tri']: compare(name,getattr(coefs,name),a[name])
    coe=np.zeros((2,ne),'i',order='F');eoc=np.zeros((me,nc),'i',order='F');sign=np.zeros((me,nc),'f',order='F')
    for e in range(n): coe[:,e]=[e+1,(e+1)%n+1]
    coe[:,n]=nc
    for c in range(n): eoc[:,c]=[c+1,(c-1)%n+1];sign[:,c]=[1,-1]
    s=dict(rho_zz=cell(.4,1.2),theta_m=cell(270,310),ru_p=cell(-.03,.03),rw_p=iface(-.03,.03),rtheta_pp=cell(-.03,.03),rtheta_pp_old=cell(-.03,.03),exner=a['p'],cqu=cell(.8,1),ruavg=cell(-.03,.03),wwavg=iface(-.03,.03),rho_pp=cell(-.03,.03),zxu=cell(-.001,.001),dss=cell(0,.1),tend_ru=cell(-.003,.003),tend_rho=cell(-.0003,.0003),tend_rt=cell(-.003,.003),tend_rw=iface(-.003,.003),w=iface(-.2,.2),ru=cell(-.2,.2),ru_save=cell(-.2,.2),rw=iface(-.2,.2),rw_save=iface(-.2,.2))
    if subnormal:
        for key in ['tend_ru','tend_rho','tend_rt','tend_rw']:
            s[key][:]=np.float32(0);s[key][0,0]=np.array(16,dtype='u4').view('f4')
    mask=np.zeros(nc,'f');maskedge=np.zeros(ne,'f')
    if regional: mask[0]=1;maskedge[0]=1;maskedge[1]=.25
    metrics=dict(dvedge=rng.uniform(.2,1,ne).astype('f'),invdcedge=rng.uniform(.0001,.0007,ne).astype('f'),invareacell=rng.uniform(.00001,.0001,nc).astype('f'))
    s.update(metrics,dcEdge=np.float32(1)/metrics['invdcedge'],cellsOnEdge=coe,edgesOnCell=eoc,nEdgesOnCell=np.array([2]*n+[0],dtype='i'),edgesOnCell_sign=sign,specZoneMaskCell=mask,specZoneMaskEdge=maskedge,zgrid=iface(0,30000),cf1=0.,cf2=0.,cf3=0.,ncells=n,nedges=n,ncellssolve=n,vertexstart=1,vertexend=0,vertexsolvestart=1,vertexsolveend=0)
    a.update({k.lower():v for k,v in s.items()})
    def gpu(value): return cp.ascontiguousarray(cp.asarray(value))
    state=CudaAcousticState(**{f.name:gpu(s[{'ru_avg':'ruavg','ww_avg':'wwavg'}.get(f.name,f.name)]) for f in fields(CudaAcousticState)})
    forcing=CudaAcousticForcing(**{f.name:gpu(s[f.name] if f.name in s else a[f.name]) for f in fields(CudaAcousticForcing)})
    mesh=SimpleNamespace(cells_on_edge=gpu((coe-1).T),edges_on_cell=gpu((eoc-1).T),n_edges_on_cell=gpu(s['nEdgesOnCell']),edge_sign_on_cell=gpu(sign.T),dv_edge=gpu(metrics['dvedge']))
    context=SimpleNamespace(inv_dc_edge=gpu(metrics['invdcedge']),inv_area_cell=gpu(metrics['invareacell']))
    for step in range(1,5):
        a['small_step']=step
        native.call('atm_advance_acoustic_step_work',a)
        advance_acoustic_step_cuda_rrfs(mesh,state,forcing,coefs,context=context,dts=dts,small_step=step,epssm=epssm,cell_solve_count=n,cell_end=n,edge_end=n,spec_zone_mask_cell=gpu(mask),spec_zone_mask_edge=gpu(maskedge),fzm=kwargs['fzm'],fzp=kwargs['fzp'],rdzw=kwargs['rdzw'],kernel_cache=cache)
        for f in fields(CudaAcousticState): compare(f'{step}:{f.name}',getattr(state,f.name),a[{'ru_avg':'ruavg','ww_avg':'wwavg'}.get(f.name,f.name)])
    receipt_dir=os.environ.get('RRFS_RECEIPTS')
    if receipt_dir:
        p=Path(receipt_dir)/f'acoustic-regional{int(regional)}-eps{epssm}-dt{dts}-subnormal{int(subnormal)}.json'
        p.write_text(json.dumps(dict(tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,regional=regional,epssm=epssm,dts=dts,subnormal=subnormal,comparisons=checks,compile=cache.compile_manifest()),indent=2)+'\n')
