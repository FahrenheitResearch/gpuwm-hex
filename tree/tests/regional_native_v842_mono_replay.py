"""Unchanged MPAS regional limited transport, complete real-call replay.

MPAS-A v8.4.2 mpas_atm_time_integration.F:5094-5785. Single-process
framework/halo stubs supply the owned domain without altering numerics.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import re
import numpy as np


def prepare(source_root,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=Path(source_root)/'src/core_atmosphere/dynamics/mpas_atm_time_integration.F'
    source=path.read_text()
    match=re.search(r'^   subroutine atm_advance_scalars_mono_work\(.*?^   end subroutine atm_advance_scalars_mono_work',source,re.M|re.S)
    assert match
    routine=match.group()
    arrays=['old','stage','rho0','rho1','uh','ww','dv','cells','neighbors','edges','counts','signs','area','tendency','fnm','fnp','rdnw','maskc','maske','advcounts','advcells','adv','adv3','target']
    declarations='''integer(c_int),value :: nc,ne,nl,ns,me,width,mono
real(c_float),value :: dt,coef
real(c_float) :: old(ns,nl,nc+1),stage(ns,nl,nc+1),rho0(nl,nc+1),rho1(nl,nc+1)
real(c_float) :: uh(nl,ne+1),ww(nl+1,nc+1),dv(ne+1),signs(me,nc+1),area(nc+1)
real(c_float) :: tendency(ns,nl,nc+1),fnm(nl+1),fnp(nl+1),rdnw(nl+1),target(nl,nc+1)
real(c_float) :: adv(width,ne+1),adv3(width,ne+1)
integer(c_int) :: cells(2,ne+1),neighbors(me,nc+1),edges(me,nc+1),counts(nc+1),maskc(nc+1),maske(ne+1)
integer(c_int) :: advcounts(ne+1),advcells(width,ne+1)
real(c_float) :: scalar_old(nl,nc+1),scalar_new(nl,nc+1),smax(nl,nc+1),smin(nl,nc+1)
real(c_float) :: wdtn(nl+1,nc+1),scale(nl,2,nc+1),flux(nl,ne+1),upwind(nl,ne+1),tmp(nl,ne+1)
type(block_type) :: block
type(mpas_pool_type) :: state'''
    code=['#define MPAS_ACC_TIMER_START(X)','#define MPAS_ACC_TIMER_STOP(X)',
          'module mpas_atm_dimensions','integer :: nVertLevels','end module',
          'module native_mono','use iso_c_binding','integer,parameter :: RKIND=kind(1.0),nRelaxZone=5,nSpecZone=2,MPAS_LOG_CRIT=1',
          'logical :: config_apply_lbcs=.true.,config_monotonic=.true.',
          'type block_type','integer :: domain=1','end type',
          'type mpas_pool_type','integer :: unused=0','end type',
          'abstract interface','subroutine halo_exchange_routine(domain,name,gpu)',
          'integer,intent(in) :: domain','character(len=*),intent(in) :: name','logical,intent(in) :: gpu','end subroutine','end interface',
          'contains',routine,
          'subroutine exchange(domain,name,gpu)','integer,intent(in) :: domain','character(len=*),intent(in) :: name','logical,intent(in) :: gpu','end subroutine',
          'subroutine mpas_log_write(message,messageType)','character(len=*),intent(in)::message','integer,intent(in)::messageType','error stop "missing density scratch"','end subroutine',
          'subroutine replay(nc,ne,nl,ns,me,width,mono,dt,coef,'+','.join(arrays)+') bind(c)',
          'use mpas_atm_dimensions',declarations,
          'nVertLevels=nl','config_monotonic=mono/=0',
          'scalar_old=0.;scalar_new=0.;smax=0.;smin=0.;wdtn=0.;scale=0.;flux=0.;upwind=0.;tmp=0.',
          "call atm_advance_scalars_mono_work('scalars',block,state,nc,ne,ns,dt, &",
          '1,nc,1,ne,1,nc,coef,nc,uh,ww,tendency,rho0,rho1,old,stage, &',
          'area,dv,cells,neighbors,edges,signs,counts,fnm,fnp,rdnw,advcounts, &',
          'advcells,adv,adv3,scalar_old,scalar_new,smax,smin,wdtn,scale,flux,upwind,tmp, &',
          'maskc,maske,.false.,exchange,.true.,target)','end subroutine','end module']
    (out/'oracle.F90').write_text('\n'.join(code)+'\n')
    flags='-cpp -DDO_PHYSICS -O3 -fPIC -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fcheck=bounds'
    (out/'Makefile').write_text('FC = gfortran\nFFLAGS = '+flags+'\nall: libmono.so\nlibmono.so: oracle.F90\n\t$(FC) $(FFLAGS) -shared $< -o $@\n')
    (out/'source-pins.json').write_text(json.dumps(dict(file=str(path),routine='atm_advance_scalars_mono_work',first_line=source[:match.start()].count('\n')+1,last_line=source[:match.end()].count('\n')+1,routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),flags=flags,arguments=arrays),indent=2)+'\n')


def install_real_capture(build,out,limit=120):
    """Compare all physical output and source-updated baseline words of 24 h."""
    import cupy as cp
    from hexcore import cuda_transport_v841 as transport
    from hexcore.cuda_physics_v841 import prepare_mpas_transport_source_baseline_v842
    build,out=Path(build),Path(out);out.mkdir(parents=True,exist_ok=True)
    pins=json.loads((build/'source-pins.json').read_text());library=ctypes.CDLL(str(build/'libmono.so'))
    fn=library.replay;fn.argtypes=[ctypes.c_int]*7+[ctypes.c_float]*2+[ctypes.c_void_p]*len(pins['arguments']);fn.restype=None
    original=transport.advance_scalar_transport_cuda_v841;records=[]
    def host(v):return cp.asnumpy(v) if isinstance(v,cp.ndarray) else np.asarray(v)
    def wrapped(mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw):
        if not kw.get('config_apply_lbcs') or kw['rk_step']!=3 or len(records)>=limit:return original(mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw)
        ns,nl,ncp=stage.shape;nc=ncp-1;ne=uh.shape[-1]-1;me=int(mesh.max_edges)
        coef=kw['coefficients'];width=int(coef.adv_coefs.shape[1]);source=kw.get('scalar_tendency');source=cp.zeros_like(old) if source is None else source
        data=dict(old=host(old),stage=host(stage),rho0=host(rho0),rho1=host(rho1),uh=host(uh),ww=host(ww),dv=host(mesh.dv_edge),
            cells=host(mesh.cells_on_edge).T+1,neighbors=host(mesh.cells_on_cell).T+1,edges=host(mesh.edges_on_cell).T+1,
            counts=host(mesh.n_edges_on_cell),signs=host(mesh.edge_sign_on_cell).T,area=host(context.inv_area_cell),tendency=host(source),
            fnm=host(kw['fzm']),fnp=host(kw['fzp']),rdnw=host(kw['rdzw']),maskc=host(kw['bdy_mask_cell']),maske=host(kw['bdy_mask_edge']),
            advcounts=host(coef.n_adv_cells_for_edge),advcells=host(coef.adv_cells_for_edge).T+1,adv=host(coef.adv_coefs).T,adv3=host(coef.adv_coefs_3rd).T,target=np.zeros((nl,ncp),np.float32))
        data['rdnw']=np.pad(data['rdnw'],(0,max(0,nl+1-data['rdnw'].size)))
        # MPAS-A v8.4.2 mpas_atm_core.F:1555 scales this at initialization.
        data['adv3']=np.multiply(data['adv3'],np.float32(kw['config_coef_3rd_order']),dtype=np.float32)
        arrays={n:np.array(v,order='F',copy=True) for n,v in data.items()}
        if not records:np.savez(out/'inputs-first.npz',**arrays)
        fn(nc,ne,nl,ns,me,width,int(kw['config_monotonic']),np.float32(dt),np.float32(kw['config_coef_3rd_order']),*(arrays[n].ctypes.data_as(ctypes.c_void_p) for n in pins['arguments']))
        result=original(mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw)
        expected=np.ascontiguousarray(arrays['stage'][...,:nc]);actual=np.ascontiguousarray(host(result.scalars)[...,:nc]);bad=expected.view(np.uint32)!=actual.view(np.uint32)
        baseline=prepare_mpas_transport_source_baseline_v842(before=old,source=source,rho_old=rho0,dt=dt,n_cells_solve=nc,kernel_cache=kw['kernel_cache'])
        baseline_bad=arrays['old'][...,:nc].view(np.uint32)!=host(baseline)[...,:nc].view(np.uint32)
        record=dict(call=len(records),words=int(expected.size),differences=int(bad.sum()),source_nonzero_words=int(np.count_nonzero(data['tendency'][...,:nc])),baseline_differences=int(baseline_bad.sum()),native_nonzero_subnormals=int(np.count_nonzero(((expected.view(np.uint32)&0x7fffffff)>0)&((expected.view(np.uint32)&0x7fffffff)<0x00800000))),by_mask={str(z):int(bad[...,data['maskc'][:nc]==z].sum()) for z in np.unique(data['maskc'][:nc])},first_indices=np.argwhere(bad)[:8].tolist())
        records.append(record)
        if record['differences'] or record['baseline_differences']:np.savez(out/f"call-{record['call']:03}.npz",native=expected,cuda=actual,old_native=arrays['old'],old_cuda=host(baseline))
        receipt=dict(schema='d4.whole-scalar-mono-real-replay/v1',forecast_proof=False,process_count=1,compile_manifest=kw['kernel_cache'].compile_manifest(),source_pins=pins,binary_sha256=hashlib.sha256((build/'libmono.so').read_bytes()).hexdigest(),records=records)
        (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print('D4_MONO_REPLAY '+json.dumps(record),flush=True)
        return result
    transport.advance_scalar_transport_cuda_v841=wrapped

    from hexcore import cuda_driver
    cuda_driver.advance_scalar_transport_cuda_v841=wrapped
