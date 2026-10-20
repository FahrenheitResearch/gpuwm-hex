"""Replay whole native scalar transport on real resident regional inputs.

MPAS-A v8.4.2 mpas_atm_time_integration.F:4673-4938 is copied unchanged.
Framework constants and the C ABI only supply storage and dimensions.
This is supporting evidence; acceptance still requires the whole forecast.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import re
import numpy as np


def prepare(source_root, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=Path(source_root)/'src/core_atmosphere/dynamics/mpas_atm_time_integration.F'
    source=path.read_text()
    match=re.search(r'^   subroutine atm_advance_scalars_work\(.*?^   end subroutine atm_advance_scalars_work',source,re.M|re.S)
    assert match
    routine=match.group()
    declarations='''integer(c_int),value :: nc,ne,nl,ns,me,width,rk,order
real(c_float),value :: dt,coef
real(c_float) :: old(ns,nl,nc+1),stage(ns,nl,nc+1),rho0(nl,nc+1),rho1(nl,nc+1)
real(c_float) :: uh(nl,ne+1),ww(nl+1,nc+1),dv(ne+1),signs(me,nc+1),area(nc+1)
real(c_float) :: tendency(ns,nl,nc+1),fnm(nl+1),fnp(nl+1),rdnw(nl+1)
real(c_float) :: adv(width,ne+1),adv3(width,ne+1),horizontal(ns,nl,ne+1)
integer(c_int) :: cells(2,ne+1),edges(me,nc+1),counts(nc+1),maskc(nc+1),maske(ne+1)
integer(c_int) :: advcounts(ne+1),advcells(width,ne+1)'''
    arrays=['old','stage','rho0','rho1','uh','ww','dv','cells','edges','counts','signs','area','tendency','fnm','fnp','rdnw','maskc','maske','advcounts','advcells','adv','adv3','horizontal']
    code=['#define MPAS_ACC_TIMER_START(X)','#define MPAS_ACC_TIMER_STOP(X)',
          'module mpas_atm_dimensions','integer :: nVertLevels','end module',
          'module native_transport','use iso_c_binding','integer,parameter :: RKIND=kind(1.0),nRelaxZone=5',
          'logical :: config_apply_lbcs=.true.','contains',routine,
          'subroutine replay(nc,ne,nl,ns,me,width,rk,order,dt,coef,'+','.join(arrays)+') bind(c)',
          'use mpas_atm_dimensions',declarations,'nVertLevels=nl',
          'call atm_advance_scalars_work(nc,ns,dt,1,ne,1,nc,coef,old,stage,rho0,rho1, &',
          'uh,ww,dv,cells,edges,counts,signs,area,tendency,fnm,fnp,rdnw,maskc,maske, &',
          'advcounts,advcells,adv,adv3,ne,horizontal,rk,order,.true.)','end subroutine','end module']
    (out/'oracle.F90').write_text('\n'.join(code)+'\n')
    flags='-cpp -DDO_PHYSICS -O3 -fPIC -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -ffree-line-length-none -fcheck=bounds'
    (out/'Makefile').write_text('FC = gfortran\nFFLAGS = '+flags+'\nall: libtransport.so\nlibtransport.so: oracle.F90\n\t$(FC) $(FFLAGS) -shared $< -o $@\n')
    (out/'source-pins.json').write_text(json.dumps(dict(file=str(path),routine='atm_advance_scalars_work',
        first_line=source[:match.start()].count('\n')+1,last_line=source[:match.end()].count('\n')+1,
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),flags=flags,arguments=arrays),indent=2)+'\n')


def install_real_capture(build,out,limit=240):
    """Compare all physical scalar outputs on every unlimited stage of 24 h."""
    import cupy as cp
    from hexcore import cuda_regional_forecast_v841 as regional
    build,out=Path(build),Path(out);out.mkdir(parents=True,exist_ok=True)
    pins=json.loads((build/'source-pins.json').read_text());library=ctypes.CDLL(str(build/'libtransport.so'))
    fn=library.replay;fn.argtypes=[ctypes.c_int]*8+[ctypes.c_float]*2+[ctypes.c_void_p]*len(pins['arguments']);fn.restype=None
    original=regional.advance_scalars_regional_cuda_v841;records=[]
    def host(value):return cp.asnumpy(value) if isinstance(value,cp.ndarray) else np.asarray(value)
    def wrapped(runtime,mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw):
        if len(records)>=limit:return original(runtime,mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw)
        ns,nl,_=stage.shape;nc=runtime.n_cells_solve;ne=runtime.n_edges_solve;me=int(mesh.max_edges)
        coefficients=kw['coefficients'];width=int(coefficients.adv_coefs.shape[1])
        data=dict(old=host(old),stage=host(stage),rho0=host(rho0),rho1=host(rho1),uh=host(uh),ww=host(ww),
            dv=host(mesh.dv_edge),cells=host(mesh.cells_on_edge).T+1,edges=host(mesh.edges_on_cell).T+1,
            counts=host(mesh.n_edges_on_cell),signs=host(mesh.edge_sign_on_cell).T,area=host(context.inv_area_cell),
            tendency=np.zeros(old.shape,np.float32) if kw.get('scalar_tendency') is None else host(kw['scalar_tendency']),
            fnm=host(kw['fzm']),fnp=host(kw['fzp']),rdnw=host(kw['rdzw']),maskc=host(runtime.bdy_mask_cell),maske=host(runtime.bdy_mask_edge),
            advcounts=host(coefficients.n_adv_cells_for_edge),advcells=host(coefficients.adv_cells_for_edge).T+1,
            adv=host(coefficients.adv_coefs).T,adv3=host(coefficients.adv_coefs_3rd).T,
            horizontal=np.zeros((ns,nl,ne+1),np.float32))
        # MPAS declares rdnw with an assumed extent and consumes only nl words.
        data['rdnw']=np.pad(data['rdnw'],(0,max(0,nl+1-data['rdnw'].size)))
        # MPAS-A v8.4.2 mpas_atm_core.F:1555 scales this at initialization.
        data['adv3']=np.multiply(data['adv3'],np.float32(kw['config_coef_3rd_order']),dtype=np.float32)
        arrays={n:np.asfortranarray(value) for n,value in data.items()}
        if not records:np.savez(out/'inputs-first.npz',**arrays)
        fn(nc,ne,nl,ns,me,width,int(kw['rk_step']),int(kw.get('config_time_integration_order',3)),np.float32(dt),np.float32(kw['config_coef_3rd_order']),
           *(arrays[n].ctypes.data_as(ctypes.c_void_p) for n in pins['arguments']))
        output,target=original(runtime,mesh,context,old,stage,rho0,rho1,uh,ww,dt,**kw)
        expected=np.ascontiguousarray(arrays['stage'][...,:nc]);actual=np.ascontiguousarray(host(output)[...,:nc])
        bad=expected.view(np.uint32)!=actual.view(np.uint32)
        record=dict(call=len(records),rk_step=int(kw['rk_step']),words=int(expected.size),differences=int(bad.sum()),
                    source_nonzero_words=int(np.count_nonzero(arrays['tendency'][...,:nc])),
                    native_nonzero_subnormals=int(np.count_nonzero(((expected.view(np.uint32)&0x7fffffff)>0)&((expected.view(np.uint32)&0x7fffffff)<0x00800000))),
                    native_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),cuda_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    first_indices=np.argwhere(bad)[:8].tolist())
        records.append(record)
        if record['differences']:np.savez(out/f"call-{record['call']:03}.npz",native=expected,cuda=actual,source=arrays['tendency'])
        receipt=dict(schema='d4.whole-scalar-transport-real-replay/v1',forecast_proof=False,process_count=1,
                     compile_manifest=runtime.kernel_cache.compile_manifest(),source_pins=pins,binary_sha256=hashlib.sha256((build/'libtransport.so').read_bytes()).hexdigest(),
                     launcher_sha256=hashlib.sha256(Path(regional.__file__).read_bytes()).hexdigest(),records=records)
        (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('D4_TRANSPORT_REPLAY '+json.dumps(record),flush=True)
        return output,target
    regional.advance_scalars_regional_cuda_v841=wrapped


def replay_signed_subnormals(build, inputs, out):
    """Replay the whole routine, both RK orders, three stages and tiny inputs."""
    from types import SimpleNamespace
    import cupy as cp
    from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels
    from hexcore import cuda_transport, cuda_transport_v841
    from hexcore import cuda_regional_forecast_v841 as regional

    with np.load(inputs) as archive:
        fixture={name:np.array(archive[name],copy=True) for name in archive.files}
    ns,nl,ncp=fixture['old'].shape
    nc=ncp-1;ne=fixture['uh'].shape[-1]-1
    cache=MpasPhysicsKernelCache();kernels=CudaRegionalKernels(cache)
    def launch(module,name,count,args):
        module._kernel(name,cache)(((count+127)//128,),(128,),args)
    install_real_capture(build,out,limit=12)
    for order in (2,3):
        for rk in (1,2,3):
            for tiny in (False,True):
                data={name:np.array(value,copy=True) for name,value in fixture.items()}
                if tiny:
                    words=np.array([1,0x80000001,0x40000,0x80040000],np.uint32).view(np.float32)
                    for name in ('old','stage','tendency'):
                        data[name][...]=np.resize(words,data[name].shape)
                    for name in ('rho0','rho1'):data[name][...]=1
                    for name in ('uh','ww'):data[name][...]=0
                device={name:cp.asarray(np.ascontiguousarray(value)) for name,value in data.items()}
                mesh=SimpleNamespace(max_edges=data['edges'].shape[0],
                    dv_edge=device['dv'],cells_on_edge=cp.ascontiguousarray(device['cells'].T-1),
                    edges_on_cell=cp.ascontiguousarray(device['edges'].T-1),
                    n_edges_on_cell=device['counts'],
                    edge_sign_on_cell=cp.ascontiguousarray(device['signs'].T))
                coefficients=SimpleNamespace(adv_coefs=cp.ascontiguousarray(device['adv'].T),
                    adv_coefs_3rd=cp.ascontiguousarray(device['adv3'].T*np.float32(4)),
                    n_adv_cells_for_edge=device['advcounts'],
                    adv_cells_for_edge=cp.ascontiguousarray(device['advcells'].T-1))
                context=SimpleNamespace(inv_area_cell=device['area'])
                flag=cp.zeros(1,cp.int32)
                def validate(target,invalid):
                    cuda_transport_v841._check_density(target,name='synthetic target',
                        count=nl*ncp,invalid_flag=invalid,kernel_cache=cache)
                runtime=SimpleNamespace(n_cells_solve=nc,n_edges_solve=ne,
                    kernel_cache=cache,kernels=kernels,bdy_mask_cell=device['maskc'],
                    bdy_mask_edge=device['maske'],validate_density=validate,
                    shared_transport_launch=lambda name,count,args:launch(cuda_transport,name,count,args),
                    shared_transport_v841_launch=lambda name,count,args:launch(cuda_transport_v841,name,count,args))
                regional.advance_scalars_regional_cuda_v841(runtime,mesh,context,
                    device['old'],device['stage'],device['rho0'],device['rho1'],
                    device['uh'],device['ww'],1.,coefficients=coefficients,
                    fzm=device['fnm'],fzp=device['fnp'],rdzw=device['rdnw'][:nl],
                    rk_step=rk,config_time_integration_order=order,
                    config_coef_3rd_order=.25,validation_flag=flag,
                    scalar_tendency=device['tendency'])
                assert int(cp.asnumpy(flag)[0])==0
    path=Path(out)/'result.json';receipt=json.loads(path.read_text())
    receipt['qualification']='Whole unchanged native routine, real mesh masks, both RK orders and signed subnormal inputs; supporting replay only.'
    receipt['signed_subnormal_input_cases']=6
    receipt['compile_manifest']=cache.compile_manifest()
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    assert len(receipt['records'])==12
    assert sum(row['differences'] for row in receipt['records'])==0
    assert sum(row['native_nonzero_subnormals'] for row in receipt['records'][1::2])>0
