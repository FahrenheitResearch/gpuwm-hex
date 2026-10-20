"""CUDA glue vs pristine strict Fortran, synthetic columns on x1.2562 extent.

No forecast claim. Run only inside the GPU queue and the build queue (NVRTC compilation).
The native authority must first be built by mpasphys_reference_v842.py.
"""
import ctypes
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('MPAS_PHYSICS_GPU_TESTS') != '1',
                               reason='explicit GPU slot allocation required')

NAMES = ('qv','qc','qr','qi','qs','qg','ni','nr','nc','nifa','nwfa')
RATES = ('rthblten','rqvblten','rqcblten','rqiblten','rqsblten','rniblten',
         'rncblten','rnifablten','rnwfablten','rthcuten','rqvcuten','rqccuten',
         'rqicuten','rqrcuten','rqscuten','rthratenlw','rthratensw')


@pytest.fixture(scope='module')
def authority():
    folder=Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])
    lib=ctypes.CDLL(str(folder/'libauthority.so'))
    lib.couple.argtypes=[ctypes.c_int]*13+[ctypes.c_void_p]*11
    rad=ctypes.CDLL(str(folder/'libradiation.so'))
    rad.radiation.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,
                           ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
    from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
    import cupy as cp
    return lib,rad,MpasPhysicsKernelCache(),cp


def f32(a):
    return np.asfortranarray(a,dtype=np.float32)


@pytest.fixture(scope='module',params=['synthetic'] +
                (['native-init'] if os.environ.get('MPAS_PHYSICS_REAL_COLUMNS') else []))
def columns(request):
    rng=np.random.default_rng(442)
    # Test arrays are synthetic columns, not a claimed MPAS mesh forecast.
    nl,nc,ne,ns=8,2562,7680,len(NAMES)
    m=f32(rng.uniform(.1,10,(nl,nc)))
    me=f32(rng.uniform(.1,10,(nl,ne)))
    theta=f32(rng.uniform(200,360,(nl,nc)))
    sc=f32(rng.uniform(-.005,.03,(ns,nl,nc)))
    if request.param=='native-init':
        folder=Path(os.environ['MPAS_PHYSICS_REAL_COLUMNS'])
        shape=json.loads((folder/'shape.json').read_text())
        nl,nc=shape['n_levels'],shape['n_cells'];ne=3*nc-6
        def read(name):
            # Raw Rust-decoded words, packed to the native [level,cell] ABI.
            return f32(np.frombuffer((folder/(name+'.bin')).read_bytes(),dtype='<f4').reshape(nc,nl).T)
        rho,zz,dry_theta=read('rho'),read('zz'),read('theta')
        sc=f32(np.zeros((ns,nl,nc)))
        for name in ('qv','qc','qr'): sc[NAMES.index(name)]=read(name)
        m,theta=f32(np.empty((nl,nc))),f32(np.empty((nl,nc)))
        init=ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libinitial_carriers.so'))
        init.initialize_carriers.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*6
        qv=f32(sc[:1])
        init.initialize_carriers(nl,nc,*[a.ctypes.data for a in (rho,zz,dry_theta,qv,m,theta)])
        # Edge masses and scheme rates remain synthetic: the published file is
        # initialization, not a captured physics boundary or completed forecast.
        me=f32(rng.uniform(.1,10,(nl,ne)))
    rates=f32(rng.choice([-1,1],size=(nl,nc,17))*10.**rng.uniform(-12,2,(nl,nc,17)))
    rates[0,0,:]=np.float32(1e-45)
    rates[0,1,:]=np.float32(-1e-45)
    rates[0,2,:]=np.float32(-0.)
    be=f32(rng.uniform(-1e-3,1e-3,(nl,ne)))
    ce=f32(rng.uniform(-1e-3,1e-3,(nl,ne)))
    diag=f32(rng.uniform(-1e-3,1e-3,(nl,ne)))
    return m,me,theta,sc,rates,be,ce,diag


_COMBOS=list(itertools.product(range(3),range(5),range(2),range(2)))
_COMBOS += [(2,cu,lw,sw) for cu,lw,sw in itertools.product(range(5),range(2),range(2))]


@pytest.mark.parametrize('case,combo',list(enumerate(_COMBOS)))
def test_rk_coupling_word_identity(authority,columns,case,combo):
    from hexcore.cuda_physics_v841 import couple_mpas_physics_tendencies_v842
    lib,_,cache,cp=authority
    pbl,cu,lw,sw=combo
    micro=int(case>=60)
    rk,split=[(1,1),(2,1),(3,2)][case%3]
    m,me,theta,sc,rates,be,ce,diag=columns
    nl,nc=m.shape;ne=me.shape[1];ns=len(NAMES)
    solve,esolve=nc-7,ne-11
    nt=f32(np.zeros_like(m));nsc=f32(np.zeros_like(sc));nu=f32(np.zeros_like(me))
    nd=diag.copy(order='F')
    args=[nl,nc,ne,solve,esolve,ns,pbl,cu,micro,lw,sw,rk,split]
    arrays=[m,me,theta,sc,rates,be,ce,nd,nt,nsc,nu]
    # C wrapper has eleven array arguments, all in native Fortran order.
    lib.couple.argtypes=[ctypes.c_int]*13+[ctypes.c_void_p]*11
    lib.couple(*args,*[a.ctypes.data for a in arrays])
    def device(a):
        return cp.asarray(np.ascontiguousarray(a))
    dd=device(diag)
    out=couple_mpas_physics_tendencies_v842(
        mass=device(m),mass_edge=device(me),theta_m=device(theta),scalars=device(sc),
        scalar_names=NAMES,tendencies={n:device(rates[:,:,i]) for i,n in enumerate(RATES)},
        selectors={'config_pbl_scheme':['off','bl_ysu','bl_mynn'][pbl],
                   'config_convection_scheme':['off','cu_grell_freitas','cu_kain_fritsch','cu_tiedtke','cu_ntiedtke'][cu],
                   'config_microp_scheme':'mp_thompson_aerosols' if micro else 'mp_wsm6',
                   'config_radt_lw_scheme':'rrtmg_lw' if lw else 'off',
                   'config_radt_sw_scheme':'rrtmg_sw' if sw else 'off'},
        pbl_edge=device(be),convection_edge=device(ce),tend_u_phys=dd,
        n_cells_solve=solve,n_edges_solve=esolve,rk_step=rk,dynamics_substep=split,
        kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    for name,native in [('tend_rtheta_physics',nt),('scalars_tend',nsc),('tend_ru_physics',nu)]:
        got=cp.asnumpy(out[name])
        assert np.array_equal(got.view(np.uint32),np.ascontiguousarray(native).view(np.uint32)), (
            name,int(np.count_nonzero(got.view(np.uint32)!=np.ascontiguousarray(native).view(np.uint32))))
    assert np.array_equal(cp.asnumpy(dd).view(np.uint32),np.ascontiguousarray(nd).view(np.uint32))
    assert not cp.asnumpy(out['tend_rho_physics']).any()


@pytest.mark.parametrize('ring,bucket',list(itertools.product([False,True],[0.,-10.,100.])))
def test_radiation_bucket_word_identity(authority,ring,bucket):
    from hexcore.cuda_physics_v841 import (
        MPAS_RADIATION_FLUX_FIELDS_V842,update_mpas_radiation_diagnostics_v842)
    _,rad,cache,cp=authority
    rng=np.random.default_rng(57)
    nc=2562;solve=nc-9;dt=360.
    flux=f32(rng.uniform(-1,5,(nc,16)))
    acc=f32(rng.uniform(0,100,(nc,16)))
    flux[0,:]=0.;acc[0,:]=bucket # Equality must not bucket.
    flux[1,:]=0.;acc[1,:]=3*bucket # Multiple buckets subtract exactly once.
    counts=np.asfortranarray(rng.integers(0,9,(nc,16),dtype=np.int32))
    fields={}
    for j,name in enumerate(MPAS_RADIATION_FLUX_FIELDS_V842):
        fields[name]=cp.asarray(np.ascontiguousarray(flux[:,j]))
        fields['ac'+name]=cp.asarray(np.ascontiguousarray(acc[:,j]))
        fields['i_ac'+name]=cp.asarray(np.ascontiguousarray(counts[:,j]))
    rad.radiation(nc,solve,dt,bucket,int(ring),flux.ctypes.data,acc.ctypes.data,counts.ctypes.data)
    update_mpas_radiation_diagnostics_v842(fields=fields,dt=dt,bucket=bucket,ringing=ring,
                                          n_cells_solve=solve,kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    for j,name in enumerate(MPAS_RADIATION_FLUX_FIELDS_V842):
        assert np.array_equal(cp.asnumpy(fields['ac'+name]).view(np.uint32),acc[:,j].view(np.uint32))
        assert np.array_equal(cp.asnumpy(fields['i_ac'+name]),counts[:,j])


def test_radiation_subnormal_words_and_effective_nvrtc_flags(authority):
    from hexcore.cuda_physics_v841 import (
        MPAS_RADIATION_FLUX_FIELDS_V842,update_mpas_radiation_diagnostics_v842)
    _,rad,cache,cp=authority
    nc=2562
    flux=f32(np.full((nc,16),np.float32(1e-45)))
    flux[1,:]=np.float32(-1e-45)
    acc=f32(np.zeros((nc,16)));counts=np.zeros((nc,16),np.int32,order='F')
    fields={}
    for j,name in enumerate(MPAS_RADIATION_FLUX_FIELDS_V842):
        fields[name]=cp.asarray(np.ascontiguousarray(flux[:,j]))
        fields['ac'+name]=cp.asarray(np.ascontiguousarray(acc[:,j]))
        fields['i_ac'+name]=cp.asarray(np.ascontiguousarray(counts[:,j]))
    rad.radiation(nc,nc,1.,1.,0,flux.ctypes.data,acc.ctypes.data,counts.ctypes.data)
    update_mpas_radiation_diagnostics_v842(fields=fields,dt=1.,bucket=1.,ringing=False,
                                          n_cells_solve=nc,kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    for j,name in enumerate(MPAS_RADIATION_FLUX_FIELDS_V842):
        assert np.array_equal(cp.asnumpy(fields['ac'+name]).view(np.uint32),acc[:,j].view(np.uint32))
    evidence=cache._compile_manifest.snapshot()['modules']['hexcore.mpasphys.coupling_v842']['effective_compile']
    assert evidence['status']=='resolved'
    assert evidence['observations']
    for observation in evidence['observations']:
        flags=observation['effective_flags']
        assert '--ftz=false' in flags and '-ftz=true' not in flags
        assert '--fmad=false' in flags


@pytest.mark.parametrize('convection',['off','cu_tiedtke','cu_grell_freitas','cu_ntiedtke'])
@pytest.mark.parametrize('monotonic',[False,True])
@pytest.mark.parametrize('dt',[90.,700.,.125])
def test_post_transport_word_identity(authority,columns,convection,monotonic,dt):
    from hexcore.cuda_physics_v841 import prepare_mpas_post_transport_v842
    _,_,cache,cp=authority
    lib=ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libtransport.so'))
    lib.transport.argtypes=[ctypes.c_int]*7+[ctypes.c_float]+[ctypes.c_void_p]*5
    _,_,theta,sc,_,_,_,_=columns
    ns,nl,nc=sc.shape;solve=nc-7
    rng=np.random.default_rng(92)
    before=sc.copy(order='F');after=f32(sc+rng.uniform(-.01,.01,sc.shape))
    after[0,0,:3]=[-1e-45,-0.,0.]
    after[1,0,:3]=[np.nan,-0.,-1e-45]
    rq=f32(rng.uniform(-1e-3,1e-3,theta.shape))
    rt=f32(rng.uniform(-1e-3,1e-3,theta.shape))
    da=cp.asarray(np.ascontiguousarray(after))
    dq=cp.asarray(np.ascontiguousarray(rq));dr=cp.asarray(np.ascontiguousarray(rt))
    lib.transport(nl,nc,solve,ns,1,int(convection in ('cu_grell_freitas','cu_ntiedtke')),
                  int(monotonic),dt,*[a.ctypes.data for a in [before,after,theta,rq,rt]])
    prepare_mpas_post_transport_v842(before=cp.asarray(np.ascontiguousarray(before)),after=da,
        theta_m=cp.asarray(np.ascontiguousarray(theta)),scalar_names=NAMES,rqvdynten=dq,
        rthdynten=dr,convection_scheme=convection,monotonic=monotonic,dt=dt,
        n_cells_solve=solve,kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    for name,got,want in [('after',da,after),('rqvdynten',dq,rq),('rthdynten',dr,rt)]:
        device=cp.asnumpy(got).view(np.uint32)
        native=np.ascontiguousarray(want).view(np.uint32)
        assert np.array_equal(device,native),(name,int(np.count_nonzero(device!=native)))


@pytest.mark.parametrize('dt', [90., 700., .125])
def test_source_updated_transport_baseline_word_identity(authority, columns, dt):
    """Physical sources must not enter resolved vapor advection a second time."""
    from hexcore.cuda_physics_v841 import prepare_mpas_transport_source_baseline_v842
    _, _, cache, cp = authority
    lib = ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libsource_baseline.so'))
    lib.source_baseline.argtypes = [ctypes.c_int]*4+[ctypes.c_float]+[ctypes.c_void_p]*3
    rho, _, _, scalars, _, _, _, _ = columns
    ns, nl, nc = scalars.shape
    solve = nc-7
    source = f32(np.random.default_rng(2048).uniform(-1e-5, 1e-5, scalars.shape))
    source[0, 0, :3] = [1e-45, -1e-45, -0.]
    baseline = scalars.copy(order='F')
    native_source = source.copy(order='F')
    old = cp.asarray(np.ascontiguousarray(scalars))
    device_source = cp.asarray(np.ascontiguousarray(source))
    lib.source_baseline(nl, nc, solve, ns, dt, baseline.ctypes.data,
                        native_source.ctypes.data, rho.ctypes.data)
    got = prepare_mpas_transport_source_baseline_v842(before=old, source=device_source,
        rho_old=cp.asarray(np.ascontiguousarray(rho)), dt=dt,
        n_cells_solve=solve, kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    for name, device, native in [('baseline', got, baseline), ('held source', device_source, source),
                                  ('outer scalars', old, scalars)]:
        words = cp.asnumpy(device).view(np.uint32)
        wanted = np.ascontiguousarray(native).view(np.uint32)
        assert np.array_equal(words, wanted), (name, int(np.count_nonzero(words != wanted)))


@pytest.mark.parametrize('seed',[32,47,108])
def test_projection_word_identity(authority,seed):
    from hexcore.cuda_physics_v841 import CudaPhysicsGeometryV841,project_mpas_momentum_v842
    from hexcore.cuda_backend.containers import TransferStats
    _,_,cache,cp=authority
    lib=ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libprojection.so'))
    lib.projection.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*7
    nl,nc,ne=8,2562,7680
    rng=np.random.default_rng(seed)
    cells=np.asfortranarray(rng.integers(1,nc+1,(2,ne),dtype=np.int32))
    cells[:,:3]=[[nc,1,1],[1,nc,nc]]
    east=f32(rng.uniform(-1,1,(3,nc)));north=f32(rng.uniform(-1,1,(3,nc)))
    normal=f32(rng.uniform(-1,1,(3,ne)))
    u=f32(rng.uniform(-100,100,(nl,nc)));v=f32(rng.uniform(-100,100,(nl,nc)))
    u[0,:3]=[-1e-45,1e-45,-0.];v[0,:3]=[1e-45,-1e-45,0.]
    u[:,-1]=0.;v[:,-1]=0.
    expected=f32(np.empty((nl,ne),dtype=np.float32))
    lib.projection(nl,nc,ne,nc-7,*[a.ctypes.data for a in [cells,east,north,normal,u,v,expected]])
    geometry=CudaPhysicsGeometryV841(
        cells_on_edge=cp.asarray(np.ascontiguousarray(cells.T-1)),
        east_cell=cp.asarray(np.ascontiguousarray(east.T)),
        north_cell=cp.asarray(np.ascontiguousarray(north.T)),
        edge_normal_vectors=cp.asarray(np.ascontiguousarray(normal.T)),
        n_cells=nc,n_edges=ne,h2d=TransferStats(0,0.))
    invalid=cp.zeros(1,dtype=cp.int32)
    result=project_mpas_momentum_v842(geometry=geometry,zonal=cp.asarray(np.ascontiguousarray(u)),
        meridional=cp.asarray(np.ascontiguousarray(v)),invalid_flag=invalid,kernel_cache=cache)
    cp.cuda.Stream.null.synchronize()
    assert cp.asnumpy(invalid)[0] == 0
    assert np.array_equal(cp.asnumpy(result).view(np.uint32),np.ascontiguousarray(expected).view(np.uint32))


@pytest.mark.parametrize('aerosol,mynn,flags',list(itertools.product(
    (False,True),(False,True),itertools.product((False,True),repeat=4))))
def test_number_carrier_selection_word_identity(authority,aerosol,mynn,flags):
    from types import SimpleNamespace
    from hexcore.cuda_physics_prep_v841 import prepare_mpas_number_carriers_v842
    _,_,cache,cp=authority
    lib=ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libnumbers.so'))
    lib.numbers.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*3
    nl,nc,solve=8,2562,2555
    names=('nc','ni','nifa','nwfa')
    rng=np.random.default_rng(510)
    # max(+0,q) has already run in the shared preparation. Exercise positive
    # subnormal words, +0 and widely separated number-carrier magnitudes.
    data=f32(10.**rng.uniform(-6,10,(4,nl,nc)))
    data[:,0,0]=np.float32(1e-45);data[:,0,1]=np.float32(0)
    native=f32(np.zeros((nc,nl,4)))
    f=np.asarray(flags,dtype=np.int32)
    lib.numbers(nl,nc,solve,int(aerosol),int(mynn),f.ctypes.data,data.ctypes.data,native.ctypes.data)
    positive={name:cp.asarray(np.ascontiguousarray(data[i])) for i,name in enumerate(names)}
    positive['qv']=cp.zeros((nl,nc),dtype=cp.float32)
    prepared=SimpleNamespace(species_p=positive)
    got=prepare_mpas_number_carriers_v842(prepared,
        microphysics='mp_thompson_aerosols' if aerosol else 'mp_thompson',
        pbl='bl_mynn' if mynn else 'bl_ysu',scalar_flags=dict(zip(names,flags)),
        n_cells_solve=solve,kernel_cache=cache)
    allocated=names if mynn else ('nifa','nwfa') if aerosol else ()
    assert tuple(got)==allocated
    for name in allocated:
        actual=cp.asnumpy(got[name]);expected=np.ascontiguousarray(native[:,:,names.index(name)].T)
        assert np.array_equal(actual.view(np.uint32),expected.view(np.uint32)),name


@pytest.mark.parametrize('pbl,cu',[(1,4),(1,0),(0,4)])
def test_projected_rk_sequence_word_identity(authority,columns,pbl,cu):
    from hexcore.cuda_physics_v841 import CudaPhysicsGeometryV841,CudaMpasCouplingV842
    from hexcore.cuda_backend.containers import TransferStats
    lib,_,cache,cp=authority
    projection=ctypes.CDLL(str(Path(os.environ['MPAS_PHYSICS_REFERENCE_DIR'])/'libprojection.so'))
    projection.projection.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*7
    m,me,theta,sc,rates,_,_,diag=columns
    nl,nc=m.shape;ne=me.shape[1];ns=len(NAMES);solve,esolve=nc-7,ne-11
    rng=np.random.default_rng(675)
    cells=np.asfortranarray(rng.integers(1,nc+1,(2,ne),dtype=np.int32))
    east=f32(rng.uniform(-1,1,(3,nc)));north=f32(rng.uniform(-1,1,(3,nc)));normal=f32(rng.uniform(-1,1,(3,ne)))
    vectors=[f32(rng.uniform(-.01,.01,(nl,nc))) for _ in range(4)]
    for v in vectors:v[:,-1]=0.
    be=f32(np.zeros((nl,ne)));ce=f32(np.zeros((nl,ne)))
    for active,target,u,v in [(pbl,be,*vectors[:2]),(cu,ce,*vectors[2:])]:
        if active:projection.projection(nl,nc,ne,solve,*[a.ctypes.data for a in [cells,east,north,normal,u,v,target]])
    def device(a):return cp.asarray(np.ascontiguousarray(a))
    geometry=CudaPhysicsGeometryV841(cells_on_edge=device(cells.T-1),east_cell=device(east.T),
        north_cell=device(north.T),edge_normal_vectors=device(normal.T),n_cells=nc,n_edges=ne,h2d=TransferStats(0,0.))
    selectors={'config_pbl_scheme':'bl_ysu' if pbl else 'off',
        'config_convection_scheme':'cu_ntiedtke' if cu else 'off',
        'config_microp_scheme':'mp_wsm6','config_radt_lw_scheme':'rrtmg_lw','config_radt_sw_scheme':'rrtmg_sw'}
    ledger=[];invalid=cp.zeros(1,dtype=cp.int32);dd=device(diag);nd=diag.copy(order='F')
    adapter=CudaMpasCouplingV842(selectors=selectors,scalar_names=NAMES,geometry=geometry,
        kernel_cache=cache,exchange_halo_group=ledger.append,pbl_edge=cp.zeros((nl,ne),dtype=cp.float32),
        convection_edge=cp.zeros((nl,ne),dtype=cp.float32),tend_u_phys=dd)
    tend={n:device(rates[:,:,i]) for i,n in enumerate(RATES)}
    tend.update({n:device(a) for n,a in zip(('rublten','rvblten','rucuten','rvcuten'),vectors)})
    # Each split/RK insertion gets current mass; level-one theta/scalars stay
    # bound to the outer step. Cached unweighted edges are never recomputed.
    for split in (1,2,3):
        for rk in (1,2,3):
            factor=np.float32(1.+(split*3+rk)/32.)
            mass=f32(m*factor);edge_mass=f32(me*factor)
            nt=f32(np.zeros_like(m));nsc=f32(np.zeros_like(sc));nu=f32(np.zeros_like(me))
            args=[nl,nc,ne,solve,esolve,ns,pbl,cu,0,1,1,rk,split]
            lib.couple(*args,*[a.ctypes.data for a in [mass,edge_mass,theta,sc,rates,be,ce,nd,nt,nsc,nu]])
            got=adapter.get_tend(mass=device(mass),mass_edge=device(edge_mass),theta_m=device(theta),
                scalars=device(sc),tendencies=tend,n_cells_solve=solve,n_edges_solve=esolve,
                rk_step=rk,dynamics_substep=split,invalid_flag=invalid)
            for actual,expected in [(got.rho_theta,nt),(got.rho_u,nu),(got.scalars,nsc),(dd,nd)]:
                assert np.array_equal(cp.asnumpy(actual).view(np.uint32),np.ascontiguousarray(expected).view(np.uint32))
    assert cp.asnumpy(invalid)[0]==0
    assert ledger==(['physics:blten'] if pbl else [])+(['physics:cuten'] if cu else [])
