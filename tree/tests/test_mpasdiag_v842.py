"""Lifecycle checks plus strict Fortran/CUDA word replay (opt-in, queued GPU).

MPAS_DIAG_REFERENCE names the built reference.so. MPAS_DIAG_GPU=1 opts into the
replay; on a shared host run it through the GPU and build queues. Receipts have no tolerances.
"""
from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasdiag.cloud_v842 import CloudDiagnostics, FIELDS, _compile_cuda
from hexcore.mpasdiag.pv_v842 import PVDiagnostics, CUDA_SOURCE
from hexcore.mpasdiag.soundings_v842 import Soundings, CUDA_SOURCE as SOUND_SOURCE

RECEIPT_ROWS=[]


def test_no_cloud_demand_needs_no_arrays():
    CloudDiagnostics({}, {}, {}).compute(["theta"])


def test_no_pv_demand_needs_no_arrays():
    PVDiagnostics({}, {}, {}).compute(["theta"])


def test_pv_demand_keeps_diagnostic_enabled():
    class Diagnostic(PVDiagnostics):
        def atm_compute_pv_diagnostics(self, **kwargs): self.executed=True
    pv=Diagnostic({}, {}, {})
    pv.compute(["ertel_pv"])
    assert pv.executed


def test_distributed_pv_requires_communication():
    with pytest.raises(ValueError,match="halo exchange"):
        PVDiagnostics({}, {}, {}, distributed=True)


class Clock:
    def __init__(self):
        self.alarms=[]
        self.ringing=False
        self.resets=[]
    def now(self): return "2000-01-01_00:00:00"
    def add_alarm(self,*args): self.alarms.append(args)
    def is_ringing(self,name): return self.ringing
    def reset_alarm(self,name): self.resets.append(name)


class IO:
    def __init__(self,locations=None): self.locations=locations;self.reads=[]
    def read_locations(self,path): self.reads.append(path);return self.locations


@pytest.mark.parametrize("interval",[None,"none"])
def test_soundings_disabled_without_io(interval):
    clock,io=Clock(),IO()
    s=Soundings({}, {}, {}, clock,io)
    s.setup(interval);s.compute();s.cleanup()
    assert not clock.alarms and not io.reads


def test_missing_sounding_list_does_not_add_alarm():
    clock,io=Clock(),IO()
    Soundings({}, {}, {},clock,io).setup("00:30:00")
    assert io.reads==["sounding_locations.txt"] and not clock.alarms


def test_empty_sounding_list_has_native_alarm():
    locations=type("Locations",(),{"names":[]})()
    clock,io=Clock(),IO(locations)
    s=Soundings({}, {}, {},clock,io)
    s.setup("00:30:00")
    assert clock.alarms==[("soundingAlarm",clock.now(),"00:30:00")]
    clock.ringing=True;s.compute()
    assert clock.resets==["soundingAlarm"]


@pytest.fixture(scope="module")
def gpu():
    if os.environ.get("MPAS_DIAG_GPU")!="1":
        pytest.skip("queued CUDA replay requires MPAS_DIAG_GPU=1")
    return pytest.importorskip("cupy")


@pytest.fixture(scope="module")
def native():
    path=os.environ.get("MPAS_DIAG_REFERENCE")
    if not path:
        pytest.skip("strict standalone Fortran reference not provided")
    return native_library(path)


def native_library(path):
    lib=ctypes.CDLL(str(path))
    def call(name,*args):
        values=[]
        for a in args:
            if isinstance(a,np.ndarray):
                assert a.flags.f_contiguous
                values.append(ctypes.c_void_p(a.ctypes.data))
            elif isinstance(a,(float,np.floating)):
                values.append(ctypes.c_float(a))
            else:
                values.append(ctypes.c_int(a))
        f=getattr(lib,name);f.restype=None;f(*values)
    return call


@pytest.fixture(scope="module",autouse=True)
def receipt():
    yield
    path=os.environ.get("MPAS_DIAG_RECEIPT")
    if path:
        fingerprints={}
        for name in ('MPAS_DIAG_REFERENCE','GPUWM_MPAS_SOUNDINGS_BRIDGE'):
            artifact=os.environ.get(name)
            if artifact and Path(artifact).is_file():fingerprints[name]=hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
        patch=os.environ.get('MPAS_DIAG_REAL_PATCH')
        if patch:fingerprints['real_patch_manifest']=hashlib.sha256((Path(patch)/'patch.json').read_bytes()).hexdigest()
        for name in ('cloud_v842.py','pv_v842.py','soundings_v842.py'):
            source=Path(__file__).resolve().parents[1]/'src/hexcore/mpasdiag'/name
            fingerprints[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        Path(path).write_text(json.dumps({"tier":"strict standalone synthetic/real columns and native executable station files",
            "whole_forecast_proof":False,"process_count":1,
            "domain":"single-process standalone and captured native patch replay",
            "artifact_sha256":fingerprints,"rows":RECEIPT_ROWS},indent=2)+"\n")


def words(name,expected,actual):
    a=np.ascontiguousarray(expected)
    b=np.ascontiguousarray(actual)
    different=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
    row=dict(field=name,words=a.size,different_words=different,
        native_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        cuda_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    RECEIPT_ROWS.append(row)
    assert different==0,row


def F(a,dtype=np.float32): return np.array(a,dtype=dtype,order="F")


def test_cloud_pressure_edges_and_owned_extent(gpu,native):
    cp=gpu
    nc,nz=9,13
    thresholds=F([64200,35000,15000])
    ps=np.concatenate([thresholds,np.nextafter(thresholds,np.float32(-np.inf)),
        np.nextafter(thresholds,np.float32(np.inf)),F([100000,9000,80000,20000])])
    ps[-1]=np.float32(np.nan) # native ordered comparisons all false
    pb=F(np.broadcast_to(ps[:,None],(nz,nc)))
    pp=F(np.zeros_like(pb))
    cf=F(np.random.default_rng(203).uniform(-0.1,1.1,(nz,nc)))
    cf[:,0]=0.0 # zero maximum
    expected=[F(np.zeros(nc)) for _ in FIELDS]
    native("replay_cloud",nc,nz,pp,pb,cf,*expected)
    diag={"pressure_p":cp.asarray(pp,order="F"),"pressure_base":cp.asarray(pb,order="F")}
    diag.update({name:cp.full(nc+1,-999.0,dtype=cp.float32) for name in FIELDS})
    port=CloudDiagnostics({"nCellsSolve":nc,"nVertLevels":nz},diag,{"cldfrac":cp.asarray(cf,order="F")})
    port.compute([FIELDS[0]])
    for name,want in zip(FIELDS,expected):
        words(name,want,diag[name][:nc].get())
        assert diag[name][nc].get()==-999.0


def test_sounding_both_saturation_polynomials(gpu,native):
    cp=gpu
    n=4096
    t=F(np.linspace(170.0,320.0,n));p=F(np.linspace(10000,105000,n))
    t[:6]=F([193.16,193.15,193.17,273.16,273.15,273.17])
    liquid=F(np.zeros(n));ice=F(np.zeros(n))
    native("replay_saturation",n,p,t,liquid,ice)
    module=_compile_cuda("#define MPAS_SOUNDINGS_SATURATION_ONLY\n"+SOUND_SOURCE)
    output=[cp.empty(n,dtype=cp.float32) for _ in range(2)]
    module.get_function("saturation")(((n+127)//128,),(128,),
        (cp.int32(n),cp.asarray(p),cp.asarray(t),*output))
    words("rslf",liquid,output[0].get());words("rsif",ice,output[1].get())


def mesh_fixture(cp,seed=14,nz=9):
    rng=np.random.default_rng(seed)
    nc,me=4,3
    pairs=list(itertools.combinations(range(1,nc+1),2))
    triangles=list(itertools.combinations(range(1,nc+1),3))
    ne,nv=len(pairs),len(triangles)
    nec=F(np.full(nc+1,me),np.int32)
    coc=F(np.ones((me,nc+1)),np.int32)
    eoc=coc.copy(order="F");voc=coc.copy(order="F")
    for c in range(1,nc+1):
        coc[:,c-1]=[n for n in range(1,nc+1) if n!=c]
        eoc[:,c-1]=[pairs.index(tuple(sorted((c,n))))+1 for n in coc[:,c-1]]
        voc[:,c-1]=[i+1 for i,t in enumerate(triangles) if c in t]
    tangent=F(rng.uniform(-1,1,(3,2,nc+1)))
    vertical=F(rng.uniform(-1,1,(3,nc+1)))
    normal=F(rng.uniform(-1,1,(3,ne)))
    z=F(np.cumsum(rng.uniform(40,2000,(nz+1,nc+1)),axis=0))
    hosts=dict(nEdgesOnCell=nec,cellsOnCell=coc,edgesOnCell=eoc,cellsOnEdge=F(np.array(pairs).T,np.int32),
        verticesOnCell=voc,cellsOnVertex=F(np.array(triangles).T,np.int32),
        kiteAreasOnVertex=F(rng.uniform(1e6,1e7,(3,nv))),areaCell=F(rng.uniform(1e7,1e8,nc+1)),
        dvEdge=F(rng.uniform(1e3,1e5,ne)),edgeNormalVectors=normal,
        cellTangentPlane=tangent,localVerticalUnitVectors=vertical,zgrid=z,
        latCell=F([0.4,-0.8,0.0,-0.0,0.0]))
    mesh={name:cp.asarray(a,order="F") for name,a in hosts.items()}
    mesh.update(nCells=nc,nCellsSolve=nc,nVertLevels=nz,maxEdges=me,nEdges=ne,nVertices=nv)
    fields={name:F(rng.uniform(-30,30,(nz,nc+1))) for name in ("uReconstructX","uReconstructY","uReconstructZ","theta")}
    fields["rho"]=F(rng.uniform(0.01,1.1,(nz,nc+1)))
    fields["vorticity"]=F(rng.uniform(-0.001,0.001,(nz,nv+1)))
    diag={name:cp.asarray(a,order="F") for name,a in fields.items()}
    diag["ertel_pv"]=cp.full((nz,nc+1),-999.0,dtype=cp.float32,order="F")
    diag["inTropo"]=cp.zeros((nz,nc+1),dtype=cp.int32,order="F")
    diag["iLev_DT"]=cp.full(nc+1,-777,dtype=cp.int32)
    w=F(rng.uniform(-3,3,(nz+1,nc+1)))
    state={"w":cp.asarray(w,order="F")}
    return mesh,state,diag,hosts,fields,w


@pytest.mark.parametrize("seed",[14,901,731])
def test_defined_pv_mesh_and_vertical_branches(gpu,native,seed):
    cp=gpu
    m,s,d,h,f,w=mesh_fixture(cp,seed)
    nc,nz,me,ne,nv=(m[n] for n in ("nCells","nVertLevels","maxEdges","nEdges","nVertices"))
    expected=F(np.full((nz,nc+1),-999.0))
    args=[h[name] for name in ("nEdgesOnCell","cellsOnCell","edgesOnCell","cellsOnEdge","verticesOnCell",
        "cellsOnVertex","kiteAreasOnVertex","areaCell","dvEdge","edgeNormalVectors","cellTangentPlane",
        "localVerticalUnitVectors","zgrid")]
    native("replay_mesh",nc,nz,me,nv,ne,*args,nc,nz-1,
        f["uReconstructX"],f["uReconstructY"],f["uReconstructZ"],w,f["vorticity"],f["rho"],f["theta"],expected)
    PVDiagnostics(m,s,d).calc_epv(level_count=nz-1)
    actual=d["ertel_pv"].get()
    words(f"ertel_pv_defined_levels_seed_{seed}",expected[:nz-1,:nc],actual[:nz-1,:nc])
    assert np.all(actual[-1,:]==-999) and np.all(actual[:,nc]==-999)


@pytest.mark.parametrize("mode",["tropo","strato"])
@pytest.mark.parametrize("seed",[0,1,2,17,89])
def test_flood_fill_disconnected_blobs_and_hemispheres(gpu,native,mode,seed):
    cp=gpu
    m,s,d,h,f,w=mesh_fixture(cp,seed)
    nc,nz,me=(m[n] for n in ("nCells","nVertLevels","maxEdges"))
    pv=F(np.random.default_rng(seed).choice([-13.,-3.,-2.,-1.,0.,1.,2.,3.,13.],(nz,nc+1)))
    if seed==0: pv.fill(13)
    if seed==1: pv.fill(0)
    if seed==2: pv[:3,:]=0;pv[5:,:]=0 # connected lateral/vertical regions
    want=F(np.full(nc,-777),np.int32);mask=F(np.zeros((nz,nc+1)),np.int32)
    native("replay_"+mode,nc,nz,me,h["nEdgesOnCell"],h["cellsOnCell"],h["latCell"][:nc].copy(),
        pv,want,mask,np.float32(2),np.float32(10))
    d["ertel_pv"][...]=cp.asarray(pv,order="F")
    port=PVDiagnostics(m,s,d)
    if mode=="tropo": port.flood_fill_tropo();actualmask=d["inTropo"]
    else: actualmask=port.flood_fill_strato()
    words(f"{mode}_levels_seed_{seed}",want,d["iLev_DT"][:nc].get())
    words(f"{mode}_mask_seed_{seed}",mask,actualmask.get())


def test_pv_interpolation_endpoints_and_small_pv_delta(gpu,native):
    cp=gpu
    nc,nz=36,9
    rng=np.random.default_rng(837)
    lat=F(np.tile([0.0,-0.0,0.5,-0.8],9))
    lev=F(np.tile([0,2,3,4,7,8,9,10,11],4),np.int32)
    pv=F(rng.uniform(-4,4,(nz,nc)))
    field=F(rng.uniform(-100,300,(nz,nc)))
    pv[:,2]=2.0;pv[:,3]=np.arange(nz,dtype=np.float32)*1.e-7
    expected=F(np.zeros(nc))
    native("replay_interp",nc,nz,lat,lev,pv,field,np.float32(2.0),expected)
    d={"ertel_pv":cp.asarray(pv,order="F"),"iLev_DT":cp.asarray(lev)}
    port=PVDiagnostics({"nCells":nc,"nCellsSolve":nc,"nVertLevels":nz,"maxEdges":3,"latCell":cp.asarray(lat)}, {}, d)
    out=cp.empty(nc,dtype=cp.float32)
    port.interp_pv(cp.asarray(field,order="F"),out)
    words("interp_pv",expected,out.get())


def test_budget_order_and_density_uncoupling(gpu,native):
    cp=gpu
    nc,nz,ne=5,9,7
    rng=np.random.default_rng(954)
    rho=F(rng.uniform(0.001,2,(nz,nc)))
    euler=F(rng.uniform(-10,10,(nz,ne)));rhoedge=F(rng.uniform(0.001,2,(nz,ne)))
    phys=F(rng.uniform(-10,10,(nz,ne)));w=F(rng.uniform(-10,10,(nz+1,nc)))
    terms=[F(rng.uniform(-0.1,0.1,(nz,nc))) for _ in range(6)]
    terms[0][0,0]=1e20;terms[1][0,0]=-1e20;terms[2][0,0]=1 # order adversary
    want_phys=phys.copy(order="F");want_w=w.copy(order="F");want_sum=F(np.zeros((nz,nc)))
    native("replay_budget",nc,nz,ne,rho,euler,rhoedge,want_phys,want_w,*terms,want_sum)
    p=PVDiagnostics({}, {}, {})
    dev_phys=cp.asarray(phys,order="F");dev_w=cp.asarray(w,order="F");out=cp.empty((nz,nc),dtype=cp.float32,order="F")
    p._kernel("uncouple_u",nz*ne,(cp.int32(nz*ne),cp.asarray(euler,order="F"),cp.asarray(rhoedge,order="F"),dev_phys))
    p._kernel("uncouple_w",(nz+1)*nc,(cp.int32(nc),cp.int32(nz),cp.asarray(rho,order="F"),dev_w))
    p._kernel("diab_sum",nz*nc,(cp.int32(nz*nc),*(cp.asarray(t,order="F") for t in terms),out))
    words("tend_u_phys_mutation",want_phys,dev_phys.get());words("tend_w_euler_mutation",want_w,dev_w.get())
    words("depv_dt_diab_order",want_sum,out.get())


def test_vertical_curl_native_edge_order(gpu,native):
    cp=gpu
    nz,ne,nv=9,12,8
    rng=np.random.default_rng(904)
    voe=F(rng.integers(1,nv+2,(2,ne)),np.int32)
    dc=F(rng.uniform(100,1e5,ne));area=F(rng.uniform(1e3,1e7,nv));u=F(rng.uniform(-20,20,(nz,ne)))
    want=F(np.zeros((nz,nv+1)))
    native("replay_curl",ne,nv,nz,voe,dc,area,u,want)
    out=cp.empty((nz,nv+1),dtype=cp.float32,order="F")
    PVDiagnostics({}, {}, {})._kernel("vertical_curl",nz,(cp.int32(ne),cp.int32(nv),cp.int32(nz),
        cp.asarray(voe,order="F"),cp.asarray(dc),cp.asarray(area),cp.asarray(u,order="F"),out))
    words("vertical_curl",want,out.get())
    reversed_want=F(np.zeros((nz,nv+1)))
    reversed_args=(F(voe[:,::-1],np.int32),F(dc[::-1]),area,F(u[:,::-1]))
    native("replay_curl",ne,nv,nz,*reversed_args,reversed_want)
    PVDiagnostics()._kernel("vertical_curl",nz,(cp.int32(ne),cp.int32(nv),cp.int32(nz),
        *(cp.asarray(a,order="F") for a in reversed_args),out))
    words("vertical_curl_reversed_native_edge_order",reversed_want,out.get())
    assert np.any(want.view(np.uint32)!=reversed_want.view(np.uint32))


def test_global_pool_restores_native_block_reduction(gpu,native):
    cp=gpu
    nz,ne,nv=9,5,4
    rng=np.random.default_rng(409)
    host_order=np.array([2,0,4,3,1],dtype=np.int32)
    voe=F(np.array([[1,2,3,4,1],[2,3,4,1,3]]),np.int32)
    dc=F(rng.uniform(100,1e5,ne));area=F(rng.uniform(1e3,1e7,nv))
    u=F(rng.uniform(-20,20,(nz,ne)))
    mesh=dict(nCells=2,nVertLevels=nz,maxEdges=3,nEdges=ne,nVertices=nv,
        nEdgesOnCell=cp.asarray([3,3],dtype=cp.int32),
        edgesOnCell=cp.asarray(F(np.array([[3,5],[1,4],[5,2]]),np.int32)),
        verticesOnEdge=cp.asarray(voe,order="F"),dcEdge=cp.asarray(dc),
        areaTriangle=cp.asarray(area))
    diagnostic=PVDiagnostics(mesh)
    assert np.array_equal(diagnostic.set_global_edge_order().get(),host_order)
    want=F(np.zeros((nz,nv+1)))
    native("replay_curl",ne,nv,nz,F(voe[:,host_order],np.int32),
        F(dc[host_order]),area,F(u[:,host_order]),want)
    out=cp.empty((nz,nv+1),dtype=cp.float32,order="F")
    diagnostic.vertical_curl(cp.asarray(u,order="F"),out)
    words("global_pool_native_block_curl",want,out.get())


def test_moist_conversion_and_mix(gpu,native):
    cp=gpu
    n=4096
    rng=np.random.default_rng(604)
    tm,q,rz,zz=[F(a) for a in (rng.uniform(180,400,n),rng.uniform(0,0.03,n),
        rng.uniform(0.001,10,n),rng.uniform(0.1,1,n))]
    want=[F(np.zeros(n)) for _ in range(3)]
    native("replay_prepare",n,tm,q,rz,zz,*want)
    p=PVDiagnostics({}, {}, {})
    out=[cp.empty(n,dtype=cp.float32) for _ in range(3)]
    p._kernel("prepare",n,(cp.int32(n),*(cp.asarray(a) for a in (tm,q,rz,zz)),p.rvord,*out[:2]))
    p._kernel("mix_theta",n,(cp.int32(n),cp.asarray(rz),cp.asarray(q),p.rvord,out[2]))
    for name,want_,actual in zip(("theta","rho","dtheta_dt_mix"),want,out): words(name,want_,actual.get())


def test_sounding_six_columns_shared_glibc(gpu,native):
    cp=gpu
    ns,nz,nc=7,55,8
    rng=np.random.default_rng(307)
    arrays=[F(a) for a in (rng.uniform(230,380,(nz,nc)),rng.uniform(0,0.03,(nz,nc)),
        rng.uniform(0.6,1.1,(nz,nc)),rng.uniform(10000,90000,(nz,nc)),
        rng.uniform(-3000,3000,(nz,nc)),rng.uniform(-30,30,(nz,nc)),rng.uniform(-30,30,(nz,nc)),
        np.cumsum(rng.uniform(50,300,(nz+1,nc)),axis=0))]
    arrays[1][:8,0]=0 # relative humidity floor
    arrays[5][:5,0]=0;arrays[6][:5,0]=0 # calm wind
    arrays[5][5:8,0]=F([0,10,-10]);arrays[6][5:8,0]=0 # wind cardinal directions
    cells=F([1,3,7,2,6,5,0],np.int32);owned=F([1,1,1,1,1,1,0],np.int32)
    want=F(np.full((6,nz,ns),-999.0))
    native("replay_sounding",ns,nz,nc,cells,owned,*arrays,want)
    state={name:cp.asarray(a,order="F") for name,a in zip(("theta_m","qv"),arrays[:2])}
    diag={name:cp.asarray(a,order="F") for name,a in zip(("exner","pressure_base","pressure_p",
        "uReconstructZonal","uReconstructMeridional"),arrays[2:7])}
    port=Soundings({"nVertLevels":nz,"zgrid":cp.asarray(arrays[7],order="F")},state,diag,Clock(),IO())
    out=cp.full((6,nz,ns),-999.0,dtype=cp.float32,order="F")
    port._kernel("sounding")(((ns*nz+127)//128,),(128,),(
        cp.int32(ns),cp.int32(nz),cp.asarray(cells),cp.asarray(owned),
        *(cp.asarray(a,order="F") for a in arrays),port.rvord,port.pi,out))
    for i,name in enumerate(("PRES","HGHT","TMPC","DWPC","DRCT","SPED")):
        words("sounding_"+name,want[i],out[i].get())


def test_sounding_greedy_nearest_ties_and_partition_ownership(gpu,native):
    cp=gpu
    nc,ns,me,solve=9,8,2,6
    # Connected ring, duplicate station, poles and dateline traversal.
    nec=F(np.full(nc,me),np.int32)
    coc=F([[((c-1)%nc)+1 for c in range(nc)], [((c+1)%nc)+1 for c in range(nc)]],np.int32)
    latcell=F(np.linspace(-1.2,1.2,nc));loncell=F(np.linspace(0,5,nc))
    lat=F([0,30,-30,60,-60,0,0,89]);lon=F([0,60,120,180,240,0,360,-180])
    cells=F(np.zeros(ns),np.int32);owned=F(np.zeros(ns),np.int32)
    native("replay_locate",ns,nc,solve,me,nec,coc,latcell,loncell,lat,lon,cells,owned)
    port=Soundings({}, {}, {},Clock(),IO())
    actualcells=cp.empty(ns,dtype=cp.int32);actualowned=cp.empty(ns,dtype=cp.int32)
    port._kernel("locate")((1,),(1,),(cp.int32(ns),cp.int32(nc),cp.int32(solve),cp.int32(me),
        *(cp.asarray(a,order="F") for a in (nec,coc,latcell,loncell,lat,lon)),port.pi,actualcells,actualowned))
    words("stationCells",cells,actualcells.get());words("stationOwned",owned,actualowned.get())


def test_manager_setup_queries_current_stream_alarm():
    class Demand:
        active=False
        def field_will_be_written(self,name): return self.active and name=="ertel_pv"
    utils=Demand()
    pools=dict(mesh={},state={},diag={},diag_physics={})
    class Diagnostic(PVDiagnostics):
        calls=0
        def atm_compute_pv_diagnostics(self,**kwargs): self.calls+=1
    cloud,pv=CloudDiagnostics(),Diagnostic()
    cloud.setup(pools,Clock(),utils);pv.setup(pools,Clock(),utils)
    cloud.compute();pv.compute()
    assert pv.calls==0
    utils.active=True
    pv.compute()
    assert pv.calls==1


def test_soundings_manager_setup_disabled():
    clock=Clock()
    port=Soundings()
    port.setup({"config_sounding_interval":"none"},dict(mesh={},state={},diag={}),clock,object())
    port.compute()
    assert not clock.alarms


def test_unused_form_ertel_helper_and_mutation(gpu,native):
    cp=gpu
    n=1024
    rng=np.random.default_rng(156)
    curl=F(rng.uniform(-1,1,(3,n)));gradient=F(rng.uniform(-100,100,(3,n)))
    density=F(rng.uniform(0.001,2,n));coordinates=F(rng.uniform(-2,2,(3,3,n)))
    want=F(np.zeros(n));wantcurl=curl.copy(order="F")
    native("replay_form_ertel",n,wantcurl,gradient,density,coordinates,want)
    actual=cp.empty(n,dtype=cp.float32);actualcurl=cp.asarray(curl,order="F")
    PVDiagnostics().form_ertel_pv(actualcurl,cp.asarray(gradient,order="F"),cp.asarray(density),
        cp.asarray(coordinates,order="F"),actual)
    words("formErtelPV",want,actual.get());words("formErtelPV_gradxu_mutation",wantcurl,actualcurl.get())


def test_budget_cartesian_reconstruction_and_qv_selection(gpu,native):
    from hexcore.mpasdiag.cloud_v842 import _qv_array
    cp=gpu
    nc,nz,me,ne=11,9,6,31
    rng=np.random.default_rng(205)
    nec=F(rng.integers(3,me+1,nc),np.int32);eoc=F(rng.integers(1,ne+1,(me,nc)),np.int32)
    coeff=F(rng.uniform(-1,1,(3,me,nc)));u=F(rng.uniform(-20,20,(nz,ne)))
    want=[F(np.zeros((nz,nc))) for _ in range(3)]
    native("replay_reconstruct",nc,nz,me,ne,nec,eoc,coeff,u,*want)
    out=[cp.full((nz,nc+1),-999,dtype=cp.float32,order="F") for _ in range(3)]
    PVDiagnostics()._kernel("reconstruct",nc*nz,(cp.int32(nc),cp.int32(nz),cp.int32(me),
        cp.asarray(nec),cp.asarray(eoc,order="F"),cp.asarray(coeff,order="F"),cp.asarray(u,order="F"),*out))
    for name,a,b in zip("XYZ",want,out):
        words("reconstruct_"+name,a,b[:,:nc].get());assert np.all(b[:,nc].get()==-999)
    scalars=F(rng.uniform(0,0.03,(6,nz,nc)))
    state={"scalars":cp.asarray(scalars,order="F"),"index_qv":3}
    words("scalars_index_qv",scalars[2],_qv_array(state).get())


@pytest.mark.parametrize("seed,solve,present",[(24,4,[1]*6),(25,2,[1,0,1,0,1,0]),(26,2,[0]*6)])
def test_complete_budget_owned_and_halo_scratch(gpu,native,seed,solve,present):
    m,s,d,h,f,w=mesh_fixture(gpu,seed,nz=55)
    replay_complete_budget(gpu,native,m,s,d,h,f,w,seed,solve,present)


def replay_complete_budget(cp,native,m,s,d,h,f,w,seed,solve,present,real_geometry=False):
    nc,nz,me,ne,nv=(m[n] for n in ("nCells","nVertLevels","maxEdges","nEdges","nVertices"))
    m["nCellsSolve"]=solve
    rng=np.random.default_rng(seed)
    coeff=h['coeffs_reconstruct'] if real_geometry else F(rng.normal(0,.01,(3,me,nc+1)))
    m["coeffs_reconstruct"]=cp.asarray(coeff,order="F")
    voe=h['verticesOnEdge'] if real_geometry else F(np.array([[1,2,3,4,1,2],[2,3,4,1,3,4]]),np.int32)
    dc=h['dcEdge'] if real_geometry else F(rng.uniform(.1,30,ne))
    area=h['areaTriangle'] if real_geometry else F(rng.uniform(.1,300,nv))
    for name,v in (("verticesOnEdge",voe),("dcEdge",dc),("areaTriangle",area)):m[name]=cp.asarray(v,order="F")
    euler=F(rng.normal(0,.1,(nz,ne)));rhoedge=F(rng.uniform(.1,2,(nz,ne)))
    phys=F(rng.normal(0,.1,(nz,ne)));we=F(rng.normal(0,.1,(nz+1,nc+1)))
    terms=F(rng.normal(0,.001,(nz,nc+1,6)));out=F(np.full((nz,nc+1,8),-999.))
    expected_phys=phys.copy(order="F");expected_w=we.copy(order="F")
    mesh_args=[h[name] for name in ("nEdgesOnCell","cellsOnCell","edgesOnCell","cellsOnEdge","verticesOnCell","cellsOnVertex","kiteAreasOnVertex","areaCell","dvEdge","edgeNormalVectors","cellTangentPlane","localVerticalUnitVectors","zgrid")]
    native("replay_full_budget",nc,nz,me,nv,ne,solve,*mesh_args,voe,dc,area,coeff,f['theta'],f['rho'],f['uReconstructX'],f['uReconstructY'],f['uReconstructZ'],w,f['vorticity'],F(present,np.int32),terms,euler,rhoedge,expected_phys,expected_w,out)
    td={"u_euler":cp.asarray(euler,order="F"),"w_euler":cp.asarray(we,order="F")}
    tp={}
    for i,(scheme,name) in enumerate(zip(('lw','sw','bl','cu','mp','mix'),('rthratenlw','rthratensw','rthblten','rthcuten','dtheta_dt_mp','dtheta_dt_mix'))):
        (tp if i<4 else d)[name]=cp.asarray(terms[:,:,i],order="F") if present[i] else None
        d['depv_dt_'+scheme]=cp.full((nz,nc+1),-999.,dtype=cp.float32,order="F")
    for name,v in (("tend_u_phys",phys),("rho_edge",rhoedge)):d[name]=cp.asarray(v,order="F")
    for name in ('diab','fric'):d['depv_dt_'+name]=cp.full((nz,nc+1),-999.,dtype=cp.float32,order="F")
    poison=tuple(cp.full((nz,nc+1),cp.nan,dtype=cp.float32,order="F") for _ in range(3))
    PVDiagnostics(m,s,d,td,tp).calc_pv_budget(captured_reconstructed_halos=poison)
    for i,scheme in enumerate(('lw','sw','bl','cu','mp','mix','diab','fric')):
        words('complete-'+scheme,out[:-1,:solve,i],d['depv_dt_'+scheme][:-1,:solve].get())
        words('complete-'+scheme+'-halo',out[:,solve:,i],d['depv_dt_'+scheme][:,solve:].get())
        if (i<6 and not present[i]) or (i==6 and not any(present)):
            words('complete-'+scheme+'-absent-top',out[-1,:solve,i],d['depv_dt_'+scheme][-1,:solve].get())
        else:
            bits=np.full(solve,np.uint32(-(nz+1)&0xffffffff),dtype=np.uint32)
            words('stable-native-'+scheme+'-top',bits.view(np.float32),d['depv_dt_'+scheme][-1,:solve].get())
    words('complete-phys-mutation',expected_phys,d['tend_u_phys'].get())
    words('complete-w-mutation',expected_w,td['w_euler'].get())


def test_rust_sounding_clock_lifecycle():
    from hexcore.mpasdiag.soundings_v842 import RustSoundingIO,RustSoundingClock
    binary=os.environ.get('GPUWM_MPAS_SOUNDINGS_BRIDGE')
    if not binary:pytest.skip('Rust clock binary not supplied')
    current=['2000-02-28_23:59:59']
    clock=RustSoundingClock(lambda:current[0],RustSoundingIO(executable=binary))
    assert not clock.is_ringing('soundingAlarm')
    clock.add_alarm('soundingAlarm',clock.now(),'00:00:02')
    assert clock.is_ringing('soundingAlarm')
    clock.reset_alarm('soundingAlarm')
    assert not clock.is_ringing('soundingAlarm')
    current[0]='2000-02-29_00:00:06'
    assert clock.is_ringing('soundingAlarm')
    clock.reset_alarm('soundingAlarm')
    assert not clock.is_ringing('soundingAlarm')
    assert clock.alarms['soundingAlarm']['previous']=='2000-02-29_00:00:05'
    clock.add_alarm('soundingAlarm',clock.now(),'00:00:03')
    assert clock.alarms['soundingAlarm']['interval']=='00:00:02'


def test_real_captured_columns_cloud_pv_soundings(gpu,native):
    cp=gpu
    path=os.environ.get('MPAS_DIAG_REAL_PATCH')
    if not path:pytest.skip('Rust real-column capture not supplied')
    root=Path(path);manifest=json.loads((root/'patch.json').read_text())
    h={row['field']:np.fromfile(root/(row['field']+'.raw'),dtype=row['dtype']).reshape(row['shape'],order='F') for row in manifest['fields']}
    dev={name:cp.asarray(a,order='F') for name,a in h.items()}
    m={name:manifest[name] for name in ('nCells','nCellsSolve','nVertLevels','nEdges','nVertices','maxEdges')}
    m.update({name:dev[name] for name in ('nEdgesOnCell','cellsOnCell','edgesOnCell','cellsOnEdge','verticesOnCell','cellsOnVertex','kiteAreasOnVertex','areaCell','dvEdge','edgeNormalVectors','cellTangentPlane','localVerticalUnitVectors','zgrid','latCell')})
    nc,ns,nz=(manifest[name] for name in ('nCells','nCellsSolve','nVertLevels'))
    d={name:dev[name] for name in ('theta','rho','uReconstructX','uReconstructY','uReconstructZ','vorticity','ertel_pv','iLev_DT','uReconstructZonal','uReconstructMeridional','u_pv','v_pv','theta_pv','vort_pv')}
    p=PVDiagnostics(m,{'w':dev['w']},d)
    p.calc_epv()
    words('real-native-ertel_pv',h['ertel_pv'][:,:ns],d['ertel_pv'][:,:ns].get())
    p.interp_pv_diagnostics()
    for name in ('u_pv','v_pv','theta_pv','vort_pv'):words('real-native-'+name,h[name][:ns],d[name][:ns].get())
    dcloud={name:dev[name] for name in ('pressure_base','pressure_p',*FIELDS)}
    CloudDiagnostics(m,dcloud,{'cldfrac':dev['cldfrac']}).compute(FIELDS)
    for name in FIELDS:words('real-native-'+name,h[name][:ns],dcloud[name][:ns].get())
    names=('theta_m','qv','exner','pressure_base','pressure_p','uReconstructZonal','uReconstructMeridional','zgrid')
    arrays=[h[name] for name in names]
    cells=F(np.arange(1,ns+1),np.int32);owned=F(np.ones(ns),np.int32)
    want=F(np.zeros((6,nz,ns)))
    native('replay_sounding',ns,nz,nc+1,cells,owned,*arrays,want)
    port=Soundings();out=cp.empty((6,nz,ns),dtype=cp.float32,order='F')
    port._kernel('sounding')(((ns*nz+127)//128,),(128,),(
        cp.int32(ns),cp.int32(nz),cp.asarray(cells),cp.asarray(owned),
        *(dev[name] for name in names),port.rvord,port.pi,out))
    words('real-column-six-sounding-values',want,out.get())


def test_interpolation_native_prefix_alias_and_nan(gpu,native):
    cp=gpu
    nc,nz=4,9
    pv=F(np.linspace(-5,5,nc*nz).reshape(nz,nc));field=F(np.linspace(-50,50,nc*nz).reshape(nz,nc))
    pv[-1,2]=np.array([0xffffffc8],dtype=np.uint32).view(np.float32)[0]
    lev=F([1,1,0,1],np.int32);lat=F([.2,-.2,0.,-.1]);expected=F(np.zeros(nc))
    native('replay_interp_prefix',nc,nz,lat,lev,F(np.r_[np.float32(0),pv.ravel(order='F')]),F(np.r_[np.float32(0),field.ravel(order='F')]),np.float32(2.),expected)
    d={'ertel_pv':cp.asarray(pv,order='F'),'iLev_DT':cp.asarray(lev)}
    p=PVDiagnostics({'nCells':nc,'nCellsSolve':nc,'nVertLevels':nz,'maxEdges':3,'latCell':cp.asarray(lat)}, {},d)
    out=cp.empty(nc,dtype=cp.float32);p.interp_pv(cp.asarray(field,order='F'),out)
    words('interp-native-prefix-previous-column-NaN',expected,out.get())


@pytest.mark.parametrize("present",[(1,1,1,1,1,1),(1,0,1,0,1,0),(0,0,0,0,0,0)])
def test_real_column_budget_strict_driver(gpu,native,present):
    path=os.environ.get('MPAS_DIAG_REAL_PATCH')
    if not path:pytest.skip('Rust real-column capture not supplied')
    root=Path(path);manifest=json.loads((root/'patch.json').read_text())
    h={r['field']:np.fromfile(root/(r['field']+'.raw'),dtype=r['dtype']).reshape(r['shape'],order='F') for r in manifest['fields']}
    m={name:manifest[name] for name in ('nCells','nCellsSolve','nVertLevels','nEdges','nVertices','maxEdges')}
    m.update({n:gpu.asarray(a,order='F') for n,a in h.items()})
    f={n:h[n] for n in ('theta','rho','uReconstructX','uReconstructY','uReconstructZ','vorticity')}
    d={n:gpu.asarray(a,order='F') for n,a in f.items()};w=h['w'];state={'w':gpu.asarray(w,order='F')}
    # Real native state/geometry; synthetic tendencies reach every association.
    replay_complete_budget(gpu,native,m,state,d,h,f,w,730,manifest['nCellsSolve'],present,real_geometry=True)


def test_unused_index_and_height_helpers(gpu,native):
    cp=gpu;nc,nz,sz=6,9,4
    values=F([1,2,3,4,9,1],np.int32)
    arrays=F(np.tile([1,2,3,4],(nc,1)).T,np.int32);arrays[:,5]=F([1,1,2,3],np.int32)
    z=F(np.cumsum(np.random.default_rng(680).uniform(-500,1500,(nz+1,nc)),axis=0))
    expected=[F(np.zeros(nc),np.int32),F(np.zeros((nz,nc))),F(np.zeros((nz,nc)))]
    native('replay_helpers',nc,nz,sz,values,arrays,z,*expected)
    outputs=[cp.empty_like(cp.asarray(a,order='F')) for a in expected]
    PVDiagnostics()._kernel('helper_replay',nc,(cp.int32(nc),cp.int32(nz),cp.int32(sz),
        cp.asarray(values),cp.asarray(arrays,order='F'),cp.asarray(z,order='F'),*outputs))
    for name,want,out in zip(('native-index-first-absent','native-cell-height','native-edge-height'),expected,outputs):words(name,want,out.get())


@pytest.mark.parametrize("mode,scenario",[("tropo","bridge"),("strato","bridge"),("strato","detach"),("strato","equal")])
def test_flood_fill_directional_bridges_and_surface_detachment(gpu,native,mode,scenario):
    cp=gpu;m,state,d,h,f,w=mesh_fixture(cp,117,nz=12)
    nc,nz,me=(m[n] for n in ('nCells','nVertLevels','maxEdges'))
    nec=F([2,1,2,1,0],np.int32);coc=F(np.full((me,nc+1),nc+1),np.int32)
    coc[:2,0]=[2,3];coc[0,1]=1;coc[:2,2]=[1,4];coc[0,3]=3
    lat=F(np.ones(nc+1));pv=F(np.full((nz,nc+1),13. if mode=='tropo' else 0.))
    if mode=='tropo':
        pv[:,2]=0.;pv[-1,0]=0.;pv[5:,1]=0.
    elif scenario=='bridge':
        pv[:,2]=13.;pv[2,0]=13.;pv[2:6,1]=13.
    elif scenario=='detach':
        pv[:,2]=13.;pv[0,0]=13.;pv[3:,0]=13.;pv[:,1]=3.
    else:
        pv[:,0]=3.;pv[:,1:3]=13.;pv[0,2]=2.
    m.update(nEdgesOnCell=cp.asarray(nec),cellsOnCell=cp.asarray(coc,order='F'),latCell=cp.asarray(lat))
    want=F(np.full(nc,-777),np.int32);mask=F(np.zeros((nz,nc+1)),np.int32)
    native('replay_'+mode,nc,nz,me,nec[:nc].copy(),coc[:,:nc].copy(order='F'),lat[:nc].copy(),pv,want,mask,np.float32(2),np.float32(10))
    d['ertel_pv'][...]=cp.asarray(pv,order='F');port=PVDiagnostics(m,state,d)
    if mode=='tropo':port.flood_fill_tropo();actual=d['inTropo']
    else:actual=port.flood_fill_strato()
    words(mode+'-'+scenario+'-levels',want,d['iLev_DT'][:nc].get())
    words(mode+'-'+scenario+'-mask',mask,actual.get())


def test_real_sounding_lifecycle_rust_files_match_native(gpu,native):
    """Actual Rust parser/alarm/writer plus GPU locate/profile, on real state."""
    import struct,subprocess
    from hexcore.mpasdiag.soundings_v842 import RustSoundingIO,RustSoundingClock
    binary=os.environ.get('GPUWM_MPAS_SOUNDINGS_BRIDGE');patch=os.environ.get('MPAS_DIAG_REAL_PATCH')
    if not binary or not patch:pytest.skip('real patch and Rust bridge required')
    oracle=Path(binary).with_name('sounding_io')
    if not oracle.exists():pytest.skip('strict native sounding writer required')
    root=Path(patch);manifest=json.loads((root/'patch.json').read_text())
    tmp_path=root.resolve().parents[1]/'scratch'/f'gpu-sounding-lifecycle-{os.getpid()}'
    tmp_path.mkdir(parents=True,exist_ok=True)
    h={r['field']:np.fromfile(root/(r['field']+'.raw'),dtype=r['dtype']).reshape(r['shape'],order='F') for r in manifest['fields']}
    cp=gpu;nc,solve,nz,me=(manifest[n] for n in ('nCells','nCellsSolve','nVertLevels','maxEdges'))
    # Last owned column's neighborhood and the initial halo cell both occur in
    # this captured patch. Duplicate station and byte names exercise real I/O.
    pi=np.float32(float.fromhex('0x1.921fb6p+1'));indices=[solve-1,nc-1,solve-1]
    lat=F(h['latCell'][indices]*np.float32(180)/pi);lon=F(h['lonCell'][indices]*np.float32(180)/pi)
    names=[b'owned',b'halo','é☁'.encode()]
    lines=[f'{a:.9g} {b:.9g} '.encode()+b"'"+name+b"'\n" for a,b,name in zip(lat,lon,names)]
    (tmp_path/'sounding_locations.txt').write_bytes(b''.join(lines))
    expected_cells=F(np.zeros(3),np.int32);expected_owned=F(np.zeros(3),np.int32)
    native('replay_locate',3,nc,solve,me,h['nEdgesOnCell'][:nc].copy(),h['cellsOnCell'][:,:nc].copy(order='F'),h['latCell'][:nc].copy(),h['lonCell'][:nc].copy(),lat,lon,expected_cells,expected_owned)
    assert expected_owned[0] and not expected_owned[1] and expected_owned[2]
    expected_rows=F(np.zeros((6,nz,3)))
    fields=('theta_m','qv','exner','pressure_base','pressure_p','uReconstructZonal','uReconstructMeridional','zgrid')
    native('replay_sounding',3,nz,nc+1,expected_cells,expected_owned,*(h[n] for n in fields),expected_rows)
    mesh={n:manifest[n] for n in ('nCells','nCellsSolve','nVertLevels','maxEdges')}
    mesh.update({n:cp.asarray(h[n],order='F') for n in ('nEdgesOnCell','cellsOnCell','latCell','lonCell','zgrid')})
    state={n:cp.asarray(h[n],order='F') for n in ('theta_m','qv')}
    diag={n:cp.asarray(h[n],order='F') for n in fields[2:-1]}
    io=RustSoundingIO(executable=binary,directory=tmp_path);now=['2000-02-29_23:59:03']
    clock=RustSoundingClock(lambda:now[0],io)
    port=Soundings(mesh,state,diag,clock,io);port.setup('00:06:00')
    words('real-lifecycle-station-cells',expected_cells,port.cells.get())
    words('real-lifecycle-station-owned',expected_owned,port.owned.get())
    native_directory=tmp_path/'native';native_directory.mkdir()
    raw=tmp_path/'profiles.raw';raw.write_bytes(struct.pack('<I',nz)+expected_owned.tobytes()+expected_rows.tobytes(order='F'))
    for time in (now[0],'2000-03-01_00:05:03'):
        now[0]=time;port.compute();assert not clock.is_ringing('soundingAlarm')
        subprocess.run([str(oracle), 'write',str(tmp_path/'sounding_locations.txt'),str(raw),time],cwd=native_directory,check=True,capture_output=True)
        for file in native_directory.glob('*.snd'):
            expected=file.read_bytes();actual=(tmp_path/file.name).read_bytes();assert actual==expected
            RECEIPT_ROWS.append(dict(field='real-lifecycle-sounding-file-'+file.name,bytes=len(actual),differing_bytes=0,sha256=hashlib.sha256(actual).hexdigest()))
        assert {f.name for f in native_directory.glob('*.snd')}=={f.name for f in tmp_path.glob('*.snd')}
    port.cleanup();assert port.locations is None and port.cells is None and port.owned is None
    for file in native_directory.iterdir():file.unlink()
    native_directory.rmdir()
    for file in tmp_path.iterdir():file.unlink()
    tmp_path.rmdir()


def test_real_native_executable_sounding_files(gpu):
    """Compare column launcher/Rust files with the actual pinned native binary.

    The capture stations are at the four selected global cell centers; this
    test supplies their captured cell mapping, rather than a partial mesh walk.
    The native-executable fixture has a zero-duration initialized run, so it
    does not substitute for either-suite forecast or distributed proof.
    """
    from hexcore.mpasdiag.soundings_v842 import RustSoundingIO,RustSoundingClock
    binary=os.environ.get('GPUWM_MPAS_SOUNDINGS_BRIDGE')
    if not binary:pytest.skip('Rust sounding bridge required')
    root=Path(binary).parent;patch=root/'native-station-patch';files=root/'native-station-files'
    if not (patch/'patch.json').exists() or not files.exists():pytest.skip('actual native station capture required')
    manifest=json.loads((patch/'patch.json').read_text())
    h={r['field']:np.fromfile(patch/(r['field']+'.raw'),dtype=r['dtype']).reshape(r['shape'],order='F') for r in manifest['fields']}
    scratch=root.resolve().parent/'scratch'/f'actual-native-soundings-{os.getpid()}';scratch.mkdir(parents=True,exist_ok=True)
    (scratch/'sounding_locations.txt').write_bytes((files/'sounding_locations.txt').read_bytes())
    cp=gpu;mesh={'nVertLevels':manifest['nVertLevels'],'zgrid':cp.asarray(h['zgrid'],order='F')}
    state={n:cp.asarray(h[n],order='F') for n in ('theta_m','qv')}
    diag={n:cp.asarray(h[n],order='F') for n in ('exner','pressure_base','pressure_p','uReconstructZonal','uReconstructMeridional')}
    # Timestamp comes from the actual native minute filename, not a default.
    native_files=sorted(files.glob('col*.snd'));assert len(native_files)==manifest['nCellsSolve']==4
    stamp=native_files[0].name.split('.')[1]
    timestamp=f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}_{stamp[8:10]}:{stamp[10:12]}:00'
    io=RustSoundingIO(executable=binary,directory=scratch);clock=RustSoundingClock(lambda:timestamp,io)
    port=Soundings(mesh,state,diag,clock,io);port.locations=io.read_locations('sounding_locations.txt')
    port.cells=cp.arange(1,5,dtype=cp.int32);port.owned=cp.ones(4,dtype=cp.int32)
    clock.add_alarm('soundingAlarm',timestamp,'00:06:00');port.compute()
    for file in native_files:
        expected=file.read_bytes();actual=(scratch/file.name).read_bytes();assert expected==actual
        RECEIPT_ROWS.append(dict(field='actual-native-sounding-file-'+file.name,bytes=len(actual),differing_bytes=0,sha256=hashlib.sha256(actual).hexdigest()))
    assert {f.name for f in native_files}=={f.name for f in scratch.glob('*.snd')}
    port.cleanup()
    for file in scratch.iterdir():file.unlink()
    scratch.rmdir()


def test_cloud_native_nan_and_signed_zero_max(gpu,native):
    cp=gpu;nc,nz=5,4
    cf=F([[.3,.3,-0.,np.nan,-0.],[np.nan,.7,-0.,np.nan,-0.],
          [.5,np.nan,-0.,np.nan,-0.],[.4,.6,-0.,np.nan,-0.]])
    # Distinct NaN payloads and signs must survive the native MAX selection.
    cf[3,3]=np.array([0xffc01234],dtype=np.uint32).view(np.float32)[0]
    cf[0,0]=np.float32(np.nan)
    pb=F(np.full((nz,nc),100000));pp=F(np.zeros((nz,nc)))
    want=[F(np.zeros(nc)) for _ in FIELDS]
    native('replay_cloud',nc,nz,pp,pb,cf,*want)
    diag={'pressure_p':cp.asarray(pp,order='F'),'pressure_base':cp.asarray(pb,order='F')}
    diag.update({name:cp.empty(nc,dtype=cp.float32) for name in FIELDS})
    CloudDiagnostics({'nCellsSolve':nc,'nVertLevels':nz},diag,{'cldfrac':cp.asarray(cf,order='F')}).compute(FIELDS)
    for name,expected in zip(FIELDS,want):words('native-MAX-'+name,expected,diag[name].get())


def test_native_subnormal_cloud_and_pv_arithmetic(gpu, native):
    # Keep subnormal selection words and Ertel multiplication/addition results.
    # Forced FTZ makes the native nonzero PV results zero.
    cp = gpu
    cf = F(np.array([[1, 2, 0x007fffff]], dtype=np.uint32).view(np.float32))
    pb = F(np.full(cf.shape, 70000.0, dtype=np.float32))
    pp = F(np.zeros_like(pb))
    expected = [F(np.zeros(cf.shape[1])) for _ in FIELDS]
    native("replay_cloud", cf.shape[1], 1, pp, pb, cf, *expected)
    diag = {"pressure_p": cp.asarray(pp, order="F"),
            "pressure_base": cp.asarray(pb, order="F")}
    diag.update({name: cp.zeros(cf.shape[1], dtype=cp.float32) for name in FIELDS})
    CloudDiagnostics({"nCellsSolve": cf.shape[1], "nVertLevels": 1}, diag,
                     {"cldfrac": cp.asarray(cf, order="F")}).compute(FIELDS)
    for name, want in zip(FIELDS, expected):
        words("subnormal_" + name, want, diag[name].get())
    assert diag[FIELDS[0]].get().view(np.uint32).tolist() == [1, 2, 0x007fffff]
    n = cf.shape[1]
    curl = F(np.broadcast_to(np.array([1, 0, 0], dtype=np.float32)[:, None], (3, n)))
    gradient = F(np.zeros((3, n)))
    gradient[0] = cf[0]
    density = F(np.ones(n))
    coordinates = F(np.repeat(np.eye(3, dtype=np.float32)[:, :, None], n, axis=2))
    wantcurl = curl.copy(order="F")
    want = F(np.zeros(n))
    native("replay_form_ertel", n, wantcurl, gradient, density, coordinates, want)
    actualcurl = cp.asarray(curl, order="F")
    actual = cp.zeros(n, dtype=cp.float32)
    PVDiagnostics().form_ertel_pv(actualcurl, cp.asarray(gradient, order="F"),
        cp.asarray(density), cp.asarray(coordinates, order="F"), actual)
    words("subnormal_pv_curl_mutation", wantcurl, actualcurl.get())
    words("subnormal_pv", want, actual.get())
    assert np.all(want > 0)


def test_real_native_pool_output_adapter_refresh_and_alarm(gpu, native, tmp_path):
    """Exercise the output adapter with real f3 pools, streams and clock alarms."""
    from types import SimpleNamespace
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.session_v842 import NativePhysicsSession
    from hexcore.mpasdiag.cloud_v842 import NativeCloudPVSoundingsOutput
    library = os.environ.get('MPAS_DIAG_POOL_LIBRARY')
    if not library:
        pytest.skip('real native pool provider required')
    nc, nz = 3, 6
    dimensions = dict(nCells=nc, nCellsSolve=nc, nVertLevels=nz)
    pp = F(np.zeros((nz, nc+1)))
    pb = F(np.broadcast_to(np.array([90000, 64200, 50000, 35000, 20000, 15000],
                                    dtype=np.float32)[:, None], pp.shape))
    fraction = F(np.arange(nz*(nc+1), dtype=np.float32).reshape(pp.shape) / 32)
    with NativePhysics(library, test_only=True) as provider:
        session = NativePhysicsSession.__new__(NativePhysicsSession)
        session.native = provider
        session.layout = SimpleNamespace(dimensions=dimensions, indices={},
                                         configs={'config_sounding_interval': 'none'})
        for name, value in dimensions.items():
            provider.dimension('mesh', name, value)
        provider.put_text_field('state', 'xtime', '2023-03-31_12:00:00')
        for pool, name, value in (('diag', 'pressure_p', pp), ('diag', 'pressure_base', pb),
                                 ('diag_physics', 'cldfrac', fraction)):
            provider.put_field(pool, name, value)
        for name in FIELDS:
            provider.put_field('diag', name, F(np.full(nc+1, -17)))
        provider.clock(start='2023-03-31_12:00:00', stop='2023-03-31_14:00:00', dt=1800)
        provider.streams()
        provider.block()
        provider.decomposition('nCells', global_ids=np.arange(1, nc+1, dtype=np.int32), owned=nc)
        for name in FIELDS:
            provider.register_field('diag', name, dimensions=('nCells',), decomposed=True)
        provider.stream('diag_o4', filename=str(tmp_path/'cloud.nc'), direction='output',
                        input_interval='none', output_interval='0_01:00:00')
        streams = (SimpleNamespace(name='diag_o4', fields=FIELDS, directions=('output',),
                                   output_interval='0_01:00:00'),)
        adapter = NativeCloudPVSoundingsOutput(session, streams, tmp_path, gpu)
        missing = {'diag_o4': [*FIELDS, 'another_lane_field']}
        adapter.attach(missing)
        assert missing == {'diag_o4': ['another_lane_field']}
        for step in (0, 1, 2):
            due = step != 1
            assert adapter.field_will_be_written(FIELDS[0]) == due
            if step == 2:
                fraction.fill(np.float32(.875))
                provider.put_field('diag_physics', 'cldfrac', fraction)
            adapter.compute(step)
            assert bool(adapter.calls[-1]['fields']) == due
            if due:
                wanted = [F(np.zeros(nc+1)) for _ in FIELDS]
                native('replay_cloud', nc, nz, pp, pb, fraction, *wanted)
                for name, want in zip(FIELDS, wanted):
                    got = provider.get_field('diag', name)
                    words('native-pool-adapter-'+str(step)+'-'+name, want[:nc], got[:nc])
                    assert got[-1] == np.float32(-17)
                provider.reset_alarm('diag_o4:out')
            provider.advance_clock()
        # The Registry name differs from its tendency-pool name. Exercise the
        # actual collective through that alias rather than a mock callback.
        euler = F(np.arange(nz*(nc+1), dtype=np.float32).reshape((nz, nc+1)))
        provider.put_field('tend', 'u_euler', euler)
        provider.register_field('tend', 'u_euler', dimensions=('nVertLevels', 'nCells'),
                                decomposed=True, registry_name='euler_tend_u')
        view = adapter.pools['tend']['u_euler']
        adapter.exchange('u_euler', view)
        words('native-pool-registry-tendency-alias', euler, view.get())
        adapter.cleanup()
        assert all(not pool.arrays for pool in adapter.pools.values())
