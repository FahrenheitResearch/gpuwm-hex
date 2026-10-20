"""Whole native fork initialization routines against the Rust bridges.

RRFSx/MPAS-Model src/core_atmosphere/mpas_atm_core.F:1266-1309 and
src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
"""
import ctypes as ct
import json,os,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

def call(path,name,fields):
    lib=ct.CDLL(str(Path(path).resolve()))
    src=Path(path).with_name('oracle.f90').read_text()
    signature=re.search(r'\bsubroutine\s+'+name+r'\((.*?)\)\s*bind',src,re.I|re.S)[1]
    keys=re.sub(r'[&\s]','',signature).lower().split(',');args=[];keep=[]
    for key in keys:
        x=fields[key]
        if isinstance(x,np.ndarray):
            assert x.flags.f_contiguous,key
            args.append(x.ctypes.data_as(ct.c_void_p))
        else:
            v=ct.c_int(x) if isinstance(x,int) else ct.c_float(x);keep.append(v);args.append(ct.byref(v))
    fn=getattr(lib,name);fn.restype=None;fn(*args)

@pytest.mark.parametrize('zd,xnutr',[(0.,.2),(16000.,.2),(40000.,.2),(16000.,0.)])
def test_full_native_damping(zd,xnutr):
    from hexcore.damping_rrfs import build_rrfs_vertical_velocity_damping
    path=os.environ.get('RRFS_DAMPING_ORACLE');bridge=os.environ.get('RW_MPAS_RRFS_DAMPING')
    if not path or not bridge:pytest.skip('native oracle and Rust bridge are required')
    nl=55;nc=12;rng=np.random.default_rng(543)
    heights=np.asfortranarray((np.linspace(0,32000,nl+1)[:,None]*rng.uniform(.8,1.4,nc)).astype('f4'))
    density=rng.uniform(.02,1.6,nc).astype('f4');dss=np.zeros((nl,nc),'f4',order='F')
    call(path,'atm_compute_damping_coefs',dict(ncells=nc,nvertlevels=nl,zgrid=heights,meshdensity=density,config_zd=zd,config_xnutr=xnutr,dss=dss))
    got=build_rrfs_vertical_velocity_damping(heights,density,xnutr=xnutr,damping_start_z=zd,bridge=bridge)
    diff=int(np.count_nonzero(got.view('u4')!=dss.view('u4')))
    if diff:
        key=tuple(np.argwhere(got.view('u4')!=dss.view('u4'))[0]);print(key,got[key],dss[key])
    assert diff==0
    out=os.environ.get('RRFS_RECEIPTS')
    if out:Path(out,f'damping-zd{zd}-xnutr{xnutr}.json').write_text(json.dumps(dict(tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,words=dss.size,differing_words=diff),indent=2)+'\n')

@pytest.mark.parametrize('sphere',[False,True])
@pytest.mark.parametrize('wide',[False,True])
def test_full_native_deformation(sphere,wide):
    from hexcore.deformation_rrfs import initialize_deformation_weights_rrfs
    path=os.environ.get('RRFS_DEFORMATION_ORACLE');bridge=os.environ.get('RW_MPAS_RRFS_DEFORMATION')
    if not path or not bridge:pytest.skip('native oracle and Rust bridge are required')
    nc=4;me=6;nv=nc*me;ne=nc*me;radius=np.float32(6371220. if sphere else 1.)
    counts=np.array([6,5,4,6],'i4');centers=np.array([[0,0,1],[1,0,0],[0,1,0],[0,0,-1]],'f4') if sphere else np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],'f4')
    coords=[]
    for center in centers:
        tangent=np.cross(center,[1,0,0]) if sphere and abs(center[0])<.5 else np.array([0,1,0]);other=np.cross(center,tangent) if sphere else np.array([1,0,0])
        for j in range(me):
            a=2*np.pi*j/me;point=center+.15*(np.cos(a)*tangent+np.sin(a)*other)
            if sphere:point=point/np.linalg.norm(point)
            coords.append(point)
    coords=np.array(coords,'f4')*radius;centers*=radius
    if wide:
        coords=coords.astype('f8')+np.float64(.00000003)
        centers=centers.astype('f8')+np.float64(.00000003)
    eoc=np.arange(ne,dtype='i4').reshape(nc,me);voc=np.arange(nv,dtype='i4').reshape(nc,me);coc=np.zeros((nc,me),'i4');coc[3,:]=nc
    coc[1,0]=-1  # native regional zero sentinel does not skip this cell
    coe=np.zeros((ne,2),'i4')
    for c in range(nc):
        for j in range(me):coe[c*me+j]=[c,(c+1)%nc] if j%2 else [(c+1)%nc,c]
    mesh=SimpleNamespace(nEdgesOnCell=counts,edgesOnCell=eoc,cellsOnCell=coc,verticesOnCell=voc,cellsOnEdge=coe,
        xCell=centers[:,0],yCell=centers[:,1],zCell=centers[:,2],xVertex=coords[:,0],yVertex=coords[:,1],zVertex=coords[:,2])
    got=initialize_deformation_weights_rrfs(mesh,on_a_sphere=sphere,sphere_radius=radius,bridge=bridge)
    a=dict(ncells=nc,nedges=ne,nvertices=nv,maxedges=me,on_a_sphere=int(sphere),sphere_radius=float(radius),nedgesoncell=counts,
        cellsonedge=np.asfortranarray((coe+1).T),edgesoncell=np.asfortranarray((eoc+1).T),cellsoncell=np.asfortranarray((coc+1).T),verticesoncell=np.asfortranarray((voc+1).T))
    for name in ('xCell','yCell','zCell','xVertex','yVertex','zVertex'):a[name.lower()]=np.ascontiguousarray(getattr(mesh,name),dtype='f4')
    for name in got:a[name]=np.zeros((me,nc),'f4',order='F')
    call(path,'atm_initialize_deformation_weights',a)
    checks=[]
    for name,actual in got.items():
        want=np.ascontiguousarray(a[name].T);diff=int(np.count_nonzero(actual.view('u4')!=want.view('u4')))
        checks.append(dict(field=name,words=actual.size,differing_words=diff))
        if diff:
            key=tuple(np.argwhere(actual.view('u4')!=want.view('u4'))[0]);print(name,key,actual[key],want[key])
        assert diff==0
    out=os.environ.get('RRFS_RECEIPTS')
    if out:Path(out,f'deformation-sphere{int(sphere)}-wide{int(wide)}.json').write_text(json.dumps(dict(tier='BITWISE SUBSYSTEM ONLY',native_ranks=1,branches=['polygon 4/5/6','sign reversal','outside halo','regional zero sentinel','f64 pool narrowing' if wide else 'f32 coordinates','north pole' if sphere else 'plane'],comparisons=checks),indent=2)+'\n')
