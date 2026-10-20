"""Strict fork coupling/carriers vs CUDA, synthetic x1.2562 column extent.

Momentum projection is supplied by the harness; this is not forecast proof.
Run only on a card; on a shared host go through its GPU and build queues
(HEX_GPU_QUEUE, HEX_BUILD_QUEUE) for the direct NVRTC compilation.
"""
import ctypes
from itertools import product
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_mpasphys_numeric_v842 import columns, f32, NAMES, RATES

pytestmark=pytest.mark.skipif(os.environ.get('RRFS_PHYSICS_GPU_TESTS')!='1',
                            reason='explicit GPU/build slot allocation required')
PBL=('off','bl_ysu','bl_mynn','bl_mynnedmf','bl_myj')
CONVECTION=('off','cu_grell_freitas','cu_kain_fritsch','cu_tiedtke','cu_ntiedtke','cu_grell_freitas_li')
MICRO=('mp_wsm6','mp_thompson_aerosols','mp_tempo','mp_tempo')


@pytest.fixture(scope='module')
def authority():
    folder=Path(os.environ['RRFS_PHYSICS_REFERENCE_DIR'])
    lib=ctypes.CDLL(str(folder/'librrfs.so'))
    lib.couple.argtypes=[ctypes.c_int]*13+[ctypes.c_void_p]*11
    numbers=ctypes.CDLL(str(folder/'libnumbers.so'))
    numbers.numbers.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*3
    from hexcore.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache
    import cupy as cp
    return lib,numbers,StrictPhysicsKernelCache(),cp


@pytest.mark.parametrize('pbl,cu,micro,lw,sw',list(product(range(5),range(6),range(4),range(2),range(2))))
def test_coupling_bitwise(authority,columns,pbl,cu,micro,lw,sw):
    from hexcore.mpasphys.rrfs.driver_rrfs import couple_physics_tendencies
    lib,_,cache,cp=authority
    m,me,theta,sc,rates,be,ce,diag=columns
    nl,nc=m.shape;ne=me.shape[1];solve,esolve=nc-7,ne-11
    nt=f32(np.zeros_like(m));nsc=f32(np.zeros_like(sc));nu=f32(np.zeros_like(me));nd=diag.copy(order='F')
    rk,split=(1,1)
    lib.couple(nl,nc,ne,solve,esolve,len(NAMES),pbl,cu,micro,lw,sw,rk,split,
        *[a.ctypes.data for a in (m,me,theta,sc,rates,be,ce,nd,nt,nsc,nu)])
    def device(a):return cp.asarray(np.ascontiguousarray(a))
    # Native's persistent diagnostic writes 1:nEdges, although its descriptor
    # contains a dummy edge. A distinct tail catches accidental shared-kernel
    # writes beyond that source range for every active selector combination.
    padded_diag=np.concatenate((diag,np.full((nl,1),0.125,dtype='f4')),axis=1)
    dd=device(padded_diag)
    def padded_edge(a):
        return device(np.concatenate((a,np.zeros((nl,1),dtype='f4')),axis=1))
    options=dict(config_pbl_scheme=PBL[pbl],config_convection_scheme=CONVECTION[cu],
        config_microp_scheme=MICRO[micro],config_tempo_aerosolaware=micro==2,
        config_radt_lw_scheme='rrtmg_lw' if lw else 'off',
        config_radt_sw_scheme='rrtmg_sw' if sw else 'off')
    out=couple_physics_tendencies(options=options,mass=device(m),mass_edge=padded_edge(me),
        theta_m=device(theta),scalars=device(sc),scalar_names=NAMES,
        tendencies={n:device(rates[:,:,i]) for i,n in enumerate(RATES)},
        pbl_edge=padded_edge(be),convection_edge=padded_edge(ce),tend_u_phys=dd,
        n_cells_solve=solve,n_edges_solve=esolve,n_edges=ne,rk_step=rk,dynamics_substep=split,kernel_cache=cache)
    for name,expected in [('tend_rtheta_physics',nt),('scalars_tend',nsc),('tend_ru_physics',nu)]:
        actual=cp.asnumpy(out[name])
        if name=='tend_ru_physics':
            assert np.array_equal(actual[:,ne:].view(np.uint32),np.zeros((nl,1),dtype='u4'))
            actual=actual[:,:ne]
        actual=np.ascontiguousarray(actual).view(np.uint32)
        assert np.array_equal(actual,np.ascontiguousarray(expected).view(np.uint32)),name
    expected_diag=np.concatenate((nd,padded_diag[:,ne:]),axis=1)
    assert np.array_equal(cp.asnumpy(dd).view(np.uint32),np.ascontiguousarray(expected_diag).view(np.uint32))


@pytest.mark.parametrize('micro,pbl,flags',list(product(range(4),range(5),product([False,True],repeat=4))))
def test_number_carriers_bitwise(authority,micro,pbl,flags):
    from hexcore.mpasphys.rrfs.driver_rrfs import prepare_number_carriers
    _,lib,cache,cp=authority
    nl,nc,solve=8,2562,2555;names=('nc','ni','nifa','nwfa')
    rng=np.random.default_rng(510)
    # Earlier shared preparation applies max(+0,q); isolate the fork's selection.
    data=f32(10.**rng.uniform(-6,10,(4,nl,nc)))
    data[:,0,0]=np.float32(1e-45);data[:,0,1]=np.float32(0)
    native=f32(np.zeros((nc,nl,4)));f=np.asarray(flags,dtype=np.int32)
    lib.numbers(nl,nc,solve,micro,pbl,f.ctypes.data,data.ctypes.data,native.ctypes.data)
    fields={name:cp.asarray(np.ascontiguousarray(data[i])) for i,name in enumerate(names)}
    fields['qv']=cp.zeros((nl,nc),dtype=cp.float32)
    options=dict(config_microp_scheme=MICRO[micro],config_pbl_scheme=PBL[pbl],
                 config_tempo_aerosolaware=micro==2)
    result=prepare_number_carriers(SimpleNamespace(species_p=fields),options=options,
        scalar_flags=dict(zip(names,flags)),n_cells_solve=solve,kernel_cache=cache)
    allocated=names if pbl in (2,3) else ('nifa','nwfa') if micro in (1,2) else ()
    assert tuple(result)==allocated
    for name in allocated:
        expected=np.ascontiguousarray(native[:,:,names.index(name)].T).view(np.uint32)
        assert np.array_equal(cp.asnumpy(result[name]).view(np.uint32),expected),name


def test_direct_nvrtc_preserves_minimum_subnormal(authority):
    _,_,cache,cp=authority
    source='''extern "C" __global__ void subnormal(const float *x,float *y) {
        y[0]=__fmul_rn(x[0],1.0f);
    }'''
    x=cp.asarray(np.array([1],dtype=np.uint32).view(np.float32));y=cp.zeros_like(x)
    cache.raw_kernel('subnormal',source,module_key='hexcore.mpasphys.rrfs.subnormal')((1,),(1,),(x,y))
    assert int(cp.asnumpy(y).view(np.uint32)[0])==1
