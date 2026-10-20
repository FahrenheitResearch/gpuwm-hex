"""Whole boundary pool replay on f1-captured real native fields (one process).

Native f1 global snapshots retain their zero boundary masks. The separately
registered 154-case regional corpus covers real active masks and LBC changes.
"""
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from hexcore import regional_v841 as regional
from hexcore.lbc import LbcFile
from test_regional_native_v842 import read_arrays, record, record_cuda_compile
from test_regional_native_v842_real import START, compare_pool


@pytest.mark.parametrize('case', range(2))
@pytest.mark.parametrize('backend', ['cpu', 'cuda', 'padded-cuda'])
def test_native_tap_boundary_pool(case, backend, monkeypatch):
    raw = os.environ.get('MPAS_REGIONAL_TAP_ORACLE')
    if not raw:
        pytest.skip('selected f1 native tap oracle is required')
    if backend != 'cpu' and os.environ.get('MPAS_REGIONAL_GPU') != '1':
        pytest.skip('GPU replay must run through the GPU queue')
    path = Path(raw)/f'real-{case:02}'
    nc,ne,nv,nl,me,ns=np.fromfile(path/'dimensions.bin',dtype='i4').tolist()
    mesh_arrays=read_arrays(path/'mesh.bin',[
        ('bc',(nc+1,)),('be',(ne+1,)),('bv',(nv+1,)),('counts',(nc+1,)),
        ('coc',(me,nc+1)),('coe',(ne+1,2)),('x',(nc+1,)),('y',(nc+1,)),
        ('z',(nc+1,)),('zz',(nl,nc+1))])
    mesh=SimpleNamespace(bdyMaskCell=mesh_arrays['bc'][:-1],bdyMaskEdge=mesh_arrays['be'][:-1],
        bdyMaskVertex=mesh_arrays['bv'][:-1],nEdgesOnCell=mesh_arrays['counts'][:-1],
        cellsOnCell=mesh_arrays['coc'][:,:-1].T-1,cellsOnEdge=mesh_arrays['coe'][:-1]-1,
        xCell=mesh_arrays['x'][:-1],yCell=mesh_arrays['y'][:-1],zCell=mesh_arrays['z'][:-1],dcEdge=np.empty(ne,dtype='f4'))
    masks=regional.derive_regional_masks(mesh,np.dtype('f4'))
    expected=read_arrays(path/'masks.bin',[('smc',(nc+1,)),('sme',(ne+1,)),('smv',(nv+1,)),('nearest',(nc+1,))])
    for name,actual in [('smc',masks.spec_zone_mask_cell),('sme',masks.spec_zone_mask_edge),
                        ('smv',masks.spec_zone_mask_vertex),('nearest',masks.nearest_relaxation_cell.astype('i4')+1)]:
        record(f'tap-{case}-{backend}-masks',name,actual,expected[name][:-1])
    primary=[('u',(nl,ne+1)),('w',(nl+1,nc+1)),('theta',(nl,nc+1)),
             ('rho',(nl,nc+1)),('scalars',(ns,nl,nc+1))]
    names=('lbc_qv','lbc_qc','lbc_qr','lbc_qi','lbc_qs','lbc_qg')
    assert ns==len(names)
    def admitted(filename):
        value=read_arrays(path/filename,primary)
        fields={f'lbc_{n}':np.ascontiguousarray(value[n][...,:-1].T) for n in ('u','w','theta','rho')}
        fields.update({n:np.ascontiguousarray(value['scalars'][j,:,:-1].T) for j,n in enumerate(names)})
        return fields
    fields,next_fields=admitted('primary.bin'),admitted('next-primary.bin')
    end=START+timedelta(seconds=10800)
    first=LbcFile(path,START,fields);following=LbcFile(path,end,next_fields)
    from hexcore import lbc
    monkeypatch.setattr(lbc,'read_lbc_file',lambda value:value)
    inventory=SimpleNamespace(latest_before=lambda _:first,earliest_strictly_after=lambda _:following)
    pool_fields=[(n,((nl+1 if n=='w' else nl),(ne+1 if n in ('u','ru','rho_edge') else nc+1)))
                 for n in ('u','ru','rho_edge','w','theta','rtheta_m','rho_zz','rho')]+[('scalars',(ns,nl,nc+1))]
    a=SimpleNamespace(case=f'tap-{case}',path=path,nl=nl,nc=nc,ns=ns,third=False,end=end,
        pool_fields=pool_fields,retrieved_scalar='qc')
    if backend=='cpu':
        driving=regional.RegionalDrivingState(inventory,mesh,zz=mesh_arrays['zz'][:,:-1],scalar_names=names)
        compare_pool(a,driving,backend,np.asarray)
    else:
        import cupy as cp
        from hexcore.cuda_backend import KernelCache
        from hexcore.cuda_regional_v841 import CudaRegionalKernels,DeviceRegionalDrivingState,DeviceRegionalMasks
        from hexcore.cuda_regional_forecast_v841 import PaddedRegionalDrivingState
        kernels=CudaRegionalKernels(KernelCache())
        if backend=='cuda':
            driving=DeviceRegionalDrivingState(inventory,DeviceRegionalMasks.from_host(masks),cells_on_edge=cp.asarray(mesh.cellsOnEdge),zz=cp.asarray(np.ascontiguousarray(mesh_arrays['zz'][:,:-1])),n_vert_levels=nl,kernels=kernels,scalar_names=names)
        else:
            # The resident regional adapter constructs padded geometry from
            # physical mesh columns: unit zz and a garbage-cell edge. Native
            # boundary math never reads global mesh padding. Preserve every
            # captured physical word and compare every padded output word.
            padded_coe=np.full((ne+1,2),nc,dtype='i4')
            padded_coe[:-1]=mesh.cellsOnEdge
            padded_zz=np.ascontiguousarray(np.pad(mesh_arrays['zz'][:,:-1],((0,0),(0,1)),constant_values=1))
            driving=PaddedRegionalDrivingState(inventory,SimpleNamespace(n_cells=nc+1,n_edges=ne+1),
                cells_on_edge=cp.asarray(padded_coe),zz=cp.asarray(padded_zz),
                n_vert_levels=nl,kernels=kernels,scalar_names=names,garbage_cell=nc,garbage_edge=ne)
        compare_pool(a,driving,backend,cp.asnumpy,padded=backend=='padded-cuda')
        cp.cuda.Stream.null.synchronize();record_cuda_compile(kernels,'native-tap')
