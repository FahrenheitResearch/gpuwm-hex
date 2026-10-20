"""Replay the entire strict-native boundary module's public interfaces.

The Fortran authority is the unchanged pinned module, including its admission
branches. File/stream association is synthetic; no real forecast is claimed.
"""
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest

from hexcore.lbc import LbcFile
from hexcore import regional_v841 as regional
from test_regional_native_v842 import read_arrays, record, record_cuda_compile

NC,NE,NL,NS=5,5,3,2
FIELDS=[(n,(NL+(n=='w'),NC)) for n in ('u','ru','rho_edge','w','theta','rtheta_m','rho_zz','rho')]
FIELDS.append(('scalars',(NS,NL,NC)))
INPUT=[('coe',(NE,2)),('zz',(NL,NC)),('prior_edge',(NL,NE)),('u',(NL,NE)),
       ('w',(NL+1,NC)),('theta',(NL,NC)),('rho',(NL,NC)),('scalars',(NS,NL,NC))]
MESH=[(n,(NC,)) for n in ('bc','be','bv','counts')]+[('coc',(NC,2))]+[(n,(NC,)) for n in ('x','y','z')]
START=datetime(2000,1,1)
END=START+timedelta(seconds=10800.25)
STEP=START+timedelta(seconds=7200.125)


def root(case):
    value=os.environ.get('MPAS_REGIONAL_MODULE_ORACLE')
    if not value:
        pytest.skip('MPAS_REGIONAL_MODULE_ORACLE is unset')
    return Path(value)/f'whole-{case:02}'


def file_fixture(a,when):
    # Feed exact native primary slabs, in the public LbcFile layout.
    fields={f'lbc_{n}':np.ascontiguousarray(a[n][:,:-1].T)
            for n in ('u','w','theta','rho')}
    fields.update({f'lbc_{n}':np.ascontiguousarray(a['scalars'][j,:,:-1].T)
                   for j,n in enumerate(('qv','qc'))})
    return LbcFile(Path('synthetic-native-input'),when,fields)


def admitted_inventory(path, monkeypatch):
    # Framework stream fixture: only file admission is prescribed, as in the
    # native framework stub. Production pool/driver arithmetic is exercised.
    from hexcore import lbc
    first=file_fixture(read_arrays(path/'first-input.bin',INPUT),START)
    following=file_fixture(read_arrays(path/'next-input.bin',INPUT),END)
    monkeypatch.setattr(lbc,'read_lbc_file',lambda admitted: admitted)
    return SimpleNamespace(latest_before=lambda when: first,
                           earliest_strictly_after=lambda when: following)


def masks_fixture(path):
    a=read_arrays(path/'mesh-input.bin',MESH)
    mesh=SimpleNamespace(bdyMaskCell=a['bc'][:-1],bdyMaskEdge=a['be'][:-1],bdyMaskVertex=a['bv'][:-1],
                         nEdgesOnCell=a['counts'][:-1],cellsOnCell=a['coc'][:-1]-1,
                         xCell=a['x'][:-1],yCell=a['y'][:-1],zCell=a['z'][:-1])
    return regional.derive_regional_masks(mesh,np.dtype('f4'))


@pytest.mark.parametrize('case',range(1,7))
def test_native_whole_cpu(case,monkeypatch):
    path=root(case)
    first=read_arrays(path/'first-input.bin',INPUT)
    masks=masks_fixture(path)
    mesh=SimpleNamespace(cellsOnEdge=first['coe'][:-1]-1,dcEdge=np.ones(NE-1))
    driving=regional.RegionalDrivingState(admitted_inventory(path,monkeypatch),mesh,
                                         first['zz'][:,:-1],scalar_names=('lbc_qv','lbc_qc'))
    driving._rho_edge_slot[:]=first['prior_edge'][:,:-1]
    for phase,when in [('first',START),('next',END)]:
        a=read_arrays(path/f'{phase}-input.bin',INPUT)
        if phase=='first':
            driving.start(when)
        else:
            driving.advance(START)
        state,scalars=driving._state,driving._state_scalars
        expected=read_arrays(path/f'{phase}-derived.bin',FIELDS)
        for n in state:
            record(f'whole-{case:02}-cpu-{phase}',n,state[n],expected[n][:,:-1])
        record(f'whole-{case:02}-cpu-{phase}','scalars',scalars,expected['scalars'][:,:,:-1])
    expected=read_arrays(path/'tendencies.bin',FIELDS)
    for n,actual in {**driving._tend,'scalars':driving._tend_scalars}.items():
        record(f'whole-{case:02}-cpu-formed-tend',n,actual,expected[n][...,:-1])
    # Raw file pool uses the same clock conversion. Compare its rho tendency
    # against the native lbc_rho slot, before any coupled-field conversion.
    record(f'whole-{case:02}-cpu-file-tend','rho',driving._pool.tendency('lbc_rho').T,
           expected['rho'][:,:-1])
    expected=read_arrays(path/'interpolated.bin',[('rho',(NL,NC)),('qc',(NL,NC)),('scalars',(NS,NL,NC))])
    # Exercise native fractional clock seconds as well as the RK offset.
    record(f'whole-{case:02}-cpu-state','rho',driving.state_at('rho_zz',STEP,np.float32(.375)),expected['rho'][:,:-1])
    record(f'whole-{case:02}-cpu-state','scalars',driving.state_at('scalars',STEP,np.float32(.375)),expected['scalars'][:,:,:-1])
    record(f'whole-{case:02}-cpu-state','qc',driving.state_at('qc',STEP,np.float32(.375)),expected['qc'][:,:-1])
    expected=read_arrays(path/'retrieved.bin',[('rho',(NL,NC)),('qc',(NL,NC))])
    record(f'whole-{case:02}-cpu-tend','rho',driving.tendency('rho_zz'),expected['rho'][:,:-1])
    record(f'whole-{case:02}-cpu-tend','qc',driving.tendency('qc'),expected['qc'][:,:-1])
    expected=read_arrays(path/'masks.bin',[(n,(NC,)) for n in ('smc','sme','smv','nearest')])
    for n,actual in [('smc',masks.spec_zone_mask_cell),('sme',masks.spec_zone_mask_edge),('smv',masks.spec_zone_mask_vertex),
                     ('nearest',masks.nearest_relaxation_cell.astype('i4')+1)]:
        record(f'whole-{case:02}-cpu-masks',n,actual,expected[n][:-1])


@pytest.mark.parametrize('case',range(1,7))
def test_native_whole_admission(case):
    path=root(case);masks=masks_fixture(path)
    status=np.fromfile(path/'status.bin',dtype='i4')
    # Both native stream-read error paths and the prescribed read directions.
    assert status[:6].tolist()==[1,0,1,0,2,1]
    empty=masks_fixture(path)
    empty.bdy_mask_cell[:]=0
    combinations=[(masks,True,True),(masks,True,False),(masks,True,False),
                  (masks,False,True),(empty,False,True),(empty,True,True)]
    got=[]
    for mask,apply,valid in combinations:
        try:
            regional.regional_bdy_checks(mask,config_apply_lbcs=apply,lbc_input_interval_valid=valid)
            got.append(0)
        except regional.RegionalAdmissionError:
            got.append(1)
    record(f'whole-{case:02}-admission','status',np.asarray(got,dtype='i4'),status[6:])


@pytest.mark.parametrize('case',range(1,7))
def test_native_whole_cuda(case,monkeypatch):
    if os.environ.get('MPAS_REGIONAL_GPU')!='1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import (
        CudaRegionalKernels,DeviceRegionalMasks,DeviceRegionalDrivingState)
    path=root(case)
    first=read_arrays(path/'first-input.bin',INPUT)
    kernels=CudaRegionalKernels(KernelCache())
    driving=DeviceRegionalDrivingState(
        admitted_inventory(path,monkeypatch),DeviceRegionalMasks.from_host(masks_fixture(path)),
        cells_on_edge=cp.asarray(first['coe'][:-1]-1,dtype='i4'),
        zz=cp.asarray(np.ascontiguousarray(first['zz'][:,:-1])),n_vert_levels=NL,
        kernels=kernels,scalar_names=('lbc_qv','lbc_qc'))
    driving._rho_edge_zero[:]=cp.asarray(first['prior_edge'][:,:-1])
    for phase,when in [('first',START),('next',END)]:
        a=read_arrays(path/f'{phase}-input.bin',INPUT)
        if phase=='first':
            driving.start(when)
        else:
            driving.advance(START)
        state,scalars=driving._state,driving._state_scalars
        expected=read_arrays(path/f'{phase}-derived.bin',FIELDS)
        for n in state:
            record(f'whole-{case:02}-cuda-{phase}',n,cp.asnumpy(state[n]),expected[n][:,:-1])
        record(f'whole-{case:02}-cuda-{phase}','scalars',cp.asnumpy(scalars),expected['scalars'][:,:,:-1])
    tend={**driving._tend,'scalars':driving._tend_scalars}
    expected=read_arrays(path/'tendencies.bin',FIELDS)
    for n in tend:
        record(f'whole-{case:02}-cuda-tend',n,cp.asnumpy(tend[n]),expected[n][...,:-1])
    expected=read_arrays(path/'retrieved.bin',[('rho',(NL,NC)),('qc',(NL,NC))])
    for actual,n in [('rho_zz','rho'),('qc','qc')]:
        record(f'whole-{case:02}-cuda-get-tend',n,cp.asnumpy(driving.tendency(actual)),expected[n][:,:-1])
    expected=read_arrays(path/'interpolated.bin',[('rho',(NL,NC)),('qc',(NL,NC)),('scalars',(NS,NL,NC))])
    for actual,n in [('rho_zz','rho'),('qc','qc'),('scalars','scalars')]:
        out=cp.empty_like(tend[actual] if actual!='qc' else tend['scalars'][1])
        result=driving.state_at(actual,STEP,np.float32(.375),out=out)
        assert result is out
        record(f'whole-{case:02}-cuda-get-state',n,cp.asnumpy(result),expected[n][...,:-1])
    cp.cuda.Stream.null.synchronize()
    if case==1:
        record_cuda_compile(kernels,"whole-module")
