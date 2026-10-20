"""Compare direct pinned-Fortran regional stages word for word.

Build with regional_native_v842_build.py through the build queue on the run host.
Set MPAS_REGIONAL_ORACLE to its fixture directory. GPU tests additionally
require MPAS_REGIONAL_GPU=1 and must run through the GPU queue. These are component
replays, not whole-forecast proof.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pytest

from hexcore import regional_v841 as regional

NC, NE, NV, NL, NT = 11, 11, 11, 4, 3  # allocated garbage elements included
INPUT = [
    ('dt', ()), ('dt_rk', ()), ('divdamp', ()),
    *[(n, (NC,)) for n in ('bc','be','bv','counts')],
    *[(n, (NC,2)) for n in ('coc','coe','eoc','voe','eov')],
    *[(n, (NC,)) for n in ('x','y','z','scale_c','scale_e','dc','dv','idc','idv','iac','iat')],
    ('esc',(NC,2)), ('esv',(NC,2)),
    *[(n, (NL,NC)) for n in ('rho','theta','ru','rd','td','ud','tr','tt','tu')],
    ('tw',(NL+1,NC)),
    *[(n, (NL,NC)) for n in ('diabatic','rp','base')],
    ('q',(NT,NL,NC)), ('qd',(NT,NL,NC)),
]
INTS = {'bc','be','bv','counts','coc','coe','eoc','voe','eov','nearest'}


def read_arrays(path, fields):
    data = path.read_bytes()
    result, offset = {}, 0
    for name, shape in fields:
        dtype = np.int32 if name in INTS else np.float32
        count = int(np.prod(shape)) if shape else 1
        value = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
        # Mesh arrays are Fortran (slot,owner), state is (level,owner),
        # scalars are (tracer,level,owner); convert to device layout.
        if len(shape) == 2 and shape[-1] == 2:
            value = value.reshape(shape)
        elif shape:
            value = value.reshape(shape, order='F')
        else:
            value = value[0]
        result[name] = value
        offset += count*4
    assert offset == len(data), (path, offset, len(data))
    return result


def oracle_root():
    raw = os.environ.get('MPAS_REGIONAL_ORACLE')
    if not raw:
        pytest.skip('MPAS_REGIONAL_ORACLE is unset; direct Fortran replay needs its artifact')
    root = Path(raw)
    assert (root/'source-pins.json').exists()
    return root


def record(stage, name, actual, expected):
    actual = np.ascontiguousarray(actual).view(np.uint32)
    expected = np.ascontiguousarray(expected).view(np.uint32)
    assert actual.shape == expected.shape
    count = int(np.count_nonzero(actual != expected))
    receipt_dir = os.environ.get('MPAS_REGIONAL_RECEIPTS')
    if receipt_dir:
        directory = Path(receipt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/f'{stage}-{name}.json').write_text(json.dumps({
            'stage':stage, 'payload':name, 'words':actual.size,
            'different_words':count, 'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
            'native_sha256':hashlib.sha256(expected.tobytes()).hexdigest(),
            'native_process_count':1,
            'tier':'direct-source component replay; not whole forecast',
        },indent=2)+'\n')
    assert count == 0, f'{stage}/{name}: {count}/{actual.size} different words'


def record_cuda_compile(kernels, stage):
    """Save the shared cache's compiler-bound manifest inside the GPU queue."""
    raw=os.environ.get('MPAS_REGIONAL_RECEIPTS')
    if raw:
        target=Path(raw).parent/f'cuda-compile-{stage}.json'
        target.write_text(json.dumps(kernels.cache.compile_manifest(),indent=2)+'\n')


def mask_fixture(a):
    return regional.RegionalMasks(
        a['bc'],a['be'],a['bv'],(a['bc']>5).astype('f4'),
        (a['be']>5).astype('f4'),(a['bv']>5).astype('f4'),np.full(NC,NC-1),
        np.flatnonzero(a['bc']>5),np.flatnonzero(a['be']>5),
        np.flatnonzero((a['bc']>1)&(a['bc']<=5)),
        np.flatnonzero((a['be']>1)&(a['be']<=5)),np.flatnonzero(a['bc']>1))


@pytest.mark.parametrize('case',range(1,17))
def test_native_masks(case):
    root=oracle_root()/f'case-{case:02}'
    a=read_arrays(root/'input.bin',INPUT)
    expected=read_arrays(root/'spec.bin',[
        ('tr',(NL,NC)),('tt',(NL,NC)),('tu',(NL,NE)),('tw',(NL+1,NC)),
        ('diabatic',(NL,NC)),('smc',(NC,)),('sme',(NE,)),('smv',(NV,)),('nearest',(NC,))])
    mesh=SimpleNamespace(bdyMaskCell=a['bc'][:-1],bdyMaskEdge=a['be'][:-1],
        bdyMaskVertex=a['bv'][:-1],nEdgesOnCell=a['counts'][:-1],
        cellsOnCell=a['coc'][:-1]-1,xCell=a['x'][:-1],yCell=a['y'][:-1],zCell=a['z'][:-1])
    masks=regional.derive_regional_masks(mesh,np.dtype('f4'))
    record(f'case-{case:02}-masks','nearest',masks.nearest_relaxation_cell.astype('i4'),expected['nearest'][:-1]-1)
    for field, actual in [('smc',masks.spec_zone_mask_cell),('sme',masks.spec_zone_mask_edge),('smv',masks.spec_zone_mask_vertex)]:
        record(f'case-{case:02}-masks',field,actual,expected[field][:-1])


@pytest.mark.parametrize('case',range(1,17))
def test_native_cuda_stages(case):
    if os.environ.get('MPAS_REGIONAL_GPU') != '1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels
    root=oracle_root()/f'case-{case:02}'
    a=read_arrays(root/'input.bin',INPUT)
    masks=mask_fixture(a)
    dev={n:cp.asarray(np.ascontiguousarray(v)) for n,v in a.items() if np.ndim(v)}
    # Native connectivity is 1-based; preserve the explicit garbage slot.
    for n in ('coc','coe','eoc','voe','eov'):
        dev[n]-=np.int32(1)
    for n, v in [('sc',masks.spec_cells),('se',masks.spec_edges),('rc',masks.relax_cells),('re',masks.relax_edges),('nudged',masks.nudged_cells)]:
        dev[n]=cp.asarray(v,dtype='i4')
    kernels=CudaRegionalKernels(KernelCache())
    I=np.int32
    F=np.float32
    d=dev
    def launch(name, *args):
        kernels.launch(name,NT*NL*NC,args)
    def check(stage,fields):
        expected=read_arrays(root/f'{stage}.bin',fields)
        for name in expected:
            record(f'case-{case:02}-{stage}',name,cp.asnumpy(d[name]),expected[name])
    launch('regional_speczone_tend_cell_v841',I(NL),I(NC),I(len(masks.spec_cells)),d['sc'],d['rd'],d['td'],d['tr'],d['tt'],d['tw'])
    launch('regional_speczone_tend_edge_v841',I(NL),I(NE),I(len(masks.spec_edges)),d['se'],d['ud'],d['tu'])
    # The native driver also clears diabatic heating. The regional runtime
    # uses this existing zero kernel for a supplied diabatic tendency pool.
    launch('regional_zero_speczone_w_v841',I(NL),I(NC),I(len(masks.spec_cells)),d['sc'],d['diabatic'])
    expected=read_arrays(root/'spec.bin',[
        ('tr',(NL,NC)),('tt',(NL,NC)),('tu',(NL,NE)),('tw',(NL+1,NC)),
        ('diabatic',(NL,NC)),('smc',(NC,)),('sme',(NE,)),('smv',(NV,)),('nearest',(NC,))])
    for name in ('tr','tt','tu','tw','diabatic'):
        record(f'case-{case:02}-spec',name,cp.asnumpy(d[name]),expected[name])
    fifty_dt=F(F(50)*a['dt']); ten_dt=F(F(10)*a['dt']); rz=F(5)
    launch('regional_relaxzone_rayleigh_cell_v841',I(NL),I(NC),I(len(masks.relax_cells)),fifty_dt,rz,d['rc'],d['bc'],d['scale_c'],d['rho'],d['theta'],d['rd'],d['td'],d['tr'],d['tt'])
    launch('regional_relaxzone_rayleigh_edge_v841',I(NL),I(NE),I(len(masks.relax_edges)),fifty_dt,rz,d['re'],d['be'],d['scale_e'],d['ru'],d['ud'],d['tu'])
    launch('regional_relaxzone_filter_cell_v841',I(NL),I(NC),I(NE),I(2),I(len(masks.relax_cells)),ten_dt,rz,d['rc'],d['bc'],d['scale_c'],d['counts'],d['eoc'],d['coe'],d['esc'],d['dv'],d['idc'],d['rho'],d['theta'],d['rd'],d['td'],d['tr'],d['tt'])
    launch('regional_relaxzone_filter_edge_v841',I(NL),I(NC),I(NE),I(NV),I(2),I(2),I(len(masks.relax_edges)),ten_dt,a['divdamp'],rz,d['re'],d['be'],d['scale_e'],d['counts'],d['eoc'],d['coe'],d['voe'],d['eov'],d['esc'],d['esv'],d['dc'],d['dv'],d['idc'],d['idv'],d['iac'],d['iat'],d['ru'],d['ud'],d['tu'])
    check('relax',[('tr',(NL,NC)),('tt',(NL,NC)),('tu',(NL,NE))])
    launch('regional_zero_speczone_w_v841',I(NL+1),I(NC),I(len(masks.spec_cells)),d['sc'],d['tw'])
    d['rho_theta']=cp.zeros((NL,NC),dtype='f4')
    launch('regional_reset_speczone_values_v841',I(NL),I(NC),I(len(masks.spec_cells)),d['sc'],d['td'],d['rd'],d['base'],d['theta'],d['rho_theta'],d['rp'])
    check('reset',[('tw',(NL+1,NC)),('theta',(NL,NC)),('rp',(NL,NC))])
    d['updates']=cp.zeros((NT,NL,NC),dtype='f4')
    launch('regional_bdy_adjust_scalars_compute_v841',I(NT),I(NL),I(NC),I(NE),I(2),I(len(masks.relax_cells)),I(len(masks.spec_cells)),ten_dt,a['dt_rk'],rz,d['rc'],d['sc'],d['bc'],d['scale_c'],d['counts'],d['eoc'],d['coe'],d['esc'],d['dv'],d['idc'],d['q'],d['qd'],d['updates'])
    launch('regional_bdy_adjust_scalars_copyback_v841',I(NT),I(NL),I(NC),I(len(masks.nudged_cells)),d['nudged'],d['updates'],d['q'])
    check('scalar-adjust',[('q',(NT,NL,NC))])
    launch('regional_bdy_set_scalars_v841',I(NT),I(NL),I(NC),I(len(masks.spec_cells)),d['sc'],d['qd'],d['q'])
    check('scalar-set',[('q',(NT,NL,NC))])
    cp.cuda.get_current_stream().synchronize()
    if case==1:
        record_cuda_compile(kernels,"boundary")


@pytest.mark.parametrize('case',range(1,17))
def test_native_mesh_scaling(case):
    root=oracle_root()/f'case-{case:02}'
    a=read_arrays(root/'input.bin',INPUT)
    lbc=read_arrays(root/'lbc-input.bin',LBC_INPUT)
    expected=read_arrays(root/'mesh-scaling.bin',[('scaling_c',(NC,)),('scaling_e',(NE,))])
    mesh=SimpleNamespace(meshDensity=lbc['density'][:-1],cellsOnEdge=a['coe'][:-1]-1,dcEdge=a['dc'][:-1])
    # The last physical edge points to the zero-initialized garbage density.
    mesh.cellsOnEdge[-1,1]=-1
    sc,se=regional.compute_mesh_scaling_regional(mesh,np.dtype('f4'),config_h_scale_with_mesh=case%2==0)
    record(f'case-{case:02}-scaling','cell',sc,expected['scaling_c'][:-1])
    record(f'case-{case:02}-scaling','edge',se,expected['scaling_e'][:-1])


LBC_INPUT=[('lbc_coe',(NE,2)),('zz',(NL,NC)),('lbc_re',(NL,NE)),
           ('old_u',(NL,NE)),('old_ru',(NL,NE)),('old_re',(NL,NE)),('old_w',(NL+1,NC)),
           *[(n,(NL,NC)) for n in ('old_theta','old_rt','old_rho','old_r')],
           ('old_q',(NT,NL,NC)),('interval',()),('remaining',()),('density',(NC,))]
INTS.add('lbc_coe')


@pytest.mark.parametrize('case',range(1,17))
def test_native_cuda_driving_pool(case):
    if os.environ.get('MPAS_REGIONAL_GPU') != '1':
        pytest.skip('MPAS_REGIONAL_GPU=1 requires the GPU queue')
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_v841 import CudaRegionalKernels
    root=oracle_root()/f'case-{case:02}'
    a=read_arrays(root/'input.bin',INPUT)
    lbc=read_arrays(root/'lbc-input.bin',LBC_INPUT)
    kernels=CudaRegionalKernels(KernelCache())
    I=np.int32
    # DeviceRegionalDrivingState owns physical columns; its later padded
    # dycore adapter supplies garbage values. Native rho_edge averaging
    # stops at nEdges, so including its allocated garbage edge in this
    # unpadded kernel would exercise a different interface than the driver.
    nc,ne=NC-1,NE-1
    fields={n:a[n] for n in ('rho','theta','ru','tw','q')}
    fields.update({n:v for n,v in lbc.items() if np.ndim(v)})
    d={n:cp.asarray(np.ascontiguousarray(v[:-1] if n=='lbc_coe' else v[...,:-1]))
       for n,v in fields.items()}
    d['lbc_coe']-=I(1)
    d['qv']=cp.ascontiguousarray(d['q'][0])
    for n in ('lbc_rho','lbc_rt','lbc_ru'):
        d[n]=cp.empty((NL,nc),dtype='f4')
    def launch(name,*args):
        kernels.launch(name,NT*NL*NC,args)
    launch('regional_lbc_derive_v841',I(NL*nc),regional.RVORD_F32,d['zz'],d['rho'],d['theta'],d['qv'],d['lbc_rho'],d['lbc_rt'])
    launch('regional_lbc_rho_edge_v841',I(NL),I(nc),I(ne),d['lbc_coe'],d['lbc_rho'],d['ru'],d['lbc_re'],d['lbc_ru'])
    expected=read_arrays(root/'lbc-derived.bin',[(n,(NL,NC)) for n in ('zz','lbc_rho','lbc_rt','lbc_re','lbc_ru')])
    for n in expected:
        record(f'case-{case:02}-lbc-derived',n,cp.asnumpy(d[n]),expected[n][...,:-1])
    pairs={'old_u':'ru','old_ru':'lbc_ru','old_re':'lbc_re','old_w':'tw','old_theta':'theta',
           'old_rt':'lbc_rt','old_rho':'lbc_rho','old_r':'rho','old_q':'q'}
    expected=read_arrays(root/'lbc-tendency.bin',[(n,a[pairs[n]].shape if pairs[n] in a else (NL,NC)) for n in pairs])
    inv_dt=np.float32(np.float32(1)/lbc['interval'])
    for old,new in pairs.items():
        # Read and write may alias exactly as native's shifted time-level pool.
        launch('regional_lbc_tendency_v841',I(d[old].size),inv_dt,d[new],d[old],d[old])
        record(f'case-{case:02}-lbc-tendency',old,cp.asnumpy(d[old]),expected[old][...,:-1])
    d['at_q']=cp.empty_like(d['q'])
    launch('regional_lbc_state_at_v841',I(d['q'].size),lbc['remaining'],d['q'],d['old_q'],d['at_q'])
    expected=read_arrays(root/'lbc-state.bin',[('at_q',(NT,NL,NC))])
    record(f'case-{case:02}-lbc-state','scalars',cp.asnumpy(d['at_q']),expected['at_q'][...,:-1])
    if case==1:
        record_cuda_compile(kernels,"driving-pool")


@pytest.mark.parametrize('case',range(1,17))
def test_native_cpu_boundary_stages(case):
    root=oracle_root()/f'case-{case:02}'
    a=read_arrays(root/'input.bin',INPUT)
    masks=mask_fixture(a)
    mesh=SimpleNamespace(nEdgesOnCell=a['counts'],edgesOnCell=a['eoc']-1,
        cellsOnEdge=a['coe']-1,verticesOnEdge=a['voe']-1,edgesOnVertex=a['eov']-1,
        dcEdge=a['dc'],dvEdge=a['dv'],areaCell=a['dc']*a['dv'],
        areaTriangle=(a['dc']*a['dv'])/np.float32(3))
    regional.adjust_dynamics_speczone_tend(masks=masks,tend_ru=a['tu'],tend_rho=a['tr'],
        tend_rt=a['tt'],tend_rw=a['tw'],ru_driving_tend=a['ud'],rt_driving_tend=a['td'],
        rho_driving_tend=a['rd'],rt_diabatic_tend=a['diabatic'])
    expected=read_arrays(root/'spec.bin',[
        ('tr',(NL,NC)),('tt',(NL,NC)),('tu',(NL,NE)),('tw',(NL+1,NC)),
        ('diabatic',(NL,NC)),('smc',(NC,)),('sme',(NE,)),('smv',(NV,)),('nearest',(NC,))])
    for name in ('tr','tt','tu','tw','diabatic'):
        record(f'case-{case:02}-cpu-spec',name,a[name],expected[name])
    regional.adjust_dynamics_relaxzone_tend(mesh,masks=masks,
        mesh_scaling_regional_cell=a['scale_c'],mesh_scaling_regional_edge=a['scale_e'],
        tend_ru=a['tu'],tend_rho=a['tr'],tend_rt=a['tt'],ru=a['ru'],theta_m=a['theta'],rho_zz=a['rho'],
        ru_driving_values=a['ud'],rt_driving_values=a['td'],rho_driving_values=a['rd'],
        dt=a['dt'],config_relax_zone_divdamp_coef=a['divdamp'])
    expected=read_arrays(root/'relax.bin',[(n,(NL,NC)) for n in ('tr','tt','tu')])
    for name in expected:
        record(f'case-{case:02}-cpu-relax',name,a[name],expected[name])
    regional.bdy_adjust_scalars(mesh,masks=masks,mesh_scaling_regional_cell=a['scale_c'],
        scalars_new=a['q'],scalars_driving=a['qd'],dt=a['dt'],dt_rk=a['dt_rk'])
    expected=read_arrays(root/'scalar-adjust.bin',[('q',(NT,NL,NC))])
    record(f'case-{case:02}-cpu-scalar-adjust','q',a['q'],expected['q'])
    regional.bdy_set_scalars(masks=masks,scalars_new=a['q'],scalars_driving=a['qd'])
    expected=read_arrays(root/'scalar-set.bin',[('q',(NT,NL,NC))])
    record(f'case-{case:02}-cpu-scalar-set','q',a['q'],expected['q'])
    regional.zero_speczone_w(masks=masks,w=a['tw'])
    rtheta=a['rp']+a['base']
    regional.reset_speczone_values(masks=masks,theta_m=a['theta'],rho_theta=rtheta,
        rt_driving_values=a['td'],rho_driving_values=a['rd'])
    expected=read_arrays(root/'reset.bin',[('tw',(NL+1,NC)),('theta',(NL,NC)),('rp',(NL,NC))])
    record(f'case-{case:02}-cpu-reset','tw',a['tw'],expected['tw'])
    record(f'case-{case:02}-cpu-reset','theta',a['theta'],expected['theta'])
    # CPU state carries total rtheta rather than native's perturbation pool.
    record(f'case-{case:02}-cpu-reset','rtheta_spec',rtheta[:,masks.spec_cells]-a['base'][:,masks.spec_cells],expected['rp'][:,masks.spec_cells])
