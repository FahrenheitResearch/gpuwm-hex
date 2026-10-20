"""Wordwise whole-module replay, including native stream and interval behavior."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasdiag.convective_v842 import CUDA_SOURCE, FIELDS, MAXIMA, SOURCE_SHA256, _mask
from mpas_convective_reference import Reference


def columns(nc=19, nl=32):
    """Synthetic physical, boundary and deliberately awkward branch soundings."""
    rng = np.random.default_rng(634927)
    z = np.tile(np.linspace(0, 17000, nl+1, dtype=np.float32), (nc, 1))
    z += np.arange(nc, dtype=np.float32)[:, None]*np.float32(137.25)
    z[3] = np.linspace(0, 4500, nl+1, dtype=np.float32)+np.float32(700)
    pressure = np.tile(np.linspace(99500, 6000, nl, dtype=np.float32), (nc, 1))
    pressure[4] = np.linspace(49000, 6000, nl, dtype=np.float32)
    pressure[5] = np.linspace(50000, 20000, nl, dtype=np.float32)
    pressure[6] = np.linspace(100000, 99700, nl, dtype=np.float32)  # dp < pinc
    temp = np.empty((nc, nl), dtype=np.float32)
    for c in range(nc):
        temp[c] = np.linspace(304-c*1.1, 202+c*0.3, nl, dtype=np.float32)
    temp[1] = np.linspace(286, 270, nl, dtype=np.float32)  # stable
    temp[2] = np.linspace(310, 170, nl, dtype=np.float32)  # strong convection
    temp[7, 1:8] += np.float32(15)  # inhibition and crossing signs
    temp[8, :10] -= np.float32(8)  # elevated source parcel
    qv = (np.linspace(0.022, 0.00001, nl, dtype=np.float32)[None, :]
          *np.linspace(0.95, 0.08, nc, dtype=np.float32)[:, None]).astype(np.float32)
    qv[9] = np.float32(0)  # vapor-pressure floor
    qv[10] = np.float32(-1.e-9)  # same floor, finite source dewpoint
    # Saturation/dewpoint-equals-T getthe branch as well as unsaturated branch.
    ev = np.float32(611.2)*np.exp(np.float32(17.67)*(temp[11]-np.float32(273.15))
                                 /(temp[11]-np.float32(29.65)))
    qv[11] = np.float32(0.622)*ev/(pressure[11]-ev)
    ex = ((pressure/np.float32(100000)).astype(np.float64)
          **float(np.float32(287.0)/np.float32(1004.5))).astype(np.float32)
    theta = temp/ex*(np.float32(1)+np.float32(461.6/287.0)*qv)
    u = rng.uniform(-25, 40, (nc, nl)).astype(np.float32)
    v = rng.uniform(-30, 30, (nc, nl)).astype(np.float32)
    u[0] = 3; v[0] = 4  # zero shear clamp, exact speed=5
    u[12] = np.linspace(0, 32, nl, dtype=np.float32)
    v[12] = np.linspace(0, 16, nl, dtype=np.float32)
    w = rng.uniform(-4, 18, (nc, nl+1)).astype(np.float32)
    w[:, -1] = np.float32(10000)  # top interface excluded by w maximum
    w[13, :-1] = np.float32(-4)  # downward motion clipped for UH
    nv, degree = 2*nc+3, 3
    cov = rng.integers(1, nc+2, (nv, degree), dtype=np.int32)
    cov[0] = (1, 1, nc+1)  # repeated cell and native sentinel
    kite = rng.uniform(0.1, 4e6, (nv, degree)).astype(np.float32)
    vort = rng.uniform(-0.005, 0.005, (nv, nl)).astype(np.float32)
    return dict(zgrid=z, w=w, uReconstructZonal=u, uReconstructMeridional=v,
                theta_m=np.ascontiguousarray(theta), qv=np.ascontiguousarray(qv),
                exner=ex, pressure_base=np.ascontiguousarray(pressure-np.float32(37.125)),
                pressure_p=np.full((nc,nl),37.125,np.float32),
                areaCell=rng.uniform(1e6, 4e6,nc).astype(np.float32),
                cellsOnVertex=cov, kiteAreasOnVertex=kite, vorticity=vort)


@pytest.fixture(scope='session')
def references():
    path = os.environ.get('MPAS_CONVECTIVE_REFERENCE')
    if not path:
        pytest.skip('set MPAS_CONVECTIVE_REFERENCE to a standalone build (tests/mpas_convective_reference.py)')
    return Reference(Path(path)), Reference(Path(path), host=True)


@pytest.fixture(scope='session')
def receipt():
    rows = []
    yield rows
    path = os.environ.get('MPAS_CONVECTIVE_RECEIPT')
    if path:
        reference = Path(os.environ['MPAS_CONVECTIVE_REFERENCE'])
        provenance = json.loads((reference/'provenance.json').read_text())
        import gpuwm
        headers = Path(gpuwm.__file__).resolve().parent/'core'/'kernels'
        header_hashes = {name: hashlib.sha256((headers/name).read_bytes()).hexdigest()
                         for name in ('glibc_flt32.cuh','flt32_expf_fma.cuh')}
        result = {'shared_math_headers': header_hashes, 'tier': 'standalone CUDA whole-module' if os.environ.get('MPAS_CONVECTIVE_GPU')
                  else ('host C++ with shared device math emulated (debug only)'
                        if os.environ.get('MPAS_CONVECTIVE_HOST_DEVICE_MATH')=='1'
                        else 'host C++ source replay (debug only)'), 'rows': rows,
                  'source': provenance, 'cuda_source_sha256': hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
                  'native_mpi_ranks': 1, 'standalone_processes': 1,
                  'cuda_compiler': 'direct NVRTC; no RawModule FTZ override',
                  'compile_options': ['--std=c++17', '--fmad=false', '--ftz=false',
                                      '--prec-div=true', '--prec-sqrt=true'],
                  'whole_model_proof': False, 'differing_words': sum(r['different'] for r in rows)}
        import platform
        result['execution_host'] = platform.node()
        if os.environ.get('MPAS_CONVECTIVE_GPU') == '1':
            import cupy as cp
            result['device_compute_capability'] = cp.cuda.Device().compute_capability
            image = os.environ.get('MPAS_CONVECTIVE_IMAGE') or os.environ.get('MPAS_CONVECTIVE_PTX')
            if image:
                result['device_image'] = json.loads(Path(image).with_suffix(Path(image).suffix+'.json').read_text())
                result['device_image_sha256'] = hashlib.sha256(Path(image).read_bytes()).hexdigest()
        utils_source = os.environ.get('MPAS_DIAGNOSTIC_UTILS_SOURCE')
        if utils_source:
            result['stream_utils_source_sha256'] = hashlib.sha256(Path(utils_source).read_bytes()).hexdigest()
        Path(path).write_text(json.dumps(result,indent=2)+'\n')


def compare(expected, actual, receipt, label):
    a, b = expected.view(np.uint32), actual.view(np.uint32)
    byfield = {name: int(np.count_nonzero(a[i]!=b[i])) for i,name in enumerate(FIELDS)}
    receipt.append(dict(label=label, words=int(a.size), different=sum(byfield.values()),
                        by_field=byfield, reference_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                        actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
    wrong = np.argwhere(a!=b)
    if len(wrong):
        examples = [(FIELDS[i],int(c),float(expected[i,c]),float(actual[i,c]),hex(int(a[i,c])),hex(int(b[i,c])))
                    for i,c in wrong[:12]]
        pytest.fail(f'{label}: {len(wrong)} differing words: {examples}')


def initial(nc):
    out = np.arange(len(FIELDS)*nc,dtype=np.float32).reshape(len(FIELDS),nc)+np.float32(123.125)
    out[14:] = 0
    out[14:,-2:] = np.float32(91.75)  # unsolved halos
    return out


def signed_zero_replay(native, host, rows, device=False):
    """Check native tie/zero signs with quiescent winds and restored maxima."""
    arrays = columns()
    nc, nl = arrays['theta_m'].shape
    for name in ('w', 'uReconstructZonal', 'uReconstructMeridional', 'vorticity'):
        arrays[name].fill(np.float32(0))
        arrays[name][::2] = np.float32(-0.)
        arrays[name][1::2, ::2] = np.float32(-0.)
    expected = initial(nc)
    expected[2:4, ::2] = np.float32(-0.)
    expected[14:, :nc-2:2] = np.float32(-0.)
    actual = expected.copy()
    if device:
        import cupy as cp
        from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
        inputs = {name: cp.asarray(value) for name, value in arrays.items()}
        cuda = ConvectiveDiagnostics(cp.asarray(actual), nc-2, nl, MAXIMA)
    for action, requested in ((1, FIELDS), (2, ()), (3, MAXIMA)):
        native(arrays, expected, nc-2, MAXIMA, requested, action)
        if device:
            if action == 1:
                cuda.compute(inputs, requested)
            elif action == 2:
                cuda.update(inputs)
            else:
                cuda.reset(requested)
            actual = cp.asnumpy(cuda.outputs)
        else:
            host(arrays, actual, nc-2, MAXIMA, requested, action)
        compare(expected, actual, rows, f'signed zero action {action}')


def test_signed_zero_host(references, receipt):
    signed_zero_replay(*references, receipt)


def subnormal_replay(native, host, rows, device=False):
    """Keep subnormal w maxima and UH products through device arithmetic."""
    arrays = columns()
    nc, nl = arrays['theta_m'].shape
    tiny = np.arange(1, nc+1, dtype=np.uint32).view(np.float32)
    arrays['w'][:] = tiny[:, None]
    arrays['vorticity'][:] = np.float32(0.001)
    expected = initial(nc)
    actual = expected.copy()
    if device:
        import cupy as cp
        from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
        inputs = {name: cp.asarray(value) for name, value in arrays.items()}
        cuda = ConvectiveDiagnostics(cp.asarray(actual), nc-2, nl, MAXIMA)
    for action in (2, 3):
        native(arrays, expected, nc-2, MAXIMA, MAXIMA, action)
        if action == 2:
            assert np.array_equal(expected[15, :nc-2].view(np.uint32),
                                  tiny[:nc-2].view(np.uint32))
        if device:
            if action == 2:
                cuda.update(inputs)
            else:
                cuda.reset(MAXIMA)
            actual = cp.asnumpy(cuda.outputs)
        else:
            host(arrays, actual, nc-2, MAXIMA, MAXIMA, action)
        compare(expected, actual, rows, f'subnormal action {action}')


def test_subnormal_host(references, receipt):
    subnormal_replay(*references, receipt)


def maxval_replay(native, host, rows, device=False):
    """Exercise GNU MAXVAL's first-tie and leading/all-NaN branches."""
    arrays = columns()
    nc, nl = arrays['theta_m'].shape
    for name in ('w', 'uReconstructZonal', 'uReconstructMeridional', 'vorticity'):
        arrays[name].fill(np.float32(0))
    payload = np.array([0x7fc12345], np.uint32).view(np.float32)[0]
    arrays['w'][0, 0] = payload
    arrays['w'][0, nl//2] = np.float32(2)
    arrays['w'][1, :nl] = payload
    arrays['w'][1, nl] = np.float32(np.inf)  # excluded top cannot rescue MAXVAL
    arrays['w'][2, :nl] = np.float32(-np.inf)
    arrays['w'][3, ::2] = np.float32(-0.)  # first zero wins over later +0
    arrays['w'][4, 0] = payload
    arrays['w'][4, 1] = np.float32(np.inf)
    arrays['vorticity'][0] = payload
    expected = initial(nc)
    expected[15, 1] = np.float32(-3)
    expected[15, 3] = np.float32(-1)
    expected[14:, 5] = payload  # GNU final MAX preserves the prior payload
    actual = expected.copy()
    native(arrays, expected, nc-2, MAXIMA, (), 2)
    if device:
        import cupy as cp
        from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
        cuda = ConvectiveDiagnostics(cp.asarray(actual), nc-2, nl, MAXIMA)
        cuda.update({name: cp.asarray(value) for name, value in arrays.items()})
        actual = cp.asnumpy(cuda.outputs)
    else:
        host(arrays, actual, nc-2, MAXIMA, (), 2)
    compare(expected, actual, rows, 'MAXVAL first tie and NaN operands')


def test_maxval_host(references, receipt):
    maxval_replay(*references, receipt)


def test_pristine_source_pin():
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('MPAS_REF required for pinned source recheck')
    source = Path(root)/'src/core_atmosphere/diagnostics/mpas_convective_diagnostics.F'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256


def test_mask_validation():
    assert _mask(['cape','cape','cin']) == 3
    assert _mask([]) == 0
    with pytest.raises(ValueError,match='unknown'):
        _mask(['new_field'])


@pytest.mark.parametrize('requested', [FIELDS[:14], ('lcl',), ('lfc',), ('cin',),
                                        ('srh_0_1km',), ('srh_0_3km',),
                                        ('uzonal_6km','umeridional_1km'), (),
                                        ('temperature_surface',), MAXIMA])
def test_compute_host(references, receipt, requested):
    native, host = references
    arrays = columns();nc,nl = arrays['theta_m'].shape
    expected = initial(nc);actual = expected.copy()
    native(arrays,expected,nc-2,MAXIMA,requested,1)
    host(arrays,actual,nc-2,MAXIMA,requested,1)
    compare(expected,actual,receipt,'host compute '+','.join(requested))
    assert np.array_equal(expected[2:4], initial(nc)[2:4])
    assert np.array_equal(expected[:,-2:], initial(nc)[:,-2:])


@pytest.mark.parametrize('included', [MAXIMA, (MAXIMA[0],), (MAXIMA[1],), (MAXIMA[2],), ()])
def test_intervals_host(references, receipt, included):
    native, host = references
    arrays = columns();nc,nl = arrays['theta_m'].shape
    expected = initial(nc);actual = expected.copy()
    for step in range(5):
        arrays['w'] = np.ascontiguousarray(arrays['w']*np.float32(0.9))
        native(arrays,expected,nc-2,included,(),2)
        host(arrays,actual,nc-2,included,(),2)
        compare(expected,actual,receipt,f'host update {included} {step}')
        if step in (1,3):
            written = (MAXIMA[0],MAXIMA[2]) if step==1 else (MAXIMA[1],)
            native(arrays,expected,nc-2,included,written,3)
            host(arrays,actual,nc-2,included,written,3)
            compare(expected,actual,receipt,f'host reset {included} {step}')
            for name in written:
                assert (expected[FIELDS.index(name)] == 0).all()


@pytest.fixture(scope='session')
def gpu():
    if os.environ.get('MPAS_CONVECTIVE_GPU') != '1':
        pytest.skip('CUDA tests require a card and MPAS_CONVECTIVE_GPU=1')
    import cupy as cp
    return cp


@pytest.mark.gpu
@pytest.mark.parametrize('requested', [FIELDS[:14], ('lcl',), ('lfc',), ('cin',),
                                        ('srh_0_1km',), ('srh_0_3km',), (),
                                        ('temperature_surface',)])
def test_compute_gpu(references, receipt, gpu, requested):
    from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
    native, _ = references;cp=gpu
    arrays=columns();nc,nl=arrays['theta_m'].shape
    expected=initial(nc)
    device=ConvectiveDiagnostics(cp.asarray(expected),nc-2,nl,MAXIMA)
    native(arrays,expected,nc-2,MAXIMA,requested,1)
    device.compute({n:cp.asarray(a) for n,a in arrays.items()},requested)
    compare(expected,cp.asnumpy(device.outputs),receipt,'CUDA compute '+','.join(requested))


@pytest.mark.gpu
@pytest.mark.parametrize('included', [MAXIMA, (MAXIMA[0],), (MAXIMA[1],), (MAXIMA[2],), ()])
def test_intervals_gpu(references, receipt, gpu, included):
    from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
    native, _ = references;cp=gpu
    arrays=columns();nc,nl=arrays['theta_m'].shape
    expected=initial(nc)
    device=ConvectiveDiagnostics(cp.asarray(expected),nc-2,nl,included)
    for step in range(5):
        arrays['w'] = np.ascontiguousarray(arrays['w']*np.float32(0.9))
        native(arrays,expected,nc-2,included,(),2)
        inputs = {n:cp.asarray(a) for n,a in arrays.items()}
        device.update(inputs)
        compare(expected,cp.asnumpy(device.outputs),receipt,f'CUDA update {included} {step}')
        if step in (1,3):
            written=(MAXIMA[0],MAXIMA[2]) if step==1 else (MAXIMA[1],)
            native(arrays,expected,nc-2,included,written,3)
            device.reset(written)
            compare(expected,cp.asnumpy(device.outputs),receipt,f'CUDA reset {included} {step}')


def varied_columns(nl, nc=67):
    arrays=columns(nc=nc,nl=nl)
    rng=np.random.default_rng(873455+nl)
    zg=np.linspace(0,25000,nl+1,dtype=np.float32)
    arrays['zgrid']=np.ascontiguousarray(np.tile(zg,(nc,1)))
    zp=np.float32(0.5)*(zg[:-1]+zg[1:])
    scale=rng.uniform(6500,9500,(nc,1)).astype(np.float32)
    pressure=np.float32(100000)*np.exp(-zp[None,:]/scale)
    trop=rng.uniform(6500,15000,(nc,1)).astype(np.float32)
    lapse=rng.uniform(0.004,0.01,(nc,1)).astype(np.float32)
    surface=rng.uniform(275,320,(nc,1)).astype(np.float32)
    temp=surface-lapse*np.minimum(zp[None,:],trop)
    temp+=rng.uniform(0,0.006,(nc,1)).astype(np.float32)*np.maximum(zp[None,:]-trop,0)
    temp=np.maximum(temp,np.float32(180))
    temp[:,0:2]+=rng.uniform(-8,8,(nc,2)).astype(np.float32)
    es=np.float32(611.2)*np.exp(np.float32(17.67)*(temp-np.float32(273.15))/(temp-np.float32(29.65)))
    qsat=np.float32(0.622)*es/(pressure-es)
    qv=qsat*rng.uniform(0.04,1.02,(nc,nl)).astype(np.float32)
    ex=(pressure.astype(np.float64)/float(np.float32(100000)))**float(np.float32(287/1004.5))
    arrays['exner']=np.ascontiguousarray(ex.astype(np.float32))
    arrays['qv']=np.ascontiguousarray(qv)
    arrays['theta_m']=np.ascontiguousarray(temp/arrays['exner']*(np.float32(1)+np.float32(461.6/287)*qv))
    arrays['pressure_base']=np.ascontiguousarray(pressure-np.float32(37.125))
    # Reach the source's Bunkers-bottom interpolation branch with a synthetic
    # inverted lowest interface, expressly outside physical height monotonicity.
    arrays['zgrid'][0,1]=np.float32(-100)
    return arrays


@pytest.mark.parametrize('nl',[3,7,55])
def test_varied_host(references,receipt,nl):
    native,host=references
    arrays=varied_columns(nl);nc=arrays['theta_m'].shape[0]
    expected=initial(nc);actual=expected.copy()
    native(arrays,expected,nc-2,MAXIMA,FIELDS,1)
    host(arrays,actual,nc-2,MAXIMA,FIELDS,1)
    compare(expected,actual,receipt,f'host varied {nl}')


@pytest.mark.gpu
@pytest.mark.parametrize('nl',[3,7,55])
def test_varied_gpu(references,receipt,gpu,nl):
    from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
    native,_=references;cp=gpu
    arrays=varied_columns(nl);nc=arrays['theta_m'].shape[0]
    expected=initial(nc)
    device=ConvectiveDiagnostics(cp.asarray(expected),nc-2,nl,MAXIMA)
    native(arrays,expected,nc-2,MAXIMA,FIELDS,1)
    device.compute({n:cp.asarray(a) for n,a in arrays.items()},FIELDS)
    compare(expected,cp.asnumpy(device.outputs),receipt,f'CUDA varied {nl}')


@pytest.mark.gpu
def test_parallel_workspace(references,receipt,gpu):
    """Independent columns retain their workspace boundaries across launches."""
    from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnostics
    native,_=references;cp=gpu
    arrays=varied_columns(55,nc=259);nc,nl=arrays['theta_m'].shape
    expected=initial(nc)
    device=ConvectiveDiagnostics(cp.asarray(expected),nc-2,nl,MAXIMA)
    inputs={name:cp.asarray(value) for name,value in arrays.items()}
    canary=np.float32(-9182.375)
    for attempt in range(3):
        device._work.fill(canary)
        native(arrays,expected,nc-2,MAXIMA,FIELDS,1)
        device.compute(inputs,FIELDS)
        compare(expected,cp.asnumpy(device.outputs),receipt,f'CUDA parallel workspace {attempt}')
        # The unused final word in each allocation is a memory boundary canary.
        assert np.all(cp.asnumpy(device._work[:,-1]) == canary)


@pytest.fixture(scope='session')
def stream_utils():
    # Until o2 lands, load its actual stream service from an explicitly supplied
    # source path. No mocked stream or diagnostic service participates.
    try:
        from hexcore.mpasdiag.utils import DiagnosticsUtils, Direction, StreamView
    except ImportError:
        source=os.environ.get('MPAS_DIAGNOSTIC_UTILS_SOURCE')
        if not source:
            pytest.skip('o2 stream service needed for pool adapter replay')
        import importlib.util
        import sys
        name='hexcore.mpasdiag.utils'
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
        DiagnosticsUtils,Direction,StreamView=(getattr(module,n) for n in
                                             ('DiagnosticsUtils','Direction','StreamView'))
    return DiagnosticsUtils,Direction,StreamView


@pytest.mark.gpu
@pytest.mark.parametrize('sentinel_output',[False,True])
def test_live_pool_adapter(references,receipt,gpu,stream_utils,sentinel_output):
    from hexcore.mpasdiag.convective_v842 import ConvectiveDiagnosticsModule
    native,_=references;cp=gpu
    Utils,Direction,StreamView=stream_utils
    arrays=columns();nc,nl=arrays['theta_m'].shape
    # Include native's sentinel in resident mass/interface fields, independently
    # of whether the allocated output field also includes that final slot.
    resident={}
    for name,value in arrays.items():
        if name in ('cellsOnVertex','kiteAreasOnVertex','vorticity'):
            resident[name]=cp.ascontiguousarray(cp.asarray(value).T)
        elif value.ndim==2:
            resident[name]=cp.ascontiguousarray(cp.asarray(np.concatenate((value,value[-1:]),axis=0)).T)
        else:
            resident[name]=cp.asarray(np.concatenate((value,value[-1:])))
    nalloc=nc+int(sentinel_output)
    expected=initial(nalloc)
    diag={name:cp.asarray(expected[i]) for i,name in enumerate(FIELDS)}
    identities={name:diag[name] for name in FIELDS}
    for name in ('uReconstructZonal','uReconstructMeridional','exner','pressure_base','pressure_p'):
        diag[name]=resident[name]
    # The native vorticity pool also has a sentinel vertex.
    diag['vorticity']=cp.ascontiguousarray(cp.concatenate((resident['vorticity'],resident['vorticity'][:,-1:]),axis=1))
    mesh={name:resident[name] for name in ('zgrid','areaCell','cellsOnVertex','kiteAreasOnVertex')}
    mesh.update(nCells=nc,nCellsSolve=nc-2,nVertLevels=nl,nVertices=arrays['vorticity'].shape[0],vertexDegree=3)
    state={'theta_m':resident['theta_m'],'w':resident['w'],
           'scalars':cp.stack((cp.zeros_like(resident['qv']),resident['qv'])),'index_qv':2}
    streams=[StreamView('diagnostics',Direction.OUTPUT,True,True,
                        {name:True for name in FIELDS},output_interval='01:00:00')]
    utils=Utils();utils.init(lambda:streams)
    adapter=ConvectiveDiagnosticsModule()
    pools=dict(mesh=mesh,state=state,diag=diag)
    adapter.setup(pools,None,utils)
    ref_arrays={}
    for name,value in arrays.items():
        if name in ('cellsOnVertex','kiteAreasOnVertex','vorticity') or nalloc==nc:
            ref_arrays[name]=value
        else:
            ref_arrays[name]=np.ascontiguousarray(np.concatenate((value,value[-1:]),axis=0))
    # Native initial reset/update/compute/write/reset ordering.
    for action in (3,2,1,3):
        native(ref_arrays,expected,nc-2,MAXIMA,FIELDS,action)
        {1:adapter.compute,2:adapter.update,3:adapter.reset}[action]()
        result=cp.asnumpy(cp.stack([diag[name] for name in FIELDS]))
        compare(expected,result,receipt,f'pool initial {sentinel_output} {action}')
        assert all(diag[name] is identities[name] for name in FIELDS)
    # Dynamic stream alarms and field membership, plus live replacement state.
    for step,requested in enumerate(((),('cin','srh_0_1km'),('wind_speed_level1_max',),FIELDS)):
        streams[:]=[StreamView('diagnostics',Direction.OUTPUT,True,bool(requested),
                               {name:True for name in requested},output_interval='01:00:00')]
        # Existing halo and LCL/LFC values may be restored/exchanged externally.
        expected[:,-1] += np.float32(7.5)
        expected[2:4] += np.float32(1.25)
        for i,name in enumerate(FIELDS):
            diag[name][-1] = cp.float32(expected[i,-1])
        diag['lcl'][:] = cp.asarray(expected[2])
        diag['lfc'][:] = cp.asarray(expected[3])
        # Seed a live changed maximum; update must read the pool's value.
        expected[14,0] = np.float32(900)
        diag['updraft_helicity_max'][0] = cp.float32(900)
        if step==2:
            state=dict(state)
            pools['state']=state  # Native level-1 rotation replaces the pool view.
        state['w']=cp.ascontiguousarray(state['w']*cp.float32(0.5))
        ref_arrays['w']=np.ascontiguousarray(ref_arrays['w']*np.float32(0.5))
        for action in (2,1,3):
            native(ref_arrays,expected,nc-2,MAXIMA,requested,action)
            {1:adapter.compute,2:adapter.update,3:adapter.reset}[action]()
            compare(expected,cp.asnumpy(cp.stack([diag[name] for name in FIELDS])),receipt,
                    f'pool live {sentinel_output} {step} {action}')


@pytest.mark.gpu
def test_real_native_session_adapter(references, receipt, gpu, tmp_path):
    """Replay native ABI pools and actual stream alarms, without a forecast claim."""
    from types import SimpleNamespace
    from hexcore.mpasdiag.convective_v842 import NativeConvectiveDiagnostics
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.session_v842 import NativePhysicsSession
    from hexcore.mpasphys.forecast_v842 import Stream
    library = os.environ.get('MPAS_CONVECTIVE_NATIVE_LIBRARY')
    if not library:
        pytest.skip('explicit strict native session library required')
    strict, _ = references
    arrays = columns()
    nc, nl = arrays['theta_m'].shape
    nv = arrays['vorticity'].shape[0]
    extended = {name: np.ascontiguousarray(value if name in
                ('cellsOnVertex','kiteAreasOnVertex','vorticity') else
                np.concatenate((value,value[-1:]),axis=0)) for name,value in arrays.items()}
    expected = initial(nc+1)
    with NativePhysics(Path(library), test_only=True) as native:
        dims = dict(nCells=nc,nCellsSolve=nc-2,nVertices=nv,vertexDegree=3,nVertLevels=nl)
        for name, value in dims.items():
            native.dimension('mesh',name,value)
        native.block()
        native.clock(start='2000-01-01_00:00:00',stop='2000-01-01_00:04:00',dt=30.)
        native.streams()
        stream = Stream('convective',('output',),str(tmp_path/'unused.nc'),'none','00:01:00',FIELDS)
        native.stream(stream.name,filename=stream.filename,direction='output',output_interval=stream.output_interval)
        for name, value in extended.items():
            pool = 'mesh' if name in ('zgrid','areaCell','cellsOnVertex','kiteAreasOnVertex') else 'state' if name in ('theta_m','w','qv') else 'diag'
            if name == 'qv':
                native.put_field('state','scalars',np.stack((np.zeros_like(value.T),value.T)))
                continue
            # Vertex arrays retain their native allocated sentinel too.
            if name in ('cellsOnVertex','kiteAreasOnVertex','vorticity'):
                value = np.concatenate((value,value[-1:]),axis=0)
            native.put_field(pool,name,np.ascontiguousarray(value.T if value.ndim==2 else value))
        for i,name in enumerate(FIELDS):
            native.put_field('diag',name,expected[i])
        # Bind the real session copy method to already populated native pools;
        # scheme initialization/dycore are outside this subsystem ABI replay.
        session = object.__new__(NativePhysicsSession)
        session.native = native
        session.layout = SimpleNamespace(dimensions=dims,indices={('state','index_qv'):2})
        adapter = NativeConvectiveDiagnostics(session,[stream])
        for step in range(5):
            requested = FIELDS if native.alarm_ringing('convective:out') else ()
            assert bool(requested) == (step % 2 == 0)
            if step == 0:
                strict(extended,expected,nc-2,MAXIMA,requested,3)
                adapter.reset()
            # Restore externally held maxima and change the live input pool.
            if step == 2:
                expected[14,0] = np.float32(777)
                native.put_field('diag',MAXIMA[0],expected[14])
                extended['w'] *= np.float32(0.5)
                native.put_field('state','w',np.ascontiguousarray(extended['w'].T))
            for action in (2,1,3):
                strict(extended,expected,nc-2,MAXIMA,requested,action)
                {1:adapter.compute,2:adapter.update,3:adapter.reset}[action]()
                actual = np.stack([native.get_field('diag',name) for name in FIELDS])
                compare(expected,actual,receipt,f'native session {step} {action}')
                receipt[-1].update(native_output_alarm=bool(requested),dt_seconds=30,
                                  output_interval_seconds=60,
                                  native_library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
            if requested:
                native.reset_alarm('convective:out')
            native.advance_clock()
