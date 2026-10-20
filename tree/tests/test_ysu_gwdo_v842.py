"""Whole YSU GWDO comparison to pristine MPAS v8.4.2 GNU strict Fortran.

Build the support Makefile (through the build queue on a shared host). GPU tests
must run via the GPU queue there.
Set YSU_GWDO_ORACLE to the resulting library. Synthetic inputs are test-only;
no Python production data path or Python translation is used as an oracle.
"""
from __future__ import annotations

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys import ysu_gwdo_v842 as port


def synthetic_columns(nlev, ncol=256, seed=2718281):
    rng = np.random.default_rng(seed)
    dz = np.linspace(40, 1000, nlev, dtype=np.float32)
    interfaces = np.r_[np.float32(0), np.cumsum(dz)]
    z = np.float32(.5) * (interfaces[:-1] + interfaces[1:])
    height = rng.uniform(0, 3000, ncol).astype(np.float32)
    p2 = np.float32(101325) * np.exp(-(interfaces[:, None]+height[None, :])/np.float32(8500))
    p = np.float32(.5) * (p2[:-1] + p2[1:])
    pi = (p/np.float32(100000))**np.float32(287/1004.5)
    lapse = np.choose(np.arange(ncol)%4, [-.005, .012, .0005, .02]).astype(np.float32)
    theta = np.float32(295) + z[:, None] * lapse[None, :]
    angle = rng.uniform(-np.pi, np.pi, ncol).astype(np.float32)
    speed = rng.uniform(.1, 30, ncol).astype(np.float32)
    speed[0] = 0
    shear = rng.uniform(-.001, .003, ncol).astype(np.float32)
    shear[0] = 0
    wind = speed[None, :] + z[:, None]*shear[None, :]
    wind[z[:, None] > rng.uniform(1500, 10000, ncol)] *= np.float32(-.6)
    u = wind * np.sin(angle)
    v = wind * np.cos(angle)
    q = np.float32(.016) * np.exp(-z[:, None]/np.float32(2500)) * np.ones((1,ncol),np.float32)
    var = rng.uniform(0, 1300, ncol).astype(np.float32)
    var[1] = 0
    var[2] = np.float32(1e6)  # no crossing: native falls back to kbl=2
    var[3] = 900
    u[:,3] = 3; v[:,3] = 0
    theta[:,3] = np.float32(295)+np.float32(.02)*z
    # Signed-zero, all directional boundaries, limiting and blocked profiles.
    u[:,4] = np.float32(-0.0); v[:,4] = np.float32(0.0)
    u[0,5] = -15; u[1:,5] = 15; v[:,5] = 0
    u[:,-1] = 80; v[:,-1] = 30; var[-1] = 10
    arrays = dict(u_p=u,v_p=v,t_p=theta*pi,qv_p=q,pres_hydd_p=p,pres2_hydd_p=p2,
                  pi_p=pi,zmid_p=np.broadcast_to(z[:,None]+height[None,:],p.shape).copy(),
                  rublten=rng.uniform(-.001,.001,p.shape).astype(np.float32),
                  rvblten=rng.uniform(-.001,.001,p.shape).astype(np.float32),
                  meshDensity=rng.uniform(.02,1,ncol).astype(np.float32),
                  var2d=var,con=rng.uniform(0,8,ncol).astype(np.float32))
    arrays['con'][7] = 0  # non-disabled drag with zero incident stress (:505)
    arrays['var2d'][7] = 100
    arrays['u_p'][:,7] = 15; arrays['v_p'][:,7] = 0
    # Native bl_gwdo.F90:228-231 uses virtual temperature directly and has
    # no qv>=0 gate. The former port guard aborted these defined columns.
    arrays['qv_p'][:,8] = np.float32(-.001)
    # MPAS bl_gwdo.F90:337-355: NINT selects directional terrain statistics.
    # Exercise float neighbors of each angular half-sector, rather than
    # relying on random winds to approach those discontinuities.
    for sector in range(8):
        boundary = np.float32(-np.pi + (sector + .5) * np.pi / 4)
        for offset, phase in enumerate((np.nextafter(boundary, np.float32(-np.inf)),
                                        boundary,
                                        np.nextafter(boundary, np.float32(np.inf)))):
            column = 16 + sector * 3 + offset
            arrays['u_p'][:,column] = np.float32(15) * np.sin(phase)
            arrays['v_p'][:,column] = np.float32(15) * np.cos(phase)
            arrays['t_p'][:,column] = (np.float32(295) + np.float32(.01)*z) * pi[:,column]
            arrays['var2d'][column] = 100
    for direction in range(1,5):
        arrays[f'oa{direction}'] = rng.uniform(-1,1,ncol).astype(np.float32)
        arrays[f'ol{direction}'] = rng.uniform(0,1,ncol).astype(np.float32)
        arrays[f'ol{direction}'][6] = 0
    return {k:np.ascontiguousarray(v,dtype=np.float32) for k,v in arrays.items()}


def native_oracle(arrays, dt, length, *, library_path=None, symbol="ysu_gwdo_oracle"):
    path = library_path or os.environ.get('YSU_GWDO_ORACLE')
    if not path:
        pytest.skip('YSU_GWDO_ORACLE must name the built pristine MPAS library')
    library = ctypes.CDLL(path)
    function = getattr(library, symbol)
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    names = ('u_p','v_p','t_p','qv_p','pres_hydd_p','pres2_hydd_p','pi_p','zmid_p',
             'rublten','rvblten',*port.STATIC_INPUTS)
    function.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_float,ctypes.c_float,*([pointer]*len(names)),pointer,ctypes.POINTER(ctypes.c_int)]
    function.restype = None
    nlev,ncol = arrays['u_p'].shape
    words=nlev*ncol
    result=np.empty(words*6+ncol*2,dtype=np.float32)
    err=ctypes.c_int(-1)
    function(nlev,ncol,dt,length,*(arrays[k] for k in names),result,ctypes.byref(err))
    assert err.value == 0
    outputs = {name:result[i*words:(i+1)*words].reshape(nlev,ncol) for i,name in enumerate(port.LAYER_OUTPUTS)}
    for i,name in enumerate(port.SURFACE_OUTPUTS):
        outputs[name] = result[6*words+i*ncol:6*words+(i+1)*ncol]
    return outputs


def test_pristine_source_recheck():
    root = os.environ.get('MPAS_REF')
    if not root:
        pytest.skip('pinned MPAS_REF needed for source recheck')
    root = Path(root)/'src/core_atmosphere/physics'
    for name, expected in port.SOURCE_SHA256.items():
        assert sha256((root/name).read_bytes()).hexdigest() == expected
    assert port.bl_gwdo_init() == ('bl_gwdo_init OK',0)
    assert port.bl_gwdo_finalize() == ('bl_gwdo_finalize OK',0)


@pytest.mark.gpu
@pytest.mark.parametrize('nlev,dt,length', [(2,20,480000),(3,120,240000),(8,600,3000),
                                          (40,20,12000),(55,120,25000),(72,3600,480000)])
def test_whole_module_bitwise(nlev,dt,length):
    import cupy as cp
    arrays = synthetic_columns(nlev)
    expected = native_oracle(arrays,dt,length)
    resident = {k:cp.asarray(v) for k,v in arrays.items()}
    artifact = None
    if directory := os.environ.get('YSU_GWDO_KERNEL_ARTIFACTS'):
        prefix = Path(directory) / f'ysu-{nlev}-dx0'
        artifact = json.loads(prefix.with_suffix('.json').read_text())
        artifact['binary'] = prefix.with_suffix('.cubin').read_bytes()
    actual = port.run_bl_ysu_gwdo_cuda_v842(dt_seconds=dt,config_len_disp=length,
                                          kernel_artifact=artifact,**resident)
    counts = {}
    details = {}
    for name in expected:
        got = actual[name].get()
        counts[name] = int(np.count_nonzero(got.view(np.uint32)!=expected[name].view(np.uint32)))
        if counts[name]:
            indexes = np.argwhere(got.view(np.uint32)!=expected[name].view(np.uint32))[:8]
            details[name] = [dict(index=idx.tolist(),got=float(got[tuple(idx)]),native=float(expected[name][tuple(idx)]),
                                 got_bits=int(got.view(np.uint32)[tuple(idx)]),native_bits=int(expected[name].view(np.uint32)[tuple(idx)])) for idx in indexes]
    receipt = dict(nlev=nlev,ncol=256,dt_seconds=dt,config_len_disp=length,counts=counts,details=details,process_count=1,
                   input_sha256={name:sha256(value.tobytes()).hexdigest() for name,value in arrays.items()},
                   source_sha256=port.SOURCE_SHA256,kernel_sha256=sha256(port.kernel_source(nlev).encode()).hexdigest(),
                   compile_options=list(port.COMPILE_OPTIONS),authority='pristine MPAS v8.4.2 module + wrapper + YSU driver slice, GNU strict')
    directory = os.environ.get('YSU_GWDO_RECEIPTS')
    if directory:
        Path(directory).mkdir(exist_ok=True,parents=True)
        (Path(directory)/f'synthetic-{nlev}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert not any(counts.values()), details
    # Actual MPAS driver integration, including rounded diagnostics and unchanged thermal carrier.
    tend={k:resident[k].copy() for k in ('rublten','rvblten')}
    thermal=cp.full((nlev,256),np.float32(3e-5));tend['rthblten']=thermal
    diag={name:cp.empty_like(value) for name,value in actual.items() if name not in ('rublten','rvblten')}
    prepared = dict(resident)
    for name in port.LAYER_INPUTS[:-2]:
        prepared[name] = cp.concatenate((resident[name], cp.full((1,256), cp.nan, dtype=cp.float32)))
    port.driver_gwdo(prepared=prepared,mesh=resident,sfc_input=resident,tend_physics=tend,
                     diag_physics=diag,dt_pbl=dt,config_len_disp=length,kernel_artifact=artifact)
    for name in expected:
        destination=tend if name in tend else diag
        np.testing.assert_array_equal(destination[name].get().view(np.uint32),expected[name].view(np.uint32))
    assert tend['rthblten'] is thermal
    np.testing.assert_array_equal(thermal.get(),np.full((nlev,256),np.float32(3e-5)))
    for name in arrays:
        np.testing.assert_array_equal(resident[name].get().view(np.uint32),arrays[name].view(np.uint32))


@pytest.mark.parametrize('defect', ['source', 'binary'])
def test_compiled_artifact_rejects_stale_or_corrupted_kernel(defect):
    # Validation precedes the device context, so this check needs no GPU slot.
    source_hash = sha256(port.kernel_source(55, True).encode()).hexdigest()
    binary = b'corrupted-cubin'
    binary_hash = sha256(binary).hexdigest()
    if defect == 'source':
        source_hash = '0' * 64
    else:
        binary_hash = '0' * 64
    with pytest.raises(ValueError, match='source|binary hash'):
        port._artifact_module(55, 0, True, source_hash, binary_hash,
                              (*port.COMPILE_OPTIONS, '--gpu-architecture=sm_120'), binary)


@pytest.mark.parametrize('nlev,dt,length', [(2,20,480000),(3,120,240000),(8,600,3000),
                                          (40,20,12000),(55,120,25000),(72,3600,480000)])
def test_host_glibc_witness(nlev,dt,length):
    bridge = os.environ.get('YSU_GWDO_HOST_BRIDGE')
    if not bridge:
        pytest.skip('YSU_GWDO_HOST_BRIDGE needs the source-derived C++ bridge')
    arrays = synthetic_columns(nlev)
    expected = native_oracle(arrays,dt,length)
    actual = native_oracle(arrays,dt,length,library_path=bridge,symbol='ysu_gwdo_bridge')
    counts = {name:int(np.count_nonzero(actual[name].view(np.uint32)!=expected[name].view(np.uint32))) for name in expected}
    details = {}
    for name in expected:
        if counts[name]:
            indexes=np.argwhere(actual[name].view(np.uint32)!=expected[name].view(np.uint32))[:8]
            details[name]=[dict(index=idx.tolist(),got=float(actual[name][tuple(idx)]),native=float(expected[name][tuple(idx)]),
                                got_bits=int(actual[name].view(np.uint32)[tuple(idx)]),native_bits=int(expected[name].view(np.uint32)[tuple(idx)])) for idx in indexes]
    directory = os.environ.get('YSU_GWDO_RECEIPTS')
    if directory:
        Path(directory).mkdir(exist_ok=True,parents=True)
        (Path(directory)/f'host-glibc-{nlev}.json').write_text(json.dumps(dict(nlev=nlev,ncol=256,dt_seconds=dt,process_count=1,
              input_sha256={name:sha256(value.tobytes()).hexdigest() for name,value in arrays.items()},
              config_len_disp=length,counts=counts,details=details,source_sha256=port.SOURCE_SHA256,
              body_sha256=sha256(port._KERNEL_BODY.encode()).hexdigest(),
              authority='production CUDA body on host GNU strict with glibc 2.43 elementary functions versus pristine MPAS Fortran'),indent=2)+'\n')
    assert not any(counts.values()), details


def subnormal_columns_and_oracle():
    # Native bl_gwdo.F90:630-634 adds zero drag to incoming tendencies.
    # With FTZ enabled, those additions destroy the subnormal input words.
    arrays = synthetic_columns(2)
    arrays['var2d'].fill(0)
    patterns = np.array([1, 0x007fffff, 0x80000001, 0x807fffff], np.uint32)
    for name in ('rublten', 'rvblten'):
        arrays[name].view(np.uint32)[:] = np.resize(patterns, arrays[name].shape)
    expected = native_oracle(arrays, 20, 480000)
    for name in ('rublten', 'rvblten'):
        np.testing.assert_array_equal(expected[name].view(np.uint32), arrays[name].view(np.uint32))
    return arrays, expected


def compare_subnormal_outputs(backend, arrays, expected, actual):
    counts = {name:int(np.count_nonzero(actual[name].view(np.uint32) != value.view(np.uint32)))
              for name,value in expected.items()}
    directory = os.environ.get('YSU_GWDO_RECEIPTS')
    if directory:
        Path(directory).mkdir(exist_ok=True, parents=True)
        (Path(directory)/f'subnormal-{backend}.json').write_text(json.dumps(dict(
            backend=backend, counts=counts, output_words=sum(value.size for value in expected.values()),
            input_sha256={name:sha256(value.tobytes()).hexdigest() for name,value in arrays.items()},
            kernel_sha256=sha256(port.kernel_source(2).encode()).hexdigest(),
            compile_options=list(port.COMPILE_OPTIONS), process_count=1,
            authority='pristine MPAS single-precision strict Fortran; preserved subnormal additions'),
            indent=2)+'\n')
    assert not any(counts.values()), counts


def test_subnormal_host_glibc_witness():
    arrays, expected = subnormal_columns_and_oracle()
    bridge = os.environ.get('YSU_GWDO_HOST_BRIDGE')
    if not bridge:
        pytest.skip('YSU_GWDO_HOST_BRIDGE needs the compiled production body')
    actual = native_oracle(arrays, 20, 480000, library_path=bridge, symbol='ysu_gwdo_bridge')
    compare_subnormal_outputs('host-glibc', arrays, expected, actual)


@pytest.mark.gpu
def test_subnormal_cuda_tendencies_survive_native_additions():
    import cupy as cp
    arrays, expected = subnormal_columns_and_oracle()
    artifacts = os.environ.get('YSU_GWDO_KERNEL_ARTIFACTS')
    artifact = None
    if artifacts:
        prefix = Path(artifacts) / 'ysu-2-dx0'
        artifact = json.loads(prefix.with_suffix('.json').read_text())
        artifact['binary'] = prefix.with_suffix('.cubin').read_bytes()
    result = port.run_bl_ysu_gwdo_cuda_v842(
        dt_seconds=20, config_len_disp=480000, kernel_artifact=artifact,
        **{name:cp.asarray(value) for name,value in arrays.items()})
    actual = {name:value.get() for name,value in result.items()}
    compare_subnormal_outputs('hex-cuda', arrays, expected, actual)


def test_foundation_binding_uses_native_manager_and_constants(monkeypatch):
    from types import SimpleNamespace
    pools = {name: {} for name in ('prepared', 'mesh', 'sfc_input',
                                   'tend_physics', 'diag_physics')}
    constants = dict(gravity=np.float32(9.80616), cp=np.float32(1004.5),
                     r_d=np.float32(287), ep_1=np.float32(.608),
                     pii=np.float32(3.1415927))
    pools.update(constants=constants, scalar_state={'dt_pbl': np.float32(600)},
                 configs={'config_len_disp': np.float32(480000)})
    call = SimpleNamespace(dt=20, pools=port.bind_forecast(pools, None))
    observed = {}
    monkeypatch.setattr(port, 'driver_gwdo', lambda **arguments: observed.update(arguments))
    port.forecast_callback(call)
    assert observed['dt_pbl'] == np.float32(600)
    assert observed['config_len_disp'] == np.float32(480000)
    assert observed['constants'] == {name: constants[native] for name, native in
                                    (('gravity', 'gravity'), ('cp', 'cp'), ('rd', 'r_d'),
                                     ('fv', 'ep_1'), ('pi', 'pii'))}
    for name in ('prepared', 'mesh', 'sfc_input', 'tend_physics', 'diag_physics'):
        assert observed[name] is pools[name]


def test_native_prepared_ghosts_do_not_enter_physical_driver_views():
    arrays = synthetic_columns(55)
    prepared = dict(arrays)
    for name in port.LAYER_INPUTS[:-2]:
        prepared[name] = np.concatenate((arrays[name], np.full((1,256), np.nan, np.float32)))
    result = port.gwdo_from_MPAS(prepared, arrays, arrays, arrays)
    for name in (*port.LAYER_INPUTS, 'pres2_hydd_p', *port.STATIC_INPUTS):
        np.testing.assert_array_equal(result[name].view(np.uint32), arrays[name].view(np.uint32))
        assert np.shares_memory(result[name], prepared[name] if name in port.LAYER_INPUTS[:-2] else arrays[name])
