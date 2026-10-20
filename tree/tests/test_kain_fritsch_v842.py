"""Strict, complete-column comparisons against the pristine MPAS module.

Build once using tests/kain_fritsch_oracle/build.py, then make -j4 (through the build queue on a shared host).
Set MPAS_KF_ORACLE to that directory. CUDA tests also require MPAS_KF_GPU=1
and must run through the GPU queue. These checks do not establish forecast identity.
"""
import ctypes as C
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from hexcore.mpasphys import kain_fritsch_v842 as kf

sys.path.insert(0,str(Path(__file__).parent))
from kain_fritsch_oracle.support import PARAMS,fixture,load_oracles,native_column,word_diff
from kain_fritsch_oracle.replay import replay
from kain_fritsch_oracle.helpers import replay_helpers


@pytest.fixture(scope='module')
def oracle_dir():
    value=os.environ.get('MPAS_KF_ORACLE')
    if not value:pytest.skip('standalone MPAS oracle not built: set MPAS_KF_ORACLE')
    return Path(value)


@pytest.fixture(scope='module')
def oracle(oracle_dir):
    return load_oracles(oracle_dir)


def test_complete_module_host_glibc(oracle_dir):
    result=replay(oracle_dir,'host-glibc')
    assert result['different_columns']==0
    assert result['nca_300_columns']>0


def test_register_callback_signature():
    # No numerical mock: this checks only registration metadata at the f4 seam.
    class Registry:
        def register(self,*args):self.row=args
    registry=Registry();kf.register(registry)
    assert registry.row==('convection','cu_kain_fritsch','hex-cuda',kf.launch)


def test_measured_native_argument_preserves_raw_logical_word():
    for word in (0, 1, 32764, 32765, -2147483648, 2147483647):
        arguments = kf.load_native_arguments(
            {'warm_rain_word': word, 'witness_sha256': 'a' * 64}, None)
        assert arguments['warm_rain_word'].dtype == np.dtype('int32')
        assert int(arguments['warm_rain_word']) == word


def test_install_packet_loads_and_binds_only_where_warm_rain_is_unread():
    import pytest
    arguments = kf.load_native_arguments(kf.install_packet(), None)
    assert int(arguments['warm_rain_word']) == 0 and arguments['warm_rain'] == kf.INSTALL_WARM_RAIN
    kf.check_warm_rain(arguments, np.int32(1))
    with pytest.raises(RuntimeError, match='no cloud ice'):
        kf.check_warm_rain(arguments, np.int32(0))
    measured = kf.load_native_arguments({'warm_rain_word': 32765, 'witness_sha256': 'a' * 64}, None)
    kf.check_warm_rain(measured, np.int32(0))
    with pytest.raises(ValueError, match='install_packet'):
        kf.load_native_arguments(kf.install_packet() | {'warm_rain_word': 1}, None)


def test_native_namelist_selects_only_kf_cuda():
    from hexcore.mpasphys.driver_v842 import Backend, SlotRegistry, route_namelist
    config = route_namelist("""&physics
 config_physics_suite='none'
 config_convection_scheme='cu_kain_fritsch'
 /
 &hex_mpasphys
 config_convection_backend='hex-cuda'
 /""")
    assert config.selectors['convection'] == kf.SCHEME
    assert config.backends['convection'] == Backend.HEX_CUDA
    # Suite 'none' leaves every other slot off, whatever its (default hex-cuda) backend label.
    assert not any(config.active(slot) for slot in config.backends if slot != 'convection')
    registry = SlotRegistry()
    kf.register(registry)
    registry.require(config)


GPU=pytest.mark.skipif(os.environ.get('MPAS_KF_GPU')!='1',reason='GPU tests require the GPU queue and MPAS_KF_GPU=1')


@GPU
def test_native_initialization_and_restart(oracle):
    import cupy as cp
    native,_,_=oracle
    for restart in (0,1):
        for qi,qs in ((0,0),(0,1),(1,0),(1,1)):
            _,s,v=fixture(55);s[:]=np.float32(.0125);v[4]=300
            ns,nv=s.copy(),v.copy()
            native.kf_native_init(55,ns,nv,restart,qi,qs,PARAMS)
            ds,dv=cp.asarray(s[:,:,None]),cp.asarray(v[:,None])
            kf.initialize_packed(ds,dv,restart=restart,f_qi=qi,f_qs=qs)
            assert word_diff(cp.asnumpy(ds)[:,:,0].copy(),ns)==0
            assert word_diff(cp.asnumpy(dv)[:,0].copy(),nv)==0


@GPU
def test_complete_module_cuda(oracle_dir):
    result=replay(oracle_dir,'hex-cuda')
    assert result['different_columns']==0,result
    assert result['nca_300_columns']>0


@GPU
def test_subnormal_input_survives_native_hold(oracle):
    import cupy as cp
    native, _, _ = oracle
    f, s, v = fixture(55)
    # Positive NCA holds the input tendencies. The pack/run/unpack path must
    # retain raw subnormal words, including the minimum positive/negative ones.
    s[1, :4] = np.array([1, 0x80000001, 0x007fffff, 0x807fffff], np.uint32).view(np.float32)
    v[4] = 300
    flags = np.array([5, 2, 0, 0, 1, 1, 1], np.int32)
    ns, nv, na = native_column(native, f.copy(), s.copy(), v.copy(), flags)
    ds, dv, active = cp.asarray(s[:, :, None]), cp.asarray(v[:, None]), cp.zeros(1, cp.uint8)
    kf.run_packed(cp.asarray(f[:, :, None]), ds, dv, active,
                  cp.asarray(PARAMS), cp.asarray(flags))
    assert word_diff(cp.asnumpy(ds)[:, :, 0].copy(), ns) == 0
    assert word_diff(cp.asnumpy(dv)[:, 0].copy(), nv) == 0
    assert cp.asnumpy(active)[0] == na


@GPU
def test_gradual_underflow_in_native_ice_tendency(oracle):
    import cupy as cp
    from kain_fritsch_oracle.support import cases
    native, _, _ = oracle
    # This exact strict-native input exposed two differing words when CuPy
    # appended --ftz=true. A byte-only hold test alone cannot detect that bug.
    _, (f, s, v), flags = next(row for row in cases()
                               if row[0] == ('dry', 65, 15, .1, 320, 5))
    ns, nv, na = native_column(native, f.copy(), s.copy(), v.copy(), flags)
    words = ns.view(np.uint32) & np.uint32(0x7fffffff)
    assert np.any((words > 0) & (words < 0x00800000))
    ds, dv, active = cp.asarray(s[:, :, None]), cp.asarray(v[:, None]), cp.zeros(1, cp.uint8)
    kf.run_packed(cp.asarray(f[:, :, None]), ds, dv, active,
                  cp.asarray(PARAMS), cp.asarray(flags))
    assert word_diff(cp.asnumpy(ds)[:, :, 0].copy(), ns) == 0
    assert word_diff(cp.asnumpy(dv)[:, 0].copy(), nv) == 0
    assert cp.asnumpy(active)[0] == na


@GPU
def test_driver_pack_unpack(oracle):
    import cupy as cp
    native,_,_=oracle
    nz,nc=55,3
    f,s,v=fixture(nz,8,.95,300,2)
    fa=[cp.asarray(np.stack([f[i,:nz]]*nc,axis=1)) for i in range(10)]
    fa[4]=cp.asarray(np.stack([f[4]]*nc,axis=1))
    prepared=dict(zip(kf._PREPARED_FIELDS,fa))
    diag={'w0avg':cp.asarray(np.stack([s[0,:nz]]*nc,axis=1))}
    diag.update({name:cp.asarray(np.full(nc,v[idx],np.float32)) for idx,name in enumerate(kf._SURFACE_POOL) if idx>=2})
    tend={name:cp.asarray(np.full((nz,nc),np.float32(i+.125),np.float32)) for i,name in enumerate(kf.STATE[1:])}
    density=np.array([.01,1,10000],np.float32)
    mesh={'meshDensity':cp.asarray(density),'areaCell':cp.asarray(np.full(nc,v[1],np.float32))}
    fields,state,surface,pointers=kf.convection_from_MPAS(prepared,mesh,diag,tend,100000)
    expected_dx=np.empty(nc,np.float32);native.kf_native_dx(nc,100000,density,expected_dx)
    assert word_diff(cp.asnumpy(surface[0]),expected_dx)==0
    for i,name in enumerate(kf.STATE[1:],1):
        assert word_diff(cp.asnumpy(state[i,:nz]),cp.asnumpy(tend[name]))==0
    kf.convection_to_MPAS(state,surface,pointers)
    for i,name in enumerate(kf.STATE[1:],1):
        assert word_diff(cp.asnumpy(state[i,:nz]),cp.asnumpy(tend[name]))==0
    for i,name in enumerate(kf._SURFACE_POOL):
        if i>=2:assert word_diff(cp.asnumpy(surface[i]),cp.asnumpy(diag[name]))==0


@GPU
def test_driver_hold_expiry_and_bucket(oracle):
    import cupy as cp
    native,_,_=oracle
    for nca in (-100,0,29,30,89,90,300):
        _,s,v=fixture(55);s[:]=np.float32(.0125);v[4]=nca;v[2:4]=.25
        ns,nv=s.copy(),v.copy();native.kf_native_step1(55,60,ns,nv)
        ds,dv=cp.asarray(s[:,:,None]),cp.asarray(v[:,None])
        kf.update_convection_step1(ds,dv,60)
        assert word_diff(cp.asnumpy(ds)[:,:,0].copy(),ns)==0
        assert word_diff(cp.asnumpy(dv)[:,0].copy(),nv)==0
    for alarm,bucket in ((False,1),(True,0),(True,1)):
        # Exact threshold, multiple bucket overflow and negative-rate preservation.
        rates=np.array([0,.01,.1,-.1],np.float32);rain=np.array([1,.4,4,.4],np.float32)
        nr,ncount=rain.copy(),np.array([0,1,2,3],np.int32)
        native.kf_native_step2(4,60,bucket,int(alarm),rates,nr,ncount)
        dr,dc=cp.asarray(rain),cp.asarray(np.array([0,1,2,3],np.int32))
        kf.update_convection_step2(cp.asarray(rates),dr,dc,60,bucket,alarm)
        assert word_diff(cp.asnumpy(dr),nr)==0
        np.testing.assert_array_equal(cp.asnumpy(dc),ncount)


@GPU
def test_slot_callback_updates_native_pools(oracle):
    import cupy as cp
    native,_,_=oracle
    nz,nc=55,2
    columns=[fixture(nz,8,.95,300,2),fixture(nz,3,.3,285,0)]
    f=np.stack([x[0] for x in columns],axis=2)
    state=np.stack([x[1] for x in columns],axis=2)
    surface=np.stack([x[2] for x in columns],axis=1)
    prepared={name:cp.asarray(f[i,:nz+(name=='w_p')].copy())
              for i,name in enumerate(kf._PREPARED_FIELDS)}
    diag={'w0avg':cp.asarray(state[0,:nz].copy())}
    diag.update({name:cp.asarray(surface[i].copy()) for i,name in enumerate(kf._SURFACE_POOL) if i>=2})
    tend={name:cp.asarray(state[i,:nz].copy()) for i,name in enumerate(kf.STATE) if i>=1}
    pools={'prepared':prepared,'diag_physics':diag,'tend_physics':tend,
           'mesh':{'areaCell':cp.asarray(surface[1].copy()),'meshDensity':cp.asarray(np.ones(nc,np.float32))},
           'dt_dyn':60,'n_cu':5,'len_disp':np.float32(100000),'kain_fritsch':{'params':cp.asarray(PARAMS),'flags':cp.asarray(np.array([5,1,0,0,1,1,1],np.int32))}}
    pools['kain_fritsch']['tables']=kf.build_tables(pools['kain_fritsch']['params'])
    # Actual column physics and pool writeback; SlotCall metadata is independent
    # of f4 imports so this can run before the common driver has landed.
    for step in (1,2,3,4,5):
        flags=np.array([5,step if step==1 else step+1,0,0,1,1,1],np.int32)
        expected_active=[]
        for c in range(nc):
            ns,nv,na=native_column(native,f[:,:,c].copy(),state[:,:,c].copy(),surface[:,c].copy(),flags)
            state[:,:,c]=ns;surface[:,c]=nv;expected_active.append(na)
        # MPAS core can resolve its live config pointer after namelist parsing.
        call=SimpleNamespace(pools=pools,dt=300,itimestep=step,options={'config_len_disp':0})
        active=kf.launch(call)
        np.testing.assert_array_equal(cp.asnumpy(active),expected_active)
        assert word_diff(cp.asnumpy(diag['w0avg']),state[0,:nz].copy())==0
        for i,name in enumerate(kf.STATE):
            if i>=1:assert word_diff(cp.asnumpy(tend[name]),state[i,:nz].copy())==0
        for i,name in enumerate(kf._SURFACE_POOL):
            if i>=2:assert word_diff(cp.asnumpy(diag[name]),surface[i].copy())==0


def test_scalar_boundaries_host(oracle_dir):
    result=replay_helpers(oracle_dir,'host-glibc')
    assert result['differing_words']==0,result


@GPU
def test_scalar_boundaries_cuda(oracle_dir):
    result=replay_helpers(oracle_dir,'hex-cuda')
    assert result['differing_words']==0,result


def test_compiled_image_retains_selected_source_and_strict_options(tmp_path):
    source = 'complete selected translation unit'
    image = b'compiled artifact'
    (tmp_path / 'module.cubin').write_bytes(image)
    manifest = dict(schema='mpas-kf-cuda-image/v1',
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    options=list(kf.CUDA_OPTIONS), architecture='120', format='cubin',
                    image='module.cubin', image_sha256=hashlib.sha256(image).hexdigest())
    path = tmp_path / 'module.json'
    path.write_text(json.dumps(manifest))
    assert kf.cuda_image(path, source, '120') == image
    with pytest.raises(ValueError, match='source, strict options or architecture'):
        kf.cuda_image(path, source + ' changed', '120')
    manifest['options'] = [option.replace('--ftz=false', '--ftz=true')
                           for option in kf.CUDA_OPTIONS]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='strict options'):
        kf.cuda_image(path, source, '120')
    manifest['options'] = list(kf.CUDA_OPTIONS)
    path.write_text(json.dumps(manifest))
    (tmp_path / 'module.cubin').write_bytes(b'different artifact')
    with pytest.raises(ValueError, match='bytes differ'):
        kf.cuda_image(path, source, '120')


@GPU
def test_actual_native_device_pool_driver(tmp_path):
    """Whole native driver updates through the real f4 copy/commit provider."""
    import subprocess
    library = os.environ.get('MPAS_KF_NATIVE_LIBRARY')
    loader = os.environ.get('MPAS_KF_NATIVE_LOADER')
    if not library or not loader:
        pytest.skip('actual paired f3 artifact required for native device-pool replay')
    receipt = tmp_path / 'provider-driver.json'
    script = Path(__file__).parent / 'kain_fritsch_oracle/provider_driver.py'
    subprocess.run([sys.executable, str(script), '--library', library,
                    '--loader', loader, '--receipt', str(receipt)], check=True)
    result = json.loads(receipt.read_text())
    assert result['steps'] == 5 and result['differences'] == 0
    assert result['halo_columns'] == 1
