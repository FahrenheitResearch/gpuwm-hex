"""Framework boundary checks for the ordered driver; no numerical proof claim."""
from dataclasses import asdict
from types import SimpleNamespace
import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hexcore.mpasphys.driver_v842 import (
    NativeDriverHooks, NativePoolArguments, PhysicsDriver, PhysicsManager,
    SlotCall, SlotRegistry, Backend, StepFlags, resolve_namelist,
)
from test_mpasphys_driver_v842 import RecordingHooks, deck


def hooks(entries=(),external=None):
    native=SimpleNamespace(abi={'entries':{k:{'arguments':[]} for k in entries}},
                           config=lambda *args:None,call=lambda *args,**kwargs:None)
    return NativeDriverHooks(native,lambda *args:{},after_call=lambda *args:None,
                             external=external or {},save_fields=lambda:{},restore_fields=lambda p:None)


def test_preflight_covers_unconditional_bucket_and_radiation_deallocation():
    config=resolve_namelist(deck('none',config_sfclayer_scheme='sf_monin_obukhov_rev',
                                config_bucket_update='01:00:00',config_bucket_radt=100.))
    assert 'update_radiation_diagnostics' in hooks().missing_operations(config)
    config=resolve_namelist(deck('none',config_radt_lw_scheme='rrtmg_lw'),
                           backends={'cloudiness':'off'})
    missing=hooks().missing_operations(config)
    assert 'deallocate_cloudiness' in missing
    assert 'allocate_cloudiness' not in missing
    config=resolve_namelist(deck('none',config_sfclayer_scheme='sf_monin_obukhov_rev',
                                config_radt_cld_scheme='cld_fraction'))
    assert 'allocate_cloudiness' not in hooks().missing_operations(config)
    with pytest.raises(RuntimeError,match='not bound'): hooks().preflight(config)
    hooks().preflight(resolve_namelist(deck()))


def test_native_boundary_errors_precede_calls():
    config=resolve_namelist(deck('none',config_microp_scheme='mp_wsm6'))
    call=SlotCall('microphysics','mp_wsm6',Backend.NATIVE_CPU,1,0,1,StepFlags(),None,config.options)
    adapter=hooks()
    with pytest.raises(RuntimeError,match='not bound'):adapter.invoke('absent',call)
    with pytest.raises(RuntimeError,match='no slot entry'):adapter.register_slots(SlotRegistry(),config)
    adapter=hooks(['entry']);adapter.arguments=lambda *args:{'extra':1}
    with pytest.raises(ValueError,match='argument names differ'):adapter.invoke('entry',call)
    with pytest.raises(ValueError,match='one-based'):
        NativePoolArguments(adapter.native,pools={},its=0,ite=1,synchronize=lambda *a:None)
    native=SimpleNamespace(abi={'entries':{'entry':{'arguments':['required']}}})
    args=NativePoolArguments(native,pools={},its=1,ite=1,synchronize=lambda *a:None)
    with pytest.raises(RuntimeError,match='no bound pool'):args('entry',call)


def test_lifetime_and_restart_errors_and_cleanup():
    config=resolve_namelist(deck());manager=PhysicsManager(config,1);registry=SlotRegistry()
    driver=PhysicsDriver(config,manager,registry,RecordingHooks())
    with pytest.raises(RuntimeError,match='requires initialization'):driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    with pytest.raises(RuntimeError,match='completed initialized'):driver.finalize(None)
    with pytest.raises(RuntimeError,match='no physics step'):driver.finish_step(pools=None)
    with pytest.raises(RuntimeError,match='outside'):driver.physics_get_tend(rk_step=1,dynamics_substep=1,pools=None)
    driver.initialize(None)
    with pytest.raises(RuntimeError,match='already initialized'):driver.initialize(None)
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    with pytest.raises(RuntimeError,match='completed timestep'):driver.restart_state()
    with pytest.raises(ValueError,match='invalid physics restart'):driver.restore_restart_state({})
    driver.finish_step(pools=None)
    payload=driver.restart_state();driver.restore_restart_state(payload);driver.finalize(None)
    config=resolve_namelist(deck('none',config_microp_scheme='mp_wsm6'))
    registry=SlotRegistry();registry.register('microphysics','mp_wsm6','native-cpu',lambda c:None)
    seen=RecordingHooks();driver=PhysicsDriver(config,PhysicsManager(config,1),registry,seen)
    driver.initialize(None);payload=driver.restart_state()
    old=driver.manager.restart_state()
    def fail(payload): raise RuntimeError('transaction rejected')
    seen.restore_restart_state=fail
    with pytest.raises(RuntimeError,match='transaction rejected'):driver.restore_restart_state(payload)
    assert driver.manager.restart_state()==old
    driver.finalize(None);assert seen.operations[-1]=='atmphys_finalize'


@pytest.mark.parametrize('scheme',['sf_noah','sf_noahmp'])
def test_optional_manager_surface_updates_and_land_order(scheme):
    config=resolve_namelist(deck('none',config_lsm_scheme=scheme,
        config_sfclayer_scheme='sf_monin_obukhov_rev',config_sst_update=True,
        config_sstdiurn_update=True,config_deepsoiltemp_update=True,config_oml1d=True))
    registry=SlotRegistry();seen=RecordingHooks()
    for slot in config.selectors:
        if config.active(slot):registry.register(slot,config.selectors[slot],'native-cpu',lambda c:None)
    driver=PhysicsDriver(config,PhysicsManager(config,1,streams={'surface':'01:00:00'}),registry,seen)
    driver.initialize(None);seen.operations.clear();driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    assert seen.operations[:5]==['physics_timetracker_state','physics_update_surface',
        'physics_update_sst','physics_update_sstskin','physics_update_deepsoiltemp']
    assert 'driver_oml1d' not in seen.operations
    assert any(call['slot']=='oml' for call in driver.calls)
    assert ('allocate_lsm' in seen.operations)==(scheme=='sf_noah')
    driver.finish_step(pools=None)


def test_native_suite_door_route_preserves_existing_default():
    from hexcore.mpasphys import route_namelist
    from hexcore.physics_suites import suite_names
    assert route_namelist('&physics\n/') is None
    assert route_namelist(deck(suite_names()[0])) is None
    for suite in ('none','mesoscale_reference','convection_permitting'):
        assert route_namelist(deck(suite)).suite==suite
    with pytest.raises(ValueError,match='explicit MPAS suite'):
        route_namelist('&hex_mpasphys\nconfig_pbl_backend="off"\n/')


@pytest.mark.parametrize('lw,sw', [('cam_lw','off'), ('off','cam_sw'), ('cam_lw','cam_sw')])
@pytest.mark.parametrize('streams', [
    {'output':'01:00:00','diagnostics':'00:15:00'},
    {'output':'00:15:00','diagnostics':'01:00:00'},
    {'diagnostics':'00:15:00'},
])
def test_cam_surface_and_diagnostic_flags_match_native_clock(lw,sw,streams):
    """manager.F:549-644,263-375, including diagnostic reset without lookahead."""
    folder=os.environ.get('MPAS_PHYSICS_REFERENCE_DIR')
    if not folder: pytest.skip('set MPAS_PHYSICS_REFERENCE_DIR')
    lib=ctypes.CDLL(str(Path(folder)/'libclock.so'))
    lib.schedule.argtypes=[ctypes.c_int64,ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,
                           ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    streams={'surface':'00:20:00','restart':'01:00:00',**streams}
    config=resolve_namelist(deck('none',config_radt_lw_scheme=lw,config_radt_sw_scheme=sw,
        config_sst_update=True,config_greeness_update='00:20:00',
        config_camrad_abs_update='00:30:00',config_bucket_update='00:40:00'))
    manager=PhysicsManager(config,700,streams=streams)
    times=np.arange(0,9000,700,dtype=np.int64)
    before=manager.restart_state()
    records=[asdict(manager.advance(int(t))) for t in times]
    for name in ('greeness','surface','cam_abs','cam_restart','radiation_bucket','diagnostics'):
        got=[int(r[name]) for r in records]
        if name not in before['alarms']:
            assert not any(got)
            continue
        alarm=before['alarms'][name]
        period,start=int(alarm['period']),int(alarm['next_ring'])
        ringing=np.zeros(len(times),dtype=np.int32)
        nexts=np.zeros(len(times),dtype=np.int64)
        lookahead=700 if name in ('cam_restart','radiation_bucket') else 0
        lib.schedule(period,start,len(times),times.ctypes.data,lookahead,1,
                     ringing.ctypes.data,nexts.ctypes.data)
        expected=((nexts <= times+700).astype(np.int32) if name=='diagnostics' else ringing)
        assert got==expected.tolist(), name
        assert manager.alarms[name].next_ring==int(nexts[-1])
    assert all(not r['rain_bucket'] for r in records)
    restored=PhysicsManager(config,700,streams=streams)
    restored.restore_restart_state(manager.restart_state())
    assert restored.advance(9100)==manager.advance(9100)


def test_disabled_cloud_and_surface_backends_preserve_native_driver_gates():
    config=resolve_namelist(deck('none',config_radt_lw_scheme='rrtmg_lw',
        config_pbl_scheme='bl_mynn',config_bucket_update='01:00:00',config_bucket_radt=100.),
        backends={'cloudiness':'off','surface_layer':'off'})
    registry=SlotRegistry();seen=RecordingHooks()
    for slot in config.selectors:
        if config.active(slot):
            registry.register(slot,config.selectors[slot],'native-cpu',
                              lambda call:seen.operations.append(call.slot))
    driver=PhysicsDriver(config,PhysicsManager(config,700),registry,seen)
    driver.initialize(None);seen.operations.clear()
    driver.begin_step(itimestep=1,time_seconds=0,pools=None)
    assert 'allocate_cloudiness' not in seen.operations
    assert 'deallocate_cloudiness' in seen.operations
    assert 'radiation_lw' in seen.operations
    assert 'update_radiation_diagnostics' in seen.operations
    assert 'allocate_pbl' not in seen.operations
    assert 'pbl' not in seen.operations
    driver.finish_step(pools=None)

@pytest.mark.parametrize('fail',[False,True])
def test_ngw_mixed_native_parent_masks_actual_config_and_restores_on_error(fail):
    config=resolve_namelist(deck('mesoscale_reference',config_ngw_scheme=True,config_gwdo_scheme='bl_ugwp_gwdo'),backends={'ngw':'hex-cuda'})
    actual={};observed=[]
    adapter=hooks(['driver_gwdo']);adapter.native.config=lambda name,value:actual.__setitem__(name,value)
    def invoke(*args,**kwargs):
        observed.append(actual['config_ngw_scheme'])
        if fail:raise RuntimeError('native parent failed')
    adapter.native.call=invoke;adapter.configure(config)
    options={**config.execution_options,'config_ngw_scheme':False}
    call=SlotCall('gwdo','bl_ugwp_gwdo',Backend.NATIVE_CPU,1,0,360,StepFlags(),None,options)
    if fail:
        with pytest.raises(RuntimeError,match='parent failed'):adapter.invoke('driver_gwdo',call)
    else:adapter.invoke('driver_gwdo',call)
    assert observed==[False] and actual['config_ngw_scheme'] is True


def test_ngw_all_native_uses_parent_registration_and_mixed_needs_snapshot():
    config=resolve_namelist(deck('none',config_ngw_scheme=True,
        config_gwdo_scheme='bl_ugwp_gwdo',config_sfclayer_scheme='sf_monin_obukhov_rev'))
    entries={'driver_gwdo','driver_sfclayer'}
    adapter=hooks(entries);registry=SlotRegistry();adapter.register_slots(registry,config)
    assert not any(slot=='ngw' for slot,_,_ in registry._callbacks)
    registry.require(config)
    assert 'snapshot_gwdo_tendencies' not in adapter.missing_operations(config)
    mixed=resolve_namelist(deck('mesoscale_reference',config_ngw_scheme=True,config_gwdo_scheme='bl_ugwp_gwdo'),backends={'ngw':'hex-cuda'})
    assert 'snapshot_gwdo_tendencies' in adapter.missing_operations(mixed)
