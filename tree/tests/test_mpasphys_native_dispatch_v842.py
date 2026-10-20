"""Actual f3 native driver called through f4's slot registry; column evidence only."""
import importlib.util
import ctypes as ct
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.mpasphys import Backend, NativeCalendarState, NativeDriverHooks, NativePoolArguments, SlotCall, SlotRegistry, StepFlags, resolve_namelist


@pytest.fixture
def native_class():
    source = os.environ.get('MPAS_NATIVE_LOADER')
    if not source:
        pytest.skip('set MPAS_NATIVE_LOADER until f3 is merged forward')
    spec = importlib.util.spec_from_file_location('f3_native_loader', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativePhysics


def test_session_restores_strict_native_optional_pool_policy(native_class):
    from hexcore.mpasphys.session_v842 import NativePhysicsSession
    with native_class(os.environ['MPAS_NATIVE_LIBRARY'], test_only=True) as native:
        getter = getattr(native.lib, '__mpas_pool_routines_MOD_mpas_pool_get_error_level')
        getter.argtypes = []
        getter.restype = ct.c_int
        seen = []
        session = NativePhysicsSession.__new__(NativePhysicsSession)
        session.native = native
        session.driver = SimpleNamespace(initialize=lambda context: seen.append(getter()))
        session._initialized = False
        session.initialize()
        assert seen == [1001] and getter() == 1001 and session._initialized


@pytest.mark.parametrize('scheme',['cld_incidence','cld_fraction','cld_fraction_thompson'])
@pytest.mark.parametrize('flags',[(True,True),(True,False),(False,False)])
def test_registered_slot_matches_separate_native_driver(native_class,scheme,flags):
    library=Path(os.environ['MPAS_NATIVE_LIBRARY'])
    probe=library.with_name('probe_cloudiness')
    direct=subprocess.check_output([str(probe),scheme,*[str(int(f)) for f in flags]],text=True)
    words=np.asarray([int(w,16) for w in direct.splitlines()],dtype=np.uint32)
    config=resolve_namelist('&physics\nconfig_physics_suite="none"\n'
                           f'config_radt_lw_scheme="rrtmg_lw"\nconfig_radt_cld_scheme="{scheme}"\n/')
    with native_class(library,test_only=True) as native:
        nc,nz=6,3
        for prefix,size in [('i',nc),('j',1),('k',nz)]:
            for scope in ['d','m','t']:
                native.put_variable(prefix+scope+'s',np.asarray(1,dtype=np.int32))
                native.put_variable(prefix+scope+'e',np.asarray(size,dtype=np.int32))
        native.put_variable('f_qc',np.asarray(flags[0]))
        native.put_variable('f_qi',np.asarray(flags[1]))
        native.config('config_radt_cld_scheme',scheme)
        native.config('config_len_disp',480000.)
        native.put_field('mesh','areaCell',np.full(nc,1e10,dtype=np.float32))
        native.put_field('mesh','meshDensity',np.ones(nc,dtype=np.float32))
        native.put_field('sfc_input','xland',np.asarray([2,1,2,1,2,1],dtype=np.float32))
        native.put_field('diag_physics','cldfrac',np.full((nz,nc),-999.,dtype=np.float32))
        for name,values in [
                ('t_p',[290,260,240,280,275,255]),('qv_p',[.015,.001,.00001,.000001,.010,.002]),
                ('qc_p',[.00001,.000001,.0000005,0,.00002,.00003]),
                ('qi_p',[0,.00002,.0000002,0,.00001,.00005]),('qs_p',[0,0,.000001,0,0,.00001])]:
            data=np.empty((nc,nz,1),dtype=np.float32,order='F')
            data[:,:,0]=np.asarray(values,dtype=np.float32)[:,None]
            native.put_variable(name,data)
        for name,value in [('pres_p',80000.),('pres_hyd_p',80000.),('rho_p',1.)]:
            native.put_variable(name,np.full((nc,nz,1),value,dtype=np.float32))
        ledger=[]
        def synchronize(entry,call):
            ledger.append(entry)
        arguments=NativePoolArguments(native,pools={name:name for name in
            ['configs','mesh','diag_physics','sfc_input']},its=1,ite=5,synchronize=synchronize)
        def after(entry,call):
            if entry=='driver_cloudiness':
                return native.get_field('diag_physics','cldfrac')
        hooks=NativeDriverHooks(native,arguments,after_call=after,external={},
                               save_fields=lambda:{},restore_fields=lambda p:None)
        registry=SlotRegistry();hooks.register_slots(registry,config)
        missing=hooks.missing_operations(config)
        assert 'physics_timetracker_state' in missing
        assert 'physics_get_tend' not in missing
        assert 'post_transport_prepare' in missing
        with pytest.raises(RuntimeError,match='not bound'):
            hooks.preflight(config)
        call=SlotCall('cloudiness',scheme,Backend.NATIVE_CPU,1,0,360,
                      StepFlags(radiation_lw=True),None,config.execution_options)
        hooks.invoke('allocate_cloudiness',call)
        result=registry.run(call)
        hooks.invoke('deallocate_cloudiness',call)
        assert ledger == ['allocate_cloudiness','driver_cloudiness','deallocate_cloudiness']
        assert np.array_equal(result.ravel(order='F').view(np.uint32),words)
        assert np.all(result[:,5] == np.float32(-999.))
        with pytest.raises(RuntimeError,match='not bound'):
            hooks.invoke('physics_timetracker_state',call)


@pytest.mark.parametrize('start,next_day,julday,leap',[
    ('2024-02-28_23:59:59','2024-02-29',60,True),
    ('1900-02-28_23:59:59','1900-03-01',60,False),
    ('2000-02-28_23:59:59','2000-02-29',60,True),
    ('2023-12-31_23:59:59','2024-01-01',1,True)])
@pytest.mark.parametrize('dt',[1.,700.,.125])
def test_native_calendar_globals_flags_and_argument_binding(native_class,start,next_day,julday,leap,dt):
    config=resolve_namelist('&physics\nconfig_physics_suite="none"\n/')
    with native_class(os.environ['MPAS_NATIVE_LIBRARY'],test_only=True) as native:
        native.clock(start=start,stop='2100-01-01_00:00:00',dt=dt)
        arguments=NativePoolArguments(native,pools={name:name for name in
            ['configs','mesh','state','diag','diag_physics','sfc_input','tend','tend_physics']},
            its=1,ite=5,synchronize=lambda entry,call:None)
        # Framework handles belong to the loaded production library. A shim
        # from a different native build cannot read its derived-type storage.
        update=NativeCalendarState(native,native.lib,arguments)
        before=SlotCall('driver','driver',Backend.NATIVE_CPU,1,0,dt,StepFlags(),None,config.execution_options)
        first=update(before)
        assert first['current_date'].startswith(start)
        bits=(True,False,True,False,True,False,True,False)
        # Eight fractional steps cross midnight without rounding away dt.
        steps=8 if dt<1 else 1
        for step in range(steps):
            native.advance_clock()
        flags=StepFlags(radiation_lw=bits[0],radiation_sw=bits[1],convection=bits[2],
            cam_abs=bits[3],cam_restart=bits[4],rain_bucket=bits[5],radiation_bucket=bits[6],diagnostics=bits[7])
        call=SlotCall('driver','driver',Backend.NATIVE_CPU,2,dt*steps,dt,flags,None,config.execution_options)
        result=update(call)
        assert result['current_date'].startswith(next_day)
        assert result['leapyear'] == leap
        current=native.get_table('mpas_atmphys_manager.curr_julday')[0].item()
        # Strict source arithmetic, including its integer fractional-second division.
        minutes,seconds=(11,39) if dt==700 else (0,0)
        utc_h=np.float32(np.float32(minutes)/np.float32(60)+np.float32(seconds)/np.float32(3600))
        expected=np.float32(np.float32(julday-1)+utc_h/np.float32(24))
        assert np.float32(current).view(np.uint32) == expected.view(np.uint32)
        assert result['julian'] == current
        for name,bit in zip(['l_radtlw','l_radtsw','l_conv','doabsems','l_camlw','l_acrain','l_acradt','l_diags'],bits):
            assert bool(native.get_variable(name)[0].item()) == bit
        mapped=arguments('physics_update_deepsoiltemp',call)
        assert mapped['leapyear'] == leap
        assert mapped['julian_in'] == current
        assert mapped['dt'] == dt
        assert arguments('driver_microphysics',call)['time_lev'] == 2
        assert arguments('mpas_to_physics',call)['time_lev'] == 1
        assert arguments('physics_update_surface',call)['current_date'] == result['current_date']
