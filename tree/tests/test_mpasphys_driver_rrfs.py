"""Fork physics driver order, restart and routing checks; no forecast identity is inferred from these."""
from dataclasses import asdict
from itertools import product
import json

import pytest

from hexcore.mpasphys.rrfs.driver_rrfs import (
    Backend, PhysicsDriver, PhysicsManager, SlotRegistry, package_selection,
    resolve_namelist, route_namelist, scalar_registry, specification,
)


def deck(suite='none', **options):
    def value(v):
        return '.true.' if v is True else '.false.' if v is False else repr(v)
    return '&physics\n' + '\n'.join(f'{k}={value(v)}' for k,v in
            {'config_physics_suite': suite, **options}.items()) + '\n/'


class TraceHooks:
    """Operation trace only; the unchanged native driver is the branch referee."""
    def invoke(self,operation,call):
        return operation
    def restart_state(self):
        return {'sentinel': [7,11]}
    def restore_restart_state(self,payload):
        assert payload == self.restart_state()


def driver(config, dt=60, streams=None):
    registry = SlotRegistry()
    for slot,scheme in config.selectors.items():
        if config.active(slot) and not config.native_embedded(slot):
            registry.register(slot,scheme,config.backends[slot],lambda call: None)
    instance = PhysicsDriver(config,PhysicsManager(config,dt,streams=streams),registry,TraceHooks())
    instance.initialize({})
    return instance


def test_tendency_is_held_reset_follows_output_and_restart_keeps_phase():
    instance = driver(resolve_namelist(deck('hrrrv5')))
    instance.begin_step(itimestep=1,time_seconds=0,pools={})
    first = instance.physics_get_tend(rk_step=1,dynamics_substep=1,pools={})
    assert first == instance.physics_get_tend(rk_step=3,dynamics_substep=2,pools={})
    assert sum(c['operation']=='physics_get_tend' for c in instance.calls)==1
    with pytest.raises(RuntimeError,match='completed timestep'):
        instance.after_output(pools={},diagnostics_alarm=True,itimestep=1,time_seconds=60)
    instance.finish_step(pools={})
    instance.after_output(pools={},diagnostics_alarm=False,itimestep=1,time_seconds=60)
    assert instance.calls[-1]['operation']=='microphysics'
    instance.after_output(pools={},diagnostics_alarm=True,itimestep=1,time_seconds=60)
    assert instance.calls[-1]['operation']=='atm_reset_diagnostics'
    state=json.loads(json.dumps(instance.restart_state()))
    restored=driver(instance.config)
    restored.restore_restart_state(state)
    assert restored.restart_state()==state
    assert asdict(restored.manager.advance(60))==asdict(instance.manager.advance(60))


@pytest.mark.parametrize('aerosol,hail',list(product([False,True],repeat=2)))
def test_tempo_registry_flags_follow_native_packages(aerosol,hail):
    c=resolve_namelist(deck('hrrrv5')+f"\n&physics_mp_tempo config_tempo_aerosolaware={'.true.' if aerosol else '.false.'}, config_tempo_hailaware={'.true.' if hail else '.false.'} /")
    names={r['name'] for r in scalar_registry(c)}
    assert {'nc','nifa','nwfa'}.issubset(names)==aerosol
    assert {'ng','volg'}.issubset(names)==hail
    assert 'qb' not in names
    assert package_selection(c)['mp_tempo_in']


def test_the_door_runs_every_ported_slot_on_the_gpu():
    # Breakage prevented: the RRFS stage ran all of its physics native-cpu although
    # the fork's GPU ports were registered (table cuda_entries), because the row's
    # default backend applied to every slot.
    from hexcore.mpasphys.driver_v842 import gpu_port_absence
    c=route_namelist(deck('hrrrv5'))
    for slot,scheme in c.selectors.items():
        if scheme=='off':
            continue
        ported=gpu_port_absence('mpasphys_rrfs',slot,scheme) is None
        assert str(c.backends[slot])==('hex-cuda' if ported else 'native-cpu'),(slot,scheme)
    assert {s for s,b in c.backends.items() if str(b)=='hex-cuda'}>={
        'microphysics','pbl','surface_layer','land_surface','seaice','radiation_lw','radiation_sw','cloudiness'}
    # Library callers keep the explicit all-native default; a namelist selection still wins.
    assert {str(b) for b in resolve_namelist(deck('hrrrv5')).backends.values()}=={'native-cpu'}
    c=route_namelist(deck('hrrrv5')+"\n&hex_mpasphys config_microp_backend='native-cpu' /")
    assert str(c.backends['microphysics'])=='native-cpu' and str(c.backends['pbl'])=='hex-cuda'


def test_routes_preserve_existing_unqualified_suites():
    assert route_namelist(deck('hrrrv5')).suite=='hrrrv5'
    assert route_namelist(deck('mesoscale_reference')) is None
    assert route_namelist(deck('convection_permitting')) is None
    c=route_namelist(deck('rrfs_mesoscale_reference'))
    assert c.options['config_physics_suite']=='mesoscale_reference'
    assert c.receipt()['dycore_path']=='v8.3.1-noaa'


def test_ngw_native_call_is_embedded_and_source_gated():
    c=resolve_namelist(deck('hrrrv5',config_ngw_scheme=True))
    assert c.native_embedded('ngw')
    d=driver(c)
    d.begin_step(itimestep=1,time_seconds=0,pools={})
    assert 'ngw' not in [r['operation'] for r in d.calls]
    assert not resolve_namelist(deck('none',config_ngw_scheme=True)).active('ngw')


@pytest.mark.parametrize('parent,child',[('native-cpu','hex-cuda'),('hex-cuda','native-cpu'),
                                         ('hex-cuda','hex-cuda')])
def test_mixed_ngw_retains_native_allocation_until_both_callbacks(parent,child):
    config=resolve_namelist(deck('hrrrv5',config_ngw_scheme=True),
                            backends={'gwdo':parent,'ngw':child})
    assert config.execution_options['config_ngw_scheme'] is True
    instance=driver(config);instance.begin_step(itimestep=1,time_seconds=0,pools={})
    operations=[c['operation'] for c in instance.calls]
    start=operations.index('allocate_gwdo')
    assert operations[start:start+5]==['allocate_gwdo','snapshot_gwdo_tendencies',
                                      'gwdo','ngw','deallocate_gwdo']
