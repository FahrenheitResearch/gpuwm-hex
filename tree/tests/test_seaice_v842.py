"""Contract tests for the MPAS seaice slot; numerical acceptance uses the Fortran replay."""
from hexcore.mpasphys.seaice_v842 import options_from_MPAS, register_backend, launch_slot


def test_frac_seaice_preserves_native_threshold_and_fixed_controls():
    fractional=options_from_MPAS(120.,config_frac_seaice=True)
    binary=options_from_MPAS(120.,config_frac_seaice=False)
    assert fractional.xice_threshold == .02
    assert binary.xice_threshold == .5
    assert fractional.dt == binary.dt == 120.
    assert fractional.seaice_thickness_default == 3.
    assert fractional.seaice_snowdepth_min == .001
    assert fractional.seaice_snowdepth_max == 1.e10
    assert fractional.frpcpn is False


def test_both_noah_schemes_register_on_independent_seaice_slot():
    class Registry:
        def __init__(self): self.calls=[]
        def register(self,*args): self.calls.append(args)
    registry=Registry()
    register_backend(registry)
    assert registry.calls == [('seaice','sf_noah','hex-cuda',launch_slot),
                              ('seaice','sf_noahmp','hex-cuda',launch_slot)]


def test_written_forecast_proof_rejects_bit_changes_new_gaps_and_disappearing_fields():
    import copy
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parent / 'seaice_oracle/written_forecast_proof.py'
    spec = importlib.util.spec_from_file_location('seaice_written_proof', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    deck = dict(id='instrument', pin_set='pin', mesh='mesh', date='date',
                process_count=1, duration_steps=1, expected_samples={'stream': [0, 1]})
    model = dict(completed=True, steps_completed=1, steps_requested=1,
                 physics={'backends': {'seaice': 'hex-cuda', 'land_surface': 'native-cpu'}})
    rows = [dict(stream='stream', field=field, step=step, status=status,
                 compared_words=3, differing_words=0 if status == 'compared' else 3)
            for step in (0, 1) for field, status in
            [('written', 'compared'), ('deferred', 'field missing in hex')]]
    receipt = dict(deck='instrument', pin_set='pin', mesh='mesh', date='date',
                   process_count=1, model_execution=True, evidence_kind='FORECAST',
                   failed_model_exit=None, comparison={'fields': rows})
    baseline = {'entries': [dict(mesh='mesh', pin_set='pin', process_count=1,
                 stream='stream', field='deferred', statuses=['field missing in hex'],
                 missing_samples=2, differing_words=6)]}
    assert module.check(receipt, deck, model, baseline, rows) == []
    altered = copy.deepcopy(receipt)
    altered['comparison']['fields'][0]['differing_words'] = 1
    assert module.check(altered, deck, model, baseline, rows)
    altered = copy.deepcopy(receipt)
    altered['comparison']['fields'][1]['field'] = 'unrecorded'
    assert module.check(altered, deck, model, baseline, rows)
    altered = copy.deepcopy(receipt)
    altered['comparison']['fields'][0].update(status='field missing in hex', differing_words=3)
    assert module.check(altered, deck, model, baseline, rows)
    altered = copy.deepcopy(receipt)
    altered['comparison']['fields'].pop()
    assert module.check(altered, deck, model, baseline, rows)
    assert module.check(receipt, deck, {**model, 'steps_completed': 0}, baseline, rows)
    stored = [dict(file='stream.nc', stream='stream', exit_code=1, words=7,
                   differing_words=0, errors=['variable set differs: deferred'],
                   fields=[dict(field='XTime', netcdf_type=2, words=4, differing_words=0),
                           dict(field='written', netcdf_type=5, words=3, differing_words=0)])]
    assert module.check_stored(stored, ['stream.nc'], rows) == []
    altered = copy.deepcopy(stored)
    altered[0]['fields'][0]['differing_words'] = 1
    assert module.check_stored(altered, ['stream.nc'], rows)
    assert module.check_stored(stored, ['stream.nc', 'missing.nc'], rows)
    altered = copy.deepcopy(stored)
    altered[0]['errors'] = ['unsupported NetCDF type: XTime']
    assert module.check_stored(altered, ['stream.nc'], rows)


def test_native_table_routes_fractional_option_to_independent_cuda_slot():
    from hexcore.mpasphys.driver_v842 import Backend, resolve_namelist, SlotRegistry
    from hexcore.physics_suites import load_table

    table = load_table()['mpasphys_v842']
    group = table['backend_namelist_group']
    for scheme in ('sf_noah', 'sf_noahmp'):
        for fractional in (False, True):
            config = resolve_namelist(f'''&physics
config_physics_suite='none'
config_lsm_scheme='{scheme}'
config_sfclayer_scheme='sf_monin_obukhov'
config_frac_seaice={'.true.' if fractional else '.false.'}
/
&{group}
config_seaice_backend='hex-cuda'
/''')
            assert config.selectors['seaice'] == scheme
            assert config.backends['seaice'] == Backend.HEX_CUDA
            assert config.options['config_frac_seaice'] is fractional
            assert all(value == Backend.NATIVE_CPU for slot, value in config.backends.items()
                       if slot != 'seaice' and config.active(slot))
            descriptor = table['slots']['seaice']['cuda_entries'][scheme]
            assert descriptor['module'] == 'seaice_v842'
            assert descriptor['call'] == 'launch_slot'
    registry = SlotRegistry()
    register_backend(registry)

def test_native_host_closure_rejects_bits_lost_fields_and_changed_clocks(monkeypatch):
    from pathlib import Path
    import copy
    import pytest
    directory = Path(__file__).parent / 'seaice_oracle'
    monkeypatch.syspath_prepend(str(directory))
    from native_host_closure import require_complete_comparison
    field = dict(stream='output', step=1, time='native-clock', field='tsk',
                 status='compared', differing_words=0, compared_words=4)
    previous = dict(comparison=dict(fields=[field]))
    assert require_complete_comparison(dict(fields=[field]), previous) == 4
    for attribute, value in [('differing_words', 1), ('status', 'shape-mismatch'),
                             ('step', 2), ('time', 'other-clock'), ('field', 'other-field')]:
        changed = copy.deepcopy(field)
        changed[attribute] = value
        with pytest.raises(ValueError):
            require_complete_comparison(dict(fields=[changed]), previous)
    for rows in ([], [field, field]):
        with pytest.raises(ValueError, match='field/time inventory'):
            require_complete_comparison(dict(fields=rows), previous)
