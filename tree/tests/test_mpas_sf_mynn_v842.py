"""Completeness and association audit against the pinned MPAS source.

Numerical native/CUDA replay lives in tests/mpas_sf_mynn/compare*.py.
This test ensures the installed CUDA transcription has not drifted from
the complete source transcribed and validated by those replay programs.
"""
import site_support
import importlib.util
import os
from pathlib import Path
import pytest
from hexcore.mpasphys import sf_mynn_v842 as port


def test_complete_pinned_module_transcription():
    ref=site_support.mpas_source(preferred=('MPAS_V842_REF',))
    source=ref/'src/core_atmosphere/physics/physics_mmm/sf_mynn.F90'
    if not source.exists():pytest.skip('pinned MPAS v8.4.2 source not installed')
    script=Path(__file__).parent/'mpas_sf_mynn/transcribe.py'
    spec=importlib.util.spec_from_file_location('sf_mynn_transcription_audit',script)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.parse(source.read_text())
    arithmetic=[r for n,r in audit.ROUTINES.items() if n not in ('sf_mynn_init','sf_mynn_finalize')]
    assert len(arithmetic)==35
    for routine in arithmetic:
        assert routine.emit() in port.SOURCE, routine.name
    run=audit.ROUTINES['sf_mynn_run']
    assert port.FIELDS==tuple(a for a in run.args if a in run.arrays)
    assert port.OUTPUTS==tuple(a for a in run.args if a in run.arrays and run.intent[a]!='in')
    assert port.OPTIONAL==tuple(run.optional)


def test_finalize_native_status():
    assert port.sf_mynn_finalize()==(' ',0)


def test_rp3_exact_source_and_correct_declaration():
    """RP3 closes this slot's gap without changing its exact arithmetic text."""
    import hashlib
    from hexcore.mpasphys.driver_v842 import (Backend, resolve_namelist,
        settle_corrections, table_declarations)
    assert hashlib.sha256(port._cuda_source().encode()).hexdigest() == (
        '1cb5e0f931bf2b384aa3a11f87c379aecb271202dd9576e1ef5a7d7dd96a139b')
    assert port._driver_source({'unrelated'}) == port._driver_source()
    assert 'float recovered=' in port._driver_source({'RP3'})
    assert table_declarations('surface_layer','sf_mynn',Backend.HEX_CUDA) == {'RP3'}
    config = resolve_namelist("&physics config_physics_suite='convection_permitting' /",
        backends={'surface_layer':'hex-cuda'}, physics_mode='correct')
    applied, gaps = settle_corrections(config, {'RP3'})
    assert not any(g.slot=='surface_layer' for g in gaps)
    assert 'RP3' not in applied  # The native-cpu seaice slot still owns its gap.
    applied, gaps = settle_corrections(config, {'RP3'}, declarations=lambda *args:{'RP3'})
    assert applied == {'RP3'} and not gaps


def test_registered_native_selector_and_slot_backend():
    from hexcore.mpasphys.driver_v842 import Backend, SlotRegistry, route_namelist
    config=route_namelist("""&physics
 config_physics_suite='none', config_sfclayer_scheme='sf_mynn'
/
&hex_mpasphys
 config_sfclayer_backend='hex-cuda'
/
""")
    assert config.selectors['surface_layer']=='sf_mynn'
    assert config.backends['surface_layer']==Backend.HEX_CUDA
    registry=SlotRegistry()
    port.register_backend(registry)
    assert registry.declared('surface_layer','sf_mynn',Backend.HEX_CUDA) == {'RP3'}
    registry.require(config)


def test_launcher_reads_native_prepared_surface_pressure(monkeypatch):
    """Native sfc_input has no psfc; interface.F:521 owns prepared pressure."""
    from types import SimpleNamespace
    pressure, skin, diagnostics = object(), object(), {}
    surface = {'skintemp': skin, 'xland': object(), 'snowh': object()}
    pools = dict(mesh={}, diag_physics=diagnostics, sfc_input=surface,
                 atm_input={}, constants={}, prepared={'psfc_p': pressure},
                 configs={'config_len_disp': 1., 'config_do_restart': False},
                 scalar_state={'xice_threshold': .01})
    call = SimpleNamespace(slot='surface_layer', scheme='sf_mynn',
                           backend='hex-cuda', pools=pools, itimestep=1,
                           options={'config_frac_seaice': True}, corrections=frozenset({'RP3'}))

    def check_driver(mesh, diag, bound_surface, profiles, constants, **options):
        assert bound_surface['psfc'] is pressure
        assert bound_surface['skintemp'] is skin
        assert diag is diagnostics
        assert options['config_frac_seaice'] is True
        assert options['corrections'] == {'RP3'}
        return diag

    monkeypatch.setattr(port, 'driver', check_driver)
    assert port.launch_slot(call) is diagnostics
    assert 'psfc' not in surface


def test_whole_forecast_scope_rejects_new_omissions_and_word_differences(tmp_path):
    """Recorded omissions cannot hide changed words or reduced native coverage."""
    from copy import deepcopy
    import hashlib
    script = Path(__file__).parent / 'mpas_sf_mynn/check_forecast_scope.py'
    spec = importlib.util.spec_from_file_location('mynn_forecast_scope', script)
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)
    library = tmp_path / 'provider'
    library.write_bytes(b'provider-identity-fixture')
    deck = dict(pin_set='pins', mesh='mesh', date='date', suite='suite',
                process_count=1, duration_steps=72, native={'sha256':'native'},
                expected_samples={'output':[0,72]},
                hex_command=['--native-physics-library',str(library)])
    samples = [dict(stream='output',field=field,step=step,time=str(step),
                    words=10,missing_allowed=missing)
               for field,step,missing in [('pressure',0,False),('qv',72,False),
                                          ('deferred',72,True)]]
    inventory = {**{k:deck[k] for k in ('pin_set','mesh','date','suite',
                                       'process_count','duration_steps')},
                 'native_manifest_sha256':'native','samples':samples}
    completion = dict(completed=True,steps_completed=72,steps_requested=72,
                      native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                      physics=dict(suite='suite',selectors={'surface_layer':'sf_mynn'},
                                   backends={'surface_layer':'hex-cuda','pbl':'native-cpu'}))
    receipt = {**{k:deck[k] for k in ('pin_set','mesh','date','process_count')},
               'model_execution':True,'evidence_kind':'FORECAST',
               'provenance':{'deck_sha256':'deck','native_manifest_sha256':'native'},
               'passed':False,'tier':None,
               'comparison':{'fields':[{**{k:s[k] for k in ('stream','field','step','time')},
                    'compared_words':10,'differing_words':10 if s['missing_allowed'] else 0,
                    'status':'field missing in hex' if s['missing_allowed'] else 'compared'}
                    for s in samples]},
               'failures':['output/deferred step 72: field missing in hex']}
    accepted = scope.check(receipt,completion,deck,inventory,'deck')
    assert accepted['passed'] and accepted['computed_words'] == 20
    assert accepted['missing_words'] == 10 and not accepted['full_field_bitwise']
    mutations = []
    altered = deepcopy(receipt); altered['comparison']['fields'][0]['differing_words']=1
    mutations.append(altered)
    altered = deepcopy(receipt); altered['comparison']['fields'][0]['status']='field missing in hex'
    mutations.append(altered)
    altered = deepcopy(receipt); altered['comparison']['fields'][2]['compared_words']=11
    mutations.append(altered)
    altered = deepcopy(receipt); altered['comparison']['fields'].pop(0)
    mutations.append(altered)
    altered = deepcopy(receipt); altered['comparison']['fields'].append(deepcopy(altered['comparison']['fields'][0]))
    mutations.append(altered)
    for altered in mutations:
        with pytest.raises(ValueError):
            scope.check(altered,completion,deck,inventory,'deck')
    incomplete = {**completion,'steps_completed':71}
    with pytest.raises(ValueError,match='not completed'):
        scope.check(receipt,incomplete,deck,inventory,'deck')


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get('MYNN_SUBNORMAL_GPU') != '1',
                    reason='GPU check must run through the GPU queue')
def test_native_seaice_blend_preserves_subnormals():
    """MPAS driver_sfclayer.F:672-747: half-weight flux retains tiny words."""
    import cupy as cp
    import numpy as np
    inputs=np.array([2,4,8,0x007ffffe],dtype=np.uint32)
    expected=np.array([1,2,4,0x003fffff],dtype=np.uint32)
    host=np.zeros((len(port.FIELDS),len(inputs)),dtype=np.float32)
    index=port.FIELDS.index('flhc')
    host[index]=inputs.view(np.float32)
    packed=cp.asarray(host)
    sea=cp.asarray(np.zeros_like(host))
    ice=cp.asarray(np.full(len(inputs),.5,dtype=np.float32))
    hold=cp.asarray(np.zeros(len(inputs),dtype=np.float32))
    port._module().get_function('sf_mynn_to_mpas')((1,),(64,),
        (packed,sea,ice,hold,np.float32(.15),np.int32(len(inputs))))
    actual=cp.asnumpy(packed)[index].view(np.uint32)
    np.testing.assert_array_equal(actual,expected)
