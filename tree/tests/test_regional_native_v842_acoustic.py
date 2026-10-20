"""Full strict-native acoustic routine replay on resident regional inputs."""
import json
import os
from pathlib import Path

import pytest


def test_whole_real_acoustic_replay():
    raw = os.environ.get('MPAS_REGIONAL_ACOUSTIC_RECEIPT')
    if not raw:
        pytest.skip('Set MPAS_REGIONAL_ACOUSTIC_RECEIPT to the whole acoustic replay artifact')
    receipt = json.loads(Path(raw).read_text())
    assert receipt['forecast_proof'] is False
    assert receipt['process_count'] == 1
    assert receipt['source_pins']['routine'] == 'atm_advance_acoustic_step_work'
    assert len(receipt['records']) == 24
    assert {row['small_step'] for row in receipt['records']} == {1, 2, 3, 4}
    for row in receipt['records']:
        assert {field['field'] for field in row['fields']} == {
            'ru_p', 'ruAvg', 'rw_p', 'wwAvg', 'rho_pp', 'rtheta_pp', 'rtheta_pp_old'}
        for field in row['fields']:
            assert field['words'] > 0
            assert field['differences'] == 0, (row['call'], field)
            assert field['native_sha256'] == field['cuda_sha256']


def test_whole_acoustic_signed_subnormal_replay():
    raw = os.environ.get('MPAS_REGIONAL_ACOUSTIC_SYNTHETIC_RECEIPT')
    if not raw:
        pytest.skip('Set MPAS_REGIONAL_ACOUSTIC_SYNTHETIC_RECEIPT to the synthetic replay artifact')
    receipt = json.loads(Path(raw).read_text())
    assert receipt['forecast_proof'] is False
    assert len(receipt['records']) == 24
    assert sum(field['native_nonzero_subnormals']
               for row in receipt['records'][12:] for field in row['fields']) > 0
    for row in receipt['records']:
        for field in row['fields']:
            assert field['differences'] == 0, (row['call'], field)
            assert field['cuda_sha256'] == field['native_sha256']
    from hexcore.cuda_regional_v841 import MODULE_KEY
    observations = receipt['compile_manifest']['modules'][MODULE_KEY]['effective_compile']['observations']
    assert observations
    for observation in observations:
        flags = observation['effective_flags']
        assert '--fmad=false' in flags and '--ftz=false' in flags
        assert '--ftz=true' not in flags


@pytest.mark.parametrize(('environment','routine','calls'), [
    ('MPAS_REGIONAL_SCALAR_UNLIMITED_RECEIPT','atm_advance_scalars_work',240),
    ('MPAS_REGIONAL_SCALAR_LIMITED_RECEIPT','atm_advance_scalars_mono_work',120),
])
def test_whole_real_scalar_transport_and_subnormals(environment,routine,calls):
    raw=os.environ.get(environment)
    if not raw:
        pytest.skip('Set '+environment+' to the complete 24h native scalar replay')
    receipt=json.loads(Path(raw).read_text())
    assert receipt['forecast_proof'] is False and receipt['process_count']==1
    assert receipt['source_pins']['routine']==routine
    assert len(receipt['records'])==calls
    assert all(row['words']>0 and row['differences']==0 for row in receipt['records'])
    assert all(row.get('baseline_differences',0)==0 for row in receipt['records'])
    assert any(row['source_nonzero_words'] for row in receipt['records'])
    assert sum(row['native_nonzero_subnormals'] for row in receipt['records'])>0
    for module in receipt['compile_manifest']['modules'].values():
        observations=module['effective_compile']['observations']
        assert observations
        for observation in observations:
            flags=observation['effective_flags']
            assert '--fmad=false' in flags and '--ftz=false' in flags
            assert '--ftz=true' not in flags


def test_whole_scalar_signed_subnormal_inputs():
    raw=os.environ.get('MPAS_REGIONAL_SCALAR_SYNTHETIC_RECEIPT')
    if not raw:
        pytest.skip('Set MPAS_REGIONAL_SCALAR_SYNTHETIC_RECEIPT to the whole tiny-input replay')
    receipt=json.loads(Path(raw).read_text())
    assert receipt['forecast_proof'] is False and receipt['process_count']==1
    assert receipt['signed_subnormal_input_cases']==6
    assert len(receipt['records'])==12
    assert {row['rk_step'] for row in receipt['records']}=={1,2,3}
    assert all(row['words']>0 and row['differences']==0 for row in receipt['records'])
    assert sum(row['native_nonzero_subnormals'] for row in receipt['records'][1::2])>0
    for module in receipt['compile_manifest']['modules'].values():
        assert module['effective_compile']['observations']
        for observation in module['effective_compile']['observations']:
            assert '--ftz=false' in observation['effective_flags']
            assert '--ftz=true' not in observation['effective_flags']
