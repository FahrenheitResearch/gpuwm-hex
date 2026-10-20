"""Landing evidence must reject a missing case, changed word or stale module."""
import importlib.util
import json
from pathlib import Path

import pytest

HELPERS = Path(__file__).with_name('radshared_v842')


@pytest.fixture
def seal(monkeypatch):
    monkeypatch.syspath_prepend(str(HELPERS))
    spec = importlib.util.spec_from_file_location('p21_landing_seal', HELPERS / 'seal_landing_replay.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path):
    cloud = dict(module_sha256='science', interfaces=['cal_cldfra3', 'calc_cldfraction', 'calc_cldincidence'],
                 all_selected_passed=True, differing_words=0, calls=11, total_words=100)
    driver = dict(source_sha256='science', captures=[{}, {}],
                  all_outputs_bitwise_equal=True, differing_words=0, words=20)
    names = {'ozone_initialization', 'pressure_table', 'ozone_CAM_time',
             'ozone_monthly_date', 'ozone_vertical', 'radiation_controls'}
    names.update('radius_' + d + '_' + p + n for d, p in (('from', 're_'), ('to', 'rre_'))
                 for n in ('cloud', 'ice', 'snow'))
    support = dict(artifacts={'/checkout/radshared_v842.py': 'science'},
                   all_outputs_bitwise_equal=True, differing_words=0, words=40,
                   captures=[dict(arrays={name: dict(comparison_complete=True, bitwise_equal=True,
                       different_words=0) for name in names}) for _ in range(2)])
    synthetic = dict(artifact_sha256={'/checkout/radshared_v842.py': 'science'}, differing_words=0,
                     subnormal_negative_control=dict(expected_failure_detected=True))
    for name, value in (('cloud-callees/matrix.json', cloud), ('cloud-driver.json', driver),
                        ('tapped-support.json', support), ('synthetic-subnormal.json', synthetic)):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(value))
    return tmp_path


def test_complete_stated_real_case_set(seal, evidence):
    result = seal.qualify(evidence, 'science')
    assert result['real_words'] == 160 and result['differing_words'] == 0
    assert not result['all_forecast_calls_claimed']


@pytest.mark.parametrize('fault', ('scheme', 'call', 'case', 'word', 'array', 'comparison',
                                  'module', 'support_module', 'ftz'))
def test_faults_cannot_seal_a_landing_replay(seal, evidence, fault):
    cloud_path = evidence / 'cloud-callees/matrix.json'
    support_path = evidence / 'tapped-support.json'
    synthetic_path = evidence / 'synthetic-subnormal.json'
    cloud = json.loads(cloud_path.read_text())
    support = json.loads(support_path.read_text())
    synthetic = json.loads(synthetic_path.read_text())
    if fault == 'scheme': cloud['interfaces'].pop()
    elif fault == 'call': cloud['calls'] -= 1
    elif fault == 'case': support['captures'].pop()
    elif fault == 'word': support['captures'][0]['arrays']['ozone_CAM_time']['different_words'] = 1
    elif fault == 'array': support['captures'][0]['arrays'].pop('ozone_initialization')
    elif fault == 'comparison': support['captures'][0]['arrays']['ozone_CAM_time']['comparison_complete'] = False
    elif fault == 'module': cloud['module_sha256'] = 'old science'
    elif fault == 'support_module': support['artifacts']['/checkout/radshared_v842.py'] = 'old science'
    elif fault == 'ftz': synthetic['subnormal_negative_control']['expected_failure_detected'] = False
    cloud_path.write_text(json.dumps(cloud))
    support_path.write_text(json.dumps(support))
    synthetic_path.write_text(json.dumps(synthetic))
    with pytest.raises(ValueError):
        seal.qualify(evidence, 'science')
