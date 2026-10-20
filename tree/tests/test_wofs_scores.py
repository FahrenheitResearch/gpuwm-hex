"""Scalar score invariants, including missing coverage and object misses."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec = importlib.util.spec_from_file_location('wofs_score', Path(__file__).parents[1]/'verification/wofs-ensemble/score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)

def test_perfect_and_missing_echo_scores():
    lon, lat = np.meshgrid(np.arange(6)*.03-88,np.arange(6)*.03+42)
    observed = np.full((6,6), -99., dtype=np.float32)
    observed[1:4,1:4] = 45
    observed[0,0] = np.nan
    perfect = score.scores(observed,observed,lat,lon)
    assert perfect['csi_35'] == 1 and perfect['valid_pixels'] == 35
    assert all(x == 1 for x in perfect['fss'].values())
    assert perfect['position_error_km'] == pytest.approx(0,abs=.001)
    dry = score.scores(np.full_like(observed,-99),observed,lat,lon)
    assert dry['csi_35'] == 0 and dry['position_error_km'] is None
    assert dry['max_reflectivity_bias_dbz'] == -144

def test_probability_and_spread_are_member_based():
    lon, lat = np.meshgrid(np.arange(6)*.03-88,np.arange(6)*.03+42)
    observed = np.full((6,6),45,dtype=np.float32)
    models = np.stack([observed,np.full_like(observed,25)])
    result = score.ensemble_scores(models,observed,lat,lon)
    assert result['brier_score_35'] == pytest.approx(.25)
    assert result['weighted_mean_spread_dbz'] == pytest.approx(20/np.sqrt(2))
    assert result['reliability'][1]['observed_frequency'] == 1


def test_case_roster_preserves_distinct_sources_and_applies_only_declared_physics():
    import json
    case_path=Path(__file__).parents[1]/'verification/wofs-ensemble/case.json'
    case=json.loads(case_path.read_text())
    roster_spec=importlib.util.spec_from_file_location('roster',case_path.parent/'roster.py')
    roster=importlib.util.module_from_spec(roster_spec)
    roster_spec.loader.exec_module(roster)
    data={'members':[dict(id=name,source=name,init=name+'.nc') for name in case['members']]}
    result=roster.apply_roster(data,case)
    assert len(set(m['init'] for m in result['members'])) == 8
    assert all(m['forecast_flags']==[] for m in result['members'][:4])
    assert all(m['forecast_flags']==['--physics','config_microp_scheme=mp_thompson'] for m in result['members'][4:])
    with pytest.raises(ValueError,match='roster differs'):
        roster.apply_roster({'members':result['members'][:1]},case)
