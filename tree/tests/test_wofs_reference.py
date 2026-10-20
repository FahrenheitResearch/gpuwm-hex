"""References must exclude absent coverage and preserve no-echo pixels."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('reference',Path(__file__).parents[1]/'verification/wofs-ensemble/reference.py')
reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)

def test_reference_coverage_no_echo_and_boundary():
    lon,lat=np.meshgrid(np.arange(4.),np.full(3,42.))
    observed=np.full((3,4),-99.); observed[:,0]=np.nan; observed[:,1]=45
    row=reference.summarize(observed,lat,lon)
    assert row['missing_pixels']==3 and row['valid_pixels']==9
    assert row['no_event_brier']==pytest.approx(1/3)
    assert row['same_window_frequency_brier']==pytest.approx(2/9)
    assert row['largest_object_bounds']['touches_score_boundary']
    observed[:,1]=-99
    row=reference.summarize(observed,lat,lon)
    assert row['no_event_brier']==0 and row['largest_object_bounds'] is None
