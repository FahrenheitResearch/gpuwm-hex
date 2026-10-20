"""Validate the physical diagnostics used to compare long trajectories."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'tools/native_dry_reference/forecast_divergence.py'
spec=importlib.util.spec_from_file_location('forecast_divergence',path)
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_log_pressure_interpolation_has_analytic_height_solution():
    pressure=np.array([[100000.,80000.],[50000.,40000.],[25000.,20000.]])
    height=8000*np.log(100000/pressure)
    actual=analysis.at_pressure(pressure,height,50000)
    assert np.allclose(actual,8000*np.log(2),rtol=0,atol=1e-10)
    for invalid in (pressure[::-1],np.full_like(pressure,np.nan),pressure[:1]):
        with pytest.raises(ValueError):analysis.at_pressure(invalid,height[:len(invalid)],50000)
    with pytest.raises(ValueError):analysis.at_pressure(pressure,height,15000)


def test_vector_wind_and_cell_area_are_used_in_rms():
    zero={'z500':np.zeros(2),'u250':np.zeros(2),'v250':np.zeros(2)}
    # Both wind speeds are five. Their vector difference is still six.
    first={'z500':np.array([1.,3.]),'u250':np.array([3.,3.]),'v250':np.array([4.,4.])}
    second={'z500':np.zeros(2),'u250':np.array([-3.,-3.]),'v250':np.array([4.,4.])}
    result=analysis.rms_pair(first,second,np.array([3.,1.]))
    assert result['wind250_ms']==6
    assert result['z500_m']==pytest.approx(np.sqrt(3))


def test_empty_reconstruction_cannot_silently_report_zero_wind():
    mesh={'nEdgesOnCell':np.array([2]),'edgesOnCell':np.array([[0,1]]),
          'coeffs_reconstruct':np.zeros((1,2,3)), 'latCell':np.array([0.]),'lonCell':np.array([0.])}
    normal=np.array([[3.,4.]])
    with pytest.raises(ValueError,match='zero wind'):analysis.reconstruct(normal,mesh)
    mesh['coeffs_reconstruct'][0,0,1]=1
    mesh['coeffs_reconstruct'][0,1,2]=1
    u,v=analysis.reconstruct(normal,mesh)
    assert u[0,0]==3 and v[0,0]==4


def test_inversion_away_from_target_does_not_invent_a_second_crossing():
    pressure=np.array([[100000.],[99000.],[99500.],[60000.],[40000.],[20000.]])
    height=8000*np.log(100000/pressure)
    assert analysis.at_pressure(pressure,height,50000,allow_outside=True)[0]==pytest.approx(8000*np.log(2))
    ambiguous=np.array([[100000.],[40000.],[80000.],[20000.]])
    assert np.isnan(analysis.at_pressure(ambiguous,np.arange(4.)[:,None],50000,allow_outside=True)[0])


def test_constant_offset_does_not_earn_native_error_envelope():
    rows=[]
    for hours in range(0,121,6):
        native=max(1e-12,hours/24)
        rows.append({'hours':hours,'native_strict_vs_optimized':{'z500_m':native,'wind250_ms':native},
          'hex_vs_strict':{'z500_m':native+1,'wind250_ms':native+1},
          'hex_vs_optimized':{'z500_m':native+2,'wind250_ms':native+2}})
    verdict=analysis.summarize(rows)
    assert not verdict['z500_m']['hex_vs_strict']['within_native_envelope_at_every_sample']
    assert verdict['z500_m']['hex_vs_strict']['no_faster_log_growth']


def test_identically_zero_error_has_no_log_fit_and_cannot_grow():
    rows=[]
    for hours in range(0,121,6):
        native=float(np.exp(0.1*hours/24))
        pair={'z500_m':native,'wind250_ms':native}
        rows.append({'hours':hours,'native_strict_vs_optimized':pair,
          'hex_vs_strict':{'z500_m':0.,'wind250_ms':0.},'hex_vs_optimized':pair})
    summary=analysis.summarize(rows)
    for metric in summary.values():
        exact=metric['hex_vs_strict']
        assert exact['within_native_envelope_at_every_sample']
        assert exact['identically_zero_at_every_sample']
        assert exact['log_rms_growth_per_day'] is None
        assert exact['native_log_rms_growth_per_day']==pytest.approx(.1)
        assert exact['no_faster_log_growth']
    rows[1]['hex_vs_strict']['z500_m']=1e-15
    assert not analysis.summarize(rows)['z500_m']['hex_vs_strict']['identically_zero_at_every_sample']
