"""Short showcase forecasts must accept float hours without admitting partial steps."""
from fractions import Fraction

import pytest

from hexcore.mpasphys.forecast_v842 import integral_duration


@pytest.mark.parametrize('hours,dt,seconds', [
    (5/60, '3.75', 300), (1/60, '0.9375', 60), (0.125, '3.75', 450),
])
def test_float_hour_conversion(hours, dt, seconds):
    assert integral_duration(Fraction(str(hours))*3600, Fraction(dt)) == seconds


@pytest.mark.parametrize('seconds', [None, Fraction(0), Fraction(-1), Fraction('300.01')])
def test_real_partial_or_invalid_duration_is_refused(seconds):
    with pytest.raises(ValueError, match='integral number'):
        integral_duration(seconds, Fraction('3.75'))
