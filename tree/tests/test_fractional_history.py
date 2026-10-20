"""An 18-second smoke must write both boundaries without changing its timestep."""
import pytest
from hexcore.forecast_door import _schedule, ForecastDoorRefusal


def test_one_step_boundary_cadence_and_unchanged_six_hour_schedule():
    assert _schedule(18 / 3600, 18 / 60, 18, "input-mesh") == (1, 2)
    assert _schedule(6, 360, 18, "input-mesh") == (1200, 2)


def test_fractional_cadence_still_refuses_between_step_output():
    with pytest.raises(ForecastDoorRefusal, match="whole number"):
        _schedule(18 / 3600, 0.25, 18, "input-mesh")
