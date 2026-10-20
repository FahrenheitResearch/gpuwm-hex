import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("s2s_terrain",
    Path(__file__).parents[1] / "src/hexcore/s2s_terrain.py")
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


def test_identical_atmosphere_on_different_terrain_has_zero_anomaly():
    # A warm anomaly must not be invented by comparing different elevations.
    source_height = np.array([0., 4000., 3000., 0.], dtype="f4")
    model_height = np.array([2000., 2000., 3000., 0.], dtype="f4")
    normal = 290. - .0065 * source_height
    model = 290. - .0065 * model_height
    adjusted = terrain.temperature_on_model_terrain(np, normal, model_height, source_height)
    np.testing.assert_allclose(model - adjusted, 0, atol=3e-5)
    # Equal-height ice/ocean columns retain their original temperature signal.
    np.testing.assert_array_equal(adjusted[2:], normal[2:])


def test_real_weather_signal_survives_terrain_adjustment():
    source = np.array([270.], dtype="f4")
    adjusted = terrain.temperature_on_model_terrain(np, source,
        np.array([3000.]), np.array([1000.]))
    assert float(adjusted[0]) == 257.
    assert float((adjusted + 4. - adjusted)[0]) == 4.


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
def test_invalid_lapse_rate_refuses(bad):
    a = np.zeros(2)
    with pytest.raises(ValueError, match="lapse rate"):
        terrain.temperature_on_model_terrain(np, a, a, a, lapse_rate=bad)


def test_missing_or_nonfinite_terrain_refuses():
    a = np.zeros(2)
    with pytest.raises(ValueError, match="identical grids"):
        terrain.temperature_on_model_terrain(np, a, a[:1], a)
    with pytest.raises(ValueError, match="finite"):
        terrain.temperature_on_model_terrain(np, a, a, a + np.nan)


class Variable:
    def __init__(self, values, units):
        self.values, self.units = values, units

    def __getitem__(self, key):
        return self.values[key]

    def getncattr(self, name):
        return getattr(self, name)


def test_supplied_climatology_cannot_bypass_terrain_contract():
    class Dataset:
        variables = {}
    ds = Dataset()
    with pytest.raises(ValueError, match="requires source orography"):
        terrain.normal_orography(np, ds, (2,))
    ds.variables = {"orography": Variable(np.zeros(2), "m**2 s**-2")}
    with pytest.raises(ValueError, match="units must be m"):
        terrain.normal_orography(np, ds, (2,))
    ds.variables["orography"] = Variable(np.zeros(2), "m")
    np.testing.assert_array_equal(terrain.normal_orography(np, ds, (2,)), np.zeros(2))
