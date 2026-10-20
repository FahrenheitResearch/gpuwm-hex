import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("score", Path(__file__).parents[1] / "tools/varmesh_baseline/score_six_hour.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def test_pressure_interpolation_and_no_terrain_extrapolation():
    p = np.array([[100000., 50000., 25000.], [40000., 20000., 10000.]])
    values = 3 * np.log(p) + 7
    result = score.pressure_level(p, values, 50000)
    assert np.isclose(result[0], 3 * np.log(50000) + 7)
    assert np.isnan(result[1])
    assert np.isclose(score.pressure_level(p[:1], values[:1], 25000)[0], values[0, -1])


def test_cyclic_longitude_and_area_weighting():
    f = np.array([[10., 20., 30., 40.], [0., 10., 20., 30.]])
    result = score.bilinear(f, np.array([1., 0.]), np.array([0., 90., 180., 270.]),
                            np.array([.5]), np.array([-45.]))
    assert result[0] == 20
    m = score.Metrics()
    m.add(np.array([1., 3., np.nan]), np.array([3., 1., 999.]))
    assert m.result()["bias"] == 1.5
    assert np.isclose(m.result()["rmse"], np.sqrt(3))
