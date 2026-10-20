"""Exercise the real product builder with bounded in-memory I/O doubles.

NumPy substitutes for CUDA only in these small CPU contract tests. Production
still imports CuPy and Rust NetCDF transport without a CPU forecast fallback.
"""
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


def load(monkeypatch):
    root = Path(__file__).parents[1] / "src/hexcore"
    package = types.ModuleType("terrain_contract")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    door = types.ModuleType("terrain_contract.s2s_door")
    door.S2SError = type("S2SError", (RuntimeError,), {})
    door.cores = lambda: 2
    door.write_json = lambda *args: None
    monkeypatch.setitem(sys.modules, door.__name__, door)
    cp = types.ModuleType("cupy")
    cp.__dict__.update(np.__dict__)
    cp.asnumpy = np.asarray
    cp.cuda = types.SimpleNamespace(Device=lambda: types.SimpleNamespace(id=0))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    engines = types.ModuleType("terrain_contract.engines")
    engines.RENDERER = "test-only-renderer"
    engines.resolve = lambda _: Path("test-only-renderer")
    monkeypatch.setitem(sys.modules, engines.__name__, engines)
    spec = importlib.util.spec_from_file_location("terrain_contract.s2s_products", root / "s2s_products.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Variable:
    def __init__(self, values, units="m"):
        self.values, self.units = np.asarray(values), units

    def __getitem__(self, index):
        return self.values[index]

    def getncattr(self, name):
        return getattr(self, name)


class Dataset:
    def __init__(self, **variables):
        self.variables = variables

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def getncattr(self, name):
        return {"s2s_climatology_source": "ERA5", "s2s_climatology_start_mmdd": "10-17"}[name]


@pytest.mark.parametrize("baseline", ["ERA5", "missing_terrain", "ensemble"])
def test_builder_uses_corrected_baseline_or_refuses(monkeypatch, tmp_path, baseline):
    products = load(monkeypatch)
    height = np.array([0., 2000., 4000., 0.], dtype="f4")
    yy, xx = np.zeros((2, 2), dtype="f4"), np.ones((2, 2), dtype="f4")
    normal = Dataset(XLAT=Variable(yy), XLONG=Variable(xx))
    for field, units in products.FIELDS.items():
        normal.variables["week_1_" + field] = Variable(np.full((2, 2), 290.), units)
    if baseline == "ERA5":
        normal.variables["orography"] = Variable(np.zeros((2, 2)))
    datasets = {"init": Dataset(zgrid=Variable(np.stack([height, height + 10], axis=1))),
                "grid": Dataset(latCell=Variable(height * 0), lonCell=Variable(height * 0)),
                "normal": normal}
    bridge = types.ModuleType("gpuwm.netcdf_bridge")
    bridge.open_dataset = lambda path: datasets[str(path)]
    monkeypatch.setitem(sys.modules, "gpuwm", types.ModuleType("gpuwm"))
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.setattr(products, "product_grid", lambda *args: (yy, xx, np.arange(4)))
    monkeypatch.setattr(products, "climatology_path", lambda *args:
        (None, "ensemble-relative baseline; not a historical normal") if baseline == "ensemble"
        else ("normal", "ERA5"))
    start = datetime(2026, 10, 17)
    monkeypatch.setattr(products, "validate_history", lambda *args:
        [(start + timedelta(days=n), Path(str(n))) for n in range(8)])
    monkeypatch.setattr(products, "read_native", lambda *args: {})
    monkeypatch.setattr(products, "native_diagnostics", lambda *args:
        {field: 290. - .0065 * height if field == "temperature2" else height * 0
         for field in products.FIELDS})
    written = []

    def write(path, cp, lat, lon, fields, *args):
        written.append((fields, args[-1]))
        path.write_bytes(b"test-only")

    monkeypatch.setattr(products, "write_aggregate", write)
    monkeypatch.setattr(products.subprocess, "run", lambda *args, **kwargs:
        types.SimpleNamespace(returncode=0, stdout="s2s-window global-ensemble-projection"))
    value = dict(start_time="2026-10-17_00:00:00", paths=dict(work=str(tmp_path), grid="grid"),
                 members=[dict(init="init", run_id="m1")])
    args = types.SimpleNamespace(days=7, history_hours=24, row="global-120km")
    if baseline == "missing_terrain":
        with pytest.raises(products.S2SError, match="requires source orography"):
            products.build_products(value, dict(attempts=[]), args, {})
        assert not written
    else:
        result = products.build_products(value, dict(attempts=[]), args, {})
        assert len(written) == 1
        np.testing.assert_allclose(written[0][0]["temperature2"]["anomaly"], 0, atol=3e-5)
        np.testing.assert_array_equal(written[0][1]["model_orography"].ravel(), height)
        correction = result["temperature2_terrain_correction"]
        assert correction["applied"] == (baseline == "ERA5")
        assert correction["lapse_rate_K_per_m"] == (.0065 if baseline == "ERA5" else 0.)
        if baseline == "ensemble":
            assert "not a historical normal" in result["baseline"]
            assert "6.5 K/km" not in result["baseline"]
            assert "climatology_orography" not in written[0][1]


def test_unavailable_auto_normal_retains_explicit_ensemble_fallback(monkeypatch, tmp_path):
    from hexcore import s2s_climatology, s2s_products
    def unavailable(*args):
        raise OSError("test archive unavailable")
    monkeypatch.delenv("GPUWM_HEX_S2S_ERA5_CLIMATOLOGY_URL", raising=False)
    monkeypatch.setattr(s2s_climatology, "fetch_normals", unavailable)
    args = types.SimpleNamespace(climatology=None)
    path, label = s2s_products.climatology_path({"paths": {"work": str(tmp_path)}}, args, {})
    assert path is None and "not a historical normal" in label
    assert "test archive unavailable" in (tmp_path / "era5-climatology-unavailable.json").read_text()
