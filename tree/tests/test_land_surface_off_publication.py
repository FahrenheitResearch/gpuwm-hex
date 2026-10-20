"""Land surface off publishes a forecast with no soil column.

THE BREAKAGE: with sf_surface_physics=0 the engine still exports tslb/smois at
its allocated 4-level width, a column no scheme set.  The suite seam refused
the run at the first committed boundary (guard hexphys-no-soil-publication,
global 30 km 2026-10-11; golden 0.4.0 W7 arm "land surface off"), because
the publication required both fields.  The seam now leaves them out for a
zero-level row and the publication carries the absent column: no required
soil fields, no soil-moisture range, no nSoilLevels axis in the history file.
A soil row still has to carry both fields.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from _layout import PACKAGE_DIR

RUNNER_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"
SUITE_PATH = PACKAGE_DIR / "cuda_suite_physics_v841.py"


@pytest.fixture()
def runner(monkeypatch):
    name = "_test_land_off_runner"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for attribute, value in (("N_CELLS", 6), ("N_EDGES", 12), ("N_LEVELS", 3),
                             ("N_INTERFACES", 4), ("N_SOIL_LEVELS", 0)):
        monkeypatch.setattr(module, attribute, value)
    yield module
    sys.modules.pop(name, None)


def _arrays(runner, *, soil: bool) -> dict:
    n, z = runner.N_CELLS, runner.N_LEVELS
    arrays = {name: np.full((z, n), 1.0e-4, dtype=np.float32) for name in runner.SCALAR_NAMES}
    arrays.update(
        rho=np.ones((z, n), np.float32), theta=np.full((z, n), 300.0, np.float32),
        pressure=np.full((z, n), 9.0e4, np.float32), surface_pressure=np.full(n, 1.0e5, np.float32),
        rainc=np.zeros(n, np.float32), rainnc=np.zeros(n, np.float32),
        snownc=np.zeros(n, np.float32), graupelnc=np.zeros(n, np.float32),
        tsk=np.full(n, 290.0, np.float32),
    )
    arrays["qv"][:] = 1.0e-3
    if soil:
        arrays["smois"] = np.full((4, n), 0.3, np.float32)
        arrays["tslb"] = np.full((4, n), 285.0, np.float32)
    return arrays


def test_the_gate_admits_a_frame_without_soil_for_a_zero_level_row(runner):
    gate = runner.physical_snapshot_gate({"arrays": _arrays(runner, soil=False)},
                                         allow_initial_negative_qv=False)
    assert gate["soil_moisture_range"] is None


def test_the_gate_refuses_a_soil_column_on_a_zero_level_row(runner):
    with pytest.raises(ValueError, match="zero-level soil row published a soil column"):
        runner.physical_snapshot_gate({"arrays": _arrays(runner, soil=True)},
                                      allow_initial_negative_qv=False)


def test_a_soil_row_still_gates_soil_moisture(runner, monkeypatch):
    monkeypatch.setattr(runner, "N_SOIL_LEVELS", 4)
    arrays = _arrays(runner, soil=True)
    assert runner.physical_snapshot_gate({"arrays": arrays}, allow_initial_negative_qv=False)[
        "soil_moisture_range"] == [pytest.approx(0.3), pytest.approx(0.3)]
    arrays["smois"][0, 0] = 1.5
    with pytest.raises(FloatingPointError, match="soil moisture"):
        runner.physical_snapshot_gate({"arrays": arrays}, allow_initial_negative_qv=False)


def test_the_history_file_declares_no_soil_axis(runner, tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    n, e = runner.N_CELLS, runner.N_EDGES
    static = {
        "indexToCellID": np.arange(1, n + 1, dtype=np.int32), "latCell": np.zeros(n, np.float32),
        "lonCell": np.zeros(n, np.float32), "ter": np.zeros(n, np.float32),
        "indexToEdgeID": np.arange(1, e + 1, dtype=np.int32), "latEdge": np.zeros(e, np.float32),
        "lonEdge": np.zeros(e, np.float32),
    }
    snapshot = {"arrays": _arrays(runner, soil=False),
                "receipt": {"schema": runner.SNAPSHOT_SCHEMA, "label": "TEST", "step": 1}}
    path = tmp_path / "frame.nc"
    runner.write_snapshot_netcdf(path, snapshot, static)
    with netCDF4.Dataset(path) as dataset:
        assert "nSoilLevels" not in dataset.dimensions
        assert not any(dim.isunlimited() for dim in dataset.dimensions.values())
        assert "smois" not in dataset.variables and "tslb" not in dataset.variables
        assert "tsk" in dataset.variables


def test_the_suite_seam_no_longer_refuses_land_surface_off():
    source = SUITE_PATH.read_text(encoding="utf-8")
    assert "publishing that column" not in source
    assert "retires guard hexphys-no-soil-publication" in source


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("broken", [None, "rainnc", "rainc", "extra"])
def test_snapshot_fallout_policy_keeps_native_words_and_checks_other_buckets(runner, signed, broken):
    from hexcore.species_row import SurfaceAccumulatorDeclaration
    arrays = _arrays(runner, soil=False)
    arrays["extra"] = np.zeros(runner.N_CELLS, np.float32)
    arrays["rainnc"][0] = np.float32(-1.3981764490221711e-14)
    if broken == "rainnc":
        arrays[broken][0] = np.nan
    elif broken:
        arrays[broken][0] = -1.0
    names = ("rainnc", "snownc", "graupelnc", "extra")
    declarations = tuple(SurfaceAccumulatorDeclaration(name, "float32", name.upper(), name,
        signed_fallout=(signed and name != "extra")) for name in names)
    before = {name: value.tobytes() for name, value in arrays.items()}
    def gate():
        return runner.physical_snapshot_gate({"arrays": arrays}, allow_initial_negative_qv=False,
            surface_accumulators=names, surface_accumulator_declarations=declarations)
    if signed and broken is None:
        result = gate()
        assert not result["precipitation_nonnegative"]
        assert result["precipitation_matches_scheme"]
        assert result["signed_fallout_accumulators"] == sorted(names[:3])
    else:
        with pytest.raises(FloatingPointError):
            gate()
    assert {name: value.tobytes() for name, value in arrays.items()} == before
