"""The native-free vertical artifact declares the engine's met-state schema.

The concrete breakage this guards: ``rw_mpas_init`` lays out its output from
the capsule schema VERBATIM and refuses when a computed value has no variable
to land in ("this run computed N value(s) the init file has no variable
for").  A constructed artifact that carries only mesh/statics/vertical fields
therefore fails EVERY native-free init at the engine, which is exactly what
the first native-free run against real x1.40962 assets measured on
2026-08-24.  These tests pin the landing-site declarations.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mpas_port import vertical_spec  # noqa: E402

netCDF4 = pytest.importorskip("netCDF4")

N_CELLS = 24
N_EDGES = 72
N_LEV = 5


@pytest.fixture()
def template(tmp_path) -> Path:
    path = tmp_path / "artifact-template.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("nCells", N_CELLS)
        dataset.createDimension("nEdges", N_EDGES)
        dataset.createDimension("nVertLevels", N_LEV)
        dataset.createDimension("nVertLevelsP1", N_LEV + 1)
        variable = dataset.createVariable("ter", "f8", ("nCells",))
        variable[:] = np.zeros(N_CELLS)
    return path


def test_every_engine_computed_variable_is_declared(template):
    with netCDF4.Dataset(template, "r+") as dataset:
        declared = vertical_spec._declare_met_state_schema(dataset)
    engine_computed = {name for name, *_ in vertical_spec.MET_STATE_VARIABLES}
    engine_computed.add("initial_time")
    engine_computed.update(
        name for name, *_ in vertical_spec.REFERENCE_PROFILE_VARIABLES)
    engine_computed.update(
        name for name, *_ in vertical_spec.ZERO_STATE_VARIABLES)
    engine_computed.add("Time")
    with netCDF4.Dataset(template) as dataset:
        present = set(dataset.variables)
        missing = engine_computed - present
        assert not missing, f"landing sites missing for: {sorted(missing)}"
        assert set(declared) == engine_computed
        assert dataset.dimensions["Time"].isunlimited()
        assert len(dataset.dimensions["nSoilLevels"]) == (
            vertical_spec.MET_STATE_SOIL_LEVELS)
        # One explicit zero record: carried fields read as defined zeros.
        assert len(dataset.dimensions["Time"]) == 1
        assert dataset.variables["theta"].dtype == np.float32
        assert dataset.variables["theta"].dimensions == (
            "Time", "nCells", "nVertLevels")
        assert dataset.variables["u"].dimensions == (
            "Time", "nEdges", "nVertLevels")
        assert dataset.variables["w"].dimensions == (
            "Time", "nCells", "nVertLevelsP1")
        assert float(np.abs(dataset.variables["theta"][0]).max()) == 0.0
        assert dataset.variables["initial_time"].dimensions == ("StrLen",)
        # Reference profiles: declared, zero, NOT record variables.
        assert dataset.variables["u_init"].dimensions == ("nVertLevels",)
        assert dataset.variables["t_init"].dimensions == (
            "nCells", "nVertLevels")
        assert float(np.abs(dataset.variables["u_init"][:]).max()) == 0.0


def test_the_format_zero_slots_complete_the_native_variable_set(template):
    """Time, dz and h_oml_initial are init-stream slots the engine never
    computes; the native v8.4.1 real-data init carries all three as exact
    zeros (measured on x1.40962, 2026-08-24).  Without them a native-free
    mint is three variables short of the native schema and any consumer
    that reads the init stream by the format's own contract refuses."""
    with netCDF4.Dataset(template, "r+") as dataset:
        vertical_spec._declare_met_state_schema(
            dataset, start_time="2026-08-12_06:00:00")
    with netCDF4.Dataset(template) as dataset:
        time = dataset.variables["Time"]
        assert time.dimensions == ("Time",)
        assert time.dtype == np.float32
        assert float(time[0]) == 0.0
        assert time.units == "seconds since 2026-08-12 06:00:00"
        dz = dataset.variables["dz"]
        assert dz.dimensions == ("Time", "nCells", "nSoilLevels")
        assert float(np.abs(dz[0]).max()) == 0.0
        oml = dataset.variables["h_oml_initial"]
        assert oml.dimensions == ("Time", "nCells")
        assert float(np.abs(oml[0]).max()) == 0.0


def test_landing_sites_carry_the_registry_attributes(template):
    """The engine's emitter copies variable attributes from the capsule
    verbatim, so a mint reads as an init only if the landing sites carry
    the Registry units/long_name the native file carries."""
    with netCDF4.Dataset(template, "r+") as dataset:
        vertical_spec._declare_met_state_schema(
            dataset, start_time="2026-08-12_06:00:00")
    with netCDF4.Dataset(template) as dataset:
        assert dataset.variables["theta"].units == "K"
        assert dataset.variables["theta"].long_name == "Potential temperature"
        assert dataset.variables["u"].units == "m s^{-1}"
        assert dataset.variables["surface_pressure"].units == "Pa"
        assert dataset.variables["u_init"].long_name == "u reference profile"
        assert dataset.variables["initial_time"].units == (
            "YYYY-MM-DD_hh:mm:ss")


def test_existing_landing_site_with_wrong_shape_refuses_by_name(template):
    with netCDF4.Dataset(template, "r+") as dataset:
        dataset.createDimension("Time", None)
        dataset.createVariable("theta", "f4", ("Time", "nCells"))
        with pytest.raises(vertical_spec.VerticalSpecError, match="theta"):
            vertical_spec._declare_met_state_schema(dataset)


def test_fixed_time_dimension_refuses_by_name(template):
    with netCDF4.Dataset(template, "r+") as dataset:
        dataset.createDimension("Time", 3)
        with pytest.raises(vertical_spec.VerticalSpecError, match="Time"):
            vertical_spec._declare_met_state_schema(dataset)
