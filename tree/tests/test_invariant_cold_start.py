"""The native invariant stream and the cold-start coupled diagnostics.

Breakage these tests prevent (2026-10-05): on RRFS
v2's conus12km mesh with physics off, WOOF hex sat 5 to 300 times above
NOAA's own rounding envelope from hour 1 because

* hex read zgrid/zz/zb/zb3/zxu from the init file while MPAS-A, with an
  invariant stream configured, reads them from the invariant file (they
  differ by up to 2.7 mm in zgrid and 8e-5 relative in zz), and
* hex consumed the init file's pressure_p and pressure_base while MPAS-A's
  cold start (atm_init_coupled_diagnostics, v8.4.1
  mpas_atm_time_integration.F:7590-7650) always recomputes them.

Real-file tests: set HEXCORE_INVARIANT_CASE to a folder holding grid.nc,
init.nc and invariant.nc of one MPAS mesh whose init and invariant vertical
grids differ (the RRFS v2 conus12km L60 case does).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def case():
    folder = os.environ.get("HEXCORE_INVARIANT_CASE")
    if not folder:
        pytest.skip("set HEXCORE_INVARIANT_CASE to a folder with grid.nc, init.nc, invariant.nc")
    folder = Path(folder)
    from hexcore.mesh import Mesh

    paths = {name: folder / f"{name}.nc" for name in ("grid", "init", "invariant")}
    mesh = Mesh.from_netcdf(paths["grid"], paths["init"], validate=False)
    return paths, mesh


def _file_field(path, name):
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        variable = dataset.variables[name]
        value = np.asarray(variable[:])
        # (Time, entity, level) in an init file, (entity, level) in an
        # invariant file: return the logical (level, entity) either way.
        return (value[0] if variable.dimensions[0] == "Time" else value).T


def test_invariant_stream_supplies_the_vertical_grid(case):
    from hexcore.driver import load_mpas_vertical_grid

    paths, mesh = case
    plain = load_mpas_vertical_grid(paths["init"], mesh, allow_regional_sentinels=True)
    overlaid = load_mpas_vertical_grid(
        paths["init"], mesh, allow_regional_sentinels=True, invariant_path=paths["invariant"]
    )
    invariant_zz = _file_field(paths["invariant"], "zz")
    invariant_zgrid = _file_field(paths["invariant"], "zgrid")
    assert np.array_equal(overlaid.vertical_grid.zz, invariant_zz)
    assert np.array_equal(overlaid.vertical_grid.zgrid, invariant_zgrid)
    # The premise: this case's init and invariant grids really differ, so the
    # equality above cannot pass by coincidence.
    assert not np.array_equal(plain.vertical_grid.zz, overlaid.vertical_grid.zz)
    assert not np.array_equal(plain.terrain_metrics.zb_cell, overlaid.terrain_metrics.zb_cell)


def test_cold_start_recomputes_the_coupled_diagnostics(case):
    from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid

    paths, mesh = case
    native = load_mpas_vertical_grid(
        paths["init"], mesh, allow_regional_sentinels=True, invariant_path=paths["invariant"]
    )
    common = dict(
        scalar_names=("qv",),
        terrain_metrics=native.terrain_metrics,
        allow_regional_sentinels=True,
        return_saved_diagnostics=True,
    )
    _, file_ref, file_saved = load_mpas_initial_state(
        paths["init"], mesh, native.vertical_grid, **common
    )
    _, cold_ref, cold_saved = load_mpas_initial_state(
        paths["init"], mesh, native.vertical_grid, cold_start=True, **common
    )
    file_pressure_base = _file_field(paths["init"], "pressure_base")
    file_pressure_p = _file_field(paths["init"], "pressure_p")
    # Default: the oracle-stream behaviour the whole-step gates pin.
    assert np.array_equal(file_ref.pressure_base, file_pressure_base)
    assert np.array_equal(file_saved.pressure_perturbation, file_pressure_p)
    # Cold start: F:7649 and F:7645 from the model's own zz, not the file's.
    f32 = np.float32
    zz = np.asarray(native.vertical_grid.zz, dtype=f32)
    expected_base = zz * f32(287.0) * cold_ref.exner_base * cold_ref.rho_theta_base
    assert np.array_equal(cold_ref.pressure_base, expected_base)
    assert not np.array_equal(cold_ref.pressure_base, file_pressure_base)
    assert not np.array_equal(cold_saved.pressure_perturbation, file_pressure_p)
    # Prognostics are untouched by the choice.
    assert np.array_equal(cold_ref.exner_base, file_ref.exner_base)
    assert np.array_equal(cold_saved.theta_m, file_saved.theta_m)
