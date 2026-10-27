"""Full-step LTS at the forecast door: the class clock takes the plan's real top class (HEX-LAM-VS-ZOOM-2026-10-27
defect 1).

The door's clock took the config dataclass's ladder ceiling (16) while both executors plan at most 8, so a bare LTS
run on the 96 -> 3 km zoom globe stepped 12.5 s instead of 15 s (135.1 vs 105.9 s per forecast hour on an RTX 5090).
"""
from argparse import Namespace
import dataclasses

import numpy as np

from hexcore import forecast_door, mpas_defaults
from hexcore.fullstep_lts_census import DOOR_DEFAULT_MAX_RATE, mesh_class_census, read_mesh_edges


def _args(**values):
    base = dict(fullstep_local_timestep=True, fullstep_lts_mode="active", fullstep_lts_max_rate=None,
                grid=None, init=None, history_every_minutes=60, hours=3.0)
    base.update(values)
    return Namespace(**base)


def _chain_mesh(path, spacings_m):
    """A closed 1-D chain of cells (cell i joins cell i+1) whose edge lengths are ``spacings_m``, as an MPAS file."""
    from netCDF4 import Dataset
    n_edges = len(spacings_m)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("nCells", n_edges + 1)
        dataset.createDimension("nEdges", n_edges)
        dataset.createDimension("TWO", 2)
        dataset.createVariable("dcEdge", "f8", ("nEdges",))[:] = np.asarray(spacings_m, dtype=np.float64)
        coe = np.stack([np.arange(n_edges), np.arange(1, n_edges + 1)], axis=1) + 1   # MPAS files are 1-based
        dataset.createVariable("cellsOnEdge", "i4", ("nEdges", "TWO"))[:] = coe
    return path


def _graded(top_km):
    """3 km cells, then a ramp doubling to ``top_km``, then a long run of ``top_km`` cells."""
    ramp = [3000.0 * 2 ** k for k in range(int(np.log2(top_km / 3.0)) + 1)]
    return [3000.0] * 30 + [s for s in ramp for _ in range(6)] + [top_km * 1000.0] * 200


def test_the_door_default_top_rate_is_the_executors():
    """One default for both executors and the door's clock; the clock read the dataclass's 16 before the fix."""
    fields = {field.name: field.default for field in dataclasses.fields(forecast_door.ForecastRequest)}
    assert fields["fullstep_lts_max_rate"] == DOOR_DEFAULT_MAX_RATE == 8


def test_a_bare_lts_run_on_the_zoom_globe_steps_15_s_not_12_5_s():
    """The exact defect: the 18 s default with 1800 s radiation, hourly history, 3 h. 8 x 15 s = 120 s fits; the
    fix moves to 15 s. Before it, the door planned for 16 x the step and took 12.5 s (16 x 12.5 = 200 s)."""
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    assert fixed["dt_seconds"] == 18.0
    moved = forecast_door.lts_step(_args(), fixed, 1800.0)
    assert moved["dt_seconds"] == 15.0 and moved["lts_max_rate"] == 8


def test_the_clock_takes_the_mesh_top_class(tmp_path):
    """A mesh that grades only to 4 keeps its 18 s default (4 x 18 = 72 s divides every event); one that grades
    to 8 or past the cap moves to 15 s; an explicit --fullstep-lts-max-rate 16 plans 16 where the mesh reaches it."""
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    to12 = _chain_mesh(tmp_path / "to12.nc", _graded(12))
    to96 = _chain_mesh(tmp_path / "to96.nc", _graded(96))
    assert max(mesh_class_census(*read_mesh_edges(to12), max_rate=8)) == 4
    assert forecast_door.lts_step(_args(grid=to12), fixed, 1800.0)["dt_seconds"] == 18.0
    assert forecast_door.lts_step(_args(grid=to96), fixed, 1800.0)["dt_seconds"] == 15.0
    assert max(mesh_class_census(*read_mesh_edges(to96), max_rate=16)) == 16
    assert forecast_door.lts_step(_args(grid=to96, fullstep_lts_max_rate=16), fixed, 1800.0)["dt_seconds"] == 12.5


def test_the_census_holds_convective_cells_and_grades():
    """Cells finer than 8 km stay at rate 1 and no two neighbours differ by more than a factor 2."""
    spacing = np.asarray(_graded(96))
    n_cells = spacing.size + 1
    coe = np.stack([np.arange(spacing.size), np.arange(1, n_cells)], axis=1)
    census = mesh_class_census(spacing, coe, n_cells, max_rate=8)
    assert sorted(census) == [1, 2, 4, 8] and sum(census.values()) == n_cells
    assert census[1] >= 30 + 12   # the 3 and 6 km cells


def test_a_plain_run_never_reads_the_mesh(tmp_path):
    """LTS off: the selection is returned as given and no census is taken."""
    fixed = mpas_defaults.default_dt_seconds(3000.0)
    args = _args(fullstep_local_timestep=False, grid=_chain_mesh(tmp_path / "to96.nc", _graded(96)))
    assert forecast_door.lts_step(args, fixed, 1800.0) is fixed
    assert not hasattr(args, "_fullstep_lts_top")
