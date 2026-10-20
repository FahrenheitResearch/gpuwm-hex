"""Native IAU on hex's step (hexcore.iau_native): what reaches the MPAS kernels, per class, on host
arrays.  The kernels themselves are proven against native MPAS in test_iau_v842 /
test_iau_correct_v842; here the binding, the clock and the release are checked."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import iau_native as native
from hexcore.iau_v842 import atm_iau_coef, atm_iau_coef_window

WSM6 = ("qv", "qc", "qr", "qi", "qs", "qg")
N_LEV = 2


class _HostCp:
    ndarray = np.ndarray

    @staticmethod
    def asarray(a):
        return np.asarray(a)

    @staticmethod
    def ascontiguousarray(a):
        return np.ascontiguousarray(a)

    @staticmethod
    def asnumpy(a):
        return np.asarray(a)


def _increments(n_cells=5, n_edges=7, seed=0):
    rng = np.random.default_rng(seed)
    out = {"u_amb": rng.standard_normal((N_LEV, n_edges)).astype(np.float32),
           "rho_amb": rng.standard_normal((N_LEV, n_cells)).astype(np.float32),
           "theta_amb": rng.standard_normal((N_LEV, n_cells)).astype(np.float32)}
    for name in WSM6:
        out[f"{name}_amb"] = rng.standard_normal((N_LEV, n_cells)).astype(np.float32)
    return out


class _Recorder:
    """Stands in for hexcore.iau_v842.atm_add_tend_anal_incr: records, and keeps MPAS's clock."""

    def __init__(self):
        self.calls = []

    def __call__(self, config, structs, itimestep, dt, tend_ru, tend_rtheta, tend_rho, *, cache):
        if config.correct:
            position = structs["diag"]["iau_window_elapsed"]
            weight = atm_iau_coef_window(config, float(position), dt)
            position[...] = np.float32(position + np.float32(dt))
        else:
            weight = atm_iau_coef(config, itimestep, dt)
        self.calls.append(dict(structs=structs, itimestep=itimestep, dt=dt, weight=float(weight),
                               tend=(tend_ru, tend_rtheta, tend_rho)))


def _step(n_cells, n_edges, n_scalars=6, garbage=None):
    held = SimpleNamespace(rho=np.zeros((N_LEV, n_cells), np.float32),
                           rho_theta=np.zeros((N_LEV, n_cells), np.float32),
                           rho_u=np.zeros((N_LEV, n_edges), np.float32),
                           scalars=np.zeros((n_scalars, N_LEV, n_cells), np.float32))
    state = SimpleNamespace(rho=np.full((N_LEV, n_cells), 2.0, np.float32),
                            rho_theta=np.full((N_LEV, n_cells), 600.0, np.float32),
                            scalars=np.zeros((n_scalars, N_LEV, n_cells), np.float32))
    cells_on_edge = np.stack([np.arange(n_edges) % n_cells, (np.arange(n_edges) + 1) % n_cells], axis=1).astype(np.int32)
    driver = SimpleNamespace(atmosphere=SimpleNamespace(vertical=SimpleNamespace(
        zz=np.full((N_LEV, n_cells), 0.5, np.float32))), cache="kernel-cache")
    context = dict(driver=driver, rho_edge=np.ones((N_LEV, n_edges), np.float32),
                   normal_velocity=np.zeros((N_LEV, n_edges), np.float32),
                   geometry=SimpleNamespace(cells_on_edge=cells_on_edge, regional_garbage_cell=garbage),
                   state=state)
    return held, context


def test_moist_block_and_the_species_that_carry_increments():
    assert native.moist_block(WSM6) == 6
    assert native.moist_block(WSM6 + ("nc", "ni", "nr")) == 6
    assert native.moist_block(("qv", "qc", "qr", "qi", "qs", "qg", "qh", "nh")) == 7
    with pytest.raises(ValueError, match="first model scalar"):
        native.moist_block(("qc", "qv"))
    with pytest.raises(ValueError, match="outside the leading moist block"):
        native.moist_block(("qv", "nc", "qc"))
    tend = native.physical_increment(_increments(), ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "nh"))
    assert tend["moist_end"] == 7 and tend["species"] == WSM6
    assert np.all(tend["scalars"][6] == 0)


@pytest.mark.parametrize("mode,window,dt,expect_fraction", [
    ("correct", 450.0, 100.0, 1.0),       # IN3: overlap weights, one dose whatever the step
    ("exact", 450.0, 100.0, 500.0 / 450),  # MPAS v8.4.2: NINT(4.5)=5 whole steps, 11 % too much
    ("exact", 600.0, 100.0, 1.0),
])
def test_one_class_is_dosed_by_mpas_rules_and_released_after_the_window(mode, window, dt, expect_fraction):
    recorder = _Recorder()
    tend = native.physical_increment(_increments(), WSM6)
    forcing = native.NativeIauForcing(tend, [native.identity_layout(5, 7)], window_seconds=window, mode=mode,
                                      scalar_names=WSM6, cp=_HostCp, apply=recorder)
    t = 0.0
    while t < 1200.0:
        held, context = _step(5, 7)
        forcing.add(0, held, t, dt, **context)
        t += dt
    receipt = forcing.receipt()
    assert receipt["fraction_admitted"][0] == pytest.approx(expect_fraction, rel=1e-6)
    assert receipt["complete"] == (abs(expect_fraction - 1) < 1e-6)
    assert all(call["weight"] > 0 for call in recorder.calls)
    assert len(recorder.calls) == receipt["classes"][0]["active_calls"]
    assert forcing.classes[0].released and forcing.classes[0].rho is None
    assert receipt["corrections"] == (["IN1", "IN2", "IN3"] if mode == "correct" else [])


def test_the_kernels_see_hex_state_under_mpas_pool_names():
    recorder = _Recorder()
    tend = native.physical_increment(_increments(), WSM6)
    forcing = native.NativeIauForcing(tend, [native.identity_layout(5, 7)], window_seconds=600.0, mode="correct",
                                      scalar_names=WSM6, cp=_HostCp, apply=recorder)
    held, context = _step(5, 7)
    forcing.add(0, held, 0.0, 60.0, **context)
    call = recorder.calls[0]
    structs = call["structs"]
    assert call["tend"] == (held.rho_u, held.rho_theta, held.rho)
    assert structs["tend"]["scalars_tend"] is held.scalars
    assert structs["state"]["rho_zz"] is context["state"].rho
    np.testing.assert_array_equal(structs["state"]["theta_m"], np.full((N_LEV, 5), 300.0, np.float32))
    assert structs["state"]["moist_end"] == 6 and structs["state"]["index_qv"] == 1
    assert structs["diag"]["rho_edge"] is context["rho_edge"]
    assert structs["mesh"]["nCellsSolve"] == 5 and structs["mesh"]["nEdgesSolve"] == 7
    np.testing.assert_array_equal(structs["tend_iau"]["rho"], tend["rho"])
    np.testing.assert_array_equal(structs["tend_iau"]["u"], tend["u"])


def test_each_class_takes_its_own_cells_and_edges_and_keeps_its_own_clock():
    recorder = _Recorder()
    tend = native.physical_increment(_increments(), WSM6)
    layouts = [SimpleNamespace(cell_l2g=np.array([0, 1, 2]), edge_l2g=np.array([0, 1, 2, 3])),
               SimpleNamespace(cell_l2g=np.array([2, 3, 4]), edge_l2g=np.array([3, 4, 5, 6]))]
    forcing = native.NativeIauForcing(tend, layouts, window_seconds=600.0, mode="correct", scalar_names=WSM6,
                                      cp=_HostCp, apply=recorder)
    for index, dt in ((0, 18.75), (1, 150.0)):
        t = 0.0
        while t < 900.0:
            held, context = _step(3, 4)
            forcing.add(index, held, t, dt, **context)
            t += dt
    receipt = forcing.receipt()
    assert receipt["fraction_admitted"] == pytest.approx([1.0, 1.0])
    first_fine = recorder.calls[0]["structs"]["tend_iau"]
    first_coarse = next(c for c in recorder.calls if c["dt"] == 150.0)["structs"]["tend_iau"]
    np.testing.assert_array_equal(first_fine["rho"], tend["rho"][:, [0, 1, 2]])
    np.testing.assert_array_equal(first_coarse["u"], tend["u"][:, [3, 4, 5, 6]])
    assert receipt["classes"][0]["max_position_lag_seconds"] < 1e-3


def test_a_limited_area_garbage_column_is_outside_the_solve_and_an_exterior_endpoint():
    recorder = _Recorder()
    tend = native.physical_increment(_increments(n_cells=4), WSM6)
    forcing = native.NativeIauForcing(tend, [native.identity_layout(4, 7)], window_seconds=600.0,
                                      mode="correct", scalar_names=WSM6, cp=_HostCp, apply=recorder)
    held, context = _step(5, 7, garbage=4)
    forcing.add(0, held, 0.0, 60.0, **context)
    structs = recorder.calls[0]["structs"]
    assert structs["mesh"]["nCellsSolve"] == 4 and structs["mesh"]["nCells"] == 5
    assert np.all(structs["tend_iau"]["rho"][:, 4] == 0)
    edges = structs["mesh"]["cellsOnEdge"]
    assert -1 in edges and 4 not in edges


def test_the_increment_must_fit_the_runs_scalars():
    tend = native.physical_increment(_increments(), WSM6)
    with pytest.raises(ValueError, match="moist block"):
        native.NativeIauForcing(tend, [native.identity_layout(5, 7)], window_seconds=600.0, mode="exact",
                                scalar_names=WSM6 + ("qh",), cp=_HostCp, apply=_Recorder())
