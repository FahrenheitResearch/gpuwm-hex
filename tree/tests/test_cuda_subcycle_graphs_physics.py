"""Graph replay against eager for the real v8.4.1 subcycle WITH physics.

Gate (F1, 2026-10-15). Breakage it prevents: a default single-card forecast
silently diverging from the proven eager path. Graph replay staged only rho,
rho_u and rho_theta of the held physics tendencies, so SG2's lagged
microphysics heating (``diabatic_theta``: None on step 1, an array from step
2) never reached a replay; every default route-A run dropped it from step 2
on while eager applied it. The older replay check
(test_cuda_subcycle_graphs_dycore.py) is dry and single-step, so it could not
see a physics field that appears after capture.

This drives the real CudaDryDycoreDriver subcycle with held physics
tendencies, moist coefficients and the lagged heating, four outer steps of
three dynamics subcycles each, new tendency arrays every step and the heating
absent at capture, and requires every result array to be bitwise equal to an
eager twin. It makes no native or forecast claim.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import os
from typing import Any

import numpy as np
import pytest

from test_cuda_subcycle_graphs_dycore import _host_case


@dataclass(frozen=True, slots=True)
class _Held:
    """The held-tendency carrier's shape (CudaHeldMpasPhysicsTendenciesV841)."""

    rho: Any
    rho_u: Any
    rho_theta: Any
    scalars: Any
    time_seconds: float
    diabatic_theta: Any = None


def _compare_exact(cp, actual, expected, path="result"):
    if isinstance(expected, cp.ndarray):
        assert isinstance(actual, cp.ndarray), path
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        assert cp.asnumpy(actual).tobytes(order="C") == cp.asnumpy(expected).tobytes(order="C"), path
    elif is_dataclass(expected):
        for field in fields(expected):
            if field.name in ("stage_physics_tendencies", "time_seconds", "h2d"):
                continue  # the carrier is read back only with a stage callback
            _compare_exact(cp, getattr(actual, field.name), getattr(expected, field.name),
                           f"{path}.{field.name}")
    elif expected is None:
        assert actual is None, path


@pytest.mark.gpu
def test_physics_subcycle_replays_bitwise_when_lagged_heating_appears(tmp_path):
    if any(os.environ.get(name, "") not in ("", "0") for name in
           ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")):
        pytest.skip("local GPU access is disabled")
    cp = pytest.importorskip("cupy")
    pytest.importorskip("scipy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA device")
    if count == 0:
        pytest.skip("no CUDA device")

    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_driver import CudaDryDycoreDriver, _CudaV841MoistDynamicsCoefficients
    from hexcore.cuda_subcycle_graphs import attach_subcycle_graphs
    from hexcore.dynamics_v841 import V841ReferenceWindProfiles
    from hexcore.integration import RKSchedule

    mesh, state, vertical, reference, terrain = _host_case()
    nlev, ncells, nedges = 4, mesh.dimensions["nCells"], mesh.dimensions["nEdges"]
    config = V841DryDycoreConfig(
        config_dt=3.0, config_dynamics_split_steps=3,
        config_split_dynamics_transport=True, config_monotonic=False,
        config_positive_definite=False, config_apvm_upwinding=0.5,
    )
    wind = V841ReferenceWindProfiles(np.zeros(4, dtype=np.float32),
                                    np.zeros(4, dtype=np.float32))
    cache = KernelCache(cache_dir=tmp_path / "kernels")
    drivers = [CudaDryDycoreDriver.from_host(
        mesh, state.copy(), vertical, reference, config,
        terrain_metrics=terrain, reference_wind_profiles=wind, kernel_cache=cache,
    ) for _ in range(2)]
    graphs = attach_subcycle_graphs(drivers[1])
    schedule = RKSchedule.from_mpas(
        config.config_dt, order=config.config_time_integration_order,
        acoustic_substeps=config.config_number_of_sub_steps, dynamics_splits=3,
    )
    diagnostics = [driver.horizontal.solve_diagnostics(
        driver.atmosphere.state.rho, driver.atmosphere.state.rho_u,
        dt=config.config_dt, apvm_upwinding=config.config_apvm_upwinding,
        normal_velocity=driver.atmosphere.saved.normal_velocity, rk_step=3,
    ) for driver in drivers]
    n_scalars = int(drivers[0].atmosphere.state.scalars.shape[0])

    def plane(rng, shape, scale, offset=0.0):
        return cp.asarray((offset + scale * rng.standard_normal(shape)).astype(np.float32))

    steps, subcycles = 4, 3
    for step in range(steps):
        # The step builder hands each step NEW arrays; SG2's heating is absent
        # on the first step and present from the second, as in ArWen route A.
        rng = np.random.default_rng(1000 + step)
        held = _Held(
            rho=plane(rng, (nlev, ncells), 1.0e-6),
            rho_u=plane(rng, (nlev, nedges), 1.0e-5),
            rho_theta=plane(rng, (nlev, ncells), 1.0e-4),
            scalars=plane(rng, (n_scalars, nlev, ncells), 1.0e-8),
            time_seconds=step * config.config_dt,
            diabatic_theta=None if step == 0 else plane(rng, (nlev, ncells), 2.0e-3),
        )
        moist = _CudaV841MoistDynamicsCoefficients(
            qtot=cp.abs(plane(rng, (nlev, ncells), 1.0e-3)),
            cqw=plane(rng, (nlev, ncells), 1.0e-4, 1.0),
            cqu=plane(rng, (nlev, nedges), 1.0e-4, 1.0),
        )
        for substep in range(subcycles):
            flags = [cp.zeros((1,), dtype=cp.int32) for _ in drivers]
            results = []
            for index, driver in enumerate(drivers):
                driver.atmosphere.state.time_seconds = float(step * config.config_dt)
                results.append(driver._advance_dynamics_subcycle_v841(
                    driver.atmosphere.state, time_level_one=driver.atmosphere.saved,
                    initial_diagnostics=diagnostics[index], schedule=schedule,
                    outer_dt=config.config_dt, validation_flag=flags[index],
                    physics_tendencies=held, moist_coefficients=moist,
                    dynamics_substep=substep + 1,
                ))
            cp.cuda.get_current_stream().synchronize()
            where = f"step {step + 1} subcycle {substep + 1}"
            _compare_exact(cp, results[1], results[0], where)
            _compare_exact(cp, flags[1], flags[0], f"{where} invalid_flag")
            assert int(cp.asnumpy(flags[0])[0]) == 0, where
            for index, (driver, result) in enumerate(zip(drivers, results)):
                driver.atmosphere.state = result.state
                driver.atmosphere.saved = result.saved
                diagnostics[index] = result.diagnostics
    receipt = graphs.receipt()
    # One capture at the first step and one when the heating appears.
    assert receipt["captured_subcycles"] == 2, receipt
    assert receipt["replayed_subcycles"] == steps * subcycles, receipt
    assert receipt["eager_subcycles"] == 0, receipt
    print(json.dumps({"instrument": "actual_v841_physics_subcycle_graph_identity",
                      "cells": ncells, "levels": nlev, "steps": steps,
                      "subcycles_compared": steps * subcycles,
                      "lagged_heating_from_step": 2, "all_result_fields_bitwise": True,
                      "graphs": receipt}, sort_keys=True))
