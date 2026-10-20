"""Actual v8.4.1 RK/acoustic graph smoke on a small synthetic closed mesh.

This compares the real subcycle's complete result against its eager path.
It supplies no native authority and makes no forecast or regional claim.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import json
import os

import numpy as np
import pytest


def _float32_fields(value):
    return replace(value, **{
        field.name: np.ascontiguousarray(getattr(value, field.name), dtype=np.float32)
        for field in fields(value)
        if isinstance(getattr(value, field.name), np.ndarray)
    })


def _host_case():
    from test_mesh_validation import graded_sphere_mesh
    from hexcore.driver import TerrainMetrics, make_synthetic_x1_case
    from hexcore.vertical import runtime_vertical_vectors

    mesh = graded_sphere_mesh(level=1, zoom=1.0)
    cells = mesh.dimensions["nCells"]
    edges = mesh.dimensions["nEdges"]
    vertices = mesh.dimensions["nVertices"]
    maximum = mesh.dimensions["maxEdges"]
    mesh.arrays.update({
        "angleEdge": np.zeros(edges, dtype=np.float32),
        "fVertex": np.zeros(vertices, dtype=np.float32),
        "fEdge": np.zeros(edges, dtype=np.float32),
        "meshDensity": np.ones(cells, dtype=np.float32),
        "defc_a": np.zeros((cells, maximum), dtype=np.float32),
        "defc_b": np.zeros((cells, maximum), dtype=np.float32),
        "nominalMinDc": np.asarray(mesh.dcEdge.min(), dtype=np.float32),
        "deriv_two": np.zeros((edges, 2, 15), dtype=np.float32),
    })
    case = make_synthetic_x1_case(
        mesh, n_vert_levels=4, perturbation_amplitude=2.0e-5,
        wind_speed=2.0, n_scalars=1,
    )
    vertical = replace(case.vertical_grid, **runtime_vertical_vectors(case.vertical_grid))
    terrain_shape = (5, cells, maximum)
    return (
        mesh, _float32_fields(case.state), _float32_fields(vertical),
        _float32_fields(case.reference),
        TerrainMetrics(np.zeros(terrain_shape, dtype=np.float32),
                       np.zeros(terrain_shape, dtype=np.float32)),
    )


def _compare_exact(cp, actual, expected, path="result"):
    if isinstance(expected, cp.ndarray):
        assert isinstance(actual, cp.ndarray), path
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        assert cp.asnumpy(actual).tobytes(order="C") == cp.asnumpy(expected).tobytes(order="C"), path
    elif is_dataclass(expected):
        assert type(actual) is type(expected), path
        for field in fields(expected):
            _compare_exact(cp, getattr(actual, field.name), getattr(expected, field.name),
                           f"{path}.{field.name}")
    else:
        assert actual == expected, path


@pytest.mark.gpu
def test_actual_v841_subcycle_replays_bitwise(tmp_path):
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
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.cuda_subcycle_graphs import attach_subcycle_graphs
    from hexcore.dynamics_v841 import V841ReferenceWindProfiles
    from hexcore.integration import RKSchedule

    mesh, state, vertical, reference, terrain = _host_case()
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
    for subcycle in range(4):
        flags = [cp.zeros((1,), dtype=cp.int32) for _ in drivers]
        results = []
        for index, driver in enumerate(drivers):
            driver.atmosphere.state.time_seconds = float(subcycle)
            advance = driver._advance_dynamics_subcycle_v841
            results.append(advance(
                driver.atmosphere.state, time_level_one=driver.atmosphere.saved,
                initial_diagnostics=diagnostics[index], schedule=schedule,
                outer_dt=config.config_dt, validation_flag=flags[index],
            ))
        cp.cuda.get_current_stream().synchronize()
        _compare_exact(cp, results[1], results[0])
        _compare_exact(cp, flags[1], flags[0], "invalid_flag")
        assert int(cp.asnumpy(flags[0])[0]) == 0
        for index, (driver, result) in enumerate(zip(drivers, results)):
            driver.atmosphere.state = result.state
            driver.atmosphere.saved = result.saved
            diagnostics[index] = result.diagnostics
    receipt = graphs.receipt()
    assert receipt["captured_subcycles"] > 0, receipt
    assert receipt["replayed_subcycles"] == 4, receipt
    assert receipt["eager_subcycles"] == 0, receipt
    print(json.dumps({"instrument": "actual_v841_subcycle_graph_smoke",
                      "cells": mesh.dimensions["nCells"], "levels": 4,
                      "subcycles_compared": 4, "all_result_fields_bitwise": True,
                      "invalid_flags_bitwise": True, "graphs": receipt}, sort_keys=True))
