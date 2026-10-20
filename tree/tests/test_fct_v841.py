"""Nonzero native-flux replay and regional FCT pool contracts."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native_case():
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.driver import DryDycoreDriver, load_mpas_initial_state, load_mpas_vertical_grid
    from hexcore.dynamics_v841 import load_v841_reference_wind_profiles, precomputed_mesh_inverse_v841
    from hexcore.mesh import Mesh
    folder = os.environ.get("HEXCORE_NATIVE_TRANSPORT_FIXTURE")
    if not folder:
        pytest.skip("set HEXCORE_NATIVE_TRANSPORT_FIXTURE to the real native nonzero-tracer fixture")
    folder = Path(folder)
    manifest = json.loads((folder / "fixture.json").read_text())
    for relative, digest in manifest["hashes"].items():
        assert hashlib.sha256((folder / relative).read_bytes()).hexdigest() == digest, relative
    init = folder / manifest["input_file"]
    mesh = Mesh.from_netcdf(init)
    native = load_mpas_vertical_grid(init, mesh)
    state, reference, saved = load_mpas_initial_state(init, mesh, native.vertical_grid,
        scalar_names=("qv",), reference_path=init, terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True)
    profiles = load_v841_reference_wind_profiles(init, n_vert_levels=native.vertical_grid.n_vert_levels)
    config = V841DryDycoreConfig(config_dt=20, config_time_integration_order=2,
        config_number_of_sub_steps=4, config_dynamics_split_steps=3,
        config_split_dynamics_transport=True, config_scalar_advection=True,
        config_monotonic=True, config_positive_definite=False, config_apvm_upwinding=0.5)
    cpu = DryDycoreDriver(mesh, native.vertical_grid, reference, config,
        terrain_metrics=native.terrain_metrics, reference_wind_profiles=profiles)
    def array(time, name):
        meta = json.loads((folder / time / "variables.json").read_text())[name]
        return np.fromfile(folder / time / (name + ".bin"), dtype=meta["dtype"]).reshape(meta["shape"])[0].T.copy()
    return dict(mesh=mesh, native=native, state=state, reference=reference, saved=saved,
        profiles=profiles, config=config, cpu=cpu,
        rho0=array("t0", "rho_zz"), rho1=array("t1", "rho_zz"),
        q0=array("t0", "qv")[None], target=array("t1", "qv")[None],
        hu=array("t1", "ruAvg"), hw=array("t1", "wwAvg"),
        inv_area=precomputed_mesh_inverse_v841(mesh, "areaCell", np.dtype(np.float32)))


def replay_cpu(case):
    from hexcore.transport import advance_scalar_transport
    q = case["q0"].copy()
    for stage, dt in enumerate((10.0, 10.0, 20.0), 1):
        q = advance_scalar_transport(case["mesh"], case["q0"], q, case["rho0"], case["rho1"],
            case["hu"], case["hw"], dt, rk_step=stage, config_time_integration_order=2,
            config_monotonic=True, config_positive_definite=False, config_split_dynamics_transport=True,
            coefficients=case["cpu"].advection_coefficients, fzm=case["native"].vertical_grid.fzm,
            fzp=case["native"].vertical_grid.fzp, rdzw=case["native"].vertical_grid.rdzw,
            inv_area_cell=case["inv_area"]).scalars
    return q


def assert_native_budget(actual, case):
    # Source-derived accumulation budget, not a fitted endpoint tolerance:
    # three passes, two accumulations, 10 reconstruction + 6 edge + 4
    # vertical slots = 120, rounded up to 128 binary32 rounding slots.
    unit_roundoff = float(np.finfo(np.float32).eps) / 2
    budget = 128 * unit_roundoff / (1 - 128 * unit_roundoff) * float(np.max(np.abs(case["q0"])))
    error = float(np.max(np.abs(actual.astype(np.float64) - case["target"].astype(np.float64))))
    assert np.count_nonzero(case["q0"] != case["target"]) > 0
    assert error <= budget
    return error, budget


def test_nonzero_native_flux_cpu_replay_and_mutation(native_case):
    actual = replay_cpu(native_case)
    assert_native_budget(actual, native_case)
    with pytest.raises(AssertionError):
        assert_native_budget(native_case["q0"], native_case)


def test_limiter_is_available_through_the_generic_forecast_door():
    from hexcore.drivers.run_cuda_v841_forecast import build_forecast_config
    config = build_forecast_config(dt_seconds=20, time_integration_order=2,
        acoustic_substeps=4, radiation_seconds=900, monotonic=True)
    assert config.config_monotonic is True
    config.validate()


@pytest.mark.parametrize("value", [1, "true", None])
def test_limiter_flag_cannot_open_from_a_nonlogical_value(value):
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.errors import ConfigurationRefusal
    with pytest.raises(ConfigurationRefusal, match="config_monotonic"):
        V841DryDycoreConfig(config_monotonic=value).validate()


@pytest.mark.gpu
def test_nonzero_native_flux_cuda_replay_and_production_dispatch(native_case, monkeypatch):
    import cupy as cp
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.cuda_transport_v841 import advance_scalar_transport_cuda_v841
    from hexcore.drivers.run_cuda_v841_full_physics_x4 import attach_inactive_zero_deformation
    case = native_case
    if "defc_a" not in case["mesh"].arrays:
        attach_inactive_zero_deformation(case["mesh"])
    def driver():
        return CudaDryDycoreDriver.from_host(case["mesh"], case["state"].copy(),
            case["native"].vertical_grid, case["reference"], case["config"],
            saved_diagnostics=case["saved"], terrain_metrics=case["native"].terrain_metrics,
            advection_coefficients=case["cpu"].advection_coefficients,
            reference_wind_profiles=case["profiles"])
    gpu = driver()
    import hexcore.cuda_transport_v841 as transport
    launch = transport._launch
    kernel_calls = []
    limited_values = []
    def recording(name, count, arguments, cache):
        result = launch(name, count, arguments, cache)
        kernel_calls.append(name)
        if name == "fct_scale":
            limited_values.append(int(cp.count_nonzero(arguments[-2] < 1)) +
                                  int(cp.count_nonzero(arguments[-1] < 1)))
        return result
    monkeypatch.setattr(transport, "_launch", recording)
    outputs = []
    for _ in range(2):
        q = cp.asarray(case["q0"])
        for stage, dt in enumerate((10.0, 10.0, 20.0), 1):
            q = advance_scalar_transport_cuda_v841(gpu.atmosphere.mesh, gpu.v841_context,
                cp.asarray(case["q0"]), q, cp.asarray(case["rho0"]), cp.asarray(case["rho1"]),
                cp.asarray(case["hu"]), cp.asarray(case["hw"]), dt,
                rk_step=stage, config_time_integration_order=2, config_monotonic=True,
                config_scalar_advection=True, config_positive_definite=False,
                config_split_dynamics_transport=True, config_scalar_adv_order=3,
                config_scalar_vadv_order=3, config_coef_3rd_order=0.25,
                coefficients=gpu.coefficients, fzm=gpu.atmosphere.vertical.fzm,
                fzp=gpu.atmosphere.vertical.fzp, rdzw=gpu.atmosphere.vertical.rdzw).scalars
        outputs.append(cp.asnumpy(q))
    np.testing.assert_array_equal(outputs[0].view(np.uint32), outputs[1].view(np.uint32))
    # The transcription contract has a stricter, bitwise CPU comparison.
    np.testing.assert_array_equal(outputs[0].view(np.uint32), replay_cpu(case).view(np.uint32))
    error, budget = assert_native_budget(outputs[0], case)
    assert max(limited_values) > 0, "the real tracer must exercise limiting"
    steps = []
    for _ in range(2):
        kernel_calls.clear()
        result = driver().step().state
        assert kernel_calls.count("fct_finish_v841") == 1, "production RK stage 3 must run FCT"
        steps.append({name: hashlib.sha256(np.asarray(getattr(result, name)).tobytes()).hexdigest()
            for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")})
    assert steps[0] == steps[1]
    receipt = os.environ.get("HEXCORE_FCT_RECEIPT")
    if receipt:
        Path(receipt).write_text(json.dumps(dict(native_changed=int(np.count_nonzero(case["q0"] != case["target"])),
            values=int(case["q0"].size), rounding_slots=128, absolute_budget=budget,
            limited_values=limited_values,
            cuda_native_max_absolute_error=error, cpu_cuda_different_bits=0,
            cuda_sha256=hashlib.sha256(outputs[0].tobytes()).hexdigest(),
            native_sha256=hashlib.sha256(case["target"].tobytes()).hexdigest(),
            production_repeat_a=steps[0], production_repeat_b=steps[1]), indent=2) + "\n")


@pytest.mark.gpu
def test_regional_fct_masks_include_garbage_and_preserve_specified_values():
    import cupy as cp
    from hexcore.cuda_transport_v841 import _launch
    deck = json.loads((Path(__file__).resolve().parents[1] /
        "verification/scalar-transport/native-boundary-v841.json").read_text())
    masks = cp.asarray(deck["masks"], dtype=cp.int32)
    stage = cp.asarray(deck["stage"], dtype=cp.float32)
    output = cp.asarray(deck["before_copy"], dtype=cp.float32)
    _launch("fct_copy_back_v841", 7, (np.int32(1), np.int32(1), np.int32(7),
        np.int32(1), masks, stage, output), None)
    np.testing.assert_array_equal(cp.asnumpy(output).view(np.uint32),
        np.asarray(deck["native_copy_bits"], np.uint32))
    residual = cp.asarray(deck["before_residual"], dtype=cp.float32)
    _launch("fct_boundary_residual_v841", 7, (np.int32(1), np.int32(1), np.int32(7), masks, residual), None)
    np.testing.assert_array_equal(cp.asnumpy(residual).view(np.uint32),
        np.asarray(deck["native_residual_bits"], np.uint32))
    density = cp.asarray([[2, 3, 0], [4, 5, 0]], dtype=cp.float32)
    _launch("fct_pool_density_v841", 2, (np.int32(2), np.int32(3), np.int32(2), density), None)
    np.testing.assert_array_equal(cp.asnumpy(density).view(np.uint32),
        np.asarray([[2, 3, 1], [4, 5, 1]], np.float32).view(np.uint32))


@pytest.mark.gpu
def test_fct_density_pool_survives_the_real_regional_cleanup_hook():
    import cupy as cp
    from hexcore.cuda_backend import KernelCache
    from hexcore.cuda_regional_forecast_v841 import RegionalGarbageDiscipline
    from hexcore.cuda_transport_v841 import _launch, _check_density
    cache = KernelCache()
    discipline = RegionalGarbageDiscipline(n_cells_solve=2,
        n_edges_solve=4, n_vertices_solve=6)
    cache.post_launch = discipline.scrub
    density = cp.asarray([[2, 3, 0], [4, 5, 0]], dtype=cp.float32)
    flag = cp.zeros(1, dtype=cp.int32)
    args = (np.int32(2), np.int32(3), np.int32(2), density)
    # Reproduce the failure: the post-launch cleanup undoes an unregistered
    # density scratch write before the following validation launch.
    _launch("fct_pool_density_v841", 2, args, cache)
    _check_density(density, name="target", count=6, invalid_flag=flag, kernel_cache=cache)
    assert int(cp.asnumpy(flag)[0]) == 1
    discipline.bind_unit_pool(density)
    flag.fill(0)
    _launch("fct_pool_density_v841", 2, args, cache)
    _check_density(density, name="target", count=6, invalid_flag=flag, kernel_cache=cache)
    assert int(cp.asnumpy(flag)[0]) == 0
    np.testing.assert_array_equal(cp.asnumpy(density), [[2, 3, 1], [4, 5, 1]])
