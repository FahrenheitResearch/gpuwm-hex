"""Order-2 dispatch contracts on the existing native-source operators."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from hexcore.config_v841 import V841DryDycoreConfig
from hexcore.integration import RKSchedule
from hexcore.regional_v841 import rk_timestep_f32, transport_rk_timestep_f32

ROOT = Path(__file__).resolve().parents[1]


def _ruler():
    name = "_rk2_numeric_ruler"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools/run_cuda_v841_numeric_ruler.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize("dt,substeps", [(15.0, 2), (20.0, 4), (60.0, 2)])
def test_native_order_two_schedule(dt, substeps):
    config = V841DryDycoreConfig(config_dt=dt, config_time_integration_order=2,
                                config_number_of_sub_steps=substeps)
    config.validate()
    schedule = RKSchedule.from_mpas(dt, order=2, acoustic_substeps=substeps, dynamics_splits=3)
    assert [s.acoustic_steps for s in schedule.stages] == [substeps // 2, substeps // 2, substeps]
    for stage in schedule.stages:
        assert np.float32(stage.large_timestep).tobytes() == rk_timestep_f32(
            outer_dt=dt, dynamics_split=3, rk_step=stage.stage, order=2).tobytes()
    assert transport_rk_timestep_f32(outer_dt=dt, rk_step=1, order=2) == np.float32(dt * 0.5)


def test_step_certificates_hold_their_receipts_and_cannot_borrow_rk3():
    """The certificates are evidence; a schedule they do not hold runs and
    is recorded as unmatched (retired as a gate 2026-10-10), and an RK3
    schedule never reads an RK2 certificate."""

    import hashlib
    from hexcore.config_v841 import V841MpasColumnPhysicsConfig
    from hexcore.step_anchors import step_anchor_record
    table = json.loads((ROOT / "src/hexcore/data/dycore-step-anchors.v1.json").read_text())
    assert {(r["dt_seconds"], r["order"], r["acoustic_substeps"]) for r in table["anchors"]} == {(15.0, 2, 2), (20.0, 2, 4), (60.0, 2, 2)}
    for row in table["anchors"]:
        data = (ROOT / row["certificate"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["certificate_sha256"]
        receipt = json.loads(data)
        assert receipt["cuda_repeat_a"] == receipt["cuda_repeat_b"]
        assert receipt["vertical_coefficient_calls"] == 3
        dt = row["dt_seconds"]
        if dt == 15:
            assert receipt["first_scalar_density_bitwise_cpu_authority"] is True
            assert receipt["native_schedule_bits_passed"] is True
            assert receipt["fixture"]["producer"] == "rw_mpas_mesh and rw_mpas_static"
        config = V841MpasColumnPhysicsConfig(config_dt=dt, config_time_integration_order=2,
            config_number_of_sub_steps=row["acoustic_substeps"], config_bldt_seconds=dt,
            config_cudt_seconds=dt, config_radt_seconds=900.0 if dt == 20 else 1800.0)
        config.validate()
        assert step_anchor_record(config)["matched"] is True
        rk3 = replace(config, config_time_integration_order=3)
        rk3.validate()
        assert step_anchor_record(rk3)["matched"] is False


@pytest.mark.parametrize("dt,count,radiation", [(15.0, 2, 600.0), (20.0, 4, 900.0), (60.0, 2, 1800.0)])
def test_the_late_constructor_receipt_uses_the_same_dycore_certificate(dt, count, radiation):
    from hexcore.drivers.run_cuda_v841_forecast import build_forecast_config, step_admission_receipt
    config = build_forecast_config(dt_seconds=dt, time_integration_order=2,
        acoustic_substeps=count, radiation_seconds=radiation)
    receipt = step_admission_receipt(config)
    dycore = receipt["dycore_anchor"]
    assert receipt["refuses"] is False
    assert (dycore["matched"], dycore["scope"], dycore["dt_seconds"], dycore["order"],
            dycore["acoustic_substeps"]) == (True, "dycore", dt, 2, count)
    other = step_admission_receipt(replace(config, config_number_of_sub_steps=6))
    assert other["dycore_anchor"]["matched"] is False
    assert other["x4_recorded_schedule"]["matched"] is False


def test_schedule_bits_against_the_compiled_native_block():
    receipt = os.environ.get("HEXCORE_NATIVE_RK_SCHEDULE")
    if not receipt:
        pytest.skip("set HEXCORE_NATIVE_RK_SCHEDULE to the compiled native schedule receipt")
    for row in json.loads(Path(receipt).read_text())["rows"]:
        dt, count, order = row["dt"], row["substeps"], row["order"]
        schedule = RKSchedule.from_mpas(dt, order=order, acoustic_substeps=count, dynamics_splits=3)
        for index, stage in enumerate(schedule.stages):
            assert int(np.float32(stage.large_timestep).view(np.int32)) == row["large_bits"][index]
            assert int(np.float32(stage.acoustic_timestep).view(np.int32)) == row["acoustic_bits"][index]
            assert stage.acoustic_steps == row["counts"][index]


def _case(dt, count):
    ruler = _ruler()
    if not ruler.DEFAULT_GRID.is_file() or not ruler.DEFAULT_STATIC.is_file():
        pytest.skip("the real x1.2562 mesh pair is absent")
    config = replace(ruler.ruler_config(), config_dt=dt, config_time_integration_order=2,
                     config_number_of_sub_steps=count)
    # Retain the established fixture's state and native-source coefficients;
    # only the driver's schedule changes. The older ruler audits its own
    # fixed schedule, so build that fixture before selecting this schedule.
    case = ruler.prepare_synthetic_case(config=ruler.ruler_config())
    case.config = config
    return ruler, case


@pytest.mark.parametrize("dt,count", [(20.0, 4), (60.0, 2)])
def test_order_two_cpu_keeps_first_stage_vertical_coefficients(dt, count):
    from hexcore.driver import DryDycoreDriver
    _, case = _case(dt, count)
    driver = DryDycoreDriver(case.mesh, case.vertical, case.reference, case.config,
                            terrain_metrics=case.terrain, advection_coefficients=case.coefficients,
                            reference_wind_profiles=case.profiles)
    original = driver._vertical_coefficients
    calls = []
    def recording(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    driver._vertical_coefficients = recording
    result = driver.step(case.state.copy(), saved_diagnostics=case.saved)
    assert len(calls) == 3, "order 2 computes coefficients once per dynamics split"
    assert result.state.time_seconds == dt


@pytest.mark.gpu
@pytest.mark.parametrize("dt,count", [(20.0, 4), (60.0, 2)])
def test_order_two_cuda_contract_and_repeatability(dt, count):
    import cupy as cp
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.drivers.run_cuda_v841_full_physics_x4 import attach_inactive_zero_deformation
    from hexcore.driver import DryDycoreDriver
    ruler, case = _case(dt, count)
    if "defc_a" not in case.mesh.arrays:
        attach_inactive_zero_deformation(case.mesh)
    cpu = DryDycoreDriver(case.mesh, case.vertical, case.reference, case.config,
                         terrain_metrics=case.terrain, advection_coefficients=case.coefficients,
                         reference_wind_profiles=case.profiles)
    expected = cpu.step(case.state.copy(), saved_diagnostics=case.saved).state
    outputs = []
    for _ in range(2):
        gpu = CudaDryDycoreDriver.from_host(case.mesh, case.state.copy(), case.vertical,
            case.reference, case.config, saved_diagnostics=case.saved,
            terrain_metrics=case.terrain, advection_coefficients=case.coefficients,
            reference_wind_profiles=case.profiles)
        original = gpu._vertical_coefficients
        calls = []
        def recording(*args, **kwargs):
            calls.append(args[1])
            return original(*args, **kwargs)
        gpu._vertical_coefficients = recording
        state = gpu.step().state
        assert len(calls) == 3
        arrays = {name: np.asarray(getattr(state, name)) for name in ruler.STATE_FIELDS}
        for name, array in arrays.items():
            assert np.isfinite(array).all()
            # Whole-step rounding is recorded separately. The new branch's
            # strict contracts are schedule bits, coefficient reuse and
            # scalar target-density bits, not an unearned endpoint envelope.
        outputs.append({name: ruler.array_sha256(value) for name, value in arrays.items()})
    assert outputs[0] == outputs[1]
    receipts = os.environ.get("HEXCORE_RK2_RECEIPT_DIR")
    if receipts:
        out = Path(receipts)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"rk2-{int(dt)}-{count}.json").write_text(json.dumps({
            "dt_seconds": dt, "order": 2, "acoustic_substeps": count,
            "dynamics_splits": 3, "vertical_coefficient_calls": len(calls),
            "cpu": {name: ruler.array_sha256(getattr(expected, name)) for name in ruler.STATE_FIELDS},
            "cuda_repeat_a": outputs[0], "cuda_repeat_b": outputs[1],
            "cpu_cuda_max_absolute_difference": {
                name: float(np.max(np.abs(outputs_array.astype(np.float64)
                    - np.asarray(getattr(expected, name)).astype(np.float64))))
                for name, outputs_array in arrays.items()
            },
            "whole_step_comparison": "differences measured; no native endpoint qualification",
        }, indent=2) + "\n")


@pytest.mark.gpu
def test_order_two_first_scalar_density_is_bitwise_the_cpu_authority():
    import cupy as cp
    from hexcore.cuda_driver import CudaDryDycoreDriver
    from hexcore.cuda_transport_v841 import advance_scalar_transport_cuda_v841
    from hexcore.drivers.run_cuda_v841_full_physics_x4 import attach_inactive_zero_deformation
    from hexcore.transport import advance_scalar_transport
    _, case = _case(20.0, 4)
    if "defc_a" not in case.mesh.arrays:
        attach_inactive_zero_deformation(case.mesh)
    old = case.state.rho
    new = old + np.float32(0.0625)
    hu = np.zeros_like(case.state.rho_u)
    hw = np.zeros_like(case.state.rho_w)
    arguments = dict(rk_step=1, config_time_integration_order=2,
                     config_scalar_advection=True, config_monotonic=False,
                     config_positive_definite=False, config_split_dynamics_transport=True,
                     config_scalar_adv_order=3, config_scalar_vadv_order=3,
                     config_coef_3rd_order=0.25)
    expected = advance_scalar_transport(case.mesh, case.state.scalars, case.state.scalars,
        old, new, hu, hw, 10.0, coefficients=case.coefficients, fzm=case.vertical.fzm,
        fzp=case.vertical.fzp, rdzw=case.vertical.rdzw, **arguments)
    gpu = CudaDryDycoreDriver.from_host(case.mesh, case.state.copy(), case.vertical,
        case.reference, case.config, saved_diagnostics=case.saved,
        terrain_metrics=case.terrain, advection_coefficients=case.coefficients,
        reference_wind_profiles=case.profiles)
    actual = advance_scalar_transport_cuda_v841(gpu.atmosphere.mesh, gpu.v841_context,
        cp.asarray(case.state.scalars), cp.asarray(case.state.scalars), cp.asarray(old),
        cp.asarray(new), cp.asarray(hu), cp.asarray(hw), 10.0,
        coefficients=gpu.coefficients, fzm=gpu.atmosphere.vertical.fzm,
        fzp=gpu.atmosphere.vertical.fzp, rdzw=gpu.atmosphere.vertical.rdzw, **arguments)
    np.testing.assert_array_equal(cp.asnumpy(actual.density).view(np.uint32), expected.density.view(np.uint32))
