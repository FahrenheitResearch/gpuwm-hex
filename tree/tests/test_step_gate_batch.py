"""Pinned gate transport, exact refusals and same-step publication rules."""
from types import SimpleNamespace

import numpy as np
import pytest


def test_health_refusal_is_never_a_bonus_history_frame():
    from hexcore.drivers.run_cuda_v841_forecast import may_capture_refusal_state
    assert not may_capture_refusal_state({"refused_in": "step health gate"})
    assert may_capture_refusal_state({"refused_in": "history capture"})
    assert may_capture_refusal_state({})


def test_empty_envelope_and_malformed_flag_need_no_device(monkeypatch):
    from hexcore import cuda_solve_region_v841 as solve
    def forbidden():
        raise AssertionError("empty envelope contacted CUDA")
    monkeypatch.setattr(solve, "_cp", forbidden)
    kernels = solve.SolveRegionKernels(None)
    assert kernels.envelope([]) == {}
    with pytest.raises(TypeError, match="resident int32"):
        kernels.envelope_with_flags([], [np.zeros(2, dtype=np.int32)])


def cache():
    from hexcore.cuda_backend import KernelCache, require_cuda
    return KernelCache(capability=require_cuda(min_compute=(12, 0)))


def test_pinned_batch_on_a_nondefault_stream_and_reused_storage():
    import cupy as cp
    from hexcore.cuda_solve_region_v841 import PinnedWordBatch
    batch = PinnedWordBatch()
    with cp.cuda.Stream(non_blocking=True):
        arrays = [cp.arange(17, dtype=cp.int64), cp.asarray([7], dtype=cp.int32)]
        result = batch.read(arrays)
        assert result[0].tolist() == list(range(17))
        assert result[1].tolist() == [7]
        pinned = batch.pinned
        arrays[1].fill(9)
        assert batch.read(arrays)[1].tolist() == [9]
        assert batch.pinned is pinned
        assert batch.copies == 2 and batch.bytes == 2 * (17 * 8 + 4)
        arrays[1].fill(11)
        assert batch.read([arrays[1]])[0].tolist() == [11]
        assert batch.pinned is pinned
        assert batch.copies == 3 and batch.bytes == 2 * (17 * 8 + 4) + 4


def test_endpoint_envelope_and_flags_use_one_copy():
    import cupy as cp
    from hexcore.cuda_solve_region_v841 import SolveRegionKernels
    kernels = SolveRegionKernels.for_cache(cache())
    fields = [("value", cp.asarray([[1, 4, -3, np.nan]], dtype=cp.float32), 3)]
    result, flags = kernels.envelope_with_flags(fields, [cp.asarray([0], dtype=cp.int32),
                                                        cp.asarray([1], dtype=cp.int32)])
    assert result == {"value": (-3.0, 4.0, False, 1)}
    assert flags == [0, 1]
    assert kernels.word_batch.copies == 1
    assert kernels.word_batch.bytes == 40


def health_stack(cp):
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    state = SimpleNamespace(**{name: cp.ones((2, 5), dtype=cp.float32)
                               for name in forecast._STATE_HEALTH_FIELDS})
    state.scalars = cp.zeros((6, 2, 5), dtype=cp.float32)
    saved = SimpleNamespace(**{name: cp.ones((2, 5), dtype=cp.float32)
                               for name in forecast._SAVED_HEALTH_FIELDS})
    for owner in (state, saved):
        for array in vars(owner).values():
            array[..., -1] = cp.nan
    atmosphere = SimpleNamespace(state=state, saved=saved)
    return {"driver": SimpleNamespace(atmosphere=atmosphere, cache=cache()),
            "solve_cells": 4, "bdy_mask_cell": [0, 1, 2, 3]}


@pytest.mark.parametrize("condition", ["pass", "rho_nan", "rho_zero", "theta_zero",
                                       "exner_zero", "velocity", "hydrometeor"])
def test_health_batch_keeps_the_exact_refusal_and_no_second_copy(condition):
    import cupy as cp
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    from hexcore.cuda_solve_region_v841 import SolveRegionKernels
    stack = health_stack(cp)
    driver = stack["driver"]
    if condition == "rho_nan":
        driver.atmosphere.state.rho[0, 0] = cp.nan
    elif condition == "rho_zero":
        driver.atmosphere.state.rho[0, 0] = 0
    elif condition == "theta_zero":
        driver.atmosphere.saved.theta_m[0, 0] = 0
    elif condition == "exner_zero":
        driver.atmosphere.saved.exner[0, 0] = 0
    elif condition == "velocity":
        driver.atmosphere.saved.vertical_velocity[1, 2] = 201
    elif condition == "hydrometeor":
        driver.atmosphere.state.scalars[2, 0, 1] = -1
    def outcome():
        try:
            return forecast.step_health_gate(stack, 3, cp, trace_hot_cell=True)
        except FloatingPointError as error:
            return str(error)
    expected = outcome()
    forecast.install_step_gate_batch(stack, cp)
    kernels = SolveRegionKernels.for_cache(driver.cache)
    before = kernels.word_batch.copies
    assert driver.final_validation_reader(driver.atmosphere, cp.zeros(1, dtype=cp.int32)) == 0
    assert outcome() == expected
    assert kernels.word_batch.copies == before + 1
    assert driver._step_gate_envelope is None
    if condition != "pass":
        assert isinstance(expected, str)
        assert not forecast.may_capture_refusal_state({"refused_in": "step health gate"})


def test_failed_numeric_word_cannot_leave_a_cached_health_result():
    import cupy as cp
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    stack = health_stack(cp)
    driver = stack["driver"]
    forecast.install_step_gate_batch(stack, cp)
    assert driver.final_validation_reader(driver.atmosphere, cp.ones(1, dtype=cp.int32)) == 1
    assert driver._step_gate_envelope is None


@pytest.mark.parametrize("gate", ["qv", "clamp"])
def test_dependency_refusals_are_exact_and_clamp_writes_nothing(gate):
    import cupy as cp
    from hexcore.cuda_physics_v841 import (WSM6_SCALAR_NAMES, nonnegative_qv_scratch_v841,
                                          clamp_wsm6_scalars_in_place_v841)
    scalars = cp.full((6, 2, 7), -0.5, dtype=cp.float32)
    scalars[0, 0, 0] = cp.nan
    before = scalars.get().tobytes()
    function = nonnegative_qv_scratch_v841 if gate == "qv" else clamp_wsm6_scalars_in_place_v841
    text = ("non-finite qv refused before aggregate phase one" if gate == "qv" else
            "non-finite WSM6 carrier refused before phase two")
    with pytest.raises(FloatingPointError) as caught:
        function(scalars, scalar_names=WSM6_SCALAR_NAMES, kernel_cache=cache())
    assert str(caught.value) == text
    assert scalars.get().tobytes() == before
