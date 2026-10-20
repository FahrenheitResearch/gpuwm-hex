"""Hardware checks for graph inputs, output lifetimes, and eager fallback."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from hexcore.cuda_backend.containers import (
    DeviceAtmosphere,
    DevicePrognosticState,
    DeviceSavedDiagnostics,
    TransferStats,
)
from hexcore.cuda_driver import _CudaDynamicsSubcycleResult
from hexcore.cuda_horizontal import CudaSolveDiagnostics
from hexcore.integration import RKSchedule


@dataclass
class _Constants:
    bias: Any


@dataclass
class _Moist:
    qtot: Any
    cqw: Any
    cqu: Any


@dataclass
class _Tendencies:
    rho: Any
    rho_theta: Any
    rho_u: Any
    rho_w: Any
    scalars: Any


@pytest.fixture
def gpu_case():
    if any(
        os.environ.get(name, "") not in ("", "0")
        for name in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")
    ):
        pytest.skip("local GPU access is disabled")
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    if not available:
        pytest.skip("no CUDA device")

    kernel = cp.RawKernel(
        r'''
        extern "C" __global__ void graph_input_check(
            const int count, const float *state, const float *saved,
            const float *diagnostics, const float *moist,
            const float *tendencies, const float *bias,
            int *invalid, float *output)
        {
            const int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= count) return;
            float value = __fadd_rn(state[i], saved[i]);
            value = __fadd_rn(value, diagnostics[i]);
            value = __fadd_rn(value, moist[i]);
            value = __fadd_rn(value, tendencies[i]);
            output[i] = __fadd_rn(value, bias[0]);
            if (i == 0) atomicOr(invalid, 2);
        }
        ''',
        "graph_input_check",
        options=("--fmad=false",),
    )
    kernel.compile()
    transfer = TransferStats(0, 0.0)

    def inputs(width=4, seed=1, time_seconds=0.0):
        def array(shape, offset):
            return cp.full(shape, np.float32(seed + offset), dtype=cp.float32)

        shape = (2, width)
        state = DevicePrognosticState(
            array(shape, 0), array(shape, 1), array(shape, 2),
            array((3, width), 3), array((1, 2, width), 4),
            time_seconds, np.dtype(np.float32), transfer,
        )
        saved = DeviceSavedDiagnostics(
            *(array(shape if index < 6 else (3, width), index + 5)
              for index in range(7)),
            np.dtype(np.float32), transfer,
        )
        diagnostics = CudaSolveDiagnostics(
            *(array(shape, index + 12) for index in range(11))
        )
        moist = _Moist(array(shape, 23), array(shape, 24), array(shape, 25))
        tendencies = _Tendencies(
            *(array(shape if index < 3 else (3, width) if index == 3
                    else (1, 2, width), index + 26) for index in range(5))
        )
        return state, {
            "time_level_one": saved,
            "initial_diagnostics": diagnostics,
            "schedule": RKSchedule.from_mpas(30.0, dynamics_splits=3),
            "outer_dt": 30.0,
            "validation_flag": cp.array([8], dtype=cp.int32),
            "physics_tendencies": tendencies,
            "moist_coefficients": moist,
        }

    state, kwargs = inputs()
    atmosphere = DeviceAtmosphere(
        _Constants(cp.array([0], dtype=cp.float32)),
        state,
        _Constants(cp.array([0], dtype=cp.float32)),
        _Constants(cp.array([3], dtype=cp.float32)),
        kwargs["time_level_one"],
        None,
        transfer,
    )
    driver = SimpleNamespace(
        cp=cp,
        cache=SimpleNamespace(post_launch=None),
        atmosphere=atmosphere,
        v841_context=_Constants(cp.array([0], dtype=cp.float32)),
        config=SimpleNamespace(config_dt=30.0),
        configuration_sha256="0" * 64,
        regional_v841=None,
        halo_exchanger_v841=None,
        mixing_config_v841=None,
        mixing_calls_v841=0,
        nlev=2,
        ncells=4,
        nedges=4,
    )

    def eager(value, **arguments):
        from hexcore.cuda_subcycle_graphs import launch_timed

        # The scratch is deliberately absent from the returned object. A graph
        # must retain its allocation after the Python function returns.
        scratch = cp.empty_like(value.rho)
        launch_timed(
            kernel, ((value.rho.size + 63) // 64,), (64,),
            (
                np.int32(value.rho.size), value.rho,
                arguments["time_level_one"].theta_m,
                arguments["initial_diagnostics"].h_edge,
                arguments["moist_coefficients"].qtot,
                arguments["physics_tendencies"].rho,
                driver.atmosphere.reference.bias,
                arguments["validation_flag"], scratch,
            ),
            warmup=0,
            repeats=1,
        )
        result_state = replace(value, rho=cp.array(scratch, copy=True))
        driver.mixing_calls_v841 += 1
        return _CudaDynamicsSubcycleResult(
            state=result_state,
            saved=arguments["time_level_one"],
            diagnostics=arguments["initial_diagnostics"],
            mass_flux_u=cp.array(value.rho_u, copy=True),
            mass_flux_w=cp.array(value.rho_w, copy=True),
            modified_theta_dynamics_rate=cp.array(scratch, copy=True),
        )

    return cp, driver, inputs, eager


def _assert_arrays_equal(cp, actual, expected):
    for field in fields(expected):
        wanted = getattr(expected, field.name)
        found = getattr(actual, field.name)
        if isinstance(wanted, cp.ndarray):
            assert cp.asnumpy(found).tobytes() == cp.asnumpy(wanted).tobytes(), field.name
        elif hasattr(wanted, "__dataclass_fields__"):
            _assert_arrays_equal(cp, found, wanted)


@pytest.mark.gpu
def test_replay_updates_inputs_flags_and_preserves_scratch(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    caller = cp.cuda.Stream(non_blocking=True)
    for step in range(8):
        with caller:
            state, kwargs = inputs(seed=step + 1, time_seconds=step * 30.0)
            driver.atmosphere.state = state
            expected_flag = cp.array(kwargs["validation_flag"], copy=True)
            expected = eager(state, **{**kwargs, "validation_flag": expected_flag})
            driver.mixing_calls_v841 -= 1
            actual = graphs.advance(eager, state, **kwargs)
            # Encourage a freed capture allocation to be recycled before replay.
            churn = [cp.full_like(state.rho, -99) for _ in range(20)]
        caller.synchronize()
        _assert_arrays_equal(cp, actual, expected)
        assert actual.state.time_seconds == state.time_seconds
        assert int(cp.asnumpy(kwargs["validation_flag"])[0]) == 10
        assert driver.mixing_calls_v841 == step + 1
        del churn
    receipt = graphs.receipt()
    assert receipt["captured_subcycles"] >= 1
    assert receipt["replayed_subcycles"] == 8
    assert receipt["eager_subcycles"] == 0
    assert receipt["fallbacks"] == []
    assert receipt["retained_scratch_bytes"] > 0


@pytest.mark.gpu
def test_shape_change_recaptures_before_first_replay(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    for _ in range(3):
        state, kwargs = inputs()
        graphs.advance(eager, state, **kwargs)
    before = graphs.receipt().copy()
    state, kwargs = inputs(width=7, seed=9)
    result = graphs.advance(eager, state, **kwargs)
    after = graphs.receipt()
    assert after["eager_subcycles"] == 0
    assert after["captured_subcycles"] == before["captured_subcycles"] + 1
    assert result.state.rho.shape == (2, 7)
    expected = eager(state, **kwargs)
    _assert_arrays_equal(cp, result, expected)
    graphs.advance(eager, state, **kwargs)
    assert graphs.receipt()["captured_subcycles"] == before["captured_subcycles"] + 1


@pytest.mark.gpu
def test_same_shape_static_pointer_change_invalidates_graph(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    for _ in range(3):
        state, kwargs = inputs()
        graphs.advance(eager, state, **kwargs)
    before = graphs.receipt().copy()
    driver.atmosphere.reference.bias = cp.array([19], dtype=cp.float32)
    state, kwargs = inputs(seed=4)
    actual = graphs.advance(eager, state, **kwargs)
    assert graphs.receipt()["eager_subcycles"] == 0
    assert graphs.receipt()["captured_subcycles"] == before["captured_subcycles"] + 1
    expected = eager(state, **kwargs)
    _assert_arrays_equal(cp, actual, expected)
    graphs.advance(eager, state, **kwargs)
    replay = graphs.advance(eager, state, **kwargs)
    _assert_arrays_equal(cp, replay, expected)


@pytest.mark.gpu
def test_retained_warm_allocation_failure_uses_eager(gpu_case, monkeypatch):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs, _Allocations

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    state, kwargs = inputs()
    attempts = 0

    def exhausted(self, size):
        nonlocal attempts
        attempts += 1
        raise cp.cuda.memory.OutOfMemoryError(size, size)

    monkeypatch.setattr(_Allocations, "__call__", exhausted)
    state, kwargs = inputs(seed=7)
    actual = graphs.advance(eager, state, **kwargs)
    expected = eager(state, **kwargs)
    _assert_arrays_equal(cp, actual, expected)
    assert int(cp.asnumpy(kwargs["validation_flag"])[0]) == 10
    assert graphs.receipt()["captured_subcycles"] == 0
    assert graphs.receipt()["eager_subcycles"] == 1
    assert attempts == 1
    graphs.advance(eager, state, **kwargs)
    assert graphs.receipt()["eager_subcycles"] == 2
    fallback = graphs.receipt()["fallbacks"]
    assert [row["subcycle"] for row in fallback] == [1, 2]
    assert all(row["capture_error"]["type"] == "OutOfMemoryError" for row in fallback)
    assert all(row["breakage_prevented"] for row in fallback)
    assert attempts == 1


@pytest.mark.gpu
def test_capture_body_error_keeps_site_and_receipts_for_every_fallback(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)

    def forbidden_upload(state, **kwargs):
        if cp.cuda.get_current_stream().is_capturing():
            # Match the first recorded scratch request, so the failure is the
            # forbidden upload rather than an earlier allocation-size guard.
            cp.asarray(np.ones(state.rho.shape, dtype=np.float32))
        return eager(state, **kwargs)

    for index in range(3):
        state, kwargs = inputs(seed=index + 1, time_seconds=index * 30.0)
        actual = graphs.advance(forbidden_upload, state, **kwargs)
        expected = eager(state, **kwargs)
        _assert_arrays_equal(cp, actual, expected)
    rows = graphs.receipt()["fallbacks"]
    assert [row["subcycle"] for row in rows] == [1, 2, 3]
    assert all("H2D" in row["capture_error"]["message"] for row in rows)
    assert all(any("forbidden_upload" in frame for frame in row["capture_error"]["site"])
               for row in rows)
    assert all([len(stage["small_steps"]) for stage in row["acoustic_subcycles"]] == [1, 3, 6]
               for row in rows)
    assert graphs.receipt()["replayed_subcycles"] == 0


@pytest.mark.gpu
def test_require_replay_refuses_first_fallback_with_receipt(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    _, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    graphs.require_replay = True
    driver.halo_exchanger_v841 = object()
    state, kwargs = inputs()
    with pytest.raises(RuntimeError, match="host_halo_exchange"):
        graphs.advance(eager, state, **kwargs)
    assert graphs.receipt()["eager_subcycles"] == 0
    assert graphs.receipt()["fallbacks"][0]["subcycle"] == 1


@pytest.mark.gpu
def test_preparation_memory_failure_restores_counters_before_eager(gpu_case):
    from hexcore.cuda_subcycle_graphs import SubcycleGraphs

    cp, driver, inputs, eager = gpu_case
    graphs = SubcycleGraphs(driver)
    calls = 0

    def exhausted_once(state, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            driver.mixing_calls_v841 += 1
            raise cp.cuda.memory.OutOfMemoryError(4, 4)
        return eager(state, **kwargs)

    state, kwargs = inputs()
    actual = graphs.advance(exhausted_once, state, **kwargs)
    assert driver.mixing_calls_v841 == 1
    expected = eager(state, **kwargs)
    _assert_arrays_equal(cp, actual, expected)
    assert graphs.receipt()["fallbacks"][0]["reason"] == "prepare_memory_exhausted"
    assert graphs.receipt()["eager_subcycles"] == 1


@pytest.mark.gpu
def test_timed_launch_in_capture_scope_preserves_launch_count(gpu_case):
    from hexcore.cuda_subcycle_graphs import graph_launch_scope, launch_timed

    cp, _, _, _ = gpu_case
    kernel = cp.RawKernel(
        r'''
        extern "C" __global__ void graph_count(int *count)
        {
            if (threadIdx.x == 0) atomicAdd(count, 1);
        }
        ''',
        "graph_count",
    )
    kernel.compile()
    count = cp.zeros((1,), dtype=cp.int32)
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        with graph_launch_scope():
            timing = launch_timed(kernel, (1,), (1,), (count,), warmup=2, repeats=3)
        graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    assert timing.measured is False
    assert int(cp.asnumpy(count)[0]) == 6
