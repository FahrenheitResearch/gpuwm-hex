"""A refused step whose physics rollback breaks still reports the refusal.

THE BREAKAGE (hex arbitrary-physics lane, 2026-10-11, defect 5): with
``--stop-on-refusal`` a refused step rolls the physics adapter back by
rebuilding its seam from the boundary export.  When that rebuild itself
raised, the adapter was left in phase ``broken``.  The loop still recorded
the refusal and stopped, but the end of the run then asked the adapter for
``restart_state`` (the final-state digest, and the carried identity when
fingerprints are on), which is legal only at a step boundary.  The run died
with "restart_state is legal only at a step boundary", wrote no receipt, and
the step's own refusal was never reported.  The composite step meanwhile
reported a phase-one rebuild failure as a clean abort, and a phase-two one
under "abort_step requires a begun or finished transaction".

Reproduced 2026-10-11 on a real run: x1.163842, one card, default physics,
--stop-on-refusal, CuPy pool limited to 80% of the run's peak.  Step 2 ran
out of memory in the monotonic scalar limiter, the seam rebuild ran out
again, and the run exited 2 with that message and no receipt.  With the fix
the same command exits 0 with status truncated_by_model_refusal, the
refusal's original_refusal is the limiter's OutOfMemoryError, and
physics_seam names phase "broken".

What these tests hold shut, on a CPU-only box, against the real adapter
state machine, the real composite step and the real forecast loop:

* a rollback that cannot rebuild the seam leaves phase ``broken`` and the
  composite step says the seam is retired, with the step's own refusal at
  the end of the exception chain;
* the forecast records that refusal (``original_refusal`` names the step's
  words), marks the physics seam retired, never asks a non-boundary adapter
  for its restart state, and returns its receipt.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import cuda_arwen_physics_v841 as arwen
from _layout import PACKAGE_DIR
from test_late_health_pipeline import Atmosphere, Backend, CPUArrays, Carrier, receipt_result

RUNNER_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"
ORIGINAL = "negative accumulated precipitation in rainc"


def _load_runner():
    name = "_test_refusal_retired_physics_runner"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _armed_backend_whose_rebuild_fails(monkeypatch, dt: float):
    """The real adapter, armed, with a seam whose rebuild runs out of memory."""

    class _Seam:
        elapsed_seconds = 0.0

        def _restart_manifest(self):
            return {}

        def export_state(self):
            return {"identity": {}, "arrays": {}, "scalars": {}}

    class _Snapshot:
        def __init__(self, seam):
            self.seam = seam

        def export_state(self):
            return {"identity": {}, "arrays": {}, "scalars": {}}

        def to_host(self, snapshot):
            return {"identity": {}, "arrays": {}, "scalars": {}}

    backend = object.__new__(arwen.PersistentTwoPhaseCudaPhysicsBackendV841)
    backend._rollback_snapshot = True
    backend._phase = "boundary"
    backend._boundary_snapshot = None
    backend._step_start = None
    backend._candidate_scalar_target = None
    backend._candidate_scalar_backup = None
    backend._pending_gwdo_result = None
    backend._pending_refl10cm = None
    backend._prep_geometry = None
    backend._kernel_cache = None
    backend._last_receipt = {"phase": "boundary"}
    backend._seam = _Seam()
    backend._constructor = SimpleNamespace(dt=dt)

    def refuse(*_args, **_kwargs):
        raise FloatingPointError(ORIGINAL)

    def rebuild():
        raise MemoryError("Out of memory allocating 1,073,741,824 bytes rebuilding the seam")

    monkeypatch.setattr(arwen, "prepare_mpas_to_phys_cuda_v841", refuse)
    monkeypatch.setattr(arwen, "DeviceBoundarySnapshotV841", _Snapshot)
    monkeypatch.setattr(backend, "_new_seam", rebuild)
    return backend


class _Driver:
    def __init__(self) -> None:
        start = SimpleNamespace(
            time_seconds=0.0,
            rho=np.ones((1, 1), dtype=np.float32),
            rho_u=np.ones((1, 1), dtype=np.float32),
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        self.atmosphere = SimpleNamespace(state=start)


def _broken_step_error(monkeypatch, runner=None):
    """Run one composite step whose phase one refuses and whose rollback breaks."""

    runner = runner or _load_runner()
    backend = _armed_backend_whose_rebuild_fails(monkeypatch, runner.DT_SECONDS)
    with pytest.raises(runner.CompositeTransactionError) as caught:
        runner.execute_composite_step(
            driver=_Driver(),
            backend=backend,
            scalar_names=runner.SCALAR_NAMES,
            physics_geometry=object(),
            kernel_cache=object(),
            previous_surface_updates=None,
            couple=lambda *a, **k: object(),
            clamp=lambda *a, **k: object(),
            recover=lambda *a, **k: object(),
        )
    return runner, backend, caught.value


def test_a_rollback_that_cannot_rebuild_the_seam_retires_it_and_says_so(
        monkeypatch, cpu_physics_phase_release):
    _, backend, error = _broken_step_error(monkeypatch)
    assert backend._phase == "broken"
    message = str(error)
    assert "rollback was incomplete" in message
    assert "could not rebuild its seam" in message
    assert "aborted without publication" not in message
    assert "abort_step requires" not in message
    with pytest.raises(RuntimeError, match="restart_state is legal only at a step boundary"):
        backend.restart_state()

    from hexcore.drivers import run_cuda_v841_forecast as forecast

    details = forecast.refusal_exception_details(error)
    assert details["original_refusal"] == {"exception": "FloatingPointError", "message": ORIGINAL}
    names = [row["exception"] for row in details["exception_chain"]]
    assert names == ["CompositeTransactionError", "RuntimeError", "MemoryError", "FloatingPointError"]
    assert forecast.retired_physics_seam(backend)["phase"] == "broken"


class _RetiringBackend(Backend):
    """The late-health harness backend with the adapter's restart contract."""

    def restart_state(self):
        if self._phase != "boundary":
            raise RuntimeError("restart_state is legal only at a step boundary")
        return {"seam": self._seam.export_state()}


@pytest.mark.parametrize("fingerprint_every", [0, 1])
def test_the_forecast_reports_the_refusal_and_returns_without_a_physics_restart(
        monkeypatch, tmp_path, fingerprint_every, cpu_physics_phase_release):
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    from hexcore import cuda_backend, cuda_driver_lts

    # The loop catches its own runner module's transaction error class.
    _, _, step_error = _broken_step_error(monkeypatch, forecast.proof)
    backend = _RetiringBackend(CPUArrays)
    driver = SimpleNamespace(atmosphere=Atmosphere(Carrier(np.ones(2)), Carrier(np.ones(1))),
                             cache=object(), config=SimpleNamespace(config_dynamics_split_steps=1))
    stack = {"driver": driver, "backend": backend, "physics_geometry": None}
    written = {}
    monkeypatch.setitem(sys.modules, "cupy", CPUArrays)
    monkeypatch.setattr(forecast, "_pin_for_run", lambda *args: {})
    capability = SimpleNamespace(compute=(12, 0), as_dict=lambda: {})
    monkeypatch.setattr(cuda_backend, "require_cuda", lambda **kwargs: capability)
    monkeypatch.setattr(cuda_backend, "KernelCache", lambda **kwargs: object())
    monkeypatch.setattr(forecast.proof, "gpu_memory_admission", lambda *args, **kwargs: {})
    monkeypatch.setattr(forecast.proof, "_construct_device_stack", lambda **kwargs: stack)
    monkeypatch.setattr(cuda_driver_lts, "attach_local_timestep", lambda *args, **kwargs: None)
    monkeypatch.setattr(forecast.proof, "_static_output_fields", lambda host: {})
    monkeypatch.setattr(forecast.proof, "_previous_surface_updates", lambda *args: {})
    monkeypatch.setattr(forecast.history_writer, "open_writer", lambda *args: None)
    monkeypatch.setattr(forecast.proof, "require_arwen_v2_surface_execution", lambda *args, **kwargs: {})
    monkeypatch.setattr(forecast.proof, "fingerprint_execution_boundary", lambda stack: {})
    monkeypatch.setattr(forecast.time_ledger, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(forecast, "_physics_cadence_proof", lambda *args: None)
    monkeypatch.setattr(forecast, "_mixing_treatment_proof", lambda *args: None)
    # The real digest reads the atmosphere first and then the adapter's
    # restart state; the harness atmosphere is not a full one, so the
    # digest is reduced to the read this test is about.
    from hexcore import final_identity
    monkeypatch.setattr(final_identity, "final_digest",
                        lambda stack, *args: {"physics": stack["backend"].restart_state()})
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "0")

    def install(stack, cp):
        def numeric(atmosphere, flag):
            step = int(atmosphere.state.time_seconds / forecast.DT_SECONDS)
            driver._step_gate_envelope = (atmosphere, {"step": step})
            return 0
        driver.final_validation_reader = numeric

    monkeypatch.setattr(forecast, "install_step_gate_batch", install)

    def execute(**kwargs):
        step = int(driver.atmosphere.state.time_seconds / forecast.DT_SECONDS) + 1
        if step == 2:
            # The adapter's rollback could not rebuild its seam.
            backend._phase = "broken"
            raise step_error
        endpoint = Atmosphere(Carrier(np.full(2, step), step * forecast.DT_SECONDS),
                              Carrier(np.full(1, step)))
        driver.final_validation_reader(endpoint, None)
        driver.atmosphere = endpoint
        result = receipt_result()
        result.committed = SimpleNamespace(receipt=Carrier(step), surface_updates={"value": np.full(1, step)})
        result.dynamics_tendencies = Carrier(np.full(1, step))
        return result

    monkeypatch.setattr(forecast.proof, "execute_composite_step", execute)
    monkeypatch.setattr(forecast, "step_health_gate", lambda stack, step, cp, **kwargs: {"step": step})

    def write(path, value):
        written[path.name] = value

    monkeypatch.setattr(forecast.proof, "_write_exclusive_json", write)
    monkeypatch.setattr(forecast, "BoundaryFingerprintWriter",
                        lambda path: SimpleNamespace(write=lambda *a: None, close=lambda: None, steps=[]))
    host = {"scalar_names": ("qv",), "surface_accumulators": (),
            "prepared": SimpleNamespace(expected_execution_fingerprint={"sha256": "0" * 64},
                                        mesh=SimpleNamespace(arrays={})),
            "constructor_receipt": None, "scalar_receipt": None, "mesh_evidence": None,
            "reconstruction_coefficients": None, "edge_normal_vectors": None, "defc": None,
            "case_pin_relaxation": None, "f000_start_humidity": {}}
    result = forecast.execute_forecast(
        host=host, schedule={"capture_steps": [], "labels": {}, "steps": 4, "valid_times": {},
                             "start_time": "2026-10-03_12:00:00"},
        cache_root=tmp_path, output_root=tmp_path, arwen_checkout=tmp_path,
        source_receipt={}, authority_receipt={},
        fingerprint_every=fingerprint_every, stop_on_refusal=True)

    refusal = result["refusal"]
    assert result["steps_executed"] == 1 and refusal["step"] == 2
    assert refusal["original_refusal"] == {"exception": "FloatingPointError", "message": ORIGINAL}
    assert "could not rebuild its seam" in refusal["message"]
    assert refusal["physics_seam"]["phase"] == "broken"
    assert refusal["physics_seam"]["restart_state"] is None
    assert result["final_state_digest"] is None
    if fingerprint_every:
        assert written["carried-identity.json"] == {"final_state_digest": None,
                                                    "physics_seam": refusal["physics_seam"]}
