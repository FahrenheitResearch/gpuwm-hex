"""The per-step physics rollback export is taken only when a run reads it.

Before phase one of every step the physics adapter exported every persisted
seam array to the host so a refused step could be rolled back: 447 MB and
192 device-to-host copies a step on a 43,884-cell mesh, about 24 ms of the
host blocked every step.  The forecast reads that export only under
``--stop-on-refusal``, which keeps the last committed state and writes its
frame; without it a refused step ends the run.  So the forecast arms the
rollback exactly when ``--stop-on-refusal`` is given, and every other caller
(the proof harness included) keeps the armed default.

THE BREAKAGES THESE TESTS HOLD SHUT:

* an unarmed backend that fails must not pretend it rolled back: it retires
  its seam (phase ``rollback_not_armed``), publishes nothing, and refuses
  every later step instead of continuing from a seam that ran part of the
  refused step;
* the composite step must not try to verify a boundary an unarmed backend
  never had, which would bury the real refusal under a verification error;
* the proof harness keeps its rollback: the default is armed.

Everything here runs on a CPU-only box.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import cuda_arwen_physics_v841 as arwen
from _layout import PACKAGE_DIR

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_full_physics_x4.py"
FORECAST_PATH = PACKAGE_DIR / "drivers" / "run_cuda_v841_forecast.py"


def _load_runner():
    name = "_test_rollback_policy_runner"
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


# ---------------------------------------------------------------------------
# the adapter's state machine
# ---------------------------------------------------------------------------
def _bare_backend(*, armed: bool, phase: str = "boundary"):
    """The adapter with no engine behind it: only the fields the transaction
    state machine reads."""

    backend = object.__new__(arwen.PersistentTwoPhaseCudaPhysicsBackendV841)
    backend._rollback_snapshot = armed
    backend._phase = phase
    backend._boundary_snapshot = None
    backend._step_start = None
    backend._candidate_scalar_target = None
    backend._candidate_scalar_backup = None
    backend._pending_gwdo_result = None
    backend._pending_refl10cm = None
    backend._prep_geometry = None
    backend._kernel_cache = None
    backend._last_receipt = {"phase": phase}
    return backend


def test_the_backend_keeps_its_rollback_unless_told_otherwise():
    signature = inspect.signature(arwen.PersistentTwoPhaseCudaPhysicsBackendV841.__init__)
    assert signature.parameters["rollback_snapshot"].default is True


def test_an_unarmed_begin_takes_no_export_and_retires_the_seam_on_failure(monkeypatch):
    exports: list[str] = []

    class _Seam:
        elapsed_seconds = 0.0

        def export_state(self):
            exports.append("export")
            return {}

    backend = _bare_backend(armed=False)
    backend._seam = _Seam()
    backend._constructor = SimpleNamespace(dt=5.0)

    def refuse(*_args, **_kwargs):
        raise FloatingPointError("preparation refused the columns")

    monkeypatch.setattr(arwen, "prepare_mpas_to_phys_cuda_v841", refuse)
    atmosphere = SimpleNamespace(state=SimpleNamespace(time_seconds=0.0))
    with pytest.raises(FloatingPointError, match="preparation refused"):
        backend.begin_step(
            atmosphere=atmosphere,
            scalar_names=("qv", "qc", "qr", "qi", "qs", "qg"),
            dt=5.0,
        )
    assert exports == []
    assert backend._phase == "rollback_not_armed"
    assert backend._last_receipt["phase"] == "rollback_not_armed"
    with pytest.raises(RuntimeError, match="begin_step refused: no boundary export"):
        backend.begin_step(
            atmosphere=atmosphere,
            scalar_names=("qv", "qc", "qr", "qi", "qs", "qg"),
            dt=5.0,
        )


def test_an_armed_begin_still_exports_the_boundary(monkeypatch):
    exports: list[str] = []
    restored: list[object] = []

    class _Seam:
        elapsed_seconds = 0.0
        _cp = SimpleNamespace(ndarray=np.ndarray)

        def _restart_manifest(self):
            return {}

        def export_state(self):
            exports.append("export")
            return {"identity": {}, "arrays": {}, "scalars": {}}

    backend = _bare_backend(armed=True)
    backend._seam = _Seam()
    backend._constructor = SimpleNamespace(dt=5.0)
    monkeypatch.setattr(
        backend, "_restore_boundary", lambda snapshot: restored.append(snapshot)
    )

    def refuse(*_args, **_kwargs):
        raise FloatingPointError("preparation refused the columns")

    monkeypatch.setattr(arwen, "prepare_mpas_to_phys_cuda_v841", refuse)
    with pytest.raises(FloatingPointError):
        backend.begin_step(
            atmosphere=SimpleNamespace(state=SimpleNamespace(time_seconds=0.0)),
            scalar_names=("qv", "qc", "qr", "qi", "qs", "qg"),
            dt=5.0,
        )
    assert exports == ["export"]
    assert restored == [{"identity": {}, "arrays": {}, "scalars": {}}]


def test_an_unarmed_abort_restores_the_candidate_scalars_and_retires_the_seam():
    backend = _bare_backend(armed=False, phase="finished")
    backend._boundary_snapshot = arwen._ROLLBACK_NOT_ARMED
    target = np.ones(4, dtype=np.float32)
    backend._candidate_scalar_target = target
    backend._candidate_scalar_backup = np.zeros(4, dtype=np.float32)
    backend.abort_step()
    assert np.all(target == 0.0)
    assert backend._phase == "rollback_not_armed"
    assert backend._last_receipt["phase"] == "rollback_not_armed"
    assert "retired" in backend._last_receipt["rollback"]
    assert backend._boundary_snapshot is None


# ---------------------------------------------------------------------------
# the composite step
# ---------------------------------------------------------------------------
class _Backend:
    def __init__(self, log: list[str], *, finish_refuses: bool = False) -> None:
        self.log = log
        self.phase = "boundary"
        self.finish_refuses = finish_refuses

    def begin_step(self, **_: object) -> object:
        self.log.append("begin")
        self.phase = "begun"
        return object()

    def finish_step(self, **_: object) -> object:
        self.log.append("phase2")
        if self.finish_refuses:
            self.phase = "rollback_not_armed"
            raise FloatingPointError("phase-two numeric refusal")
        self.phase = "finished_unpublished"
        return object()

    def commit_step(self) -> None:
        self.log.append("adapter_commit")
        self.phase = "complete"

    def abort_step(self) -> None:
        self.log.append("adapter_abort")
        self.phase = "rollback_not_armed"

    def step_receipt(self) -> dict[str, object]:
        return {
            "schema": "fake-transaction/v1",
            "phase": self.phase,
            "transaction_rollback": {"armed": False},
        }

    def restart_state(self) -> dict[str, object]:
        raise AssertionError("an unarmed backend has no boundary to verify")


class _Driver:
    def __init__(self, log: list[str]) -> None:
        self.log = log
        start = SimpleNamespace(
            time_seconds=0.0,
            rho=np.ones((1, 1), dtype=np.float32),
            rho_u=np.ones((1, 1), dtype=np.float32),
            scalars=np.zeros((6, 1, 1), dtype=np.float32),
        )
        self.atmosphere = SimpleNamespace(state=start)
        self.horizontal = SimpleNamespace(
            recover_edge_fields=lambda *_: SimpleNamespace(
                rho_edge=np.ones((1, 1), dtype=np.float32)
            )
        )

    def step_device_with_physics(self, _: object) -> object:
        self.log.append("moist_rk")
        endpoint = SimpleNamespace(
            time_seconds=120.0, scalars=np.zeros((6, 1, 1), dtype=np.float32)
        )
        return SimpleNamespace(atmosphere=SimpleNamespace(state=endpoint))

    def commit_post_wsm6_candidate(self, candidate: object, recovery: object) -> object:
        self.log.append("driver_commit")
        return SimpleNamespace(atmosphere=candidate.atmosphere, surface_updates={})

    def abort_post_wsm6_candidate(self, _: object) -> None:
        self.log.append("driver_abort")


def _callables(log: list[str], *, recovery_fails: bool = False):
    def couple(*_: object, **__: object) -> object:
        return object()

    def clamp(*_: object, **__: object) -> object:
        return object()

    def recover(*_: object, **__: object) -> object:
        if recovery_fails:
            log.append("recover_refused")
            raise RuntimeError("recovery refused the candidate")
        return object()

    return couple, clamp, recover


def _step(runner, driver, backend, callables):
    couple, clamp, recover = callables
    return runner.execute_composite_step(
        driver=driver,
        backend=backend,
        scalar_names=runner.SCALAR_NAMES,
        physics_geometry=object(),
        kernel_cache=object(),
        previous_surface_updates=None,
        couple=couple,
        clamp=clamp,
        recover=recover,
    )


def test_an_unarmed_refusal_after_phase_two_aborts_both_owners_and_says_why(cpu_physics_phase_release):
    runner = _load_runner()
    log: list[str] = []
    with pytest.raises(runner.CompositeTransactionError, match="armed no physics rollback") as caught:
        _step(runner, _Driver(log), _Backend(log), _callables(log, recovery_fails=True))
    assert log[-2:] == ["driver_abort", "adapter_abort"]
    assert "driver_commit" not in log
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "rollback was incomplete" not in str(caught.value)


def test_an_unarmed_refusal_inside_phase_two_is_reported_not_verified(cpu_physics_phase_release):
    runner = _load_runner()
    log: list[str] = []
    with pytest.raises(runner.CompositeTransactionError, match="armed no physics rollback") as caught:
        _step(runner, _Driver(log), _Backend(log, finish_refuses=True), _callables(log))
    assert "adapter_abort" not in log
    assert isinstance(caught.value.__cause__, FloatingPointError)


# ---------------------------------------------------------------------------
# who arms it
# ---------------------------------------------------------------------------
def test_the_harness_keeps_the_rollback_by_default():
    runner = _load_runner()
    parameters = inspect.signature(runner._construct_device_stack).parameters
    assert parameters["physics_rollback_snapshot"].default is True
    source = inspect.getsource(runner._construct_device_stack)
    assert "rollback_snapshot=bool(physics_rollback_snapshot)" in source


def test_the_forecast_arms_the_rollback_exactly_under_stop_on_refusal():
    source = FORECAST_PATH.read_text(encoding="utf-8")
    assert "physics_rollback_snapshot=bool(stop_on_refusal)," in source
    assert '"physics_rollback": {' in source
