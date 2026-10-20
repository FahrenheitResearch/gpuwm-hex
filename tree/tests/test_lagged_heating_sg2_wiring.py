"""SG2 wiring: the lagged-heating carry reaches the dycore, survives restart,
and exists on every backend that can run a step.

The formula tests (test_lagged_heating_sg2.py) prove the producer's
arithmetic.  These prove the plumbing around it with the project's own
objects and fakes standing in for the card (the kernels themselves are
exercised on a card by test_lagged_heating_sg2_kernels.py):

* the suite / generic-seam backend's __init__ sets every slot the frozen
  __init__ sets, and its restart_state() works before the first committed
  step (verifier 2026-10-13: it raised AttributeError '_lagged_diabatic_theta'
  on every suite row at the step-0 boundary fingerprint);
* finish -> commit publishes the carry, abort drops it, restart round-trips
  it, a pre-SG2 payload resumes with none, a wrong plane is refused;
* both step drivers hand the backend's committed carry to the dycore
  (deleting either attach line fails here);
* a selection without microphysics carries nothing (native calls the
  producer only when config_microp_scheme /= 'off');
* on a limited-area run the specified-zone zeroing sits after the
  specified-zone tendency and before the final recovery.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from _layout import PACKAGE_DIR

from hexcore import cuda_arwen_physics_v841 as arwen
from hexcore import cuda_suite_physics_v841 as suite

F32 = np.float32
WSM6 = ("qv", "qc", "qr", "qi", "qs", "qg")


# ---------------------------------------------------------------- helpers
def _init_slots(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    slots: set[str] = set()
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Assign):
                            targets = sub.targets
                        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                            targets = [sub.target]
                        else:
                            continue
                        for target in targets:
                            if (isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"):
                                slots.add(target.attr)
                    return slots
    raise AssertionError(f"{class_name}.__init__ not found in {path}")


class _NumpyCupy(types.ModuleType):
    """Stands in for cupy in the restart path, so no card is touched."""

    float32 = np.float32

    def __init__(self) -> None:
        super().__init__("cupy")

    @staticmethod
    def asnumpy(value):
        return np.array(value, copy=True)

    @staticmethod
    def asarray(value):
        return np.array(value, copy=True)


@pytest.fixture
def numpy_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", _NumpyCupy())


class _Seam:
    _cfg = object()
    step_index = 0
    elapsed_seconds = 0.0

    def export_state(self):
        return {"seam": "exported"}

    def restore_state(self, payload):
        self.restored = payload


def _boundary_backend(cls, *, n_levels=2, n_columns=3):
    """A backend at a step boundary, with only what the lifecycle reads."""

    backend = object.__new__(cls)
    backend._constructor = types.SimpleNamespace(n_levels=n_levels, n_columns=n_columns)
    backend._seam = _Seam()
    backend._phase = "boundary"
    backend._boundary_snapshot = None
    backend._step_start = None
    backend._candidate_scalar_target = None
    backend._candidate_scalar_backup = None
    backend._pending_gwdo_result = None
    backend._last_gwdo_result = None
    backend._pending_refl10cm = None
    backend._committed_refl10cm = None
    backend._gwdo_static = None
    backend._gwdo_calls = 0
    backend._last_receipt = {"phase": "boundary"}
    backend._pending_lagged_diabatic_theta = None
    backend._lagged_diabatic_theta = None
    backend._lagged_heating_restored = None
    return backend


@pytest.fixture
def stub_identity(monkeypatch):
    monkeypatch.setattr(arwen.PersistentTwoPhaseCudaPhysicsBackendV841,
                        "_restart_identity", lambda self: {"identity": "fixed"})
    monkeypatch.setattr(suite.PersistentSuiteCudaPhysicsBackendV841,
                        "_restart_identity", lambda self: {"identity": "fixed"})
    for cls in (arwen.PersistentTwoPhaseCudaPhysicsBackendV841,
                suite.PersistentSuiteCudaPhysicsBackendV841):
        monkeypatch.setattr(cls, "_new_seam", lambda self: _Seam())
        monkeypatch.setattr(cls, "_private_binding_guard", lambda self: None)
        monkeypatch.setattr(cls, "_base_receipt", lambda self: {})


def _finished(backend, pending):
    """The state finish_step leaves: staged, unpublished, carry pending."""

    backend._phase = "finished"
    backend._boundary_snapshot = arwen._ROLLBACK_NOT_ARMED
    backend._candidate_scalar_target = np.zeros(1, F32)
    backend._candidate_scalar_backup = np.zeros(1, F32)
    backend._pending_lagged_diabatic_theta = pending


# ------------------------------------------------- suite backend slots (1)
def test_suite_init_sets_every_slot_the_frozen_init_sets():
    """The suite __init__ does not call the frozen one; any slot it forgets
    is an AttributeError on the first method that reads it.  Fails before
    the SG2 repair (the three lagged-heating slots were missing)."""

    frozen = _init_slots(PACKAGE_DIR / "cuda_arwen_physics_v841.py",
                         "PersistentTwoPhaseCudaPhysicsBackendV841")
    bound = _init_slots(PACKAGE_DIR / "cuda_suite_physics_v841.py",
                        "PersistentSuiteCudaPhysicsBackendV841")
    assert {"_pending_lagged_diabatic_theta", "_lagged_diabatic_theta",
            "_lagged_heating_restored"} <= frozen
    assert frozen - bound == set(), sorted(frozen - bound)


def test_suite_restart_state_before_the_first_commit(monkeypatch, numpy_cupy, stub_identity):
    """The real suite __init__, then restart_state() at the step-0 boundary
    (the boundary fingerprint, samehost identity and cycling step-0 capture
    all call it there).  Raised AttributeError before the repair."""

    selection = {key: "x" for key in suite._SELECTOR_KEYS}
    constructor = object.__new__(suite.SealedSuiteConstructorV841)
    object.__setattr__(constructor, "_seal", suite._CONSTRUCTOR_SEAL)
    object.__setattr__(constructor, "_values", {
        **selection, "microphysics_scheme": "wsm6", "n_columns": 3, "n_levels": 2})
    geometry = object.__new__(suite.CudaMpasToPhysGeometryV841)
    object.__setattr__(geometry, "n_cells", 3)
    monkeypatch.setattr(suite.CudaMpasToPhysGeometryV841, "validate", lambda self: None)
    monkeypatch.setattr(suite, "physics_suite",
                        lambda profile_id: {"id": profile_id, "selection": dict(selection)})
    monkeypatch.setattr(suite, "_load_suite_factory",
                        lambda checkout, selection, scheme: (None, Path("."), object()))
    import hexcore.cuda_arwen_helper_scope_v2 as helper_scope
    monkeypatch.setattr(helper_scope, "bind_arwen_helpers_v2", lambda root: None)

    from hexcore import kernel_prewarm
    static_runs = []
    monkeypatch.setattr(kernel_prewarm, "start_static", static_runs.append)

    backend = suite.PersistentSuiteCudaPhysicsBackendV841(
        constructor=constructor, prep_geometry=geometry, kernel_cache=None, profile_id="p")
    assert backend.lagged_diabatic_theta is None
    assert static_runs == [backend._seam._cfg]
    payload = backend.restart_state()
    assert payload["schema"] == suite.CUDA_SUITE_PHYSICS_V841_SCHEMA
    assert payload["lagged_heating"] == {"rt_diabatic_tend": None}


# --------------------------------------------------- lifecycle (finish/commit/abort)
@pytest.mark.parametrize("cls", [arwen.PersistentTwoPhaseCudaPhysicsBackendV841,
                                 suite.PersistentSuiteCudaPhysicsBackendV841])
def test_commit_publishes_and_abort_drops_the_carry(cls, stub_identity):
    backend = _boundary_backend(cls)
    first = np.full((2, 3), F32(0.25))
    _finished(backend, first)
    assert backend.lagged_diabatic_theta is None      # staged, not yet carried
    backend.commit_step()
    assert backend.lagged_diabatic_theta is first
    assert backend._pending_lagged_diabatic_theta is None

    second = np.full((2, 3), F32(-1.0))
    _finished(backend, second)
    backend.abort_step()                              # the retried step keeps `first`
    assert backend.lagged_diabatic_theta is first
    assert backend._pending_lagged_diabatic_theta is None


def test_a_commit_with_nothing_staged_keeps_the_carry(stub_identity):
    """Microphysics off stages no rate; the carry it began with is kept,
    as native's tend pool keeps rt_diabatic_tend when nothing writes it."""

    backend = _boundary_backend(arwen.PersistentTwoPhaseCudaPhysicsBackendV841)
    first = np.full((2, 3), F32(0.5))
    backend._lagged_diabatic_theta = first
    _finished(backend, None)
    backend.commit_step()
    assert backend.lagged_diabatic_theta is first


# ---------------------------------------------------------- restart round trip
@pytest.mark.parametrize("cls", [arwen.PersistentTwoPhaseCudaPhysicsBackendV841,
                                 suite.PersistentSuiteCudaPhysicsBackendV841])
def test_restart_round_trips_the_carry(cls, numpy_cupy, stub_identity):
    source = _boundary_backend(cls)
    plane = np.arange(6, dtype=F32).reshape(2, 3) - F32(2.5)
    _finished(source, plane)
    source.commit_step()
    payload = source.restart_state()
    written = payload["lagged_heating"]["rt_diabatic_tend"]
    assert written.dtype == F32 and written.tobytes() == plane.tobytes()

    resumed = _boundary_backend(cls)
    resumed.restore_restart_state(payload)
    assert resumed.lagged_diabatic_theta.tobytes() == plane.tobytes()
    assert resumed._lagged_heating_restored is True
    assert resumed._pending_lagged_diabatic_theta is None


def test_a_pre_sg2_payload_resumes_with_no_carry(numpy_cupy, stub_identity):
    cls = arwen.PersistentTwoPhaseCudaPhysicsBackendV841
    payload = dict(_boundary_backend(cls).restart_state())
    del payload["lagged_heating"]
    resumed = _boundary_backend(cls)
    resumed._lagged_diabatic_theta = np.ones((2, 3), F32)   # whatever it held before
    resumed.restore_restart_state(payload)
    assert resumed.lagged_diabatic_theta is None
    assert resumed._lagged_heating_restored is False


@pytest.mark.parametrize("plane", [np.zeros((3, 2), F32), np.zeros((2, 3), np.float64)])
def test_a_restart_plane_of_another_shape_or_type_is_refused(plane, numpy_cupy, stub_identity):
    cls = arwen.PersistentTwoPhaseCudaPhysicsBackendV841
    payload = dict(_boundary_backend(cls).restart_state())
    payload["lagged_heating"] = {"rt_diabatic_tend": plane}
    with pytest.raises(ValueError, match="rt_diabatic_tend must be float32"):
        _boundary_backend(cls).restore_restart_state(payload)


# ------------------------------------------------------- microphysics off
def test_only_the_generic_seam_with_microphysics_off_runs_none():
    frozen = object.__new__(arwen.PersistentTwoPhaseCudaPhysicsBackendV841)
    assert frozen.runs_microphysics() is True
    bound = object.__new__(suite.PersistentSuiteCudaPhysicsBackendV841)
    bound._constructor = types.SimpleNamespace(physics=None)
    assert bound.runs_microphysics() is True
    bound._constructor = types.SimpleNamespace(physics={"mp_physics": 8})
    assert bound.runs_microphysics() is True
    bound._constructor = types.SimpleNamespace(physics={"mp_physics": 0})
    assert bound.runs_microphysics() is False


def test_the_producer_is_gated_on_microphysics():
    source = (PACKAGE_DIR / "cuda_arwen_physics_v841.py").read_text(encoding="utf-8")
    body = source.split("def finish_step", 1)[1].split("def abort_step", 1)[0]
    gate = body.index("if self.runs_microphysics():")
    assert gate < body.index("lagged_theta_m_rate(") < body.index("self._pending_lagged_diabatic_theta = pending_lagged")


# ------------------------------------------------------- the two attach sites
@dataclass(frozen=True)
class _Held:
    rho_theta: object = None
    diabatic_theta: object = None


class _State:
    def __init__(self, t):
        self.time_seconds = t
        self.scalars = np.zeros((6, 2, 3), F32)
        self.rho = self.rho_u = None


class _Atmosphere:
    def __init__(self, t):
        self.state = _State(t)


class _Driver:
    def __init__(self, t0, dt):
        self.atmosphere = _Atmosphere(t0)
        self.dt = dt
        self.horizontal = types.SimpleNamespace(
            recover_edge_fields=lambda rho, rho_u: types.SimpleNamespace(rho_edge=None))
        self.seen = []

    def step_device_with_physics(self, held):
        self.seen.append(held)
        return types.SimpleNamespace(atmosphere=_Atmosphere(self.atmosphere.state.time_seconds + self.dt),
                                     dynamics_tendencies=None)

    def commit_post_wsm6_candidate(self, candidate, recovery):
        self.atmosphere = candidate.atmosphere
        return candidate

    def abort_post_wsm6_candidate(self, candidate):  # pragma: no cover - not reached
        raise AssertionError("no abort expected")


class _Backend:
    def __init__(self, carry):
        self._carry = carry
        self._receipt = {"phase": "boundary"}

    @property
    def lagged_diabatic_theta(self):
        return self._carry

    def step_receipt(self):
        return dict(self._receipt)

    def begin_step(self, **_):
        self._receipt = {"phase": "begun"}
        return "raw"

    def finish_step(self, **_):
        self._receipt = {"phase": "finished_unpublished"}
        return "update"

    def commit_step(self):
        self._receipt = {"phase": "complete"}


@pytest.mark.parametrize("carry_present", [True, False])
def test_the_lts_class_step_hands_the_carry_to_the_dycore(monkeypatch, carry_present):
    import hexcore.cuda_physics_v841 as physics
    from hexcore.drivers import fullstep_lts_forecast_v841 as lts

    monkeypatch.setattr(physics, "couple_raw_column_physics_v841", lambda raw, **_: _Held())
    monkeypatch.setattr(physics, "clamp_wsm6_scalars_in_place_v841", lambda *a, **k: None)
    monkeypatch.setattr(physics, "recover_post_rk_wsm6_state_v841", lambda *a, **k: "recovery")
    carry = np.full((2, 3), F32(0.125)) if carry_present else None
    driver = _Driver(0.0, 37.5)
    lts.class_composite_step(
        driver=driver, backend=_Backend(carry), scalar_names=WSM6, physics_geometry=None,
        kernel_cache=None, dt=37.5, previous_surface_updates=None, dynamics_tendencies=None,
        refl_10cm_due=False, before_phase_two=lambda atmosphere: None)
    (held,) = driver.seen
    assert held.diabatic_theta is carry


@pytest.mark.parametrize("carry_present", [True, False])
def test_the_full_step_hands_the_carry_to_the_dycore(carry_present, cpu_physics_phase_release):
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as x4

    carry = np.full((2, 3), F32(-0.5)) if carry_present else None
    driver = _Driver(0.0, x4.DT_SECONDS)
    result = x4.execute_composite_step(
        driver=driver, backend=_Backend(carry), scalar_names=WSM6, physics_geometry=None,
        kernel_cache=None, previous_surface_updates=None,
        couple=lambda raw, **_: _Held(), clamp=lambda *a, **k: None,
        recover=lambda *a, **k: "recovery")
    (held,) = driver.seen
    assert held.diabatic_theta is carry
    assert result.recovery == "recovery"


# ------------------------------------------------- limited-area ordering
def test_the_specified_zone_zeroing_sits_between_the_tendency_and_the_recovery():
    """F:7949 zeroes rt_diabatic_tend inside the specified-zone tendency
    routine of every RK stage; the stage-3 recovery (F:4436-4437) then
    removes nothing there.  The hex stage loop must keep that order."""

    source = (PACKAGE_DIR / "cuda_driver.py").read_text(encoding="utf-8")
    adjust = source.index("self.regional_v841.adjust_dynamics_tendencies(")
    zero = source.index("self._zero_speczone_lagged_heating(diabatic_theta)")
    recover = source.index("diabatic_theta=diabatic_theta, large_timestep=stage.large_timestep")
    add = source.index('"add_lagged_diabatic_theta_v842_f32", self.ncells,')
    assert add < adjust < zero < recover
    kernel = source.split("zero_speczone_lagged_diabatic_theta_v842_f32(", 1)[1].split("}", 1)[0]
    assert "diabatic_theta[C2(k, spec_cells[slot], ncells)] = 0.0f;" in kernel
