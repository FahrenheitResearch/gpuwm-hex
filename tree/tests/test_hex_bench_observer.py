"""The common observer must retain every declared carried-state member."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def observer():
    path = Path(__file__).resolve().parents[1] / "tools/hex_timeline/bench_case.py"
    spec = importlib.util.spec_from_file_location("hex_bench_observer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def carriers():
    from hexcore.cuda_dualrun import STATE_FIELDS, SIDECAR_FIELDS
    state = SimpleNamespace(**{name: np.zeros((2, 3), np.float32) for name in STATE_FIELDS}, time_seconds=10800.)
    saved = SimpleNamespace(**{name: np.ones((2, 3), np.float32) for name in SIDECAR_FIELDS})
    driver = SimpleNamespace(atmosphere=SimpleNamespace(state=state, saved=saved))
    restart = {"fields": {"rain": np.ones(2, np.float32)}, "step": 2160}
    backend = SimpleNamespace(restart_state=lambda: restart)
    result = SimpleNamespace(dynamics_tendencies={"u": np.ones(3, np.float32)},
        committed=SimpleNamespace(surface_updates={"soil": np.ones(4, np.float32)}))
    return driver, backend, result, restart


def test_observer_uses_real_canonical_digest_and_all_carrier_groups():
    driver, backend, result, restart = carriers()
    digest = observer().carried_digest(driver, backend, result)
    assert digest["schema"] == "gpuwm-canonical-state-v2"
    for array in (driver.atmosphere.state.rho, driver.atmosphere.saved.theta_m,
                  restart["fields"]["rain"], result.dynamics_tendencies["u"],
                  result.committed.surface_updates["soil"]):
        array.flat[0] += 1
        changed = observer().carried_digest(driver, backend, result)
        assert changed["sha256"] != digest["sha256"]
        array.flat[0] -= 1
        assert observer().carried_digest(driver, backend, result) == digest
    restart["step"] += 1
    assert observer().carried_digest(driver, backend, result)["sha256"] != digest["sha256"]


def test_observer_refuses_an_unregistered_carrier():
    driver, backend, result, restart = carriers()
    restart["opaque"] = object()
    with pytest.raises(TypeError, match="unregistered carried state backend/opaque"):
        observer().carried_digest(driver, backend, result)
