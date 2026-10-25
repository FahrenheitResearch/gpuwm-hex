"""The suite seam takes the parent's history snapshot call.

THE BREAKAGE THIS PREVENTS: hex 0.4.4's history path asks a backend for
``history_diagnostic_snapshot()`` (falling back to ``diagnostic_snapshot``),
and the parent's version calls ``diagnostic_snapshot(selective=True,
resident=True)``.  PersistentSuiteCudaPhysicsBackendV841 overrode
``diagnostic_snapshot`` with no keywords, so every column-suite run
(rrfs_v2_dev, the HRRR-shape overrides) died with a TypeError at its hour-0
frame (2025-05-19 hex 60-15 km zoom globe).  Both calls must reach the
snapshot body; off a committed boundary that body refuses with its own
RuntimeError, which is what proves the call bound.
"""

import inspect

import pytest

from hexcore.cuda_arwen_physics_v841 import PersistentTwoPhaseCudaPhysicsBackendV841
from hexcore.cuda_suite_physics_v841 import PersistentSuiteCudaPhysicsBackendV841


def _off_boundary_backend():
    backend = object.__new__(PersistentSuiteCudaPhysicsBackendV841)
    backend._phase = "phase1"
    return backend


def test_parent_history_call_binds_on_the_suite_seam():
    signature = inspect.signature(PersistentSuiteCudaPhysicsBackendV841.diagnostic_snapshot)
    signature.bind(object(), selective=True, resident=True)
    with pytest.raises(RuntimeError, match="committed boundary"):
        _off_boundary_backend().diagnostic_snapshot(selective=True, resident=True)


def test_suite_history_snapshot_takes_the_full_host_path(monkeypatch):
    calls = []

    def record(self, **keywords):
        calls.append(keywords)
        return "snapshot"

    monkeypatch.setattr(PersistentSuiteCudaPhysicsBackendV841, "diagnostic_snapshot", record)
    backend = _off_boundary_backend()
    assert backend.history_diagnostic_snapshot() == "snapshot"
    # The suite seam exports through the host: no selective/resident pair.
    assert calls == [{}]
    assert (PersistentSuiteCudaPhysicsBackendV841.history_diagnostic_snapshot
            is not PersistentTwoPhaseCudaPhysicsBackendV841.history_diagnostic_snapshot)
