"""The refused-step value restore for graph runs (F1 follow-through, 2026-10-15).

Breakage prevented: a --stop-on-refusal forecast with CUDA graphs ending on the
refused step's dynamics. Graph replay returns its own output buffers, so the
committed state shares memory with the graph and the refused step's replays
overwrite it; the transaction restores the start-state identity only. Measured
on x1.40962 with a step refused after its dycore: final-state digest 3c1062ef
under graphs against 101d8174 under eager before the restore, 101d8174 in both
after it. numpy stands in for cupy.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.cuda_backend.containers import (
    DeviceAtmosphere,
    DevicePrognosticState,
    DeviceSavedDiagnostics,
    TransferStats,
)
from hexcore.cuda_seam_snapshot_v841 import TwoDeepForecastSnapshotV841
from hexcore.drivers.run_cuda_v841_forecast import restore_refused_dycore_values


def _atmosphere(value: float) -> DeviceAtmosphere:
    plane = lambda: np.full((2, 3), value, dtype=np.float32)  # noqa: E731
    transfer = TransferStats(0, 0.0)
    state = DevicePrognosticState(plane(), plane(), plane(), plane(),
                                  np.full((1, 2, 3), value, np.float32), 0.0,
                                  np.dtype(np.float32), transfer)
    saved = DeviceSavedDiagnostics(*(plane() for _ in range(7)), np.dtype(np.float32), transfer)
    return DeviceAtmosphere(None, state, None, None, saved, None, transfer)


def _snapshot(driver, committed_step, committed):
    snapshot = TwoDeepForecastSnapshotV841.__new__(TwoDeepForecastSnapshotV841)
    snapshot.cp = SimpleNamespace(ndarray=np.ndarray, copyto=np.copyto)
    snapshot.stack = {"driver": driver}
    snapshot.slots = [{}, {}]
    snapshot.slots[committed_step % 2] = {"step": committed_step, "atmosphere": committed}
    return snapshot


def test_values_come_back_and_identities_stay():
    live = _atmosphere(7.0)   # the refused step's replays wrote 7s into the committed buffers
    driver = SimpleNamespace(atmosphere=live)
    rho, theta = live.state.rho, live.saved.theta_m
    restore_refused_dycore_values(_snapshot(driver, 4, _atmosphere(1.0)), 4)
    assert driver.atmosphere is live and live.state.rho is rho and live.saved.theta_m is theta
    checked = 0
    for part in (live.state, live.saved):
        for field in fields(part):
            value = getattr(part, field.name)
            if isinstance(value, np.ndarray):
                assert np.all(value == 1.0), field.name
                checked += 1
    assert checked == 12


def test_a_slot_that_holds_another_step_refuses():
    driver = SimpleNamespace(atmosphere=_atmosphere(7.0))
    with pytest.raises(RuntimeError, match="lost committed boundary 5"):
        restore_refused_dycore_values(_snapshot(driver, 3, _atmosphere(1.0)), 5)
