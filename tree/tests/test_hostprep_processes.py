"""Deformation weights computed across worker processes are the serial loop's bytes.

THE BREAKAGE THIS PREVENTS: a fresh box spent about 71 s of its first forecast's
start in ``initialize_deformation_weights_v841`` on one core (x1.163842, 2026-10-16);
the process path removes that, and these tests hold it to the serial bytes and to
the serial loop's first-failing-row refusal.
"""
from __future__ import annotations

import numpy as np
import pytest

from hexcore import hostprep_processes, mixing_v841, numerics
from hexcore.mixing_v841 import initialize_deformation_weights_v841
from test_mesh_validation import graded_sphere_mesh

NAMES = ("coef_c2", "coef_s2", "coef_cs", "coef_c", "coef_s")


def _weights(mesh, monkeypatch, processes):
    monkeypatch.setenv(hostprep_processes.ENVIRONMENT, str(processes))
    return initialize_deformation_weights_v841.__wrapped_prep_cache__(mesh)


def _bits(value):
    value = np.ascontiguousarray(value)
    return value.dtype.str, value.shape, value.tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_process_rows_are_the_serial_bytes(monkeypatch, dtype):
    mesh = graded_sphere_mesh(level=5, dtype=dtype)  # 10,242 cells: 4 processes
    monkeypatch.setenv(hostprep_processes.ENVIRONMENT, "4")
    assert hostprep_processes.process_count(np.asarray(mesh.nEdgesOnCell).size) == 4
    serial = _weights(mesh, monkeypatch, 1)
    spread = _weights(mesh, monkeypatch, 4)
    for name in NAMES:
        assert _bits(getattr(spread, name)) == _bits(getattr(serial, name)), name


def test_a_worker_refusal_is_the_first_failing_row(monkeypatch):
    mesh = graded_sphere_mesh(level=5)
    counts = np.asarray(mesh.nEdgesOnCell).copy()
    late, early = counts.size - 10, counts.size // 2 + 7  # two bad rows, in different worker slices
    counts[late] = 2
    counts[early] = 2
    mesh.nEdgesOnCell = counts
    messages = []
    for processes in (1, 4):
        with pytest.raises(ValueError) as caught:
            _weights(mesh, monkeypatch, processes)
        messages.append(str(caught.value))
    assert messages[0] == messages[1] == f"cell {early} has invalid nEdgesOnCell 2"


def test_a_rebound_libm_hook_stays_in_process(monkeypatch):
    # A rebound fast hook cannot be delegated to an unmodified worker.
    monkeypatch.setattr(numerics, "_mode", "fast")
    mesh = graded_sphere_mesh(level=5)
    monkeypatch.setenv(hostprep_processes.ENVIRONMENT, "4")
    calls = []
    original = mixing_v841.geometry_math.sin

    def hooked(*values):
        calls.append(1)
        return original(*values)

    monkeypatch.setattr(mixing_v841.geometry_math, "sin", hooked)
    initialize_deformation_weights_v841.__wrapped_prep_cache__(mesh)
    assert calls, "a worker process cannot see a hook bound in this process"


def test_the_process_count_follows_the_rows(monkeypatch):
    monkeypatch.delenv(hostprep_processes.ENVIRONMENT, raising=False)
    assert hostprep_processes.process_count(hostprep_processes.MIN_ROWS_PER_PROCESS - 1) == 1
    monkeypatch.setenv(hostprep_processes.ENVIRONMENT, "0")
    with pytest.raises(ValueError):
        hostprep_processes.process_count(10**6)
