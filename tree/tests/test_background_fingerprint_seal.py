"""The prepared-input fingerprint runs on a thread and is joined before use."""

from __future__ import annotations

from types import SimpleNamespace
import threading

import pytest

from hexcore import cuda_dualrun
from hexcore.drivers import run_cuda_v841_forecast as forecast
from hexcore.drivers import samehost_forecast_v841 as samehost


def _prepared():
    return SimpleNamespace(expected_execution_fingerprint=None)


def test_join_stores_the_unchanged_fingerprint(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fingerprint(prepared, config):
        started.set()
        release.wait(5)
        return {"sha256": "abc", "config": config}

    monkeypatch.setattr(cuda_dualrun, "fingerprint_prepared_execution", fingerprint)
    prepared = _prepared()
    seal = cuda_dualrun.BackgroundFingerprintSeal(prepared, "config")
    assert started.wait(5)
    assert prepared.expected_execution_fingerprint is None  # never half-built
    release.set()
    assert seal.join() is prepared
    assert prepared.expected_execution_fingerprint == {"sha256": "abc", "config": "config"}
    seal.join()  # idempotent
    assert prepared.expected_execution_fingerprint["sha256"] == "abc"


def test_join_reraises_the_validation_refusal(monkeypatch):
    def refuse(prepared, config):
        raise cuda_dualrun.CudaDualRunError("execution-input arrays must be finite")

    monkeypatch.setattr(cuda_dualrun, "fingerprint_prepared_execution", refuse)
    prepared = _prepared()
    seal = cuda_dualrun.BackgroundFingerprintSeal(prepared, "config")
    with pytest.raises(cuda_dualrun.CudaDualRunError, match="must be finite"):
        seal.join()
    assert prepared.expected_execution_fingerprint is None


def test_a_sealed_input_is_not_sealed_twice():
    with pytest.raises(cuda_dualrun.CudaDualRunError, match="already sealed"):
        cuda_dualrun.BackgroundFingerprintSeal(SimpleNamespace(expected_execution_fingerprint={}), None)


def test_driver_join_is_idempotent_and_tolerates_no_seal(monkeypatch):
    joined = []
    host = {"prepared_fingerprint_seal": SimpleNamespace(join=lambda: joined.append(1))}
    forecast.join_prepared_fingerprint(host)
    forecast.join_prepared_fingerprint({})
    assert joined == [1]


def test_samehost_adapter_defers_the_join_only_while_installed():
    assert forecast.DEFER_PREPARED_FINGERPRINT is False
    with samehost.installed(forecast, devices=(0, 1), layouts_path=None, part_path=None):
        assert forecast.DEFER_PREPARED_FINGERPRINT is True
    assert forecast.DEFER_PREPARED_FINGERPRINT is False
