"""Parallel boundary fingerprints are the same bytes as the serial walk (unified image single-core fix, 2026-10-16)."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof


def _copying_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _payload():
    rng = np.random.default_rng(7)
    return {
        "theta": rng.random((55, 400), dtype=np.float32),
        "strided": rng.random((40, 30))[:, ::3],
        "fortran": np.asfortranarray(rng.random((9, 11))),
        "big_endian": np.arange(12, dtype=">f8").reshape(3, 4),
        "empty": np.zeros((0, 5), np.float32),
        "flags": rng.random(17) > 0.5,
        "objects": np.array(["a", 1, None], dtype=object),
        "nested": {"b": [np.arange(5, dtype=np.int32), 3.5, "x"], "a": np.float32(2.0)},
        "scalar": 4,
    }


def test_array_digest_is_the_copying_digest():
    for value in list(_payload().values())[:7]:
        assert proof.array_sha256(value) == _copying_sha256(value)


def test_threaded_fingerprint_matches_one_thread(monkeypatch):
    many = proof.fingerprint_nested_arrays(_payload(), prefix="backend")
    monkeypatch.setenv("GPUWM_HEX_FINGERPRINT_THREADS", "1")
    one = proof.fingerprint_nested_arrays(_payload(), prefix="backend")
    assert many == one
    assert json.dumps(many, sort_keys=True) == json.dumps(one, sort_keys=True)
    assert many["arrays"]["backend/theta"]["sha256"] == _copying_sha256(_payload()["theta"])
