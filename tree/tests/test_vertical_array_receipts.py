"""Vertical receipt metadata preserves its original binary64 authority."""

import hashlib
import json

import numpy as np
import pytest

from hexcore.vertical_spec import _raw_array_receipt


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", ">f4", "i8", "u8"])
@pytest.mark.parametrize("kind", ["finite", "nonfinite", "zeros", "empty", "scalar"])
def test_receipt_keeps_hash_and_binary64_extrema(dtype, kind):
    if kind == "finite":
        values = [0, 1, 2**53 - 1, 2**53 + 1] if dtype[-2:] in ("i8", "u8") else [1, -2, 3, 4]
    elif kind == "nonfinite":
        if dtype[-2:] in ("i8", "u8"):
            pytest.skip("integer fields have no nonfinite values")
        values = [np.nan, np.inf, -np.inf, 1]
    elif kind == "zeros":
        values = [-0., 0., -0., 0.]
    elif kind == "empty":
        values = []
    else:
        values = 1
    array = np.asarray(values, dtype=dtype)
    if array.size == 4:
        array = array.reshape(2, 2).T[::-1]
    value = np.ascontiguousarray(array)
    converted = np.asarray(value, dtype=np.float64).ravel()
    finite = np.isfinite(converted)
    header = (str(value.dtype).encode("ascii") + b"\0" +
              json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
    expected = dict(shape=list(value.shape), dtype=str(value.dtype),
                    raw_sha256=hashlib.sha256(header + value.tobytes()).hexdigest(),
                    finite=bool(np.all(finite)),
                    minimum=float(np.min(converted[finite])) if np.any(finite) else None,
                    maximum=float(np.max(converted[finite])) if np.any(finite) else None)
    before = array.tobytes()
    assert json.dumps(_raw_array_receipt(array), sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert array.tobytes() == before


@pytest.mark.parametrize("length", [1, 7, 32, 129, 8193])
def test_zero_extrema_keep_the_binary64_reduction_sign(length):
    values = np.resize(np.array([-0., 0., 0., -0.], dtype="f4"), length)
    actual = _raw_array_receipt(values)
    expected = values.astype("f8")
    assert np.signbit(actual["minimum"]) == np.signbit(np.min(expected))
    assert np.signbit(actual["maximum"]) == np.signbit(np.max(expected))


def test_no_finite_values_have_no_extrema():
    receipt = _raw_array_receipt(np.array([np.nan, np.inf, -np.inf], dtype="f4"))
    assert receipt["finite"] is False
    assert receipt["minimum"] is None and receipt["maximum"] is None
