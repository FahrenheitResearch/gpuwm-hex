"""Batch words equal scalar words from the selected fast or strict authority."""
import shutil

import numpy as np
import pytest

from hexcore import native_geometry_math as geometry, numerics
from hexcore import native_libm_batch as batch


@pytest.fixture(params=["fast", "strict"], autouse=True)
def geometry_mode(request, monkeypatch):
    monkeypatch.setattr(numerics, "_mode", request.param)


def authority(symbol, nargs=1):
    return (geometry._strict_intrinsic(symbol, nargs) if numerics.is_strict()
            else getattr(geometry._LIBM, symbol))


def scalar(values):
    function = authority("sinf")
    return np.fromiter((function(float(x)) for x in values.flat),
                       dtype=np.float32, count=values.size).reshape(values.shape)


@pytest.mark.parametrize("workers", [1, 4])
def test_finite_bits_and_unequal_worker_slices(monkeypatch, workers):
    if geometry._LIBM is None or shutil.which("rustc") is None:
        pytest.skip("GNU libm and Rust bridge required")
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", str(workers))
    rng = np.random.default_rng(817)
    bits = rng.integers(0, 2**32, 300007, dtype=np.uint32)
    # Both signs, subnormals, largest finite values and damping phase endpoints.
    bits[:10] = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                 0x7f7fffff, 0xff7fffff, 0x3fc90fdb, 0x40490fdb]
    values = bits.view(np.float32)
    values = values[np.isfinite(values)]
    got = batch.sin_batch(authority("sinf"), values)
    assert got is not None
    np.testing.assert_array_equal(got.view(np.uint32), scalar(values).view(np.uint32))


@pytest.mark.parametrize("layout", ["strided", "fortran", "broadcast"])
def test_array_layout_and_nonfinite_payload_fallback(layout):
    if geometry._LIBM is None:
        pytest.skip("GNU libm required")
    values = np.linspace(-4, 4, 8192, dtype=np.float32).reshape(64, 128)
    if layout == "strided":
        values = values[:, ::-1]
    elif layout == "fortran":
        values = np.asfortranarray(values)
    else:
        values = np.broadcast_to(values[0], values.shape)
    np.testing.assert_array_equal(geometry.sin(values).view(np.uint32), scalar(values).view(np.uint32))
    values = values.copy()
    values.view(np.uint32).flat[:4] = [0x7fc01234, 0xff801234, 0x7f800000, 0xff800000]
    assert batch.sin_batch(geometry._LIBM.sinf, values.reshape(-1)) is None
    np.testing.assert_array_equal(geometry.sin(values).view(np.uint32), scalar(values).view(np.uint32))


def test_missing_compiler_preserves_scalar_path(monkeypatch, tmp_path):
    if geometry._LIBM is None:
        pytest.skip("GNU libm required")
    batch._library.cache_clear()
    monkeypatch.setenv("HEXCORE_NATIVE_CPU_CACHE", str(tmp_path))
    # Compile the strict authority before removing only the optional batch compiler.
    function = authority("sinf")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    try:
        values = np.linspace(0, 2, 4097, dtype=np.float32)
        assert batch.sin_batch(function, values) is None
        np.testing.assert_array_equal(geometry.sin(values).view(np.uint32), scalar(values).view(np.uint32))
    finally:
        batch._library.cache_clear()


def test_scalar_empty_and_float64_keep_original_routes():
    assert geometry.sin(np.empty((2, 0), np.float32)).shape == (2, 0)
    x = np.linspace(-10, 10, 5000, dtype=np.float64)
    np.testing.assert_array_equal(geometry.sin(x).view(np.uint64), np.sin(x).view(np.uint64))
    if geometry._LIBM is not None:
        for x in [np.float32(0), np.float32(-0.0), np.float32(1.1)]:
            assert geometry.sin(x).tobytes() == np.float32(authority("sinf")(float(x))).tobytes()


@pytest.mark.parametrize("name,symbol", [("cos", "cosf"), ("asin", "asinf"),
                                        ("atan2", "atan2f")])
def test_other_intrinsics_use_exact_libm_words(monkeypatch, name, symbol):
    if geometry._LIBM is None or shutil.which("rustc") is None:
        pytest.skip("GNU libm and Rust bridge required")
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "4")
    rng = np.random.default_rng(842)
    bits = rng.integers(0, 2**32, 150007, dtype=np.uint32)
    bits[:8] = [0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff,
                0x3f800000, 0xbf800000]
    left = bits.view(np.float32)
    left = left[np.isfinite(left)]
    right = left[::-1].copy()
    function = authority(symbol, 2 if name == "atan2" else 1)
    values = (left, right) if name == "atan2" else (left,)
    expected = np.fromiter((function(*(float(v[i]) for v in values))
                            for i in range(left.size)), dtype=np.float32)
    actual = getattr(geometry, name)(*values)
    assert actual.tobytes() == expected.tobytes()
