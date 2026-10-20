"""FP32 cell-length batching must retain the native scalar oracle."""
import numpy as np
import pytest

from hexcore.cuda_gwdo_v841 import native_cell_dx_m, _div, _pow


def scalar_authority(density, length):
    md = np.ascontiguousarray(np.asarray(density, dtype=np.float32))
    return np.array([_div(np.float32(length), _pow(x, np.float32(.25)))
                     for x in md], dtype=np.float32)


@pytest.mark.parametrize("length", [np.nextafter(np.float32(0), np.float32(1)),
                                   3000., 30000., 60000., np.finfo(np.float32).max])
def test_cell_dx_random_binary32_and_extremes(length):
    rng = np.random.default_rng(51981)
    words = rng.integers(1, 0x3f800001, size=100003, dtype=np.uint32)
    # Every exponent, denormal/normal boundary, adjacent values around one.
    words[:128] = np.arange(128, dtype=np.uint32) << 23
    words[0] = 1
    words[128:135] = [1, 2, 0x7fffff, 0x800000, 0x800001, 0x3f7fffff, 0x3f800000]
    md = words.view(np.float32)
    before = md.tobytes()
    with np.errstate(over="ignore", under="ignore"):
        expected = scalar_authority(md, length)
        actual = native_cell_dx_m(md, length)
    assert actual.tobytes() == expected.tobytes()
    assert md.tobytes() == before
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert not actual.flags.writeable
    assert not np.shares_memory(actual, md)


@pytest.mark.parametrize("layout", ["slice", "reverse", "readonly", "f64", "list"])
def test_cell_dx_carrier_layouts(layout):
    md = np.linspace(.001, 1, 8195, dtype=np.float32)
    if layout == "slice": md = md[::2]
    elif layout == "reverse": md = md[::-1]
    elif layout == "readonly": md.setflags(write=False)
    elif layout == "f64": md = md.astype(np.float64)
    elif layout == "list": md = md.tolist()
    actual = native_cell_dx_m(md, 3000)
    assert actual.tobytes() == scalar_authority(md, 3000).tobytes()
    assert not actual.flags.writeable


@pytest.mark.parametrize("md, message", [([], "non-empty"), ([[1]], "non-empty"),
    ([0], "within"), ([-0.], "within"), ([-1], "within"), ([1.0001], "within"),
    ([np.nan], "within"), ([np.inf], "within")])
def test_cell_dx_density_refusals_precede_length(md, message):
    with pytest.raises(ValueError, match=message):
        native_cell_dx_m(md, np.nan)


@pytest.mark.parametrize("length", [0, -1, np.nan, np.inf])
def test_cell_dx_length_refusals(length):
    with pytest.raises(ValueError, match="finite and positive"):
        native_cell_dx_m([1], length)


def test_cell_dx_raised_fp_overflow():
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            native_cell_dx_m([np.float32(1.e-20)], np.finfo(np.float32).max)
