"""Parallel field packing keeps complete netCDF bytes and caller-owned I/O."""
from threading import get_ident

import numpy as np
import pytest

from hexcore import host_parallel, vertical_spec as v


@pytest.fixture
def copy_workers(monkeypatch):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "16")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 7)


def source(dtype):
    words = np.resize(np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                               0xff800000, 0x7fc12345, 0x7f812345], dtype="u4"),
                      (17, 2, 101))
    with np.errstate(invalid="ignore"):
        return words.view("f4").astype(dtype).transpose(2, 1, 0)[:, :, ::-1]


@pytest.mark.parametrize("format", ["NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF4"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_complete_file_words_and_attributes_unchanged(copy_workers, tmp_path, monkeypatch, format, dtype):
    netcdf = pytest.importorskip("netCDF4")
    data = source(dtype)
    original_words = data.tobytes()
    outputs = [tmp_path / (arm + ".nc") for arm in ("before", "after")]
    prepare = v._artifact_field_array
    with np.errstate(invalid="ignore"):
        for arm, path in enumerate(outputs):
            monkeypatch.setattr(v, "_artifact_field_array", prepare if arm else
                                lambda x: np.asarray(x, dtype=np.float32))
            with netcdf.Dataset(path, "w", format=format) as dataset:
                for name, size in zip(("edge", "side", "level"), data.shape):
                    dataset.createDimension(name, size)
                v._write_variable(dataset, "zb", data, ("edge", "side", "level"),
                                  units="unitless", description="edge metric")
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert data.tobytes() == original_words


def test_netcdf_receives_packed_array_on_caller_thread(copy_workers, monkeypatch):
    data = source("f4")
    caller = get_ident()
    workers = []
    executor = host_parallel.ThreadPoolExecutor

    def pool(*args, **kwargs):
        workers.append(kwargs["max_workers"])
        return executor(*args, **kwargs)

    class Variable:
        dimensions = ("edge", "side", "level")
        shape = data.shape

        def __setitem__(self, key, value):
            assert get_ident() == caller
            assert value.flags.c_contiguous
            assert value.tobytes() == np.asarray(data, dtype=np.float32).tobytes()

    class Dataset:
        variables = {"zb": Variable()}

    monkeypatch.setattr(host_parallel, "ThreadPoolExecutor", pool)
    v._write_variable(Dataset(), "zb", data, Variable.dimensions,
                      units="unitless", description="edge metric")
    assert workers == [4]


def test_scalar_shape_and_contiguous_alias_are_preserved():
    scalar = np.asarray(-0.0, dtype=np.float32)
    assert v._artifact_field_array(scalar) is scalar
    data = np.arange(30, dtype=np.float32).reshape(10, 3)
    data.flags.writeable = False
    assert v._artifact_field_array(data) is data
