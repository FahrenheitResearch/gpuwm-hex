"""NetCDF access stays on the caller while bounded owned copies overlap it."""
from pathlib import Path
import threading

import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import mesh


def test_real_netcdf_words_attributes_order_and_independence(tmp_path, monkeypatch):
    path = tmp_path / "mesh.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("cell", 31)
        dataset.createDimension("slot", 6)
        dataset.file_id = "grid"
        dataset.version = np.int32(841)
        for name, dtype in (("xCell", "f8"), ("cellsOnCell", "i4"), ("packed", "i2")):
            var = dataset.createVariable(name, dtype, ("cell", "slot"))
            var.units = "stored"
            if name == "packed":
                var.scale_factor = np.float32(.25)
                var.add_offset = np.float32(-2.)
            var[:] = np.arange(186).reshape(31, 6)
        dataset.createVariable("scalar", "f8", ())[:] = -0.
        dataset.createVariable("text", str, ("cell",))[:] = np.array(["mesh"] * 31, object)
    monkeypatch.setattr(mesh, "_NETCDF_COPY_MIN_BYTES", 0)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "1")
    reference = mesh._read_netcdf(path)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "2")
    actual = mesh._read_netcdf(path)
    assert reference[:2] == actual[:2]
    assert reference[3:] == actual[3:]
    assert list(reference[2]) == list(actual[2])
    for name, value in reference[2].items():
        result = actual[2][name]
        assert result.dtype == value.dtype and result.shape == value.shape
        if result.dtype.hasobject:
            assert result.tolist() == value.tolist()
        else:
            assert result.tobytes() == value.tobytes()
        assert result.flags == value.flags
        assert not np.shares_memory(value, result)


def fake_dataset(monkeypatch, read, *, attribute_error=None, close_error=None):
    caller = threading.get_ident()

    class Variable:
        dimensions = ("cell",)

        def __init__(self, index):
            self.index = index

        def __getitem__(self, key):
            assert threading.get_ident() == caller
            return read(self.index)

        def ncattrs(self):
            assert threading.get_ident() == caller
            if attribute_error is not None:
                raise attribute_error
            return []

    class Reader:
        dimensions = {"cell": range(3)}
        variables = {str(i): Variable(i) for i in range(3)}

        def __enter__(self):
            assert threading.get_ident() == caller
            return self

        def __exit__(self, *args):
            assert threading.get_ident() == caller
            if close_error is not None:
                raise close_error

        def set_auto_mask(self, mask):
            assert threading.get_ident() == caller and mask is False

        def ncattrs(self):
            assert threading.get_ident() == caller
            return []

    monkeypatch.setattr(mesh, "Dataset", lambda *a, **k: Reader())
    monkeypatch.setattr(mesh, "_NETCDF_COPY_MIN_BYTES", 0)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "2")


def test_real_workers_overlap_reads_with_only_two_pending_buffers(monkeypatch):
    first_started, allow_first, third_read = (threading.Event() for _ in range(3))
    first_finished = threading.Event()
    sources = [np.arange(3) + i for i in range(3)]
    worker_names = set()

    def read(i):
        if i == 1:
            assert first_started.wait(5), "first copy did not run beside next read"
            allow_first.set()
        if i == 2:
            assert first_finished.is_set(), "read buffers were not bounded"
            third_read.set()
        return sources[i]

    def copy(value):
        worker_names.add(threading.current_thread().name)
        if value is sources[0]:
            first_started.set()
            assert allow_first.wait(5)
            first_finished.set()
        elif value is sources[1]:
            assert third_read.wait(5), "copy blocked further caller reads"
        return np.array(value, copy=True)

    fake_dataset(monkeypatch, read)
    monkeypatch.setattr(mesh, "_copy_netcdf_array", copy)
    result = mesh._read_netcdf(Path("fake"))[2]
    assert list(result) == ["0", "1", "2"]
    assert worker_names and all(n.startswith("hex-netcdf-copy") for n in worker_names)
    for i, value in enumerate(sources):
        assert result[str(i)].tobytes() == value.tobytes()
        assert not np.shares_memory(result[str(i)], value)


@pytest.mark.parametrize("later", ["read", "attribute", "close", "copy"])
@pytest.mark.parametrize("first_fails", [False, True])
def test_original_copy_error_precedence_and_identity(monkeypatch, later, first_fails):
    first_error, later_error = MemoryError("first copy"), RuntimeError("later " + later)
    source = [np.full(3, i) for i in range(3)]

    def read(i):
        if i == 1 and later == "read":
            raise later_error
        return source[i]

    def copy(value):
        if value is source[0] and first_fails:
            raise first_error
        if value is source[1] and later == "copy":
            raise later_error
        return np.array(value, copy=True)

    fake_dataset(monkeypatch, read, attribute_error=later_error if later == "attribute" else None,
                 close_error=later_error if later == "close" else None)
    monkeypatch.setattr(mesh, "_copy_netcdf_array", copy)
    with pytest.raises(BaseException) as error:
        mesh._read_netcdf(Path("fake"))
    assert error.value is (first_error if first_fails and later != "close" else later_error)


@pytest.mark.parametrize("fallback", ["single_thread", "small", "object"])
def test_small_object_and_single_thread_copies_stay_inline(monkeypatch, fallback):
    caller = threading.get_ident()
    source = np.array(["x"], object) if fallback == "object" else np.ones(3)
    fake_dataset(monkeypatch, lambda _: source)
    if fallback == "single_thread":
        monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "1")
    elif fallback == "small":
        monkeypatch.setattr(mesh, "_NETCDF_COPY_MIN_BYTES", source.nbytes + 1)

    def copy(value):
        assert threading.get_ident() == caller
        return np.array(value, copy=True)

    monkeypatch.setattr(mesh, "_copy_netcdf_array", copy)
    assert len(mesh._read_netcdf(Path("fake"))[2]) == 3
