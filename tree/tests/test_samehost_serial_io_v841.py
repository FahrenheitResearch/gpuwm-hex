"""Legacy I/O stays on the coordinator without changing file or field bytes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from types import ModuleType, SimpleNamespace
import sys

import netCDF4
import cftime
import numpy as np
import pytest

from hexcore.drivers.samehost_serial_io_v841 import (
    SerialExtensionIoError, coordinator_only_extension_io,
)


def _worker(operation):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(operation).result()


def _fake_extensions():
    calls = []

    class Variable:
        @property
        def shape(self):
            calls.append("variable.shape")
            return (2,)

        def __getitem__(self, key):
            calls.append("variable.getitem")
            return np.array([1., -0.], dtype=np.float32)

        def __setitem__(self, key, item):
            calls.append("variable.setitem")

        def read(self):
            calls.append("variable.read")
            return "read"

    class Dimension:
        def __len__(self):
            calls.append("dimension.len")
            return 2

    class Dataset:
        def __init__(self, *args, **kwargs):
            calls.append("dataset.create")
            self.variables = {"field": Variable()}
            self.dimensions = {"columns": Dimension()}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("dataset.exit")

    def conversion(*args, **kwargs):
        calls.append("conversion")
        return "converted"

    netcdf = SimpleNamespace(__name__="netCDF4", Dataset=Dataset, Variable=Variable, Dimension=Dimension,
                             num2date=conversion, date2num=conversion, date2index=conversion)
    dates = SimpleNamespace(__name__="cftime", num2date=conversion, date2num=conversion, date2index=conversion)
    return netcdf, dates, calls


@pytest.mark.parametrize("operation", ["dataset", "attribute", "index", "write", "method", "dimension",
                                      "mapping", "private_value", "netcdf_date", "cftime_date"])
def test_worker_refuses_before_any_raw_extension_call(operation):
    netcdf, dates, calls = _fake_extensions()
    with coordinator_only_extension_io(netcdf=netcdf, cftime=dates) as guard:
        dataset = netcdf.Dataset("unused")
        variable = dataset.variables["field"]
        dimension = dataset.dimensions["columns"]
        method = variable.read
        mapping = dataset.variables
        operations = {
            "dataset": lambda: netcdf.Dataset("unused"), "attribute": lambda: variable.shape,
            "index": lambda: variable[:], "write": lambda: variable.__setitem__(slice(None), [1, 2]),
            "method": method, "dimension": lambda: len(dimension), "mapping": lambda: mapping["field"],
            "private_value": lambda: variable._value,
            "netcdf_date": lambda: netcdf.num2date(0, "unused"), "cftime_date": lambda: dates.date2num(0, "unused"),
        }
        before = list(calls)
        with pytest.raises(SerialExtensionIoError, match="forecast coordinator"):
            _worker(operations[operation])
        assert calls == before
        assert guard.receipt()["worker_refusals"] == 1
    assert guard.receipt()["module_entrypoints_restored"] is True


def test_existing_imported_reader_aliases_are_guarded_and_restored(monkeypatch):
    netcdf, dates, calls = _fake_extensions()
    alias = ModuleType("hexcore.test_reader_alias")
    alias.ImportedDataset = netcdf.Dataset
    alias.convert_date = netcdf.date2num
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    originals = (alias.ImportedDataset, alias.convert_date)
    with coordinator_only_extension_io(netcdf=netcdf, cftime=dates) as guard:
        before = list(calls)
        with pytest.raises(SerialExtensionIoError):
            _worker(lambda: alias.ImportedDataset("unused"))
        with pytest.raises(SerialExtensionIoError):
            _worker(lambda: alias.convert_date(0, "unused"))
        assert calls == before
        assert set(guard.receipt()["existing_reader_aliases_patched"]) >= {
            "hexcore.test_reader_alias.ImportedDataset", "hexcore.test_reader_alias.convert_date"}
    assert (alias.ImportedDataset, alias.convert_date) == originals


def test_entrypoints_and_reader_aliases_restore_after_caller_exception(monkeypatch):
    netcdf, dates, _calls = _fake_extensions()
    alias = ModuleType("gpuwm.test_reader_alias")
    alias.Dataset = netcdf.Dataset
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    original_dataset, original_conversion = netcdf.Dataset, dates.num2date
    with pytest.raises(ValueError, match="caller failed"):
        with coordinator_only_extension_io(netcdf=netcdf, cftime=dates) as guard:
            raise ValueError("caller failed")
    assert netcdf.Dataset is original_dataset and alias.Dataset is original_dataset
    assert dates.num2date is original_conversion
    assert guard.receipt()["module_entrypoints_restored"] is True


def test_aliases_first_bound_during_guard_restore_after_failure(monkeypatch):
    netcdf, dates, _calls = _fake_extensions()
    original_dataset, original_date = netcdf.Dataset, dates.date2num
    alias = ModuleType("hexcore.test_new_reader_alias")
    with pytest.raises(ValueError, match="after import"):
        with coordinator_only_extension_io(netcdf=netcdf, cftime=dates) as guard:
            # This mirrors a module's from-import while the CLI is loading.
            alias.Dataset, alias.date2num = netcdf.Dataset, dates.date2num
            monkeypatch.setitem(sys.modules, alias.__name__, alias)
            with pytest.raises(SerialExtensionIoError):
                _worker(lambda: alias.Dataset("unused"))
            raise ValueError("after import")
    assert alias.Dataset is original_dataset and alias.date2num is original_date
    assert set(guard.receipt()["new_reader_aliases_restored"]) >= {
        "hexcore.test_new_reader_alias.Dataset", "hexcore.test_new_reader_alias.date2num"}


def _write_file(path):
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("columns", 3)
        variable = dataset.createVariable("field", "f4", ("columns",), zlib=True, complevel=1)
        variable.setncattr("units", "K")
        variable[:] = np.array([1., -0., 3.], dtype=np.float32)
        dataset.setncattr("schema", "byte-test")


def test_real_coordinator_read_write_preserves_full_file_hash_and_signed_zero(tmp_path):
    original_dataset = netCDF4.Dataset
    original_date = cftime.date2num
    plain, guarded = tmp_path / "plain.nc", tmp_path / "guarded.nc"
    _write_file(plain)
    with coordinator_only_extension_io() as guard:
        _write_file(guarded)
        with netCDF4.Dataset(guarded, "r") as dataset:
            assert len(dataset.dimensions["columns"]) == 3
            assert tuple(dataset.variables) == ("field",)
            variable = dataset.variables["field"]
            variable.set_auto_maskandscale(False)
            payload = np.asarray(variable[:])
            assert payload.tobytes() == np.array([1., -0., 3.], dtype=np.float32).tobytes()
            assert payload.view(np.uint32)[1] == 0x80000000
            assert variable.shape == (3,)
            assert variable.getncattr("units") == "K"
        units = "hours since 2000-01-01"
        assert netCDF4.date2num(datetime(2000, 1, 2), units) == 24
        assert cftime.date2num(datetime(2000, 1, 2), units) == 24
    assert netCDF4.Dataset is original_dataset and cftime.date2num is original_date
    assert hashlib.sha256(plain.read_bytes()).hexdigest() == hashlib.sha256(guarded.read_bytes()).hexdigest()
    receipt = guard.receipt()
    assert receipt["worker_refusals"] == 0
    assert receipt["module_entrypoints_restored"] is True
    assert receipt["coordinator_calls"]["netCDF4.Dataset"] == 2


def test_real_variable_cannot_be_used_from_a_rank_thread(tmp_path):
    path = tmp_path / "field.nc"
    _write_file(path)
    with coordinator_only_extension_io() as guard:
        with netCDF4.Dataset(path, "r") as dataset:
            variable = dataset.variables["field"]
            with pytest.raises(SerialExtensionIoError, match="coordinator"):
                _worker(lambda: variable[:])
            assert np.asarray(variable[:]).shape == (3,)
        assert guard.receipt()["worker_refusals"] == 1


def test_existing_mesh_reader_preserves_every_loaded_byte_and_attribute(tmp_path):
    from hexcore import mesh

    path = tmp_path / "mesh-reader.nc"
    _write_file(path)
    plain = mesh._read_netcdf(path)
    original_alias = mesh.Dataset
    with coordinator_only_extension_io() as guard:
        guarded = mesh._read_netcdf(path)
        assert "hexcore.mesh.Dataset" in guard.receipt()["existing_reader_aliases_patched"]
    assert mesh.Dataset is original_alias
    for index in (0, 1, 3, 4):
        assert plain[index] == guarded[index]
    assert set(plain[2]) == set(guarded[2])
    for name, array in plain[2].items():
        candidate = guarded[2][name]
        assert array.dtype == candidate.dtype and array.shape == candidate.shape
        assert array.tobytes() == candidate.tobytes()
