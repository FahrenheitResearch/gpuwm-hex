"""Whole-variable reads of classic netCDF files on threads return netCDF4-python's arrays (no GPU).

THE COST THIS REMOVES (30 km globe, V5090 box, 2026-10-27): Mesh.from_netcdf read every grid/init variable through
netCDF4 one after another on the forecast's main thread (a 30 km init: 6.3 s for 135 variables, 0.7 s on threads).
"""
import numpy as np
import pytest
from netCDF4 import Dataset

from hexcore import classic_netcdf
from hexcore.mesh import _read_netcdf


def _write(path, data_model):
    with Dataset(path, "w", format=data_model) as ds:
        ds.createDimension("Time", None)
        ds.createDimension("nCells", 7)
        ds.createDimension("nVertLevels", 3)
        ds.createDimension("StrLen", 4)
        ds.title = "classic"
        rng = np.random.default_rng(3)
        for name, kind, dims in (("areaCell", "f8", ("nCells",)), ("zgrid", "f4", ("nCells", "nVertLevels")),
                                 ("cellsOnCell", "i4", ("nCells", "nVertLevels")), ("mask", "i1", ("nCells",)),
                                 ("short", "i2", ("nCells",)), ("theta", "f4", ("Time", "nCells", "nVertLevels")),
                                 ("u", "f4", ("Time", "nCells"))):
            var = ds.createVariable(name, kind, dims)
            var.units = "x"
            shape = tuple(1 if d == "Time" else len(ds.dimensions[d]) for d in dims)
            var[...] = (rng.standard_normal(shape) * 100).astype(kind)
        ds.createVariable("xtime", "S1", ("Time", "StrLen"))[0, :] = np.array(list("2025"), dtype="S1")
        scaled = ds.createVariable("scaled", "i2", ("nCells",))
        scaled.scale_factor = 0.5
        scaled[...] = np.arange(7) * 0.5
        ds.createVariable("scalar", "f4", ())[...] = 3.5


@pytest.mark.parametrize("data_model", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA"])
def test_threaded_reads_are_netcdf4s_arrays(tmp_path, monkeypatch, data_model):
    path = tmp_path / "file.nc"
    _write(path, data_model)
    table, records = classic_netcdf.extents(path)
    assert records == 1
    readable = {name for name in table if classic_netcdf.readable(table[name], records, ())}
    assert {"xtime", "scalar"}.isdisjoint(readable) and {"zgrid", "theta", "u", "areaCell"} <= readable
    monkeypatch.setenv("GPUWM_HEX_CLASSIC_READ", "off")
    reference = _read_netcdf(path)
    monkeypatch.setenv("GPUWM_HEX_CLASSIC_READ", "on")
    threaded = _read_netcdf(path)
    assert reference[0] == threaded[0] and reference[3] == threaded[3]
    assert list(reference[2]) == list(threaded[2])
    assert str(reference[1]) == str(threaded[1]) and str(reference[4]) == str(threaded[4])
    for name, value in reference[2].items():
        assert value.dtype == threaded[2][name].dtype and value.shape == threaded[2][name].shape, name
        assert value.tobytes() == threaded[2][name].tobytes(), name


def test_a_netcdf4_file_is_not_classic(tmp_path):
    path = tmp_path / "hdf5.nc"
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", 2)
        ds.createVariable("a", "f4", ("x",))[...] = 1
    assert classic_netcdf.extents(path) is None
