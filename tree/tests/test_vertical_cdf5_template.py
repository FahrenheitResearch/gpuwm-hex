"""Real NetCDF format and stored-bit checks for dense vertical templates."""
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from hexcore.vertical_spec import _copy_vertical_template


@pytest.fixture
def transport():
    binary = os.environ.get("GPUWM_HEX_NETCDF_COPY")
    if not binary or not Path(binary).is_file():
        pytest.skip("the real Rust CDF5 transport is not built")
    return binary


def template(path):
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.set_fill_off()
        ds.createDimension("nEdges", 19_464_192)
        ds.createDimension("nCells", 3)
        ds.createDimension("Time", None)
        ds.title = "raw template copy"
        scalar = ds.createVariable("nominalMinDc", "f4", ())
        scalar[...] = np.float32(3000)
        floats = ds.createVariable("stored", "f4", ("nCells",), fill_value=np.float32(-9999))
        floats.units = "K"
        floats[:] = np.array([0x80000000, 0x7FC00123, 0x3F800001], dtype="u4").view("f4")
        records = ds.createVariable("records", "i4", ("Time", "nCells"))
        records[:] = np.array([[1, -2, 3], [4, 5, 6]], dtype="i4")
    return nc


def test_dense_template_supports_two_large_edge_fields_without_changing_bits(tmp_path, transport):
    original, output = tmp_path / "input.nc", tmp_path / "output.nc"
    nc = template(original)
    assert _copy_vertical_template(original, output, n_edges=19_464_192, n_levels=55) == "rust-cdf5-upgrade"
    with nc.Dataset(original) as left, nc.Dataset(output, "r+") as right:
        assert right.data_model == "NETCDF3_64BIT_DATA"
        assert right.title == left.title
        assert right.dimensions["Time"].isunlimited()
        for name in left.variables:
            a, b = left.variables[name], right.variables[name]
            a.set_auto_mask(False)
            b.set_auto_mask(False)
            assert a[:].tobytes() == b[:].tobytes(), name
            assert a.dimensions == b.dimensions
            for attr in a.ncattrs():
                assert a.getncattr(attr) == b.getncattr(attr)
        right.set_fill_off()
        right.createDimension("TWO", 2)
        right.createDimension("nVertLevelsP1", 56)
        right.createVariable("zb", "f4", ("nEdges", "TWO", "nVertLevelsP1"))
        right.createVariable("zb3", "f4", ("nEdges", "TWO", "nVertLevelsP1"))
        right.sync()


def test_header_growth_keeps_first_data_offset_and_stored_bits(tmp_path, transport):
    original, output = tmp_path / "input.nc", tmp_path / "output.nc"
    nc = template(original)
    subprocess.run([transport, str(original), str(output)], check=True)
    # The first scalar's unique bit pattern identifies its stored offset.
    before = output.read_bytes()
    needle = np.array([3000], dtype=">f4").tobytes()
    offset = before.find(needle)
    assert offset > 1024 * 1024
    with nc.Dataset(output, "r+") as ds:
        for index in range(100):
            ds.setncattr(f"growth_{index}", "x" * 500)
        ds.sync()
        ds["stored"].set_auto_mask(False)
        assert ds["stored"][:].view("u4").tolist() == [0x80000000, 0x7FC00123, 0x3F800001]
    assert output.read_bytes().find(needle) == offset


def test_transport_refuses_an_existing_destination(tmp_path, transport):
    original, output = tmp_path / "input.nc", tmp_path / "output.nc"
    template(original)
    output.write_bytes(b"must survive")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    result = subprocess.run([transport, str(original), str(output)], capture_output=True, text=True)
    assert result.returncode != 0
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest


def test_small_edge_fields_keep_the_existing_format(tmp_path, monkeypatch):
    original, output = tmp_path / "input.nc", tmp_path / "output.nc"
    template(original)
    monkeypatch.setenv("GPUWM_HEX_NETCDF_COPY", "/missing/transport")
    assert _copy_vertical_template(original, output, n_edges=10, n_levels=55) == "source-format-copy"
    assert original.read_bytes() == output.read_bytes()
