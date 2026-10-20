"""Chunk-aligned copies preserve the complete NetCDF file and float words."""
import hashlib
import threading

import netCDF4
import numpy as np
import pytest

from hexcore import history_chunks


@pytest.mark.parametrize("chunks", [(1, 31, 7), (1, 83, 17), None])
@pytest.mark.parametrize("reverse", [False, True])
def test_chunk_order_padding_and_special_words(tmp_path, monkeypatch, chunks, reverse):
    monkeypatch.setattr(history_chunks, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "2")
    words = np.resize(np.array([0, 0x80000000, 0x7fc01234, 0x7f800000,
                               0xff800000, 0x3f800001], dtype=np.uint32), (17, 83))
    source = words.view(np.float32)
    if reverse:
        source = source[::-1, ::-1]
    payload = source.T[None]
    caller = threading.get_ident()

    class OnCaller:
        def __init__(self, variable):
            self.variable = variable

        def chunking(self):
            return self.variable.chunking()

        def __setitem__(self, selection, value):
            assert threading.get_ident() == caller
            self.variable[selection] = value

    for fast in (False, True):
        path = tmp_path / f"{fast}.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
            for name, size in (("Time", 1), ("cell", 83), ("level", 17)):
                dataset.createDimension(name, size)
            options = {"chunksizes": chunks} if chunks else {"contiguous": True}
            if chunks:
                options.update(zlib=True, complevel=1)
            v = dataset.createVariable("field", "f4", ("Time", "cell", "level"), **options)
            v.units = "K"
            if fast:
                history_chunks.write_variable(OnCaller(v), payload)
            else:
                v[:] = np.ascontiguousarray(payload)
            dataset.createVariable("after", "i4", ("cell",))[:] = np.arange(83)
        with netCDF4.Dataset(path) as dataset:
            dataset.set_auto_mask(False)
            assert np.array_equal(dataset["field"][:].view(np.uint32), payload.view(np.uint32))
    assert hashlib.sha256((tmp_path / "False.nc").read_bytes()).digest() == hashlib.sha256(
        (tmp_path / "True.nc").read_bytes()).digest()
