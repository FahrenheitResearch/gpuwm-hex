"""Independent provenance tasks overlap without changing the recorded bytes."""
import hashlib
import threading

import numpy as np
import pytest

from hexcore import vertical_spec as vs
from hexcore.mesh import Mesh


def test_input_hashes_overlap_mesh_read(tmp_path, monkeypatch):
    grid, static = tmp_path / "grid", tmp_path / "static"
    grid.write_bytes(b"grid authority")
    static.write_bytes(b"static authority")
    barrier = threading.Barrier(3, timeout=5)
    original_hash = vs._sha256_file
    caller = threading.get_ident()
    mesh = object()

    def hash_file(path):
        barrier.wait()
        return original_hash(path)

    def read_mesh(cls, selected_grid, selected_static):
        assert (selected_grid, selected_static) == (grid, static)
        assert threading.get_ident() == caller
        barrier.wait()
        return mesh

    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", "off")
    monkeypatch.setattr(vs, "_sha256_file", hash_file)
    monkeypatch.setattr(Mesh, "from_netcdf", classmethod(read_mesh))
    actual = vs._mesh_and_input_hashes(grid, static)
    assert actual == (mesh, hashlib.sha256(grid.read_bytes()).hexdigest(),
                      hashlib.sha256(static.read_bytes()).hexdigest())


def test_file_and_array_receipts_overlap_and_remain_exact(tmp_path, monkeypatch):
    output = tmp_path / "artifact"
    output.write_bytes(bytes(range(256)) * 5000)
    arrays = {"transposed": np.arange(120, dtype=np.float32).reshape(10, 12).T,
              "masked": np.array([0.0, np.nan, np.inf, -2.5], dtype=np.float64)}
    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", "off")
    expected = {"arrays": vs._raw_array_receipts(arrays),
                "output": {"path": str(output), "bytes": output.stat().st_size,
                           "sha256": vs._sha256_file(output)}}
    barrier = threading.Barrier(2, timeout=5)
    original_hash, original_arrays = vs._sha256_file, vs._raw_array_receipts

    def hash_file(path):
        barrier.wait()
        return original_hash(path)

    def array_receipts(values):
        barrier.wait()
        return original_arrays(values)

    monkeypatch.setattr(vs, "_sha256_file", hash_file)
    monkeypatch.setattr(vs, "_raw_array_receipts", array_receipts)
    assert vs._artifact_receipts(arrays, output) == expected
    assert output.read_bytes() == bytes(range(256)) * 5000


def test_failed_output_hash_does_not_return_a_receipt(tmp_path, monkeypatch):
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        vs._artifact_receipts({"value": np.ones(2, dtype=np.float32)}, output)


def test_array_failure_waits_for_hash_worker(tmp_path, monkeypatch):
    output = tmp_path / "artifact"
    output.write_bytes(b"complete")
    barrier = threading.Barrier(2, timeout=5)
    finished = threading.Event()

    def hash_file(path):
        barrier.wait()
        finished.set()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def array_receipts(values):
        barrier.wait()
        raise ValueError("invalid array")

    monkeypatch.setattr(vs, "_sha256_file", hash_file)
    monkeypatch.setattr(vs, "_raw_array_receipts", array_receipts)
    with pytest.raises(ValueError, match="invalid array"):
        vs._artifact_receipts({}, output)
    assert finished.is_set()
