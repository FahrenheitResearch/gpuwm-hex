"""Join template transport before artifact use or failed-output cleanup."""
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, get_ident

import numpy as np
import pytest

from hexcore import vertical_spec as v


@pytest.mark.parametrize("format", [
    "NETCDF3_64BIT_OFFSET", "NETCDF3_64BIT_DATA", "NETCDF4",
])
def test_template_copy_preserves_complete_bytes_before_publication(tmp_path, monkeypatch, format):
    from netCDF4 import Dataset

    source, partial, published = [tmp_path / name for name in ("static", "partial", "artifact")]
    with Dataset(source, "w", format=format) as dataset:
        dataset.createDimension("nCells", 12)
        dataset.createVariable("ter", "f4", ("nCells",))[:] = np.arange(12, dtype=np.float32)
    original = source.read_bytes()
    monkeypatch.setattr(v, "_netcdf_copy_executable", lambda: None)
    with v._background_vertical_template(source, partial, n_edges=36, n_levels=5) as ready:
        assert ready() == "source-format-copy"
        assert partial.read_bytes() == original
        partial.replace(published)
    assert not partial.exists()
    assert source.read_bytes() == published.read_bytes() == original


def test_template_copy_failure_does_not_leave_partial_output(tmp_path, monkeypatch):
    output = tmp_path / "partial"

    def fail(*args, **unused):
        output.write_bytes(b"incomplete")
        raise OSError("template transport failed")

    monkeypatch.setattr(v, "_copy_vertical_template", fail)
    with pytest.raises(OSError, match="template transport failed"):
        with v._background_vertical_template(tmp_path / "static", output, n_edges=36, n_levels=5):
            pass  # Context exit must also propagate a failure if not consumed.
    assert not output.exists()


def test_default_construction_overlaps_copy_and_joins_before_failure_cleanup(tmp_path, monkeypatch):
    from hexcore import rust_geometry

    grid, static, spec, output = [tmp_path / name for name in ("grid", "static", "spec", "artifact")]
    grid.write_bytes(b"grid")
    static.write_bytes(b"static")
    spec.write_text(json.dumps(v.VerticalSpec().to_mapping()))
    output.write_bytes(b"previous artifact")
    mesh = type("Mesh", (), {"dimensions": {"nCells": 12, "nEdges": 36}})()
    monkeypatch.setattr(v, "_mesh_and_input_hashes", lambda *unused: (mesh, "grid-sha", "static-sha"))
    monkeypatch.setattr(rust_geometry, "geometry_engine", lambda: ("unchanged-engine", "engine-sha"))
    entered, release, construction_failed = Event(), Event(), Event()
    copy_threads = []

    def copy(source, partial, **unused):
        copy_threads.append(get_ident())
        partial.write_bytes(b"early partial")
        entered.set()
        assert release.wait(5)
        # Recreating the file after a premature cleanup would leak this write.
        partial.write_bytes(b"late partial")
        raise OSError("secondary copy failure")

    def build(*args, **unused):
        assert entered.wait(5)
        assert copy_threads == [copy_threads[0]]
        assert copy_threads[0] != get_ident()
        construction_failed.set()
        raise ValueError("vertical construction failed")

    monkeypatch.setattr(v, "_copy_vertical_template", copy)
    monkeypatch.setattr(v, "build_vertical_from_spec", build)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(v.materialize_vertical_artifact,
                             grid=grid, static=static, spec_path=spec, output=output)
        try:
            assert construction_failed.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ValueError, match="vertical construction failed"):
            future.result(timeout=5)
    assert output.read_bytes() == b"previous artifact"
    assert not output.with_name(output.name + ".receipt.json").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))
