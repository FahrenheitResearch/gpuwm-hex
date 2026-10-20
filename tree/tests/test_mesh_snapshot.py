"""A served dynamics mesh is the loader's mesh: same arrays, values, types and sharing."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from hexcore import mesh as mesh_module
from hexcore import mesh_snapshot
from hexcore.mesh import Mesh


def _mesh(grid: Path, static: Path) -> tuple[Mesh, dict]:
    rng = np.random.default_rng(841)
    arrays = {
        "cellsOnEdge": rng.integers(-1, 50, size=(70, 2)).astype(np.int64),
        "xCell": rng.standard_normal(50),
        "areaCell": rng.uniform(1.0, 2.0, size=50).astype(np.float32),
        "zgrid": np.asfortranarray(rng.standard_normal((5, 50))),
        "frozen": rng.standard_normal(3),
    }
    arrays["frozen"].flags.writeable = False
    mesh = Mesh(
        arrays=arrays,
        dimensions={"nCells": 50, "nEdges": 70},
        attrs={"sphere_radius": 6371229.0, "on_a_sphere": "YES", "np_scalar": np.float32(1.5),
               "np_int": np.int32(7), "vector": np.arange(3, dtype=np.int16), "raw": b"\x00\x01"},
        grid_attrs={"file_id": "abc"},
        static_attrs={"parent_id": "abc"},
        variable_dimensions={"xCell": ("nCells",), "cellsOnEdge": ("nEdges", "TWO")},
        variable_attrs={"xCell": {"units": "m", "missing": np.float64(-1e30)}},
        variable_sources={"xCell": "grid", "areaCell": "static"},
        grid_path=grid, static_path=static, converted_connectivity=("cellsOnEdge",),
    )
    evidence = {"schema": "test", "scale": 1.0 / 3.0, "nested": [1, (2, 3.5), None, True],
                "path": PurePosixPath("/x/y")}
    mesh.provenance["precision_preserving_mesh_pair"] = evidence
    return mesh, evidence


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_PREP_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", "off")
    monkeypatch.delenv(mesh_snapshot.ENVIRONMENT, raising=False)
    grid, static = tmp_path / "x.grid.nc", tmp_path / "x.static.nc"
    grid.write_bytes(b"grid bytes")
    static.write_bytes(b"static bytes")
    calls = []

    def loader(grid_path, static_path):
        calls.append((grid_path, static_path))
        built, evidence = _mesh(Path(grid_path), Path(static_path))
        return built, object(), evidence

    monkeypatch.setattr(mesh_module, "load_precision_preserving_mesh_pair", loader)
    mesh_snapshot._ledger.clear()
    return grid, static, calls


def _same(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.flags.c_contiguous == right.flags.c_contiguous
        assert left.flags.f_contiguous == right.flags.f_contiguous
        assert left.flags.writeable == right.flags.writeable
        assert left.tobytes(order="A") == right.tobytes(order="A")
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    elif isinstance(left, np.generic):
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    else:
        assert left == right


def test_hit_returns_the_loader_mesh_with_its_sharing(files):
    grid, static, calls = files
    first, first_evidence = mesh_snapshot.load_dynamics_mesh(grid, static)
    second, second_evidence = mesh_snapshot.load_dynamics_mesh(grid, static)
    assert len(calls) == 1
    events = [row["event"] for row in mesh_snapshot.receipt()["calls"]]
    assert events == ["miss", "hit"]
    assert isinstance(second, Mesh)
    _same(dict(vars(first)), dict(vars(second)))
    _same(first_evidence, second_evidence)
    assert second.variables is second.arrays
    assert second.provenance["variable_sources"] is second.variable_sources
    assert second.provenance["precision_preserving_mesh_pair"] is second_evidence
    assert second.cells_on_edge is second.arrays["cellsOnEdge"]


def test_a_changed_byte_misses(files):
    grid, static, calls = files
    mesh_snapshot.load_dynamics_mesh(grid, static)
    static.write_bytes(b"static bytes, edited")
    mesh_snapshot.load_dynamics_mesh(grid, static)
    assert len(calls) == 2


def test_a_damaged_entry_is_discarded_and_the_loader_runs(files):
    grid, static, calls = files
    mesh_snapshot.load_dynamics_mesh(grid, static)
    root = Path(mesh_snapshot.prep_cache.cache_root())
    entry = next(path for path in root.glob("*/*") if (path / "manifest.json").is_file())
    stored = entry / "1.npy"
    raw = bytearray(stored.read_bytes())
    raw[-1] ^= 0xFF
    stored.write_bytes(bytes(raw))
    mesh_snapshot.load_dynamics_mesh(grid, static)
    assert len(calls) == 2
    assert [row["event"] for row in mesh_snapshot.receipt()["calls"]][-2:] == ["discarded", "miss"]


def test_an_unencodable_value_runs_the_loader_every_time(files, monkeypatch):
    grid, static, calls = files
    original = mesh_module.load_precision_preserving_mesh_pair

    def with_view(grid_path, static_path):
        built, other, evidence = original(grid_path, static_path)
        built.arrays["view"] = built.arrays["xCell"][::2]
        return built, other, evidence

    monkeypatch.setattr(mesh_module, "load_precision_preserving_mesh_pair", with_view)
    mesh_snapshot.load_dynamics_mesh(grid, static)
    mesh_snapshot.load_dynamics_mesh(grid, static)
    assert len(calls) == 2
    rows = mesh_snapshot.receipt()["calls"]
    assert [row["stored"] for row in rows] == [False, False]
    assert "array view" in rows[0]["reason"]


def test_off_switch_runs_the_loader(files, monkeypatch):
    grid, static, calls = files
    monkeypatch.setenv(mesh_snapshot.ENVIRONMENT, "off")
    mesh_snapshot.load_dynamics_mesh(grid, static)
    mesh_snapshot.load_dynamics_mesh(grid, static)
    assert len(calls) == 2
    assert mesh_snapshot.receipt()["calls"] == []


def test_manifest_records_the_producer_and_every_array_digest(files):
    grid, static, _ = files
    mesh_snapshot.load_dynamics_mesh(grid, static)
    root = Path(mesh_snapshot.prep_cache.cache_root())
    manifest = json.loads(next(root.glob("*/*/manifest.json")).read_text())
    assert manifest["schema"] == mesh_snapshot.SCHEMA
    assert manifest["producer_digest"] == mesh_snapshot.prep_cache.producer_digest()
    assert all(len(row["sha256"]) == 64 for row in manifest["arrays"])
