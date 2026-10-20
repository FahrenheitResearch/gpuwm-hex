"""Layout-generation checks through the existing mesh NetCDF loader."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import build_samehost_layouts_v841 as instrument
from hexcore.drivers import samehost_layouts_v841 as production
from hexcore.mesh import Mesh
from hexcore.partition_assets_v841 import load_layouts_npz, sha256_file


def test_cold_grid_hash_overlaps_mesh_read_without_threading_netcdf(tmp_path, monkeypatch):
    from hexcore import file_digest

    grid = _grid(tmp_path)
    expected = Mesh.from_netcdf(grid, validate=False)
    digest = sha256_file(grid)
    caller = threading.get_ident()
    started = threading.Event()
    read = threading.Event()
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)

    def hash_grid(path):
        assert path == grid and threading.get_ident() != caller
        started.set()
        assert read.wait(5), "mesh read did not overlap the hash"
        return digest

    def read_grid(path, *, validate):
        assert path == grid and not validate and threading.get_ident() == caller
        assert started.wait(5), "grid hash did not run on its worker"
        read.set()
        return expected

    monkeypatch.setattr(production, "file_version_sha256", hash_grid)
    monkeypatch.setattr(production.Mesh, "from_netcdf", read_grid)
    assert production._load_mesh_and_digest(grid) == (digest, expected)


@pytest.mark.parametrize("workers,threshold", [(1, 0), (4, 1 << 30)])
def test_small_or_single_worker_grid_keeps_hash_then_read_inline(tmp_path, monkeypatch, workers, threshold):
    from hexcore import file_digest

    grid = _grid(tmp_path)
    calls = []
    caller = threading.get_ident()
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", str(workers))
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", threshold)

    def hash_grid(path):
        assert threading.get_ident() == caller
        calls.append("hash")
        return "digest"

    def read_grid(path, *, validate):
        assert threading.get_ident() == caller and not validate
        calls.append("read")
        return "mesh"

    monkeypatch.setattr(production, "file_version_sha256", hash_grid)
    monkeypatch.setattr(production.Mesh, "from_netcdf", read_grid)
    assert production._load_mesh_and_digest(grid) == ("digest", "mesh")
    assert calls == ["hash", "read"]


@pytest.mark.parametrize("hash_fails,read_fails", [(True, True), (True, False), (False, True)])
def test_overlapping_grid_failures_preserve_original_exception(tmp_path, monkeypatch, hash_fails, read_fails):
    from hexcore import file_digest

    grid = _grid(tmp_path)
    monkeypatch.setenv("GPUWM_HEX_HOST_COPY_THREADS", "4")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    hash_error, read_error = OSError("hash unavailable"), ValueError("mesh unreadable")

    def hash_grid(path):
        if hash_fails:
            raise hash_error
        return "digest"

    def read_grid(path, *, validate):
        if read_fails:
            raise read_error
        return "mesh"

    monkeypatch.setattr(production, "file_version_sha256", hash_grid)
    monkeypatch.setattr(production.Mesh, "from_netcdf", read_grid)
    expected = hash_error if hash_fails else read_error
    with pytest.raises(type(expected)) as caught:
        production._load_mesh_and_digest(grid)
    assert caught.value is expected


def _grid(tmp_path):
    grid = tmp_path / "regional.grid.nc"
    with Dataset(grid, "w") as dataset:
        for name, count in (("nCells", 12), ("nEdges", 14), ("nVertices", 16), ("maxEdges", 2), ("two", 2), ("three", 3)):
            dataset.createDimension(name, count)
        coc = np.array([[cell - 1 if cell else -1, cell + 1 if cell < 11 else -1] for cell in range(12)], dtype=np.int32)
        coe = np.array([[-1, 0]] + [[cell, cell + 1] for cell in range(11)] + [[11, -1], [-1, 0]], dtype=np.int32)
        cov = np.array([[cell, -1, -1] for cell in range(12)] + [[0, 1, -1], [3, 4, -1], [7, 8, -1], [11, -1, -1]], dtype=np.int32)
        for name, dimensions, values in (("cellsOnCell", ("nCells", "maxEdges"), coc + 1), ("cellsOnEdge", ("nEdges", "two"), coe + 1), ("cellsOnVertex", ("nVertices", "three"), cov + 1), ("nEdgesOnCell", ("nCells",), np.full(12, 2, dtype=np.int32))):
            dataset.createVariable(name, "i4", dimensions)[:] = values
    return grid


@pytest.mark.parametrize("ranks", [2, 4])
def test_contiguous_parts_make_bounded_layouts_bound_to_original_bytes(tmp_path, ranks):
    grid = _grid(tmp_path)
    part = tmp_path / f"official.part.{ranks}"
    original = "\n".join(str(cell * ranks // 12) for cell in range(12)) + "\n"
    part.write_text(original)
    receipt = instrument.build_assets(grid, ranks=ranks, output_dir=tmp_path / f"assets{ranks}", part=part)
    assert part.read_text() == original
    assert receipt["grid_sha256"] == sha256_file(grid)
    assert receipt["part_sha256"] == sha256_file(part)
    assert receipt["layouts_sha256"] == sha256_file(receipt["layouts"])
    assert receipt["owned_connected_components"] == [1] * ranks
    assert receipt["cut_edges"] == ranks - 1
    assert receipt["exact_cover"] == {"cells": 12, "edges": 14, "vertices": 16}
    assert all(row["n_owned_cells"] < row["n_local_cells"] <= 12 for row in receipt["partitions"])
    assert receipt["replicated_cells"] == [11]
    assert Path(receipt["graph"]).read_text().splitlines()[0] == "12 11"
    layouts = load_layouts_npz(receipt["layouts"], expect_mesh_sha256=receipt["grid_sha256"], expect_part_sha256=receipt["part_sha256"], expect_halo_rings=3)
    assert len(layouts) == ranks
    assert all(layout._replicated_cells == (11,) and 11 in layout.cell_l2g for layout in layouts)
    assert json.loads(Path(receipt["receipt"]).read_text())["layouts_sha256"] == receipt["layouts_sha256"]
    assert instrument.build_assets(grid, ranks=ranks, output_dir=tmp_path / f"assets{ranks}", part=part)["layouts_sha256"] == receipt["layouts_sha256"]


def test_gpmetis_invocation_uses_cut_objective_contiguity_and_fixed_seed(tmp_path, monkeypatch):
    grid = _grid(tmp_path)
    destination = tmp_path / "generated.part.2"
    commands = []
    monkeypatch.setattr(instrument.shutil, "which", lambda name: "/bin/gpmetis")
    def run(command, **kwargs):
        commands.append((command, kwargs))
        Path(f"{command[-2]}.part.{command[-1]}").write_text("0\n" * 6 + "1\n" * 6)
        return subprocess.CompletedProcess(command, 0, "METIS completed\n", "")
    monkeypatch.setattr(instrument.subprocess, "run", run)
    receipt = instrument.build_assets(grid, ranks=2, output_dir=tmp_path / "assets", part=destination)
    command, kwargs = commands[0]
    assert "-contig" in command and "-objtype=cut" in command and "-seed=841" in command
    assert kwargs["timeout"] == 600
    assert destination.read_bytes() == Path(receipt["part"]).read_bytes()
    assert receipt["metis_command"] == command
    assert Path(receipt["metis_log"]).read_text() == "METIS completed\n"


def test_missing_slots_are_not_neighbors_and_active_range_is_checked(tmp_path):
    grid = _grid(tmp_path)
    mesh = Mesh.from_netcdf(grid, validate=False)
    rows = instrument.graph_rows(mesh)
    assert rows[0] == (1,) and rows[-1] == (10,)
    assert instrument.component_count(rows) == 1
    mesh.arrays["cellsOnCell"][0, 0] = -2
    with pytest.raises(ValueError, match="outside nCells"):
        instrument.graph_rows(mesh)


def test_asymmetric_graph_and_noncontiguous_part_refuse(tmp_path):
    grid = _grid(tmp_path)
    mesh = Mesh.from_netcdf(grid, validate=False)
    mesh.arrays["cellsOnCell"][1, 0] = -1
    with pytest.raises(ValueError, match="not reciprocal"):
        instrument.graph_rows(mesh)
    part = tmp_path / "alternating.part.2"
    part.write_text("0\n1\n" * 6)
    with pytest.raises(ValueError, match="not contiguous"):
        instrument.build_assets(grid, ranks=2, output_dir=tmp_path / "assets", part=part)


def test_missing_metis_names_a_concrete_recovery(tmp_path, monkeypatch):
    grid = _grid(tmp_path)
    monkeypatch.setattr(instrument.shutil, "which", lambda name: None)
    from hexcore import engines
    monkeypatch.setattr(engines, "bundled_program", lambda name: None)
    with pytest.raises(ValueError, match="existing --part file"):
        instrument.build_assets(grid, ranks=2, output_dir=tmp_path / "assets")


def test_one_rank_needs_no_metis_and_cli_emits_receipt(tmp_path, capsys):
    grid = _grid(tmp_path)
    assert instrument.main(["--grid", str(grid), "--ranks", "1", "--output-dir", str(tmp_path / "assets")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["ranks"] == 1 and summary["cut_edges"] == 0
    assert Path(summary["receipt"]).is_file()


def test_asset_builder_uses_the_active_edge_stencil_for_completeness(tmp_path):
    grid = _grid(tmp_path)
    with Dataset(grid, "a") as dataset:
        dataset.createDimension("maxEdges2", 1)
        stencil = np.zeros((14, 1), np.int32)
        stencil[9, 0] = 1  # edge 9 refers to edge 0 beyond this rank's K3 set
        dataset.createVariable("edgesOnEdge", "i4", ("nEdges", "maxEdges2"))[:] = stencil
        dataset.createVariable("nEdgesOnEdge", "i4", ("nEdges",))[:] = (stencil[:, 0] > 0).astype(np.int32)
    part = tmp_path / "part2"
    part.write_text("0\n" * 6 + "1\n" * 6)
    receipt = instrument.build_assets(grid, ranks=2, part=part, output_dir=tmp_path / "assets")
    layouts = load_layouts_npz(receipt["layouts"], expect_mesh_sha256=receipt["grid_sha256"],
                              expect_part_sha256=receipt["part_sha256"], expect_halo_rings=3)
    layout = layouts[1]
    edge = int(np.flatnonzero(layout.edge_l2g == 9)[0])
    # Replicated cell 11 is retained, but edge 0 does not enter its K3 cone.
    assert 0 not in layout.edge_l2g
    assert not layout.edge_row_complete[edge]


def test_a_verified_receipt_is_reused_without_recomputing(tmp_path, monkeypatch):
    """A warm run reads the receipt instead of re-walking every cell in Python."""

    from hexcore.drivers import samehost_layouts_v841 as layouts_module

    grid = _grid(tmp_path)
    part = tmp_path / "official.part.2"
    part.write_text("\n".join(str(cell * 2 // 12) for cell in range(12)) + "\n")
    output = tmp_path / "assets"
    first = layouts_module.build_assets(grid, ranks=2, output_dir=output, part=part)

    def walked(*args, **kwargs):
        raise AssertionError("the verified receipt should have been reused")

    monkeypatch.setattr(layouts_module, "PackedGraph", walked)
    reused = layouts_module.build_assets(grid, ranks=2, output_dir=output, part=part)
    assert reused == json.loads(json.dumps(first)) == json.loads(Path(first["receipt"]).read_text())
    monkeypatch.undo()

    # Any named file that no longer carries its recorded digest rebuilds and
    # re-verifies everything, exactly as a cold run does.
    graph = Path(first["graph"])
    graph.write_text(graph.read_text() + "\n")
    assert layouts_module.reuse_verified_receipt(grid, ranks=2, output_dir=output, part=part) is None
    with pytest.raises(ValueError, match="existing asset has different bytes"):
        layouts_module.build_assets(grid, ranks=2, output_dir=output, part=part)


def test_a_receipt_from_another_builder_version_is_not_reused(tmp_path):
    from hexcore.drivers import samehost_layouts_v841 as layouts_module

    grid = _grid(tmp_path)
    part = tmp_path / "official.part.2"
    part.write_text("\n".join(str(cell * 2 // 12) for cell in range(12)) + "\n")
    output = tmp_path / "assets"
    receipt = layouts_module.build_assets(grid, ranks=2, output_dir=output, part=part)
    stored = json.loads(Path(receipt["receipt"]).read_text())
    stored["builder_version"] = "older"
    Path(receipt["receipt"]).write_text(json.dumps(stored))
    assert layouts_module.reuse_verified_receipt(grid, ranks=2, output_dir=output, part=part) is None


def test_cold_builder_remembers_grid_digest_for_verified_reuse(tmp_path, monkeypatch):
    from hexcore import file_digest

    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", str(tmp_path / "digests"))
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    monkeypatch.setattr(file_digest, "RACY_SECONDS", 0)
    grid = _grid(tmp_path)
    calls = []
    original_hash = file_digest._hash

    def traced_hash(path):
        calls.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(file_digest, "_hash", traced_hash)
    first = production.build_assets(grid, ranks=1, output_dir=tmp_path / "assets")
    assert first["grid_sha256"] == sha256_file(grid)
    assert production.build_assets(grid, ranks=1, output_dir=tmp_path / "assets") == first
    assert calls.count(grid) == 1


@pytest.mark.parametrize("digest_cache", ["enabled", "off"])
def test_cold_builder_rejects_grid_changed_after_layout_construction(tmp_path, monkeypatch, digest_cache):
    from hexcore import file_digest

    monkeypatch.setenv("GPUWM_HEX_DIGEST_CACHE", str(tmp_path / "digests") if digest_cache == "enabled" else "off")
    monkeypatch.setattr(file_digest, "MIN_CACHED_BYTES", 0)
    monkeypatch.setattr(file_digest, "RACY_SECONDS", 0)
    grid = _grid(tmp_path)
    original_save = production.save_layouts_npz

    def save_then_change_grid(*args, **kwargs):
        original_save(*args, **kwargs)
        with grid.open("ab") as stream:
            stream.write(b"changed during asset construction")

    monkeypatch.setattr(production, "save_layouts_npz", save_then_change_grid)
    output = tmp_path / "assets"
    with pytest.raises(ValueError, match="mesh or part file changed while layouts were being built"):
        production.build_assets(grid, ranks=1, output_dir=output)
    assert not (output / "regional.part1.K3.json").exists()


@pytest.mark.parametrize("instrument", [instrument, production])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
@pytest.mark.parametrize("workers", [1, 4])
def test_graph_blocks_preserve_sorted_unique_active_neighbors(monkeypatch, dtype, workers, instrument):
    from hexcore import host_parallel

    monkeypatch.setenv(host_parallel.ENVIRONMENT, str(workers))
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 4)
    size = 79
    rng = np.random.default_rng(841)
    adjacency = np.full((size, 9), np.iinfo(dtype).max, dtype=dtype)
    expected = []
    for cell in range(size):
        neighbors = sorted({(cell + offset) % size for offset in (-3, -2, -1, 1, 2, 3)})
        expected.append(tuple(neighbors))
        # Duplicate entries are collapsed; invalid inactive padding is ignored.
        adjacency[cell, :7] = rng.permutation(neighbors + neighbors[:1])
    mesh = SimpleNamespace(dimensions={"nCells": size}, arrays={
        "cellsOnCell": adjacency, "nEdgesOnCell": np.full(size, 7, np.int64)})
    assert instrument.graph_rows(mesh) == tuple(expected)


@pytest.mark.parametrize("instrument", [instrument, production])
@pytest.mark.parametrize("workers", [1, 4])
def test_graph_blocks_preserve_first_refusal_and_reciprocal_order(monkeypatch, workers, instrument):
    from hexcore import host_parallel

    monkeypatch.setenv(host_parallel.ENVIRONMENT, str(workers))
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 1)
    adjacency = np.array([[2, 1], [0, -1], [0, -1], [-1, -1]], np.int64)
    counts = np.full(4, 2, np.int64)
    mesh = SimpleNamespace(dimensions={"nCells": 4}, arrays={
        "cellsOnCell": adjacency, "nEdgesOnCell": counts})
    # Earlier self-edge wins over a later invalid count, even across workers.
    adjacency[0, 0] = 0
    counts[3] = 3
    with pytest.raises(ValueError, match="cell 0: a self edge"):
        instrument.graph_rows(mesh)
    # Within one row, count then bounds then self matches the scalar checks.
    counts[0] = -1
    with pytest.raises(ValueError, match="cell 0: nEdgesOnCell"):
        instrument.graph_rows(mesh)
    counts[0] = 2
    adjacency[0, 1] = -2
    with pytest.raises(ValueError, match="cell 0: adjacency"):
        instrument.graph_rows(mesh)
    counts[3] = 2
    adjacency[0] = [2, 1]
    adjacency[1:] = -1
    with pytest.raises(ValueError, match="not reciprocal at 0->1"):
        instrument.graph_rows(mesh)


@pytest.mark.parametrize("instrument", [instrument, production])
def test_graph_blocks_accept_empty_graph_and_zero_width(instrument):
    for size in (0, 3):
        mesh = SimpleNamespace(dimensions={"nCells": size}, arrays={
            "cellsOnCell": np.empty((size, 0), np.int32),
            "nEdgesOnCell": np.zeros(size, np.int32)})
        assert instrument.graph_rows(mesh) == ((),) * size
