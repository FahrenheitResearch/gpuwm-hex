"""Parallel deflate: the history file is the bytes the single-thread HDF5 path writes."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pytest

from hexcore import history_deflate, history_writer
from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof

CRATE = Path(__file__).resolve().parents[1] / "rust/history-deflate"


@pytest.fixture(scope="module")
def library():
    """The crate's release build (seconds once built); the hook is Linux-only."""

    if not sys.platform.startswith("linux"):
        pytest.skip("the HDF5 filter-table hook is built for Linux")
    if history_deflate.library_path() is None:
        cargo = shutil.which("cargo")
        if cargo is None:
            pytest.skip("cargo is not on PATH to build rust/history-deflate")
        subprocess.run([cargo, "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")], check=True)
    ready = history_deflate.prepare()
    assert ready["engaged"], ready
    return ready


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def mesh(monkeypatch):
    # Cells not a multiple of any chunk, so the edge chunks carry the fill padding.
    for name, value in (("N_CELLS", 40_961), ("N_EDGES", 122_877), ("N_LEVELS", 55),
                        ("N_INTERFACES", 56), ("N_SOIL_LEVELS", 4)):
        monkeypatch.setattr(proof, name, value)
    return 40_961, 122_877, 55, 4


def _frame(seed, cells, edges, levels, soil):
    rng = np.random.default_rng(seed)
    smooth = np.linspace(250, 320, cells, dtype=np.float32)
    arrays = {name: (rng.standard_normal((levels, cells)) * 1e-4).astype(np.float32) for name in proof.SCALAR_NAMES}
    arrays.update(
        theta=(smooth[None, :] + np.arange(levels, dtype=np.float32)[:, None]).astype(np.float32),
        noise=rng.standard_normal((levels, cells)).astype(np.float32),  # incompressible: left to HDF5
        normal_u=np.sin(np.arange(levels * edges, dtype=np.float32)).reshape(levels, edges),
        w=np.zeros((levels + 1, cells), dtype=np.float32),
        smois=rng.uniform(0, 0.4, (soil, cells)).astype(np.float32),
        t2=smooth.copy(),
    )
    static = {
        "indexToCellID": np.arange(1, cells + 1, dtype=np.int32),
        "latCell": rng.uniform(-1, 1, cells).astype(np.float32),
        "lonCell": rng.uniform(0, 6, cells).astype(np.float32),
        "ter": rng.uniform(0, 3000, cells).astype(np.float32),
        "indexToEdgeID": np.arange(1, edges + 1, dtype=np.int32),
        "latEdge": rng.uniform(-1, 1, edges).astype(np.float32),
        "lonEdge": rng.uniform(0, 6, edges).astype(np.float32),
    }
    return {"arrays": arrays, "receipt": {"step": 6}}, static


def test_parallel_frames_are_the_single_thread_bytes(tmp_path, library, mesh):
    snapshot, static = _frame(1, *mesh)
    plain, fast = tmp_path / "plain.nc", tmp_path / "fast.nc"
    proof.write_snapshot_netcdf(plain, snapshot, static)
    with history_deflate.frame(proof.write_snapshot_netcdf, (fast, snapshot, static)) as stats:
        proof.write_snapshot_netcdf(fast, snapshot, static)
    assert _sha(plain) == _sha(fast)
    assert stats["engaged"] and stats["restored"] and stats["hits"] > 0
    assert stats["shuffle_hits"] > 0
    # Every matched compressed chunk fits the existing HDF5 allocation.
    assert stats["inplace_hits"] == stats["hits"] > 0
    assert stats["hits"] + stats["misses"] >= stats["chunks"] - stats["deferred"] - stats["unused"]
    # After the frame HDF5's own deflate is back: a third write is the same bytes again.
    again = tmp_path / "again.nc"
    proof.write_snapshot_netcdf(again, snapshot, static)
    assert _sha(again) == _sha(plain)


def test_unprepared_shuffle_chunks_use_original_filter(tmp_path, library, mesh):
    prepared, static = _frame(4, *mesh)
    actual, _ = _frame(5, *mesh)
    plain, fast = tmp_path / "unprepared-plain.nc", tmp_path / "unprepared-fast.nc"
    proof.write_snapshot_netcdf(plain, actual, static)
    # Precompute different bytes. Neither sampled matches nor missing chunks
    # may change the fallback pipeline, including its fill-padded edge chunks.
    with history_deflate.frame(proof.write_snapshot_netcdf, (fast, prepared, static)) as stats:
        proof.write_snapshot_netcdf(fast, actual, static)
    assert _sha(plain) == _sha(fast)
    assert stats["restored"] and stats["shuffle_misses"] > 0
    assert stats["misses"] > 0


def test_the_background_writer_engages_it_and_keeps_the_bytes(tmp_path, library, mesh, monkeypatch):
    monkeypatch.delenv(history_writer.ENVIRONMENT, raising=False)
    monkeypatch.delenv(history_deflate.ENVIRONMENT, raising=False)
    snapshot, static = _frame(2, *mesh)
    direct = tmp_path / "direct.nc"
    proof.write_snapshot_netcdf(direct, snapshot, static)
    writer = history_writer.open_writer(proof.write_snapshot_netcdf)
    assert writer is not None and writer.ready["deflate"]["engaged"], writer.ready
    background = tmp_path / "background.nc"
    writer.result(writer.submit(background, snapshot, static))
    writer.close()
    assert _sha(direct) == _sha(background)
    row = writer.receipt()["frames"][0]
    assert row["how"] == "background" and row["deflate"]["engaged"] and row["deflate"]["hits"] > 0


def test_switched_off_the_writer_says_so(monkeypatch):
    monkeypatch.setenv(history_deflate.ENVIRONMENT, "off")
    monkeypatch.setattr(history_deflate._State, "library", None)
    monkeypatch.setattr(history_deflate._State, "fault", None)
    assert history_deflate.prepare() == {**history_deflate.status(), "engaged": False}
    assert "off" in history_deflate.status()["why_not"]


def test_retained_chunks_respect_budget_while_writer_is_delayed(tmp_path, library, mesh, monkeypatch):
    monkeypatch.setenv(history_deflate.THREADS_ENVIRONMENT, "2")
    monkeypatch.setenv(history_deflate.BUDGET_ENVIRONMENT, "64")
    snapshot, static = _frame(3, *mesh)
    plain, bounded = tmp_path / "plain.nc", tmp_path / "bounded.nc"
    proof.write_snapshot_netcdf(plain, snapshot, static)
    with history_deflate.frame(proof.write_snapshot_netcdf, (bounded, snapshot, static)) as stats:
        # Let compression run ahead while the consumer has not started. This
        # exercises insertion backpressure rather than a writer-paced cache.
        time.sleep(1)
        proof.write_snapshot_netcdf(bounded, snapshot, static)
    assert _sha(plain) == _sha(bounded)
    assert stats["engaged"] and stats["restored"] and stats["hits"] > 0
    assert 0 < stats["peak_bytes"] <= 64 * 1024 * 1024
    assert stats["budget_bytes"] == 64 * 1024 * 1024


def test_samehost_child_engages_parallel_deflate_and_keeps_file_hash(tmp_path, library, mesh, monkeypatch):
    from hexcore.drivers.samehost_forecast_v841 import _AsyncHistoryWriter
    monkeypatch.setenv('GPUWM_HEX_HISTORY_SPOOL', str(tmp_path))
    snapshot, static = _frame(3, *mesh)
    direct = tmp_path / 'direct-samehost.nc'
    proof.write_snapshot_netcdf(direct, snapshot, static)
    records, hashes = {}, {}
    writer = _AsyncHistoryWriter(proof.write_snapshot_netcdf, lambda a: 'checked', records, hashes, workers=1)
    path = tmp_path / 'samehost.nc'
    writer.submit(0, path, lambda: (snapshot, static))
    writer.close()
    assert _sha(direct) == _sha(path)
    stats = records['0']['deflate']
    assert stats['engaged'] and stats['restored'] and stats['hits'] > 0
    assert history_writer.durable_marker(path).is_file()
