"""Process history handoff preserves native bytes and ready/error ordering."""
import threading

import numpy as np
import pytest

from hexcore.history_frames import ProcessHistoryFrames
from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof


@pytest.fixture
def frame(monkeypatch):
    for name, value in (("N_CELLS", 12), ("N_EDGES", 24), ("N_LEVELS", 2),
                        ("N_INTERFACES", 3), ("N_SOIL_LEVELS", 0)):
        monkeypatch.setattr(proof, name, value)
    static = {name: np.arange(size, dtype=dtype) for name, size, dtype in (
        ("indexToCellID", 12, "i4"), ("latCell", 12, "f4"),
        ("lonCell", 12, "f4"), ("ter", 12, "f4"),
        ("indexToEdgeID", 24, "i4"), ("latEdge", 24, "f4"), ("lonEdge", 24, "f4"))}
    return {"arrays": {"theta": np.arange(24, dtype="f4").reshape(2, 12)}}, static


def test_frames_native_bytes_views_and_order(tmp_path, frame):
    snapshot, static = frame
    original = snapshot["arrays"]["theta"]
    main_thread = threading.get_ident()
    ready = []
    def on_written(step, path, record):
        assert threading.get_ident() == main_thread
        assert path.is_file()
        ready.append((step, record))
    frames = ProcessHistoryFrames(proof.write_snapshot_netcdf, on_written)
    try:
        for step in range(3):
            original[:] = step + np.arange(24, dtype="f4").reshape(2, 12)
            snapshot["receipt"] = {"step": step}
            proof.write_snapshot_netcdf(tmp_path / f"direct-{step}.nc", snapshot, static)
            frames.submit(step, tmp_path / f"child-{step}.nc", snapshot, static)
        original[:] = -123  # held views cannot follow subsequent forecast state
        frames.drain(until_step=1)
        assert [step for step, _ in ready] == [0, 1]
        frames.drain()
        assert [step for step, _ in ready] == [0, 1, 2]
        for step in range(3):
            assert (tmp_path / f"direct-{step}.nc").read_bytes() == (tmp_path / f"child-{step}.nc").read_bytes()
        assert frames.receipt()["broken"] is None
        assert frames.receipt()["max_pending"] == 2
    finally:
        frames.close()


def test_refused_frame_is_never_announced_or_replaced(tmp_path, frame):
    snapshot, static = frame
    path = tmp_path / "existing.nc"
    path.write_bytes(b"preserve existing history")
    ready = []
    frames = ProcessHistoryFrames(proof.write_snapshot_netcdf, lambda *args: ready.append(args))
    frames.submit(1, path, snapshot, static)
    with pytest.raises(FileExistsError):
        frames.close()
    assert ready == []
    assert path.read_bytes() == b"preserve existing history"
    assert frames.writer._child.poll() == 0


def test_ready_only_never_waits_for_unfinished_frame():
    from concurrent.futures import Future
    from pathlib import Path
    frames = ProcessHistoryFrames.__new__(ProcessHistoryFrames)
    frames.pending = {1: (Path("not-written.nc"), Future())}
    frames.drain(ready_only=True)
    assert list(frames.pending) == [1]
