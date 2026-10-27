"""hexcore.column_store: the any-point column store of a hex run, every output hour.

Breakage it prevents: a hex run whose soundings have no store for its last hour (a run ending off the hour), a
surface-only export asked to write a store it has no 3-D fields for, and a store failure that fails the door.
"""
from __future__ import annotations

from pathlib import Path

from hexcore import column_store


def test_whole_hours_and_the_last_frame():
    frames = [Path(f"/x/wrfout_d01_2024-04-26_{h:02d}_{m:02d}_00") for h in (12, 13) for m in (0, 30)]
    picked = [f.name for f in column_store.chosen(frames)]
    assert picked == ["wrfout_d01_2024-04-26_12_00_00", "wrfout_d01_2024-04-26_13_00_00",
                      "wrfout_d01_2024-04-26_13_30_00"]


def test_surface_only_and_switched_off_write_nothing(monkeypatch):
    assert column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"), field_set="surface",
                              say=lambda *a, **k: None)["files"] == 0
    monkeypatch.setenv("GPUWM_COLUMNS", "off")
    assert column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"),
                              say=lambda *a, **k: None)["skipped"] == "GPUWM_COLUMNS=off"


def test_a_frame_the_live_loop_stored_is_kept_not_written_again(monkeypatch, tmp_path):
    """The zoom globe's products pass converts every frame again for its whole-run maps; the frames whose store the live
    loop wrote keep it (RESULT.md defect 3: 162 s after a 3 h 3 km zoom globe, 15-20 min after a 24 h one)."""
    import gpuwm.soundings as soundings

    calls = []

    def fake(frame, out):
        m = column_store._FRAME.match(Path(frame).name)
        path = Path(out) / m[1] / f"{m[2]}{m[3]}{m[4]}T{m[5]}{m[6]}{m[7]}Z.wcol"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RWSTORE2")
        calls.append(Path(frame).name)
        return [{"path": str(path), "bytes": 8}]
    monkeypatch.setattr(soundings, "write_columns", fake)
    frames = [tmp_path / f"wrfout_d01_2025-05-19_{h:02d}_00_00" for h in (18, 19, 20, 21)]
    out = tmp_path / "columns"
    # The live loop: one frame at a time as each lands.
    for frame in frames[:3]:
        assert column_store.write([frame], out, say=lambda *a, **k: None)["files"] == 1
    # The products pass: every frame; only the one the live loop did not reach is written.
    calls.clear()
    receipt = column_store.write(frames, out, say=lambda *a, **k: None)
    assert calls == ["wrfout_d01_2025-05-19_21_00_00"]
    assert (receipt["files"], receipt["kept"]) == (1, 3)
    # Everything stored already: nothing written, and the writer is not even loaded.
    calls.clear()
    assert column_store.write(frames, out, say=lambda *a, **k: None)["kept"] == 4 and calls == []
    # An empty file (never left by rw-store, which renames whole files in) is written again.
    (out / "d01" / "20250519T180000Z.wcol").write_bytes(b"")
    column_store.write(frames, out, say=lambda *a, **k: None)
    assert calls == ["wrfout_d01_2025-05-19_18_00_00"]


def test_a_frame_that_fails_is_named_and_the_door_goes_on(monkeypatch):
    import gpuwm.soundings as soundings

    def boom(frame, out):
        raise RuntimeError("column store not written: no PH")
    monkeypatch.setattr(soundings, "write_columns", boom)
    receipt = column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"), say=lambda *a, **k: None)
    assert receipt["files"] == 0 and "no PH" in receipt["failed"][0]["error"]
