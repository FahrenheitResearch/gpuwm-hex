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


def test_a_frame_that_fails_is_named_and_the_door_goes_on(monkeypatch):
    import gpuwm.soundings as soundings

    def boom(frame, out):
        raise RuntimeError("column store not written: no PH")
    monkeypatch.setattr(soundings, "write_columns", boom)
    receipt = column_store.write([Path("/x/wrfout_d01_2024-04-26_12_00_00")], Path("/o"), say=lambda *a, **k: None)
    assert receipt["files"] == 0 and "no PH" in receipt["failed"][0]["error"]
