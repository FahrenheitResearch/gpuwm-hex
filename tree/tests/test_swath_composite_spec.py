"""The composite documents are a function of the plan, for any number of grids.

THE BREAKAGE THESE PREVENT.  The first composite of several placed grids was
hand-written JSON for three sources.  Hand-written JSON silently drifts from
the cycle: the placement layer decides how many swaths exist and where, so a
composite built by editing a file draws whatever the last editor typed --
three rings when four grids ran, or four rings when one run failed.  These
assert the two documents come out of the plan itself, that a slot with no
finished frames is DROPPED AND NAMED rather than quietly drawn, and that the
composed hours are the intersection every source really carries.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_composite_spec as builder  # noqa: E402


HOURS = ("2026-08-12_06.00.00", "2026-08-12_09.00.00", "2026-08-12_12.00.00")


def _plan(slots):
    return {
        "admitted": [
            {
                "slot_id": slot,
                "threat_class": threat,
                "centroid_deg": [lat, lon],
                "ring_deg": [[lat - 1, lon - 1], [lat + 1, lon - 1],
                             [lat + 1, lon + 1], [lat - 1, lon + 1]],
            }
            for slot, threat, lat, lon in slots
        ]
    }


def _frames(directory: Path, stamps) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        (directory / f"cuda-history.{stamp}.nc").write_bytes(b"")
    return {s: directory / f"cuda-history.{s}.nc" for s in stamps}


@pytest.fixture()
def four_slot_plan(tmp_path):
    plan = _plan([
        ("s01", "atmospheric_river", 14.23, -140.89),
        ("s02", "winter_storm", -66.41, 159.52),
        ("s03", "atmospheric_river", 28.31, 164.82),
        ("s04", "extratropical_cyclone", -60.09, 139.46),
    ])
    path = tmp_path / "swath-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_four_grids_produce_four_overlays_and_four_rings(tmp_path, four_slot_plan):
    base = _frames(tmp_path / "coarse", HOURS)
    slot_dirs, slot_meshes = {}, {}
    for slot in ("s01", "s02", "s03", "s04"):
        _frames(tmp_path / slot, HOURS)
        slot_dirs[slot] = tmp_path / slot
        slot_meshes[slot] = tmp_path / f"{slot}.static.nc"

    compose, overlay, _ = builder.build(
        four_slot_plan, tmp_path / "g96.static.nc", base,
        slot_dirs, slot_meshes, None)

    assert len(compose["overlays"]) == 4
    assert [o["label"] for o in compose["overlays"]] == [
        "s01-atmospheric_river", "s02-winter_storm",
        "s03-atmospheric_river", "s04-extratropical_cyclone",
    ]
    # Every source names the SAME hours, in the same order: a composite that
    # pairs t+0 of one run with t+3 of another is a picture of nothing.
    assert len(compose["base"]["history"]) == len(HOURS)
    for source in compose["overlays"]:
        assert [Path(p).name.split(".", 1)[1] for p in source["history"]] == \
               [Path(p).name.split(".", 1)[1] for p in compose["base"]["history"]]
    assert len(overlay["lines"]) == 4
    assert len(overlay["labels"]) == 4
    assert all(line["closed"] for line in overlay["lines"])


def test_a_slot_with_no_frames_is_dropped_and_named(tmp_path, four_slot_plan):
    base = _frames(tmp_path / "coarse", HOURS)
    slot_dirs, slot_meshes = {}, {}
    for slot in ("s01", "s02", "s03", "s04"):
        directory = tmp_path / slot
        # s03's run failed: the directory exists and is empty.
        _frames(directory, () if slot == "s03" else HOURS)
        slot_dirs[slot] = directory
        slot_meshes[slot] = tmp_path / f"{slot}.static.nc"

    compose, overlay, notes = builder.build(
        four_slot_plan, tmp_path / "g96.static.nc", base,
        slot_dirs, slot_meshes, None)

    assert len(compose["overlays"]) == 3
    assert len(overlay["lines"]) == 3
    assert any("s03" in note and "no history frames" in note for note in notes)


def test_the_composed_hours_are_the_intersection(tmp_path, four_slot_plan):
    base = _frames(tmp_path / "coarse", HOURS)
    slot_dirs, slot_meshes = {}, {}
    for slot in ("s01", "s02", "s03", "s04"):
        # s02 stopped early and carries only the first two hours.
        stamps = HOURS[:2] if slot == "s02" else HOURS
        _frames(tmp_path / slot, stamps)
        slot_dirs[slot] = tmp_path / slot
        slot_meshes[slot] = tmp_path / f"{slot}.static.nc"

    compose, _, notes = builder.build(
        four_slot_plan, tmp_path / "g96.static.nc", base,
        slot_dirs, slot_meshes, None)

    assert len(compose["base"]["history"]) == 2
    assert any("2 shared hour(s)" in note for note in notes)


def test_an_hour_no_source_carries_is_refused_by_name(tmp_path, four_slot_plan):
    base = _frames(tmp_path / "coarse", HOURS)
    slot_dirs, slot_meshes = {}, {}
    for slot in ("s01", "s02", "s03", "s04"):
        _frames(tmp_path / slot, HOURS)
        slot_dirs[slot] = tmp_path / slot
        slot_meshes[slot] = tmp_path / f"{slot}.static.nc"

    with pytest.raises(SystemExit) as excinfo:
        builder.build(four_slot_plan, tmp_path / "g96.static.nc", base,
                      slot_dirs, slot_meshes, ["2026-08-12_18.00.00"])
    assert "2026-08-12_18.00.00" in str(excinfo.value)


def test_no_finished_run_refuses_rather_than_composing_nothing(tmp_path, four_slot_plan):
    base = _frames(tmp_path / "coarse", HOURS)
    with pytest.raises(SystemExit) as excinfo:
        builder.build(four_slot_plan, tmp_path / "g96.static.nc", base,
                      {}, {}, None)
    assert "nothing to compose" in str(excinfo.value)


def test_a_fifth_grid_needs_no_edit(tmp_path):
    """The arbitrary test: N is read, never declared."""

    plan = _plan([
        ("s01", "atmospheric_river", 14.23, -140.89),
        ("s02", "winter_storm", -66.41, 159.52),
        ("s03", "atmospheric_river", 28.31, 164.82),
        ("s04", "extratropical_cyclone", -60.09, 139.46),
        ("s05", "fire_weather", 37.0, -119.5),
    ])
    path = tmp_path / "swath-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    base = _frames(tmp_path / "coarse", HOURS)
    slot_dirs, slot_meshes = {}, {}
    for slot in ("s01", "s02", "s03", "s04", "s05"):
        _frames(tmp_path / slot, HOURS)
        slot_dirs[slot] = tmp_path / slot
        slot_meshes[slot] = tmp_path / f"{slot}.static.nc"

    compose, overlay, _ = builder.build(
        path, tmp_path / "g96.static.nc", base, slot_dirs, slot_meshes, None)
    assert len(compose["overlays"]) == 5
    assert len(overlay["lines"]) == 5
    assert overlay["labels"][-1]["text"] == "s05 fire_weather"
