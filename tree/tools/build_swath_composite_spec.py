"""Turn a swath plan and its finished runs into the two documents a composite
render needs: ``rw_mpas_convert``'s ``--compose`` source list, and
``rw_wrfbatch``'s ``--overlays`` ring-and-label file.

WHY THIS EXISTS.  A composite of N placed grids was hand-written as JSON the
first time it was made, for three sources, one of which was fixture-placed.
Hand-written JSON does not scale with the cycle: the placement layer decides
how many swaths there are and where, so the documents that draw them have to
be a function of the plan rather than a file someone edits when the count
changes.  This reads:

* ``swath-plan.json`` -- the admitted slots, each with its ``slot_id``,
  ``threat_class``, ``centroid_deg`` and its ``ring_deg`` polygon;
* one run directory per slot, holding that slot's history frames;
* the coarse parent's own history frames and static,

and writes the compose source list (base + one overlay per slot, over the
hours EVERY source carries) and the overlay document (one closed ring and one
label per slot).  A slot with no finished run is left out of both and named on
stderr, so a composite of three grids out of four says which one is missing
instead of silently drawing three.

It is a document builder.  It opens no history file, decodes no field and
renders nothing: the frames it names are matched by their valid-time stamp,
which is in the filename the forecast door writes.

Usage::

    python tools/build_swath_composite_spec.py \\
        --plan PLAN/swath-plan.json \\
        --base-mesh coarse.static.nc --base-history COARSE/*.nc \\
        --slot s01=RUN-s01/out --slot s02=RUN-s02/out ... \\
        --slot-mesh s01=build-s01/s01.static.nc ... \\
        --compose-out compose.json --overlay-out overlay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence

#: ``cuda-history.2026-08-12_09.30.00.nc`` -> ``2026-08-12_09.30.00``.  The
#: forecast door writes the valid time into the name, so the hours two runs
#: share are matched on the stamp rather than on file order -- two runs with
#: different history cadences must still compose on the hours they agree on.
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2})")

#: One colour per drawn ring, cycled.  These are ring outlines over a weather
#: field, not a data encoding: they mark where a grid is, and nothing about
#: the colour carries a value.
RING_COLOURS = ("#000000", "#1a1a1a", "#000000", "#1a1a1a")


def stamp_of(path: Path) -> str | None:
    match = _STAMP.search(path.name)
    return match.group(1) if match else None


def frames_in(directory: Path) -> dict[str, Path]:
    """Every history frame in ``directory``, keyed by its valid-time stamp."""

    found: dict[str, Path] = {}
    for item in sorted(directory.glob("*.nc")):
        stamp = stamp_of(item)
        if stamp is not None:
            found[stamp] = item
    return found


def _pairs(values: Sequence[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"{flag} wants slot=path, got {value!r}")
        slot, _, path = value.partition("=")
        out[slot.strip()] = path.strip()
    return out


def build(plan_path: Path, base_mesh: Path, base_frames: dict[str, Path],
          slot_dirs: dict[str, Path], slot_meshes: dict[str, Path],
          only_stamps: Sequence[str] | None) -> tuple[dict, dict, list[str]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    admitted = plan.get("admitted", [])
    notes: list[str] = []

    usable: list[dict] = []
    for swath in admitted:
        slot = swath.get("slot_id")
        if slot not in slot_dirs:
            notes.append(f"{slot}: no run directory given; left out of the composite")
            continue
        if slot not in slot_meshes:
            notes.append(f"{slot}: no mesh given; left out of the composite")
            continue
        directory = Path(slot_dirs[slot])
        found = frames_in(directory)
        if not found:
            notes.append(f"{slot}: no history frames under {directory}; left out")
            continue
        usable.append({"swath": swath, "slot": slot, "frames": found,
                       "mesh": Path(slot_meshes[slot])})

    if not usable:
        raise SystemExit(
            "no admitted slot has a finished run, so there is nothing to "
            "compose. A composite is a picture of several grids; one grid is "
            "a fine render, and that is what --window mesh already draws."
        )

    shared = set(base_frames)
    for entry in usable:
        shared &= set(entry["frames"])
    if only_stamps:
        wanted = set(only_stamps)
        missing = wanted - shared
        if missing:
            raise SystemExit(
                "the hours asked for are not carried by every source: "
                f"{sorted(missing)} missing from the shared set "
                f"{sorted(shared)}"
            )
        shared &= wanted
    if not shared:
        raise SystemExit(
            "the base and the overlays carry no valid time in common, so "
            "there is no hour to compose. Every source must have written a "
            "frame at the same instant."
        )
    order = sorted(shared)
    notes.append(
        f"{len(order)} shared hour(s): {order[0]} .. {order[-1]} "
        f"(base carried {len(base_frames)}, "
        + ", ".join(f"{e['slot']} carried {len(e['frames'])}" for e in usable)
        + ")"
    )

    compose = {
        "base": {
            "label": "coarse-96km",
            "mesh": str(base_mesh),
            "history": [str(base_frames[s]) for s in order],
        },
        "overlays": [],
    }
    overlay = {"labels": [], "lines": []}

    for index, entry in enumerate(usable):
        swath = entry["swath"]
        slot = entry["slot"]
        threat = str(swath.get("threat_class", "unknown"))
        compose["overlays"].append({
            "label": f"{slot}-{threat}",
            "mesh": str(entry["mesh"]),
            "history": [str(entry["frames"][s]) for s in order],
        })
        ring = swath.get("ring_deg") or []
        if ring:
            overlay["lines"].append({
                "closed": True,
                "color": RING_COLOURS[index % len(RING_COLOURS)],
                "points": [[float(lat), float(lon)] for lat, lon in ring],
            })
        centroid = swath.get("centroid_deg")
        if centroid:
            overlay["labels"].append({
                "lat": float(centroid[0]),
                "lon": float(centroid[1]),
                "text": f"{slot} {threat}",
            })
    return compose, overlay, notes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--base-mesh", required=True, type=Path)
    parser.add_argument("--base-history", required=True, nargs="+", type=Path)
    parser.add_argument("--slot", action="append", default=[],
                        metavar="SLOT=RUNDIR")
    parser.add_argument("--slot-mesh", action="append", default=[],
                        metavar="SLOT=STATIC.nc")
    parser.add_argument("--hour", action="append", default=[],
                        metavar="YYYY-MM-DD_HH.MM.SS",
                        help="restrict to these valid times; every one must "
                             "be carried by every source")
    parser.add_argument("--compose-out", required=True, type=Path)
    parser.add_argument("--overlay-out", required=True, type=Path)
    args = parser.parse_args(argv)

    base_frames: dict[str, Path] = {}
    for item in args.base_history:
        stamp = stamp_of(item)
        if stamp is None:
            raise SystemExit(f"no valid-time stamp in {item.name}")
        base_frames[stamp] = item

    compose, overlay, notes = build(
        args.plan, args.base_mesh, base_frames,
        {k: Path(v) for k, v in _pairs(args.slot, "--slot").items()},
        {k: Path(v) for k, v in _pairs(args.slot_mesh, "--slot-mesh").items()},
        args.hour or None,
    )
    args.compose_out.write_text(
        json.dumps(compose, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    args.overlay_out.write_text(
        json.dumps(overlay, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for note in notes:
        print(note, file=sys.stderr)
    print(json.dumps({
        "compose": str(args.compose_out),
        "overlay": str(args.overlay_out),
        "overlays": len(compose["overlays"]),
        "hours": len(compose["base"]["history"]),
        "rings": len(overlay["lines"]),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
