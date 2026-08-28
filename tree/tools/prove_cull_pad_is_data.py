#!/usr/bin/env python3
"""A differently-rungged ladder is one number in one row, and nothing else.

the ruling on the cascade was "build the ladder if it earns it, and the
ratios are ours to choose", with the standing condition that the number of
levels must be DATA -- tunable later without a lane.

In MPAS an intermediate resolution level is not another forecast.  The parent
is a variable-resolution mesh that ramps from the swath's own spacing out to
the background, so the atmosphere between the fine core and the cut is already
resolved at intermediate resolutions.  Cutting the limited-area domain at the
swath ring throws all of that away and hands a 71 km parent state to cells the
fine core's own size; cutting wider keeps it.  How much to keep is
``cull_pad_scale``, one number on one swath row.

So "the number of levels" is not even a discrete rung list -- it is a
continuous choice of how much of the parent's ramp to hold, and a
differently-rungged ladder is a different number in a JSON row.

THIS ASSERTS THE ABSENCE OF THE EDIT, which is the half a unit test cannot
see.  ``tests/test_swath_plan.py`` proves a row declaring 1.35 gets a cut
1.35x wider while its mesh spec is untouched; this proves the tree did not
move while that happened.

    python tools/prove_cull_pad_is_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.swath import registry  # noqa: E402
from hexcore.swath.geometry import polygon_area_km2  # noqa: E402
from hexcore.swath.history import HistoryReader  # noqa: E402
from hexcore.swath.plan import plan_cycle, plan_document  # noqa: E402

SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0,
     "bearing_deg": 300.0, "speed_km_per_hour": 22.0, "radius_km": 420.0,
     "amplitude": 4200.0},
]
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0]

#: Three ladders, declared only here and only as numbers.  1.0 is what every
#: shipped row declares and what the code did before the column existed; the
#: other two are the pads this lane measured on a real placed swath.
LADDERS = (1.0, 1.35, 1.70)


def _porcelain(repo: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prove_cull_pad_is_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells", type=int, default=10242)
    parser.add_argument("--repo", type=Path, default=ROOT.parent)
    arguments = parser.parse_args(argv)

    before = _porcelain(arguments.repo)
    rows = []
    with tempfile.TemporaryDirectory(prefix="cull-pad-") as scratch:
        work = Path(scratch)
        policy = registry.load_policy()
        history = fixture.build(
            work / "coarse.nc", cells=arguments.cells, hours=HOURS,
            scenario=SCENARIO,
        )
        for pad in LADDERS:
            document = json.loads(
                registry.DEFAULT_METRICS.read_text(encoding="utf-8")
            )
            # ONE ASSIGNMENT.  Everything else about the cycle is identical.
            for row in document["metrics"]:
                row["swath"]["cull_pad_scale"] = pad
            path = work / f"metrics-pad-{pad}.json"
            path.write_text(
                json.dumps(document, indent=2), encoding="utf-8", newline="\n"
            )
            metrics = registry.load_metrics(path)
            with HistoryReader(history) as reader:
                result = plan_cycle(reader, metrics, policy)
                plan = json.loads(
                    json.dumps(plan_document(reader, metrics, policy, result))
                )
            if not plan["admitted"]:
                print(
                    f"REFUSED: pad {pad} placed nothing, so this run proves "
                    "only that the document loaded. The claim is that a ladder "
                    "reaches an admitted swath.",
                    file=sys.stderr,
                )
                return 1
            admitted = plan["admitted"][0]
            ring = [(a, b) for a, b in admitted["ring_deg"]]
            cut = [(a, b) for a, b in admitted["cull_region"]["vertices_deg"]]
            rows.append(
                {
                    "cull_pad_scale": pad,
                    "slot_id": admitted["slot_id"],
                    "metric_id": admitted["metric_id"],
                    "ring_area_km2": round(polygon_area_km2(ring), 1),
                    "cut_area_km2": round(polygon_area_km2(cut), 1),
                    "cut_over_ring": round(
                        polygon_area_km2(cut) / polygon_area_km2(ring), 5
                    ),
                    "mesh_spec_regions": len(admitted["mesh_spec"]["regions"]),
                    "mesh_spec_spacing_km": [
                        region["spacing_km"]
                        for region in admitted["mesh_spec"]["regions"]
                    ],
                }
            )
    after = _porcelain(arguments.repo)

    report = {
        "ladders": rows,
        "git_status_unchanged": before == after,
        "source_files_touched": 0 if before == after else -1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if before != after:
        print(
            "REFUSED: 'git status --porcelain' changed while a ladder was "
            "declared. The number of levels is supposed to be data; a tree "
            "that moved means it is a code path wearing JSON's clothes.\n"
            f"before:\n{before}\nafter:\n{after}",
            file=sys.stderr,
        )
        return 1

    baseline = rows[0]
    if baseline["cut_over_ring"] != 1.0:
        print(
            f"REFUSED: at cull_pad_scale 1.0 the cut is {baseline['cut_over_ring']}x "
            "the ring rather than exactly the ring. 1.0 has to reproduce the "
            "behaviour that shipped before this column existed, or every "
            "registered regional row's bdyMask digest moves under it.",
            file=sys.stderr,
        )
        return 1
    for row in rows[1:]:
        expected = row["cull_pad_scale"] ** 2
        if not (0.97 * expected <= row["cut_over_ring"] <= 1.03 * expected):
            print(
                f"REFUSED: cull_pad_scale {row['cull_pad_scale']} produced a cut "
                f"{row['cut_over_ring']}x the ring's area, against {expected:.4f} "
                "for a similarity of that factor. The pad reached the plan but "
                "did not scale the region it names.",
                file=sys.stderr,
            )
            return 1
    if len({tuple(row["mesh_spec_spacing_km"]) for row in rows}) != 1:
        print(
            "REFUSED: the mesh spec moved with the pad. A pad moves the CUT and "
            "never a cell centre; if the refinement changes too, the fine core "
            "is no longer the same core at every pad and nothing measured "
            "across pads is a like-for-like comparison.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
