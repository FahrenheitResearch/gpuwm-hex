"""the arbitrary acceptance test, run against the working tree itself.

THE CLAIM: adding a COMPOUND threat -- one whose definition is a conjunction
over quantities that share no unit -- costs rows in a JSON document and
nothing else.  The unit tests assert the behaviour; this asserts the
absence of the edit, which is the half a test cannot see.

It takes ``git status --porcelain`` before, adds a three-condition threat to
a COPY of the shipped document in a scratch directory, drives the whole
placement layer with it against a fixture history, and takes
``git status --porcelain`` again.  If the two disagree, something under
``src/`` was touched to make a threat work and the property is false.

The threat is the same one ``tests/test_swath_registry.py`` defines, so the
two cannot drift: a low that is also warm and also windy, which is three
margins in pascals, kelvin and metres per second and one ``extremum_of``
row to take the weakest of them.
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
from hexcore.swath.history import HistoryReader  # noqa: E402
from hexcore.swath.plan import plan_cycle, plan_document  # noqa: E402

SCENARIO = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0,
     "bearing_deg": 300.0, "speed_km_per_hour": 22.0, "radius_km": 420.0,
     "amplitude": 4200.0},
]
HOURS = [0.0, 3.0, 6.0, 9.0, 12.0]


def _porcelain(repo: Path) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo), capture_output=True, text=True, check=True,
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prove_compound_threat_is_data",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cells", type=int, default=10242)
    parser.add_argument("--repo", type=Path, default=ROOT.parent)
    arguments = parser.parse_args(argv)

    from test_swath_registry import with_compound_threat  # noqa: PLC0415

    before = _porcelain(arguments.repo)
    with tempfile.TemporaryDirectory(prefix="compound-threat-") as scratch:
        work = Path(scratch)
        document = json.loads(registry.DEFAULT_METRICS.read_text(encoding="utf-8"))
        for row in document["metrics"]:
            row["enabled"] = False
        with_compound_threat(document)
        metrics_path = work / "compound-metrics.json"
        metrics_path.write_text(
            json.dumps(document, indent=2), encoding="utf-8", newline="\n"
        )
        metrics = registry.load_metrics(metrics_path)
        policy = registry.load_policy()

        history = fixture.build(
            work / "coarse.nc", cells=arguments.cells, hours=HOURS,
            scenario=SCENARIO,
        )
        with HistoryReader(history) as reader:
            result = plan_cycle(reader, metrics, policy)
            plan = json.loads(
                json.dumps(plan_document(reader, metrics, policy, result))
            )
    after = _porcelain(arguments.repo)

    admitted = plan["admitted"]
    report = {
        "armed": [row.id for row in metrics.armed],
        "publication_manifest": list(metrics.publication_manifest()),
        "tracks": len(plan["tracks"]),
        "admitted": [
            {
                "slot_id": row["slot_id"],
                "threat_class": row["threat_class"],
                "cull_region_kind": row["cull_region"]["kind"],
                "spacing_km": row["mesh_spec"]["regions"][0]["spacing_km"],
                "score": row["rank"]["score"],
            }
            for row in admitted
        ],
        "git_status_unchanged": before == after,
        "source_files_touched": 0 if before == after else -1,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not admitted:
        print(
            "REFUSED: the compound threat placed nothing, so the run proves "
            "only that the document loaded. The claim is that it reaches an "
            "admitted swath.",
            file=sys.stderr,
        )
        return 1
    if before != after:
        print(
            "REFUSED: 'git status --porcelain' changed while adding a threat. "
            "The arbitrary acceptance test says a phenomenon is metadata; a "
            "tree that moved means it is a code path wearing JSON's clothes.\n"
            f"before:\n{before}\nafter:\n{after}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
