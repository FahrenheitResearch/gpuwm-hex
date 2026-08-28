#!/usr/bin/env python3
"""Compare the 5 s welded and held arms over the IDENTICAL steps they share.

THE PROBLEM THIS SOLVES.  The 5 s held-cadence arm did not complete: it ran
964 of 1,440 composite steps, every one finite, and then the transactional
dycore refused to publish step 965 ("composite step at 4820.0 s was aborted
without publication"), in two independent processes, at the same step.  Its
band therefore cannot be read against the welded arm's FULL-RUN band without
comparing a 964-step run to a 1,440-step one, which would answer a different
question and flatter whichever run was shorter.

So this reads BOTH configurations over the identical first 964 steps, from
the same mesh, the same init and the same card, with convection off in both.
The only thing that differs is how often the surface layer, the land-surface
model and the PBL are called: 720 times an hour welded, 30 times an hour
held -- the proven rate.

The welded arm's per-step receipts are the ones ``convection-off-20260826``
wrote on this card; nothing is re-run, so nothing can drift.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _campaign(tree: Path):
    spec = importlib.util.spec_from_file_location(
        "_campaign", tree / "tools" / "run_dt_anchor_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == key:
                return value
            found = find(value, key)
            if found is not None:
                return found
    if isinstance(node, list):
        for value in node:
            found = find(value, key)
            if found is not None:
                return found
    return None


def summarise(campaign, health: list[dict[str, Any]]) -> dict[str, Any]:
    band = campaign.band(health)
    trend = campaign.trend(health)
    return {
        "steps": band["steps"],
        "finite_every_step": band["finite_every_step"],
        "w_max": band["vertical_velocity_abs_max"]["max"],
        "theta_m_max": band["theta_m_max"]["max"],
        "qv_max": band["qv_max"]["max"],
        "w_mean_by_quarter": [
            w["vertical_velocity_abs_max"]["mean"] for w in trend["windows"]
        ],
        "w_max_by_quarter": [
            w["vertical_velocity_abs_max"]["max"] for w in trend["windows"]
        ],
        "theta_m_max_by_quarter": [
            w["theta_m_max"]["max"] for w in trend["windows"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--welded-a", type=Path, required=True)
    parser.add_argument("--welded-b", type=Path, required=True)
    parser.add_argument("--held-a", type=Path, required=True)
    parser.add_argument("--held-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    campaign = _campaign(arguments.tree)
    held = {
        "a": json.loads(arguments.held_a.read_text(encoding="utf-8")),
        "b": json.loads(arguments.held_b.read_text(encoding="utf-8")),
    }
    welded = {
        "a": json.loads(arguments.welded_a.read_text(encoding="utf-8")),
        "b": json.loads(arguments.welded_b.read_text(encoding="utf-8")),
    }
    held_health = {k: find(v, "step_health") for k, v in held.items()}
    welded_health = {k: find(v, "step_health") for k, v in welded.items()}
    steps = len(held_health["a"])

    record: dict[str, Any] = {
        "schema": "gpuwm-hex.pbl-cadence-dt5-step-matched/v1",
        "what_this_is": (
            "the 5 s welded and held configurations compared over the "
            f"identical first {steps} steps -- same mesh, same init, same "
            "card, convection off in both.  The only difference is how "
            "often the surface layer, the land-surface model and the PBL "
            "are called: 720 times an hour welded, 30 held"
        ),
        "steps_compared": steps,
        "surface_pbl_calls_per_hour": {"welded": 720.0, "held": 30.0},
        "truncation": {
            "held_arms": {
                name: {
                    "status": value["status"],
                    "last_committed_step": find(value, "last_committed_step"),
                    "steps_requested": find(value, "steps_requested"),
                    "message": find(value, "message"),
                }
                for name, value in held.items()
            },
            "reproduces_at_the_same_step": (
                find(held["a"], "last_committed_step")
                == find(held["b"], "last_committed_step")
            ),
            "welded_completed_every_step": all(
                len(v) == 1440 for v in welded_health.values()
            ),
        },
        "welded": {
            f"arm-{k}": summarise(campaign, v[:steps])
            for k, v in welded_health.items()
        },
        "held": {
            f"arm-{k}": summarise(campaign, v)
            for k, v in held_health.items()
        },
    }
    record["held_arms_agree"] = (
        record["held"]["arm-a"] == record["held"]["arm-b"]
    )
    record["welded_arms_agree"] = (
        record["welded"]["arm-a"] == record["welded"]["arm-b"]
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({k: record[k] for k in (
        "steps_compared", "held_arms_agree", "welded_arms_agree",
    )}, indent=2))
    for label in ("welded", "held"):
        arm = record[label]["arm-a"]
        print(f"{label:7s} |w| max {arm['w_max']:9.4f}  "
              f"quarters {[round(x, 3) for x in arm['w_mean_by_quarter']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
