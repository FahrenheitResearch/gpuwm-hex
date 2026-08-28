"""What the hysteresis rule buys, counted both ways.

GATE LAW.  ``hexcore.swath.hysteresis`` names the breakage it prevents:
two candidates whose rank scores differ by less than the arithmetic that
produced them trade the last admitted slot every cycle, and each trade
discards a fine domain that already ran.  A named breakage without a
number is half a gate, so this tool runs the SAME placement sequence twice
-- once with the shipped rule and once with every hysteresis knob at zero
-- and counts what changed.

THE SCENARIO IS BUILT TO MAKE THE RULE WORK FOR ITS KEEP.  Two features
sit close in rank and cross over the run: one deepening, one filling.  A
rule that never has to arbitrate anything is untested, and a measurement
taken on a sequence where nothing was ever close would report zero
evictions for the armed arm, zero for the disarmed arm, and prove nothing.

The GPU-minute figures are the shipped cascade ladder's own per-slot
costs, quoted as the conversion they are.  This tool measures CHURN; the
minutes are what churn costs at the rates another lane measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(ROOT / "src"), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.swath import registry  # noqa: E402
from hexcore.swath.history import HistoryReader  # noqa: E402
from hexcore.swath.hysteresis import SwathState  # noqa: E402
from hexcore.swath.plan import plan_cycle, plan_document  # noqa: E402

#: Per-slot GPU cost on the shipped cascade ladder, the proving RTX 5070 Ti.
#: Quoted from the 0.2 program plan's L3 row (56.5 min forecast, 9.0 min
#: init); this tool does not measure them and says so.
SLOT_FORECAST_MINUTES = 56.5
SLOT_INIT_MINUTES = 9.0

#: Two near-equal features that cross, plus one that outranks both and one
#: that never competes.  Amplitudes are Pa of depression.
CHURN_SCENARIO: list[dict[str, Any]] = [
    {"kind": "low", "latitude_deg": 16.0, "longitude_deg": -52.0, "bearing_deg": 300.0,
     "speed_km_per_hour": 22.0, "radius_km": 420.0, "amplitude": 4200.0},
    {"kind": "low", "latitude_deg": 14.0, "longitude_deg": 132.0, "bearing_deg": 310.0,
     "speed_km_per_hour": 24.0, "radius_km": 390.0, "amplitude": 2800.0,
     "amplitude_swing": 320.0, "amplitude_period_hours": 12.0,
     "amplitude_phase_hours": 0.0},
    {"kind": "low", "latitude_deg": -18.0, "longitude_deg": 62.0, "bearing_deg": 240.0,
     "speed_km_per_hour": 20.0, "radius_km": 390.0, "amplitude": 2800.0,
     "amplitude_swing": 320.0, "amplitude_period_hours": 12.0,
     "amplitude_phase_hours": 6.0},
]

DISARMED = {
    "promotion_margin": 0.0,
    "minimum_dwell_cycles": 0,
    "regenerate_centroid_km": 0.001,
    "regenerate_overlap_below": 1.0,
}


def _policy_with(base: Path | None, overrides: dict[str, Any], scratch: Path) -> Path:
    document = json.loads(
        (base or registry.DEFAULT_POLICY).read_text(encoding="utf-8")
    )
    document["hysteresis"].update(overrides)
    document["budget"]["maximum_swaths"] = 2
    target = scratch / f"policy-{'-'.join(str(v) for v in overrides.values()) or 'armed'}.json"
    target.write_text(json.dumps(document, indent=2), encoding="utf-8", newline="\n")
    return target


def _sequence(
    histories: Sequence[Path], policy_path: Path, metrics: registry.MetricRegistry
) -> dict[str, Any]:
    policy = registry.load_policy(policy_path)
    state = SwathState.empty()
    cycles: list[dict[str, Any]] = []
    for index, history in enumerate(histories):
        with HistoryReader(history) as reader:
            result = plan_cycle(reader, metrics, policy, state=state, cycle_index=index)
            document = plan_document(reader, metrics, policy, result)
        state = result.state
        cycles.append({
            "cycle_index": index,
            "slots": [
                {
                    "slot_id": row["slot_id"],
                    "metric_id": row["metric_id"],
                    "centroid_deg": row["centroid_deg"],
                    "rank_score": row["rank"]["score"],
                    "mesh_action": (row["hysteresis"] or {}).get("mesh_action"),
                }
                for row in document["admitted"]
            ],
            "churn": document["churn"],
        })
    totals = {
        "cycles": len(cycles),
        "evictions": sum(entry["churn"]["evictions"] for entry in cycles),
        "mesh_generate": sum(entry["churn"]["mesh_generate"] for entry in cycles),
        "mesh_reuse": sum(entry["churn"]["mesh_reuse"] for entry in cycles),
        "new_slots": sum(entry["churn"]["new_slots"] for entry in cycles),
    }
    # A slot that changes which feature it holds between cycles is an
    # identity break -- the thing a reader sees as a cut in an animation.
    breaks = 0
    for before, after in zip(cycles, cycles[1:]):
        earlier = {row["slot_id"]: row["centroid_deg"] for row in before["slots"]}
        for row in after["slots"]:
            previous = earlier.get(row["slot_id"])
            if previous is None:
                continue
            from hexcore.swath.geometry import great_circle_km

            if great_circle_km(previous[0], previous[1], *row["centroid_deg"]) > 1000.0:
                breaks += 1
    totals["slot_identity_breaks"] = breaks
    return {"cycles": cycles, "totals": totals}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="measure_swath_hysteresis",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--cycle-hours", type=float, default=6.0)
    parser.add_argument("--cells", type=int, default=40962)
    arguments = parser.parse_args(argv)

    out = arguments.out.expanduser().resolve()
    (out / "histories").mkdir(parents=True, exist_ok=True)
    mesh = fixture.fibonacci_mesh(arguments.cells)
    histories = [
        fixture.build(
            out / "histories" / f"cycle-{index:02d}.nc",
            scenario=CHURN_SCENARIO,
            hours=[0.0, 3.0, 6.0],
            offset_hours=index * arguments.cycle_hours,
            mesh=mesh,
        )
        for index in range(arguments.cycles)
    ]

    metrics = registry.load_metrics()
    armed_path = _policy_with(None, {}, out)
    disarmed_path = _policy_with(None, DISARMED, out)
    armed = _sequence(histories, armed_path, metrics)
    disarmed = _sequence(histories, disarmed_path, metrics)

    def abandoned_minutes(totals: dict[str, Any]) -> float:
        """GPU minutes whose product nothing continues.

        An evicted slot ran a full fine forecast last cycle and this cycle
        no swath stands on that ground: the spin-up is spent, the
        animation breaks, and the next cycle starts a cold slot somewhere
        else.  The per-slot rate is QUOTED from the program plan's L3 row
        and is not measured here.
        """

        return totals["evictions"] * (SLOT_FORECAST_MINUTES + SLOT_INIT_MINUTES)

    summary = {
        "schema": "gpuwm-hex.swath-hysteresis-measurement.v1",
        "cycles": arguments.cycles,
        "cycle_hours": arguments.cycle_hours,
        "mesh_cells": arguments.cells,
        "scenario": CHURN_SCENARIO,
        "conversion": {
            "slot_forecast_minutes": SLOT_FORECAST_MINUTES,
            "slot_init_minutes": SLOT_INIT_MINUTES,
            "basis": (
                "quoted from the gpuwm-hex 0.2 program plan's L3 row; NOT measured "
                "by this tool, which measures churn only"
            ),
        },
        "armed": armed["totals"],
        "disarmed": disarmed["totals"],
        "difference": {
            key: disarmed["totals"][key] - armed["totals"][key]
            for key in armed["totals"]
            if key != "cycles"
        },
        "abandoned_gpu_minutes_avoided": round(
            abandoned_minutes(disarmed["totals"]) - abandoned_minutes(armed["totals"]), 1
        ),
        "mesh_regenerations_avoided": (
            disarmed["totals"]["mesh_generate"] - armed["totals"]["mesh_generate"]
        ),
        "armed_detail": armed["cycles"],
        "disarmed_detail": disarmed["cycles"],
        "armed_policy": str(armed_path),
        "disarmed_policy": str(disarmed_path),
    }
    (out / "HYSTERESIS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(json.dumps({
        "armed": summary["armed"],
        "disarmed": summary["disarmed"],
        "difference": summary["difference"],
        "abandoned_gpu_minutes_avoided": summary["abandoned_gpu_minutes_avoided"],
        "mesh_regenerations_avoided": summary["mesh_regenerations_avoided"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
