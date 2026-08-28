"""``gpuwm-hex cycle`` -- follow weather across cycles, not just once.

THE GAP THIS CLOSES.  Everything the cascade had was a snapshot: one coarse
run, one placement, one cull, one fine forecast, one set of images.  The plan
layer already knew how to decide whether a slot MOVES or STAYS between cycles
-- continuation radius, promotion margin, dwell, regeneration thresholds --
and there was nothing to hand that decision to.  the project law is that
engine-proven is not shipped, so the loop gets a front door and a demo, not a
driver script in an evidence folder.

TWO LEGS, because a user meets this in two ways.

``plan``  answers "what would this cascade do?" without opening a device.  It
    runs every cycle's detection, placement and hysteresis, prices the culls,
    and prints the slots each cycle would run, what each would cost, and where
    a delayed start would save the run from integrating hours nobody placed a
    grid for.  Nothing is cut and no card is touched.

``run``   does it.  Cull, mid-window initial condition when the threat is
    late, boundaries, a contract deck on this cull's own rings, a registered
    row written from the cull's own bytes, the full-physics fine forecast, and
    the frame.

NO PHENOMENON APPEARS ON THIS DOOR.  There is no ``--tropical-cyclone``, no
``--convection`` mode and no per-threat flag: what to look for is
``threat-metrics.v3``, and adding one is a row in that document.  The door's
own arguments are all about WHERE THE FILES ARE and HOW LONG TO RUN.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..errors import MpasPortError
from ..swath import registry as registry_module
from ..swath.history import HistoryReader
from ..swath.hysteresis import SwathState
from ..swath.plan import plan_cycle, plan_document
from .chain import (
    CascadeConfig,
    read_parent_stream,
    run_cascade,
    window_receipt,
    _stamp,
)
from .errors import CycleRefusal


def _require(value: Any, flag: str, breakage: str) -> Any:
    if value is None:
        raise CycleRefusal(f"{flag} was not given: {breakage}")
    return value


def _config(arguments: argparse.Namespace) -> CascadeConfig:
    repo = Path(
        arguments.repo if arguments.repo is not None
        else Path(__file__).resolve().parents[3]
    )
    return CascadeConfig(
        out=Path(_require(arguments.out, "--out", "a cascade writes a folder per cycle and there is no default place for it")),
        parent_row=str(_require(arguments.parent_row, "--parent-row", "every cull registers itself as a cull OF something, and lineage stops at a row a person registered")),
        parent_grid=Path(_require(arguments.parent_grid, "--parent-grid", "the fine grid is CUT from the parent; without the parent's grid there is nothing to cut")),
        parent_static=Path(_require(arguments.parent_static, "--parent-static", "the cull needs the parent's static or the child has no geography")),
        parent_init=Path(_require(arguments.parent_init, "--parent-init", "culling the parent's own init is the supported route to a limited-area initial condition; building the parent's own init took 775 s on v4.75.121182 against about 1 s to cull it")),
        parent_history=(
            None if arguments.parent_history is None else Path(arguments.parent_history)
        ),
        coarse_history=Path(_require(arguments.coarse_history, "--coarse-history", "the cascade detects in a coarse forecast this project produced and drives its boundaries from the same one; with none there is nothing to place a swath from")),
        coarse_parent_grid=Path(_require(arguments.coarse_parent_grid, "--coarse-parent-grid", "rw_mpas_lbc reads the coarse parent's own grid to build the boundary series")),
        gpuwm_checkout=Path(_require(arguments.gpuwm_checkout, "--gpuwm-checkout", "the cascade drives the forecast lane, whose proof harness verifies a gpuwm GIT WORKING TREE and records its HEAD, tree and dirty paths into every receipt; an install carries the pinned bytes and no commit, so it cannot name what it executed")),
        repo=repo,
        mesh_exe=None if arguments.mesh_exe is None else Path(arguments.mesh_exe),
        lbc_exe=None if arguments.lbc_exe is None else Path(arguments.lbc_exe),
        cycles=int(arguments.cycles),
        cycle_hours=float(arguments.cycle_hours),
        plan_window_hours=float(arguments.plan_window_hours),
        fine_hours=float(arguments.fine_hours),
        history_every_minutes=int(arguments.history_every_minutes),
        dt_seconds=float(arguments.dt),
        nominal_dx_m=float(arguments.nominal_dx_m),
        class_id=str(arguments.class_id),
        max_slots_per_cycle=int(arguments.max_slots),
        lbc_interval_seconds=int(arguments.lbc_interval_seconds),
        render=bool(arguments.render),
        delayed_start=bool(arguments.delayed_start),
        state=None if arguments.state is None else Path(arguments.state),
        metrics=None if arguments.metrics is None else Path(arguments.metrics),
        policy=None if arguments.policy is None else Path(arguments.policy),
    )


def run_cycle_plan(arguments: argparse.Namespace) -> int:
    """Every cycle's decision, priced, with no device opened and nothing cut."""

    config = _config(arguments)
    coarse = read_parent_stream(config.coarse_history)
    parent = (
        read_parent_stream(config.parent_history)
        if config.parent_history is not None else None
    )
    metrics = registry_module.load_metrics(config.metrics)
    policy = registry_module.load_policy(config.policy)
    state = SwathState.load(config.state) if config.state else SwathState.empty()
    scratch = Path(config.out)
    scratch.mkdir(parents=True, exist_ok=True)

    cycles: list[dict[str, Any]] = []
    lines: list[str] = []
    for index in range(int(config.cycles)):
        cycle_start = coarse.start + timedelta(hours=index * config.cycle_hours)
        end = cycle_start + timedelta(hours=config.plan_window_hours)
        receipt = window_receipt(
            coarse, cycle_start, end, scratch / f"cycle-{index + 1:02d}-window.json"
        )
        with HistoryReader(receipt) as reader:
            result = plan_cycle(reader, metrics, policy, state=state, cycle_index=index + 1)
            document = json.loads(
                json.dumps(plan_document(reader, metrics, policy, result))
            )
        state = SwathState.from_document(document["state"])
        rows = []
        lines.append(
            f"cycle {index + 1}  valid {_stamp(cycle_start)}  "
            f"admitted {len(document['admitted'])}  "
            f"declined {len(document['declined'])}  "
            f"churn {json.dumps(document['churn'], sort_keys=True)}"
        )
        for row in document["admitted"][: config.max_slots_per_cycle]:
            metric = metrics.metric_rows[row["metric_id"]]
            ignite = float(row["ignite_at_seconds"])
            start = cycle_start + timedelta(seconds=ignite)
            hysteresis = row.get("hysteresis") or {}
            # The gap a delayed start removes: everything between the PARENT's
            # init hour and the hour this swath actually wants.  Without one,
            # that gap is fine-grid integration over atmosphere nobody placed
            # a grid for.
            parent_init_time = parent.start if parent is not None else cycle_start
            gap_hours = (start - parent_init_time).total_seconds() / 3600.0
            delayed_ok = parent is not None and any(
                frame[0] == start for frame in parent.frames
            )
            rows.append({
                "slot_id": row["slot_id"],
                "metric_id": row["metric_id"],
                "threat_class": row["threat_class"],
                "cull_pad_scale": metric.swath.cull_pad_scale,
                "ignite_at_hours": round(ignite / 3600.0, 3),
                "start_time": _stamp(start),
                "predicted_cells": row["sizing"].get("predicted_cells"),
                "mesh_action": hysteresis.get("mesh_action"),
                "incumbent": hysteresis.get("incumbent"),
                "delayed_start_needed": gap_hours > 0.0,
                "delayed_start_available": delayed_ok,
                "hours_not_integrated": round(max(gap_hours, 0.0), 3),
                "fine_hours_with_delayed_start": round(config.fine_hours, 3),
                "fine_hours_without_it": round(
                    max(gap_hours, 0.0) + config.fine_hours, 3
                ),
            })
            lines.append(
                f"  {row['slot_id']:<8} {row['metric_id']:<28} "
                f"pad {metric.swath.cull_pad_scale:<5g} "
                f"ignite +{ignite / 3600.0:.2f} h  start {_stamp(start)}  "
                f"mesh {hysteresis.get('mesh_action')}  "
                f"delayed start: {gap_hours:.2f} h not integrated "
                f"({config.fine_hours:.1f} h instead of "
                f"{max(gap_hours, 0.0) + config.fine_hours:.1f} h)"
                + ("" if delayed_ok or gap_hours <= 0.0 else "  -- NOT AVAILABLE")
            )
        cycles.append({
            "cycle_index": index + 1,
            "valid_time": _stamp(cycle_start),
            "admitted": len(document["admitted"]),
            "declined": len(document["declined"]),
            "churn": document["churn"],
            "slots": rows,
        })

    document = {
        "schema": "gpuwm-hex.cascade-plan/v1",
        "cycles": cycles,
        "nothing_was_cut": True,
        "no_device_was_opened": True,
    }
    (scratch / "cascade-plan.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    print("\n".join(lines))
    print(f"\nplan {scratch / 'cascade-plan.json'} -- nothing cut, no device opened")
    return 0


def run_cycle_run(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    receipt = run_cascade(config)
    print(json.dumps({
        "out": str(config.out),
        "cycles": [
            {
                "cycle_index": cycle["cycle_index"],
                "valid_time": cycle["valid_time"],
                "admitted": cycle["admitted"],
                "ran": cycle["ran"],
                "churn": cycle["churn"],
            }
            for cycle in receipt["cycles"]
        ],
        "wall_seconds": receipt["wall_seconds"],
    }, indent=2, sort_keys=True))
    failed = [
        slot["tag"]
        for cycle in receipt["cycles"]
        for slot in cycle["slots"]
        if slot.get("ran") and slot["forecast"]["returncode"] != 0
    ]
    if failed:
        print(
            "gpuwm-hex: these fine forecasts did not finish and their logs say "
            f"why: {', '.join(failed)}",
        )
        return 1
    return 0


def add_cycle_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "cycle",
        help="follow weather across cycles: plan, cull, force, forecast, render",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    legs = parser.add_subparsers(dest="cycle_command", required=True)
    for name, handler, blurb in (
        ("plan", run_cycle_plan,
         "what every cycle would place and what a delayed start would save "
         "-- nothing is cut and no device is opened"),
        ("run", run_cycle_run,
         "run the loop: cull, mid-window init, boundaries, contract deck, "
         "full-physics fine forecast, frame"),
    ):
        leg = legs.add_parser(name, help=blurb, description=blurb)
        _add_arguments(leg)
        leg.set_defaults(handler=handler)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=Path, default=None, metavar="DIR",
                        help="a folder per cycle, plus one cascade receipt")
    parser.add_argument("--parent-row", default=None, metavar="NAME",
                        help="the REGISTERED mesh row the fine grids are cut "
                             "from. Lineage stops at a row a person "
                             "registered: a cull of an unregistered parent is "
                             "how a shape nobody looked at becomes a forecast")
    parser.add_argument("--parent-grid", type=Path, default=None, metavar="FILE")
    parser.add_argument("--parent-static", type=Path, default=None, metavar="FILE")
    parser.add_argument("--parent-init", type=Path, default=None, metavar="FILE")
    parser.add_argument(
        "--parent-history", type=Path, default=None, metavar="RECEIPT",
        help="the run receipt of the PARENT's own integration. THIS IS WHAT "
             "MAKES A DELAYED START POSSIBLE: a swath placed for weather that "
             "arrives at hour 12 is initialised from the parent's state at "
             "hour 12 instead of integrating the fine grid from hour 0 and "
             "throwing the first half away. Without it a late swath is "
             "refused by name rather than silently starting early")
    parser.add_argument(
        "--coarse-history", type=Path, default=None, metavar="RECEIPT",
        help="the run receipt of the COARSE forecast the cascade detects in "
             "and drives its boundaries from")
    parser.add_argument("--coarse-parent-grid", type=Path, default=None, metavar="FILE",
                        help="the coarse run's own grid/init, for rw_mpas_lbc")
    parser.add_argument("--gpuwm-checkout", type=Path, default=None, metavar="DIR")
    parser.add_argument("--repo", type=Path, default=None, metavar="DIR")
    parser.add_argument("--mesh-exe", type=Path, default=None, metavar="FILE")
    parser.add_argument("--lbc-exe", type=Path, default=None, metavar="FILE")
    parser.add_argument("--cycles", type=int, default=2, metavar="N")
    parser.add_argument("--cycle-hours", type=float, default=6.0, metavar="H",
                        help="how far apart two cycles are")
    parser.add_argument("--plan-window-hours", type=float, default=12.0, metavar="H",
                        help="how much of the coarse forecast each cycle "
                             "detects in")
    parser.add_argument("--fine-hours", type=float, default=6.0, metavar="H",
                        help="how long each fine forecast runs")
    parser.add_argument("--history-every-minutes", type=int, default=30, metavar="M")
    parser.add_argument("--dt", type=float, default=20.0, metavar="SECONDS",
                        help="the fine grid's timestep; must hold an anchor")
    parser.add_argument("--nominal-dx-m", type=float, default=4000.0, metavar="METRES")
    parser.add_argument("--class-id", default="graded-4457m-dt20-z7", metavar="ID",
                        help="the minted regional configuration class these "
                             "culls belong to")
    parser.add_argument("--max-slots", type=int, default=1, metavar="N",
                        help="how many admitted swaths a cycle actually runs")
    parser.add_argument("--lbc-interval-seconds", type=int, default=3600, metavar="S")
    parser.add_argument("--state", type=Path, default=None, metavar="FILE",
                        help="a swath-state to continue from, so cycle 1 here "
                             "is not cycle 1 of the world")
    parser.add_argument("--metrics", type=Path, default=None, metavar="FILE")
    parser.add_argument("--policy", type=Path, default=None, metavar="FILE")
    parser.add_argument("--no-render", dest="render", action="store_false", default=True,
                        help="skip the frame; the forecast still runs")
    parser.add_argument(
        "--no-delayed-start", dest="delayed_start", action="store_false", default=True,
        help="REFUSE a late swath instead of starting it mid-window. This is "
             "the A/B arm for what a delayed start saves, not a mode anybody "
             "should run: without it a swath placed for hour 12 either "
             "integrates from hour 0 or does not run")


def _dispatch(arguments: argparse.Namespace) -> int:
    handler = getattr(arguments, "handler", None)
    if handler is None:
        raise MpasPortError("gpuwm-hex cycle: no leg was named")
    return int(handler(arguments))


__all__ = ["add_cycle_parser", "run_cycle_plan", "run_cycle_run"]
