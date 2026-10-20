"""``gpuwm-hex global --members``: a global ensemble, resident on the card.

    gpuwm-hex global --row global-120km --source gefs --members 1-30 --cycle latest --work-dir RUN

WHAT WAS MISSING (B5, the 2026-10-16 global operations gap list): the global
door ran one state.  An ensemble of global members meant one door command per
member, each paying its own process, imports, mesh bind, kernel load and
device-stack construction (measured on the 120 km row, warm kernel cache:
22.5 s of the 32 s forecast door), and one init after another.

WHAT THIS DOES, composing doors that exist:

  init      every member's ``init-source --member N`` runs as its own process,
            ``--init-jobs`` at a time (downloads and the Rust init overlap);
  forecast  members run in GROUPS RESIDENT ON ONE CARD through the forecast
            door's own roster (``forecast --members``, :mod:`hexcore.multi_member`):
            one process binds the mesh and loads the kernels once, every member
            of the group keeps private state, a private stream and private
            history, and the card interleaves their kernels.  Groups run back to
            back on each card, one queue per card over ``--devices`` (default:
            the visible card).  The group size is ``--members-per-card``, or
            the co-tenancy table's limit for this mesh and card
            (``device_admission.CO_TENANCY_ROWS``, whose admission still
            charges each member its full solo floor); a group the card refuses
            before construction is split in two and retried.
  the rest  products, export and score run per member on its own history.

Each member's history lands in ``RUN/ensemble/gNN/memNNN/`` and the plan
(``global-plan.json``) lists it; ``ensemble.json`` maps every member to its
init, group, card, forecast folder and outcome.

A member resident with others writes the same bytes it writes alone: the
roster gives each member private module state (the forecast door's own
resident-member proof); the one-member command, ``--member N``, is the
reference this is checked against.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

SCHEMA = "gpuwm-hex.global-ensemble/v1"
REFUSED_GROUP = "co-tenancy refuses"


class EnsembleDoorError(MpasPortError):
    """The ensemble path refuses, and says why."""


def parse_members(spec: str) -> list[int]:
    """``1-30``, ``0,1,2`` or a mix; every member once, in order."""

    found: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            if high < low:
                raise EnsembleDoorError(f"--members range {part!r} runs backwards")
            found.extend(range(low, high + 1))
        else:
            found.append(int(part))
    if not found or len(set(found)) != len(found):
        raise EnsembleDoorError(f"--members {spec!r} must name each member once")
    return found


def member_plan(value: Mapping[str, Any], members: Sequence[int], lags=None) -> list[dict[str, Any]]:
    """Source cycles and leads are distinct inputs at one common valid time."""
    from .init_sources import Cycle, load_table, resolve_source
    from .lagged_members import expand_lags

    work = Path(value["paths"]["work"])
    base = list(value["commands"]["init"])
    source = resolve_source(load_table(), value["source"])
    selected = members if source.get("members") else [None]
    expanded = expand_lags(source, Cycle.parse(value["cycle"]), selected, lags or [0])
    rows = []
    for index, provenance in enumerate(expanded, 1):
        number = provenance["source_member"]
        run_id = (f"mem{number:03d}" if lags is None and number is not None else f"mem{index:03d}")
        init_dir = work / "members" / run_id / "init"
        argv = list(base)
        argv[argv.index("--out-dir") + 1] = str(init_dir)
        argv[argv.index("--cycle") + 1] = provenance["source_cycle"]
        if number is not None:
            argv += ["--member", str(number)]
        if provenance["lead_hours"]:
            argv += ["--lead-hours", str(provenance["lead_hours"])]
        rows.append({"member": number, "run_id": run_id, "init_dir": str(init_dir),
                     "init": str(init_dir / "init.nc"), "init_argv": argv, "provenance": provenance})
    return rows


def group_limit(value: Mapping[str, Any], args: argparse.Namespace) -> int:
    """``--members-per-card``, else the co-tenancy table's limit for this mesh on the smallest card."""

    if getattr(args, "members_per_card", None) not in (None, "auto"):
        limit = int(args.members_per_card)
        if not 1 <= limit <= 4:
            raise EnsembleDoorError("--members-per-card is 1 to 4: the forecast door's roster holds 1 to 4 members")
        return limit
    from .device_admission import CO_TENANCY_ROWS
    from .global_door import _visible_cards

    memory = _visible_cards()
    if not memory:
        return 1
    rows = [row for row in CO_TENANCY_ROWS if row.mesh_kind == "global"
            and int(value["mesh"]["cells"]) <= row.maximum_cells and min(memory) >= row.minimum_card_bytes]
    return max((row.maximum_members for row in rows), default=1)


def _init_done(member: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
    try:
        receipt = json.loads((Path(member["init_dir"]) / "init-source-receipt.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = member.get("provenance", {})
    return (Path(member["init"]).is_file() and (receipt.get("init") or {}).get("path") == member["init"]
            and receipt.get("member") == member["member"] and str(receipt.get("source")) == str(value["source"])
            and receipt.get("cycle") == expected.get("source_cycle")
            and receipt.get("lead_hours", 0) == expected.get("lead_hours", 0)
            and receipt.get("valid_time", receipt.get("cycle")) == expected.get("valid_time"))


def _cli(argv: Sequence[str], log: Path, env: Mapping[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run([sys.executable, "-m", "hexcore.cli", *argv], stdout=stream,
                              stderr=subprocess.STDOUT, env={**os.environ, **env}).returncode


#: Peak resident bytes of one member's rw_mpas_init per mesh cell (55 levels), with headroom.
#: Measured 2026-10-16 (unified image): x1.40962 (120 km GEFS) peak RSS 1,223 MiB (30 KiB per cell; 60 km x1.163842 4,857 MiB).
INIT_BYTES_PER_CELL = 64 * 1024


def _available_host_bytes() -> int | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def init_jobs(requested: Any, value: Mapping[str, Any], todo: int) -> int:
    """Member inits built at once: ``--init-jobs N``, or ``auto`` = bounded by cores and host RAM.

    THE BREAKAGE THIS PREVENTS, MEASURED (home RTX 5090 host, unified image 2026-10-16, 6 GEFS members at
    120 km): a fixed 4 left two of six members waiting on a 24-core box.  Each rw_mpas_init runs
    multi-threaded phases (rayon) beside single-threaded ones, so ``auto`` allows one init per four
    cores, and no more than host RAM holds at INIT_BYTES_PER_CELL per mesh cell each.
    """

    if requested not in (None, "", "auto"):
        jobs = int(requested)
        if jobs < 1:
            raise EnsembleDoorError("--init-jobs is a positive integer or auto")
        return jobs
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    jobs = max(1, min(todo, max(1, cores // 4)))
    available = _available_host_bytes()
    cells = int((value.get("mesh") or {}).get("cells") or 0)
    if available is not None and cells > 0:
        jobs = max(1, min(jobs, available // (cells * INIT_BYTES_PER_CELL)))
    return jobs


def run_inits(value: Mapping[str, Any], members: list[dict[str, Any]], args, status, env) -> dict[str, Any]:
    todo = [m for m in members if args.rebuild_init or not _init_done(m, value)]
    for member in members:
        member["init_skipped"] = member not in todo
    status.stage("init", step_count=len(todo))
    if not todo:
        status.note(f"every member's init is already built for {value['source']} {value['cycle']}")
    done = 0
    failed = []
    jobs = init_jobs(getattr(args, "init_jobs", "auto"), value, len(todo))
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_cli, m["init_argv"], Path(m["init_dir"]).parent / "init.log", env): m for m in todo}
        for future in concurrent.futures.as_completed(futures):
            member = futures[future]
            member["init_returncode"] = future.result()
            done += 1
            status.step(done, note=f"{member['run_id']} init exited {member['init_returncode']}")
            if member["init_returncode"] != 0:
                failed.append(member["run_id"])
    if failed:
        raise EnsembleDoorError(
            f"the init of {', '.join(sorted(failed))} failed (logs: members/<run_id>/init.log); an ensemble "
            f"with members missing is not run as if it were complete")
    return {"built": len(todo), "skipped": len(members) - len(todo), "jobs": jobs}


def forecast_argv(value: Mapping[str, Any], roster: Path, out: Path, scratch: Path, first_init: str) -> list[str]:
    argv = list(value["commands"]["forecast"])
    argv[argv.index("--init") + 1] = first_init
    argv[argv.index("--out") + 1] = str(out)
    argv[argv.index("--scratch") + 1] = str(scratch)
    if "--samehost-devices" in argv:  # a card per group here, never one member over several cards
        at = argv.index("--samehost-devices")
        del argv[at:at + 2]
    return [*argv, "--members", str(roster)]


def run_forecasts(value: Mapping[str, Any], members: list[dict[str, Any]], args, status, env) -> dict[str, Any]:
    work = Path(value["paths"]["work"])
    limit = group_limit(value, args)
    cards = [c.strip() for c in (args.devices or os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",") if c.strip()]
    pending: "queue.Queue[list[dict[str, Any]]]" = queue.Queue()
    for start in range(0, len(members), limit):
        pending.put(members[start:start + limit])
    groups: list[dict[str, Any]] = []
    lock = threading.Lock()
    counter = [0]
    status.stage("forecast", step_count=len(members))
    finished = [0]

    def run_group(card: str, group: list[dict[str, Any]]) -> None:
        with lock:
            counter[0] += 1
            name = f"g{counter[0]:02d}"
        out, scratch = work / "ensemble" / name, work / "scratch" / name
        roster = work / "ensemble" / f"{name}.roster.json"
        roster.parent.mkdir(parents=True, exist_ok=True)
        scratch.parent.mkdir(parents=True, exist_ok=True)
        from .stochastic import member_number, member_options
        selected = getattr(args, "stochastic", None)
        overrides = {} if selected is None else json.loads(selected)
        if "member" in overrides:
            raise EnsembleDoorError('ensemble stochastic member identity comes from the source roster')
        stochastic_seed = overrides.pop("seed", getattr(args, "stochastic_seed", 53))
        for m in group:
            identity = m["member"] if m["member"] is not None and not m["provenance"]["lag_hours"] else member_number(m["run_id"])
            m["stochastic"] = member_options(identity, seed=stochastic_seed,
                source_members=len({item["member"] for item in members}), ensemble_size=len(members), **overrides)
        roster.write_text(json.dumps([
            {"run_id": m["run_id"], "init": m["init"], "stochastic": m["stochastic"],
             "init_source": json.dumps(m["provenance"], sort_keys=True)}
            for m in group], indent=2) + "\n", encoding="utf-8")
        argv = forecast_argv(value, roster, out, scratch, group[0]["init"])
        log = work / "ensemble" / f"{name}.log"
        started = time.monotonic()
        rc = _cli(argv, log, {**env, "CUDA_VISIBLE_DEVICES": card})
        row = {"group": name, "card": card, "members": [m["run_id"] for m in group], "returncode": rc,
               "seconds": round(time.monotonic() - started, 1), "log": str(log), "out": str(out)}
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        if rc != 0 and REFUSED_GROUP in text and len(group) > 1:
            # Refused before construction (the roster's folders are removed): split and queue again.
            row["split"] = True
            half = len(group) // 2
            pending.put(group[:half])
            pending.put(group[half:])
            with lock:
                status.note(f"{name}: card {card} refused {len(group)} resident members; split in two")
        else:
            for member in group:
                member.update(group=name, card=card, forecast=str(out / member["run_id"]), forecast_returncode=rc)
            with lock:
                finished[0] += len(group)
                status.step(finished[0], note=f"{name} on card {card}: {', '.join(row['members'])} exited {rc} "
                                              f"in {row['seconds']} s")
        with lock:
            groups.append(row)

    def worker(card: str) -> None:
        while True:
            try:
                group = pending.get_nowait()
            except queue.Empty:
                with lock:
                    if finished[0] >= len(members):
                        return
                time.sleep(0.5)  # a split group may still be queued by another card
                continue
            run_group(card, group)

    threads = [threading.Thread(target=worker, args=(card,), daemon=True) for card in cards]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [m["run_id"] for m in members if m.get("forecast_returncode") != 0]
    return {"members_per_group": limit, "cards": cards, "groups": groups, "failed": failed}


def restore_forecasts(value, members):
    """A postprocessing-only invocation uses the recorded member outputs, never init folders."""
    work = Path(value["paths"]["work"])
    try:
        prior = json.loads((work / "ensemble.json").read_text(encoding="utf-8"))
        if any(prior.get(key) != value[key] for key in ("source", "cycle")) or prior.get("row") != value["row"]["id"]:
            raise ValueError("the recorded ensemble belongs to a different row, source or cycle")
        saved = {m["run_id"]: m for m in prior["members"]}
        for member in members:
            row = saved[member["run_id"]]
            forecast = Path(row["forecast"])
            if row["member"] != member["member"] or not forecast.is_dir() or not forecast.resolve().is_relative_to((work / "ensemble").resolve()):
                raise ValueError(f"no matching forecast folder for {member['run_id']}")
            member.update({key: row[key] for key in ("forecast", "group", "card", "forecast_returncode") if key in row})
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise EnsembleDoorError(f"cannot resume member postprocessing from {work / 'ensemble.json'}: {error}; run the forecast stage first") from None


def run_ensemble(value: dict[str, Any], args: argparse.Namespace, stages: list[str], status,
                 env: Mapping[str, str]) -> dict[str, Any]:
    from .global_door import _run_door

    work = Path(value["paths"]["work"])
    members = value["members"]
    if "forecast" not in stages and any(name in stages for name in ("products", "export", "score")):
        restore_forecasts(value, members)
    record: dict[str, Any] = {"schema": SCHEMA, "row": value["row"]["id"], "source": value["source"],
                              "cycle": value["cycle"], "members": members, "stages": {}}

    def save() -> None:
        (work / "ensemble.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        for name in stages:
            started = time.monotonic()
            if name == "init":
                record["stages"]["init"] = run_inits(value, members, args, status, env)
            elif name == "forecast":
                result = run_forecasts(value, members, args, status, env)
                record["stages"]["forecast"] = result
                if result["failed"]:
                    raise EnsembleDoorError(
                        f"the forecast of {', '.join(result['failed'])} failed (logs: ensemble/gNN.log)")
            elif name in ("products", "export", "score"):
                status.stage(name, step_count=len(members))
                for index, member in enumerate(members, 1):
                    forecast = member["forecast"]
                    out = work / "members" / member["run_id"]
                    if name == "products":
                        argv = ["products", forecast, "--out", str(out / "products")]
                    elif name == "export":
                        argv = ["export", forecast, "--out", str(out / "export"), "--window", "global",
                                "--field-set", args.export_field_set, "--overwrite"]
                    else:
                        config = out / "observation-scorecard.config.json"
                        config.write_text(json.dumps({**value["score_config"], "init": member["init"]}, indent=2)
                                          + "\n", encoding="utf-8")
                        argv = ["score", forecast, "--config", str(config)]
                    rc = _run_door(argv, env)
                    if name == "score" and rc == 3:
                        from .observation_scorecard import completed_scorecard
                        completed = completed_scorecard(forecast, config)
                        if completed is not None:
                            member["scorer_returncode"] = rc
                            member["score_status"] = completed["status"]
                            member["measured_metrics"] = completed["measured_metrics"]
                            status.note(completed["note"])
                            rc = 0
                    member[f"{name}_returncode"] = rc
                    status.step(index, note=f"{member['run_id']} {name} exited {rc}")
                    if rc != 0:
                        raise EnsembleDoorError(f"{name} of {member['run_id']} exited {rc}")
            else:
                continue  # mesh and static are the one-state stages, run by the caller
            record["stages"].setdefault(name, {})["seconds"] = round(time.monotonic() - started, 1)
            save()
    finally:
        save()
    return record


def add_ensemble_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lagged", default=None, help="earlier source-cycle offsets in hours, e.g. 0,6,12,18; Cartesian product with --members")
    parser.add_argument("--stochastic", default=None, metavar="JSON",
                        help="member options; default SPPT+SKEB on when fewer than 10 source members")
    parser.add_argument("--stochastic-seed", type=int, default=53)
    parser.add_argument("--members", default=None, metavar="SPEC",
                        help="ensemble members of an ensemble source, e.g. 1-30 or 0,1,2: each member's init, "
                             "then the members resident on the card in groups (see --members-per-card)")
    parser.add_argument("--members-per-card", default="auto", metavar="N|auto",
                        help="resident members per group, 1 to 4 (default: the co-tenancy table's limit)")
    parser.add_argument("--init-jobs", default="auto", metavar="N|auto",
                        help="member inits built at once (default auto: one per four cores, bounded by host RAM)")
