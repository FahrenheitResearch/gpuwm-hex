"""Prepare table-driven source members and earlier cycles at one valid time."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from .ensemble import SCHEMA, EnsembleRefusal, digest, write_json
from .init_sources import Cycle, load_table, resolve_source
from .lagged_members import expand_lags, parse_lags


def prepare_plan(args):
    row = resolve_source(load_table(getattr(args, "table", None)), getattr(args, "source", "gefs"))
    lags = parse_lags(getattr(args, "lagged", None))
    if args.members < 2:
        raise EnsembleRefusal("an ensemble needs at least two distinct inputs")
    if row.get("members"):
        block = row["members"]["perturbed"]
        numbers = list(range(block["first"], block["last"] + 1))
    else:
        numbers = [None]
    # Balance the requested total over lags, preserving native members first.
    per_cycle = (args.members + len(lags) - 1) // len(lags)
    expanded = expand_lags(row, Cycle.parse(args.source_cycle), numbers[:per_cycle], lags, args.source_hour)
    if len(expanded) < args.members:
        raise EnsembleRefusal(f"source members and lags supply {len(expanded)} distinct inputs, requested {args.members}")
    quota, remainder = divmod(args.members, len(lags))
    limits = {lag: quota + (i < remainder) for i, lag in enumerate(lags)}
    used = dict.fromkeys(lags, 0)
    result = []
    for record in expanded:
        lag = record["lag_hours"]
        if used[lag] < limits[lag]:
            result.append(record)
            used[lag] += 1
    return result


def run_prepare(args):
    plans = prepare_plan(args)
    if args.jobs is not None and args.jobs < 1:
        raise EnsembleRefusal("--jobs must be positive")
    if args.vtable is not None:
        from .init_sources import resolve_vtable
        row = resolve_source(load_table(args.table), args.source)
        if args.vtable.read_bytes() != resolve_vtable(row).read_bytes():
            raise EnsembleRefusal("--vtable differs from the source row; use --table to declare a field map")
    output = args.out.resolve()
    if output.exists():
        raise EnsembleRefusal(f"prepared ensemble directory exists: {output}; reuse its manifest")
    output.mkdir(parents=True)
    manifest = {"schema": SCHEMA, "resolution_km": args.resolution_km, "mesh": args.mesh,
                "grid": str(args.grid.resolve()), "static": str(args.static.resolve()),
                "mesh_rows": str(args.mesh_rows.resolve()), "native_template": "member-001/init.nc",
                "start_time": plans[0]["valid_time"], "members": [], "forecast_flags": []}
    receipt = {"schema": SCHEMA, "status": "preparing", "members": [], "commands": [], "deleted": []}
    started = time.monotonic()
    def prepare(index, provenance):
        name = f"member-{index:03d}"
        root = output / name
        command = [sys.executable, "-m", "hexcore.cli", "init-source", "--source", provenance["source"],
                   "--cycle", provenance["source_cycle"], "--lead-hours", str(provenance["lead_hours"]),
                   "--out-dir", str(root), "--grid", str(args.grid), "--static", str(args.static),
                   "--vertical-spec", str(args.vertical_spec)]
        if provenance["source_member"] is not None:
            command += ["--member", str(provenance["source_member"])]
        for flag, value in (("--writer", args.met_engine), ("--table", args.table)):
            if value is not None:
                command += [flag, str(value)]
        root.mkdir()
        log = root / "prepare.log"
        before = time.monotonic()
        with log.open("w") as stream:
            rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", GPUWM_HEX_NO_LOCAL_GPU="1")).returncode
        record = {"command": command, "log": str(log), "returncode": rc, "wall_seconds": time.monotonic()-before}
        if rc:
            return record, None, []
        init = root / "init.nc"
        member = {"id": name, "init": f"{name}/init.nc", "provenance": provenance,
                  "source": json.dumps(provenance, sort_keys=True), "init_sha256": digest(init)}
        deleted = []
        # Each input's number is already captured in init-source-receipt.json.
        for folder in (root / "products", root / "intermediate", root / "remap"):
            if folder.is_dir():
                for path in folder.rglob("*"):
                    if path.is_file() and not path.name.endswith((".json", ".log")):
                        deleted.append({"path": str(path), "bytes": path.stat().st_size})
                        path.unlink()
        for pattern in ("MET:*", "*.vertical.nc", "derived-geometry-*.npz"):
            for path in root.glob(pattern):
                if path.is_file():
                    deleted.append({"path": str(path), "bytes": path.stat().st_size})
                    path.unlink()
        return record, member, deleted
    try:
        jobs = min(len(plans), args.jobs or max(1, (os.cpu_count() or 1)//4))
        receipt["jobs"] = jobs
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(prepare, i, p) for i, p in enumerate(plans, 1)]
            for future in as_completed(futures):
                record, member, deleted = future.result()
                receipt["commands"].append(record)
                receipt["deleted"].extend(deleted)
                if member:
                    manifest["members"].append(member)
                write_json(output / "prepare-receipt.json", receipt)
        if any(c["returncode"] for c in receipt["commands"]):
            raise EnsembleRefusal(f"source preparation failed; see {output}/member-*/prepare.log")
        manifest["members"].sort(key=lambda m: m["id"])
        receipt["members"] = manifest["members"]
        receipt["status"] = "prepared"
        write_json(output / "ensemble-input.json", manifest)
        return 0
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["wall_seconds"] = time.monotonic()-started
        write_json(output / "prepare-receipt.json", receipt)


def add_ensemble_prepare_parser(commands):
    parser = commands.add_parser("ensemble-prepare", help="prepare source members and lagged cycles with a common valid time")
    parser.add_argument("--source", default="gefs")
    parser.add_argument("--source-cycle", required=True, help="YYYYMMDDHH or YYYY-MM-DDTHH")
    parser.add_argument("--source-hour", type=int, default=0)
    parser.add_argument("--members", type=int, default=20, help="total inputs across source members and lagged cycles")
    parser.add_argument("--lagged", default=None, help="cycle offsets in hours, e.g. 0,6,12,18")
    parser.add_argument("--resolution-km", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--met-engine", type=Path)
    parser.add_argument("--vtable", type=Path, help="legacy check: must equal the source row Vtable")
    for key in ("grid", "static", "mesh-rows", "vertical-spec", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--mesh", required=True)
    parser.set_defaults(handler=run_prepare)
