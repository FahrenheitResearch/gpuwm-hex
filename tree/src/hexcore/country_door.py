"""Compose existing mesh, init, forecast, products and score doors from data."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import time
from datetime import datetime

from .errors import MpasPortError

DATA = Path(__file__).with_name("data")
TABLE = DATA / "countries.v1.json"
SCHEMA = "gpuwm-hex.countries/v1"
STAGES = ("prepare", "forecast", "products", "score")


def load_row(name, table=TABLE):
    table = Path(table)
    data = json.loads(table.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise MpasPortError("country table schema differs; the row's stage bindings cannot be interpreted")
    rows = [r for r in data["countries"] if r["id"] == name]
    if len(rows) != 1:
        raise MpasPortError(f"country row {name!r} must occur exactly once")
    row = rows[0]
    for key in ("mesh", "init_source", "vertical_spec", "scorecard", "product_set"):
        if key not in row:
            raise MpasPortError(f"country row lacks {key}; its operational stages would be disconnected")
    from .init_sources import load_table, resolve_source
    resolve_source(load_table(), row["init_source"])
    if row["product_set"] not in data["product_sets"]:
        raise MpasPortError("country row names an absent product set")
    return row, data["product_sets"][row["product_set"]], table.parent


def plan(args):
    try:
        datetime.strptime(args.cycle, "%Y-%m-%dT%H")
    except ValueError as error:
        raise MpasPortError("country cycle must be YYYY-MM-DDTHH; it determines the init's valid time") from error
    row, products, data_dir = load_row(args.row, args.table)
    root = args.work_dir.resolve()
    mesh = row["mesh"]
    grid = args.grid or root / "mesh.grid.nc"
    static = args.static or root / "mesh.static.nc"
    init = args.init or root / "init" / "init.nc"
    spec = (data_dir / mesh["spec"]).resolve()
    vertical = (data_dir / row["vertical_spec"]).resolve()
    start = args.cycle.replace("T", "_") + ":00:00"
    common = ["--grid", str(grid), "--static", str(static)]
    initialization = ["init-source", "--source", row["init_source"], "--cycle", args.cycle,
                      "--out-dir", str(root / "init"), *common, "--vertical-spec", str(vertical)]
    for staged in args.staged:
        initialization += ["--staged", str(staged)]
    forecast = ["forecast", "--mesh", mesh["id"], *common, "--init", str(init),
                "--start-time", start, "--init-source", f"{row['init_source']} {args.cycle}; country row {args.row}",
                "--hours", str(args.hours), "--dt", str(mesh["dt_seconds"]),
                "--history-every-minutes", str(args.history_every_minutes), "--out", str(root / "forecast"),
                "--scratch", str(root / "scratch"), "--no-products"]
    if args.gpuwm_checkout is not None:
        # Without a checkout the forecast door runs the installed engine and
        # records it by version and RECORD digest, so a pip install reaches
        # this door too.
        forecast += ["--gpuwm-checkout", str(args.gpuwm_checkout)]
    for key in ("samehost_devices", "samehost_layouts", "samehost_part"):
        if getattr(args, key, None) is not None:
            forecast += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    product_argv = ["products", str(root / "forecast"), "--out", str(root / "products"), *products]
    config = dict(row["scorecard"])
    if args.observation_config:
        config.update(json.loads(args.observation_config.read_text(encoding="utf-8")))
    # Region belongs to the country row; observation file metadata does not move it.
    config["region"] = row["scorecard"]["region"]
    score = ["score", str(root / "forecast"), "--config", str(root / "observation-scorecard.config.json")]
    return {"schema": "gpuwm-hex.country-plan/v1", "row": row, "cycle": args.cycle,
            "paths": {"grid": str(grid), "static": str(static), "init": str(init), "spec": str(spec)},
            "commands": {"init": initialization, "forecast": forecast, "products": product_argv, "score": score},
            "scorecard": config}


def _prepare(args, value, status):
    from .cli import main
    root = args.work_dir.resolve()
    rows = root / "mesh-rows.json"
    from . import mesh_rows
    from .engines import resolve, ENGINES
    def resolve_engine(name):
        return resolve(next(spec for spec in ENGINES if spec.name == name))
    from .prep_store import run_or_fetch
    paths = {k: Path(v) for k, v in value["paths"].items()}
    # Mesh and static depend on the mesh request alone, never the cycle:
    # fetched from the per-mesh store when it holds them (hexcore.prep_store).
    if not paths["grid"].exists():
        run_or_fetch([str(resolve_engine("rw_mpas_mesh")), "--spec", str(paths["spec"]),
                      "--out", str(paths["grid"]), "--receipt", str(root / "mesh-receipt.json")])
    if not paths["static"].exists():
        def build_static(argv):
            from .engines import refusal_text
            log = root / "static.log"
            result = subprocess.run(argv, capture_output=True, text=True)
            log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise MpasPortError(f"rw_mpas_static exited {result.returncode} (log {log}):\n"
                                    f"{refusal_text(result.stderr)}")
            return result
        # Let the static helper resolve the same root that fetch-geog stages,
        # as the global door does. It also owns missing-archive diagnostics.
        run_or_fetch([str(resolve_engine("rw_mpas_static")), "--grid", str(paths["grid"]),
                      *(["--geog", str(args.geog.expanduser().absolute())] if args.geog else []),
                      "--out", str(paths["static"]),
                      "--receipt", str(root / "static-receipt.json")], runner=build_static)
    mesh = value["row"]["mesh"]
    registered = mesh_rows.describe_generated(name=mesh["id"], grid=paths["grid"], static=paths["static"],
        spec_path=paths["spec"], generator_receipt=root / "mesh-receipt.json", static_receipt=root / "static-receipt.json",
        point_deg=tuple(mesh["point_deg"]), fine_dx_m=mesh["fine_dx_m"], core_radius_km=mesh["core_radius_km"],
        background_km=mesh["background_km"], dt_seconds=mesh["dt_seconds"], n_levels=mesh["n_levels"],
        admission={"scope": "country row; geometry admissions remain at forecast binding"})
    mesh_rows.write_rows(rows, [registered])
    if paths["init"].is_file():
        status.note(f"init already present: {paths['init']}")
        return 0
    status.note("gpuwm-hex " + " ".join(value["commands"]["init"]))
    return main(value["commands"]["init"])


def _run_stage(name, args, value, status):
    from .cli import main
    if name == "prepare":
        return {"returncode": _prepare(args, value, status)}
    root = args.work_dir.resolve()
    if not (root / "mesh-rows.json").is_file():
        raise MpasPortError("country stages need 'country --stage prepare' first; the mesh has no registered byte identity")
    config = root / "observation-scorecard.config.json"
    if name == "score":
        from .observation_scorecard import completed_scorecard, join_automatic_scorer
        joined = (completed_scorecard(root / "forecast", config)
                  or join_automatic_scorer(root / "forecast", config))
        if joined is not None:
            status.note(joined["note"])
            print(json.dumps(joined), flush=True)
            return {**joined, "returncode": 0}
    status.note("gpuwm-hex " + " ".join(value["commands"][name]))
    rc = int(main(value["commands"][name]))
    if name == "score" and rc == 3:
        completed = completed_scorecard(root / "forecast", config)
        if completed is not None:
            status.note(completed["note"])
            return {**completed, "scorer_returncode": rc}
    return {"returncode": rc}


def run_country(args):
    from .run_status import StatusWriter, utc_now
    value = plan(args)
    if args.stage == "plan":
        print(json.dumps(value, indent=2))
        return 0
    stages = [name for name in STAGES if (args.stage == "all" or args.stage == name)
              and name not in args.skip]
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "country-plan.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    config = root / "observation-scorecard.config.json"
    config.write_text(json.dumps(value["scorecard"], indent=2) + "\n", encoding="utf-8")
    status = StatusWriter(root, "country", stages=tuple(stages), row=args.row)
    if status.log_path is not None:
        print(f"log: {status.log_path}", flush=True)
    receipt = {"schema": "gpuwm-hex.country-run/v1", "plan": str(root / "country-plan.json"),
               "row": args.row, "cycle": args.cycle, "stages": {}}
    rows_env = os.pathsep.join(filter(None, [str(root / "mesh-rows.json"), os.environ.get("GPUWM_HEX_MESH_ROWS")]))
    env = {"GPUWM_HEX_MESH_ROWS": rows_env, "GPUWM_HEX_OBSERVATION_CONFIG": str(config)}
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        for name in stages:
            started = time.monotonic()
            status.stage(name)
            timing = receipt["stages"][name] = {"started_utc": utc_now()}
            try:
                result = _run_stage(name, args, value, status)
            finally:
                timing.update(seconds=time.monotonic() - started, finished_utc=utc_now())
            timing.update(result)
            if result["returncode"] != 0:
                status.failed(f"the {name} stage exited {result['returncode']}")
                return result["returncode"]
        status.done(f"{args.row} {args.cycle}: {', '.join(stages)}")
        return 0
    except MpasPortError as error:
        status.refused(str(error))
        receipt["refused"] = str(error)
        raise
    except Exception as error:
        status.failed(f"{type(error).__name__}: {error}")
        receipt["failed"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        (root / "country-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def add_country_parser(commands):
    parser = commands.add_parser("country", help="compose operational doors from a country row")
    parser.add_argument("--row", required=True)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--cycle", required=True, help="YYYY-MM-DDTHH, the init cycle")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all",
                        help="run all stages (default), one named stage, or print the plan")
    parser.add_argument("--skip", action="append", default=[], choices=STAGES, metavar="STAGE",
                        help="leave a stage out; repeat to leave out several")
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--geog", type=Path, default=None,
                        help="WPS_GEOG root; defaults to rw_mpas_static's lookup "
                             "(the root `gpuwm fetch-geog` stages)")
    parser.add_argument("--staged", type=Path, action="append", default=[])
    parser.add_argument("--gpuwm-checkout", type=Path,
                        help="gpuwm tree whose seam files the forecast runs (default: the installed engine)")
    parser.add_argument("--observation-config", type=Path)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--history-every-minutes", type=int, default=60)
    parser.add_argument("--samehost-devices")
    parser.add_argument("--samehost-layouts", type=Path)
    parser.add_argument("--samehost-part", type=Path)
    parser.set_defaults(handler=run_country)
