"""``gpuwm-hex regional`` -- a limited-area hex forecast at a point, in one command.

WHY THIS DOOR EXISTS.  A fine hex core at an arbitrary point, initialised and
driven at its boundary by a limited-area source, was five commands a user had
to chain by hand (``mesh-plan --point --generate``, ``intermediate``,
``init``, ``lbc``, then ``forecast --lbc-dir`` with ``$GPUWM_HEX_MESH_ROWS``
set by hand), plus the contract deck on the cull's own rings that the regional
opener requires.  Every one of those steps existed; nothing joined them, so a
headline regional capability had no front door.

This door composes the shipped doors in order and changes none of them:

1. ``mesh``         -- ``mesh-plan --point --generate`` (ladder spec, build,
                       admit, register the runtime rows, cull, vertical);
2. ``intermediate`` -- the source's projected-to-lat-lon intermediate door
                       over the cull's box, every hour of the run;
3. ``init``         -- the init door on the culled static and vertical, with
                       the source row's first guess and the table's run
                       switches (the same switches every global source uses);
4. ``lbc``          -- ``rw_mpas_lbc`` over the hourly intermediates;
5. ``contract``     -- the regional contract deck on THIS cull's rings, the
                       per-geometry admission the forecast opener requires;
6. ``forecast``     -- the forecast door with ``--lbc-dir``, its runtime rows
                       and contract ledger set here;
7. ``products``     -- the products door, completed before this command exits.

WHAT IS DATA.  The source is a ``limited_area`` row of the init-source table
(``data/init-sources.v1.json``) that names its intermediate row; the run
shape (fine spacing, core radius, vertical, hours, history cadence) is a
``limited_area`` row of ``data/countries.v1.json``.  Nothing below branches on
a source or case name: a second limited-area source is a row in each table.

Preparation stages reuse their outputs in ``--work-dir``. Forecast output
must have a fresh destination, as required by the forecast door.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .errors import MpasPortError

DATA = Path(__file__).with_name("data")
TABLE = DATA / "countries.v1.json"
ROWS_KEY = "limited_area"
DEFAULT_KEY = "limited_area_default"
STAGES = ("mesh", "intermediate", "init", "lbc", "contract", "forecast", "products")
RECEIPT_SCHEMA = "gpuwm-hex.regional-run/v1"


class RegionalRefusal(MpasPortError):
    """The regional run cannot proceed, and the message names what would break."""


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------
def load_row(name: str | None, table: Path = TABLE) -> tuple[dict[str, Any], Path]:
    table = Path(table)
    data = json.loads(table.read_text(encoding="utf-8"))
    rows = data.get(ROWS_KEY) or []
    wanted = name or data.get(DEFAULT_KEY)
    if not wanted:
        raise RegionalRefusal(
            f"no --row was given and {table} names no {DEFAULT_KEY}; the run shape "
            "(fine spacing, core radius, vertical) has nothing to come from")
    found = [row for row in rows if row.get("id") == wanted]
    if len(found) != 1:
        known = ", ".join(row.get("id", "?") for row in rows) or "(none)"
        raise RegionalRefusal(
            f"limited-area row {wanted!r} must occur exactly once in {table}; known: {known}")
    row = found[0]
    for key in ("init_source", "fine_dx_m", "radius_km", "vertical_spec", "hours",
                "history_every_minutes"):
        if key not in row:
            raise RegionalRefusal(
                f"limited-area row {wanted!r} lacks {key}; the stage that reads it would "
                "have no value and nothing here guesses one")
    return row, table.parent


def resolve_regional_source(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from .init_sources import load_table, resolve_source

    table = load_table()
    source = resolve_source(table, name, coverage="limited_area")
    return table, source


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def _cycle(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H")
    except ValueError as error:
        raise RegionalRefusal(
            f"--cycle {text!r} is not YYYY-MM-DDTHH; it is the source cycle the "
            "init's valid time and every boundary time are counted from") from error


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d_%H:%M:%S")


def plan(args: argparse.Namespace) -> dict[str, Any]:
    from .mesh_point import parse_point

    row, data_dir = load_row(args.row, args.table)
    source_name = args.source or row["init_source"]
    table, source = resolve_regional_source(source_name)
    point = parse_point(args.point)
    cycle = _cycle(args.cycle)
    hours = args.hours if args.hours is not None else row["hours"]
    if float(hours) != int(hours) or int(hours) < 1:
        raise RegionalRefusal(
            f"--hours {hours} is not a whole number of hours of at least 1; the boundary "
            "series is one file per source hour, and a run ending between two of them "
            "would have no boundary file for its last stretch")
    hours = int(hours)
    start = cycle
    stop = cycle + timedelta(hours=hours)
    vertical = Path(args.vertical_spec) if args.vertical_spec else (data_dir / row["vertical_spec"])
    root = Path(args.work_dir).expanduser().absolute()
    first_guess = source["first_guess"]
    switches = table["run_switches"]
    return {
        "schema": "gpuwm-hex.regional-plan/v1",
        "row": row,
        "source": {"id": source["id"], "display_name": source.get("display_name"),
                   "intermediate_row": source["intermediate"]["row"],
                   "first_guess": first_guess},
        "point_deg": list(point),
        "fine_dx_m": float(args.fine_dx_m if args.fine_dx_m is not None else row["fine_dx_m"]),
        "radius_km": float(args.radius_km if args.radius_km is not None else row["radius_km"]),
        "card": args.card,
        "cycle": args.cycle,
        "hours": hours,
        "history_every_minutes": int(args.history_every_minutes
                                     if args.history_every_minutes is not None
                                     else row["history_every_minutes"]),
        "start_time": _stamp(start),
        "stop_time": _stamp(stop),
        "run_switches": {key: value for key, value in switches.items() if key != "about"},
        "paths": {
            "work": str(root),
            "mesh": str(root / "mesh"),
            "grib": str(Path(args.grib_dir).expanduser().absolute() if args.grib_dir else root / "grib"),
            "met": str(root / "met"),
            "init": str(root / "init" / "init.nc"),
            "contract": str(root / "contract"),
            "forecast": str(root / "forecast"),
            "scratch": str(root / "scratch"),
            "vertical_spec": str(vertical.resolve()),
        },
    }


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------
def _generated(value: Mapping[str, Any]) -> dict[str, Any]:
    receipts = sorted(Path(value["paths"]["mesh"]).glob("*.point-generate.json"))
    if len(receipts) != 1:
        raise RegionalRefusal(
            f"{value['paths']['mesh']} holds {len(receipts)} point-generate receipts; the "
            "later stages read the cull, its rows file and its vertical from exactly one")
    return json.loads(receipts[0].read_text(encoding="utf-8")) | {"_path": str(receipts[0])}


def _cli(argv: list[str], status) -> int:
    from . import cli

    status.note("gpuwm-hex " + " ".join(str(item) for item in argv))
    # The forecast runs in its own process, so its device memory ends with it (cli.DEVICE_COMMANDS).
    return cli.run_stage([str(item) for item in argv])


def _stage_mesh(args, value, status) -> dict[str, Any]:
    mesh_dir = Path(value["paths"]["mesh"])
    if list(mesh_dir.glob("*.point-generate.json")):
        status.note(f"mesh already generated in {mesh_dir}")
        return {"returncode": 0, "reused": True}
    lat, lon = value["point_deg"]
    argv = ["mesh-plan", "--point", f"{lat},{lon}", "--fine-dx-m", repr(value["fine_dx_m"]),
            "--radius-km", repr(value["radius_km"]), "--generate", "--out-dir", str(mesh_dir),
            "--vertical-spec", value["paths"]["vertical_spec"]]
    if value["card"]:
        argv += ["--card", value["card"]]
    if args.geog is not None:
        argv += ["--geog", str(Path(args.geog).expanduser().absolute())]
    return {"returncode": _cli(argv, status)}


def _stage_intermediate(args, value, status) -> dict[str, Any]:
    met = Path(value["paths"]["met"])
    if (met / "intermediate-receipt.json").is_file():
        status.note(f"intermediates already written in {met}")
        return {"returncode": 0, "reused": True}
    generated = _generated(value)
    Path(value["paths"]["grib"]).mkdir(parents=True, exist_ok=True)
    argv = ["intermediate", "--source", value["source"]["intermediate_row"],
            "--from-plan", generated["_path"], "--grib-dir", value["paths"]["grib"],
            "--cycle", value["cycle"], "--hours", f"0-{value['hours']}", "--out-dir", str(met),
            "--workers", str(args.workers)]
    return {"returncode": _cli(argv, status)}


def _met_file(value: Mapping[str, Any]) -> Path:
    receipt = json.loads((Path(value["paths"]["met"]) / "intermediate-receipt.json").read_text(encoding="utf-8"))
    files = [Path(item["path"]) for item in receipt["files"]]
    stamp = value["start_time"][:13]
    found = [path for path in files if path.name.endswith(":" + stamp)]
    if len(found) != 1:
        raise RegionalRefusal(
            f"the intermediates {[p.name for p in files]} hold no single file valid at "
            f"{stamp}; the init would start from another hour than the run's start time")
    return found[0]


def _stage_init(args, value, status) -> dict[str, Any]:
    init = Path(value["paths"]["init"])
    if init.is_file():
        status.note(f"init already present: {init}")
        return {"returncode": 0, "reused": True}
    generated = _generated(value)
    vertical = (generated.get("vertical") or {}).get("cull_artifact")
    if not vertical:
        raise RegionalRefusal(
            f"{generated['_path']} carries no culled vertical artifact; a limited-area init "
            "takes its vertical from the parent's culled artifact, never from exterior state")
    init.parent.mkdir(parents=True, exist_ok=True)
    guess = value["source"]["first_guess"]
    switches = value["run_switches"]
    argv = ["init", "--met", str(_met_file(value)), "--static", generated["cull"]["static"],
            "--capsule", vertical, "--reference", vertical, "--out", str(init),
            "--start-time", value["start_time"],
            "--nfglevels", str(guess["nfglevels"]), "--nfgsoillevels", str(guess["nfgsoillevels"]),
            "--use-spechumd", guess["use_spechumd"],
            "--extrap-airtemp", switches["extrap_airtemp"],
            "--theta-adv-order", str(switches["theta_adv_order"]),
            "--coef-3rd-order", repr(float(switches["coef_3rd_order"])),
            "--virtual-factor", switches["virtual_factor"],
            "--deep-soil-moisture", switches["deep_soil_moisture"],
            "--landuse-table", switches["landuse_table"],
            "--frac-seaice", switches["frac_seaice"],
            "--tsk-seaice-threshold", repr(float(switches["tsk_seaice_threshold"])),
            "--oned-underflow", switches["oned_underflow"]]
    return {"returncode": _cli(argv, status)}


def _stage_lbc(args, value, status) -> dict[str, Any]:
    generated = _generated(value)
    lbc_dir = Path(generated["cull"]["lbc_dir"])
    if (lbc_dir / "lbc-door-receipt.json").is_file():
        status.note(f"boundary files already written in {lbc_dir}")
        return {"returncode": 0, "reused": True}
    guess = value["source"]["first_guess"]
    switches = value["run_switches"]
    argv = ["lbc", "--grid", value["paths"]["init"], "--met-dir", value["paths"]["met"],
            "--out-dir", str(lbc_dir), "--start-time", value["start_time"],
            "--stop-time", value["stop_time"], "--nfglevels", str(guess["nfglevels"]),
            "--extrap-airtemp", switches["extrap_airtemp"], "--use-spechumd", guess["use_spechumd"],
            "--theta-adv-order", str(switches["theta_adv_order"]),
            "--coef-3rd-order", repr(float(switches["coef_3rd_order"])),
            "--oned-underflow", switches["oned_underflow"]]
    return {"returncode": _cli(argv, status)}


def regional_class_key(grid: Path, static: Path, *, n_vert_levels: int, dt_seconds: float):
    """The configuration key measured from this cull, before mint or admission.

    The same key the forecast opener measures (zone width, column count,
    finest edge, timestep, kernel set), so the deck is run under the class
    the forecast will be admitted against.  The mesh is the grid AND static
    pair the forecast binds: a generated grid is written on the unit sphere,
    and its lengths in metres are the static's.
    """

    import numpy as np

    from .cuda_backend.regional_admission import (
        RegionalClassKey,
        kernel_set_sha256,
    )
    from .mesh import Mesh

    mesh = Mesh.from_netcdf(grid, static)
    zone = int(np.max(np.asarray(mesh.arrays["bdyMaskCell"])))
    finest = float(np.min(np.asarray(mesh.dcEdge)))
    key = RegionalClassKey.build(boundary_zone_width=zone, n_vert_levels=int(n_vert_levels),
                                 finest_edge_m=finest, dt_seconds=float(dt_seconds),
                                 kernel_set=kernel_set_sha256())
    return key


def _stage_contract(args, value, status) -> dict[str, Any]:
    out = Path(value["paths"]["contract"]) / "contract.json"
    from .cuda_backend.regional_admission import admitted_class_for_key, contract_receipt_defects
    from .mesh import Mesh, regional_boundary_mask_digest
    from .regional_mint import mint

    generated = _generated(value)
    grid = Path(generated["cull"]["grid"])
    static = Path(generated["cull"]["static"])
    key = regional_class_key(grid, static, n_vert_levels=int(generated["vertical"]["n_vert_levels"]),
                             dt_seconds=float(generated["parent"]["dt_seconds"]))
    klass = admitted_class_for_key(key)
    if klass is None:
        klass = mint(key, generated, value, status)
    class_id = klass.class_id
    if out.is_file():
        held = json.loads(out.read_text(encoding="utf-8"))
        mesh = Mesh.from_netcdf(grid, static)
        defects = contract_receipt_defects(
            held, bdy_mask_sha256=regional_boundary_mask_digest(mesh.arrays),
            n_cells=mesh.nCells, kernel_set=key.kernel_set_sha256)
        if not defects and held.get("class_id") == class_id:
            status.note(f"contract deck receipt already present: {out}")
            return {"returncode": 0, "reused": True, "class_id": class_id,
                    "mint_receipts": list(klass.mint_receipts), "receipt": str(out)}
        status.note(f"{out} does not match this class and geometry; running the deck again")
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "hexcore.drivers.run_cuda_regional_contract",
            "--grid", str(grid), "--init", value["paths"]["init"],
            "--lbc-dir", generated["cull"]["lbc_dir"], "--class-id", class_id,
            "--mesh-row", generated["cull"]["row"], "--start-time", value["start_time"],
            "--out", str(out)]
    status.note(" ".join(argv))
    log = out.parent / "contract.log"
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT).returncode
    if code != 0 or not out.is_file():
        raise RegionalRefusal(
            f"the contract deck refused this cull (exit {code}; log {log}).  The regional "
            "kernels are indexed by ring, so without a deck on THESE rings nothing has "
            "checked this cull's specified and relaxation zones, and the forecast opener "
            "refuses it")
    return {"returncode": 0, "class_id": class_id, "mint_receipts": list(klass.mint_receipts),
            "receipt": str(out), "log": str(log)}


def _stage_forecast(args, value, status) -> dict[str, Any]:
    generated = _generated(value)
    cull = generated["cull"]
    argv = ["forecast", "--mesh", cull["row"], "--grid", cull["grid"], "--static", cull["static"],
            "--init", value["paths"]["init"], "--lbc-dir", cull["lbc_dir"],
            "--init-source", f"{value['source']['display_name']} {value['cycle']}; "
                             f"regional row {value['row']['id']} at {value['point_deg']}",
            "--start-time", value["start_time"], "--hours", str(value["hours"]),
            "--history-every-minutes", str(value["history_every_minutes"]),
            "--dt", "row", "--out", value["paths"]["forecast"],
            "--scratch", value["paths"]["scratch"], "--no-products"]
    if args.gpuwm_checkout is not None:
        argv += ["--gpuwm-checkout", str(args.gpuwm_checkout)]
    return {"returncode": _cli(argv, status)}


def _stage_products(args, value, status) -> dict[str, Any]:
    from .forecast_door import (history_files, products_command_for_this_install,
                                release_device_memory)

    generated = _generated(value)
    forecast = Path(value["paths"]["forecast"])
    request = argparse.Namespace(out=forecast, grid=Path(generated["cull"]["grid"]),
                                 init=Path(value["paths"]["init"]), start_time=value["start_time"])
    command = products_command_for_this_install(request, history_files(forecast))
    released = release_device_memory()
    if released is not None:
        status.note(f"device memory before products: {released}")
    code = _cli(command[1:], status)
    manifests = sorted((forecast / "products").glob("*/products-manifest.json"))
    if len(manifests) != 1:
        raise RegionalRefusal(f"products exited {code} but wrote {len(manifests)} manifests in "
                              f"{forecast / 'products'}; expected exactly one")
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    counts = {key: sum((view.get("counts") or {}).get(key, 0) for view in manifest["views"])
              for key in ("rendered", "skipped", "failed")}
    return {"returncode": code, "manifest": str(manifests[0]), "counts": counts,
            "views": [view["view"] for view in manifest["views"]]}


_RUNNERS = {
    "mesh": _stage_mesh,
    "intermediate": _stage_intermediate,
    "init": _stage_init,
    "lbc": _stage_lbc,
    "contract": _stage_contract,
    "forecast": _stage_forecast,
    "products": _stage_products,
}


def run_regional(args: argparse.Namespace) -> int:
    from .run_status import StatusWriter, utc_now

    value = plan(args)
    if args.stage == "plan":
        print(json.dumps(value, indent=2))
        return 0
    stages = [name for name in STAGES if args.stage in ("all", name) and name not in args.skip
              and not (name == "products" and args.no_products)]
    root = Path(value["paths"]["work"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "regional-plan.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    status = StatusWriter(root, "regional", stages=tuple(stages), row=value["row"]["id"])
    if status.log_path is not None:
        print(f"log: {status.log_path}", flush=True)
    started = time.monotonic()
    receipt: dict[str, Any] = {"schema": RECEIPT_SCHEMA, "plan": str(root / "regional-plan.json"),
                               "row": value["row"]["id"], "source": value["source"]["id"],
                               "point_deg": value["point_deg"], "cycle": value["cycle"],
                               "started_utc": utc_now(), "outcome": "running", "mesh": None,
                               "products": None, "stages": {}}

    def write_receipt():
        temporary = root / "regional-receipt.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(root / "regional-receipt.json")

    write_receipt()
    # The forecast door finds the cull's registered row through
    # $GPUWM_HEX_MESH_ROWS and a fresh cull's contract deck through
    # $GPUWM_HEX_REGIONAL_CONTRACT_DIR; both are this run's own files.
    rows_file = Path(value["paths"]["mesh"]) / "mesh-rows.json"
    env = {
        "GPUWM_HEX_MESH_ROWS": os.pathsep.join(filter(None, [str(rows_file),
                                                             os.environ.get("GPUWM_HEX_MESH_ROWS")])),
        "GPUWM_HEX_REGIONAL_CONTRACT_DIR": os.pathsep.join(filter(None, [
            value["paths"]["contract"], os.environ.get("GPUWM_HEX_REGIONAL_CONTRACT_DIR")])),
    }
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        for name in stages:
            stage_started = time.monotonic()
            status.stage(name)
            timing = receipt["stages"][name] = {"started_utc": utc_now(), "outcome": "running"}
            write_receipt()
            try:
                result = _RUNNERS[name](args, value, status)
                timing.update(result)
                timing["outcome"] = "done" if result["returncode"] == 0 else "failed"
            except BaseException as error:
                timing.update(outcome="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                timing.update(seconds=round(time.monotonic() - stage_started, 2), finished_utc=utc_now())
                write_receipt()
            if name == "mesh" and result["returncode"] == 0:
                receipt["mesh"] = _generated(value)["cull"]
            if name == "products":
                receipt["products"] = result
            if result["returncode"] != 0:
                receipt.update(outcome="failed", returncode=int(result["returncode"]))
                status.failed(f"the {name} stage exited {result['returncode']}")
                return int(result["returncode"])
        receipt.update(outcome="done", returncode=0)
        status.done(f"{value['row']['id']} {value['cycle']} at {value['point_deg']}: {', '.join(stages)}")
        return 0
    except MpasPortError as error:
        status.refused(str(error))
        receipt.update(outcome="refused", returncode=2)
        receipt["refused"] = str(error)
        raise
    except Exception as error:
        status.failed(f"{type(error).__name__}: {error}")
        receipt.update(outcome="failed", returncode=1)
        receipt["failed"] = f"{type(error).__name__}: {error}"
        raise
    except BaseException as error:
        receipt.update(outcome="interrupted", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        receipt.update(seconds=round(time.monotonic() - started, 2), finished_utc=utc_now())
        write_receipt()


def add_regional_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "regional",
        help="one command: limited-area hex at a point from a regional source "
             "(mesh, intermediate, init, boundaries, contract deck, forecast, products)",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--point", required=True, metavar="LAT,LON", help="centre of the fine core")
    parser.add_argument("--cycle", required=True, metavar="YYYY-MM-DDTHH", help="the source cycle (UTC)")
    parser.add_argument("--work-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--row", default=None,
                        help="limited-area row of data/countries.v1.json (default: the table's own default)")
    parser.add_argument("--table", type=Path, default=TABLE, help="alternative row table (tests)")
    parser.add_argument("--source", default=None,
                        help="limited-area init-source row (default: the row's init_source); "
                             "`gpuwm-hex sources` lists them")
    parser.add_argument("--hours", type=float, default=None, help="forecast length (default: the row's)")
    parser.add_argument("--history-every-minutes", type=int, default=None)
    parser.add_argument("--fine-dx-m", type=float, default=None, help="fine spacing (default: the row's)")
    parser.add_argument("--radius-km", type=float, default=None, help="fine core radius (default: the row's)")
    parser.add_argument("--card", default=None,
                        help="a measured card key, so the mesh plan prices the cull on it")
    parser.add_argument("--vertical-spec", type=Path, default=None, help="default: the row's")
    parser.add_argument("--grib-dir", type=Path, default=None,
                        help="already-fetched source files (default: <work-dir>/grib; missing hours are fetched)")
    parser.add_argument("--geog", type=Path, default=None, help="WPS_GEOG root for the static")
    parser.add_argument("--workers", type=int, default=4, help="intermediate decoder workers")
    parser.add_argument("--gpuwm-checkout", type=Path, default=None,
                        help="gpuwm tree whose seam files the forecast runs (default: the installed engine)")
    parser.add_argument("--no-products", action="store_true", help="end at the history frames")
    parser.add_argument("--stage", choices=("plan", "all", *STAGES), default="all",
                        help="run all stages (default), one named stage, or print the plan")
    parser.add_argument("--skip", action="append", default=[], choices=STAGES, metavar="STAGE",
                        help="leave a stage out; repeat to leave out several")
    parser.set_defaults(handler=run_regional)


__all__ = ["STAGES", "RegionalRefusal", "add_regional_parser", "load_row", "plan",
           "regional_class_key", "run_regional"]
