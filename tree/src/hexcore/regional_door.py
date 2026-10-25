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

FROM A KEPT HEX RUN.  ``--parent-grid --parent-static --parent-init
--parent-history-dir --parent-start`` cut the region from a hex forecast that
was kept as a start (its native history frames, mesh and init, as WOOF stores
them) instead of a limited-area source.  The stages are then

1. ``mesh``      -- as above, the region's own point mesh at its spacing;
2. ``analysis``  -- the parent's state at ``--parent-start`` as an analysis
                    file (``hexcore.parent_state``: the frame rehydrated into a
                    copy of the parent's init, its clocks stamped);
3. ``init``      -- that analysis remapped onto the cull with the moving nest's
                    conservative operators (``hexcore.analysis_remap``,
                    vertical-template kind on the cull's own vertical, the
                    parent cut to a cap around the region first);
4. ``lbc``       -- the boundaries from every parent frame through the window
                    (``hexcore.nest_boundary``, capped parent, frames at once);
5. ``contract``, ``forecast``, ``products`` -- as above.

The boundary interval is the parent's history interval.  Nothing here names a
case: the parent is whatever hex run wrote those files.
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
#: The stages of a region cut from a kept hex run (``--parent-*``): the parent's state replaces the source's files.
PARENT_STAGES = ("mesh", "analysis", "init", "lbc", "contract", "forecast", "products")
#: A parent history frame's valid time, from its name (``cuda-history.2025-05-19_18.15.00.nc``).
_FRAME_TIME = __import__("re").compile(r"(\d{4}-\d{2}-\d{2}_\d{2}[.:]\d{2}[.:]\d{2})")
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
    """A limited-area row, or a global row: a global model covers any region, so it initialises and drives a cull
    the way GFS drives a WRF domain (its Vtable intermediates, one per boundary time)."""
    from .init_sources import _norm, load_table, resolve_source

    table = load_table()
    held = next((row.get("coverage", "global") for row in table["sources"]
                 if _norm(name) in {_norm(n) for n in (row["id"], *row.get("aliases", ()))}), "limited_area")
    source = resolve_source(table, name, coverage=held)
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


def parent_frames(directory: Path) -> list[tuple[datetime, Path]]:
    """A kept hex run's native history frames in ``directory``, by the valid time in their names, in time order."""
    found = []
    for path in sorted(Path(directory).glob("*history.*.nc")):
        match = _FRAME_TIME.search(path.name)
        if match:
            try:
                found.append((datetime.strptime(match.group(1).replace(".", ":"), "%Y-%m-%d_%H:%M:%S"), path))
            except ValueError:
                continue  # not a valid time: not a frame of the parent's
    return sorted(found)


def _parent(args: argparse.Namespace, hours: int) -> dict[str, Any] | None:
    """The kept hex parent (--parent-*), or None: its files, and the frames from the start through the window."""
    given = [getattr(args, name, None) for name in ("parent_grid", "parent_static", "parent_init", "parent_history_dir")]
    if not any(given) and not getattr(args, "parent_start", None):
        return None
    names = ("--parent-grid", "--parent-static", "--parent-init", "--parent-history-dir")
    missing = [flag for flag, value in zip(names, given) if value is None]
    if missing or not getattr(args, "parent_start", None):
        raise RegionalRefusal(
            f"{', '.join(missing) or '--parent-start'} was not given: a region cut from a kept hex run reads the "
            "parent's grid, static, init and history frames, and starts at one of its frames")
    for flag, value in zip(names[:3], given[:3]):
        if not Path(value).is_file():
            raise RegionalRefusal(f"{flag} {value} is not a file: the region is cut from the parent's own mesh and state")
    text = str(args.parent_start).replace("T", " ")
    try:
        start = datetime.strptime(text[:16], "%Y-%m-%d %H:%M") if len(text) > 13 else datetime.strptime(text, "%Y-%m-%d %H")
    except ValueError as error:
        raise RegionalRefusal(f"--parent-start {args.parent_start!r} is not YYYY-MM-DDTHH[:MM]") from error
    stop = start + timedelta(hours=hours)
    frames = parent_frames(Path(given[3]))
    at = [path for moment, path in frames if moment == start]
    window = [path for moment, path in frames if start < moment <= stop]
    if not at:
        raise RegionalRefusal(
            f"the parent has no history frame at {start:%Y-%m-%d %H:%M}; the region starts from the parent's state "
            f"at one of its frames ({', '.join(f'{m:%Y-%m-%d %H:%M}' for m, _ in frames[:6]) or 'none found'})")
    if not window or max(m for m, _ in frames) < stop:
        last = max((m for m, _ in frames), default=start)
        raise RegionalRefusal(
            f"the parent's frames end at {last:%Y-%m-%d %H:%M}, before this {hours} h region would "
            f"({stop:%Y-%m-%d %H:%M}): its boundaries come from the parent's frames through the window")
    return {"grid": str(Path(given[0]).absolute()), "static": str(Path(given[1]).absolute()),
            "init": str(Path(given[2]).absolute()), "frame": str(Path(at[0]).absolute()),
            "history": [str(Path(p).absolute()) for p in window], "start": start.strftime("%Y-%m-%dT%H:%M")}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    from . import mesh_shape
    from .mesh_point import drawn_shape, parse_point

    row, data_dir = load_row(args.row, args.table)
    # A drawn core (--shape/--ellipse) is the region; the circle is --point.
    drawn = drawn_shape(args)
    if drawn is not None and args.point is not None:
        raise RegionalRefusal("--point and a drawn core (--shape/--ellipse) were both given; name one")
    if drawn is not None and drawn["kind"] == "cap":
        point = (float(drawn["center_deg"][0]), float(drawn["center_deg"][1]))
        args.radius_km = float(drawn["radius_km"])
        drawn = None
    elif drawn is not None:
        point = mesh_shape.centroid(drawn)
    elif args.point is None:
        raise RegionalRefusal(
            "no fine core was named: pass --point LAT,LON, --shape FILE or "
            "--ellipse LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]")
    else:
        point = parse_point(args.point)
    probe_hours = args.hours if args.hours is not None else row["hours"]
    parent = _parent(args, int(probe_hours)) if float(probe_hours) == int(probe_hours) else None
    if parent is not None:
        # A region cut from a kept hex run: no limited-area source; its start is the parent's frame.
        table = {"run_switches": {}}
        source = {"id": "hex-parent", "display_name": "kept hex run",
                  "intermediate": {"row": None}, "first_guess": None}
        cycle = datetime.strptime(parent["start"], "%Y-%m-%dT%H:%M")
    else:
        if not getattr(args, "cycle", None):
            raise RegionalRefusal(
                "--cycle was not given: the source cycle is what the init's valid time and every boundary time "
                "are counted from (or cut the region from a kept hex run with --parent-*)")
        source_name = args.source or row["init_source"]
        table, source = resolve_regional_source(source_name)
        cycle = _cycle(args.cycle)
    lead = int(getattr(args, "lead_hours", None) or 0)
    if lead and parent is not None:
        raise RegionalRefusal("--lead-hours applies to a source cycle; a region cut from a kept hex run starts at "
                              "--parent-start")
    hours = args.hours if args.hours is not None else row["hours"]
    if float(hours) != int(hours) or int(hours) < 1:
        raise RegionalRefusal(
            f"--hours {hours} is not a whole number of hours of at least 1; the boundary "
            "series is one file per source hour, and a run ending between two of them "
            "would have no boundary file for its last stretch")
    hours = int(hours)
    # INIT FROM A FORECAST LEAD: the source's fields at cycle + lead are the start state, valid
    # at cycle + lead, and the boundaries are the same cycle's later leads.  Lead 0 is the analysis, as before.
    interval = None
    if parent is None and not (source.get("intermediate") or {}).get("row"):
        cadence = source.get("forecast") or {}
        interval = int(cadence.get("interval_hours") or 0)
        if not interval:
            raise RegionalRefusal(
                f"source {source['id']!r} declares no forecast cadence; its boundaries are its own later leads, "
                "so a region driven by it has no boundary file to read")
        if lead % interval or hours % interval or lead + hours > int(cadence.get("max_hours", 0)):
            raise RegionalRefusal(
                f"--lead-hours {lead} and --hours {hours} must be multiples of {source['id']}'s "
                f"{interval} h output and end by its {cadence.get('max_hours')} h range; a boundary time between "
                "two of its leads has no file")
    elif lead:
        raise RegionalRefusal(
            f"--lead-hours needs a source that writes forecast leads through its table row; {source['id']!r} "
            "reaches the region through its own intermediate door, which reads one cycle's hours from 0")
    start = cycle + timedelta(hours=lead)
    stop = start + timedelta(hours=hours)
    vertical = Path(args.vertical_spec) if args.vertical_spec else (data_dir / row["vertical_spec"])
    root = Path(args.work_dir).expanduser().absolute()
    first_guess = source["first_guess"]
    switches = table["run_switches"]
    return {
        "schema": "gpuwm-hex.regional-plan/v1",
        "row": row,
        "source": {"id": source["id"], "display_name": source.get("display_name"),
                   "intermediate_row": (source.get("intermediate") or {}).get("row"),
                   "first_guess": first_guess},
        "lead_hours": lead,
        "boundary_interval_hours": interval,
        "point_deg": list(point),
        "drawn_shape": drawn,
        "fine_dx_m": float(args.fine_dx_m if args.fine_dx_m is not None else row["fine_dx_m"]),
        "radius_km": (float(mesh_shape.equal_area_radius_km(drawn)) if drawn is not None else
                      float(args.radius_km if args.radius_km is not None else row["radius_km"])),
        # The halving ladder the region's mesh is built on: given, the row's, or the generator's default (120 km).
        # Any spacing is a rung of some ladder, so the caller (WOOF) names the one that holds its spacing instead of
        # being limited to the 120 km ladder's 937.5 m / 1.875 km / 3.75 km.
        "background_km": (float(args.background_km) if getattr(args, "background_km", None) is not None
                          else (float(row["background_km"]) if row.get("background_km") is not None else None)),
        "card": args.card,
        "cycle": args.cycle if parent is None else parent["start"],
        "parent": parent,
        "forecast_args": [str(item) for item in (getattr(args, "forecast_arg", None) or ())],
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
            "analysis": str(root / "parent-analysis" / "analysis.nc"),
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
    if value.get("drawn_shape"):
        # The drawn region rides to the mesh door as the row it resolved to.
        mesh_dir.mkdir(parents=True, exist_ok=True)
        drawn = Path(value["paths"]["work"]) / "drawn-shape.json"
        drawn.write_text(json.dumps(value["drawn_shape"], indent=2) + "\n", encoding="utf-8")
        core = ["--shape", str(drawn)]
    else:
        core = ["--point", f"{lat},{lon}", "--radius-km", repr(value["radius_km"])]
    argv = ["mesh-plan", *core, "--fine-dx-m", repr(value["fine_dx_m"]),
            "--generate", "--out-dir", str(mesh_dir),
            "--vertical-spec", value["paths"]["vertical_spec"]]
    if value.get("background_km") is not None:
        argv += ["--background-km", repr(value["background_km"])]
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
    if value["source"]["intermediate_row"] is None:
        return _stage_intermediate_from_table(args, value, status)
    generated = _generated(value)
    Path(value["paths"]["grib"]).mkdir(parents=True, exist_ok=True)
    argv = ["intermediate", "--source", value["source"]["intermediate_row"],
            "--from-plan", generated["_path"], "--grib-dir", value["paths"]["grib"],
            "--cycle", value["cycle"], "--hours", f"0-{value['hours']}", "--out-dir", str(met),
            "--workers", str(args.workers)]
    return {"returncode": _cli(argv, status)}


def _stage_intermediate_from_table(args, value, status) -> dict[str, Any]:
    """A table source's WPS intermediates, one per boundary time, from ``init-source --stop-after decode``.

    Every lead from the start lead to the end goes through the same acquire and Vtable decode a global init uses
    (the source row decides products, records and Vtable), several at once, and lands in the met directory under
    its valid time, where the init and boundary engines read it.
    """
    from concurrent.futures import ThreadPoolExecutor

    met = Path(value["paths"]["met"])
    met.mkdir(parents=True, exist_ok=True)
    first = int(value["lead_hours"])
    step = int(value["boundary_interval_hours"])
    leads = list(range(first, first + int(value["hours"]) + 1, step))

    def one(lead: int) -> dict[str, Any]:
        out = met / "leads" / f"f{lead:03d}"
        out.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "hexcore.cli", "init-source", "--source", value["source"]["id"],
                "--cycle", value["cycle"], "--lead-hours", str(lead), "--out-dir", str(out),
                "--stop-after", "decode"]
        with (out / "init-source.log").open("w", encoding="utf-8") as handle:
            code = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT).returncode
        made = sorted(out.glob("MET:*"))
        if code != 0 or len(made) != 1:
            raise RegionalRefusal(f"init-source for lead {lead} h exited {code} with {len(made)} intermediates; "
                                  f"see {out / 'init-source.log'}")
        target = met / made[0].name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(made[0])
        return {"lead_hours": lead, "path": str(target), "source_dir": str(out)}

    status.note(f"{value['source']['id']} {value['cycle']} leads {leads[0]}-{leads[-1]} h every {step} h")
    workers = max(1, min(len(leads), int(getattr(args, "workers", 4) or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(one, leads))
    (met / "intermediate-receipt.json").write_text(json.dumps(
        {"schema": "gpuwm-hex.table-intermediates/v1", "source": value["source"]["id"], "cycle": value["cycle"],
         "files": files}, indent=2) + "\n", encoding="utf-8")
    return {"returncode": 0, "files": len(files)}


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


def _stage_analysis(args, value, status) -> dict[str, Any]:
    """The kept parent's state at the region's start, as an analysis file (hexcore.parent_state)."""
    from .parent_state import analysis_at

    out = Path(value["paths"]["analysis"])
    if out.is_file():
        status.note(f"parent analysis already present: {out}")
        return {"returncode": 0, "reused": True}
    parent = value["parent"]
    status.note(f"the parent's state at {parent['start']} from {Path(parent['frame']).name}")
    numbers, scheme = _region_numbers(value)
    receipt = analysis_at(Path(parent["grid"]), Path(parent["static"]), Path(parent["init"]), Path(parent["frame"]), out,
                          numbers=numbers, scheme=scheme)
    return {"returncode": 0, "valid_time": receipt["valid_time"], "carried": len(receipt["carried"]),
            "numbers": receipt.get("numbers")}


def _region_numbers(value: Mapping[str, Any]) -> tuple[tuple[str, ...], int | None]:
    """The number species a region cut from a kept parent carries, and the scheme that diagnoses missing ones.

    THE BREAKAGE THIS PREVENTS (init-from-any-run proof, 2026-10-24): the region's start and boundaries carried the
    six water masses only, so a number-carrying row got mass with no number (hexcore.handover_numbers)."""
    from .handover_numbers import diagnose_scheme, region_number_species

    parent = value["parent"]
    extra = value.get("forecast_args") or []
    dx = float(value["fine_dx_m"]) if value.get("fine_dx_m") is not None else None
    return (region_number_species(extra, Path(parent["frame"]), nominal_dx_m=dx),
            diagnose_scheme(extra, nominal_dx_m=dx))


def _stage_init_from_parent(args, value, status) -> dict[str, Any]:
    """The region's start: the parent's analysis remapped onto the cull (vertical template: the cull's own vertical,
    the land and surface state carried from the parent), the parent cut to a cap around the region first."""
    from types import SimpleNamespace

    from .analysis_remap import SPECIES, transfer
    from .nest_boundary import SUBSET_MARGIN_KM

    generated = _generated(value)
    vertical = (generated.get("vertical") or {}).get("cull_artifact")
    if not vertical:
        raise RegionalRefusal(
            f"{generated['_path']} carries no culled vertical artifact; the region's start takes its vertical grid "
            "from it and its state from the parent")
    init = Path(value["paths"]["init"])
    work = init.parent
    work.mkdir(parents=True, exist_ok=True)
    parent = value["parent"]
    receipt = transfer(SimpleNamespace(
        source_grid=Path(parent["grid"]), source_static=Path(parent["static"]), analysis=Path(value["paths"]["analysis"]),
        target_grid=Path(generated["cull"]["grid"]), target_static=Path(generated["cull"]["static"]),
        template=Path(vertical), template_kind="vertical", operator=work / "remap-operator.npz", output=init,
        receipt=work / "remap-receipt.json", species=",".join(_analysis_species(Path(value["paths"]["analysis"]))),
        parent_subset="auto",
        parent_subset_margin_km=SUBSET_MARGIN_KM))
    return {"returncode": 0, "valid_time": receipt.get("valid_time"), "receipt": str(work / "remap-receipt.json")}


def _analysis_species(analysis: Path) -> tuple[str, ...]:
    """The six water masses plus the number species the parent analysis carries (written by _stage_analysis)."""
    from .analysis_remap import SPECIES
    from .handover_numbers import frame_number_species

    return SPECIES + frame_number_species(analysis)


def _stage_lbc_from_parent(args, value, status) -> dict[str, Any]:
    """The boundaries from every parent frame through the window (hexcore.nest_boundary: the parent cut to a cap
    around the region, the frames remapped at once), at the parent's history interval."""
    from types import SimpleNamespace

    from .nest_boundary import SUBSET_MARGIN_KM, produce

    generated = _generated(value)
    lbc_dir = Path(generated["cull"]["lbc_dir"])
    parent = value["parent"]
    records = produce(SimpleNamespace(
        output=lbc_dir, analysis=Path(value["paths"]["analysis"]), history=[Path(p) for p in parent["history"]],
        parent_manifest=None, wait_parent_seconds=3600.0, parent_subset="auto", parent_subset_margin_km=SUBSET_MARGIN_KM,
        source_grid=Path(parent["grid"]), source_static=Path(parent["static"]),
        target_grid=Path(generated["cull"]["grid"]), target_static=Path(generated["cull"]["static"]),
        target_init=Path(value["paths"]["init"]), operator=lbc_dir.with_name(lbc_dir.name + ".operator.npz"),
        numbers=_region_numbers(value)[0], number_scheme=_region_numbers(value)[1],
        jobs=int(getattr(args, "lbc_jobs", None) or max(1, min(8, (os.cpu_count() or 2) // 2)))))
    return {"returncode": 0, "frames": len(records)}


def _stage_init(args, value, status) -> dict[str, Any]:
    init = Path(value["paths"]["init"])
    if init.is_file():
        status.note(f"init already present: {init}")
        return {"returncode": 0, "reused": True}
    if value.get("parent"):
        return _stage_init_from_parent(args, value, status)
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
    argv += _init_suite_argv(value)
    return {"returncode": _cli(argv, status)}


def _init_suite_argv(value: Mapping[str, Any]) -> list[str]:
    """A hex suite chosen through --forecast-arg also shapes the init: its soil levels (RUC's nine) and species
    row (aerosol-aware Thompson's number concentrations).  THE BREAKAGE IT PREVENTS: a region run on rrfs_v2_dev
    from a Noah four-level init.  MPAS-A's own suites start from the init with no suite flag (case 7, Noah), which
    the init door requires, so they add nothing here."""
    from .global_door import forecast_arg_physics
    from .physics_suites import mpas_suite_names, suite_names

    suite = forecast_arg_physics(value.get("forecast_args") or []).get("suite")
    if suite and suite in suite_names() and suite not in mpas_suite_names():
        return ["--physics-suite", suite]
    return []


def _stage_lbc(args, value, status) -> dict[str, Any]:
    generated = _generated(value)
    lbc_dir = Path(generated["cull"]["lbc_dir"])
    if (lbc_dir / "lbc-door-receipt.json").is_file():
        status.note(f"boundary files already written in {lbc_dir}")
        return {"returncode": 0, "reused": True}
    if value.get("parent"):
        return _stage_lbc_from_parent(args, value, status)
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
            "--init-source", f"{value['source']['display_name']} {value['cycle']}"
                             + (f" at lead {value['lead_hours']} h (valid {value['start_time']})"
                                if value.get("lead_hours") else "") + "; "
                             f"regional row {value['row']['id']} at {value['point_deg']}",
            "--start-time", value["start_time"], "--hours", str(value["hours"]),
            "--history-every-minutes", str(value["history_every_minutes"]),
            "--dt", "row", "--out", value["paths"]["forecast"],
            "--scratch", value["paths"]["scratch"], "--no-products"]
    if args.gpuwm_checkout is not None:
        argv += ["--gpuwm-checkout", str(args.gpuwm_checkout)]
    # The caller's forecast flags (--forecast-arg) come last, so a --physics-suite given here wins.  `--dt row` stays
    # even then: the contract stage minted and admitted the class at the row step (Courant-stepped on the real min
    # dcEdge), and without it the suite route took the bare 6 s/km rule, so the forecast measured an unminted class and
    # refused after the whole preparation (release review 20, 2.9.7).  The suite route takes `--dt row`.  A physics
    # namelist whose config_dt names another step is refused here by name, since the forecast door would let it win.
    extra = value.get("forecast_args") or []
    if extra:
        from .global_door import forecast_arg_physics

        named = forecast_arg_physics(extra)
        if named.get("namelist"):
            _refuse_namelist_step(named["namelist"], float(generated["parent"]["dt_seconds"]))
        argv += extra
    return {"returncode": _cli(argv, status)}


def _refuse_namelist_step(namelist: str, row_dt_seconds: float) -> None:
    """Refuse a --physics-namelist whose config_dt is not the step the region's contract was minted for."""
    from .mpasphys.forecast_v842 import namelist_option

    try:
        text = Path(namelist).expanduser().read_text(encoding="utf-8", errors="replace")
    except OSError:
        return  # the forecast door refuses an unreadable namelist by name
    named = namelist_option(text, "nhyd_model", "config_dt")
    if named is None:
        return
    try:
        step = float(str(named).strip().replace("d", "e").replace("D", "e"))
    except ValueError:
        return
    if step != float(row_dt_seconds):
        raise RegionalRefusal(
            f"--physics-namelist {namelist} sets config_dt = {step:g} s, but this region's contract was minted and "
            f"admitted at the row step {float(row_dt_seconds):g} s (the Courant-checked step for its mesh); the forecast "
            "would measure an unminted class and refuse. Drop config_dt from the namelist, or set it to the row step.")


def _stage_products(args, value, status) -> dict[str, Any]:
    from .forecast_door import (history_files, products_command_for_this_install,
                                release_device_memory)

    generated = _generated(value)
    forecast = Path(value["paths"]["forecast"])
    request = argparse.Namespace(out=forecast, grid=Path(generated["cull"]["grid"]),
                                 init=Path(value["paths"]["init"]), start_time=value["start_time"])
    from .products_door import label_argv

    command = products_command_for_this_install(request, history_files(forecast), label_argv(args))
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
    "analysis": _stage_analysis,
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
    order = PARENT_STAGES if value.get("parent") else STAGES
    stages = [name for name in order if args.stage in ("all", name) and name not in args.skip
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
    parser.add_argument("--point", default=None, metavar="LAT,LON",
                        help="centre of a circular fine core (or draw one: --shape, --ellipse)")
    parser.add_argument("--shape", type=Path, default=None, metavar="FILE",
                        help="a drawn fine core: a Shape row JSON (polygon with [lat, lon] "
                             "vertices, lat_lon_box or cap) or GeoJSON; the outline is grown by "
                             "one ramp so the whole drawn area gets the fine spacing")
    parser.add_argument("--ellipse", default=None,
                        metavar="LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]",
                        help="a drawn elliptical fine core (azimuth of the major axis from north)")
    parser.add_argument("--cycle", default=None, metavar="YYYY-MM-DDTHH",
                        help="the source cycle (UTC); not given when the region is cut from a kept hex run (--parent-*)")
    parser.add_argument("--work-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument("--row", default=None,
                        help="limited-area row of data/countries.v1.json (default: the table's own default)")
    parser.add_argument("--table", type=Path, default=TABLE, help="alternative row table (tests)")
    parser.add_argument("--source", default=None,
                        help="init-source row (default: the row's init_source): a limited-area row, or a global "
                             "row such as gfs, which drives the region from its own later leads; "
                             "`gpuwm-hex sources` lists them")
    parser.add_argument("--lead-hours", type=int, default=0, metavar="L",
                        help="start from the source cycle's forecast at lead L: its fields at cycle + L are the "
                             "start state, valid at cycle + L, and its later leads drive the boundary (table "
                             "sources; default 0, the analysis)")
    parser.add_argument("--hours", type=float, default=None, help="forecast length (default: the row's)")
    parser.add_argument("--history-every-minutes", type=int, default=None)
    parser.add_argument("--fine-dx-m", type=float, default=None, help="fine spacing (default: the row's)")
    parser.add_argument("--radius-km", type=float, default=None, help="fine core radius (default: the row's)")
    parser.add_argument("--background-km", type=float, default=None,
                        help="the halving ladder's coarsest spacing; --fine-dx-m must be a rung of it "
                             "(background / 2^k). Default: the row's, else the mesh door's 120 km")
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
    from .products_door import add_label_arguments

    add_label_arguments(parser)
    parser.add_argument("--stage", choices=("plan", "all", *dict.fromkeys((*STAGES, *PARENT_STAGES))), default="all",
                        help="run all stages (default), one named stage, or print the plan")
    parent = parser.add_argument_group(
        "from a kept hex run", "cut the region from a hex forecast kept as a start, instead of a limited-area source")
    parent.add_argument("--parent-grid", type=Path, default=None, metavar="FILE", help="the parent's grid")
    parent.add_argument("--parent-static", type=Path, default=None, metavar="FILE", help="the parent's static")
    parent.add_argument("--parent-init", type=Path, default=None, metavar="FILE",
                        help="the parent's init (its geometry, vertical grid and reference state)")
    parent.add_argument("--parent-history-dir", type=Path, default=None, metavar="DIR",
                        help="the parent's native history frames (*history.YYYY-MM-DD_HH.MM.SS.nc)")
    parent.add_argument("--parent-start", default=None, metavar="YYYY-MM-DDTHH[:MM]",
                        help="the parent frame the region starts from")
    parent.add_argument("--lbc-jobs", type=int, default=None, metavar="N", help="boundary frames remapped at once")
    parser.add_argument("--forecast-arg", action="append", default=[], metavar="ARG",
                        help="a forecast-door flag passed through as written (e.g. --forecast-arg=--physics-suite=NAME)")
    parser.add_argument("--skip", action="append", default=[], choices=tuple(dict.fromkeys((*STAGES, *PARENT_STAGES))), metavar="STAGE",
                        help="leave a stage out; repeat to leave out several")
    parser.set_defaults(handler=run_regional)


__all__ = ["PARENT_STAGES", "STAGES", "RegionalRefusal", "add_regional_parser", "load_row", "parent_frames", "plan",
           "regional_class_key", "run_regional"]
