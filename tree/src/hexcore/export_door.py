"""``gpuwm-hex export``: a forecast's history on a lat-lon grid, without drawing.

Until this door the only way a hex history became a regular latitude-longitude
NetCDF file was inside ``render``: the converter wrote the frames into the
render scratch, the renderer drew them, and the scratch was removed.  A user who
wanted the numbers -- to feed another model, to score with their own tool, to
open in any NetCDF reader -- had no command for it (gpuwm-global has had
``export`` since 0.1).

What it does, and nothing more: it runs the same Rust converter the render
door runs (``rw_mpas_convert``, resolved the same way), over every history
frame of a run, onto the window asked for, and files each frame as
``<out>/<view>/<valid-day>/<frame>``.  The frames are WRF-shaped
(``wrfout_d01_<valid>``: XLAT/XLONG on a regular grid, the WRF names for the
fields, the MPAS provenance in global attributes), which is what every WRF
reader and the ArWen engine's own wrfout readers open.  The converter's own
JSON receipt -- mesh, weights and per-frame output digests and the WRF fields
the history did not carry -- is kept beside them as ``export-receipt.json``.

The window vocabulary is the render door's: ``global`` (the whole sphere at the
mesh's own spacing), ``global-0.25``, ``mesh``, ``focus``, ``region:`` and
``lambert:`` specs.  ``--remap product`` (the default) interpolates the state
linearly and remaps accumulations conservatively; ``nearest`` copies the nearest
cell, which is exact at cell centres and steps at cell edges.

No numerics are done here: Python schedules, Rust converts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
import shutil
import time
from typing import Any

from . import render_door
from .errors import MpasPortError

RECEIPT_NAME = "export-receipt.json"
RECEIPT_SCHEMA = "gpuwm-hex.export/v1"


class ExportDoorError(MpasPortError):
    """The export door refuses, and says why."""


def _valid_day(frame: Path) -> str:
    """``YYYY-MM-DD`` from a ``wrfout_d01_YYYY-MM-DD_HH_MM_SS`` frame name."""

    parts = frame.name.split("_")
    for token in parts:
        if len(token) == 10 and token[4] == "-" and token[7] == "-":
            return token
    return "undated"


def run_export(args: argparse.Namespace) -> int:
    from .products_door import (FORECAST_RECEIPT, discover_histories, forecast_receipt,
                                receipt_text, simulation_start)

    started = time.monotonic()
    forecast_dir = Path(args.forecast_dir) if args.forecast_dir else None
    receipt = forecast_receipt(forecast_dir)
    if args.history:
        histories = [Path(p) for p in args.history]
    elif forecast_dir is not None:
        histories = discover_histories(forecast_dir)
    else:
        raise ExportDoorError("give the forecast folder (the forecast door's --out) or --history FILE...")
    if not histories:
        raise ExportDoorError(f"no history frames in {forecast_dir}; nothing to export")
    for path in histories:
        if not path.is_file():
            raise ExportDoorError(f"history frame {path} is missing")
    mesh_text = args.mesh or receipt_text(receipt, "mesh", "grid")
    if not mesh_text:
        raise ExportDoorError(
            f"no mesh: --mesh was not given and the forecast folder carries no {FORECAST_RECEIPT} "
            "naming the grid; the converter places the mesh's cells on the lat-lon grid from it")
    mesh = Path(mesh_text)
    if not mesh.is_file():
        raise ExportDoorError(f"the mesh {mesh} is missing; pass --mesh GRID.nc")
    init_text = args.init or receipt_text(receipt, "init", "path")
    init = Path(init_text) if init_text else None
    if init is not None and not init.is_file():
        raise ExportDoorError(
            f"the init {init} is missing; zgrid, the heights behind every pressure-level field, "
            "lives there; pass --init INIT.nc")
    start = simulation_start(args.simulation_start, forecast_dir, histories)
    if args.out:
        out = Path(args.out)
    elif forecast_dir is not None:
        out = forecast_dir / "export"
    else:
        raise ExportDoorError("no --out: with --history there is no forecast folder to file the export beside")
    scratch = Path(args.scratch) if args.scratch else out.parent / (out.name + ".export-scratch")
    render_door.refuse_scratch_inside_out(scratch, out)
    window = render_door.converter_window(args.window)
    remap = render_door.resolve_remap(args.remap, composite=False)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    view = args.view or render_door.default_view(args.window)
    print(f"EXPORT frames={len(histories)} window={window} remap={remap} out={out}", flush=True)
    report, command = render_door.convert_frames(
        convert_exe, histories=histories, mesh=mesh, scratch=scratch, window=window,
        field_set=args.field_set, simulation_start=start, nc_format=args.nc_format,
        init=init, model_label=args.model_label, remap=remap, jobs=max(1, int(args.jobs)))
    produced = sorted((scratch / "frames").glob("wrfout_*"))
    if not produced:
        raise ExportDoorError("rw_mpas_convert finished and wrote no frame; nothing was exported")
    delivered: list[dict[str, Any]] = []
    for frame in produced:
        target = out / view / _valid_day(frame) / frame.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not args.overwrite:
                raise ExportDoorError(f"{target} exists; pass --overwrite to replace it")
            target.unlink()
        shutil.move(str(frame), str(target))
        delivered.append({"frame": str(target), "bytes": target.stat().st_size})
    by_name = {Path(row.get("output", "")).name: row for row in report.get("frames", [])}
    for row in delivered:
        source = by_name.get(Path(row["frame"]).name, {})
        row["sha256"] = source.get("output_sha256")
        row["valid_time"] = source.get("valid_time")
    value = {
        "schema": RECEIPT_SCHEMA,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "forecast": str(forecast_dir) if forecast_dir else None,
        "histories": [str(p) for p in histories],
        "mesh": str(mesh),
        "init": str(init) if init else None,
        "simulation_start": start,
        "window": window,
        "view": view,
        "field_set": args.field_set,
        "remap": remap,
        "nc_format": args.nc_format,
        "frames": delivered,
        "converter": {"command": command, "report": report},
        "seconds": round(time.monotonic() - started, 3),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / RECEIPT_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"EXPORT done frames={len(delivered)} receipt={out / RECEIPT_NAME} "
          f"seconds={value['seconds']}", flush=True)
    return 0


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("forecast_dir", nargs="?", default=None, metavar="RUN_DIR",
                        help="the forecast door's --out folder (its receipt names the mesh and init)")
    parser.add_argument("--history", nargs="+", default=None, metavar="FILE",
                        help="history frames, instead of a forecast folder")
    parser.add_argument("--mesh", default=None, metavar="FILE", help="the run's grid (default: the receipt's)")
    parser.add_argument("--init", default=None, metavar="FILE", help="the run's init (default: the receipt's)")
    parser.add_argument("--out", default=None, metavar="DIR", help="default: <RUN_DIR>/export")
    parser.add_argument("--window", default="global", type=render_door.window_argument,
                        help="global (the sphere at the mesh's spacing), global-0.25, mesh, focus, "
                             "region:CLAT,CLON,W_KM,H_KM, region-latlon:CLAT,CLON,W_KM,H_KM (a regular "
                             "lat-lon box at any latitude: the window an ArWen domain takes as its parent, "
                             "with --field-set full) or lambert:... (default global)")
    parser.add_argument("--view", default=None, metavar="NAME",
                        help="folder name under --out (default: the window's view name)")
    parser.add_argument("--field-set", default="full", choices=("full", "surface"))
    parser.add_argument("--remap", default="product", choices=("product", "nearest"),
                        help="product: linear state, conservative accumulations (default); nearest: "
                             "the nearest cell's value")
    parser.add_argument("--nc-format", default="cdf5", choices=("cdf2", "cdf5"),
                        help="NetCDF variant (default cdf5: a global frame passes the CDF-2 size limit)")
    parser.add_argument("--simulation-start", default=None, metavar="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--model-label", default=render_door.DEFAULT_MODEL_LABEL)
    parser.add_argument("--jobs", type=int, default=1, metavar="N", help="frames converted at once")
    parser.add_argument("--scratch", default=None, metavar="DIR")
    parser.add_argument("--keep-scratch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--convert-exe", default=None, metavar="FILE")


def add_export_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "export",
        help="a forecast's history on a lat-lon (or Lambert) grid as NetCDF, without drawing",
        description=__doc__.split("\n\n")[0])
    add_export_arguments(parser)
    parser.set_defaults(handler=run_export)
