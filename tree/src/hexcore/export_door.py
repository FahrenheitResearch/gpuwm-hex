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


READY_NAME = "export-ready.jsonl"
HISTORY_READY_NAME = "history-ready.jsonl"


def _history_ready_paths(forecast_dir: Path) -> list[Path]:
    """Frames the running forecast has closed, in the order it listed them.

    ``history-ready.jsonl`` is appended by every forecast driver after a frame
    is durable (``{"step", "path"}``, or the products worker's longer row);
    a line still being written is skipped until it is whole.  An executor that
    keeps no such list (the variable-resolution door's) is followed by its
    durable sidecars instead (``_durable_frames``)."""

    try:
        text = (forecast_dir / HISTORY_READY_NAME).read_text(encoding="utf-8")
    except OSError:
        return _durable_frames(forecast_dir)
    seen: list[Path] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        path = row.get("path") if isinstance(row, dict) else None
        if path and Path(path) not in seen:
            seen.append(Path(path))
    return seen


HISTORY_DURABLE_GLOB = "*history.*.nc.durable.json"


def _durable_frames(forecast_dir: Path) -> list[Path]:
    """Closed frames by their durable sidecars, in valid-time (name) order.

    THE BREAKAGE THIS PREVENTS: the variable-resolution door's executor writes
    each frame and then its ``<frame>.durable.json`` sidecar, but no
    ``history-ready.jsonl``, so ``--follow`` beside it waited for a list that
    never came and exported nothing (2026-10-23).  The sidecar is written
    after the frame, so a frame whose sidecar names its full size is done (the
    same rule as hex-fast-start c21eeba2's first look).  Frames are taken in
    order and the scan stops at the first one not yet whole, so a later frame
    is never exported ahead of an earlier one."""

    frames: list[Path] = []
    for sidecar in sorted(forecast_dir.glob(HISTORY_DURABLE_GLOB)):
        frame = sidecar.with_name(sidecar.name[: -len(".durable.json")])
        try:
            whole = frame.stat().st_size == int(json.loads(sidecar.read_text(encoding="utf-8"))["bytes"])
        except (OSError, ValueError, KeyError, TypeError):
            whole = False
        if not whole:
            break
        frames.append(frame)
    return frames


def _history_folder(forecast_dir: Path) -> Path:
    """The folder holding ``history-ready.jsonl``: the run folder or a door's ``forecast/``."""

    for candidate in (forecast_dir, forecast_dir / "forecast"):
        if (candidate / HISTORY_READY_NAME).is_file():
            return candidate
    for depth in ("*", "*/*"):
        found = sorted(forecast_dir.glob(f"{depth}/{HISTORY_READY_NAME}"))
        if found:
            return found[0].parent
    for candidate in (forecast_dir, forecast_dir / "forecast"):
        if next(candidate.glob(HISTORY_DURABLE_GLOB), None) is not None:
            return candidate
    return forecast_dir


_STAMP = __import__("re").compile(r"(\d{4}-\d{2}-\d{2})[_T](\d{2})[.:_-]?(\d{2})[.:_-]?(\d{2})")


def history_valid_time(path: Path) -> _dt.datetime | None:
    """The valid time a history file's name carries (``...YYYY-MM-DD_HH.MM.SS.nc``), or None."""

    match = None
    for match in _STAMP.finditer(Path(path).name):
        pass
    if match is None:
        return None
    day, hh, mm, ss = match.groups()
    return _dt.datetime.strptime(f"{day} {hh}:{mm}:{ss}", "%Y-%m-%d %H:%M:%S")


def _follow_from(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    return _dt.datetime.strptime(text.replace("T", "_").rstrip("Z"), "%Y-%m-%d_%H:%M:%S")


def run_follow(args: argparse.Namespace) -> int:
    """``export --follow``: export each frame the moment the forecast closes it.

    THE BREAKAGE THIS PREVENTS: an ArWen domain driven by a hex parent could only
    start after the whole hex forecast had run, because the export read a finished
    run.  A nest fed this way waits hours for boundaries it needs one at a time.
    Followed, each closed frame is converted onto the window at once and filed by an
    atomic rename, then listed in ``<out>/export-ready.jsonl`` (frame path, valid time,
    bytes, sha256), which a consumer (``python -m gpuwm.local_posting``) reads as the
    frame's ready signal.  Same converter, window and files as the one-shot export;
    only the timing differs.
    """

    started = time.monotonic()
    if args.forecast_dir is None:
        raise ExportDoorError("--follow watches a forecast folder; give RUN_DIR (the forecast's --out)")
    if not args.mesh or not args.init:
        raise ExportDoorError("--follow runs beside a forecast whose receipt is not written yet; "
                              "give --mesh GRID.nc and --init INIT.nc")
    if not args.simulation_start:
        raise ExportDoorError("--follow needs --simulation-start YYYY-MM-DD_HH:MM:SS (the forecast's start), "
                              "because lead hours are numbered from it before any receipt exists")
    if not args.follow_frames or args.follow_frames < 1:
        raise ExportDoorError("--follow needs --follow-frames N, the number of history frames the forecast writes")
    forecast_dir = Path(args.forecast_dir)
    mesh, init = Path(args.mesh), Path(args.init)
    out = Path(args.out) if args.out else forecast_dir / "export"
    scratch_root = Path(args.scratch) if args.scratch else out.parent / (out.name + ".export-scratch")
    render_door.refuse_scratch_inside_out(scratch_root, out)
    window = render_door.converter_window(args.window)
    remap = render_door.resolve_remap(args.remap, composite=False)
    convert_exe = render_door.resolve_convert_exe(args.convert_exe)
    view = args.view or render_door.default_view(args.window)
    out.mkdir(parents=True, exist_ok=True)
    ready_path = out / READY_NAME
    quiet_limit = float(args.follow_timeout_minutes) * 60.0
    # A nest added while the parent runs (gpuwm-hex add-nest) starts at a frame
    # the parent closed earlier: frames before it are not the nest's.
    follow_from = _follow_from(getattr(args, "follow_from", None))
    # --follow-every-minutes: only frames on the consumer's lead grid
    # (follow_from, or the simulation start, plus k * M).  Taking every frame
    # of a parent that writes more often than the nest's boundary interval
    # stopped --follow-frames short of the nest's end (release review 30,
    # finding 2).
    every = getattr(args, "follow_every_minutes", None)
    every_s = None if not every else float(every) * 60.0
    anchor = follow_from or _follow_from(args.simulation_start)
    last_progress = time.monotonic()
    print(f"EXPORT follow frames={args.follow_frames} window={window} out={out}", flush=True)
    done: list[dict[str, Any]] = []
    converted: set[Path] = set()
    while len(done) < args.follow_frames:
        pending = [p for p in _history_ready_paths(_history_folder(forecast_dir))
                   if p not in converted and not _before(p, follow_from)]
        if not pending:
            if time.monotonic() - last_progress > quiet_limit:
                raise ExportDoorError(
                    f"--follow: {len(done)} of {args.follow_frames} frames, and the forecast listed no new "
                    f"frame for {args.follow_timeout_minutes:g} min")
            time.sleep(float(args.follow_poll_seconds))
            continue
        history = pending[0]
        if every_s is not None and not _on_grid(history, anchor, every_s):
            converted.add(history)  # between two of the consumer's leads: never its frame
            continue
        began = time.monotonic()
        scratch = scratch_root / f"frame-{len(done):04d}"
        shutil.rmtree(scratch, ignore_errors=True)
        report, _command = render_door.convert_frames(
            convert_exe, histories=[history], mesh=mesh, scratch=scratch, window=window,
            field_set=args.field_set, simulation_start=args.simulation_start, nc_format=args.nc_format,
            init=init, model_label=args.model_label, remap=remap, jobs=1)
        produced = sorted((scratch / "frames").glob("wrfout_*"))
        if len(produced) != 1:
            raise ExportDoorError(f"rw_mpas_convert wrote {len(produced)} frames for {history}; expected one")
        frame = produced[0]
        target = out / view / _valid_day(frame) / frame.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not args.overwrite:
                raise ExportDoorError(f"{target} exists; pass --overwrite to replace it")
            target.unlink()
        # Scratch sits beside --out on the same filesystem, so this is a rename:
        # a reader of the folder never sees half a frame.
        shutil.move(str(frame), str(target))
        row_report = (report.get("frames") or [{}])[0]
        row = {"frame": str(target), "history": str(history), "bytes": target.stat().st_size,
               "sha256": row_report.get("output_sha256"), "valid_time": row_report.get("valid_time"),
               "seconds": round(time.monotonic() - began, 3), "ready_unix": round(time.time(), 3)}
        with ready_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        converted.add(history)
        done.append(row)
        last_progress = time.monotonic()
        if not args.keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"EXPORT frame {len(done)}/{args.follow_frames} {target.name} {row['seconds']} s", flush=True)
    value = {
        "schema": RECEIPT_SCHEMA, "mode": "follow",
        "created_utc": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "forecast": str(forecast_dir), "mesh": str(mesh), "init": str(init),
        "simulation_start": args.simulation_start, "window": window, "view": view,
        "field_set": args.field_set, "remap": remap, "nc_format": args.nc_format,
        "frames": done, "seconds": round(time.monotonic() - started, 3),
    }
    (out / RECEIPT_NAME).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_scratch:
        shutil.rmtree(scratch_root, ignore_errors=True)
    print(f"EXPORT follow done frames={len(done)} seconds={value['seconds']}", flush=True)
    return 0


def _on_grid(path: Path, anchor: _dt.datetime, every_s: float) -> bool:
    valid = history_valid_time(path)
    if valid is None:
        raise ExportDoorError(f"--follow-every-minutes needs the valid time in the history name; {path.name} has none")
    offset = (valid - anchor).total_seconds()
    return offset >= 0 and abs(offset / every_s - round(offset / every_s)) < 1e-9


def _before(path: Path, start: _dt.datetime | None) -> bool:
    if start is None:
        return False
    valid = history_valid_time(path)
    if valid is None:
        raise ExportDoorError(f"--follow-from needs the valid time in the history name; {path.name} has none")
    return valid < start


def run_export(args: argparse.Namespace) -> int:
    from .products_door import (FORECAST_RECEIPT, discover_histories, forecast_receipt,
                                receipt_text, simulation_start)

    if getattr(args, "follow", False):
        return run_follow(args)
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
    # The any-point column store, every output hour (column_store.py): from the frames just delivered.
    from .column_store import write as write_columns
    value["columns"] = write_columns([Path(row["frame"]) for row in delivered], out / "columns",
                                     field_set=args.field_set)
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
    parser.add_argument("--follow", action="store_true",
                        help="export each frame as the running forecast closes it (history-ready.jsonl), "
                             "listing it in <out>/export-ready.jsonl; needs --mesh, --init, "
                             "--simulation-start and --follow-frames")
    parser.add_argument("--follow-frames", type=int, default=None, metavar="N",
                        help="with --follow: the number of history frames the forecast writes")
    parser.add_argument("--follow-timeout-minutes", type=float, default=240.0, metavar="MIN",
                        help="with --follow: stop when the forecast lists no new frame for this long (default 240)")
    parser.add_argument("--follow-poll-seconds", type=float, default=1.0, metavar="S")
    parser.add_argument("--follow-from", default=None, metavar="YYYY-MM-DD_HH:MM:SS",
                        help="with --follow: skip the forecast's frames valid before this time "
                             "(a nest added while the parent runs starts here)")
    parser.add_argument("--follow-every-minutes", type=float, default=None, metavar="M",
                        help="with --follow: export only frames valid at --follow-from (or the simulation "
                             "start) plus a whole multiple of M minutes, the consumer's boundary interval "
                             "(default: every frame)")


def add_export_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "export",
        help="a forecast's history on a lat-lon (or Lambert) grid as NetCDF, without drawing",
        description=__doc__.split("\n\n")[0])
    add_export_arguments(parser)
    parser.set_defaults(handler=run_export)
