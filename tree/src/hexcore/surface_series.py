"""CLI and configuration only. Acquisition, decode, remap and I/O run in Rust."""
from pathlib import Path
import subprocess

from .engines import EngineSpec, resolve
from .init_sources import TABLE_PATH, VTABLE_ROOT, load_table, resolve_source

SERIES = EngineSpec(name="rw_sfc_series", flag="--engine", env_names=("GPUWM_HEX_RW_SFC_SERIES",),
                    gpuwm_env="GPUWM_RW_SFC_SERIES", subject="the surface time-series producer",
                    what_breaks="timed SST and sea-ice forcing cannot be produced", cargo_package="rw-sfc-series")


def run_surface_series(args):
    row = resolve_source(load_table(args.table), args.source)
    engine = resolve(SERIES, args.engine)
    command = [str(engine), "--table", str(args.table or TABLE_PATH), "--source", row["id"],
               "--cycle", args.cycle, "--hours", str(args.hours), "--static", str(args.static),
               "--out", str(args.out), "--receipt", str(args.receipt or args.out.with_suffix(".receipt.json")),
               "--vtable-root", str(VTABLE_ROOT)]
    for name in ("jobs", "threads", "interval_hours", "staged", "forecast_cycle"):
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-"), str(value)]
    return subprocess.run(command).returncode


def add_surface_series_parser(commands):
    parser = commands.add_parser("surface-series", help="produce timed SST and sea-ice forcing with the Rust case-8 remap")
    for name in ("source", "cycle"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--forecast-cycle", help="use one source forecast cycle for every dated record")
    parser.add_argument("--hours", type=int, required=True)
    for name in ("static", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("receipt", "engine", "table", "staged"):
        parser.add_argument("--" + name, type=Path)
    for name in ("jobs", "threads", "interval-hours"):
        parser.add_argument("--" + name, type=int)
    parser.set_defaults(handler=run_surface_series)
