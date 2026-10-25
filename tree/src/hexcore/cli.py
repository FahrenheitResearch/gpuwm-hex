"""The ``gpuwm-hex`` console script: every front door this package ships."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .errors import MpasPortError
# ``.mesh`` and ``.oracle`` are imported inside the handlers that use them:
# they pull numpy/netCDF4, and the ``render`` door must run on a render node
# whose Python has neither -- its data path is entirely the two Rust
# binaries it drives.  ``.init_door`` and ``.forecast_door`` are stdlib-only
# at import; their netCDF4/cupy/driver needs are named runtime refusals.
from .cull_door import add_cull_parser
from .init_door import add_init_parser, add_surface_update_parser
# The checkout probe lives with the forecast door because that door is the
# one that cannot open without a checkout at all.  Stated once: two copies
# of "is this a checkout" would be two things to keep true, and the answer
# decides both this module's mesh defaults and that door's existence.
from .forecast_door import PROJECT_ROOT, add_forecast_parser
DEFAULT_GRID = (
    PROJECT_ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
    if PROJECT_ROOT is not None
    else None
)
DEFAULT_STATIC = (
    PROJECT_ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"
    if PROJECT_ROOT is not None
    else None
)
DEFAULT_ORACLE = (
    PROJECT_ROOT / "oracle" / "x1.2562" if PROJECT_ROOT is not None else None
)


def _require_mesh_paths(arguments: argparse.Namespace) -> None:
    for flag, value in (("--grid", arguments.grid), ("--static", arguments.static)):
        if value is None:
            raise MpasPortError(
                f"{flag} was not given and there is no default outside a source "
                "checkout.  Mesh grid and static files are external assets: "
                "gpuwm-hex ships neither and has no general fetch path for them "
                "(verify-dycore downloads only its fixed small test mesh), so a "
                "guessed default would name a file that does not exist.  Obtain "
                "the pair from the MPAS-Atmosphere mesh downloads and pass both "
                "--grid and --static explicitly (see README, 'Assets you must "
                "supply')."
            )


def _mesh(arguments: argparse.Namespace):
    from .mesh import Mesh

    _require_mesh_paths(arguments)
    return Mesh.from_netcdf(arguments.grid, arguments.static)


def _version(arguments: argparse.Namespace) -> int:
    from . import DISTRIBUTION_NAME, __version__

    print(
        json.dumps(
            {
                "distribution": DISTRIBUTION_NAME,
                "version": __version__,
                "package": str(Path(__file__).resolve().parent),
                "source_checkout": (
                    str(PROJECT_ROOT) if PROJECT_ROOT is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _mesh_check(arguments: argparse.Namespace) -> int:
    from .dual_edge_admission import DualEdgeAdmissionError, admit_dual_edges
    from .oracle import sha256_file

    if getattr(arguments, "grid_only", False):
        # A culled regional grid exists before its static does (the culler
        # emits the grid first), and refusing to examine it until a static
        # exists would leave the one file the whole regional chain builds on
        # unvalidatable at the door users are told to run first.
        if arguments.grid is None:
            raise MpasPortError(
                "--grid was not given.  --grid-only validates one grid file "
                "and there is no default outside a source checkout; pass "
                "--grid explicitly."
            )
        from .mesh import Mesh

        mesh = Mesh.from_netcdf(arguments.grid)
        static_path = None
        static_sha = None
    else:
        mesh = _mesh(arguments)
        static_path = str(Path(arguments.static).resolve())
        static_sha = sha256_file(arguments.static)
    receipt = {
        "passed": True,
        "grid": str(Path(arguments.grid).resolve()),
        "grid_sha256": sha256_file(arguments.grid),
        "static": static_path,
        "static_sha256": static_sha,
        "dimensions": {
            name: int(mesh.dimensions[name])
            for name in ("nCells", "nEdges", "nVertices", "maxEdges")
        },
        "connectivity_indexing": mesh.provenance["connectivity_indexing"],
    }
    if mesh.is_regional:
        import numpy as np

        from .mesh import (
            REGIONAL_BOUNDARY_MASK_NAMES,
            regional_boundary_mask_digest,
        )

        mask_cell = np.asarray(mesh.arrays["bdyMaskCell"])
        receipt["regional"] = {
            "boundary_zone_width": int(
                max(
                    int(np.asarray(mesh.arrays[name]).max())
                    for name in REGIONAL_BOUNDARY_MASK_NAMES
                )
            ),
            "euler_characteristic": int(
                mesh.dimensions["nCells"]
                - mesh.dimensions["nEdges"]
                + mesh.dimensions["nVertices"]
            ),
            "ring_cell_counts": [
                int(np.count_nonzero(mask_cell == ring)) for ring in range(8)
            ],
            "bdy_mask_sha256": regional_boundary_mask_digest(mesh.arrays),
        }
    # THE BREAKAGE THIS PREVENTS: a pair that passes mesh-check and then dies
    # inside step 0 of every forecast.  The registered v15.150.38857 did
    # exactly that -- this receipt said "passed": true while its worst Voronoi
    # edge was 6.5 m -- and mesh-check is the door a user is told to run
    # BEFORE anything expensive touches the mesh.  The forecast bind applies
    # the same floor; agreeing here is what makes the earlier door worth
    # running.
    try:
        admission = admit_dual_edges(
            mesh.dvEdge,
            mesh.dcEdge,
            cells_on_edge=mesh.cellsOnEdge,
            cells_on_edge_base=0,
            mesh_name=str(
                Path(arguments.grid).name
                if static_path is None
                else Path(arguments.static).name
            ),
        )
    except DualEdgeAdmissionError as error:
        receipt["passed"] = False
        receipt["dual_edge_refusal"] = str(error)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 1
    receipt["dual_edge_admission"] = admission.as_dict()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _oracle_gate(arguments: argparse.Namespace) -> int:
    from .oracle import run_m1_oracle

    if arguments.fixtures is None:
        raise MpasPortError(
            "--fixtures was not given and there is no default outside a source "
            "checkout.  The M1 oracle replays source-extracted Fortran fixtures "
            "that live in the port's own checkout, not in the installed wheel; "
            "without them there is nothing to replay.  Pass --fixtures pointing "
            "at the checkout's oracle directory."
        )
    report = run_m1_oracle(_mesh(arguments), arguments.fixtures)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed else 1


def _render(arguments: argparse.Namespace) -> int:
    from .render_door import run_render

    return run_render(arguments)


def _add_mesh_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuwm-hex",
        description=(
            "gpuwm-hex: front doors for the GPU-native global variable-resolution "
            "model core (a port of MPAS-Atmosphere v8.4.2)"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    from .country_door import add_country_parser
    add_country_parser(commands)

    version = commands.add_parser(
        "version", help="report the installed distribution and version"
    )
    version.set_defaults(handler=_version)

    from .verify_dycore_door import add_verify_dycore_parser
    add_verify_dycore_parser(commands)

    from .verify_mpas import add_verify_mpas_parser
    add_verify_mpas_parser(commands)
    from .verify_mpas import run_cached_dycore
    cached = commands.choices["verify-dycore"]
    cached.set_defaults(legacy_dycore_handler=cached.get_default("handler"), handler=run_cached_dycore)
    cached.add_argument("--cached-deck", type=Path, help="verify a pinned cached dycore forecast without rebuilding native")
    cached.add_argument("--receipt", type=Path)
    cached.add_argument("--ratchet", type=Path)
    cached.add_argument("--comparator", type=Path)

    # First after `version`, because it is the command a fresh install
    # should reach for.  This distribution cannot carry the Rust engines,
    # the CUDA runtime or the mesh assets, so a bare `pip install` lands
    # something real that is not yet able to run: doctor is where that
    # gap gets NAMED, with the command that closes each half, instead of
    # a user meeting a refusal at the end of a long argument vector.
    from .doctor import add_doctor_parser

    add_doctor_parser(commands)

    # Before `mesh-check`, because that is the order a mesh happens in: a
    # spec is asked whether it can be built, and only then is a built mesh
    # validated.  The gap this closes is measured -- `rw_mpas_mesh --dry-run`
    # sizes a spec its own build then refuses, once on pre-run arithmetic and
    # once after 711 s of relaxation.
    from .mesh_plan_door import add_mesh_plan_parser

    add_mesh_plan_parser(commands)

    mesh_check = commands.add_parser("mesh-check", help="validate an MPAS mesh")
    _add_mesh_paths(mesh_check)
    mesh_check.add_argument(
        "--grid-only",
        action="store_true",
        help=(
            "validate the grid file alone (a culled regional grid exists "
            "before its static does); any --static is not read in this mode "
            "and the receipt records static: null"
        ),
    )
    mesh_check.set_defaults(handler=_mesh_check)

    oracle_gate = commands.add_parser(
        "oracle-gate", help="replay the source-extracted Fortran M1 fixtures"
    )
    _add_mesh_paths(oracle_gate)
    oracle_gate.add_argument("--fixtures", type=Path, default=DEFAULT_ORACLE)
    oracle_gate.set_defaults(handler=_oracle_gate)

    add_cull_parser(commands)

    # Between cull and init, because that is where a regional case meets it:
    # a cull needs a regional meteorological source the init and boundary
    # engines can read, and a projected product (HRRR) is not one until it
    # is resampled onto a regular lat-lon intermediate.  Deferred import:
    # the door pulls numpy and the engine's ingest.
    from .hrrr_intermediate import add_intermediate_parser, add_lbc_parser

    add_intermediate_parser(commands)

    add_init_parser(commands)
    add_surface_update_parser(commands)

    # Beside init, because it ends in init: one table-driven path from any
    # declared source (GFS, IFS, AIFS, ENS/GEFS members, ERA5, ...) to the
    # same init door.  A new source model is a table row, not a code path.
    from .init_sources import add_init_source_parser

    add_init_source_parser(commands)
    from .surface_series import add_surface_series_parser
    add_surface_series_parser(commands)

    # After init, because it consumes the culled init: one boundary file per
    # intermediate valid time, through rw_mpas_lbc on its wps-intermediate
    # route.
    add_lbc_parser(commands)

    # Between init and render, because that is the order a user meets them:
    # an init is what a forecast starts from and history is what a render
    # consumes.  The door needs a source checkout and says so by name; it is
    # listed here unconditionally so that `gpuwm-hex --help` on a wheel
    # names the command and its one requirement, rather than hiding a
    # capability the distribution has.
    add_forecast_parser(commands)

    # Right after forecast, because a moving run is a chain of forecasts: the
    # tracker on the model's own frames, member preparation ahead of each move
    # and the balanced remap between members.  Stdlib-only at import.
    from .moving_door import add_moving_parser

    add_moving_parser(commands)

    from .observation_scorecard import add_score_parser

    add_score_parser(commands)

    from .ensemble import add_ensemble_parser
    from .ensemble_prepare import add_ensemble_prepare_parser
    add_ensemble_parser(commands)
    add_ensemble_prepare_parser(commands)

    # Immediately after forecast, because it IS the forecast door: `pair`
    # takes every forecast flag from `add_forecast_arguments`, runs that one
    # authored request twice -- once with a point-source table and once
    # without -- and differences the two.  The import is deferred for the
    # same reason `forecast` is not: this module is stdlib-only, but keeping
    # the two together here is what makes the adjacency visible in --help.
    from .pair_door import add_pair_parser

    add_pair_parser(commands)

    # After forecast and before render, because that is the order a cascade
    # meets them: a coarse forecast is what the placement reads, and the
    # swaths it places are what the next forecast and the render draw.  The
    # import is deferred: this door pulls numpy, netCDF4 and scipy, and the
    # render door must stay usable on a node whose Python has none of them.
    from .swath.door import add_swath_parser

    add_swath_parser(commands)

    from .cycle.door import add_cycle_parser

    add_cycle_parser(commands)

    # The named physics suites, resolved on the host: stdlib + json only.
    from .physics_door import add_physics_parser

    add_physics_parser(commands)

    # Right after `physics`: the MPAS suites it lists read tables and device
    # maths that this command builds from the install (stdlib at import).
    from .mpasphys.resources_build import add_physics_resources_parser

    add_physics_resources_parser(commands)
    from .assets import add_fetch_assets_parser
    add_fetch_assets_parser(commands)

    render = commands.add_parser(
        "render",
        help="MPAS history -> product PNGs through the Rust path "
             "(rw_mpas_convert + rw_wrfbatch)",
    )
    from .render_door import add_render_arguments

    add_render_arguments(render)
    render.set_defaults(handler=_render)

    # After render, because it drives render: the standard product set (maps
    # on every view a mesh has, soundings, meteograms) in one command, and
    # the forecast door's own last stage by default.
    from .products_door import add_products_parser

    add_products_parser(commands)

    # After products, because it is products without the drawing: the same
    # Rust converter, frames filed as NetCDF for any reader.
    from .export_door import add_export_parser

    add_export_parser(commands)

    # Beside export: it composes export --follow with the engine's local
    # posting, as-posted prep and sim, for a nest added while hex runs.
    from .add_nest_door import add_add_nest_parser

    add_add_nest_parser(commands)

    # Last, because it composes every door above from a data row: mesh,
    # static, init-source, forecast, products, export and score in one
    # command, with its plan, status.json and events.jsonl.
    from .global_door import add_global_parser, add_run_plan_parser, add_sources_parser

    add_global_parser(commands)
    from .s2s_door import add_s2s_parser

    add_s2s_parser(commands)
    add_run_plan_parser(commands)
    add_sources_parser(commands)
    from .chem_emissions import add_parser as add_chem_sources_parser

    add_chem_sources_parser(commands)
    from .global_cycle import add_global_cycle_parser

    add_global_cycle_parser(commands)
    # The limited-area counterpart of `global`: mesh-plan --point --generate,
    # intermediate, init, lbc, the contract deck and forecast --lbc-dir from
    # one limited-area row and a point.
    from .regional_door import add_regional_parser

    add_regional_parser(commands)
    _add_numerics_option(commands)
    return parser


NUMERICS_HELP = (
    "kernel numerics: 'fast' (the default) compiles with FMA contraction and flush-to-zero (IEEE division and "
    "square root kept; land surface keeps its subnormals); 'strict' compiles every kernel the way the bitwise "
    "proofs measured it and is the opt-in for byte-identity to native MPAS-A. Same key as the namelist's "
    "config_numerics and the GPUWM_HEX_NUMERICS environment variable")


def _add_numerics_option(commands: argparse._SubParsersAction) -> None:
    """``--numerics {fast,strict}`` on every door (owner ruling 2026-10-16: strict is opt-in)."""
    for sub in commands.choices.values():
        if any("--numerics" in action.option_strings for action in sub._actions):
            continue
        sub.add_argument("--numerics", choices=("fast", "strict"), default=None, help=NUMERICS_HELP)


def _select_numerics(arguments: argparse.Namespace) -> None:
    """Fix the process's numerics before any kernel compiles; ranks and subprocesses inherit it."""
    from . import numerics
    text = None
    path = getattr(arguments, "physics_namelist", None)
    if path is not None:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = None
    chosen, source = numerics.resolve(getattr(arguments, "numerics", None), text)
    numerics.set_mode(chosen, source=source)
    arguments.numerics = chosen
    arguments.numerics_source = source


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        _select_numerics(arguments)
        if arguments.command in {"init", "lbc"}:
            from .time_ledger import stage

            with stage(f"prep.{arguments.command}"):
                return int(arguments.handler(arguments))
        return int(arguments.handler(arguments))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"gpuwm-hex: {error}", file=sys.stderr)
        return 2


#: Door commands that hold the device.  Composing doors (global, regional, country) run these in a child
#: process so the CUDA context and every pooled byte end with the stage.  THE BREAKAGE THIS PREVENTS
#: (rolling 2.9.4 run, hex global-60km 24 h, 2026-10-21): the in-process forecast kept 11.6 GB of the card
#: through products, export and observation fetches for scoring, more than 10 min at 0 % GPU, so no other
#: job could admit on the card.  The stages after it are host work (Rust renderer, NetCDF export, scorer).
DEVICE_COMMANDS = frozenset({"forecast"})


def run_stage(argv: Sequence[str], env: dict[str, str] | None = None) -> int:
    """Run one door command for a composing door: device commands in a child process, the rest in-process."""

    argv = [str(item) for item in argv]
    if argv and argv[0] in DEVICE_COMMANDS:
        import os
        import subprocess
        child = {**os.environ, **(env or {})}
        return subprocess.run([sys.executable, "-m", "hexcore.cli", *argv], env=child).returncode
    return int(main(argv))


def run_exit_handlers() -> None:
    """What the interpreter runs at a normal exit, in its order, without the module teardown after it.

    threading's exit hooks first (``threading._shutdown``: concurrent.futures stops its process pools and
    the kernel plans' compile pools drop their workers), then the ``atexit`` functions. THE BREAKAGE THIS
    PREVENTS (the hl-3 headline run, 2026-10-24): running ``atexit`` alone skipped the threading hooks, so
    multiprocessing's exit function joined an idle spawn compile worker that nothing had told to stop, and a
    finished forecast (receipt and products written) never exited.
    """
    import atexit
    import threading
    try:
        shutdown = getattr(threading, "_shutdown", None)
        if shutdown is not None:
            shutdown()
    finally:
        atexit._run_exitfuncs()


def device_process_exit(code: int) -> None:
    """End a device-command process without the interpreter's module teardown.

    THE BREAKAGE THIS PREVENTS (HEX-PHYSICS-PARITY run group 2 #6): a moving segment on the MPAS suite route
    finished, wrote its receipt ("completed": true) and its carrier, then died with SIGSEGV (rc -11) while the
    interpreter tore modules down: cupy's ``Event.__del__`` ran after the CUDA context was gone
    ("cudaErrorDeviceUninitialized: invalid device context").  The controller counted the finished segment as
    failed and launched a recovery.  Here the exit handlers run exactly as at a normal exit (kernel-cache and
    prewarm manifests, history writers), the streams are flushed, and the process ends with the door's code
    before any late finalizer can touch the device (the forked member slot in hexcore.forecast_batch does the
    same).
    """
    import atexit
    import os
    try:
        run_exit_handlers()
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(code))


def entry() -> int:
    """The console door: :func:`main`, with a device command's process ended by :func:`device_process_exit`."""
    argv = sys.argv[1:]
    code = main(argv)
    if argv and argv[0] in DEVICE_COMMANDS:
        device_process_exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(entry())
