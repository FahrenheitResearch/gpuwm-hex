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
from .init_door import add_init_parser
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
                "gpuwm-hex ships neither and has no fetch path for them, so a "
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

    mesh = _mesh(arguments)
    receipt = {
        "passed": True,
        "grid": str(Path(arguments.grid).resolve()),
        "grid_sha256": sha256_file(arguments.grid),
        "static": str(Path(arguments.static).resolve()),
        "static_sha256": sha256_file(arguments.static),
        "dimensions": {
            name: int(mesh.dimensions[name])
            for name in ("nCells", "nEdges", "nVertices", "maxEdges")
        },
        "connectivity_indexing": mesh.provenance["connectivity_indexing"],
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
            mesh_name=str(Path(arguments.static).name),
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
            "model core (a port of MPAS-Atmosphere v8.4.1)"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    version = commands.add_parser(
        "version", help="report the installed distribution and version"
    )
    version.set_defaults(handler=_version)

    # First after `version`, because it is the command a fresh install
    # should reach for.  This distribution cannot carry the Rust engines,
    # the CUDA runtime or the mesh assets, so a bare `pip install` lands
    # something real that is not yet able to run: doctor is where that
    # gap gets NAMED, with the command that closes each half, instead of
    # a user meeting a refusal at the end of a long argument vector.
    from .doctor import add_doctor_parser

    add_doctor_parser(commands)

    mesh_check = commands.add_parser("mesh-check", help="validate an MPAS mesh")
    _add_mesh_paths(mesh_check)
    mesh_check.set_defaults(handler=_mesh_check)

    oracle_gate = commands.add_parser(
        "oracle-gate", help="replay the source-extracted Fortran M1 fixtures"
    )
    _add_mesh_paths(oracle_gate)
    oracle_gate.add_argument("--fixtures", type=Path, default=DEFAULT_ORACLE)
    oracle_gate.set_defaults(handler=_oracle_gate)

    add_init_parser(commands)

    # Between init and render, because that is the order a user meets them:
    # an init is what a forecast starts from and history is what a render
    # consumes.  The door needs a source checkout and says so by name; it is
    # listed here unconditionally so that `gpuwm-hex --help` on a wheel
    # names the command and its one requirement, rather than hiding a
    # capability the distribution has.
    add_forecast_parser(commands)

    render = commands.add_parser(
        "render",
        help="MPAS history -> product PNGs through the Rust path "
             "(rw_mpas_convert + rw_wrfbatch)",
    )
    from .render_door import add_render_arguments

    add_render_arguments(render)
    render.set_defaults(handler=_render)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (MpasPortError, OSError, ValueError) as error:
        print(f"gpuwm-hex: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
