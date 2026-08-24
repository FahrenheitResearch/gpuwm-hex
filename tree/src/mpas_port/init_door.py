"""The init front door: native-free construction or explicit capsule compatibility.

The Rust ``rw_mpas_init`` engine continues to own meteorological interpolation
and state construction.  This door owns fail-closed argument resolution,
construction of the vertical artifact when requested, compatibility-capsule
preflight, and provenance.

Normal native-free mode is:

``grid + static + WPS met + versioned vertical JSON -> constructed artifact -> rw_mpas_init``

The generated artifact is passed through the engine's existing capsule/reference
ABI as both arguments.  A native init file is opened only when the caller
explicitly selects compatibility mode with ``--capsule`` and ``--reference``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import MpasPortError

SURFACE_LEVEL_TAG = 200100.0
_PROFILE_FIELDS = {"TT", "RH", "SPECHUMD", "GHT", "PRES", "PRESSURE", "UU", "VV"}
_SOIL_PREFIXES = ("ST", "SM", "SOILT", "SOILM")

_STATIC_VARIABLES = (
    "latCell",
    "lonCell",
    "latEdge",
    "lonEdge",
    "angleEdge",
    "cellsOnEdge",
    "edgesOnCell",
    "nEdgesOnCell",
    "landmask",
    "ter",
    "soiltemp",
    "ivgtyp",
    "isltyp",
    "snoalb",
    "isice_lu",
    "greenfrac",
    "albedo12m",
    "dcEdge",
    "dvEdge",
    "areaCell",
    "cellsOnCell",
    "deriv_two",
)
_COMPATIBILITY_VARIABLES = (
    "ter",
    "zgrid",
    "zz",
    "fzm",
    "fzp",
    "dzu",
    "rdzw",
    "zb",
    "zb3",
)


class InitDoorRefusal(MpasPortError):
    """A named refusal: wrong result prevented, then remedy."""


@dataclass(frozen=True)
class _Switch:
    flag: str
    namelist_key: str
    breakage: str


_SWITCHES = (
    _Switch("--nfglevels", "config_nfglevels", "it caps the first-guess level table; a wrong count truncates or over-allocates every profile"),
    _Switch("--nfgsoillevels", "config_nfgsoillevels", "it declares the first-guess soil column; a wrong count drops or zero-fills layers"),
    _Switch("--extrap-airtemp", "config_extrap_airtemp", "it selects temperature extrapolation past the first-guess column"),
    _Switch("--use-spechumd", "config_use_spechumd", "it selects the moisture source and changes qv everywhere"),
    _Switch("--theta-adv-order", "config_theta_adv_order", "it selects the z-edge/zb3 branch and initial vertical mass-flux correction"),
    _Switch("--coef-3rd-order", "config_coef_3rd_order", "it scales the third-order rw correction"),
    _Switch("--virtual-factor", "(reference build property)", "it selects the virtual-temperature arithmetic reproduced by the engine"),
    _Switch("--deep-soil-moisture", "(reference build property)", "it selects the deepest-soil first-guess anchor"),
    _Switch("--landuse-table", "config_landuse_data", "the wrong table silently renumbers vegetation categories"),
    _Switch("--frac-seaice", "config_frac_seaice", "it changes fractional versus binary sea-ice handling"),
    _Switch("--tsk-seaice-threshold", "config_tsk_seaice_threshold", "it controls cold-water conversion to ice-covered land"),
    _Switch("--oned-underflow", "(reference compiler property)", "it selects preserve versus ifx FTZ behavior at interpolation guards"),
    _Switch("--start-time", "config_start_time", "it stamps the init and selects monthly statics interpolation"),
)


def _refuse(message: str) -> None:
    raise InitDoorRefusal(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _described(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().absolute()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


# ---------------------------------------------------------------------------
# engine and met resolution
# ---------------------------------------------------------------------------
def resolve_engine(explicit: Path | None) -> Path:
    """The ``rw_mpas_init`` binary, or a refusal naming what supplies it.

    Delegates to :mod:`mpas_port.engines`, which is the ONE ladder every
    door in this distribution resolves through.  This door used to read
    ``$RW_MPAS_INIT`` and nothing else, which meant a user who had run
    ``gpuwm fetch-bridges`` -- the single command that stages this exact
    binary onto the machine -- still met a refusal telling them to build
    it with cargo.  The shared ladder reads that staging directory, and
    the refusal now names the command that fills it.

    ``$RW_MPAS_INIT`` stays in the ladder permanently as a legacy
    spelling: an install line that already works must not be invalidated
    by a rename.
    """

    from .engines import INIT, EngineRefusal, resolve

    try:
        return resolve(INIT, explicit)
    except EngineRefusal as error:
        _refuse(str(error))
        raise  # pragma: no cover - _refuse always raises


def _probe_wps_header(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return False
    if len(head) < 8:
        return False
    for endian in (">", "<"):
        marker, version = struct.unpack(f"{endian}ii", head)
        if marker == 4 and version in (3, 4, 5):
            return True
    return False


def resolve_met_source(met: Path) -> Path:
    if not met.exists():
        _refuse(
            f"met source {met} does not exist; point --met at ungrib output (one WPS "
            "intermediate file) or a directory containing exactly one valid-time file"
        )
    if met.is_file():
        return met
    candidates = sorted(
        child for child in met.iterdir() if child.is_file() and _probe_wps_header(child)
    )
    if not candidates:
        _refuse(
            f"no WPS intermediate file found in {met}; nothing there has a valid "
            "Fortran-framed WPS v3/v4/v5 header"
        )
    if len(candidates) > 1:
        _refuse(
            f"{met} holds {len(candidates)} candidate WPS files ({', '.join(p.name for p in candidates)}); "
            "the engine requires exactly one valid time, so pass the intended file explicitly"
        )
    return candidates[0]


@dataclass(frozen=True)
class MetInventory:
    fields: tuple[str, ...]
    valid_times: tuple[str, ...]
    upper_air_levels: int
    surface_level_present: bool


def scan_met_file(met: Path, *, nfglevels: int, start_time: str) -> MetInventory:
    from .wps_intermediate import WpsIntermediateReader

    names: set[str] = set()
    valid_times: set[str] = set()
    profile_levels: set[float] = set()
    try:
        with WpsIntermediateReader(met) as reader:
            for field in reader.iter_fields(load_values=False):
                names.add(field.field)
                valid_times.add(field.valid_time)
                if field.field in _PROFILE_FIELDS:
                    profile_levels.add(field.level)
    except MpasPortError as error:
        _refuse(
            f"{met} is not readable as a WPS intermediate file: {error}. Re-run ungrib "
            "or select its actual output; a truncated source cannot produce a trustworthy init"
        )
    if "LANDSEA" not in names:
        _refuse(
            f"the intermediate file {met} carries no LANDSEA field; coastal soil and sea-ice "
            "interpolation would use the wrong surface type. Use a Vtable that emits LANDSEA"
        )
    if SURFACE_LEVEL_TAG not in profile_levels:
        _refuse(
            f"no profile level in {met} is tagged {SURFACE_LEVEL_TAG:.1f}; the case-7 "
            "surface column would be absent and t2m/rh2 would remain structurally wrong"
        )
    if len(profile_levels) > nfglevels:
        _refuse(
            f"the intermediate file has {len(profile_levels)} distinct first-guess levels but "
            f"--nfglevels declares {nfglevels}; declare at least {len(profile_levels)}"
        )
    if not any(name.startswith(_SOIL_PREFIXES) for name in names):
        _refuse(
            f"the intermediate file {met} carries no ST*/SM*/SOILT*/SOILM* soil layers; "
            "tslb/smois cannot be initialized. Use a source/Vtable with the declared soil column"
        )
    stamp = start_time[:13]
    stamps = sorted(value[:13] for value in valid_times)
    if stamps and stamp not in stamps:
        _refuse(
            f"--start-time {start_time} is not among met valid times {', '.join(stamps)}; "
            "using it would mislabel the meteorology"
        )
    return MetInventory(
        fields=tuple(sorted(names)),
        valid_times=tuple(sorted(valid_times)),
        upper_air_levels=len(profile_levels),
        surface_level_present=True,
    )


# ---------------------------------------------------------------------------
# mesh and vertical-source contracts
# ---------------------------------------------------------------------------
def check_static_and_grid(static: Path, grid: Path | None) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover
        _refuse(
            f"python package netCDF4 is unavailable ({error}); the door will not launch "
            "without preflighting grid/static dimensions and variables"
        )
    if not static.is_file():
        _refuse(
            f"mesh static file {static} does not exist; it supplies the physical mesh and statics"
        )
    with Dataset(static) as dataset:
        missing_dims = [
            name
            for name in ("nCells", "nEdges", "maxEdges", "nMonths")
            if name not in dataset.dimensions
        ]
        missing_vars = [name for name in _STATIC_VARIABLES if name not in dataset.variables]
        if missing_dims or missing_vars:
            _refuse(
                f"static file {static} is missing {', '.join(missing_dims + missing_vars)}; "
                "vertical construction/interpolation would have no complete topology or statics. "
                "Use a complete generated static file"
            )
        summary = {
            "nCells": int(len(dataset.dimensions["nCells"])),
            "nEdges": int(len(dataset.dimensions["nEdges"])),
            "static_dcEdge_shape": list(dataset.variables["dcEdge"].shape),
        }
    if grid is not None:
        if not grid.is_file():
            _refuse(
                f"mesh grid file {grid} does not exist; native-free vertical construction requires "
                "the grid/static pair minted together"
            )
        with Dataset(grid) as dataset:
            for name in ("nCells", "nEdges"):
                if name not in dataset.dimensions:
                    _refuse(f"--grid {grid} has no {name} dimension; it is not an MPAS mesh")
            grid_cells = int(len(dataset.dimensions["nCells"]))
            grid_edges = int(len(dataset.dimensions["nEdges"]))
        if grid_cells != summary["nCells"] or grid_edges != summary["nEdges"]:
            _refuse(
                f"grid/static pairing mismatch: grid has nCells={grid_cells}, nEdges={grid_edges}; "
                f"static has nCells={summary['nCells']}, nEdges={summary['nEdges']}. "
                "Pass the pair generated together"
            )
        summary["grid_nCells"] = grid_cells
        summary["grid_nEdges"] = grid_edges
    return summary


def check_mesh_and_capsule(
    static: Path,
    capsule: Path,
    reference: Path,
    grid: Path | None,
) -> dict[str, Any]:
    """Compatibility-mode preflight; this function retains its old public name."""

    try:
        from netCDF4 import Dataset
    except ImportError as error:  # pragma: no cover
        _refuse(f"python package netCDF4 is unavailable ({error}); install it")
    summary = check_static_and_grid(static, grid)
    for label, path in (("capsule", capsule), ("reference", reference)):
        if not path.is_file():
            _refuse(
                f"compatibility {label} {path} does not exist; provide the native file or use "
                "--vertical-spec for the native-free path"
            )
    with Dataset(capsule) as dataset:
        missing = [
            name for name in ("nCells", "nVertLevels") if name not in dataset.dimensions
        ] + [name for name in _COMPATIBILITY_VARIABLES if name not in dataset.variables]
        if missing:
            _refuse(
                f"compatibility capsule {capsule} is missing {', '.join(missing)}; its vertical contract is incomplete"
            )
        capsule_cells = int(len(dataset.dimensions["nCells"]))
        if capsule_cells != summary["nCells"]:
            _refuse(
                f"capsule nCells={capsule_cells} differs from static nCells={summary['nCells']}; "
                "vertical arrays would be attached to the wrong columns"
            )
        capsule_shape = tuple(int(value) for value in dataset.variables["zgrid"].shape)
        summary["nVertLevels"] = int(len(dataset.dimensions["nVertLevels"]))
    with Dataset(reference) as dataset:
        if "zgrid" not in dataset.variables:
            _refuse(
                f"compatibility reference {reference} carries no zgrid; the capsule identity oracle cannot run"
            )
        reference_shape = tuple(int(value) for value in dataset.variables["zgrid"].shape)
    if capsule_shape != reference_shape:
        _refuse(
            f"compatibility capsule zgrid{capsule_shape} differs from reference zgrid{reference_shape}; "
            "pass the native reference that minted the capsule"
        )
    summary["zgrid_shape"] = list(capsule_shape)
    return summary


@dataclass(frozen=True)
class VerticalSource:
    mode: str
    capsule: Path
    reference: Path
    summary: dict[str, Any]
    vertical_spec: Path | None = None
    vertical_artifact_receipt: Path | None = None


def select_vertical_mode(arguments: argparse.Namespace) -> str:
    constructed_requested = (
        getattr(arguments, "vertical_spec", None) is not None
        or getattr(arguments, "vertical_artifact", None) is not None
    )
    compatibility_requested = (
        getattr(arguments, "capsule", None) is not None
        or getattr(arguments, "reference", None) is not None
    )
    if constructed_requested and compatibility_requested:
        _refuse(
            "constructed and compatibility vertical sources were both declared. Mixing them can "
            "hide which bytes produced zgrid/zb. Use --vertical-spec (optionally --vertical-artifact) "
            "or use --capsule plus --reference, never both"
        )
    if constructed_requested:
        if getattr(arguments, "vertical_spec", None) is None:
            _refuse(
                "--vertical-artifact was given without --vertical-spec; there is no declaration to construct from"
            )
        if getattr(arguments, "grid", None) is None:
            _refuse(
                "native-free mode requires --grid as well as --static; topology and deriv_two must come "
                "from the grid/static pair generated together"
            )
        return "constructed"
    if compatibility_requested:
        if getattr(arguments, "capsule", None) is None or getattr(arguments, "reference", None) is None:
            _refuse(
                "compatibility mode requires both --capsule and --reference; a capsule asserted only "
                "against itself or without its oracle is not admitted"
            )
        return "compatibility-capsule"
    _refuse(
        "no vertical source was declared. Normal native-free use requires --vertical-spec and --grid; "
        "legacy compatibility requires both --capsule and --reference"
    )
    raise AssertionError("unreachable")


def prepare_vertical_source(arguments: argparse.Namespace) -> VerticalSource:
    mode = select_vertical_mode(arguments)
    if mode == "compatibility-capsule":
        summary = check_mesh_and_capsule(
            arguments.static, arguments.capsule, arguments.reference, arguments.grid
        )
        return VerticalSource(
            mode=mode,
            capsule=arguments.capsule,
            reference=arguments.reference,
            summary=summary,
        )

    summary = check_static_and_grid(arguments.static, arguments.grid)
    from .vertical_spec import materialize_vertical_artifact

    out: Path = arguments.out
    artifact = (
        arguments.vertical_artifact
        if arguments.vertical_artifact is not None
        else out.with_name(out.name + ".vertical.nc")
    )
    artifact = artifact.expanduser().absolute()
    receipt = artifact.with_name(artifact.name + ".receipt.json")
    payload = materialize_vertical_artifact(
        grid=arguments.grid,
        static=arguments.static,
        spec_path=arguments.vertical_spec,
        output=artifact,
        receipt_path=receipt,
        # The emitter copies capsule global attributes verbatim and the
        # forecast driver parses them from the init, so the artifact must
        # carry THIS run's declarations, not the static template's
        # placeholders.
        run_config={
            "config_start_time": arguments.start_time,
            "config_nfglevels": int(arguments.nfglevels),
            "config_nfgsoillevels": int(arguments.nfgsoillevels),
            "config_nsoillevels": 4,
            "config_use_spechumd": arguments.use_spechumd == "yes",
            "config_frac_seaice": arguments.frac_seaice == "yes",
            "config_extrap_airtemp": arguments.extrap_airtemp,
            "config_tsk_seaice_threshold": float(
                arguments.tsk_seaice_threshold),
            "config_landuse_data": arguments.landuse_table,
        },
    )
    summary = {
        **summary,
        "vertical_artifact": payload["output"],
        "vertical_spec_canonical_sha256": payload["inputs"]["vertical_spec"]["canonical_sha256"],
        "vertical_invariants": payload["invariants"],
    }
    # The current Rust ABI performs a capsule/reference zgrid identity check.
    # Passing the one constructed artifact as both inputs keeps that check while
    # removing any native-file dependency.  The receipt names this explicitly.
    return VerticalSource(
        mode=mode,
        capsule=artifact,
        reference=artifact,
        summary=summary,
        vertical_spec=arguments.vertical_spec,
        vertical_artifact_receipt=receipt,
    )


# ---------------------------------------------------------------------------
# CLI and execution
# ---------------------------------------------------------------------------
def add_init_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "init",
        help="build MPAS initial conditions with native-free vertical construction or explicit capsule compatibility",
        description=(
            "Drive rw_mpas_init. Normal mode constructs the vertical contract from --grid, "
            "--static, and --vertical-spec; compatibility mode is selected only by passing "
            "both --capsule and --reference. Every numerical switch remains explicit."
        ),
    )
    parser.add_argument("--met", type=Path, help="WPS intermediate file, or a directory holding exactly one")
    parser.add_argument("--static", type=Path, help="generated/published static file")
    parser.add_argument("--grid", type=Path, default=None, help="grid file; required in native-free mode")
    parser.add_argument("--vertical-spec", type=Path, default=None, help="versioned JSON declaration for native-free construction")
    parser.add_argument("--vertical-artifact", type=Path, default=None, help="durable constructed artifact path (default: <out>.vertical.nc)")
    parser.add_argument("--capsule", type=Path, default=None, help="explicit compatibility-only native init-class capsule")
    parser.add_argument("--reference", type=Path, default=None, help="explicit compatibility-only native reference")
    parser.add_argument("--out", type=Path, help="init NetCDF to write")
    parser.add_argument("--start-time", help="YYYY-MM-DD_HH:MM:SS (config_start_time)")
    parser.add_argument("--nfglevels", type=int, help="config_nfglevels")
    parser.add_argument("--nfgsoillevels", type=int, help="config_nfgsoillevels")
    parser.add_argument("--extrap-airtemp", choices=("constant", "linear", "lapse-rate"), help="config_extrap_airtemp")
    parser.add_argument("--use-spechumd", choices=("yes", "no"), help="config_use_spechumd")
    parser.add_argument("--theta-adv-order", type=int, help="config_theta_adv_order; must agree with vertical spec")
    parser.add_argument("--coef-3rd-order", type=float, help="config_coef_3rd_order")
    parser.add_argument("--virtual-factor", choices=("reproduce-fortran", "consistent"))
    parser.add_argument("--deep-soil-moisture", choices=("reproduce-fortran", "corrected"))
    parser.add_argument("--landuse-table", help="config_landuse_data")
    parser.add_argument("--frac-seaice", choices=("yes", "no"), help="config_frac_seaice")
    parser.add_argument("--tsk-seaice-threshold", type=float, help="config_tsk_seaice_threshold, K")
    parser.add_argument("--oned-underflow", choices=("preserve", "reproduce-ifx-ftz"))
    parser.add_argument("--engine", type=Path, default=None, help="rw_mpas_init executable (default $RW_MPAS_INIT)")
    parser.add_argument("--receipt", type=Path, default=None, help="provenance receipt path")
    parser.set_defaults(handler=run_init_command)


def _require_switches(arguments: argparse.Namespace) -> None:
    for switch in _SWITCHES:
        attribute = switch.flag.lstrip("-").replace("-", "_")
        if getattr(arguments, attribute, None) is None:
            _refuse(
                f"{switch.flag} was not given and has no default: {switch.breakage}. "
                f"State it explicitly; native key is {switch.namelist_key}"
            )
    for flag, value in (
        ("--met", arguments.met),
        ("--static", arguments.static),
        ("--out", arguments.out),
    ):
        if value is None:
            _refuse(f"{flag} was not given; it cannot be inferred")
    select_vertical_mode(arguments)


def _validate_spec_switch_agreement(arguments: argparse.Namespace) -> None:
    if arguments.vertical_spec is None:
        return
    from .vertical_spec import VerticalSpec

    spec = VerticalSpec.from_file(arguments.vertical_spec)
    if int(arguments.theta_adv_order) != spec.theta_adv_order:
        _refuse(
            f"--theta-adv-order={arguments.theta_adv_order} disagrees with vertical spec "
            f"theta_adv_order={spec.theta_adv_order}; zb/zb3 and the engine would use different branches. "
            "Make the two declarations identical"
        )
    if not abs(float(arguments.coef_3rd_order) - spec.coef_3rd_order) <= 8.0 * max(
        1.0, abs(spec.coef_3rd_order)
    ) * sys.float_info.epsilon:
        _refuse(
            f"--coef-3rd-order={arguments.coef_3rd_order} disagrees with vertical spec "
            f"coef_3rd_order={spec.coef_3rd_order}; declare one value in both contracts"
        )


def run_init_command(arguments: argparse.Namespace) -> int:
    started = time.time()
    _require_switches(arguments)
    _validate_spec_switch_agreement(arguments)

    out: Path = arguments.out.expanduser().absolute()
    if not out.parent.is_dir():
        _refuse(
            f"output directory {out.parent} does not exist; create it before an expensive init run"
        )
    receipt_path = (
        arguments.receipt.expanduser().absolute()
        if arguments.receipt is not None
        else out.with_name(out.name + ".provenance.json")
    )
    if not receipt_path.parent.is_dir():
        _refuse(f"receipt directory {receipt_path.parent} does not exist; create it")

    met = resolve_met_source(arguments.met)
    inventory = scan_met_file(
        met, nfglevels=arguments.nfglevels, start_time=arguments.start_time
    )
    engine = resolve_engine(arguments.engine)
    vertical = prepare_vertical_source(arguments)

    argv = [
        str(engine),
        "--met", str(met),
        "--static", str(arguments.static),
        "--capsule", str(vertical.capsule),
        "--reference", str(vertical.reference),
        "--out", str(out),
        "--start-time", arguments.start_time,
        "--nfglevels", str(arguments.nfglevels),
        "--nfgsoillevels", str(arguments.nfgsoillevels),
        "--extrap-airtemp", arguments.extrap_airtemp,
        "--use-spechumd", arguments.use_spechumd,
        "--theta-adv-order", str(arguments.theta_adv_order),
        "--coef-3rd-order", repr(arguments.coef_3rd_order),
        "--virtual-factor", arguments.virtual_factor,
        "--deep-soil-moisture", arguments.deep_soil_moisture,
        "--landuse-table", arguments.landuse_table,
        "--frac-seaice", arguments.frac_seaice,
        "--tsk-seaice-threshold", repr(arguments.tsk_seaice_threshold),
        "--oned-underflow", arguments.oned_underflow,
    ]
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        sys.stderr.write(
            "gpuwm-hex init: rw_mpas_init refused or failed "
            f"(rc {completed.returncode}). No init/provenance receipt was written. "
            + (
                f"The constructed vertical artifact remains at {vertical.capsule} with its own receipt for diagnosis.\n"
                if vertical.mode == "constructed"
                else "\n"
            )
        )
        return int(completed.returncode)
    try:
        engine_receipt = json.loads(completed.stdout)
    except json.JSONDecodeError:
        engine_receipt = {"unparsed_stdout": completed.stdout}
    if not out.is_file():
        _refuse(
            f"the engine reported success but {out} does not exist; rebuild the current rw_mpas_init "
            "or inspect an ABI mismatch between door and engine"
        )

    from . import __version__

    inputs: dict[str, Any] = {
        "met": _described(met),
        "static": _described(arguments.static),
        "grid": _described(arguments.grid) if arguments.grid else None,
    }
    if vertical.mode == "constructed":
        inputs["vertical_spec"] = _described(vertical.vertical_spec)  # type: ignore[arg-type]
        inputs["vertical_artifact"] = _described(vertical.capsule)
        inputs["vertical_artifact_receipt"] = _described(vertical.vertical_artifact_receipt)  # type: ignore[arg-type]
        inputs["capsule"] = None
        inputs["reference"] = None
    else:
        inputs["vertical_spec"] = None
        inputs["vertical_artifact"] = None
        inputs["vertical_artifact_receipt"] = None
        inputs["capsule"] = _described(vertical.capsule)
        inputs["reference"] = _described(vertical.reference)

    receipt = {
        "schema": "gpuwm-hex.init-provenance/v2",
        "tool": "gpuwm-hex init",
        "door_version": __version__,
        "created_utc": _utc_now(),
        "host": platform.node(),
        "engine": _described(engine),
        "vertical_source": {
            "mode": vertical.mode,
            "native_runtime_dependency": vertical.mode != "constructed",
            "engine_abi_mapping": {
                "capsule_argument": str(vertical.capsule),
                "reference_argument": str(vertical.reference),
                "constructed_artifact_used_for_both": vertical.mode == "constructed",
            },
            "summary": vertical.summary,
        },
        "inputs": inputs,
        "met_inventory": {
            "fields": list(inventory.fields),
            "valid_times": list(inventory.valid_times),
            "distinct_upper_air_levels": inventory.upper_air_levels,
        },
        "engine_argv": argv,
        "engine_receipt": engine_receipt,
        "output": _described(out),
        "door_seconds": round(time.time() - started, 3),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "init": str(out),
                "provenance": str(receipt_path),
                "vertical_mode": vertical.mode,
                "vertical_artifact": (
                    str(vertical.capsule) if vertical.mode == "constructed" else None
                ),
                "n_cells": engine_receipt.get("n_cells"),
                "n_vert_levels": engine_receipt.get("n_vert_levels"),
                "met_records_used": engine_receipt.get("met_records_used"),
                "engine_seconds": engine_receipt.get("seconds"),
            },
            indent=2,
        )
    )
    return 0
