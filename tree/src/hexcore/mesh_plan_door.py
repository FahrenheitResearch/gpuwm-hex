"""``gpuwm-hex mesh-plan`` -- ask a resolution spec every question before building.

WHY THIS DOOR EXISTS.  ``mesh-check`` validates a mesh that already exists.
Until this leg there was nothing to ask about a mesh that does NOT exist yet,
so the only way to find out whether a spec was buildable was to build it --
and ``rw_mpas_mesh --dry-run``, the thing that looks like the answer, sizes
without applying the build's gates.  Measured 2026-08-28 on this desktop:
one design sized clean and was refused 87 ms later by the build's pre-run
arithmetic; a second sized clean, cleared that gate, relaxed 217,621 cells
over **1,251 s** and was then refused on the 200 m shortest-dual-edge floor
at edge 562175 (36.8 m over an 886 m dcEdge), with no grid written.  The
report that opened this lane measured the same pair at 711 s on another box.

``--point LAT,LON`` (:mod:`hexcore.mesh_point`) writes the spec itself --
the registered 937.5 m ladder moved to the point -- prices its CULL against
the named card on the one admission surface, and with ``--generate``
builds, admits, registers (a runtime row beside the mesh) and culls it.

This leg costs about a fifth of a second and answers three things:

* what the spec costs -- cells, and the device footprint when a card is named;
* every gate the build applies that a SPEC decides, applied here, so a spec
  the build would refuse is refused here first, with the remedy in numbers;
* every gate the build applies that a spec does NOT decide, named, with the
  quantities that decide it and the earliest point at which it can be known.

The third list is not an apology.  A sizing path that silently omits a gate
is how a user spends twenty minutes to learn something; a sizing path that
says which gate it cannot evaluate, and why, spends nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .errors import MpasPortError


def _load(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MpasPortError(
            f"--spec {path} cannot be read: {error}. A resolution spec is the "
            "one input this leg has; there is no default that would name a "
            "file that exists"
        ) from error
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as error:
        raise MpasPortError(f"--spec {path} is not valid JSON: {error}") from error
    if not isinstance(spec, dict) or "background_km" not in spec:
        raise MpasPortError(
            f"--spec {path} carries no background_km, so it is not a "
            "resolution spec. The shape is "
            '{"background_km": KM, "regions": [{"shape": {...}, '
            '"spacing_km": KM, "transition_cells": N}]}'
        )
    return spec


def _band(edge: Mapping[str, Any]) -> str:
    pair = edge.get("predicted_shortest_dual_edge_m")
    if not pair:
        return "not predicted"
    low, high = pair
    return f"{low:.1f} to {high:.1f} m"


def _nearest(edge: Mapping[str, Any], count: int = 2) -> list[str]:
    """The measured rows closest in finest spacing, so a verdict has evidence."""

    finest = edge.get("finest_requested_spacing_km") or 0.0
    samples = edge.get("measured_samples") or []
    if finest <= 0.0 or not samples:
        return []
    ordered = sorted(
        samples, key=lambda row: abs(math.log(row["finest_km"] / finest))
    )
    return [
        f"      measured          {row['label']}: {row['finest_km']:g} km finest, "
        f"{row['min_dv_edge_m']:.1f} m shortest dual edge, "
        f"{'emitted' if row['emitted'] else 'REFUSED'}"
        for row in ordered[:count]
    ]


def _report(spec: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    gates = receipt.get("gates_applied_by_hexcore") or {}
    band = gates.get("transition_band") or {}
    edge = gates.get("short_dual_edge_floor") or {}
    name = spec.get("name") or "(unnamed spec)"
    lines = [
        f"mesh-plan: {name}",
        f"  cells            {receipt.get('predicted_cells', float('nan')):,.0f} "
        "(the generator's own sizing integral)",
    ]
    if receipt.get("card"):
        lines.append(
            f"  footprint        {receipt.get('footprint_mib'):,.1f} MiB on "
            f"{receipt['card']}"
        )
    else:
        lines.append(
            "  footprint        not sized: no --card, and the fixed term is a "
            "property of the part"
        )
    lines.extend([
        "",
        "  GATES THIS SPEC DECIDES, applied here:",
        f"    transition band   PASS  {band.get('steepest_gradient_percent_per_cell', float('nan')):.4f} "
        f"%/cell -> {band.get('band_cells', float('nan')):.2f} cells "
        f"(floor {band.get('band_cells_floor', float('nan')):.0f}, ceiling "
        f"{band.get('gradient_percent_per_cell_ceiling', float('nan')):.4f} %/cell)",
        "",
        "  GATES ONLY A BUILD DECIDES, named rather than skipped:",
        f"    shortest dual edge  {str(edge.get('verdict', 'unknown')).upper()}",
        f"      limit             {edge.get('limit_m', float('nan')):.0f} m "
        f"({edge.get('gate', 'unknown gate')})",
        f"      predicted         {_band(edge)} at a "
        f"{edge.get('finest_requested_spacing_km', float('nan')):g} km finest "
        f"spacing, over {edge.get('prediction_samples', 0)} measured meshes",
        f"      why not decidable {edge.get('why_not', '')}",
        f"      basis             {edge.get('prediction_basis', '')}",
    ])
    lines.extend(_nearest(edge))
    lines.extend([
        f"      earliest known    {edge.get('earliest_this_can_be_known', '')}",
        f"      to clear it       {edge.get('if_it_refuses_change', '')}",
    ])
    return lines


def run_mesh_plan(arguments: argparse.Namespace) -> int:
    from .swath.sizing import dry_run

    if getattr(arguments, "global_mesh", False):
        return run_global_mesh(arguments)
    if getattr(arguments, "run", False):
        raise MpasPortError("--run requires --global-mesh --generate")

    if getattr(arguments, "point", None) is not None and arguments.spec is not None:
        raise MpasPortError(
            "--spec and --point were both given; a plan prices one request. "
            "--spec hands the generator an authored spec unchanged, --point "
            "writes the ladder spec for you"
        )
    if getattr(arguments, "point", None) is not None:
        from .mesh_point import run_point

        return run_point(arguments)
    if arguments.spec is None:
        raise MpasPortError(
            "neither --spec nor --point was given; a plan needs a request. "
            "Pass --spec SPEC.json for an authored resolution spec, or "
            "--point LAT,LON [--fine-dx-m M --radius-km KM --card KEY] for a "
            "fine core at a point"
        )
    spec = _load(arguments.spec)
    receipt = dry_run(
        spec,
        engine=arguments.mesh_exe,
        card=arguments.card,
        vram_gib=arguments.vram_gib,
    )
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print("\n".join(_report(spec, receipt)))
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return 0


def add_mesh_plan_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "mesh-plan",
        help="price a resolution spec and apply the gates its build applies",
    )
    parser.add_argument(
        "--spec", type=Path, default=None,
        help="the resolution spec JSON rw_mpas_mesh would be given "
             "(or --point, below)",
    )
    parser.add_argument(
        "--card", default=None,
        help="a measured card key, so the receipt carries a device footprint",
    )
    parser.add_argument(
        "--vram-gib", type=float, default=None,
        help="a device budget instead of the named card's own memory",
    )
    parser.add_argument("--mesh-exe", type=Path, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="write the receipt here as well"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the whole receipt instead of the report",
    )
    from .mesh_point import add_point_arguments

    add_point_arguments(parser)
    global_group = parser.add_argument_group("generated global variable-resolution forecast")
    global_group.add_argument("--global-mesh", action="store_true",
                              help="price/build the whole global mesh, without a limited-area cull")
    global_group.add_argument("--run", action="store_true",
                              help="with --global-mesh --generate: init-source, forecast and Rust products")
    global_group.add_argument("--source", default=None, help="an init-source table row")
    global_group.add_argument("--cycle", default=None, help="YYYY-MM-DDTHH")
    global_group.add_argument("--hours", type=float, default=12.0)
    global_group.add_argument("--history-every-minutes", type=int, default=180)
    global_group.add_argument("--staged", type=Path, action="append", default=[])
    global_group.add_argument("--source-table", type=Path, default=None)
    global_group.add_argument("--member", type=int, default=None)
    global_group.add_argument("--jobs", type=int, default=8)
    global_group.add_argument("--region", action="append", default=[],
                              help="products view NAME=region:CLAT,CLON,W_KM,H_KM")
    parser.set_defaults(handler=run_mesh_plan)


__all__ = ["add_mesh_plan_parser", "run_mesh_plan"]


def run_global_mesh(arguments: argparse.Namespace) -> int:
    """A generated spec to a bound global forecast through existing doors.

    No per-event registry edits, source adapters, boundary inventing or
    replacement data processing. The Rust generator and static builder make
    the mesh; init-source supplies meteorology; forecast rechecks admissions.
    """
    import datetime
    import os
    import sys
    import time
    from . import card_fit, mesh_point, mesh_rows
    from .engines import MESH, STATIC, resolve
    from .swath.sizing import dry_run
    from .vertical_spec import VerticalSpec

    if arguments.point is not None and arguments.spec is not None:
        raise MpasPortError("--spec and --point are mutually exclusive")
    if arguments.spec is not None:
        spec = dict(_load(arguments.spec))
    elif arguments.point is not None:
        spec = mesh_point.ladder_spec(
            mesh_point.parse_point(arguments.point), fine_dx_m=arguments.fine_dx_m,
            radius_km=arguments.radius_km, background_km=arguments.background_km,
            transition_factor=arguments.transition_factor, ring_factor=arguments.ring_factor,
            name=arguments.name)
    else:
        raise MpasPortError("--global-mesh requires --spec or --point")
    regions = spec.get("regions") or []
    finest = min([float(spec["background_km"])] + [float(r["spacing_km"]) for r in regions]) * 1000
    if not math.isfinite(finest) or finest <= 0:
        raise MpasPortError("the finest mesh spacing must be finite and positive")
    card = mesh_point.resolve_card(arguments.card)
    if card is None:
        raise MpasPortError("--global-mesh requires --card to price the whole forecast")
    vertical = (arguments.vertical_spec or
                Path(__file__).with_name("data") / "countries" / "tc55-v1.json").resolve()
    levels = VerticalSpec.from_file(vertical).n_vert_levels
    from .mpas_defaults import default_dt_seconds
    dt = float(default_dt_seconds(finest)["dt_seconds"])
    if arguments.run:
        if not arguments.generate or arguments.out_dir is None:
            raise MpasPortError("--run requires --generate --out-dir")
        if not arguments.source or not arguments.cycle:
            raise MpasPortError("--run requires --source and --cycle")
        from .init_sources import Cycle, load_table, resolve_source, resolve_writer, resolve_vtable
        cycle = Cycle.parse(arguments.cycle)
        source_table = load_table(arguments.source_table)
        source = resolve_source(source_table, arguments.source)
        resolve_writer(None)
        resolve_vtable(source)
        seconds = arguments.hours * 3600
        cadence = arguments.history_every_minutes * 60
        if (not math.isfinite(seconds) or seconds <= 0 or cadence <= 0 or
                abs(seconds / cadence - round(seconds / cadence)) > 1e-8 or
                abs(cadence / dt - round(cadence / dt)) > 1e-8):
            raise MpasPortError("history cadence must divide the positive run length in whole timesteps")
    engine = resolve(MESH, arguments.mesh_exe)
    sizing = dry_run(spec, engine=engine)
    fit = card_fit.predict(
        sizing["predicted_cells"], card=card, levels=levels, dt_seconds=dt, hours=arguments.hours,
        budget_mib=None if arguments.vram_gib is None else arguments.vram_gib * 1024)
    receipt: dict[str, Any] = {
        "schema": "gpuwm-hex.global-vr/v1", "status": "planned", "spec": spec,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sizing": sizing, "prediction_before_generation": fit,
        "vertical_spec": str(vertical), "stages": [],
    }
    if arguments.json:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"global mesh: {fit['cells']:,} predicted cells, "
              f"{fit['predicted_peak_mib']:,.1f} MiB peak, "
              f"{fit['required_free_mib']:,.1f} MiB required, fits={fit['fits']}; "
              f"forecast runtime {fit['runtime']['seconds']} s (extrapolated)", flush=True)
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not arguments.generate:
        return 0
    if not fit["fits"]:
        raise MpasPortError(
            f"whole global mesh requires {fit['required_free_mib']:,.1f} MiB; "
            f"budget {fit['budget_mib']:,.1f} MiB. Coarsen or shrink the refinement region")
    if arguments.out_dir is None:
        raise MpasPortError("--global-mesh --generate requires --out-dir")
    root = arguments.out_dir.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "global-vr-receipt.json"
    if receipt_path.exists():
        raise MpasPortError(f"{receipt_path} exists; use a fresh --out-dir to preserve its evidence")
    spec_path, grid, static = [root / name for name in ("resolution-spec.json", "mesh.grid.nc", "mesh.static.nc")]
    rows_file = root / mesh_rows.MESH_ROWS_FILENAME
    for path in (spec_path, grid, static, rows_file):
        if path.exists():
            raise MpasPortError(f"{path} exists; use a fresh --out-dir")
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    receipt["spec_sha256"] = mesh_rows.sha256_file(spec_path)
    receipt["vertical_spec_sha256"] = mesh_rows.sha256_file(vertical)
    receipt["rows_file"] = str(rows_file)
    env = dict(os.environ)
    receipt["numerics_environment"] = {k: env.get(k) for k in ("GPUWM_HEX_NUMERICS", "GPUWM_WRF_EXACT", "GPUWM_HEX_CUDA_GRAPHS")}
    started = time.monotonic()

    def save() -> None:
        receipt["wall_seconds"] = round(time.monotonic() - started, 3)
        pending = receipt_path.with_suffix(".json.pending")
        pending.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n")
        pending.replace(receipt_path)

    def stage(name: str, argv: list[Any]) -> dict[str, Any]:
        receipt["status"] = "running"
        receipt["active_stage"] = name
        save()
        print(f"STAGE {name}", flush=True)
        result = card_fit.run_stage([str(a) for a in argv], name=name, directory=root / "logs", env=env)
        receipt["stages"].append(result)
        save()
        if result["returncode"]:
            raise MpasPortError(f"{name} exited {result['returncode']}; log {result['log']}")
        return result

    try:
        static_engine = resolve(STATIC, arguments.static_exe)
        geog = mesh_point.resolve_geog(arguments.geog)
        generator_receipt = root / "mesh-generator.json"
        static_receipt = root / "mesh-static.json"
        receipt["engines"] = {
            "mesh": {"path": str(engine), "sha256": mesh_rows.sha256_file(engine)},
            "static": {"path": str(static_engine), "sha256": mesh_rows.sha256_file(static_engine)}}
        stage("mesh", [engine, "--spec", spec_path, "--out", grid, "--receipt", generator_receipt])
        stage("static", [static_engine, "--grid", grid, "--out", static, "--geog", geog,
                         "--nominal-dx-m", finest, "--native-init", "--native-gwd-static",
                         "--receipt", static_receipt])
        errors = mesh_point.static_coordinate_errors(grid, static)
        if errors:
            raise MpasPortError(errors[0])
        admission = mesh_point.admit_pair(grid, static, fine_dx_m=finest)
        dt = admission["timestep_choice"]["dt_seconds"]
        # The point fields remain descriptive legacy metadata. The complete
        # authored spec, rather than an invented cap, is the row's authority.
        shape = min(regions, key=lambda region: float(region["spacing_km"]))["shape"] if regions else {}
        point = tuple(shape.get("center_deg", [0.0, 0.0]))
        row = mesh_rows.describe_resolution_spec(
            spec=spec, name=f"hex-vr-{receipt['spec_sha256'][:16]}-{admission['n_cells']}",
            grid=grid, static=static, spec_path=spec_path,
            generator_receipt=generator_receipt, static_receipt=static_receipt,
            point_deg=point, fine_dx_m=finest, core_radius_km=float(shape.get("radius_km", 0)),
            background_km=float(spec["background_km"]), dt_seconds=dt,
            n_levels=levels, admission=admission)
        mesh_rows.write_rows(rows_file, [row])
        receipt["mesh_row"] = row.as_dict()
        measured_fit = card_fit.predict(
            row.n_cells, card=card, levels=levels, dt_seconds=dt, hours=arguments.hours,
            budget_mib=None if arguments.vram_gib is None else arguments.vram_gib * 1024)
        receipt["prediction_on_generated_mesh"] = measured_fit
        save()
        if not measured_fit["fits"]:
            raise MpasPortError("the emitted global mesh exceeds its card budget; see prediction_on_generated_mesh")
        if arguments.run:
            cadence = arguments.history_every_minutes * 60
            if abs(cadence / dt - round(cadence / dt)) > 1e-8:
                raise MpasPortError(f"history cadence does not divide the measured mesh timestep {dt} s")
            env[mesh_rows.MESH_ROWS_ENVIRONMENT] = os.pathsep.join(
                [str(rows_file), *[str(p) for p in mesh_rows.row_files()]])
            prefix = [sys.executable, "-m", "hexcore.cli"]
            init_argv = [*prefix, "init-source", "--source", arguments.source,
                         "--cycle", arguments.cycle, "--out-dir", root / "init",
                         "--grid", grid, "--static", static, "--vertical-spec", vertical,
                         "--jobs", arguments.jobs]
            for path in arguments.staged:
                init_argv += ["--staged", path]
            if arguments.source_table:
                init_argv += ["--table", arguments.source_table]
            if arguments.member is not None:
                init_argv += ["--member", arguments.member]
            stage("init-source", init_argv)
            init_receipt = root / "init" / "init-source-receipt.json"
            receipt["source_binding"] = json.loads(init_receipt.read_text())
            run = root / "forecast"
            measured = stage("forecast", [*prefix, "forecast", "--mesh", row.name,
                "--grid", grid, "--static", static, "--init", root / "init" / "init.nc",
                "--init-source", f"{arguments.source} {arguments.cycle}; spec {receipt['spec_sha256']}",
                "--start-time", cycle.start_time, "--hours", arguments.hours,
                "--dt", dt, "--history-every-minutes", arguments.history_every_minutes,
                "--out", run, "--scratch", root / "scratch", "--no-products"])
            receipt["predicted_vs_measured"] = card_fit.comparison(measured_fit, measured)
            products = [*prefix, "products", run, "--out", root / "products", "--jobs", arguments.jobs]
            for region in arguments.region:
                products += ["--region", region]
            stage("products", products)
            receipt["products_manifest"] = str(root / "products")
        receipt["status"] = "complete" if arguments.run else "generated"
        receipt.pop("active_stage", None)
        save()
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        save()
        raise
    print(f"RECEIPT {receipt_path}", flush=True)
    return 0
