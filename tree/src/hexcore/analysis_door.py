"""``gpuwm-hex forecast --analysis BUNDLE``: a forecast that starts from an analysis bundle.

The bundle (``hexcore.analysis_bundle``, ``docs/analysis-bundle.md``) is checked byte for byte,
its mesh, static and vertical grid are checked against the run's, and the request is then
rewritten into the door's existing inputs, so the forecast itself runs exactly the code a
plain ``--init`` run does:

* direct insertion (``--iau off``, the default for a bundle with no increment.nc):
  ``--init analysis.nc``;
* incremental analysis update (``--iau on``, the default when the bundle carries
  increment.nc): ``--init`` is the background (analysis minus increment, written beside the
  run), and the increment enters over ``iau.window_s`` through ``--iau-route`` (``native``:
  MPAS-A v8.4.2's own IAU kernels, ``hexcore.iau_native``; ``hex``: ``hexcore.iau_forcing``);
* DA-cycling restart (``--restart-from CHECKPOINT``): the checkpoint's physics with the
  analysis's dycore state (``hexcore.jedi_contract.recouple_da_cycling``); ``--init`` is then
  the original run's init, as for any restart;
* ``--remap ROW``: the analysis (and its background) moved onto another mesh with the moving
  nest's operators (``hexcore.analysis_remap``), the run then on that mesh; a limited-area
  target takes its boundaries from ``--lbc-dir`` (the parent forecast's trajectory through
  ``python -m hexcore.nest_boundary``, or rw_mpas_lbc).

A bundle that carries its mesh (grid.nc, static.nc, mesh-row.json) needs no ``--mesh``,
``--grid`` or ``--static``: the door binds the carried files (``<work>/mesh-rows.json``, added
to ``$GPUWM_HEX_MESH_ROWS`` for this run).  Explicit ones are checked against the bundle's
digests as before.

Everything the door derived is written into a fresh directory beside ``--out``
(``<out>.analysis-work`` unless ``--analysis-work`` names one) and listed in the run receipt's
``analysis_bundle`` block.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .errors import MpasPortError

#: How an IAU increment enters (hexcore.iau_native).  "native": MPAS-A v8.4.2's own IAU kernels on the
#: uncoupled increments, every step against that step's state.  "hex": hexcore.iau_forcing's coupled
#: analysis-minus-background as a constant tendency.  The two differ by construction (second-order
#: terms, and in exact mode MPAS's own IN1/IN2 defects), so both stay selectable.
IAU_ROUTES = ("native", "hex")
#: The route a run takes when it names none.  Exact mode is the MPAS match, so its IAU is MPAS's
#: own; correct mode keeps hexcore.iau_forcing, the route already measured in cycling (about 70 % of
#: the direct-insertion surface-pressure shock removed on a cycled convective case).
IAU_ROUTE_DEFAULT = {"exact": "native", "correct": "hex"}


class AnalysisDoorRefusal(MpasPortError):
    """A ``--analysis`` request the door will not run; the message names why."""


def _refuse(message: str) -> AnalysisDoorRefusal:
    return AnalysisDoorRefusal(f"gpuwm-hex forecast --analysis: {message}")


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis", default=None, metavar="BUNDLE",
        help="start from an MPAS analysis bundle (hexcore.analysis_bundle, docs/analysis-bundle.md): "
             "a directory, its manifest.json, or s3://bucket/prefix/.  Every file is checked by "
             "SHA-256, and a mesh, static or vertical grid other than the bundle's is refused by name. "
             "Replaces --init (the analysis, or with --iau on its background, is the initial state); "
             "with --restart-from it is a DA-cycling restart and --init stays the original run's")
    parser.add_argument(
        "--iau", choices=("on", "off"), default=None,
        help="with --analysis: 'on' starts from the background (analysis minus increment.nc) and "
             "admits the increment over the bundle's IAU window; 'off' inserts the analysis at the "
             "start.  Default: on when the bundle carries increment.nc, off otherwise")
    parser.add_argument(
        "--iau-route", choices=IAU_ROUTES, default=None,
        help="how an IAU increment enters (--analysis with --iau on, or --iau-analysis/--iau-increment): "
             "'native' runs MPAS-A v8.4.2's own IAU kernels (hexcore.iau_v842) on the uncoupled "
             "increments every step, word for word in exact mode and with IN1-IN3 in correct mode; "
             "'hex' adds hexcore.iau_forcing's coupled analysis-minus-background as a constant "
             "tendency.  Default: native in exact mode, hex in correct mode")
    parser.add_argument(
        "--iau-increment", type=Path, default=None, metavar="NC",
        help="MPAS IAU increments (u_amb, rho_amb, theta_amb, q*_amb) of --init's analysis; "
             "with --iau-window-seconds, the native route's input")
    parser.add_argument(
        "--remap", default=None, metavar="ROW",
        help="with --analysis: move the analysis onto this registered mesh row (regional or finer) "
             "with the moving nest's conservative operators (hexcore.analysis_remap) and run there. "
             "--grid/--static then name the BUNDLE's mesh; --remap-grid/--remap-static/--remap-template "
             "name the target's.  A limited-area target needs --lbc-dir from the parent forecast")
    parser.add_argument("--remap-grid", type=Path, default=None, metavar="FILE")
    parser.add_argument("--remap-static", type=Path, default=None, metavar="FILE")
    parser.add_argument(
        "--remap-template", type=Path, default=None, metavar="FILE",
        help="an initial-conditions file of the target mesh: its terrain, reference state and land "
             "state are kept, every atmospheric field comes from the analysis")
    parser.add_argument("--remap-template-kind", choices=("initialized", "vertical"), default="initialized",
                        help="'vertical': the template carries only the vertical grid (the land and "
                             "surface state are carried from the analysis)")
    parser.add_argument("--remap-operator", type=Path, default=None, metavar="NPZ",
                        help="cache of the conservative remap weights (built and saved when absent)")
    parser.add_argument("--analysis-work", type=Path, default=None, metavar="DIR",
                        help="fresh directory for what the door derives from the bundle (background, "
                             "remapped files, fetched bundle); default <out>.analysis-work")


def resolve_iau_route(arguments: Any, physics_mode: str) -> str:
    route = getattr(arguments, "iau_route", None)
    if route is not None:
        return str(route)
    return IAU_ROUTE_DEFAULT[physics_mode]


def resolve_analysis(arguments: Any, *, registry: Mapping[str, Any] | None = None,
                     registry_loader: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Rewrite ``arguments`` for ``--analysis``; return the receipt block, or None without it.

    ``registry_loader`` is called after a carried mesh is bound, so the row it registers is in
    the registry the door checks ``--mesh``/``--remap`` against; ``registry`` is a fixed one.
    """

    source = getattr(arguments, "analysis", None)
    if source is not None:
        # numpy and netCDF4 load only for a run that reads a bundle (this module is imported by
        # the console script's parser, which must start without them).
        from .analysis_bundle import (BundleRefusal, check_run, load_bundle, vertical_fingerprint,
                                      write_background, write_increment)
    iau = getattr(arguments, "iau", None)
    remap = getattr(arguments, "remap", None)
    if source is None:
        for flag, value in (("--iau", iau), ("--remap", remap),
                            ("--remap-grid", getattr(arguments, "remap_grid", None)),
                            ("--remap-template", getattr(arguments, "remap_template", None)),
                            ("--analysis-work", getattr(arguments, "analysis_work", None))):
            if value is not None:
                raise _refuse(f"{flag} acts on an analysis bundle and no --analysis was given; it would be "
                              "accepted and never used")
        return None
    # Retired (named gap: bundles for member rosters): a roster resolves each member's bundle through this
    # function (hexcore.multi_member.roster_requests, the record's "analysis" or the roster-wide --analysis).
    for flag, name in (("--iau-analysis", "iau_analysis"), ("--iau-increment", "iau_increment"),
                       ("--da-analysis", "da_analysis")):
        if getattr(arguments, name, None) is not None:
            raise _refuse(f"{flag} names a second analysis; the bundle carries its own")
    restart = getattr(arguments, "restart_from", None)
    if restart is None and getattr(arguments, "init", None) is not None:
        raise _refuse("--init and --analysis both name the initial state; the bundle's analysis (or, with "
                      "--iau on, its background) is the run's init")
    if restart is not None and getattr(arguments, "init", None) is None:
        raise _refuse("a DA-cycling restart (--restart-from) needs --init, the original run's init, for the "
                      "reference state the analysis is recoupled on")
    # Retired (named gap W1-G1): checkpoints carry their IAU window and a restart resolves it by rule
    # (hexcore.iau_native.resolve_restart_window), so --iau on with --restart-from runs: the checkpoint is the
    # background and the bundle's increment.nc is admitted over a window that begins at the restart.
    if restart is not None and iau == "on" and getattr(arguments, "iau_route", None) == "hex":
        raise _refuse("--iau-route hex forms its increment as analysis minus --init, which is not a restarted "
                      "run's background (its checkpoint state); a DA-cycling restart with --iau on admits the "
                      "bundle's increment.nc on the native route (drop --iau-route or name native)")
    if restart is not None and remap is not None:
        raise _refuse("--remap moves an analysis onto another mesh for a cold start; a restart continues "
                      "the checkpoint's own mesh")
    if remap is not None:
        for flag, name in (("--remap-grid", "remap_grid"), ("--remap-static", "remap_static"),
                           ("--remap-template", "remap_template")):
            if getattr(arguments, name, None) is None:
                raise _refuse(f"--remap {remap} needs {flag} (the target mesh's file)")
            if not Path(getattr(arguments, name)).is_file():
                raise _refuse(f"{flag} {getattr(arguments, name)} is not a file")
    out = getattr(arguments, "out", None)
    if out is None:
        raise _refuse("--out is required (the derived files go beside it)")
    preflight = bool(getattr(arguments, "preflight", False))
    if not preflight:
        # The door's own cheap refusals, asked before a background or remap is written for a run
        # that would then be refused on them.
        if Path(out).expanduser().exists():
            raise _refuse(f"--out {out} exists; a forecast writes into a fresh directory")
        for flag, name in (("--hours", "hours"), ("--history-every-minutes", "history_every_minutes")):
            if getattr(arguments, name, None) is None:
                raise _refuse(f"{flag} is required")
    work = Path(getattr(arguments, "analysis_work", None) or
                Path(out).expanduser().absolute().with_name(Path(out).name + ".analysis-work")).absolute()
    if not preflight:
        if work.exists():
            raise _refuse(f"{work} exists; what the door derives from a bundle goes into a fresh directory so "
                          "a background or remap of another bundle cannot be read in its place")
        work.mkdir(parents=True)
    try:
        return _resolve(arguments, source=source, iau=iau, remap=remap, restart=restart, work=work,
                        preflight=preflight, registry=registry, registry_loader=registry_loader)
    except BaseException:
        if not preflight and work.exists():
            import shutil

            shutil.rmtree(work, ignore_errors=True)
        raise


def _bind_mesh(arguments: Any, bundle: Any, work: Path, preflight: bool) -> dict[str, Any]:
    """--mesh/--grid/--static from the bundle's carried mesh where the run names none."""

    import os

    from .analysis_bundle import BundleRefusal, bind_carried_mesh
    from .mesh_rows import MESH_ROWS_ENVIRONMENT

    named = {flag: getattr(arguments, name, None) for flag, name in
             (("--mesh", "mesh"), ("--grid", "grid"), ("--static", "static"))}
    record: dict[str, Any] = {"carried": bundle.carried_static is not None, "bound": []}
    if bundle.carried_static is None:
        for flag in ("--grid", "--static"):
            if named[flag] is None:
                raise _refuse(f"{flag} must name the bundle's mesh file (the bundle carries no mesh); the door "
                              "checks it against the bundle's SHA-256 rather than guess which mesh the analysis "
                              "is on")
        if named["--mesh"] is None and not preflight:
            raise _refuse("--mesh is required (the bundle carries no mesh row)")
        return record
    try:
        if preflight:
            # a preflight writes nothing beside --out; the carried row is bound from a private temporary
            # folder so the registry the preflight checks against holds it
            import tempfile

            bound = bind_carried_mesh(bundle, Path(tempfile.mkdtemp(prefix="hex-analysis-preflight-")))
        else:
            bound = bind_carried_mesh(bundle, work)
    except BundleRefusal as error:
        raise _refuse(str(error)) from None
    if bound.get("row") and named["--mesh"] is None:
        # A row of the same name already registered for this process ($GPUWM_HEX_MESH_ROWS, a box that made the
        # mesh) would be registered twice, which the registry refuses.  The same bytes are the same mesh: that
        # registration binds the run.  Other bytes under the bundle's mesh name are refused by name.
        from .mesh_rows import MeshRowRefusal, load_rows

        try:
            existing = [row for row in load_rows() if row.name == bound["row"]]
        except MeshRowRefusal as error:
            raise _refuse(f"${MESH_ROWS_ENVIRONMENT} cannot be read: {error}") from None
        if existing:
            row = existing[0]
            manifest = bundle.manifest
            if row.grid_sha256 != manifest["mesh"]["sha256"] or row.static_sha256 != manifest["static"]["sha256"]:
                raise _refuse(f"${MESH_ROWS_ENVIRONMENT} already registers mesh {row.name!r} with other bytes (grid "
                              f"{row.grid_sha256[:16]}..., static {str(row.static_sha256)[:16]}...) than the bundle's "
                              f"(grid {manifest['mesh']['sha256'][:16]}..., static {manifest['static']['sha256'][:16]}...); "
                              "one name cannot bind two meshes")
            bound = {"row": row.name, "grid": row.grid, "static": row.static, "rows": None,
                     "existing_registration": True}
            record["existing_registration"] = True
    if named["--grid"] is None:
        arguments.grid = Path(bound["grid"])
        record["bound"].append("--grid")
    if named["--static"] is None:
        arguments.static = Path(bound["static"])
        record["bound"].append("--static")
    if named["--mesh"] is None:
        if bundle.carried_row is None:
            raise _refuse("--mesh is required: the bundle carries its grid and static but no mesh-row.json, so "
                          "the row that binds them (timestep, levels, admission) must be named")
        arguments.mesh = bound["row"]
        record["bound"].append("--mesh")
    if bound.get("rows") and named["--mesh"] is None:
        previous = os.environ.get(MESH_ROWS_ENVIRONMENT)
        os.environ[MESH_ROWS_ENVIRONMENT] = os.pathsep.join(filter(None, [bound["rows"], previous]))
        record["mesh_rows"] = bound["rows"]
    record.update(row=bound.get("row"), grid=str(arguments.grid), static=str(arguments.static))
    if record["bound"]:
        print(f"ANALYSIS the bundle carries its mesh: {', '.join(record['bound'])} bound to the bundle's files"
              + (f" (row {arguments.mesh})" if "--mesh" in record["bound"] else ""), file=sys.stderr, flush=True)
    return record


def _apply_forecast_defaults(arguments: Any, bundle: Any) -> dict[str, Any] | None:
    """The producer's forecast configuration (manifest ``forecast.args``) for every door option the
    run left at its default; an option the run names is the run's.  THE BREAKAGE THIS PREVENTS: a
    forecast leg that passes --analysis alone integrated the door's default timestep, executor and
    physics, not the model the analysis was cycled with (measured: a leg ran dt 18 s where the
    producer's own forecast ran 18.75 s, so the two diverged from the first step)."""

    args = list((bundle.manifest.get("forecast") or {}).get("args") or [])
    if not args:
        return None
    from .forecast_door import add_forecast_arguments

    parser = argparse.ArgumentParser(add_help=False)
    add_forecast_arguments(parser)
    try:
        produced = parser.parse_args(args)
    except SystemExit:
        raise _refuse(f"the bundle's forecast.args {args} do not parse as door flags") from None
    applied: list[str] = []
    run_named: list[str] = []
    for action in parser._actions:
        dest = action.dest
        default = parser.get_default(dest)
        value = getattr(produced, dest, default)
        if value == default:
            continue
        if getattr(arguments, dest, default) == default:
            setattr(arguments, dest, value)
            applied.append(dest)
        else:
            run_named.append(dest)
    print(f"ANALYSIS the producer's forecast configuration: {' '.join(args)} (applied: {', '.join(applied) or 'none'}"
          + (f"; the run names {', '.join(run_named)}" if run_named else "") + ")", file=sys.stderr, flush=True)
    return {"args": args, "applied": applied, "run_named": run_named}


def _resolve(arguments: Any, *, source: Any, iau: Any, remap: Any, restart: Any, work: Path,
             preflight: bool, registry: Mapping[str, Any] | None,
             registry_loader: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    from .analysis_bundle import (BundleRefusal, check_run, increment_variables, load_bundle,
                                  vertical_fingerprint, write_background, write_increment)

    try:
        bundle = load_bundle(source, fetch_to=None if preflight and str(source).startswith("s3://") else work / "bundle")
    except BundleRefusal as error:
        raise _refuse(str(error)) from None
    mesh_binding = _bind_mesh(arguments, bundle, work, preflight)
    forecast_defaults = _apply_forecast_defaults(arguments, bundle)
    if registry is None and registry_loader is not None:
        registry = registry_loader()
    if remap is not None and registry is not None and remap not in registry:
        raise _refuse(f"--remap {remap!r} is not a registered mesh row ({', '.join(sorted(registry))})")
    mesh_name = getattr(arguments, "mesh", None)
    levels = None
    if registry is not None and mesh_name in (registry or {}):
        levels = getattr(registry[mesh_name], "n_levels", None)
    vertical_sources = []
    if getattr(arguments, "invariant", None) is not None and not getattr(arguments, "no_invariant", False):
        vertical_sources.append(arguments.invariant)
    if restart is not None:
        vertical_sources.append(arguments.init)
    try:
        checks = check_run(bundle, grid=arguments.grid, static=arguments.static, levels=levels,
                           vertical_sources=vertical_sources)
    except BundleRefusal as error:
        raise _refuse(str(error)) from None
    start = getattr(arguments, "start_time", None)
    receipt: dict[str, Any] = {**bundle.receipt(), "checks": checks, "work": str(work), "mesh_binding": mesh_binding,
                               "forecast_defaults": forecast_defaults}
    if restart is None:
        if start is not None and str(start) != bundle.valid_time:
            raise _refuse(f"--start-time {start} is not the analysis's valid time {bundle.valid_time}")
        arguments.start_time = bundle.valid_time
    if getattr(arguments, "init_source", None) is None:
        arguments.init_source = bundle.provenance_sentence()
    run_mode = getattr(arguments, "physics_mode", None)
    if run_mode is not None and run_mode != bundle.manifest["physics"]["mode"]:
        print(f"ANALYSIS the bundle was cycled with physics.mode {bundle.manifest['physics']['mode']}; this "
              f"forecast runs {run_mode}", file=sys.stderr, flush=True)
    if restart is not None and iau == "on":
        if bundle.increment is None:
            raise _refuse("--iau on: the bundle carries no increment.nc, so a restarted run has nothing to admit "
                          "over a window (its checkpoint is the background)")
        window = getattr(arguments, "iau_window_seconds", None)
        arguments.iau_increment = bundle.increment
        arguments.iau_route = "native"
        arguments.iau_window_seconds = float(window if window is not None else bundle.iau_window_seconds)
        receipt["run_iau"] = {"on": True, "selected": "explicit", "route": "native",
                              "window_seconds": arguments.iau_window_seconds, "increment": str(bundle.increment),
                              "window_source": "--iau-window-seconds" if window is not None else "manifest iau.window_s"}
        receipt["start"] = ("DA-cycling restart with IAU (--restart-from, --iau on): the checkpoint is the background "
                            "and the bundle's increment is admitted over a window that begins at the restart")
        return receipt
    if restart is not None:
        arguments.da_analysis = bundle.analysis
        receipt["start"] = "DA-cycling restart (--restart-from): checkpoint physics, analysis dycore state"
        return receipt
    if bundle.form != "init":
        raise _refuse(f"the bundle's analysis is in {bundle.form} form, which carries no reference state of "
                      "its own; it starts only a DA-cycling restart (--restart-from with the original --init)")
    if iau == "on" and bundle.increment is None:
        raise _refuse("--iau on: the bundle carries no increment.nc, so there is no background to start the "
                      "window from")
    iau_on = iau == "on" or (iau is None and bundle.increment is not None)
    window = getattr(arguments, "iau_window_seconds", None)
    if window is not None and not iau_on:
        raise _refuse("--iau-window-seconds was given and the run inserts the analysis directly (--iau off)")
    receipt["run_iau"] = {"on": bool(iau_on), "selected": "explicit" if iau is not None else "bundle default"}
    analysis_file = bundle.analysis
    background_file = None
    increment_file = bundle.increment
    if preflight:
        arguments.init = analysis_file
        receipt["start"] = "preflight: the background and any remap are not built"
        return receipt
    if iau_on:
        background_file = work / "background.nc"
        receipt["background"] = write_background(bundle.analysis, bundle.increment, background_file)
    if remap is not None:
        from .analysis_remap import SPECIES, transfer

        operator = Path(getattr(arguments, "remap_operator", None) or (work / "remap-operator.npz"))
        remapped = {}
        for role, path in (("analysis", analysis_file), ("background", background_file)):
            if path is None:
                continue
            target = work / f"{role}.remapped.nc"
            try:
                remapped[role] = transfer(SimpleNamespace(
                    source_grid=Path(arguments.grid), source_static=Path(arguments.static), analysis=Path(path),
                    target_grid=Path(arguments.remap_grid), target_static=Path(arguments.remap_static),
                    template=Path(arguments.remap_template), operator=operator, output=target,
                    receipt=work / f"{role}.remap.json", species=",".join(SPECIES),
                    template_kind=getattr(arguments, "remap_template_kind", "initialized")))
            except (ValueError, KeyError) as error:
                # The operators bind meshes made by hex's own generator (gpuwm-hex mesh-plan, the
                # moving nest): an archived mesh's stored areas are not the generator's sphere,
                # and the transfer refuses it deep inside after reading both meshes.
                raise _refuse(f"--remap {remap}: the transfer of the {role} refused: {type(error).__name__}: "
                              f"{error}.  The remap moves an analysis onto a mesh made by hex's own generator "
                              "(a regional cull or nest of the analysis's region)") from None
            if role == "analysis":
                analysis_file = target
            else:
                background_file = target
        if iau_on:
            increment_file = work / "increment.remapped.nc"
            remapped["increment"] = write_increment(analysis_file, background_file, increment_file)
        receipt["remap"] = {"row": remap, "source_mesh": bundle.manifest["mesh"]["id"],
                            "target_grid": str(arguments.remap_grid), "target_static": str(arguments.remap_static),
                            "template": str(arguments.remap_template), "operator": str(operator),
                            "transfers": remapped,
                            "target_vertical": vertical_fingerprint(analysis_file),
                            "boundaries": ("--lbc-dir " + str(arguments.lbc_dir)
                                           if getattr(arguments, "lbc_dir", None) else "none (global target)")}
        arguments.mesh, arguments.grid, arguments.static = remap, arguments.remap_grid, arguments.remap_static
    if iau_on:
        mpas_stream = {"u_amb", "rho_amb", "theta_amb", "qv_amb", "qc_amb", "qr_amb", "qi_amb", "qs_amb", "qg_amb"}
        carried = sorted(increment_variables(increment_file))
        extra = [name for name in carried if name not in mpas_stream]
        receipt["run_iau"]["increment_variables"] = carried
        if extra:
            route = resolve_iau_route(arguments, getattr(arguments, "physics_mode", None) or "correct")
            if route == "native":
                receipt["run_iau"]["native_route_not_admitted"] = extra
                print(f"ANALYSIS --iau-route native admits MPAS's iau stream (u rho theta qv..qg); {', '.join(extra)} "
                      "start at the background's values and are not admitted (MPAS IAU has no such fields).  "
                      "--iau-route hex admits every prognostic", file=sys.stderr, flush=True)
        arguments.init = background_file
        arguments.iau_analysis = analysis_file
        arguments.iau_increment = increment_file
        arguments.iau_window_seconds = float(window if window is not None else bundle.iau_window_seconds)
        receipt["run_iau"].update(window_seconds=arguments.iau_window_seconds,
                              window_source="--iau-window-seconds" if window is not None else "manifest iau.window_s",
                              background=str(background_file), increment=str(increment_file))
        receipt["start"] = "IAU: the background, with the increment admitted over the window"
    else:
        arguments.init = analysis_file
        receipt["start"] = "direct insertion: the analysis is the initial state"
    receipt["init"] = str(arguments.init)
    return receipt
