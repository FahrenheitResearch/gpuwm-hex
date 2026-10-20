"""``gpuwm-hex physics``: the named physics suites, on the host, with no card.

Three verbs, all read from the one suite table (hexcore.physics_suites):

* ``list``    -- every named suite, MPAS's own suites, and every option the
  column seams carry or refuse (with the breakage and the follow-up);
* ``resolve`` -- what a ``--physics-suite``/``--physics`` request runs: the
  seam, the backend row, the convection request, the differences;
* ``import``  -- a ``namelist.atmosphere``, option by option.

The forecast door takes the same flags, so a request checked here is the
request the door resolves.  Exit status: 0 resolved, 3 refused.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from .errors import ConfigurationRefusal

REFUSED_EXIT = 3


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=False, default=str))


def _mesh_spacing(mesh: str | None) -> float | None:
    """The registered row's declared spacing, so ``auto`` cumulus resolves
    here as the forecast door resolves it; ``None`` for no or an unknown mesh."""

    if mesh is None:
        return None
    from .forecast_door import load_registry

    row = load_registry().get(mesh)
    return None if row is None else getattr(row, "nominal_dx_m", None)


def _list(arguments: argparse.Namespace) -> int:
    from .physics_suites import load_table, mpas_suite_names, resolve_physics, suite_names

    table = load_table()
    suites = []
    for name in suite_names() + mpas_suite_names():
        try:
            resolved = resolve_physics(
                name, mesh=arguments.mesh, nominal_dx_m=_mesh_spacing(arguments.mesh))
            verdict = {"runs": True, "seam": resolved.seam, "backend_row": resolved.backend_row,
                       "convection_request": resolved.convection}
        except ConfigurationRefusal as error:
            verdict = {"runs": False, "refusals": [
                item.line() for item in getattr(error, "refusals", ())] or [str(error)]}
        row = next((item for item in table["suites"] if item["name"] == name), None)
        suites.append({
            "name": name,
            "kind": "hex" if row is not None else "MPAS",
            "default": bool(row is not None and row.get("default")),
            "describes": None if row is None else row["describes"],
            "selectors": dict(row["selectors"]) if row is not None
            else dict(table["mpas_suites"]["suites"][name]),
            **verdict,
        })
    options = {
        key: {value: (f"refused: {item['refused']}" if "refused" in item
                      else "hex rule (decided at the mesh bind)" if not item.get("engine")
                      else "WOOF " + ", ".join(f"{k}={v!r}" for k, v in item["engine"].items()))
              for value, item in rows.items()}
        for key, rows in table["options"].items()
    }
    # What exists is the engine's to say: every implemented option of WOOF's
    # registry, selectable by id or selector value under the same key.
    from . import engine_registry

    registry: dict[str, Any] = {}
    try:
        for key, route in table["engine_selectors"]["selectors"].items():
            if "component" in route:
                registry[key] = {
                    ident: {route["selector"]: option["selectors"][route["selector"]],
                            "label": option.get("label"), "maturity": option.get("maturity")}
                    for ident, option in engine_registry.implemented_options(
                        route["component"]).items()
                    if route["selector"] in option["selectors"]
                }
            else:
                row = engine_registry.parameter_row(route["parameter"]) or {}
                registry[key] = {route["parameter"]: row.get("enum")}
        registry_source = str(engine_registry.engine_root())
        seam_problem = engine_registry.generic_seam_problem()
    except engine_registry.EngineRegistryUnavailable as error:
        registry_source, seam_problem = None, str(error)
    _print({"mesh": arguments.mesh, "suites": suites, "options": options,
            "woof_registry": {"engine": registry_source, "options": registry,
                              "generic_seam_problem": seam_problem},
            "generation_selectors": {
                key: list(row["values"]) for key, row in table["generation_selectors"].items()}})
    return 0


def _resolve(arguments: argparse.Namespace) -> int:
    from .physics_suites import parse_override, resolve_physics

    try:
        overrides = dict(parse_override(text) for text in arguments.physics or ())
        resolved = resolve_physics(
            arguments.physics_suite, overrides, mesh=arguments.mesh,
            nominal_dx_m=_mesh_spacing(arguments.mesh))
    except ConfigurationRefusal as error:
        _print({"resolved": False, "refusals": [
            {"option": item.knob, "value": item.value, "reason": item.reason,
             "follow_up": item.follow_up}
            for item in getattr(error, "refusals", ())] or [str(error)]})
        return REFUSED_EXIT
    _print({"resolved": True, **resolved.receipt()})
    return 0


def _import(arguments: argparse.Namespace) -> int:
    from .physics_namelist import read_physics_namelist

    result = read_physics_namelist(arguments.namelist, mesh=arguments.mesh)
    _print(result.report())
    return 0 if result.admitted else REFUSED_EXIT


def add_physics_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "physics",
        help="list, resolve and import named physics suites (no GPU)",
        description=(
            "The physics a forecast runs is a named suite plus overrides, read "
            "from one table (data/profiles/mpas-physics-suites.v1.json). These "
            "verbs answer, on the host, what `gpuwm-hex forecast` would run for "
            "the same flags."
        ),
    )
    verbs = parser.add_subparsers(dest="physics_verb", required=True)
    listing = verbs.add_parser("list", help="every suite and every option")
    listing.add_argument("--mesh", default=None,
                         help="a registered mesh row, for suites with per-mesh selectors")
    listing.set_defaults(handler=_list)
    resolve = verbs.add_parser("resolve", help="what one request runs")
    resolve.add_argument("--physics-suite", default=None, metavar="NAME")
    resolve.add_argument("--physics", action="append", default=None, metavar="KEY=VALUE")
    resolve.add_argument("--mesh", default=None)
    resolve.set_defaults(handler=_resolve)
    importer = verbs.add_parser("import", help="a namelist.atmosphere, option by option")
    importer.add_argument("namelist", metavar="NAMELIST")
    importer.add_argument("--mesh", default=None)
    importer.set_defaults(handler=_import)
