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
        "--spec", type=Path, required=True,
        help="the resolution spec JSON rw_mpas_mesh would be given",
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
    parser.set_defaults(handler=run_mesh_plan)


__all__ = ["add_mesh_plan_parser", "run_mesh_plan"]
