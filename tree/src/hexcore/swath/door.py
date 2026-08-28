"""``gpuwm-hex swath`` -- the front door of the placement layer.

Three legs, because a user meets this capability in three ways: they want
a plan for a cycle (``plan``), they want to know what the machine is armed
to look for and what the coarse run must publish for it (``metrics``), or
they want to know why a particular storm did or did not get a grid
(``explain``).

``plan`` PRICES BY DEFAULT.  It drives the real ``rw_mpas_mesh --dry-run``
on every admitted swath, on the CPU, writing nothing, because a plan whose
cells were never sized is the plan that discovers a swath does not fit the
card forty minutes into a forecast.  ``--no-size`` is the explicit opt-out
and stamps every row with the weaker basis so no receipt can quietly
present arithmetic as a measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..errors import MpasPortError
from . import registry as registry_module
from . import sizing
from .errors import SwathError
from .history import HistoryReader
from .hysteresis import SwathState
from .plan import plan_cycle, plan_document


def _write(path: Path, document: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _price(document: dict[str, Any], arguments: argparse.Namespace) -> None:
    """Replace every admitted row's derived cell count with a measured one."""

    if not arguments.size:
        for row in document["admitted"]:
            row["sizing"]["engine"] = None
            row["sizing"]["basis"] = "area_integral"
            row["sizing"]["not_sized_because"] = (
                "--no-size was given, so no swath in this plan has been sized by the "
                "generator's own sizing integral. Every cell figure here is this "
                "layer's area arithmetic and none of it has been checked against "
                "what the mesh would actually deliver"
            )
        return
    engine = sizing.resolve_engine(arguments.mesh_exe)
    for row in document["admitted"]:
        widest = max(row["half_widths_km"]) if row["half_widths_km"] else 0.0
        result = sizing.size_swath_spec(
            row["mesh_spec"],
            region_index=0,
            probe_centre=(row["centroid_deg"][0], row["centroid_deg"][1]),
            probe_radius_km=widest,
            ring=[tuple(vertex) for vertex in row["ring_deg"]],
            engine=engine,
            card=arguments.card,
        )
        row["sizing"] = {
            **row["sizing"],
            **result.as_row(),
            "engine": str(engine),
            "swath_cells_at_requested_spacing": row["sizing"].get("predicted_cells"),
        }
        row["sizing"].pop("predicted_cells", None)
        row["sizing"].pop("basis", None)
        sizing.refuse_over_ceiling(
            result.swath_cells,
            float(row["sizing"]["maximum_cells_per_swath"]),
            row["slot_id"],
        )


def run_plan(arguments: argparse.Namespace) -> int:
    metrics = registry_module.load_metrics(arguments.metrics)
    policy = registry_module.load_policy(arguments.policy)
    state = SwathState.load(arguments.state)
    with HistoryReader(arguments.history, grid=arguments.grid) as reader:
        result = plan_cycle(
            reader, metrics, policy, state=state, cycle_index=arguments.cycle_index
        )
        document = json.loads(json.dumps(plan_document(reader, metrics, policy, result)))
    _price(document, arguments)

    if arguments.out is not None:
        out = Path(arguments.out).expanduser()
        _write(out / "swath-plan.json", document)
        _write(
            out / "threat-decision.json",
            {
                "schema": "gpuwm-hex.threat-decision.v1",
                "history": document["history"],
                "metrics_document": document["metrics_document"],
                "tracks": document["tracks"],
                "drops": document["drops"],
                "declined": document["declined"],
            },
        )
        _write(out / "swath-state.json", document["state"])
        for row in document["admitted"]:
            _write(out / "specs" / f"{row['slot_id']}.mesh-spec.json", row["mesh_spec"])
            _write(out / "specs" / f"{row['slot_id']}.cull-region.json", row["cull_region"])
        print(json.dumps({
            "out": str(out),
            "admitted": len(document["admitted"]),
            "declined": len(document["declined"]),
            "churn": document["churn"],
        }, indent=2, sort_keys=True))
    else:
        print(json.dumps(document, indent=2, sort_keys=True))

    if not document["admitted"]:
        print(
            "gpuwm-hex swath: this cycle placed NO swath. That is a result, not a "
            "failure: the armed metric rows found nothing above threshold in this "
            "forecast, or everything they found was declined by name. The reasons "
            "are in 'declined' and 'drops'.",
        )
    return 0


def run_metrics(arguments: argparse.Namespace) -> int:
    metrics = registry_module.load_metrics(arguments.metrics)
    if arguments.publication_manifest:
        print(json.dumps(list(metrics.publication_manifest()), indent=2))
        return 0
    print(json.dumps({
        "schema": metrics.schema,
        "sha256": metrics.sha256,
        "path": str(metrics.source_path) if metrics.source_path else None,
        "publication_manifest": list(metrics.publication_manifest()),
        "fields": [
            {
                "id": row.id,
                "source_variables": list(row.source_variables),
                "inputs": list(row.inputs),
                "leaf_variables": list(
                    metrics.leaf_variables(row.id, whose=f"field row {row.id!r}")
                ),
                "derivation": row.derivation_kind,
                "units": row.units,
            }
            for row in metrics.field_rows.values()
        ],
        "metrics": [
            {
                "id": row.id,
                "threat_class": row.threat_class,
                "field": row.field,
                "enabled": row.enabled,
                "detector": row.detector.kind,
                "confirm_with": [item.field for item in row.confirm_with],
                "start_policy": row.start_policy.kind,
                "region": dict(row.region.as_row()),
                "intensity_reference": row.rank.intensity_reference,
                "spacing_km": row.swath.spacing_km,
                "lead_hours": row.swath.lead_hours,
                "half_width_km": [row.swath.half_width_km, row.swath.maximum_half_width_km],
                "needs": list(
                    metrics.leaf_variables(row.field, whose=f"metric {row.id!r}")
                ),
            }
            for row in metrics.metric_rows.values()
        ],
        "vocabularies": {
            "derivation": list(registry_module.DERIVATION_KINDS),
            "detector": list(registry_module.DETECTOR_KINDS),
            "aggregation": list(registry_module.AGGREGATION_KINDS),
            "start_policy": list(registry_module.START_POLICY_KINDS),
            "rank_terms": list(registry_module.RANK_TERM_KINDS),
            "region": list(registry_module.REGION_KINDS),
        },
    }, indent=2, sort_keys=True))
    return 0


def run_explain(arguments: argparse.Namespace) -> int:
    path = Path(arguments.plan).expanduser()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SwathError(f"cannot read the swath plan at {path}: {error}") from error
    lines: list[str] = []
    lines.append(f"cycle {document.get('cycle_index')} from {document['history']['path']}")
    lines.append(f"  history sha256 {document['history']['sha256']}")
    lines.append(
        f"  metrics {document['metrics_document']['sha256'][:16]} "
        f"policy {document['policy_document']['sha256'][:16]}"
    )
    lines.append("")
    lines.append("ADMITTED")
    for row in document["admitted"]:
        if arguments.slot and row["slot_id"] != arguments.slot:
            continue
        lines.append(
            f"  {row['slot_id']}  {row['metric_id']}  {row['threat_class']}"
        )
        lines.append(
            f"    score {row['rank']['score']:.4f}  effective {row['effective_score']:.4f}"
        )
        for term in row["rank"]["terms"]:
            lines.append(
                f"      {term['id']:<14} {term['kind']:<24} raw {term['raw']:>12.3f}"
                f" / reference {term.get('reference', 1.0):<10g}"
                f" contributes {term['contribution']:.4f}"
            )
        hyst = row.get("hysteresis") or {}
        lines.append(
            f"    hysteresis: incumbent={hyst.get('incumbent')} "
            f"dwell_protected={hyst.get('dwell_protected')} "
            f"mesh={hyst.get('mesh_action')} -- {hyst.get('reason')}"
        )
        lines.append(
            f"    ignition {row['ignite_at_seconds'] / 3600.0:.2f} h, lead "
            f"{row['lead_hours']:.2f} h, idle {row['idle_lead_hours']:.2f} h, "
            f"extrapolated {row['extrapolated_hours']:.2f} h"
        )
        lines.append(
            f"    axis smoothing {row['axis_smoothing_passes']} passes, drift "
            f"{row['axis_smoothing_drift_km']:.1f} km"
        )
        size = row["sizing"]
        lines.append(
            f"    shape {row['cull_region']['kind']}, swath cells "
            f"{size.get('swath_cells', size.get('predicted_cells', 0)):,.0f} "
            f"(basis {size.get('swath_basis', size.get('basis'))}) against ceiling "
            f"{size.get('maximum_cells_per_swath', 0):,.0f}"
        )
        lines.append(
            f"    graded parent {size.get('parent_cells', 0):,.0f} cells "
            f"(basis {size.get('parent_basis', 'not sized')}), footprint "
            f"{size.get('parent_footprint_mib')} MiB on {size.get('card')}"
        )
        lines.append(
            f"    spacing requested {row['mesh_spec']['regions'][0]['spacing_km']} km, "
            f"ATTAINED {size.get('attained_spacing_km')} km "
            f"(basis {size.get('attained_basis', 'not measured')})"
        )
        lines.append("")
    lines.append("DECLINED")
    for row in document["declined"]:
        lines.append(f"  {row['track_id']}  score {row['rank']['score']:.4f}")
        lines.append(f"    {row['reason']}")
    lines.append("")
    lines.append(f"CHURN {json.dumps(document['churn'], sort_keys=True)}")
    print("\n".join(lines))
    return 0


def add_swath_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "swath",
        help="decide where the fine grid goes, from a coarse forecast (CPU)",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    legs = parser.add_subparsers(dest="swath_command", required=True)

    plan_parser = legs.add_parser(
        "plan",
        help="place this cycle's swaths and write the plan",
        description=(
            "Read a coarse forecast this project produced, find what is worth "
            "resolving, project where it will be, and emit one mesh-spec row and "
            "one cull-region row per admitted swath -- both in the grammar "
            "rw_mpas_mesh already reads. Nothing here opens a device."
        ),
    )
    plan_parser.add_argument(
        "--history", type=Path,
        help=(
            "the coarse forecast to read: either a self-describing history "
            "file, or the run receipt 'gpuwm-hex forecast' writes beside its "
            "frames (which names the whole sequence, its times and its grid)"
        ),
    )
    plan_parser.add_argument(
        "--grid", type=Path, default=None,
        help=(
            "the mesh the forecast was integrated on, when the history does "
            "not carry its own connectivity and no run receipt names it"
        ),
    )
    plan_parser.add_argument(
        "--metrics", type=Path, default=None,
        help="a threat-metrics document, replacing the one that ships",
    )
    plan_parser.add_argument(
        "--policy", type=Path, default=None,
        help="a placement-policy document, replacing the one that ships",
    )
    plan_parser.add_argument(
        "--state", type=Path, default=None,
        help="the previous cycle's swath-state, which is what makes a slot continue",
    )
    plan_parser.add_argument("--out", type=Path, default=None, help="write documents here")
    plan_parser.add_argument("--cycle-index", type=int, default=None)
    plan_parser.add_argument(
        "--card", default=None,
        help="a measured card key for the footprint column of the sizing receipt",
    )
    plan_parser.add_argument("--mesh-exe", type=Path, default=None)
    plan_parser.add_argument(
        "--no-size", dest="size", action="store_false", default=True,
        help=(
            "skip the generator dry-run. Every cell figure then carries "
            "basis 'area_integral' and nothing in the plan has been checked "
            "against what a mesh would deliver"
        ),
    )
    plan_parser.set_defaults(handler=_dispatch_plan)

    metrics_parser = legs.add_parser(
        "metrics",
        help="print the armed threat rows and what the coarse run must publish",
    )
    metrics_parser.add_argument("--metrics", type=Path, default=None)
    metrics_parser.add_argument(
        "--publication-manifest", action="store_true",
        help="print only the history variables the armed rows need",
    )
    metrics_parser.set_defaults(handler=run_metrics)

    explain_parser = legs.add_parser(
        "explain", help="why each candidate was placed or declined"
    )
    explain_parser.add_argument("--plan", type=Path, help="a swath-plan.json")
    explain_parser.add_argument("--slot", default=None)
    explain_parser.set_defaults(handler=_dispatch_explain)


def _dispatch_plan(arguments: argparse.Namespace) -> int:
    if arguments.history is None:
        raise MpasPortError(
            "--history was not given. The plan is a pure function of a coarse "
            "forecast and two documents; without the forecast there is nothing to "
            "place a swath from, and a default would name a file that does not exist"
        )
    return run_plan(arguments)


def _dispatch_explain(arguments: argparse.Namespace) -> int:
    if arguments.plan is None:
        raise MpasPortError(
            "--plan was not given. 'swath explain' reads a swath-plan.json written "
            "by 'swath plan --out'"
        )
    return run_explain(arguments)


__all__ = ["add_swath_parser", "run_explain", "run_metrics", "run_plan"]
