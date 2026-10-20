#!/usr/bin/env python3
"""Reject forecast graph evidence that contains any eager subcycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


class GraphReplayEvidenceError(ValueError):
    """A completed forecast did not prove the required graph treatment."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not value:
        raise GraphReplayEvidenceError(f"{name} must be a nonempty object")
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise GraphReplayEvidenceError(f"{name} must be a nonnegative integer")
    return value


def validate_graph_replay_receipt(document: Mapping[str, Any]) -> dict[str, Any]:
    document = _mapping(document, "receipt")
    if document.get("status") != "passed":
        raise GraphReplayEvidenceError("forecast status must be passed")
    forecast = _mapping(document.get("forecast"), "forecast")
    requested = _count(forecast.get("steps_requested"), "steps_requested")
    executed = _count(forecast.get("steps_executed"), "steps_executed")
    if requested < 2 or requested != executed:
        raise GraphReplayEvidenceError("at least two requested steps must all be committed")
    if forecast.get("refusal"):
        raise GraphReplayEvidenceError("a completed graph proof cannot contain a refusal")
    if "step_receipt_count" in forecast:
        if _count(forecast["step_receipt_count"], "step_receipt_count") != executed:
            raise GraphReplayEvidenceError("step receipt count differs from committed steps")
    schedule = forecast.get("schedule")
    if isinstance(schedule, dict) and schedule.get("steps", requested) != requested:
        raise GraphReplayEvidenceError("schedule differs from requested steps")

    graph = _mapping(forecast.get("cuda_graphs"), "forecast.cuda_graphs")
    subcycles = _mapping(graph.get("subcycles"), "cuda_graphs.subcycles")
    captured = _count(subcycles.get("captured_subcycles"), "captured_subcycles")
    replayed = _count(subcycles.get("replayed_subcycles"), "replayed_subcycles")
    eager = _count(subcycles.get("eager_subcycles"), "eager_subcycles")
    if captured < 1 or captured > replayed:
        raise GraphReplayEvidenceError("at least one captured graph must have replayed")
    if replayed != executed * 3 or eager != 0:
        raise GraphReplayEvidenceError("every subcycle, including the first, must replay")
    reasons = subcycles.get("fallback_reasons")
    if not isinstance(reasons, dict) or reasons or subcycles.get("fallbacks"):
        raise GraphReplayEvidenceError("a complete graph proof cannot contain fallback")
    fully = _count(graph.get("fully_replayed_committed_steps"),
                   "fully_replayed_committed_steps")
    partly = _count(graph.get("partly_replayed_committed_steps"),
                    "partly_replayed_committed_steps")
    if fully != executed or partly != 0:
        raise GraphReplayEvidenceError("every committed step must replay all three subcycles")
    return {"status": "passed", "steps": executed, "captured_subcycles": captured,
            "replayed_subcycles": replayed, "eager_subcycles": eager,
            "fully_replayed_committed_steps": fully,
            "partly_replayed_committed_steps": partly}


def _require_hashes(value: Any, name: str) -> None:
    hashes: list[Any] = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "sha256":
                    hashes.append(child)
                else:
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    if not hashes or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item)
                         for item in hashes):
        raise GraphReplayEvidenceError(f"{name} lacks complete SHA-256 evidence")


def validate_identity_pair(base_snapshots: Mapping[str, Any], snapshots: Mapping[str, Any],
                           base_identity: Mapping[str, Any], identity: Mapping[str, Any],
                           *, steps: int) -> dict[str, Any]:
    for name, item in (("base snapshots", base_snapshots), ("candidate snapshots", snapshots)):
        item = _mapping(item, name)
        projection = _mapping(item.get("projection"), f"{name}.projection")
        if "q2" not in item or "0" not in projection or str(steps) not in projection:
            raise GraphReplayEvidenceError(f"{name} lacks initial or final history hashes")
        for frame, arrays in projection.items():
            arrays = _mapping(arrays, f"{name}.projection.{frame}")
            if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in arrays.values()):
                raise GraphReplayEvidenceError(f"{name} has an invalid array SHA-256")
    if base_snapshots != snapshots:
        raise GraphReplayEvidenceError("full history array hash maps differ")
    for name, item in (("base final identity", base_identity), ("candidate final identity", identity)):
        item = _mapping(item, name)
        if type(item.get("rc")) is not int or item["rc"] != 0 or item.get("complete_run_identity_eligible") is not True:
            raise GraphReplayEvidenceError(f"{name} is not eligible complete-run identity")
        if item.get("completed_composite_steps") != steps:
            raise GraphReplayEvidenceError(f"{name} does not name every committed step")
        boundary = _mapping(item.get("execution_boundary"), f"{name}.execution_boundary")
        for group in ("atmosphere", "backend"):
            _require_hashes(_mapping(boundary.get(group), f"{name}.{group}"), f"{name}.{group}")
        atmosphere = boundary["atmosphere"]
        required = {
            "state": {"rho", "rho_theta", "rho_u", "rho_w", "scalars"},
            "saved_diagnostics": {"theta_m", "exner", "density_perturbation",
                                  "rho_theta_perturbation", "pressure_perturbation",
                                  "normal_velocity", "vertical_velocity"},
        }
        for group, names in required.items():
            fields = _mapping(_mapping(atmosphere.get(group), f"{name}.{group}").get("fields"),
                              f"{name}.{group}.fields")
            if not names.issubset(fields):
                raise GraphReplayEvidenceError(f"{name}.{group} lacks required mutable fields")
            for field in fields.values():
                _require_hashes(field, f"{name}.{group} field")
        _mapping(boundary["backend"].get("arrays"), f"{name}.backend.arrays")
        forcing = _mapping(item.get("dynamics_tendencies"), f"{name}.dynamics_tendencies")
        _require_hashes(forcing, f"{name}.dynamics_tendencies")
        arrays = _mapping(forcing.get("arrays"), f"{name}.dynamics_tendencies.arrays")
        if not {"rthdynten", "rqvdynten"}.issubset(arrays):
            raise GraphReplayEvidenceError(f"{name} lacks both dynamics forcing carriers")
    for key in ("execution_boundary", "dynamics_tendencies"):
        if base_identity[key] != identity[key]:
            raise GraphReplayEvidenceError(f"complete final {key} hash maps differ")
    return {"history_hash_maps_identical": True, "final_execution_boundary_identical": True,
            "final_dynamics_tendencies_identical": True}


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    for name in ("base-snapshots", "snapshots", "base-identity", "identity"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    paths = [args.base_snapshots, args.snapshots, args.base_identity, args.identity]
    if any(paths) and not all(paths):
        parser.error("identity verification requires all four paired evidence paths")
    try:
        result = validate_graph_replay_receipt(read_json(args.receipt))
        if all(paths):
            result.update(validate_identity_pair(*(read_json(path) for path in paths),
                                                 steps=result["steps"]))
    except (OSError, ValueError, TypeError) as error:
        print(f"graph replay evidence refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
