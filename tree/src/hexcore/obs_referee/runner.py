"""End-to-end manifest runner for the observational referee."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .align import align_grid_field, align_station_field
from .bundle import GridBundle, StationBundle, load_source
from .canonical import resolve_path, sha256_file
from .errors import MeasurementUnavailable
from .manifest import BASE_COMMIT, Manifest, load_manifest
from .metrics import MetricResult, calculate_metric
from .report import write_evidence_directory
from .scorecard import build_scorecard
from .treatment import validate_treatment_receipt


def run_suite(
    manifest_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    arms = {str(arm["arm_id"]): arm for arm in manifest.arms}
    case_metrics: list[dict[str, Any]] = []
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    loaded_sources: dict[tuple[str, str, str], GridBundle | StationBundle] = {}

    for case in sorted(manifest.cases, key=lambda item: str(item["case_id"])):
        case_id = str(case["case_id"])
        case_role = str(case["role"])
        if str(case.get("selection_status", "selected")) == "pending":
            for arm_id in sorted(arms):
                for metric in manifest.metrics:
                    case_metrics.append(
                        _not_measured(
                            case_id,
                            case_role,
                            arm_id,
                            str(metric["metric_id"]),
                            "case identity/date is intentionally pending in the manifest",
                        )
                    )
            continue
        for arm_id in sorted(arms):
            model_spec = case["model_inputs"].get(arm_id)
            if model_spec is None:
                for metric in manifest.metrics:
                    case_metrics.append(
                        _not_measured(
                            case_id,
                            case_role,
                            arm_id,
                            str(metric["metric_id"]),
                            "case manifest supplies no model input for this arm",
                        )
                    )
                continue
            try:
                model = _load_cached(
                    manifest,
                    model_spec,
                    loaded_sources,
                    cache_key=(case_id, arm_id, "model"),
                    expected_kind="grid",
                )
                assert isinstance(model, GridBundle)
                _record_artifact(
                    artifacts,
                    logical_id=f"case/{case_id}/model/{arm_id}",
                    bundle=model,
                )
                _validate_case_treatment(
                    manifest,
                    case_id=case_id,
                    arm_id=arm_id,
                    arm=arms[arm_id],
                )
            except MeasurementUnavailable as exc:
                for metric in manifest.metrics:
                    case_metrics.append(
                        _not_measured(
                            case_id,
                            case_role,
                            arm_id,
                            str(metric["metric_id"]),
                            str(exc),
                        )
                    )
                continue

            for metric in sorted(manifest.metrics, key=lambda item: str(item["metric_id"])):
                metric_id = str(metric["metric_id"])
                source_name = str(metric["source"])
                observation_spec = case["observations"].get(source_name)
                if observation_spec is None:
                    case_metrics.append(
                        _not_measured(
                            case_id,
                            case_role,
                            arm_id,
                            metric_id,
                            f"case manifest has no observation source {source_name!r}",
                        )
                    )
                    continue
                expected_kind = (
                    "stations"
                    if str(observation_spec["adapter"]) == "canonical-stations-v1"
                    else "grid"
                )
                try:
                    observation = _load_cached(
                        manifest,
                        observation_spec,
                        loaded_sources,
                        cache_key=(case_id, source_name, "observation"),
                        expected_kind=expected_kind,
                    )
                    _record_artifact(
                        artifacts,
                        logical_id=f"case/{case_id}/observation/{source_name}",
                        bundle=observation,
                    )
                    aligned = _align(
                        model,
                        observation,
                        field=str(metric["field"]),
                        time_tolerance_seconds=int(
                            metric.get("time_tolerance_seconds", 900)
                        ),
                        space_tolerance_km=float(
                            metric.get("space_tolerance_km", 25.0)
                        ),
                    )
                    result = calculate_metric(
                        aligned.forecast,
                        aligned.observation,
                        family=str(metric["family"]),
                        statistic=str(metric["statistic"]),
                        threshold=(
                            float(metric["threshold"])
                            if "threshold" in metric
                            else None
                        ),
                        neighborhood_radius_cells=(
                            int(metric["neighborhood_radius_cells"])
                            if "neighborhood_radius_cells" in metric
                            else None
                        ),
                        minimum_object_cells=int(
                            metric.get("minimum_object_cells", 1)
                        ),
                        maximum_object_match_km=float(
                            metric.get("maximum_object_match_km", 100.0)
                        ),
                        latitude_deg=aligned.latitude_deg,
                        longitude_deg=aligned.longitude_deg,
                        minimum_valid_samples=int(
                            metric.get("minimum_valid_samples", 1)
                        ),
                    )
                    case_metrics.append(
                        _metric_record(
                            case_id,
                            case_role,
                            arm_id,
                            metric_id,
                            result,
                            alignment=aligned.audit,
                        )
                    )
                except MeasurementUnavailable as exc:
                    case_metrics.append(
                        _not_measured(
                            case_id,
                            case_role,
                            arm_id,
                            metric_id,
                            str(exc),
                        )
                    )

    case_metrics.sort(
        key=lambda item: (
            str(item["case_id"]),
            str(item["arm_id"]),
            str(item["metric_id"]),
        )
    )
    evidence_class = "synthetic" if manifest.mode == "synthetic" else "production"
    scorecard = build_scorecard(
        manifest,
        case_metrics,
        evidence_class=evidence_class,
    )
    measured = sum(item["status"] == "MEASURED" for item in case_metrics)
    status = (
        "NOT_MEASURED"
        if measured == 0
        else "MEASURED"
        if measured == len(case_metrics)
        else "PARTIALLY_MEASURED"
    )
    run_receipt = {
        "schema": "gpuwm-hex.obs-referee-run/v1",
        "suite_id": manifest.suite_id,
        "manifest_sha256": manifest.digest,
        "base_commit": BASE_COMMIT,
        "evidence_class": evidence_class,
        "status": status,
        "not_measured_reason": None,
        "case_count": len(manifest.cases),
        "arm_count": len(manifest.arms),
        "metric_count": len(manifest.metrics),
        "case_metric_records": len(case_metrics),
        "measured_case_metric_records": measured,
        "input_artifacts": [
            artifacts[key] for key in sorted(artifacts)
        ],
        "determinism_contract": {
            "wall_clock_in_outputs": False,
            "case_block_bootstrap": True,
            "raw_observation_parser_in_package": False,
            "automatic_default_promotion": False,
        },
    }
    inventory = write_evidence_directory(
        output_directory,
        run_receipt=run_receipt,
        case_metrics=case_metrics,
        scorecard=scorecard,
    )
    return {
        "run_receipt": run_receipt,
        "scorecard": scorecard,
        "output_inventory": inventory,
    }


def emit_not_measured(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """Create an explicit unrun/absent-data record without fabricated metrics."""

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("NOT MEASURED reason must be non-empty")
    manifest = load_manifest(manifest_path)
    case_metrics = [
        _not_measured(
            str(case["case_id"]),
            str(case["role"]),
            str(arm["arm_id"]),
            str(metric["metric_id"]),
            reason.strip(),
        )
        for case in sorted(manifest.cases, key=lambda item: str(item["case_id"]))
        for arm in sorted(manifest.arms, key=lambda item: str(item["arm_id"]))
        for metric in sorted(manifest.metrics, key=lambda item: str(item["metric_id"]))
    ]
    scorecard = build_scorecard(
        manifest,
        case_metrics,
        evidence_class="synthetic" if manifest.mode == "synthetic" else "production",
    )
    if manifest.mode == "production":
        scorecard["scientific_verdict"] = "NOT_MEASURED"
        scorecard["scientific_reason"] = reason.strip()
    run_receipt = {
        "schema": "gpuwm-hex.obs-referee-run/v1",
        "suite_id": manifest.suite_id,
        "manifest_sha256": manifest.digest,
        "base_commit": BASE_COMMIT,
        "evidence_class": "synthetic" if manifest.mode == "synthetic" else "production",
        "status": "NOT_MEASURED",
        "not_measured_reason": reason.strip(),
        "case_count": len(manifest.cases),
        "arm_count": len(manifest.arms),
        "metric_count": len(manifest.metrics),
        "case_metric_records": len(case_metrics),
        "measured_case_metric_records": 0,
        "input_artifacts": [],
        "determinism_contract": {
            "wall_clock_in_outputs": False,
            "case_block_bootstrap": True,
            "raw_observation_parser_in_package": False,
            "automatic_default_promotion": False,
        },
    }
    inventory = write_evidence_directory(
        output_directory,
        run_receipt=run_receipt,
        case_metrics=case_metrics,
        scorecard=scorecard,
    )
    return {
        "run_receipt": run_receipt,
        "scorecard": scorecard,
        "output_inventory": inventory,
    }


def _align(
    model: GridBundle,
    observation: GridBundle | StationBundle,
    *,
    field: str,
    time_tolerance_seconds: int,
    space_tolerance_km: float,
):
    if isinstance(observation, StationBundle):
        return align_station_field(
            model,
            observation,
            field=field,
            time_tolerance_seconds=time_tolerance_seconds,
            space_tolerance_km=space_tolerance_km,
        )
    return align_grid_field(
        model,
        observation,
        field=field,
        time_tolerance_seconds=time_tolerance_seconds,
        space_tolerance_km=space_tolerance_km,
    )


def _load_cached(
    manifest: Manifest,
    source: Mapping[str, Any],
    cache: dict[tuple[str, str, str], GridBundle | StationBundle],
    *,
    cache_key: tuple[str, str, str],
    expected_kind: str,
) -> GridBundle | StationBundle:
    if cache_key not in cache:
        cache[cache_key] = load_source(
            manifest,
            source,
            expected_kind=expected_kind,
        )
    return cache[cache_key]


def _record_artifact(
    artifacts: dict[tuple[str, str], dict[str, Any]],
    *,
    logical_id: str,
    bundle: GridBundle | StationBundle,
) -> None:
    key = (logical_id, bundle.artifact_sha256)
    artifacts[key] = {
        "logical_id": logical_id,
        "artifact_name": bundle.artifact_path.name,
        "sha256": bundle.artifact_sha256,
        "producer": bundle.producer,
        "producer_version": bundle.producer_version,
    }


def _validate_case_treatment(
    manifest: Manifest,
    *,
    case_id: str,
    arm_id: str,
    arm: Mapping[str, Any],
) -> None:
    treatment = arm.get("treatment", {"kind": "none"})
    if treatment["kind"] == "none":
        return
    raw_receipt = str(treatment["receipt"]).format(
        case_id=case_id,
        arm_id=arm_id,
    )
    path = resolve_path(
        raw_receipt,
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    if not path.exists():
        raise MeasurementUnavailable(
            f"experimental arm is unscorable because treatment receipt is absent: {raw_receipt}"
        )
    validate_treatment_receipt(
        path,
        expected_name=str(treatment["expected_name"]),
        expected_mode=str(treatment["expected_mode"]),
        expected_value=float(treatment["expected_value"]),
    )


def _metric_record(
    case_id: str,
    case_role: str,
    arm_id: str,
    metric_id: str,
    result: MetricResult,
    *,
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_role": case_role,
        "arm_id": arm_id,
        "metric_id": metric_id,
        "status": result.status,
        "value": result.value,
        "n_valid": result.n_valid,
        "components": result.components,
        "alignment": dict(alignment),
        "reason": result.reason,
    }


def _not_measured(
    case_id: str,
    case_role: str,
    arm_id: str,
    metric_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_role": case_role,
        "arm_id": arm_id,
        "metric_id": metric_id,
        "status": "NOT_MEASURED",
        "value": None,
        "n_valid": 0,
        "components": {},
        "alignment": {},
        "reason": reason,
    }
