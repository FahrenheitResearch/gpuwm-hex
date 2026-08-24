"""Paired scorecard, claim classification, and no-auto-promotion policy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .bootstrap import paired_case_interval, raw_case_interval
from .manifest import Manifest


def build_scorecard(
    manifest: Manifest,
    case_metrics: Sequence[Mapping[str, Any]],
    *,
    evidence_class: str,
) -> dict[str, Any]:
    metric_by_id = {str(metric["metric_id"]): metric for metric in manifest.metrics}
    policy = manifest.policy
    candidate = str(policy["candidate_arm"])
    reference = str(policy["reference_arm"])
    bootstrap = manifest.bootstrap
    required_roles = set(policy["required_case_roles"])
    role_by_case = {str(case["case_id"]): str(case["role"]) for case in manifest.cases}

    values: dict[tuple[str, str], dict[str, float]] = {}
    for item in case_metrics:
        if item.get("status") != "MEASURED" or item.get("value") is None:
            continue
        key = (str(item["arm_id"]), str(item["metric_id"]))
        values.setdefault(key, {})[str(item["case_id"])] = float(item["value"])

    comparisons: list[dict[str, Any]] = []
    for metric_id in sorted(metric_by_id):
        metric = metric_by_id[metric_id]
        candidate_values = values.get((candidate, metric_id), {})
        reference_values = values.get((reference, metric_id), {})
        eligible_cases = sorted(
            case_id
            for case_id in set(candidate_values) & set(reference_values)
            if role_by_case.get(case_id) in required_roles
        )
        interval = paired_case_interval(
            {case_id: candidate_values[case_id] for case_id in eligible_cases},
            {case_id: reference_values[case_id] for case_id in eligible_cases},
            direction=str(metric["direction"]),
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            minimum_cases=int(bootstrap["minimum_cases"]),
            seed=int(bootstrap["seed"]),
            seed_context=(manifest.suite_id, metric_id, candidate, reference),
        )
        if interval.status != "MEASURED":
            verdict = "NOT_MEASURED"
        elif interval.lower is not None and interval.lower > 0.0:
            verdict = "FAVORS_CANDIDATE"
        elif interval.upper is not None and interval.upper < 0.0:
            verdict = "FAVORS_REFERENCE"
        else:
            verdict = "INDISTINGUISHABLE"
        comparisons.append(
            {
                "metric_id": metric_id,
                "candidate_arm": candidate,
                "reference_arm": reference,
                "direction": str(metric["direction"]),
                "primary": bool(metric.get("primary", True)),
                "guardrail": bool(metric.get("guardrail", False)),
                "paired_case_ids": eligible_cases,
                "improvement_interval": interval.as_dict(),
                "verdict": verdict,
            }
        )

    scientific_verdict, scientific_reason = _overall_verdict(
        comparisons,
        evidence_class=evidence_class,
        guardrail_tolerance=float(policy.get("guardrail_tolerance", 0.0)),
    )
    claims = [
        _evaluate_claim(
            manifest,
            claim,
            values=values,
            comparisons=comparisons,
            role_by_case=role_by_case,
        )
        for claim in manifest.claims
    ]
    return {
        "schema": "gpuwm-hex.obs-referee-scorecard/v1",
        "suite_id": manifest.suite_id,
        "manifest_sha256": manifest.digest,
        "evidence_class": evidence_class,
        "candidate_arm": candidate,
        "reference_arm": reference,
        "metric_comparisons": comparisons,
        "claim_assessments": claims,
        "scientific_verdict": scientific_verdict,
        "scientific_reason": scientific_reason,
        "default_on_decision": "DO_NOT_ENABLE",
        "default_on_reason": (
            "This lane forbids automatic default promotion. Real measured evidence, "
            "all guardrails, and explicit owner acceptance are required."
        ),
        "owner_acceptance_required": True,
        "owner_acceptance_recorded": False,
    }


def _overall_verdict(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    evidence_class: str,
    guardrail_tolerance: float,
) -> tuple[str, str]:
    if evidence_class == "synthetic":
        return (
            "SYNTHETIC_ONLY",
            "Synthetic fixtures validate machinery, not atmospheric skill.",
        )
    primary = [item for item in comparisons if item["primary"]]
    if not primary or any(item["verdict"] == "NOT_MEASURED" for item in primary):
        return (
            "NOT_MEASURED",
            "At least one required primary comparison lacks enough paired real cases.",
        )
    for item in comparisons:
        if not item["guardrail"]:
            continue
        interval = item["improvement_interval"]
        lower = interval.get("lower")
        if lower is not None and lower < -guardrail_tolerance:
            return (
                "DISFAVORED",
                f"Guardrail {item['metric_id']} exceeds the allowed adverse tolerance.",
            )
    if any(item["verdict"] == "FAVORS_REFERENCE" for item in primary):
        return (
            "DISFAVORED",
            "At least one primary metric confidently favors the reference arm.",
        )
    if all(item["verdict"] == "FAVORS_CANDIDATE" for item in primary):
        return (
            "SUPPORTED",
            "Every primary metric confidently favors the candidate on required cases.",
        )
    return (
        "INDISTINGUISHABLE",
        "The paired case-block intervals do not separate candidate and reference.",
    )


def _evaluate_claim(
    manifest: Manifest,
    claim: Mapping[str, Any],
    *,
    values: Mapping[tuple[str, str], dict[str, float]],
    comparisons: Sequence[Mapping[str, Any]],
    role_by_case: Mapping[str, str],
) -> dict[str, Any]:
    claim_id = str(claim["claim_id"])
    rule = str(claim["rule"])
    if rule == "outside_primary_scope":
        return {
            "claim_id": claim_id,
            "status": "UNRESOLVED",
            "reason": str(claim["reason"]),
            "evidence": None,
        }
    metric_id = str(claim["metric_id"])
    bootstrap = manifest.bootstrap
    required_roles = set(manifest.policy["required_case_roles"])
    if rule == "arm_delta":
        candidate = str(claim["candidate_arm"])
        reference = str(claim["reference_arm"])
        metric = next(item for item in manifest.metrics if item["metric_id"] == metric_id)
        candidate_values = values.get((candidate, metric_id), {})
        reference_values = values.get((reference, metric_id), {})
        cases = sorted(
            case_id
            for case_id in set(candidate_values) & set(reference_values)
            if role_by_case.get(case_id) in required_roles
        )
        interval = paired_case_interval(
            {case_id: candidate_values[case_id] for case_id in cases},
            {case_id: reference_values[case_id] for case_id in cases},
            direction=str(metric["direction"]),
            replicates=int(bootstrap["replicates"]),
            confidence=float(bootstrap["confidence"]),
            minimum_cases=int(bootstrap["minimum_cases"]),
            seed=int(bootstrap["seed"]),
            seed_context=(manifest.suite_id, "claim", claim_id),
        )
        if interval.status != "MEASURED":
            status = "NOT_MEASURED"
            reason = interval.reason
        elif interval.lower is not None and interval.lower > 0.0:
            status = "SUPPORTED"
            reason = "Paired improvement interval is entirely favorable."
        elif interval.upper is not None and interval.upper < 0.0:
            status = "DISFAVORED"
            reason = "Paired improvement interval is entirely adverse."
        else:
            status = "INDISTINGUISHABLE"
            reason = "Paired improvement interval spans no effect."
        return {
            "claim_id": claim_id,
            "status": status,
            "reason": reason,
            "evidence": {
                "rule": rule,
                "metric_id": metric_id,
                "candidate_arm": candidate,
                "reference_arm": reference,
                "interval": interval.as_dict(),
            },
        }

    arm = str(claim["arm"])
    raw_values = values.get((arm, metric_id), {})
    eligible = {
        case_id: value
        for case_id, value in raw_values.items()
        if role_by_case.get(case_id) in required_roles
    }
    interval = raw_case_interval(
        eligible,
        replicates=int(bootstrap["replicates"]),
        confidence=float(bootstrap["confidence"]),
        minimum_cases=int(bootstrap["minimum_cases"]),
        seed=int(bootstrap["seed"]),
        seed_context=(manifest.suite_id, "claim", claim_id),
    )
    expected = str(claim["expected_sign"])
    if interval.status != "MEASURED":
        status = "NOT_MEASURED"
        reason = interval.reason
    elif expected == "negative":
        if interval.upper is not None and interval.upper < 0.0:
            status, reason = "SUPPORTED", "Metric interval is entirely negative."
        elif interval.lower is not None and interval.lower > 0.0:
            status, reason = "DISFAVORED", "Metric interval is entirely positive."
        else:
            status, reason = "INDISTINGUISHABLE", "Metric interval spans zero."
    elif expected == "positive":
        if interval.lower is not None and interval.lower > 0.0:
            status, reason = "SUPPORTED", "Metric interval is entirely positive."
        elif interval.upper is not None and interval.upper < 0.0:
            status, reason = "DISFAVORED", "Metric interval is entirely negative."
        else:
            status, reason = "INDISTINGUISHABLE", "Metric interval spans zero."
    else:
        if interval.lower is not None and interval.upper is not None and interval.lower <= 0.0 <= interval.upper:
            status, reason = "SUPPORTED", "Metric interval includes zero."
        else:
            status, reason = "DISFAVORED", "Metric interval excludes zero."
    return {
        "claim_id": claim_id,
        "status": status,
        "reason": reason,
        "evidence": {
            "rule": rule,
            "metric_id": metric_id,
            "arm": arm,
            "expected_sign": expected,
            "interval": interval.as_dict(),
        },
    }
