"""Strict manifest contract for the observational referee.

Case identities, dates, source locations, arm commands, thresholds, and policy
all live in JSON manifests.  Python contains no hidden weather cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .canonical import canonical_json_bytes, parse_utc, read_json, sha256_bytes
from .errors import SchemaError


SCHEMA = "gpuwm-hex.obs-referee/v1"
BASE_COMMIT = "c14bc1045b0d4516f18004f13265d900ec12563b"

_CASE_ROLES = frozenset(
    ("known_divergence", "independent_control", "weak_convection_control")
)
_ARM_ROLES = frozenset(("candidate", "baseline", "context", "experiment"))
_SOURCE_ADAPTERS = frozenset(
    ("canonical-grid-v1", "canonical-stations-v1", "mpas-netcdf-v1")
)
_METRIC_FAMILIES = frozenset(
    ("continuous", "categorical", "fss", "objects")
)
_DIRECTIONS = frozenset(("higher", "lower"))
_CLAIM_RULES = frozenset(
    ("outside_primary_scope", "arm_delta", "absolute_metric_sign")
)


@dataclass(frozen=True, slots=True)
class Manifest:
    path: Path
    raw: Mapping[str, Any]
    digest: str

    @property
    def directory(self) -> Path:
        return self.path.parent

    @property
    def suite_id(self) -> str:
        return str(self.raw["suite_id"])

    @property
    def mode(self) -> str:
        return str(self.raw["mode"])

    @property
    def cases(self) -> Sequence[Mapping[str, Any]]:
        return self.raw["cases"]

    @property
    def arms(self) -> Sequence[Mapping[str, Any]]:
        return self.raw["arms"]

    @property
    def metrics(self) -> Sequence[Mapping[str, Any]]:
        return self.raw["metrics"]

    @property
    def policy(self) -> Mapping[str, Any]:
        return self.raw["policy"]

    @property
    def bootstrap(self) -> Mapping[str, Any]:
        return self.raw["bootstrap"]

    @property
    def claims(self) -> Sequence[Mapping[str, Any]]:
        return self.raw.get("claims", ())

    def arm_ids(self) -> tuple[str, ...]:
        return tuple(str(item["arm_id"]) for item in self.arms)

    def metric_ids(self) -> tuple[str, ...]:
        return tuple(str(item["metric_id"]) for item in self.metrics)


_TOP_KEYS = frozenset(
    (
        "schema",
        "suite_id",
        "mode",
        "base_commit",
        "description",
        "path_variables",
        "producer_allowlist",
        "cases",
        "arms",
        "metrics",
        "bootstrap",
        "policy",
        "claims",
        "metadata",
    )
)
_REQUIRED_TOP = frozenset(
    (
        "schema",
        "suite_id",
        "mode",
        "base_commit",
        "cases",
        "arms",
        "metrics",
        "bootstrap",
        "policy",
    )
)


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).resolve()
    raw = read_json(manifest_path)
    if not isinstance(raw, dict):
        raise SchemaError("manifest root must be a JSON object")
    _strict_keys(raw, allowed=_TOP_KEYS, required=_REQUIRED_TOP, where="manifest")
    _require_equal(raw["schema"], SCHEMA, "manifest.schema")
    _identifier(raw["suite_id"], "manifest.suite_id")
    if raw["mode"] not in ("production", "synthetic"):
        raise SchemaError("manifest.mode must be 'production' or 'synthetic'")
    if raw["base_commit"] != BASE_COMMIT:
        raise SchemaError(
            f"manifest.base_commit must pin {BASE_COMMIT}; got {raw['base_commit']!r}"
        )
    path_variables = raw.get("path_variables", {})
    if not isinstance(path_variables, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in path_variables.items()
    ):
        raise SchemaError("manifest.path_variables must map strings to strings")
    producers = raw.get("producer_allowlist", ["rustwx", "gpuwm-hex"])
    if not _string_list(producers, nonempty=True):
        raise SchemaError("manifest.producer_allowlist must be a non-empty string list")
    if raw["mode"] == "production" and any(p.startswith("synthetic") for p in producers):
        raise SchemaError("production manifests may not allow synthetic producers")

    _validate_arms(raw["arms"])
    arm_ids = {str(arm["arm_id"]) for arm in raw["arms"]}
    _validate_cases(raw["cases"], arm_ids=arm_ids)
    _validate_metrics(raw["metrics"])
    metric_ids = {str(metric["metric_id"]) for metric in raw["metrics"]}
    _validate_bootstrap(raw["bootstrap"])
    _validate_policy(raw["policy"], arm_ids=arm_ids)
    _validate_claims(raw.get("claims", []), arm_ids=arm_ids, metric_ids=metric_ids)

    immutable = _freeze(raw)
    digest = sha256_bytes(canonical_json_bytes(raw))
    return Manifest(path=manifest_path, raw=immutable, digest=digest)


def _validate_arms(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise SchemaError("manifest.arms must be a non-empty list")
    seen: set[str] = set()
    allowed = frozenset(
        (
            "arm_id",
            "role",
            "description",
            "eligible_for_promotion",
            "treatment",
            "metadata",
        )
    )
    for index, arm in enumerate(value):
        where = f"manifest.arms[{index}]"
        if not isinstance(arm, dict):
            raise SchemaError(f"{where} must be an object")
        _strict_keys(arm, allowed=allowed, required={"arm_id", "role"}, where=where)
        arm_id = _identifier(arm["arm_id"], f"{where}.arm_id")
        if arm_id in seen:
            raise SchemaError(f"duplicate arm_id {arm_id!r}")
        seen.add(arm_id)
        if arm["role"] not in _ARM_ROLES:
            raise SchemaError(f"{where}.role must be one of {sorted(_ARM_ROLES)}")
        eligible = arm.get("eligible_for_promotion", False)
        if not isinstance(eligible, bool):
            raise SchemaError(f"{where}.eligible_for_promotion must be boolean")
        treatment = arm.get("treatment", {"kind": "none"})
        _validate_treatment(treatment, where=f"{where}.treatment")


def _validate_treatment(value: Any, *, where: str) -> None:
    if not isinstance(value, dict):
        raise SchemaError(f"{where} must be an object")
    allowed = frozenset(
        ("kind", "receipt", "expected_name", "expected_mode", "expected_value", "metadata")
    )
    _strict_keys(value, allowed=allowed, required={"kind"}, where=where)
    if value["kind"] not in ("none", "external-receipt-v1"):
        raise SchemaError(f"{where}.kind must be 'none' or 'external-receipt-v1'")
    if value["kind"] == "none":
        extras = set(value) - {"kind", "metadata"}
        if extras:
            raise SchemaError(f"{where}: disabled treatment may not declare {sorted(extras)}")
    else:
        for key in ("receipt", "expected_name", "expected_mode"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise SchemaError(f"{where}.{key} must be a non-empty string")
        expected = value.get("expected_value")
        if isinstance(expected, bool) or not isinstance(expected, (int, float)):
            raise SchemaError(f"{where}.expected_value must be a real number")


def _validate_cases(value: Any, *, arm_ids: set[str]) -> None:
    if not isinstance(value, list) or not value:
        raise SchemaError("manifest.cases must be a non-empty list")
    seen: set[str] = set()
    allowed = frozenset(
        (
            "case_id",
            "role",
            "cycle_time",
            "window_start",
            "window_end",
            "description",
            "selection_status",
            "observations",
            "model_inputs",
            "metadata",
        )
    )
    for index, case in enumerate(value):
        where = f"manifest.cases[{index}]"
        if not isinstance(case, dict):
            raise SchemaError(f"{where} must be an object")
        _strict_keys(
            case,
            allowed=allowed,
            required={
                "case_id",
                "role",
                "cycle_time",
                "window_start",
                "window_end",
                "observations",
                "model_inputs",
            },
            where=where,
        )
        case_id = _identifier(case["case_id"], f"{where}.case_id")
        if case_id in seen:
            raise SchemaError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        if case["role"] not in _CASE_ROLES:
            raise SchemaError(f"{where}.role must be one of {sorted(_CASE_ROLES)}")
        selection_status = case.get("selection_status", "selected")
        if selection_status not in ("selected", "pending"):
            raise SchemaError(f"{where}.selection_status must be 'selected' or 'pending'")
        if selection_status == "selected":
            cycle = parse_utc(case["cycle_time"], name=f"{where}.cycle_time")
            start = parse_utc(case["window_start"], name=f"{where}.window_start")
            end = parse_utc(case["window_end"], name=f"{where}.window_end")
            if not start <= cycle <= end:
                raise SchemaError(f"{where}: cycle_time must fall inside the verification window")
            if end <= start:
                raise SchemaError(f"{where}: window_end must be after window_start")
        else:
            if any(case[name] is not None for name in ("cycle_time", "window_start", "window_end")):
                raise SchemaError(f"{where}: pending cases must use null cycle/window times")
        observations = case["observations"]
        if not isinstance(observations, dict):
            raise SchemaError(f"{where}.observations must be an object")
        for source_name, source in observations.items():
            _identifier(source_name, f"{where}.observations source name")
            _validate_source(source, where=f"{where}.observations.{source_name}")
        model_inputs = case["model_inputs"]
        if not isinstance(model_inputs, dict):
            raise SchemaError(f"{where}.model_inputs must be an object")
        if selection_status == "pending" and (observations or model_inputs):
            raise SchemaError(f"{where}: pending cases must not bind observation/model artifacts")
        unknown = set(model_inputs) - arm_ids
        if unknown:
            raise SchemaError(f"{where}.model_inputs names unknown arms: {sorted(unknown)}")
        for arm_id, source in model_inputs.items():
            _validate_source(source, where=f"{where}.model_inputs.{arm_id}")


def _validate_source(value: Any, *, where: str) -> None:
    if not isinstance(value, dict):
        raise SchemaError(f"{where} must be an object")
    allowed = frozenset(
        (
            "adapter",
            "path",
            "receipt",
            "sha256",
            "optional",
            "producer_command",
            "producer_environment",
            "options",
            "metadata",
        )
    )
    _strict_keys(value, allowed=allowed, required={"adapter", "path"}, where=where)
    if value["adapter"] not in _SOURCE_ADAPTERS:
        raise SchemaError(f"{where}.adapter must be one of {sorted(_SOURCE_ADAPTERS)}")
    if not isinstance(value["path"], str) or not value["path"]:
        raise SchemaError(f"{where}.path must be a non-empty string")
    if "receipt" in value and (
        not isinstance(value["receipt"], str) or not value["receipt"]
    ):
        raise SchemaError(f"{where}.receipt must be a non-empty string")
    if "sha256" in value:
        digest = value["sha256"]
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise SchemaError(f"{where}.sha256 must be null or lowercase SHA-256")
    if not isinstance(value.get("optional", False), bool):
        raise SchemaError(f"{where}.optional must be boolean")
    command = value.get("producer_command")
    if command is not None and not _string_list(command, nonempty=True):
        raise SchemaError(f"{where}.producer_command must be a non-empty string list")
    environment = value.get("producer_environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in environment.items()
    ):
        raise SchemaError(f"{where}.producer_environment must map strings to strings")
    if not isinstance(value.get("options", {}), dict):
        raise SchemaError(f"{where}.options must be an object")


def _validate_metrics(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise SchemaError("manifest.metrics must be a non-empty list")
    seen: set[str] = set()
    allowed = frozenset(
        (
            "metric_id",
            "source",
            "field",
            "family",
            "statistic",
            "direction",
            "threshold",
            "neighborhood_radius_cells",
            "minimum_object_cells",
            "maximum_object_match_km",
            "time_tolerance_seconds",
            "space_tolerance_km",
            "minimum_valid_samples",
            "primary",
            "guardrail",
            "metadata",
        )
    )
    for index, metric in enumerate(value):
        where = f"manifest.metrics[{index}]"
        if not isinstance(metric, dict):
            raise SchemaError(f"{where} must be an object")
        _strict_keys(
            metric,
            allowed=allowed,
            required={
                "metric_id",
                "source",
                "field",
                "family",
                "statistic",
                "direction",
            },
            where=where,
        )
        metric_id = _identifier(metric["metric_id"], f"{where}.metric_id")
        if metric_id in seen:
            raise SchemaError(f"duplicate metric_id {metric_id!r}")
        seen.add(metric_id)
        _identifier(metric["source"], f"{where}.source")
        _identifier(metric["field"], f"{where}.field")
        if metric["family"] not in _METRIC_FAMILIES:
            raise SchemaError(f"{where}.family must be one of {sorted(_METRIC_FAMILIES)}")
        if metric["direction"] not in _DIRECTIONS:
            raise SchemaError(f"{where}.direction must be one of {sorted(_DIRECTIONS)}")
        if metric["family"] in ("categorical", "fss", "objects"):
            _finite_real(metric.get("threshold"), f"{where}.threshold")
        if metric["family"] == "fss":
            _positive_int(
                metric.get("neighborhood_radius_cells"),
                f"{where}.neighborhood_radius_cells",
                allow_zero=True,
            )
        if metric["family"] == "objects":
            _positive_int(
                metric.get("minimum_object_cells", 1),
                f"{where}.minimum_object_cells",
            )
            _positive_real(
                metric.get("maximum_object_match_km", 100.0),
                f"{where}.maximum_object_match_km",
            )
        _positive_int(
            metric.get("time_tolerance_seconds", 900),
            f"{where}.time_tolerance_seconds",
            allow_zero=True,
        )
        _positive_real(
            metric.get("space_tolerance_km", 25.0),
            f"{where}.space_tolerance_km",
            allow_zero=True,
        )
        _positive_int(
            metric.get("minimum_valid_samples", 1),
            f"{where}.minimum_valid_samples",
        )
        if not isinstance(metric.get("primary", True), bool):
            raise SchemaError(f"{where}.primary must be boolean")
        if not isinstance(metric.get("guardrail", False), bool):
            raise SchemaError(f"{where}.guardrail must be boolean")


def _validate_bootstrap(value: Any) -> None:
    if not isinstance(value, dict):
        raise SchemaError("manifest.bootstrap must be an object")
    _strict_keys(
        value,
        allowed={"replicates", "confidence", "minimum_cases", "seed"},
        required={"replicates", "confidence", "minimum_cases", "seed"},
        where="manifest.bootstrap",
    )
    _positive_int(value["replicates"], "manifest.bootstrap.replicates")
    confidence = _finite_real(value["confidence"], "manifest.bootstrap.confidence")
    if not 0.0 < confidence < 1.0:
        raise SchemaError("manifest.bootstrap.confidence must be in (0, 1)")
    _positive_int(value["minimum_cases"], "manifest.bootstrap.minimum_cases")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise SchemaError("manifest.bootstrap.seed must be an integer")


def _validate_policy(value: Any, *, arm_ids: set[str]) -> None:
    if not isinstance(value, dict):
        raise SchemaError("manifest.policy must be an object")
    allowed = frozenset(
        (
            "candidate_arm",
            "reference_arm",
            "required_case_roles",
            "owner_acceptance_required",
            "allow_automatic_default",
            "guardrail_tolerance",
            "metadata",
        )
    )
    _strict_keys(
        value,
        allowed=allowed,
        required={
            "candidate_arm",
            "reference_arm",
            "required_case_roles",
            "owner_acceptance_required",
            "allow_automatic_default",
        },
        where="manifest.policy",
    )
    for key in ("candidate_arm", "reference_arm"):
        if value[key] not in arm_ids:
            raise SchemaError(f"manifest.policy.{key} names unknown arm {value[key]!r}")
    if value["candidate_arm"] == value["reference_arm"]:
        raise SchemaError("candidate_arm and reference_arm must differ")
    roles = value["required_case_roles"]
    if not _string_list(roles, nonempty=True) or not set(roles) <= _CASE_ROLES:
        raise SchemaError(
            f"manifest.policy.required_case_roles must use {sorted(_CASE_ROLES)}"
        )
    if not isinstance(value["owner_acceptance_required"], bool):
        raise SchemaError("manifest.policy.owner_acceptance_required must be boolean")
    if not value["owner_acceptance_required"]:
        raise SchemaError("lane 283 requires explicit owner acceptance before default-on")
    if value["allow_automatic_default"] is not False:
        raise SchemaError("lane 283 forbids automatic default promotion")
    tolerance = _finite_real(
        value.get("guardrail_tolerance", 0.0),
        "manifest.policy.guardrail_tolerance",
    )
    if tolerance < 0.0:
        raise SchemaError("manifest.policy.guardrail_tolerance cannot be negative")


def _validate_claims(value: Any, *, arm_ids: set[str], metric_ids: set[str]) -> None:
    if not isinstance(value, list):
        raise SchemaError("manifest.claims must be a list")
    seen: set[str] = set()
    allowed = frozenset(
        (
            "claim_id",
            "description",
            "rule",
            "metric_id",
            "arm",
            "candidate_arm",
            "reference_arm",
            "expected_sign",
            "reason",
            "metadata",
        )
    )
    for index, claim in enumerate(value):
        where = f"manifest.claims[{index}]"
        if not isinstance(claim, dict):
            raise SchemaError(f"{where} must be an object")
        _strict_keys(claim, allowed=allowed, required={"claim_id", "rule"}, where=where)
        claim_id = _identifier(claim["claim_id"], f"{where}.claim_id")
        if claim_id in seen:
            raise SchemaError(f"duplicate claim_id {claim_id!r}")
        seen.add(claim_id)
        if claim["rule"] not in _CLAIM_RULES:
            raise SchemaError(f"{where}.rule must be one of {sorted(_CLAIM_RULES)}")
        if claim["rule"] == "outside_primary_scope":
            if not isinstance(claim.get("reason"), str) or not claim["reason"]:
                raise SchemaError(f"{where}.reason must explain the scope limitation")
            continue
        metric_id = claim.get("metric_id")
        if metric_id not in metric_ids:
            raise SchemaError(f"{where}.metric_id names unknown metric {metric_id!r}")
        if claim["rule"] == "arm_delta":
            for key in ("candidate_arm", "reference_arm"):
                if claim.get(key) not in arm_ids:
                    raise SchemaError(f"{where}.{key} names unknown arm {claim.get(key)!r}")
        else:
            if claim.get("arm") not in arm_ids:
                raise SchemaError(f"{where}.arm names unknown arm {claim.get('arm')!r}")
            if claim.get("expected_sign") not in ("negative", "positive", "zero"):
                raise SchemaError(
                    f"{where}.expected_sign must be negative, positive, or zero"
                )


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name} must be a non-empty string")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-_.")
    if value.lower() != value or any(ch not in allowed for ch in value):
        raise SchemaError(
            f"{name} must use lowercase ASCII letters, digits, '-', '_', or '.': {value!r}"
        )
    return value


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str],
    where: str,
) -> None:
    missing = set(required) - set(value)
    extra = set(value) - set(allowed)
    if missing or extra:
        raise SchemaError(f"{where} keys invalid: missing={sorted(missing)}, extra={sorted(extra)}")


def _string_list(value: Any, *, nonempty: bool) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(item, str) and bool(item) for item in value)
    )


def _finite_real(value: Any, name: str) -> float:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{name} must be finite")
    return result


def _positive_real(value: Any, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_real(value, name)
    if result < 0.0 if allow_zero else result <= 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SchemaError(f"{name} must be {qualifier}")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{name} must be an integer")
    if value < 0 if allow_zero else value <= 0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise SchemaError(f"{name} must be {qualifier}")
    return value


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise SchemaError(f"{name} must equal {expected!r}; got {actual!r}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
