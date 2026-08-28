"""Case-block uncertainty for paired verification comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .canonical import stable_seed
from .errors import SchemaError


@dataclass(frozen=True, slots=True)
class Interval:
    status: str
    estimate: float | None
    lower: float | None
    upper: float | None
    n_cases: int
    confidence: float
    replicates: int
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "n_cases": self.n_cases,
            "confidence": self.confidence,
            "replicates": self.replicates,
            "reason": self.reason,
        }


def paired_case_interval(
    candidate_by_case: dict[str, float],
    reference_by_case: dict[str, float],
    *,
    direction: str,
    replicates: int,
    confidence: float,
    minimum_cases: int,
    seed: int,
    seed_context: Iterable[object] = (),
) -> Interval:
    """Bootstrap the mean paired improvement by resampling whole cases.

    Positive values always mean the candidate is better. Spatial pixels and
    times within a case are never treated as independent bootstrap samples.
    """

    if direction not in ("higher", "lower"):
        raise SchemaError("direction must be 'higher' or 'lower'")
    paired_ids = sorted(set(candidate_by_case) & set(reference_by_case))
    differences = []
    for case_id in paired_ids:
        candidate = float(candidate_by_case[case_id])
        reference = float(reference_by_case[case_id])
        if not np.isfinite(candidate) or not np.isfinite(reference):
            continue
        raw = candidate - reference
        differences.append(raw if direction == "higher" else -raw)
    values = np.asarray(differences, dtype=np.float64)
    if values.size < minimum_cases:
        return Interval(
            status="NOT_MEASURED",
            estimate=None,
            lower=None,
            upper=None,
            n_cases=int(values.size),
            confidence=float(confidence),
            replicates=int(replicates),
            reason=(
                f"{values.size} paired cases is below minimum {minimum_cases}; "
                "no uncertainty claim is permitted"
            ),
        )
    rng = np.random.default_rng(
        stable_seed(seed, "paired-case-bootstrap", *tuple(seed_context))
    )
    indices = rng.integers(0, values.size, size=(replicates, values.size), endpoint=False)
    bootstrap_means = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return Interval(
        status="MEASURED",
        estimate=float(np.mean(values)),
        lower=float(lower),
        upper=float(upper),
        n_cases=int(values.size),
        confidence=float(confidence),
        replicates=int(replicates),
    )


def raw_case_interval(
    values_by_case: dict[str, float],
    *,
    replicates: int,
    confidence: float,
    minimum_cases: int,
    seed: int,
    seed_context: Iterable[object] = (),
) -> Interval:
    values = np.asarray(
        [
            float(values_by_case[case_id])
            for case_id in sorted(values_by_case)
            if np.isfinite(float(values_by_case[case_id]))
        ],
        dtype=np.float64,
    )
    if values.size < minimum_cases:
        return Interval(
            status="NOT_MEASURED",
            estimate=None,
            lower=None,
            upper=None,
            n_cases=int(values.size),
            confidence=float(confidence),
            replicates=int(replicates),
            reason=f"{values.size} cases is below minimum {minimum_cases}",
        )
    rng = np.random.default_rng(stable_seed(seed, "raw-case-bootstrap", *tuple(seed_context)))
    indices = rng.integers(0, values.size, size=(replicates, values.size), endpoint=False)
    bootstrap_means = np.mean(values[indices], axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        bootstrap_means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return Interval(
        status="MEASURED",
        estimate=float(np.mean(values)),
        lower=float(lower),
        upper=float(upper),
        n_cases=int(values.size),
        confidence=float(confidence),
        replicates=int(replicates),
    )
