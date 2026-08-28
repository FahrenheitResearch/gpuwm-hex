"""Frozen-evidence loading and numerical gates for the M1 operator oracle."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import EvidenceError
from .horizontal import (
    cell_to_edge,
    cell_to_vertex,
    edge_scalar_gradient,
    edge_to_cell_divergence,
    edge_to_vertex_curl,
)
from .vector import tangential_velocity


@dataclass(frozen=True)
class OracleTolerance:
    """Declared absolute, relative, and ULP limits for one oracle field."""

    atol: float
    rtol: float
    max_ulp: int


@dataclass(frozen=True)
class OracleComparison:
    """One numerical comparison, including diagnostics useful on failure."""

    operator: str
    count: int
    max_abs: float
    max_rel: float
    max_ulp: int
    worst_index: int
    passed: bool
    tolerance: OracleTolerance


@dataclass(frozen=True)
class OracleReport:
    """Complete result of replaying a fixture against the Python authority."""

    evidence_class: str
    fixture_directory: str
    comparisons: tuple[OracleComparison, ...]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(item.passed for item in self.comparisons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_class": self.evidence_class,
            "fixture_directory": self.fixture_directory,
            "passed": self.passed,
            "comparisons": [
                {
                    "operator": item.operator,
                    "count": item.count,
                    "max_abs": item.max_abs,
                    "max_rel": item.max_rel,
                    "max_ulp": item.max_ulp,
                    "worst_index": item.worst_index,
                    "passed": item.passed,
                    "tolerance": {
                        "atol": item.tolerance.atol,
                        "rtol": item.tolerance.rtol,
                        "max_ulp": item.tolerance.max_ulp,
                    },
                }
                for item in self.comparisons
            ],
        }


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading a large fixture or mesh into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(directory: Path) -> Mapping[str, Any]:
    path = directory / "manifest.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read oracle manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise EvidenceError(f"oracle manifest {path} is not a JSON object")
    return manifest


def _verify_fixture_files(directory: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceError("oracle manifest has no hashed files")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise EvidenceError("oracle file hashes must map names to SHA-256 strings")
        path = directory / name
        if not path.is_file():
            raise EvidenceError(f"oracle fixture is missing {path}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise EvidenceError(
                f"oracle fixture hash mismatch for {name}: {actual} != {expected}"
            )


def load_csv_fields(
    path: str | Path, *, label_column: str
) -> dict[str, NDArray[np.float64]]:
    """Load indexed CSV vectors and enforce unique contiguous one-based indices."""

    source = Path(path)
    grouped: dict[str, list[tuple[int, float]]] = {}
    try:
        with source.open("r", encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {label_column, "index", "value"}
            if reader.fieldnames is None or set(reader.fieldnames) != expected_columns:
                raise EvidenceError(
                    f"{source} columns are {reader.fieldnames!r}, expected "
                    f"{sorted(expected_columns)!r}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    label = row[label_column].strip()
                    index = int(row["index"])
                    value = float(row["value"])
                except (AttributeError, TypeError, ValueError) as error:
                    raise EvidenceError(
                        f"malformed oracle row {source}:{line_number}"
                    ) from error
                if not label or index < 1 or not np.isfinite(value):
                    raise EvidenceError(f"invalid oracle row {source}:{line_number}")
                grouped.setdefault(label, []).append((index, value))
    except OSError as error:
        raise EvidenceError(f"cannot read oracle CSV {source}: {error}") from error

    result: dict[str, NDArray[np.float64]] = {}
    for label, records in grouped.items():
        indices = [record[0] for record in records]
        expected_indices = list(range(1, len(records) + 1))
        if indices != expected_indices:
            raise EvidenceError(
                f"oracle vector {label!r} does not have contiguous one-based indices"
            )
        result[label] = np.asarray([record[1] for record in records], dtype=np.float64)
    if not result:
        raise EvidenceError(f"oracle CSV {source} contains no vectors")
    return result


def ulp_distance(expected: ArrayLike, actual: ArrayLike) -> NDArray[np.uint64]:
    """Return elementwise IEEE-754 binary64 ULP distance, including negatives."""

    left = np.ascontiguousarray(expected, dtype=np.float64)
    right = np.ascontiguousarray(actual, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch for ULP distance: {left.shape} != {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("ULP distance is defined here only for finite values")
    sign = np.uint64(1 << 63)
    magnitude_mask = ~sign
    left_bits = left.view(np.uint64)
    right_bits = right.view(np.uint64)
    left_negative = (left_bits & sign) != 0
    right_negative = (right_bits & sign) != 0
    same_sign_distance = np.maximum(left_bits, right_bits) - np.minimum(
        left_bits, right_bits
    )
    # Across zero, count magnitude steps and collapse -0.0/+0.0 to the same
    # numerical value.  This keeps nextafter(-tiny, +0) one ULP as expected.
    cross_zero_distance = (left_bits & magnitude_mask) + (
        right_bits & magnitude_mask
    )
    return np.where(left_negative == right_negative, same_sign_distance, cross_zero_distance)


def compare_oracle_vector(
    operator: str,
    expected: ArrayLike,
    actual: ArrayLike,
    tolerance: OracleTolerance,
) -> OracleComparison:
    """Apply all three declared tolerances and retain worst-point diagnostics."""

    reference = np.asarray(expected, dtype=np.float64)
    candidate = np.asarray(actual, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise EvidenceError(
            f"oracle shape mismatch for {operator}: {candidate.shape} != {reference.shape}"
        )
    if reference.ndim != 1 or reference.size == 0:
        raise EvidenceError(f"oracle vector {operator} must be non-empty and one-dimensional")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise EvidenceError(f"oracle vector {operator} contains non-finite values")

    absolute = np.abs(candidate - reference)
    relative = absolute / np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    ulps = ulp_distance(reference, candidate)
    worst = int(np.argmax(absolute))
    numeric_ok = np.all(
        absolute <= tolerance.atol + tolerance.rtol * np.abs(reference)
    )
    ulp_ok = int(np.max(ulps)) <= tolerance.max_ulp
    return OracleComparison(
        operator=operator,
        count=int(reference.size),
        max_abs=float(np.max(absolute)),
        max_rel=float(np.max(relative)),
        max_ulp=int(np.max(ulps)),
        worst_index=worst + 1,
        passed=bool(numeric_ok and ulp_ok),
        tolerance=tolerance,
    )


def run_m1_oracle(mesh: object, fixture_directory: str | Path) -> OracleReport:
    """Replay every M1 fixture and return a fail-closed numerical report."""

    directory = Path(fixture_directory).resolve(strict=True)
    manifest = _load_manifest(directory)
    _verify_fixture_files(directory, manifest)
    inputs = load_csv_fields(directory / "inputs.csv", label_column="field")
    expected = load_csv_fields(directory / "outputs.csv", label_column="operator")

    try:
        phi = inputs["phi_cell"]
        normal_velocity = inputs["u_edge"]
    except KeyError as error:
        raise EvidenceError(f"oracle inputs are missing {error.args[0]!r}") from error

    actual = {
        "edge_scalar_gradient": edge_scalar_gradient(phi, mesh),
        "edge_to_cell_divergence": edge_to_cell_divergence(normal_velocity, mesh),
        "edge_to_vertex_curl": edge_to_vertex_curl(normal_velocity, mesh),
        "tangential_velocity": tangential_velocity(normal_velocity, mesh),
        "cell_to_edge": cell_to_edge(phi, mesh),
        "cell_to_vertex": cell_to_vertex(phi, mesh),
    }
    declared = manifest.get("tolerances")
    if not isinstance(declared, dict):
        raise EvidenceError("oracle manifest has no tolerance declarations")

    comparisons: list[OracleComparison] = []
    if set(expected) != set(actual):
        raise EvidenceError(
            "oracle output set differs from implemented operator set: "
            f"{sorted(expected)} != {sorted(actual)}"
        )
    for name, candidate in actual.items():
        raw = declared.get(name)
        if not isinstance(raw, dict):
            raise EvidenceError(f"oracle manifest has no tolerance for {name}")
        try:
            tolerance = OracleTolerance(
                atol=float(raw["atol"]),
                rtol=float(raw["rtol"]),
                max_ulp=int(raw["max_ulp"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceError(f"invalid tolerance declaration for {name}") from error
        if tolerance.atol < 0 or tolerance.rtol < 0 or tolerance.max_ulp < 0:
            raise EvidenceError(f"negative tolerance declaration for {name}")
        comparisons.append(
            compare_oracle_vector(name, expected[name], candidate, tolerance)
        )

    evidence_class = manifest.get("evidence_class")
    if not isinstance(evidence_class, str) or not evidence_class:
        raise EvidenceError("oracle manifest has no evidence_class")
    return OracleReport(evidence_class, str(directory), tuple(comparisons))
