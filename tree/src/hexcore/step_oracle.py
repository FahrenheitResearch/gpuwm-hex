"""Integrity-checked gate against a complete frozen MPAS-A RK3 trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import EvidenceError


FloatArray = NDArray[np.floating[Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ulp_distance_f32(reference: NDArray[np.float32], candidate: NDArray[np.float32]) -> NDArray[np.uint32]:
    """Binary32 ULP distance with -0 and +0 treated as the same value."""

    left = np.ascontiguousarray(reference, dtype="<f4")
    right = np.ascontiguousarray(candidate, dtype="<f4")
    sign = np.uint32(1 << 31)
    magnitude = ~sign
    left_bits = left.view("<u4")
    right_bits = right.view("<u4")
    left_negative = (left_bits & sign) != 0
    right_negative = (right_bits & sign) != 0
    same_sign = np.maximum(left_bits, right_bits) - np.minimum(left_bits, right_bits)
    cross_zero = (left_bits & magnitude) + (right_bits & magnitude)
    return np.where(left_negative == right_negative, same_sign, cross_zero)


@dataclass(frozen=True, slots=True)
class StepTolerance:
    atol: float
    rtol: float


@dataclass(frozen=True, slots=True)
class StepFieldComparison:
    time: str
    field: str
    count: int
    max_abs: float
    max_rel: float
    rms: float
    max_ulp_f32: int
    failed_count: int
    nonfinite_count: int
    worst_index_level_first: tuple[int, int]
    tolerance: StepTolerance
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "time": self.time,
            "field": self.field,
            "count": self.count,
            "max_abs": self.max_abs,
            "max_rel": self.max_rel,
            "rms": self.rms,
            "max_ulp_f32": self.max_ulp_f32,
            "failed_count": self.failed_count,
            "nonfinite_count": self.nonfinite_count,
            "worst_index_level_first_zero_based": list(self.worst_index_level_first),
            "tolerance": {"atol": self.tolerance.atol, "rtol": self.tolerance.rtol},
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class StepOracleReport:
    evidence_kind: str
    verification_status: str
    fixture_directory: str
    manifest_sha256: str
    comparisons: tuple[StepFieldComparison, ...]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(item.passed for item in self.comparisons)

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_kind": self.evidence_kind,
            "verification_status": self.verification_status,
            "fixture_directory": self.fixture_directory,
            "manifest_sha256": self.manifest_sha256,
            "passed": self.passed,
            "comparisons": [item.as_dict() for item in self.comparisons],
        }

    def require_pass(self) -> None:
        if self.passed:
            return
        failed = [
            f"{item.time}/{item.field}: {item.failed_count} failed, "
            f"max_abs={item.max_abs:.9g}"
            for item in self.comparisons
            if not item.passed
        ]
        raise EvidenceError("frozen whole-step oracle mismatch: " + "; ".join(failed))


class FrozenStepOracle:
    """Loader and fail-closed comparator for the frozen two-endpoint fixture.

    Raw fixture files preserve the NetCDF logical order ``(horizontal, level)``.
    The public array and comparison APIs default to ``level_first=True`` because
    that is the Python port's declared logical state convention.
    """

    def __init__(self, fixture_directory: str | Path) -> None:
        self.directory = Path(fixture_directory).resolve(strict=True)
        self.manifest_path = self.directory / "manifest.json"
        try:
            self.manifest: Mapping[str, Any] = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(f"cannot read step-oracle manifest: {error}") from error
        if not isinstance(self.manifest, dict):
            raise EvidenceError("step-oracle manifest is not a JSON object")
        self.manifest_sha256 = _sha256(self.manifest_path)
        self._validate_manifest()
        self._verify_checksums()

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["fields"])

    @property
    def time_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["times"])

    @property
    def evidence_kind(self) -> str:
        return str(self.manifest["evidence"]["kind"])

    @property
    def verification_status(self) -> str:
        return str(self.manifest["evidence"]["verification_status"])

    def _validate_manifest(self) -> None:
        if self.manifest.get("schema") != "mpas-port.frozen-fortran-trajectory.v1":
            raise EvidenceError("unsupported step-oracle manifest schema")
        evidence = self.manifest.get("evidence")
        if not isinstance(evidence, dict) or evidence.get("kind") != "full-model frozen-Fortran trajectory":
            raise EvidenceError("fixture is not labeled full-model frozen-Fortran trajectory evidence")
        fields = self.manifest.get("fields")
        times = self.manifest.get("times")
        files = self.manifest.get("files")
        if not isinstance(fields, dict) or not fields:
            raise EvidenceError("step-oracle manifest has no fields")
        if not isinstance(times, dict) or set(times) != {"t0", "t1"}:
            raise EvidenceError("step-oracle manifest must contain t0 and t1")
        if not isinstance(files, dict):
            raise EvidenceError("step-oracle manifest has no file records")
        expected_files = {f"{time}_{field}.f32le" for time in times for field in fields}
        if set(files) != expected_files:
            raise EvidenceError("step-oracle file records do not span every time/field pair")
        for field, definition in fields.items():
            if not isinstance(definition, dict):
                raise EvidenceError(f"invalid field declaration for {field}")
            shape = definition.get("shape")
            tolerance = definition.get("tolerance")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(not isinstance(value, int) or value <= 0 for value in shape)
            ):
                raise EvidenceError(f"invalid shape declaration for {field}")
            if not isinstance(tolerance, dict):
                raise EvidenceError(f"missing tolerance declaration for {field}")
            try:
                atol = float(tolerance["atol"])
                rtol = float(tolerance["rtol"])
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError(f"invalid tolerance declaration for {field}") from error
            if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
                raise EvidenceError(f"invalid tolerance declaration for {field}")

    def _verify_checksums(self) -> None:
        sums_path = self.directory / "SHA256SUMS"
        try:
            records = {}
            for line in sums_path.read_text(encoding="ascii").splitlines():
                digest, name = line.split("  ", maxsplit=1)
                records[name] = digest
        except (OSError, ValueError) as error:
            raise EvidenceError(f"cannot read step-oracle SHA256SUMS: {error}") from error
        expected_names = {"manifest.json", *self.manifest["files"]}
        if set(records) != expected_names:
            raise EvidenceError("SHA256SUMS does not exactly cover manifest and payloads")
        if records["manifest.json"].lower() != self.manifest_sha256.lower():
            raise EvidenceError("step-oracle manifest hash mismatch")
        for filename, declaration in self.manifest["files"].items():
            path = self.directory / filename
            if not path.is_file():
                raise EvidenceError(f"step-oracle payload is missing: {filename}")
            actual = _sha256(path)
            declared = declaration.get("sha256")
            if actual.lower() != str(declared).lower() or actual.lower() != records[filename].lower():
                raise EvidenceError(f"step-oracle payload hash mismatch: {filename}")
            if path.stat().st_size != int(declaration.get("bytes", -1)):
                raise EvidenceError(f"step-oracle payload size mismatch: {filename}")

    def field(
        self, time: str, field: str, *, level_first: bool = True
    ) -> NDArray[np.float32]:
        if time not in self.manifest["times"]:
            raise EvidenceError(f"unknown step-oracle time {time!r}")
        if field not in self.manifest["fields"]:
            raise EvidenceError(f"unknown step-oracle field {field!r}")
        declaration = self.manifest["files"][f"{time}_{field}.f32le"]
        shape = tuple(int(value) for value in declaration["shape"])
        values = np.memmap(
            self.directory / f"{time}_{field}.f32le",
            dtype="<f4",
            mode="r",
            shape=shape,
            order="C",
        )
        result = values.T if level_first else values
        result.flags.writeable = False
        return result

    def compare(
        self,
        time: str,
        candidate: Mapping[str, ArrayLike],
        *,
        fields: Sequence[str] | None = None,
        level_first: bool = True,
    ) -> StepOracleReport:
        selected = self.field_names if fields is None else tuple(fields)
        if not selected:
            raise EvidenceError("step-oracle comparison selected no fields")
        unknown = set(selected) - set(self.field_names)
        if unknown:
            raise EvidenceError(f"unknown step-oracle comparison fields: {sorted(unknown)}")
        missing = set(selected) - set(candidate)
        if missing:
            raise EvidenceError(f"candidate is missing step-oracle fields: {sorted(missing)}")
        comparisons = tuple(
            self._compare_field(
                time,
                field,
                candidate[field],
                level_first=level_first,
            )
            for field in selected
        )
        return StepOracleReport(
            evidence_kind=self.evidence_kind,
            verification_status=self.verification_status,
            fixture_directory=str(self.directory),
            manifest_sha256=self.manifest_sha256,
            comparisons=comparisons,
        )

    def compare_trajectory(
        self,
        candidate: Mapping[str, Mapping[str, ArrayLike]],
        *,
        fields: Sequence[str] | None = None,
        level_first: bool = True,
    ) -> StepOracleReport:
        missing = set(self.time_ids) - set(candidate)
        if missing:
            raise EvidenceError(f"candidate trajectory is missing times: {sorted(missing)}")
        comparisons: list[StepFieldComparison] = []
        for time in self.time_ids:
            comparisons.extend(
                self.compare(
                    time,
                    candidate[time],
                    fields=fields,
                    level_first=level_first,
                ).comparisons
            )
        return StepOracleReport(
            evidence_kind=self.evidence_kind,
            verification_status=self.verification_status,
            fixture_directory=str(self.directory),
            manifest_sha256=self.manifest_sha256,
            comparisons=tuple(comparisons),
        )

    def _compare_field(
        self,
        time: str,
        field: str,
        candidate: ArrayLike,
        *,
        level_first: bool,
    ) -> StepFieldComparison:
        reference = np.asarray(self.field(time, field, level_first=level_first))
        values = np.asarray(candidate)
        if values.shape != reference.shape:
            raise EvidenceError(
                f"step-oracle shape mismatch for {time}/{field}: "
                f"{values.shape} != {reference.shape}"
            )
        if values.dtype.kind != "f":
            raise EvidenceError(f"step-oracle candidate {time}/{field} is not floating point")

        tolerance_raw = self.manifest["fields"][field]["tolerance"]
        tolerance = StepTolerance(
            atol=float(tolerance_raw["atol"]),
            rtol=float(tolerance_raw["rtol"]),
        )
        reference64 = reference.astype(np.float64)
        candidate64 = values.astype(np.float64)
        finite = np.isfinite(candidate64)
        nonfinite_count = int(candidate64.size - np.count_nonzero(finite))
        difference = np.full(candidate64.shape, np.inf, dtype=np.float64)
        difference[finite] = np.abs(candidate64[finite] - reference64[finite])
        envelope = tolerance.atol + tolerance.rtol * np.abs(reference64)
        failed = (~finite) | (difference > envelope)
        worst_flat = int(np.argmax(difference))
        worst = tuple(int(index) for index in np.unravel_index(worst_flat, difference.shape))
        relative = difference / np.maximum(np.abs(reference64), np.finfo(np.float64).tiny)

        rounded = values.astype("<f4")
        rounded_finite = np.isfinite(rounded)
        if np.all(rounded_finite):
            max_ulp = int(np.max(_ulp_distance_f32(reference, rounded)))
        else:
            max_ulp = int(np.iinfo(np.uint32).max)
        finite_difference = difference[np.isfinite(difference)]
        rms = (
            float(np.sqrt(np.mean(finite_difference * finite_difference)))
            if finite_difference.size
            else float("inf")
        )
        return StepFieldComparison(
            time=time,
            field=field,
            count=int(reference.size),
            max_abs=float(np.max(difference)),
            max_rel=float(np.max(relative)),
            rms=rms,
            max_ulp_f32=max_ulp,
            failed_count=int(np.count_nonzero(failed)),
            nonfinite_count=nonfinite_count,
            worst_index_level_first=worst if level_first else (worst[1], worst[0]),
            tolerance=tolerance,
            passed=not np.any(failed),
        )

