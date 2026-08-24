"""Portable native-state control for the frozen no-mixing MPAS-A step.

The fixture consumed here is stock-Fortran control data.  It is deliberately
not evidence that the Python port matches a complete timestep.  Its purpose is
to make that comparison exact, local, and reproducible without another MPAS
build or authority run.
"""

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
_STATE_FIELDS = ("rho_zz", "theta_m", "ru", "rw", "qv")
_REFERENCE_FIELDS = (
    "rho_base",
    "theta_base",
    "rtheta_base",
    "exner_base",
    "pressure_base",
)
_VERTICAL_FIELDS = (
    "zgrid",
    "rdzw",
    "dzu",
    "rdzu",
    "fzm",
    "fzp",
    "zz",
    "zxu",
    "dss",
    "zb",
    "zb3",
    "cf1",
    "cf2",
    "cf3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ulp_distance_f32(
    reference: NDArray[np.float32], candidate: NDArray[np.float32]
) -> NDArray[np.uint32]:
    """Return binary32 ULP distance, treating the two signed zeros equally."""

    left = np.ascontiguousarray(reference, dtype="<f4")
    right = np.ascontiguousarray(candidate, dtype="<f4")
    sign = np.uint32(1 << 31)
    magnitude = ~sign
    left_bits = left.view("<u4")
    right_bits = right.view("<u4")
    same_sign = np.maximum(left_bits, right_bits) - np.minimum(left_bits, right_bits)
    cross_zero = (left_bits & magnitude) + (right_bits & magnitude)
    return np.where((left_bits & sign) == (right_bits & sign), same_sign, cross_zero)


@dataclass(frozen=True, slots=True)
class NomixTolerance:
    atol: float
    rtol: float


@dataclass(frozen=True, slots=True)
class NomixFieldComparison:
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
    tolerance: NomixTolerance
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
            "worst_index_level_first_zero_based": list(
                self.worst_index_level_first
            ),
            "tolerance": {
                "atol": self.tolerance.atol,
                "rtol": self.tolerance.rtol,
            },
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class NomixOracleReport:
    evidence_kind: str
    verification_status: str
    fixture_directory: str
    manifest_sha256: str
    comparisons: tuple[NomixFieldComparison, ...]

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
        failures = [
            f"{item.time}/{item.field}: {item.failed_count} failed, "
            f"max_abs={item.max_abs:.9g}"
            for item in self.comparisons
            if not item.passed
        ]
        raise EvidenceError(
            "frozen no-mixing native-step mismatch: " + "; ".join(failures)
        )


class FrozenNomixOracle:
    """Integrity-check and compare a frozen native MPAS state trajectory.

    Payloads retain the source NetCDF logical order.  State and reference
    accessors default to the port's ``(vertical, horizontal)`` convention.
    The generic :meth:`array` accessor can expose either source order or move
    the declared ``level``/``interface`` axis first.
    """

    def __init__(self, fixture_directory: str | Path) -> None:
        self.directory = Path(fixture_directory).resolve(strict=True)
        self.manifest_path = self.directory / "manifest.json"
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(f"cannot read no-mixing oracle manifest: {error}") from error
        if not isinstance(raw, dict):
            raise EvidenceError("no-mixing oracle manifest is not a JSON object")
        self.manifest: Mapping[str, Any] = raw
        self.manifest_sha256 = _sha256(self.manifest_path)
        self._validate_manifest()
        self._verify_checksums_and_payloads()

    @property
    def time_ids(self) -> tuple[str, ...]:
        return ("t0", "t1")

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["groups"]["t0"])

    @property
    def reference_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["groups"]["reference"])

    @property
    def vertical_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["groups"]["vertical"])

    @property
    def evidence_kind(self) -> str:
        return str(self.manifest["evidence"]["kind"])

    @property
    def verification_status(self) -> str:
        return str(self.manifest["evidence"]["verification_status"])

    def _validate_manifest(self) -> None:
        if self.manifest.get("schema") != "mpas-port.frozen-fortran-native-step.v1":
            raise EvidenceError("unsupported no-mixing oracle manifest schema")
        evidence = self.manifest.get("evidence")
        if not isinstance(evidence, dict):
            raise EvidenceError("no-mixing oracle evidence declaration is missing")
        if evidence.get("kind") != "stock-Fortran native-state control":
            raise EvidenceError("fixture is not labeled stock-Fortran native-state control")
        if evidence.get("port_claim") != "none":
            raise EvidenceError("no-mixing control must explicitly make no port claim")

        groups = self.manifest.get("groups")
        expected_groups = {"t0", "t1", "reference", "vertical"}
        if not isinstance(groups, dict) or set(groups) != expected_groups:
            raise EvidenceError("no-mixing oracle groups are incomplete")
        expected_members = {
            "t0": _STATE_FIELDS,
            "t1": _STATE_FIELDS,
            "reference": _REFERENCE_FIELDS,
            "vertical": _VERTICAL_FIELDS,
        }
        for group, expected in expected_members.items():
            members = groups.get(group)
            if not isinstance(members, list) or tuple(members) != expected:
                raise EvidenceError(f"unexpected no-mixing oracle {group} members")

        state_fields = self.manifest.get("state_fields")
        if not isinstance(state_fields, dict) or set(state_fields) != set(_STATE_FIELDS):
            raise EvidenceError("no-mixing oracle state-field declarations are incomplete")
        for field, definition in state_fields.items():
            if not isinstance(definition, dict):
                raise EvidenceError(f"invalid no-mixing field declaration for {field}")
            tolerance = definition.get("tolerance")
            if not isinstance(tolerance, dict):
                raise EvidenceError(f"missing no-mixing tolerance for {field}")
            try:
                atol = float(tolerance["atol"])
                rtol = float(tolerance["rtol"])
            except (KeyError, TypeError, ValueError) as error:
                raise EvidenceError(f"invalid no-mixing tolerance for {field}") from error
            if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0.0 or rtol < 0.0:
                raise EvidenceError(f"invalid no-mixing tolerance for {field}")

        payloads = self.manifest.get("payloads")
        if not isinstance(payloads, dict):
            raise EvidenceError("no-mixing oracle payload declarations are missing")
        expected_keys = {
            f"{group}/{name}"
            for group, members in expected_members.items()
            for name in members
        }
        if set(payloads) != expected_keys:
            raise EvidenceError("no-mixing payloads do not exactly span every group member")

        filenames: set[str] = set()
        for key, declaration in payloads.items():
            if not isinstance(declaration, dict):
                raise EvidenceError(f"invalid no-mixing payload declaration for {key}")
            group, name = key.split("/", maxsplit=1)
            if declaration.get("group") != group or declaration.get("name") != name:
                raise EvidenceError(f"no-mixing payload identity mismatch for {key}")
            filename = declaration.get("file")
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or filename in filenames
            ):
                raise EvidenceError(f"unsafe or duplicate no-mixing payload filename for {key}")
            filenames.add(filename)
            if declaration.get("dtype") != "<f4" or declaration.get("order") != "C":
                raise EvidenceError(f"unsupported no-mixing payload layout for {key}")
            shape = declaration.get("shape")
            axes = declaration.get("axes")
            if not isinstance(shape, list) or not isinstance(axes, list) or len(shape) != len(axes):
                raise EvidenceError(f"invalid no-mixing payload shape/axes for {key}")
            if any(not isinstance(value, int) or value <= 0 for value in shape):
                raise EvidenceError(f"invalid no-mixing payload shape for {key}")
            if any(not isinstance(axis, str) or not axis for axis in axes) or len(set(axes)) != len(axes):
                raise EvidenceError(f"invalid no-mixing payload axes for {key}")
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if declaration.get("count") != count or declaration.get("bytes") != count * 4:
                raise EvidenceError(f"invalid no-mixing payload size declaration for {key}")
            if declaration.get("finite_count") != count:
                raise EvidenceError(f"no-mixing authority payload is not declared fully finite: {key}")

    def _verify_checksums_and_payloads(self) -> None:
        sums_path = self.directory / "SHA256SUMS"
        try:
            records: dict[str, str] = {}
            for line in sums_path.read_text(encoding="ascii").splitlines():
                digest, filename = line.split("  ", maxsplit=1)
                if filename in records or len(digest) != 64:
                    raise ValueError
                int(digest, 16)
                records[filename] = digest.lower()
        except (OSError, UnicodeError, ValueError) as error:
            raise EvidenceError(f"cannot read no-mixing oracle SHA256SUMS: {error}") from error

        payloads = self.manifest["payloads"]
        expected_names = {"manifest.json", *(item["file"] for item in payloads.values())}
        if set(records) != expected_names:
            raise EvidenceError("SHA256SUMS does not exactly cover manifest and payloads")
        if records["manifest.json"] != self.manifest_sha256.lower():
            raise EvidenceError("no-mixing oracle manifest hash mismatch")

        for key, declaration in payloads.items():
            filename = declaration["file"]
            path = self.directory / filename
            if not path.is_file():
                raise EvidenceError(f"no-mixing oracle payload is missing: {filename}")
            if path.stat().st_size != declaration["bytes"]:
                raise EvidenceError(f"no-mixing oracle payload size mismatch: {filename}")
            actual_hash = _sha256(path)
            if (
                actual_hash.lower() != str(declaration.get("sha256", "")).lower()
                or actual_hash.lower() != records[filename]
            ):
                raise EvidenceError(f"no-mixing oracle payload hash mismatch: {filename}")
            values = np.memmap(path, dtype="<f4", mode="r", shape=(declaration["count"],))
            finite_count = int(np.count_nonzero(np.isfinite(values)))
            if finite_count != declaration["finite_count"]:
                raise EvidenceError(f"no-mixing oracle finite-count mismatch: {key}")

    def _declaration(self, group: str, name: str) -> Mapping[str, Any]:
        groups = self.manifest["groups"]
        if group not in groups:
            raise EvidenceError(f"unknown no-mixing oracle group {group!r}")
        if name not in groups[group]:
            raise EvidenceError(f"unknown no-mixing oracle array {group}/{name}")
        return self.manifest["payloads"][f"{group}/{name}"]

    def array(
        self,
        group: str,
        name: str,
        *,
        vertical_first: bool = False,
    ) -> NDArray[np.float32]:
        declaration = self._declaration(group, name)
        shape = tuple(int(value) for value in declaration["shape"])
        mapped_shape = shape if shape else (1,)
        mapped = np.memmap(
            self.directory / declaration["file"],
            dtype="<f4",
            mode="r",
            shape=mapped_shape,
            order="C",
        )
        result: NDArray[np.float32] = mapped if shape else mapped.reshape(())
        if vertical_first:
            axes = tuple(declaration["axes"])
            candidates = [
                index for index, axis in enumerate(axes) if axis in {"level", "interface"}
            ]
            if len(candidates) > 1:
                raise EvidenceError(f"ambiguous vertical axis for {group}/{name}")
            if candidates:
                result = np.moveaxis(result, candidates[0], 0)
        result.flags.writeable = False
        return result

    def field(
        self, time: str, field: str, *, level_first: bool = True
    ) -> NDArray[np.float32]:
        if time not in self.time_ids:
            raise EvidenceError(f"unknown no-mixing oracle time {time!r}")
        if field not in self.field_names:
            raise EvidenceError(f"unknown no-mixing oracle field {field!r}")
        return self.array(time, field, vertical_first=level_first)

    def reference(
        self, name: str, *, level_first: bool = True
    ) -> NDArray[np.float32]:
        return self.array("reference", name, vertical_first=level_first)

    def vertical(
        self, name: str, *, vertical_first: bool = False
    ) -> NDArray[np.float32]:
        return self.array("vertical", name, vertical_first=vertical_first)

    def compare(
        self,
        time: str,
        candidate: Mapping[str, ArrayLike],
        *,
        fields: Sequence[str] | None = None,
        level_first: bool = True,
    ) -> NomixOracleReport:
        if time not in self.time_ids:
            raise EvidenceError(f"unknown no-mixing oracle time {time!r}")
        selected = self.field_names if fields is None else tuple(fields)
        if not selected:
            raise EvidenceError("no-mixing oracle comparison selected no fields")
        if len(set(selected)) != len(selected):
            raise EvidenceError("no-mixing oracle comparison contains duplicate fields")
        unknown = set(selected) - set(self.field_names)
        if unknown:
            raise EvidenceError(f"unknown no-mixing comparison fields: {sorted(unknown)}")
        missing = set(selected) - set(candidate)
        if missing:
            raise EvidenceError(f"candidate is missing no-mixing fields: {sorted(missing)}")
        comparisons = tuple(
            self._compare_field(
                time,
                field,
                candidate[field],
                level_first=level_first,
            )
            for field in selected
        )
        return NomixOracleReport(
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
    ) -> NomixOracleReport:
        missing = set(self.time_ids) - set(candidate)
        if missing:
            raise EvidenceError(f"candidate trajectory is missing times: {sorted(missing)}")
        comparisons: list[NomixFieldComparison] = []
        for time in self.time_ids:
            comparisons.extend(
                self.compare(
                    time,
                    candidate[time],
                    fields=fields,
                    level_first=level_first,
                ).comparisons
            )
        return NomixOracleReport(
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
    ) -> NomixFieldComparison:
        reference = np.asarray(self.field(time, field, level_first=level_first))
        values = np.asarray(candidate)
        if values.shape != reference.shape:
            raise EvidenceError(
                f"no-mixing oracle shape mismatch for {time}/{field}: "
                f"{values.shape} != {reference.shape}"
            )
        if values.dtype.kind != "f":
            raise EvidenceError(f"no-mixing candidate {time}/{field} is not floating point")

        tolerance_raw = self.manifest["state_fields"][field]["tolerance"]
        tolerance = NomixTolerance(
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

        with np.errstate(over="ignore", invalid="ignore"):
            rounded = values.astype("<f4")
        if np.all(np.isfinite(rounded)):
            max_ulp = int(np.max(_ulp_distance_f32(reference, rounded)))
        else:
            max_ulp = int(np.iinfo(np.uint32).max)
        finite_difference = difference[np.isfinite(difference)]
        rms = (
            float(np.sqrt(np.mean(finite_difference * finite_difference)))
            if finite_difference.size
            else float("inf")
        )
        canonical_worst = worst if level_first else (worst[1], worst[0])
        return NomixFieldComparison(
            time=time,
            field=field,
            count=int(reference.size),
            max_abs=float(np.max(difference)),
            max_rel=float(np.max(relative)),
            rms=rms,
            max_ulp_f32=max_ulp,
            failed_count=int(np.count_nonzero(failed)),
            nonfinite_count=nonfinite_count,
            worst_index_level_first=canonical_worst,
            tolerance=tolerance,
            passed=not np.any(failed),
        )


__all__ = [
    "FrozenNomixOracle",
    "NomixFieldComparison",
    "NomixOracleReport",
    "NomixTolerance",
]
