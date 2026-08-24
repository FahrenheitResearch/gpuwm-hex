"""Integrity-checked stock-MPAS authority for a transported nonzero tracer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .errors import EvidenceError


STATE_FIELDS = ("rho_zz", "theta_m", "ru", "rw", "qv")
DIAGNOSTIC_FIELDS = ("rho_p", "rtheta_p", "exner", "pressure_p", "u", "w")
REFERENCE_FIELDS = ("rho_base", "rtheta_base", "exner_base", "pressure_base")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenTracerOracle:
    """Read the compiled nonzero-qv trajectory without granting it a port claim."""

    def __init__(
        self,
        directory: str | Path,
        *,
        repository_root: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory).resolve(strict=True)
        if repository_root is None:
            if self.directory.parent.name != "oracle":
                raise EvidenceError(
                    "cannot infer repository root for nonzero-tracer dependencies"
                )
            self.repository_root = self.directory.parent.parent.resolve(strict=True)
        else:
            self.repository_root = Path(repository_root).resolve(strict=True)
        self.manifest_path = self.directory / "manifest.json"
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceError(
                f"cannot read nonzero-tracer oracle manifest: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise EvidenceError("nonzero-tracer oracle manifest is not a JSON object")
        self.manifest: Mapping[str, Any] = raw
        self.manifest_sha256 = _sha256(self.manifest_path)
        self._validate_manifest()
        self._verify_payloads()
        self._verify_dependencies()

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("qv",)

    @property
    def diagnostic_names(self) -> tuple[str, ...]:
        return ()

    @property
    def reference_names(self) -> tuple[str, ...]:
        return ()

    def _validate_manifest(self) -> None:
        if (
            self.manifest.get("schema")
            != "mpas-port.frozen-fortran-nonzero-tracer-step.v1"
        ):
            raise EvidenceError("unsupported nonzero-tracer oracle schema")
        evidence = self.manifest.get("evidence")
        if not isinstance(evidence, dict):
            raise EvidenceError("nonzero-tracer evidence declaration is missing")
        if evidence.get("kind") != "stock-Fortran nonzero-tracer native-state control":
            raise EvidenceError(
                "fixture is not labeled stock-Fortran nonzero-tracer control"
            )
        if evidence.get("port_claim") != "none":
            raise EvidenceError(
                "stock-Fortran nonzero-tracer control must make no port claim"
            )

        expected_groups = {"t0": ["qv"], "t1": ["qv"]}
        if self.manifest.get("groups") != expected_groups:
            raise EvidenceError("nonzero-tracer oracle groups are incomplete")
        expected_keys = {
            f"{group}/{name}"
            for group, names in expected_groups.items()
            for name in names
        }
        payloads = self.manifest.get("payloads")
        if not isinstance(payloads, dict) or set(payloads) != expected_keys:
            raise EvidenceError("nonzero-tracer payload inventory is incomplete")

        proof = self.manifest.get("tracer_proof")
        if not isinstance(proof, dict):
            raise EvidenceError("nonzero-tracer proof is missing")
        required_true = (
            "input_qv_matches_native_t0_bitwise",
            "strictly_positive_at_both_endpoints",
            "horizontal_and_vertical_structure",
        )
        if not all(proof.get(name) is True for name in required_true):
            raise EvidenceError("nonzero-tracer proof does not establish its controls")
        if int(proof.get("changed_element_count", 0)) <= 0:
            raise EvidenceError("stock trajectory did not change the nonzero tracer")
        if float(proof.get("max_abs_change", 0.0)) <= 0.0:
            raise EvidenceError("stock trajectory declares no nonzero tracer amplitude")

        coupling = self.manifest.get("coupling_observation")
        if (
            not isinstance(coupling, dict)
            or coupling.get("qv_can_be_dynamically_active_in_stock_mpas") is not True
            or coupling.get("selected_amplitude_is_bitwise_dynamically_inert")
            is not True
            or coupling.get("state_and_sidecars_hash_linked_without_duplication")
            is not True
        ):
            raise EvidenceError("nonzero-tracer inertness controls are not declared")
        scan = self.manifest.get("amplitude_scan")
        if (
            not isinstance(scan, dict)
            or float(scan.get("selected_scale", 0.0)) <= 0.0
            or float(scan.get("first_non_inert_upper_bracket_scale", 0.0))
            <= float(scan.get("selected_scale", 0.0))
        ):
            raise EvidenceError("nonzero-tracer amplitude scan is not bracketed")
        budget = self.manifest.get("tracer_comparison_budget")
        if (
            not isinstance(budget, dict)
            or float(budget.get("absolute_ceiling", 0.0)) <= 0.0
            or budget.get("frozen_qv_no_transport_mutation", {}).get(
                "decisively_rejected"
            )
            is not True
        ):
            raise EvidenceError("nonzero-tracer comparison budget is invalid")

        for key, declaration in payloads.items():
            if not isinstance(declaration, dict):
                raise EvidenceError(f"invalid nonzero-tracer declaration for {key}")
            group, name = key.split("/", maxsplit=1)
            if declaration.get("group") != group or declaration.get("name") != name:
                raise EvidenceError(f"nonzero-tracer identity mismatch for {key}")
            filename = declaration.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise EvidenceError(f"unsafe nonzero-tracer filename for {key}")
            if declaration.get("dtype") != "<f4" or declaration.get("order") != "C":
                raise EvidenceError(f"unsupported nonzero-tracer layout for {key}")
            shape = declaration.get("shape")
            axes = declaration.get("axes")
            if (
                not isinstance(shape, list)
                or not isinstance(axes, list)
                or len(shape) != len(axes)
            ):
                raise EvidenceError(f"invalid nonzero-tracer shape/axes for {key}")
            if any(not isinstance(value, int) or value <= 0 for value in shape):
                raise EvidenceError(f"invalid nonzero-tracer shape for {key}")
            count = int(np.prod(shape, dtype=np.int64))
            if (
                declaration.get("count") != count
                or declaration.get("bytes") != count * 4
            ):
                raise EvidenceError(f"invalid nonzero-tracer size for {key}")
            if declaration.get("finite_count") != count:
                raise EvidenceError(f"non-finite stock payload declared for {key}")

    def _verify_payloads(self) -> None:
        try:
            sums: dict[str, str] = {}
            for line in (
                (self.directory / "SHA256SUMS").read_text(encoding="ascii").splitlines()
            ):
                digest, filename = line.split("  ", maxsplit=1)
                if filename in sums or len(digest) != 64:
                    raise ValueError
                int(digest, 16)
                sums[filename] = digest.lower()
        except (OSError, UnicodeError, ValueError) as error:
            raise EvidenceError(
                f"cannot read nonzero-tracer SHA256SUMS: {error}"
            ) from error

        payloads = self.manifest["payloads"]
        expected_files = {
            "manifest.json",
            *(item["file"] for item in payloads.values()),
        }
        if set(sums) != expected_files:
            raise EvidenceError(
                "nonzero-tracer checksums do not exactly cover the fixture"
            )
        actual_files = {
            path.name for path in self.directory.iterdir() if path.is_file()
        }
        if actual_files != {"SHA256SUMS", *expected_files}:
            raise EvidenceError("nonzero-tracer fixture contains an unexpected file")
        if sums["manifest.json"] != self.manifest_sha256:
            raise EvidenceError("nonzero-tracer manifest hash mismatch")
        for key, declaration in payloads.items():
            path = self.directory / declaration["file"]
            if not path.is_file() or path.stat().st_size != declaration["bytes"]:
                raise EvidenceError(f"nonzero-tracer payload size mismatch for {key}")
            digest = _sha256(path)
            if digest != declaration.get("sha256") or digest != sums[path.name]:
                raise EvidenceError(f"nonzero-tracer payload hash mismatch for {key}")
            values = np.memmap(
                path, dtype="<f4", mode="r", shape=(declaration["count"],)
            )
            if int(np.count_nonzero(np.isfinite(values))) != declaration["count"]:
                raise EvidenceError(
                    f"nonzero-tracer payload contains non-finite values: {key}"
                )

    def _resolve_dependency(self, raw: object) -> Path:
        if not isinstance(raw, str):
            raise EvidenceError("nonzero-tracer dependency path is not a string")
        posix = PurePosixPath(raw)
        if posix.is_absolute() or not posix.parts or ".." in posix.parts:
            raise EvidenceError("unsafe nonzero-tracer dependency path")
        try:
            candidate = (self.repository_root / Path(*posix.parts)).resolve(strict=True)
        except OSError as error:
            raise EvidenceError(
                "nonzero-tracer dependency cannot be resolved"
            ) from error
        try:
            candidate.relative_to(self.repository_root)
        except ValueError as error:
            raise EvidenceError(
                "nonzero-tracer dependency escapes repository"
            ) from error
        if not candidate.is_file():
            raise EvidenceError("nonzero-tracer dependency is not a file")
        return candidate

    def _verify_dependencies(self) -> None:
        authority = self.manifest.get("authority")
        if not isinstance(authority, dict):
            raise EvidenceError("nonzero-tracer authority declaration is missing")
        try:
            declarations = {
                "base_vertical_fixture": authority["base_vertical_fixture"],
                "native_state_manifest": authority["linked_dry_controls"][
                    "native_state_manifest"
                ],
                "internal_sidecar_manifest": authority["linked_dry_controls"][
                    "internal_sidecar_manifest"
                ],
            }
        except (KeyError, TypeError) as error:
            raise EvidenceError(
                "nonzero-tracer dry dependency declarations are missing"
            ) from error
        resolved: dict[str, Path] = {}
        for name, declaration in declarations.items():
            if not isinstance(declaration, dict):
                raise EvidenceError(f"invalid nonzero-tracer dependency {name}")
            path = self._resolve_dependency(declaration.get("path"))
            digest = declaration.get("sha256")
            if not isinstance(digest, str) or _sha256(path) != digest:
                raise EvidenceError(
                    f"nonzero-tracer dependency hash mismatch for {name}"
                )
            resolved[name] = path
        if resolved["base_vertical_fixture"] != resolved["native_state_manifest"]:
            raise EvidenceError(
                "nonzero-tracer native-state dependency is inconsistent"
            )
        self.dependency_paths = resolved

    def array(
        self,
        group: str,
        name: str,
        *,
        vertical_first: bool = True,
    ) -> NDArray[np.float32]:
        groups = self.manifest["groups"]
        if group not in groups or name not in groups[group]:
            raise EvidenceError(f"unknown nonzero-tracer array {group}/{name}")
        declaration = self.manifest["payloads"][f"{group}/{name}"]
        shape = tuple(int(value) for value in declaration["shape"])
        result = np.memmap(
            self.directory / declaration["file"],
            dtype="<f4",
            mode="r",
            shape=shape,
            order="C",
        )
        if vertical_first:
            axes = tuple(declaration["axes"])
            candidates = [
                index
                for index, axis in enumerate(axes)
                if axis in {"level", "interface"}
            ]
            if len(candidates) > 1:
                raise EvidenceError(f"ambiguous vertical axis for {group}/{name}")
            if candidates:
                result = np.moveaxis(result, candidates[0], 0)
        result.flags.writeable = False
        return result

    def field(self, time: str, name: str) -> NDArray[np.float32]:
        if time not in {"t0", "t1"} or name != "qv":
            raise EvidenceError(f"unknown nonzero-tracer state field {time}/{name}")
        return self.array(time, name)


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "FrozenTracerOracle",
    "REFERENCE_FIELDS",
    "STATE_FIELDS",
]
