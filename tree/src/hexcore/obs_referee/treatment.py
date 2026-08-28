"""Optional external GF-subsidence treatment contract.

gpuwm-hex does not own GF tendency arithmetic. An experimental sibling hook is
accepted only when it emits a narrow receipt proving that the requested
treatment touched GF subsidence and that disabled mode is byte-neutral.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import read_json, sha256_file
from .errors import IntegrityError, SchemaError


TREATMENT_RECEIPT_SCHEMA = "gpuwm.gf-subsidence-treatment/v1"


def validate_treatment_receipt(
    path: str | Path,
    *,
    expected_name: str,
    expected_mode: str,
    expected_value: float,
) -> Mapping[str, Any]:
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise SchemaError("treatment receipt must be an object")
    required = {
        "schema",
        "treatment_name",
        "mode",
        "value",
        "enabled",
        "scope",
        "call_count",
        "columns_touched",
        "pre_tendency_sha256",
        "post_tendency_sha256",
        "producer_commit",
        "metadata",
    }
    if set(receipt) != required:
        raise SchemaError(
            f"treatment receipt keys must be exactly {sorted(required)}"
        )
    if receipt["schema"] != TREATMENT_RECEIPT_SCHEMA:
        raise SchemaError("unsupported treatment receipt schema")
    if receipt["treatment_name"] != expected_name:
        raise IntegrityError(
            f"treatment receipt names {receipt['treatment_name']!r}, expected {expected_name!r}"
        )
    if receipt["mode"] != expected_mode:
        raise IntegrityError(
            f"treatment receipt mode {receipt['mode']!r}, expected {expected_mode!r}"
        )
    if isinstance(receipt["value"], bool) or not isinstance(
        receipt["value"], (int, float)
    ):
        raise SchemaError("treatment receipt value must be numeric")
    if float(receipt["value"]) != float(expected_value):
        raise IntegrityError(
            f"treatment receipt value {receipt['value']!r}, expected {expected_value!r}"
        )
    if receipt["enabled"] is not True:
        raise IntegrityError("experiment receipt says treatment was not enabled")
    if receipt["scope"] != "gf_subsidence_only":
        raise IntegrityError(
            f"treatment scope must be 'gf_subsidence_only', got {receipt['scope']!r}"
        )
    if isinstance(receipt["call_count"], bool) or not isinstance(receipt["call_count"], int):
        raise SchemaError("treatment call_count must be an integer")
    if receipt["call_count"] <= 0:
        raise IntegrityError("enabled treatment recorded zero GF calls")
    if isinstance(receipt["columns_touched"], bool) or not isinstance(
        receipt["columns_touched"], int
    ):
        raise SchemaError("treatment columns_touched must be an integer")
    if receipt["columns_touched"] <= 0:
        raise IntegrityError("enabled treatment recorded zero touched columns")
    for key in ("pre_tendency_sha256", "post_tendency_sha256"):
        value = receipt[key]
        if not isinstance(value, str) or len(value) != 64 or any(
            ch not in "0123456789abcdef" for ch in value
        ):
            raise SchemaError(f"{key} must be lowercase SHA-256")
    producer_commit = receipt["producer_commit"]
    if (
        not isinstance(producer_commit, str)
        or len(producer_commit) != 40
        or any(ch not in "0123456789abcdef" for ch in producer_commit)
    ):
        raise SchemaError("producer_commit must be a full lowercase hexadecimal commit")
    if receipt["pre_tendency_sha256"] == receipt["post_tendency_sha256"]:
        raise IntegrityError(
            "enabled non-neutral treatment did not change the GF subsidence tendency digest"
        )
    if not isinstance(receipt["metadata"], dict):
        raise SchemaError("treatment metadata must be an object")
    return receipt


def compare_output_trees(
    first: str | Path,
    second: str | Path,
    *,
    include: Iterable[str] = ("*.nc", "*.json", "*.npz"),
    exclude: Iterable[str] = ("*treatment-receipt*.json",),
) -> dict[str, Any]:
    """Compare selected output artifacts by relative path and SHA-256."""

    first_root = Path(first).resolve()
    second_root = Path(second).resolve()
    include_patterns = tuple(include)
    exclude_patterns = tuple(exclude)
    first_files = _inventory(first_root, include_patterns, exclude_patterns)
    second_files = _inventory(second_root, include_patterns, exclude_patterns)
    all_paths = sorted(set(first_files) | set(second_files))
    differences = []
    for relative in all_paths:
        a = first_files.get(relative)
        b = second_files.get(relative)
        if a != b:
            differences.append(
                {
                    "path": relative,
                    "first_sha256": a,
                    "second_sha256": b,
                }
            )
    result = {
        "schema": "gpuwm-hex.byte-identity/v1",
        "status": "IDENTICAL" if not differences else "DIFFERENT",
        "first_file_count": len(first_files),
        "second_file_count": len(second_files),
        "differences": differences,
    }
    if differences:
        raise IntegrityError(
            f"disabled treatment is not byte-identical: {len(differences)} differing paths"
        )
    return result


def validate_disabled_receipt(path: str | Path) -> Mapping[str, Any]:
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise SchemaError("disabled treatment receipt must be an object")
    required = {
        "schema",
        "treatment_name",
        "mode",
        "value",
        "enabled",
        "scope",
        "call_count",
        "columns_touched",
        "pre_tendency_sha256",
        "post_tendency_sha256",
        "producer_commit",
        "metadata",
    }
    if set(receipt) != required:
        raise SchemaError("disabled treatment receipt has unexpected keys")
    if receipt["schema"] != TREATMENT_RECEIPT_SCHEMA:
        raise SchemaError("unsupported disabled treatment receipt schema")
    if receipt["enabled"] is not False:
        raise IntegrityError("disabled receipt says treatment was enabled")
    if receipt["call_count"] != 0 or receipt["columns_touched"] != 0:
        raise IntegrityError("disabled treatment must report zero calls and zero touched columns")
    if receipt["pre_tendency_sha256"] != receipt["post_tendency_sha256"]:
        raise IntegrityError("disabled treatment changed the GF tendency digest")
    return receipt


def _inventory(
    root: Path,
    include: tuple[str, ...],
    exclude: tuple[str, ...],
) -> dict[str, str]:
    if not root.is_dir():
        raise IntegrityError(f"output tree is absent: {root}")
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if not any(fnmatch(relative, pattern) or fnmatch(path.name, pattern) for pattern in include):
            continue
        if any(fnmatch(relative, pattern) or fnmatch(path.name, pattern) for pattern in exclude):
            continue
        result[relative] = sha256_file(path)
    if not result:
        raise IntegrityError(f"no files matched identity patterns under {root}")
    return result
