"""Compare same-host compile inputs without turning cache hits into evidence.

Source, ordered requested options, kernel inventory and the measured compiler
platform must agree. Effective flags and captured images are compared wherever
both ranks recorded them. Missing NVRTC capture is retained as missing evidence,
not inferred from another rank's compile or from a matching source digest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
import json
import operator
import re
from typing import Any

from .cuda_backend.compile_contract import (
    COMPILE_MANIFEST_SCHEMA, CompileContractError, canonical_sha256,
    validate_compile_platform_fingerprint, validate_module_key,
)
from .partition_executor_v841 import ExecutorError

__all__ = ["SameHostCompileMismatch", "compare_samehost_compile_manifests"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
#: cuda_shape.specialize: a generic module key plus the first 20 hex digits of
#: the specialized source's SHA-256. Each rank bakes its own mesh dimensions.
_SHAPE_KEY = re.compile(r"^(?P<base>.+)\.shape_(?P<tag>[0-9a-f]{20})$")


class SameHostCompileMismatch(ExecutorError):
    """Rank compiler inputs or mutually observed compile evidence disagree."""


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise SameHostCompileMismatch(f"{where}: an exact SHA-256 digest is required")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SameHostCompileMismatch(f"{where}: a sequence of nonempty strings is required")
    return tuple(value)


def _observations(entry: Mapping[str, Any], where: str) -> tuple[tuple[str, ...] | None, dict[str, str]]:
    effective = entry.get("effective_compile")
    if not isinstance(effective, Mapping):
        raise SameHostCompileMismatch(f"{where}: effective compile evidence is missing")
    status = effective.get("status")
    if status == "unavailable":
        if effective.get("observations"):
            raise SameHostCompileMismatch(f"{where}: unavailable evidence contains observations")
        return None, {}
    if status != "resolved":
        raise SameHostCompileMismatch(f"{where}: unknown effective compile evidence status {status!r}")
    observations = effective.get("observations")
    if not isinstance(observations, list) or not observations:
        raise SameHostCompileMismatch(f"{where}: resolved evidence has no NVRTC observations")
    flags = None
    images = {}
    for index, observation in enumerate(observations):
        label = f"{where} observation {index}"
        if not isinstance(observation, Mapping):
            raise SameHostCompileMismatch(f"{label}: observation is not a mapping")
        if observation.get("source_sha256") != entry["source_sha256"]:
            raise SameHostCompileMismatch(f"{label}: observed source contradicts the declared source")
        observed_flags = _strings(observation.get("effective_flags"), f"{label} flags")
        if flags is not None and flags != observed_flags:
            raise SameHostCompileMismatch(f"{where}: NVRTC observations contain contradictory effective flags")
        flags = observed_flags
        image = observation.get("compiled_image")
        if not isinstance(image, Mapping) or image.get("status") not in ("resolved", "unavailable"):
            raise SameHostCompileMismatch(f"{label}: compiled image availability is malformed")
        if image["status"] == "resolved":
            kind = image.get("kind")
            if kind not in ("ptx", "cubin"):
                raise SameHostCompileMismatch(f"{label}: unknown captured image kind {kind!r}")
            digest = _digest(image.get("sha256"), f"{label} image")
            if kind in images and images[kind] != digest:
                raise SameHostCompileMismatch(f"{where}: contradictory captured {kind} images")
            images[kind] = digest
    return flags, images


def _validate_manifest(rank: int, manifest: Any) -> dict[str, Any]:
    where = f"rank {rank}"
    if not isinstance(manifest, Mapping) or manifest.get("schema") != COMPILE_MANIFEST_SCHEMA:
        raise SameHostCompileMismatch(f"{where}: a current CUDA compile manifest is required")
    platform = manifest.get("compile_platform")
    if not isinstance(platform, Mapping):
        raise SameHostCompileMismatch(f"{where}: measured compile platform is missing")
    try:
        fingerprint = validate_compile_platform_fingerprint(platform.get("fingerprint"))
    except CompileContractError as error:
        raise SameHostCompileMismatch(f"{where}: {error}") from error
    digest = _digest(platform.get("sha256"), f"{where} platform")
    if canonical_sha256(fingerprint) != digest:
        raise SameHostCompileMismatch(f"{where}: compiler platform digest contradicts its document")
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise SameHostCompileMismatch(f"{where}: no compiled module inventory was recorded")
    origins = manifest.get("shape_origins", {})
    if not isinstance(origins, Mapping):
        raise SameHostCompileMismatch(f"{where}: shape origins are not a mapping")
    normalized = {}
    shapes: dict[str, list[dict[str, Any]]] = {}
    for key, entry in modules.items():
        try:
            validate_module_key(key)
        except CompileContractError as error:
            raise SameHostCompileMismatch(f"{where}: {error}") from error
        label = f"{where} module {key!r}"
        if not isinstance(entry, Mapping):
            raise SameHostCompileMismatch(f"{label}: module entry is not a mapping")
        source = _digest(entry.get("source_sha256"), f"{label} source")
        options = _strings(entry.get("requested_options"), f"{label} requested options")
        kernels = _strings(entry.get("resolved_kernels"), f"{label} kernel inventory")
        if not kernels or len(set(kernels)) != len(kernels):
            raise SameHostCompileMismatch(f"{label}: kernel inventory is empty or contains duplicates")
        if entry.get("compile_platform_fingerprint_sha256") != digest:
            raise SameHostCompileMismatch(f"{label}: module compiler binding contradicts its platform")
        flags, images = _observations(entry, label)
        shape = _SHAPE_KEY.fullmatch(key)
        if shape is not None:
            # The rank's dimensions are its own by design, so its specialized
            # source and image differ from the other rank's. What must agree is
            # the generic source, the dimension names, options, compiler and
            # kernel inventory; the key must name its own specialized source.
            if shape["tag"] != source[:20]:
                raise SameHostCompileMismatch(f"{label}: shape key does not name its specialized source")
            origin = origins.get(key)
            if not isinstance(origin, Mapping):
                raise SameHostCompileMismatch(f"{label}: shape module has no recorded generic source")
            generic = _digest(origin.get("generic_source_sha256"), f"{label} generic source")
            dimensions = origin.get("dimensions")
            if (not isinstance(dimensions, list) or not dimensions or any(
                    not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                    or isinstance(item[1], bool) or not isinstance(item[1], int) for item in dimensions)):
                raise SameHostCompileMismatch(f"{label}: shape dimensions are malformed")
            shapes.setdefault(shape["base"], []).append({
                "key": key,
                "identity": {"generic_source_sha256": generic,
                    "dimension_names": [item[0] for item in dimensions],
                    "requested_options": list(options),
                    "compile_platform_fingerprint_sha256": digest, "resolved_kernels": sorted(kernels)},
                "dimensions": {item[0]: item[1] for item in dimensions},
                "flags": flags, "images": images,
            })
            continue
        normalized[key] = {
            "identity": {"source_sha256": source, "requested_options": list(options),
                "compile_platform_fingerprint_sha256": digest, "resolved_kernels": sorted(kernels)},
            "flags": flags, "images": images,
            "module_cache_key": entry.get("module_cache_key"),
        }
    for base, variants in shapes.items():
        generic = normalized.get(base)
        if generic is None:
            continue
        for variant in variants:
            if (variant["identity"]["generic_source_sha256"] != generic["identity"]["source_sha256"]
                    or variant["identity"]["requested_options"] != generic["identity"]["requested_options"]):
                raise SameHostCompileMismatch(
                    f"{where} module {variant['key']!r}: shape module does not derive from its generic module {base!r}")
    return {"platform": {"fingerprint": fingerprint, "sha256": digest}, "modules": normalized, "shapes": shapes}


def _shape_identities(variants: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(variant["identity"], sort_keys=True) for variant in variants)


def compare_samehost_compile_manifests(manifests: Mapping[int, Any]) -> dict[str, Any]:
    """Return evidence of compatible rank compiles, refusing contradictions.

    Cache bookkeeping and observation counts are not numerical inputs.
    Original manifests are preserved in the receipt. Captured PTX equality
    is distinguished from captured CUBIN equality; neither records a digest
    of the final device-loaded kernel, so loaded binary identity is unproved.
    """
    if not isinstance(manifests, Mapping) or not manifests:
        raise SameHostCompileMismatch("no rank compile manifests were supplied")
    originals = {}
    normalized = {}
    for rank, manifest in manifests.items():
        try:
            ordinal = operator.index(rank)
        except TypeError as error:
            raise SameHostCompileMismatch("rank ordinals must be integers") from error
        if ordinal < 0 or ordinal in normalized:
            raise SameHostCompileMismatch("rank ordinals must be unique and nonnegative")
        normalized[ordinal] = _validate_manifest(ordinal, manifest)
        try:
            originals[str(ordinal)] = json.loads(json.dumps(manifest))
        except (TypeError, ValueError) as error:
            raise SameHostCompileMismatch(f"rank {ordinal}: manifest cannot be preserved as JSON") from error
    ranks = sorted(normalized)
    reference = normalized[ranks[0]]
    for rank in ranks[1:]:
        current = normalized[rank]
        if current["platform"] != reference["platform"]:
            raise SameHostCompileMismatch(f"rank {rank}: measured compiler platform differs from rank {ranks[0]}")
        if set(current["modules"]) != set(reference["modules"]):
            raise SameHostCompileMismatch(f"rank {rank}: compiled module inventory differs from rank {ranks[0]}")
        for key, entry in current["modules"].items():
            if entry["identity"] != reference["modules"][key]["identity"]:
                differing = [field for field in entry["identity"] if entry["identity"][field] != reference["modules"][key]["identity"][field]]
                raise SameHostCompileMismatch(f"rank {rank} module {key!r}: stable compile inputs differ at {differing}")
        if set(current["shapes"]) != set(reference["shapes"]):
            raise SameHostCompileMismatch(f"rank {rank}: shape-specialized module inventory differs from rank {ranks[0]}")
        for base, variants in current["shapes"].items():
            if _shape_identities(variants) != _shape_identities(reference["shapes"][base]):
                raise SameHostCompileMismatch(
                    f"rank {rank} shape modules of {base!r}: generic source, dimension names, options or kernels differ")
    modules = {}
    for key, reference_entry in sorted(reference["modules"].items()):
        entries = {rank: normalized[rank]["modules"][key] for rank in ranks}
        comparisons = []
        observed_flags = {entry["flags"] for entry in entries.values() if entry["flags"] is not None}
        if len(observed_flags) > 1:
            raise SameHostCompileMismatch(f"module {key!r}: mutually observed effective NVRTC flags differ")
        for left, right in combinations(ranks, 2):
            left_images, right_images = entries[left]["images"], entries[right]["images"]
            for kind in sorted(set(left_images) & set(right_images)):
                if left_images[kind] != right_images[kind]:
                    raise SameHostCompileMismatch(f"module {key!r}: captured {kind} image differs between ranks {left} and {right}")
                comparisons.append({"left_rank": left, "right_rank": right,
                    "kind": kind, "sha256": left_images[kind]})
        common_kinds = set.intersection(*(set(entry["images"]) for entry in entries.values()))
        cache_keys = {str(rank): entry["module_cache_key"] for rank, entry in entries.items()}
        modules[key] = {
            "inputs": reference_entry["identity"], "module_cache_keys": cache_keys,
            "cache_keys_identical": all(value == next(iter(cache_keys.values())) for value in cache_keys.values()),
            "effective_flags": list(next(iter(observed_flags))) if observed_flags else None,
            "ranks_without_effective_flags": [rank for rank in ranks if entries[rank]["flags"] is None],
            "captured_images": {str(rank): entries[rank]["images"] for rank in ranks},
            "ranks_without_captured_images": [rank for rank in ranks if not entries[rank]["images"]],
            "mutually_compared_images": comparisons,
            "all_rank_common_image_kinds": sorted(common_kinds) if len(ranks) > 1 else [],
        }
    shape_modules = {}
    for base in sorted(reference["shapes"]):
        observed_flags = {variant["flags"] for rank in ranks for variant in normalized[rank]["shapes"][base]
                          if variant["flags"] is not None}
        if len(observed_flags) > 1:
            raise SameHostCompileMismatch(f"shape modules of {base!r}: mutually observed effective NVRTC flags differ")
        shape_modules[base] = {
            "inputs": [json.loads(item) for item in _shape_identities(reference["shapes"][base])],
            "effective_flags": list(next(iter(observed_flags))) if observed_flags else None,
            "rank_modules": {str(rank): [{"key": variant["key"], "dimensions": variant["dimensions"],
                "captured_images": variant["images"]} for variant in normalized[rank]["shapes"][base]]
                for rank in ranks},
        }
    return {
        "schema": "mpas-port.samehost-compile-comparison/v1", "ranks": ranks,
        "inputs_compatible": True, "compile_platform": reference["platform"],
        "modules": modules, "shape_modules": shape_modules,
        "shape_module_images_law": "each rank bakes its own dimensions into a shape module, so its source and image are its own; the generic source, dimension names, options, compiler and kernels are compared",
        "rank_manifests": originals,
        "captured_image_identity_complete": len(ranks) > 1 and all(entry["all_rank_common_image_kinds"] for entry in modules.values()),
        "captured_cubin_identity_complete": len(ranks) > 1 and all("cubin" in entry["all_rank_common_image_kinds"] for entry in modules.values()),
        "loaded_kernel_binary_identity_proved": False,
        "binary_evidence_law": "only mutually captured images of the same kind are compared; cache hits and missing images remain unproved, and device-loaded kernel binaries are not observed",
    }
