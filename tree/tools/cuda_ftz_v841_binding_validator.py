#!/usr/bin/env python3
"""Isolated live-replay validator for one emitted v8.4.1 FTZ binding.

This file is executed only through the byte-pinned isolated bootstrap.  It
creates a caller-supplied cache root exclusively, makes that root the parent
of every temporary four-pass cache, and asks the production validator to
rebuild the complete binding from live source, compiler, GPU, and gpuwm
evidence.  Success is one exact canonical JSON document on stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
VALIDATION_SCHEMA = "mpas-port.cuda-ftz-v841-isolated-live-replay/v1"
CACHE_RECORD_SCHEMA = "mpas-port.cuda-ftz-v841-replay-cache/v1"


class ValidationError(RuntimeError):
    """The serialized binding did not survive the isolated live replay."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValidationError(f"JSON contains non-finite numeric token {value!r}")


def _load_binding(path: Path) -> dict[str, Any]:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValidationError(f"binding must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_file():
        raise ValidationError(f"binding is not a real regular file: {selected}")
    before = selected.stat(follow_symlinks=False)
    payload = selected.read_bytes()
    after = selected.stat(follow_symlinks=False)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(payload) != after.st_size
    ):
        raise ValidationError("binding changed while its exact bytes were read")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"binding is not strict UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError("binding JSON root is not an object")
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _real_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValidationError(f"{label} must not be a symlink: {raw}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise ValidationError(f"{label} is not a real directory: {selected}")
    return selected


def _create_fresh_cache(path: Path, *, protected: tuple[Path, ...]) -> Path:
    raw = path.expanduser()
    if raw.is_symlink():
        raise ValidationError(f"validation cache must not be a symlink: {raw}")
    selected = raw.resolve()
    if selected.exists():
        raise ValidationError(f"validation cache must be absent: {selected}")
    for source in protected:
        resolved = source.resolve()
        if (
            selected == resolved
            or selected in resolved.parents
            or resolved in selected.parents
        ):
            raise ValidationError(
                f"validation cache overlaps protected input {resolved}"
            )
    selected.parent.mkdir(parents=True, exist_ok=True)
    try:
        selected.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ValidationError(
            f"validation cache lost exclusive-create race: {selected}"
        ) from error
    return selected


def _write_cache_record(path: Path, payload: object) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValidationError(f"validation cache record already exists: {path}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild and live-replay one saved v8.4.1 FTZ binding."
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--gpuwm-root", type=Path, required=True)
    parser.add_argument("--gpuwm-receipt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    binding_path = args.binding.expanduser().resolve()
    gpuwm_root = _real_directory(args.gpuwm_root, label="gpuwm root")
    gpuwm_receipt = _real_directory(args.gpuwm_receipt, label="gpuwm receipt")
    cache = _create_fresh_cache(
        args.cache_dir,
        protected=(
            binding_path,
            gpuwm_root,
            gpuwm_receipt,
            ROOT / "src",
            ROOT / "tests",
            ROOT / "tools",
            ROOT / "oracle",
            ROOT / "receipts",
        ),
    )

    # tempfile is intentionally imported only after these exact settings.  The
    # production four-pass runner creates each pass with TemporaryDirectory,
    # so all fresh compile images are born under this exclusively-created root.
    for key in ("TMP", "TEMP", "TMPDIR", "CUPY_CACHE_DIR", "MPAS_PORT_CUDA_CACHE_DIR"):
        os.environ[key] = str(cache)
    import tempfile

    tempfile.tempdir = None
    if Path(tempfile.gettempdir()).resolve() != cache:
        raise ValidationError("temporary-directory parent is not the fresh cache")
    if any(cache.iterdir()):
        raise ValidationError("fresh validation cache was not born empty")

    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from mpas_port.cuda_ftz import validate_mpas_ftz_binding_v841

    binding = _load_binding(binding_path)
    binding_sha = _canonical_sha256(binding)
    rebuilt = validate_mpas_ftz_binding_v841(
        binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=gpuwm_receipt,
    )
    rebuilt_sha = _canonical_sha256(rebuilt)
    if rebuilt_sha != binding_sha or rebuilt != binding:
        raise ValidationError("live replay is not canonically identical to the binding")
    audit = rebuilt.get("kernel_audit")
    kernels = audit.get("kernels") if isinstance(audit, dict) else None
    if not isinstance(kernels, dict):
        raise ValidationError("live replay returned no exact kernel inventory")
    disabled_red = sum(
        1
        for row in kernels.values()
        if isinstance(row, dict) and row.get("mutation_red") is True
    )
    if len(kernels) != 95 or disabled_red != 78:
        raise ValidationError("live replay returned the wrong 95/78 kernel surface")

    cache_record = {
        "schema": CACHE_RECORD_SCHEMA,
        "binding_canonical_sha256": binding_sha,
        "validated_binding_canonical_sha256": rebuilt_sha,
        "temporary_directory_parent": str(cache),
        "cache_was_absent_and_exclusively_created": True,
        "four_pass_runner_requires_each_nested_cache_to_be_born_empty": True,
    }
    _write_cache_record(cache / "validation-cache.json", cache_record)
    summary = {
        "schema": VALIDATION_SCHEMA,
        "status": "live-replay-validated",
        "binding": str(binding_path),
        "binding_canonical_sha256": binding_sha,
        "validated_binding_canonical_sha256": rebuilt_sha,
        "canonical_binding_equal": True,
        "validation_cache_directory": str(cache),
        "validation_cache_record_sha256": hashlib.sha256(
            (cache / "validation-cache.json").read_bytes()
        ).hexdigest(),
        "kernel_count": len(kernels),
        "disabled_fallback_red_count": disabled_red,
        "four_pass_live_replay": True,
        "authority_claim": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
