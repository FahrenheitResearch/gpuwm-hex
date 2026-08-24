"""Canonical serialization, hashing, paths, and UTC helpers.

No result file written by the referee contains wall-clock time, temporary paths,
random UUIDs, or platform-dependent JSON formatting.  That makes identical
inputs produce byte-identical evidence directories.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .errors import IntegrityError, SchemaError


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict, sorted UTF-8 JSON terminated by one LF."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not canonical-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_pretty_json_bytes(value: Any) -> bytes:
    """Return stable human-readable JSON terminated by one LF."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"value is not strict-JSON serializable: {exc}") from exc
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | os.PathLike[str], expected: str) -> str:
    if not isinstance(expected, str) or not _is_sha256(expected):
        raise SchemaError(f"expected SHA-256 is not 64 lowercase hexadecimal characters: {expected!r}")
    actual = sha256_file(path)
    if actual != expected:
        raise IntegrityError(
            f"SHA-256 mismatch for {Path(path).name}: expected {expected}, got {actual}"
        )
    return actual


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path: str | os.PathLike[str], value: Any, *, pretty: bool = True) -> None:
    data = canonical_pretty_json_bytes(value) if pretty else canonical_json_bytes(value)
    atomic_write_bytes(path, data)


def read_json(path: str | os.PathLike[str]) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise SchemaError(f"cannot read JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON in {path}: {exc}") from exc


def parse_utc(value: str, *, name: str = "time") -> datetime:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name} must be a non-empty ISO-8601 UTC string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SchemaError(f"{name} is not ISO-8601: {value!r}") from exc
    if result.tzinfo is None:
        raise SchemaError(f"{name} must include a UTC offset")
    result = result.astimezone(timezone.utc)
    return result


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise SchemaError("cannot format a naive datetime as UTC")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def unix_seconds(value: str, *, name: str = "time") -> int:
    return int(parse_utc(value, name=name).timestamp())


def resolve_path(
    raw: str,
    *,
    manifest_dir: Path,
    variables: dict[str, str] | None = None,
) -> Path:
    """Resolve a manifest path without invoking a shell.

    `${NAME}` expansion is limited to the explicit mapping plus environment
    variables. Undefined names are a refusal, not an empty string.
    """

    if not isinstance(raw, str) or not raw:
        raise SchemaError("artifact path must be a non-empty string")
    mapping = dict(os.environ)
    if variables:
        mapping.update(variables)

    pattern = re_compile_env()

    def replace(match: Any) -> str:
        name = match.group(1)
        if name not in mapping:
            raise SchemaError(f"undefined path variable ${{{name}}} in {raw!r}")
        return mapping[name]

    expanded = pattern.sub(replace, raw)
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


_ENV_PATTERN = None


def re_compile_env():
    global _ENV_PATTERN
    if _ENV_PATTERN is None:
        import re
        _ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
    return _ENV_PATTERN


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big", signed=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)
