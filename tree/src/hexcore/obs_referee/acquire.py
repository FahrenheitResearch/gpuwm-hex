"""Fail-closed acquisition and external-producer bridge.

The referee can cache already-normalized artifacts or invoke an explicitly
configured producer command. It never implements a second raw MRMS/ASOS parser.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.request import Request, urlopen

from .canonical import require_sha256, resolve_path, sha256_file, write_json
from .errors import IntegrityError, ProducerError, SchemaError
from .manifest import Manifest


def run_producer(
    manifest: Manifest,
    source: Mapping[str, Any],
    *,
    case_id: str,
    arm_id: str | None = None,
    timeout_seconds: int = 3600,
) -> Path:
    """Run a manifest-owned command without a shell.

    Placeholders are limited to `{output}`, `{receipt}`, `{case_id}`, and
    `{arm_id}`. The command must atomically create the canonical artifact and
    its normalization receipt. Existing valid output is not overwritten.
    """

    command = source.get("producer_command")
    if command is None:
        raise ProducerError("source has no producer_command")
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise SchemaError("producer_command must be a string list")
    output = resolve_path(
        str(source["path"]),
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    receipt_raw = str(source.get("receipt", f"{source['path']}.receipt.json"))
    receipt = resolve_path(
        receipt_raw,
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    substitutions = {
        "output": str(output),
        "receipt": str(receipt),
        "case_id": case_id,
        "arm_id": arm_id or "",
    }
    argv = [_format_token(str(token), substitutions) for token in command]
    environment = _minimal_environment()
    for key, value in dict(source.get("producer_environment", {})).items():
        if key.startswith("GPUWM_GF_"):
            raise ProducerError(
                "observation/model producer_environment may not smuggle treatment variables"
            )
        environment[str(key)] = _format_token(str(value), substitutions)

    if output.exists() and receipt.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        argv,
        cwd=manifest.directory,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace")[-4000:]
        raise ProducerError(
            f"producer exited {completed.returncode} for {case_id}/{arm_id or 'obs'}: {stderr}"
        )
    if not output.is_file() or not receipt.is_file():
        raise ProducerError(
            f"producer succeeded but did not create {output.name} and {receipt.name}"
        )
    return output


class ContentAddressedCache:
    """Small SHA-256 cache with exclusive locks and atomic publication."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.objects = self.root / "objects"
        self.receipts = self.root / "receipts"
        self.locks = self.root / "locks"

    def object_path(self, digest: str) -> Path:
        _validate_digest(digest)
        return self.objects / digest[:2] / digest[2:]

    def fetch(
        self,
        *,
        url: str,
        expected_sha256: str,
        online: bool,
        timeout_seconds: int = 120,
    ) -> Path:
        destination = self.object_path(expected_sha256)
        if destination.exists():
            require_sha256(destination, expected_sha256)
            return destination
        if not online:
            raise IntegrityError(
                f"offline cache miss for {expected_sha256}; network was not authorized"
            )
        self.locks.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.locks / f"{expected_sha256}.lock"):
            if destination.exists():
                require_sha256(destination, expected_sha256)
                return destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{expected_sha256}.", suffix=".part", dir=destination.parent
            )
            os.close(fd)
            temp = Path(temp_name)
            try:
                request = Request(url, headers={"User-Agent": "gpuwm-hex-obs-referee/1"})
                with urlopen(request, timeout=timeout_seconds) as response, temp.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                    response_url = response.geturl()
                    content_type = response.headers.get("Content-Type")
                    etag = response.headers.get("ETag")
                require_sha256(temp, expected_sha256)
                os.replace(temp, destination)
                self.receipts.mkdir(parents=True, exist_ok=True)
                write_json(
                    self.receipts / f"{expected_sha256}.json",
                    {
                        "schema": "gpuwm-hex.cache-fetch/v1",
                        "sha256": expected_sha256,
                        "requested_url": url,
                        "response_url": response_url,
                        "content_type": content_type,
                        "etag": etag,
                        "size_bytes": destination.stat().st_size,
                    },
                )
            finally:
                temp.unlink(missing_ok=True)
        return destination


def link_cached_object(cache_path: Path, destination: Path) -> None:
    """Publish a cached object by hard link when possible, copy otherwise."""

    if destination.exists():
        if sha256_file(destination) != sha256_file(cache_path):
            raise IntegrityError(f"refusing to overwrite non-identical {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cache_path, destination)
    except OSError:
        shutil.copyfile(cache_path, destination)


def _format_token(value: str, substitutions: Mapping[str, str]) -> str:
    try:
        return value.format_map(_StrictFormatMap(substitutions))
    except (KeyError, ValueError) as exc:
        raise SchemaError(f"invalid producer placeholder in {value!r}: {exc}") from exc


class _StrictFormatMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _minimal_environment() -> dict[str, str]:
    keep = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "WINDIR")
    result = {key: os.environ[key] for key in keep if key in os.environ}
    result["PYTHONHASHSEED"] = "0"
    result["TZ"] = "UTC"
    return result


@contextmanager
def _exclusive_lock(path: Path, *, timeout_seconds: int = 300) -> Iterator[None]:
    start = time.monotonic()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() - start > timeout_seconds:
                raise IntegrityError(f"timed out waiting for cache lock {path.name}")
            time.sleep(0.1)
    try:
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            os.close(fd)
        path.unlink(missing_ok=True)


def _validate_digest(value: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SchemaError("expected_sha256 must be lowercase SHA-256")
