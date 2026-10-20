"""SHA-256 of large input files, computed once per file version.

THE BREAKAGE THIS PREVENTS, MEASURED (8-card RTX PRO 6000 host, 2026-10-10): a 15 km global
forecast re-read and re-hashed about 42 GB of unchanged bytes before its first
step -- the 3.6 GB grid and 5.0 GB static at the mesh-row bind and again at
the authority check, and the 24.6 GB init that the init door had hashed a
minute earlier -- about 55 s of the 191 s host start (prep.door 25.8 s, and
29.5 s of prep.host before the first cache load).  Every digest is still the
file's real SHA-256; this only stops computing the same one twice.

A digest is reused only for the same file VERSION: same device, inode, size,
modification time and status-change time (st_ctime_ns, which no user call can
set back).  Rewriting, truncating, replacing or touching a file changes at
least one of them and the next call hashes it again.  Small files are always
hashed directly.

``GPUWM_HEX_DIGEST_CACHE`` names the cache directory (default
``~/.cache/gpuwm-hex/digests``); ``off`` disables reuse entirely.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

#: Below this size a file is hashed every time: the stat and the cache entry
#: cost more than the read.
MIN_CACHED_BYTES = 64 * 1024 * 1024
#: A version modified this recently is never remembered ("racily clean").
RACY_SECONDS = 2.0
_BLOCK = 16 * 1024 * 1024


def _cache_dir() -> Path | None:
    text = os.environ.get("GPUWM_HEX_DIGEST_CACHE", "")
    if text.lower() in ("off", "0", "none"):
        return None
    return Path(text).expanduser() if text else Path.home() / ".cache" / "gpuwm-hex" / "digests"


def _version(stat: os.stat_result) -> str:
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"


def _entry(cache: Path, path: Path) -> Path:
    return cache / (hashlib.sha256(str(path).encode("utf-8")).hexdigest() + ".json")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def lookup(path: str | Path) -> str | None:
    """The cached digest of this exact file version, or None."""
    cache = _cache_dir()
    selected = Path(path).resolve()
    if cache is None:
        return None
    try:
        stat = selected.stat()
        record = json.loads(_entry(cache, selected).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("version") == _version(stat) and record.get("path") == str(selected):
        return str(record["sha256"])
    return None


def record(path: str | Path, sha256: str, stat: os.stat_result | None = None) -> None:
    """Remember ``sha256`` for the file version ``stat`` (taken before hashing)."""
    cache = _cache_dir()
    selected = Path(path).resolve()
    if cache is None:
        return
    stat = stat or selected.stat()
    if stat.st_size < MIN_CACHED_BYTES:
        return
    if time.time() - stat.st_mtime < RACY_SECONDS:
        # Too fresh to trust: a second write inside the same timestamp tick
        # would leave size, mtime and ctime unchanged (measured on the 8-card RTX PRO 6000 host
        # overlay: two writes 1 ms apart shared both stamps), so the entry
        # is only made once the version is older than any tick.
        return
    try:
        cache.mkdir(parents=True, exist_ok=True)
        entry = _entry(cache, selected)
        partial = entry.with_name(f"{entry.name}.{os.getpid()}.part")
        partial.write_text(json.dumps({"path": str(selected), "version": _version(stat),
                                       "sha256": sha256}), encoding="utf-8")
        os.replace(partial, entry)
    except OSError:
        pass  # a cache that cannot be written only costs the next run a read


def sha256_file(path: str | Path) -> str:
    """The file's SHA-256, reused when this exact file version was hashed before."""
    selected = Path(path)
    stat = selected.stat()
    if stat.st_size >= MIN_CACHED_BYTES:
        cached = lookup(selected)
        if cached is not None:
            return cached
    digest = _hash(selected)
    # Only remember the digest if the file did not change while it was read.
    after = selected.stat()
    if _version(after) == _version(stat):
        record(selected, digest, stat)
    return digest
