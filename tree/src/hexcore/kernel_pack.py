"""The kernel-pack codec: compiler requests, and optionally their cubins, bound to one compiler.

THE BREAKAGE THIS ENDS.  ``hexcore.cuda_backend.aot`` read its shipped pack
through ``gpuwm.kernel_pack``, a module no published gpuwm ever carried.  The
import failed on every install, so the wheel build skipped the pack and every
forecast logged ``pack_loader_unavailable`` and compiled every CUDA module
through NVRTC: 206 s to the first step on a fresh 2x RTX PRO 6000 box on
2026-10-16 against 16 s warm.  The codec is pure JSON and belongs with the
pack it reads, so it lives here.

THE DOCUMENT (schema ``gpuwm.kernel-pack/v1``, the same envelope
``hexcore.kernel_prewarm`` writes for its replay manifest)::

    {"schema": "gpuwm.kernel-pack/v1",
     "compiler": {...the identity the entries were produced under...},
     "entries": [{"sha256": <sha256 of the canonical JSON of arguments>,
                  "arguments": {...one row...}}, ...]}

:func:`read` refuses (``ValueError``) a document of another schema, one bound
to a different compiler identity, or one whose rows do not hash to their
recorded digests, so an edited or stale row can never reach a kernel load.
A path ending in ``.xz`` is LZMA-compressed on disk.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
from pathlib import Path
import tempfile

SCHEMA = "gpuwm.kernel-pack/v1"
MAX_ENTRIES = 16000


#: The identity fields that decide what a compile produces: the loaded NVRTC
#: (its version and the SHA-256 of the library bytes) and CuPy's own key
#: inputs.  ``cupy_package`` and ``nvrtc_package`` are pip metadata: they name
#: what to install, not what is loaded.  THE BREAKAGE THIS ENDS
#: (2026-10-24): a venv whose CuPy loaded the system CUDA 13.0 libnvrtc while
#: pip metadata named another NVRTC, or one that carried the same library bytes
#: under a different wheel, never matched any pack and compiled every module
#: on one core (about 4 min per new box).
CODE_KEYS = ("cupy", "nvrtc", "nvrtc_library_sha256", "cache_key", "use_ptx", "use_pch", "debug")


def same_compiler(built, measured) -> bool:
    """Whether a pack's recorded compiler produces what this process's compiler produces."""
    if not isinstance(built, dict) or not isinstance(measured, dict):
        return False
    if any(key not in built or key not in measured for key in CODE_KEYS):
        return built == measured
    return all(built[key] == measured[key] for key in CODE_KEYS)


def digest(arguments) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True).encode("utf-8")).hexdigest()


def _load(path: Path):
    data = Path(path).read_bytes()
    if str(path).endswith(".xz"):
        data = lzma.decompress(data)
    return json.loads(data.decode("utf-8"))


def compiler_of(path) -> dict | None:
    """The identity a pack was built under, or None when the file is unreadable."""
    try:
        document = _load(Path(path))
    except (OSError, ValueError, lzma.LZMAError):
        return None
    compiler = document.get("compiler") if isinstance(document, dict) else None
    return compiler if isinstance(compiler, dict) else None


def read(path, compiler) -> list[dict]:
    document = _load(Path(path))
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} document")
    if not same_compiler(document.get("compiler"), compiler):
        raise ValueError(f"{path}: built for compiler {document.get('compiler')!r}, "
                         f"this process has {compiler!r}")
    records = document.get("entries")
    if not isinstance(records, list) or len(records) > MAX_ENTRIES:
        raise ValueError(f"{path}: entries missing or over {MAX_ENTRIES}")
    rows = []
    for record in records:
        arguments = record.get("arguments") if isinstance(record, dict) else None
        if not isinstance(arguments, dict) or record.get("sha256") != digest(arguments):
            raise ValueError(f"{path}: an entry does not match its recorded digest")
        rows.append(arguments)
    return rows


def write(path, compiler, entries) -> None:
    path = Path(path)
    entries = list(entries)
    if len(entries) > MAX_ENTRIES:
        raise ValueError(f"{len(entries)} entries exceed the {MAX_ENTRIES} cap")
    document = {"schema": SCHEMA, "compiler": compiler,
                "entries": [{"sha256": digest(row), "arguments": row} for row in entries]}
    data = (json.dumps(document, sort_keys=True) + "\n").encode("utf-8")
    if str(path).endswith(".xz"):
        data = lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".kernel-pack.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        # mkstemp creates 0600; a pack is package data every user must read.
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
