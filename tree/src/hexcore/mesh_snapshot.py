"""Serve a forecast's dynamics mesh from a verified snapshot of the same grid/static bytes.

THE COST THIS REMOVES, MEASURED (rented box, 2026-10-13, 20 km global, 1,472,532
cells, warm box, main thread sampled): ``load_precision_preserving_mesh_pair``
took 44.8 s of every forecast's host preparation -- three netCDF reads of the
grid and static (about 7 GB), connectivity canonicalisation, and four whole-
mesh validations (25.5 s of them) -- while every card waited for the first
step.  The answer is a function of the grid and static bytes and of this
port's code only, and a production line runs the same mesh every cycle with a
new init, so every cycle after the first repeated it for nothing.

WHAT A SNAPSHOT IS.  The dynamics mesh and its evidence record exactly as the
loader returned them -- every array with its dtype, shape, memory order,
writeable flag and bytes, and every attribute, dimension, source label and
provenance entry with its Python type and the object sharing between them
(``mesh.variables is mesh.arrays``; the evidence record is the same object
the mesh's provenance names) -- written right after the loader returned and
before any caller overlays anything.  A snapshot is served only for:

* the same grid and static bytes, by their real SHA-256
  (:mod:`hexcore.file_digest`), at the same resolved paths (the mesh records
  them);
* the same producer: :func:`hexcore.prep_cache.producer_digest` (every
  ``hexcore`` source file, the NumPy build, the interpreter, the C library and
  NumPy's CPU dispatch), plus the netCDF4 and libnetcdf versions that decoded
  the files.

It is the loader's own validated result, so serving it skips the
re-validation of bytes that already passed it; a different byte anywhere is a
different key and the loader runs (and validates) exactly as before.

THE GUARDS, AND THE BREAKAGE EACH PREVENTS.
* Every stored array carries its SHA-256 and is re-hashed on load; a mismatch
  discards the snapshot and the loader runs.  Prevents: a truncated write or a
  flipped bit feeding wrong geometry into a forecast that would otherwise run
  to completion and look plausible.
* Only plain data is stored (arrays, numbers, strings, containers); loading
  never unpickles.  A value of any other type makes the mesh unsnapshotable
  and the loader simply runs.  Prevents: a cache directory executing code.
* Entries live in the host-preparation cache's directory and layout, so its
  size cap and least-recently-used eviction (``GPUWM_HEX_PREP_CACHE_MAX_GB``)
  bound them too.  Prevents: one multi-GB entry per mesh filling the disk.

``GPUWM_HEX_MESH_SNAPSHOT=off`` (or the host-preparation cache being off)
runs the loader every time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePath
import shutil
import tempfile
import threading
import time
from typing import Any

import numpy as np

from . import prep_cache, time_ledger

SCHEMA = "gpuwm-hex.mesh-snapshot/v1"
ENVIRONMENT = "GPUWM_HEX_MESH_SNAPSHOT"
LABEL = "mesh-snapshot:precision-preserving-dynamics-mesh"

_lock = threading.Lock()
_ledger: list[dict[str, Any]] = []


class Unencodable(Exception):
    """A value with no plain-data encoding; the loader runs instead."""


def enabled() -> bool:
    flag = os.environ.get(ENVIRONMENT, "").strip().lower()
    return flag not in ("off", "0", "no", "false") and prep_cache.enabled()


def _note(row: dict[str, Any]) -> None:
    with _lock:
        _ledger.append(row)


def receipt() -> dict[str, Any]:
    with _lock:
        return {"schema": SCHEMA, "enabled": enabled(), "calls": [dict(row) for row in _ledger]}


def _workers() -> int:
    from .host_parallel import threads

    return threads()


def _netcdf_identity() -> list[str]:
    import netCDF4

    return [str(getattr(netCDF4, "__version__", "?")),
            str(getattr(netCDF4, "__netcdf4libversion__", "?"))]


def snapshot_key(grid: Path, static: Path) -> str:
    from .file_digest import sha256_file

    digest = hashlib.sha256()
    parts = [SCHEMA, prep_cache.producer_digest(), *_netcdf_identity()]
    for path in (grid, static):
        parts += [str(path), str(path.stat().st_size), sha256_file(path)]
    for part in parts:
        digest.update(part.encode("utf-8") + b"\0")
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# plain-data encoding with object sharing
# ---------------------------------------------------------------------------
class _Encoder:
    def __init__(self) -> None:
        self.arrays: list[np.ndarray] = []
        self._shared: dict[int, int] = {}
        self._keep: list[Any] = []   # holds encoded objects so ids stay unique
        self._next = 0

    def _share(self, value: Any) -> tuple[int | None, int]:
        marker = id(value)
        if marker in self._shared:
            return self._shared[marker], -1
        index = self._next
        self._next += 1
        self._shared[marker] = index
        self._keep.append(value)
        return None, index

    def encode(self, value: Any, depth: int = 0) -> Any:
        if depth > 16:
            raise Unencodable("nesting deeper than the snapshot walks")
        if value is None or isinstance(value, bool):
            return {"v": value}
        if isinstance(value, int) and type(value) is int:
            return {"i": str(value)}
        if type(value) is float:
            return {"f": value.hex()}
        if type(value) is str:
            return {"s": value}
        if type(value) is bytes:
            return {"y": value.hex()}
        if isinstance(value, PurePath):
            return {"p": [type(value).__name__, str(value)]}
        if isinstance(value, np.generic):
            if value.dtype.hasobject:
                raise Unencodable("object-dtype scalar")
            return {"g": [value.dtype.str, value.tobytes().hex()]}
        if type(value) is np.ndarray:
            if value.dtype.hasobject:
                raise Unencodable("object-dtype array")
            if value.base is not None:
                # A view shares memory with another array; a stored copy would
                # not, so a write through one would stop reaching the other.
                raise Unencodable("array view")
            seen, index = self._share(value)
            if seen is not None:
                return {"ref": seen}
            self.arrays.append(value)
            order = "F" if (value.flags.f_contiguous and not value.flags.c_contiguous) else "C"
            return {"a": len(self.arrays) - 1, "id": index, "order": order,
                    "writeable": bool(value.flags.writeable)}
        if type(value) in (dict, list):
            seen, index = self._share(value)
            if seen is not None:
                return {"ref": seen}
            if type(value) is dict:
                if not all(type(key) is str for key in value):
                    raise Unencodable("mapping with non-string keys")
                return {"d": [[key, self.encode(item, depth + 1)] for key, item in value.items()],
                        "id": index}
            return {"l": [self.encode(item, depth + 1) for item in value], "id": index}
        if type(value) is tuple:
            return {"t": [self.encode(item, depth + 1) for item in value]}
        raise Unencodable(f"value of type {type(value).__module__}.{type(value).__qualname__}")


def _decode(node: Any, arrays: list[np.ndarray], shared: dict[int, Any]) -> Any:
    if "v" in node:
        return node["v"]
    if "i" in node:
        return int(node["i"])
    if "f" in node:
        return float.fromhex(node["f"])
    if "s" in node:
        return node["s"]
    if "y" in node:
        return bytes.fromhex(node["y"])
    if "p" in node:
        kind, text = node["p"]
        import pathlib

        return getattr(pathlib, kind)(text)
    if "g" in node:
        dtype, raw = node["g"]
        return np.frombuffer(bytes.fromhex(raw), dtype=np.dtype(dtype))[0]
    if "ref" in node:
        return shared[node["ref"]]
    if "a" in node:
        array = arrays[node["a"]]
        if not node["writeable"]:
            array.flags.writeable = False
        shared[node["id"]] = array
        return array
    if "d" in node:
        result: dict[str, Any] = {}
        shared[node["id"]] = result
        for key, item in node["d"]:
            result[key] = _decode(item, arrays, shared)
        return result
    if "l" in node:
        result_list: list[Any] = []
        shared[node["id"]] = result_list
        result_list.extend(_decode(item, arrays, shared) for item in node["l"])
        return result_list
    if "t" in node:
        return tuple(_decode(item, arrays, shared) for item in node["t"])
    raise ValueError(f"unknown mesh-snapshot node {sorted(node)}")


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


# ---------------------------------------------------------------------------
# the disk layer (the prep cache's directory and entry layout)
# ---------------------------------------------------------------------------
def _entry(root: Path, key: str) -> Path:
    return root / key[:2] / key


def _store(root: Path, key: str, structure: Any, arrays: list[np.ndarray]) -> None:
    final = _entry(root, key)
    if final.exists():
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{key[:12]}.", dir=final.parent))
    try:
        def write(index: int) -> dict[str, Any]:
            array = arrays[index]
            np.save(staging / f"{index}.npy", array, allow_pickle=False)
            return {"dtype": array.dtype.str, "shape": list(array.shape), "sha256": _digest(array)}

        with ThreadPoolExecutor(max_workers=max(1, min(_workers(), len(arrays) or 1)),
                                thread_name_prefix="hex-mesh-snapshot") as pool:
            rows = list(pool.map(write, range(len(arrays))))
        manifest = {
            "schema": SCHEMA, "key": key, "label": LABEL,
            "producer_digest": prep_cache.producer_digest(),
            "structure": structure, "arrays": rows,
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        try:
            os.rename(staging, final)
        except OSError:
            pass  # another writer published the same key first; same bytes
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    prep_cache.evict(root, {key})


def _load(root: Path, key: str) -> tuple[Any, list[np.ndarray]] | None:
    entry = _entry(root, key)
    manifest_path = entry / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("key") != key:
            raise ValueError("manifest does not describe this key")
        rows = manifest["arrays"]

        def read(index: int) -> np.ndarray:
            array = np.load(entry / f"{index}.npy", allow_pickle=False)
            row = rows[index]
            if (array.dtype.str != row["dtype"] or list(array.shape) != row["shape"]
                    or _digest(array) != row["sha256"]):
                raise ValueError(f"stored array {index} does not match its recorded digest")
            return array

        with ThreadPoolExecutor(max_workers=max(1, min(_workers(), len(rows) or 1)),
                                thread_name_prefix="hex-mesh-snapshot") as pool:
            arrays = list(pool.map(read, range(len(rows))))
    except Exception as error:  # a damaged entry is discarded and the loader runs
        _note({"event": "discarded", "key": key, "reason": f"{type(error).__name__}: {error}"})
        shutil.rmtree(entry, ignore_errors=True)
        return None
    try:
        os.utime(manifest_path)  # least-recently-used order for the size cap
    except OSError:
        pass
    return manifest["structure"], arrays


def _restore_orders(node: Any, arrays: list[np.ndarray]) -> None:
    """np.save keeps Fortran order; a C array stays C.  Nothing to convert,
    but a recorded order that disagrees with the loaded one is a damaged entry."""

    stack = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if "a" in item:
                array = arrays[item["a"]]
                fortran = array.flags.f_contiguous and not array.flags.c_contiguous
                if ("F" if fortran else "C") != item["order"]:
                    raise ValueError("stored array memory order differs from the recorded order")
            stack.extend(value for value in item.values() if isinstance(value, (dict, list)))
        elif isinstance(item, list):
            stack.extend(value for value in item if isinstance(value, (dict, list)))


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------
def load_dynamics_mesh(grid_path: str | Path, static_path: str | Path) -> tuple[Any, dict[str, Any]]:
    """``load_precision_preserving_mesh_pair``'s dynamics mesh and evidence, served
    from a verified snapshot of the same bytes when one exists."""

    from .mesh import Mesh, load_precision_preserving_mesh_pair

    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    root = prep_cache.cache_root() if enabled() else None
    if root is None:
        mesh, output_mesh, evidence = load_precision_preserving_mesh_pair(grid, static)
        del output_mesh
        return mesh, evidence
    started = time.perf_counter()
    key = snapshot_key(grid, static)
    time_ledger.emit("prep.cache_load", "begin", attrs={"key": key, "label": LABEL})
    try:
        loaded = _load(root, key)
        if loaded is not None:
            structure, arrays = loaded
            try:
                _restore_orders(structure, arrays)
                state, evidence = _decode(structure, arrays, {})
                mesh = Mesh.__new__(Mesh)
                mesh.__dict__.update(state)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                _note({"event": "discarded", "key": key, "reason": f"{type(error).__name__}: {error}"})
                shutil.rmtree(_entry(root, key), ignore_errors=True)
            else:
                _note({"event": "hit", "key": key, "arrays": len(arrays),
                       "bytes": int(sum(array.nbytes for array in arrays)),
                       "seconds": round(time.perf_counter() - started, 3)})
                return mesh, evidence
    finally:
        time_ledger.emit("prep.cache_load", "end", attrs={"key": key, "label": LABEL})
    computed = time.perf_counter()
    mesh, output_mesh, evidence = load_precision_preserving_mesh_pair(grid, static)
    del output_mesh
    compute_seconds = time.perf_counter() - computed
    try:
        encoder = _Encoder()
        structure = encoder.encode((dict(vars(mesh)), evidence))
        _store(root, key, structure, encoder.arrays)
        stored = True
        reason = None
    except (Unencodable, OSError) as error:
        stored = False
        reason = f"{type(error).__name__}: {error}"
    _note({"event": "miss", "key": key, "stored": stored, "reason": reason,
           "compute_seconds": round(compute_seconds, 3),
           "seconds": round(time.perf_counter() - started, 3)})
    return mesh, evidence
