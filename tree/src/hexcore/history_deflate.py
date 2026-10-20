"""Parallel deflate for history frames: the file HDF5 writes, on every core.

THE SINGLE-THREAD STAGE THIS REMOVES (B4, the 2026-10-16 global operations gap list,
the project rule: no single-CPU path on the critical path): a history
frame is a netCDF-4 file whose variables carry the shuffle and zlib level-1
filters, and HDF5 runs that pipeline chunk by chunk on the one thread that
called it.  A 15 km frame (11.1 GiB) spent 115 s in zlib on one core.

WHAT THIS DOES.  ``write_snapshot_netcdf`` is unchanged and still makes every
netCDF call.  Before it runs, the Rust library ``rust/history-deflate`` is
handed the frame's arrays and the chunk layout netCDF-C picks for them; its
worker threads build each chunk exactly as HDF5 will hand it to deflate and
compress it with the same ``compress2`` HDF5 links.  For the length of the
frame HDF5's shuffle and deflate entries look chunks up by content and return the
precomputed bytes; a chunk they do not hold goes to HDF5's own functions.
So the file is byte for byte the file the single-thread path writes, and a
wrong guess here costs speed, never bytes.  The receipt row of each frame
carries the counts (``hits`` precomputed, ``misses`` compressed by HDF5).

Only the background history writer process engages it
(:mod:`hexcore.history_writer`): that process does nothing but write frames,
so no other HDF5 call can meet the swapped entry.

``GPUWM_HEX_HISTORY_DEFLATE=off`` writes every frame the single-thread way.
``GPUWM_HEX_HISTORY_DEFLATE_THREADS`` sets the worker count (default: half
the CPUs this process may run on, at least 2 and at most 16), ``..._BUDGET_MIB`` the retained
precomputed chunk limit (default 2048, minimum 64 MiB). Worker scratch arrays,
input arrays and HDF5 allocations are additional memory, outside that limit.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterator, Mapping

import numpy as np

ENVIRONMENT = "GPUWM_HEX_HISTORY_DEFLATE"
LIBRARY_ENVIRONMENT = "GPUWM_HEX_HISTORY_DEFLATE_LIB"
THREADS_ENVIRONMENT = "GPUWM_HEX_HISTORY_DEFLATE_THREADS"
BUDGET_ENVIRONMENT = "GPUWM_HEX_HISTORY_DEFLATE_BUDGET_MIB"
MARKER = b"hex-history-deflate-v2"
LIBRARY_NAME = "libhex_history_deflate.so"
WRITER = "write_snapshot_netcdf"


class _Job(ctypes.Structure):
    _fields_ = [("src", ctypes.c_void_p), ("levels", ctypes.c_uint64), ("cells", ctypes.c_uint64),
                ("chunk_cells", ctypes.c_uint64), ("chunk_levels", ctypes.c_uint64),
                ("level", ctypes.c_int), ("shuffle", ctypes.c_int), ("fill_bits", ctypes.c_uint32)]


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT, "on").strip().lower() not in ("off", "0", "no", "false")


def library_path() -> Path | None:
    """``$GPUWM_HEX_HISTORY_DEFLATE_LIB``, the copy an install ships in
    ``hexcore/libexec``, or the crate's release build in a source checkout."""

    explicit = os.environ.get(LIBRARY_ENVIRONMENT)
    if explicit:
        return Path(explicit)
    shipped = Path(__file__).with_name("libexec") / LIBRARY_NAME
    if shipped.is_file():
        return shipped
    built = Path(__file__).resolve().parents[2] / "rust/history-deflate/target/release" / LIBRARY_NAME
    return built if built.is_file() else None


def _loaded_hdf5() -> str | None:
    """The HDF5 shared library netCDF4 loaded into this process (a wheel bundles its own)."""

    import netCDF4  # noqa: F401 - loads the library whose path is wanted

    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            paths = {line.split()[-1] for line in maps if "/" in line}
    except OSError:
        return None
    found = sorted(p for p in paths if Path(p).name.startswith("libhdf5")
                   and not any(tag in Path(p).name for tag in ("_hl", "_cpp", "fortran")))
    return found[0] if found else None


class _State:
    library: ctypes.CDLL | None = None
    fault: str | None = None
    hdf5: str | None = None
    layouts: dict[tuple, dict[str, Any]] = {}


def prepare() -> dict[str, Any]:
    """Load and bind the library once per process; say why when it cannot be used."""

    if _State.library is not None or _State.fault is not None:
        return status()
    if not enabled():
        _State.fault = f"{ENVIRONMENT}=off"
        return status()
    if not sys.platform.startswith("linux"):
        _State.fault = "the HDF5 filter-table hook is built for Linux"
        return status()
    path = library_path()
    if path is None or not path.is_file():
        _State.fault = (f"{LIBRARY_NAME} is not built (cargo build --release --manifest-path "
                        "rust/history-deflate/Cargo.toml, or set " + LIBRARY_ENVIRONMENT + ")")
        return status()
    try:
        library = ctypes.CDLL(str(path))
        library.hex_hd_version.restype = ctypes.c_char_p
        if library.hex_hd_version() != MARKER:
            raise OSError(f"{path} is not a {MARKER.decode()} library")
        hdf5 = _loaded_hdf5()
        if hdf5 is None:
            raise OSError("no HDF5 library is mapped in this process after importing netCDF4")
        error = ctypes.create_string_buffer(1024)
        library.hex_hd_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        if library.hex_hd_init(hdf5.encode(), error, len(error)) != 0:
            raise OSError(error.value.decode("utf-8", "replace"))
        library.hex_hd_begin.argtypes = [ctypes.POINTER(_Job), ctypes.c_size_t, ctypes.c_uint,
                                         ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
        library.hex_hd_end.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    except OSError as error:
        _State.fault = f"{type(error).__name__}: {error}"
        return status()
    _State.library, _State.hdf5 = library, hdf5
    return status()


def status() -> dict[str, Any]:
    return {"engaged": _State.library is not None, "hdf5": _State.hdf5, "why_not": _State.fault,
            "library": str(library_path()) if library_path() else None}


def _threads() -> int:
    explicit = os.environ.get(THREADS_ENVIRONMENT)
    if explicit:
        return max(1, int(explicit))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 2
    # Chunk construction and shuffle share memory bandwidth with the frame
    # transpose and hashes. The million-cell replay took 6.23 s with 16
    # workers versus 7.20/7.30 s with 24, also retaining more in-flight bytes.
    return min(16, max(2, cpus // 2))


def _layout(lengths: Mapping[str, int], dims_list: list[tuple[str, ...]]) -> dict[tuple[str, ...], dict[str, Any]]:
    """The chunking, filters and fill netCDF-C gives each dimension tuple of the
    frame, read from a throwaway file defined the way the writer defines it."""

    from netCDF4 import Dataset, default_fillvals

    key = (tuple(sorted(lengths.items())), tuple(dims_list))
    if key in _State.layouts:
        return _State.layouts[key]
    found: dict[tuple[str, ...], dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="hex-deflate-layout-") as scratch:
        with Dataset(Path(scratch) / "layout.nc", "w", format="NETCDF4_CLASSIC") as probe:
            for name, size in lengths.items():
                probe.createDimension(name, size)
            for index, dims in enumerate(dims_list):
                variable = probe.createVariable(f"v{index}", "f4", dims, zlib=True, complevel=1)
                chunking = variable.chunking()
                filters = variable.filters() or {}
                found[dims] = {"chunks": list(chunking) if chunking != "contiguous" else None,
                               "shuffle": bool(filters.get("shuffle")), "zlib": bool(filters.get("zlib")),
                               "level": int(filters.get("complevel") or 0)}
    fill = int(np.asarray(default_fillvals["f4"], dtype=np.float32).view(np.uint32))
    for value in found.values():
        value["fill_bits"] = fill
    _State.layouts[key] = found
    return found


def snapshot_jobs(module: Any, snapshot: Mapping[str, Any]) -> list[tuple[str, np.ndarray, int, int, tuple[str, ...]]]:
    """The deflated variables ``write_snapshot_netcdf`` writes, in its order, with its shape rules."""

    cells, levels, edges = int(module.N_CELLS), int(module.N_LEVELS), int(module.N_EDGES)
    interfaces, soil = int(module.N_INTERFACES), int(getattr(module, "N_SOIL_LEVELS", 0) or 0)
    jobs = []
    for name, value in sorted(snapshot["arrays"].items()):
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (levels, cells):
            dims, nz, n = ("Time", "nCells", "nVertLevels"), levels, cells
        elif array.shape == (levels, edges):
            dims, nz, n = ("Time", "nEdges", "nVertLevels"), levels, edges
        elif array.shape == (interfaces, cells):
            dims, nz, n = ("Time", "nCells", "nVertLevelsP1"), interfaces, cells
        elif soil and array.shape == (soil, cells):
            dims, nz, n = ("Time", "nCells", "nSoilLevels"), soil, cells
        elif array.shape == (cells,):
            dims, nz, n = ("Time", "nCells"), 0, cells
        else:
            continue
        jobs.append((name, np.ascontiguousarray(array), nz, n, dims))
    return jobs


@contextlib.contextmanager
def frame(function: Any, args: tuple) -> Iterator[dict[str, Any]]:
    """Around one ``function(path, snapshot, static)`` call: precompute its
    deflate streams in parallel.  Yields the dict the frame's counts land in."""

    stats: dict[str, Any] = {}
    if getattr(function, "__name__", "") != WRITER or len(args) < 2 or _State.library is None:
        stats.update(status(), engaged=False)
        if getattr(function, "__name__", "") != WRITER:
            stats["why_not"] = f"{getattr(function, '__qualname__', function)!r} is not {WRITER}"
        yield stats
        return
    module = sys.modules[function.__module__]
    if not all(hasattr(module, name) for name in ("N_CELLS", "N_EDGES", "N_LEVELS", "N_INTERFACES")):
        stats.update(status(), engaged=False, why_not="writer has no forecast mesh binding")
        yield stats
        return
    started = time.perf_counter()
    jobs = snapshot_jobs(module, args[1])
    lengths = {"Time": 1, "nCells": int(module.N_CELLS), "nVertLevels": int(module.N_LEVELS),
               "nEdges": int(module.N_EDGES), "nVertLevelsP1": int(module.N_INTERFACES)}
    if int(getattr(module, "N_SOIL_LEVELS", 0) or 0):
        lengths["nSoilLevels"] = int(module.N_SOIL_LEVELS)
    layout = _layout(lengths, sorted({dims for *_, dims in jobs}))
    keep = []
    table = []
    for name, array, nz, n, dims in jobs:
        row = layout[dims]
        if not row["zlib"] or not row["chunks"]:
            continue
        chunks = row["chunks"]
        table.append(_Job(array.ctypes.data, nz, n, chunks[1], chunks[2] if nz else 1,
                          row["level"], int(row["shuffle"]), row["fill_bits"]))
        keep.append(array)
    if not table:
        stats.update(status(), engaged=False, why_not="no deflated variable in this frame")
        yield stats
        return
    library = _State.library
    error = ctypes.create_string_buffer(1024)
    budget = int(os.environ.get(BUDGET_ENVIRONMENT, "2048")) * 1024 * 1024
    array_type = _Job * len(table)
    if library.hex_hd_begin(array_type(*table), len(table), _threads(), budget, error, len(error)) != 0:
        stats.update(status(), engaged=False, why_not=error.value.decode("utf-8", "replace"))
        yield stats
        return
    stats["setup_seconds"] = round(time.perf_counter() - started, 3)
    try:
        yield stats
    finally:
        out = ctypes.create_string_buffer(4096)
        library.hex_hd_end(out, len(out))
        try:
            stats.update(json.loads(out.value.decode("utf-8")))
        except ValueError:
            stats["end"] = out.value.decode("utf-8", "replace")
        stats["engaged"] = True
        del keep
