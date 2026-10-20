"""Shared surface-series discovery and packaged Rust decoder resolution."""
from datetime import datetime
import os
from pathlib import Path
import sys

SERIES_NAME = "surface_series.nc"


def reader_library():
    name = {"win32": "rw_surface.dll", "darwin": "librw_surface.dylib"}.get(
        sys.platform, "librw_surface.so")
    explicit = os.environ.get("GPUWM_HEX_SURFACE_READER")
    path = Path(explicit) if explicit else Path(__file__).parent / "libexec" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"surface series requires the packaged Rust decoder {path}; "
            "build tools/rustwx crate rw-surface or install a wheel carrying it")
    return path


def discover(work_dir):
    """The forecast output directory or its containing run work directory."""
    explicit = os.environ.get("GPUWM_HEX_SURFACE_SERIES")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"selected surface series does not exist: {path}")
        return path
    work = Path(work_dir)
    return next((p for p in (work / SERIES_NAME, work.parent / SERIES_NAME)
                 if p.is_file()), None)


def covering_reader(path, *, n_cells, start, end, library=None):
    from .mpasphys.surface_update_v842 import RustSurfaceRecordReader
    reader = RustSurfaceRecordReader(library or reader_library(), path, n_cells=n_cells)
    extent = reader.library.surface_cell_count
    extent.argtypes, extent.restype = [reader.ct.c_char_p], reader.ct.c_int64
    actual = extent(reader.path)
    if actual != n_cells:
        raise ValueError(f"surface series nCells {actual} differs from run nCells {n_cells}")
    start, end = (datetime.fromisoformat(str(s).replace("_", "T")) for s in (start, end))
    if reader.times[0] > start or reader.times[-1] < end:
        raise ValueError(f"surface series {path} does not cover {start} through {end}")
    if len(reader.times) < 2:
        raise ValueError("surface series needs at least two record times")
    intervals = {b-a for a, b in zip(reader.times, reader.times[1:])}
    if len(intervals) != 1:
        raise ValueError("surface series record interval must be uniform for SurfaceAlarms")
    reader.record_interval = intervals.pop()
    return reader
