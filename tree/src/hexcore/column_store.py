"""The any-point column store of a hex run: every output hour, from the converted lat-lon frames.

Every run (ArWen and hex) writes the tiled compact column store for every output hour, so a
sounding at any point and any hour costs nothing extra. The store is rw-store version 2 (``RWSTORE2``, written by
the engine's ``rw_soundings --columns`` through :func:`gpuwm.soundings.write_columns`; an engine before 2.9.7 writes
the same tiles as ``GPWMCOL1``, which every reader of the store also reads). It needs the 3-D fields, so it is
written from frames converted with the full field set: the render door's (``--field-set full``, its default) and
the export door's when it exports the full set.

Files: ``<out>/<domain>/<YYYYMMDDTHHMMSSZ>.wcol``, the frames on whole hours plus the last frame when the run ends
off the hour. ``GPUWM_COLUMNS=off`` writes none. A store that cannot be written is said and never fails the door.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[_:](\d{2})[_:](\d{2})")


def enabled(environ=None) -> bool:
    value = (os.environ if environ is None else environ).get("GPUWM_COLUMNS", "").strip().lower()
    return value not in ("0", "off", "false", "no")


def chosen(frames: list[Path], every_minutes: int = 60) -> list[Path]:
    """The frames on the cadence, plus the last one."""
    frames = sorted(frames, key=lambda p: p.name)
    keep = []
    for f in frames:
        m = _STAMP.search(f.name)
        if m is None or every_minutes <= 0 or ((int(m[2]) * 60 + int(m[3])) % every_minutes == 0 and int(m[4]) == 0):
            keep.append(f)
    if frames and frames[-1] not in keep:
        keep.append(frames[-1])
    return keep


def write(frames: list[Path], out: Path, *, field_set: str = "full", say=print) -> dict:
    """Write the store for these converted frames under ``out``; returns a receipt (files, bytes, seconds) or the
    reason none was written."""
    started = time.monotonic()
    if not enabled():
        return {"files": 0, "skipped": "GPUWM_COLUMNS=off"}
    if field_set != "full":
        return {"files": 0, "skipped": f"field set {field_set!r} carries no 3-D fields for soundings"}
    picked = chosen([Path(f) for f in frames])
    if not picked:
        return {"files": 0, "skipped": "no frames"}
    try:
        from gpuwm.soundings import write_columns
    except ImportError as error:
        return {"files": 0, "skipped": f"the engine has no column store writer ({error})"}
    rows, failed = [], []
    for frame in picked:
        try:
            rows += write_columns(frame, out)
        except Exception as error:  # a store never fails the door
            failed.append({"frame": str(frame), "error": f"{type(error).__name__}: {error}"[:300]})
    receipt = {"files": len(rows), "bytes": int(sum(r.get("bytes", 0) for r in rows)), "out": str(out),
               "seconds": round(time.monotonic() - started, 3), "failed": failed}
    say(f"COLUMNS files={receipt['files']} bytes={receipt['bytes']} out={out} seconds={receipt['seconds']}"
        + (f" failed={len(failed)}: {failed[0]['error']}" if failed else ""), flush=True)
    return receipt
