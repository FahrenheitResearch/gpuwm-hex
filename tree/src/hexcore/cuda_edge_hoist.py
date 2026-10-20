"""Selection and first-call self-check for the hoisted stencil kernels.

The element kernels walk a flat (level, owner) index, so every level re-reads
the owner's stencil rows (neighbour indices and weights, stored [edge][slot]).
On a 15 km global rank that is about 160 MB of strided rows per level, and in
the nsys profile of 2026-10-10 (8-card RTX PRO 6000 host, 8 ranks, dt 60) the four edge-gather
kernels and the scalar finish were 393 ms of an 897 ms kernel step.

The hoisted kernels read each owner's stencil once into registers and walk the
levels inside the thread.  Every output element runs the element body
unchanged -- the same operands, the same slot order, the same helper calls --
so the result carries the same bits; only the launch mapping moved.

``GPUWM_HEX_EDGE_HOIST``: ``auto`` (default) selects the hoisted kernels when
the rank has at least ``HOIST_MIN_EDGES`` edges (below that, one thread per
edge under-fills the card and the element mapping is faster); ``1`` forces
them, ``0`` restores the element kernels.

``GPUWM_HEX_EDGE_HOIST_CHECK=N`` also runs the element kernel on the first N
calls of each hoisted kernel on each device and compares the two outputs word
for word (uint32 views, so NaN payloads and signed zeros count); a single
differing word raises.  It synchronizes on those calls only.
"""
from __future__ import annotations

import json
import os
import threading

HOIST_MIN_EDGES = 400_000
_LOCK = threading.Lock()
_CALLS: dict[tuple[str, int], int] = {}


def enabled(nedges: int) -> bool:
    mode = os.environ.get("GPUWM_HEX_EDGE_HOIST", "auto")
    if mode not in ("auto", "0", "1"):
        raise ValueError("GPUWM_HEX_EDGE_HOIST must be auto, 0 or 1")
    return mode == "1" or (mode == "auto" and int(nedges) >= HOIST_MIN_EDGES)


_LPT: dict[tuple[str, int, int, int], int] = {}
_LPT_TIMES: dict[tuple[str, int, int, int], dict[int, float]] = {}


def chunks(rows: int, lpt: int) -> int:
    return (int(rows) + int(lpt) - 1) // int(lpt)


def levels_per_thread(name: str, owners: int, rows: int, launch, cp) -> int:
    """Levels per thread for a hoisted kernel: tuned once per kernel, size and card.

    Every candidate writes the same bits (each (level, owner) runs the same
    body), so the first call times each candidate with CUDA events on the
    caller's stream and keeps the fastest.  ``GPUWM_HEX_HOIST_LPT=N`` fixes it
    (N >= rows is one thread per owner, the round-1 geometry).
    """
    rows = int(rows)
    fixed = os.environ.get("GPUWM_HEX_HOIST_LPT", "auto")
    if fixed != "auto":
        return max(1, min(rows, int(fixed)))
    key = (name, int(owners), rows, int(cp.cuda.Device().id))
    with _LOCK:
        chosen = _LPT.get(key)
    if chosen is not None:
        return chosen
    candidates = sorted({rows, (rows + 1) // 2, (rows + 3) // 4, (rows + 7) // 8, (rows + 15) // 16} - {0})
    stream = cp.cuda.get_current_stream()
    times = {}
    for lpt in candidates:
        launch(lpt)
        began, ended = cp.cuda.Event(), cp.cuda.Event()
        began.record(stream)
        launch(lpt)
        launch(lpt)
        ended.record(stream)
        ended.synchronize()
        times[lpt] = float(cp.cuda.get_elapsed_time(began, ended)) / 2.0
    chosen = min(times, key=times.get)
    with _LOCK:
        _LPT[key] = chosen
        _LPT_TIMES[key] = times
    print(json.dumps({"edge_hoist_lpt": {"kernel": name, "owners": int(owners), "rows": rows,
                                         "device": key[3], "chosen": chosen,
                                         "ms_by_lpt": {str(k): round(v, 4) for k, v in times.items()}}}),
          flush=True)
    return chosen


def tuning_receipt() -> list:
    with _LOCK:
        return [{"kernel": k[0], "owners": k[1], "rows": k[2], "device": k[3], "chosen": v,
                 "ms_by_lpt": {str(a): b for a, b in _LPT_TIMES.get(k, {}).items()}}
                for k, v in _LPT.items()]


def check_wanted(name: str, cp) -> bool:
    limit = int(os.environ.get("GPUWM_HEX_EDGE_HOIST_CHECK", "0") or 0)
    if limit <= 0:
        return False
    key = (name, int(cp.cuda.Device().id))
    with _LOCK:
        calls = _CALLS.get(key, 0)
        if calls >= limit:
            return False
        _CALLS[key] = calls + 1
        return True


def compare(name: str, hoisted, element, cp) -> None:
    new = cp.ascontiguousarray(hoisted).view(cp.uint32)
    old = cp.ascontiguousarray(element).view(cp.uint32)
    if new.shape != old.shape:
        raise RuntimeError(f"{name}: hoisted and element outputs differ in shape")
    differing = int(cp.count_nonzero(new != old))
    if differing:
        raise RuntimeError(
            f"{name}: hoisted kernel differs from the element kernel in "
            f"{differing} of {new.size} words; set GPUWM_HEX_EDGE_HOIST=0")
