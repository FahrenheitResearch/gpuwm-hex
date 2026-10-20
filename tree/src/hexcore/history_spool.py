"""Bounded history IPC without copying every array again in the child.

The caller owns the temporary directory until its writer child exits. Protocol
5 keeps NumPy's dtype, shape, order and alias encoding; its contiguous buffers
live in one spool file mapped privately by the child. No NetCDF call is made
on these threads. A writer that modifies a buffer gets private copy-on-write
pages, as it did with an ordinary unpickled array.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import mmap
import os
from pathlib import Path
import pickle

from .host_parallel import threads


def dump_payload(payload, path: Path) -> dict:
    buffers = []
    encoded = pickle.dumps(payload, protocol=5, buffer_callback=buffers.append)
    spans, work = [], []
    total = 0
    for buffer in buffers:
        view = buffer.raw()
        total = (total + 63) // 64 * 64
        spans.append((total, len(view)))
        for start in range(0, len(view), 16 * 1024 * 1024):
            work.append((total + start, view[start:start + 16 * 1024 * 1024]))
        total += len(view)
    workers = min(4, threads(), max(1, len(work)))
    positioned_write = getattr(os, "pwrite", None)
    if positioned_write is None:
        workers = 1
    with path.with_suffix(".buffers").open("w+b", buffering=0) as stream:
        stream.truncate(total)

        def write(item):
            offset, view = item
            while view:
                if positioned_write is None:
                    stream.seek(offset)
                    count = stream.write(view)
                else:
                    count = positioned_write(stream.fileno(), view, offset)
                if count <= 0:
                    raise OSError("history spool write made no progress")
                offset += count
                view = view[count:]

        if workers == 1:
            for item in work:
                write(item)
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-history-spool") as pool:
                for _ in pool.map(write, work):
                    pass
    with path.open("wb") as stream:
        pickle.dump((encoded, spans), stream, protocol=5)
    return {"mode": "mapped-protocol5", "buffer_bytes": total,
            "buffers": len(buffers), "metadata_bytes": path.stat().st_size,
            "write_threads": workers}


def load_payload(path: Path):
    with path.open("rb") as stream:
        encoded, spans = pickle.load(stream)
    if not any(length for _, length in spans):
        return pickle.loads(encoded, buffers=[bytearray() for _ in spans])
    with path.with_suffix(".buffers").open("rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
    # NumPy's reconstructed arrays retain their buffer views and the mapping.
    # It is released when those arrays die, including at writer-child exit.
    return pickle.loads(encoded, buffers=(memoryview(mapping)[offset:offset + length]
                                         for offset, length in spans))
