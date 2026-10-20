"""A committed frame's stream-ordered D2H copies with explicit lifetime ownership.

Copies are enqueued on the producer stream before the next step can mutate
state. The background consumer waits on one event, replacing the old capture
fence; host buffers and device sources are held until that event completes.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .host_parallel import threads


class SnapshotTransfer:
    def __init__(self, cp):
        self.cp = cp
        self.device = cp.cuda.Device().id
        self.stream = cp.cuda.get_current_stream()
        self.sources = []
        self.event = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            # Construction may fail after a copy was queued but before seal().
            # Keep both ends alive until that producer stream has drained.
            self.wait()

    def download(self, value, cp):
        if not isinstance(value, cp.ndarray):
            return np.array(value, copy=True, order="C")
        source = cp.ascontiguousarray(value)
        storage = cp.cuda.alloc_pinned_memory(source.nbytes)
        target = np.ndarray(source.shape, dtype=source.dtype, buffer=storage)
        self.sources.append((source, storage, target))
        source.get(out=target, stream=self.stream, blocking=False)
        return target

    def seal(self):
        event = self.cp.cuda.Event(disable_timing=True)
        event.record(self.stream)
        self.event = event

    def wait(self):
        with self.cp.cuda.Device(self.device):
            (self.event if self.event is not None else self.stream).synchronize()
        self.sources.clear()  # targets themselves retain their pinned allocations


class PendingSnapshot:
    def __init__(self, transfer, finish):
        self.transfer, self.finish = transfer, finish

    def result(self):
        self.transfer.wait()
        return self.finish()


def array_records(arrays, digest):
    def record(name):
        value = arrays[name]
        return name, {"dtype": value.dtype.str, "shape": list(value.shape),
                      "sha256": digest(value),
                      "minimum": float(value.min()) if value.size else None,
                      "maximum": float(value.max()) if value.size else None,
                      "nonzero": int(np.count_nonzero(value)),
                      "negative": int(np.count_nonzero(value < 0)) if value.dtype.kind == "f" else 0}
    with ThreadPoolExecutor(max_workers=min(threads(), max(1, len(arrays))),
                            thread_name_prefix="hex-frame-check") as pool:
        return dict(pool.map(record, sorted(arrays)))
