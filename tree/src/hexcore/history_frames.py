"""Bounded process history delivery at forecast-thread completion boundaries.

Reuse the native writer process rather than calling NetCDF from a thread
concurrently with regional boundary reads. No device operations occur here.
"""
from __future__ import annotations

from .history_writer import BackgroundWriter, private_arrays


class ProcessHistoryFrames:
    def __init__(self, function, on_written, *, max_pending=2):
        self.writer = BackgroundWriter(function, max_pending=max_pending)
        self.on_written = on_written
        self.pending = {}

    def submit(self, step, path, snapshot, static):
        if step in self.pending:
            raise ValueError(f"history step already pending: {step}")
        # Snapshot views must not follow later device/host state mutations.
        payload = {"arrays": private_arrays(snapshot["arrays"]),
                   "receipt": {"step": step}}
        self.pending[step] = (path, self.writer.submit(path, payload, static))

    def drain(self, *, until_step=None, ready_only=False):
        for step in sorted(self.pending):
            if until_step is not None and step > until_step:
                break
            path, future = self.pending[step]
            if ready_only and not future.done():
                break
            # result() also recovers a lost child reply on the forecast thread.
            # A writer refusal propagates before any ready notification.
            record = self.writer.result(future)
            self.on_written(step, path, record)
            del self.pending[step]

    def close(self):
        try:
            self.drain()
        finally:
            self.writer.close()

    def receipt(self):
        return {**self.writer.receipt(),
                "parallelism": {"gpu": False, "writer_processes": 1,
                                "threads_per_writer": 1},
                "completion_poll": "forecast macro boundaries; no device synchronization"}
