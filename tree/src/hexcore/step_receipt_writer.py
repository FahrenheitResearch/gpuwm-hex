"""Compact, ordered host receipts written off the CUDA launch thread."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
import threading
import uuid
from pathlib import Path


class StepReceiptWriter:
    def __init__(self, path: Path, *, document: Path | None = None, schema: str | None = None) -> None:
        self.path = path
        if (document is None) != (schema is None):
            raise ValueError("step receipt document and schema must be supplied together")
        self.document = document
        self.schema = schema
        self._document_tmp = (document.with_name(f".{document.name}.{uuid.uuid4().hex}.tmp")
                              if document is not None else None)
        self.stream = path.open("x", encoding="utf-8")
        try:
            self._document_stream = (self._document_tmp.open("x", encoding="utf-8")
                                     if self._document_tmp is not None else None)
        except BaseException:
            self.stream.close()
            raise
        self._document_rows = 0
        self._published = False
        if self._document_stream is not None:
            self._document_stream.write('{\n  "receipts": [')
        # Keep the final join bounded, rather than accumulating a long-run
        # serialization backlog while the launch thread submits steps.
        self._slots = threading.BoundedSemaphore(32)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-receipts")
        self.futures = []
        self.closed = False

    def submit(self, step, elapsed, result, surface_execution):
        if self.closed:
            raise RuntimeError("step receipt writer is closed")
        self._slots.acquire()
        try:
            self.futures.append(self.executor.submit(
                self._write_bounded, step, elapsed, result, surface_execution))
        except BaseException:
            self._slots.release()
            raise

    def _write_bounded(self, *args):
        try:
            return self._write(*args)
        finally:
            self._slots.release()

    def _write(self, step, elapsed, result, surface_execution):
        row = {"step": step, "seconds": elapsed,
               "driver": asdict(result.committed.receipt),
               "backend": dict(result.backend_receipt),
               "arwen_v2_surface_execution": surface_execution,
               "clamp_d2h": result.clamp_d2h.as_dict(),
               "recovery": result.recovery.receipt()}
        self.stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.stream.flush()
        if self._document_stream is not None:
            # Match json.dumps({"receipts": rows, "schema": schema},
            # indent=2, sort_keys=True, allow_nan=False) byte for byte.
            encoded = json.dumps(row, indent=2, sort_keys=True, allow_nan=False)
            prefix = ",\n" if self._document_rows else "\n"
            self._document_stream.write(prefix + "\n".join("    " + line for line in encoded.split("\n")))
            self._document_rows += 1
        return row

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True)
            self.stream.close()
        # A disk/serialization failure must be reported, never hidden by a
        # successful forecast. No receipt from a speculative step is submitted.
        try:
            rows = [future.result() for future in self.futures]
            if self._document_stream is not None and not self._published:
                ending = "\n  ],\n" if self._document_rows else "],\n"
                self._document_stream.write(ending + '  "schema": ' + json.dumps(self.schema) + "\n}\n")
                self._document_stream.close()
                # Publish exclusively after the complete document is closed.
                os.link(self._document_tmp, self.document)
                self._published = True
            return rows
        finally:
            if self._document_stream is not None:
                self._document_stream.close()
                self._document_tmp.unlink(missing_ok=True)
