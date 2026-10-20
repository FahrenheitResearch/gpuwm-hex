"""Bound closed-history product work without running it on the writer pump.

A session factory owns a persistent renderer/context on this worker. It must
return a context manager whose callable consumes file metadata and returns
closed-product records. No renderer command or CUDA policy is selected here.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable, Iterable, Mapping
import threading
import time
from typing import Any


class HistoryReady:
    def __init__(self, callback: Callable[[dict[str, Any]], Any] | None = None, *,
                 session_factory: Callable[[], Any] | None = None, max_pending: int = 2,
                 required_product_ids: Iterable[str] = ()) -> None:
        if (callback is None) == (session_factory is None):
            raise ValueError("supply one callback or one product session factory")
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._callback = callback
        self._session_factory = session_factory
        self._session = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-products-ready")
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_pending)
        self._max_pending = max_pending
        self._pending = 0
        self._peak_pending = 0
        self._queue_wait = 0.0
        self._closed = False
        self._seen: set[str] = set()
        self._tasks: list[Future] = []
        self._rows: list[dict[str, Any]] = []
        self._required = set(required_product_ids)
        self._products: dict[str, dict[str, Any]] = {}
        self._forecast_end: tuple[float, float] | None = None

    def _enqueue(self, function, *args) -> None:
        # Only capture/forecast callers enqueue. A full queue must never block
        # a writer Future's done-callback or its pump thread.
        mark = time.perf_counter()
        self._slots.acquire()
        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError("history product worker is closed")
                self._queue_wait += time.perf_counter() - mark
                self._pending += 1
                self._peak_pending = max(self._peak_pending, self._pending)
                try:
                    task = self._worker.submit(function, *args)
                except BaseException:
                    self._pending -= 1
                    raise
                self._tasks.append(task)
        except BaseException:
            self._slots.release()
            raise
        def finished(task):
            with self._lock:
                self._pending -= 1
            self._slots.release()
        task.add_done_callback(finished)

    def submit(self, file: Mapping[str, Any], *, step: int, label: str, valid_time: str) -> None:
        """Queue a closed file; bounded backpressure belongs to the caller."""
        with self._lock:
            if str(file["path"]) in self._seen:
                return
        self._enqueue(self._invoke, file, step, label, valid_time)

    def watch(self, future: Future, *, step: int, label: str, valid_time: str) -> None:
        """Wait on this worker, never in a completion callback on the writer pump."""
        self._enqueue(self._await_file, future, step, label, valid_time)

    def _await_file(self, future, step, label, valid_time):
        try:
            file = future.result()
        except BaseException:
            return  # the forecast raises the original write failure
        if isinstance(file, Mapping):
            self._invoke(file, step, label, valid_time)
        # A failed writer's sentinel is notified only after forecast recovery.

    def _invoke(self, file, step, label, valid_time) -> None:
        event = {"schema": "gpuwm-hex.history-ready/v1", "step": int(step),
                 "label": label, "valid_time": valid_time,
                 "path": str(file["path"]), "bytes": int(file["bytes"]),
                 "sha256": str(file["sha256"])}
        with self._lock:
            if event["path"] in self._seen:
                return
            self._seen.add(event["path"])
        started = time.monotonic()
        row = {"path": event["path"], "step": event["step"],
               "started_monotonic": started, "started_unix": time.time()}
        try:
            if self._session_factory is not None and self._session is None:
                session = self._session_factory()
                callback = session.__enter__()
                self._session, self._callback = session, callback
            result = self._callback(event)
            # The consumer returns only products already closed and available.
            # IDs are explicit view/product/valid-time identities from the run
            # manifest. Missing IDs can never certify a complete product tail.
            if result is not None:
                row["result"] = result
                if isinstance(result, Mapping):
                    finished, unix = time.monotonic(), time.time()
                    with self._lock:
                        for product in result.get("products", ()):
                            if product.get("status", "ready") != "ready":
                                continue
                            for key in ("id", "path", "bytes", "sha256"):
                                if key not in product:
                                    raise ValueError(f"closed product record lacks {key}")
                            self._products[str(product["id"])] = {
                                **product, "ready_monotonic": finished, "ready_unix": unix}
        except BaseException as error:
            row["error"] = repr(error)
            raise
        finally:
            row["finished_monotonic"] = time.monotonic()
            row["finished_unix"] = time.time()
            row["seconds"] = row["finished_monotonic"] - started
            with self._lock:
                self._rows.append(row)

    def forecast_ended(self, monotonic: float, unix: float) -> None:
        """Record the existing host committed-step clock, not GPU busy time."""
        with self._lock:
            self._forecast_end = (monotonic, unix)

    def _close_session(self):
        if self._session is not None:
            self._session.__exit__(None, None, None)

    def close(self) -> None:
        """Drain and close the session on the same worker that created it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        finalizer = self._worker.submit(self._close_session)
        self._worker.shutdown(wait=True)
        # Surface product errors after releasing the renderer session.
        for task in self._tasks:
            task.result()
        finalizer.result()

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            missing = sorted(self._required - self._products.keys())
            complete = bool(self._required) and not missing and not any("error" in r for r in self._rows)
            last = max((self._products[key]["ready_monotonic"] for key in self._required), default=None) if complete else None
            last_unix = max((self._products[key]["ready_unix"] for key in self._required), default=None) if complete else None
            return {"schema": "gpuwm-hex.history-ready-callbacks/v2", "workers": 1,
                    "max_pending": self._max_pending, "peak_pending": self._peak_pending,
                    "queue_wait_seconds": self._queue_wait,
                    "persistent_session_requested": self._session_factory is not None,
                    "submitted": len(self._seen), "completed": list(self._rows),
                    "required_product_ids": sorted(self._required),
                    "products": list(self._products.values()), "missing_product_ids": missing,
                    "required_products_complete": complete,
                    "forecast_end_monotonic": self._forecast_end[0] if self._forecast_end else None,
                    "forecast_end_unix": self._forecast_end[1] if self._forecast_end else None,
                    "forecast_end_origin": "host last committed step return; not a device timeline endpoint",
                    "last_required_ready_monotonic": last, "last_required_ready_unix": last_unix,
                    "last_required_ready_minus_forecast_end_seconds":
                        last - self._forecast_end[0] if last is not None and self._forecast_end else None}
