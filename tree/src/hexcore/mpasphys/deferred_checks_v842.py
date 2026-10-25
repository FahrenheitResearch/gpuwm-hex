"""Device-side refusals read off the forecast's critical path.

Native MPAS stops at a fatal condition before the next slot runs. The CUDA
slots record the same conditions in device words. Reading each word as it is
written makes the single host thread wait for the GPU queue to drain several
times per step, so host launch work and device work never overlap.

THE BREAKAGE THIS PREVENTS: U-MPASPHYS-1. The MPAS-physics route spent most
of its step in an outer-physics stage that ran on one host thread while the
card idled between per-scheme synchronizations (GWDO, Noah, sea ice and the
outer wind reconstruction each read a flag every step).

Every check stays on. Inside a forecast step loop a slot hands its flag to
``defer``: the flag is reduced on the device at once (stream ordered, so it
captures the value at that point), and ``flush`` at the end of the step queues
one asynchronous copy of all of that step's words into pinned host memory. The
copy is inspected without blocking when it has landed, at the latest at the
next flush, and ``finish`` waits for every outstanding copy before the forecast
writes its last output. A refusal therefore still stops the forecast, at most
one step later than an immediate read, and names the condition and the step.

Outside a forecast loop (unit replays, referee helpers, other routes) no
collector is active and ``defer`` reads the flag immediately, as before.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable

_ACTIVE: contextvars.ContextVar["DeferredChecks | None"] = contextvars.ContextVar(
    "hexcore_mpasphys_deferred_checks", default=None)


def _message(message: str | Callable[[int], str], value: int) -> str:
    return message(value) if callable(message) else message.format(value=value)


def defer(flag: Any, message: str | Callable[[int], str], *, reduce: str = "max",
          error: type[Exception] = RuntimeError) -> None:
    """Refuse with ``error`` when ``flag`` reduces to a nonzero word.

    ``flag`` is a device integer array; ``reduce`` is ``max`` (the largest word,
    e.g. a native source line) or ``first`` (the first nonzero word in order).
    ``message`` is a format string with ``{value}``, or a callable of it.
    ``error`` is the exception type the slot raised before; both paths keep it.
    """
    checks = _ACTIVE.get()
    if checks is None:
        value = int(_reduce(flag, reduce).item())
        if value:
            raise error(_message(message, value))
        return
    checks.defer(flag, message, reduce=reduce, error=error)


def _array_module(flag: Any) -> Any:
    """The flag's own array module: CuPy for a device word, NumPy for a host one.

    THE BREAKAGE THIS PREVENTS (hex 0.4.5 gate, public CI battery on a CPU runner with no CuPy): the
    resident-synchronization replays hand host flags to ``defer`` outside a forecast loop, and an
    unconditional ``import cupy`` refused them with "No module named 'cupy'". A device flag still
    reduces on the device, exactly as before.
    """
    if type(flag).__module__.partition(".")[0] == "cupy":
        import cupy
        return cupy
    import numpy
    return numpy


def _reduce(flag: Any, reduce: str) -> Any:
    cp = _array_module(flag)
    if reduce == "max":
        return cp.max(flag)
    if reduce == "first":
        nonzero = flag != 0
        return flag.ravel()[cp.argmax(nonzero.ravel())]
    raise ValueError(f"unknown deferred reduction {reduce!r}")


class DeferredChecks:
    """One forecast's deferred device refusals."""

    def __init__(self, cp: Any):
        self.cp = cp
        self.step = 0
        self._pending: list[tuple[Any, Any, Any]] = []
        self._inflight: list[tuple[Any, Any, list[Any], int]] = []
        self.flushed = 0
        self.words = 0

    def defer(self, flag: Any, message: Any, *, reduce: str = "max",
              error: type[Exception] = RuntimeError) -> None:
        value = _reduce(flag, reduce).astype(self.cp.int64, copy=False)
        # A slot may run on its own stream: the flush stream waits for this point.
        self._pending.append((value, (message, error), self.cp.cuda.get_current_stream().record()))

    def flush(self, step: int | None = None) -> None:
        """Queue this step's words to pinned memory; inspect copies that have landed."""
        if step is not None:
            self.step = step
        self.poll(block=False)
        if not self._pending:
            return
        cp = self.cp
        stream = cp.cuda.get_current_stream()
        for _, _, ready in self._pending:
            stream.wait_event(ready)
        values = cp.stack([value for value, _, _ in self._pending])
        messages = [message for _, message, _ in self._pending]
        self._pending = []
        host = cp.cuda.alloc_pinned_memory(values.nbytes)
        import numpy as np
        out = np.frombuffer(host, dtype=np.int64, count=values.size)
        values.get(out=out, stream=stream, blocking=False)
        event = stream.record()
        # values must outlive its asynchronous copy; host keeps the pinned block alive.
        self._inflight.append((event, (values, host, out), messages, self.step))
        self.flushed += 1
        self.words += int(values.size)

    def poll(self, *, block: bool) -> None:
        keep = []
        for event, buffers, messages, step in self._inflight:
            if block:
                event.synchronize()
            elif not event.done:
                keep.append((event, buffers, messages, step))
                continue
            out = buffers[2]
            for value, (message, error) in zip(out.tolist(), messages):
                if value:
                    self._inflight = []
                    raise error(f"{_message(message, int(value))} (deferred device check, step {step})")
        self._inflight = keep

    def finish(self) -> None:
        """Read every outstanding word (the forecast's last boundary)."""
        self.flush()
        self.poll(block=True)

    def receipt(self) -> dict[str, Any]:
        return dict(flushes=self.flushed, words=self.words,
                    reader="pinned asynchronous copy per step, inspected when landed; blocking at the end",
                    parallelism="GPU reductions; host inspection off the step path")


@contextlib.contextmanager
def collecting(checks: DeferredChecks):
    token = _ACTIVE.set(checks)
    try:
        yield checks
    finally:
        _ACTIVE.reset(token)
