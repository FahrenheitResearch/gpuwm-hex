"""Write forecast history frames in a background process.

THE COST THIS REMOVES, MEASURED (2026-10-05, start-up lane receipts in
``evidence/hex-startup-20261005/``): the step-0 history frame is one netCDF4
file with zlib level-1 variables, written by ``write_snapshot_netcdf`` at
about 40 MB/s on one core.  It sat between the step-0 capture and the first
forecast step: 1.15 GB and most of the 37.6 s step-0 output phase of the
135 s warm start on the published 30 km mesh, 7.2 s of 12.9 s on conus12km.
Every later frame stalled the step loop the same way.

WHAT THIS DOES.  The forecast still captures each frame on its own thread,
exactly as before (capture allocates device memory, and that order is kept).
The captured host arrays are then handed to ONE writer process, which calls
the SAME function from the SAME module file with the SAME module constants
the forecast process holds at hand-off (the mesh binding rebinds
``N_CELLS`` and friends at run time; the writer receives their values), so
each history file is byte-for-byte the file the forecast would have written
itself.  Frames are written in the order they were handed over.  The
forecast waits for every frame before it writes its receipts, so a finished
run has every file and every file digest it had before.
The pipe carries only control messages. Protocol-5 array buffers are copied
in parallel to one temporary spool per frame and privately mapped by the
child. The spool survives until the reply, then is removed. This uses the
same bounded transport as the same-host writer; GPUWM_HEX_HISTORY_SPOOL
selects its directory (default the frame's output directory).

WHY A PROCESS AND NOT A THREAD.  The forecast runs on the free-threaded
interpreter, and the netCDF4/HDF5 extension is not safe for concurrent use
(``drivers/samehost_serial_io_v841.py`` keeps it on one thread for that
reason).  A regional forecast reads its boundary files while it steps; a
writer thread would call into the same library at the same time.  A
separate process has its own copy of the library.

THE GUARDS, AND THE BREAKAGE EACH PREVENTS.
* The child imports the writer's module by name and refuses unless it
  resolved to the same file the forecast imported.  Prevents: a different
  installed copy of hexcore writing the frames.
* At most ``max_pending`` frames are in flight; a capture waits for the
  oldest before handing over another.  Prevents: a run that writes history
  every step holding every captured frame in host memory at once.
* A frame the writer did not complete (the process died, it could not
  start, the hand-off failed) is written again on the forecast thread, the
  way it always was, after any partial file is removed; the receipt says
  so.  An error the writer function itself raised (a refusal such as an
  existing file) is raised in the forecast exactly as a direct call would.
* At interpreter exit, frames already handed over are finished.  Prevents:
  a forecast that raises mid-run losing frames a direct write would have
  left on disk.

THE COMPRESSION ON EVERY CORE (B4, 2026-10-16).  Inside the writer process
each frame of ``write_snapshot_netcdf`` runs with :mod:`hexcore.history_deflate`
open: the Rust library ``rust/history-deflate`` compresses every chunk on
worker threads ahead of HDF5 and hands HDF5 the stream its own single-thread
deflate would have made, so the file is the same bytes and the 115 s a 15 km
frame spent in zlib on one core is spread over the CPUs.  Each frame's
receipt row carries the counts under ``deflate`` (``hits`` precomputed,
``misses`` left to HDF5).  ``GPUWM_HEX_HISTORY_DEFLATE=off`` writes the frames
the single-thread way.

``GPUWM_HEX_HISTORY_BACKGROUND=off`` writes every frame on the forecast
thread, as before.
"""

from __future__ import annotations

import atexit
import collections
import concurrent.futures
import importlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Mapping

import numpy as np
from . import history_deflate, history_spool, time_ledger

SCHEMA = "gpuwm-hex.history-writer/v1"
ENVIRONMENT = "GPUWM_HEX_HISTORY_BACKGROUND"
_SIMPLE = (bool, int, float, str, bytes, tuple, np.generic)


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT, "on").strip().lower() not in ("off", "0", "no", "false")


def _unwrap(function: Callable[..., Any]) -> Callable[..., Any]:
    """A tracer or timer that declares itself transparent is looked through."""

    for _ in range(16):
        if getattr(function, "__prep_cache_transparent__", False) and hasattr(function, "__wrapped__"):
            function = function.__wrapped__
        else:
            break
    return function


#: What each hand-off copied, for the receipt (one row per frame).
handoffs: list[dict[str, Any]] = []


def private_arrays(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """The frame's arrays, safe to hand to the writer while the forecast steps on.

    An array that owns its memory and is referenced only by ``arrays`` (a
    fresh device download, which is what a capture returns) is handed over as
    it is; anything else -- a view, or an array some other object also holds,
    which the next step could write into -- is copied first.  MEASURED
    (2026-10-05, RTX 5090 host): copying a whole conus12km frame (1.1 GB) took 3.8 s
    on the forecast thread, as long as the direct netCDF write it replaces.

    The ownership test is inline, not a helper call: the three references a
    sole-owned array has here are the mapping's entry, the loop variable and
    getrefcount's argument, and a helper (or a tracer wrapped around one)
    would add its own and copy everything.
    """

    started = time.perf_counter()
    handed: dict[str, Any] = {}
    copied: list[str] = []
    copied_bytes = 0
    # dict.items() retains a tuple on CPython 3.12, adding a fourth
    # reference even when the frame alone owns this array.
    for name in arrays:
        value = arrays[name]
        if (
            type(value) is np.ndarray
            and value.base is None
            and value.flags.owndata
            and sys.getrefcount(value) <= 3
        ):
            handed[name] = value
        else:
            handed[name] = np.array(value, copy=True)
            copied.append(name)
            copied_bytes += int(handed[name].nbytes)
    handoffs.append({"arrays": len(handed), "copied": copied, "copied_bytes": copied_bytes,
                     "seconds": round(time.perf_counter() - started, 3)})
    return handed


def _module_constants(function: Callable[..., Any]) -> dict[str, Any]:
    """The module-level constants ``function`` reads, as the forecast holds them now."""

    module = sys.modules[function.__module__]
    names: set[str] = set()
    stack = [function.__code__]
    while stack:
        code = stack.pop()
        names |= set(code.co_names)
        stack.extend(c for c in code.co_consts if hasattr(c, "co_names"))
    namespace = vars(module)
    return {
        name: namespace[name]
        for name in sorted(names)
        if name in namespace and isinstance(namespace[name], _SIMPLE)
    }


_FALLBACK = object()  # the writer did not complete this frame


def durable_marker(path: Path) -> Path:
    return path.with_name(path.name + ".durable.json")


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def mark_closed_frame_durable(path: Path) -> None:
    """Sync a native stream file after its owning library has closed it.

    Native PnetCDF owns the filename and cannot use atomic_write's temporary
    name. The marker is published only after the successful run closes all
    native streams, and after the same file/directory barriers as atomic_write.
    """
    path = Path(path)
    marker = durable_marker(path)
    temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
    # An old marker must not certify a newly written file if the sync fails.
    marker.unlink(missing_ok=True)
    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
        stat = path.stat()
        receipt = {"schema": "gpuwm.hex-history-durable/v1", "file": path.name,
                   "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(marker)
        _sync_directory(path.parent)
        time_ledger.emit("forecast.output.history.durable", root=path.parent, attrs=receipt)
    finally:
        temporary.unlink(missing_ok=True)
def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def atomic_write(function: Callable[..., Any], path: Path, *args: Any, replace: bool = False) -> Any:
    """Publish only a closed, synced frame, then its matching durability receipt.

    Hidden temporary files never match the site's history glob. A process crash
    before the marker leaves nothing eligible for conversion on the marker route.
    The original writer still performs every byte of the NetCDF serialization.
    """
    path = Path(path)
    if not replace and (path.exists() or durable_marker(path).exists()):
        raise FileExistsError(path)
    protocol = path.parent / ".history-durable-v1"
    try:
        stream = protocol.open("x", encoding="utf-8")
    except FileExistsError:
        pass  # Another frame writer has already installed the shared marker.
    else:
        with stream:
            stream.write("gpuwm.hex-history-durable/v1\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    marker = durable_marker(path)
    marker_tmp = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
    try:
        digest = getattr(function, "__history_digest__", None)
        if digest is None:
            value = function(tmp, *args)
            _sync_file(tmp)
        else:
            # The native snapshot writer closes NetCDF before returning.
            # Hash its immutable file while fsync flushes the same bytes;
            # neither result may be published until both have succeeded.
            value = function(tmp, *args, _hash_file=False)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                      thread_name_prefix="hex-history-digest") as pool:
                hashed = pool.submit(digest, tmp)
                _sync_file(tmp)
                value["sha256"] = hashed.result()
        tmp.replace(path)
        _sync_directory(path.parent)
        stat = path.stat()
        receipt = {"schema": "gpuwm.hex-history-durable/v1", "file": path.name,
                   "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        with marker_tmp.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        marker_tmp.replace(marker)
        _sync_directory(path.parent)
        time_ledger.emit("forecast.output.history.durable", root=path.parent,
                         attrs=receipt)
        if isinstance(value, dict) and "path" in value:
            value = {**value, "path": str(path)}
        return value
    finally:
        tmp.unlink(missing_ok=True)
        marker_tmp.unlink(missing_ok=True)


class _Frame:
    __slots__ = ("future", "args", "path", "submitted")

    def __init__(self, path: Path, args: tuple) -> None:
        self.future: concurrent.futures.Future = concurrent.futures.Future()
        self.args = args
        self.path = path
        self.submitted = time.perf_counter()


class BackgroundWriter:
    """One writer process; ``submit`` returns a future of the function's result."""

    def __init__(self, function: Callable[..., Any], *, max_pending: int = 2) -> None:
        function = _unwrap(function)
        self.function = function
        self.module = function.__module__
        self.qualname = function.__qualname__
        self.module_file = str(Path(sys.modules[self.module].__file__).resolve())
        self.max_pending = min(2, max(1, int(max_pending)))
        self._pending: collections.deque[_Frame] = collections.deque()
        self._outbox: collections.deque[_Frame | None] = collections.deque()
        self._cv = threading.Condition()
        self._broken: str | None = None
        self.receipt_rows: list[dict[str, Any]] = []
        handoffs.clear()
        self.wait_seconds = 0.0
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [p for p in sys.path if p] + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        try:
            self._child = subprocess.Popen(
                [sys.executable, "-m", "hexcore.history_writer"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment, close_fds=True,
            )
        except OSError as error:
            self._child = None
            self._broken = f"writer process did not start: {error}"
        self._stderr: collections.deque[str] = collections.deque(maxlen=40)
        self.ready: dict[str, Any] = {}
        if self._child is not None:
            threading.Thread(target=self._drain_stderr, name="hex-history-writer-stderr", daemon=True).start()
            try:
                # Finish interpreter and writer-module startup before capture(0).
                pickle.dump({"module": self.module, "qualname": self.qualname,
                             "module_file": self.module_file, "constants": {},
                             "prepare": True}, self._child.stdin, protocol=5)
                self._child.stdin.flush()
                self.ready = pickle.load(self._child.stdout)
                if "fault" in self.ready:
                    self._broken = self.ready["fault"]
            except Exception as error:
                self._broken = f"writer readiness failed: {type(error).__name__}: {error}"
        self._closed = False
        self._pump = threading.Thread(target=self._run, name="hex-history-writer", daemon=True)
        self._pump.start()
        atexit.register(self.close)

    # -- forecast side ----------------------------------------------------
    def submit(self, path: Path, *args: Any, defer_fallback: bool = False) -> concurrent.futures.Future:
        """Hand over owned arrays; optionally defer broken-writer recovery to result().

        A capture worker must defer recovery: NetCDF fallback belongs on the
        forecast thread, after integration stops reading regional boundaries.
        """

        mark = time.perf_counter()
        write_here = False
        with self._cv:
            if self._closed:
                raise RuntimeError("history writer is closed; submitting would leave a frame unwritten")
            while len(self._pending) >= self.max_pending:
                self._cv.wait()
            frame = _Frame(Path(path), args)
            if self._broken is None:
                self._pending.append(frame)
                self._outbox.append(frame)
                self._cv.notify_all()
            else:
                frame.future.frame = frame  # type: ignore[attr-defined]
                frame.future.set_result(_FALLBACK)
                write_here = not defer_fallback
        self.wait_seconds += time.perf_counter() - mark
        if write_here:  # the writer was broken before this submission
            frame.future = self._fallback(frame)
        return frame.future

    def result(self, future: concurrent.futures.Future) -> Any:
        """The frame's result; a frame the writer could not complete is written here."""

        mark = time.perf_counter()
        try:
            value = future.result()
            if value is _FALLBACK:
                if not hasattr(future, "recovered"):
                    future.recovered = self._fallback(future.frame)  # type: ignore[attr-defined]
                value = future.recovered.result()  # type: ignore[attr-defined]
            return value
        finally:
            self.wait_seconds += time.perf_counter() - mark

    def recover_if_needed(self, future: concurrent.futures.Future) -> Any:
        """Recover a failed hand-off on the forecast thread without waiting for healthy I/O."""
        if future.done() and future.result() is _FALLBACK:
            return self.result(future)
        return None

    def _fallback(self, frame: _Frame) -> concurrent.futures.Future:
        """Write ``frame`` on the calling (forecast) thread, as a direct call would."""

        started = time.perf_counter()
        done: concurrent.futures.Future = concurrent.futures.Future()
        time_ledger.emit("forecast.output.history", "begin", root=frame.path.parent,
                         attrs={"file": frame.path.name, "writer": "fallback", "span_id": frame.path.name})
        try:
            # A lost reply after publication is safe to recover. A pre-existing
            # destination is still a refusal from atomic_write, never deleted.
            if durable_marker(frame.path).is_file():
                receipt = json.loads(durable_marker(frame.path).read_text())
                stat = frame.path.stat()
                if (receipt.get("file"), receipt.get("bytes"), receipt.get("mtime_ns")) != (
                    frame.path.name, stat.st_size, stat.st_mtime_ns
                ):
                    raise RuntimeError(f"durable history changed after publication: {frame.path}")
                with frame.path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                done.set_result({"path": str(frame.path), "bytes": stat.st_size, "sha256": digest})
            else:
                done.set_result(atomic_write(self.function, frame.path, *frame.args))
        except BaseException as error:  # noqa: BLE001 - raised where the caller waits
            done.set_exception(error)
        finally:
            time_ledger.emit("forecast.output.history", "end", root=frame.path.parent,
                             attrs={"file": frame.path.name, "writer": "fallback", "span_id": frame.path.name})
        frame.args = ()
        self.receipt_rows.append({"path": str(frame.path), "how": "in-process fallback",
                                  "seconds": round(time.perf_counter() - started, 3)})
        return done

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._outbox.append(None)
            self._cv.notify_all()
        self._pump.join()

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": "background-process" if self._broken is None else "background-process, broken",
            "writer": f"{self.module}.{self.qualname}",
            "module_file": self.module_file,
            "ready": self.ready,
            "max_pending": self.max_pending,
            "frames": list(self.receipt_rows),
            "handoffs": list(handoffs),
            "forecast_wait_seconds": round(self.wait_seconds, 3),
            "broken": self._broken,
            "stderr_tail": list(self._stderr) if self._broken else [],
        }

    # -- pump thread ------------------------------------------------------
    def _drain_stderr(self) -> None:
        assert self._child is not None and self._child.stderr is not None
        for line in iter(self._child.stderr.readline, b""):
            self._stderr.append(line.decode("utf-8", "replace").rstrip())

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._outbox:
                    self._cv.wait()
                frame = self._outbox.popleft()
            if frame is None:
                break
            self._write(frame)
            with self._cv:
                self._pending.remove(frame)
                self._cv.notify_all()
        if self._child is not None:
            try:
                self._child.stdin.close()  # type: ignore[union-attr]
                self._child.wait(timeout=60)
            except Exception:  # noqa: BLE001 - a child that will not exit is killed
                self._child.kill()

    def _write(self, frame: _Frame) -> None:
        """Runs on the pump thread: never calls the writer function itself."""

        started = time.perf_counter()
        reply: dict[str, Any] | None = None
        transport: dict[str, Any] = {}
        if self._broken is None:
            try:
                request = {
                    "module": self.module, "qualname": self.qualname,
                    "module_file": self.module_file,
                    "constants": _module_constants(self.function),
                    "args": (frame.path, *frame.args),
                }
                # Keep large array buffers out of the serial control pipe.
                # The child privately maps them; the originals stay available
                # until its reply, including when transport recovery is needed.
                spool = Path(os.environ.get("GPUWM_HEX_HISTORY_SPOOL", frame.path.parent))
                spool.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="hex-history-", dir=spool) as work:
                    payload = Path(work) / "frame.pkl"
                    spooled = time.perf_counter()
                    transport = history_spool.dump_payload(request, payload)
                    transport["spool_seconds"] = time.perf_counter() - spooled
                    pickle.dump({"payload": str(payload)}, self._child.stdin, protocol=5)  # type: ignore[union-attr,arg-type]
                    self._child.stdin.flush()  # type: ignore[union-attr]
                    reply = pickle.load(self._child.stdout)  # type: ignore[union-attr,arg-type]
            except Exception as error:  # noqa: BLE001 - the frame is written in process below
                self._broken = f"{type(error).__name__}: {error}"
                if self._child is not None and self._child.poll() is None:
                    self._child.kill()
                    self._child.wait()
                reply = None
            if reply is not None and "fault" in reply:
                self._broken = reply["fault"]
                reply = None
        if reply is None:
            # Written on the forecast thread when the caller asks for it.
            frame.future.frame = frame  # type: ignore[attr-defined]
            frame.future.set_result(_FALLBACK)
            return
        frame.args = ()
        row = {"path": str(frame.path), "how": "background",
               "seconds": round(time.perf_counter() - started, 3),
               "transport": transport}
        if "error" in reply:
            row["error"] = repr(reply["error"])
            self.receipt_rows.append(row)
            frame.future.set_exception(reply["error"])
            return
        if reply.get("deflate") is not None:
            row["deflate"] = reply["deflate"]
        self.receipt_rows.append(row)
        frame.future.set_result(reply["value"])


def open_writer(function: Callable[..., Any], *, max_pending: int = 2) -> BackgroundWriter | None:
    """A background writer for ``function``, or None when disabled (write in process)."""

    if not enabled():
        return None
    return BackgroundWriter(function, max_pending=max_pending)


# ---------------------------------------------------------------------------
# the writer process
# ---------------------------------------------------------------------------
def _serve() -> int:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # a print in the writer must not corrupt the reply stream
    priority = None
    if sys.platform.startswith("linux"):
        try:
            priority = subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                                      capture_output=True, text=True)
        except OSError:
            pass
    while True:
        try:
            request = pickle.load(stdin)
        except EOFError:
            return 0
        try:
            if "payload" in request:
                request = history_spool.load_payload(Path(request["payload"]))
            module = importlib.import_module(request["module"])
            found = str(Path(module.__file__).resolve())
            if found != request["module_file"]:
                raise ImportError(f"{request['module']} resolved to {found}, not {request['module_file']}")
            target: Any = module
            for part in request["qualname"].split("."):
                target = getattr(target, part)
            for name, value in request["constants"].items():
                setattr(module, name, value)
        except Exception:  # noqa: BLE001 - reported; the forecast writes the frame itself
            pickle.dump({"fault": traceback.format_exc()}, stdout, protocol=5)
            stdout.flush()
            continue
        if request.get("prepare"):
            deflate = (history_deflate.prepare() if request["qualname"] == history_deflate.WRITER
                       else {"engaged": False, "why_not": f"{request['qualname']} is not {history_deflate.WRITER}"})
            pickle.dump({"pid": os.getpid(), "prepared": True,
                         "io_priority": "idle" if priority is not None and priority.returncode == 0 else "unavailable",
                         "deflate": deflate},
                        stdout, protocol=5)
            stdout.flush()
            continue
        try:
            time_ledger.emit("forecast.output.history", "begin", root=Path(request["args"][0]).parent,
                             attrs={"file": Path(request["args"][0]).name, "writer": "background",
                                    "span_id": Path(request["args"][0]).name})
            with history_deflate.frame(target, tuple(request["args"])) as deflate:
                value = atomic_write(target, *request["args"])
            reply = {"value": value, "deflate": deflate}
        except Exception as error:  # noqa: BLE001 - re-raised in the forecast
            reply = {"error": error}
        finally:
            time_ledger.emit("forecast.output.history", "end", root=Path(request["args"][0]).parent,
                             attrs={"file": Path(request["args"][0]).name, "writer": "background",
                                    "span_id": Path(request["args"][0]).name})
        del request
        try:
            pickle.dump(reply, stdout, protocol=5)
        except Exception:  # noqa: BLE001 - an unpicklable error is sent as text
            pickle.dump({"error": RuntimeError(repr(reply.get("error")))}, stdout, protocol=5)
        stdout.flush()


if __name__ == "__main__":
    raise SystemExit(_serve())
