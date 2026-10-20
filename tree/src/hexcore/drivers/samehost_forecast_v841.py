"""Resident same-host execution adapter for the existing forecast front door.

Uses the existing host preparation, composite transaction, diagnostic CUDA
prep and NetCDF writer. Only rank orchestration and byte movement are added.
Regional ranks retain the native garbage element and project the global LBC
pool through the same local entity maps used for initialized state.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from collections import defaultdict
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np


def _prepared_attribute_view(prepared):
    return SimpleNamespace(**{field.name: getattr(prepared, field.name) for field in fields(prepared)})


def _initial_rank_surface_updates(proof, stacks, devices, surface_accumulators, cp, streams=None):
    """Upload each rank's initial accumulator snapshot on its assigned card.

    Ranks on distinct cards take their snapshots at the same time, each on its
    own thread and its card's default stream, exactly as the coordinator took
    them one after another.  Breakage the serial loop caused: each snapshot
    exports the rank's whole physics state to the host (diagnostic_snapshot),
    7.7 s for four 20 km ranks on a rented box (2026-10-13) with every card idle.
    Ranks sharing a card keep the serial order.
    """
    if len(stacks) != len(devices):
        raise ValueError("initial surface snapshots need one device per rank")

    def snapshot(rank):
        stack, device = stacks[rank], devices[rank]
        with cp.cuda.Device(device), (streams[rank] if streams is not None else nullcontext()):
            values = proof._previous_surface_updates(stack, surface_accumulators)
            # Snapshot uploads use the default stream. Finish them before
            # nonblocking rank streams can consume their initial old values.
            cp.cuda.get_current_stream().synchronize()
            if values is not None:
                for name, value in values.items():
                    if isinstance(value, cp.ndarray) and int(value.device.id) != int(device):
                        raise RuntimeError(
                            f"rank {rank} previous surface accumulator {name!r} is on device "
                            f"{value.device.id}; assigned device is {device}"
                        )
        return values

    workers = len(stacks) if len(set(devices)) == len(devices) else 1
    return _parallel_map(snapshot, range(len(stacks)), workers)


def _checked_boundary_comparison(candidate, reference, step):
    from hexcore.drivers.samehost_checks_v841 import compare_boundary_records

    verdict = compare_boundary_records(candidate, reference)
    if not verdict["state_invariant"] or verdict["unexpected_metadata_diffs"]:
        raise RuntimeError(
            f"partition invariance failed at step {step}: "
            f"state={verdict['state_diffs'][:4]}, "
            f"metadata={verdict['unexpected_metadata_diffs'][:4]}"
        )
    return verdict


def device_list(text):
    try:
        tokens = str(text).split(",")
        devices = tuple(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError("--samehost-devices needs comma-separated device ordinals") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--samehost-devices needs non-negative device ordinals")
    return devices


def resolved_fingerprint_every(arguments):
    """Boundary fingerprint interval of an explicit same-host request.

    An unset interval means every step only when a single-card reference is
    given, because the identity gate compares every recorded boundary.  With no
    reference it means zero (initial and final boundaries only).  Breakage the
    zero default prevents: a 438,918-cell two-card forecast started without the
    option hashed the whole committed state after every step, which took about
    7 s per 0.6 s model step and grew the host process until the box killed it
    at step 126 (agency demo, 2026-10-12).
    """
    every = getattr(arguments, "samehost_fingerprint_every", None)
    if every is None:
        return 1 if getattr(arguments, "samehost_reference", None) is not None else 0
    every = int(every)
    if every < 0:
        raise ValueError("--samehost-fingerprint-every must be non-negative")
    return every


#: Layouts loaded and halo tables built in this process, keyed by what they
#: were built from (see :func:`_layouts_and_tables`).
_LAYOUT_TABLES: dict[tuple, tuple] = {}


def _layouts_and_tables(path, *, mesh_sha256, part_sha256):
    """``load_layouts_npz`` and ``build_all`` once per process for the same files.

    Breakage it prevents: the door's ownership check, its rank placement, the
    driver's own load and the field-halo planner each loaded the same layouts
    and rebuilt the same halo tables -- four ``build_all`` passes of 2.4-3.9 s
    on the 20 km mesh (rented box, 2026-10-13) before the first step.  The tables are
    frozen dataclasses of read-only arrays, so one build serves every reader;
    the key binds the layouts file's own SHA-256 and the mesh and part digests
    the load verifies, so a changed file is loaded and built again.
    """
    from hexcore.file_digest import sha256_file
    from hexcore.partition_assets_v841 import load_layouts_npz
    from hexcore.partition_samehost_tables_v841 import build_all

    path = Path(path).resolve()
    key = (str(path), sha256_file(path), str(mesh_sha256), str(part_sha256))
    held = _LAYOUT_TABLES.get(key)
    if held is None:
        layouts = load_layouts_npz(path, expect_mesh_sha256=mesh_sha256,
                                   expect_part_sha256=part_sha256, expect_halo_rings=3)
        held = (layouts, build_all(layouts))
        _LAYOUT_TABLES[key] = held
    return held


def request_options(arguments, request):
    """Check ownership and destinations before a same-host run touches CUDA."""
    # The file's real SHA-256, reused for the same file version (hexcore.file_digest):
    # the 2 GB 20 km grid was hashed six times before the first step.
    from hexcore.file_digest import sha256_file
    devices = device_list(arguments.samehost_devices)
    if arguments.samehost_layouts is None and arguments.samehost_part is None:
        # Neither given: build them from this grid with the verified builder automatic card selection
        # uses (cached per grid digest).  Breakage the old refusal caused: a moving region crosses many
        # member meshes, and layouts given once were built for one member's grid only.
        arguments = built_layout_arguments(arguments, request, len(devices))
    if arguments.samehost_layouts is None or arguments.samehost_part is None:
        raise ValueError("same-host ranks need both --samehost-layouts and --samehost-part (or neither, to "
                         "build them from the grid): the layouts' halo ownership is verified against the part "
                         "file and the grid together")
    fingerprint_every = resolved_fingerprint_every(arguments)
    layouts, _ = _layouts_and_tables(arguments.samehost_layouts, mesh_sha256=sha256_file(request.grid),
                                     part_sha256=sha256_file(arguments.samehost_part))
    if len(layouts) != len(devices) or layouts[0]._global_cells != request.cells:
        raise ValueError("device assignments and layout coverage must match this mesh and partition count")
    return {"devices": devices, "layouts_path": arguments.samehost_layouts, "part_path": arguments.samehost_part,
            "transport": arguments.samehost_transport, "reference": arguments.samehost_reference,
            "fingerprint_every": fingerprint_every, "layouts": layouts}


def built_layout_arguments(arguments, request, count):
    """``arguments`` with ``count``-rank layouts and part built for ``request.grid`` (the existing builder)."""
    from hexcore.partition_assets_v841 import BUILDER_VERSION
    from hexcore.file_digest import sha256_file
    from hexcore.drivers.samehost_layouts_v841 import build_assets
    from hexcore.static_cache import _lock
    from hexcore import time_ledger

    # This is the existing source-tree tool, not a second partition algorithm.
    # A new cycle/work directory must reuse the mesh's verified partition.
    # The old scratch-parent default rebuilt a 2.6M-cell graph/layout in
    # 49.6 s on one core with both cards idle. Keep the existing content key
    # and receipt verification; only move its default lifetime to the mesh.
    # Mesh inputs may be installed read-only or shared between users. Keep
    # generated assets in the writable user cache, never beside the grid.
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    root = Path(os.environ.get("GPUWM_HEX_CACHE_ROOT", cache_home / "gpuwm-hex")).expanduser()
    assets = root / "samehost-layouts" / sha256_file(request.grid) / BUILDER_VERSION.replace("/", "-") / f"ranks{count}"
    with time_ledger.stage("prep.partition"):
        # Several cycles can start together. The next caller verifies the
        # completed receipt under the same lock instead of racing METIS or
        # reading partially written layouts. Reuse the static-cache lock.
        with _lock(assets / ".build.lock"):
            receipt = build_assets(request.grid, ranks=count, output_dir=assets)
    selected = SimpleNamespace(**vars(arguments))
    selected.samehost_layouts = Path(receipt["layouts"])
    selected.samehost_part = Path(receipt["part"])
    return selected


def visible_card_memory():
    """Read only CUDA-visible cards, after the rental queue masks its grant."""
    for name in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(name, "") not in ("", "0"):
            raise ValueError(f"{name} is set; automatic card selection cannot query a local card")
    import cupy as cp
    count = int(cp.cuda.runtime.getDeviceCount())
    if count < 1:
        raise ValueError("automatic card selection needs a CUDA-visible card to avoid allocating a forecast on no device")
    return tuple(int(cp.cuda.runtime.getDeviceProperties(device)["totalGlobalMem"])
                 for device in range(count))


def automatic_options(arguments, request):
    """Use capacity policy rows, then the existing verified layout builder."""
    from hexcore.device_admission import select_card_count, single_card_rows
    from hexcore import time_ledger

    # LTS is a one-card feature unless the caller explicitly names resident
    # ranks. Do this before the capacity census or layout construction: a
    # large mesh and extra visible cards must not turn LTS into multi-card.
    if (getattr(request, "fullstep_local_timestep", False)
            or getattr(request, "local_timestep", False)):
        time_ledger.emit("forecast.construct", attrs={"card_selection": {
            "selected_cards": 1,
            "basis": "local time stepping defaults to one card; multi-card needs --samehost-devices",
        }})
        return None

    # No device query or step-loop replacement for rows that always select one.
    policy = select_card_count(request.cells, 0, 1)
    if single_card_rows(request.cells):
        grant = os.environ.get("GPU_MUTEX_CARDS", "").replace(",", " ").split()
        visible = grant or [token for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token]
        policy.update(visible_cards=len(visible) if visible else None, card_memory_bytes=None,
                      basis="every memory tier for this mesh selects one card; no extra CUDA census")
        time_ledger.emit("forecast.construct", attrs={"card_selection": policy})
        return None
    memory = visible_card_memory()
    policy = select_card_count(request.cells, min(memory), len(memory))
    time_ledger.emit("forecast.construct", attrs={"card_selection": policy})
    count = policy["selected_cards"]
    if count == 1:
        return None
    from hexcore.file_digest import sha256_file

    selected = built_layout_arguments(arguments, request, count)
    selected.samehost_devices = ",".join(str(device) for device in range(count))
    selected.samehost_fingerprint_every = int(getattr(request, "fingerprint_every", 0))
    options = request_options(selected, request)
    from hexcore.partition_topology_v841 import place_ranks
    _, tables = _layouts_and_tables(options["layouts_path"], mesh_sha256=sha256_file(request.grid),
                                    part_sha256=sha256_file(options["part_path"]))
    placement = place_ranks(tables, options["devices"])
    options["devices"] = tuple(placement["devices"])
    policy["topology_placement"] = placement
    options["card_selection"] = policy
    return options


def rank_event(output_root, stage, phase, rank, device, *, attrs=None):
    """Emit through the shared ledger with physical card and rank identity."""
    try:
        from gpuwm.timeline.ledger import emit
    except ModuleNotFoundError:
        # gpuwm 2.8.7 and earlier carry no gpuwm.timeline; a rank event is
        # telemetry and must not stop a same-host forecast.
        return
    grant = os.environ.get("GPU_MUTEX_CARDS", "").replace(",", " ").split()
    visible = grant or os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    token = visible[device] if device < len(visible) else ""
    card = int(token) if token.isdecimal() else (None if token else device)
    emit(stage, phase, path=os.environ.get("GPUWM_TIME_LEDGER") or str(Path(output_root) / "timeline.jsonl"),
         product="hex", card=card, attrs={"rank": rank, "visible_device": device,
                                         "span_id": f"rank{rank}", **(attrs or {})})


@contextmanager
def output_span(stage, step):
    """Observe the existing coordinator capture and writer without replacing it."""
    from hexcore import time_ledger
    time_ledger.emit(stage, "begin", attrs={"step": step, "scope": "samehost_coordinator"})
    try:
        yield
    finally:
        time_ledger.emit(stage, "end", attrs={"step": step, "scope": "samehost_coordinator"})


def _parallel_map(function, items, workers):
    """Run ``function`` over ``items`` on ``workers`` threads, results in item order.

    The interpreter runs with the GIL disabled, so numpy gathers and hashes on
    different arrays proceed in parallel.  The first exception is re-raised.
    """
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [function(item) for item in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items)), thread_name_prefix="hex-gather") as pool:
        return list(pool.map(function, items))


def _off_path_workers(name, default):
    text = os.environ.get(name, str(default))
    try:
        value = int(text)
    except ValueError:
        value = -1
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; prevents an ambiguous writer receipt")
    return value


class _AsyncHistoryWriter:
    """Build and write history frames on background threads, off the step path.

    Breakage it prevents: the coordinator wrote each 4.6 GB global-15 km
    frame inline (about 100 s per frame on the 8-card RTX PRO 6000 host, 2026-10-10 smoke), so every
    history time stalled all ranks for the whole NetCDF write; after that
    moved off the path (58373c1) the capture still held every rank for about
    27 s per frame (smoke timeline: 126.5 s capture, 96.6 s of it the write),
    because the owned-array assembly, the f000 overlay and the physical gate
    ran on the coordinator.  Now only the per-rank device gather stays on the
    step path; ``submit`` takes a ``build`` callable that assembles the
    snapshot from those host copies, runs the unchanged physical gate and
    returns ``(snapshot, static)`` on a writer thread. Array hashes run on
    a bounded thread pool and the NetCDF write runs in a CHILD PROCESS: the
    legacy netCDF4/HDF5 extension is not thread-safe and the free-threaded
    serial-I/O guard (samehost_serial_io_v841) refuses any extension call
    off the coordinator thread, so the child is a separate interpreter
    calling the unchanged ``write_snapshot_netcdf`` with the module constants
    this process bound for the mesh.

    Array buffers are spooled on up to four threads while their hashes run,
    then mapped privately by the child. Every hash must succeed before the
    child launches. This avoids reading a second multi-GB array copy into the
    child's heap; the bounded frame slots also bound the spool lifetime.

    ``GPUWM_HEX_HISTORY_WRITERS`` (default 2) frames, including capture,
    are in flight at once (submission waits before gathering another frame),
    each in its own child: one writer took about 100 s per 15 km frame, longer
    than the interval between hourly frames at MPAS-A's dt.  Records, hashes
    and gates are re-keyed in step order on ``close``, so receipts do not
    depend on which frame finished first.  A failure (gate refusal, child
    error) is re-raised on the next ``submit`` and by ``close``, so a frame is
    never dropped silently.  When ``ready_path`` is given, one JSON line per
    finished frame (step, path, seconds, UTC) is appended there, which is how
    a renderer starts on a frame the moment it is complete.
    ``GPUWM_HEX_ASYNC_HISTORY=0`` restores the inline writer.
    """

    def __init__(self, write, array_sha256, frame_records, frame_hashes, *, workers=1,
                 ready_path=None, gates=None):
        import queue
        import threading
        self._module = write.__module__
        module = __import__("sys").modules[self._module]
        # Mesh-bound constants (N_CELLS, N_LEVELS, SCALAR_NAMES ...) are module
        # globals rebound at run time; the child must see the same values.
        self._constants = {name: value for name, value in vars(module).items()
                           if name.isupper() and isinstance(value, (int, float, str, bool, tuple))}
        self._write, self._sha = write, array_sha256
        self._records, self._hashes = frame_records, frame_hashes
        self._gates = gates
        self._ready = Path(ready_path) if ready_path is not None else None
        self._queue = queue.Queue()
        self._slots = threading.BoundedSemaphore(int(workers))
        self._lock = threading.Lock()
        self._error = None
        self.write_seconds = 0.0
        self.frames = 0
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("history writers must be positive")
        # Bound queued plus active frames, including each build closure's
        # captured host arrays. Queue(maxsize) alone excludes active frames.
        self._slots = threading.BoundedSemaphore(self.workers)
        self._closed = False
        self._threads = [threading.Thread(target=self._run, name=f"hex-history-writer-{index}", daemon=True)
                         for index in range(self.workers)]
        for thread in self._threads:
            thread.start()

    def _reserve(self):
        if self._closed:
            raise RuntimeError("history writer is closed")
        self._slots.acquire()
        if self._error is not None:
            self._slots.release()
            raise RuntimeError(f"history writer failed earlier: {self._error!r}") from self._error

    def _submit_reserved(self, step, path, build):
        from concurrent.futures import Future
        future = Future()
        self._queue.put((step, path, build, future))
        return future

    def submit(self, step, path, build):
        self._reserve()
        return self._submit_reserved(step, path, build)

    def capture(self, step, path, gather, build):
        # Reserve BEFORE allocating another frame's pinned storage. A slow
        # disk or product consumer cannot grow the history buffer inventory.
        self._reserve()
        try:
            arrays = gather()
        except BaseException:
            self._slots.release()
            raise
        return self._submit_reserved(step, path, lambda: build(arrays))

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            step, path, build, future = item
            began = time.perf_counter()
            try:
                if self._error is not None:
                    raise RuntimeError(f"history writer failed earlier: {self._error!r}") from self._error
                with output_span("forecast.output.history", step):
                    snapshot, static = build()
                    items = list(snapshot["arrays"].items())
                    from concurrent.futures import ThreadPoolExecutor
                    hash_workers = min(len(items) or 1,
                                       _off_path_workers("GPUWM_HEX_GATHER_THREADS", 8))
                    # Both readers see the same captured arrays. Hash while
                    # the handoff copies them, then require every hash before
                    # launching a child that can publish a history file.
                    with ThreadPoolExecutor(max_workers=hash_workers,
                                            thread_name_prefix="hex-history-hash") as pool:
                        pending = [(name, pool.submit(self._sha, array)) for name, array in items]

                        def checked_hashes():
                            return {name: future.result() for name, future in pending}

                        record = self._write_in_child(step, path, snapshot, static,
                                                      before_launch=checked_hashes)
                        hashes = checked_hashes()
                seconds = time.perf_counter() - began
                with self._lock:
                    self._hashes[str(step)] = hashes
                    self._records[str(step)] = record
                    self.frames += 1
                    if self._ready is not None:
                        from datetime import datetime, timezone
                        with self._ready.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({
                                "step": int(step), "path": str(path), "seconds": round(seconds, 3),
                                "utc": datetime.now(timezone.utc).isoformat()}) + "\n")
                future.set_result(record)  # file closed, hashes and ready manifest published
            except BaseException as error:  # re-raised on the coordinator by submit()/close()
                future.set_exception(error)
                with self._lock:
                    if self._error is None:
                        self._error = error
            finally:
                with self._lock:
                    self.write_seconds += time.perf_counter() - began
                # Do not retain the last frame while this worker waits idle.
                item = build = snapshot = static = items = pending = checked_hashes = None
                self._slots.release()

    _CHILD = (
        "import importlib, json, sys\n"
        "from pathlib import Path\n"
        "from hexcore.history_spool import load_payload\n"
        "payload = load_payload(Path(sys.argv[1]))\n"
        "module = importlib.import_module(payload['module'])\n"
        "for name, value in payload['constants'].items():\n"
        "    setattr(module, name, value)\n"
        "from pathlib import Path\n"
        "from hexcore import history_deflate, history_writer\n"
        "history_deflate.prepare()\n"
        "target = module.write_snapshot_netcdf\n"
        "args = (Path(payload['path']), payload['snapshot'], payload['static'])\n"
        "with history_deflate.frame(target, args) as stats:\n"
        "    record = history_writer.atomic_write(target, *args)\n"
        "record['deflate'] = stats\n"
        "open(sys.argv[2], 'w').write(json.dumps(record))\n"
    )

    def _write_in_child(self, step, path, snapshot, static, *, before_launch=None):
        from hexcore.history_spool import dump_payload
        import subprocess
        import sys
        import tempfile
        spool = Path(os.environ.get("GPUWM_HEX_HISTORY_SPOOL", "/dev/shm"))
        with tempfile.TemporaryDirectory(prefix=f"hex-history-{step}-", dir=spool) as work:
            frame, out = Path(work) / "frame.pkl", Path(work) / "record.json"
            began = time.perf_counter()
            transport = dump_payload({"module": self._module, "constants": self._constants, "path": str(path),
                                      "snapshot": snapshot, "static": static}, frame)
            transport["seconds"] = time.perf_counter() - began
            if before_launch is not None:
                checked = time.perf_counter()
                before_launch()
                transport["hash_wait_seconds"] = time.perf_counter() - checked
            done = subprocess.run([sys.executable, "-c", self._CHILD, str(frame), str(out)],
                                  capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"history child failed rc={done.returncode}: {done.stderr[-4000:]}")
            record = json.loads(out.read_text())
            record["handoff"] = transport
            return record

    def close(self):
        if not self._closed:
            self._closed = True
            for _ in self._threads:
                self._queue.put(None)
            for thread in self._threads:
                thread.join()
        for mapping in (self._records, self._hashes, self._gates):
            if mapping is not None:
                ordered = sorted(mapping.items(), key=lambda item: int(item[0]))
                mapping.clear()
                mapping.update(ordered)
        if self._error is not None:
            raise RuntimeError(f"history writer failed: {self._error!r}") from self._error


class _AsyncBoundaryWriter:
    """Reconstruct, hash and record boundary fingerprints on one background thread.

    Breakage it prevents: each boundary fingerprint held every rank while the
    coordinator assembled the full restart state of all ranks and hashed it
    (8-card RTX PRO 6000 host smoke, 2026-10-10: 76 s between the f000 capture and the end of
    forecast.construct, and again after the last step).  The device-to-host
    collection (``collect_owned`` on every rank) stays on the step path, so
    the bytes fingerprinted are the committed state at that step; the
    reconstruction, hashing, JSONL append and reference comparison run here,
    one boundary at a time in step order, so the file and the comparisons
    are the ones the inline path writes.  ``close`` joins and re-raises the
    first failure (a partition-invariance refusal included).
    ``GPUWM_HEX_ASYNC_BOUNDARY=0`` restores the inline path.
    """

    def __init__(self, process):
        import queue
        import threading
        self._process = process
        self._queue = queue.Queue()
        self._error = None
        self.seconds = 0.0
        self._thread = threading.Thread(target=self._run, name="hex-boundary-writer", daemon=True)
        self._thread.start()

    def submit(self, step, collected):
        if self._error is not None:
            raise RuntimeError(f"boundary writer failed earlier: {self._error!r}") from self._error
        self._queue.put((step, collected))

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            if self._error is not None:
                continue
            began = time.perf_counter()
            try:
                self._process(*item)
            except BaseException as error:
                self._error = error
            finally:
                self.seconds += time.perf_counter() - began

    def close(self):
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError(f"boundary writer failed: {self._error!r}") from self._error


class _StepLoopGarbage:
    """Keep CPython's cyclic collector off the rank threads during stepping.

    Breakage it prevents: the free-threaded collector stops every thread of
    the process for each pass, and its automatic passes fire on allocation
    counts from eight rank threads launching thousands of kernels per step.
    A pass in the middle of a step leaves all eight cards idle until it ends.
    The policy freezes everything built during setup (gc.freeze), disables
    automatic passes while the step loop runs, and runs one explicit pass on
    the coordinator between steps every GPUWM_HEX_GC_EVERY steps (default
    30), when no rank is launching.  Reference-counted frees are unchanged;
    only cycles wait for the next explicit pass.  GPUWM_HEX_GC=python keeps
    the interpreter's automatic collector.
    """

    def __init__(self):
        import gc
        self._gc = gc
        self.mode = os.environ.get("GPUWM_HEX_GC", "freeze")
        if self.mode not in ("freeze", "python"):
            raise ValueError("GPUWM_HEX_GC must be freeze or python")
        self.every = int(os.environ.get("GPUWM_HEX_GC_EVERY", "30"))
        if self.every < 1:
            raise ValueError("GPUWM_HEX_GC_EVERY must be a positive step count")
        self.passes, self.seconds, self.max_seconds, self.collected = 0, 0.0, 0.0, 0
        self._was_enabled = gc.isenabled()
        self._active = False

    def enter(self):
        if self.mode != "freeze":
            return
        began = time.perf_counter()
        self.collected += self._gc.collect()
        self._gc.freeze()
        self._gc.disable()
        self._active = True
        self.setup_seconds = time.perf_counter() - began

    def after_step(self, step):
        if not self._active or step % self.every:
            return
        began = time.perf_counter()
        self.collected += self._gc.collect()
        elapsed = time.perf_counter() - began
        self.passes += 1
        self.seconds += elapsed
        self.max_seconds = max(self.max_seconds, elapsed)

    def exit(self):
        if not self._active:
            return
        self._active = False
        self._gc.unfreeze()
        if self._was_enabled:
            self._gc.enable()

    def receipt(self):
        return {"mode": self.mode, "every_steps": self.every, "explicit_passes": self.passes,
                "explicit_seconds": self.seconds, "max_pass_seconds": self.max_seconds,
                "collected_objects": self.collected,
                "setup_freeze_seconds": getattr(self, "setup_seconds", None)}


def _exchange_before_scalar_boundary_adjust(regional, exchanger):
    """Make a rank's stage scalars owner truth on its halo before the boundary adjust.

    MPAS atm_srk3 exchanges the 'dynamics:scalars' halo group after each stage's
    transport and before atm_bdy_adjust_scalars; the relaxation filter reads the
    stage output on neighbour cells. Without this round a relaxation-zone cell on a
    partition seam reads its halo neighbours' unexchanged transport output (public
    conus12km: N2, N4 and N8 differed from N1 and from each other from step 1, in the
    moisture species at seam cells with bdyMaskCell 1-5). One card has no exchanger
    and is unchanged; the kernels and cuda_driver.py are unchanged.
    """
    original = regional.bdy_adjust_scalars
    if getattr(original, "_samehost_exchange_first", False):
        return

    def adjust(scalars, rk_step):
        exchanger.round_transport(scalars)
        return original(scalars, rk_step)

    adjust._samehost_exchange_first = True
    regional.bdy_adjust_scalars = adjust


def admit_request(request, registry, options, door):
    """Price each resident rank, summing ranks that share a physical card.

    Per-partition application of the existing footprint row is derived until
    multi-card high-water receipts replace it. Every card's free bytes and
    shape are read now, with one card margin after its rank-footprint sum.
    """
    from hexcore.device_admission import card_profile_from_attributes, model_for_card, samehost_rank_model
    from hexcore.forecast_door import stochastic_array_ledger
    from hexcore.stochastic import StochasticConfig
    from hexcore.cuda_backend.arch_admission import PROVEN_COMPUTE, admitted_architecture, below_floor_refusal
    if len(options["devices"]) > 1 and StochasticConfig.parse(getattr(request, "stochastic", None)).enabled:
        raise ValueError("partitioned stochastic forcing needs halo-aware streamfunction curl and "
                         "global dissipation reductions; these are not implemented, so this route "
                         "would apply inconsistent forcing between ranks. Use one global rank per member")
    for name in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU"):
        if os.environ.get(name, "") not in ("", "0"):
            raise ValueError(f"{name} is set; same-host admission cannot open a local card")
    import cupy as cp
    count = cp.cuda.runtime.getDeviceCount()
    records, floors, verdicts = [], {}, []
    for device in sorted(set(options["devices"])):
        if device >= count:
            raise ValueError(f"device {device} is outside the visible {count} cards")
        properties = cp.cuda.runtime.getDeviceProperties(device)
        compute = (int(properties["major"]), int(properties["minor"]))
        if compute < PROVEN_COMPUTE and admitted_architecture(compute) is None:
            raise ValueError(f"device {device}: " + below_floor_refusal(compute, PROVEN_COMPUTE))
        raw_name = properties["name"]
        name = raw_name.decode("utf-8", "replace") if isinstance(raw_name, bytes) else str(raw_name)
        threads_per_sm = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor, device)
        card = card_profile_from_attributes(name, {"MultiProcessorCount": properties["multiProcessorCount"],
                     "MaxThreadsPerMultiProcessor": threads_per_sm})
        configuration = door.request_configuration(request)
        model = request.model.with_card(card, configuration) if isinstance(request.model, door._SuppliedRow) else model_for_card(card, configuration)
        if request.model is None:
            model = samehost_rank_model(model)
        owned_ranks = [rank for rank, target in enumerate(options["devices"]) if target == device]
        predicted = sum(int(round(model.predict_bytes(options["layouts"][rank].n_local_cells))) for rank in owned_ranks)
        stochastic_arrays = {}
        for rank in owned_ranks:
            for key, size in stochastic_array_ledger(
                    options["layouts"][rank].n_local_cells,
                    getattr(request, "stochastic", None)).items():
                stochastic_arrays[key] = stochastic_arrays.get(key, 0) + size
        stochastic_basis = stochastic_arrays.get("harmonic_basis", 0)
        stochastic_bytes = sum(stochastic_arrays.values())
        predicted += stochastic_bytes
        margin = model.margin_bytes() if request.headroom_bytes is None else int(request.headroom_bytes)
        required = predicted + margin
        with cp.cuda.Device(device):
            free, total = map(int, cp.cuda.runtime.memGetInfo())
        floors[device] = required
        records.append({"device": device, "ranks": owned_ranks, "card": card.as_dict(),
                        "compute_capability": list(compute),
                        "predicted_peak_bytes": predicted, "required_free_bytes": required,
                        "stochastic_basis_bytes": stochastic_basis,
                        "stochastic_array_bytes": stochastic_bytes,
                        "stochastic_arrays": stochastic_arrays,
                        "measured_free_bytes": free, "measured_total_bytes": total,
                        "model": model.as_dict(),
                        "derivation": "sum of resident local-rank predictions plus one card margin; not a measured N-rank fit"})
        if free < required:
            local = [options["layouts"][rank].n_local_cells for rank in owned_ranks]
            raise ValueError(
                f"device memory admission refused {request.cells:,} mesh cells on device {device}: "
                f"ranks {owned_ranks} ({', '.join(f'{n:,}' for n in local)} local columns) need "
                f"{required} free bytes ({required / 2**20:,.0f} MiB), measured {free} "
                f"({free / 2**20:,.0f} MiB), short by {required - free} bytes "
                f"({(required - free) / 2**20:,.0f} MiB); one rank on this card admits at most "
                f"{model.max_cells(free, margin):,} local columns, so this mesh needs more cards; "
                "prevents out of memory while constructing or integrating the resident local ranks; "
                f"priced row: {model.provenance}"
            )
        verdicts.append(door.AdmissionVerdict(free >= required, request.mesh, request.cells, predicted, margin,
            required, free, total, max(0, required - free), 0, (),
            "resident-rank sum on device " + str(device) + "; " + model.provenance, model,
            stochastic_basis, stochastic_bytes, stochastic_arrays))
    options["memory_floor_by_device"] = floors
    options["admission_records"] = records
    return max(verdicts, key=lambda verdict: verdict.required_bytes - verdict.free_bytes)


@contextmanager
def installed(forecast, *, devices, layouts_path, part_path, transport="auto", reference=None, fingerprint_every=1,
              layouts=None, memory_floor_by_device=None, admission_records=None, card_selection=None):
    """Select rank orchestration for this invocation, restoring the door after."""
    previous = forecast.execute_forecast
    def execute(**kwargs):
        # The driver always forwards the full-step option (None when off).
        options = kwargs.pop("fullstep_lts_options", None)
        kwargs["fingerprint_every"] = fingerprint_every

        def native(**native_kwargs):
            native_kwargs.pop("fullstep_lts_options", None)
            native_kwargs["fingerprint_every"] = fingerprint_every
            return execute_samehost(forecast, devices=devices, layouts_path=layouts_path,
                                    part_path=part_path, transport_policy=transport,
                                    reference=reference, memory_floor_by_device=memory_floor_by_device,
                                    admission_records=admission_records, card_selection=card_selection,
                                    **native_kwargs)

        if options is not None:
            # Active full-step LTS across these cards: every rate class cut
            # into blocks (hexcore.drivers.fullstep_lts_forecast_v841).  A mesh
            # that classes entirely at rate one runs these same-host ranks,
            # unchanged, so on a uniform mesh "on" is "off".  The shadow has
            # no partitioned form (its donor census would count halo
            # duplicates), so it keeps its refusal.
            if options.get("fullstep_lts_mode") != "active":
                raise ValueError(
                    "the full-step shadow runs on one device: the partitioned shadow has no "
                    "owner-only donor reduction (halo duplicates would corrupt the rate census)")
            if int(options.get("fullstep_lts_max_rate", 8)) != 1:
                from hexcore.drivers import fullstep_lts_forecast_v841 as fullstep_forecast

                join = getattr(forecast, "join_prepared_fingerprint", None)
                if join is not None and "host" in kwargs:
                    join(kwargs["host"])

                return fullstep_forecast.execute_fullstep_lts(
                    forecast, native_execute=native, policy=dict(options), devices=tuple(devices),
                    memory_floor_by_device=memory_floor_by_device, **kwargs)
        return native(**kwargs)
    forecast.execute_forecast = execute
    # The prepared-input fingerprint (validation only: a same-host receipt
    # does not record it) keeps running while the ranks are built, and is
    # joined before the first step (_execute_samehost).  Breakage the early
    # join caused: 20.8 s of host preparation spent waiting on it with every
    # card idle (20 km, rented box, 2026-10-13).
    previous_defer = getattr(forecast, "DEFER_PREPARED_FINGERPRINT", False)
    forecast.DEFER_PREPARED_FINGERPRINT = True
    try:
        yield
    finally:
        forecast.execute_forecast = previous
        forecast.DEFER_PREPARED_FINGERPRINT = previous_defer


def _capture_rank(proof, host, stack, step, label, *, deferred=False):
    import cupy as cp
    from hexcore.snapshot_transfer import SnapshotTransfer, PendingSnapshot

    if not deferred:
        return _capture_rank_arrays(proof, host, stack, step, label, proof._host_array, resident=False)
    with SnapshotTransfer(cp) as transfer:
        arrays = _capture_rank_arrays(proof, host, stack, step, label, transfer.download, resident=True)
        transfer.seal()
    # SameHostRanks.call retains its existing final stream fence. Batching
    # avoids per-field blocking copies; this does not remove that rank fence.
    return PendingSnapshot(transfer, lambda: arrays)


def _capture_rank_arrays(proof, host, stack, step, label, download, *, resident):
    import cupy as cp
    from hexcore.cuda_physics_prep_v841 import prepare_mpas_to_phys_cuda_v841

    driver, backend = stack["driver"], stack["backend"]
    if driver.atmosphere.state.time_seconds != step * host["config"].config_dt:
        raise RuntimeError("history capture is not at the shared clock")
    # The backend validates its own local constructor census. The sealed
    # proof's whole-mesh census is not a rank's column count.
    if proof._phase_from_receipt(backend) not in ("complete", "boundary", "restored"):
        raise RuntimeError("history requires a committed backend boundary")
    diagnostic_snapshot = (getattr(backend, "history_diagnostic_snapshot", backend.diagnostic_snapshot)
                           if resident else backend.diagnostic_snapshot)
    diagnostic = proof.backend_diagnostic_mapping(diagnostic_snapshot())
    service = getattr(backend, "_surface_updates", None)
    classification = (dict(stack["constructor"].expected_surface_classification())
                      if service is None else dict(service.classification))
    used = classification if service is None else service.physics_classification or classification
    # The Noah-MP execution census is Noah-MP's: a rank whose land surface is
    # another scheme (a column-suite or generic-seam selection) publishes its
    # own census, or none, and has no Noah-MP classes to compare.  A suite
    # backend carries the census as last_land_surface_census (it renames the
    # frozen backend's last_noahmp_census in its receipts).
    selection = getattr(stack["constructor"], "selection", None)
    noahmp = selection is None or selection.get("land_surface_scheme") == "noahmp"
    census = None
    if noahmp:
        census = {"land": used["sflx_land_columns"], "water": used["open_water_columns"],
                  "sea_ice": used["sea_ice_columns"], "glacier": used["glacier_columns"]}
        if census["glacier"]:
            census["glacier_path"] = proof.ARWEN_GLACIER_CUDA_PROVENANCE
    for payload in (backend.step_receipt(), diagnostic["metadata"], diagnostic["receipt"]):
        if dict(payload["surface_classification"]) != classification:
            raise RuntimeError("history backend census differs from its local constructor")
        expected_census = census if step else None
        executed = payload.get("last_noahmp_census")
        if executed is None:
            executed = payload.get("last_land_surface_census")
        if noahmp and executed != expected_census:
            raise RuntimeError(
                "history NoahMP execution census differs from its local constructor: "
                f"executed {executed}, constructor {expected_census}")
    prepared = prepare_mpas_to_phys_cuda_v841(driver.atmosphere,
        scalar_names=host["scalar_names"], geometry=stack["prep_geometry"], kernel_cache=driver.cache,
        post_rk_wsm6=False)
    state, saved = driver.atmosphere.state, driver.atmosphere.saved
    arrays = {name: download(state.scalars[index], cp) for index, name in enumerate(host["scalar_names"])}
    arrays.update({name: download(array, cp) for name, array in (
        ("u_zonal", prepared.u_p), ("v_meridional", prepared.v_p), ("normal_u", saved.normal_velocity),
        ("rho", prepared.rho_dry), ("theta", prepared.th_p), ("pressure", prepared.pres_p),
        ("surface_pressure", prepared.psfc_p), ("w", saved.vertical_velocity))})
    # The snapshot taken above, at this same committed boundary: nothing in
    # between touches the physics seam (the census checks read it, the prep
    # reads the dycore atmosphere), and a second diagnostic_snapshot exported
    # the whole physics state to the host again for every rank and frame.
    diagnostics, _ = proof._flatten_backend_diagnostics(diagnostic, cp, download)
    arrays.update(diagnostics)
    refl = backend.take_history_refl10cm()
    if refl is None and step == 0 and hasattr(backend, "initial_refl10cm"):
        # Hour 0: the scheme's own reflectivity of the initial hydrometeors
        # (declared divergence from native MPAS-A's allocated zero; see
        # PersistentTwoPhaseCudaPhysicsBackendV841.initial_refl10cm).
        refl, _ = backend.initial_refl10cm(atmosphere=driver.atmosphere, scalar_names=host["scalar_names"])
    if refl is None:
        # A selection with no microphysics forms no reflectivity, and a scheme
        # whose reflectivity the engine cannot separate from its step has none
        # before its first call; such a frame carries the allocated default.
        if step and getattr(backend, "computes_reflectivity", lambda: True)():
            raise RuntimeError("history step committed no reflectivity")
        refl = cp.zeros_like(state.rho)
    arrays["refl10cm"] = download(refl, cp)
    regional = getattr(driver, "regional_v841", None)
    if regional is not None:
        arrays = {name: value if name == "normal_u" else regional.history_slice(value)
                  for name, value in arrays.items()}
    return arrays


def execute_samehost(forecast, **kwargs):
    host = kwargs["host"]
    if kwargs.get("balance_series"):
        # THE BREAKAGE THIS PREVENTS: the balance series is sampled by the one-card executors; the ranks
        # have no sampler, so the flag would be accepted and nothing written.
        raise ValueError("the balance series is sampled on one card; the same-host ranks have no balance-series "
                         "sampler.  Run it on one card (--samehost-devices with one ordinal, or none)")
    # Retired: the IAU refusal that stood here.  The ranks' step now carries the held-tendency hook
    # (each rank admits the increment on its own local cells and edges, halos included, through
    # hexcore.iau_native.build_iau_forcing), so IAU runs on the same-host ranks and on the automatic
    # multi-card selection.  The balance-series keyword is still popped: the rank executor takes none.
    kwargs.pop("balance_series", None)
    # The backend row is judged once, below, by the batch its row binds
    # (the frozen batch and the column-suite module's seams); this first
    # check holds only what the ranks' schedule cannot carry.
    if (kwargs.get("park_physics_tier") or kwargs.get("local_timestep_classing_path") is not None
            or getattr(host["config"], "config_local_timestep", False)):
        raise ValueError("same-host execution uses resident physics on the native step schedule: "
                         "a parked physics tier or local time stepping would step ranks on different clocks")
    if kwargs.get("grid_path") is None:
        raise ValueError("same-host layouts need the grid path for their mesh hash")
    # Legacy radiation caches contain mutable workspaces and device tables.
    # Their normal single-stream scope cannot be shared by resident ranks.
    from hexcore.drivers.samehost_physics_scope_v841 import scoped_physics_resources
    with scoped_physics_resources() as physics_scope:
        result = _execute_samehost(forecast, physics_scope=physics_scope, **kwargs)
    result["physics_resource_scope"] = physics_scope.receipt()
    (Path(kwargs["output_root"]) / "samehost-receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def _execute_samehost(forecast, *, physics_scope, devices, layouts_path, part_path, transport_policy, reference,
                     memory_floor_by_device=None, admission_records=None, card_selection=None,
                     host, schedule, cache_root, output_root, arwen_checkout,
                     source_receipt, authority_receipt, fingerprint_every,
                     stop_on_refusal=False, grid_path=None, park_physics_tier=False, required_free_bytes=None,
                     local_timestep_classing_path=None, cycling=None, iau=None,
                     on_history_ready=None, history_product_session=None,
                     history_product_max_pending=2, required_product_ids=()):
    if on_history_ready is not None and history_product_session is not None:
        raise ValueError("supply one history callback or one product session factory")
    if history_product_max_pending < 1:
        raise ValueError("history_product_max_pending must be positive")
    if required_product_ids and on_history_ready is None and history_product_session is None:
        raise ValueError("required products need a history consumer")
    from hexcore.file_digest import sha256_file
    from hexcore.partition_local_mesh_v841 import slice_prepared_host
    from hexcore.partition_samehost_transport_v841 import SameHostTransport
    from hexcore.partition_samehost_v841 import SameHostHaloExchanger, SameHostRanks
    from hexcore.partition_executor_v841 import expected_round_sequence
    from hexcore.partition_samehost_compile_v841 import compare_samehost_compile_manifests
    from hexcore.drivers.samehost_identity_v841 import collect_owned, reconstruct_boundary, assemble_owned_arrays
    from hexcore.drivers.samehost_checks_v841 import read_boundary_jsonl
    from hexcore.iau_native import IAU_WINDOW_KEY, window_record

    wall_categories = defaultdict(float)
    @contextmanager
    def measure_wall(category):
        began = time.perf_counter()
        try:
            yield
        finally:
            wall_categories[category] += time.perf_counter() - began
    proof = forecast.proof
    # Rank checkpoints persist local columns, halos, saved diagnostics,
    # backend accumulators and pending tendencies. The global JEDI contract
    # still requires assembly and analysis scatter that this path lacks.
    if cycling is not None and (cycling.get("jedi_background") or cycling.get("da_analysis") is not None):
        raise ValueError(
            "same-host --jedi-background and --da-analysis remain unsupported: "
            "rank-local physics fields and edge winds are not assembled into the "
            "global JEDI contract, and recoupled analysis is not scattered to rank "
            "halos. Dropping either would silently use the wrong initial state. "
            "Run the DA selection on one card"
        )
    # Resident frozen physics means the frozen engine batch, not the WSM6
    # row: a row binding gpuwm.core.mpas_column_batch (thompson_aero_column)
    # is the same resident seam with another species row, and this driver
    # already sizes everything from host["scalar_names"] and builds each
    # rank's backend through the row's own builder.  Any other batch (the
    # column suite, a provider's sibling batch) is still refused.
    from hexcore.physics_backend_admission import resolve_backend as _resolve_backend
    _row = _resolve_backend(str(host.get("physics_backend", "wsm6_column")))
    # 2026-10-11: the column-suite module's seams (the slot-named suite and
    # the generic entry that runs any selection of WOOF's registry) are the
    # same resident two-phase seam built per rank by the row's own builder,
    # so they are admitted beside the frozen batch.  A provider's sibling
    # batch is still refused: nothing here knows how to build it per rank.
    _resident = ("gpuwm.core.mpas_column_batch", "gpuwm.core.mpas_column_suite")
    if park_physics_tier or local_timestep_classing_path is not None or getattr(host["config"], "config_local_timestep", False):
        raise ValueError("same-host execution uses resident physics on the native step schedule: "
                         "a parked physics tier or local time stepping would step ranks on different clocks")
    if _row.batch_module not in _resident:
        raise ValueError(
            f"same-host execution builds each rank's physics from a resident column seam {_resident}; "
            f"backend row {_row.name} binds {_row.batch_module}, which no rank builder constructs")
    if grid_path is None:
        raise ValueError("same-host layouts need the grid path for their mesh hash")
    layouts, tables = _layouts_and_tables(layouts_path, mesh_sha256=sha256_file(grid_path),
                                          part_sha256=sha256_file(part_path))
    if len(layouts) != len(devices):
        raise ValueError(f"{len(layouts)} partitions need {len(layouts)} device assignments, got {len(devices)}")
    from hexcore.drivers import samehost_restart_v841 as restart
    restart_identity = restart.identity(host=host, grid_sha256=sha256_file(grid_path),
        layouts_sha256=sha256_file(layouts_path), part_sha256=sha256_file(part_path))
    checkpoint = None
    start_step = 0
    if cycling is not None and cycling.get("restart_from") is not None:
        import pickle
        with Path(cycling["restart_from"]).open("rb") as stream:
            checkpoint = pickle.load(stream)
        start_step = restart.validate(checkpoint, expected=restart_identity, schedule=schedule,
                                      rank_count=len(devices))
        from hexcore.iau_native import resolve_restart_window

        iau = resolve_restart_window(iau, carried=checkpoint.get(IAU_WINDOW_KEY),
                                     restart_seconds=float(start_step) * float(host["config"].config_dt))
        if host.get("regional") is not None:
            from datetime import datetime
            host = dict(host, regional=dict(host["regional"]))
            host["regional"]["start_time"] = datetime.strptime(checkpoint["valid_time"], "%Y-%m-%d_%H:%M:%S")
    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841
    arwen_pin = dict(forecast._pin_for_run(host, arwen_checkout, pin_arwen_physics_v841))
    from hexcore.cuda_backend import KernelCache, require_cuda
    import cupy as cp

    # Declare the alignment used by the existing local slicer before it runs.
    mesh = host["prepared"].mesh
    declarations = dict(getattr(mesh, "variable_dimensions", {}) or {})
    for name in ("defc_a", "defc_b"):
        if name in mesh.arrays:
            declarations.setdefault(name, ("nCells", "maxEdges"))
    mesh.variable_dimensions = declarations
    stacks, caches = [], []
    memory = []
    admitted_devices = set()
    # One compute stream per rank, built before the ranks and kept for the
    # whole run: construction, the step-0 snapshots and every step use it, so
    # each rank's arrays live in one CuPy pool arena (a freed block returns
    # to the arena of the stream that allocated it). Breakage it prevents:
    # ranks built on the default stream (or the setup group's own streams)
    # left every construction-time block freed later in an arena the
    # stepping streams never reuse, 1.5 GB of free blocks held beside one
    # 120 km rank (x1.40962, io-halo census 2026-10-15).
    # GPUWM_HEX_SAMEHOST_SETUP_STREAM=default builds on the default stream as before.
    setup_stream = os.environ.get("GPUWM_HEX_SAMEHOST_SETUP_STREAM", "rank")
    if setup_stream not in ("rank", "default"):
        raise ValueError("GPUWM_HEX_SAMEHOST_SETUP_STREAM must be rank or default; prevents an ambiguous memory receipt")
    rank_streams = None
    if setup_stream == "rank":
        rank_streams = []
        for device in devices:
            with cp.cuda.Device(device):
                rank_streams.append(cp.cuda.Stream(non_blocking=True))
    # When a rank exchanges fields: "fence" (default with rank streams) drops
    # the transport's references once the rank's consumer fence is queued,
    # "completion" once every rank's receive event has completed, "step" at
    # end_step (the schedule-slot transport before 2026-10-15).
    field_release = os.environ.get("GPUWM_HEX_HALO_FIELD_RELEASE", "fence")
    if field_release not in ("fence", "completion", "step"):
        raise ValueError("GPUWM_HEX_HALO_FIELD_RELEASE must be fence, completion or step")
    if field_release == "fence" and rank_streams is None:
        # The fence argument needs every rank array in its compute stream's
        # arena; default-stream construction keeps the completion contract.
        field_release = "completion"
    compile_cache_root = Path(os.environ.get("MPAS_PORT_CUDA_CACHE_DIR")
                              or os.environ.get("CUPY_CACHE_DIR")
                              or Path.home() / ".cache" / "gpuwm-hex" / "cuda")
    def construct_rank(rank):
        layout, device = layouts[rank], devices[rank]
        rank_event(output_root, "prep.device_upload", "begin", rank, device,
                   attrs={"local_cells": layout.n_local_cells, "card_selection": card_selection})
        with (cp.cuda.Device(device), rank_streams[rank] if rank_streams is not None else nullcontext(),
              physics_scope.rank(rank, device)):
            if device >= cp.cuda.runtime.getDeviceCount():
                raise ValueError(f"device {device} is not visible")
            # Keep compiler images across forecasts. CuPy's source/options/compiler
            # key includes every mesh dimension embedded into CUDA source.
            rank_cache = compile_cache_root / f"rank{rank}"
            rank_cache.mkdir(parents=True, exist_ok=True)
            capability = require_cuda(device_id=device, min_compute=(12, 0), cache_dir=rank_cache)
            cache = KernelCache(capability=capability, cache_dir=rank_cache)
            # Same-card rank checks need the SUM of resident rank footprints.
            # A whole-mesh door estimate is a lower bound, not a per-rank cap.
            floor = (memory_floor_by_device or {}).get(device, required_free_bytes or proof.MIN_FREE_DEVICE_BYTES)
            if device not in admitted_devices:
                memory.append({"device": device, **proof.gpu_memory_admission(cp, minimum=int(floor))})
                admitted_devices.add(device)
            local_host = dict(host)
            for name in ("prepared", "constructor_values", "gwdo_host", "f000_surface_diagnostics"):
                # PreparedCudaInputs has init=False derived fields. Its legacy
                # dataclass slicer can mutate the source on replace() failure.
                # A plain attribute view leaves the global preparation intact.
                item = _prepared_attribute_view(host[name]) if name == "prepared" else host[name]
                local_host[name] = slice_prepared_host(item, layout)
            local_host["constructor_values"] = dict(local_host["constructor_values"])
            local_host["constructor_values"]["n_columns"] = layout.n_local_cells + int(host.get("regional") is not None)
            try:
                stack = proof._construct_device_stack(host=local_host, cache=cache, arwen_checkout=arwen_checkout,
                                                       backend_builder=forecast._backend_builder(host, arwen_checkout),
                                                       physics_rollback_snapshot=bool(stop_on_refusal),
                                                       **(restart.constructor_kwargs(checkpoint["ranks"][rank])
                                                          if checkpoint is not None else {}))
                from hexcore.arwen_surface_update import attach_host
                attach_host(stack["backend"], host, schedule, output_root,
                            dt=float(forecast.DT_SECONDS), layout=layout)
                stack["restart_forcing"] = (restart.restore_forcing(checkpoint["ranks"][rank], stack)
                                             if checkpoint is not None else None)
                from hexcore.partition_overlap_v841 import interior_edges
                edge_mask = interior_edges(local_host["prepared"].mesh.arrays["cellsOnEdge"],
                                           n_owned_cells=layout.n_owned_cells,
                                           n_owned_edges=layout.n_owned_edges)
                edge_extent = stack["driver"].horizontal.nedges
                if edge_extent < edge_mask.size:
                    raise ValueError("local device has fewer edges than the partition; prevents incomplete damping")
                edge_mask = np.pad(edge_mask, (0, edge_extent - edge_mask.size))
                stack["interior_edge_mask"] = cp.asarray(edge_mask)
                # Halo work is disposable. Native field widths do not refresh
                # every outer K3 value, and those discarded values can exceed
                # health limits while the owned forecast remains unchanged.
                # Every global cell and edge is checked by exactly one rank.
                stack["owned_health_extents"] = {
                    stack["driver"].horizontal.ncells: layout.n_owned_cells,
                    edge_extent: layout.n_owned_edges,
                }
            finally:
                # Construction also enqueues device-table uploads. Finish
                # them before a failed constructor releases scoped owners.
                cp.cuda.get_current_stream().synchronize()
        rank_event(output_root, "prep.device_upload", "end", rank, device)
        return stack, cache
    setup_workers = int(os.environ.get("GPUWM_HEX_SETUP_WORKERS", str(len(devices))))
    if setup_workers < 1 or setup_workers > len(devices):
        raise ValueError("GPUWM_HEX_SETUP_WORKERS must be between one and the rank count")
    setup_started = time.perf_counter()
    # The existing workers already isolate CUDA device, stream and physics resources.
    # Same-card ranks retain serial admission of their summed resident footprint.
    if setup_workers > 1 and len(set(devices)) == len(devices):
        with SameHostRanks(devices, None, cp=cp, physics_scope=physics_scope, streams=rank_streams) as setup_ranks:
            results = setup_ranks.call(construct_rank)
    else:
        results = [construct_rank(rank) for rank in range(len(devices))]
    stacks = [result[0] for result in results]
    caches = [result[1] for result in results]
    # The prepared-input validation ran beside the rank construction; it must
    # pass before any rank steps (run_cuda_v841_forecast.join_prepared_fingerprint).
    join = getattr(forecast, "join_prepared_fingerprint", None)
    if join is not None:
        join(host)
    memory.sort(key=lambda item: item["device"])
    print(json.dumps({"rank_setup_seconds":time.perf_counter()-setup_started,
                      "rank_setup_workers":len(devices) if setup_workers > 1 and len(set(devices)) == len(devices) else 1}), flush=True)
    if len({stack["driver"].configuration_sha256 for stack in stacks}) != 1:
        raise RuntimeError("rank configurations differ")
    # The round string follows the run's own schedule.  It used to assume the
    # default RK3 / 6-substep schedule (acoustic rounds 1, 3, 6 per split) and
    # no limiter on regional meshes, so RRFS v2 settings (RK2, 4 substeps:
    # 2, 2, 4; limiter on) aborted at step 1 round 1 with "expected B, got C"
    # on every N-rank run (rrfs-3km-proof, conus3km N4, 2026-10-05).  The
    # regional limiter now fires round E like the global one (cuda_driver).
    from hexcore.integration import RKSchedule
    _config = host["config"]
    monotonic = bool(_config.config_monotonic or _config.config_positive_definite)
    _rk = RKSchedule.from_mpas(float(_config.config_dt), order=int(_config.config_time_integration_order),
                               acoustic_substeps=int(_config.config_number_of_sub_steps),
                               dynamics_splits=int(_config.config_dynamics_split_steps))
    round_schedule = {"dynamics_splits": int(_config.config_dynamics_split_steps),
                      "acoustic_steps": tuple(stage.acoustic_steps for stage in _rk.stages)}
    # Schedule-slot packets have their own storage and end_step drains them
    # before reuse. The byte exchange preserves the numerical operations,
    # while avoiding the second rendezvous at every halo round.
    pipeline = os.environ.get("GPUWM_HEX_PIPELINED_EXCHANGE", "1")
    if pipeline not in ("0", "1"):
        raise ValueError("GPUWM_HEX_PIPELINED_EXCHANGE must be 0 or 1")
    transport_type = SameHostTransport
    if pipeline == "1":
        from hexcore.partition_samehost_pipeline_v841 import StepSlotTransport
        transport_type = StepSlotTransport
    from hexcore.partition_field_halos_v841 import build_field_indices, FIELD_WIDTHS
    field_indices = build_field_indices(layouts, mesh.arrays["cellsOnEdge"], tables=tables)
    timing = os.environ.get("GPUWM_HEX_EXCHANGE_TIMING", "0")
    if timing not in ("0", "1"):
        raise ValueError("GPUWM_HEX_EXCHANGE_TIMING must be 0 or 1; prevents ambiguous halo timing receipts")
    transport = transport_type(tables, devices, xp=cp, copy_policy=transport_policy, timeout=120,
                               field_halo_indices=field_indices, measure_gpu=timing == "1")
    transport.release_fields_after_fence = field_release == "fence"
    transport.release_fields_on_retire = field_release != "step"
    regional = host.get("regional") is not None
    # One round-E slot per monotonic-limiter species pass (hex memory census
    # item 4): the limiter splits its scale-factor exchange the same way.
    from hexcore.cuda_transport_v841 import limiter_species_per_pass
    n_species = int(stacks[0]["driver"].atmosphere.state.scalars.shape[0])
    fct_parts = -(-n_species // limiter_species_per_pass(n_species)) if n_species else 1
    native_rounds = expected_round_sequence(monotonic=monotonic, regional=regional,
                                            fct_parts=fct_parts, **round_schedule)
    physics_rounds = native_rounds[:-1] + ("P", native_rounds[-1])
    exchangers = [SameHostHaloExchanger(table, transport, expected_sequence=physics_rounds) for table in tables]
    for stack, exchanger in zip(stacks, exchangers):
        stack["driver"].halo_exchanger_v841 = exchanger
        exchanger.interior_edge_mask = stack["interior_edge_mask"]
    # IAU on the ranks: one class per rank, its local cells and edges (halos included, as each rank
    # steps them), the increment's copies on that rank's card.  The same forcing objects as one card.
    iau_forcing = iau_receipt = None
    if iau is not None:
        from hexcore.iau_native import build_iau_forcing

        iau_forcing, iau_receipt = build_iau_forcing(
            iau, prepared=host["prepared"], scalar_names=host["scalar_names"], layouts=layouts, cp=cp,
            work_dir=cache_root, n_cells=int(layouts[0]._global_cells), n_edges=int(layouts[0]._global_edges),
            n_levels=int(stacks[0]["driver"].atmosphere.state.rho.shape[0]), devices=list(devices),
            cell_widths=[int(stack["driver"].atmosphere.state.rho.shape[1]) for stack in stacks])
        print(json.dumps({"iau": iau_receipt}, default=str), flush=True)
        if regional:
            _exchange_before_scalar_boundary_adjust(stack["driver"].regional_v841, exchanger)
    reference_records = read_boundary_jsonl(reference) if reference else None
    compared = []
    writer = forecast.BoundaryFingerprintWriter(Path(output_root) / "boundary-fingerprints.jsonl")
    previous = _initial_rank_surface_updates(proof, stacks, devices, host["surface_accumulators"], cp,
                                             streams=rank_streams)
    forcing = [stack["restart_forcing"] for stack in stacks]
    checkpoint_files = {}
    step_rows, frame_records, frame_hashes = [], {}, {}
    health, physical_gates = [], {}
    refusal = None
    final_forcing_hashes = None
    compile_identity = None
    static = proof._static_output_fields(host)
    captures = {step for step in schedule["capture_steps"] if step > start_step or step == start_step == 0}
    active_intervals = [False] * len(devices)
    dt = float(host["config"].config_dt)
    async_history = os.environ.get("GPUWM_HEX_ASYNC_HISTORY", "1")
    if async_history not in ("0", "1"):
        raise ValueError("GPUWM_HEX_ASYNC_HISTORY must be 0 or 1; prevents an ambiguous history-writer receipt")
    history_async = (_AsyncHistoryWriter(proof.write_snapshot_netcdf, proof.array_sha256, frame_records, frame_hashes,
                                         workers=_off_path_workers("GPUWM_HEX_HISTORY_WRITERS", 2),
                                         ready_path=Path(output_root) / "history-ready.jsonl",
                                         gates=physical_gates)
                     if async_history == "1" else None)
    gc_policy = _StepLoopGarbage()
    async_boundary = os.environ.get("GPUWM_HEX_ASYNC_BOUNDARY", "1")
    if async_boundary not in ("0", "1"):
        raise ValueError("GPUWM_HEX_ASYNC_BOUNDARY must be 0 or 1; prevents an ambiguous fingerprint receipt")

    def record_boundary(step, collected):
        record = reconstruct_boundary(layouts, collected, proof=proof)
        writer.write(step, record)
        if reference_records is not None:
            if step not in reference_records:
                raise RuntimeError(f"reference has no boundary at step {step}")
            verdict = _checked_boundary_comparison(record, reference_records[step], step)
            compared.append({"step": step, **verdict})

    boundary_async = _AsyncBoundaryWriter(record_boundary) if async_boundary == "1" else None
    started = time.perf_counter()
    product_ready = None
    last_step_clock = None
    if on_history_ready is not None or history_product_session is not None:
        from hexcore.history_ready import HistoryReady
        product_ready = HistoryReady(on_history_ready, session_factory=history_product_session,
                                     max_pending=history_product_max_pending,
                                     required_product_ids=required_product_ids)
    try:
        with SameHostRanks(devices, transport, cp=cp, physics_scope=physics_scope, streams=rank_streams) as ranks:
            def boundary(step):
                with measure_wall("fingerprints"):
                    collected = ranks.call(lambda rank: collect_owned(layouts, rank, stacks[rank]))
                    if boundary_async is not None:
                        boundary_async.submit(step, collected)
                        return
                    record_boundary(step, collected)
            def capture(step):
                with measure_wall("capture_inclusive"):
                    with output_span("forecast.output.capture", step):
                        return capture_arrays(step)
            def assemble_snapshot(step, arrays):
                # Initial reflectivity is a per-rank zero allocation, sized to
                # that rank, then assembled with every other published field.
                kinds = {name: "edge" if name == "normal_u" else "cell" for name in arrays[0]}
                snapshot = {"arrays": assemble_owned_arrays(layouts, arrays, kinds=kinds,
                                native_pads=("normal_u",) if host.get("regional") is not None else ()),
                            "receipt": {"step": step, "arwen_v2_surface_execution": {"last_noahmp_census": None}}}
                if step == 0:
                    proof.overlay_f000_initialized_surface_diagnostics(snapshot["arrays"], host["f000_surface_diagnostics"]["arrays"])
                    forecast.complete_f000_surface_diagnostics(snapshot, host["f000_start_humidity"])
                physical_gates[str(step)] = proof.physical_snapshot_gate(snapshot, allow_initial_negative_qv=(step == 0),
                    scalar_names=host["scalar_names"], surface_accumulators=host["surface_accumulators"],
                    surface_accumulator_declarations=host.get("surface_accumulator_declarations"))
                return snapshot
            def capture_arrays(step):
                def gather(rank):
                    return _capture_rank(proof, host, stacks[rank], step, schedule["labels"][step],
                                         deferred=history_async is not None)
                history_path = Path(output_root) / f"cuda-history.{schedule['labels'][step]}.nc"
                if history_async is not None:
                    def build(pending):
                        arrays = [frame.result() for frame in pending]
                        return assemble_snapshot(step, arrays), static
                    with measure_wall("history_submission_with_gather"):
                        future = history_async.capture(step, history_path, lambda: ranks.call(gather), build)
                    if product_ready is not None:
                        product_ready.watch(future, step=step, label=schedule["labels"][step],
                                            valid_time=schedule["valid_times"][step])
                    return
                arrays = ranks.call(gather)
                snapshot = assemble_snapshot(step, arrays)
                frame_hashes[str(step)] = {name: proof.array_sha256(array) for name, array in snapshot["arrays"].items()}
                with measure_wall("history_nested_in_capture"), output_span("forecast.output.history", step):
                    frame_records[str(step)] = proof.write_snapshot_netcdf(history_path, snapshot, static)
                if product_ready is not None:
                    product_ready.submit(frame_records[str(step)], step=step, label=schedule["labels"][step],
                                         valid_time=schedule["valid_times"][step])
            if 0 in captures:
                capture(0)
            boundary(start_step)
            from hexcore import time_ledger
            time_ledger.end_if_open("forecast.construct")
            gc_policy.enter()
            for step in range(start_step + 1, int(schedule["steps"]) + 1):
                mark = time.perf_counter()
                def advance(rank):
                    if step == 1:
                        rank_event(output_root, "forecast.integrate", "begin", rank, devices[rank],
                                   attrs={"step": step, "card_selection": card_selection})
                        active_intervals[rank] = True
                    return advance_rank(rank)
                def advance_rank(rank):
                    stack, exchanger = stacks[rank], exchangers[rank]
                    exchanger.begin_step(step)
                    result = proof.execute_composite_step(driver=stack["driver"], backend=stack["backend"],
                        scalar_names=host["scalar_names"], physics_geometry=stack["physics_geometry"],
                        kernel_cache=stack["driver"].cache, previous_surface_updates=previous[rank],
                        dynamics_tendencies=forcing[rank], refl_10cm_due=(step in captures),
                        before_phase_two=exchanger.round_physics_candidate,
                        **({} if iau_forcing is None else {"held_forcing": (
                            lambda held, start, length, **context:
                            iau_forcing.add(rank, held, start, length, **context))}))
                    atmosphere = stack["driver"].atmosphere
                    exchanger.round_step_boundary(atmosphere.state, atmosphere.saved)
                    exchanger.end_step()
                    previous[rank] = result.committed.surface_updates
                    forcing[rank] = result.dynamics_tendencies
                    return {"driver": asdict(result.committed.receipt),
                            "backend": dict(result.backend_receipt), "clamp_d2h": result.clamp_d2h.as_dict(),
                            "recovery": result.recovery.receipt()}
                def completed(rank):
                    rank_event(output_root, "forecast.integrate", "mark", rank, devices[rank], attrs={"step": step})
                    if step == int(schedule["steps"]) or ranks.failed:
                        rank_event(output_root, "forecast.integrate", "end", rank, devices[rank], attrs={"step": step})
                        active_intervals[rank] = False
                advance.after_synchronize = completed
                try:
                    if pipeline == "1":
                        transport.begin_step(step, len(exchangers[0]._expected))
                    if step == 1:
                        time_ledger.emit("forecast.first_step", "begin")
                    rank_receipts = ranks.step(advance, lambda rank: stacks[rank]["driver"].atmosphere.state.time_seconds, dt=dt)
                    if product_ready is not None:
                        last_step_clock = (time.monotonic(), time.time())
                    if pipeline == "1":
                        transport.end_step(step)
                except Exception as error:
                    for rank, device in enumerate(devices):
                        # step() has drained every submitted worker. Include
                        # ranks that completed before another rank failed.
                        if active_intervals[rank]:
                            rank_event(output_root, "forecast.integrate", "end", rank, device,
                                       attrs={"step": step, "failed": True})
                            active_intervals[rank] = False
                    first_failure = {"step": step, "last_committed_shared_step": step - 1,
                                     "exception": type(error).__name__, "message": str(error),
                                     "notes": list(getattr(error, "__notes__", ())),
                                     "traceback": traceback.format_exc(),
                                     "transport_abort_reason": str(getattr(transport, "_error", ""))}
                    # Persist the first visible failure before a sticky CUDA
                    # error can make shutdown fail at a later API call.
                    try:
                        Path(output_root, "samehost-first-failure.json").write_text(
                            json.dumps(first_failure, indent=2) + "\n")
                    except OSError as receipt_error:
                        error.add_note(f"first-failure receipt could not be written: {receipt_error}")
                        first_failure["notes"] = list(getattr(error, "__notes__", ()))
                    if not stop_on_refusal:
                        raise
                    refusal = {**first_failure,
                               "final_state_retained": False,
                               "reason": "a failed rank group stops; histories from completed shared boundaries remain"}
                    break
                finally:
                    if step == 1:
                        time_ledger.end_if_open("forecast.first_step")
                elapsed = time.perf_counter() - mark
                step_rows.append({"step": step, "seconds": elapsed, "ranks": rank_receipts})
                with measure_wall("health_with_sync"):
                    health.append(ranks.call(lambda rank: forecast.step_health_gate(stacks[rank], step, cp)))
                if step == 1:
                    with measure_wall("compile_manifest_check"):
                        compile_identity = compare_samehost_compile_manifests({rank: cache.compile_manifest() for rank, cache in enumerate(caches)})
                if (fingerprint_every > 0 and step % fingerprint_every == 0) or step == int(schedule["steps"]):
                    boundary(step)
                if step in captures:
                    capture(step)
                if cycling is not None and step in cycling["checkpoint_steps"]:
                    entries = ranks.call(lambda rank: restart.download_rank(stacks[rank], layouts[rank], forcing[rank]))
                    target = Path(output_root) / f"checkpoint.{schedule['labels'][step]}.pkl"
                    checkpoint_files[str(step)] = restart.write(target, entries=entries,
                        expected=restart_identity, step=step, schedule=schedule, dt=dt,
                        **({} if iau is None else {"extra": {IAU_WINDOW_KEY: window_record(iau)}}))
                    checkpoint_files[str(step)]["sha256"] = sha256_file(target)
                    del entries
                with measure_wall("step_log"):
                    print(json.dumps({"step": step, "seconds": elapsed, "samehost_ranks": len(devices)}), flush=True)
                gc_policy.after_step(step)
            if refusal is None and all(carrier is not None for carrier in forcing):
                def collect_forcing(rank):
                    carrier = forcing[rank]
                    if float(carrier.time_seconds) != ranks.time_seconds:
                        raise RuntimeError("final dynamics forcing does not share the committed clock")
                    return {name: proof._host_array(getattr(carrier, name), cp)
                            for name in ("rthdynten", "rqvdynten")}
                arrays = assemble_owned_arrays(layouts, ranks.call(collect_forcing),
                                               kinds={name: "cell" for name in ("rthdynten", "rqvdynten")},
                                               native_pads=("rthdynten", "rqvdynten") if host.get("regional") is not None else ())
                final_forcing_hashes = {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                                              "sha256": proof.array_sha256(value)} for name, value in arrays.items()}
    finally:
        gc_policy.exit()
        if product_ready is not None and last_step_clock is not None:
            product_ready.forecast_ended(*last_step_clock)
        try:
            if boundary_async is not None:
                with measure_wall("boundary_async_drain"):
                    boundary_async.close()
                wall_categories["boundary_async_writer_thread"] = boundary_async.seconds
        finally:
            try:
                writer.close()
            finally:
                try:
                    if history_async is not None:
                        with measure_wall("history_async_drain"):
                            history_async.close()
                        wall_categories["history_async_writer_thread"] = history_async.write_seconds
                finally:
                    if product_ready is not None:
                        with measure_wall("history_product_drain"):
                            product_ready.close()
    iau_record = None
    if iau_forcing is not None:
        iau_record = {**iau_receipt, **iau_forcing.receipt()}
        if (refusal is None and float(ranks.time_seconds) >= float(iau.get("window_start_seconds") or 0.0)
                + float(iau["window_seconds"]) and not iau_record.get("complete")):
            raise RuntimeError(f"IAU window not fully applied on the same-host ranks: {iau_record}")
    result = {"compile_timing_by_rank": {str(rank): cache.timing_receipt() for rank, cache in enumerate(caches)},
        "stochastic_by_rank": {str(rank): stack["driver"].stochastic.receipt()
                               for rank, stack in enumerate(stacks)
                               if hasattr(stack["driver"], "stochastic")},
        **({"iau": iau_record} if iau_record is not None else {}),
        **({"history_product_ready": product_ready.receipt()} if product_ready is not None else {}),
        "wall_accounting": {"seconds": dict(wall_categories),
            "rank_stream_sync_seconds": list(ranks.sync_seconds),
            "rank_stream_sync_calls": list(ranks.sync_calls),
            "scope": "host timers, no added CUDA fences; health and fingerprints include existing rank synchronization; "
                     "with GPUWM_HEX_ASYNC_HISTORY=0 assembly, gate, hashes and the NetCDF write are nested in capture; "
                     "by default only the per-rank device gather is in capture_inclusive and assembly, gate, hashes and "
                     "the NetCDF write run on GPUWM_HEX_HISTORY_WRITERS writer threads (history_async_writer_thread sums "
                     "their busy seconds, off the step path; history_async_drain is the end-of-run join); with "
                     "GPUWM_HEX_ASYNC_BOUNDARY=1 (default) fingerprints holds only the device collection and "
                     "boundary_async_writer_thread the reconstruction and hashing; compile load/sync remain inside integration",
            "async_history": history_async is not None,
            "history_writers": history_async.workers if history_async is not None else 0,
            "history_max_pending_frames": history_async.workers if history_async is not None else 0,
            "history_capture": "owned pinned batch; existing rank fence retained" if history_async else "synchronous",
            "async_boundary": boundary_async is not None,
            "garbage_collection": gc_policy.receipt()},
        "schema": "mpas-port.samehost-forecast/v1", "devices": list(devices),
        "card_selection": card_selection,
        "restart": None if checkpoint is None else {"path": str(cycling["restart_from"]), "step": start_step},
        "checkpoint_files": checkpoint_files,
        "steps_executed": len(step_rows), "steps_requested": int(schedule["steps"]),
        "arwen_pin": arwen_pin, "memory": memory, "step_receipts": step_rows,
        "admission": admission_records,
        "compile_identity": compile_identity,
        "health": health, "physical_gates": physical_gates, "refusal": refusal,
        "final_gf_tendency_hashes": final_forcing_hashes,
        "walls": {"steps": len(step_rows), "loop_seconds": time.perf_counter() - started, "integration_seconds": sum(row["seconds"] for row in step_rows),
                  "step_seconds": [row["seconds"] for row in step_rows]},
        "snapshot_files": frame_records, "all_published_field_hashes": frame_hashes,
        "identity": {"reference": str(reference) if reference else None, "comparisons": compared,
                     "all_state_invariant": refusal is None and bool(compared)
                         and compared[-1]["step"] == int(schedule["steps"])
                         and all(row["state_invariant"] and not row["unexpected_metadata_diffs"]
                                 for row in compared)},
        "exchange": transport.receipt(), "field_halo_widths": FIELD_WIDTHS,
        "memory_policy": {"setup_stream": setup_stream, "halo_field_release": field_release},
        "rank_exchanges": [exchanger.receipt() for exchanger in exchangers],
        "layouts_sha256": sha256_file(layouts_path), "part_sha256": sha256_file(part_path)}
    return result
