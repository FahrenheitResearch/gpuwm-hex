#!/usr/bin/env python3
"""Per-allocation device-memory ledger for the MPAS-A v8.4.1 CUDA port.

Runs the REAL forecast driver (tools/run_cuda_v841_forecast.py) through the
REAL mesh binder (tools/mpas_mesh_binding.py) and records where device memory
actually goes.  Nothing in the port tree is modified: the instruments are

  1. a cupy.cuda.MemoryHook that records (nbytes, site) for every pool
     malloc/free, keyed on the pooled block pointer, with the site resolved to
     the innermost frame inside the port tree -> file:line;
  2. a device CENSUS taken at phase boundaries: every live cupy.ndarray is
     enumerated (dtype, shape, nbytes, block ptr) and named by walking the
     device stack the driver returns, so each resident block gets a real name;
  3. cupy.get_default_memory_pool().used_bytes()/total_bytes();
  4. cudaMemGetInfo (whole-device used) sampled at every real cudaMalloc, so
     the driver-side high-water mark is captured, not projected;
  5. nvidia-smi per-process used_memory for this PID.

The delta (4) - (3.total) is what the pool does not see: CUDA context, module
images, cuBLAS/cuFFT workspaces.

ARMS.  --arm clean installs nothing.  --arm hook installs only the allocation
hook.  --arm full installs the hook AND the boundary censuses.  Comparing the
snapshot digests of the three arms is what demonstrates -- rather than assumes
-- that the instrument did not move a single output bit, which matters here
because the frozen phase-1 longwave seam depends on device-pool contents.

--selftest validates the instrument against known quantities in BOTH
directions before any of it is trusted.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024.0 * 1024.0

# The hook's own frames, excluded from site resolution.  Excluding the whole
# probe FILE would also hide a caller that happens to live in it.
_HOOK_FRAMES = frozenset(
    {"_site", "malloc_postprocess", "free_postprocess", "alloc_postprocess"}
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def nvsmi_process_mib(pid: int):
    """Per-process device memory as the driver reports it, MiB, or None."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------
# the allocation hook
# --------------------------------------------------------------------------
def build_hook(cupy, roots, depth: int, sample_device: bool):
    probe_file = os.path.abspath(__file__)

    class LedgerHook(cupy.cuda.MemoryHook):
        name = "gpuwm-hex-ledger"

        def __init__(self):
            super().__init__()
            self.roots = tuple(roots)
            self.depth = int(depth)
            self.sample_device = bool(sample_device)
            self.enabled = True

            self.live = {}               # block ptr -> (nbytes, site)
            self.site_alloc_bytes = {}
            self.site_alloc_count = {}
            self.site_live = {}
            self.site_peak_live = {}
            self.site_max_single = {}
            self.site_phases = {}
            self.site_chain = {}

            self.live_bytes = 0
            self.peak_live_bytes = 0
            self.peak_site_live = {}
            self.peak_phase = None
            self._last_peak_copy = 0
            self._peak_copy_step = 4 * 1024 * 1024

            self.driver_alloc_bytes = 0
            self.driver_alloc_count = 0
            self.driver_free_count = 0
            self.peak_device_used = 0

            self.phase = "preload"
            self.phase_alloc_bytes = {}
            self.phase_alloc_count = {}
            self.phase_peak_live = {}

        # -- site resolution ------------------------------------------------
        def _site(self):
            f = sys._getframe(1)
            chain = []
            best = None
            n = 0
            while f is not None and n < self.depth:
                fn = f.f_code.co_filename
                nm = f.f_code.co_name
                # Skip only the hook's OWN frames, not the whole probe file:
                # excluding the file would also hide a caller living in it.
                if not (fn == probe_file and nm in _HOOK_FRAMES):
                    entry = (fn, f.f_lineno, nm)
                    chain.append(entry)
                    if best is None and fn.startswith(self.roots):
                        best = entry
                f = f.f_back
                n += 1
            if best is None:
                for entry in chain:
                    if "cupy" not in entry[0]:
                        best = entry
                        break
            if best is None:
                best = (chain[0] if chain else ("<cupy-internal>", 0, "?"))
            key = "%s:%d" % (best[0], best[1])
            if key not in self.site_chain:
                self.site_chain[key] = [
                    "%s:%d %s" % (c[0], c[1], c[2]) for c in chain[:14]
                ]
            return key

        # -- pool level -----------------------------------------------------
        def malloc_postprocess(self, **kw):
            if not self.enabled:
                return
            ptr = kw.get("mem_ptr", 0)
            if not ptr:
                return
            size = int(kw.get("mem_size", 0))
            site = self._site()
            self.live[ptr] = (size, site)
            self.live_bytes += size
            self.site_alloc_bytes[site] = self.site_alloc_bytes.get(site, 0) + size
            self.site_alloc_count[site] = self.site_alloc_count.get(site, 0) + 1
            cur = self.site_live.get(site, 0) + size
            self.site_live[site] = cur
            if cur > self.site_peak_live.get(site, 0):
                self.site_peak_live[site] = cur
            if size > self.site_max_single.get(site, 0):
                self.site_max_single[site] = size
            ph = self.site_phases.setdefault(site, {})
            ph[self.phase] = ph.get(self.phase, 0) + 1
            self.phase_alloc_bytes[self.phase] = (
                self.phase_alloc_bytes.get(self.phase, 0) + size
            )
            self.phase_alloc_count[self.phase] = (
                self.phase_alloc_count.get(self.phase, 0) + 1
            )
            if self.live_bytes > self.peak_live_bytes:
                self.peak_live_bytes = self.live_bytes
                if self.live_bytes - self._last_peak_copy > self._peak_copy_step:
                    self.peak_site_live = dict(self.site_live)
                    self.peak_phase = self.phase
                    self._last_peak_copy = self.live_bytes
            if self.live_bytes > self.phase_peak_live.get(self.phase, 0):
                self.phase_peak_live[self.phase] = self.live_bytes

        def free_postprocess(self, **kw):
            if not self.enabled:
                return
            ptr = kw.get("mem_ptr", 0)
            rec = self.live.pop(ptr, None)
            if rec is None:
                return
            size, site = rec
            self.live_bytes -= size
            self.site_live[site] = self.site_live.get(site, 0) - size

        # -- driver level (real cudaMalloc) ---------------------------------
        def alloc_postprocess(self, **kw):
            if not self.enabled:
                return
            self.driver_alloc_bytes += int(kw.get("mem_size", 0))
            self.driver_alloc_count += 1
            if self.sample_device:
                try:
                    free_b, total_b = cupy.cuda.runtime.memGetInfo()
                    used = total_b - free_b
                    if used > self.peak_device_used:
                        self.peak_device_used = used
                except Exception:
                    pass

        # -- reporting ------------------------------------------------------
        def snapshot(self):
            return {
                "live_bytes": self.live_bytes,
                "live_blocks": len(self.live),
                "peak_live_bytes": self.peak_live_bytes,
                "driver_alloc_bytes": self.driver_alloc_bytes,
                "driver_alloc_count": self.driver_alloc_count,
                "peak_device_used_bytes": self.peak_device_used,
            }

    return LedgerHook()


# --------------------------------------------------------------------------
# device census: name every live cupy.ndarray
# --------------------------------------------------------------------------
_CONTAINER_TYPES = (dict, list, tuple, set, frozenset)


def name_device_arrays(cupy, root, root_name="stack", max_depth=9, max_nodes=400000):
    """Walk an object graph and map pooled block ptr -> dotted names.

    Read-only.  Never calls gc.collect(); never mutates anything it walks.
    """
    ndarray = cupy.ndarray
    names = {}
    seen = set()
    stack = [(root, root_name, 0)]
    nodes = 0
    while stack and nodes < max_nodes:
        obj, path, depth = stack.pop()
        nodes += 1
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, ndarray):
            try:
                ptr = int(obj.data.mem.ptr)
            except Exception:
                continue
            names.setdefault(ptr, []).append(path)
            continue
        if depth >= max_depth:
            continue
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str):
                        stack.append((v, "%s.%s" % (path, k), depth + 1))
                    else:
                        stack.append((v, "%s[%r]" % (path, k), depth + 1))
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj[:512]):
                    stack.append((v, "%s[%d]" % (path, i), depth + 1))
            elif isinstance(obj, (set, frozenset)):
                continue
            elif hasattr(obj, "__dict__") and not isinstance(obj, type):
                for k, v in list(vars(obj).items()):
                    if isinstance(k, str) and not k.startswith("__"):
                        stack.append((v, "%s.%s" % (path, k), depth + 1))
            elif hasattr(obj, "__slots__") and not isinstance(obj, type):
                for k in obj.__slots__:
                    try:
                        stack.append((getattr(obj, k), "%s.%s" % (path, k), depth + 1))
                    except Exception:
                        pass
        except Exception:
            continue
    return names


def census_device_arrays(cupy, hook, name_map, label):
    """Enumerate every live cupy.ndarray. Read-only; no gc.collect()."""
    ndarray = cupy.ndarray
    blocks = {}
    arrays = 0
    objs = gc.get_objects()
    for o in objs:
        if type(o) is not ndarray:
            continue
        arrays += 1
        try:
            mem = o.data.mem
            ptr = int(mem.ptr)
            block = int(getattr(mem, "size", 0) or 0)
            nb = int(o.nbytes)
            dt = str(o.dtype)
            shape = tuple(int(s) for s in o.shape)
        except Exception:
            continue
        rec = blocks.get(ptr)
        if rec is None:
            rec = {
                "block_bytes": block,
                "views": 0,
                "view_bytes": 0,
                "dtypes": {},
                "shapes": {},
            }
            blocks[ptr] = rec
        rec["views"] += 1
        rec["view_bytes"] = max(rec["view_bytes"], nb)
        rec["dtypes"][dt] = rec["dtypes"].get(dt, 0) + 1
        key = "x".join(str(s) for s in shape)
        rec["shapes"][key] = rec["shapes"].get(key, 0) + 1
    del objs

    rows = []
    total_block = 0
    for ptr, rec in blocks.items():
        site = None
        if hook is not None:
            live = hook.live.get(ptr)
            if live is not None:
                site = live[1]
        nm = name_map.get(ptr) if name_map else None
        total_block += rec["block_bytes"]
        rows.append(
            {
                "ptr": ptr,
                "block_bytes": rec["block_bytes"],
                "largest_view_bytes": rec["view_bytes"],
                "views": rec["views"],
                "dtype": max(rec["dtypes"].items(), key=lambda kv: kv[1])[0],
                "shape": max(rec["shapes"].items(), key=lambda kv: kv[1])[0],
                "names": sorted(set(nm))[:4] if nm else [],
                "site": site,
            }
        )
    rows.sort(key=lambda r: -r["block_bytes"])
    return {
        "label": label,
        "arrays": arrays,
        "blocks": len(blocks),
        "resident_block_bytes": total_block,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# seam trace: which physics scheme moves the NON-POOL residue
# --------------------------------------------------------------------------
def install_seam_trace(cp, stack, ledger, hook, max_steps):
    """Sample whole-device usage around each physics scheme call.

    The non-pool residue is invisible to the allocation hook by construction,
    so the only way to attribute it is to bracket the calls with
    cudaMemGetInfo.  Observation only: no argument, order or result is
    touched.
    """
    trace = ledger.setdefault("seam_trace", [])
    pool = cp.get_default_memory_pool()
    state = {"step": 0}

    def sample():
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return total_b - free_b, int(pool.total_bytes())

    def wrap(obj, name, label):
        # Patch the CLASS, never the instance: gpuwm's restart manifest
        # classifies every PhysicsDriver INSTANCE attribute and refuses an
        # unclassified one by name (gpuwm/io/restart.py).  That refusal is
        # correct; a class-level wrapper leaves the instance __dict__ alone.
        cls = type(obj)
        orig = getattr(cls, name, None)
        if orig is None or not callable(orig):
            return False

        def wrapped(self, *a, **k):
            if state["step"] > max_steps:
                return orig(self, *a, **k)
            d0, p0 = sample()
            try:
                return orig(self, *a, **k)
            finally:
                cp.cuda.get_current_stream().synchronize()
                d1, p1 = sample()
                if (d1 - d0) - (p1 - p0) != 0 or (p1 - p0) != 0:
                    trace.append(
                        {
                            "label": label,
                            "step": state["step"],
                            "device_delta": d1 - d0,
                            "pool_total_delta": p1 - p0,
                            "non_pool_delta": (d1 - d0) - (p1 - p0),
                            "device_after": d1,
                        }
                    )

        try:
            setattr(cls, name, wrapped)
            return True
        except Exception:
            return False

    installed = []
    try:
        seam = stack["backend"]._seam
        for nm in ("run_phase1", "run_phase2"):
            if wrap(seam, nm, "seam." + nm):
                installed.append("seam." + nm)
        drv = seam._driver
        cls = type(drv)
        for nm in dir(cls):
            if not nm.startswith("_run_"):
                continue
            if not callable(getattr(cls, nm, None)):
                continue
            if wrap(drv, nm, "driver." + nm):
                installed.append("driver." + nm)
        # the dispatch-selected scheme methods (sfclay / land / PBL)
        try:
            from gpuwm.core.physics import resolve_physics_dispatch

            for key, nm in resolve_physics_dispatch(seam._cfg).items():
                if isinstance(nm, str) and wrap(drv, nm, "dispatch.%s=%s" % (key, nm)):
                    installed.append("dispatch." + nm)
        except Exception as exc:
            ledger["seam_trace_dispatch_error"] = repr(exc)
    except Exception as exc:
        ledger["seam_trace_error"] = repr(exc)
    ledger["seam_trace_installed"] = installed

    orig_step_marker = ledger.setdefault("_seam_step_marker", {})
    orig_step_marker["state"] = state
    return state


# --------------------------------------------------------------------------
# instrument validation -- both directions, against known quantities
# --------------------------------------------------------------------------
def selftest(args) -> int:
    import cupy as cp

    failures = []

    def check(label, ok, detail=""):
        print("[selftest] %-6s %s  %s" % ("PASS" if ok else "FAIL", label, detail))
        if not ok:
            failures.append(label)

    pool = cp.get_default_memory_pool()
    # roots mirror the real run: only OUR code counts as an attribution site,
    # so the innermost frame under a root is this file, not cupy's internals.
    hook = build_hook(
        cp, roots=(os.path.dirname(os.path.abspath(__file__)),), depth=30,
        sample_device=True,
    )
    hook.__enter__()

    # Warm the context and the pool so the baseline is a steady state.
    warm = cp.zeros(1024, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()
    del warm
    pool.free_all_blocks()

    free0, total0 = cp.cuda.runtime.memGetInfo()
    used0 = total0 - free0
    pool_used0 = pool.used_bytes()
    live0 = hook.live_bytes
    smi0 = nvsmi_process_mib(os.getpid())

    KNOWN = 256 * 1024 * 1024  # exactly 256 MiB of float32
    n = KNOWN // 4
    known_line = sys._getframe().f_lineno + 1
    known = cp.zeros(n, dtype=cp.float32)
    cp.cuda.get_current_stream().synchronize()

    free1, total1 = cp.cuda.runtime.memGetInfo()
    used1 = total1 - free1
    pool_used1 = pool.used_bytes()
    live1 = hook.live_bytes
    smi1 = nvsmi_process_mib(os.getpid())

    # POSITIVE 1: the hook saw exactly the known quantity.
    check(
        "hook sees a known 256.000 MiB device array",
        live1 - live0 == KNOWN,
        "delta=%.3f MiB (want %.3f)" % ((live1 - live0) / MIB, KNOWN / MIB),
    )
    # POSITIVE 2: the pool agrees.
    check(
        "pool.used_bytes() agrees with the hook",
        pool_used1 - pool_used0 == KNOWN,
        "delta=%.3f MiB" % ((pool_used1 - pool_used0) / MIB),
    )
    # POSITIVE 3: the driver agrees to within one pool chunk.
    check(
        "cudaMemGetInfo moved by at least the known quantity",
        used1 - used0 >= KNOWN,
        "delta=%.3f MiB" % ((used1 - used0) / MIB),
    )
    # POSITIVE 4: nvidia-smi per-process agrees.
    if smi0 is not None and smi1 is not None:
        check(
            "nvidia-smi per-process moved by at least the known quantity",
            (smi1 - smi0) * MIB >= KNOWN * 0.98,
            "delta=%.1f MiB (smi %.1f -> %.1f)" % (smi1 - smi0, smi0, smi1),
        )
    else:
        check("nvidia-smi per-process query works", False, "PID not listed")

    # POSITIVE 5: the site is THIS file at the allocating line, and the census
    # finds the array with the right dtype/shape/nbytes.
    ptr = int(known.data.mem.ptr)
    site = hook.live.get(ptr, (0, None))[1]
    want_site = "%s:%d" % (os.path.abspath(__file__), known_line)
    check(
        "hook attributes the block to the EXACT allocating file:line",
        site == want_site,
        "site=%s (want %s)" % (site, want_site),
    )
    # NEGATIVE 0: the attribution is not a constant -- a second allocation on a
    # different line must be attributed to THAT line, not the first one.
    other_line = sys._getframe().f_lineno + 1
    other = cp.zeros(1024 * 1024, dtype=cp.float64)
    other_site = hook.live.get(int(other.data.mem.ptr), (0, None))[1]
    check(
        "a different allocating line gets a DIFFERENT site",
        other_site == "%s:%d" % (os.path.abspath(__file__), other_line)
        and other_site != site,
        "other_site=%s" % other_site,
    )
    del other
    cen = census_device_arrays(cp, hook, {ptr: ["selftest.known"]}, "selftest")
    row = next((r for r in cen["rows"] if r["ptr"] == ptr), None)
    check(
        "census finds the known array with correct dtype/shape/bytes",
        row is not None
        and row["dtype"] == "float32"
        and row["shape"] == str(n)
        and row["largest_view_bytes"] == KNOWN,
        "row=%s" % (
            None if row is None
            else "%s %s %.3f MiB names=%s"
            % (row["dtype"], row["shape"], row["largest_view_bytes"] / MIB, row["names"])
        ),
    )

    # NEGATIVE 1: a HOST array of the same size moves nothing.
    import numpy as np

    hb_live = hook.live_bytes
    hb_pool = pool.used_bytes()
    hb_free, hb_total = cp.cuda.runtime.memGetInfo()
    host = np.zeros(n, dtype=np.float32)
    host[0] = 1.0
    ha_free, ha_total = cp.cuda.runtime.memGetInfo()
    check(
        "a 256 MiB HOST array moves none of the three device instruments",
        hook.live_bytes == hb_live
        and pool.used_bytes() == hb_pool
        and (hb_total - hb_free) == (ha_total - ha_free),
        "hook=%d pool=%d device=%d (all deltas must be 0)"
        % (
            hook.live_bytes - hb_live,
            pool.used_bytes() - hb_pool,
            (ha_total - ha_free) - (hb_total - hb_free),
        ),
    )
    del host

    # NEGATIVE 2: freeing the known array retracts it from hook and pool.
    del known
    cen2 = census_device_arrays(cp, hook, {}, "selftest-after-free")
    check(
        "freeing the known array retracts it from the hook",
        hook.live_bytes == live0,
        "live=%.3f MiB (baseline %.3f)" % (hook.live_bytes / MIB, live0 / MIB),
    )
    check(
        "freeing the known array retracts it from the pool",
        pool.used_bytes() == pool_used0,
        "pool.used=%.3f MiB" % (pool.used_bytes() / MIB),
    )
    check(
        "census no longer lists the freed block",
        not any(r["ptr"] == ptr for r in cen2["rows"]),
    )

    # NEGATIVE 3: a deliberately wrong expectation must NOT be reported as met.
    check(
        "an untrue expectation is refused (512 MiB claim against a 256 MiB array)",
        (live1 - live0) != 512 * 1024 * 1024,
    )

    # Resolution limits, stated.
    print(
        "[selftest] RESOLUTION LIMIT: the hook sees the CuPy default pool only. "
        "CUDA context, module images and library workspaces are NOT pool "
        "allocations; they are measured as (cudaMemGetInfo used) minus "
        "(pool total_bytes) and are reported as one residual class, not "
        "itemised."
    )
    print(
        "[selftest] RESOLUTION LIMIT: block_bytes is the POOL BLOCK size, "
        "which is >= the array nbytes (CuPy rounds up); the ledger reports "
        "block bytes because that is what the card actually loses."
    )
    print(
        "[selftest] RESOLUTION LIMIT: nvidia-smi per-process granularity is "
        "1 MiB and on some drivers lags by up to a sampling interval."
    )
    hook.__exit__(None, None, None)
    print("[selftest] %s" % ("ALL PASS" if not failures else "FAILURES: %r" % failures))
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# the instrumented run
# --------------------------------------------------------------------------
def run(args) -> int:
    repo = Path(args.repo).resolve(strict=True)
    sys.path.insert(0, str(repo / "src"))
    binding = _load("mpas_mesh_binding", repo / "tools" / "mpas_mesh_binding.py")
    forecast = _load("v841_forecast", repo / "tools" / "run_cuda_v841_forecast.py")
    proof = forecast.proof

    # The ArWen checkout is a root too.  Without it every allocation made
    # inside gpuwm/core collapses onto the single port line that calls the
    # seam, which hides the largest pool class in the run behind one number.
    roots = (str(repo / "src"), str(repo / "tools"),
             str(Path(args.arwen_checkout).resolve()))

    rest = [
        "--grid", str(args.grid),
        "--static", str(args.static),
        "--init", str(args.init),
        "--init-source", args.init_source,
        "--hours", str(args.hours),
        "--history-every-minutes", str(args.history_every_minutes),
        "--arwen-checkout", str(args.arwen_checkout),
        "--cache-root", str(args.cache_root),
        "--output", str(args.output),
        "--case-label", args.case_label,
    ]
    if args.required_free_bytes is not None:
        # Forwarded verbatim to the driver's admission, exactly as the
        # forecast door forwards its own verdict: a card's measured row may
        # admit a mesh the default model's fixed term refuses.
        rest += ["--required-free-bytes", str(int(args.required_free_bytes))]

    # The real mesh bind, fail-closed, exactly as tools/run_cuda_v841_forecast_mesh.py does.
    bind_receipt = binding.bind_mesh(
        proof, args.mesh, grid=Path(args.grid), static=Path(args.static),
        forecast=forecast,
    )

    ledger = {
        "mesh": args.mesh,
        "arm": args.arm,
        "n_cells": binding.MESH_BINDINGS[args.mesh].n_cells,
        "n_edges": binding.MESH_BINDINGS[args.mesh].n_edges,
        "n_levels": binding.MESH_BINDINGS[args.mesh].n_levels,
        "bind_rebound": bind_receipt.get("rebound"),
        "pid": os.getpid(),
        "argv": rest,
        "phases": [],
        "censuses": [],
    }

    hook = None
    cp = None
    name_map = {}

    if args.arm in ("hook", "full"):
        import cupy as _cp

        cp = _cp
        hook = build_hook(cp, roots=roots, depth=args.depth,
                          sample_device=args.arm == "full")
        hook.__enter__()
        ledger["instrument"] = {
            "hook": True,
            "census": args.arm == "full",
            "device_sample_in_hook": args.arm == "full",
            "cupy": cp.__version__,
        }
    else:
        ledger["instrument"] = {"hook": False, "census": False}

    def sample(label, do_census=False, root_obj=None):
        rec = {"label": label, "t": time.time()}
        if cp is not None:
            pool = cp.get_default_memory_pool()
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                rec["device_used_bytes"] = total_b - free_b
                rec["device_total_bytes"] = total_b
            except Exception:
                pass
            rec["pool_used_bytes"] = int(pool.used_bytes())
            rec["pool_total_bytes"] = int(pool.total_bytes())
            rec["nvsmi_process_mib"] = nvsmi_process_mib(os.getpid())
        if hook is not None:
            rec.update(hook.snapshot())
        ledger["phases"].append(rec)
        if do_census and hook is not None and args.arm == "full":
            nm = dict(name_map)
            if root_obj is not None:
                nm.update(name_device_arrays(cp, root_obj))
            cen = census_device_arrays(cp, hook, nm, label)
            cen["pool_used_bytes"] = rec.get("pool_used_bytes")
            cen["pool_total_bytes"] = rec.get("pool_total_bytes")
            cen["device_used_bytes"] = rec.get("device_used_bytes")
            ledger["censuses"].append(cen)
        print(
            "[phase] %-28s live=%.1f MiB pool_used=%.1f pool_total=%.1f "
            "device=%.1f smi=%s"
            % (
                label,
                rec.get("live_bytes", 0) / MIB,
                rec.get("pool_used_bytes", 0) / MIB,
                rec.get("pool_total_bytes", 0) / MIB,
                rec.get("device_used_bytes", 0) / MIB,
                rec.get("nvsmi_process_mib"),
            ),
            flush=True,
        )

    # ---- phase instrumentation: pure observation, no behaviour changed ----
    stack_holder = {}
    if hook is not None:
        orig_prepare = forecast.prepare_forecast_host
        orig_admission = proof.gpu_memory_admission
        orig_construct = proof._construct_device_stack
        orig_capture = proof.capture_snapshot
        orig_step = proof.execute_composite_step
        orig_health = forecast.step_health_gate
        orig_write = proof.write_snapshot_netcdf

        def w_prepare(*a, **k):
            hook.phase = "host_prepare"
            sample("host_prepare:begin")
            out = orig_prepare(*a, **k)
            sample("host_prepare:end")
            return out

        def w_admission(*a, **k):
            # Called immediately after require_cuda() and `import cupy`, and
            # BEFORE KernelCache is constructed.  This is the CUDA-context
            # baseline: context + whatever cupy loaded, no port kernels yet.
            hook.phase = "cuda_context"
            sample("cuda_context:after_require_cuda")
            out = orig_admission(*a, **k)
            hook.phase = "kernel_cache_build"
            return out

        def w_construct(*a, **k):
            hook.phase = "device_stack_construct"
            sample("construct:begin")
            out = orig_construct(*a, **k)
            stack_holder["stack"] = out
            if args.seam_trace:
                install_seam_trace(cp, out, ledger, hook, args.seam_trace_steps)
            hook.phase = "resident"
            sample("construct:end", do_census=True, root_obj=out)
            if args.arm == "full":
                name_map.update(name_device_arrays(cp, out))
            return out

        def w_capture(*a, **k):
            prev = hook.phase
            hook.phase = "capture_snapshot"
            out = orig_capture(*a, **k)
            hook.phase = prev
            return out

        step_no = {"n": 0}

        def w_step(*a, **k):
            step_no["n"] += 1
            n = step_no["n"]
            marker = ledger.get("_seam_step_marker", {}).get("state")
            if marker is not None:
                marker["step"] = n
            hook.phase = "step_integrate"
            out = orig_step(*a, **k)
            hook.phase = "resident"
            return out

        def w_health(stack, step, cpmod):
            hook.phase = "health_gate"
            out = orig_health(stack, step, cpmod)
            hook.phase = "resident"
            if step <= args.census_steps:
                sample("after_step_%d" % step, do_census=True,
                       root_obj=stack_holder.get("stack"))
            else:
                sample("after_step_%d" % step)
            return out

        def w_write(*a, **k):
            prev = hook.phase
            hook.phase = "history_write"
            out = orig_write(*a, **k)
            hook.phase = prev
            return out

        forecast.prepare_forecast_host = w_prepare
        proof.gpu_memory_admission = w_admission
        proof._construct_device_stack = w_construct
        proof.capture_snapshot = w_capture
        proof.execute_composite_step = w_step
        forecast.step_health_gate = w_health
        proof.write_snapshot_netcdf = w_write

    t0 = time.perf_counter()
    rc = forecast.main(rest)
    wall = time.perf_counter() - t0
    ledger["rc"] = rc
    ledger["wall_seconds"] = wall

    # ---- kernel inventory ------------------------------------------------
    # Every launched kernel reserves local memory for the WHOLE device:
    # local_size_bytes * resident_threads, taken as a high-water mark by the
    # context.  That reservation is NOT a pool allocation and scales with the
    # CARD (SM count), not with the mesh.
    if cp is not None:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            mtpsm = cp.cuda.runtime.deviceGetAttribute(
                cp.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor, 0
            )
            ledger["device"] = {
                "name": props["name"].decode(),
                "sms": int(props["multiProcessorCount"]),
                "max_threads_per_sm": int(mtpsm),
                "resident_threads": int(props["multiProcessorCount"]) * int(mtpsm),
                "total_bytes": int(props["totalGlobalMem"]),
            }
        except Exception as exc:
            ledger["device_error"] = repr(exc)

    # EVERY RawKernel alive in the process, not just the port's KernelCache --
    # the ArWen physics seam (gpuwm.core.*) compiles into its OWN cache, and
    # those column kernels are the ones with big local frames.
    kernels = []
    try:
        port_cache_keys = set()
        cache_obj = None
        if stack_holder.get("stack") is not None:
            cache_obj = stack_holder["stack"]["driver"].cache
            port_cache_keys = {
                id(k) for k in getattr(cache_obj, "_kernels", {}).values()
            }
            ledger["port_kernel_modules"] = len(getattr(cache_obj, "_modules", {}))
            ledger["port_compile_seconds_total"] = float(
                sum(getattr(cache_obj, "_compile_seconds", {}).values())
            )
        seen_k = set()
        for o in gc.get_objects():
            # Duck-typed, not `type(o) is cp.RawKernel`: the ArWen seam wraps
            # or subclasses its launchers, and an exact-type test silently
            # misses exactly the high-local physics columns we are hunting.
            if not hasattr(type(o), "local_size_bytes"):
                continue
            if isinstance(o, type):
                continue
            try:
                _ = o.local_size_bytes
            except Exception:
                continue
            if id(o) in seen_k:
                continue
            seen_k.add(id(o))
            row = {
                "owner": "port_cache" if id(o) in port_cache_keys else "other",
                "pytype": type(o).__name__,
            }
            for attr in (
                "local_size_bytes",
                "shared_size_bytes",
                "const_size_bytes",
                "num_regs",
                "max_threads_per_block",
            ):
                try:
                    row[attr] = int(getattr(o, attr))
                except Exception:
                    row[attr] = None
            try:
                row["name"] = o.name
            except Exception:
                row["name"] = "?"
            kernels.append(row)
        ledger["kernel_modules"] = ledger.get("port_kernel_modules")
        ledger["compile_seconds_total"] = ledger.get("port_compile_seconds_total", 0.0)
    except Exception as exc:
        ledger["kernel_inventory_error"] = repr(exc)
    kernels.sort(key=lambda r: -(r.get("local_size_bytes") or 0))
    ledger["kernels"] = kernels
    if kernels and ledger.get("device"):
        top = kernels[0]
        rt = ledger["device"]["resident_threads"]
        by_owner = {}
        for k in kernels:
            o = k["owner"]
            b = by_owner.setdefault(o, {"count": 0, "max_local": 0, "nonzero_local": 0})
            b["count"] += 1
            L = k.get("local_size_bytes") or 0
            b["max_local"] = max(b["max_local"], L)
            if L:
                b["nonzero_local"] += 1
        ledger["local_memory_model"] = {
            "max_local_size_bytes": top.get("local_size_bytes"),
            "max_local_kernel": top.get("name"),
            "max_local_owner": top.get("owner"),
            "kernel_count": len(kernels),
            "by_owner": by_owner,
            "resident_threads": rt,
            "predicted_reservation_bytes": (top.get("local_size_bytes") or 0) * rt,
            "note": (
                "predicted is the UPPER bound local_size * resident_threads; "
                "CUDA reserves for the ACHIEVABLE occupancy, which falls for "
                "high-local kernels, so the measured reservation is <= this. "
                "It is a per-CONTEXT high-water mark over launched kernels, "
                "and it scales with the CARD (SM count), never with the mesh."
            ),
        }

    if cp is None:
        try:
            import cupy as cp  # noqa: F811
        except Exception:
            cp = None
    sample("final")

    if hook is not None:
        hook.enabled = False
        pool = cp.get_default_memory_pool()
        ledger["pool_final"] = {
            "used_bytes": int(pool.used_bytes()),
            "total_bytes": int(pool.total_bytes()),
        }
        ledger["hook_totals"] = hook.snapshot()
        ledger["peak_phase"] = hook.peak_phase
        ledger["phase_alloc_bytes"] = hook.phase_alloc_bytes
        ledger["phase_alloc_count"] = hook.phase_alloc_count
        ledger["phase_peak_live_bytes"] = hook.phase_peak_live
        sites = []
        for site, peak in hook.site_peak_live.items():
            sites.append(
                {
                    "site": site,
                    "peak_live_bytes": peak,
                    "live_at_global_peak_bytes": hook.peak_site_live.get(site, 0),
                    "total_alloc_bytes": hook.site_alloc_bytes.get(site, 0),
                    "alloc_count": hook.site_alloc_count.get(site, 0),
                    "max_single_bytes": hook.site_max_single.get(site, 0),
                    "phases": hook.site_phases.get(site, {}),
                    "still_live_bytes": max(0, hook.site_live.get(site, 0)),
                    "chain": hook.site_chain.get(site, [])[:8],
                }
            )
        sites.sort(key=lambda s: -s["live_at_global_peak_bytes"])
        ledger["sites"] = sites
        hook.__exit__(None, None, None)

    # ---- the drain: what the card still holds with the pool given back ----
    # Everything mesh-scaling lives in the pool.  Returning every free block to
    # the driver leaves exactly the mesh-INDEPENDENT residue resident: CUDA
    # context, module images, library workspaces and the local-memory
    # reservation.  Measured, not modelled.  Runs AFTER the forecast, so it
    # cannot touch a single output bit.
    if cp is not None:
        try:
            stack_holder.pop("stack", None)
            gc.collect()
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            free_b, total_b = cp.cuda.runtime.memGetInfo()
            ledger["drain"] = {
                "device_used_bytes": total_b - free_b,
                "device_total_bytes": total_b,
                "pool_used_bytes": int(pool.used_bytes()),
                "pool_total_bytes": int(pool.total_bytes()),
                "nvsmi_process_mib": nvsmi_process_mib(os.getpid()),
                "note": (
                    "device_used here is the mesh-INDEPENDENT resident residue: "
                    "CUDA context + module images + library workspaces + "
                    "per-thread local-memory reservation.  No pool blocks remain."
                ),
            }
            print(
                "[drain] mesh-independent residue = %.1f MiB "
                "(pool_total now %.1f MiB, smi %s)"
                % (
                    (total_b - free_b) / MIB,
                    pool.total_bytes() / MIB,
                    ledger["drain"]["nvsmi_process_mib"],
                ),
                flush=True,
            )
        except Exception as exc:
            ledger["drain_error"] = repr(exc)

    # snapshot digests: the bitwise comparison surface across arms
    receipt_path = Path(args.output) / "receipt.json"
    if not receipt_path.exists():
        cands = sorted(Path(args.output).glob("*.json"))
        receipt_path = cands[0] if cands else None
    if receipt_path is not None and receipt_path.exists():
        try:
            rec = json.loads(receipt_path.read_text())
            ledger["receipt_file"] = str(receipt_path)
            ledger["snapshot_projection"] = rec.get("snapshot_projection")
            ledger["snapshot_q2"] = rec.get("snapshot_q2")
        except Exception as exc:
            ledger["receipt_read_error"] = repr(exc)

    Path(args.ledger_json).write_text(json.dumps(ledger, indent=1, default=str))
    print("[ledger] wrote %s  rc=%d wall=%.1fs" % (args.ledger_json, rc, wall))
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--repo", type=Path)
    p.add_argument("--mesh")
    p.add_argument("--grid", type=Path)
    p.add_argument("--static", type=Path)
    p.add_argument("--init", type=Path)
    p.add_argument("--init-source", default="instrumented allocation-ledger run")
    p.add_argument("--hours", type=float, default=0.2)
    p.add_argument("--history-every-minutes", type=int, default=12)
    p.add_argument("--arwen-checkout", type=Path)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--case-label", default="hex-ledger")
    p.add_argument("--ledger-json", type=Path)
    p.add_argument(
        "--required-free-bytes",
        type=int,
        default=None,
        help="forwarded to the forecast driver's admission floor (a card's "
             "own measured row, computed by the caller); default: the "
             "mesh-bound floor from the shared admission surface",
    )
    p.add_argument("--arm", choices=("clean", "hook", "full"), default="full")
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--census-steps", type=int, default=2)
    p.add_argument("--seam-trace", action="store_true",
                   help="bracket each physics scheme call with cudaMemGetInfo "
                        "to attribute the NON-POOL residue")
    p.add_argument("--seam-trace-steps", type=int, default=2)
    args = p.parse_args(argv)
    if args.selftest:
        return selftest(args)
    for req in ("repo", "mesh", "grid", "static", "init", "cache_root", "output",
                "ledger_json", "arwen_checkout"):
        if getattr(args, req) is None:
            p.error("--%s is required for a run" % req.replace("_", "-"))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
