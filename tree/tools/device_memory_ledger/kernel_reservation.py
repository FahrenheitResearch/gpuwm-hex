#!/usr/bin/env python3
"""Per-kernel device local-memory reservation tracer for the CUDA MPAS port.

The CuPy memory pool cannot see the single largest mesh-independent device
allocation the port makes.  CUDA reserves a *local-memory backing store* the
first time a kernel with a non-zero local frame is launched:

    reservation_bytes ~= local_size_bytes * (threads resident at the kernel's
                                             achievable occupancy)

It is taken once per CUDA context, it is never returned while the context
lives, and it scales with the CARD (SM count), not with the mesh.  A pool
hook records ``nbytes`` for pool allocations only, so this class shows up as
an unattributed non-pool residue in every pool-based ledger.

This module attributes it kernel by kernel by bracketing the FIRST launches
of every kernel with ``cudaMemGetInfo`` and subtracting the pool's own
movement over the same window.

Why a proxy and not a gc scan: ``RawModule.get_function`` returns a fresh
``RawKernel`` that callers keep in a local variable (gpuwm/core/gf.py:166 is
the example that matters -- it is rebound on every cumulus call).  By the
time a post-run ``gc.get_objects()`` sweep runs, that kernel object is gone,
so a scan reports the port's retained kernels and silently misses the
biggest local frame in the process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

MIB = 1024.0 * 1024.0

# CUfunction_attribute (cuda.h).  Queried through the driver so the numbers
# are the driver's, not a Python-side guess.
CU_FUNC_ATTRIBUTE = {
    "max_threads_per_block": 0,
    "shared_size_bytes": 1,
    "const_size_bytes": 2,
    "local_size_bytes": 3,
    "num_regs": 4,
    "ptx_version": 5,
    "binary_version": 6,
}


def nvsmi_process_mib(pid):
    """This process's device memory as the DRIVER reports it, not the pool."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False).stdout
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


def func_attributes(cp, fn):
    """Driver-reported attributes of a compiled function, or {} if unreachable."""
    ptr = None
    for attr in ("ptr", "kernel"):
        obj = getattr(fn, attr, None)
        if obj is None:
            continue
        ptr = getattr(obj, "ptr", obj)
        if isinstance(ptr, int):
            break
        ptr = None
    if not isinstance(ptr, int):
        # RawKernel exposes the same numbers as properties.
        out = {}
        for name in CU_FUNC_ATTRIBUTE:
            try:
                out[name] = int(getattr(fn, name))
            except Exception:
                pass
        return out
    out = {}
    for name, code in CU_FUNC_ATTRIBUTE.items():
        try:
            out[name] = int(cp.cuda.driver.funcGetAttribute(code, ptr))
        except Exception:
            pass
    return out


class KernelReservationTracer:
    """Bracket first launches with cudaMemGetInfo; attribute the non-pool jump."""

    def __init__(self, cp, roots=(), max_traced_launches=2, depth=24):
        self.cp = cp
        self.roots = tuple(str(r) for r in roots)
        self.max_traced = int(max_traced_launches)
        self.depth = int(depth)
        self.records = {}          # kernel name -> record
        self.order = []            # kernel names in first-launch order
        self.modules = []          # every RawModule built, with its size
        self.module_loads = []     # every module IMAGE load, sized per process
        self._local_cache = {}
        self.installed = False
        self._real_module = None
        self._real_kernel = None
        self._pool = cp.get_default_memory_pool()
        self._t0 = time.perf_counter()

    # -- sampling ---------------------------------------------------------
    def sample(self):
        free_b, total_b = self.cp.cuda.runtime.memGetInfo()
        return total_b - free_b, int(self._pool.total_bytes())

    def _site(self):
        frames = []
        for fr in traceback.extract_stack()[:-3][-self.depth:]:
            fn = fr.filename
            if self.roots and not any(fn.startswith(r) for r in self.roots):
                continue
            if "kernel_reservation.py" in fn:
                continue
            frames.append("%s:%d" % (fn, fr.lineno))
        return frames[-1] if frames else "", frames

    # -- per-process cross-check ------------------------------------------
    def local_size_of(self, fn):
        """local_size_bytes BEFORE the first launch, so the expensive
        per-process bracket is spent only on kernels that can reserve."""
        key = id(fn)
        val = self._local_cache.get(key)
        if val is None:
            val = func_attributes(self.cp, fn).get("local_size_bytes") or 0
            self._local_cache[key] = val
        return val

    # -- the record -------------------------------------------------------
    def note_launch(self, fn, name, dev_delta, pool_delta, wall, nvsmi=None):
        rec = self.records.get(name)
        if rec is None:
            site, stack = self._site()
            rec = {
                "kernel": name,
                "launches_traced": 0,
                "first_site": site,
                "first_stack": stack[-6:],
                "attrs": func_attributes(self.cp, fn),
                "deltas": [],
            }
            self.records[name] = rec
            self.order.append(name)
        rec["launches_traced"] += 1
        entry = {
            "device_delta_bytes": int(dev_delta),
            "pool_total_delta_bytes": int(pool_delta),
            "non_pool_delta_bytes": int(dev_delta - pool_delta),
            "t": round(time.perf_counter() - self._t0, 3),
            "wall_s": round(wall, 6),
        }
        if nvsmi is not None:
            entry["nvsmi_process_mib_before"] = nvsmi[0]
            entry["nvsmi_process_mib_after"] = nvsmi[1]
            if nvsmi[0] is not None and nvsmi[1] is not None:
                entry["nvsmi_process_delta_mib"] = round(nvsmi[1] - nvsmi[0], 1)
        rec["deltas"].append(entry)

    def should_trace(self, name):
        rec = self.records.get(name)
        return rec is None or rec["launches_traced"] < self.max_traced

    def note_module_load(self, kind, nv0, nv1, d0, d1, detail=None):
        """Device bytes taken by loading one module IMAGE (code + constants)."""
        rec = {
            "kind": kind,
            "nvsmi_process_delta_mib": (None if (nv0 is None or nv1 is None)
                                        else round(nv1 - nv0, 1)),
            "device_delta_bytes": int(d1 - d0),
            "site": self._site()[0],
            "t": round(time.perf_counter() - self._t0, 3),
        }
        if detail:
            rec.update(detail)
        self.module_loads.append(rec)

    def bracket_module(self, kind, thunk, detail=None):
        pid = os.getpid()
        nv0 = nvsmi_process_mib(pid)
        d0, _ = self.sample()
        try:
            return thunk()
        finally:
            self.cp.cuda.runtime.deviceSynchronize()
            d1, _ = self.sample()
            self.note_module_load(kind, nv0, nvsmi_process_mib(pid), d0, d1, detail)

    def traced_call(self, fn, name, a, k):
        """One bracketed launch.

        cudaMemGetInfo is DEVICE-WIDE: a sibling process allocating during the
        window would land on this kernel's row.  For any kernel that can
        actually reserve (local_size_bytes > 0) the window is bracketed again
        with this process's own nvidia-smi row, which no sibling can move.
        """
        cp = self.cp
        want_pp = (self.records.get(name) is None) and self.local_size_of(fn) > 0
        pid = os.getpid()
        nv0 = nvsmi_process_mib(pid) if want_pp else None
        d0, p0 = self.sample()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            cp.cuda.get_current_stream().synchronize()
            wall = time.perf_counter() - t0
            d1, p1 = self.sample()
            nv1 = nvsmi_process_mib(pid) if want_pp else None
            self.note_launch(fn, name, d1 - d0, p1 - p0, wall,
                             nvsmi=((nv0, nv1) if want_pp else None))

    # -- installation -----------------------------------------------------
    def install(self):
        if self.installed:
            return
        cp = self.cp
        tracer = self
        self._real_module = cp.RawModule
        self._real_kernel = cp.RawKernel

        class TracedFunction(object):
            """Transparent proxy around a compiled kernel."""

            def __init__(self, fn, name):
                object.__setattr__(self, "_fn", fn)
                object.__setattr__(self, "_name", name)

            def __call__(self, *a, **k):
                fn = object.__getattribute__(self, "_fn")
                name = object.__getattribute__(self, "_name")
                if not tracer.should_trace(name):
                    return fn(*a, **k)
                return tracer.traced_call(fn, name, a, k)

            def __getattr__(self, item):
                return getattr(object.__getattribute__(self, "_fn"), item)

            def __setattr__(self, item, value):
                setattr(object.__getattribute__(self, "_fn"), item, value)

            def __repr__(self):
                return "Traced<%s>" % object.__getattribute__(self, "_name")

        class TracedRawModule(object):
            def __init__(self, *a, **k):
                m = tracer._real_module(*a, **k)
                object.__setattr__(self, "_m", m)
                object.__setattr__(self, "_loaded", False)
                code = k.get("code")
                if code is None and a:
                    code = a[0]
                tracer.modules.append({
                    "source_chars": len(code) if isinstance(code, str) else None,
                    "options": list(k.get("options") or ()),
                    "backend": k.get("backend"),
                    "t": round(time.perf_counter() - tracer._t0, 3),
                })

            def get_function(self, name):
                m = object.__getattribute__(self, "_m")
                loaded = object.__getattribute__(self, "_loaded")
                if loaded:
                    return TracedFunction(m.get_function(name), name)
                object.__setattr__(self, "_loaded", True)
                # NVRTC + cuModuleLoadData happen here on the first resolve.
                out = {}
                tracer.bracket_module(
                    "RawModule.get_function(first)",
                    lambda: out.setdefault("fn", m.get_function(name)),
                    detail={"first_symbol": name})
                return TracedFunction(out["fn"], name)

            def __getattr__(self, item):
                return getattr(object.__getattribute__(self, "_m"), item)

            def __setattr__(self, item, value):
                setattr(object.__getattribute__(self, "_m"), item, value)

        class TracedCudaModule(object):
            """cupy.cuda.function.Module -- the direct-NVRTC route.

            RRTMG compiles through cupy.cuda.compiler.compile_using_nvrtc and
            loads the PTX itself (gpuwm/core/rrtmg_lw.py:3708) because CuPy's
            RawModule route appends -ftz=true after caller options.  That
            bypasses cp.RawModule entirely, so without this proxy the module
            image and every RRTMG kernel are invisible to the ledger.
            """

            def __init__(self, *a, **k):
                object.__setattr__(self, "_m", tracer._real_cuda_module(*a, **k))

            def load(self, *a, **k):
                m = object.__getattribute__(self, "_m")
                out = {}
                tracer.bracket_module(
                    "cuda.function.Module.load",
                    lambda: out.setdefault("r", m.load(*a, **k)),
                    detail={"ptx_bytes": (len(a[0]) if a and hasattr(a[0], "__len__")
                                          else None)})
                return out.get("r")

            def load_file(self, *a, **k):
                m = object.__getattribute__(self, "_m")
                out = {}
                tracer.bracket_module(
                    "cuda.function.Module.load_file",
                    lambda: out.setdefault("r", m.load_file(*a, **k)))
                return out.get("r")

            def get_function(self, name):
                m = object.__getattribute__(self, "_m")
                return TracedFunction(m.get_function(name), name)

            def __getattr__(self, item):
                return getattr(object.__getattribute__(self, "_m"), item)

            def __setattr__(self, item, value):
                setattr(object.__getattribute__(self, "_m"), item, value)

        class TracedRawKernel(object):
            def __init__(self, *a, **k):
                kern = tracer._real_kernel(*a, **k)
                object.__setattr__(self, "_fn", kern)
                name = k.get("name")
                if name is None and len(a) > 1:
                    name = a[1]
                object.__setattr__(self, "_name", name or getattr(kern, "name", "?"))

            def __call__(self, *a, **k):
                fn = object.__getattribute__(self, "_fn")
                name = object.__getattribute__(self, "_name")
                if not tracer.should_trace(name):
                    return fn(*a, **k)
                return tracer.traced_call(fn, name, a, k)

            def __getattr__(self, item):
                return getattr(object.__getattribute__(self, "_fn"), item)

            def __setattr__(self, item, value):
                setattr(object.__getattribute__(self, "_fn"), item, value)

        cp.RawModule = TracedRawModule
        cp.RawKernel = TracedRawKernel
        try:
            import cupy.cuda.function as _cufunc

            self._cufunc = _cufunc
            self._real_cuda_module = _cufunc.Module
            _cufunc.Module = TracedCudaModule
        except Exception as exc:  # pragma: no cover
            self._cufunc = None
            self.cuda_module_patch_error = repr(exc)
        self.installed = True

    def uninstall(self):
        if not self.installed:
            return
        self.cp.RawModule = self._real_module
        self.cp.RawKernel = self._real_kernel
        if getattr(self, "_cufunc", None) is not None:
            self._cufunc.Module = self._real_cuda_module
        self.installed = False

    # -- output -----------------------------------------------------------
    def device_info(self):
        cp = self.cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        mtpsm = cp.cuda.runtime.deviceGetAttribute(
            cp.cuda.runtime.cudaDevAttrMaxThreadsPerMultiProcessor, 0)
        return {
            "name": props["name"].decode(),
            "sms": int(props["multiProcessorCount"]),
            "max_threads_per_sm": int(mtpsm),
            "resident_threads": int(props["multiProcessorCount"]) * int(mtpsm),
            "total_bytes": int(props["totalGlobalMem"]),
        }

    def snapshot(self):
        dev = None
        try:
            dev = self.device_info()
        except Exception as exc:  # pragma: no cover - device query is cheap
            dev = {"error": repr(exc)}
        rows = []
        for name in self.order:
            rec = self.records[name]
            first = rec["deltas"][0] if rec["deltas"] else {}
            later = rec["deltas"][1:]
            local = rec["attrs"].get("local_size_bytes") or 0
            np_first = first.get("non_pool_delta_bytes", 0)
            implied = (np_first / local) if local else None
            rows.append({
                "kernel": name,
                "local_size_bytes": local,
                "num_regs": rec["attrs"].get("num_regs"),
                "shared_size_bytes": rec["attrs"].get("shared_size_bytes"),
                "const_size_bytes": rec["attrs"].get("const_size_bytes"),
                "max_threads_per_block": rec["attrs"].get("max_threads_per_block"),
                "first_launch_non_pool_bytes": np_first,
                "first_launch_device_bytes": first.get("device_delta_bytes", 0),
                "later_launch_non_pool_bytes": [d["non_pool_delta_bytes"] for d in later],
                "implied_threads": round(implied) if implied else None,
                "nvsmi_process_delta_mib": first.get("nvsmi_process_delta_mib"),
                "first_site": rec["first_site"],
                "first_stack": rec["first_stack"],
                "launches_traced": rec["launches_traced"],
            })
        rows.sort(key=lambda r: -r["first_launch_non_pool_bytes"])
        return {
            "device": dev,
            "kernels": rows,
            "kernel_count": len(rows),
            "modules_built": len(self.modules),
            "modules": self.modules,
            "module_loads": self.module_loads,
            "module_image_nvsmi_total_mib": round(sum(
                (r.get("nvsmi_process_delta_mib") or 0.0)
                for r in self.module_loads), 1),
            "module_image_device_total_bytes": sum(
                r.get("device_delta_bytes", 0) for r in self.module_loads),
            "non_pool_first_launch_total_bytes":
                sum(r["first_launch_non_pool_bytes"] for r in rows),
            "note": (
                "first_launch_non_pool_bytes is device memory taken during the "
                "kernel's first launch that the CuPy pool did not take.  For a "
                "kernel with local_size_bytes > 0 that is the CUDA local-memory "
                "backing store: it is a context-wide high-water mark, it is not "
                "returned, and it scales with the card's resident-thread count, "
                "not with the mesh."
            ),
        }


# --------------------------------------------------------------------------
# instrument validation -- known answers, in both directions
# --------------------------------------------------------------------------
_SELFTEST_SRC = r"""
extern "C" __global__ void probe_nolocal(float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.5f * (float)i;
}

extern "C" __global__ void probe_local(float* out, int n, int m)
{
    float buf[LOCALN];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (int j = 0; j < LOCALN; ++j) buf[j] = 0.001f * (float)(i + j);
    float s = 0.0f;
    /* dynamic index: the compiler cannot keep this frame in registers */
    for (int j = 0; j < LOCALN; ++j) s += buf[(j * m + 7) % LOCALN];
    out[i] = s;
}
"""


def selftest(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--localn", type=int, default=2048,
                   help="floats in the probe kernel's local frame")
    p.add_argument("--alloc-mib", type=float, default=512.0)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    import cupy as cp

    out = {"localn": args.localn, "pid": os.getpid(), "cupy": cp.__version__}
    pool = cp.get_default_memory_pool()

    def dev():
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        return total_b - free_b

    # --- context baseline -------------------------------------------------
    cp.cuda.runtime.free(cp.cuda.runtime.malloc(8))   # force context creation
    cp.cuda.runtime.deviceSynchronize()
    base_dev = dev()
    out["context_baseline_mib"] = round(base_dev / MIB, 1)
    out["nvsmi_baseline_mib"] = nvsmi_process_mib(os.getpid())

    # --- A: a KNOWN array must show up in every meter ---------------------
    want = int(args.alloc_mib * MIB)
    d0, p0 = dev(), pool.used_bytes()
    a = cp.zeros(want // 4, dtype=cp.float32)
    a[0] = 1.0
    cp.cuda.runtime.deviceSynchronize()
    d1, p1 = dev(), pool.used_bytes()
    nv1 = nvsmi_process_mib(os.getpid())
    out["A_known_alloc"] = {
        "requested_bytes": want,
        "array_nbytes": int(a.nbytes),
        "pool_used_delta_bytes": int(p1 - p0),
        "device_delta_bytes": int(d1 - d0),
        "nvsmi_delta_mib": (None if (nv1 is None or out["nvsmi_baseline_mib"] is None)
                            else round(nv1 - out["nvsmi_baseline_mib"], 1)),
        "pool_sees_it": (p1 - p0) == a.nbytes,
        "device_sees_it": (d1 - d0) >= a.nbytes,
    }

    # --- A-negative: no allocation must move nothing ----------------------
    d2, p2 = dev(), pool.used_bytes()
    b = a[10:20]           # a view: no new device bytes
    b += 1.0
    cp.cuda.runtime.deviceSynchronize()
    d3, p3 = dev(), pool.used_bytes()
    out["A_negative_view"] = {
        "pool_used_delta_bytes": int(p3 - p2),
        "device_delta_bytes": int(d3 - d2),
        "passes": (p3 - p2) == 0,
    }
    del a, b
    pool.free_all_blocks()

    # --- B: a KNOWN local frame must show up as NON-POOL device bytes -----
    tracer = KernelReservationTracer(cp, roots=(), max_traced_launches=2)
    tracer.install()
    src = _SELFTEST_SRC.replace("LOCALN", str(args.localn))
    mod = cp.RawModule(code=src, options=("-std=c++17",), backend="nvrtc")
    k_no = mod.get_function("probe_nolocal")
    k_lo = mod.get_function("probe_local")
    n = 1 << 20
    buf = cp.zeros(n, dtype=cp.float32)
    threads = 256
    blocks = (n + threads - 1) // threads
    k_no((blocks,), (threads,), (buf, n))
    k_lo((blocks,), (threads,), (buf, n, 3))
    k_no((blocks,), (threads,), (buf, n))
    k_lo((blocks,), (threads,), (buf, n, 3))
    cp.cuda.runtime.deviceSynchronize()
    tracer.uninstall()
    snap = tracer.snapshot()
    out["B_local_frame"] = snap
    by = {r["kernel"]: r for r in snap["kernels"]}
    nolocal = by.get("probe_nolocal", {})
    haslocal = by.get("probe_local", {})
    out["B_verdict"] = {
        "nolocal_local_size_bytes": nolocal.get("local_size_bytes"),
        "nolocal_first_non_pool_bytes": nolocal.get("first_launch_non_pool_bytes"),
        "local_local_size_bytes": haslocal.get("local_size_bytes"),
        "local_first_non_pool_bytes": haslocal.get("first_launch_non_pool_bytes"),
        "local_second_non_pool_bytes": (haslocal.get("later_launch_non_pool_bytes") or [None])[0],
        "implied_threads": haslocal.get("implied_threads"),
        "resident_threads": snap["device"].get("resident_threads"),
        # the two directions
        "zero_local_reserves_nothing":
            (nolocal.get("first_launch_non_pool_bytes", 1) or 0) < 4 * 1024 * 1024,
        "nonzero_local_reserves_a_lot":
            (haslocal.get("first_launch_non_pool_bytes") or 0) > 64 * 1024 * 1024,
        "reservation_is_one_time":
            ((haslocal.get("later_launch_non_pool_bytes") or [1]) [0] or 0) == 0,
    }
    out["nvsmi_final_mib"] = nvsmi_process_mib(os.getpid())
    txt = json.dumps(out, indent=1)
    print(txt)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(txt)
    v = out["B_verdict"]
    ok = (out["A_known_alloc"]["pool_sees_it"] and out["A_negative_view"]["passes"]
          and v["zero_local_reserves_nothing"] and v["nonzero_local_reserves_a_lot"]
          and v["reservation_is_one_time"])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(selftest(sys.argv[1:]))
