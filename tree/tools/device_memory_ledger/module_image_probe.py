#!/usr/bin/env python3
"""Size every physics module IMAGE, per process, on the real device.

Two module routes exist in this stack and only one of them is reachable from
a cupy.RawModule proxy:

  * cp.RawModule(...)  -- the port's KernelCache and most of gpuwm.core
  * cupy.cuda.compiler.compile_using_nvrtc + cupy.cuda.function.Module.load
    -- RRTMG, because CuPy appends -ftz=true after caller options and that
    would flush the subnormal transmittances (gpuwm/core/rrtmg_lw.py:3708)

A proxy on cupy.cuda.function.Module is not an option: CuPy's own Cython
elementwise path constructs and type-checks that exact class, so patching it
kills every cupy expression in the process.  This probe instead calls the
real module builders in a fresh process and brackets each one with this
process's nvidia-smi row, which no sibling lane on the card can move.

It also reports every resolved kernel's local_size_bytes, so a large local
frame hiding in one of these modules cannot escape the ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MIB = 1024.0 * 1024.0

# (import path, builder attribute, builder args, one symbol to resolve)
TARGETS = [
    ("gpuwm.core.rrtmg_lw", "_gpu_module", (), None),
    ("gpuwm.core.rrtmg_mcica", "_mcica_gpu_module", (), None),
    ("gpuwm.core.gf", "_gf_module", (55,), "gf_gfdrv_stage"),
    ("gpuwm.core.noahmp_driver_gpu", "driver_module", (), None),
    ("gpuwm.core.noahmp_energy_gpu", "energy_module", (), None),
    ("gpuwm.core.noahmp_thermal_gpu", "thermal_module", (), None),
    ("gpuwm.core.noahmp_glacier_gpu", "glacier_module", (), None),
    ("gpuwm.core.nest_interp", "_nest_module", (), None),
]


def nvsmi_process_mib(pid):
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


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arwen-checkout", required=True)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    sys.path.insert(0, args.arwen_checkout)

    import cupy as cp

    pid = os.getpid()
    cp.cuda.runtime.free(cp.cuda.runtime.malloc(8))
    cp.cuda.runtime.deviceSynchronize()
    base = nvsmi_process_mib(pid)
    out = {"pid": pid, "baseline_mib": base, "cupy": cp.__version__,
           "targets": []}

    from kernel_reservation import func_attributes  # same directory

    prev = base
    for mod_path, builder, bargs, symbol in TARGETS:
        rec = {"module": mod_path, "builder": builder, "args": list(bargs)}
        t0 = time.perf_counter()
        try:
            mod = __import__(mod_path, fromlist=["*"])
            fn = getattr(mod, builder)
            obj = fn(*bargs)
            if symbol is not None:
                obj.get_function(symbol)
            cp.cuda.runtime.deviceSynchronize()
            now = nvsmi_process_mib(pid)
            rec["nvsmi_after_mib"] = now
            rec["image_mib"] = (None if (now is None or prev is None)
                                else round(now - prev, 1))
            rec["build_seconds"] = round(time.perf_counter() - t0, 2)
            prev = now
            # every kernel this module exposes, with its local frame
            kernels = []
            names = rec_symbols(mod)
            for nm in names:
                try:
                    k = obj.get_function(nm)
                except Exception:
                    continue
                a = func_attributes(cp, k)
                if a:
                    a["name"] = nm
                    kernels.append(a)
            cp.cuda.runtime.deviceSynchronize()
            after_syms = nvsmi_process_mib(pid)
            if after_syms is not None and now is not None and after_syms > now:
                rec["extra_on_symbol_resolve_mib"] = round(after_syms - now, 1)
                prev = after_syms
            rec["kernels"] = sorted(kernels,
                                    key=lambda r: -(r.get("local_size_bytes") or 0))
            rec["max_local_size_bytes"] = max(
                [k.get("local_size_bytes") or 0 for k in kernels] or [0])
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
        out["targets"].append(rec)

    out["final_mib"] = nvsmi_process_mib(pid)
    out["module_image_total_mib"] = (
        None if (out["final_mib"] is None or base is None)
        else round(out["final_mib"] - base, 1))
    txt = json.dumps(out, indent=1)
    if args.json:
        open(args.json, "w").write(txt)
    print("%-34s %10s %10s %14s" % ("module", "image MiB", "build s", "max local B"))
    for r in out["targets"]:
        print("%-34s %10s %10s %14s  %s"
              % (r["module"], r.get("image_mib"), r.get("build_seconds"),
                 r.get("max_local_size_bytes"), r.get("error", "")[:40]))
    print("baseline %.1f MiB -> final %.1f MiB   module images total %.1f MiB"
          % (base or -1, out["final_mib"] or -1, out["module_image_total_mib"] or -1))
    return 0


KERNEL_DECL = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)')


def rec_symbols(mod):
    """Every kernel this module declares.

    A name list attribute is the easy case; most of these modules do not have
    one, and an empty symbol list silently reports "max local frame 0" for a
    module that was never inspected.  So the source itself is parsed: the
    builders all expose the exact translation unit they compile.
    """
    names = []
    for attr in ("_GPU_KERNEL_NAMES", "KERNEL_NAMES", "_KERNEL_NAMES"):
        v = getattr(mod, attr, None)
        if isinstance(v, (list, tuple, set)):
            names.extend(str(x) for x in v)
    if not names:
        for attr in ("_gpu_source", "_source", "_mcica_gpu_source",
                     "_module_source", "_kernel_source"):
            fn = getattr(mod, attr, None)
            if not callable(fn):
                continue
            for call in ((), (55,)):
                try:
                    src = fn(*call)
                except Exception:
                    continue
                if isinstance(src, str):
                    names.extend(KERNEL_DECL.findall(src))
                    break
            if names:
                break
    if not names:
        # last resort: the .cu files that sit beside the module
        try:
            kdir = Path(mod.__file__).parent / "kernels"
            stem = mod.__name__.rsplit(".", 1)[-1]
            for path in sorted(kdir.glob(stem + "*.cu")):
                names.extend(KERNEL_DECL.findall(
                    path.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass
    return sorted(set(names))


if __name__ == "__main__":
    raise SystemExit(main())
