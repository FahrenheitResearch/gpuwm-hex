#!/usr/bin/env python3
"""Join the two-mesh pool ledger and the kernel-reservation ledger into ONE
per-allocation device memory table with measured fixed/per-cell split.

Two meshes on one card make the split arithmetic, not a guess:

    footprint(cells) = FIXED + SLOPE * cells

Every row carries: MiB at each mesh, whether it scales with cells, dtype,
lifetime, and the file:line that allocates it.  Anything the rows do not
cover is printed as a named residue, never absorbed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

M = 1024.0 * 1024.0
BASE = Path(__file__).resolve().parent


def _shortening_prefixes():
    """The tree roots this table abbreviates, read from the environment.

    THE BREAKAGE THIS PREVENTS: these four prefixes were absolute paths
    under one person's home directory, hard-coded, and public since 0.1.0
    (evidence/assembly-rehearsal-20260827/ §6).  They exposed a home
    directory and a private working-tree layout, in a table that could not
    abbreviate anything on any other machine.  Unset variables simply mean
    no abbreviation -- the table still prints, with full paths, which is a
    readable table rather than a refusal in a reporting tool.

        HEX_REPO        the gpuwm-hex checkout's tree/ directory
        ARWEN_CHECKOUT  the gpuwm source checkout at the pinned commit
        HEX_WORK        the campaign scratch directory
    """

    roots = []
    hex_repo = os.environ.get("HEX_REPO")
    if hex_repo:
        base = Path(hex_repo).as_posix().rstrip("/")
        roots.append((f"{base}/src/hexcore/", "port:"))
        roots.append((f"{base}/", "port:"))
    arwen = os.environ.get("ARWEN_CHECKOUT")
    if arwen:
        roots.append((f"{Path(arwen).as_posix().rstrip('/')}/gpuwm/", "arwen:"))
    work = os.environ.get("HEX_WORK")
    if work:
        roots.append((f"{Path(work).as_posix().rstrip('/')}/", ""))
    return tuple(roots)


def short(p):
    text = str(p).replace("\\", "/")
    for prefix, label in _shortening_prefixes():
        text = text.replace(prefix, label)
    return text


def ph(d, label, key, default=None):
    for p in d["phases"]:
        if p["label"] == label:
            return p.get(key, default)
    return default


def proc_dev(d, label):
    """Device bytes charged to THIS process.

    cudaMemGetInfo is device-wide: on a shared card a sibling lane's
    allocation lands inside any window measured that way.  nvidia-smi's
    per-process row cannot be moved by another process, so it is the
    authority here and memGetInfo is only the fallback.
    """
    nv = ph(d, label, "nvsmi_process_mib")
    if nv is not None:
        return float(nv) * M
    return float(ph(d, label, "device_used_bytes", 0))


def load(mesh, suffix="kern"):
    led = json.load(open(BASE / ("ledger-%s-%s.json" % (mesh, suffix))))
    kf = BASE / ("kernels-%s.json" % mesh)
    ker = json.load(open(kf)) if kf.exists() else {"kernels": []}
    return led, ker


def dtype_hint(led):
    hint = {}
    for c in led.get("censuses", []):
        for r in c["rows"]:
            if r.get("site"):
                hint.setdefault(r["site"], (r["dtype"], (r["names"] or [""])[0]))
    return hint


def lifetime_of(rec_a, rec_b):
    """process / run / step / round / transient, from where it was allocated."""
    rec = rec_b or rec_a or {}
    phases = set((rec.get("phases") or {}).keys())
    still = max((rec_a or {}).get("still_live_bytes", 0),
                (rec_b or {}).get("still_live_bytes", 0))
    peak = max((rec_a or {}).get("peak_live_bytes", 0),
               (rec_b or {}).get("peak_live_bytes", 0))
    if still > 0.5 * max(peak, 1):
        return "run"
    if phases == {"device_stack_construct"}:
        return "run"
    if phases == {"capture_snapshot"}:
        return "round"
    if phases == {"resident"}:
        return "run"
    n = rec.get("alloc_count", 0)
    return "step" if n >= 6 else "transient"


def main():
    l1, k1 = load("x1")
    l4, k4 = load("x4")
    c1, c4 = l1["n_cells"], l4["n_cells"]
    dc = c4 - c1
    h1, h4 = dtype_hint(l1), dtype_hint(l4)

    dev1 = proc_dev(l1, "final")
    dev4 = proc_dev(l4, "final")
    pt1 = ph(l1, "final", "pool_total_bytes")
    pt4 = ph(l4, "final", "pool_total_bytes")
    np1, np4 = dev1 - pt1, dev4 - pt4
    peak1 = l1["hook_totals"]["peak_live_bytes"]
    peak4 = l4["hook_totals"]["peak_live_bytes"]

    out = {"meshes": {"x1": {"cells": c1}, "x4": {"cells": c4}}, "rows": []}
    lines = []
    A = lines.append

    A("=" * 132)
    A("gpuwm-hex PER-ALLOCATION DEVICE MEMORY LEDGER  --  RTX 5090 32,607 MiB, %d SM, float32, nVertLevels=%d"
      % (l1.get("device", {}).get("sms", 0), l1["n_levels"]))
    A("x1.40962: %d cells / %d edges     x4.163842: %d cells / %d edges     both meshes, one card, 6 steps"
      % (c1, l1["n_edges"], c4, l4["n_edges"]))
    A("=" * 132)
    A("%-46s %10s %10s %8s %6s %-9s %-10s  %s"
      % ("class", "x1 MiB", "x4 MiB", "B/cell", "scale", "dtype", "lifetime", "allocated at"))
    A("-" * 132)

    def row(name, a, b, dtype, life, site, scale=None, note=None):
        slope = (b - a) / dc
        scaling = ("yes" if abs(slope) > 100 else "no") if scale is None else scale
        A("%-46s %10.1f %10.1f %8.0f %6s %-9s %-10s  %s"
          % (name[:46], a / M, b / M, slope, scaling, dtype, life, short(site)[:44]))
        out["rows"].append({
            "class": name, "x1_mib": round(a / M, 1), "x4_mib": round(b / M, 1),
            "bytes_per_cell": round(slope), "scales_with_cells": scaling,
            "dtype": dtype, "lifetime": life, "site": short(site),
            "note": note,
        })
        return slope

    # ---------------- FIXED: non-pool -------------------------------------
    A("FIXED TERM -- device memory the CuPy pool never sees (cudaMemGetInfo minus pool.total_bytes)")
    ctx1 = proc_dev(l1, "construct:begin")
    ctx4 = proc_dev(l4, "construct:begin")
    acc1 = acc4 = 0
    row("CUDA context + CuPy/NVRTC baseline", ctx1, ctx4, "-", "process",
        "cupy first-touch (probe: cuda_context:after_require_cuda)", scale="no")
    acc1 += ctx1; acc4 += ctx4

    # kernel local-memory backing stores, measured at first launch
    kk1 = {r["kernel"]: r for r in k1.get("kernels", [])}
    kk4 = {r["kernel"]: r for r in k4.get("kernels", [])}
    for name in sorted(set(kk1) | set(kk4),
                       key=lambda n: -max(kk1.get(n, {}).get("first_launch_non_pool_bytes", 0),
                                          kk4.get(n, {}).get("first_launch_non_pool_bytes", 0))):
        a = kk1.get(name, {}).get("first_launch_non_pool_bytes", 0)
        b = kk4.get(name, {}).get("first_launch_non_pool_bytes", 0)
        if max(a, b) < 4 * M:
            continue
        rec = kk4.get(name) or kk1[name]
        local = rec.get("local_size_bytes") or 0
        acc1 += a; acc4 += b
        row("kernel local-memory frame: %s (%d B/thread)" % (name, local),
            a, b, "local", "process", rec.get("first_site") or "?", scale="no",
            note="CUDA reserves local_size_bytes x resident threads at first "
                 "launch; context-wide high-water, never returned")

    mods1 = (proc_dev(l1, "construct:end")
             - ph(l1, "construct:end", "pool_total_bytes") - ctx1)
    mods4 = (proc_dev(l4, "construct:end")
             - ph(l4, "construct:end", "pool_total_bytes") - ctx4)
    acc1 += mods1; acc4 += mods4
    row("non-pool taken during device-stack construct", mods1, mods4,
        "cubin+", "process", "port:cuda_backend/runtime.py:270",
        note="module images for the port's compiled translation units plus "
             "any non-pool device bytes the construct path takes")

    # module images loaded after construct, sized per process by the tracer
    ml1 = [r for r in k1.get("module_loads", []) if (r.get("nvsmi_process_delta_mib") or 0) > 0]
    ml4 = [r for r in k4.get("module_loads", []) if (r.get("nvsmi_process_delta_mib") or 0) > 0]
    if ml1 or ml4:
        A("")
        A("  module IMAGE loads measured per process (nvidia-smi bracket around the load):")
        for r in sorted(ml1, key=lambda r: -(r.get("nvsmi_process_delta_mib") or 0))[:12]:
            A("    %-58s %7.1f MiB  %s" % (r["kind"][:58],
                                           r.get("nvsmi_process_delta_mib") or 0.0,
                                           short(r.get("site") or "")[:52]))
        A("    x1 module-image total %.1f MiB   x4 module-image total %.1f MiB"
          % (sum((r.get("nvsmi_process_delta_mib") or 0) for r in ml1),
             sum((r.get("nvsmi_process_delta_mib") or 0) for r in ml4)))
        out["module_images"] = {
            "x1_mib": round(sum((r.get("nvsmi_process_delta_mib") or 0) for r in ml1), 1),
            "x4_mib": round(sum((r.get("nvsmi_process_delta_mib") or 0) for r in ml4), 1),
            "loads": ml1,
        }
        A("")

    A("-" * 132)
    res1, res4 = np1 - acc1, np4 - acc4
    A("%-46s %10.1f %10.1f %8.0f %6s"
      % ("UNATTRIBUTED fixed residue", res1 / M, res4 / M, (res4 - res1) / dc, "-"))
    A("%-46s %10.1f %10.1f %8.0f %6s"
      % ("FIXED TOTAL (non-pool, measured)", np1 / M, np4 / M, (np4 - np1) / dc, "no"))
    out["fixed"] = {"x1_mib": round(np1 / M, 1), "x4_mib": round(np4 / M, 1),
                    "attributed_x1_mib": round(acc1 / M, 1),
                    "attributed_x4_mib": round(acc4 / M, 1),
                    "residue_x1_mib": round(res1 / M, 1),
                    "residue_x4_mib": round(res4 / M, 1)}

    # ---------------- SLOPE: the pool -------------------------------------
    A("")
    A("=" * 132)
    A("PER-CELL TERM -- the CuPy pool high-water.  Rows are LIVE AT THE GLOBAL PEAK INSTANT, so they sum to one")
    A("simultaneous footprint rather than to a sum of peaks that never coexist.")
    A("=" * 132)
    A("%-46s %10s %10s %8s %6s %-9s %-10s  %s"
      % ("class", "x1 MiB", "x4 MiB", "B/cell", "scale", "dtype", "lifetime", "allocated at"))
    A("-" * 132)
    s1 = {s["site"]: s for s in l1["sites"]}
    s4 = {s["site"]: s for s in l4["sites"]}
    merged = []
    for site in set(s1) | set(s4):
        a = s1.get(site, {}).get("live_at_global_peak_bytes", 0)
        b = s4.get(site, {}).get("live_at_global_peak_bytes", 0)
        merged.append((site, a, b))
    merged.sort(key=lambda r: -max(r[1], r[2]))
    tot_a = tot_b = 0.0
    shown_a = shown_b = 0.0
    for site, a, b in merged:
        tot_a += a; tot_b += b
        if max(a, b) < 40 * M:
            continue
        shown_a += a; shown_b += b
        dt, nm = h4.get(site) or h1.get(site) or ("float32", "")
        life = lifetime_of(s1.get(site), s4.get(site))
        label = (nm.replace("stack.", "").replace("driver.atmosphere.", "atm.")
                 if nm else site.rsplit("/", 1)[-1])
        row(label or "?", a, b, dt, life, site)
    A("-" * 132)
    A("%-46s %10.1f %10.1f %8.0f" % ("rows above 40 MiB, subtotal",
                                     shown_a / M, shown_b / M, (shown_b - shown_a) / dc))
    A("%-46s %10.1f %10.1f %8.0f" % ("all %d pool sites at the peak instant" % len(merged),
                                     tot_a / M, tot_b / M, (tot_b - tot_a) / dc))
    A("%-46s %10.1f %10.1f %8.0f" % ("hook peak_live (cross-check)",
                                     peak1 / M, peak4 / M, (peak4 - peak1) / dc))
    over1, over4 = pt1 - peak1, pt4 - peak4
    A("%-46s %10.1f %10.1f %8.0f  %s"
      % ("pool blocks held above the live peak", over1 / M, over4 / M,
         (over4 - over1) / dc,
         "best-fit rounding + free blocks the pool keeps"))
    A("%-46s %10.1f %10.1f %8.0f" % ("POOL TOTAL (high-water, measured)",
                                     pt1 / M, pt4 / M, (pt4 - pt1) / dc))
    out["pool"] = {"x1_mib": round(pt1 / M, 1), "x4_mib": round(pt4 / M, 1),
                   "peak_live_x1_mib": round(peak1 / M, 1),
                   "peak_live_x4_mib": round(peak4 / M, 1),
                   "sites_at_peak_x1_mib": round(tot_a / M, 1),
                   "sites_at_peak_x4_mib": round(tot_b / M, 1)}

    # ---------------- whole-footprint reconciliation ----------------------
    A("")
    A("=" * 132)
    A("WHOLE FOOTPRINT")
    A("=" * 132)
    slope = (dev4 - dev1) / dc
    fixed = dev1 - c1 * slope
    A("measured device_used   x1 = %10.1f MiB   x4 = %10.1f MiB" % (dev1 / M, dev4 / M))
    A("nvidia-smi this process x1 = %10.1f MiB   x4 = %10.1f MiB"
      % (ph(l1, "final", "nvsmi_process_mib") or -1, ph(l4, "final", "nvsmi_process_mib") or -1))
    A("two-point fit          FIXED = %.0f MiB   SLOPE = %.0f B/cell" % (fixed / M, slope))
    A("state (prognostic)     = %.0f B/cell  (%.1f%% of the slope)"
      % (4628, 100.0 * 4628 / slope))
    A("attributed fixed       = %.1f MiB of %.1f MiB   (residue %.1f MiB)"
      % (acc1 / M, np1 / M, res1 / M))
    out["fit"] = {"fixed_mib": round(fixed / M, 1), "slope_bytes_per_cell": round(slope),
                  "device_x1_mib": round(dev1 / M, 1), "device_x4_mib": round(dev4 / M, 1)}

    txt = "\n".join(lines)
    print(txt)
    (BASE / "LEDGER-TABLE.txt").write_text(txt + "\n")
    (BASE / "LEDGER-TABLE.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
