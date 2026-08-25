#!/usr/bin/env python3
import json, sys
MIB = 1024.0 * 1024.0
d = json.load(open(sys.argv[1]))
print("=" * 100)
print("mesh=%s cells=%d edges=%d levels=%d rc=%s wall=%.1fs"
      % (d["mesh"], d["n_cells"], d["n_edges"], d["n_levels"], d["rc"], d["wall_seconds"]))
dev = d.get("device", {})
print("device: %s SMs=%s resident_threads=%s" % (dev.get("name"), dev.get("sms"), dev.get("resident_threads")))
print()
print("--- PHASES (MiB) " + "-" * 60)
print("%-34s %10s %10s %10s %10s %10s" % ("phase", "hook_live", "pool_used", "pool_total", "device", "non_pool"))
for p in d["phases"]:
    dv = p.get("device_used_bytes", 0) / MIB
    pt = p.get("pool_total_bytes", 0) / MIB
    print("%-34s %10.1f %10.1f %10.1f %10.1f %10.1f"
          % (p["label"], p.get("live_bytes", 0) / MIB, p.get("pool_used_bytes", 0) / MIB,
             pt, dv, dv - pt))
dr = d.get("drain")
if dr:
    print("%-34s %10s %10.1f %10.1f %10.1f %10.1f"
          % ("DRAIN (pool given back)", "-", dr["pool_used_bytes"] / MIB,
             dr["pool_total_bytes"] / MIB, dr["device_used_bytes"] / MIB,
             (dr["device_used_bytes"] - dr["pool_total_bytes"]) / MIB))
print()
lm = d.get("local_memory_model")
if lm:
    print("--- KERNELS " + "-" * 66)
    print("kernels=%d modules=%s compile_seconds=%.1f"
          % (lm["kernel_count"], d.get("kernel_modules"), d.get("compile_seconds_total", 0)))
    print("max local_size_bytes=%d in %s -> upper-bound reservation %.1f MiB"
          % (lm["max_local_size_bytes"], lm["max_local_kernel"],
             lm["predicted_reservation_bytes"] / MIB))
    print("%-52s %9s %9s %7s %7s" % ("kernel", "local_B", "shared_B", "regs", "maxthr"))
    for k in d["kernels"][:18]:
        print("%-52s %9s %9s %7s %7s"
              % (k["name"][:52], k["local_size_bytes"], k["shared_size_bytes"],
                 k["num_regs"], k["max_threads_per_block"]))
    nz = [k for k in d["kernels"] if (k.get("local_size_bytes") or 0) > 0]
    print("kernels with local_size_bytes > 0: %d of %d" % (len(nz), len(d["kernels"])))
print()
print("--- POOL SITES at global peak (MiB) " + "-" * 42)
print("%10s %10s %10s %8s  %s" % ("at_peak", "site_peak", "total_all", "count", "site"))
tot = 0.0
for s in d.get("sites", [])[:28]:
    tot += s["live_at_global_peak_bytes"] / MIB
    print("%10.1f %10.1f %10.1f %8d  %s"
          % (s["live_at_global_peak_bytes"] / MIB, s["peak_live_bytes"] / MIB,
             s["total_alloc_bytes"] / MIB, s["alloc_count"], s["site"]))
print("top-28 at peak sum = %.1f MiB ; hook peak_live = %.1f MiB"
      % (tot, d["hook_totals"]["peak_live_bytes"] / MIB))
print()
print("--- PHASE ALLOC TOTALS (MiB churned) " + "-" * 41)
for k, v in sorted(d.get("phase_alloc_bytes", {}).items(), key=lambda kv: -kv[1]):
    print("%-28s %12.1f  n=%d  peak_live=%.1f"
          % (k, v / MIB, d["phase_alloc_count"][k], d["phase_peak_live_bytes"].get(k, 0) / MIB))
print()
for cen in d.get("censuses", []):
    print("--- CENSUS %s : %d arrays in %d blocks, resident %.1f MiB (pool_used %.1f) ---"
          % (cen["label"], cen["arrays"], cen["blocks"], cen["resident_block_bytes"] / MIB,
             (cen.get("pool_used_bytes") or 0) / MIB))
    print("%10s %8s %-9s %-22s %-42s %s" % ("MiB", "views", "dtype", "shape", "name", "site"))
    for r in cen["rows"][:26]:
        nm = r["names"][0] if r["names"] else ""
        print("%10.2f %8d %-9s %-22s %-42s %s"
              % (r["block_bytes"] / MIB, r["views"], r["dtype"], r["shape"][:22],
                 nm[:42], (r["site"] or "")[-58:]))
    print()
