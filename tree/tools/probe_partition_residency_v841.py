#!/usr/bin/env python
"""Measure per-partition persistent device residency for the streamed executor.

Brief-2 requires a measurement before choosing the residency branch:
  * per-partition driver static device bytes
  * per-partition Arwen backend restart-state bytes
Both are reported for the smallest and the largest partition, and extrapolated
across all P partitions so the CAP decision is made on numbers, not a guess.

Run from the repo root.  Writes JSON to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arwen-checkout", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-parts", type=int, default=16)
    parser.add_argument("--halo-rings", type=int, default=60)
    parser.add_argument("--parts", type=int, nargs="*", default=None)
    args = parser.parse_args()

    import cupy as cp

    import run_cuda_v841_full_physics_x4 as R
    import v841_partstream_common as C
    from hexcore.cuda_backend import KernelCache
    from hexcore.partition_local_mesh_v841 import slice_prepared_host

    report: dict[str, object] = {"schema": "mpas-port.partstream-residency-probe/v1"}

    paths = C.real_authority_paths(args.asset_root)
    authority = R.verify_authorities(paths)
    report["authority_sha256"] = authority["sha256"]

    host = R._prepare_host_execution(paths, authority)
    print("[probe] host prepared", flush=True)

    layouts = C.load_layouts(
        args.assets, parts=args.n_parts, halo_rings=args.halo_rings,
        mesh=host["prepared"].mesh,
    )
    report["n_partitions"] = len(layouts)
    report["layouts"] = [
        {
            "rank": i,
            "owned_cells": int(l.n_owned_cells),
            "local_cells": int(l.n_local_cells),
            "owned_edges": int(l.n_owned_edges),
            "local_edges": int(l.n_local_edges),
            "owned_vertices": int(l.n_owned_vertices),
            "local_vertices": int(l.n_local_vertices),
        }
        for i, l in enumerate(layouts)
    ]

    order = sorted(range(len(layouts)), key=lambda i: layouts[i].n_local_cells)
    selected = args.parts if args.parts else [order[0], order[-1]]

    pool = cp.get_default_memory_pool()
    measurements = []
    for rank in selected:
        layout = layouts[rank]
        pool.free_all_blocks()
        before = int(pool.used_bytes())
        sliced_prepared = slice_prepared_host(host["prepared"], layout)
        sliced_constructor = slice_prepared_host(host["constructor_values"], layout)
        sliced_gwdo = slice_prepared_host(host["gwdo_host"], layout)
        sliced_f000 = slice_prepared_host(host["f000_surface_diagnostics"], layout)
        local_host = {
            "config": host["config"],
            "prepared": sliced_prepared,
            "constructor_values": sliced_constructor,
            "gwdo_host": sliced_gwdo,
            "f000_surface_diagnostics": sliced_f000,
        }
        cache = KernelCache(str(args.cache_root))
        stack = R._construct_device_stack(
            host=local_host,
            cache=cache,
            arwen_checkout=args.arwen_checkout,
        )
        cp.cuda.get_current_stream().synchronize()
        after_construct = int(pool.used_bytes())
        backend_state = stack["backend"].restart_state()
        backend_fp = R.fingerprint_nested_arrays(backend_state)
        backend_bytes = 0
        for entry in backend_fp["arrays"].values():
            shape = entry["shape"]
            n = 1
            for d in shape:
                n *= int(d)
            backend_bytes += n * np.dtype(entry["dtype"]).itemsize
        record = {
            "rank": rank,
            "local_cells": int(layout.n_local_cells),
            "owned_cells": int(layout.n_owned_cells),
            "device_bytes_after_construct": after_construct - before,
            "backend_restart_host_bytes": int(backend_bytes),
            "backend_restart_array_count": len(backend_fp["arrays"]),
        }
        print("[probe] " + json.dumps(record), flush=True)
        measurements.append(record)
        del stack, cache, sliced_prepared, sliced_constructor, sliced_gwdo
        del sliced_f000, local_host, backend_state, backend_fp
        pool.free_all_blocks()

    report["measurements"] = measurements
    total_local = sum(l.n_local_cells for l in layouts)
    biggest = max(measurements, key=lambda m: m["local_cells"])
    per_cell = biggest["device_bytes_after_construct"] / biggest["local_cells"]
    report["projected_all_partition_resident_bytes"] = int(per_cell * total_local)
    report["projected_all_partition_resident_gib"] = round(
        per_cell * total_local / 1024**3, 3
    )
    report["sum_local_cells"] = int(total_local)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), "utf-8")
    print(json.dumps({"status": "ok", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
