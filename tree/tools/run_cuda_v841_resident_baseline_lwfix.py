#!/usr/bin/env python
"""Resident single-GPU x4.163842 baseline at the FIXED (LW-purity) Arwen pin.

This is the reference arm of brief-2's proof matrix.  It runs the ordinary
resident stack through the frozen runner's own ``_prepare_host_execution`` /
``_construct_device_stack`` / ``_run_steps``, and emits exactly the artifacts
the partitioned arm emits:

  boundary-fingerprints.jsonl   one line per committed boundary (step 0..N)
  snapshot-hashes.json          ``_snapshot_hash_projection`` at capture steps
  snapshots/*.nc                ``write_snapshot_netcdf`` capsules
  run-receipt.json              pins, source manifest, compile manifest, VRAM

Nothing here edits the frozen runner; the Arwen pin comes from the new pin
table in ``v841_partstream_common`` so the runner's own ``ARWEN_COMMIT``
constant stays untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def parse_steps(text: str) -> list[int]:
    return [int(piece) for piece in text.split(",") if piece.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--arwen-checkout", type=Path, required=True)
    parser.add_argument("--arwen-pin", default="lwfix-20260812")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--capture-steps", default="0,15,30")
    parser.add_argument("--no-netcdf", action="store_true")
    args = parser.parse_args(argv)

    import numpy as np  # noqa: F401  (imported for parity of the numeric env)
    import cupy as cp

    import run_cuda_v841_full_physics_x4 as R
    import v841_partstream_common as C
    from mpas_port.cuda_backend import KernelCache

    root = args.evidence_root
    root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    arwen = C.verify_arwen_checkout_pinned(args.arwen_checkout, args.arwen_pin)
    print("[baseline] arwen " + json.dumps(arwen), flush=True)

    paths = C.real_authority_paths(args.asset_root)
    authority = R.verify_authorities(paths)
    manifest = C.port_source_manifest(ROOT)
    print("[baseline] authority sha256 " + authority["sha256"], flush=True)
    print("[baseline] port source sha256 " + manifest["sha256"], flush=True)

    admission = R.gpu_memory_admission(cp)
    host = R._prepare_host_execution(paths, authority)
    print("[baseline] host prepared", flush=True)

    cache = KernelCache(str(args.cache_root))
    stack = R._construct_device_stack(
        host=host, cache=cache, arwen_checkout=args.arwen_checkout
    )
    print("[baseline] device stack constructed", flush=True)

    capture = sorted(
        step
        for step in parse_steps(args.capture_steps)
        if step in R.SNAPSHOT_LABELS and step <= args.steps
    )

    free_floor = {"bytes": int(cp.cuda.runtime.memGetInfo()[0])}
    writer = C.BoundaryFingerprintWriter(root / "boundary-fingerprints.jsonl")

    def record(step: int, current_stack) -> None:
        cp.cuda.get_current_stream().synchronize()
        writer.write(step, R.fingerprint_execution_boundary(current_stack))
        free = int(cp.cuda.runtime.memGetInfo()[0])
        free_floor["bytes"] = min(free_floor["bytes"], free)
        print(f"[baseline] boundary step={step} free_bytes={free}", flush=True)

    record(0, stack)
    snapshots, _previous, receipts = R._run_steps(
        stack=stack,
        start_step=0,
        end_step=args.steps,
        capture_steps=set(capture),
        boundary_observer=record,
    )
    writer.close()

    hashes = {
        str(step): R._snapshot_hash_projection(snapshot)
        for step, snapshot in sorted(snapshots.items())
    }
    q2 = {
        str(step): (
            R.array_sha256(snapshot["arrays"]["q2"])
            if "q2" in snapshot["arrays"]
            else None
        )
        for step, snapshot in sorted(snapshots.items())
    }
    (root / "snapshot-hashes.json").write_text(
        json.dumps({"projection": hashes, "q2": q2}, indent=2, sort_keys=True), "utf-8"
    )

    written = {}
    if not args.no_netcdf:
        static = R._static_output_fields(host)
        out_dir = root / "snapshots"
        out_dir.mkdir(exist_ok=True)
        for step, snapshot in sorted(snapshots.items()):
            label = R.SNAPSHOT_LABELS[step]
            written[label] = R.write_snapshot_netcdf(
                out_dir / R.SNAPSHOT_FILE_NAMES[label], snapshot, static
            )

    pool = cp.get_default_memory_pool()
    receipt = {
        "schema": "mpas-port.v841-partstream-resident-baseline/v1",
        "mode": "resident-single-gpu",
        "arwen": arwen,
        "authority_sha256": authority["sha256"],
        "port_source_manifest": manifest,
        "gpu_memory_admission": admission,
        "steps": args.steps,
        "capture_steps": capture,
        "dt_seconds": R.DT_SECONDS,
        "boundary_fingerprint_steps": writer.steps,
        "snapshot_receipts": {
            str(step): snapshot["receipt"] for step, snapshot in sorted(snapshots.items())
        },
        "step_receipt_count": len(receipts),
        "compile_manifest": cache.compile_manifest(),
        "compile_manifest_sha256": R.canonical_json_sha256(cache.compile_manifest()),
        "device": {
            "name": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "min_free_bytes_observed": free_floor["bytes"],
            "mempool_total_bytes_peak": int(pool.total_bytes()),
        },
        "netcdf": written,
        "wall_seconds": round(time.time() - started, 3),
    }
    (root / "run-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str), "utf-8"
    )
    (root / "step-receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True, default=str), "utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "resident-single-gpu",
                "evidence_root": str(root),
                "steps": args.steps,
                "wall_seconds": receipt["wall_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
