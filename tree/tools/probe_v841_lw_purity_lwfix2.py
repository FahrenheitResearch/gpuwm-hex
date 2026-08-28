#!/usr/bin/env python3
"""Mechanism discrimination for the v841 LW pool-layout impurity.

Extends tools/probe_v841_lw_purity.py (which is preserved as-is) with
perturbation arms that hold the CuPy pool LAYOUT fixed while varying the
CONTENTS of the recycled blocks:

  --perturb none      no perturbation (baseline)
  --perturb device    original device arm (cp.empty keep + freed scratch)
  --perturb devzero   same allocation/free sequence, every block memset 0.0
  --perturb devnan    same allocation/free sequence, every block filled NaN

devzero and devnan present bit-identical pool layouts; only the bytes the
pool will recycle differ.  If phase-1 results differ between devzero and
devnan (or go NaN), an uninitialized device read exists (mechanism B).
If devzero == devnan bitwise while both differ from none, the impurity is
keyed on allocation addresses/layout, not contents (mechanism A).

Optionally logs the device pointer of every pool allocation made during
seam construction + begin_step (--trace-alloc) so the differing arms'
address maps can be diffed.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import run_cuda_v841_full_physics_x4 as runner  # noqa: E402
from diagnose_v841_restart_step16_x4 import (  # noqa: E402
    _authority_paths,
    _fresh_dir,
    _host,
)


def _perturb_device_fill(cp, fill):
    """Original _perturb_device allocation sequence; optional fill value.

    fill=None reproduces the original arm exactly (cp.empty, untouched).
    Otherwise every block (kept and scratch) is filled with `fill` before
    the scratch blocks are freed, so the pool's recycled chunks carry that
    bit pattern while the chunk geometry stays identical.
    """
    keep = []
    for i in range(40):
        block = cp.empty(100_003 + 5077 * i, dtype=cp.float32)
        if fill is not None:
            block.fill(fill)
        keep.append(block)
    scratch = []
    for i in range(20):
        block = cp.empty(1_000_003 + 9973 * i, dtype=cp.float32)
        if fill is not None:
            block.fill(fill)
        scratch.append(block)
    if fill is not None:
        cp.cuda.runtime.deviceSynchronize()
    del scratch
    return keep


def _resolve_arwen_checkout(value: Path | None) -> Path:
    """Refuse by name rather than pin against a path nobody supplied.

    There is no default checkout: a baked-in one would make every run on a
    machine that lacks it fail as a missing directory instead of naming the
    setting that was never made.
    """
    if value is None:
        declared = os.environ.get("ARWEN_CHECKOUT")
        if not declared:
            raise SystemExit(
                "no Arwen checkout: pass --arwen-checkout or set ARWEN_CHECKOUT"
            )
        value = Path(declared)
    return value.expanduser().absolute()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument(
        "--perturb",
        choices=("none", "device", "devzero", "devnan"),
        default="none",
    )
    parser.add_argument("--trace-alloc", action="store_true")
    parser.add_argument("--skip-pins", action="store_true", help=(
        "skip the frozen-checkout git/manifest pins so a MODIFIED "
        "work copy (arwen-lwfix-work) can be probed; the printed "
        "gpuwm module path is the provenance record"))
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=None,
        help="Arwen checkout to pin; defaults to $ARWEN_CHECKOUT",
    )
    parser.add_argument("--assets-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cache_root = _fresh_dir(args.cache_root, "cache root")
    dump_root = _fresh_dir(args.dump_root, "dump root")
    arwen_checkout = _resolve_arwen_checkout(args.arwen_checkout)
    assets_root = (
        None if args.assets_root is None else args.assets_root.expanduser().absolute()
    )

    paths = _authority_paths(assets_root)
    # Source pins verify first, unconditionally: the checkout guard imports
    # the seam manifest from a pinned module, so that module's bytes are
    # proven before its constants are trusted.  --skip-pins waives only the
    # Arwen checkout guard, never the port's own frozen-source pins.
    runner.require_frozen_execution_sources()
    if args.skip_pins:
        import hexcore.cuda_arwen_physics_v841 as _pin_mod
        _pin_mod._verify_checkout_root = lambda root: None
        print("[probe] ARWEN CHECKOUT GUARD SKIPPED (modified work checkout)", flush=True)
    else:
        runner.verify_arwen_checkout_git(arwen_checkout)
    authority_receipt = runner.verify_authorities(paths)
    host = runner._prepare_host_execution(paths, authority_receipt)
    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(arwen_checkout)
    import gpuwm

    print(f"[probe:{args.perturb}] gpuwm module: {gpuwm.__file__}", flush=True)
    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    runner.gpu_memory_admission(cp, minimum=22 * 1024**3)
    cache = KernelCache(capability=capability, cache_dir=cache_root)

    anchors = []
    if args.perturb == "device":
        anchors.append(_perturb_device_fill(cp, None))
    elif args.perturb == "devzero":
        anchors.append(_perturb_device_fill(cp, 0.0))
    elif args.perturb == "devnan":
        anchors.append(_perturb_device_fill(cp, float("nan")))

    trace = []
    if args.trace_alloc:
        base_pool = cp.get_default_memory_pool()

        def traced_alloc(size):
            mem = base_pool.malloc(size)
            trace.append({"size": int(size), "ptr": int(mem.ptr)})
            return mem

        cp.cuda.set_allocator(traced_alloc)

    with args.checkpoint.expanduser().absolute().open("rb") as stream:
        checkpoint = pickle.load(stream)
    stack = runner._construct_device_stack(
        host=host,
        cache=cache,
        arwen_checkout=arwen_checkout,
        state=checkpoint.state,
        saved_diagnostics=checkpoint.saved_diagnostics,
        backend_restart=checkpoint.backend_state,
    )
    restored = runner.fingerprint_execution_boundary(stack)
    runner.require_fingerprint_identity(
        "F030 restored MPAS atmosphere (lwfix probe)",
        checkpoint.atmosphere_fingerprint,
        restored["atmosphere"],
    )
    runner.require_fingerprint_identity(
        "F030 restored Arwen backend (lwfix probe)",
        checkpoint.backend_fingerprint,
        restored["backend"],
    )
    print(f"[probe:{args.perturb}] F030 rehydration bitwise identical", flush=True)

    backend = stack["backend"]
    raw = backend.begin_step(
        atmosphere=stack["driver"].atmosphere,
        scalar_names=runner.SCALAR_NAMES,
        dt=runner.DT_SECONDS,
    )
    receipt = dict(backend.step_receipt())
    arrays = {
        "raw/du": raw.du,
        "raw/dv": raw.dv,
        "raw/dtheta": raw.dtheta,
        **{f"raw/d{name}": raw.dscalars[name] for name in runner.SCALAR_NAMES},
    }
    arrays.update(
        {
            f"post_phase1/{key}": value
            for key, value in backend._seam._restart_manifest().items()
        }
    )
    host_arrays = {name: _host(value) for name, value in sorted(arrays.items())}
    nan_report = {
        name: int(np.count_nonzero(~np.isfinite(value)))
        for name, value in host_arrays.items()
        if np.issubdtype(value.dtype, np.floating)
        and np.count_nonzero(~np.isfinite(value))
    }
    manifest = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": runner.array_sha256(value),
        }
        for name, value in host_arrays.items()
    }
    for name, record in sorted(manifest.items()):
        print(f"[probe:{args.perturb}] {name} sha256={record['sha256']}", flush=True)
    print(
        f"[probe:{args.perturb}] nonfinite-arrays={json.dumps(nan_report, sort_keys=True)}",
        flush=True,
    )
    np.savez(dump_root / "phase1.npz", **host_arrays)
    if args.trace_alloc:
        (dump_root / "alloc-trace.json").write_text(
            json.dumps(trace) + "\n", encoding="utf-8"
        )
        print(f"[probe:{args.perturb}] traced {len(trace)} pool mallocs", flush=True)
    (dump_root / "phase1.manifest.json").write_text(
        json.dumps(
            {
                "perturb": args.perturb,
                "cadence": receipt.get("cadence"),
                "arrays": manifest,
                "nonfinite": nan_report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "perturb": args.perturb,
                "radiation_ran": bool(
                    (receipt.get("cadence") or {}).get("radiation_ran")
                ),
                "dump_root": str(dump_root),
                "anchors": len(anchors),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
