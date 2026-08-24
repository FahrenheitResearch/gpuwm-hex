#!/usr/bin/env python3
"""LW purity probe: is the step-16 phase-1 (radiation recompute) a pure
function of the restored F030 state, or does it depend on the process's
heap / device-pool history?

Each invocation is one fresh process:
  1. optional heap/pool perturbation (--perturb none|host|device|both)
  2. restore the given F030 checkpoint through the exact restart path
  3. verify bitwise F030 rehydration
  4. run backend.begin_step once (radiation is due at step 16)
  5. print sha256 of raw du/dv/dtheta/dq* and of every post-phase1
     seam-manifest array (held rthratenlw among them), then write them
     to the dump root as npz.

Compare the printed hashes across processes with different --perturb
values (and across two identical --perturb none runs for repeatability).
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


def _perturb_host() -> list:
    # Shift the host heap: many odd-sized allocations kept alive plus a
    # burst of freed ones so subsequent numpy buffers land elsewhere.
    keep = []
    for i in range(20_000):
        keep.append(bytearray(37 + (i * 13) % 251))
    for i in range(300):
        keep.append(np.empty(1021 + 7 * i, dtype=np.float32))
    scratch = [np.empty(4093 + 11 * i, dtype=np.float64) for i in range(200)]
    del scratch
    return keep


def _perturb_device(cp) -> list:
    keep = []
    for i in range(40):
        keep.append(cp.empty(100_003 + 5077 * i, dtype=cp.float32))
    scratch = [cp.empty(1_000_003 + 9973 * i, dtype=cp.float32) for i in range(20)]
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
        "--perturb", choices=("none", "host", "device", "both"), default="none"
    )
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

    anchors = []
    if args.perturb in ("host", "both"):
        anchors.append(_perturb_host())

    paths = _authority_paths(assets_root)
    # Source pins verify first: the checkout guard imports the seam manifest
    # from a pinned module, so that module's bytes are proven before its
    # constants are trusted.
    runner.require_frozen_execution_sources()
    runner.verify_arwen_checkout_git(arwen_checkout)
    authority_receipt = runner.verify_authorities(paths)
    host = runner._prepare_host_execution(paths, authority_receipt)
    from mpas_port.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(arwen_checkout)
    from mpas_port.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    runner.gpu_memory_admission(cp, minimum=22 * 1024**3)
    cache = KernelCache(capability=capability, cache_dir=cache_root)

    if args.perturb in ("device", "both"):
        anchors.append(_perturb_device(cp))

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
        "F030 restored MPAS atmosphere (purity probe)",
        checkpoint.atmosphere_fingerprint,
        restored["atmosphere"],
    )
    runner.require_fingerprint_identity(
        "F030 restored Arwen backend (purity probe)",
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
    np.savez(dump_root / "phase1.npz", **host_arrays)
    (dump_root / "phase1.manifest.json").write_text(
        json.dumps(
            {
                "perturb": args.perturb,
                "cadence": receipt.get("cadence"),
                "arrays": manifest,
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
