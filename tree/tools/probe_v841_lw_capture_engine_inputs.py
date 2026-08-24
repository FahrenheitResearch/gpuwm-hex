#!/usr/bin/env python3
"""Capture the FIRST gpu_rrtmg_lw_batched call's array arguments to npz.

Same skeleton as tools/probe_v841_lw_localize.py, --perturb none only.
Wraps the pinned checkout's gpu_rrtmg_lw_batched; on call #1 it saves
every positional argument (arrays as float32/int arrays, scalars as
0-d arrays; the coefficient dict C is rebuilt by the consumer via
gpuwm.core.rrtmg_legacy._lw_coeffs()) plus column_chunk, then exits the
process immediately -- the model step is NOT completed.  Output:
<out>/lwengine_call1_inputs.npz + a done marker printed on stdout.
"""

from __future__ import annotations

import argparse
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
)

ARG_NAMES = (
    "ncol", "nlay", "icld", "play", "plev", "tlay", "tlev", "tsfc",
    "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "n2ovmr", "o2vmr",
    "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr", "emis", "inflglw",
    "iceflglw", "liqflglw", "cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl",
    "cswpmcl", "reicmcl", "relqmcl", "resnmcl", "tauaer",
)


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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=None,
        help="Arwen checkout to pin; defaults to $ARWEN_CHECKOUT",
    )
    parser.add_argument("--assets-root", type=Path, default=None)
    args = parser.parse_args(argv)

    cache_root = _fresh_dir(args.cache_root, "cache root")
    _fresh_dir(args.dump_root, "dump root")
    arwen_checkout = _resolve_arwen_checkout(args.arwen_checkout)
    assets_root = (
        None if args.assets_root is None
        else args.assets_root.expanduser().absolute()
    )
    out_path = args.out.expanduser().absolute()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    import gpuwm

    print(f"[capture] gpuwm module: {gpuwm.__file__}", flush=True)
    from mpas_port.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    runner.gpu_memory_admission(cp, minimum=22 * 1024**3)
    cache = KernelCache(capability=capability, cache_dir=cache_root)

    from gpuwm.core import rrtmg_lw as _lw_mod

    real = _lw_mod.gpu_rrtmg_lw_batched

    def shim(*a, **kw):
        payload = {}
        for name, value in zip(ARG_NAMES, a):
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                try:
                    value = cp.asnumpy(value)
                except Exception:
                    value = np.asarray(value)
                payload[name] = np.ascontiguousarray(value)
            else:
                payload[name] = np.asarray(value)
        cc = kw.get("column_chunk")
        payload["column_chunk"] = np.asarray(-1 if cc is None else int(cc))
        np.savez_compressed(out_path, **payload)
        print(f"[capture] saved {out_path} ncol={payload['ncol']}"
              f" nlay={payload['nlay']}", flush=True)
        sys.stdout.flush()
        os._exit(0)
        return real(*a, **kw)

    _lw_mod.gpu_rrtmg_lw_batched = shim

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
        "F030 restored MPAS atmosphere (capture probe)",
        checkpoint.atmosphere_fingerprint,
        restored["atmosphere"],
    )
    runner.require_fingerprint_identity(
        "F030 restored Arwen backend (capture probe)",
        checkpoint.backend_fingerprint,
        restored["backend"],
    )
    print("[capture] F030 rehydration bitwise identical", flush=True)

    backend = stack["backend"]
    backend.begin_step(
        atmosphere=stack["driver"].atmosphere,
        scalar_names=runner.SCALAR_NAMES,
        dt=runner.DT_SECONDS,
    )
    print("[capture] ERROR: begin_step returned without LW call",
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
