#!/usr/bin/env python3
"""Standalone LW-engine purity harness: replay captured inputs under a
poisoned CuPy pool and ledger every kernel launch.

Loads the npz written by probe_v841_lw_capture_engine_inputs.py, pins
the frozen Arwen checkout, rebuilds the coefficient dict via
gpuwm.core.rrtmg_legacy._lw_coeffs(), poisons the memory pool with the
proven allocation sequence (fill 0.0 or NaN), then calls
gpu_rrtmg_lw_batched on the first --ncol columns with
_lw._gpu_kernel wrapped: after each launch (device-synchronized) it
prints a sha256 for every cupy array argument.  Diffing the zero vs nan
ledgers pins the FIRST kernel whose array set moves -> the kernel that
reads foreign memory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

ARG_NAMES = (
    "ncol", "nlay", "icld", "play", "plev", "tlay", "tlev", "tsfc",
    "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "n2ovmr", "o2vmr",
    "cfc11vmr", "cfc12vmr", "cfc22vmr", "ccl4vmr", "emis", "inflglw",
    "iceflglw", "liqflglw", "cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl",
    "cswpmcl", "reicmcl", "relqmcl", "resnmcl", "tauaer",
)
# leading-axis-is-column arrays; mcica arrays are (NGPTLW, ncol, nlay)
COL_AXIS0 = {
    "play", "plev", "tlay", "tlev", "tsfc", "h2ovmr", "o3vmr",
    "co2vmr", "ch4vmr", "n2ovmr", "o2vmr", "cfc11vmr", "cfc12vmr",
    "cfc22vmr", "ccl4vmr", "emis", "reicmcl", "relqmcl", "resnmcl",
    "tauaer",
}
COL_AXIS1 = {"cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl"}


def _perturb_device_fill(cp, fill):
    """Proven poison sequence from probe_v841_lw_purity_lwfix2.py."""
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
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--fill", choices=("zero", "nan", "none"),
                        required=True)
    parser.add_argument("--ncol", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True,
                        help="npz path for the engine outputs")
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=None,
        help="Arwen checkout to pin; defaults to $ARWEN_CHECKOUT",
    )
    parser.add_argument("--skip-pins", action="store_true")
    parser.add_argument("--stage-dump", type=Path, default=None,
                        help="npz path: dump taug/fracs/laytrop/isv/fs "
                             "after the first rlw_taugb1 launch")
    args = parser.parse_args(argv)

    checkout = _resolve_arwen_checkout(args.arwen_checkout)
    if args.skip_pins:
        sys.path.insert(0, str(checkout))
    else:
        import run_cuda_v841_full_physics_x4 as runner
        # Source pins verify first: the checkout guard imports the seam
        # manifest from a pinned module, so that module's bytes are proven
        # before its constants are trusted.
        runner.require_frozen_execution_sources()
        runner.verify_arwen_checkout_git(checkout)
        from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841
        pin_arwen_physics_v841(checkout)
    import gpuwm

    print(f"[repro:{args.fill}] gpuwm module: {gpuwm.__file__}", flush=True)

    import cupy as cp
    from gpuwm.core import rrtmg_legacy as _legacy
    from gpuwm.core import rrtmg_lw as _lw

    data = np.load(args.inputs)
    n = int(args.ncol)
    total = int(data["ncol"])
    n = min(n, total)
    call = {}
    for name in ARG_NAMES:
        v = data[name]
        if name in COL_AXIS0:
            v = np.ascontiguousarray(v[:n])
        elif name in COL_AXIS1:
            v = np.ascontiguousarray(v[:, :n])
        elif v.ndim == 0:
            v = v.item()
        call[name] = v
    call["ncol"] = n
    print(f"[repro:{args.fill}] ncol={n}/{total} nlay={call['nlay']}",
          flush=True)

    # Build the coefficient dict BEFORE poisoning (constant host data).
    C = _legacy._lw_coeffs()

    # Compile + preflight BEFORE poisoning so arm layouts match at launch.
    _lw.gpu_preflight()

    # ---- ledger every kernel launch ---------------------------------
    real_gk = _lw._gpu_kernel
    counter = {}

    def _h(a):
        return hashlib.sha256(
            cp.asnumpy(a).tobytes()).hexdigest()[:16]

    def ledgered(name):
        fn = real_gk(name)

        def launch(grid, block, kargs):
            r = fn(grid, block, kargs)
            cp.cuda.runtime.deviceSynchronize()
            k = counter[name] = counter.get(name, 0) + 1
            for i, a in enumerate(kargs):
                if isinstance(a, cp.ndarray):
                    if a.dtype == cp.uint64:
                        print(f"[k:{name}#{k}] a{i} ptrtable", flush=True)
                    else:
                        print(f"[k:{name}#{k}] a{i} {a.dtype.str}"
                              f"{tuple(a.shape)} {_h(a)}", flush=True)
            if (args.stage_dump is not None and name == "rlw_taugb1"
                    and k == 1):
                np.savez_compressed(
                    args.stage_dump,
                    taug=cp.asnumpy(kargs[9]),
                    fracs=cp.asnumpy(kargs[10]),
                    laytrop=cp.asnumpy(kargs[2]),
                    fs=cp.asnumpy(kargs[3]),
                    isv=cp.asnumpy(kargs[4]))
                print(f"[k:{name}#{k}] stage dump saved", flush=True)
            return r

        return launch

    _lw._gpu_kernel = ledgered

    fill = {"zero": 0.0, "nan": float("nan"), "none": None}[args.fill]
    anchors = _perturb_device_fill(cp, fill)

    res = _lw.gpu_rrtmg_lw_batched(
        call["ncol"], call["nlay"], call["icld"], call["play"],
        call["plev"], call["tlay"], call["tlev"], call["tsfc"],
        call["h2ovmr"], call["o3vmr"], call["co2vmr"], call["ch4vmr"],
        call["n2ovmr"], call["o2vmr"], call["cfc11vmr"],
        call["cfc12vmr"], call["cfc22vmr"], call["ccl4vmr"],
        call["emis"], call["inflglw"], call["iceflglw"],
        call["liqflglw"], call["cldfmcl"], call["taucmcl"],
        call["ciwpmcl"], call["clwpmcl"], call["cswpmcl"],
        call["reicmcl"], call["relqmcl"], call["resnmcl"],
        call["tauaer"], C, column_chunk=n)

    for k in sorted(res):
        v = res[k]
        digest = hashlib.sha256(np.ascontiguousarray(v).tobytes())
        nonfin = int(np.size(v) - np.isfinite(v).sum())
        print(f"[out] {k} {digest.hexdigest()[:16]} nonfinite={nonfin}",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **res)
    print(f"[repro:{args.fill}] anchors={len(anchors)} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
