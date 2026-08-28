#!/usr/bin/env python3
"""Localize the v841 LW pool-layout impurity to a chain stage.

Same skeleton as tools/probe_v841_lw_purity_lwfix.py, but before the
begin_step call it wraps the pinned checkout's LW-chain entry points --
cal_cldfra1, lwrad_prep_batch, gpu_generate_lw_subcolumns,
gpu_rrtmg_lw_batched, lwrad_outputs_batch -- and prints a sha256 for
every array argument and every array result, in call order.  Diffing the
printed ledger between --perturb none and --perturb device shows the
FIRST stage whose outputs move while its inputs hold, i.e. the stage
that owns the impurity.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _perturb_device(cp):
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


def _sha(value):
    try:
        import cupy as cp

        if isinstance(value, cp.ndarray):
            host = cp.asnumpy(cp.ascontiguousarray(value))
            return hashlib.sha256(host.tobytes()).hexdigest()[:16]
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return hashlib.sha256(
            np.ascontiguousarray(value).tobytes()
        ).hexdigest()[:16]
    return None


def _ledger_args(tag, args, kwargs):
    lines = []
    for i, a in enumerate(args):
        digest = _sha(a)
        if digest is not None:
            lines.append(f"{tag} arg[{i}] {digest}")
    for k in sorted(kwargs):
        digest = _sha(kwargs[k])
        if digest is not None:
            lines.append(f"{tag} kw[{k}] {digest}")
    return lines


def _ledger_result(tag, res):
    lines = []
    if isinstance(res, dict):
        for k in sorted(res):
            digest = _sha(res[k])
            if digest is not None:
                lines.append(f"{tag} out[{k}] {digest}")
    else:
        digest = _sha(res)
        if digest is not None:
            lines.append(f"{tag} out {digest}")
    return lines


def _wrap(module, name, tag, counter):
    real = getattr(module, name)

    def shim(*args, **kwargs):
        n = counter[tag] = counter.get(tag, 0) + 1
        label = f"[stage:{tag}#{n}]"
        for line in _ledger_args(label, args, kwargs):
            print(line, flush=True)
        res = real(*args, **kwargs)
        for line in _ledger_result(label, res):
            print(line, flush=True)
        return res

    setattr(module, name, shim)
    return real


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--dump-root", type=Path, required=True)
    parser.add_argument("--perturb", choices=("none", "device"), default="none")
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
    # Source pins verify first: the checkout guard imports the seam manifest
    # from a pinned module, so that module's bytes are proven before its
    # constants are trusted.
    runner.require_frozen_execution_sources()
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
        anchors.append(_perturb_device(cp))

    # ---- instrument the pinned checkout's LW chain -------------------
    from gpuwm.core import rrtmg_legacy as _legacy
    from gpuwm.core import rrtmg_legacy_prep as _prep_mod
    from gpuwm.core import rrtmg_lw as _lw_mod
    from gpuwm.core import rrtmg_mcica as _mcica_mod
    from gpuwm.core import rrtmgp as _rrtmgp_mod

    counter = {}
    _wrap(_rrtmgp_mod, "cal_cldfra1", "cldfra", counter)
    _wrap(_prep_mod, "lwrad_prep_batch", "lwprep", counter)
    _wrap(_mcica_mod, "gpu_generate_lw_subcolumns", "lwmcica", counter)
    _wrap(_lw_mod, "gpu_rrtmg_lw_batched", "lwengine", counter)
    _wrap(_prep_mod, "lwrad_outputs_batch", "lwout", counter)
    # rrtmg_legacy binds names at module level; re-point the ones it uses.
    print(
        f"[probe:{args.perturb}] legacy adapter file: {_legacy.__file__}",
        flush=True,
    )

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
        "F030 restored MPAS atmosphere (localize probe)",
        checkpoint.atmosphere_fingerprint,
        restored["atmosphere"],
    )
    runner.require_fingerprint_identity(
        "F030 restored Arwen backend (localize probe)",
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
    for name in ("du", "dv", "dtheta"):
        print(
            f"[final] raw/{name} {_sha(getattr(raw, name))}",
            flush=True,
        )
    print(f"[probe:{args.perturb}] anchors={len(anchors)} done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
