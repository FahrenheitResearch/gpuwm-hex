#!/usr/bin/env python3
"""Diagnostic: run N whole-mesh composite steps and dump the atmosphere arrays.

Localization aid for the partition-invariance gate: pairs with the 2-GPU
runner's MG_DUMP_STEPS assembled dumps so a divergence can be mapped to mesh
locations (near-cut vs global).  Not a proof tool; touches nothing pinned.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_cuda_v841_full_physics_x4 as proof  # noqa: E402
import run_cuda_v841_forecast as forecast  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--arwen-checkout", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--horiz-mixing", default="2d_smagorinsky")
    args = parser.parse_args()

    paths = {
        "grid": args.grid.absolute(),
        "static": args.static.absolute(),
        "init": args.init.absolute(),
    }
    authority = forecast.verify_forecast_authorities(paths)
    host = forecast.prepare_forecast_host(
        paths, authority, start_time_text=None, horiz_mixing=args.horiz_mixing
    )

    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(args.arwen_checkout)
    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=args.cache_root
    )
    import cupy as cp

    cache = KernelCache(capability=capability, cache_dir=args.cache_root)
    stack = proof._construct_device_stack(
        host=host, cache=cache, arwen_checkout=args.arwen_checkout
    )
    previous = proof._previous_surface_updates(stack)
    args.out.mkdir(parents=True, exist_ok=True)

    def dump(step: int) -> None:
        atmosphere = stack["driver"].atmosphere
        arrays = {}
        for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
            arrays[f"state.{name}"] = cp.asnumpy(getattr(atmosphere.state, name))
        for name in (
            "theta_m",
            "exner",
            "density_perturbation",
            "rho_theta_perturbation",
            "pressure_perturbation",
            "normal_velocity",
            "vertical_velocity",
        ):
            arrays[f"saved.{name}"] = cp.asnumpy(getattr(atmosphere.saved, name))
        np.savez(args.out / f"whole-step{step}.npz", **arrays)
        print(f"dumped step {step}", flush=True)

    dump(0)
    for step in range(1, int(args.steps) + 1):
        result = proof.execute_composite_step(
            driver=stack["driver"],
            backend=stack["backend"],
            scalar_names=forecast.SCALAR_NAMES,
            physics_geometry=stack["physics_geometry"],
            kernel_cache=stack["driver"].cache,
            previous_surface_updates=previous,
        )
        previous = result.committed.surface_updates
        dump(step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
