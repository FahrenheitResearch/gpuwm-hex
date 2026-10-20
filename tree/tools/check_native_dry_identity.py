#!/usr/bin/env python3
"""Compare the hex dry dycore with a native MPAS-A run, output by output.

The native side is a directory of ``tracer.<time>.nc`` files written by a stock
``atmosphere_model`` at every model step (the stream of
``tools/run_nonzero_tracer_authority.sh``: rho_zz, theta_m, ru, rw, u, w, the
base and perturbation fields, exner, pressure and the scalars), from a dry
Jablonowski-Williamson initial state.  The hex side starts from the native
file at time zero and is advanced with the same schedule:

* ``cuda``: one continuous device run, compared with the native file after
  every step;
* ``cpu``: the NumPy port, one step from each native file to the next.

The report names the first output and field whose bytes differ, the largest
absolute difference and the count of differing 32-bit words.  It measures; it
does not certify and it refuses nothing.

Written for the MPAS-A v8.4.2 re-pin (2026-10-11), where the question was
whether the port stands in the same relation to native v8.4.2 as to native
v8.4.1 at two and at four acoustic sub-steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "tools"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")


def build_config(args: argparse.Namespace) -> Any:
    from hexcore.config_v841 import V841DryDycoreConfig

    values: dict[str, Any] = dict(
        config_dt=float(args.dt),
        config_time_integration_order=int(args.order),
        config_number_of_sub_steps=int(args.substeps),
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_scalar_advection=True,
        config_monotonic=True,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_smdiv=0.0,
        config_divergence_damping=False,
        config_xnutr=0.0,
        config_zd=22_000.0,
    )
    if args.mixing == "smag":
        values.update(config_horiz_mixing="2d_smagorinsky", config_visc4_2dsmag=0.05)
    else:
        values.update(config_horiz_mixing="off")
    config = V841DryDycoreConfig(**values)
    config.validate()
    return config


def field_report(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"identical": False, "why": f"{actual.shape}/{actual.dtype} against "
                                           f"{expected.shape}/{expected.dtype}"}
    if actual.tobytes() == expected.tobytes():
        return {"identical": True}
    gap = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    scale = float(np.max(np.abs(expected)))
    return {
        "identical": False,
        "max_abs": float(gap.max()),
        "max_abs_relative_to_field_max": float(gap.max() / scale) if scale else None,
        "different_words": int(np.count_nonzero(
            actual.view(np.uint32) != expected.view(np.uint32))),
        "words": int(actual.size),
    }


def compare_states(actual: Any, expected: Any) -> dict[str, Any]:
    return {name: field_report(np.asarray(getattr(actual, name)),
                               np.asarray(getattr(expected, name)))
            for name in STATE_FIELDS}


def first_difference(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in steps:
        for name in STATE_FIELDS:
            if not row["fields"][name]["identical"]:
                return {"step": row["step"], "field": name, **row["fields"][name]}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--init", type=Path, required=True,
                        help="the init file the native run read (mesh and vertical grid)")
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--order", type=int, choices=(2, 3), required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--mixing", choices=("off", "smag"), default="off")
    parser.add_argument("--backend", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args(argv)

    from hexcore.driver import (
        DryDycoreDriver, load_mpas_initial_state, load_mpas_vertical_grid)
    from hexcore.dynamics_v841 import load_v841_reference_wind_profiles
    from hexcore.mesh import Mesh

    config = build_config(args)
    native_files = sorted(args.native_dir.glob("tracer.*.nc"))
    if len(native_files) < 2:
        raise SystemExit(f"{args.native_dir} holds fewer than two native outputs")
    if args.steps is not None:
        native_files = native_files[: args.steps + 1]
    mesh = Mesh.from_netcdf(args.init)
    native = load_mpas_vertical_grid(
        args.init, mesh, config_coef_3rd_order=config.config_coef_3rd_order)
    profiles = load_v841_reference_wind_profiles(
        args.init, n_vert_levels=native.vertical_grid.n_vert_levels)

    def load(path: Path) -> tuple[Any, Any, Any]:
        return load_mpas_initial_state(
            path, mesh, native.vertical_grid, scalar_names=("qv",),
            reference_path=path, terrain_metrics=native.terrain_metrics,
            return_saved_diagnostics=True)

    loaded = [load(path) for path in native_files]
    report: dict[str, Any] = {
        "native_dir": str(args.native_dir),
        "native_outputs": [path.name for path in native_files],
        "config": {"dt": float(args.dt), "order": int(args.order),
                   "acoustic_substeps": int(args.substeps), "dynamics_splits": 3,
                   "mixing": args.mixing},
        "cells": int(np.asarray(loaded[0][0].rho).shape[1]),
        "levels": int(np.asarray(loaded[0][0].rho).shape[0]),
    }

    if args.backend in ("cpu", "both"):
        rows = []
        for index in range(len(loaded) - 1):
            state, reference, saved = loaded[index]
            driver = DryDycoreDriver(
                mesh, native.vertical_grid, reference, config,
                terrain_metrics=native.terrain_metrics,
                reference_wind_profiles=profiles)
            result = driver.step(state, saved_diagnostics=saved)
            rows.append({"step": index + 1,
                         "fields": compare_states(result.state, loaded[index + 1][0])})
        report["cpu_one_step_from_each_native_output"] = {
            "steps": rows, "first_difference": first_difference(rows),
            "identical": first_difference(rows) is None}

    if args.backend in ("cuda", "both"):
        from hexcore.cuda_backend import require_cuda
        from hexcore.cuda_backend.runtime import KernelCache
        from hexcore.cuda_driver import CudaDryDycoreDriver

        if args.cache_root is None:
            raise SystemExit("--cache-root is required for the cuda backend")
        state, reference, saved = loaded[0]
        # arrays on the mesh; with mixing off they are exact zeros, as the
        # forecast door attaches them.
        arrays = mesh.arrays
        for name in ("defc_a", "defc_b"):
            present = arrays.get(name)
            if present is None or np.asarray(present).dtype.kind != "f":
                counts = np.asarray(arrays["nEdgesOnCell"])
                edges = np.asarray(arrays["edgesOnCell"])
                width = edges.shape[1] if edges.shape[0] == counts.size else edges.shape[0]
                arrays[name] = np.zeros((counts.size, width), dtype=np.float32)
        capability = require_cuda(min_compute=(12, 0))
        cache = KernelCache(capability=capability, cache_dir=str(args.cache_root))
        driver = CudaDryDycoreDriver.from_host(
            mesh, state, native.vertical_grid, reference, config,
            saved_diagnostics=saved, terrain_metrics=native.terrain_metrics,
            kernel_cache=cache, reference_wind_profiles=profiles)
        rows = []
        for index in range(1, len(loaded)):
            result = driver.step_device()
            driver.atmosphere = result.atmosphere
            host = driver.atmosphere.state.to_host()
            rows.append({"step": index, "fields": compare_states(host, loaded[index][0])})
        report["cuda_continuous_from_native_time_zero"] = {
            "steps": rows, "first_difference": first_difference(rows),
            "identical": first_difference(rows) is None}

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({key: ({"identical": value["identical"],
                             "first_difference": value["first_difference"]}
                            if isinstance(value, dict) and "identical" in value else value)
                      for key, value in report.items() if key != "native_outputs"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
