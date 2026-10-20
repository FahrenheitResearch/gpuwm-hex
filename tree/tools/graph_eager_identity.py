"""CUDA subcycle graph replay against eager on a real MPAS mesh and init.

F1 (2026-10-15) companion to tests/test_cuda_subcycle_graphs_physics.py, for
meshes the route-A forecast door does not run (x1.2562 has no registered
route-A mesh row or static file). Two CudaDryDycoreDriver twins are built from
the same init; graphs are attached to one. Two legs, each over --hours of
model time at --dt:

  whole-steps   the real dry step (moist coefficients, three dynamics
                subcycles, scalar transport) through step_device(); state and
                saved diagnostics compared bitwise after every step.
  physics       the real v8.4.1 subcycle chained three per step with held
                physics tendencies, moist coefficients and SG2 lagged heating
                that is absent on step 1 and present from step 2 (F1's shape);
                every result array compared bitwise after every subcycle.

The physics leg's tendencies are seeded synthetic values, not a physics
suite: this is a replay-identity instrument and makes no forecast claim.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class _Held:
    rho: Any
    rho_u: Any
    rho_theta: Any
    scalars: Any
    time_seconds: float
    diabatic_theta: Any = None


def _diff(cp, actual, expected, path, rows):
    if isinstance(expected, cp.ndarray):
        a = cp.asnumpy(actual).view(np.uint32 if expected.dtype.itemsize == 4 else np.uint8)
        e = cp.asnumpy(expected).view(np.uint32 if expected.dtype.itemsize == 4 else np.uint8)
        rows.append((path, int(e.size), int(np.count_nonzero(a != e))))
    elif is_dataclass(expected):
        for field in fields(expected):
            if field.name in ("stage_physics_tendencies", "time_seconds", "h2d"):
                continue
            _diff(cp, getattr(actual, field.name), getattr(expected, field.name),
                  f"{path}.{field.name}", rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--scalars", default="qv")
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tree = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(tree / "src")]
    import cupy as cp
    from hexcore.config_v841 import V841DryDycoreConfig
    from hexcore.cuda_backend import require_cuda
    from hexcore.cuda_backend.runtime import KernelCache
    from hexcore.cuda_driver import CudaDryDycoreDriver, _CudaV841MoistDynamicsCoefficients
    from hexcore.cuda_subcycle_graphs import attach_subcycle_graphs
    from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from hexcore.dynamics_v841 import load_v841_reference_wind_profiles
    from hexcore.integration import RKSchedule
    from hexcore.mesh import Mesh
    from hexcore.vector import initialize_reconstruction_coefficients

    steps = a.hours * 3600.0 / a.dt
    if not float(steps).is_integer() or steps < 2:
        raise SystemExit("--hours must be at least two whole --dt steps")
    steps = int(steps)
    names = tuple(n.strip() for n in a.scalars.split(",") if n.strip())
    mesh = Mesh.from_netcdf(a.init)
    mesh.arrays["coeffs_reconstruct"] = initialize_reconstruction_coefficients(mesh, include_halos=True)
    grid = load_mpas_vertical_grid(a.init, mesh, config_coef_3rd_order=.25)
    profiles = load_v841_reference_wind_profiles(a.init, n_vert_levels=grid.vertical_grid.n_vert_levels)
    for name in ("defc_a", "defc_b"):
        mesh.arrays[name] = np.zeros((len(mesh.arrays["nEdgesOnCell"]),
                                      mesh.arrays["edgesOnCell"].shape[1]), np.float32)
    config = V841DryDycoreConfig(
        config_dt=a.dt, config_time_integration_order=3, config_number_of_sub_steps=6,
        config_dynamics_split_steps=3, config_split_dynamics_transport=True, config_monotonic=True,
        config_positive_definite=False, config_scalar_adv_order=3, config_scalar_vadv_order=3,
        config_coef_3rd_order=.25, config_apvm_upwinding=.5, config_smdiv=.1,
        config_divergence_damping=True, config_xnutr=.2, config_zd=22000.,
        config_visc4_2dsmag=.05, config_smagorinsky_coef=.125, config_del4u_div_factor=10.,
        config_h_ScaleWithMesh=True, config_horiz_mixing="2d_smagorinsky")
    config.validate()
    cache = KernelCache(capability=require_cuda(min_compute=(7, 5)), cache_dir=str(a.cache_root))

    def twins():
        out = []
        for _ in range(2):
            state, reference, saved = load_mpas_initial_state(
                a.init, mesh, grid.vertical_grid, scalar_names=names, reference_path=a.init,
                terrain_metrics=grid.terrain_metrics, return_saved_diagnostics=True)
            out.append(CudaDryDycoreDriver.from_host(
                mesh, state, grid.vertical_grid, reference, config, saved_diagnostics=saved,
                terrain_metrics=grid.terrain_metrics, reference_wind_profiles=profiles,
                kernel_cache=cache))
        return out, attach_subcycle_graphs(out[1])

    report: dict[str, Any] = {
        "schema": "gpuwm-hex.graph-eager-identity/v1", "init": str(a.init),
        "cells": int(mesh.dimensions["nCells"]), "edges": int(mesh.dimensions["nEdges"]),
        "levels": int(grid.vertical_grid.n_vert_levels), "dt": a.dt, "hours": a.hours,
        "steps": steps, "scalars": list(names),
        "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    }
    t0 = time.perf_counter()

    # Leg 1: whole dry steps through the real step.
    drivers, graphs = twins()
    rows_by_step = []
    for step in range(1, steps + 1):
        for driver in drivers:
            driver.atmosphere = driver.step_device().atmosphere
        cp.cuda.get_current_stream().synchronize()
        rows: list = []
        _diff(cp, drivers[1].atmosphere.state, drivers[0].atmosphere.state, "state", rows)
        _diff(cp, drivers[1].atmosphere.saved, drivers[0].atmosphere.saved, "saved", rows)
        rows_by_step.append({"step": step, "arrays": len(rows), "words": sum(r[1] for r in rows),
                             "different_words": sum(r[2] for r in rows),
                             "differing": [r[0] for r in rows if r[2]]})
    report["whole_steps"] = {"rows": rows_by_step, "graphs": graphs.receipt(),
                             "all_bitwise": all(r["different_words"] == 0 for r in rows_by_step)}
    del drivers, graphs
    cp.get_default_memory_pool().free_all_blocks()

    # Leg 2: physics-forced subcycles, lagged heating from step 2.
    drivers, graphs = twins()
    nlev, ncells, nedges = (int(grid.vertical_grid.n_vert_levels), int(mesh.dimensions["nCells"]),
                            int(mesh.dimensions["nEdges"]))
    schedule = RKSchedule.from_mpas(a.dt, order=3, acoustic_substeps=6, dynamics_splits=3)
    diagnostics = [d.horizontal.solve_diagnostics(
        d.atmosphere.state.rho, d.atmosphere.state.rho_u, dt=a.dt, apvm_upwinding=.5,
        normal_velocity=d.atmosphere.saved.normal_velocity, rk_step=3) for d in drivers]
    n_scalars = int(drivers[0].atmosphere.state.scalars.shape[0])
    rho0 = drivers[0].atmosphere.state.rho
    rt0 = drivers[0].atmosphere.state.rho_theta
    sub_rows = []
    for step in range(steps):
        rng = np.random.default_rng(2562 + step)

        def noise(shape, scale):
            return cp.asarray((scale * rng.standard_normal(shape)).astype(np.float32))

        held = _Held(
            rho=(rho0 * noise((nlev, ncells), 1.0e-8)).astype(cp.float32),
            rho_u=noise((nlev, nedges), 1.0e-4),
            rho_theta=(rt0 * noise((nlev, ncells), 1.0e-8)).astype(cp.float32),
            scalars=noise((n_scalars, nlev, ncells), 1.0e-9),
            time_seconds=step * a.dt,
            diabatic_theta=None if step == 0 else cp.abs(noise((nlev, ncells), 5.0e-4)),
        )
        moist = _CudaV841MoistDynamicsCoefficients(
            qtot=cp.abs(noise((nlev, ncells), 1.0e-3)),
            cqw=(1.0 + noise((nlev, ncells), 1.0e-4)).astype(cp.float32),
            cqu=(1.0 + noise((nlev, nedges), 1.0e-4)).astype(cp.float32))
        for substep in range(3):
            flags = [cp.zeros((1,), dtype=cp.int32) for _ in drivers]
            results = []
            for index, driver in enumerate(drivers):
                driver.atmosphere.state.time_seconds = float(step * a.dt)
                results.append(driver._advance_dynamics_subcycle_v841(
                    driver.atmosphere.state, time_level_one=driver.atmosphere.saved,
                    initial_diagnostics=diagnostics[index], schedule=schedule, outer_dt=a.dt,
                    validation_flag=flags[index], physics_tendencies=held,
                    moist_coefficients=moist, dynamics_substep=substep + 1))
            cp.cuda.get_current_stream().synchronize()
            rows = []
            _diff(cp, results[1], results[0], "result", rows)
            sub_rows.append({"step": step + 1, "subcycle": substep + 1,
                             "heating": held.diabatic_theta is not None,
                             "arrays": len(rows), "words": sum(r[1] for r in rows),
                             "different_words": sum(r[2] for r in rows),
                             "differing": [r[0] for r in rows if r[2]],
                             "invalid_flags": [int(cp.asnumpy(f)[0]) for f in flags]})
            for index, (driver, result) in enumerate(zip(drivers, results)):
                driver.atmosphere.state = result.state
                driver.atmosphere.saved = result.saved
                diagnostics[index] = result.diagnostics
    report["physics_subcycles"] = {
        "rows": sub_rows, "graphs": graphs.receipt(),
        "all_bitwise": all(r["different_words"] == 0 for r in sub_rows),
        "invalid_flags_zero": all(r["invalid_flags"] == [0, 0] for r in sub_rows)}
    report["seconds"] = time.perf_counter() - t0
    report["all_bitwise"] = report["whole_steps"]["all_bitwise"] and report["physics_subcycles"]["all_bitwise"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    w, s = report["whole_steps"], report["physics_subcycles"]
    print(json.dumps({"all_bitwise": report["all_bitwise"],
                      "whole_steps": {"steps": len(w["rows"]), "words_per_step": w["rows"][0]["words"],
                                      "different_words": sum(r["different_words"] for r in w["rows"]),
                                      "graphs": {k: w["graphs"][k] for k in ("captured_subcycles", "replayed_subcycles", "eager_subcycles")}},
                      "physics_subcycles": {"subcycles": len(s["rows"]), "words_per_subcycle": s["rows"][0]["words"],
                                            "different_words": sum(r["different_words"] for r in s["rows"]),
                                            "invalid_flags_zero": s["invalid_flags_zero"],
                                            "graphs": {k: s["graphs"][k] for k in ("captured_subcycles", "replayed_subcycles", "eager_subcycles")}}},
                     sort_keys=True))
    return 0 if report["all_bitwise"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
