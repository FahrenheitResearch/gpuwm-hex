#!/usr/bin/env python3
"""Per-step conservation reading of every transported scalar, window by window.

The composite full-physics step (``run_cuda_v841_full_physics_x4.
execute_composite_step``) has three windows that can change the mass a scalar
carries: the dycore's own transport, the ``+0`` clamp applied to the whole
scalar block before phase two, and phase two itself (the column physics and
any point-source release the selected row carries).  A reading taken only at
phase-two entry, AFTER the clamp, sees the transport and the clamp added
together, and on a global mesh it cannot say which of the two moved the
number.  This probe reads all three windows separately, in the metric the
dycore conserves::

    M_s = sum over levels k and cells c of  q[s,k,c] * rho_zz[k,c] * dzeta_k * area_c

which is the dry mass a cell holds times the mixing ratio (rho_zz dzeta =
rho_dry dz exactly, because zz = dzeta/dz by construction), and the negative
part of the same sum before the clamp, which is the number the clamp will
create.  Every reduction is FP64 over the solve cells.

Readings per step ``n``:

    R0  committed state at the start of the step (end of step n-1)
    R1  the dycore's candidate after transport, BEFORE the clamp
    R2  the same candidate AFTER the clamp (what phase two receives)
    R3  the committed state after phase two and the driver commit

    transport = R1 - R0     (0 on a global mesh for a conservative transport,
                             the lateral-boundary flux on a limited-area one)
    clamp     = R2 - R1     (= -negative part of R1; 0 for a positive field)
    phase2    = R3 - R2     (column physics and any point-source release)

Calibration arms ``--plant uniform`` (a constant mixing ratio everywhere: a
conservative transport keeps it exactly, and nothing can undershoot a
constant, so both transport and clamp must read 0) and ``--plant point`` (one
cell: it must spread without gaining).  ``--source-table`` hands the driver
a point-source table exactly as the door does.  The driver, the door's mesh
bind, the physics row and the seam are the production ones; nothing in the
step is replaced, only observed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo", required=True, help="hex tree root (holds tools/ and src/)")
    p.add_argument("--mesh", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--static", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--init-source", default="conservation probe")
    p.add_argument("--start-time", required=True)
    p.add_argument("--gpuwm-checkout", required=True)
    p.add_argument("--physics-backend", default="wsm6_column")
    p.add_argument("--source-table", default=None)
    p.add_argument("--lbc-dir", default=None)
    p.add_argument("--convection", default="auto")
    p.add_argument("--pbl-cadence", default="auto")
    p.add_argument("--required-free-bytes", type=int, default=None)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--plant", choices=("none", "uniform", "point"), default="none")
    p.add_argument("--plant-species", default="qc")
    p.add_argument("--plant-value", type=float, default=1.0e-3,
                   help="mixing ratio planted, in the scalar's own units (kg/kg for a WSM6 species)")
    p.add_argument("--plant-lat", type=float, default=None)
    p.add_argument("--plant-lon", type=float, default=None)
    p.add_argument("--plant-height", type=float, default=3300.0, help="m MSL")
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--case-label", default="scalar transport conservation probe")
    return p.parse_args(argv)


class _Reader:
    """FP64 reductions on the device, in the dycore's own mass metric."""

    def __init__(self, cp, driver, names, solve_cells, bdy_mask_cell):
        self.cp = cp
        self.names = tuple(names)
        atmosphere = driver.atmosphere
        n_cells = int(atmosphere.state.rho.shape[1])
        self.solve = n_cells if solve_cells is None else int(solve_cells)
        vertical = atmosphere.vertical
        rdzw = cp.asarray(vertical.rdzw, dtype=cp.float64).reshape(-1)
        self.dzeta = 1.0 / rdzw                                   # [nlev]
        self.area = cp.asarray(atmosphere.mesh.area_cell, dtype=cp.float64).reshape(-1)[: self.solve]
        zz = cp.asarray(vertical.zz, dtype=cp.float64)[:, : self.solve]
        zgrid = cp.asarray(vertical.zgrid, dtype=cp.float64)[:, : self.solve]
        self.dz = zgrid[1:] - zgrid[:-1]                          # [nlev, solve]
        self.zz = zz
        if bdy_mask_cell is None:
            self.interior = None
        else:
            mask = np.asarray(bdy_mask_cell).reshape(-1)[: self.solve]
            self.interior = cp.asarray((mask == 0).astype(np.float64))

    def weights(self, rho):
        rho64 = self.cp.asarray(rho, dtype=self.cp.float64)[:, : self.solve]
        return rho64 * self.dzeta[:, None] * self.area[None, :]  # [nlev, solve]

    def ledger_weights(self, rho):
        rho64 = self.cp.asarray(rho, dtype=self.cp.float64)[:, : self.solve]
        return (self.zz * rho64) * self.dz * self.area[None, :]

    def read(self, scalars, rho, *, ledger_rho=None):
        cp = self.cp
        w = self.weights(rho)
        wl = None if ledger_rho is None else self.ledger_weights(ledger_rho)
        total, negative, interior, ledger = {}, {}, {}, {}
        for index, name in enumerate(self.names):
            q = cp.asarray(scalars[index, :, : self.solve], dtype=cp.float64)
            qw = q * w
            total[name] = float(qw.sum())
            negative[name] = float((cp.minimum(q, 0.0) * w).sum())
            if self.interior is not None:
                interior[name] = float((qw * self.interior[None, :]).sum())
            if wl is not None:
                ledger[name] = float((q * wl).sum())
        return {"total": total, "negative": negative, "interior": interior, "ledger": ledger}


def _plant(cp, driver, reader, names, args, log):
    state = driver.atmosphere.state
    index = names.index(args.plant_species)
    solve = reader.solve
    if args.plant == "uniform":
        state.scalars[index, :, :solve] = cp.float32(args.plant_value)
        where = {"kind": "uniform", "cells": solve, "value": args.plant_value}
    else:
        if args.plant_lat is None or args.plant_lon is None:
            raise SystemExit("--plant point needs --plant-lat and --plant-lon")
        lat = np.asarray(cp.asnumpy(driver.atmosphere.mesh.lat_cell)).reshape(-1)[:solve]
        lon = np.asarray(cp.asnumpy(driver.atmosphere.mesh.lon_cell)).reshape(-1)[:solve]
        lat0, lon0 = math.radians(args.plant_lat), math.radians(args.plant_lon)
        lon = np.where(lon > math.pi, lon - 2.0 * math.pi, lon)
        if lon0 > math.pi:
            lon0 -= 2.0 * math.pi
        d = np.arccos(np.clip(np.sin(lat) * math.sin(lat0)
                              + np.cos(lat) * math.cos(lat0) * np.cos(lon - lon0), -1.0, 1.0))
        cell = int(np.argmin(d))
        zgrid = np.asarray(cp.asnumpy(driver.atmosphere.vertical.zgrid))[:, cell]
        zmid = 0.5 * (zgrid[1:] + zgrid[:-1])
        level = int(np.argmin(np.abs(zmid - args.plant_height)))
        state.scalars[index, level, cell] = cp.float32(args.plant_value)
        where = {"kind": "point", "cell": cell, "level": level,
                 "lat_deg": float(math.degrees(lat[cell])),
                 "lon_deg": float(math.degrees(lon[cell])),
                 "z_mid_m": float(zmid[level]), "value": args.plant_value,
                 "great_circle_rad": float(d[cell])}
    cp.cuda.get_current_stream().synchronize()
    log(f"[probe] planted {where}")
    return where


def main(argv=None):
    args = _parse(argv)
    repo = Path(args.repo).resolve()
    for entry in (str(repo / "tools"), str(repo / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    out = Path(args.out).resolve()
    scratch = Path(args.scratch).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch.parent.mkdir(parents=True, exist_ok=True)

    from hexcore import forecast_door as door

    binding, driver_module = door._load_drivers(repo)
    bind_receipt = binding.bind_mesh(
        driver_module.proof, args.mesh, grid=Path(args.grid), static=Path(args.static),
        forecast=driver_module, convection=args.convection, pbl_cadence=args.pbl_cadence,
    )
    proof = driver_module.proof
    dt = float(door.load_registry(repo)[args.mesh].dt_seconds)
    total_seconds = args.steps * dt
    if total_seconds % 60.0 != 0.0:
        raise SystemExit(f"--steps {args.steps} x {dt:g} s is not a whole number of minutes; the driver's history cadence is in minutes")
    hours = repr(total_seconds / 3600.0)
    minutes = int(total_seconds // 60)

    log_lines = []

    def log(text):
        print(text, flush=True)
        log_lines.append(text)

    holder = {}
    rows = []

    original_stack = proof._construct_device_stack
    original_step = proof.execute_composite_step

    def construct_stack(**kwargs):
        stack = original_stack(**kwargs)
        import cupy as cp
        names = tuple(kwargs["host"]["scalar_names"])
        reader = _Reader(cp, stack["driver"], names, stack.get("solve_cells"), stack.get("bdy_mask_cell"))
        holder.update(cp=cp, stack=stack, names=names, reader=reader, plant=None)
        if args.plant != "none":
            holder["plant"] = _plant(cp, stack["driver"], reader, names, args, log)
        state = stack["driver"].atmosphere.state
        holder["initial"] = reader.read(state.scalars, state.rho)
        log(f"[probe] initial totals: " + ", ".join(f"{k}={v:.6e}" for k, v in holder["initial"]["total"].items()))
        return stack

    def composite_step(**kwargs):
        cp, reader = holder["cp"], holder["reader"]
        driver, backend = kwargs["driver"], kwargs["backend"]
        state0 = driver.atmosphere.state
        r0 = reader.read(state0.scalars, state0.rho)
        rho0 = state0.rho
        captured = {}

        real_step = driver.step_device_with_physics

        def step_device_with_physics(held):
            candidate = real_step(held)
            cp.cuda.get_current_stream().synchronize()
            s = candidate.atmosphere.state
            captured["r1"] = reader.read(s.scalars, s.rho, ledger_rho=rho0)
            captured["rho_new"] = s.rho
            return candidate

        real_finish = backend.finish_step

        def finish_step(**fkw):
            s = fkw["atmosphere"].state
            cp.cuda.get_current_stream().synchronize()
            captured["r2"] = reader.read(s.scalars, s.rho, ledger_rho=rho0)
            # The same field in the CARRIER metric: the candidate's own
            # density, the one the door hands the seam as carrier_rho_dry.
            captured["r2_carrier"] = reader.read(s.scalars, s.rho, ledger_rho=s.rho)
            return real_finish(**fkw)

        driver.step_device_with_physics = step_device_with_physics
        backend.finish_step = finish_step
        try:
            result = original_step(**kwargs)
        finally:
            del driver.step_device_with_physics
            del backend.finish_step
        cp.cuda.get_current_stream().synchronize()
        state3 = driver.atmosphere.state
        r3 = reader.read(state3.scalars, state3.rho, ledger_rho=rho0)
        r1, r2 = captured["r1"], captured["r2"]
        row = {
            "step": len(rows) + 1,
            "time_seconds": float(state3.time_seconds),
            "r0": r0, "r1": r1, "r2": r2, "r3": r3,
            "r2_carrier": captured.get("r2_carrier"),
        }
        rows.append(row)
        names = holder["names"]
        focus = args.plant_species if args.plant_species in names else names[0]
        f0, f1, f2, f3 = (r["total"][focus] for r in (r0, r1, r2, r3))
        neg1 = r1["negative"][focus]
        log(
            f"[probe] step {row['step']:3d} {focus}: R0 {f0:.6e} transport {f1 - f0:+.4e} "
            f"clamp {f2 - f1:+.4e} (neg before clamp {neg1:+.4e}) phase2 {f3 - f2:+.4e} "
            f"R3 {f3:.6e}"
        )
        parts = []
        for n in names:
            parts.append(f"{n}: tr {r1['total'][n] - r0['total'][n]:+.3e} cl {r2['total'][n] - r1['total'][n]:+.3e} ({(r2['total'][n] - r1['total'][n]) / max(abs(r1['total'][n]), 1e-300):+.2e} rel)")
        log("[probe]      model species " + "; ".join(parts))
        return result

    proof._construct_device_stack = construct_stack
    proof.execute_composite_step = composite_step

    argv_driver = [
        "--grid", str(Path(args.grid).resolve()), "--static", str(Path(args.static).resolve()),
        "--init", str(Path(args.init).resolve()), "--init-source", args.init_source,
        "--hours", hours, "--history-every-minutes", str(minutes),
        "--arwen-checkout", str(Path(args.gpuwm_checkout).resolve()),
        "--horiz-mixing", "2d_smagorinsky", "--convection", args.convection,
        "--pbl-cadence", args.pbl_cadence,
        "--physics-backend", args.physics_backend,
        "--start-time", args.start_time, "--case-label", args.case_label,
        "--cache-root", str(scratch), "--output", str(out),
    ]
    if args.required_free_bytes is not None:
        argv_driver += ["--required-free-bytes", str(args.required_free_bytes)]
    if args.lbc_dir is not None:
        argv_driver += ["--lbc-dir", str(Path(args.lbc_dir).resolve())]
    if args.source_table is not None:
        argv_driver += ["--source-table", str(Path(args.source_table).resolve())]
    started = time.perf_counter()
    rc = int(driver_module.main(argv_driver))
    wall = time.perf_counter() - started
    payload = {
        "mesh": args.mesh, "dt_seconds": dt, "steps": args.steps, "plant": holder.get("plant"),
        "plant_mode": args.plant, "plant_species": args.plant_species, "plant_value": args.plant_value,
        "source_table": args.source_table, "physics_backend": args.physics_backend,
        "scalar_names": list(holder.get("names", ())), "solve_cells": holder["reader"].solve if "reader" in holder else None,
        "initial": holder.get("initial"), "rows": rows, "driver_rc": rc, "wall_seconds": wall,
        "bind": {k: v for k, v in bind_receipt.items() if isinstance(v, (str, int, float, bool))} if isinstance(bind_receipt, dict) else str(bind_receipt),
        "metric": "sum_k sum_c q * rho_zz * dzeta_k * areaCell over solve cells, FP64; 'ledger' uses zz*rho * dz * areaCell at the START-of-step rho (the phase-1 density); 'r2_carrier' repeats the post-clamp reading with its ledger column at the candidate's own rho (the carrier density the door hands the seam)",
        "log": log_lines,
    }
    (out / "conservation-probe.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"[probe] wrote {out / 'conservation-probe.json'} rc={rc} wall={wall:.1f}s", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
