#!/usr/bin/env python
"""Box-only dry-dynamics and scalar-transport arms for active full-step local time stepping.

``--arm fixed``  the pinned native step at one global ``--dt`` (no LTS module is
                 imported); the reference every other arm is scored against.
``--arm lts``    rate classes from the mesh (or ``--forced-cap`` on a uniform
                 mesh, where the geometry alone gives one class), each stepping
                 whole RK steps at ``rate * dt``.  A mesh whose cells all class
                 at rate 1 dispatches to the native step before any LTS object
                 exists, so "on" equals "off" there by construction.

Both arms run the same configuration: RK2, three dynamics splits,
``--acoustic-substeps`` acoustic sub-steps (MPAS-A v8.4.2's default, 4, unless
given; receipts before 2026-10-11 ran 2), the shipped Smagorinsky filter, no
physics.  ``--transport`` turns on
the native scalar transport with the monotonic limiter for six species: the
initial file's qv, qc and qr, and (``--tracers test``, the default) three test
tracers in the slots that are exactly zero in the initial file:

  unit   1 everywhere: stays 1 only if a scalar and the density are moved by
         one mass flux through every edge, interface edges included;
  band   1 in the cells whose spacing is 1.5 to 3 times the finest (or the
         ``--band`` annulus), 0 elsewhere: a step that sits on the rate-1/2
         interface, the hardest thing for the limiter there;
  ramp   0.5 (1 + sin lat) (k + 1) / nlev: smooth in both directions.

Each arm writes a receipt with wall time per forecast hour, the FP64 dry-mass,
coupled-theta and tracer-mass totals through the run, the extreme mixing
ratios, health, and optionally the final state for ``--compare``.
``--clamp-audit`` adds up the mass the native third stage clamps at zero, so a
change of a tracer total can be split into clamped mass and rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

STATE = ("rho", "rho_theta", "rho_u", "rho_w")


TEST_TRACERS = ("unit", "band", "ramp")


def build_dry_config(dt: float, *, mixing: str, len_disp: float, apvm: float, transport: bool = False,
                     acoustic_substeps: int):
    from run_lts_dry_gates import build_config

    extra = dict(config_time_integration_order=2, config_number_of_sub_steps=int(acoustic_substeps),
                 config_monotonic=True, config_scalar_advection=bool(transport),
                 config_apvm_upwinding=float(apvm))
    if mixing == "smagorinsky":
        extra.update(config_horiz_mixing="2d_smagorinsky", config_len_disp=float(len_disp),
                     config_visc4_2dsmag=0.05, config_smagorinsky_coef=0.125,
                     config_del4u_div_factor=10.0, config_h_ScaleWithMesh=True)
    config = replace(build_config("off", dt=dt, rates=(1, 3), rings=2), **extra)
    config.validate()
    return config


def bind_and_prepare(args, config):
    """The public mesh binder, then the dry gate's host preparation."""

    from netCDF4 import Dataset
    from hexcore import mesh_rows
    from hexcore.drivers import mpas_mesh_binding as binding
    import run_cuda_v841_forecast as door
    from run_lts_dry_gates import prepare_host

    paths = {name: getattr(args, name).resolve(strict=True) for name in ("grid", "static", "init")}
    with Dataset(paths["static"]) as dataset:
        nominal = float(dataset["nominalMinDc"][:].item())
    spec = args.out / "input-provenance.json"
    spec.write_text(json.dumps(dict(
        grid=str(paths["grid"]), static=str(paths["static"]), init=str(paths["init"]),
        init_sha256=mesh_rows.sha256_file(paths["init"]),
        scope="dry dynamics arms of the full-step LTS stage, original mesh coordinates")))
    row = replace(mesh_rows.describe_generated(
        name="fullstep-lts-dry", grid=paths["grid"], static=paths["static"], spec_path=spec,
        generator_receipt=spec, static_receipt=spec, point_deg=(0, 0), fine_dx_m=nominal,
        core_radius_km=0, background_km=nominal / 1000, dt_seconds=args.dt, n_levels=55,
        admission={"scope": "dry dynamics instrument, no forecast qualification"}),
        kind=mesh_rows.ARCHIVED_GLOBAL_KIND)
    rowfile = mesh_rows.write_rows(args.out / "mesh-row.json", [row])
    binding.MESH_BINDINGS = mesh_rows.apply_rows(binding.MESH_BINDINGS, binding.MeshBinding, rowfile)
    receipt = binding.bind_mesh(door.proof, row.name, grid=paths["grid"], static=paths["static"],
                                forecast=door, convection="off", pbl_cadence="auto",
                                time_integration_order=2, acoustic_substeps=args.acoustic_substeps,
                                dt_seconds=args.dt)
    (args.out / "binding.json").write_text(json.dumps(receipt, indent=2, default=str))
    return paths, nominal, prepare_host(paths, config, scalars="wsm6")


def metric_lengths(mesh) -> tuple[np.ndarray, np.ndarray]:
    dc = np.asarray(mesh.arrays["dcEdge"], dtype=np.float64)
    if dc.max() < 10.0:  # a unit-sphere grid file; the model's radius is 6,371,229 m
        dc = dc * 6371229.0
    return dc, np.asarray(mesh.arrays["cellsOnEdge"], dtype=np.int64)


def plant_test_tracers(args, host) -> list[str]:
    """Replace the three exactly-zero species by the test tracers; returns the species names."""

    scalars = host["state"].scalars
    names = ["qv", "qc", "qr", "qi", "qs", "qg"][:int(scalars.shape[0])]
    if args.tracers != "test" or scalars.shape[0] < 6:
        return names
    for slot in (3, 4, 5):
        if np.any(np.asarray(scalars[slot]) != 0.0):
            raise SystemExit(f"refusing: species {names[slot]} is not zero in the initial state; "
                             "planting a test tracer there would overwrite real water")
    mesh = host["mesh"]
    n_levels, n_cells = int(scalars.shape[1]), int(scalars.shape[2])
    lat = np.asarray(mesh.arrays["latCell"], np.float64)
    lon = np.asarray(mesh.arrays["lonCell"], np.float64)
    if args.band:
        lat0, lon0, inner, outer = (float(v) for v in args.band.split(","))
        a, b = np.deg2rad(lat0), np.deg2rad(lon0)
        cosine = np.sin(lat) * np.sin(a) + np.cos(lat) * np.cos(a) * np.cos(lon - b)
        distance = 6371.229 * np.arccos(np.clip(cosine, -1.0, 1.0))
        band = (distance >= inner) & (distance < outer)
    else:
        dc, coe = metric_lengths(mesh)
        spacing = np.full(n_cells, np.inf)
        np.minimum.at(spacing, coe[:, 0], dc)
        np.minimum.at(spacing, coe[:, 1], dc)
        ratio = spacing / spacing.min()
        band = (ratio >= 1.5) & (ratio < 3.0)
    scalars[3][...] = 1.0
    scalars[4][...] = band[None, :].astype(scalars.dtype)
    levels = (np.arange(n_levels, dtype=np.float64) + 1.0) / n_levels
    scalars[5][...] = (levels[:, None] * 0.5 * (1.0 + np.sin(lat))[None, :]).astype(scalars.dtype)
    names[3:6] = list(TEST_TRACERS)
    return names


def initial_donor_fraction(host, dt: float) -> np.ndarray:
    from hexcore.fullstep_lts_admission import initial_donor_fraction as library

    return library(host["mesh"], host["state"], host["vertical"], dt)


def assign_rates(args, host) -> tuple[np.ndarray, dict]:
    """Dyadic rate of every cell: geometry, initial wind and sound-speed bounds, buffer, grading."""

    from hexcore.fullstep_lts import buffer_rates, geometric_rates
    from hexcore.fullstep_lts_classes import DYADIC, grade_rates

    mesh, state, saved = host["mesh"], host["state"], host["saved"]
    dc, coe = metric_lengths(mesh)
    n_cells = int(np.asarray(state.rho).shape[1])
    info: dict = {}
    if args.forced_cap:
        lat0, lon0, *radii = (float(v) for v in args.forced_cap.split(","))
        lat, lon = np.asarray(mesh.arrays["latCell"], np.float64), np.asarray(mesh.arrays["lonCell"], np.float64)
        a, b = np.deg2rad(lat0), np.deg2rad(lon0)
        cosine = np.sin(lat) * np.sin(a) + np.cos(lat) * np.cos(a) * np.cos(lon - b)
        distance = 6371.229 * np.arccos(np.clip(cosine, -1.0, 1.0))
        rates = np.full(n_cells, 2 ** len(radii), dtype=np.int32)
        for level, radius in reversed(list(enumerate(radii))):
            rates[distance < radius] = 2 ** level
        rates = np.minimum(rates, args.max_rate).astype(np.int32)
        info["source"] = f"forced cap at ({lat0},{lon0}), radii km {radii}: a test map, not a mesh admission"
    else:
        # One admission rule for this instrument and the forecast route.
        from hexcore.fullstep_lts_admission import admit_mesh_rates

        try:
            rates, info = admit_mesh_rates(
                mesh, state, saved, host["vertical"], dt=args.dt, max_rate=args.max_rate,
                target_advection=args.target_advection, target_acoustic=args.target_acoustic,
                target_transport=args.target_transport if args.transport else None,
                acoustic_substeps=args.acoustic_substeps)
        except ValueError as error:
            raise SystemExit(f"refusing: {error}") from error
        if int(rates.max()) == 1:
            return rates, info
    for spec in args.demote_disc or ():
        lat0, lon0, radius, rate = (float(v) for v in spec.split(","))
        lat, lon = np.asarray(mesh.arrays["latCell"], np.float64), np.asarray(mesh.arrays["lonCell"], np.float64)
        a, b = np.deg2rad(lat0), np.deg2rad(lon0)
        cosine = np.sin(lat) * np.sin(a) + np.cos(lat) * np.cos(a) * np.cos(lon - b)
        inside = 6371.229 * np.arccos(np.clip(cosine, -1.0, 1.0)) < radius
        rates = np.where(inside, np.minimum(rates, int(rate)), rates).astype(np.int32)
        info.setdefault("demoted_discs", []).append(spec)
    rates = grade_rates(rates, coe)
    info["cells"] = {str(r): int(np.count_nonzero(rates == r)) for r in DYADIC}
    return rates, info


def totals_fixed(cp, driver, weight, transport: bool = False) -> dict:
    state = driver.atmosphere.state
    air = state.rho.astype(cp.float64) * weight
    result = {"dry_mass": float(cp.sum(air)),
              "rho_theta": float(cp.sum(state.rho_theta.astype(cp.float64) * weight))}
    if transport:
        count = int(state.scalars.shape[0])
        result["tracer_mass"] = [float(cp.sum(state.scalars[t].astype(cp.float64) * air)) for t in range(count)]
        result["tracer_min"] = [float(cp.min(state.scalars[t])) for t in range(count)]
        result["tracer_max"] = [float(cp.max(state.scalars[t])) for t in range(count)]
    return result


def health_fixed(cp, driver, dt) -> dict:
    atmosphere = driver.atmosphere
    state, saved = atmosphere.state, atmosphere.saved
    speed = cp.abs(saved.normal_velocity)
    return {"finite": all(bool(cp.isfinite(v).all()) for v in (state.rho, state.rho_theta, state.rho_w, state.rho_u)),
            "max_abs_w": float(cp.max(cp.abs(saved.vertical_velocity))), "max_abs_u": float(cp.max(speed)),
            "min_rho": float(cp.min(state.rho)),
            "advective_courant_per_class": [float(cp.max(speed / atmosphere.mesh.dc_edge[None, :])) * dt],
            "max_w_per_class": []}


def parse_devices(text, cp) -> tuple[int, ...]:
    if text is None:
        return (0,)
    if text == "all":
        return tuple(range(int(cp.cuda.runtime.getDeviceCount())))
    # A card may be listed twice: two ranks then share it (the partition
    # identity check on a one-card grant); it is not a speed configuration.
    devices = tuple(int(token) for token in str(text).split(","))
    if min(devices) < 0:
        raise SystemExit("refusing: --devices needs card ordinals")
    return devices


def build_plan(args, host, rates, area):
    from hexcore.fullstep_lts_classes import build_class_plan

    mesh = host["mesh"]
    return build_class_plan(
        rates=rates, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]),
        cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
        cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]), area_cell=area,
        dc_edge_m=metric_lengths(mesh)[0], base_dt=args.dt,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"),
        signal_speed=args.zone_speed, zone_factor=args.zone_factor, min_zone_rings=args.zone_min_rings)


def cell_xyz(mesh) -> np.ndarray:
    lat = np.asarray(mesh.arrays["latCell"], np.float64)
    lon = np.asarray(mesh.arrays["lonCell"], np.float64)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)


def run_multi(args, host, config, rates, devices, receipt, record, walls, area, dzw, species):
    """Both arms on several cards: rate classes cut into blocks (one class of rate 1 for the fixed arm)."""

    import cupy as cp
    from hexcore.cuda_fullstep_lts_multi import FullStepLtsMulti, build_block_drivers
    from hexcore.fullstep_lts_blocks import blocks_per_class, build_block_plan
    from hexcore.fullstep_lts_classes import plan_macros

    mesh = host["mesh"]
    n_cells = int(area.size)
    one_class = rates is None or int(rates.max()) == 1
    plan_rates = np.ones(n_cells, dtype=np.int32) if one_class else rates
    began = time.perf_counter()
    plan = build_plan(args, host, plan_rates, area)
    if args.blocks:
        counts = tuple(int(v) for v in args.blocks.split(","))
    else:
        counts = blocks_per_class(plan, len(devices), min_block_cells=args.block_cells)
        if one_class:
            counts = (len(devices),)
    bplan = build_block_plan(
        plan, devices=devices, n_blocks=counts, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
        n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]),
        cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]), cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]),
        cell_xyz=cell_xyz(mesh), area_cell=area,
        edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"))
    receipt["plan"] = plan.receipt()
    receipt["blocks"] = bplan.receipt()
    receipt["plan_seconds"] = time.perf_counter() - began
    receipt["dispatch"] = ("one rate class cut into blocks (fixed step on several cards)" if one_class
                           else "rate classes cut into blocks on several cards")
    print(json.dumps({"plan": receipt["plan"], "blocks": receipt["blocks"],
                      "plan_seconds": receipt["plan_seconds"]}), flush=True)
    drivers, areas, caches = build_block_drivers(host, config, bplan, base_dt=args.dt, cache_root=args.cache)
    active = FullStepLtsMulti(bplan, drivers, base_dt=args.dt, kernel_caches=caches, area_cell=areas,
                              dzw=dzw, clamp_audit=bool(args.clamp_audit), copy_policy=args.copy_policy)
    try:
        if active.ntr:
            active.guard = not args.no_donor_guard
            active.cap = not args.no_reserve_cap
            active.packets = not args.no_packets
            receipt["donor_guard"], receipt["reserve_cap"], receipt["packets"] = active.guard, active.cap, active.packets
        active.reflux_scale = {"full": (1.0, 1.0), "mass": (1.0, 0.0), "none": (0.0, 0.0)}[args.reflux]
        receipt["reflux"] = args.reflux
        receipt["construct_seconds"] = time.perf_counter() - began
        record(0, active.totals(), active.health())
        steps = int(receipt["steps"])
        top = 1 if one_class else min(max(plan.rates), args.macro_cap)
        macros = plan_macros(steps, top, warmup_steps=0 if one_class else args.warmup_steps)
        if one_class:
            # timed blocks of base steps, as the one-card fixed arm times them
            block = max(1, int(args.block_steps))
            grouped, run_ = [], 0
            for _ in macros:
                run_ += 1
                if run_ == block:
                    grouped.append(run_)
                    run_ = 0
            if run_:
                grouped.append(run_)
        else:
            grouped = [None] * len(macros)
        done = 0
        next_total = args.totals_every
        cursor = 0
        for group in grouped:
            began_block = time.perf_counter()
            if one_class:
                for _ in range(group):
                    active.advance_macro(1)
                count = group
            else:
                count = macros[cursor]
                active.advance_macro(count)
                cursor += 1
            walls.append({"base_steps": count, "seconds": time.perf_counter() - began_block})
            done += count
            if done >= next_total or done == steps:
                next_total = (done // args.totals_every + 1) * args.totals_every
                record(done, active.totals(), active.health())
        receipt["active"] = active.receipt()
        if not active.registers_empty():
            raise RuntimeError("an interface register is not empty at the end of the run")
        final = active.gather(STATE)
        if active.ntr:
            final["scalars"] = active.gather_scalars()
            receipt["scalar_interface"] = active.scalar_stats()
        if rates is not None:
            receipt["rates_sha256"] = hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest()
            if args.save_state:
                np.save(args.out / "rates.npy", rates)
        peak = []
        for device in devices:
            with cp.cuda.Device(device):
                free, total = cp.cuda.runtime.memGetInfo()
                pool = cp.get_default_memory_pool()
                peak.append({"device": device, "pool_used_bytes": int(pool.used_bytes()),
                             "pool_total_bytes": int(pool.total_bytes()), "device_free_bytes": int(free),
                             "device_total_bytes": int(total)})
        receipt["memory_end_of_run"] = peak
    finally:
        active.close()
    return final


def run(args) -> int:
    import cupy as cp
    from hexcore.cuda_backend import require_cuda
    from hexcore.cuda_backend.runtime import KernelCache

    args.out.mkdir(parents=True, exist_ok=False)
    from netCDF4 import Dataset
    with Dataset(args.static) as dataset:
        nominal = float(dataset["nominalMinDc"][:].item())
    config = build_dry_config(args.dt, mixing=args.mixing, len_disp=nominal, apvm=args.apvm,
                              transport=args.transport, acoustic_substeps=args.acoustic_substeps)
    paths, nominal, host = bind_and_prepare(args, config)
    species = plant_test_tracers(args, host) if args.transport else []
    steps = int(round(args.hours * 3600.0 / args.dt)) if args.steps is None else int(args.steps)
    if abs(steps * args.dt - (args.hours * 3600.0 if args.steps is None else steps * args.dt)) > 1e-6:
        raise SystemExit("refusing: the requested duration is not a whole number of base steps; "
                         "rounding it would end the arms at different times")
    area = np.asarray(host["mesh"].area_cell, dtype=np.float64)
    dzw = np.asarray(host["vertical"].dzw, dtype=np.float64)
    capability = require_cuda(min_compute=(12, 0))
    receipt: dict = {"schema": "gpuwm-hex.fullstep-lts-dry-arm/v1", "arm": args.arm, "dt": args.dt,
                     "steps": steps, "simulated_seconds": steps * args.dt,
                     "numerical_configuration": asdict(config), "inputs": {k: str(v) for k, v in paths.items()},
                     "physics": "off", "scalar_transport": "on" if args.transport else "off",
                     "species": species, "clamp_audit": bool(args.clamp_audit), "qualified": False}
    series: list[dict] = []
    walls: list[dict] = []
    began_all = time.perf_counter()

    def record(done_steps: int, totals: dict, health: dict) -> None:
        series.append({"step": done_steps, "seconds": done_steps * args.dt, **totals, **health})
        print(json.dumps(series[-1]), flush=True)
        if not health["finite"]:
            raise FloatingPointError(f"non-finite owned state after base step {done_steps}")

    rates = info = None
    if args.arm == "lts":
        rates, info = assign_rates(args, host)
        receipt["rate_assignment"] = info
        receipt["dispatch"] = "native single-class step" if int(rates.max()) == 1 else "rate classes"
        print(json.dumps({"dispatch": receipt["dispatch"], "rate_assignment": info}), flush=True)
    devices = parse_devices(args.devices, cp)
    if len(devices) > 1:
        # Several cards: the fixed arm (and an LTS arm whose mesh classes at
        # rate 1) is ONE class cut into blocks -- the same-host executor's
        # law with the same byte transport -- so both arms share their
        # partition method and exchange code.
        receipt["devices"] = list(devices)
        final = run_multi(args, host, config, rates if args.arm == "lts" else None, devices, receipt,
                          record, walls, area, dzw, species)
    elif args.arm == "fixed" or int(rates.max()) == 1:
        from hexcore.cuda_driver import CudaDryDycoreDriver

        cache = KernelCache(capability=capability, cache_dir=str(args.cache / "fixed"))
        driver = CudaDryDycoreDriver.from_host(
            host["mesh"], host["state"], host["vertical"], host["reference"], config,
            saved_diagnostics=host["saved"], terrain_metrics=host["terrain_metrics"],
            kernel_cache=cache, reference_wind_profiles=host["profiles"])
        if args.graphs:
            from hexcore.cuda_subcycle_graphs import attach_subcycle_graphs
            attach_subcycle_graphs(driver)
        weight = cp.asarray(dzw)[:, None] * cp.asarray(area)[None, :]
        audit = None
        if args.transport and args.clamp_audit:
            from hexcore.cuda_fullstep_lts_scalars import ClampAudit

            audit = ClampAudit(driver, weight, int(area.size),
                               KernelCache(capability=capability, cache_dir=str(args.cache / "audit")))
        record(0, totals_fixed(cp, driver, weight, args.transport), health_fixed(cp, driver, args.dt))
        block = max(1, int(args.block_steps))
        done = 0
        while done < steps:
            count = min(block, steps - done)
            cp.cuda.get_current_stream().synchronize()
            began = time.perf_counter()
            for _ in range(count):
                result = driver.step_device()
                driver.atmosphere = result.atmosphere
            cp.cuda.get_current_stream().synchronize()
            walls.append({"base_steps": count, "seconds": time.perf_counter() - began})
            done += count
            if done % args.totals_every == 0 or done == steps:
                record(done, totals_fixed(cp, driver, weight, args.transport),
                       health_fixed(cp, driver, args.dt))
        receipt["graphs"] = bool(args.graphs)
        final = {name: cp.asnumpy(getattr(driver.atmosphere.state, name)) for name in STATE}
        if args.transport:
            final["scalars"] = cp.asnumpy(driver.atmosphere.state.scalars)
        if audit is not None:
            receipt["scalar_interface"] = audit.read()
        receipt["fullstep_modules_loaded"] = sorted(k for k in sys.modules if "fullstep_lts" in k)
    else:
        from hexcore.cuda_fullstep_lts_active import FullStepLtsActive, build_class_drivers
        from hexcore.fullstep_lts_classes import build_class_plan, plan_macros

        mesh = host["mesh"]
        plan = build_class_plan(
            rates=rates, cells_on_cell=np.asarray(mesh.arrays["cellsOnCell"]),
            n_edges_on_cell=np.asarray(mesh.arrays["nEdgesOnCell"]),
            cells_on_edge=np.asarray(mesh.arrays["cellsOnEdge"]),
            cells_on_vertex=np.asarray(mesh.arrays["cellsOnVertex"]), area_cell=area,
            dc_edge_m=metric_lengths(mesh)[0], base_dt=args.dt,
            edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"),
            signal_speed=args.zone_speed, zone_factor=args.zone_factor, min_zone_rings=args.zone_min_rings)
        receipt["plan"] = plan.receipt()
        print(json.dumps({"plan": receipt["plan"]}), flush=True)
        began = time.perf_counter()
        drivers, areas = build_class_drivers(host, config, plan, base_dt=args.dt,
                                             cache_root=args.cache, capability=capability)
        cache = KernelCache(capability=capability, cache_dir=str(args.cache / "lts"))
        active = FullStepLtsActive(plan, drivers, base_dt=args.dt, kernel_cache=cache,
                                   area_cell=areas, dzw=dzw, clamp_audit=bool(args.clamp_audit),
                                   interface_flux=args.interface_flux)
        if active.scalars is not None:
            active.scalars.guard = not args.no_donor_guard
            active.scalars.cap = not args.no_reserve_cap
            active.scalars.packets = not args.no_packets
            receipt["donor_guard"], receipt["reserve_cap"] = active.scalars.guard, active.scalars.cap
            receipt["packets"] = active.scalars.packets
        active.reflux_scale = {"full": (1.0, 1.0), "mass": (1.0, 0.0), "none": (0.0, 0.0)}[args.reflux]
        active.time_classes = bool(args.time_classes)
        receipt["reflux"] = args.reflux
        receipt["interface_flux"] = args.interface_flux
        receipt["construct_seconds"] = time.perf_counter() - began
        record(0, active.totals(), active.health())
        macros = plan_macros(steps, min(max(plan.rates), args.macro_cap), warmup_steps=args.warmup_steps)
        done = 0
        next_total = args.totals_every
        for macro in macros:
            cp.cuda.get_current_stream().synchronize()
            began = time.perf_counter()
            active.advance_macro(macro, collect=args.reflux_stats)
            cp.cuda.get_current_stream().synchronize()
            walls.append({"base_steps": macro, "seconds": time.perf_counter() - began})
            done += macro
            if done >= next_total or done == steps:
                next_total = (done // args.totals_every + 1) * args.totals_every
                record(done, active.totals(), active.health())
        receipt["active"] = active.receipt()
        receipt["macros"] = {str(rate): int(macros.count(rate)) for rate in sorted(set(macros))}
        if args.reflux_stats:
            receipt["max_relative_density_correction"] = max(
                (row["max_relative_density_correction"] for row in active.reflux_stats), default=0.0)
        if not active.registers_empty():
            raise RuntimeError("an interface register is not empty at the end of the run")
        final = active.gather(STATE)
        if active.scalars is not None:
            final["scalars"] = active.scalars.gather()
            receipt["scalar_interface"] = active.scalars.stats()
        receipt["rates_sha256"] = hashlib.sha256(np.ascontiguousarray(rates).tobytes()).hexdigest()
        if args.save_state:
            np.save(args.out / "rates.npy", rates)

    receipt["wall_total_seconds"] = time.perf_counter() - began_all
    # Warm timing: drop the leading blocks that contain compilation.
    skipped = 0
    warm = list(walls)
    while warm and skipped < args.warm_skip_steps:
        skipped += warm.pop(0)["base_steps"]
    warm_steps = sum(row["base_steps"] for row in warm)
    warm_seconds = sum(row["seconds"] for row in warm)
    receipt["timing"] = {
        "blocks": walls, "warm_skip_base_steps": skipped, "warm_base_steps": warm_steps,
        "warm_seconds": warm_seconds,
        "warm_seconds_per_forecast_hour": (warm_seconds / (warm_steps * args.dt) * 3600.0) if warm_steps else None,
        "all_seconds_per_forecast_hour": sum(r["seconds"] for r in walls) / (steps * args.dt) * 3600.0 if steps else None,
        "scope": "integration wall with a stream synchronize at each block or macro end; excludes "
                 "construction, totals and health",
    }
    first, last = series[0], series[-1]
    receipt["series"] = series
    receipt["dry_mass_relative_drift"] = abs(last["dry_mass"] / first["dry_mass"] - 1.0)
    receipt["dry_mass_max_relative_excursion"] = max(abs(row["dry_mass"] / first["dry_mass"] - 1.0) for row in series)
    receipt["rho_theta_relative_drift"] = abs(last["rho_theta"] / first["rho_theta"] - 1.0)
    receipt["final_state_sha256"] = {name: hashlib.sha256(np.ascontiguousarray(final[name]).tobytes()).hexdigest()
                                     for name in final}
    if args.transport:
        count = len(first["tracer_mass"])
        clamped = receipt.get("scalar_interface", {}).get("clamped_mass")
        rows = []
        for t in range(count):
            start, end = first["tracer_mass"][t], last["tracer_mass"][t]
            row = {"species": species[t] if t < len(species) else str(t), "initial_mass": start,
                   "final_mass": end,
                   "relative_drift": (end / start - 1.0) if start else None,
                   "max_relative_excursion": max(abs(r["tracer_mass"][t] / start - 1.0) for r in series) if start else None,
                   "min": min(r["tracer_min"][t] for r in series), "max": max(r["tracer_max"][t] for r in series)}
            if clamped is not None and start:
                row["clamped_mass_relative"] = clamped[t] / start
                row["relative_drift_less_clamped"] = (end - clamped[t]) / start - 1.0
            withheld = receipt.get("scalar_interface", {}).get("withheld_fraction_of_throughput")
            if withheld is not None:
                row["interface_flux_withheld_fraction"] = withheld[t]
            rows.append(row)
        receipt["tracers"] = rows
        print(json.dumps({"tracers": rows, "scalar_interface": {
            k: v for k, v in receipt.get("scalar_interface", {}).items() if "per_class" not in k}}), flush=True)
    if args.save_state:
        np.savez(args.out / "final-state.npz", **final)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in (
        "arm", "steps", "dry_mass_relative_drift", "dry_mass_max_relative_excursion",
        "rho_theta_relative_drift", "final_state_sha256")} | {"timing": {
            k: v for k, v in receipt["timing"].items() if k != "blocks"}}), flush=True)
    return 0


def compare(args) -> int:
    """Distances between saved final states, split by rate class when a map is given."""

    reference = np.load(args.compare[0] / "final-state.npz")
    rows = {}
    for other in args.compare[1:]:
        candidate = np.load(other / "final-state.npz")
        row = {}
        for name in [n for n in reference.files if n in candidate.files]:
            a, b = reference[name].astype(np.float64), candidate[name].astype(np.float64)
            difference = b - a
            row[name] = {"identical": bool(np.array_equal(reference[name], candidate[name])),
                         "l2_relative": float(np.sqrt(np.mean(difference ** 2)) / np.sqrt(np.mean(a ** 2))),
                         "max_abs": float(np.abs(difference).max()),
                         "max_abs_reference": float(np.abs(a).max())}
        rates_path = other / "rates.npy"
        if rates_path.exists():
            rates = np.load(rates_path)
            a, b = reference["rho_theta"].astype(np.float64), candidate["rho_theta"].astype(np.float64)
            row["rho_theta_l2_relative_by_rate"] = {
                str(int(r)): float(np.sqrt(np.mean((b - a)[:, rates == r] ** 2)) / np.sqrt(np.mean(a[:, rates == r] ** 2)))
                for r in np.unique(rates)}
        rows[str(other)] = row
    text = json.dumps({"reference": str(args.compare[0]), "comparisons": rows}, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", type=Path, nargs="+", help="reference arm directory followed by others")
    for name in ("grid", "static", "init", "out", "cache"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--arm", choices=("fixed", "lts"))
    p.add_argument("--dt", type=float, default=18.0, help="base step, the finest class's whole step")
    from hexcore.mpas_defaults import DEFAULT_ACOUSTIC_SUBSTEPS
    p.add_argument("--acoustic-substeps", type=int, default=DEFAULT_ACOUSTIC_SUBSTEPS,
                   help="acoustic sub-steps per dynamics split (even; MPAS-A v8.4.2 default 4)")
    p.add_argument("--hours", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=None, help="base steps, overriding --hours")
    # RETIRED GUARD (2026-10-12): the top rate defaulted to 8 because a rate-16/8 interface
    # under a 77 to 96 m/s jet was unstable (48-3 km patch mesh: 193 m/s after 4.3 h against
    # 111 m/s at fixed step).  The cause was the reflux of the coarser class's own interface
    # flux estimate; the matched interface flux (--interface-flux matched, the default)
    # removed it (107 m/s through 6 h on the same class map), so the default is the ladder top (32; the forecast door defaults to 16, the top of the ladder its shadow mode also accepts).
    p.add_argument("--max-rate", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    p.add_argument("--forced-cap", default=None,
                   help="lat,lon,r1km[,r2km...]: rate 1 inside r1, 2 inside r2, ...; a test map for uniform meshes")
    p.add_argument("--demote-disc", action="append", default=None,
                   help="lat,lon,radius_km,rate: hold the cells of a disc to at most this rate (repeatable); "
                        "a test map for placing an island of a finer class")
    p.add_argument("--target-advection", type=float, default=0.80)
    p.add_argument("--target-acoustic", type=float, default=0.70)
    p.add_argument("--target-transport", type=float, default=0.90,
                   help="lts arm with --transport: largest initial-state donor fraction a class step may use")
    p.add_argument("--transport", action="store_true",
                   help="native scalar transport with the monotonic limiter (six species)")
    p.add_argument("--tracers", choices=("test", "real"), default="test",
                   help="test: unit/band/ramp in the three zero species; real: the initial file's species only")
    p.add_argument("--band", default=None,
                   help="lat,lon,inner_km,outer_km annulus of the band tracer (default: spacing 1.5 to 3 times the finest)")
    p.add_argument("--clamp-audit", action="store_true",
                   help="add up the mass the native third stage clamps at zero (one more kernel per step)")
    p.add_argument("--no-packets", action="store_true",
                   help="lts arm: diagnostic; the coarse side applies its own interface flux, so tracer "
                        "mass is NOT conserved (shows the instrument sees a broken interface)")
    p.add_argument("--no-donor-guard", action="store_true",
                   help="lts arm: diagnostic; a coarse interface column may overdraw what its packets left")
    p.add_argument("--no-reserve-cap", action="store_true",
                   help="lts arm: diagnostic; the finer class may move more out of a coarse column than it held")
    p.add_argument("--mixing", choices=("smagorinsky", "off"), default="smagorinsky")
    p.add_argument("--apvm", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="lts arm: leading macros of one base step (every class at the base step)")
    p.add_argument("--macro-cap", type=int, choices=(1, 2, 4, 8, 16, 32), default=32,
                   help="lts arm: longest macro; 1 steps every class at the base step (a plumbing check)")
    p.add_argument("--reflux", choices=("full", "mass", "none"), default="full",
                   help="lts arm: diagnostic; only 'full' conserves")
    p.add_argument("--interface-flux", choices=("matched", "own"), default="matched",
                   help="lts arm: 'matched' (default) makes a coarser class's acoustic continuity and heat "
                        "equations read the finer class's accepted mean momentum over each sub-step on an "
                        "interface edge; 'own' is the earlier coupling (the coarser class's own estimate, "
                        "repaired by the reflux after the step), kept as a diagnostic arm")
    p.add_argument("--zone-speed", type=float, default=400.0,
                   help="lts arm: fastest signal (m/s) a class's scratch zone of coarser cells must outrun")
    p.add_argument("--zone-factor", type=float, default=1.5,
                   help="lts arm: margin on the distance that signal covers between two resets of the zone")
    p.add_argument("--zone-min-rings", type=int, default=3)
    p.add_argument("--time-classes", action="store_true",
                   help="lts arm: wall time per class (a stream synchronize after every class step; slows the run)")
    p.add_argument("--totals-every", type=int, default=200, help="base steps between conservation samples")
    p.add_argument("--block-steps", type=int, default=16, help="fixed arm: base steps per timed block")
    p.add_argument("--warm-skip-steps", type=int, default=34, help="leading base steps excluded from warm timing")
    p.add_argument("--graphs", action="store_true", help="fixed arm: replay the dynamics subcycle as CUDA graphs")
    p.add_argument("--reflux-stats", action="store_true", help="lts arm: record the largest density correction (adds syncs)")
    p.add_argument("--save-state", action="store_true")
    p.add_argument("--devices", default=None,
                   help="cards of this host, comma separated, or 'all': both arms run cut into blocks across them")
    p.add_argument("--block-cells", type=int, default=16000,
                   help="several cards: a class is cut into one block per this many owned cells, at most one per card")
    p.add_argument("--blocks", default=None, help="several cards: explicit blocks per class, coarsest first")
    p.add_argument("--copy-policy", choices=("auto", "peer", "staged"), default="auto",
                   help="several cards: same-host transport route between the blocks of a class")
    args = p.parse_args()
    if args.compare:
        return compare(args)
    for name in ("grid", "static", "init", "out", "cache", "arm"):
        if getattr(args, name) is None:
            p.error(f"--{name} is required for a run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
