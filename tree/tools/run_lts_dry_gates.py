#!/usr/bin/env python
"""Local-timestep gates 2, 3 and 4, run in the dry passive-qv lane.

Why the dry lane
----------------
The conservation bound this gate enforces is 2.0e-8 on relative dry-mass and
relative qv-mass drift.  That bound is not arbitrary: it is the one
``tools/run_cuda_x1_163842_stabilized_products.py:878-885`` applies, and
``:983-984`` of the same file refuses unless the run carries no physics, with
``:1194-1199`` computing the qv budget the bound is applied to.  Water vapour in
a full-physics run has sources and sinks -- a measured 1.8e-4 drift over six
minutes on x1.40962 with WSM6 active -- so a full-physics qv budget measures the
microphysics and can say nothing about whether local time stepping conserves.
The dry lane makes qv a passively transported scalar, which is exactly the
quantity that a class-interface flux mismatch would destroy.

The three arms and what each decides
------------------------------------
``--arm off``            the pinned dry path, no local-timestep module involved.
``--arm declared-off``   the local-timestep configuration subtype with the
                         switch off.  GATE 2: this must be bit-identical to
                         ``off`` on a REFINED mesh, which is what proves the
                         option did not leak into the default path.  A uniform
                         mesh cannot decide it, because there the option is
                         inert whether or not it leaked.
``--arm on``             the switch on.  GATE 3 measures its mass and qv drift.
``--arm referee``        GATE 4: the pinned dry path at a globally smaller
                         acoustic step (``--dt`` divided, step count multiplied,
                         same physical duration).  Both ``off`` and ``on`` are
                         then scored against it, which is the only comparison
                         that separates local-timestep error from the truncation
                         error the default path already carries.  Obs-skill
                         cannot do this: a class-interface reflection can score
                         well against MRMS while being categorically wrong.

Every arm writes its final state to ``.npz`` plus a JSON receipt, so identity,
conservation and the referee distances are decided afterwards by
``--compare`` over those files rather than inside a run.

Instrument validation
---------------------
``--selftest`` checks the mass functional in both directions on synthetic
arrays: an unchanged state must report exactly zero drift, and a state with one
cell's density scaled must report the drift that scaling implies.  A functional
that cannot see a planted change cannot clear a conservation gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "tools"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")


# --------------------------------------------------------------------------
# the conserved quantities
# --------------------------------------------------------------------------
def dry_mass(area: np.ndarray, dzw: np.ndarray, rho: np.ndarray) -> float:
    """Total dry mass, the ``mass`` field of ``driver._state_metrics``."""

    weight = np.asarray(dzw, dtype=np.float64)[:, None] * np.asarray(
        area, dtype=np.float64
    )[None, :]
    return float(np.sum(np.asarray(rho, dtype=np.float64) * weight, dtype=np.float64))


def qv_mass(
    area: np.ndarray, dzw: np.ndarray, rho: np.ndarray, qv: np.ndarray
) -> float:
    """Total water-vapour mass, ``_qv_mass`` of the stabilized-products tool."""

    weight = np.asarray(dzw, dtype=np.float64)[:, None] * np.asarray(
        area, dtype=np.float64
    )[None, :]
    return float(
        np.sum(
            np.asarray(rho, dtype=np.float64)
            * np.asarray(qv, dtype=np.float64)
            * weight,
            dtype=np.float64,
        )
    )


def relative_drift(after: float, before: float) -> float:
    return abs(after - before) / max(abs(before), float(np.finfo(np.float64).tiny))


def selftest() -> None:
    rng = np.random.default_rng(20260821)
    area = rng.uniform(1e8, 2e8, size=97)
    dzw = rng.uniform(20.0, 900.0, size=11)
    rho = rng.uniform(0.2, 1.3, size=(11, 97))
    qv = rng.uniform(1e-6, 2e-2, size=(11, 97))

    base = dry_mass(area, dzw, rho)
    assert relative_drift(base, base) == 0.0, "an unchanged state must drift zero"

    # Plant a change of a known size and require the functional to see it.
    moved = rho.copy()
    cell_mass = float(np.sum(rho[:, 3] * dzw * area[3], dtype=np.float64))
    moved[:, 3] *= 2.0
    expected = cell_mass / base
    seen = relative_drift(dry_mass(area, dzw, moved), base)
    assert abs(seen - expected) < 1e-12, f"planted {expected}, saw {seen}"

    qbase = qv_mass(area, dzw, rho, qv)
    qmoved = qv.copy()
    qmoved *= 1.0 + 1e-9
    qseen = relative_drift(qv_mass(area, dzw, rho, qmoved), qbase)
    assert abs(qseen - 1e-9) < 1e-15, f"planted 1e-9, saw {qseen}"

    # And in the other direction: a drift below the gate must not trip it.
    assert relative_drift(base * (1.0 + 1.0e-9), base) <= 2.0e-8
    assert relative_drift(base * (1.0 + 1.0e-7), base) > 2.0e-8
    print("selftest 6/6 PASS: the mass functionals see a planted change and "
          "the threshold cuts on the right side", flush=True)


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------
def build_config(
    arm: str,
    *,
    dt: float,
    rates: tuple[int, ...],
    rings: int,
    sub_steps: int = 6,
) -> Any:
    from mpas_port.config_lts import V841LocalTimestepDryConfig
    from mpas_port.config_v841 import V841DryDycoreConfig

    # The CUDA v8.4.1 lane mirrors the native split-three ruler and refuses
    # anything else (cuda_driver.py:2548-2561), while the dry dataclass default
    # is one.  Every arm therefore declares split three explicitly, and the
    # referee moves ONLY config_dt so the schedule shape is held fixed and the
    # single difference between it and the default arm is the step size.
    # These are the released column-physics lane's DYNAMICS knobs, copied term
    # for term from ``config_v841.V841MpasColumnPhysicsConfig.validate``'s
    # exact table, so the dry arms run the shipped dycore with the physics
    # removed rather than some other dry configuration.  Only config_dt moves,
    # and only for the referee.
    shared = {
        "config_dt": float(dt),
        "config_time_integration_order": 3,
        "config_number_of_sub_steps": int(sub_steps),
        "config_dynamics_split_steps": 3,
        "config_split_dynamics_transport": True,
        "config_scalar_advection": True,
        "config_monotonic": False,
        "config_positive_definite": False,
        "config_apvm_upwinding": 0.5,
        "config_xnutr": 0.2,
        "config_zd": 22_000.0,
        "config_coef_3rd_order": 0.25,
        "config_smdiv": 0.1,
        "config_divergence_damping": True,
        "config_terrain_following": True,
        "config_horiz_mixing": "off",
    }
    if arm in ("off", "referee"):
        return V841DryDycoreConfig(**shared)
    if arm == "declared-off":
        return V841LocalTimestepDryConfig(
            **shared,
            config_local_timestep=False,
            config_local_timestep_rates=tuple(rates),
            config_local_timestep_buffer_rings=int(rings),
        )
    if arm == "on":
        return V841LocalTimestepDryConfig(
            **shared,
            config_local_timestep=True,
            config_local_timestep_rates=tuple(rates),
            config_local_timestep_buffer_rings=int(rings),
        )
    raise ValueError(f"unknown arm {arm!r}")


def prepare_host(
    paths: dict[str, Path], config: Any, *, scalars: str = "wsm6"
) -> dict[str, Any]:
    """Load mesh, vertical grid and initial state exactly as the door does.

    ``scalars`` selects the transported set.  ``wsm6`` carries the shipped six
    (qv/qc/qr plus exact +0 qi/qs/qg), which is the state a real forecast moves.
    ``qv`` carries water vapour alone, which is the lane the 2.0e-8 bound is
    defined in -- and it is also the only fair lane to TIME the option in,
    because scalar transport is nine full-domain passes per model step that
    local time stepping does not touch, and five extra species of it dilute the
    measurement of a feature that only changes the acoustic loop.
    """

    import run_cuda_v841_forecast as door
    import run_cuda_v841_full_physics_x4 as proof
    from mpas_port.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from mpas_port.dynamics_v841 import load_v841_reference_wind_profiles
    from mpas_port.mesh import load_precision_preserving_mesh_pair

    # The proof module pins the digests of the ONE init file it was sealed
    # against.  The door already owns the rebinding for a different case, and
    # reusing it keeps this instrument honest: the relaxation is recorded, not
    # silently skipped.
    relaxation = {"init_carriers": door.relax_init_carrier_pins(paths["init"])}

    mesh, output_mesh, mesh_evidence = load_precision_preserving_mesh_pair(
        paths["grid"], paths["static"]
    )
    del output_mesh
    proof.overlay_exact_init_reconstruction_coefficients(mesh, paths["init"])
    proof.overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    proof.attach_inactive_zero_deformation(mesh)
    native = load_mpas_vertical_grid(
        paths["init"], mesh, config_coef_3rd_order=config.config_coef_3rd_order
    )
    scalar_names = ("qv",) if scalars == "qv" else proof.SOURCE_SCALAR_NAMES
    state, reference, saved = load_mpas_initial_state(
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=scalar_names,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    relaxation["negative_qv"] = door.relax_negative_qv_pin(state)
    if scalars != "qv":
        proof.augment_exact_wsm6_scalars(state)
    relaxation["transported_scalars"] = list(scalar_names)
    n_levels = int(native.vertical_grid.n_vert_levels)
    profiles = load_v841_reference_wind_profiles(paths["init"], n_vert_levels=n_levels)
    return {
        "mesh": mesh,
        "mesh_evidence": mesh_evidence,
        "vertical": native.vertical_grid,
        "terrain_metrics": native.terrain_metrics,
        "state": state,
        "reference": reference,
        "saved": saved,
        "profiles": profiles,
        "n_levels": n_levels,
        "relaxation": relaxation,
    }


def _mesh_field(mesh: Any, name: str) -> np.ndarray:
    value = getattr(mesh, name, None)
    if value is None and hasattr(mesh, "get"):
        value = mesh.get(name)
    if value is None:
        raise KeyError(f"mesh has no {name}")
    return np.asarray(value)


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    from mpas_port.cuda_backend import require_cuda
    from mpas_port.cuda_backend.runtime import KernelCache
    from mpas_port.cuda_driver import CudaDryDycoreDriver
    from mpas_port.cuda_driver_lts import attach_local_timestep

    paths = {
        key: Path(getattr(args, key)).expanduser().resolve(strict=True)
        for key in ("grid", "static", "init")
    }
    dt = float(args.dt)
    steps = int(args.steps)
    if args.arm == "referee":
        factor = int(args.referee_factor)
        if factor < 2:
            raise SystemExit(
                "refusing: the referee arm exists to run a SMALLER global step; "
                "--referee-factor must be at least 2"
            )
        dt = dt / factor
        steps = steps * factor

    config = build_config(
        args.arm,
        dt=dt,
        rates=tuple(args.rates),
        rings=int(args.buffer_rings),
        sub_steps=int(args.sub_steps),
    )
    config.validate()
    host = prepare_host(paths, config, scalars=args.scalars)

    area = _mesh_field(host["mesh"], "areaCell")
    dzw = np.asarray(host["vertical"].dzw)
    initial = host["state"]
    initial_mass = dry_mass(area, dzw, initial.rho)
    initial_qv = qv_mass(area, dzw, initial.rho, np.asarray(initial.scalars)[0])

    capability = require_cuda(min_compute=(12, 0))
    cache = KernelCache(capability=capability, cache_dir=str(args.cache_root))
    driver = CudaDryDycoreDriver.from_host(
        host["mesh"],
        host["state"],
        host["vertical"],
        host["reference"],
        config,
        saved_diagnostics=host["saved"],
        terrain_metrics=host["terrain_metrics"],
        kernel_cache=cache,
        reference_wind_profiles=host["profiles"],
    )
    attachment = attach_local_timestep(driver, grid_path=str(paths["grid"]))
    if args.arm == "on" and attachment is None:
        raise SystemExit(
            "refusing: the ON arm did not attach local time stepping, so its "
            "delta against the OFF arm would be exactly zero for the wrong "
            "reason"
        )
    if args.arm != "on" and attachment is not None:
        raise SystemExit(
            f"refusing: the {args.arm} arm attached local time stepping"
        )

    import cupy as cp

    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        result = driver.step_device()
        driver.atmosphere = result.atmosphere
    cp.cuda.get_current_stream().synchronize()
    wall = time.perf_counter() - started

    final = driver.atmosphere.state.to_host()
    final_mass = dry_mass(area, dzw, final.rho)
    final_qv = qv_mass(area, dzw, final.rho, np.asarray(final.scalars)[0])

    out = Path(args.out).expanduser().absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(final, name)) for name in STATE_FIELDS}
    npz_path = out.with_suffix(".npz")
    np.savez(npz_path, **arrays)
    digests = {
        name: hashlib.sha256(
            np.ascontiguousarray(value).tobytes()
        ).hexdigest()
        for name, value in arrays.items()
    }

    receipt: dict[str, Any] = {
        "arm": args.arm,
        "grid": str(paths["grid"]),
        "init": str(paths["init"]),
        "config_type": type(config).__name__,
        "config_dt": float(config.config_dt),
        "config_number_of_sub_steps": int(config.config_number_of_sub_steps),
        "transported_scalars": host["relaxation"]["transported_scalars"],
        "config_dynamics_split_steps": int(config.config_dynamics_split_steps),
        "steps": steps,
        "model_seconds": float(dt * steps),
        "n_cells": int(np.asarray(final.rho).shape[1]),
        "n_levels": int(np.asarray(final.rho).shape[0]),
        "wall_seconds": wall,
        "seconds_per_step": wall / max(steps, 1),
        "dry_mass_initial": initial_mass,
        "dry_mass_final": final_mass,
        "dry_mass_relative_drift": relative_drift(final_mass, initial_mass),
        "qv_mass_initial": initial_qv,
        "qv_mass_final": final_qv,
        "qv_mass_relative_drift": relative_drift(final_qv, initial_qv),
        "conservation_threshold": 2.0e-8,
        "state_sha256": digests,
        "state_npz": str(npz_path),
        "local_timestep": (
            None if attachment is None else attachment.receipt()
        ),
        "all_finite": bool(
            all(np.all(np.isfinite(value)) for value in arrays.values())
        ),
    }
    if attachment is not None:
        attachment.detach()
    out.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps(
        {
            k: v for k, v in receipt.items()
            if k not in ("state_sha256", "local_timestep")
        },
        indent=1,
    ), flush=True)
    return receipt


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fields(receipt: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(receipt["state_npz"]) as data:
        return {name: np.asarray(data[name]) for name in STATE_FIELDS}


def _distance(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    diff = np.abs(x - y)
    scale = np.sqrt(np.mean(y ** 2)) if y.size else 0.0
    return {
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0,
        "rms_relative_to_reference": (
            float(np.sqrt(np.mean(diff ** 2)) / scale) if scale > 0 else 0.0
        ),
        "n_differing": int((diff != 0.0).sum()),
        "n": int(diff.size),
    }


def _interface_cell_mask(grid: str, rates: tuple[int, ...], rings: int) -> np.ndarray:
    """Cells that touch a rate-class boundary, from the grid file's own dcEdge."""

    from mpas_port.lts_v841 import classify_from_grid_file

    classing = classify_from_grid_file(grid, rates=rates, buffer_rings=rings)
    from netCDF4 import Dataset

    with Dataset(grid, "r") as dataset:
        coe = np.asarray(dataset.variables["cellsOnEdge"][:]).astype(np.int64) - 1
    mask = np.zeros(classing.n_cells, dtype=bool)
    if classing.interface_edges.size:
        edges = classing.interface_edges.astype(np.int64)
        mask[coe[edges, 0]] = True
        mask[coe[edges, 1]] = True
    return mask


def _locality(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    """Is the error concentrated on the class boundary, or spread over the mesh?

    A class-interface reflection is a LOCAL artifact.  It can leave a whole-mesh
    RMS looking healthy and still be categorically wrong, which is exactly why
    obs-skill cannot referee this feature.  The number that matters is the
    ratio of interface-cell error to interior-cell error: the default arm's own
    ratio is the control, because a mesh's refinement boundary is a busy place
    even without local time stepping.
    """

    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.ndim == 3:  # scalars: (n_scalars, nlev, ncells)
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
    if x.ndim != 2 or x.shape[-1] != mask.size:
        return {}
    diff = (x - y) ** 2
    inside = float(np.sqrt(diff[:, mask].mean())) if mask.any() else 0.0
    outside = float(np.sqrt(diff[:, ~mask].mean())) if (~mask).any() else 0.0
    return {
        "rms_on_interface_cells": inside,
        "rms_on_interior_cells": outside,
        "interface_over_interior": (inside / outside) if outside > 0 else float("inf"),
        "n_interface_cells": int(mask.sum()),
    }


def compare(args: argparse.Namespace) -> int:
    report: dict[str, Any] = {"threshold": 2.0e-8}
    failures: list[str] = []

    arms = {}
    for label in ("off", "declared_off", "on", "referee", "on_repeat"):
        path = getattr(args, label, None)
        if path:
            arms[label.replace("_", "-")] = _load(path)
    report["arms"] = {
        label: {
            key: receipt[key]
            for key in (
                "arm", "config_type", "config_dt", "steps", "model_seconds",
                "wall_seconds", "seconds_per_step",
                "dry_mass_relative_drift", "qv_mass_relative_drift",
                "all_finite",
            )
        }
        for label, receipt in arms.items()
    }

    # GATE 2 -- the option is inert when the switch is off.
    if "off" in arms and "declared-off" in arms:
        identical = arms["off"]["state_sha256"] == arms["declared-off"]["state_sha256"]
        report["gate2_default_off_bit_identical"] = bool(identical)
        report["gate2_verdict"] = "PASS" if identical else "FAIL"
        if not identical:
            failures.append("gate2")
            report["gate2_field_distance"] = {
                name: _distance(a, b)
                for (name, a), b in zip(
                    _fields(arms["off"]).items(), _fields(arms["declared-off"]).values()
                )
            }

    # GATE 3 -- conservation, a hard blocker.
    if "on" in arms:
        on = arms["on"]
        mass_ok = abs(on["dry_mass_relative_drift"]) <= 2.0e-8
        qv_ok = abs(on["qv_mass_relative_drift"]) <= 2.0e-8
        report["gate3_conservation"] = {
            "dry_mass_relative_drift": on["dry_mass_relative_drift"],
            "qv_mass_relative_drift": on["qv_mass_relative_drift"],
            "dry_mass_pass": bool(mass_ok),
            "qv_mass_pass": bool(qv_ok),
        }
        if "off" in arms:
            report["gate3_conservation"]["off_arm_dry_mass_relative_drift"] = arms[
                "off"
            ]["dry_mass_relative_drift"]
            report["gate3_conservation"]["off_arm_qv_mass_relative_drift"] = arms[
                "off"
            ]["qv_mass_relative_drift"]
        report["gate3_verdict"] = "PASS" if (mass_ok and qv_ok) else "FAIL"
        if not (mass_ok and qv_ok):
            failures.append("gate3")

    # GATE 4 -- distance to a globally smaller step, for both arms.
    if "referee" in arms and "off" in arms:
        reference = _fields(arms["referee"])
        report["gate4_referee"] = {
            "referee_dt": arms["referee"]["config_dt"],
            "referee_steps": arms["referee"]["steps"],
            "default_dt": arms["off"]["config_dt"],
        }
        for label in ("off", "on"):
            if label not in arms:
                continue
            other = _fields(arms[label])
            report["gate4_referee"][f"{label}_vs_referee"] = {
                name: _distance(other[name], reference[name]) for name in STATE_FIELDS
            }
        if "on" in arms:
            ratios = {}
            for name in STATE_FIELDS:
                base = report["gate4_referee"]["off_vs_referee"][name][
                    "rms_relative_to_reference"
                ]
                lts = report["gate4_referee"]["on_vs_referee"][name][
                    "rms_relative_to_reference"
                ]
                ratios[name] = (lts / base) if base > 0 else float("inf")
            report["gate4_referee"]["lts_error_over_default_error"] = ratios
            if args.grid:
                mask = _interface_cell_mask(
                    args.grid,
                    tuple(int(v) for v in str(args.rates).split(",") if v),
                    int(args.buffer_rings),
                )
                other = _fields(arms["on"])
                default = _fields(arms["off"])
                locality = {}
                for name in ("rho", "rho_theta", "scalars"):
                    lts_local = _locality(other[name], reference[name], mask)
                    base_local = _locality(default[name], reference[name], mask)
                    if not lts_local or not base_local:
                        continue
                    locality[name] = {
                        "lts": lts_local,
                        "default": base_local,
                        "lts_interface_ratio_over_default_interface_ratio": (
                            lts_local["interface_over_interior"]
                            / base_local["interface_over_interior"]
                            if base_local["interface_over_interior"] > 0
                            else float("inf")
                        ),
                    }
                report["gate4_referee"]["interface_locality"] = locality
            worst = max(ratios.values())
            report["gate4_referee"]["worst_ratio"] = worst
            report["gate4_referee"]["budget"] = float(args.gate4_budget)
            ok = worst <= float(args.gate4_budget)
            report["gate4_verdict"] = "PASS" if ok else "FAIL"
            if not ok:
                failures.append("gate4")

    # Repeatability: the settle kernel uses atomicAdd on shared coarse cells.
    if "on" in arms and "on-repeat" in arms:
        same = arms["on"]["state_sha256"] == arms["on-repeat"]["state_sha256"]
        report["lts_run_to_run_bit_identical"] = bool(same)

    # Measured speedup, wall clock, same mesh, same steps.
    if "on" in arms and "off" in arms:
        report["measured_speedup_x"] = (
            arms["off"]["wall_seconds"] / arms["on"]["wall_seconds"]
        )

    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    text = json.dumps(report, indent=1)
    print(text, flush=True)
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--arm", required=True,
                     choices=("off", "declared-off", "on", "referee"))
    run.add_argument("--grid", required=True)
    run.add_argument("--static", required=True)
    run.add_argument("--init", required=True)
    run.add_argument("--dt", type=float, default=30.0)
    run.add_argument("--steps", type=int, default=60)
    run.add_argument("--referee-factor", type=int, default=4)
    run.add_argument(
        "--scalars", choices=("wsm6", "qv"), default="wsm6",
        help=(
            "transported scalar set. 'qv' is the passive-water-vapour lane the "
            "2.0e-8 conservation bound is defined in, and the only fair lane "
            "to time the option in: scalar transport is nine full-domain "
            "passes per model step that local time stepping does not touch"
        ),
    )
    run.add_argument(
        "--sub-steps", type=int, default=6,
        help=(
            "acoustic sub-steps per dynamics subcycle. DIAGNOSTIC ONLY: "
            "raising it on the OFF arm measures how much of a model step "
            "the acoustic loop actually costs, which is the ceiling on "
            "what local time stepping can ever remove. The option itself "
            "refuses anything but the released 6"
        ),
    )
    run.add_argument("--rates", default="1,3")
    run.add_argument("--buffer-rings", type=int, default=1)
    run.add_argument("--cache-root", required=True)
    run.add_argument("--out", required=True)

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--off")
    cmp_.add_argument("--declared-off", dest="declared_off")
    cmp_.add_argument("--on")
    cmp_.add_argument("--on-repeat", dest="on_repeat")
    cmp_.add_argument("--referee")
    cmp_.add_argument("--gate4-budget", type=float, default=2.0)
    cmp_.add_argument("--grid", help="grid file, for the interface-locality check")
    cmp_.add_argument("--rates", default="1,3")
    cmp_.add_argument("--buffer-rings", type=int, default=1)
    cmp_.add_argument("--json")

    sub.add_parser("selftest")

    args = parser.parse_args(argv)
    if args.command == "selftest":
        selftest()
        return 0
    selftest()
    if args.command == "compare":
        return compare(args)
    args.rates = tuple(int(piece) for piece in str(args.rates).split(",") if piece)
    run_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
