"""Pin the v8.4.1 CUDA 2-D Smagorinsky mixing against its CPU authorities.

One-shot instrument, run on the card box before the A/B arms:

1. builds the real x4 forecast host exactly as run_cuda_v841_forecast does
   (mixing lane 2d_smagorinsky), so the pinned objects are the execution
   objects;
2. EXECUTION IDENTITY: the float32 deformation weights the driver attached
   are recomputed by the CPU authority and compared by sha256 (H2D is a byte
   copy, so equality is required, not approximate);
3. INSTRUMENT VALIDATION both directions: a rigid solid-body rotation (a
   deformation-free flow) must produce kdiff at the roundoff floor, and the
   real initial state must produce a nonzero field; both bounds are printed
   as MEASURED numbers, not asserted silently;
4. MAX-ULP PIN: the CUDA kernels (smagorinsky_v841_f32 + the armored filter
   kernels) are compared element-wise against the float32 CPU authority on
   the real initial state; the float64 scaffold of the same code path is
   reported as a max-abs delta.

Writes ``v841-smagorinsky-authority-pin.json`` into --output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (str(ROOT / "src"), str(ROOT / "tools")):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def ulp_distance_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    y = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    x = np.where(x < 0, np.int64(-2147483648) - x, x)
    y = np.where(y < 0, np.int64(-2147483648) - y, y)
    return np.abs(x - y)


def compare(name: str, dev: np.ndarray, host32: np.ndarray, host64: np.ndarray) -> dict[str, Any]:
    ulp = ulp_distance_f32(dev, host32)
    worst = int(np.argmax(ulp))
    return {
        "field": name,
        "shape": list(dev.shape),
        "max_ulp_cuda_vs_cpu_f32": int(ulp.max()),
        "mean_ulp": float(ulp.mean()),
        "mismatch_count": int(np.count_nonzero(ulp)),
        "worst_flat_index": worst,
        "worst_cuda": float(dev.reshape(-1)[worst]),
        "worst_cpu_f32": float(host32.reshape(-1)[worst]),
        "max_abs_f32_vs_f64_scaffold": float(
            np.max(np.abs(host32.astype(np.float64) - host64))
        ),
        "max_abs_value_f64": float(np.max(np.abs(host64))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--arwen-checkout", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import run_cuda_v841_forecast as runner

    proof = runner.proof

    from hexcore.mixing_v841 import (
        compute_dry_mixing_tendencies_v841,
        compute_smagorinsky_coefficients_v841,
        initialize_deformation_weights_v841,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "grid": args.grid.absolute(),
        "static": args.static.absolute(),
        "init": args.init.absolute(),
    }
    authority = runner.verify_forecast_authorities(paths)
    host = runner.prepare_forecast_host(
        paths, authority, start_time_text=None, horiz_mixing="2d_smagorinsky"
    )
    # Same ordering law as execute_forecast: the Arwen checkout pin must
    # precede KernelCache's gpuwm platform-binding construction, or the
    # frozen-package guard refuses ("gpuwm was already imported from a
    # different tree").
    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    pin_arwen_physics_v841(args.arwen_checkout.absolute())
    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=args.cache_root
    )
    import cupy as cp

    cache = KernelCache(capability=capability, cache_dir=args.cache_root)
    stack = proof._construct_device_stack(
        host=host, cache=cache, arwen_checkout=args.arwen_checkout.absolute()
    )
    driver = stack["driver"]
    cfg = driver.mixing_config_v841
    assert cfg is not None, "the mixing lane must be active for this pin"
    mesh = host["prepared"].mesh
    report: dict[str, Any] = {
        "schema": "mpas-port.v841-smagorinsky-authority-pin/v1",
        "config": {
            "type": type(host["config"]).__name__,
            "config_smagorinsky_coef": float(cfg.config_smagorinsky_coef),
            "config_visc4_2dsmag": float(cfg.config_visc4_2dsmag),
            "config_del4u_div_factor": float(cfg.config_del4u_div_factor),
            "config_len_disp": float(cfg.config_len_disp),
            "config_h_ScaleWithMesh": bool(cfg.config_h_ScaleWithMesh),
        },
        "init": {"path": str(paths["init"]), **authority["files"]["init"]},
    }
    attached_receipt = dict(driver.deformation_weights_receipt_v841)

    # -- GPU phase: device diagnostics + device mixing, then release -------
    dt = float(host["config"].config_dt)
    saved = driver.atmosphere.saved
    state = driver.atmosphere.state
    diag = driver.horizontal.solve_diagnostics(
        state.rho,
        state.rho_u,
        dt=dt,
        apvm_upwinding=float(host["config"].config_apvm_upwinding),
        normal_velocity=saved.normal_velocity,
        rk_step=3,
    )
    mix_dev = driver.horizontal.compute_dry_mixing_tendencies_v841(
        saved.normal_velocity,
        diag.tangential_velocity,
        saved.vertical_velocity,
        saved.theta_m,
        diag.h_edge,
        diag.divergence,
        diag.vorticity,
        dt=dt,
        config=cfg,
    )
    cp.cuda.get_current_stream().synchronize()

    host_inputs = {
        "u": cp.asnumpy(saved.normal_velocity),
        "v": cp.asnumpy(diag.tangential_velocity),
        "w": cp.asnumpy(saved.vertical_velocity),
        "theta_m": cp.asnumpy(saved.theta_m),
        "rho_edge": cp.asnumpy(diag.h_edge),
        "divergence": cp.asnumpy(diag.divergence),
        "vorticity": cp.asnumpy(diag.vorticity),
    }
    device_outputs = {
        name: cp.asnumpy(getattr(mix_dev, name))
        for name in (
            "kdiff",
            "tend_u_euler",
            "tend_w_euler",
            "tend_theta_euler",
            "delsq_u",
            "delsq_w",
            "delsq_theta",
        )
    }
    # Release the card: everything else in this instrument is CPU-only, and
    # the chain starts the first arm on the freed card as soon as the marker
    # below exists.
    del mix_dev, diag, saved, state, driver, stack
    import gc

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    marker = args.output / "PIN-GPU-RELEASED.marker"
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"gpu released; marker {marker}", flush=True)

    # -- 2. execution identity of the deformation weights ------------------
    started = time.perf_counter()
    weights32 = initialize_deformation_weights_v841(mesh, dtype=np.float32)
    weights64 = initialize_deformation_weights_v841(mesh, dtype=np.float64)
    report["deformation_weights_seconds"] = time.perf_counter() - started
    import hashlib

    identity = {}
    for name in ("coef_c2", "coef_s2", "coef_cs"):
        recomputed = hashlib.sha256(
            np.ascontiguousarray(
                np.asarray(getattr(weights32, name), dtype=np.float32)
            ).tobytes(order="C")
        ).hexdigest()
        attached = attached_receipt[name]["sha256"]
        identity[name] = {
            "recomputed_sha256": recomputed,
            "driver_attached_sha256": attached,
            "identical": recomputed == attached,
        }
        w32 = np.asarray(getattr(weights32, name), dtype=np.float64)
        w64 = np.asarray(getattr(weights64, name))
        identity[name]["max_abs_f32_vs_f64"] = float(np.max(np.abs(w32 - w64)))
        identity[name]["max_abs_f64"] = float(np.max(np.abs(w64)))
    report["deformation_weight_identity"] = identity

    # -- 3. instrument validation ------------------------------------------
    # Rigid solid-body rotation about the pole: u_east = omega*R*cos(lat) is a
    # deformation-free rigid motion, so kdiff must sit at the roundoff floor.
    attrs = mesh.attrs
    radius = float(attrs["sphere_radius"])
    lat_edge = np.asarray(mesh.arrays["latEdge"], dtype=np.float64)
    angle_edge = np.asarray(mesh.arrays["angleEdge"], dtype=np.float64)
    omega_test = 2.0 * np.pi / 86400.0  # one rotation per day, jet-like speeds
    u_east = omega_test * radius * np.cos(lat_edge)
    normal_rot = (u_east * np.cos(angle_edge)).astype(np.float32)
    tangential_rot = (-u_east * np.sin(angle_edge)).astype(np.float32)
    nlev = int(host["prepared"].state.rho.shape[0])
    rot_u = np.repeat(normal_rot[None, :], nlev, axis=0)
    rot_v = np.repeat(tangential_rot[None, :], nlev, axis=0)
    rot = compute_smagorinsky_coefficients_v841(
        mesh, rot_u, rot_v, weights32, dt=dt, config=cfg
    )
    speed = float(np.max(np.abs(u_east)))
    report["instrument_validation"] = {
        "rigid_rotation": {
            "description": (
                "solid-body rotation (deformation-free); kdiff should be at "
                "the discretization/roundoff floor, NOT at flow scale"
            ),
            "peak_speed_m_s": speed,
            "kdiff_max": float(rot.kdiff.max()),
            "resolution_limit_note": (
                "the floor is set by curvature of the tangent-plane fit and "
                "float32 roundoff; compare kdiff_max against the real-state "
                "kdiff below -- it must be orders smaller"
            ),
        }
    }

    # -- 4. real-state max-ULP pin -----------------------------------------
    started = time.perf_counter()
    cpu32 = compute_dry_mixing_tendencies_v841(
        mesh,
        weights32,
        normal_velocity=host_inputs["u"],
        tangential_velocity=host_inputs["v"],
        vertical_velocity=host_inputs["w"],
        theta_m=host_inputs["theta_m"],
        rho_edge=host_inputs["rho_edge"],
        divergence=host_inputs["divergence"],
        vorticity=host_inputs["vorticity"],
        dt=dt,
        config=cfg,
    )
    report["cpu_f32_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    cpu64 = compute_dry_mixing_tendencies_v841(
        mesh,
        weights64,
        normal_velocity=host_inputs["u"].astype(np.float64),
        tangential_velocity=host_inputs["v"].astype(np.float64),
        vertical_velocity=host_inputs["w"].astype(np.float64),
        theta_m=host_inputs["theta_m"].astype(np.float64),
        rho_edge=host_inputs["rho_edge"].astype(np.float64),
        divergence=host_inputs["divergence"].astype(np.float64),
        vorticity=host_inputs["vorticity"].astype(np.float64),
        dt=dt,
        config=cfg,
    )
    report["cpu_f64_seconds"] = time.perf_counter() - started

    pins = []
    for name in (
        "kdiff",
        "tend_u_euler",
        "tend_w_euler",
        "tend_theta_euler",
        "delsq_u",
        "delsq_w",
        "delsq_theta",
    ):
        pins.append(
            compare(
                name,
                device_outputs[name],
                np.asarray(getattr(cpu32, name)),
                np.asarray(getattr(cpu64, name), dtype=np.float64),
            )
        )
    report["max_ulp_pins"] = pins
    kdiff_dev = device_outputs["kdiff"]
    report["real_state_kdiff"] = {
        "max": float(kdiff_dev.max()),
        "mean": float(kdiff_dev.mean()),
        "nonzero_fraction": float(np.count_nonzero(kdiff_dev) / kdiff_dev.size),
        "ceiling": float(
            (np.float32(0.01) * (np.float32(rot.config_len_disp) ** 2))
            * (np.float32(1.0) / np.float32(dt))
        ),
        "at_ceiling_count": int(
            np.count_nonzero(
                kdiff_dev
                >= (np.float32(0.01) * np.float32(rot.config_len_disp) ** 2)
                * (np.float32(1.0) / np.float32(dt))
            )
        ),
        "h_mom_eddy_visc4": float(cpu32.h_mom_eddy_visc4),
    }
    report["verdict"] = {
        "weights_identical": all(
            entry["identical"] for entry in identity.values()
        ),
        "max_ulp_overall": max(p["max_ulp_cuda_vs_cpu_f32"] for p in pins),
        "rotation_floor_ratio": (
            float(rot.kdiff.max())
            / max(float(kdiff_dev.max()), np.finfo(np.float32).tiny)
        ),
    }
    out = args.output / "v841-smagorinsky-authority-pin.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
