#!/usr/bin/env python3
"""The regional dry byte ladder: port CPU steps against native history frames.

Runs the v8.4.1 CPU whole-step driver on a native-culled regional case
(mesh + init + lbc series, config_apply_lbcs=true, the CANDIDATE-REGIONAL-DRY
namelist) and compares, at every native history frame, the port's state and
output diagnostics BITWISE against the reference executable's payloads.
First divergence is localized to a field, level and element with both values
printed — the debugging instrument the regional transcription is pinned by.

Compared payloads per frame (native names):
``u, w, theta, rho, qv, pressure, divergence, vorticity, ke``.
``theta``/``rho``/``pressure`` are the atm_compute_output_diagnostics
transcriptions (mpas_atm_core.F:916-965); the solve diagnostics are the diag
pool the last RK3 stage leaves for the writer.

Usage (x1-cull per-step ladder, all frames present):

    python tools/run_regional_dry_ladder.py --arm x1 --steps 15

The x4 arm points the same instrument at the 44,770-cell reference
(ladder-x4 frames or the hourly run-e endpoints via --frames-dir).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from netCDF4 import Dataset  # noqa: E402

from hexcore.config_v841 import V841DryDycoreConfig  # noqa: E402
from hexcore.driver import (  # noqa: E402
    DryDycoreDriver,
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from hexcore.dynamics_v841 import load_v841_reference_wind_profiles  # noqa: E402
from hexcore.mesh import Mesh  # noqa: E402
from hexcore.regional_v841 import RVORD_F32, RegionalRuntime  # noqa: E402

_XTIME = "%Y-%m-%d_%H:%M:%S"


def dry_reference_config() -> V841DryDycoreConfig:
    """The CANDIDATE-REGIONAL-DRY namelist, as the port admits it.

    Byte-for-byte the run-e/run-f/run-g/run-h namelist
    (sha256 e623fa93f893115a1072a5ac1e1300641fbefa011803c9ffe12ee6efdf37eea4):
    dt 120, split 3/6, third-order non-monotonic transport, the v8.4.1
    release-default dissipation block, variable epssm profile, and
    config_apply_lbcs=true with physics suite none.
    """

    return V841DryDycoreConfig(
        config_dt=120.0,
        config_split_dynamics_transport=True,
        config_dynamics_split_steps=3,
        config_number_of_sub_steps=6,
        config_scalar_advection=True,
        config_monotonic=False,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_horiz_mixing="2d_smagorinsky",
        config_len_disp=25000.0,
        config_visc4_2dsmag=0.05,
        config_smagorinsky_coef=0.125,
        config_del4u_div_factor=10.0,
        config_h_ScaleWithMesh=True,
        config_smdiv=0.1,
        config_divergence_damping=True,
        config_xnutr=0.2,
        config_zd=22_000.0,
        config_apply_lbcs=True,
        config_moist_physics=True,
    )


def read_frame(path: Path) -> dict[str, np.ndarray]:
    fields = {}
    with Dataset(path) as dataset:
        dataset.set_auto_maskandscale(False)
        for name in (
            "u", "w", "theta", "rho", "qv", "pressure",
            "divergence", "vorticity", "ke",
        ):
            if name not in dataset.variables:
                continue
            slab = np.array(dataset.variables[name][0][:], dtype=np.float32)
            fields[name] = np.ascontiguousarray(slab.T)
        raw = np.asarray(dataset.variables["xtime"][0])
        text = (
            raw.tobytes().decode("utf-8", "replace")
            if raw.dtype.kind in ("S", "U")
            else bytes(raw).decode()
        )
        fields["_xtime"] = text.strip()[:19]
    return fields


def compare(name: str, port: np.ndarray, native: np.ndarray) -> dict:
    port = np.asarray(port, dtype=np.float32)
    native = np.asarray(native, dtype=np.float32)
    if port.shape != native.shape:
        return {
            "field": name,
            "bitwise_equal": False,
            "reason": f"shape {port.shape} != {native.shape}",
        }
    same = port.view(np.uint32) == native.view(np.uint32)
    if bool(same.all()):
        return {"field": name, "bitwise_equal": True, "values": int(port.size)}
    bad = ~same
    count = int(bad.sum())
    diff = np.abs(port.astype(np.float64) - native.astype(np.float64))
    diff[~np.isfinite(diff)] = np.inf
    first = tuple(int(x) for x in np.argwhere(bad)[0])
    worst = tuple(int(x) for x in np.unravel_index(np.argmax(np.where(bad, diff, -1.0)), diff.shape))
    return {
        "field": name,
        "bitwise_equal": False,
        "mismatch_count": count,
        "values": int(port.size),
        "max_abs_diff": float(np.max(diff[bad])),
        "first_mismatch": {
            "index": first,
            "port": float(port[first]),
            "native": float(native[first]),
        },
        "worst_mismatch": {
            "index": worst,
            "port": float(port[worst]),
            "native": float(native[worst]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--arm", choices=("x1", "x4"), default="x1")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--dump-dir",
        default=None,
        help="write each compared step's port fields as <dump-dir>/step<N>.npz",
    )
    parser.add_argument(
        "--fields",
        default="u,w,theta,rho,qv,pressure,divergence,vorticity,ke",
    )
    args = parser.parse_args()

    import os

    reference = Path(
        args.reference_dir
        or os.environ.get("GPUWM_HEX_REGIONAL_REFERENCE_DIR", "")
    )
    if not reference.is_dir():
        raise SystemExit(
            "reference dir not found; pass --reference-dir or set "
            "GPUWM_HEX_REGIONAL_REFERENCE_DIR"
        )
    if args.arm == "x1":
        grid = reference / "cull-x1" / "conus.grid.nc"
        init = reference / "init-x1" / "conus.init.nc"
        lbc_dir = reference / "lbc-x1"
        frames_dir = Path(args.frames_dir or (reference / "ladder-x1"))
    else:
        grid = reference / "cull-x4" / "grid" / "conus.region.nc"
        init = reference / "init" / "conus.init.nc"
        lbc_dir = reference / "lbc"
        frames_dir = Path(args.frames_dir or (reference / "ladder-x4"))
    frames = sorted(frames_dir.glob("history.*.nc"))
    if not frames:
        raise SystemExit(f"no history frames under {frames_dir}")
    lbc_paths = sorted(str(p) for p in lbc_dir.glob("lbc.*.nc"))

    compared_fields = tuple(
        f.strip() for f in args.fields.split(",") if f.strip()
    )

    t0 = time.perf_counter()
    mesh = Mesh.from_netcdf(grid, init, validate=False)
    vertical = load_mpas_vertical_grid(
        init, mesh, allow_regional_sentinels=True
    )
    state, reference_state, saved = load_mpas_initial_state(
        init,
        mesh,
        vertical.vertical_grid,
        scalar_names=("qv",),
        terrain_metrics=vertical.terrain_metrics,
        allow_regional_sentinels=True,
        return_saved_diagnostics=True,
    )
    frame0 = read_frame(frames[0])
    start_time = datetime.strptime(frame0["_xtime"], _XTIME)
    config = dry_reference_config()
    runtime = RegionalRuntime(
        mesh,
        dtype=np.dtype(np.float32),
        lbc_paths=lbc_paths,
        start_time=start_time,
        config_h_scale_with_mesh=bool(config.config_h_ScaleWithMesh),
        zz=vertical.vertical_grid.zz,
        scalar_names=("lbc_qv",),
        moist_indices=(0,),
    )
    driver = DryDycoreDriver(
        mesh,
        vertical.vertical_grid,
        reference_state,
        config,
        terrain_metrics=vertical.terrain_metrics,
        reference_wind_profiles=load_v841_reference_wind_profiles(init),
        regional=runtime,
        index_qv=0,
    )
    setup_wall = time.perf_counter() - t0

    zz32 = np.asarray(vertical.vertical_grid.zz, dtype=np.float32)
    one = np.float32(1.0)

    def port_fields(current_state, current_saved) -> dict[str, np.ndarray]:
        qv = current_state.scalars[0]
        out = {
            "u": current_saved.normal_velocity,
            "w": current_saved.vertical_velocity,
            "qv": qv,
            # atm_compute_output_diagnostics F:958-961:
            "theta": current_saved.theta_m / (one + RVORD_F32 * qv),
            "rho": current_state.rho * zz32,
            "pressure": (
                reference_state.pressure_base
                + current_saved.pressure_perturbation
            ),
        }
        if any(
            name in compared_fields
            for name in ("divergence", "vorticity", "ke")
        ):
            diagnostics = driver._diagnostics(
                current_state,
                outer_dt=config.config_dt,
                cached_v=None,
                rk_step=3,
                normal_velocity=current_saved.normal_velocity,
            )
            out["divergence"] = diagnostics.divergence
            out["vorticity"] = diagnostics.vorticity
            out["ke"] = diagnostics.kinetic_energy
        return out

    report = {
        "instrument": "run_regional_dry_ladder",
        "arm": args.arm,
        "grid": str(grid),
        "init": str(init),
        "frames_dir": str(frames_dir),
        "lbc_paths": lbc_paths,
        "start_time": frame0["_xtime"],
        "setup_wall_seconds": round(setup_wall, 2),
        "compared_fields": list(compared_fields),
        "frames": [],
    }

    def check(step_index: int, frame_path: Path, fields_port) -> bool:
        native = read_frame(frame_path)
        if args.dump_dir:
            dump_dir = Path(args.dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dump_dir / f"step{step_index}.npz",
                **{k: v for k, v in fields_port.items()},
                **{"native_" + k: v for k, v in native.items() if k != "_xtime"},
            )
        rows = []
        clean = True
        for name in compared_fields:
            if name not in native:
                rows.append({"field": name, "skipped": "absent in frame"})
                continue
            row = compare(name, fields_port[name], native[name])
            rows.append(row)
            clean = clean and bool(row.get("bitwise_equal"))
        report["frames"].append(
            {
                "step": step_index,
                "frame": frame_path.name,
                "xtime": native["_xtime"],
                "all_bitwise_equal": clean,
                "fields": rows,
            }
        )
        flag = "BITWISE" if clean else "DIVERGED"
        print(f"[{time.strftime('%H:%M:%S')}] step {step_index} "
              f"{frame_path.name}: {flag}", flush=True)
        if not clean:
            for row in rows:
                if row.get("bitwise_equal") is False:
                    print("   ", json.dumps(row), flush=True)
        return clean

    ok = check(0, frames[0], port_fields(state, saved))
    total = args.steps if args.steps is not None else len(frames) - 1
    step_walls = []
    if ok or args.keep_going:
        for step in range(1, total + 1):
            t1 = time.perf_counter()
            result = driver.step(state, saved_diagnostics=saved)
            state = result.state
            saved = result.saved_diagnostics
            step_walls.append(round(time.perf_counter() - t1, 2))
            if step < len(frames):
                ok = check(step, frames[step], port_fields(state, saved))
                if not ok and not args.keep_going:
                    break
    report["step_wall_seconds"] = step_walls
    out_path = Path(
        args.out
        or ROOT
        / "evidence"
        / "regional-cpu-l4-20260825"
        / f"ladder-{args.arm}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("report:", out_path, flush=True)
    all_clean = all(f["all_bitwise_equal"] for f in report["frames"])
    print("ladder verdict:", "BITWISE" if all_clean else "DIVERGED", flush=True)
    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
