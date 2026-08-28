#!/usr/bin/env python3
"""Benchmark the resident CUDA recovery foundation on frozen x1.2562 state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable

import cupy as cp
import numpy as np

from hexcore.cuda_backend import (
    DeviceAtmosphere,
    KernelCache,
    RECOVERY_CUDA_SOURCE,
    recover_state,
    require_cuda,
)
from hexcore.driver import DryReferenceState, DrySavedDiagnostics, TerrainMetrics
from hexcore.mesh import Mesh
from hexcore.nomix_oracle import FrozenNomixOracle
from hexcore.state import PrognosticState
from hexcore.terrain import build_terrain_coupling, recover_velocities
from hexcore.vertical import VerticalGrid


ROOT = Path(__file__).resolve().parents[1]
MESH_DIRECTORY = ROOT / "data" / "meshes" / "x1.2562"
ORACLE_DIRECTORY = ROOT / "oracle" / "jw-x1.2562-v8.2.3-nomix-native"
DEFAULT_RECEIPT = ROOT / "receipts" / "cuda-backend" / "RTX5090-x1.2562-recovery.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _native_vertical(oracle: FrozenNomixOracle) -> VerticalGrid:
    zgrid = np.array(oracle.vertical("zgrid", vertical_first=True), copy=True)
    zz = np.array(oracle.vertical("zz", vertical_first=True), copy=True)
    zxu = np.array(oracle.vertical("zxu", vertical_first=True), copy=True)
    dss = np.array(oracle.vertical("dss", vertical_first=True), copy=True)
    rdzw = np.array(oracle.vertical("rdzw"), copy=True)
    dzu = np.array(oracle.vertical("dzu"), copy=True)
    rdzu = np.array(oracle.vertical("rdzu"), copy=True)
    fzm = np.array(oracle.vertical("fzm"), copy=True)
    fzp = np.array(oracle.vertical("fzp"), copy=True)
    nlev = int(zz.shape[0])
    dtype = zz.dtype
    dzw = np.reciprocal(rdzw)
    zw = np.empty(nlev + 1, dtype=dtype)
    zw[0] = dtype.type(0.0)
    zw[1:] = np.cumsum(dzw, dtype=dtype)
    zu = dtype.type(0.5) * (zw[:-1] + zw[1:])
    rdzwp = np.zeros(nlev, dtype=dtype)
    rdzwm = np.zeros(nlev, dtype=dtype)
    rdzwp[1:] = dzw[:-1] / (dzw[1:] * (dzw[1:] + dzw[:-1]))
    rdzwm[1:] = dzw[1:] / (dzw[:-1] * (dzw[1:] + dzw[:-1]))
    return VerticalGrid(
        zw=zw,
        dzw=dzw,
        rdzw=rdzw,
        zu=zu,
        dzu=dzu,
        rdzu=rdzu,
        rdzwp=rdzwp,
        rdzwm=rdzwm,
        fzp=fzp,
        fzm=fzm,
        ah=np.zeros(nlev + 1, dtype=dtype),
        hx=np.zeros_like(zgrid),
        zgrid=zgrid,
        zz=zz,
        zxu=zxu,
        dss=dss,
        cf1=float(oracle.vertical("cf1")),
        cf2=float(oracle.vertical("cf2")),
        cf3=float(oracle.vertical("cf3")),
        first_height_level=nlev + 1,
    )


def build_case() -> tuple[Any, ...]:
    oracle = FrozenNomixOracle(ORACLE_DIRECTORY)
    mesh = Mesh.from_netcdf(
        MESH_DIRECTORY / "x1.2562.grid.nc",
        MESH_DIRECTORY / "x1.2562.static.nc",
    )
    vertical = _native_vertical(oracle)
    coupling = build_terrain_coupling(
        n_edges_on_cell=mesh.nEdgesOnCell,
        edges_on_cell=mesh.edgesOnCell,
        cells_on_edge=mesh.cellsOnEdge,
        zb=np.transpose(np.asarray(oracle.vertical("zb")), (2, 1, 0)),
        zb3=np.transpose(np.asarray(oracle.vertical("zb3")), (2, 1, 0)),
        config_coef_3rd_order=0.25,
        array_layout="logical",
    )
    terrain = TerrainMetrics(
        zb_cell=np.asarray(coupling.zb_cell),
        zb3_cell=np.asarray(coupling.zb3_cell),
    )
    reference = DryReferenceState(
        rho_base=np.array(oracle.reference("rho_base"), copy=True),
        rho_theta_base=np.array(oracle.reference("rtheta_base"), copy=True),
        pressure_base=np.array(oracle.reference("pressure_base"), copy=True),
        exner_base=np.array(oracle.reference("exner_base"), copy=True),
    )
    rho = np.array(oracle.field("t0", "rho_zz"), copy=True)
    state = PrognosticState(
        rho=rho,
        rho_theta=rho * np.array(oracle.field("t0", "theta_m"), copy=True),
        rho_u=np.array(oracle.field("t0", "ru"), copy=True),
        rho_w=np.array(oracle.field("t0", "rw"), copy=True),
        scalars=np.array(oracle.field("t0", "qv"), copy=True)[None],
    )
    dtype = state.rho.dtype
    rgas = dtype.type(287.0)
    cp_air = dtype.type(1004.5)
    p0 = dtype.type(100_000.0)

    def pressure_cpu() -> dict[str, np.ndarray]:
        theta = state.rho_theta / state.rho
        exner = (
            vertical.zz * (rgas / p0) * state.rho_theta
        ) ** (rgas / (cp_air - rgas))
        pressure = vertical.zz * rgas * state.rho_theta * exner
        density_p = state.rho - reference.rho_base
        rtheta_p = state.rho_theta - reference.rho_theta_base
        pressure_p = vertical.zz * rgas * (
            exner * rtheta_p
            + reference.rho_theta_base * (exner - reference.exner_base)
        )
        return {
            "theta_m": theta,
            "exner": exner,
            "pressure": pressure,
            "density_perturbation": density_p,
            "rho_theta_perturbation": rtheta_p,
            "pressure_perturbation": pressure_p,
        }

    pressure_values = pressure_cpu()

    def velocity_cpu() -> Any:
        return recover_velocities(
            rho_zz=state.rho,
            ru=state.rho_u,
            rw=state.rho_w,
            zz=vertical.zz,
            fzm=vertical.fzm,
            fzp=vertical.fzp,
            cf1=vertical.cf1,
            cf2=vertical.cf2,
            cf3=vertical.cf3,
            coupling=coupling,
        )

    velocity_values = velocity_cpu()
    saved = DrySavedDiagnostics(
        theta_m=pressure_values["theta_m"].copy(),
        exner=pressure_values["exner"].copy(),
        density_perturbation=pressure_values["density_perturbation"].copy(),
        rho_theta_perturbation=pressure_values["rho_theta_perturbation"].copy(),
        pressure_perturbation=pressure_values["pressure_perturbation"].copy(),
        normal_velocity=velocity_values.u.copy(),
        vertical_velocity=velocity_values.w.copy(),
    )
    truth = {
        **pressure_values,
        "normal_velocity": velocity_values.u,
        "vertical_velocity": velocity_values.w,
    }
    return (
        oracle, mesh, vertical, terrain, reference, state, saved, truth,
        pressure_cpu, velocity_cpu,
    )


def time_cpu(function: Callable[[], Any], repeats: int) -> dict[str, float | int]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1_000.0)
    return {
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
        "repeats": repeats,
    }


def comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    difference = np.abs(
        np.asarray(candidate, dtype=np.float64)
        - np.asarray(reference, dtype=np.float64)
    )
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(difference)),
        "rms": float(np.sqrt(np.mean(difference * difference))),
        "finite": bool(np.all(np.isfinite(candidate))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-repeats", type=int, default=500)
    parser.add_argument("--pressure-cpu-repeats", type=int, default=100)
    parser.add_argument("--velocity-cpu-repeats", type=int, default=3)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.gpu_repeats, args.pressure_cpu_repeats, args.velocity_cpu_repeats) < 1:
        raise ValueError("benchmark repeat counts must be positive")
    capability = require_cuda(required_compute=(12, 0))
    total_started = time.perf_counter()
    (
        oracle, mesh, vertical, terrain, reference, state, saved, truth,
        pressure_cpu, velocity_cpu,
    ) = build_case()
    cpu_pressure = time_cpu(pressure_cpu, args.pressure_cpu_repeats)
    cpu_velocity = time_cpu(velocity_cpu, args.velocity_cpu_repeats)
    upload_started = time.perf_counter()
    atmosphere = DeviceAtmosphere.from_host(
        mesh, state, vertical, reference, saved, terrain
    )
    upload_wall = time.perf_counter() - upload_started
    cache = KernelCache(capability=capability)
    recovered = recover_state(
        atmosphere,
        cache=cache,
        warmup=10,
        timing_repeats=args.gpu_repeats,
    )
    download_started = time.perf_counter()
    candidate = recovered.to_host()
    download_seconds = time.perf_counter() - download_started
    comparisons = {
        name: comparison(truth[name], candidate[name]) for name in truth
    }
    timings = {name: value.as_dict() for name, value in recovered.timings.items()}
    gpu_velocity_ms = (
        recovered.timings["normal_velocity"].mean_launch_ms
        + recovered.timings["vertical_velocity"].mean_launch_ms
    )
    payload = {
        "schema": "mpas-port.cuda-backend-benchmark.v1",
        "evidence": {
            "status": "passed",
            "classification": "real RTX 5090 resident CUDA recovery benchmark",
            "claim": "device containers and fused pressure/u/w recovery executed on sm_120 and matched CPU-derived x1.2562 recovery",
            "non_claim": "this benchmark is not a complete CUDA timestep and is not a stock-Fortran equivalence result",
        },
        "device": capability.as_dict(),
        "authority": {
            "mesh_grid_sha256": sha256_file(MESH_DIRECTORY / "x1.2562.grid.nc"),
            "mesh_static_sha256": sha256_file(MESH_DIRECTORY / "x1.2562.static.nc"),
            "oracle_manifest_sha256": oracle.manifest_sha256,
            "cuda_source_sha256": sha256_text(RECOVERY_CUDA_SOURCE),
        },
        "case": {
            "nCells": int(mesh.dimensions["nCells"]),
            "nEdges": int(mesh.dimensions["nEdges"]),
            "nVertices": int(mesh.dimensions["nVertices"]),
            "nVertLevels": int(vertical.n_vert_levels),
            "dtype": str(state.rho.dtype),
            "state_device_bytes": int(
                sum(
                    getattr(atmosphere.state, name).nbytes
                    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
                )
            ),
            "total_resident_upload_bytes": atmosphere.h2d.bytes,
        },
        "transfers": {
            "h2d": {
                **atmosphere.h2d.as_dict(),
                "outer_wall_seconds": upload_wall,
                "policy": "one-time upload before repeated kernels",
            },
            "d2h_recovered": {
                "bytes": int(sum(value.nbytes for value in candidate.values())),
                "seconds": download_seconds,
                "policy": "one-time validation/download after repeated kernels",
            },
        },
        "kernels": timings,
        "cpu": {
            "vectorized_pressure_recovery": cpu_pressure,
            "scalar_terrain_u_w_recovery": cpu_velocity,
        },
        "speedup_device_resident": {
            "pressure": float(cpu_pressure["mean_ms"]) / recovered.timings["pressure"].mean_launch_ms,
            "normal_plus_terrain_vertical": float(cpu_velocity["mean_ms"]) / gpu_velocity_ms,
        },
        "correctness": comparisons,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cupy": cp.__version__,
            "total_wall_seconds": time.perf_counter() - total_started,
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    if not all(item["finite"] for item in comparisons.values()):
        raise FloatingPointError("CUDA recovery comparison contains non-finite values")
    receipt = args.receipt.expanduser().resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
