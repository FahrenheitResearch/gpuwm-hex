#!/usr/bin/env python3
"""Advance the native JW state with the Python dry dycore for a real day.

This is a repeated full-RK/acoustic port execution.  It is not a stock-MPAS
wrapper, a quiescent fixed point, or a two-giant-step scheduling smoke test.
The receipt claims only the stability actually observed; whole-step numerical
equivalence remains the job of the separate frozen native-state gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import time
import traceback
from typing import Any, Mapping

import numpy as np

from hexcore.driver import (
    DryDycoreConfig,
    DryDycoreDriver,
    DrySavedDiagnostics,
    StabilityBounds,
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from hexcore.mesh import Mesh
from hexcore.state import PrognosticState


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT.parents[1] / "work" / "jw_step"
DEFAULT_INITIAL = WORK_ROOT / "authority_init.nc"
DEFAULT_NATIVE_T0 = WORK_ROOT / "nomix_internal_t0.nc"
DEFAULT_RECEIPT = ROOT / "receipts" / "jw-dry-day" / "JW-x1.2562-native-24h-dt3600.json"
DEFAULT_FINAL_STATE = ROOT / "receipts" / "jw-dry-day" / "JW-x1.2562-native-24h-dt3600-final.npz"

EXPECTED_INITIAL_SHA256 = "45c6879f794af984de791ca7da654a7da5d515dbdb6a131ea778f4edcf597970"
EXPECTED_NATIVE_T0_SHA256 = "01adfd13c1abe481316a610c875df961938b76b2a12a155ae66e56e348584249"
AUTHORITY_BINARY_SHA256 = "dfdfcebadb39d902ebe70ff59ed5e7540f4795d02c2348b9667cd58021b398c0"
AUTHORITY_TAG_COMMIT = "ac3866c1e5b05f6d4f5bd41aeab7d3882bace514"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative_path(
    path: str | Path, *, repository_root: str | Path = ROOT
) -> str:
    """Return a portable, containment-checked receipt path."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    target = Path(path).expanduser().resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"receipt artifact must remain inside the repository: {target}"
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid repository-relative artifact path: {relative}")
    return relative.as_posix()


def resolve_receipt_artifact(
    declaration: Mapping[str, Any],
    *,
    receipt_path: str | Path,
    repository_root: str | Path = ROOT,
) -> Path:
    """Resolve and integrity-check new portable and narrow legacy declarations.

    New declarations use explicit POSIX ``repo_relative`` paths.  Historical
    receipts omitted ``path_kind`` and recorded an absolute host path; for
    those only, resolve its single basename beside the receipt and still
    require the declared byte count and SHA-256.
    """

    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact declaration path must be a non-empty string")
    path_kind = declaration.get("path_kind")
    root = Path(repository_root).expanduser().resolve(strict=True)
    receipt = Path(receipt_path).expanduser().resolve()

    if path_kind == "repo_relative":
        if "\\" in raw_path:
            raise ValueError("repo_relative artifact paths must use POSIX separators")
        portable = PurePosixPath(raw_path)
        if portable.is_absolute() or not portable.parts or any(
            part in {"", ".", ".."} for part in portable.parts
        ):
            raise ValueError("repo_relative artifact path is absolute or contains traversal")
        candidate = (root / Path(*portable.parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("repo_relative artifact escapes the repository") from error
    elif path_kind is None:
        if "/" in raw_path and "\\" in raw_path:
            raise ValueError("legacy artifact path mixes Windows and POSIX separators")
        legacy = PureWindowsPath(raw_path) if "\\" in raw_path else PurePosixPath(raw_path)
        if not legacy.is_absolute() or legacy.name in {"", ".", ".."}:
            raise ValueError("legacy artifact path must be an absolute single-style path")
        candidate = (receipt.parent / legacy.name).resolve(strict=True)
        try:
            candidate.relative_to(receipt.parent.resolve())
        except ValueError as error:
            raise ValueError("legacy artifact basename escapes the receipt directory") from error
    else:
        raise ValueError(f"unsupported artifact path_kind {path_kind!r}")

    declared_bytes = declaration.get("bytes")
    if not isinstance(declared_bytes, int) or isinstance(declared_bytes, bool):
        raise ValueError("artifact declaration bytes must be an integer")
    if candidate.stat().st_size != declared_bytes:
        raise ValueError("artifact byte count does not match its declaration")
    declared_sha256 = declaration.get("sha256")
    if (
        not isinstance(declared_sha256, str)
        or len(declared_sha256) != 64
        or any(character not in "0123456789abcdef" for character in declared_sha256)
    ):
        raise ValueError("artifact declaration sha256 must be a lowercase SHA-256 digest")
    if sha256_file(candidate) != declared_sha256:
        raise ValueError("artifact SHA-256 does not match its declaration")
    return candidate


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(np.float64).tiny)


def derive_step_count(duration_seconds: float, dt_seconds: float) -> int:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    quotient = duration_seconds / dt_seconds
    rounded = round(quotient)
    if not np.isclose(quotient, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError("duration_seconds must be an exact integer multiple of dt_seconds")
    if rounded < 3:
        raise ValueError("a real day-run/probe must execute at least three full model steps")
    return int(rounded)


def day_config(dt_seconds: float, acoustic_substeps: int) -> DryDycoreConfig:
    """Resolve the no-mixing JW target, changing only timestep controls."""

    return DryDycoreConfig(
        config_dt=float(dt_seconds),
        config_time_integration_order=3,
        config_number_of_sub_steps=int(acoustic_substeps),
        config_dynamics_split_steps=1,
        config_apply_lbcs=False,
        config_split_dynamics_transport=True,
        config_scalar_advection=True,
        config_monotonic=True,
        config_positive_definite=False,
        config_scalar_adv_order=3,
        config_scalar_vadv_order=3,
        config_coef_3rd_order=0.25,
        config_apvm_upwinding=0.5,
        config_epssm=0.1,
        config_moist_physics=False,
        config_physics_suite="none",
        config_iau_option="off",
        config_divergence_damping=False,
        config_horiz_mixing="2d_fixed",
        config_h_theta_eddy_visc2=0.0,
        config_v_theta_eddy_visc2=0.0,
        config_h_mom_eddy_visc2=0.0,
        config_v_mom_eddy_visc2=0.0,
        config_h_theta_eddy_visc4=0.0,
        config_h_mom_eddy_visc4=0.0,
        config_smdiv=0.0,
        config_xnutr=0.0,
        config_vertical_mixing=False,
        config_rayleigh_damp_u=False,
        config_curvature_terms=False,
        config_terrain_following=True,
    )


def state_summary(state: PrognosticState) -> dict[str, Any]:
    result: dict[str, Any] = {"time_seconds": float(state.time_seconds)}
    for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
        values = np.asarray(getattr(state, name))
        finite = np.isfinite(values)
        selected = values[finite]
        result[name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "count": int(values.size),
            "finite_count": int(np.count_nonzero(finite)),
            "min": float(np.min(selected)) if selected.size else None,
            "max": float(np.max(selected)) if selected.size else None,
            "mean_float64": (
                float(np.mean(selected, dtype=np.float64)) if selected.size else None
            ),
        }
    return result


def diagnostics_summary(saved: DrySavedDiagnostics) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in fields(saved):
        values = np.asarray(getattr(saved, item.name))
        finite = np.isfinite(values)
        selected = values[finite]
        result[item.name] = {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "finite_count": int(np.count_nonzero(finite)),
            "min": float(np.min(selected)) if selected.size else None,
            "max": float(np.max(selected)) if selected.size else None,
        }
    return result


def check_bounds(
    metrics: Mapping[str, Any],
    initial: Mapping[str, Any],
    bounds: StabilityBounds,
) -> tuple[float, float]:
    mass_drift = relative_change(float(metrics["mass"]), float(initial["mass"]))
    energy_drift = relative_change(
        float(metrics["energy_proxy"]), float(initial["energy_proxy"])
    )
    if not bool(metrics["all_finite"]):
        raise FloatingPointError("state contains a non-finite value")
    if float(metrics["min_density"]) < bounds.min_density:
        raise FloatingPointError(
            f"min_density {metrics['min_density']} < {bounds.min_density}"
        )
    if mass_drift > bounds.max_mass_relative_drift:
        raise FloatingPointError(
            f"mass drift {mass_drift} > {bounds.max_mass_relative_drift}"
        )
    if energy_drift > bounds.max_energy_relative_drift:
        raise FloatingPointError(
            f"energy drift {energy_drift} > {bounds.max_energy_relative_drift}"
        )
    if float(metrics["max_abs_velocity"]) > bounds.max_abs_velocity:
        raise FloatingPointError(
            f"max velocity {metrics['max_abs_velocity']} > {bounds.max_abs_velocity}"
        )
    return mass_drift, energy_drift


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def save_final_state(
    path: Path,
    state: PrognosticState,
    saved: DrySavedDiagnostics,
) -> dict[str, Any]:
    portable_path = repository_relative_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rho": np.asarray(state.rho),
        "rho_theta": np.asarray(state.rho_theta),
        "rho_u": np.asarray(state.rho_u),
        "rho_w": np.asarray(state.rho_w),
        "scalars": np.asarray(state.scalars),
        "time_seconds": np.asarray(state.time_seconds, dtype=np.float64),
    }
    for item in fields(saved):
        arrays[f"saved_{item.name}"] = np.asarray(getattr(saved, item.name))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    return {
        "path": portable_path,
        "path_kind": "repo_relative",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "arrays": sorted(arrays),
    }


def verify_frozen_input(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: {actual} != frozen {expected_sha256}"
        )
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--native-t0", type=Path, default=DEFAULT_NATIVE_T0)
    parser.add_argument("--dt", type=float, default=3600.0)
    parser.add_argument("--duration", type=float, default=86_400.0)
    parser.add_argument("--acoustic-substeps", type=int, default=6)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--final-state", type=Path, default=DEFAULT_FINAL_STATE)
    parser.add_argument("--no-final-state", action="store_true")
    parser.add_argument("--max-mass-drift", type=float, default=1.0e-5)
    parser.add_argument("--max-energy-drift", type=float, default=0.5)
    parser.add_argument("--max-velocity", type=float, default=500.0)
    parser.add_argument("--min-density", type=float, default=1.0e-8)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser


def execute(args: argparse.Namespace) -> int:
    steps = derive_step_count(args.duration, args.dt)
    if args.acoustic_substeps < 1:
        raise ValueError("acoustic_substeps must be positive")
    if args.progress_every < 1:
        raise ValueError("progress_every must be positive")
    bounds = StabilityBounds(
        max_mass_relative_drift=args.max_mass_drift,
        max_energy_relative_drift=args.max_energy_drift,
        max_abs_velocity=args.max_velocity,
        min_density=args.min_density,
    )
    bounds.validate()
    config = day_config(args.dt, args.acoustic_substeps)
    config.validate()
    receipt_path = args.receipt.expanduser().resolve()
    final_path = args.final_state.expanduser().resolve()
    started_utc = utc_now()
    started = time.perf_counter()
    initial_record = verify_frozen_input(
        args.initial, EXPECTED_INITIAL_SHA256, "authority initial condition"
    )
    native_record = verify_frozen_input(
        args.native_t0, EXPECTED_NATIVE_T0_SHA256, "native internal t0"
    )
    driver_path = ROOT / "src" / "hexcore" / "driver.py"

    receipt: dict[str, Any] = {
        "schema": "mpas-port.jw-dry-day-receipt.v1",
        "evidence": {
            "status": "running",
            "classification": "actual repeated full-step Python-port execution",
            "claim": (
                "native non-quiescent JW state advanced through complete RK3/acoustic "
                "steps for the recorded model duration within named bounds"
            ),
            "non_claim": (
                "This stability run does not by itself establish element-wise Fortran "
                "equivalence, production forecast skill, or GPU readiness."
            ),
            "port_evidence_label": "implemented-unverified",
        },
        "authority": {
            "release": "v8.2.3",
            "tag_commit": AUTHORITY_TAG_COMMIT,
            "atmosphere_binary_sha256": AUTHORITY_BINARY_SHA256,
            "initial_condition": initial_record,
            "native_internal_t0": native_record,
            "initial_saved_diagnostics": (
                "exact native theta_m/exner/internal perturbation sidecar loaded from t0"
            ),
        },
        "configuration": {
            **asdict(config),
            "target_duration_seconds": float(args.duration),
            "target_steps": steps,
            "authority_dt_seconds": 600.0,
            "dt_deviation_reason": (
                "coarse x1.2562 24-hour stability milestone; equivalence uses the "
                "separate 600-second gate"
            ),
        },
        "bounds": asdict(bounds),
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "driver_path": str(driver_path.resolve()),
            "driver_sha256": sha256_file(driver_path),
            "started_utc": started_utc,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "progress": {
            "completed_steps": 0,
            "target_steps": steps,
            "step_records": [],
        },
    }
    write_json_atomic(receipt_path, receipt)

    try:
        mesh_load_started = time.perf_counter()
        mesh = Mesh.from_netcdf(Path(initial_record["path"]))
        native = load_mpas_vertical_grid(
            Path(initial_record["path"]),
            mesh,
            config_coef_3rd_order=config.config_coef_3rd_order,
        )
        state, reference, saved = load_mpas_initial_state(
            Path(native_record["path"]),
            mesh,
            native.vertical_grid,
            scalar_names=("qv",),
            terrain_metrics=native.terrain_metrics,
            return_saved_diagnostics=True,
        )
        driver = DryDycoreDriver(
            mesh,
            native.vertical_grid,
            reference,
            config,
            terrain_metrics=native.terrain_metrics,
        )
        initial_metrics = asdict(driver.metrics(state))
        receipt["timing_seconds"] = {
            "load_and_setup": time.perf_counter() - mesh_load_started,
        }
        receipt["state"] = {
            "initial": state_summary(state),
            "initial_metrics": initial_metrics,
            "initial_saved_diagnostics": diagnostics_summary(saved),
        }
        write_json_atomic(receipt_path, receipt)

        max_mass_drift = 0.0
        max_energy_drift = 0.0
        max_velocity = float(initial_metrics["max_abs_velocity"])
        current = state
        current_saved = saved
        step_records: list[dict[str, Any]] = receipt["progress"]["step_records"]
        for index in range(steps):
            step_started = time.perf_counter()
            result = driver.step(current, saved_diagnostics=current_saved)
            step_wall = time.perf_counter() - step_started
            current = result.state
            current_saved = result.saved_diagnostics
            metrics = asdict(result.receipt.after)
            mass_drift, energy_drift = check_bounds(metrics, initial_metrics, bounds)
            max_mass_drift = max(max_mass_drift, mass_drift)
            max_energy_drift = max(max_energy_drift, energy_drift)
            max_velocity = max(max_velocity, float(metrics["max_abs_velocity"]))
            step_record = {
                "step": index + 1,
                "model_time_seconds": float(current.time_seconds),
                "wall_seconds": step_wall,
                "mass_relative_drift_from_initial": mass_drift,
                "energy_relative_drift_from_initial": energy_drift,
                "metrics": metrics,
            }
            step_records.append(step_record)
            receipt["progress"]["completed_steps"] = index + 1
            receipt["progress"]["last_model_time_seconds"] = float(
                current.time_seconds
            )
            receipt["progress"]["wall_seconds"] = time.perf_counter() - started
            if (index + 1) % args.progress_every == 0 or index + 1 == steps:
                write_json_atomic(receipt_path, receipt)
                print(
                    json.dumps(
                        {
                            "step": index + 1,
                            "of": steps,
                            "model_hour": current.time_seconds / 3600.0,
                            "wall_seconds": step_wall,
                            "mass_drift": mass_drift,
                            "energy_drift": energy_drift,
                            "min_density": metrics["min_density"],
                            "max_velocity": metrics["max_abs_velocity"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        total_wall = time.perf_counter() - started
        receipt["evidence"]["status"] = "passed"
        receipt["evidence"]["completed_24_hours"] = bool(
            np.isclose(args.duration, 86_400.0)
            and np.isclose(current.time_seconds - state.time_seconds, 86_400.0)
        )
        receipt["progress"]["completed_steps"] = steps
        receipt["progress"]["wall_seconds"] = total_wall
        receipt["state"]["final"] = state_summary(current)
        receipt["state"]["final_metrics"] = asdict(driver.metrics(current))
        receipt["state"]["final_saved_diagnostics"] = diagnostics_summary(
            current_saved
        )
        receipt["observed"] = {
            "max_mass_relative_drift": max_mass_drift,
            "max_energy_relative_drift": max_energy_drift,
            "max_abs_velocity": max_velocity,
        }
        if not args.no_final_state:
            receipt["final_state_artifact"] = save_final_state(
                final_path, current, current_saved
            )
        receipt["timing_seconds"]["total"] = total_wall
        receipt["timing_seconds"]["mean_per_step"] = sum(
            float(item["wall_seconds"]) for item in step_records
        ) / steps
        receipt["runner"]["finished_utc"] = utc_now()
        write_json_atomic(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        receipt["evidence"]["status"] = "failed"
        receipt["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        receipt.setdefault("timing_seconds", {})["total"] = time.perf_counter() - started
        receipt["runner"]["finished_utc"] = utc_now()
        write_json_atomic(receipt_path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return execute(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
