#!/usr/bin/env python3
"""Run a real GFS-initialized forecast entirely through the Python MPAS port.

The frozen WPS intermediate file is input data only.  This runner does not
invoke a stock MPAS executable.  Dynamics use the currently admitted dry
RK3/split-explicit driver; GFS water vapour is transported as a passive scalar
and no column-physics backend is run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import time
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpas_port.driver import (
    DryDycoreConfig,
    DryDycoreDriver,
    DryReferenceState,
    DrySavedDiagnostics,
    StabilityBounds,
    TerrainMetrics,
)
from mpas_port.initialization import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    GRAVITY,
    MOIST_THETA_FACTOR,
    REFERENCE_PRESSURE,
    REFERENCE_TEMPERATURE,
    initialize_from_structured,
    load_structured_atmosphere,
)
from mpas_port.mesh import Mesh
from mpas_port.output import HistoryField, HistoryStreamOptions, write_history
from mpas_port.regrid import build_regrid_weights, write_regridded_netcdf
from mpas_port.state import PrognosticState
from mpas_port.terrain import build_terrain_coupling, recover_velocities
from mpas_port.vector import (
    initialize_reconstruction_coefficients,
    reconstruct_1d,
)
from mpas_port.vertical import VerticalGrid, build_vertical_grid


ROOT = Path(__file__).resolve().parents[1]
# The WPS intermediate file lives outside the repository and its location is
# site-specific; a baked-in default would point at a machine nobody else has.
_SOURCE_ENVIRONMENT_VARIABLE = "MPAS_GFS_INTERMEDIATE"
_SOURCE_FROM_ENVIRONMENT = os.environ.get(_SOURCE_ENVIRONMENT_VARIABLE)
DEFAULT_SOURCE = (
    Path(_SOURCE_FROM_ENVIRONMENT) if _SOURCE_FROM_ENVIRONMENT else None
)
DEFAULT_GRID = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
DEFAULT_STATIC = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "gfs-forecast"
DEFAULT_RECEIPT = (
    ROOT
    / "receipts"
    / "gfs-forecast"
    / "GFS-2026-03-26-00.x1.2562.python-port-6h.json"
)
DEFAULT_HISTORY = DEFAULT_ARTIFACT_DIR / "GFS-2026-03-26-00.x1.2562.python-port-6h.history.nc"
DEFAULT_LATLON = DEFAULT_ARTIFACT_DIR / "GFS-2026-03-26-00.x1.2562.python-port-6h.latlon.nc"
DEFAULT_PNG = DEFAULT_ARTIFACT_DIR / "GFS-2026-03-26-00.x1.2562.python-port-6h.surface-pressure.png"
DEFAULT_CHECKSUMS = ROOT / "receipts" / "gfs-forecast" / "SHA256SUMS"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        label = resolved.relative_to(ROOT.resolve()).as_posix()
        path_kind = "repo_relative"
    except ValueError:
        label = str(resolved)
        path_kind = "external_absolute"
    return {
        "path": label,
        "path_kind": path_kind,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def resolve_repo_relative_path(label: str) -> Path:
    """Resolve one receipt path beneath ROOT and reject unsafe spellings."""

    if not isinstance(label, str) or not label or "\\" in label or ":" in label:
        raise ValueError("receipt path must be nonempty repo-relative POSIX text")
    logical = PurePosixPath(label)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError("receipt path must stay beneath the repository root")
    resolved = ROOT.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("receipt path escapes the repository root") from error
    return resolved


def array_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    selected = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "count": int(array.size),
        "finite_count": int(np.count_nonzero(finite)),
        "min": float(np.min(selected)) if selected.size else None,
        "max": float(np.max(selected)) if selected.size else None,
        "mean_float64": (
            float(np.mean(selected, dtype=np.float64)) if selected.size else None
        ),
    }


def state_summary(state: PrognosticState) -> dict[str, Any]:
    return {
        "time_seconds": float(state.time_seconds),
        **{
            name: array_summary(np.asarray(getattr(state, name)))
            for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
        },
    }


def derive_step_count(duration_seconds: float, dt_seconds: float) -> int:
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration must be finite and positive")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt must be finite and positive")
    quotient = duration_seconds / dt_seconds
    rounded = round(quotient)
    if not np.isclose(quotient, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError("duration must be an integer multiple of dt")
    return int(rounded)


def forecast_config(dt_seconds: float, acoustic_substeps: int) -> DryDycoreConfig:
    """Use frozen atmosphere defaults with physics explicitly disabled."""

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
        config_len_disp=0.0,
        config_visc4_2dsmag=0.0,
        config_smagorinsky_coef=0.0,
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


def normalize_runtime_vertical(vertical: VerticalGrid) -> VerticalGrid:
    """Materialize the unused lower interpolation boundary as finite zero.

    ``vertical.py:229-236`` defines fzm/fzp only at Python levels ``1:``.
    Every dynamics use in ``driver.py:724-869`` likewise loops from level one;
    slot zero is a debug sentinel, not a coefficient.  The strict terrain
    conversion validates the complete vector before applying those same level
    bounds, so the runtime representation uses the native-style finite zero.
    Interior coefficients are asserted bit-identical.
    """

    original_fzm = np.asarray(vertical.fzm)
    original_fzp = np.asarray(vertical.fzp)
    if original_fzm.ndim != 1 or original_fzp.shape != original_fzm.shape:
        raise ValueError("vertical fzm/fzp runtime vectors disagree")
    if not np.all(np.isfinite(original_fzm[1:])) or not np.all(
        np.isfinite(original_fzp[1:])
    ):
        raise ValueError("vertical fzm/fzp contain a non-finite interior coefficient")
    runtime_fzm = original_fzm.copy()
    runtime_fzp = original_fzp.copy()
    runtime_fzm[0] = runtime_fzm.dtype.type(0.0)
    runtime_fzp[0] = runtime_fzp.dtype.type(0.0)
    np.testing.assert_array_equal(runtime_fzm[1:], original_fzm[1:])
    np.testing.assert_array_equal(runtime_fzp[1:], original_fzp[1:])
    return replace(vertical, fzm=runtime_fzm, fzp=runtime_fzp)


def validate_output_mesh(dynamics_mesh: Mesh, output_mesh: Mesh) -> None:
    """Bind grid-only render coordinates to the dynamics topology.

    The static file carries float32 copies of lat/lon whose pole rounds just
    beyond pi/2.  Regridding therefore uses the published grid-file float64
    coordinate authority directly, with no unit conversion or clipping.
    Connectivity and global IDs must be identical before the two views may be
    paired.
    """

    for dimension in ("nCells", "nEdges", "nVertices"):
        if int(dynamics_mesh.dimensions[dimension]) != int(
            output_mesh.dimensions[dimension]
        ):
            raise ValueError(f"output grid {dimension} differs from dynamics mesh")
    for name in (
        "indexToCellID",
        "indexToEdgeID",
        "indexToVertexID",
        "nEdgesOnCell",
        "cellsOnEdge",
        "edgesOnCell",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(dynamics_mesh, name)),
            np.asarray(getattr(output_mesh, name)),
            err_msg=f"output grid topology differs at {name}",
        )
    for name, bound in (("latCell", np.pi / 2.0), ("lonCell", np.pi)):
        values = np.asarray(getattr(output_mesh, name))
        units = str(output_mesh.variable_attrs.get(name, {}).get("units", ""))
        if units.strip().lower() not in {"rad", "radian", "radians"}:
            raise ValueError(f"output grid {name} must be declared in radians")
        if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > bound + 1.0e-12:
            raise ValueError(f"output grid {name} is outside its declared radian range")


def build_order2_terrain_metrics(
    mesh: Mesh, vertical: VerticalGrid, coefficient: float
) -> tuple[TerrainMetrics, Any]:
    """Build the exact order-two init-atmosphere ``zb`` branch.

    ``initialize_from_structured`` currently admits
    ``config_theta_adv_order=2``.  In that frozen branch ``z_edge`` is the
    two-cell average and ``zb3`` is zero.
    """

    cells = np.asarray(mesh.cellsOnEdge, dtype=np.int64)
    zgrid = np.asarray(vertical.zgrid)
    dtype = zgrid.dtype
    dv = np.asarray(mesh.dvEdge, dtype=dtype)
    area = np.asarray(mesh.areaCell, dtype=dtype)
    z_edge = dtype.type(0.5) * (
        zgrid[:, cells[:, 0]] + zgrid[:, cells[:, 1]]
    )
    zb = np.empty((zgrid.shape[0], 2, cells.shape[0]), dtype=dtype)
    zb[:, 0] = (
        (z_edge - zgrid[:, cells[:, 0]])
        * dv[np.newaxis, :]
        / area[cells[:, 0]][np.newaxis, :]
    )
    zb[:, 1] = (
        (z_edge - zgrid[:, cells[:, 1]])
        * dv[np.newaxis, :]
        / area[cells[:, 1]][np.newaxis, :]
    )
    coupling = build_terrain_coupling(
        n_edges_on_cell=mesh.nEdgesOnCell,
        edges_on_cell=mesh.edgesOnCell,
        cells_on_edge=mesh.cellsOnEdge,
        zb=zb,
        zb3=np.zeros_like(zb),
        config_coef_3rd_order=coefficient,
        array_layout="logical",
    )
    return (
        TerrainMetrics(
            zb_cell=np.asarray(coupling.zb_cell),
            zb3_cell=np.asarray(coupling.zb3_cell),
        ),
        coupling,
    )


def build_reference_and_sidecar(
    initialized: Any,
    mesh: Mesh,
    vertical: VerticalGrid,
    coupling: Any,
) -> tuple[DryReferenceState, DrySavedDiagnostics]:
    state = initialized.state
    dtype = state.rho.dtype
    zmid = dtype.type(0.5) * (
        np.asarray(vertical.zgrid[:-1], dtype=dtype)
        + np.asarray(vertical.zgrid[1:], dtype=dtype)
    )
    zz = np.asarray(vertical.zz, dtype=dtype)
    p0 = dtype.type(REFERENCE_PRESSURE)
    rd = dtype.type(DRY_AIR_GAS_CONSTANT)
    cp = dtype.type(DRY_AIR_CP)
    gravity = dtype.type(GRAVITY)
    tref = dtype.type(REFERENCE_TEMPERATURE)
    pressure_base = p0 * np.exp(-gravity * zmid / (rd * tref))
    rho_base = pressure_base / (rd * tref * zz)
    exner_base = (pressure_base / p0) ** (rd / cp)
    theta_base = tref / exner_base
    rho_theta_base = rho_base * theta_base
    reference = DryReferenceState(
        rho_base=np.asarray(rho_base, dtype=dtype),
        rho_theta_base=np.asarray(rho_theta_base, dtype=dtype),
        pressure_base=np.asarray(pressure_base, dtype=dtype),
        exner_base=np.asarray(exner_base, dtype=dtype),
    )
    reference.validate(state.rho.shape)

    theta_m = np.asarray(
        initialized.diagnostics.modified_potential_temperature, dtype=dtype
    )
    exner = (
        np.asarray(initialized.diagnostics.pressure, dtype=dtype) / p0
    ) ** (rd / cp)
    density_perturbation = state.rho - reference.rho_base
    rtheta_perturbation = state.rho_theta - reference.rho_theta_base
    pressure_perturbation = zz * rd * (
        exner * rtheta_perturbation
        + reference.rho_theta_base * (exner - reference.exner_base)
    )
    # The constructed VerticalGrid intentionally marks projection slot zero
    # NaN because only levels 1..nVertLevels-1 consume fzm/fzp.  The strict
    # terrain API validates whole vectors, so provide a harmless finite
    # boundary value for this recovery call; no equation reads that slot.
    fzm_recovery = np.asarray(vertical.fzm, dtype=dtype).copy()
    fzp_recovery = np.asarray(vertical.fzp, dtype=dtype).copy()
    fzm_recovery[0] = dtype.type(0.0)
    fzp_recovery[0] = dtype.type(0.0)
    recovered = recover_velocities(
        rho_zz=state.rho,
        ru=state.rho_u,
        rw=state.rho_w,
        zz=vertical.zz,
        fzm=fzm_recovery,
        fzp=fzp_recovery,
        cf1=vertical.cf1,
        cf2=vertical.cf2,
        cf3=vertical.cf3,
        coupling=coupling,
    )
    saved = DrySavedDiagnostics(
        theta_m=np.asarray(theta_m, dtype=dtype),
        exner=np.asarray(exner, dtype=dtype),
        density_perturbation=np.asarray(density_perturbation, dtype=dtype),
        rho_theta_perturbation=np.asarray(rtheta_perturbation, dtype=dtype),
        pressure_perturbation=np.asarray(pressure_perturbation, dtype=dtype),
        normal_velocity=np.asarray(recovered.u, dtype=dtype),
        vertical_velocity=np.asarray(recovered.w, dtype=dtype),
    )
    saved.validate(state.rho.shape, dtype, state.rho_u.shape[1])
    return reference, saved


def relative_change(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), np.finfo(np.float64).tiny)


def check_bounds(
    metrics: Mapping[str, Any],
    initial: Mapping[str, Any],
    bounds: StabilityBounds,
    qv: np.ndarray,
) -> tuple[float, float]:
    if not bool(metrics["all_finite"]):
        raise FloatingPointError("forecast state contains non-finite values")
    mass_drift = relative_change(float(metrics["mass"]), float(initial["mass"]))
    energy_drift = relative_change(
        float(metrics["energy_proxy"]), float(initial["energy_proxy"])
    )
    if float(metrics["min_density"]) < bounds.min_density:
        raise FloatingPointError("forecast density crossed the declared minimum")
    if mass_drift > bounds.max_mass_relative_drift:
        raise FloatingPointError("forecast mass drift crossed the declared maximum")
    if energy_drift > bounds.max_energy_relative_drift:
        raise FloatingPointError("forecast energy-proxy drift crossed the declared maximum")
    if float(metrics["max_abs_velocity"]) > bounds.max_abs_velocity:
        raise FloatingPointError("forecast velocity crossed the declared maximum")
    if not np.all(np.isfinite(qv)) or np.min(qv) < -1.0e-8 or np.max(qv) > 0.05:
        raise FloatingPointError("passive qv left its declared [-1e-8, 0.05] range")
    return mass_drift, energy_drift


def diagnose_products(
    state: PrognosticState,
    saved: DrySavedDiagnostics,
    mesh: Mesh,
    vertical: VerticalGrid,
    reconstruction_coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    dtype = state.rho.dtype
    zz = np.asarray(vertical.zz, dtype=dtype)
    rd = dtype.type(DRY_AIR_GAS_CONSTANT)
    p0 = dtype.type(REFERENCE_PRESSURE)
    cp = dtype.type(DRY_AIR_CP)
    exner = (zz * (rd / p0) * state.rho_theta) ** (rd / (cp - rd))
    pressure = zz * rd * state.rho_theta * exner
    qv = state.scalars[0]
    theta = (state.rho_theta / state.rho) / (
        dtype.type(1.0) + dtype.type(MOIST_THETA_FACTOR) * qv
    )
    temperature = theta * exner
    surface_pressure = diagnose_surface_pressure(state, vertical)
    reconstructed = reconstruct_1d(
        mesh,
        saved.normal_velocity[0],
        include_halos=True,
        coefficients=reconstruction_coefficients,
    )
    wind_speed = np.hypot(reconstructed.zonal, reconstructed.meridional)
    return {
        "pressure": np.asarray(pressure),
        "theta": np.asarray(theta),
        "surface_pressure": np.asarray(surface_pressure),
        "temperature_lowest_model_level": np.asarray(temperature[0]),
        "u_lowest_model_level": np.asarray(reconstructed.zonal),
        "v_lowest_model_level": np.asarray(reconstructed.meridional),
        "wind_speed_lowest_model_level": np.asarray(wind_speed),
    }


def diagnose_surface_pressure(
    state: PrognosticState, vertical: VerticalGrid
) -> np.ndarray:
    """Diagnose native-cell surface pressure without reconstructing wind."""

    dtype = state.rho.dtype
    zz = np.asarray(vertical.zz, dtype=dtype)
    rd = dtype.type(DRY_AIR_GAS_CONSTANT)
    p0 = dtype.type(REFERENCE_PRESSURE)
    cp = dtype.type(DRY_AIR_CP)
    exner = (zz * (rd / p0) * state.rho_theta) ** (rd / (cp - rd))
    pressure = zz * rd * state.rho_theta * exner
    qv = np.asarray(state.scalars[0], dtype=dtype)
    return np.asarray(
        dtype.type(0.5)
        * dtype.type(GRAVITY)
        / dtype.type(vertical.rdzw[0])
        * (
            dtype.type(1.25) * state.rho[0] * (dtype.type(1.0) + qv[0])
            - dtype.type(0.25) * state.rho[1] * (dtype.type(1.0) + qv[1])
        )
        + pressure[0]
    )


def mean_surface_pressure_record(
    *, step: int, model_time_seconds: float, surface_pressure: np.ndarray
) -> dict[str, float | int]:
    """Return one finite domain-mean pressure point in Pa and hPa."""

    values = np.asarray(surface_pressure, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise FloatingPointError("surface-pressure trajectory must be finite and nonempty")
    mean_pa = float(np.mean(values, dtype=np.float64))
    return {
        "step": int(step),
        "model_time_seconds": float(model_time_seconds),
        "mean_surface_pressure_pa": mean_pa,
        "mean_surface_pressure_hpa": mean_pa * 0.01,
    }


def summarize_surface_pressure_adjustment(
    trajectory: list[Mapping[str, Any]],
    *,
    adjustment_seconds: float = 7_200.0,
) -> dict[str, Any]:
    """Quantify the early pressure adjustment against the full trajectory.

    The positive ``mean_drop`` values are initial mean minus later mean.  This
    diagnostic describes model adjustment only; it is not a forecast-skill
    metric.
    """

    if len(trajectory) < 2:
        raise ValueError("surface-pressure trajectory needs t0 and a final point")
    times = np.asarray(
        [float(point["model_time_seconds"]) for point in trajectory],
        dtype=np.float64,
    )
    means = np.asarray(
        [float(point["mean_surface_pressure_pa"]) for point in trajectory],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(times))
        or not np.all(np.isfinite(means))
        or times[0] != 0.0
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("surface-pressure trajectory times/means are invalid")
    result: dict[str, Any] = {
        "classification": "initialization adjustment diagnostic",
        "non_claim": "domain-mean pressure adjustment is not a forecast-skill score",
        "available": False,
        "adjustment_window_seconds": float(adjustment_seconds),
        "forecast_window_seconds": float(times[-1]),
    }
    matches = np.flatnonzero(
        np.isclose(times, adjustment_seconds, rtol=0.0, atol=1.0e-9)
    )
    if matches.size != 1:
        result["unavailable_reason"] = (
            "trajectory has no exact point at the requested adjustment window"
        )
        return result
    adjustment_index = int(matches[0])
    initial_mean = float(means[0])
    adjustment_mean = float(means[adjustment_index])
    final_mean = float(means[-1])
    early_change = adjustment_mean - initial_mean
    full_change = final_mean - initial_mean
    early_drop = -early_change
    full_drop = -full_change
    if full_drop <= 0.0:
        result["unavailable_reason"] = (
            "full-forecast domain-mean surface pressure did not drop"
        )
        return result
    return result | {
        "available": True,
        "first_two_hours": {
            "mean_change_pa": early_change,
            "mean_change_hpa": early_change * 0.01,
            "mean_drop_pa": early_drop,
            "mean_drop_hpa": early_drop * 0.01,
        },
        "full_forecast": {
            "mean_change_pa": full_change,
            "mean_change_hpa": full_change * 0.01,
            "mean_drop_pa": full_drop,
            "mean_drop_hpa": full_drop * 0.01,
        },
        "fraction_of_full_mean_drop_in_first_two_hours": early_drop / full_drop,
    }


def write_checksum_inventory(path: Path, targets: tuple[Path, ...]) -> None:
    """Write a stable SHA-256 inventory for the receipt and its artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for target in targets:
        resolved = target.resolve(strict=True)
        try:
            label = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                "committed checksum targets must be beneath the repository root"
            ) from error
        if resolve_repo_relative_path(label) != resolved:
            raise ValueError("checksum target did not round-trip through repo path")
        lines.append(f"{sha256_file(resolved)}  {label}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def render_surface_pressure(
    path: Path,
    latitude: np.ndarray,
    longitude: np.ndarray,
    initial_pressure: np.ndarray,
    final_pressure: np.ndarray,
    *,
    initial_time: datetime,
    valid_time: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_hpa = np.asarray(initial_pressure) * 0.01
    final_hpa = np.asarray(final_pressure) * 0.01
    change_hpa = final_hpa - initial_hpa
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.5), constrained_layout=True)
    first = axes[0].pcolormesh(
        longitude, latitude, final_hpa, shading="auto", cmap="turbo"
    )
    axes[0].set_title(
        f"Python MPAS port: surface pressure (hPa)  valid {valid_time:%Y-%m-%d %H:%M UTC}"
    )
    fig.colorbar(first, ax=axes[0], label="hPa", pad=0.015)
    vmax = max(float(np.max(np.abs(change_hpa))), 0.01)
    second = axes[1].pcolormesh(
        longitude,
        latitude,
        change_hpa,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    axes[1].set_title(
        f"Surface-pressure change since {initial_time:%Y-%m-%d %H:%M UTC}"
    )
    fig.colorbar(second, ax=axes[1], label="hPa", pad=0.015)
    for axis in axes:
        axis.set_xlabel("longitude (degrees east)")
        axis.set_ylabel("latitude (degrees north)")
        axis.set_xlim(float(longitude[0]), float(longitude[-1]))
        axis.set_ylim(float(latitude[0]), float(latitude[-1]))
        axis.grid(alpha=0.18, linewidth=0.5)
    fig.suptitle(
        "Real GFS initialization · dry RK3 dynamics · qv passive-only · physics suite none",
        fontsize=13,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--levels", type=int, default=55)
    parser.add_argument("--ztop", type=float, default=30_000.0)
    parser.add_argument("--dt", type=float, default=3600.0)
    parser.add_argument("--duration", type=float, default=21_600.0)
    parser.add_argument("--acoustic-substeps", type=int, default=6)
    parser.add_argument("--max-mass-drift", type=float, default=2.0e-8)
    parser.add_argument("--max-energy-drift", type=float, default=0.50)
    parser.add_argument("--max-velocity", type=float, default=500.0)
    parser.add_argument("--min-density", type=float, default=1.0e-7)
    parser.add_argument("--latlon-resolution", type=float, default=2.0)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--latlon", type=Path, default=DEFAULT_LATLON)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.source is None:
        raise ValueError(
            "--source is required: pass the GFS WPS intermediate file, or set "
            f"{_SOURCE_ENVIRONMENT_VARIABLE}"
        )
    steps = derive_step_count(args.duration, args.dt)
    if args.levels < 3:
        raise ValueError("at least three model levels are required")
    if args.acoustic_substeps < 1:
        raise ValueError("acoustic-substeps must be positive")
    if not 0.25 <= args.latlon_resolution <= 10.0:
        raise ValueError("latlon-resolution must lie in [0.25, 10] degrees")

    source_path = args.source.expanduser().resolve(strict=True)
    grid_path = args.grid.expanduser().resolve(strict=True)
    static_path = args.static.expanduser().resolve(strict=True)
    receipt_path = args.receipt.expanduser().resolve()
    history_path = args.history.expanduser().resolve()
    latlon_path = args.latlon.expanduser().resolve()
    png_path = args.png.expanduser().resolve()
    checksums_path = args.checksums.expanduser().resolve()
    bounds = StabilityBounds(
        max_mass_relative_drift=args.max_mass_drift,
        max_energy_relative_drift=args.max_energy_drift,
        max_abs_velocity=args.max_velocity,
        min_density=args.min_density,
    )
    bounds.validate()
    config = forecast_config(args.dt, args.acoustic_substeps)
    config.validate()

    timing: dict[str, float] = {}
    started = time.perf_counter()
    source = load_structured_atmosphere(source_path)
    timing["source_load"] = time.perf_counter() - started
    checkpoint = time.perf_counter()
    mesh = Mesh.from_netcdf(grid_path, static_path)
    output_mesh = Mesh.from_netcdf(grid_path)
    validate_output_mesh(mesh, output_mesh)
    vertical = build_vertical_grid(
        mesh,
        np.asarray(mesh.ter, dtype=np.float64),
        n_vert_levels=args.levels,
        ztop=args.ztop,
        smooth_surfaces=False,
    )
    vertical = normalize_runtime_vertical(vertical)
    timing["mesh_and_vertical"] = time.perf_counter() - checkpoint
    checkpoint = time.perf_counter()
    initialized = initialize_from_structured(source, mesh, vertical)
    timing["initialization"] = time.perf_counter() - checkpoint
    terrain, coupling = build_order2_terrain_metrics(
        mesh, vertical, config.config_coef_3rd_order
    )
    reference, saved = build_reference_and_sidecar(
        initialized, mesh, vertical, coupling
    )
    driver = DryDycoreDriver(
        mesh,
        vertical,
        reference,
        config,
        terrain_metrics=terrain,
    )
    coefficients = initialize_reconstruction_coefficients(mesh)
    initial_state = initialized.state.copy()
    initial_saved = saved
    initial_metrics = asdict(driver.metrics(initial_state))
    initial_products = diagnose_products(
        initial_state, initial_saved, mesh, vertical, coefficients
    )
    surface_pressure_trajectory = [
        mean_surface_pressure_record(
            step=0,
            model_time_seconds=initial_state.time_seconds,
            surface_pressure=initial_products["surface_pressure"],
        )
    ]
    timing["forecast_setup"] = time.perf_counter() - checkpoint - timing["initialization"]

    current = initial_state.copy()
    current_saved = saved
    records: list[dict[str, Any]] = []
    max_mass_drift = 0.0
    max_energy_drift = 0.0
    max_velocity = float(initial_metrics["max_abs_velocity"])
    checkpoint = time.perf_counter()
    for index in range(steps):
        step_started = time.perf_counter()
        result = driver.step(current, saved_diagnostics=current_saved)
        current = result.state
        current_saved = result.saved_diagnostics
        metrics = asdict(result.receipt.after)
        qv = np.asarray(current.scalars[0])
        step_surface_pressure = diagnose_surface_pressure(current, vertical)
        pressure_record = mean_surface_pressure_record(
            step=index + 1,
            model_time_seconds=current.time_seconds,
            surface_pressure=step_surface_pressure,
        )
        surface_pressure_trajectory.append(pressure_record)
        mass_drift, energy_drift = check_bounds(
            metrics, initial_metrics, bounds, qv
        )
        max_mass_drift = max(max_mass_drift, mass_drift)
        max_energy_drift = max(max_energy_drift, energy_drift)
        max_velocity = max(max_velocity, float(metrics["max_abs_velocity"]))
        record = {
            "step": index + 1,
            "model_time_seconds": float(current.time_seconds),
            "wall_seconds": time.perf_counter() - step_started,
            "mass_relative_drift_from_initial": mass_drift,
            "energy_proxy_relative_drift_from_initial": energy_drift,
            "metrics": metrics,
            "qv_min": float(np.min(qv)),
            "qv_max": float(np.max(qv)),
            "mean_surface_pressure_pa": pressure_record[
                "mean_surface_pressure_pa"
            ],
            "mean_surface_pressure_hpa": pressure_record[
                "mean_surface_pressure_hpa"
            ],
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    timing["forecast"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    final_products = diagnose_products(
        current, current_saved, mesh, vertical, coefficients
    )
    np.testing.assert_allclose(
        surface_pressure_trajectory[-1]["mean_surface_pressure_pa"],
        np.mean(final_products["surface_pressure"], dtype=np.float64),
        rtol=0.0,
        atol=0.0,
    )
    pressure_adjustment = summarize_surface_pressure_adjustment(
        surface_pressure_trajectory
    )
    valid_text = str(source.provenance.get("valid_time", "2026-03-26_00:00:00"))
    initial_time = datetime.fromisoformat(valid_text.replace("_", "T", 1))
    valid_time = initial_time + timedelta(seconds=args.duration)

    history_fields = {
        "rho": HistoryField(
            np.stack((initial_state.rho, current.rho)),
            ("Time", "nVertLevels", "nCells"),
        ),
        "theta": HistoryField(
            np.stack((initial_products["theta"], final_products["theta"])),
            ("Time", "nVertLevels", "nCells"),
        ),
        "pressure": HistoryField(
            np.stack((initial_products["pressure"], final_products["pressure"])),
            ("Time", "nVertLevels", "nCells"),
        ),
        "qv": HistoryField(
            np.stack((initial_state.scalars[0], current.scalars[0])),
            ("Time", "nVertLevels", "nCells"),
            {"units": "kg kg-1", "long_name": "passively transported water-vapor mixing ratio"},
        ),
        "u": HistoryField(
            np.stack((initial_saved.normal_velocity, current_saved.normal_velocity)),
            ("Time", "nVertLevels", "nEdges"),
        ),
        "w": HistoryField(
            np.stack((initial_saved.vertical_velocity, current_saved.vertical_velocity)),
            ("Time", "nVertLevelsP1", "nCells"),
        ),
        "surface_pressure": HistoryField(
            np.stack((initial_products["surface_pressure"], final_products["surface_pressure"])),
            ("Time", "nCells"),
        ),
        "temperature_lowest_model_level": HistoryField(
            np.stack((initial_products["temperature_lowest_model_level"], final_products["temperature_lowest_model_level"])),
            ("Time", "nCells"),
            {"units": "K", "long_name": "temperature at lowest model level"},
        ),
        "wind_speed_lowest_model_level": HistoryField(
            np.stack((initial_products["wind_speed_lowest_model_level"], final_products["wind_speed_lowest_model_level"])),
            ("Time", "nCells"),
            {"units": "m s-1", "long_name": "horizontal wind speed at lowest model level"},
        ),
    }
    write_history(
        history_path,
        output_mesh,
        history_fields,
        (initial_time, valid_time),
        initial_time=initial_time,
        time_seconds=(0.0, args.duration),
        n_vert_levels=args.levels,
        global_attrs={
            "title": "Real GFS-initialized Python MPAS-port dry forecast",
            "physics_suite": "none",
            "water_vapor_treatment": "passive scalar transport only",
        },
        stream_options=HistoryStreamOptions(clobber_mode="truncate"),
    )

    latitude = np.arange(-90.0, 90.0 + 0.5 * args.latlon_resolution, args.latlon_resolution)
    longitude = np.arange(0.0, 360.0, args.latlon_resolution)
    weights = build_regrid_weights(
        output_mesh,
        target_latitude=latitude,
        target_longitude=longitude,
        method="inverse_distance",
        neighbors=4,
        power=2.0,
    )
    regrid_fields: dict[str, HistoryField] = {}
    for name in (
        "surface_pressure",
        "temperature_lowest_model_level",
        "u_lowest_model_level",
        "v_lowest_model_level",
        "wind_speed_lowest_model_level",
    ):
        units = "Pa" if name == "surface_pressure" else (
            "K" if name.startswith("temperature") else "m s-1"
        )
        regrid_fields[name] = HistoryField(
            np.stack((initial_products[name], final_products[name])),
            ("Time", "nCells"),
            {"units": units},
        )
    write_regridded_netcdf(
        latlon_path,
        weights,
        regrid_fields,
        cell_axis={name: 1 for name in regrid_fields},
        valid_time=(initial_time, valid_time),
        initial_time=initial_time,
        global_attrs={
            "title": "Real GFS-initialized Python MPAS-port forecast products",
            "physics_suite": "none",
            "water_vapor_treatment": "passive scalar transport only",
        },
        clobber=True,
    )
    gridded_initial_pressure = weights.apply(initial_products["surface_pressure"])
    gridded_final_pressure = weights.apply(final_products["surface_pressure"])
    render_surface_pressure(
        png_path,
        latitude,
        longitude,
        gridded_initial_pressure,
        gridded_final_pressure,
        initial_time=initial_time,
        valid_time=valid_time,
    )
    timing["diagnostics_and_output"] = time.perf_counter() - checkpoint
    timing["total_before_hashing"] = time.perf_counter() - started

    payload = {
        "schema": "mpas-port.gfs-forecast-receipt.v1",
        "evidence": {
            "status": "passed",
            "classification": "real GFS-initialized repeated full-step Python-port forecast",
            "claim": "the Python CPU port initialized, advanced, diagnosed, wrote, regridded, and rendered this forecast",
            "non_claim": "this is not a frozen-Fortran equivalence result and does not establish forecast skill",
        },
        "physics": {
            "config_physics_suite": "none",
            "column_backend_executed": False,
            "qv_treatment": "passive scalar advection only",
            "non_claim": "no microphysics, convection, radiation, PBL, surface-layer, or land-surface parameterization ran",
        },
        "source": {
            **file_record(source_path),
            "adapter": source.provenance.get("adapter"),
            "valid_time": source.provenance.get("valid_time"),
            "forecast_hour": source.provenance.get("forecast_hour"),
        },
        "mesh": {
            "grid": file_record(grid_path),
            "static": file_record(static_path),
            "nCells": int(mesh.dimensions["nCells"]),
            "nEdges": int(mesh.dimensions["nEdges"]),
        },
        "configuration": {
            "duration_seconds": args.duration,
            "steps": steps,
            "dt_seconds": args.dt,
            "acoustic_substeps": args.acoustic_substeps,
            "vertical_levels": args.levels,
            "ztop_m": args.ztop,
            "dry_dycore": asdict(config),
        },
        "bounds": asdict(bounds),
        "observed": {
            "max_mass_relative_drift": max_mass_drift,
            "max_energy_proxy_relative_drift": max_energy_drift,
            "max_abs_velocity": max_velocity,
            "surface_pressure_change_pa": array_summary(
                final_products["surface_pressure"] - initial_products["surface_pressure"]
            ),
            "mean_surface_pressure_trajectory": surface_pressure_trajectory,
            "surface_pressure_initialization_adjustment": pressure_adjustment,
        },
        "progress": {
            "completed_steps": len(records),
            "last_model_time_seconds": float(current.time_seconds),
            "step_records": records,
        },
        "state": {
            "initial": state_summary(initial_state),
            "final": state_summary(current),
        },
        "products": {
            "surface_pressure": array_summary(final_products["surface_pressure"]),
            "temperature_lowest_model_level": array_summary(final_products["temperature_lowest_model_level"]),
            "wind_speed_lowest_model_level": array_summary(final_products["wind_speed_lowest_model_level"]),
        },
        "artifacts": {
            "history": {**file_record(history_path), "records": 2},
            "latlon": {
                **file_record(latlon_path),
                "records": 2,
                "resolution_degrees": args.latlon_resolution,
            },
            "png": file_record(png_path),
        },
        "runner": file_record(Path(__file__)),
        "timing_seconds": timing,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_checksum_inventory(
        checksums_path,
        (receipt_path, history_path, latlon_path, png_path),
    )
    print(json.dumps({"receipt": str(receipt_path), "status": "passed"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
