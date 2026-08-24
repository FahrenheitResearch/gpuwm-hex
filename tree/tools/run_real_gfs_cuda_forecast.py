#!/usr/bin/env python3
"""Run the pinned real-GFS six-hour forecast through two CUDA arms and plot it.

This is deliberately a separate evidence lane from the committed CPU forecast.
The WPS intermediate, x1.2562 grid, and x1.2562 static file are pinned to the
bytes used by that lane, then materialized once as a C-contiguous binary32 host
state.  Two independent device uploads advance that same preparation.  No
forecast product is written until gpuwm reports total equality of the two CUDA
capsules.

The final arm-A state is downloaded only after that comparison.  History and
lat/lon products use the existing MPAS writers, and the unchanged Rust
``rw_wrfbatch`` path renders the five generic dynamic fields plus terrain.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
TOOLS_ROOT = ROOT / "tools"
for import_root in (SOURCE_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_real_gfs_forecast as cpu_gfs  # noqa: E402

from mpas_port.cuda_backend import require_cuda  # noqa: E402
from mpas_port.cuda_dualrun import (  # noqa: E402
    CudaArmRun,
    PreparedCudaInputs,
    compare_cuda_capsule_files,
    fingerprint_atmosphere,
    load_ftz_binding_record,
    load_gpuwm_dualrun,
    prepare_cuda_kernel_cache,
    run_cuda_arm_generic,
    sha256_file,
    write_json_atomic,
)
from mpas_port.driver import (  # noqa: E402
    DryDycoreConfig,
    DryDycoreDriver,
    DryReferenceState,
    DrySavedDiagnostics,
    StabilityBounds,
    TerrainMetrics,
)
from mpas_port.initialization import (  # noqa: E402
    initialize_from_structured,
    load_structured_atmosphere,
)
from mpas_port.mesh import Mesh  # noqa: E402
from mpas_port.output import (  # noqa: E402
    HistoryField,
    HistoryStreamOptions,
    write_history,
)
from mpas_port.regrid import (  # noqa: E402
    build_regrid_weights,
    write_regridded_netcdf,
)
from mpas_port.rust_renderer import (  # noqa: E402
    RendererProbe,
    discover_rust_renderer,
    inspect_renderer_products,
    materialize_gfs_rust_input,
    render_catalogued_products,
)
from mpas_port.state import PrognosticState  # noqa: E402
from mpas_port.vector import initialize_reconstruction_coefficients  # noqa: E402
from mpas_port.vertical import VerticalGrid, build_vertical_grid  # noqa: E402


GFS_STEM = "GFS-2026-03-26-00.x1.2562.cuda-port-sm120-dual-6h"
DEFAULT_SOURCE = cpu_gfs.DEFAULT_SOURCE
DEFAULT_GRID = cpu_gfs.DEFAULT_GRID
DEFAULT_STATIC = cpu_gfs.DEFAULT_STATIC
DEFAULT_GPUWM_ROOT = Path(os.environ.get("GPUWM_ROOT", str(Path.home() / "gpuwm")))
DEFAULT_GPUWM_PROBE = ROOT / "receipts" / "cuda-ftz-sm120" / "gpuwm-probe"
DEFAULT_FTZ_BINDING = ROOT / "receipts" / "cuda-ftz-sm120" / "binding.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "cuda-gfs"
DEFAULT_RECEIPT_ROOT = ROOT / "receipts" / "cuda-gfs-forecast" / "final"
DEFAULT_CACHE_ROOT = ROOT / "work" / "cuda-gfs-forecast-cache" / "fresh"

TARGET_DT_SECONDS = 3_600.0
TARGET_DURATION_SECONDS = 21_600.0
TARGET_STEPS = 6
TARGET_ACOUSTIC_SUBSTEPS = 6
TARGET_LEVELS = 55
TARGET_ZTOP_M = 30_000.0
TARGET_LATLON_DEGREES = 2.0

PROFILE = "real-gfs-20260326-x1.2562-dry-binary32"
TARGET = "real GFS 2026-03-26 00Z x1.2562 dry no-physics CUDA 6 h forecast"
PREPARATION_METHOD = (
    "load the three pinned CPU-lane inputs once, initialize with the shared "
    "structured-atmosphere path, and explicitly materialize one C-contiguous "
    "binary32 host atmosphere for two independent CUDA uploads"
)

EXPECTED_RENDERER_SHA256 = (
    "d9e8abeeb622892441f4120fa0b3062ff089ddcaf1f413d9c0817dac61c39983"
)
EXPECTED_RENDERER_BYTES = 9_030_656
WINDOWS_LEGACY_PATH_LIMIT = 259
RENDERER_FILENAME_BUDGET = 140
DEFAULT_PRODUCTS = (
    "var:wrf_surface_pressure",
    "var:wrf_temperature_lowest_model_level",
    "var:wrf_u_lowest_model_level",
    "var:wrf_v_lowest_model_level",
    "var:wrf_wind_speed_lowest_model_level",
    "terrain_height",
)


@dataclass(frozen=True, slots=True)
class InputPin:
    filename: str
    bytes: int
    sha256: str


INPUT_PINS = {
    "gfs_wps_intermediate": InputPin(
        "GFS:2026-03-26_00",
        818_178_824,
        "98b773f981b08444ea838fab8e31568bd2e92ab7237b67ae056970518ccabd49",
    ),
    "x1_2562_grid": InputPin(
        "x1.2562.grid.nc",
        3_508_132,
        "8a825312a713bbe959c33ed03c2b503e5ec626238de6b15a686cd0ad5b40c986",
    ),
    "x1_2562_static": InputPin(
        "x1.2562.static.nc",
        4_894_200,
        "c0f4516eac15f9b97abb80d5c6cf82ebb9fc06b67581858070853e44a3026cab",
    ),
}

_VERTICAL_ARRAY_FIELDS = (
    "zw",
    "dzw",
    "rdzw",
    "zu",
    "dzu",
    "rdzu",
    "rdzwp",
    "rdzwm",
    "fzp",
    "fzm",
    "ah",
    "hx",
    "zgrid",
    "zz",
    "zxu",
    "dss",
)
_STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_SAVED_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)
_LOWER_SENTINEL_FIELDS = (
    "dzu",
    "rdzu",
    "rdzwp",
    "rdzwm",
    "fzp",
    "fzm",
)


@dataclass(frozen=True, slots=True)
class PreparedGfsCase:
    cuda: PreparedCudaInputs
    output_mesh: Mesh
    source_provenance: dict[str, Any]
    input_records: dict[str, dict[str, Any]]
    grid_path: Path
    static_path: Path


@dataclass(frozen=True, slots=True)
class OutputPlan:
    artifact_root: Path
    receipt_root: Path
    history: Path
    latlon: Path
    renderer_input: Path
    renderer_store: Path
    renderer_outputs: Path
    capsule_a: Path
    capsule_b: Path
    comparison: Path
    receipt: Path
    checksums: Path


@dataclass(frozen=True, slots=True)
class VerifiedCudaRun:
    arm_a: CudaArmRun
    arm_b: CudaArmRun
    comparison: dict[str, Any]
    capsule_a_path: Path
    capsule_b_path: Path
    comparison_path: Path


@dataclass(frozen=True, slots=True)
class HostDownload:
    state: PrognosticState
    saved: DrySavedDiagnostics
    bytes: int
    seconds: float


def _as_f32_c(name: str, value: Any) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind != "f" or not np.all(np.isfinite(source)):
        raise ValueError(f"{name} must be a finite floating array")
    result = np.ascontiguousarray(source, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} overflowed during binary32 materialization")
    return result


def normalize_cuda_vertical_sentinels(vertical: VerticalGrid) -> VerticalGrid:
    """Materialize unused lower-boundary coefficient sentinels as finite zero.

    ``build_vertical_grid`` intentionally defines these six vectors only over
    ``1:``.  Their slot zero is a debug NaN; both the CPU equations and the
    CUDA kernels read the coefficients only for ``k >= 1``.  The committed GFS
    CPU runner already applies this exact policy to fzm/fzp for strict terrain
    recovery.  The CUDA host authority extends it to all six uploaded vectors
    and asserts the defined interiors are bit-identical.
    """

    updates: dict[str, np.ndarray] = {}
    for name in _LOWER_SENTINEL_FIELDS:
        source = np.asarray(getattr(vertical, name))
        if source.ndim != 1 or source.size < 2:
            raise ValueError(f"vertical.{name} must be a nonempty level vector")
        if not np.all(np.isfinite(source[1:])):
            raise ValueError(f"vertical.{name}[1:] contains a non-finite coefficient")
        if np.isfinite(source[0]) and source[0] != source.dtype.type(0.0):
            raise ValueError(
                f"vertical.{name}[0] must be the unused NaN sentinel or finite zero"
            )
        converted = source.copy()
        converted[0] = source.dtype.type(0.0)
        np.testing.assert_array_equal(converted[1:], source[1:])
        updates[name] = converted
    return replace(vertical, **updates)


def materialize_binary32_atmosphere(
    state: PrognosticState,
    saved: DrySavedDiagnostics,
    vertical: VerticalGrid,
    reference: DryReferenceState,
    terrain: TerrainMetrics,
    *,
    n_cells: int,
    n_edges: int,
) -> tuple[
    PrognosticState,
    DrySavedDiagnostics,
    VerticalGrid,
    DryReferenceState,
    TerrainMetrics,
]:
    """Make the one explicit binary32 host authority consumed by both arms."""

    converted_state = PrognosticState(
        **{
            name: _as_f32_c(f"state.{name}", getattr(state, name))
            for name in _STATE_FIELDS
        },
        time_seconds=float(state.time_seconds),
    )
    converted_saved = DrySavedDiagnostics(
        **{
            name: _as_f32_c(f"saved.{name}", getattr(saved, name))
            for name in _SAVED_FIELDS
        }
    )
    converted_vertical = replace(
        vertical,
        **{
            name: _as_f32_c(f"vertical.{name}", getattr(vertical, name))
            for name in _VERTICAL_ARRAY_FIELDS
        },
    )
    converted_reference = DryReferenceState(
        **{
            name: _as_f32_c(f"reference.{name}", getattr(reference, name))
            for name in DryReferenceState.__slots__
        }
    )
    converted_terrain = TerrainMetrics(
        **{
            name: _as_f32_c(f"terrain.{name}", getattr(terrain, name))
            for name in TerrainMetrics.__slots__
        }
    )
    nlev = int(converted_state.rho.shape[0])
    converted_state.validate(
        n_cells=n_cells,
        n_edges=n_edges,
        n_vert_levels=nlev,
    )
    converted_saved.validate(converted_state.rho.shape, np.dtype(np.float32), n_edges)
    converted_reference.validate(converted_state.rho.shape)
    converted_terrain.validate(
        nlev=nlev,
        ncells=n_cells,
        max_edges=int(converted_terrain.zb_cell.shape[2]),
    )
    if converted_state.scalars.shape[0] != 1:
        raise ValueError("the GFS CUDA lane requires exactly one qv passive scalar")
    return (
        converted_state,
        converted_saved,
        converted_vertical,
        converted_reference,
        converted_terrain,
    )


def pinned_input_record(path: str | Path, pin: InputPin) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve(strict=True)
    if not selected.is_file():
        raise ValueError(f"pinned input is not a regular file: {selected}")
    if selected.name != pin.filename:
        raise ValueError(f"pinned input name {selected.name!r} != {pin.filename!r}")
    actual_bytes = selected.stat().st_size
    if actual_bytes != pin.bytes:
        raise ValueError(
            f"pinned input {selected.name} byte count {actual_bytes} != {pin.bytes}"
        )
    actual_sha256 = sha256_file(selected)
    if actual_sha256 != pin.sha256:
        raise ValueError(
            f"pinned input {selected.name} SHA-256 {actual_sha256} != {pin.sha256}"
        )
    return {"bytes": actual_bytes, "sha256": actual_sha256}


def exact_target(
    *,
    dt_seconds: float,
    duration_seconds: float,
    acoustic_substeps: int,
    levels: int,
    ztop_m: float,
    latlon_degrees: float,
) -> tuple[DryDycoreConfig, int]:
    expected = {
        "dt_seconds": TARGET_DT_SECONDS,
        "duration_seconds": TARGET_DURATION_SECONDS,
        "acoustic_substeps": TARGET_ACOUSTIC_SUBSTEPS,
        "levels": TARGET_LEVELS,
        "ztop_m": TARGET_ZTOP_M,
        "latlon_degrees": TARGET_LATLON_DEGREES,
    }
    actual = {
        "dt_seconds": float(dt_seconds),
        "duration_seconds": float(duration_seconds),
        "acoustic_substeps": int(acoustic_substeps),
        "levels": int(levels),
        "ztop_m": float(ztop_m),
        "latlon_degrees": float(latlon_degrees),
    }
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise ValueError(
                f"the pinned CUDA-GFS lane requires {name}={expected_value}, "
                f"got {actual[name]}"
            )
    steps = cpu_gfs.derive_step_count(duration_seconds, dt_seconds)
    if steps != TARGET_STEPS:
        raise ValueError(f"the pinned CUDA-GFS lane requires {TARGET_STEPS} steps")
    config = cpu_gfs.forecast_config(dt_seconds, acoustic_substeps)
    config.validate()
    return config, steps


def build_output_plan(
    artifact_root: str | Path,
    receipt_root: str | Path,
) -> OutputPlan:
    artifacts = Path(artifact_root).expanduser().resolve()
    receipts = Path(receipt_root).expanduser().resolve()
    return OutputPlan(
        artifact_root=artifacts,
        receipt_root=receipts,
        history=artifacts / f"{GFS_STEM}.history.nc",
        latlon=artifacts / f"{GFS_STEM}.latlon.nc",
        renderer_input=artifacts / f"{GFS_STEM}.rw-wrf2d.nc",
        renderer_store=artifacts / "s",
        renderer_outputs=artifacts / "p",
        capsule_a=receipts / f"{GFS_STEM}.arm-a.json",
        capsule_b=receipts / f"{GFS_STEM}.arm-b.json",
        comparison=receipts / f"{GFS_STEM}.comparison.json",
        receipt=receipts / f"{GFS_STEM}.json",
        checksums=receipts / "SHA256SUMS",
    )


def _require_beneath(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} must stay beneath {root}") from error


def validate_fresh_output_plan(plan: OutputPlan) -> None:
    _require_beneath(
        plan.artifact_root,
        ROOT / "artifacts" / "cuda-gfs",
        "artifact root",
    )
    _require_beneath(
        plan.receipt_root,
        ROOT / "receipts" / "cuda-gfs-forecast",
        "receipt root",
    )
    if plan.artifact_root.exists():
        raise FileExistsError(
            f"CUDA-GFS artifact root must be fresh: {plan.artifact_root}"
        )
    if plan.receipt_root.exists():
        raise FileExistsError(
            f"CUDA-GFS receipt root must be fresh: {plan.receipt_root}"
        )
    worst_renderer_path = len(str(plan.renderer_outputs)) + 1 + RENDERER_FILENAME_BUDGET
    if worst_renderer_path > WINDOWS_LEGACY_PATH_LIMIT:
        raise ValueError(
            "renderer output path exceeds the declared Windows-safe budget: "
            f"{worst_renderer_path}>{WINDOWS_LEGACY_PATH_LIMIT}"
        )


def prepare_gfs_case(
    source_path: str | Path,
    grid_path: str | Path,
    static_path: str | Path,
    config: DryDycoreConfig,
) -> PreparedGfsCase:
    source_file = Path(source_path).expanduser().resolve(strict=True)
    grid_file = Path(grid_path).expanduser().resolve(strict=True)
    static_file = Path(static_path).expanduser().resolve(strict=True)
    input_records = {
        "gfs_wps_intermediate": pinned_input_record(
            source_file, INPUT_PINS["gfs_wps_intermediate"]
        ),
        "x1_2562_grid": pinned_input_record(grid_file, INPUT_PINS["x1_2562_grid"]),
        "x1_2562_static": pinned_input_record(
            static_file, INPUT_PINS["x1_2562_static"]
        ),
    }

    source = load_structured_atmosphere(source_file)
    mesh = Mesh.from_netcdf(grid_file, static_file)
    output_mesh = Mesh.from_netcdf(grid_file)
    cpu_gfs.validate_output_mesh(mesh, output_mesh)
    if (
        int(mesh.dimensions["nCells"]) != 2_562
        or int(mesh.dimensions["nEdges"]) != 7_680
    ):
        raise ValueError("pinned x1.2562 topology dimensions changed")
    vertical = build_vertical_grid(
        mesh,
        np.asarray(mesh.ter, dtype=np.float64),
        n_vert_levels=TARGET_LEVELS,
        ztop=TARGET_ZTOP_M,
        smooth_surfaces=False,
    )
    vertical = cpu_gfs.normalize_runtime_vertical(vertical)
    vertical = normalize_cuda_vertical_sentinels(vertical)
    initialized = initialize_from_structured(source, mesh, vertical)
    terrain, coupling = cpu_gfs.build_order2_terrain_metrics(
        mesh,
        vertical,
        config.config_coef_3rd_order,
    )
    reference, saved = cpu_gfs.build_reference_and_sidecar(
        initialized,
        mesh,
        vertical,
        coupling,
    )
    state32, saved32, vertical32, reference32, terrain32 = (
        materialize_binary32_atmosphere(
            initialized.state,
            saved,
            vertical,
            reference,
            terrain,
            n_cells=int(mesh.dimensions["nCells"]),
            n_edges=int(mesh.dimensions["nEdges"]),
        )
    )
    if float(state32.time_seconds) != 0.0:
        raise ValueError("the pinned GFS initial state must start at model time zero")
    provenance = dict(source.provenance)
    if str(provenance.get("valid_time")) != "2026-03-26_00:00:00":
        raise ValueError("the pinned GFS valid time changed")
    if float(provenance.get("forecast_hour", np.nan)) != 0.0:
        raise ValueError("the pinned GFS input must be forecast hour zero")
    prepared = PreparedCudaInputs.validated(
        config=config,
        profile=PROFILE,
        target=TARGET,
        preparation_method=PREPARATION_METHOD,
        mesh=mesh,
        state=state32,
        vertical=vertical32,
        reference=reference32,
        saved_diagnostics=saved32,
        terrain_metrics=terrain32,
        input_bytes=input_records,
    )
    return PreparedGfsCase(
        cuda=prepared,
        output_mesh=output_mesh,
        source_provenance=provenance,
        input_records=input_records,
        grid_path=grid_file,
        static_path=static_file,
    )


def require_current_renderer(path: str | Path | None) -> RendererProbe:
    probe = discover_rust_renderer(path)
    if (
        probe.executable_sha256 != EXPECTED_RENDERER_SHA256
        or probe.executable_bytes != EXPECTED_RENDERER_BYTES
    ):
        raise ValueError(
            "the CUDA-GFS lane requires the exact generic-product Rust renderer "
            f"{EXPECTED_RENDERER_SHA256}/{EXPECTED_RENDERER_BYTES} bytes"
        )
    return probe


def _print_arm_steps(label: str, capsule: Mapping[str, Any]) -> None:
    for record in capsule["trajectory"]["step_records"]:
        print(
            json.dumps(
                {
                    "arm": label,
                    "step": record["step"],
                    "model_time_seconds": record["end_time_seconds"],
                    "snapshot_sha256": record["snapshot"]["sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def run_verified_cuda_arms(
    prepared: PreparedCudaInputs,
    config: DryDycoreConfig,
    *,
    steps: int,
    kernel_cache: Any,
    ftz_binding: Mapping[str, Any],
    comparison_authority: Mapping[str, Any],
    gpuwm_root: str | Path,
    plan: OutputPlan,
) -> VerifiedCudaRun:
    print(f"running CUDA arm A ({steps} resident full steps)", flush=True)
    arm_a = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    _print_arm_steps("a", arm_a.capsule)
    write_json_atomic(plan.capsule_a, arm_a.capsule)
    print(f"running CUDA arm B ({steps} resident full steps)", flush=True)
    arm_b = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    _print_arm_steps("b", arm_b.capsule)
    write_json_atomic(plan.capsule_b, arm_b.capsule)
    print("running gpuwm total capsule comparison", flush=True)
    comparison = compare_cuda_capsule_files(
        plan.capsule_a,
        plan.capsule_b,
        gpuwm_root=gpuwm_root,
        report_path=plan.comparison,
    )
    gpuwm = comparison.get("gpuwm_comparison", {})
    if comparison.get("total_comparison") is not True:
        raise RuntimeError("CUDA dual-run report is not a total comparison")
    if gpuwm.get("identical") is not True or gpuwm.get("divergence_count") != 0:
        raise RuntimeError(
            "CUDA arms diverged; forecast products are intentionally withheld"
        )
    return VerifiedCudaRun(
        arm_a=arm_a,
        arm_b=arm_b,
        comparison=comparison,
        capsule_a_path=plan.capsule_a,
        capsule_b_path=plan.capsule_b,
        comparison_path=plan.comparison,
    )


def download_final_atmosphere(atmosphere: Any) -> HostDownload:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError("CuPy disappeared before the final CUDA download") from error
    started = time.perf_counter()
    state = atmosphere.state.to_host()
    saved = DrySavedDiagnostics(
        **{name: cp.asnumpy(getattr(atmosphere.saved, name)) for name in _SAVED_FIELDS}
    )
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - started
    state.validate(
        n_cells=int(atmosphere.mesh.n_cells),
        n_edges=int(atmosphere.mesh.n_edges),
        n_vert_levels=int(atmosphere.vertical.n_vert_levels),
    )
    saved.validate(state.rho.shape, state.rho.dtype, int(atmosphere.mesh.n_edges))
    byte_count = sum(int(getattr(state, name).nbytes) for name in _STATE_FIELDS)
    byte_count += sum(int(getattr(saved, name).nbytes) for name in _SAVED_FIELDS)
    return HostDownload(state=state, saved=saved, bytes=byte_count, seconds=elapsed)


def _valid_times(provenance: Mapping[str, Any]) -> tuple[datetime, datetime]:
    valid_text = str(provenance.get("valid_time", ""))
    initial = datetime.fromisoformat(valid_text.replace("_", "T", 1))
    return initial, initial + timedelta(seconds=TARGET_DURATION_SECONDS)


def _repo_relative(path: str | Path) -> str:
    selected = Path(path).expanduser().resolve(strict=True)
    try:
        logical = selected.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"evidence artifact must stay beneath the repository: {selected}"
        ) from error
    parsed = PurePosixPath(logical)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"unsafe evidence artifact path: {logical}")
    return logical


def _file_record(path: str | Path) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve(strict=True)
    return {
        "path": _repo_relative(selected),
        "path_kind": "repo_relative",
        "bytes": selected.stat().st_size,
        "sha256": sha256_file(selected),
    }


def _renderer_product(path: Path, products: Sequence[str]) -> str:
    normalized = path.stem.lower().replace(":", "_").replace(".", "_")
    matches = [
        product
        for product in products
        if product.lower().split(":", 1)[-1].replace(".", "_") in normalized
    ]
    if len(matches) != 1:
        raise RuntimeError(f"renderer output does not bind one product: {path.name}")
    return matches[0]


def write_forecast_products(
    case: PreparedGfsCase,
    verified: VerifiedCudaRun,
    config: DryDycoreConfig,
    bounds: StabilityBounds,
    *,
    plan: OutputPlan,
    renderer_probe: RendererProbe,
    latlon_resolution: float,
    renderer_width: int,
    renderer_height: int,
) -> dict[str, Any]:
    """Download arm A and write products after, and only after, total equality."""

    if (
        verified.comparison.get("total_comparison") is not True
        or verified.comparison.get("gpuwm_comparison", {}).get("identical") is not True
    ):
        raise RuntimeError("forecast products require a verified total CUDA comparison")

    download = download_final_atmosphere(verified.arm_a.final_atmosphere)
    initial_state = case.cuda.state
    initial_saved = case.cuda.saved_diagnostics
    final_state = download.state
    final_saved = download.saved
    mesh = case.cuda.mesh
    vertical = case.cuda.vertical
    reference = case.cuda.reference
    terrain = case.cuda.terrain_metrics
    if final_state.time_seconds != initial_state.time_seconds + TARGET_DURATION_SECONDS:
        raise RuntimeError("final CUDA model time differs from the six-hour target")

    diagnostics = DryDycoreDriver(
        mesh,
        vertical,
        reference,
        config,
        terrain_metrics=terrain,
    )
    initial_metrics = asdict(diagnostics.metrics(initial_state))
    final_metrics = asdict(diagnostics.metrics(final_state))
    mass_drift, energy_drift = cpu_gfs.check_bounds(
        final_metrics,
        initial_metrics,
        bounds,
        np.asarray(final_state.scalars[0]),
    )
    coefficients = initialize_reconstruction_coefficients(mesh)
    initial_products = cpu_gfs.diagnose_products(
        initial_state,
        initial_saved,
        mesh,
        vertical,
        coefficients,
    )
    final_products = cpu_gfs.diagnose_products(
        final_state,
        final_saved,
        mesh,
        vertical,
        coefficients,
    )
    initial_time, final_time = _valid_times(case.source_provenance)

    history_fields = {
        "rho": HistoryField(
            np.stack((initial_state.rho, final_state.rho)),
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
            np.stack((initial_state.scalars[0], final_state.scalars[0])),
            ("Time", "nVertLevels", "nCells"),
            {
                "units": "kg kg-1",
                "long_name": "passively transported water-vapor mixing ratio",
            },
        ),
        "u": HistoryField(
            np.stack((initial_saved.normal_velocity, final_saved.normal_velocity)),
            ("Time", "nVertLevels", "nEdges"),
        ),
        "w": HistoryField(
            np.stack((initial_saved.vertical_velocity, final_saved.vertical_velocity)),
            ("Time", "nVertLevelsP1", "nCells"),
        ),
        "surface_pressure": HistoryField(
            np.stack(
                (
                    initial_products["surface_pressure"],
                    final_products["surface_pressure"],
                )
            ),
            ("Time", "nCells"),
        ),
        "temperature_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["temperature_lowest_model_level"],
                    final_products["temperature_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "K", "long_name": "temperature at lowest model level"},
        ),
        "wind_speed_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["wind_speed_lowest_model_level"],
                    final_products["wind_speed_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {
                "units": "m s-1",
                "long_name": "horizontal wind speed at lowest model level",
            },
        ),
    }
    write_history(
        plan.history,
        case.output_mesh,
        history_fields,
        (initial_time, final_time),
        initial_time=initial_time,
        time_seconds=(0.0, TARGET_DURATION_SECONDS),
        n_vert_levels=TARGET_LEVELS,
        global_attrs={
            "title": "Real GFS-initialized dual-arm CUDA MPAS-port dry forecast",
            "physics_suite": "none",
            "water_vapor_treatment": "passive scalar transport only",
            "cuda_dual_run": "gpuwm total capsule equality required before output",
        },
        stream_options=HistoryStreamOptions(clobber_mode="truncate"),
    )

    latitude = np.arange(
        -90.0,
        90.0 + 0.5 * latlon_resolution,
        latlon_resolution,
    )
    longitude = np.arange(0.0, 360.0, latlon_resolution)
    weights = build_regrid_weights(
        case.output_mesh,
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
        units = (
            "Pa"
            if name == "surface_pressure"
            else "K"
            if name.startswith("temperature")
            else "m s-1"
        )
        regrid_fields[name] = HistoryField(
            np.stack((initial_products[name], final_products[name])),
            ("Time", "nCells"),
            {"units": units},
        )
    write_regridded_netcdf(
        plan.latlon,
        weights,
        regrid_fields,
        cell_axis={name: 1 for name in regrid_fields},
        valid_time=(initial_time, final_time),
        initial_time=initial_time,
        global_attrs={
            "title": "Real GFS-initialized dual-arm CUDA MPAS-port products",
            "physics_suite": "none",
            "water_vapor_treatment": "passive scalar transport only",
        },
        clobber=True,
    )

    materialize_gfs_rust_input(
        plan.renderer_input,
        history_path=plan.history,
        latlon_path=plan.latlon,
        grid_path=case.grid_path,
        static_path=case.static_path,
        ztop_m=TARGET_ZTOP_M,
        clobber=False,
    )
    catalog = inspect_renderer_products(
        plan.renderer_input,
        store_root=plan.renderer_store,
        probe=renderer_probe,
    )
    renderer_run = render_catalogued_products(
        plan.renderer_input,
        store_root=plan.renderer_store,
        out_dir=plan.renderer_outputs,
        products=DEFAULT_PRODUCTS,
        probe=renderer_probe,
        catalog=catalog,
        frames="1",
        width=renderer_width,
        height=renderer_height,
        source_label="MPAS-Atmosphere CUDA dry port, dual-arm verified",
    )
    renderer_records = []
    for output, expected_sha256 in zip(
        renderer_run.outputs,
        renderer_run.output_sha256,
        strict=True,
    ):
        if len(str(output)) > WINDOWS_LEGACY_PATH_LIMIT:
            raise RuntimeError(
                "renderer output path is too long for strict Windows resolution: "
                f"{output}"
            )
        record = _file_record(output)
        if record["sha256"] != expected_sha256:
            raise RuntimeError("renderer output changed after RendererRun returned")
        record["product"] = _renderer_product(output, DEFAULT_PRODUCTS)
        renderer_records.append(record)
    renderer_records.sort(key=lambda record: str(record["product"]))
    if {record["product"] for record in renderer_records} != set(DEFAULT_PRODUCTS):
        raise RuntimeError(
            "renderer output inventory does not cover every selected product"
        )

    return {
        "download": {"bytes": download.bytes, "seconds": download.seconds},
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mass_relative_drift": mass_drift,
        "energy_proxy_relative_drift": energy_drift,
        "initial_state": initial_state,
        "final_state": final_state,
        "initial_products": initial_products,
        "final_products": final_products,
        "history": _file_record(plan.history),
        "latlon": _file_record(plan.latlon),
        "renderer_input": _file_record(plan.renderer_input),
        "renderer_outputs": renderer_records,
        "renderer_catalog": {
            "summary": renderer_run.catalog.summary,
            "elapsed_seconds": renderer_run.catalog.elapsed_seconds,
            "renderer_input_sha256": renderer_run.catalog.renderer_input_sha256,
        },
        "renderer_run": {
            "elapsed_seconds": renderer_run.elapsed_seconds,
            "renderer_sha256": renderer_run.renderer_sha256,
            "frames": renderer_run.frames,
            "width": renderer_run.width,
            "height": renderer_run.height,
            "source_label": renderer_run.source_label,
        },
    }


def _write_receipt(
    case: PreparedGfsCase,
    verified: VerifiedCudaRun,
    products: Mapping[str, Any],
    config: DryDycoreConfig,
    bounds: StabilityBounds,
    renderer_probe: RendererProbe,
    plan: OutputPlan,
    *,
    timing: Mapping[str, float],
) -> None:
    initial_state = products["initial_state"]
    final_state = products["final_state"]
    initial_products = products["initial_products"]
    final_products = products["final_products"]
    comparison = verified.comparison
    step_records = verified.arm_a.capsule["trajectory"]["step_records"]
    payload = {
        "schema": "mpas-port.cuda-gfs-forecast-receipt/v1",
        "evidence": {
            "status": "passed",
            "classification": "real GFS-initialized dual-arm CUDA forecast and Rust plot lane",
            "claim": (
                "the CUDA dry port advanced the same pinned binary32 GFS preparation "
                "twice, gpuwm compared every capsule leaf equal, and only then arm A "
                "was downloaded, diagnosed, written, regridded, and rendered"
            ),
            "non_claims": [
                "not a frozen-Fortran equivalence result",
                "not evidence of forecast skill",
                "not a timing or throughput benchmark",
                "no moist or column physics ran",
            ],
        },
        "inputs": {
            label: {
                "filename": INPUT_PINS[label].filename,
                **record,
            }
            for label, record in case.input_records.items()
        },
        "preparation": {
            "profile": PROFILE,
            "target": TARGET,
            "method": PREPARATION_METHOD,
            "host_float_dtype": "float32",
            "layout": "level-major C-contiguous with horizontal entity fastest",
            "same_prepared_object_passed_to_both_arms": True,
            "initial_execution_fingerprint_sha256": (
                case.cuda.expected_execution_fingerprint["sha256"]
            ),
            "source_adapter": case.source_provenance.get("adapter"),
            "source_valid_time": case.source_provenance.get("valid_time"),
        },
        "configuration": {
            "duration_seconds": TARGET_DURATION_SECONDS,
            "steps": TARGET_STEPS,
            "dt_seconds": TARGET_DT_SECONDS,
            "acoustic_substeps": TARGET_ACOUSTIC_SUBSTEPS,
            "vertical_levels": TARGET_LEVELS,
            "ztop_m": TARGET_ZTOP_M,
            "dry_dycore": asdict(config),
        },
        "physics": {
            "config_physics_suite": "none",
            "column_backend_executed": False,
            "qv_treatment": "passive scalar advection only",
        },
        "cuda_dual_run": {
            "total_comparison": comparison["total_comparison"],
            "gpuwm_comparison": comparison["gpuwm_comparison"],
            "comparison_authority": comparison["comparison_authority"],
            "capsule_a": _file_record(verified.capsule_a_path),
            "capsule_b": _file_record(verified.capsule_b_path),
            "comparison_report": _file_record(verified.comparison_path),
            "completed_steps": len(step_records),
            "step_snapshot_sha256": [
                record["snapshot"]["sha256"] for record in step_records
            ],
            "intermediate_numeric_scope": (
                "every state and sidecar field was downloaded, finite-checked, and "
                "hashed by each arm; declared stability bounds are evaluated at t0/t6"
            ),
        },
        "bounds": asdict(bounds),
        "observed": {
            "mass_relative_drift": products["mass_relative_drift"],
            "energy_proxy_relative_drift": products["energy_proxy_relative_drift"],
            "initial_metrics": products["initial_metrics"],
            "final_metrics": products["final_metrics"],
            "surface_pressure_change_pa": cpu_gfs.array_summary(
                final_products["surface_pressure"]
                - initial_products["surface_pressure"]
            ),
            "qv_final": cpu_gfs.array_summary(final_state.scalars[0]),
        },
        "state": {
            "initial": cpu_gfs.state_summary(initial_state),
            "final": cpu_gfs.state_summary(final_state),
        },
        "artifacts": {
            "history": {**products["history"], "records": 2},
            "latlon": {
                **products["latlon"],
                "records": 2,
                "resolution_degrees": TARGET_LATLON_DEGREES,
            },
            "rust_renderer_input": products["renderer_input"],
            "rust_renderer_pngs": products["renderer_outputs"],
        },
        "renderer": {
            "executable_name": renderer_probe.executable.name,
            "executable_path_recorded": False,
            "executable_bytes": renderer_probe.executable_bytes,
            "executable_sha256": renderer_probe.executable_sha256,
            "basemap_available": True,
            "basemap_path_recorded": False,
            "catalog": products["renderer_catalog"],
            "run": products["renderer_run"],
        },
        "final_download": products["download"],
        "runner": _file_record(Path(__file__)),
        "timing_seconds": dict(timing),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    write_json_atomic(plan.receipt, payload)
    targets = (
        plan.receipt,
        plan.capsule_a,
        plan.capsule_b,
        plan.comparison,
        plan.history,
        plan.latlon,
        plan.renderer_input,
    )
    renderer_targets = tuple(
        ROOT.joinpath(*PurePosixPath(record["path"]).parts)
        for record in products["renderer_outputs"]
    )
    cpu_gfs.write_checksum_inventory(
        plan.checksums,
        targets + renderer_targets,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact pinned six-hour real-GFS CUDA forecast twice, require "
            "total gpuwm equality, then write history/regrid/Rust products."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--gpuwm-root", type=Path, default=DEFAULT_GPUWM_ROOT)
    parser.add_argument("--gpuwm-probe", type=Path, default=DEFAULT_GPUWM_PROBE)
    parser.add_argument("--ftz-binding", type=Path, default=DEFAULT_FTZ_BINDING)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--receipt-root", type=Path, default=DEFAULT_RECEIPT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dt", type=float, default=TARGET_DT_SECONDS)
    parser.add_argument("--duration", type=float, default=TARGET_DURATION_SECONDS)
    parser.add_argument(
        "--acoustic-substeps", type=int, default=TARGET_ACOUSTIC_SUBSTEPS
    )
    parser.add_argument("--levels", type=int, default=TARGET_LEVELS)
    parser.add_argument("--ztop", type=float, default=TARGET_ZTOP_M)
    parser.add_argument(
        "--latlon-resolution", type=float, default=TARGET_LATLON_DEGREES
    )
    parser.add_argument("--max-mass-drift", type=float, default=2.0e-8)
    parser.add_argument("--max-energy-drift", type=float, default=0.50)
    parser.add_argument("--max-velocity", type=float, default=500.0)
    parser.add_argument("--min-density", type=float, default=1.0e-7)
    parser.add_argument("--renderer-width", type=int, default=1_200)
    parser.add_argument("--renderer-height", type=int, default=800)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="pin and materialize the host case without importing CuPy or writing evidence",
    )
    mode.add_argument(
        "--single-arm-smoke",
        action="store_true",
        help="run one nonpublishing CUDA arm; no capsule/product/receipt is written",
    )
    return parser


def execute(args: argparse.Namespace) -> int:
    config, steps = exact_target(
        dt_seconds=args.dt,
        duration_seconds=args.duration,
        acoustic_substeps=args.acoustic_substeps,
        levels=args.levels,
        ztop_m=args.ztop,
        latlon_degrees=args.latlon_resolution,
    )
    bounds = StabilityBounds(
        max_mass_relative_drift=args.max_mass_drift,
        max_energy_relative_drift=args.max_energy_drift,
        max_abs_velocity=args.max_velocity,
        min_density=args.min_density,
    )
    bounds.validate()
    if args.renderer_width < 64 or args.renderer_height < 64:
        raise ValueError("renderer dimensions must each be at least 64 pixels")

    plan = build_output_plan(args.artifact_root, args.receipt_root)
    if not args.prepare_only and not args.single_arm_smoke:
        validate_fresh_output_plan(plan)

    timing: dict[str, float] = {}
    started = time.perf_counter()
    checkpoint = time.perf_counter()
    print("pinning and materializing the real GFS host case", flush=True)
    case = prepare_gfs_case(args.source, args.grid, args.static, config)
    timing["host_preparation"] = time.perf_counter() - checkpoint
    if args.prepare_only:

        class _HostAtmosphere:
            state = case.cuda.state
            saved = case.cuda.saved_diagnostics

        snapshot = fingerprint_atmosphere(_HostAtmosphere())
        print(
            json.dumps(
                {
                    "mode": "prepare-only",
                    "profile": PROFILE,
                    "initial_snapshot_sha256": snapshot["sha256"],
                    "initial_execution_fingerprint_sha256": (
                        case.cuda.expected_execution_fingerprint["sha256"]
                    ),
                    "input_bytes": case.input_records,
                    "published": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    gpuwm_root = args.gpuwm_root.expanduser().resolve(strict=True)
    print("validating gpuwm comparator and FTZ/compile authority", flush=True)
    _, comparison_authority = load_gpuwm_dualrun(gpuwm_root)
    ftz_binding = load_ftz_binding_record(
        args.ftz_binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=args.gpuwm_probe,
    )
    renderer_probe = None
    if not args.single_arm_smoke:
        print("probing the exact generic-product Rust renderer", flush=True)
        renderer_probe = require_current_renderer(args.renderer)

    cache_root = args.cache_root.expanduser().resolve()
    _require_beneath(
        cache_root,
        ROOT / "work" / "cuda-gfs-forecast-cache",
        "CUDA cache root",
    )
    capability = require_cuda(
        min_compute=(12, 0),
        required_compute=(12, 0),
        cache_dir=cache_root,
    )
    checkpoint = time.perf_counter()
    print("compiling the one FTZ-bound executable shared by both arms", flush=True)
    kernel_cache = prepare_cuda_kernel_cache(
        capability,
        cache_root,
        ftz_binding=ftz_binding,
    )
    timing["compile"] = time.perf_counter() - checkpoint

    if args.single_arm_smoke:
        checkpoint = time.perf_counter()
        print(f"running one nonpublishing CUDA arm ({steps} full steps)", flush=True)
        arm = run_cuda_arm_generic(
            case.cuda,
            config,
            steps=steps,
            kernel_cache=kernel_cache,
            ftz_binding=ftz_binding,
            comparison_authority=comparison_authority,
        )
        timing["single_arm"] = time.perf_counter() - checkpoint
        print(
            json.dumps(
                {
                    "mode": "single-arm-smoke",
                    "profile": PROFILE,
                    "steps": steps,
                    "final_snapshot_sha256": arm.capsule["trajectory"][
                        "final_snapshot_sha256"
                    ],
                    "compile_manifest_sha256": arm.capsule["contracts"][
                        "compile_manifest"
                    ]["sha256"],
                    "published": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    checkpoint = time.perf_counter()
    verified = run_verified_cuda_arms(
        case.cuda,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
        gpuwm_root=gpuwm_root,
        plan=plan,
    )
    timing["dual_cuda_trajectory_and_compare"] = time.perf_counter() - checkpoint
    if renderer_probe is None:
        raise AssertionError("normal execution lost its renderer probe")
    checkpoint = time.perf_counter()
    print("dual run is identical; downloading arm A and writing products", flush=True)
    products = write_forecast_products(
        case,
        verified,
        config,
        bounds,
        plan=plan,
        renderer_probe=renderer_probe,
        latlon_resolution=args.latlon_resolution,
        renderer_width=args.renderer_width,
        renderer_height=args.renderer_height,
    )
    timing["diagnostics_outputs_and_rust"] = time.perf_counter() - checkpoint
    timing["total_before_receipt"] = time.perf_counter() - started
    _write_receipt(
        case,
        verified,
        products,
        config,
        bounds,
        renderer_probe,
        plan,
        timing=timing,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": _repo_relative(plan.receipt),
                "rust_png_count": len(products["renderer_outputs"]),
                "dual_run_identical": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
