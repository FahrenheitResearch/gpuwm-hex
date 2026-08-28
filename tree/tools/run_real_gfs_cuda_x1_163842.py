"""Run the serious-resolution x1.163842 real-GFS CUDA evidence lane.

The forecast target is deliberately fixed: the official 163,842-cell global
MPAS mesh, 55 vertical levels, a 360 second large step, six acoustic substeps,
and sixty resident steps per arm (six simulated hours).  A single explicit
level-major, C-contiguous binary32 host preparation is uploaded independently
to both CUDA arms.  No forecast field or renderer input is written until the
gpuwm total comparator reports equality of every capsule leaf.

The remote CUDA host does not carry the Windows Rust renderer.  Normal forecast
mode therefore writes the history, 0.5-degree lat/lon file, and exact
``rw_wrfbatch`` input after equality.  ``--render-only`` consumes that pinned
input on the renderer host and writes the six polished global products with a
separate byte-binding receipt.  This remains a dry/no-physics forecast; qv is a
passively transported scalar only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
TOOLS_ROOT = ROOT / "tools"
for import_root in (SOURCE_ROOT, TOOLS_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_real_gfs_cuda_forecast as coarse  # noqa: E402

from hexcore.cuda_backend import require_cuda  # noqa: E402
from hexcore.cuda_driver import CudaDryDycoreDriver  # noqa: E402
from hexcore.cuda_dualrun import (  # noqa: E402
    compare_cuda_capsule_files,
    fingerprint_array,
    fingerprint_atmosphere,
    fingerprint_prepared_execution,
    fingerprint_uploaded_execution,
    load_ftz_binding_record,
    load_gpuwm_dualrun,
    prepare_cuda_kernel_cache,
    run_cuda_arm_generic,
    write_json_atomic,
)
from hexcore.driver import (  # noqa: E402
    DryDycoreDriver,
    StabilityBounds,
)
from hexcore.initialization import (  # noqa: E402
    initialize_from_structured,
    load_structured_atmosphere,
)
from hexcore.mesh import (  # noqa: E402
    LONGITUDE_TRIG_EQUIVALENCE_ATOL as _CORE_LONGITUDE_TRIG_EQUIVALENCE_ATOL,
    Mesh,
    load_precision_preserving_mesh_pair,
)
from hexcore.output import (  # noqa: E402
    HistoryField,
    HistoryStreamOptions,
    write_history,
)
from hexcore.regrid import (  # noqa: E402
    build_regrid_weights,
    write_regridded_netcdf,
)
from hexcore.rust_renderer import (  # noqa: E402
    discover_rust_renderer,
    inspect_renderer_products,
    materialize_gfs_rust_input,
    render_catalogued_products,
    sha256_file,
    write_renderer_materialization_authority,
)
from hexcore.vector import initialize_reconstruction_coefficients  # noqa: E402
from hexcore.vertical import build_vertical_grid  # noqa: E402

InputPin = coarse.InputPin
VerifiedCudaRun = coarse.VerifiedCudaRun

GFS_STEM = "GFS-2026-03-26-00.x1.163842.cuda-port-sm120-dual-6h"
TARGET_DT_SECONDS = 360.0
TARGET_DURATION_SECONDS = 21_600.0
TARGET_STEPS = 60
TARGET_ACOUSTIC_SUBSTEPS = 6
TARGET_LEVELS = 55
TARGET_ZTOP_M = 30_000.0
TARGET_LATLON_DEGREES = 0.5
TARGET_CELLS = 163_842
TARGET_EDGES = 491_520
TARGET_VERTICES = 327_680
# FROZEN CLOSED-CASE PROOF CONSTANTS — the dt derivation below is a
# RETIRED METHOD, kept as the record of the runs it made (adjudicated
# 2026-08-25, stale-guard audit #347 unknowns).  This lane derives dt from
# ``nominalMinDc`` (6 s per nominal km -> 360 s), and #300 later measured
# nominalMinDc 10.5% / 23.6% / 39.7% HIGH as a length on three meshes --
# the unsafe direction.  The live rule (tools/mpas_mesh_binding.py, the
# versioned Courant policy: 125 m/s, 0.90) admits dt against the file's
# real ``dcEdge`` and never nominalMinDc.  Whether this lane's 360 s
# would survive that policy is NOT MEASURED: at the uniform-mesh bias
# (10.5%) the policy cap is ~391 s and 360 s passes; at the worst
# recorded bias (39.7%) the cap is ~309 s and 360 s fails.  Settling it
# needs one CPU read of min(dcEdge) from the pinned x1.163842 grid, and
# that file is absent from every reachable fleet box (either proving card,
# node-4 full-disk searched 2026-08-25; node-3 unreachable), so the tool
# cannot pass its own input pins today either.  If the mesh is ever
# re-fetched, the read is the named follow-up; until then these constants
# are the receipt-bearing record of the closed 2026-03-26 evidence
# campaign, and this dt rule must not be copied into live code.
TARGET_NOMINAL_MIN_DC_M = 60_000.0
DT_SECONDS_PER_NOMINAL_KM = 6.0
LONGITUDE_TRIG_EQUIVALENCE_ATOL = _CORE_LONGITUDE_TRIG_EQUIVALENCE_ATOL
PROFILE = "real-gfs-20260326-x1.163842-approximately-60km-dry-binary32"
TARGET = (
    "real GFS 2026-03-26 00Z x1.163842 approximately-60-km dry no-physics "
    "CUDA 6 h forecast"
)
PREPARATION_METHOD = (
    "load the three pinned real-GFS/x1.163842 inputs once, initialize with the "
    "shared structured-atmosphere path, and explicitly materialize one "
    "level-major C-contiguous binary32 host atmosphere for two independent "
    "CUDA uploads"
)

INPUT_PINS = {
    "gfs_wps_intermediate": InputPin(
        "GFS_2026-03-26_00",
        818_178_824,
        "98b773f981b08444ea838fab8e31568bd2e92ab7237b67ae056970518ccabd49",
    ),
    "x1_163842_grid": InputPin(
        "x1.163842.grid.nc",
        224_139_172,
        "e90ac8fadbe1330f3e4ef26bccb3125e37cc7031c70affeb68c2b69860c69c0c",
    ),
    "x1_163842_static": InputPin(
        "x1.163842.static.nc",
        328_730_104,
        "46eeae729da60c43c2c7a70535b71887186cc720725690af2029603d762a18b9",
    ),
}

# The pinned inputs and the comparator checkout live outside this repository and
# their locations are site-specific.  Baking in a root would point every other
# machine at a directory it does not have, so the defaults come from the
# environment and each unset value is refused by name at use.
_ASSET_ROOT = os.environ.get("MPAS_ASSET_ROOT")
_ASSET_ROOT_PATH = Path(_ASSET_ROOT) if _ASSET_ROOT else None
DEFAULT_SOURCE = (
    _ASSET_ROOT_PATH / "GFS_2026-03-26_00" if _ASSET_ROOT_PATH else None
)
DEFAULT_GRID = (
    _ASSET_ROOT_PATH / "x1.163842" / "mesh" / "x1.163842.grid.nc"
    if _ASSET_ROOT_PATH
    else None
)
DEFAULT_STATIC = (
    _ASSET_ROOT_PATH / "x1.163842" / "static" / "x1.163842.static.nc"
    if _ASSET_ROOT_PATH
    else None
)
_GPUWM_ROOT = os.environ.get("GPUWM_ROOT")
DEFAULT_GPUWM_ROOT = Path(_GPUWM_ROOT) if _GPUWM_ROOT else None
DEFAULT_GPUWM_PROBE = ROOT / "receipts" / "cuda-ftz-sm120-remote" / "gpuwm-probe"
DEFAULT_FTZ_BINDING = ROOT / "receipts" / "cuda-ftz-sm120-remote" / "binding.json"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "cuda-gfs" / "x1.163842-60km-final"
DEFAULT_RECEIPT_ROOT = ROOT / "receipts" / "cuda-gfs-forecast" / "x1.163842-60km-final"
DEFAULT_CACHE_ROOT = ROOT / "work" / "cuda-gfs-60km-cache" / "fresh"
DEFAULT_RENDER_ROOT = ROOT / "artifacts" / "cuda-gfs" / "x1.163842-60km-render"
DEFAULT_RENDER_RECEIPT = (
    ROOT / "receipts" / "cuda-gfs-forecast" / "x1.163842-60km-render" / "renderer.json"
)

_TOPOLOGY_FIELDS = (
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "cellsOnCell",
    "edgesOnCell",
    "verticesOnCell",
    "cellsOnEdge",
    "verticesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
    "edgesOnEdge",
    "indexToCellID",
    "indexToEdgeID",
    "indexToVertexID",
)
_COORDINATE_FIELDS = tuple(
    f"{component}{entity}"
    for entity in ("Cell", "Edge", "Vertex")
    for component in ("x", "y", "z", "lat", "lon")
)
_STATIC_METRIC_WITNESSES = (
    "dcEdge",
    "dvEdge",
    "areaCell",
    "areaTriangle",
    "kiteAreasOnVertex",
    "nominalMinDc",
    "ter",
)


@dataclass(frozen=True, slots=True)
class PreparedGfsCase:
    cuda: Any
    output_mesh: Mesh
    source_provenance: dict[str, Any]
    input_records: dict[str, dict[str, Any]]
    grid_path: Path
    static_path: Path
    mesh_merge: dict[str, Any]


@dataclass(frozen=True, slots=True)
class OutputPlan:
    artifact_root: Path
    receipt_root: Path
    history: Path
    latlon: Path
    renderer_input: Path
    renderer_authority: Path
    renderer_store: Path
    renderer_outputs: Path
    capsule_a: Path
    capsule_b: Path
    comparison: Path
    receipt: Path
    checksums: Path


@contextmanager
def _coarse_profile_binding() -> Iterator[None]:
    """Temporarily reuse generic coarse-runner helpers under this exact profile."""

    replacements = {
        "GFS_STEM": GFS_STEM,
        "TARGET_DT_SECONDS": TARGET_DT_SECONDS,
        "TARGET_DURATION_SECONDS": TARGET_DURATION_SECONDS,
        "TARGET_STEPS": TARGET_STEPS,
        "TARGET_ACOUSTIC_SUBSTEPS": TARGET_ACOUSTIC_SUBSTEPS,
        "TARGET_LEVELS": TARGET_LEVELS,
        "TARGET_ZTOP_M": TARGET_ZTOP_M,
        "TARGET_LATLON_DEGREES": TARGET_LATLON_DEGREES,
        "PROFILE": PROFILE,
        "TARGET": TARGET,
        "PREPARATION_METHOD": PREPARATION_METHOD,
        "INPUT_PINS": INPUT_PINS,
    }
    prior = {name: getattr(coarse, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(coarse, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(coarse, name, value)


def exact_target(
    *,
    dt_seconds: float,
    duration_seconds: float,
    acoustic_substeps: int,
    levels: int,
    ztop_m: float,
    latlon_degrees: float,
) -> tuple[Any, int]:
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
                f"the x1.163842 CUDA-GFS lane requires {name}={expected_value}, "
                f"got {actual[name]}"
            )
    steps = coarse.cpu_gfs.derive_step_count(duration_seconds, dt_seconds)
    if steps != TARGET_STEPS:
        raise ValueError(f"the x1.163842 lane requires {TARGET_STEPS} steps")
    config = coarse.cpu_gfs.forecast_config(dt_seconds, acoustic_substeps)
    config.validate()
    return config, steps


def build_output_plan(
    artifact_root: str | Path, receipt_root: str | Path
) -> OutputPlan:
    with _coarse_profile_binding():
        base = coarse.build_output_plan(artifact_root, receipt_root)
    return OutputPlan(
        artifact_root=base.artifact_root,
        receipt_root=base.receipt_root,
        history=base.history,
        latlon=base.latlon,
        renderer_input=base.renderer_input,
        renderer_authority=(base.artifact_root / f"{GFS_STEM}.renderer-authority.npz"),
        renderer_store=base.renderer_store,
        renderer_outputs=base.renderer_outputs,
        capsule_a=base.capsule_a,
        capsule_b=base.capsule_b,
        comparison=base.comparison,
        receipt=base.receipt,
        checksums=base.checksums,
    )


def validate_fresh_output_plan(plan: OutputPlan) -> None:
    with _coarse_profile_binding():
        coarse.validate_fresh_output_plan(plan)
    if plan.renderer_authority.exists():
        raise FileExistsError(
            "renderer materialization authority must be fresh: "
            f"{plan.renderer_authority}"
        )


def _array_bytes_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def merge_official_x1_mesh(
    grid_path: str | Path,
    static_path: str | Path,
) -> tuple[Mesh, Mesh, dict[str, Any]]:
    """Apply the generic core repair, then bind its exact x1.163842 witnesses."""

    merged, grid_mesh, core_evidence = load_precision_preserving_mesh_pair(
        grid_path, static_path
    )
    expected_raw_failure = (
        "MPAS mesh validation failed:\n"
        " - dvEdge disagrees with spherical vertex arc length"
    )
    raw_overlay = core_evidence["raw_static_overlay"]
    if raw_overlay != {"status": "failed", "error": expected_raw_failure}:
        raise ValueError(f"raw x1.163842 overlay mutation changed: {raw_overlay!r}")
    if (
        core_evidence["grid_sphere_radius"] != 1.0
        or core_evidence["static_sphere_radius"] != 6_371_229.0
    ):
        raise ValueError("official x1.163842 sphere radii changed")

    grid_integer_fields = tuple(
        sorted(
            name
            for name, value in grid_mesh.arrays.items()
            if np.asarray(value).dtype.kind in "iu"
        )
    )
    if set(grid_integer_fields) != set(_TOPOLOGY_FIELDS):
        raise ValueError(
            f"x1.163842 grid integer/topology inventory changed: {grid_integer_fields}"
        )
    topology: dict[str, Any] = {}
    for name in _TOPOLOGY_FIELDS:
        merged_value = np.asarray(merged.arrays[name])
        grid_value = np.asarray(grid_mesh.arrays[name])
        if not np.array_equal(merged_value, grid_value):
            raise ValueError(f"x1.163842 grid/static topology differs at {name}")
        merged_record = fingerprint_array(name, merged_value)
        grid_record = fingerprint_array(name, grid_value)
        if merged_record != grid_record:
            raise AssertionError(f"equal topology did not fingerprint equally: {name}")
        topology[name] = {"grid": grid_record, "merged": merged_record}

    metric_witnesses: dict[str, Any] = {}
    for name in _STATIC_METRIC_WITNESSES:
        if merged.variable_sources.get(name) != "static":
            raise ValueError(f"x1.163842 {name} is no longer sourced from static")
        metric_witnesses[name] = fingerprint_array(name, merged.arrays[name])

    aggregate = hashlib.sha256()
    for name in (*_TOPOLOGY_FIELDS, *_COORDINATE_FIELDS):
        aggregate.update(name.encode("ascii"))
        aggregate.update(_array_bytes_sha256(merged.arrays[name]).encode("ascii"))
    receipt = {
        **core_evidence,
        "schema": "mpas-port.x1.163842-precision-preserving-mesh-merge/v2",
        "core_schema": core_evidence["schema"],
        "raw_static_overlay_mutation": {
            "passed": False,
            "expected_failure": expected_raw_failure,
            "observed_failure": raw_overlay["error"],
        },
        "official_grid_validation": "passed",
        "topology_inventory_is_every_integer_grid_field": True,
        "topology": topology,
        "static_metric_witnesses_unchanged": metric_witnesses,
        "merged_topology_coordinate_sha256": aggregate.hexdigest(),
    }
    return merged, grid_mesh, receipt


def prepare_gfs_case(
    source_path: str | Path,
    grid_path: str | Path,
    static_path: str | Path,
    config: Any,
) -> PreparedGfsCase:
    source_file = Path(source_path).expanduser().resolve(strict=True)
    grid_file = Path(grid_path).expanduser().resolve(strict=True)
    static_file = Path(static_path).expanduser().resolve(strict=True)
    input_records = {
        "gfs_wps_intermediate": coarse.pinned_input_record(
            source_file, INPUT_PINS["gfs_wps_intermediate"]
        ),
        "x1_163842_grid": coarse.pinned_input_record(
            grid_file, INPUT_PINS["x1_163842_grid"]
        ),
        "x1_163842_static": coarse.pinned_input_record(
            static_file, INPUT_PINS["x1_163842_static"]
        ),
    }

    source = load_structured_atmosphere(source_file)
    mesh, output_mesh, mesh_merge = merge_official_x1_mesh(grid_file, static_file)
    coarse.cpu_gfs.validate_output_mesh(mesh, output_mesh)
    dimensions = {
        name: int(mesh.dimensions[name]) for name in ("nCells", "nEdges", "nVertices")
    }
    expected = {
        "nCells": TARGET_CELLS,
        "nEdges": TARGET_EDGES,
        "nVertices": TARGET_VERTICES,
    }
    if dimensions != expected:
        raise ValueError(
            f"pinned x1.163842 topology changed: {dimensions} != {expected}"
        )
    nominal_min_dc = float(np.asarray(mesh.nominalMinDc))
    if nominal_min_dc != TARGET_NOMINAL_MIN_DC_M:
        raise ValueError(
            "pinned x1.163842 nominalMinDc changed: "
            f"{nominal_min_dc} != {TARGET_NOMINAL_MIN_DC_M}"
        )
    declared_dt = DT_SECONDS_PER_NOMINAL_KM * (nominal_min_dc / 1_000.0)
    if float(config.config_dt) != declared_dt:
        raise ValueError(
            "x1.163842 dt must equal 6 seconds per nominal kilometer: "
            f"{config.config_dt} != {declared_dt}"
        )

    vertical = build_vertical_grid(
        mesh,
        np.asarray(mesh.ter, dtype=np.float64),
        n_vert_levels=TARGET_LEVELS,
        ztop=TARGET_ZTOP_M,
        smooth_surfaces=False,
    )
    vertical = coarse.cpu_gfs.normalize_runtime_vertical(vertical)
    vertical = coarse.normalize_cuda_vertical_sentinels(vertical)
    initialized = initialize_from_structured(source, mesh, vertical)
    terrain, coupling = coarse.cpu_gfs.build_order2_terrain_metrics(
        mesh,
        vertical,
        config.config_coef_3rd_order,
    )
    reference, saved = coarse.cpu_gfs.build_reference_and_sidecar(
        initialized,
        mesh,
        vertical,
        coupling,
    )
    state32, saved32, vertical32, reference32, terrain32 = (
        coarse.materialize_binary32_atmosphere(
            initialized.state,
            saved,
            vertical,
            reference,
            terrain,
            n_cells=TARGET_CELLS,
            n_edges=TARGET_EDGES,
        )
    )
    if float(state32.time_seconds) != 0.0:
        raise ValueError("the pinned GFS initial state must start at model time zero")
    provenance = dict(source.provenance)
    if str(provenance.get("valid_time")) != "2026-03-26_00:00:00":
        raise ValueError("the pinned GFS valid time changed")
    if float(provenance.get("forecast_hour", np.nan)) != 0.0:
        raise ValueError("the pinned GFS input must be forecast hour zero")

    prepared = coarse.PreparedCudaInputs.validated(
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
        mesh_merge=mesh_merge,
    )


def _gpu_memory_record(cp: Any) -> dict[str, int]:
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    pool = cp.get_default_memory_pool()
    return {
        "device_total_bytes": int(total_bytes),
        "device_free_bytes": int(free_bytes),
        "device_used_bytes": int(total_bytes - free_bytes),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
    }


def run_resident_performance_probe(
    prepared: Any,
    config: Any,
    bounds: StabilityBounds,
    *,
    kernel_cache: Any,
    steps: int,
) -> dict[str, Any]:
    """Measure resident device steps without the capsule's per-step D2H hashing."""

    if steps < 1 or steps > 5:
        raise ValueError("the bounded resident probe requires 1..5 steps")
    import cupy as cp

    cp.cuda.get_current_stream().synchronize()
    before = _gpu_memory_record(cp)
    sealed_execution = prepared.expected_execution_fingerprint
    if not isinstance(sealed_execution, Mapping):
        raise RuntimeError("resident probe has no sealed execution-input fingerprint")
    verification_started = time.perf_counter()
    current_host_execution = fingerprint_prepared_execution(prepared, config)
    if current_host_execution != sealed_execution:
        raise RuntimeError("resident probe host inputs changed after preparation seal")
    upload_started = time.perf_counter()
    driver = CudaDryDycoreDriver.from_host(
        prepared.mesh,
        prepared.state,
        prepared.vertical,
        prepared.reference,
        config,
        saved_diagnostics=prepared.saved_diagnostics,
        terrain_metrics=prepared.terrain_metrics,
        kernel_cache=kernel_cache,
    )
    cp.cuda.get_current_stream().synchronize()
    upload_seconds = time.perf_counter() - upload_started
    uploaded_execution = fingerprint_uploaded_execution(driver)
    if uploaded_execution != sealed_execution:
        raise RuntimeError(
            "resident probe upload changed the complete prepared execution input"
        )
    execution_input_verification_seconds = time.perf_counter() - verification_started
    after_upload = _gpu_memory_record(cp)

    device_milliseconds: list[float] = []
    wall_seconds: list[float] = []
    memory_samples = [before, after_upload]
    for _index in range(steps):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        started = time.perf_counter()
        start_event.record()
        result = driver.step_device()
        end_event.record()
        end_event.synchronize()
        elapsed = time.perf_counter() - started
        if result.receipt.d2h.bytes != 0:
            raise RuntimeError("resident performance probe performed an internal D2H")
        driver.atmosphere = result.atmosphere
        device_milliseconds.append(
            float(cp.cuda.get_elapsed_time(start_event, end_event))
        )
        wall_seconds.append(float(elapsed))
        memory_samples.append(_gpu_memory_record(cp))

    download_started = time.perf_counter()
    download = coarse.download_final_atmosphere(driver.atmosphere)
    download_seconds = time.perf_counter() - download_started

    class _HostAtmosphere:
        state = download.state
        saved = download.saved

    fingerprint_started = time.perf_counter()
    final_snapshot = fingerprint_atmosphere(_HostAtmosphere())
    fingerprint_seconds = time.perf_counter() - fingerprint_started
    diagnostics = DryDycoreDriver(
        prepared.mesh,
        prepared.vertical,
        prepared.reference,
        config,
        terrain_metrics=prepared.terrain_metrics,
    )
    initial_metrics = asdict(diagnostics.metrics(prepared.state))
    final_metrics = asdict(diagnostics.metrics(download.state))
    mass_drift, energy_drift = coarse.cpu_gfs.check_bounds(
        final_metrics,
        initial_metrics,
        bounds,
        np.asarray(download.state.scalars[0]),
    )
    area = np.asarray(prepared.mesh.areaCell, dtype=np.float64)
    dzw = np.asarray(prepared.vertical.dzw, dtype=np.float64)

    def qv_mass(state: Any) -> float:
        rho = np.asarray(state.rho, dtype=np.float64)
        qv = np.asarray(state.scalars[0], dtype=np.float64)
        return float(np.sum(rho * qv * dzw[:, None] * area[None, :], dtype=np.float64))

    initial_qv_mass = qv_mass(prepared.state)
    final_qv_mass = qv_mass(download.state)
    qv_mass_drift = abs(final_qv_mass - initial_qv_mass) / max(
        abs(initial_qv_mass), np.finfo(np.float64).tiny
    )
    wall_total = float(sum(wall_seconds))
    simulated_seconds = float(steps * config.config_dt)
    return {
        "schema": "mpas-port.cuda-resident-performance-probe/v1",
        "steps": steps,
        "simulated_seconds": simulated_seconds,
        "upload_seconds": upload_seconds,
        "execution_input_binding": {
            "status": "passed",
            "sealed_sha256": sealed_execution["sha256"],
            "host_recomputed_sha256": current_host_execution["sha256"],
            "uploaded_sha256": uploaded_execution["sha256"],
            "verified_before_step_timing": True,
            "verification_seconds_excluded_from_step_timing": (
                execution_input_verification_seconds
            ),
        },
        "device_milliseconds": device_milliseconds,
        "device_milliseconds_mean": float(np.mean(device_milliseconds)),
        "wall_seconds": wall_seconds,
        "wall_seconds_total": wall_total,
        "wall_seconds_mean": float(np.mean(wall_seconds)),
        "simulated_hours_per_wall_hour": simulated_seconds / wall_total,
        "post_probe_download_seconds_excluded_from_timing": download_seconds,
        "post_probe_fingerprint_seconds_excluded_from_timing": fingerprint_seconds,
        "final_snapshot_sha256": final_snapshot["sha256"],
        "stability": {
            "status": "passed",
            "bounds": asdict(bounds),
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "mass_relative_drift": mass_drift,
            "energy_proxy_relative_drift": energy_drift,
            "qv_mass_initial": initial_qv_mass,
            "qv_mass_final": final_qv_mass,
            "qv_mass_relative_drift": qv_mass_drift,
        },
        "memory": {
            "before_upload": before,
            "after_upload": after_upload,
            "peak_sampled_device_used_bytes": max(
                sample["device_used_bytes"] for sample in memory_samples
            ),
            "peak_cupy_pool_total_bytes": max(
                sample["cupy_pool_total_bytes"] for sample in memory_samples
            ),
        },
    }


def run_verified_cuda_arms_profiled(
    prepared: Any,
    config: Any,
    *,
    steps: int,
    kernel_cache: Any,
    ftz_binding: Mapping[str, Any],
    comparison_authority: Mapping[str, Any],
    gpuwm_root: str | Path,
    plan: OutputPlan,
) -> tuple[VerifiedCudaRun, dict[str, Any]]:
    import cupy as cp

    print(f"running x1.163842 CUDA arm A ({steps} resident full steps)", flush=True)
    started = time.perf_counter()
    arm_a = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    arm_a_seconds = time.perf_counter() - started
    memory_after_a = _gpu_memory_record(cp)
    coarse._print_arm_steps("a", arm_a.capsule)
    write_json_atomic(plan.capsule_a, arm_a.capsule)

    print(f"running x1.163842 CUDA arm B ({steps} resident full steps)", flush=True)
    started = time.perf_counter()
    arm_b = run_cuda_arm_generic(
        prepared,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    )
    arm_b_seconds = time.perf_counter() - started
    memory_after_b = _gpu_memory_record(cp)
    coarse._print_arm_steps("b", arm_b.capsule)
    write_json_atomic(plan.capsule_b, arm_b.capsule)

    print("running gpuwm total capsule comparison", flush=True)
    started = time.perf_counter()
    comparison = compare_cuda_capsule_files(
        plan.capsule_a,
        plan.capsule_b,
        gpuwm_root=gpuwm_root,
        report_path=plan.comparison,
    )
    comparison_seconds = time.perf_counter() - started
    gpuwm = comparison.get("gpuwm_comparison", {})
    if comparison.get("total_comparison") is not True:
        raise RuntimeError("CUDA dual-run report is not a total comparison")
    if gpuwm.get("identical") is not True or gpuwm.get("divergence_count") != 0:
        raise RuntimeError("CUDA arms diverged; all forecast products are withheld")

    verified = VerifiedCudaRun(
        arm_a=arm_a,
        arm_b=arm_b,
        comparison=comparison,
        capsule_a_path=plan.capsule_a,
        capsule_b_path=plan.capsule_b,
        comparison_path=plan.comparison,
    )
    profile = {
        "arm_a_seconds_including_per_step_evidence_d2h": arm_a_seconds,
        "arm_b_seconds_including_per_step_evidence_d2h": arm_b_seconds,
        "comparison_seconds": comparison_seconds,
        "arm_a_simulated_hours_per_wall_hour_with_evidence": (
            TARGET_DURATION_SECONDS / arm_a_seconds
        ),
        "arm_b_simulated_hours_per_wall_hour_with_evidence": (
            TARGET_DURATION_SECONDS / arm_b_seconds
        ),
        "memory_after_arm_a": memory_after_a,
        "memory_after_arm_b": memory_after_b,
    }
    return verified, profile


def _valid_times(provenance: Mapping[str, Any]) -> tuple[datetime, datetime]:
    initial = datetime.fromisoformat(str(provenance["valid_time"]).replace("_", "T", 1))
    return initial, initial + timedelta(seconds=TARGET_DURATION_SECONDS)


def _qv_mass(case: PreparedGfsCase, state: Any) -> float:
    area = np.asarray(case.cuda.mesh.areaCell, dtype=np.float64)
    dzw = np.asarray(case.cuda.vertical.dzw, dtype=np.float64)
    rho = np.asarray(state.rho, dtype=np.float64)
    qv = np.asarray(state.scalars[0], dtype=np.float64)
    return float(np.sum(rho * qv * dzw[:, None] * area[None, :], dtype=np.float64))


def write_forecast_fields(
    case: PreparedGfsCase,
    verified: VerifiedCudaRun,
    config: Any,
    bounds: StabilityBounds,
    *,
    plan: OutputPlan,
) -> dict[str, Any]:
    """Write forecast fields and renderer input only after total equality."""

    if (
        verified.comparison.get("total_comparison") is not True
        or verified.comparison.get("gpuwm_comparison", {}).get("identical") is not True
    ):
        raise RuntimeError("forecast products require a verified total CUDA comparison")

    download = coarse.download_final_atmosphere(verified.arm_a.final_atmosphere)
    initial_state = case.cuda.state
    initial_saved = case.cuda.saved_diagnostics
    final_state = download.state
    final_saved = download.saved
    mesh = case.cuda.mesh
    vertical = case.cuda.vertical
    if final_state.time_seconds != TARGET_DURATION_SECONDS:
        raise RuntimeError("final CUDA model time differs from the six-hour target")

    diagnostics = DryDycoreDriver(
        mesh,
        vertical,
        case.cuda.reference,
        config,
        terrain_metrics=case.cuda.terrain_metrics,
    )
    initial_metrics = asdict(diagnostics.metrics(initial_state))
    final_metrics = asdict(diagnostics.metrics(final_state))
    mass_drift, energy_drift = coarse.cpu_gfs.check_bounds(
        final_metrics,
        initial_metrics,
        bounds,
        np.asarray(final_state.scalars[0]),
    )
    initial_qv_mass = _qv_mass(case, initial_state)
    final_qv_mass = _qv_mass(case, final_state)
    qv_mass_drift = abs(final_qv_mass - initial_qv_mass) / max(
        abs(initial_qv_mass), np.finfo(np.float64).tiny
    )

    coefficients = initialize_reconstruction_coefficients(mesh)
    initial_products = coarse.cpu_gfs.diagnose_products(
        initial_state, initial_saved, mesh, vertical, coefficients
    )
    final_products = coarse.cpu_gfs.diagnose_products(
        final_state, final_saved, mesh, vertical, coefficients
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
            {"units": "kg kg-1", "long_name": "passively transported water vapor"},
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
            {"units": "K"},
        ),
        "wind_speed_lowest_model_level": HistoryField(
            np.stack(
                (
                    initial_products["wind_speed_lowest_model_level"],
                    final_products["wind_speed_lowest_model_level"],
                )
            ),
            ("Time", "nCells"),
            {"units": "m s-1"},
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
            "title": "Real GFS x1.163842 dual-arm CUDA MPAS-port dry forecast",
            "nominal_resolution": "approximately 60 km global",
            "physics_suite": "none",
            "water_vapor_treatment": "passive scalar transport only",
            "cuda_dual_run": "gpuwm total capsule equality required before output",
        },
        stream_options=HistoryStreamOptions(clobber_mode="truncate"),
    )

    latitude = np.arange(
        -90.0, 90.0 + 0.5 * TARGET_LATLON_DEGREES, TARGET_LATLON_DEGREES
    )
    longitude = np.arange(0.0, 360.0, TARGET_LATLON_DEGREES)
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
            "title": "Real GFS x1.163842 dual-arm CUDA MPAS-port products",
            "nominal_resolution": "approximately 60 km global",
            "physics_suite": "none",
        },
        clobber=True,
    )
    write_renderer_materialization_authority(
        plan.renderer_authority,
        weights=weights,
        dynamics_mesh=case.cuda.mesh,
        output_mesh=case.output_mesh,
        grid_path=case.grid_path,
        static_path=case.static_path,
    )
    renderer_authority_sha256 = sha256_file(plan.renderer_authority)
    materialize_gfs_rust_input(
        plan.renderer_input,
        history_path=plan.history,
        latlon_path=plan.latlon,
        grid_path=case.grid_path,
        static_path=case.static_path,
        dynamics_mesh=case.cuda.mesh,
        output_mesh=case.output_mesh,
        regrid_weights_path=plan.renderer_authority,
        expected_regrid_weights_sha256=renderer_authority_sha256,
        ztop_m=TARGET_ZTOP_M,
        clobber=False,
    )
    return {
        "download": {"bytes": download.bytes, "seconds": download.seconds},
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "mass_relative_drift": mass_drift,
        "energy_proxy_relative_drift": energy_drift,
        "qv_mass_initial": initial_qv_mass,
        "qv_mass_final": final_qv_mass,
        "qv_mass_relative_drift": qv_mass_drift,
        "initial_state": initial_state,
        "final_state": final_state,
        "initial_products": initial_products,
        "final_products": final_products,
        "history": coarse._file_record(plan.history),
        "latlon": coarse._file_record(plan.latlon),
        "renderer_authority": coarse._file_record(plan.renderer_authority),
        "renderer_input": coarse._file_record(plan.renderer_input),
    }


def write_forecast_receipt(
    case: PreparedGfsCase,
    verified: VerifiedCudaRun,
    products: Mapping[str, Any],
    config: Any,
    bounds: StabilityBounds,
    plan: OutputPlan,
    *,
    timing: Mapping[str, Any],
    resident_probe: Mapping[str, Any],
    dual_profile: Mapping[str, Any],
) -> None:
    capsule = verified.arm_a.capsule
    compile_record = capsule["contracts"]["compile_manifest"]
    ftz_record = capsule["contracts"]["ftz_binding"]
    payload = {
        "schema": "mpas-port.cuda-gfs-x1.163842-forecast-receipt/v1",
        "evidence": {
            "status": "passed",
            "claim": (
                "the CUDA dry port advanced one pinned x1.163842 binary32 GFS "
                "preparation for sixty steps in each of two independent uploads; "
                "gpuwm compared every capsule leaf equal before any forecast field "
                "or Rust renderer input was written"
            ),
            "non_claims": [
                "not a frozen-Fortran equivalence result",
                "not evidence of forecast skill",
                "no moist, microphysics, PBL, land, radiation, or cumulus physics ran",
                "qv is passive transport only",
            ],
        },
        "inputs": {
            label: {"filename": INPUT_PINS[label].filename, **record}
            for label, record in case.input_records.items()
        },
        "mesh": {
            "name": "x1.163842",
            "nominal_resolution": "approximately 60 km global",
            "nCells": TARGET_CELLS,
            "nEdges": TARGET_EDGES,
            "nVertices": TARGET_VERTICES,
            "nVertLevels": TARGET_LEVELS,
            "nominalMinDc_m": TARGET_NOMINAL_MIN_DC_M,
            "dt_rule_seconds_per_nominal_km": DT_SECONDS_PER_NOMINAL_KM,
            "dt_from_nominalMinDc_seconds": (
                DT_SECONDS_PER_NOMINAL_KM * (TARGET_NOMINAL_MIN_DC_M / 1_000.0)
            ),
            "precision_preserving_merge": case.mesh_merge,
        },
        "preparation": {
            "profile": PROFILE,
            "target": TARGET,
            "method": PREPARATION_METHOD,
            "host_float_dtype": "float32",
            "layout": "[nVertLevels, horizontal entity], C-contiguous, entity fastest",
            "same_prepared_object_passed_to_both_arms": True,
            "initial_execution_fingerprint_sha256": (
                case.cuda.expected_execution_fingerprint["sha256"]
            ),
            "source_provenance": case.source_provenance,
        },
        "configuration": {
            "duration_seconds": TARGET_DURATION_SECONDS,
            "steps": TARGET_STEPS,
            "dt_seconds": TARGET_DT_SECONDS,
            "acoustic_substeps": TARGET_ACOUSTIC_SUBSTEPS,
            "ztop_m": TARGET_ZTOP_M,
            "dry_dycore": asdict(config),
        },
        "physics": {
            "suite": "none",
            "column_backend_executed": False,
            "qv_treatment": "passive scalar advection only",
        },
        "cuda_contracts": {
            "compile_manifest_sha256": compile_record["sha256"],
            "compile_manifest": compile_record["value"],
            "ftz_binding_sha256": ftz_record["sha256"],
            "ftz_binding_artifact_sha256": ftz_record["artifact_sha256"],
            "layout": capsule["contracts"]["layout"],
            "device": capsule["device"],
            "initial_execution_fingerprint_sha256": capsule["trajectory"][
                "initial_execution_fingerprint"
            ]["sha256"],
        },
        "cuda_dual_run": {
            "total_comparison": verified.comparison["total_comparison"],
            "gpuwm_comparison": verified.comparison["gpuwm_comparison"],
            "comparison_authority": verified.comparison["comparison_authority"],
            "capsule_a": coarse._file_record(verified.capsule_a_path),
            "capsule_b": coarse._file_record(verified.capsule_b_path),
            "comparison_report": coarse._file_record(verified.comparison_path),
            "completed_steps_per_arm": TARGET_STEPS,
        },
        "performance": {
            "resident_no_intermediate_d2h_probe": dict(resident_probe),
            "certified_dual_profile": dict(dual_profile),
            "timing_seconds": dict(timing),
            "interpretation": (
                "resident probe is the GPU forecast-rate number; certified arm timing "
                "also includes a roughly full-state D2H finite-check and hash each step"
            ),
        },
        "bounds": asdict(bounds),
        "observed": {
            "mass_relative_drift": products["mass_relative_drift"],
            "energy_proxy_relative_drift": products["energy_proxy_relative_drift"],
            "qv_mass_initial": products["qv_mass_initial"],
            "qv_mass_final": products["qv_mass_final"],
            "qv_mass_relative_drift": products["qv_mass_relative_drift"],
            "initial_metrics": products["initial_metrics"],
            "final_metrics": products["final_metrics"],
            "surface_pressure_change_pa": coarse.cpu_gfs.array_summary(
                products["final_products"]["surface_pressure"]
                - products["initial_products"]["surface_pressure"]
            ),
            "qv_final": coarse.cpu_gfs.array_summary(
                products["final_state"].scalars[0]
            ),
        },
        "state": {
            "initial": coarse.cpu_gfs.state_summary(products["initial_state"]),
            "final": coarse.cpu_gfs.state_summary(products["final_state"]),
        },
        "artifacts": {
            "history": {**products["history"], "records": 2},
            "latlon": {
                **products["latlon"],
                "records": 2,
                "resolution_degrees": TARGET_LATLON_DEGREES,
            },
            "renderer_materialization_authority": products["renderer_authority"],
            "rust_renderer_input": products["renderer_input"],
            "rust_renderer_pngs": "withheld here; produced by --render-only after this receipt is green",
        },
        "runner": coarse._file_record(Path(__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    write_json_atomic(plan.receipt, payload)
    coarse.cpu_gfs.write_checksum_inventory(
        plan.checksums,
        (
            plan.receipt,
            plan.capsule_a,
            plan.capsule_b,
            plan.comparison,
            plan.history,
            plan.latlon,
            plan.renderer_authority,
            plan.renderer_input,
        ),
    )


def render_existing_input(
    renderer_input: str | Path,
    renderer: str | Path | None,
    output_root: str | Path,
    receipt_path: str | Path,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    source = Path(renderer_input).expanduser().resolve(strict=True)
    outputs = Path(output_root).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    if outputs.exists():
        raise FileExistsError(f"renderer output root must be fresh: {outputs}")
    if receipt.exists():
        raise FileExistsError(f"renderer receipt must be fresh: {receipt}")
    probe = discover_rust_renderer(renderer)
    store = outputs / "s"
    png_root = outputs / "p"
    catalog = inspect_renderer_products(source, store_root=store, probe=probe)
    run = render_catalogued_products(
        source,
        store_root=store,
        out_dir=png_root,
        products=coarse.DEFAULT_PRODUCTS,
        probe=probe,
        catalog=catalog,
        frames="1",
        width=width,
        height=height,
        source_label="MPAS x1.163842 (~60 km) CUDA dry port, dual-arm verified",
    )
    product_records = []
    for output, expected_sha in zip(run.outputs, run.output_sha256, strict=True):
        record = coarse._file_record(output)
        if record["sha256"] != expected_sha:
            raise RuntimeError("Rust renderer output changed after return")
        record["product"] = coarse._renderer_product(output, coarse.DEFAULT_PRODUCTS)
        product_records.append(record)
    payload = {
        "schema": "mpas-port.cuda-gfs-x1.163842-rust-renderer-receipt/v1",
        "status": "passed",
        "forecast_input": coarse._file_record(source),
        "renderer": {
            "executable_bytes": probe.executable_bytes,
            "executable_sha256": probe.executable_sha256,
            "catalog_summary": run.catalog.summary,
            "elapsed_seconds": run.elapsed_seconds,
            "width": width,
            "height": height,
        },
        "products": sorted(product_records, key=lambda item: str(item["product"])),
        "physics": "none; dry dynamics with passive qv only",
    }
    write_json_atomic(receipt, payload)
    checksum_path = receipt.parent / "SHA256SUMS"
    coarse.cpu_gfs.write_checksum_inventory(
        checksum_path,
        (receipt, source, *run.outputs),
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--gpuwm-root", type=Path, default=DEFAULT_GPUWM_ROOT)
    parser.add_argument("--gpuwm-probe", type=Path, default=DEFAULT_GPUWM_PROBE)
    parser.add_argument("--ftz-binding", type=Path, default=DEFAULT_FTZ_BINDING)
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
    parser.add_argument("--probe-steps", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--probe-only", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    parser.add_argument("--renderer-input", type=Path)
    parser.add_argument("--renderer", type=Path)
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--render-receipt", type=Path, default=DEFAULT_RENDER_RECEIPT)
    parser.add_argument("--renderer-width", type=int, default=1_600)
    parser.add_argument("--renderer-height", type=int, default=1_000)
    return parser


def execute(args: argparse.Namespace) -> int:
    if args.render_only:
        if args.renderer_input is None:
            raise ValueError("--render-only requires --renderer-input")
        payload = render_existing_input(
            args.renderer_input,
            args.renderer,
            args.render_root,
            args.render_receipt,
            width=args.renderer_width,
            height=args.renderer_height,
        )
        print(json.dumps({"status": "passed", "pngs": len(payload["products"])}))
        return 0

    for value, flag in (
        (args.source, "--source"),
        (args.grid, "--grid"),
        (args.static, "--static"),
    ):
        if value is None:
            raise ValueError(
                f"{flag} is required: pass it explicitly, or set MPAS_ASSET_ROOT to "
                "the root holding the pinned GFS and x1.163842 inputs"
            )

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
    plan = build_output_plan(args.artifact_root, args.receipt_root)
    if not args.prepare_only and not args.probe_only:
        validate_fresh_output_plan(plan)

    timing: dict[str, Any] = {}
    total_started = time.perf_counter()
    started = time.perf_counter()
    print("preparing the x1.163842 real-GFS host authority", flush=True)
    case = prepare_gfs_case(args.source, args.grid, args.static, config)
    timing["host_preparation"] = time.perf_counter() - started
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
                    "mesh": {"nCells": TARGET_CELLS, "nEdges": TARGET_EDGES},
                    "mesh_merge": case.mesh_merge,
                    "initial_snapshot_sha256": snapshot["sha256"],
                    "initial_execution_fingerprint_sha256": (
                        case.cuda.expected_execution_fingerprint["sha256"]
                    ),
                    "input_bytes": case.input_records,
                    "host_preparation_seconds": timing["host_preparation"],
                    "published": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.gpuwm_root is None:
        raise ValueError(
            "--gpuwm-root is required: pass it explicitly, or set GPUWM_ROOT to the "
            "gpuwm checkout that provides the dual-run capsule comparator"
        )
    gpuwm_root = args.gpuwm_root.expanduser().resolve(strict=True)
    _, comparison_authority = load_gpuwm_dualrun(gpuwm_root)
    ftz_binding = load_ftz_binding_record(
        args.ftz_binding,
        gpuwm_root=gpuwm_root,
        gpuwm_receipt_root=args.gpuwm_probe,
    )
    cache_root = args.cache_root.expanduser().resolve()
    coarse._require_beneath(
        cache_root, ROOT / "work" / "cuda-gfs-60km-cache", "CUDA cache root"
    )
    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    started = time.perf_counter()
    kernel_cache = prepare_cuda_kernel_cache(
        capability, cache_root, ftz_binding=ftz_binding
    )
    timing["compile"] = time.perf_counter() - started
    resident_probe = run_resident_performance_probe(
        case.cuda,
        config,
        bounds,
        kernel_cache=kernel_cache,
        steps=args.probe_steps,
    )
    print(
        json.dumps(
            {
                "phase": "resident-probe-passed",
                "steps": resident_probe["steps"],
                "device_milliseconds_mean": resident_probe["device_milliseconds_mean"],
                "simulated_hours_per_wall_hour": resident_probe[
                    "simulated_hours_per_wall_hour"
                ],
                "peak_sampled_device_used_bytes": resident_probe["memory"][
                    "peak_sampled_device_used_bytes"
                ],
                "stability": resident_probe["stability"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.probe_only:
        print(
            json.dumps(
                {
                    "mode": "probe-only",
                    "profile": PROFILE,
                    "resident_probe": resident_probe,
                    "compile_manifest_sha256": ftz_binding["value"]["compile_relation"][
                        "compile_manifest_sha256"
                    ],
                    "published": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    started = time.perf_counter()
    verified, dual_profile = run_verified_cuda_arms_profiled(
        case.cuda,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
        gpuwm_root=gpuwm_root,
        plan=plan,
    )
    timing["dual_cuda_trajectory_and_compare"] = time.perf_counter() - started
    started = time.perf_counter()
    products = write_forecast_fields(case, verified, config, bounds, plan=plan)
    timing["diagnostics_history_regrid_renderer_input"] = time.perf_counter() - started
    timing["total_before_receipt"] = time.perf_counter() - total_started
    write_forecast_receipt(
        case,
        verified,
        products,
        config,
        bounds,
        plan,
        timing=timing,
        resident_probe=resident_probe,
        dual_profile=dual_profile,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "receipt": coarse._repo_relative(plan.receipt),
                "renderer_input": coarse._repo_relative(plan.renderer_input),
                "dual_run_identical": True,
                "resident_simulated_hours_per_wall_hour": resident_probe[
                    "simulated_hours_per_wall_hour"
                ],
            },
            sort_keys=True,
        )
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
