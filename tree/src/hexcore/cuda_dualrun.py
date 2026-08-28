"""Deterministic long-run capsules for the MPAS CUDA dry-dycore mirror.

The RTX 5090 has no ECC.  A long CUDA claim therefore consists of two
independent device uploads of one host preparation, two complete trajectories,
and gpuwm's total capsule comparison.  This module deliberately omits wall
clock values, cache paths, timestamps, and arm labels from the capsules: those
values differ by construction and would either make equality unreachable or
require an unsafe comparison ignore list.

Every model step is explicitly downloaded for instrumentation and represented
by semantic SHA-256 records for the prognostic state and saved diagnostic
sidecar.  The forecast itself remains device-resident between steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from .cuda_backend import CudaCapability, KernelCache, canonical_sha256
from .cuda_driver import (
    CUDA_AUTHORITY_INITIAL_SCHEMA,
    CUDA_AUTHORITY_RULER_SCHEMA,
    CUDA_IMPLEMENTED_UNLINKED_EVIDENCE,
    CUDA_LAYOUT_CONTRACT,
    CUDA_ORIGINAL_JW_BRANCH_EVIDENCE,
    CUDA_T0_EXACT_SIDECAR,
    CUDA_WHOLE_STEP_EVIDENCE,
    CUDA_V841_AUTHORITY_NONCLAIMS,
    CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE,
    FROZEN_CUDA_SOURCE,
    V841_CUDA_SOURCE,
    CudaAuthorityRulerBinder,
    CudaDryDycoreDriver,
    _AUTHORITY_ADVECTION_FLOAT_FIELDS,
    _AUTHORITY_ADVECTION_INDEX_FIELDS,
    _AUTHORITY_MESH_FLOAT_FIELDS,
    _AUTHORITY_MESH_INDEX_FIELDS,
    _AUTHORITY_REFERENCE_FIELDS,
    _AUTHORITY_SIDECAR_FIELDS,
    _AUTHORITY_STATE_FIELDS,
    _AUTHORITY_TERRAIN_FIELDS,
    _AUTHORITY_VERTICAL_FIELDS,
    _validate_initial_fingerprint,
    cuda_authority_initial_fingerprint,
)
from .cuda_ftz import (
    MPAS_FTZ_SCHEMA,
    MPAS_FTZ_V841_SCHEMA,
    production_translation_units,
    v841_reached_translation_units,
    validate_compile_manifest_relation,
    validate_v841_compile_manifest_relation,
    validate_mpas_ftz_binding,
    validate_mpas_ftz_binding_v841,
)
from .config_v841 import V841DryDycoreConfig, V841_SOURCE_RELEASE
from .damping_v841 import build_v841_vertical_velocity_damping
from .dynamics_v841 import (
    V841ReferenceWindProfiles,
    precomputed_mesh_inverse_v841,
)
from .driver import (
    DryDycoreConfig,
    DrySavedDiagnostics,
    SPLIT_FLUX_REDUCTION,
    load_mpas_initial_state,
    load_mpas_vertical_grid,
)
from .mesh import Mesh
from .state import PrognosticState
from .transport import build_advection_coefficients
from .integration import RKSchedule
from .offcentering_v841 import build_v841_acoustic_offcentering
from .errors import ConfigurationRefusal


CAPSULE_SCHEMA = "mpas-port.cuda-dual-run-capsule/v2"
REPORT_SCHEMA = "mpas-port.cuda-dual-run-report/v1"
COMPARISON_AUTHORITY_SCHEMA = "mpas-port.gpuwm-dualrun-authority/v1"
EXPECTED_GPUWM_COMPARISON_SCHEMA = "gpuwm.dual-run-comparison/v1"
EXECUTION_INPUT_FINGERPRINT_SCHEMA = "mpas-port.cuda-execution-input-fingerprint/v1"

STATE_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
SIDECAR_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)
V841_CONTEXT_FIELDS = (
    "inv_area_cell",
    "inv_area_triangle",
    "inv_dc_edge",
    "inv_dv_edge",
    "etp",
    "etm",
    "ewp",
    "ewm",
    "u_init",
    "v_init",
)
_BASE_STEP_CONTRACT_FIELDS = frozenset(
    {
        "evidence",
        "configuration",
        "configuration_sha256",
        "authority_ruler",
        "authority_ruler_sha256",
        "frozen_source",
        "t0_diagnostics_source",
        "stage_acoustic_steps",
        "compile_manifest_sha256",
        "layout_contract_sha256",
        "d2h_bytes_inside_step",
    }
)
_V841_STEP_CONTRACT_FIELDS = _BASE_STEP_CONTRACT_FIELDS | frozenset(
    {
        "source_release",
        "dynamics_split_steps",
        "dynamics_timestep_seconds",
        "dynamics_stage_timesteps",
        "scalar_transport_stage_timesteps",
        "split_flux_reduction",
        "authority_nonclaims",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONDETERMINISTIC_KEYS = frozenset(
    {
        "cache_directory",
        "compile_seconds",
        "elapsed_seconds",
        "finished_utc",
        "first_launch_ms",
        "first_wall_ms",
        "generated_at_utc",
        "max_launch_ms",
        "mean_launch_ms",
        "min_launch_ms",
        "started_utc",
        "timestamp",
        "wall_seconds",
    }
)


class CudaDualRunError(RuntimeError):
    """A deterministic long-run or its evidence is incomplete."""


@dataclass(frozen=True, slots=True)
class PreparedJwInputs:
    """One immutable host preparation shared by both CUDA uploads."""

    mesh: Mesh
    state: PrognosticState
    vertical: Any
    reference: Any
    saved_diagnostics: DrySavedDiagnostics
    terrain_metrics: Any
    input_bytes: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PreparedCudaInputs:
    """A profile-tagged host preparation reusable by two CUDA arms."""

    profile: str
    target: str
    preparation_method: str
    mesh: Any
    state: PrognosticState
    vertical: Any
    reference: Any
    saved_diagnostics: DrySavedDiagnostics
    terrain_metrics: Any
    input_bytes: dict[str, dict[str, Any]]
    reference_wind_profiles: V841ReferenceWindProfiles | None = None
    expected_execution_fingerprint: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    advection_coefficients: Any | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def validated(
        cls,
        *,
        config: DryDycoreConfig,
        profile: str,
        target: str,
        preparation_method: str,
        mesh: Any,
        state: PrognosticState,
        vertical: Any,
        reference: Any,
        saved_diagnostics: DrySavedDiagnostics,
        terrain_metrics: Any,
        input_bytes: dict[str, dict[str, Any]],
        reference_wind_profiles: V841ReferenceWindProfiles | None = None,
        allow_regional_sentinels: bool = False,
    ) -> "PreparedCudaInputs":
        """Seal the complete host execution boundary before either upload.

        ``allow_regional_sentinels`` says this mesh's outermost ring carries
        MPAS's limited-area one-cell-edge sentinels by construction, so the
        order-three advection stencil resolves them onto the garbage cell
        instead of refusing.  It is the mesh's own property, read off the
        bdyMask triple by the caller, never a knob a run chooses.
        """

        prepared = cls(
            profile=profile,
            target=target,
            preparation_method=preparation_method,
            mesh=mesh,
            state=state,
            vertical=vertical,
            reference=reference,
            saved_diagnostics=saved_diagnostics,
            terrain_metrics=terrain_metrics,
            input_bytes=input_bytes,
            reference_wind_profiles=reference_wind_profiles,
        )
        is_v841 = getattr(config, "source_release", "v8.2.3") == V841_SOURCE_RELEASE
        if is_v841:
            if not isinstance(config, V841DryDycoreConfig):
                raise CudaDualRunError(
                    "v8.4.1 prepared execution requires V841DryDycoreConfig"
                )
            try:
                CudaDryDycoreDriver._validate_config(config)
            except ConfigurationRefusal as error:
                raise CudaDualRunError(
                    f"v8.4.1 prepared execution is outside the CUDA lane: {error}"
                ) from error
            if reference_wind_profiles is None:
                raise CudaDualRunError(
                    "v8.4.1 prepared execution requires reference wind profiles"
                )
            reference_wind_profiles.validate(
                int(state.rho.shape[0]), np.dtype(np.float32)
            )
        elif reference_wind_profiles is not None:
            raise CudaDualRunError(
                "v8.2.3 prepared execution cannot carry v8.4.1 reference winds"
            )
        coefficients = build_advection_coefficients(
            mesh,
            config_scalar_adv_order=config.config_scalar_adv_order,
            n_vert_levels=state.rho.shape[0],
            source_order_v841=is_v841,
            allow_regional_sentinels=allow_regional_sentinels,
        )
        object.__setattr__(prepared, "advection_coefficients", coefficients)
        sealed = fingerprint_prepared_execution(prepared, config)
        object.__setattr__(prepared, "expected_execution_fingerprint", sealed)
        return prepared


@dataclass(frozen=True, slots=True)
class CudaArmRun:
    """Deterministic evidence plus the still-resident final atmosphere."""

    capsule: dict[str, Any]
    final_atmosphere: Any


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    selected = Path(path)
    if not selected.is_file():
        raise CudaDualRunError(f"required artifact is not a file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    selected = Path(path).expanduser().resolve()
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(selected.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(selected)
    return selected


def _require_sha256(value: Any, label: str) -> str:
    selected = str(value)
    if _SHA256_RE.fullmatch(selected) is None:
        raise CudaDualRunError(f"{label} is not a lowercase SHA-256")
    return selected


def _artifact_record(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    selected = Path(path).expanduser().resolve(strict=True)
    digest = sha256_file(selected)
    if expected_sha256 is not None and digest != expected_sha256:
        raise CudaDualRunError(
            f"input SHA-256 mismatch for {selected.name}: {digest} != {expected_sha256}"
        )
    return {"bytes": selected.stat().st_size, "sha256": digest}


def derive_step_count(duration_seconds: float, dt_seconds: float) -> int:
    """Return the exact step count and refuse scheduling-only smokes."""

    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    quotient = duration_seconds / dt_seconds
    rounded = round(quotient)
    if not np.isclose(quotient, rounded, rtol=0.0, atol=1.0e-12):
        raise ValueError("duration_seconds must be an exact multiple of dt_seconds")
    if rounded < 3:
        raise ValueError("a durable CUDA dual run requires at least three full steps")
    return int(rounded)


def jw_day_config(
    dt_seconds: float = 3600.0, acoustic_substeps: int = 6
) -> DryDycoreConfig:
    """The frozen no-mixing x1.2562 day target, changing timestep controls only."""

    config = DryDycoreConfig(
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
    config.validate()
    return config


def prepare_jw_inputs(
    initial_path: str | Path,
    native_t0_path: str | Path,
    config: DryDycoreConfig,
    *,
    expected_initial_sha256: str | None = None,
    expected_native_t0_sha256: str | None = None,
) -> PreparedJwInputs:
    """Read and convert the authority inputs exactly once on the host."""

    initial = Path(initial_path).expanduser().resolve(strict=True)
    native_t0 = Path(native_t0_path).expanduser().resolve(strict=True)
    input_bytes = {
        "authority_initial_condition": _artifact_record(
            initial, expected_sha256=expected_initial_sha256
        ),
        "native_internal_t0": _artifact_record(
            native_t0, expected_sha256=expected_native_t0_sha256
        ),
    }
    mesh = Mesh.from_netcdf(initial)
    native = load_mpas_vertical_grid(
        initial,
        mesh,
        config_coef_3rd_order=config.config_coef_3rd_order,
    )
    state, reference, saved = load_mpas_initial_state(
        native_t0,
        mesh,
        native.vertical_grid,
        scalar_names=("qv",),
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
    )
    return PreparedJwInputs(
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=input_bytes,
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import cupy as cp
    except Exception:
        return np.asarray(value)
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return np.asarray(value)


def fingerprint_array(name: str, value: Any) -> dict[str, Any]:
    """Hash one logical array, binding bytes to its name, dtype, and shape."""

    array = np.ascontiguousarray(_to_numpy(value))
    if array.dtype.hasobject:
        raise CudaDualRunError(f"{name} has an object dtype")
    if array.dtype.kind in "fc" and not bool(np.all(np.isfinite(array))):
        raise CudaDualRunError(f"{name} contains non-finite values")
    raw_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    core = {
        "name": str(name),
        "dtype": array.dtype.str,
        "shape": [int(value) for value in array.shape],
        "bytes": int(array.nbytes),
        "bytes_sha256": raw_sha256,
    }
    return {**core, "sha256": _json_sha256(core)}


def fingerprint_array_group(
    values: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not values:
        raise CudaDualRunError("an array fingerprint group cannot be empty")
    fields = {name: fingerprint_array(name, values[name]) for name in sorted(values)}
    core = {"metadata": dict(metadata or {}), "fields": fields}
    return {**core, "sha256": _json_sha256(core)}


def fingerprint_atmosphere(atmosphere: Any) -> dict[str, Any]:
    """Explicitly download and hash one resident state/sidecar snapshot."""

    state = atmosphere.state
    saved = atmosphere.saved
    state_record = fingerprint_array_group(
        {name: getattr(state, name) for name in STATE_FIELDS},
        metadata={"time_seconds": float(state.time_seconds)},
    )
    sidecar_record = fingerprint_array_group(
        {name: getattr(saved, name) for name in SIDECAR_FIELDS},
        metadata={"kind": "DrySavedDiagnostics"},
    )
    core = {
        "model_time_seconds": float(state.time_seconds),
        "state": state_record,
        "saved_diagnostics": sidecar_record,
    }
    return {**core, "sha256": _json_sha256(core)}


def _fingerprint_host_preparation(prepared: PreparedJwInputs) -> dict[str, Any]:
    class _HostAtmosphere:
        state = prepared.state
        saved = prepared.saved_diagnostics

    return fingerprint_atmosphere(_HostAtmosphere())


_DEVICE_MESH_NAMES = {
    "cellsOnEdge": "cells_on_edge",
    "edgesOnCell": "edges_on_cell",
    "nEdgesOnCell": "n_edges_on_cell",
    "cellsOnCell": "cells_on_cell",
    "verticesOnEdge": "vertices_on_edge",
    "edgesOnEdge": "edges_on_edge",
    "nEdgesOnEdge": "n_edges_on_edge",
    "verticesOnCell": "vertices_on_cell",
    "edgesOnVertex": "edges_on_vertex",
    "cellsOnVertex": "cells_on_vertex",
    "weightsOnEdge": "weights_on_edge",
    "dcEdge": "dc_edge",
    "dvEdge": "dv_edge",
    "areaCell": "area_cell",
    "areaTriangle": "area_triangle",
    "kiteAreasOnVertex": "kite_areas_on_vertex",
    "latCell": "lat_cell",
    "lonCell": "lon_cell",
    "latEdge": "lat_edge",
    "lonEdge": "lon_edge",
    "angleEdge": "angle_edge",
    "meshDensity": "mesh_density",
    "defc_a": "defc_a",
    "defc_b": "defc_b",
    "fVertex": "f_vertex",
    "fEdge": "f_edge",
    "spec_zone_mask_edge": "spec_zone_mask_edge",
}


def _authority_array_record(value: Any, *, dtype: Any | None = None) -> dict[str, Any]:
    array = _to_numpy(value)
    if dtype is not None:
        array = np.asarray(array, dtype=dtype)
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.hasobject:
        raise CudaDualRunError("execution-input arrays cannot use object dtype")
    if contiguous.dtype.kind in "fc" and not bool(np.all(np.isfinite(contiguous))):
        raise CudaDualRunError("execution-input arrays must be finite")
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": contiguous.dtype.str,
        "shape": [int(extent) for extent in contiguous.shape],
        "c_bytes": len(raw),
        "c_bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _host_mesh_value(mesh: Any, name: str) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    raise CudaDualRunError(f"prepared mesh is missing {name}")


def _host_edge_sign_record(mesh: Any) -> dict[str, Any]:
    """Rebuild the sole derived DeviceMesh array in upload source order."""

    edges_on_cell = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "edgesOnCell"), dtype=np.int32)
    )
    cells_on_edge = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "cellsOnEdge"), dtype=np.int32)
    )
    counts = np.ascontiguousarray(
        np.asarray(_host_mesh_value(mesh, "nEdgesOnCell"), dtype=np.int32)
    )
    signs = np.zeros(edges_on_cell.shape, dtype=np.float32)
    for cell in range(edges_on_cell.shape[0]):
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            signs[cell, slot] = (
                np.float32(1.0)
                if int(cells_on_edge[edge, 0]) == cell
                else np.float32(-1.0)
            )
    return _authority_array_record(signs)


def _execution_fingerprint_core(
    authority_initial: Mapping[str, Any],
    configuration_sha256: str,
    edge_sign_on_cell: Mapping[str, Any],
    v841_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    derived = {"mesh.edge_sign_on_cell": dict(edge_sign_on_cell)}
    if v841_context is not None:
        if set(v841_context) != set(V841_CONTEXT_FIELDS):
            raise CudaDualRunError("v8.4.1 context fingerprint inventory changed")
        derived.update(
            {
                f"v841_context.{name}": dict(v841_context[name])
                for name in V841_CONTEXT_FIELDS
            }
        )
    return {
        "schema": EXECUTION_INPUT_FINGERPRINT_SCHEMA,
        "configuration_sha256": configuration_sha256,
        "authority_initial": dict(authority_initial),
        "derived_upload_arrays": derived,
    }


def _finalize_execution_fingerprint(core: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(core), "sha256": _json_sha256(core)}


def fingerprint_prepared_execution(
    prepared: PreparedCudaInputs,
    config: DryDycoreConfig,
) -> dict[str, Any]:
    """Hash every host value that controls a generic CUDA trajectory."""

    configuration_sha256 = canonical_sha256(asdict(config))
    authority_initial = cuda_authority_initial_fingerprint(
        prepared.state,
        prepared.saved_diagnostics,
        mesh=prepared.mesh,
        vertical=prepared.vertical,
        reference=prepared.reference,
        terrain_metrics=prepared.terrain_metrics,
        config=config,
        advection_coefficients=prepared.advection_coefficients,
    )
    source_release = getattr(config, "source_release", "v8.2.3")
    v841_context = None
    if source_release == V841_SOURCE_RELEASE:
        if prepared.reference_wind_profiles is None:
            raise CudaDualRunError(
                "v8.4.1 execution fingerprint requires reference wind profiles"
            )
        selected_dss = build_v841_vertical_velocity_damping(
            np.asarray(prepared.vertical.zgrid),
            xnutr=config.config_xnutr,
            damping_start_z=config.config_zd,
        )
        vertical_group = authority_initial["execution_inputs"]["vertical"]
        vertical_group["arrays"]["dss"] = _authority_array_record(
            selected_dss, dtype=np.float32
        )
        for name in ("cf1", "cf2", "cf3"):
            vertical_group["scalars"][name] = float(
                np.float32(getattr(prepared.vertical, name))
            )
        advection_group = authority_initial["execution_inputs"]["advection"]
        for name in _AUTHORITY_ADVECTION_FLOAT_FIELDS:
            advection_group["arrays"][name] = _authority_array_record(
                getattr(prepared.advection_coefficients, name), dtype=np.float32
            )
        for name in _AUTHORITY_ADVECTION_INDEX_FIELDS:
            advection_group["arrays"][name] = _authority_array_record(
                getattr(prepared.advection_coefficients, name), dtype=np.int32
            )

        offcentering = build_v841_acoustic_offcentering(
            np.asarray(prepared.vertical.rdzw),
            minimum=config.config_epssm_minimum,
            maximum=config.config_epssm_maximum,
            transition_bottom_z=config.config_epssm_transition_bottom_z,
            transition_top_z=config.config_epssm_transition_top_z,
        )
        inverse_fields = {
            "inv_area_cell": "areaCell",
            "inv_area_triangle": "areaTriangle",
            "inv_dc_edge": "dcEdge",
            "inv_dv_edge": "dvEdge",
        }
        context_values: dict[str, Any] = {
            name: precomputed_mesh_inverse_v841(
                prepared.mesh, field_name, np.dtype(np.float32)
            )
            for name, field_name in inverse_fields.items()
        }
        context_values.update(
            {
                name: getattr(offcentering, name)
                for name in ("etp", "etm", "ewp", "ewm")
            }
        )
        context_values.update(
            {
                "u_init": prepared.reference_wind_profiles.u_init,
                "v_init": prepared.reference_wind_profiles.v_init,
            }
        )
        v841_context = {
            name: _authority_array_record(context_values[name], dtype=np.float32)
            for name in V841_CONTEXT_FIELDS
        }
    core = _execution_fingerprint_core(
        authority_initial,
        configuration_sha256,
        _host_edge_sign_record(prepared.mesh),
        v841_context,
    )
    return _finalize_execution_fingerprint(core)


def _device_authority_initial_fingerprint(
    driver: CudaDryDycoreDriver,
) -> dict[str, Any]:
    atmosphere = driver.atmosphere
    mesh = atmosphere.mesh
    vertical = atmosphere.vertical
    reference = atmosphere.reference
    terrain = atmosphere.terrain
    if terrain is None:
        raise CudaDualRunError("uploaded CUDA atmosphere has no terrain metrics")
    coefficients = driver.coefficients

    def arrays(
        owner: Any,
        names: tuple[str, ...],
        *,
        rename: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        selected = rename or {}
        return {
            name: _authority_array_record(getattr(owner, selected.get(name, name)))
            for name in names
        }

    state = atmosphere.state
    saved = atmosphere.saved
    result = {
        "schema": CUDA_AUTHORITY_INITIAL_SCHEMA,
        "start_time_seconds": float(state.time_seconds),
        "state": arrays(state, _AUTHORITY_STATE_FIELDS),
        "saved_diagnostics": arrays(saved, _AUTHORITY_SIDECAR_FIELDS),
        "execution_inputs": {
            "mesh": {
                "arrays": arrays(
                    mesh,
                    _AUTHORITY_MESH_INDEX_FIELDS + _AUTHORITY_MESH_FLOAT_FIELDS,
                    rename=_DEVICE_MESH_NAMES,
                ),
                "scalars": {
                    "n_cells": int(mesh.n_cells),
                    "n_edges": int(mesh.n_edges),
                    "n_vertices": int(mesh.n_vertices),
                    "max_edges": int(mesh.max_edges),
                    "max_edges2": int(mesh.max_edges2),
                    "vertex_degree": int(mesh.vertex_degree),
                    "nominal_min_dc": float(mesh.nominal_min_dc),
                },
            },
            "vertical": {
                "arrays": arrays(vertical, _AUTHORITY_VERTICAL_FIELDS),
                "scalars": {
                    "cf1": float(vertical.cf1),
                    "cf2": float(vertical.cf2),
                    "cf3": float(vertical.cf3),
                    "first_height_level": int(vertical.first_height_level),
                    "n_vert_levels": int(vertical.n_vert_levels),
                },
            },
            "reference": {
                "arrays": arrays(reference, _AUTHORITY_REFERENCE_FIELDS),
                "scalars": {},
            },
            "terrain": {
                "arrays": arrays(terrain, _AUTHORITY_TERRAIN_FIELDS),
                "scalars": {},
            },
            "advection": {
                "arrays": arrays(
                    coefficients,
                    _AUTHORITY_ADVECTION_FLOAT_FIELDS
                    + _AUTHORITY_ADVECTION_INDEX_FIELDS,
                ),
                "scalars": {
                    "horizontal_order": int(coefficients.horizontal_order),
                },
            },
        },
    }
    _validate_initial_fingerprint(result)
    return result


def fingerprint_uploaded_execution(
    driver: CudaDryDycoreDriver,
) -> dict[str, Any]:
    """Explicitly download and hash the complete admitted device boundary."""

    authority_initial = _device_authority_initial_fingerprint(driver)
    v841_context = None
    resident_v841 = getattr(driver, "v841_context", None)
    if resident_v841 is not None:
        v841_context = {
            name: _authority_array_record(getattr(resident_v841, name))
            for name in V841_CONTEXT_FIELDS
        }
    core = _execution_fingerprint_core(
        authority_initial,
        driver.configuration_sha256,
        _authority_array_record(driver.atmosphere.mesh.edge_sign_on_cell),
        v841_context,
    )
    return _finalize_execution_fingerprint(core)


def _validate_execution_fingerprint(
    record: Mapping[str, Any],
    *,
    configuration_sha256: str,
) -> None:
    if set(record) != {
        "schema",
        "configuration_sha256",
        "authority_initial",
        "derived_upload_arrays",
        "sha256",
    }:
        raise CudaDualRunError("execution-input fingerprint inventory changed")
    if record.get("schema") != EXECUTION_INPUT_FINGERPRINT_SCHEMA:
        raise CudaDualRunError("execution-input fingerprint schema changed")
    if record.get("configuration_sha256") != configuration_sha256:
        raise CudaDualRunError("execution-input fingerprint configuration changed")
    authority_initial = record.get("authority_initial")
    if not isinstance(authority_initial, Mapping):
        raise CudaDualRunError("execution-input fingerprint has no authority input")
    try:
        _validate_initial_fingerprint(authority_initial)
    except (TypeError, ValueError) as error:
        raise CudaDualRunError(
            f"execution-input authority fingerprint is invalid: {error}"
        ) from error
    if not isinstance(authority_initial.get("execution_inputs"), Mapping):
        raise CudaDualRunError("execution-input fingerprint is state-only")
    derived = record.get("derived_upload_arrays")
    old_derived = {"mesh.edge_sign_on_cell"}
    v841_derived = old_derived | {
        f"v841_context.{name}" for name in V841_CONTEXT_FIELDS
    }
    if not isinstance(derived, Mapping) or set(derived) not in (
        old_derived,
        v841_derived,
    ):
        raise CudaDualRunError("derived upload-array inventory changed")
    edge_sign = derived["mesh.edge_sign_on_cell"]
    if not isinstance(edge_sign, Mapping) or set(edge_sign) != {
        "dtype",
        "shape",
        "c_bytes",
        "c_bytes_sha256",
    }:
        raise CudaDualRunError("derived edge-sign fingerprint is incomplete")
    try:
        dtype = np.dtype(edge_sign["dtype"])
    except TypeError as error:
        raise CudaDualRunError("derived edge-sign dtype is invalid") from error
    if dtype != np.dtype(np.float32):
        raise CudaDualRunError("derived edge-sign dtype changed")
    shape = edge_sign["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(not isinstance(extent, int) or extent <= 0 for extent in shape)
    ):
        raise CudaDualRunError("derived edge-sign shape is invalid")
    if edge_sign["c_bytes"] != int(np.prod(shape, dtype=np.int64)) * 4:
        raise CudaDualRunError("derived edge-sign byte count is false")
    _require_sha256(
        edge_sign.get("c_bytes_sha256"),
        "derived_upload_arrays.mesh.edge_sign_on_cell",
    )
    v841_context = None
    if set(derived) == v841_derived:
        execution = authority_initial["execution_inputs"]
        mesh_scalars = execution["mesh"]["scalars"]
        vertical_scalars = execution["vertical"]["scalars"]
        nlev = int(vertical_scalars["n_vert_levels"])
        expected_shapes = {
            "inv_area_cell": [int(mesh_scalars["n_cells"])],
            "inv_area_triangle": [int(mesh_scalars["n_vertices"])],
            "inv_dc_edge": [int(mesh_scalars["n_edges"])],
            "inv_dv_edge": [int(mesh_scalars["n_edges"])],
            "etp": [nlev],
            "etm": [nlev],
            "ewp": [nlev + 1],
            "ewm": [nlev + 1],
            "u_init": [nlev],
            "v_init": [nlev],
        }
        v841_context = {}
        for name in V841_CONTEXT_FIELDS:
            key = f"v841_context.{name}"
            value = derived[key]
            if not isinstance(value, Mapping) or set(value) != {
                "dtype",
                "shape",
                "c_bytes",
                "c_bytes_sha256",
            }:
                raise CudaDualRunError(f"{key} fingerprint is incomplete")
            try:
                context_dtype = np.dtype(value["dtype"])
            except TypeError as error:
                raise CudaDualRunError(f"{key} dtype is invalid") from error
            if context_dtype != np.dtype(np.float32):
                raise CudaDualRunError(f"{key} dtype changed")
            if value["shape"] != expected_shapes[name]:
                raise CudaDualRunError(f"{key} shape changed")
            if value["c_bytes"] != int(np.prod(value["shape"])) * 4:
                raise CudaDualRunError(f"{key} byte count is false")
            _require_sha256(value.get("c_bytes_sha256"), f"{key}.c_bytes_sha256")
            v841_context[name] = value
    core = _execution_fingerprint_core(
        authority_initial,
        configuration_sha256,
        edge_sign,
        v841_context,
    )
    if record.get("sha256") != _json_sha256(core):
        raise CudaDualRunError("execution-input fingerprint digest is false")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    selected = Path(path)
    try:
        value = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CudaDualRunError(
            f"cannot read JSON object {selected}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CudaDualRunError(f"JSON artifact must be an object: {selected}")
    return value


def load_ftz_binding_record(
    path: str | Path,
    *,
    gpuwm_root: str | Path,
    gpuwm_receipt_root: str | Path,
    source_release: str = "v8.2.3",
) -> dict[str, Any]:
    """Validate the live FTZ authority and return a self-contained binding."""

    selected = Path(path).expanduser().resolve(strict=True)
    binding = _load_json_object(selected)
    if source_release == V841_SOURCE_RELEASE:
        validated = validate_mpas_ftz_binding_v841(
            binding,
            gpuwm_root=gpuwm_root,
            gpuwm_receipt_root=gpuwm_receipt_root,
        )
        schema = MPAS_FTZ_V841_SCHEMA
    elif source_release == "v8.2.3":
        validated = validate_mpas_ftz_binding(
            binding,
            gpuwm_root=gpuwm_root,
            gpuwm_receipt_root=gpuwm_receipt_root,
        )
        schema = MPAS_FTZ_SCHEMA
    else:
        raise CudaDualRunError(
            f"unsupported CUDA FTZ binding source release {source_release!r}"
        )
    if validated != binding:
        raise CudaDualRunError("validated FTZ binding changed its JSON value")
    return {
        "schema": schema,
        "artifact_sha256": sha256_file(selected),
        "sha256": _json_sha256(binding),
        "value": binding,
    }


def load_gpuwm_dualrun(gpuwm_root: str | Path) -> tuple[ModuleType, dict[str, Any]]:
    """Load gpuwm's comparator from the exact source tree recorded in evidence."""

    root = Path(gpuwm_root).expanduser().resolve(strict=True)
    source = (root / "gpuwm" / "certify" / "dualrun.py").resolve(strict=True)
    digest = sha256_file(source)
    module_name = f"_mpas_gpuwm_dualrun_{digest[:16]}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise CudaDualRunError(f"cannot load gpuwm comparator at {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    for function_name in ("compare_capsules", "compare_capsule_files"):
        if not callable(getattr(module, function_name, None)):
            raise CudaDualRunError(
                f"gpuwm comparator does not export {function_name}()"
            )
    authority = {
        "schema": COMPARISON_AUTHORITY_SCHEMA,
        "module": "gpuwm.certify.dualrun",
        "source_path": "gpuwm/certify/dualrun.py",
        "source_sha256": digest,
        "functions": ["compare_capsules", "compare_capsule_files"],
        "comparison_schema": EXPECTED_GPUWM_COMPARISON_SCHEMA,
        "comparison_scope": "total leaf comparison with no ignore list",
    }
    return module, authority


def _source_bindings(
    source_release: str = "v8.2.3",
) -> dict[str, dict[str, str]]:
    root = Path(__file__).resolve().parent
    paths = {
        "cuda_dualrun": Path(__file__).resolve(),
        "cuda_acoustic": root / "cuda_acoustic.py",
        "cuda_driver": root / "cuda_driver.py",
        "cuda_fp32": root / "cuda_fp32.py",
        "cuda_ftz": root / "cuda_ftz.py",
        "cuda_horizontal": root / "cuda_horizontal.py",
        "cuda_transport": root / "cuda_transport.py",
        "cuda_backend_compile_contract": (
            root / "cuda_backend" / "compile_contract.py"
        ),
        "cuda_backend_containers": root / "cuda_backend" / "containers.py",
        "cuda_backend_recovery": root / "cuda_backend" / "recovery.py",
        "cuda_backend_runtime": root / "cuda_backend" / "runtime.py",
        "host_driver": root / "driver.py",
        "host_integration": root / "integration.py",
        "host_mixing": root / "mixing.py",
        "host_transport": root / "transport.py",
    }
    if source_release == V841_SOURCE_RELEASE:
        paths.update(
            {
                "cuda_acoustic_v841": root / "cuda_acoustic_v841.py",
                "cuda_dynamics_v841": root / "cuda_dynamics_v841.py",
                "cuda_horizontal_v841": root / "cuda_horizontal_v841.py",
                "cuda_transport_v841": root / "cuda_transport_v841.py",
                "cuda_v841": root / "cuda_v841.py",
                "host_acoustic_v841": root / "acoustic_v841.py",
                "host_config_v841": root / "config_v841.py",
                "host_damping_v841": root / "damping_v841.py",
                "host_dynamics_v841": root / "dynamics_v841.py",
                "host_offcentering_v841": root / "offcentering_v841.py",
            }
        )
    elif source_release != "v8.2.3":
        raise CudaDualRunError(f"unsupported CUDA source release {source_release!r}")
    return {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _device_record(capability: CudaCapability) -> dict[str, Any]:
    record = capability.as_dict()
    record.pop("cache_directory", None)
    return record


def _configuration_record(config: DryDycoreConfig) -> dict[str, Any]:
    value = asdict(config)
    return {"value": value, "sha256": canonical_sha256(value)}


def _preparation_record(
    configuration: Mapping[str, Any],
    input_bytes: Mapping[str, Any],
    initial_snapshot: Mapping[str, Any],
    initial_execution_fingerprint: Mapping[str, Any],
    *,
    profile: str,
    target: str,
    method: str,
) -> dict[str, Any]:
    core = {
        "profile": profile,
        "target": target,
        "method": method,
        "configuration_sha256": configuration["sha256"],
        "input_bytes": dict(input_bytes),
        "initial_snapshot_sha256": initial_snapshot["sha256"],
        "initial_execution_fingerprint_sha256": initial_execution_fingerprint["sha256"],
    }
    return {**core, "sha256": _json_sha256(core)}


def _step_record(index: int, result: Any) -> dict[str, Any]:
    receipt = result.receipt
    snapshot = fingerprint_atmosphere(result.atmosphere)
    contract = {
        "evidence": receipt.evidence,
        "configuration": dict(receipt.configuration),
        "configuration_sha256": receipt.configuration_sha256,
        "authority_ruler": (
            None if receipt.authority_ruler is None else dict(receipt.authority_ruler)
        ),
        "authority_ruler_sha256": receipt.authority_ruler_sha256,
        "frozen_source": receipt.frozen_source,
        "t0_diagnostics_source": receipt.t0_diagnostics_source,
        "stage_acoustic_steps": list(receipt.stage_acoustic_steps),
        "compile_manifest_sha256": receipt.compile_manifest_sha256,
        "layout_contract_sha256": canonical_sha256(receipt.layout_contract),
        "d2h_bytes_inside_step": int(receipt.d2h.bytes),
    }
    if getattr(receipt, "source_release", "v8.2.3") == V841_SOURCE_RELEASE:
        contract.update(
            {
                "source_release": receipt.source_release,
                "dynamics_split_steps": int(receipt.dynamics_split_steps),
                "dynamics_timestep_seconds": float(
                    receipt.dynamics_timestep_seconds
                ),
                "dynamics_stage_timesteps": list(
                    receipt.dynamics_stage_timesteps
                ),
                "scalar_transport_stage_timesteps": (
                    None
                    if receipt.scalar_transport_stage_timesteps is None
                    else list(receipt.scalar_transport_stage_timesteps)
                ),
                "split_flux_reduction": receipt.split_flux_reduction,
                "authority_nonclaims": list(receipt.authority_nonclaims),
            }
        )
    core = {
        "step": int(index),
        "start_time_seconds": float(receipt.start_time_seconds),
        "end_time_seconds": float(receipt.end_time_seconds),
        "snapshot": snapshot,
        "step_contract": contract,
    }
    return {**core, "sha256": _json_sha256(core)}


def _compile_relation(
    manifest: Mapping[str, Any], source_release: str
) -> dict[str, Any]:
    if source_release == V841_SOURCE_RELEASE:
        return validate_v841_compile_manifest_relation(manifest)
    if source_release == "v8.2.3":
        return validate_compile_manifest_relation(manifest)
    raise CudaDualRunError(f"unsupported CUDA source release {source_release!r}")


def _resolve_full_compile_manifest(
    cache: KernelCache,
    source_release: str = "v8.2.3",
) -> dict[str, Any]:
    if source_release == V841_SOURCE_RELEASE:
        inventory = v841_reached_translation_units()
    elif source_release == "v8.2.3":
        inventory = production_translation_units()
    else:
        raise CudaDualRunError(
            f"unsupported CUDA source release {source_release!r}"
        )
    for module_key, (source, names) in inventory.items():
        cache.raw_kernels(names, source, module_key=module_key)
    manifest = cache.compile_manifest()
    _compile_relation(manifest, source_release)
    return manifest


def prepare_cuda_kernel_cache(
    capability: CudaCapability,
    cache_dir: str | Path,
    *,
    ftz_binding: Mapping[str, Any],
    source_release: str = "v8.2.3",
) -> KernelCache:
    """Compile the executable once for both independently uploaded arms."""

    selected_cache = Path(cache_dir).expanduser().resolve()
    if selected_cache.exists() and any(selected_cache.iterdir()):
        raise CudaDualRunError(
            f"prepared CUDA cache must be absent or empty: {selected_cache}"
        )
    selected_cache.mkdir(parents=True, exist_ok=True)
    cache = KernelCache(capability=capability, cache_dir=selected_cache)
    manifest = _resolve_full_compile_manifest(cache, source_release)
    manifest_sha256 = canonical_sha256(manifest)
    binding_value = ftz_binding.get("value")
    if not isinstance(binding_value, Mapping):
        raise CudaDualRunError("FTZ binding record has no value")
    expected_schema = (
        MPAS_FTZ_V841_SCHEMA
        if source_release == V841_SOURCE_RELEASE
        else MPAS_FTZ_SCHEMA
    )
    if (
        ftz_binding.get("schema") != expected_schema
        or binding_value.get("schema") != expected_schema
    ):
        raise CudaDualRunError("prepared FTZ binding schema differs from source release")
    binding_manifest_sha = binding_value.get("compile_relation", {}).get(
        "compile_manifest_sha256"
    )
    if binding_manifest_sha != manifest_sha256:
        raise CudaDualRunError(
            "prepared compile manifest does not match the validated FTZ binding: "
            f"{manifest_sha256} != {binding_manifest_sha}"
        )
    if (
        source_release == V841_SOURCE_RELEASE
        and binding_value.get("compile_relation")
        != validate_v841_compile_manifest_relation(manifest)
    ):
        raise CudaDualRunError(
            "prepared v8.4.1 FTZ binding has a false compile relation"
        )
    return cache


def run_cuda_arm_generic(
    prepared: PreparedCudaInputs,
    config: DryDycoreConfig,
    *,
    steps: int,
    kernel_cache: KernelCache,
    ftz_binding: Mapping[str, Any],
    comparison_authority: Mapping[str, Any],
    authority_ruler: CudaAuthorityRulerBinder | None = None,
) -> CudaArmRun:
    """Execute one independent upload against one shared prepared executable."""

    if steps < 3:
        raise CudaDualRunError("a durable CUDA arm requires at least three steps")
    if not prepared.profile or not prepared.target or not prepared.preparation_method:
        raise CudaDualRunError(
            "prepared CUDA inputs require nonempty profile, target, and method"
        )
    if not prepared.input_bytes:
        raise CudaDualRunError("prepared CUDA inputs require input-byte bindings")
    sealed_execution = prepared.expected_execution_fingerprint
    if not isinstance(sealed_execution, Mapping):
        raise CudaDualRunError(
            "prepared CUDA inputs must be constructed with PreparedCudaInputs.validated"
        )
    configuration = _configuration_record(config)
    _validate_execution_fingerprint(
        sealed_execution,
        configuration_sha256=configuration["sha256"],
    )
    current_host_execution = fingerprint_prepared_execution(prepared, config)
    if current_host_execution != sealed_execution:
        raise CudaDualRunError(
            "prepared host execution inputs changed after they were sealed"
        )
    manifest = kernel_cache.compile_manifest()
    source_release = getattr(config, "source_release", "v8.2.3")
    if source_release == V841_SOURCE_RELEASE and authority_ruler is not None:
        raise CudaDualRunError(
            "v8.4.1 CUDA cannot adopt a dual-run authority ruler before the native nonzero-tracer gate"
        )
    relation = _compile_relation(manifest, source_release)
    manifest_sha256 = relation["compile_manifest_sha256"]
    binding_value = ftz_binding.get("value")
    if not isinstance(binding_value, Mapping):
        raise CudaDualRunError("FTZ binding record has no value")
    expected_schema = (
        MPAS_FTZ_V841_SCHEMA
        if source_release == V841_SOURCE_RELEASE
        else MPAS_FTZ_SCHEMA
    )
    if (
        ftz_binding.get("schema") != expected_schema
        or binding_value.get("schema") != expected_schema
    ):
        raise CudaDualRunError("CUDA arm FTZ binding schema differs from source release")
    binding_manifest_sha = binding_value.get("compile_relation", {}).get(
        "compile_manifest_sha256"
    )
    if binding_manifest_sha != manifest_sha256:
        raise CudaDualRunError(
            "prepared executable does not match the validated FTZ binding: "
            f"{manifest_sha256} != {binding_manifest_sha}"
        )
    if (
        source_release == V841_SOURCE_RELEASE
        and binding_value.get("compile_relation") != relation
    ):
        raise CudaDualRunError("v8.4.1 FTZ binding has a false compile relation")

    driver_kwargs: dict[str, Any] = {
        "saved_diagnostics": prepared.saved_diagnostics,
        "terrain_metrics": prepared.terrain_metrics,
        "kernel_cache": kernel_cache,
        "advection_coefficients": prepared.advection_coefficients,
    }
    if authority_ruler is not None:
        driver_kwargs["authority_ruler"] = authority_ruler
    if source_release == V841_SOURCE_RELEASE:
        driver_kwargs["reference_wind_profiles"] = prepared.reference_wind_profiles
    driver = CudaDryDycoreDriver.from_host(
        prepared.mesh,
        prepared.state,
        prepared.vertical,
        prepared.reference,
        config,
        **driver_kwargs,
    )
    uploaded_execution = fingerprint_uploaded_execution(driver)
    if uploaded_execution != sealed_execution:
        raise CudaDualRunError(
            "independent device upload changed the complete prepared execution input"
        )
    uploaded_initial = fingerprint_atmosphere(driver.atmosphere)
    host_initial = _fingerprint_host_preparation(prepared)
    if uploaded_initial != host_initial:
        raise CudaDualRunError(
            "independent device upload changed the prepared initial state or sidecar"
        )

    records: list[dict[str, Any]] = []
    for index in range(1, steps + 1):
        result = driver.step_device()
        record = _step_record(index, result)
        if result.receipt.compile_manifest_sha256 != manifest_sha256:
            raise CudaDualRunError(
                "compile manifest changed during the CUDA trajectory"
            )
        if result.receipt.layout_contract != CUDA_LAYOUT_CONTRACT:
            raise CudaDualRunError("CUDA step receipt changed the admitted layout")
        expected_d2h = 4 if source_release == V841_SOURCE_RELEASE else 0
        if result.receipt.d2h.bytes != expected_d2h:
            raise CudaDualRunError(
                "CUDA step internal D2H differs from its release contract: "
                f"{result.receipt.d2h.bytes} != {expected_d2h}"
            )
        records.append(record)
        # The public resident-step result is the next step's atmosphere.  The
        # explicit assignment is the long-run chaining boundary; no host state
        # is fed back into the forecast.
        driver.atmosphere = result.atmosphere

    preparation = _preparation_record(
        configuration,
        prepared.input_bytes,
        host_initial,
        sealed_execution,
        profile=prepared.profile,
        target=prepared.target,
        method=prepared.preparation_method,
    )
    trajectory_core = {
        "target": prepared.target,
        "steps": int(steps),
        "dt_seconds": float(config.config_dt),
        "initial_snapshot": host_initial,
        "initial_execution_fingerprint": dict(sealed_execution),
        "step_records": records,
        "final_snapshot_sha256": records[-1]["snapshot"]["sha256"],
    }
    trajectory = {**trajectory_core, "sha256": _json_sha256(trajectory_core)}
    layout = {
        "value": dict(CUDA_LAYOUT_CONTRACT),
        "sha256": canonical_sha256(CUDA_LAYOUT_CONTRACT),
    }
    compile_record = {"value": manifest, "sha256": manifest_sha256}
    capsule = {
        "schema": CAPSULE_SCHEMA,
        "profile": prepared.profile,
        "evidence": {
            "claim": (
                "one complete resident CUDA trajectory from an independently "
                "uploaded copy of the shared host preparation"
            ),
            "comparison_requirement": (
                "gpuwm total capsule equality against the second independent arm"
            ),
            "timing_policy": "nondeterministic timings and timestamps are omitted",
        },
        "input_bytes": dict(prepared.input_bytes),
        "configuration": configuration,
        "preparation": preparation,
        "device": _device_record(kernel_cache.capability),
        "contracts": {
            "compile_manifest": compile_record,
            "layout": layout,
            "ftz_binding": dict(ftz_binding),
            "comparison_authority": dict(comparison_authority),
            "implementation_sources": _source_bindings(source_release),
        },
        "execution": {
            "upload": "independent host-to-device upload for this arm",
            "cache": "shared prepare-once KernelCache executable",
            "between_steps": "device-resident atmosphere chaining",
            "instrumentation": (
                (
                    "explicit complete execution-input D2H before the trajectory and "
                    "state+sidecar D2H after every completed step; v8.4.1 also "
                    "performs one declared four-byte pre-publication validation-flag "
                    "D2H"
                )
                if source_release == V841_SOURCE_RELEASE
                else (
                    "explicit complete execution-input D2H before the trajectory and "
                    "state+sidecar D2H after every completed step"
                )
            ),
        },
        "trajectory": trajectory,
    }
    validate_cuda_capsule(capsule)
    return CudaArmRun(capsule=capsule, final_atmosphere=driver.atmosphere)


def run_cuda_arm(
    prepared: PreparedJwInputs,
    config: DryDycoreConfig,
    *,
    steps: int,
    kernel_cache: KernelCache,
    ftz_binding: Mapping[str, Any],
    comparison_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Run unlinked JW durability; exact t0 authority is a separate gate."""

    generic = PreparedCudaInputs.validated(
        config=config,
        profile="jw-x1.2562-native-dry-nomix",
        target="JW x1.2562 native dry no-mix CUDA durability lane",
        preparation_method=(
            "one native JW host preparation reused for two independent device uploads"
        ),
        mesh=prepared.mesh,
        state=prepared.state,
        vertical=prepared.vertical,
        reference=prepared.reference,
        saved_diagnostics=prepared.saved_diagnostics,
        terrain_metrics=prepared.terrain_metrics,
        input_bytes=prepared.input_bytes,
    )
    return run_cuda_arm_generic(
        generic,
        config,
        steps=steps,
        kernel_cache=kernel_cache,
        ftz_binding=ftz_binding,
        comparison_authority=comparison_authority,
    ).capsule


def _validate_array_record(name: str, record: Mapping[str, Any]) -> None:
    if record.get("name") != name:
        raise CudaDualRunError(f"array record {name!r} changed its name")
    _require_sha256(record.get("bytes_sha256"), f"{name}.bytes_sha256")
    declared = _require_sha256(record.get("sha256"), f"{name}.sha256")
    core = {
        "name": record.get("name"),
        "dtype": record.get("dtype"),
        "shape": record.get("shape"),
        "bytes": record.get("bytes"),
        "bytes_sha256": record.get("bytes_sha256"),
    }
    if declared != _json_sha256(core):
        raise CudaDualRunError(f"array record {name!r} has a false semantic digest")


def _validate_group(record: Mapping[str, Any], label: str) -> None:
    fields = record.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise CudaDualRunError(f"{label} has no array fields")
    for name, value in fields.items():
        if not isinstance(value, Mapping):
            raise CudaDualRunError(f"{label}.{name} is not an array record")
        _validate_array_record(str(name), value)
    core = {"metadata": record.get("metadata"), "fields": dict(fields)}
    if record.get("sha256") != _json_sha256(core):
        raise CudaDualRunError(f"{label} has a false group digest")


def _validate_snapshot(record: Mapping[str, Any], label: str) -> None:
    state = record.get("state")
    saved = record.get("saved_diagnostics")
    if not isinstance(state, Mapping) or not isinstance(saved, Mapping):
        raise CudaDualRunError(f"{label} is missing state or sidecar")
    if tuple(sorted(state.get("fields", {}))) != tuple(sorted(STATE_FIELDS)):
        raise CudaDualRunError(f"{label} state inventory changed")
    if tuple(sorted(saved.get("fields", {}))) != tuple(sorted(SIDECAR_FIELDS)):
        raise CudaDualRunError(f"{label} sidecar inventory changed")
    _validate_group(state, f"{label}.state")
    _validate_group(saved, f"{label}.saved_diagnostics")
    core = {
        "model_time_seconds": record.get("model_time_seconds"),
        "state": dict(state),
        "saved_diagnostics": dict(saved),
    }
    if record.get("sha256") != _json_sha256(core):
        raise CudaDualRunError(f"{label} has a false snapshot digest")


def _reject_nondeterministic_fields(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            selected = str(key)
            child_path = f"{path}.{selected}" if path else selected
            if selected in _NONDETERMINISTIC_KEYS:
                raise CudaDualRunError(
                    f"capsule contains nondeterministic field {child_path}"
                )
            _reject_nondeterministic_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nondeterministic_fields(child, f"{path}[{index}]")


def validate_cuda_capsule(capsule: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all internal digests and mandatory long-run bindings."""

    if capsule.get("schema") != CAPSULE_SCHEMA:
        raise CudaDualRunError("CUDA dual-run capsule schema is not v2")
    _reject_nondeterministic_fields(capsule)

    configuration = capsule.get("configuration")
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get("value"), Mapping
    ):
        raise CudaDualRunError("capsule has no configuration document")
    config_sha = canonical_sha256(configuration["value"])
    if configuration.get("sha256") != config_sha:
        raise CudaDualRunError("capsule configuration digest is false")
    configuration_value = dict(configuration["value"])
    source_release = configuration_value.get("source_release", "v8.2.3")
    if source_release == V841_SOURCE_RELEASE:
        try:
            config = V841DryDycoreConfig(**configuration_value)
            config.validate()
            CudaDryDycoreDriver._validate_config(config)
        except (ConfigurationRefusal, TypeError, ValueError) as error:
            raise CudaDualRunError(
                f"capsule v8.4.1 configuration is invalid: {error}"
            ) from error
    else:
        config = DryDycoreConfig.from_mapping(configuration_value)

    profile = capsule.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise CudaDualRunError("capsule profile must be a nonempty string")

    input_bytes = capsule.get("input_bytes")
    if not isinstance(input_bytes, Mapping) or not input_bytes:
        raise CudaDualRunError("capsule input-byte inventory is empty")
    for label, record in input_bytes.items():
        if not isinstance(label, str) or not label:
            raise CudaDualRunError("capsule input-byte label is empty")
        if not isinstance(record, Mapping) or int(record.get("bytes", 0)) <= 0:
            raise CudaDualRunError(f"input record {label!r} is invalid")
        _require_sha256(record.get("sha256"), f"input_bytes.{label}.sha256")

    contracts = capsule.get("contracts")
    if not isinstance(contracts, Mapping):
        raise CudaDualRunError("capsule has no contract bindings")
    compile_record = contracts.get("compile_manifest")
    if not isinstance(compile_record, Mapping) or not isinstance(
        compile_record.get("value"), Mapping
    ):
        raise CudaDualRunError("capsule has no compile manifest")
    manifest = compile_record["value"]
    relation = _compile_relation(manifest, source_release)
    manifest_sha = relation["compile_manifest_sha256"]
    if compile_record.get("sha256") != manifest_sha:
        raise CudaDualRunError("capsule compile-manifest digest is false")

    layout = contracts.get("layout")
    if not isinstance(layout, Mapping) or layout.get("value") != CUDA_LAYOUT_CONTRACT:
        raise CudaDualRunError("capsule layout contract changed")
    if layout.get("sha256") != canonical_sha256(CUDA_LAYOUT_CONTRACT):
        raise CudaDualRunError("capsule layout digest is false")

    ftz = contracts.get("ftz_binding")
    if not isinstance(ftz, Mapping) or not isinstance(ftz.get("value"), Mapping):
        raise CudaDualRunError("capsule has no FTZ binding")
    binding = ftz["value"]
    expected_ftz_schema = (
        MPAS_FTZ_V841_SCHEMA
        if source_release == V841_SOURCE_RELEASE
        else MPAS_FTZ_SCHEMA
    )
    if binding.get("schema") != expected_ftz_schema:
        raise CudaDualRunError("capsule FTZ binding schema changed")
    _require_sha256(ftz.get("artifact_sha256"), "ftz_binding.artifact_sha256")
    if ftz.get("sha256") != _json_sha256(binding):
        raise CudaDualRunError("capsule FTZ binding digest is false")
    binding_manifest = binding.get("compile_manifest")
    if not isinstance(binding_manifest, Mapping):
        raise CudaDualRunError("FTZ binding has no compile manifest")
    if canonical_sha256(binding_manifest) != manifest_sha:
        raise CudaDualRunError("FTZ binding and arm compile manifests differ")
    if (
        binding.get("compile_relation", {}).get("compile_manifest_sha256")
        != manifest_sha
    ):
        raise CudaDualRunError("FTZ compile relation digest is false")
    if (
        source_release == V841_SOURCE_RELEASE
        and binding.get("compile_relation") != relation
    ):
        raise CudaDualRunError(
            "v8.4.1 FTZ binding does not carry the exact reached compile relation"
        )

    authority = contracts.get("comparison_authority")
    if not isinstance(authority, Mapping):
        raise CudaDualRunError("capsule has no gpuwm comparison authority")
    if authority.get("schema") != COMPARISON_AUTHORITY_SCHEMA:
        raise CudaDualRunError("gpuwm comparison-authority schema changed")
    _require_sha256(authority.get("source_sha256"), "gpuwm dualrun source")
    if authority.get("functions") != ["compare_capsules", "compare_capsule_files"]:
        raise CudaDualRunError("gpuwm comparison function inventory changed")
    if authority.get("comparison_schema") != EXPECTED_GPUWM_COMPARISON_SCHEMA:
        raise CudaDualRunError("gpuwm comparison report schema changed")

    implementation_sources = contracts.get("implementation_sources")
    expected_sources = _source_bindings(source_release)
    if implementation_sources != expected_sources:
        raise CudaDualRunError(
            "capsule implementation-source inventory differs from the live "
            "certified implementation"
        )

    preparation = capsule.get("preparation")
    if not isinstance(preparation, Mapping):
        raise CudaDualRunError("capsule has no shared-preparation binding")
    preparation_core = {
        "profile": preparation.get("profile"),
        "target": preparation.get("target"),
        "method": preparation.get("method"),
        "configuration_sha256": preparation.get("configuration_sha256"),
        "input_bytes": preparation.get("input_bytes"),
        "initial_snapshot_sha256": preparation.get("initial_snapshot_sha256"),
        "initial_execution_fingerprint_sha256": preparation.get(
            "initial_execution_fingerprint_sha256"
        ),
    }
    if preparation_core["profile"] != profile:
        raise CudaDualRunError("preparation profile binding is false")
    if (
        not isinstance(preparation_core["target"], str)
        or not preparation_core["target"]
    ):
        raise CudaDualRunError("preparation target is empty")
    if (
        not isinstance(preparation_core["method"], str)
        or not preparation_core["method"]
    ):
        raise CudaDualRunError("preparation method is empty")
    if preparation_core["configuration_sha256"] != config_sha:
        raise CudaDualRunError("preparation configuration binding is false")
    if preparation_core["input_bytes"] != input_bytes:
        raise CudaDualRunError("preparation input-byte binding is false")
    if preparation.get("sha256") != _json_sha256(preparation_core):
        raise CudaDualRunError("preparation digest is false")

    trajectory = capsule.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise CudaDualRunError("capsule has no trajectory")
    if trajectory.get("target") != preparation_core["target"]:
        raise CudaDualRunError("trajectory target differs from preparation target")
    records = trajectory.get("step_records")
    steps = int(trajectory.get("steps", -1))
    if not isinstance(records, list) or steps < 3 or len(records) != steps:
        raise CudaDualRunError("trajectory is not a complete durable run")
    if float(trajectory.get("dt_seconds", np.nan)) != float(config.config_dt):
        raise CudaDualRunError("trajectory timestep differs from configuration")
    initial_snapshot = trajectory.get("initial_snapshot")
    if not isinstance(initial_snapshot, Mapping):
        raise CudaDualRunError("trajectory has no initial snapshot")
    _validate_snapshot(initial_snapshot, "trajectory.initial_snapshot")
    if preparation_core["initial_snapshot_sha256"] != initial_snapshot.get("sha256"):
        raise CudaDualRunError("preparation initial-snapshot binding is false")
    initial_execution = trajectory.get("initial_execution_fingerprint")
    if not isinstance(initial_execution, Mapping):
        raise CudaDualRunError(
            "trajectory has no complete initial execution-input fingerprint"
        )
    _validate_execution_fingerprint(
        initial_execution,
        configuration_sha256=config_sha,
    )
    if preparation_core[
        "initial_execution_fingerprint_sha256"
    ] != initial_execution.get("sha256"):
        raise CudaDualRunError(
            "preparation execution-input fingerprint binding is false"
        )
    admitted_capsule_initial = initial_execution["authority_initial"]
    for group_name, field_names in (
        ("state", STATE_FIELDS),
        ("saved_diagnostics", SIDECAR_FIELDS),
    ):
        admitted_group = admitted_capsule_initial[group_name]
        snapshot_fields = initial_snapshot[group_name]["fields"]
        for name in field_names:
            admitted_array = admitted_group[name]
            snapshot_array = snapshot_fields[name]
            try:
                same_dtype = np.dtype(admitted_array["dtype"]) == np.dtype(
                    snapshot_array["dtype"]
                )
            except TypeError:
                same_dtype = False
            if (
                not same_dtype
                or admitted_array["shape"] != snapshot_array["shape"]
                or admitted_array["c_bytes"] != snapshot_array["bytes"]
                or admitted_array["c_bytes_sha256"] != snapshot_array["bytes_sha256"]
            ):
                raise CudaDualRunError(
                    "complete execution fingerprint differs from initial snapshot "
                    f"{group_name}.{name}"
                )

    expected_start = float(initial_snapshot["model_time_seconds"])
    is_v841 = source_release == V841_SOURCE_RELEASE
    dynamics_schedule = RKSchedule.from_mpas(
        config.config_dt,
        order=config.config_time_integration_order,
        acoustic_substeps=config.config_number_of_sub_steps,
        dynamics_splits=config.config_dynamics_split_steps,
    )
    expected_acoustic_steps = tuple(
        stage.acoustic_steps for stage in dynamics_schedule.stages
    )
    expected_dynamics_stage_timesteps = tuple(
        stage.large_timestep for stage in dynamics_schedule.stages
    )
    scalar_shape = initial_snapshot["state"]["fields"]["scalars"].get("shape")
    if (
        not isinstance(scalar_shape, list)
        or not scalar_shape
        or not isinstance(scalar_shape[0], int)
        or scalar_shape[0] < 0
    ):
        raise CudaDualRunError("initial scalar fingerprint has an invalid shape")
    expected_scalar_stage_timesteps: tuple[float, float, float] | None = None
    if is_v841 and config.config_scalar_advection and scalar_shape[0] > 0:
        scalar_schedule = RKSchedule.from_mpas(
            config.config_dt,
            order=config.config_time_integration_order,
            acoustic_substeps=config.config_number_of_sub_steps,
            dynamics_splits=1,
        )
        expected_scalar_stage_timesteps = tuple(
            stage.large_timestep for stage in scalar_schedule.stages
        )
    for expected_index, record in enumerate(records, 1):
        if not isinstance(record, Mapping) or record.get("step") != expected_index:
            raise CudaDualRunError("trajectory step indices are not contiguous")
        start = float(record.get("start_time_seconds", np.nan))
        end = float(record.get("end_time_seconds", np.nan))
        if start != expected_start or end != start + config.config_dt:
            raise CudaDualRunError(
                f"trajectory model-time sequence broke at step {expected_index}"
            )
        snapshot = record.get("snapshot")
        contract = record.get("step_contract")
        if not isinstance(snapshot, Mapping) or not isinstance(contract, Mapping):
            raise CudaDualRunError(f"trajectory step {expected_index} is incomplete")
        expected_contract_fields = (
            _V841_STEP_CONTRACT_FIELDS if is_v841 else _BASE_STEP_CONTRACT_FIELDS
        )
        if frozenset(contract) != expected_contract_fields:
            raise CudaDualRunError(
                f"trajectory step {expected_index} contract inventory changed"
            )
        _validate_snapshot(snapshot, f"trajectory.step_records[{expected_index - 1}]")
        if float(snapshot.get("model_time_seconds", np.nan)) != end:
            raise CudaDualRunError("step snapshot time differs from receipt time")
        if contract.get("compile_manifest_sha256") != manifest_sha:
            raise CudaDualRunError("step compile-manifest binding changed")
        if contract.get("layout_contract_sha256") != layout["sha256"]:
            raise CudaDualRunError("step layout binding changed")
        step_configuration = contract.get("configuration")
        if not isinstance(step_configuration, Mapping):
            raise CudaDualRunError("step has no full configuration payload")
        if dict(step_configuration) != dict(configuration["value"]):
            raise CudaDualRunError("step configuration payload changed")
        if contract.get("configuration_sha256") != config_sha:
            raise CudaDualRunError("step configuration digest changed")
        expected_frozen_source = V841_CUDA_SOURCE if is_v841 else FROZEN_CUDA_SOURCE
        if contract.get("frozen_source") != expected_frozen_source:
            raise CudaDualRunError("step frozen-source authority changed")
        if contract.get("t0_diagnostics_source") != CUDA_T0_EXACT_SIDECAR:
            raise CudaDualRunError("step t0 diagnostics source changed")
        if tuple(contract.get("stage_acoustic_steps", ())) != (expected_acoustic_steps):
            raise CudaDualRunError("step acoustic RK schedule changed")
        if is_v841:
            if contract.get("source_release") != V841_SOURCE_RELEASE:
                raise CudaDualRunError("step v8.4.1 source-release binding changed")
            if contract.get("dynamics_split_steps") != 3:
                raise CudaDualRunError("step v8.4.1 dynamics split count changed")
            if float(
                contract.get("dynamics_timestep_seconds", np.nan)
            ) != float(config.config_dt / config.config_dynamics_split_steps):
                raise CudaDualRunError("step v8.4.1 dynamics timestep changed")
            if tuple(contract.get("dynamics_stage_timesteps", ())) != (
                expected_dynamics_stage_timesteps
            ):
                raise CudaDualRunError("step v8.4.1 dynamics RK schedule changed")
            scalar_timesteps = contract.get("scalar_transport_stage_timesteps")
            if (
                None if scalar_timesteps is None else tuple(scalar_timesteps)
            ) != expected_scalar_stage_timesteps:
                raise CudaDualRunError("step v8.4.1 scalar RK schedule changed")
            if contract.get("split_flux_reduction") != SPLIT_FLUX_REDUCTION:
                raise CudaDualRunError("step v8.4.1 split-flux reduction changed")
            if tuple(contract.get("authority_nonclaims", ())) != (
                CUDA_V841_AUTHORITY_NONCLAIMS
            ):
                raise CudaDualRunError("step v8.4.1 authority nonclaims changed")
        ruler = contract.get("authority_ruler")
        ruler_sha256 = contract.get("authority_ruler_sha256")
        evidence = contract.get("evidence")
        if ruler is None:
            if ruler_sha256 is not None:
                raise CudaDualRunError("unbound step has an authority-ruler digest")
            expected_evidence = (
                CUDA_V841_IMPLEMENTED_UNLINKED_EVIDENCE
                if is_v841
                else CUDA_IMPLEMENTED_UNLINKED_EVIDENCE
            )
            if evidence != expected_evidence:
                raise CudaDualRunError("unbound step claims linked authority evidence")
        else:
            if is_v841:
                raise CudaDualRunError(
                    "v8.4.1 CUDA cannot bind an authority ruler before the nonzero-tracer gate"
                )
            if not isinstance(ruler, Mapping):
                raise CudaDualRunError("step authority ruler is not a mapping")
            if ruler.get("schema") != CUDA_AUTHORITY_RULER_SCHEMA:
                raise CudaDualRunError("step authority-ruler schema changed")
            if ruler.get("admitted_configuration_sha256") != config_sha:
                raise CudaDualRunError(
                    "step authority ruler admits a different configuration"
                )
            if ruler_sha256 != canonical_sha256(ruler):
                raise CudaDualRunError("step authority-ruler digest is false")
            if expected_index != 1:
                raise CudaDualRunError(
                    "a frozen t0 authority ruler cannot be reused after step one"
                )
            if ruler.get("admitted_start_time_seconds") != start:
                raise CudaDualRunError(
                    "step authority ruler admits a different model start time"
                )
            if start != float(initial_snapshot["model_time_seconds"]):
                raise CudaDualRunError(
                    "linked authority step does not start from the capsule t0"
                )
            if evidence not in (
                CUDA_WHOLE_STEP_EVIDENCE,
                CUDA_ORIGINAL_JW_BRANCH_EVIDENCE,
            ):
                raise CudaDualRunError(
                    "bound step does not carry a registered linked evidence label"
                )
            admitted_initial = ruler.get("admitted_initial_fingerprint")
            if not isinstance(admitted_initial, Mapping):
                raise CudaDualRunError(
                    "step authority ruler has no admitted initial fingerprint"
                )
            try:
                CudaAuthorityRulerBinder(
                    evidence=str(evidence),
                    payload=dict(ruler),
                    payload_sha256=str(ruler_sha256),
                ).validate(config_sha, admitted_initial)
            except (TypeError, ValueError) as error:
                raise CudaDualRunError(
                    f"step authority-ruler payload is invalid: {error}"
                ) from error
            if not isinstance(admitted_initial.get("execution_inputs"), Mapping):
                raise CudaDualRunError(
                    "linked authority ruler does not bind complete execution inputs"
                )
            if admitted_initial != admitted_capsule_initial:
                raise CudaDualRunError(
                    "linked authority ruler differs from the independently recorded "
                    "complete execution input"
                )
            for group_name, field_names in (
                ("state", STATE_FIELDS),
                ("saved_diagnostics", SIDECAR_FIELDS),
            ):
                admitted_group = admitted_initial.get(group_name)
                snapshot_group = initial_snapshot.get(group_name)
                if not isinstance(admitted_group, Mapping) or not isinstance(
                    snapshot_group, Mapping
                ):
                    raise CudaDualRunError(
                        f"linked authority {group_name} binding is incomplete"
                    )
                snapshot_fields = snapshot_group.get("fields")
                if not isinstance(snapshot_fields, Mapping):
                    raise CudaDualRunError(
                        f"capsule initial {group_name} inventory is missing"
                    )
                for name in field_names:
                    admitted_array = admitted_group.get(name)
                    snapshot_array = snapshot_fields.get(name)
                    if not isinstance(admitted_array, Mapping) or not isinstance(
                        snapshot_array, Mapping
                    ):
                        raise CudaDualRunError(
                            f"linked authority initial {group_name}.{name} is missing"
                        )
                    try:
                        same_dtype = np.dtype(admitted_array.get("dtype")) == np.dtype(
                            snapshot_array.get("dtype")
                        )
                    except TypeError:
                        same_dtype = False
                    if (
                        not same_dtype
                        or admitted_array.get("shape") != snapshot_array.get("shape")
                        or admitted_array.get("c_bytes") != snapshot_array.get("bytes")
                        or admitted_array.get("c_bytes_sha256")
                        != snapshot_array.get("bytes_sha256")
                    ):
                        raise CudaDualRunError(
                            "linked authority admitted t0 differs from capsule initial "
                            f"{group_name}.{name}"
                        )
        expected_d2h_bytes = 4 if is_v841 else 0
        if contract.get("d2h_bytes_inside_step") != expected_d2h_bytes:
            raise CudaDualRunError("step contains an undeclared internal D2H copy")
        step_core = {
            "step": record.get("step"),
            "start_time_seconds": record.get("start_time_seconds"),
            "end_time_seconds": record.get("end_time_seconds"),
            "snapshot": dict(snapshot),
            "step_contract": dict(contract),
        }
        if record.get("sha256") != _json_sha256(step_core):
            raise CudaDualRunError(f"step {expected_index} digest is false")
        expected_start = end

    if trajectory.get("final_snapshot_sha256") != records[-1]["snapshot"]["sha256"]:
        raise CudaDualRunError("trajectory final snapshot binding is false")
    trajectory_core = {
        "target": trajectory.get("target"),
        "steps": trajectory.get("steps"),
        "dt_seconds": trajectory.get("dt_seconds"),
        "initial_snapshot": dict(initial_snapshot),
        "initial_execution_fingerprint": dict(initial_execution),
        "step_records": records,
        "final_snapshot_sha256": trajectory.get("final_snapshot_sha256"),
    }
    if trajectory.get("sha256") != _json_sha256(trajectory_core):
        raise CudaDualRunError("trajectory digest is false")
    return json.loads(json.dumps(dict(capsule), sort_keys=True))


def compare_cuda_capsule_files(
    path_a: str | Path,
    path_b: str | Path,
    *,
    gpuwm_root: str | Path,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate both capsules and compare every leaf with gpuwm's authority."""

    capsule_a = _load_json_object(path_a)
    capsule_b = _load_json_object(path_b)
    validate_cuda_capsule(capsule_a)
    validate_cuda_capsule(capsule_b)
    module, authority = load_gpuwm_dualrun(gpuwm_root)
    for label, capsule in (("a", capsule_a), ("b", capsule_b)):
        recorded = capsule["contracts"]["comparison_authority"]
        if recorded != authority:
            raise CudaDualRunError(
                f"capsule {label} is not bound to the live gpuwm comparator source"
            )
    comparison = module.compare_capsule_files(path_a, path_b)
    gpuwm_report = comparison.report()
    if gpuwm_report.get("schema") != EXPECTED_GPUWM_COMPARISON_SCHEMA:
        raise CudaDualRunError("gpuwm returned an unexpected comparison schema")
    report = {
        "schema": REPORT_SCHEMA,
        "comparison_authority": authority,
        "capsules": {
            "a": {"sha256": sha256_file(path_a)},
            "b": {"sha256": sha256_file(path_b)},
        },
        "gpuwm_comparison": gpuwm_report,
        "total_comparison": True,
    }
    if report_path is not None:
        write_json_atomic(report_path, report)
    return report


__all__ = [
    "CAPSULE_SCHEMA",
    "COMPARISON_AUTHORITY_SCHEMA",
    "CudaArmRun",
    "CudaDualRunError",
    "EXECUTION_INPUT_FINGERPRINT_SCHEMA",
    "EXPECTED_GPUWM_COMPARISON_SCHEMA",
    "PreparedCudaInputs",
    "PreparedJwInputs",
    "REPORT_SCHEMA",
    "SIDECAR_FIELDS",
    "STATE_FIELDS",
    "compare_cuda_capsule_files",
    "derive_step_count",
    "fingerprint_array",
    "fingerprint_array_group",
    "fingerprint_atmosphere",
    "fingerprint_prepared_execution",
    "fingerprint_uploaded_execution",
    "jw_day_config",
    "load_ftz_binding_record",
    "load_gpuwm_dualrun",
    "prepare_cuda_kernel_cache",
    "prepare_jw_inputs",
    "run_cuda_arm",
    "run_cuda_arm_generic",
    "sha256_file",
    "validate_cuda_capsule",
    "write_json_atomic",
]
