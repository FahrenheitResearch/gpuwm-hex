#!/usr/bin/env python3
"""Engineering forecast driver for the MPAS-A v8.4.1 CUDA port.

DERIVED, NOT A PROOF.  This tool is a parameterized fork of the sealed proof
harness ``tools/run_cuda_v841_full_physics_x4.py``.  It exists because that
harness is a one-case proof: it pins the init file's SHA-256, its start time,
its step count and its per-case F000 surface-diagnostic bytes, and it spends
half its run comparing against a sealed native CPU authority that exists for
exactly one case.  None of that can execute a forecast from a different
initial condition.

THE FORK IS IMPORT-BASED ON PURPOSE.  Every executing model path -- mesh load,
overlays, constructor arrays, the staged two-owner composite step, the physics
configuration, the snapshot capture -- is CALLED FROM the proof module, not
copied.  Two functions are re-implemented here (``_prepare_host_execution`` and
``build_arwen_constructor_values``) because they carry case literals in their
bodies rather than in module constants; both re-implementations are faithful
transcriptions with the case assertions replaced by measurement, and the
values they hand to the model are constructed identically.  The proof module's
file bytes are never modified and its own entry point is unaffected.

WHAT IS KEPT (unchanged, still enforced, refusal on violation)
  * the fixed Arwen checkout pin: HEAD/tree/clean, verified before and after
    the run by ``verify_arwen_checkout_git`` (ARWEN_COMMIT, contract surface
    SHA-256, glacier composed-TU SHA-256);
  * ``EXECUTION_SOURCE_PINS``: exact SHA-256 of every executing port module,
    verified before CUDA is imported and again after the forecast;
  * the mesh authority pins: grid and static file byte counts and SHA-256;
  * the rotation-aware precision-preserving mesh/static pair load, and both
    in-memory init overlays (reconstruction coefficients, edge normals) with
    every structural check they carry (topology equality with the prepared
    mesh, exact +0 padding, finiteness, placeholder provenance);
  * the full physics configuration exactly as proven:
    ``V841MpasColumnPhysicsSmagorinskyGwdoConfig`` by default (native's
    Registry deformation-based 2-D Smagorinsky horizontal mixing; a NEW
    sub-series, not bit-comparable to mixing-off arms), or the pre-mixing
    ``V841MpasColumnPhysicsGwdoConfig`` control under ``--horiz-mixing off``;
    both carry WSM6 + GF + YSU + external YSU-GWDO +
    revised-MO + NoahMP (+ glacier dispatch) + cloud fraction + legacy RRTMG,
    dt = 120 s, radiation 600 s, surface/PBL/cumulus 120 s, six-species scalar
    order, ``wsm6_hail_opt = 0``, ``xice_threshold = 0.02``, dx = 25000 m;
  * the staged two-owner transaction with its rollback contract
    (``execute_composite_step``), including the no-fail commit law;
  * the surface classification receipt and the NoahMP census/glacier-path
    check at every capture (the census is now measured from this init instead
    of pinned to the proof case, but a backend census that disagrees with the
    host classification still refuses);
  * the physical snapshot gate (finite everywhere, rho/theta/pressure > 0,
    soil moisture in [0,1], non-negative precipitation, hydrometeors >= 0);
  * the exact snapshot capture and its native-grid history writer.

WHAT IS REMOVED, AND WHY (every removal deliberate; guarantees dropped)
  1. THE INIT SHA-256 PIN.  ``AUTHORITY_PINS['init']`` fixes one file.  A
     forecast from another initial condition cannot satisfy it.  REPLACED BY:
     the actual init path, byte count and SHA-256 are measured and recorded in
     the receipt, and the receipt names the init's stated source.
     GUARANTEE DROPPED: this tool cannot tell you the init is the blessed one.
     It tells you exactly which bytes it ran.
  2. THE CASE-PINNED F000 SURFACE-DIAGNOSTIC PINS.  ``t2m``/``u10``/``v10``
     array SHA-256 for the proof case.  REPLACED BY: the same three fields are
     read from the supplied init, still required FP32 [Time,nCells], finite,
     and still overlaid only onto verified exact +0 placeholders; their
     digests are recorded.  GUARANTEE DROPPED: byte identity of the F000
     diagnostic overlay to the proof case.
  3. THE NATIVE-COMPARISON STAGE.  ``compare_snapshot_to_native`` and the six
     ``native_*`` authority files (F000/F030/F001 history, validation receipt,
     launch receipt, run closure), plus their RMSE gates.  Those files are the
     CPU authority for ONE case at ONE valid time; there is no such authority
     for any other case, and comparing a new forecast against them would be
     meaningless.  GUARANTEE DROPPED: no quantitative agreement with a native
     MPAS CPU run is established for these forecasts.  Nothing here is
     verified against native MPAS.
  4. THE CHECKPOINT/RESTART PROOF STAGE.  The F030 host checkpoint, the fresh
     restart worker process, F030 rehydration identity, first-resumed-step-16
     identity and the F001 bitwise restart comparison.  A forecast runs
     uninterrupted; the restart property is a property of the port and was
     proven by the release proof.  GUARANTEE DROPPED: these runs do not
     re-establish restart bitwise identity (and produce no checkpoint).
  5. THE CASE-PINNED INITIAL-CONTENT FINGERPRINTS, each replaced by a measured
     value recorded in the receipt:
       - ``NEGATIVE_QV_PIN`` (215 negative qv values at F000 in the proof
         case).  The count is now measured from this init; the physical gate
         still requires exactly that many at F000 and exactly zero after every
         subsequent step, so a clamp regression still refuses.
       - ``INIT_RECONSTRUCTION_COEFFICIENTS_PIN.init_carrier_raw_sha256`` and
         ``INIT_EDGE_NORMAL_VECTORS_PIN.init_carrier_raw_sha256`` plus the
         edge-normal activity counts and norm envelope.  These are mesh
         geometry regenerated by ``init_atmosphere`` per init, so they are
         recorded, not pinned.  ADDED IN THEIR PLACE: an explicit physical
         bound, every edge-normal row norm within 1e-4 of unity, which the
         pinned envelope previously implied.
       - ``EXPECTED_SURFACE_CLASSIFICATION`` / ``EXPECTED_NOAHMP_CENSUS`` /
         the xland, glacier-index, sea-ice-index and threshold-delta digests.
         Land/water/sea-ice/glacier counts are date-dependent (sea ice moves).
         Measured, recorded, and the backend's own census must still equal the
         host classification at every capture.
       - ``LANDMASK_CONSTRUCTOR_CAST_PIN``.  The exactness of the int32 ->
         FP32 cast is still checked (round-trip equality and the {0,1} value
         set); only the case digests are recorded instead of pinned.
       - ``EXPECTED_ARWEN_P_TOP_PA_F32`` / ``EXPECTED_TOP_PRESSURE_RANGE_PA``.
         p_top is derived from the init's own pressure field.  This driver
         re-derives the expectation with the identical FP32 reduction and
         seeds it; the proof function then recomputes it independently and
         still asserts equality, so a transcription error refuses instead of
         passing.  The derived scalar and the per-column min/median/max are
         recorded.
       - ``START_TIME_TEXT``.  The init's ``config_start_time`` is now the
         authority; ``--start-time`` is an assertion against it, not a source.
  6. THE FIXED 30-STEP / F000-F030-F001 SCHEDULE.  Replaced by ``--hours`` and
     ``--history-every-minutes``.  dt stays 120 s and every physics cadence
     stays as proven; only the number of steps and the capture set move.
     GUARANTEE DROPPED: the proof's three-snapshot schedule and its labels.

FORK-EQUIVALENCE GATE.  Because this driver reaches the model through the
proof module's own functions, a fork defect would show up as a changed
trajectory.  ``tools/gate_v841_forecast_fork_equivalence.py`` runs THIS driver
on the AUTHORITY init for 30 steps and compares its boundary fingerprints and
snapshot array digests bitwise against the release-proof uninterrupted arm.
BITWISE-IDENTICAL is required before any showcase forecast is run.

The claim, and the non-claims, are stated in the receipt.  These are
engineering forecasts on a 92-to-25 km variable-resolution global mesh.  They
establish no forecast skill.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import MappingProxyType
from typing import Any

import numpy as np

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_cuda_v841_full_physics_x4 as proof  # noqa: E402

from hexcore import dt_admission  # noqa: E402
from hexcore.errors import ConfigurationRefusal  # noqa: E402
from hexcore.mesh import (  # noqa: E402
    REGIONAL_BOUNDARY_MASK_NAMES,
    REGIONAL_BOUNDARY_ZONE_WIDTH,
    regional_boundary_mask_digest,
)

ROOT = proof.ROOT

SCHEMA = "mpas-port.cuda-v841-engineering-forecast/v1"
RECEIPT_MODE = "engineering-forecast"
RECEIPT_NAME = "cuda-v841-forecast-receipt.json"
DERIVED_FROM = "tools/run_cuda_v841_full_physics_x4.py"

N_CELLS = proof.N_CELLS
N_EDGES = proof.N_EDGES
N_LEVELS = proof.N_LEVELS
N_INTERFACES = proof.N_INTERFACES
N_SOIL_LEVELS = proof.N_SOIL_LEVELS
DT_SECONDS = proof.DT_SECONDS
SCALAR_NAMES = proof.SCALAR_NAMES
COLD_ZERO_SCALAR_NAMES = proof.COLD_ZERO_SCALAR_NAMES
SOURCE_SCALAR_NAMES = proof.SOURCE_SCALAR_NAMES
NOMINAL_DX_M = proof.NOMINAL_DX_M
ARWEN_XICE_THRESHOLD = proof.ARWEN_XICE_THRESHOLD

#: The run's cumulus selection, and why it was made.  ``bind_mesh`` rebinds
#: this from the BOUND mesh's own finest spacing, exactly as it rebinds
#: DT_SECONDS -- one decision, one source.  The value standing here before a
#: bind is the native x4 configuration this module's constants describe, so a
#: direct invocation with no bind behaves as it always did.
CONVECTION_DECISION: dict[str, Any] = {}

#: The run's surface/PBL cadence, and why it was chosen.  Travels the same
#: road as DT_SECONDS and CONVECTION_DECISION: ``bind_mesh`` takes the
#: decision once from the bound row's own timestep and rebinds it here, and
#: the driver refuses if its own request disagrees.  Empty before a bind
#: means the weld -- ``config_bldt_seconds = config_dt``, the proven
#: configuration -- so a direct invocation with no bind behaves as it always
#: did.  See :mod:`hexcore.pbl_cadence`.
PBL_CADENCE_DECISION: dict[str, Any] = {}

# Mesh authority roles kept under exact-byte pins.  ``init`` is deliberately
# absent: see removal 1.
MESH_AUTHORITY_ROLES = ("grid", "static")

EDGE_NORMAL_UNIT_TOLERANCE = 1.0e-4
VERTICAL_VELOCITY_REFUSAL_M_S = 200.0

DROPPED_GUARANTEES = (
    "init identity is recorded, not pinned: this tool runs whatever init bytes "
    "it is given and reports their SHA-256",
    "no comparison against a native MPAS CPU run is performed; the sealed "
    "native GF+YSU-GWDO authority applies to one case only",
    "no checkpoint is written and restart bitwise identity is not "
    "re-established by these runs",
    "the F000 surface-diagnostic overlay, the initial negative-qv fingerprint, "
    "the init-carried mesh-geometry digests, the surface classification and "
    "NoahMP census, the landmask cast digests and p_top are measured from the "
    "supplied init and recorded rather than pinned to the proof case",
    "forecast skill is not established; this is a 92-to-25 km variable-"
    "resolution global mesh and 25 km is its fine limit, not a nest",
)

CLAIM = (
    "one uninterrupted real-initialized x4.163842 MPAS-A v8.4.1 CUDA forecast "
    "using WSM6 + GF + YSU + external YSU-GWDO + revised-MO + NoahMP "
    "(with the CUDA glacier path) + cloud fraction + legacy RRTMG, at "
    "dt = 120 s, from the init named in this receipt"
)
NONCLAIMS = proof.NONCLAIMS + (
    "not a proof: the init pin, the native comparison stage and the "
    "checkpoint/restart stage of the sealed harness are deliberately absent "
    "(see dropped_guarantees)",
)


# --------------------------------------------------------------------------
# authority verification (grid/static pinned, init recorded)
# --------------------------------------------------------------------------
def verify_forecast_authorities(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Pin the mesh, record the init."""

    if set(paths) != {"grid", "static", "init"}:
        raise ValueError("forecast authority roles are exactly grid, static, init")
    files: dict[str, Any] = {}
    for role in MESH_AUTHORITY_ROLES:
        files[role] = proof._file_record(role, paths[role], proof.AUTHORITY_PINS[role])
    init = proof._plain_absolute(paths["init"], "init")
    if not init.is_file():
        raise FileNotFoundError(f"missing init: {init}")
    files["init"] = {
        "path": str(init),
        "bytes": init.stat().st_size,
        "sha256": proof.sha256_file(init),
        "pinned": False,
        "policy": "recorded, not pinned (derived-driver removal 1)",
    }
    return {
        "files": files,
        "mesh_roles_pinned": list(MESH_AUTHORITY_ROLES),
        "init_role_pinned": False,
        "sha256": proof.canonical_json_sha256(files),
    }


# --------------------------------------------------------------------------
# case-pin relaxation: measure from this init, then rebind the proof module's
# case constants so its own functions execute unchanged against this case.
# Every substitution is recorded in the receipt.
# --------------------------------------------------------------------------
def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def relax_init_carrier_pins(init_path: Path) -> dict[str, Any]:
    """Rebind the two init-carried mesh-geometry digests and F000 diagnostics."""

    from netCDF4 import Dataset

    record: dict[str, Any] = {}
    with Dataset(init_path, "r") as dataset:
        coefficients = proof._read_exact_variable(
            dataset,
            "coeffs_reconstruct",
            dtype=np.float32,
            dimensions=tuple(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN["dimensions"]),
        )
        normals = proof._read_exact_variable(
            dataset,
            "edgeNormalVectors",
            dtype=np.float32,
            dimensions=tuple(proof.INIT_EDGE_NORMAL_VECTORS_PIN["dimensions"]),
        )
        surface_diagnostics = {}
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items():
            value = proof._read_exact_variable(
                dataset,
                str(pin["source"]),
                dtype=np.float32,
                dimensions=("Time", "nCells"),
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{pin['source']} shape changed: {value.shape}")
            surface_diagnostics[target] = proof.array_sha256(
                np.ascontiguousarray(value[0], dtype=np.float32)
            )

    # --- reconstruction coefficients: only the init carrier digest moves.
    reconstruction = dict(proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN)
    removed_reconstruction_pin = str(reconstruction["init_carrier_raw_sha256"])
    measured_reconstruction = _raw_sha256(coefficients)
    reconstruction["init_carrier_raw_sha256"] = measured_reconstruction
    proof.INIT_RECONSTRUCTION_COEFFICIENTS_PIN = MappingProxyType(reconstruction)
    record["reconstruction_coefficients"] = {
        "removed_proof_pin_sha256": removed_reconstruction_pin,
        "init_carrier_raw_sha256": measured_reconstruction,
        "structural_checks_retained": [
            "dtype/shape",
            "finiteness",
            "topology equality with the prepared mesh",
            "no active 3-vector is the all-+0 static placeholder",
            "padding bitwise +0",
        ],
    }

    # --- edge normals: carrier digest, activity counts and norm envelope move.
    normals = np.ascontiguousarray(normals)
    nonzero = int(np.count_nonzero(normals))
    zeros = int(normals.size - nonzero)
    zero_rows = int(np.count_nonzero(np.all(normals == np.float32(0.0), axis=1)))
    normals64 = normals.astype(np.float64)
    norms = np.sqrt(np.sum(normals64 * normals64, axis=1, dtype=np.float64))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    # Replacement physical bound for the removed exact envelope pin.
    if (
        not math.isfinite(norm_min)
        or not math.isfinite(norm_max)
        or abs(norm_min - 1.0) > EDGE_NORMAL_UNIT_TOLERANCE
        or abs(norm_max - 1.0) > EDGE_NORMAL_UNIT_TOLERANCE
    ):
        raise RuntimeError(
            "initialized edge normals are not unit vectors: "
            f"norm envelope {(norm_min, norm_max)}"
        )
    edge = dict(proof.INIT_EDGE_NORMAL_VECTORS_PIN)
    removed_edge_pin = str(edge["init_carrier_raw_sha256"])
    edge["init_carrier_raw_sha256"] = _raw_sha256(normals)
    edge["nonzero_components"] = nonzero
    edge["exact_zero_components"] = zeros
    edge["zero_rows"] = zero_rows
    edge["float64_norm_min"] = norm_min
    edge["float64_norm_max"] = norm_max
    proof.INIT_EDGE_NORMAL_VECTORS_PIN = MappingProxyType(edge)
    record["edge_normal_vectors"] = {
        "removed_proof_pin_sha256": removed_edge_pin,
        "init_carrier_raw_sha256": edge["init_carrier_raw_sha256"],
        "nonzero_components": nonzero,
        "exact_positive_zero_components": zeros,
        "zero_rows": zero_rows,
        "float64_norm_min": norm_min,
        "float64_norm_max": norm_max,
        "replacement_gate": (
            f"every row norm within {EDGE_NORMAL_UNIT_TOLERANCE} of unity"
        ),
    }

    # --- F000 surface diagnostics.
    removed_surface_pins = {
        target: str(pin["sha256"])
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
    }
    pins = {
        target: {"source": pin["source"], "sha256": surface_diagnostics[target]}
        for target, pin in proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS.items()
    }
    proof.F000_INITIALIZED_SURFACE_DIAGNOSTIC_PINS = MappingProxyType(
        {name: MappingProxyType(value) for name, value in pins.items()}
    )
    record["f000_surface_diagnostics"] = {
        "measured": pins,
        "removed_proof_pins": removed_surface_pins,
    }
    return record


def relax_negative_qv_pin(state: Any) -> dict[str, Any]:
    """Measure this init's F000 negative-qv fingerprint and rebind the pin."""

    fingerprint = proof.negative_qv_fingerprint(np.asarray(state.scalars)[0])
    proof.NEGATIVE_QV_PIN = MappingProxyType(dict(fingerprint))
    return dict(fingerprint)


def relax_surface_classification(
    classification: Mapping[str, Any], glacier_path: str
) -> dict[str, Any]:
    """Rebind the classification/census expectations to this init's counts."""

    core = {
        name: classification[name]
        for name in proof.EXPECTED_SURFACE_CLASSIFICATION
    }
    proof.EXPECTED_SURFACE_CLASSIFICATION = MappingProxyType(dict(core))
    census = {
        "land": int(core["sflx_land_columns"]),
        "water": int(core["open_water_columns"]),
        "sea_ice": int(core["sea_ice_columns"]),
        "glacier": int(core["glacier_columns"]),
    }
    # The glacier kernel's provenance is reported by the seam only when the
    # seam ran it, which it does only when the domain HAS glacier columns.
    # MEASURED (2026-08-26, r4.75.11020): a placed swath over the Southern
    # Ocean has none, and demanding the provenance anyway refused its first
    # step with "NoahMP census/provenance changed" -- an all-ocean domain
    # failing for not naming the source file of a scheme it correctly never
    # called.  The count is still checked in both directions, so a domain
    # that HAS glaciers still has to say which kernel ran on them.
    if int(core["glacier_columns"]) > 0:
        census["glacier_path"] = glacier_path
    proof.EXPECTED_NOAHMP_CENSUS = MappingProxyType(dict(census))
    return {"surface_classification": dict(core), "noahmp_census": dict(census)}


def seed_p_top_expectation(
    *,
    pressure_base: Any,
    pressure_perturbation: Any,
    zgrid: Any,
    area_cell: Any,
) -> dict[str, Any]:
    """Re-derive the FP32 area-weighted p_top and seed the proof expectation.

    This is a faithful transcription of the reduction in
    ``proof.derive_area_weighted_p_top_v841``.  It only SEEDS the expectation;
    the proof function then recomputes the same quantity independently and
    still asserts equality, so a transcription error refuses rather than
    silently passing.
    """

    base = np.asarray(pressure_base)
    perturbation = np.asarray(pressure_perturbation)
    height = np.asarray(zgrid)
    area = np.asarray(area_cell)
    pressure = np.add(base, perturbation, dtype=np.float32)
    half = np.float32(0.5)
    one = np.float32(1.0)
    z0 = height[-1]
    z1 = np.multiply(
        half, np.add(height[-1], height[-2], dtype=np.float32), dtype=np.float32
    )
    z2 = np.multiply(
        half, np.add(height[-2], height[-3], dtype=np.float32), dtype=np.float32
    )
    w1 = np.divide(
        np.subtract(z0, z2, dtype=np.float32),
        np.subtract(z1, z2, dtype=np.float32),
        dtype=np.float32,
    )
    w2 = np.subtract(one, w1, dtype=np.float32)
    logarithm = np.add(
        np.multiply(w1, np.log(pressure[-1], dtype=np.float32), dtype=np.float32),
        np.multiply(w2, np.log(pressure[-2], dtype=np.float32), dtype=np.float32),
        dtype=np.float32,
    )
    top = np.ascontiguousarray(np.exp(logarithm, dtype=np.float32))
    if not np.all(np.isfinite(top)) or np.any(top <= 0):
        raise FloatingPointError("derived F000 top pressure is invalid for this init")
    area64 = area.astype(np.float64, copy=False)
    weighted_mean64 = float(
        np.sum(top.astype(np.float64) * area64, dtype=np.float64)
        / np.sum(area64, dtype=np.float64)
    )
    scalar = np.float32(weighted_mean64)
    observed = (float(np.min(top)), float(np.median(top)), float(np.max(top)))
    proof.EXPECTED_ARWEN_P_TOP_PA_F32 = scalar
    proof.EXPECTED_TOP_PRESSURE_RANGE_PA = observed
    return {
        "area_weighted_mean_f32_pa": float(scalar),
        "per_column_minimum_pa": observed[0],
        "per_column_median_pa": observed[1],
        "per_column_maximum_pa": observed[2],
        "seeding_policy": (
            "re-derived here, independently recomputed and asserted by the "
            "proof reduction"
        ),
    }


def install_capture_labels(labels: Mapping[int, str]) -> None:
    """Rebind the proof's fixed F000/F030/F001 label table to this schedule."""

    proof.SNAPSHOT_LABELS = dict(labels)
    proof.SNAPSHOT_STEPS = tuple(sorted(labels))


# --------------------------------------------------------------------------
# constructor values (transcribed from proof.build_arwen_constructor_values,
# case literals replaced by measurement)
# --------------------------------------------------------------------------
def build_forecast_config(
    *,
    dt_seconds: float,
    convection_scheme: str = "cu_grell_freitas",
    surface_pbl_seconds: float | None = None,
    horiz_mixing: str = "2d_smagorinsky",
    local_timestep: bool = False,
    local_timestep_declared_off: bool = False,
    local_timestep_rates: tuple[int, ...] = (1, 3),
    local_timestep_buffer_rings: int = 1,
    apply_lbcs: bool = False,
) -> Any:
    """Build the run's configuration AT THE BOUND MESH'S TIMESTEP.

    THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, the proving RTX 5090):
    this function did not exist and the configuration was constructed from
    its dataclass defaults, so ``config_dt`` was 120.0 no matter what mesh
    was bound.  ``bind_mesh`` rebinds ``DT_SECONDS`` in this module and in
    the proof module, and the sealed Arwen constructor read that rebound
    value -- but the DYCORE takes its outer step from ``config.config_dt``,
    which nothing rebound.  A mesh row declaring 100 s therefore bound
    clean, allocated 18,820 MiB, spent 285 s and died inside composite
    step 0 with ``post-RK candidate time must equal the exact step
    endpoint: 120.0 != 100.0``.

    There is now ONE timestep in this path: the bound mesh's.  It reaches
    the config here, and ``build_forecast_constructor_values`` derives the
    seam's clocks from the config rather than from the module constant, so
    the two cannot disagree by construction.  An unanchored timestep is
    refused by ``config.validate()`` below, on the host, before a mesh file
    is opened -- which is where a 285-second refusal belongs.

    The two cadences welded to the timestep travel with it.  ``cudt`` has
    no choice: WRF pins ``cudt = 0`` for Grell-Freitas, so the sealed
    constructor requires ``cumulus_seconds == dt``.  ``bldt`` follows dt by
    DEFAULT because that is the proven configuration's own semantics -- the
    native x4 reference ran ``bldt = dt``, i.e. surface/PBL every step --
    and ``surface_pbl_seconds`` is how an A/B arm holds it there while dt
    moves.  ``None`` is the weld and changes no run; an explicit value is an
    instrument and records itself as one.  See :mod:`hexcore.pbl_cadence`
    for the measurement that needed it and the breakage its registry key
    prevents.
    """

    from hexcore.config_v841 import (
        V841MpasColumnPhysicsGwdoConfig,
        V841MpasColumnPhysicsSmagorinskyGwdoConfig,
    )
    from hexcore.config_lts import (
        V841LocalTimestepGwdoConfig,
        V841LocalTimestepSmagorinskyGwdoConfig,
    )

    from hexcore import convection_admission, pbl_cadence

    dt = float(dt_seconds)
    scheme = str(convection_scheme)
    if scheme not in convection_admission.ADMITTED_CONVECTION_SCHEMES:
        raise ValueError(
            f"convection_scheme must be one of "
            f"{list(convection_admission.ADMITTED_CONVECTION_SCHEMES)}, "
            f"got {convection_scheme!r}"
        )
    bldt = (
        dt
        if surface_pbl_seconds is None
        else pbl_cadence.resolve_seconds(
            dt_seconds=dt, requested=surface_pbl_seconds
        )
    )
    clocks: dict[str, Any] = {
        "config_dt": dt,
        "config_bldt_seconds": bldt,
        # WRF pins cudt=0 for Grell-Freitas, so with a scheme selected the
        # cumulus cadence IS dt.  With no scheme selected there is no cadence
        # at all -- see hexcore.convection_admission.
        "config_cudt_seconds": (
            None if scheme == convection_admission.SCHEME_OFF else dt
        ),
        "config_convection_scheme": scheme,
        # The lateral-boundary switch is NOT a user knob here: it is the
        # grid's own property, read off the bdyMask triple by the caller.
        # mpas_atm_bdy_checks refuses either mismatch by name -- boundary
        # cells with the switch off, or the switch on with none.
        "config_apply_lbcs": bool(apply_lbcs),
    }
    lts_block: dict[str, Any] = {
        "config_local_timestep": bool(local_timestep),
        "config_local_timestep_rates": tuple(local_timestep_rates),
        "config_local_timestep_buffer_rings": int(local_timestep_buffer_rings),
    }

    if horiz_mixing == "2d_smagorinsky":
        # Default: native's Registry configuration (deformation-based 2-D
        # Smagorinsky + del4), the regime the natA/natB 24-h references
        # integrate.  Numbers under this lane are a NEW SUB-SERIES: they are
        # not bit-comparable to any mixing-off arm.
        if local_timestep or local_timestep_declared_off:
            # Same released lane, plus the opt-in local-timestep block.  With
            # the switch off this subtype validates and executes identically
            # to its parent, which is what the default-off gate measures.
            return V841LocalTimestepSmagorinskyGwdoConfig(**clocks, **lts_block)
        return V841MpasColumnPhysicsSmagorinskyGwdoConfig(**clocks)
    if horiz_mixing == "off":
        # Explicit control arm: the pre-mixing configuration (the regime
        # native itself dies in on convective cases -- the 2026-08-17
        # reference-node control died at step 466 on case B).
        if local_timestep or local_timestep_declared_off:
            return V841LocalTimestepGwdoConfig(**clocks, **lts_block)
        return V841MpasColumnPhysicsGwdoConfig(**clocks)
    raise ValueError(
        f"horiz_mixing must be '2d_smagorinsky' or 'off', got {horiz_mixing!r}"
    )


def build_forecast_constructor_values(
    *,
    init_path: Path,
    mesh: Any,
    vertical: Any,
    reference: Any,
    saved_diagnostics: Any,
    start_time_text: str | None,
    config: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    from netCDF4 import Dataset

    read = proof._read_exact_variable
    with Dataset(init_path, "r") as dataset:
        unexpected_species = sorted(
            name for name in COLD_ZERO_SCALAR_NAMES if name in dataset.variables
        )
        if unexpected_species:
            raise RuntimeError(
                "cold-start assumption changed; unexpected variables "
                f"{unexpected_species}"
            )
        landmask_i = read(dataset, "landmask", dtype=np.int32, dimensions=("nCells",))
        ivgtyp = read(dataset, "ivgtyp", dtype=np.int32, dimensions=("nCells",))
        isltyp = read(dataset, "isltyp", dtype=np.int32, dimensions=("nCells",))
        xland_source = read(
            dataset, "xland", dtype=np.float32, dimensions=("Time", "nCells")
        )
        if xland_source.shape != (1, N_CELLS):
            raise ValueError(f"xland shape changed: {xland_source.shape}")
        xland = np.ascontiguousarray(xland_source[0], dtype=np.float32)
        zgrid = read(
            dataset,
            "zgrid",
            dtype=np.float32,
            dimensions=("nCells", "nVertLevelsP1"),
        )
        surface: dict[str, np.ndarray] = {}
        for source, target in (
            ("vegfra", "vegfra"),
            ("skintemp", "tsk"),
            ("tmn", "tmn"),
            ("xice", "xice"),
            ("snow", "snow"),
            ("snowh", "snow_depth"),
        ):
            value = read(
                dataset, source, dtype=np.float32, dimensions=("Time", "nCells")
            )
            if value.shape != (1, N_CELLS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            surface[target] = np.ascontiguousarray(value[0], dtype=np.float32)
        soil: dict[str, np.ndarray] = {}
        for source, target in (
            ("tslb", "soil_temperature"),
            ("smois", "soil_moisture"),
        ):
            value = read(
                dataset,
                source,
                dtype=np.float32,
                dimensions=("Time", "nCells", "nSoilLevels"),
            )
            if value.shape != (1, N_CELLS, N_SOIL_LEVELS):
                raise ValueError(f"{source} shape changed: {value.shape}")
            soil[target] = np.ascontiguousarray(value[0].T, dtype=np.float32)
        nominal_min_dc = read(dataset, "nominalMinDc", dtype=np.float32, dimensions=())
        start_text = str(getattr(dataset, "config_start_time", ""))

    if not start_text:
        raise RuntimeError("init carries no config_start_time")
    try:
        start_datetime = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
    except ValueError as error:
        raise RuntimeError(f"init config_start_time is unparseable: {start_text!r}") from error
    if start_time_text is not None and start_time_text != start_text:
        raise RuntimeError(
            f"--start-time {start_time_text!r} disagrees with the init's "
            f"config_start_time {start_text!r}"
        )

    # Lake columns: the frozen physics stack carries NO lake model, and the
    # arwen vegetation parameter tables end at category 20, so a lake column
    # (MODIS category 21 -- present in with-lakes land use and in generated-mesh
    # statics, absent from the x4 proof's own landuse, whose maximum is 19)
    # routed to the land path indexes off the SLA table inside the Noah-MP cold
    # start.  WRF applies its rule at this same boundary when sf_lake_physics
    # is off: a lake column IS open water.  Fold before classification so every
    # downstream consumer -- the partition, the census, the seam -- sees one
    # consistent water column, and put the count in the receipt.  On the native
    # x4 case the mask is empty and every array passes through untouched.
    lake_mask = ivgtyp == np.int32(21)
    lake_fold = {
        "lake_category": 21,
        "lake_columns": int(np.count_nonzero(lake_mask)),
        "rule": (
            "no lake model in the frozen v8.4.1 physics: lake columns become "
            "open water before classification (ivgtyp 21->17, isltyp ->14, "
            "landmask ->0, xland ->2.0), the same conversion WRF applies when "
            "sf_lake_physics is off"
        ),
        "pre_fold_landmask_sha256": proof.array_sha256(landmask_i),
        "pre_fold_ivgtyp_sha256": proof.array_sha256(ivgtyp),
    }
    if lake_fold["lake_columns"]:
        ivgtyp = np.ascontiguousarray(np.where(lake_mask, np.int32(17), ivgtyp))
        isltyp = np.ascontiguousarray(np.where(lake_mask, np.int32(14), isltyp))
        landmask_i = np.ascontiguousarray(
            np.where(lake_mask, np.int32(0), landmask_i)
        )
        xland = np.ascontiguousarray(
            np.where(lake_mask, np.float32(2.0), xland)
        )

    source_xland_sha256 = proof.array_sha256(xland_source)
    flat_xland_sha256 = proof.array_sha256(xland)
    xland_unique, xland_counts = np.unique(xland, return_counts=True)
    # MEASURED, not pinned: the proof asserted (1.0, 2.0) with the case counts.
    # The value SET is still a hard requirement -- MPAS xland is 1 (land/ice)
    # or 2 (water) and nothing else -- but requiring BOTH values requires the
    # domain to contain a coastline, and a limited-area domain need not.
    #
    # THE BREAKAGE THIS PREVENTS: xland selects which surface scheme a column
    # runs.  A third value would send columns to neither the land-surface
    # model nor the open-water branch and the surface fluxes would be whatever
    # the uninitialised path left behind.
    #
    # THE BREAKAGE IT MUST NOT INVENT, MEASURED (2026-08-26, r4.75.11020): a
    # placed swath over the Southern Ocean is all water, so its xland set is
    # (2.0,).  Demanding a land column refuses an all-ocean domain for having
    # no coastline in it, which is a property of the case, not a defect.
    if not set(float(value) for value in xland_unique) <= {1.0, 2.0}:
        raise RuntimeError(
            f"init xland carries values outside (1.0, 2.0): {xland_unique}; "
            "MPAS xland is 1 for land or ice and 2 for water, and a column "
            "carrying anything else reaches neither surface branch"
        )
    if xland_unique.size == 0:
        raise RuntimeError("init xland is empty")
    xice = surface["xice"]
    if float(np.min(xice)) < 0.0 or float(np.max(xice)) > 1.0:
        raise RuntimeError("init xice lies outside [0,1]")
    sea_ice_mask = np.ascontiguousarray(xice >= ARWEN_XICE_THRESHOLD)
    open_water_mask = np.ascontiguousarray((xland >= np.float32(1.5)) & ~sea_ice_mask)
    land_mask = np.ascontiguousarray(~(sea_ice_mask | open_water_mask))
    glacier_mask = np.ascontiguousarray(land_mask & (ivgtyp == np.int32(15)))
    sflx_land_mask = np.ascontiguousarray(land_mask & ~glacier_mask)
    surface_classification = {
        "xland_source": "native",
        "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
        "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "sea_ice_columns": int(np.count_nonzero(sea_ice_mask)),
        "open_water_columns": int(np.count_nonzero(open_water_mask)),
        "sflx_land_columns": int(np.count_nonzero(sflx_land_mask)),
        "glacier_columns": int(np.count_nonzero(glacier_mask)),
    }
    if (
        surface_classification["sflx_land_columns"]
        + surface_classification["open_water_columns"]
        + surface_classification["sea_ice_columns"]
        + surface_classification["glacier_columns"]
        != N_CELLS
    ):
        raise RuntimeError("surface classification does not partition the mesh")
    glacier_indices = np.ascontiguousarray(np.flatnonzero(glacier_mask))
    sea_ice_indices = np.ascontiguousarray(np.flatnonzero(sea_ice_mask))
    threshold_delta_indices = np.ascontiguousarray(
        np.flatnonzero((xice >= ARWEN_XICE_THRESHOLD) & (xice < np.float32(0.5)))
    )
    # KEPT: the category signature.  Every sea-ice and glacier column must
    # carry the native MPAS ice vegetation category and a consistent landmask.
    if not (
        np.all(ivgtyp[sea_ice_mask] == np.int32(15))
        and np.all(landmask_i[sea_ice_mask] == np.int32(0))
        and np.all(xland[sea_ice_mask] == np.float32(1.0))
        and np.all(ivgtyp[glacier_mask] == np.int32(15))
        and np.all(landmask_i[glacier_mask] == np.int32(1))
        and np.all(xland[glacier_mask] == np.float32(1.0))
        and np.all(xice[glacier_mask] == np.float32(0.0))
    ):
        raise RuntimeError("native xland/ice category signature is inconsistent")
    surface_classification_receipt = {
        **surface_classification,
        "lake_fold": lake_fold,
        "source_field": "init xland[0,:]",
        "source_shape": list(xland_source.shape),
        "source_array_sha256": source_xland_sha256,
        "constructor_shape": list(xland.shape),
        "constructor_array_sha256": flat_xland_sha256,
        "first_glacier_column": (
            int(glacier_indices[0]) if glacier_indices.size else None
        ),
        "glacier_index_sha256": proof.array_sha256(glacier_indices),
        "sea_ice_index_sha256": proof.array_sha256(sea_ice_indices),
        "threshold_0p02_to_0p5_delta_columns": int(threshold_delta_indices.size),
        "threshold_delta_index_sha256": proof.array_sha256(threshold_delta_indices),
        "native_xland_consumed_verbatim": True,
        "counts_measured_not_pinned": True,
    }

    if np.float32(nominal_min_dc).view(np.uint32) != NOMINAL_DX_M.view(np.uint32):
        raise RuntimeError("init nominalMinDc is not exact FP32 25000 m")

    # KEPT: exactness of the int32 -> FP32 landmask cast.  Digests recorded.
    source_landmask_sha256 = proof.array_sha256(landmask_i)
    source_unique_values = tuple(int(value) for value in np.unique(landmask_i))
    if landmask_i.shape != (N_CELLS,) or source_unique_values not in ((0, 1), (0,), (1,)):
        raise RuntimeError(
            f"init landmask identity is not int32 {{0,1}}[nCells]: "
            f"{landmask_i.shape} {source_unique_values}"
        )
    landmask = np.ascontiguousarray(landmask_i, dtype=np.float32)
    target_landmask_sha256 = proof.array_sha256(landmask)
    target_uint32_values = tuple(
        int(value) for value in np.unique(landmask.view(np.uint32))
    )
    if landmask.dtype != np.dtype(np.float32) or not np.array_equal(
        landmask.astype(np.int32), landmask_i
    ):
        raise RuntimeError("landmask int32 -> FP32 constructor cast is not exact")
    landmask_receipt = {
        "source_field": "init landmask",
        "source_dimensions": ["nCells"],
        "source_shape": list(landmask_i.shape),
        "source_dtype": landmask_i.dtype.str,
        "source_array_sha256": source_landmask_sha256,
        "source_unique_values": list(source_unique_values),
        "target_field": "SealedArwenConstructorV841.landmask",
        "target_shape": list(landmask.shape),
        "target_dtype": landmask.dtype.str,
        "target_array_sha256": target_landmask_sha256,
        "target_uint32_values": list(target_uint32_values),
        "value_preserving_exact_fp32_cast": True,
        "digests_measured_not_pinned": True,
    }

    if zgrid.shape != (N_CELLS, N_INTERFACES):
        raise ValueError(f"zgrid shape changed: {zgrid.shape}")

    lat = np.asarray(proof._mesh_value(mesh, "latCell"), dtype=np.float64)
    lon = np.asarray(proof._mesh_value(mesh, "lonCell"), dtype=np.float64)
    if lat.shape != (N_CELLS,) or lon.shape != (N_CELLS,):
        raise ValueError("reconciled mesh latitude/longitude shape changed")
    latitude_deg = np.ascontiguousarray(lat * (180.0 / np.pi), dtype=np.float32)
    longitude_deg = np.ascontiguousarray(lon * (180.0 / np.pi), dtype=np.float32)
    terrain = np.ascontiguousarray(zgrid[:, 0], dtype=np.float32)
    nominal_z = np.ascontiguousarray(np.asarray(vertical.zw))
    if nominal_z.shape != (N_INTERFACES,) or nominal_z.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise TypeError("loaded vertical.zw is not the exact 56-interface host vector")
    if not np.all(np.isfinite(nominal_z)) or np.any(np.diff(nominal_z) <= 0.0):
        raise ValueError("loaded vertical.zw is not finite and strictly increasing")

    # GF's per-cell length scale, native's own construction and the same one
    # this port already feeds GWDO: len_disp/meshDensity**0.25, with a
    # non-positive config_len_disp resolved to the mesh nominalMinDc.
    from hexcore.cuda_gwdo_v841 import native_cell_dx_m

    dx_column_m = native_cell_dx_m(
        proof._mesh_value(mesh, "meshDensity"), float(NOMINAL_DX_M)
    )
    if dx_column_m.shape != (N_CELLS,):
        raise ValueError(
            f"per-cell GF dx must have shape {(N_CELLS,)}, got {dx_column_m.shape}"
        )

    p_top_seed = seed_p_top_expectation(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(proof._mesh_value(mesh, "areaCell")),
    )
    p_top_pa, p_top_receipt = proof.derive_area_weighted_p_top_v841(
        pressure_base=reference.pressure_base,
        pressure_perturbation=saved_diagnostics.pressure_perturbation,
        zgrid=vertical.zgrid,
        area_cell=np.ascontiguousarray(proof._mesh_value(mesh, "areaCell")),
    )

    # The seam's four clocks come from the CONFIG, never from the module
    # constant.  That is the whole of the 2026-08-26 rebinding fix: one
    # timestep travels from the bound mesh row through config_dt to here, so
    # the dycore's outer step and the frozen seam's step are the same number
    # by construction rather than by two rebinds agreeing.
    from hexcore import convection_admission

    cumulus_scheme = convection_admission.constructor_scheme(
        config.config_convection_scheme
    )
    gf_ishallow = convection_admission.gf_ishallow(config.config_convection_scheme)

    values: dict[str, Any] = {
        "n_levels": N_LEVELS,
        "n_columns": N_CELLS,
        "dt": float(config.config_dt),
        "radiation_seconds": float(config.config_radt_seconds),
        "surface_pbl_seconds": float(config.config_bldt_seconds),
        # The cumulus selection comes from the CONFIG, never from a literal
        # here.  it was ruled on 2026-08-26 that convection is off below 3 km,
        # so "gf" written in this mapping would have silently reinstated the
        # closure the ruling switches off -- the config would say off and the
        # sealed constructor would be handed on.  See
        # hexcore.convection_admission.
        "cumulus_seconds": (
            None
            if config.config_cudt_seconds is None
            else float(config.config_cudt_seconds)
        ),
        "cumulus_scheme": cumulus_scheme,
        "start_time": start_datetime,
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_deg,
        "terrain_height_m": terrain,
        "z_interface_nominal_m": nominal_z,
        "p_top_pa": p_top_pa,
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_m": dx_column_m,
        # Native MPAS v8.4.1 hardwires GF's shallow scheme on
        # (mpas_atmphys_vars.F:340).  Derived from the selection rather than
        # written as 1: the sealed constructor refuses gf_ishallow=1 with no
        # GF selected, so a literal here would refuse every convection-off
        # run at host preparation.
        "gf_ishallow": gf_ishallow,
        "landmask": landmask,
        "xland": xland,
        "xice_threshold": float(ARWEN_XICE_THRESHOLD),
        "ivgtyp": np.ascontiguousarray(ivgtyp, dtype=np.int32),
        "isltyp": np.ascontiguousarray(isltyp, dtype=np.int32),
        **surface,
        **soil,
        "wsm6_hail_opt": 0,
    }
    arrays = {
        name: value for name, value in values.items() if isinstance(value, np.ndarray)
    }
    receipt = {
        "source": "exact initialized x4.163842 fields from the supplied init",
        "init_path": str(Path(init_path).absolute()),
        "config_start_time": start_text,
        "mapping": {
            "latitude_deg": "reconciled mesh latCell radians -> FP32 degrees",
            "longitude_deg": "reconciled mesh lonCell radians -> FP32 degrees",
            "terrain_height_m": "init zgrid[:,0]",
            "landmask": "init int32 {0,1} -> exact FP32 sealed-constructor cast",
            "xland": "init native xland[0,:] consumed verbatim",
            "xice_threshold": "explicit MPAS config_frac_seaice threshold 0.02",
            "z_interface_nominal_m": "loaded native vertical.zw",
            "tsk": "init skintemp[0,:]",
            "snow_depth": "init snowh[0,:] (m)",
            "soil_temperature": "init tslb[0,:,:].T",
            "soil_moisture": "init smois[0,:,:].T",
        },
        "p_top_pa": p_top_pa,
        "p_top_seed": p_top_seed,
        "landmask_exact_cast": landmask_receipt,
        "surface_classification": surface_classification_receipt,
        "p_top_derivation": p_top_receipt,
        "p_top_policy": "exact areaCell-weighted F000 native pres2_p top",
        "dx_m": float(NOMINAL_DX_M),
        "dx_column_policy": "native len_disp/meshDensity**0.25 per cell",
        "dx_column_min_m": float(dx_column_m.min()),
        "dx_column_max_m": float(dx_column_m.max()),
        "gf_ishallow": gf_ishallow,
        "cumulus_scheme": cumulus_scheme,
        "config_convection_scheme": config.config_convection_scheme,
        "defaults_used": False,
        "arrays": {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": proof.array_sha256(value),
            }
            for name, value in sorted(arrays.items())
        },
    }
    static_for_gwdo = {
        "meshDensity": np.asarray(proof._mesh_value(mesh, "meshDensity")),
        "nominalMinDc": np.asarray(NOMINAL_DX_M),
    }
    for name in ("var2d", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4"):
        with Dataset(init_path, "r") as dataset:
            static_for_gwdo[name] = read(
                dataset, name, dtype=np.float32, dimensions=("nCells",)
            )
    return values, receipt, static_for_gwdo, surface_classification


# --------------------------------------------------------------------------
# host preparation (transcribed from proof._prepare_host_execution, with the
# measurement/rebinding interleaved)
# --------------------------------------------------------------------------
def prepare_forecast_host(
    paths: Mapping[str, Path],
    authority_receipt: Mapping[str, Any],
    *,
    start_time_text: str | None,
    horiz_mixing: str = "2d_smagorinsky",
    convection: str = "auto",
    pbl_cadence: str = "auto",
    local_timestep: bool = False,
    local_timestep_declared_off: bool = False,
    local_timestep_rates: tuple[int, ...] = (1, 3),
    local_timestep_buffer_rings: int = 1,
    lbc_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    from hexcore.cuda_arwen_physics_v841 import SealedArwenConstructorV841
    from hexcore.cuda_dualrun import PreparedCudaInputs
    from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from hexcore.dynamics_v841 import load_v841_reference_wind_profiles
    from hexcore.mesh import load_precision_preserving_mesh_pair

    relaxation: dict[str, Any] = {"init_carriers": relax_init_carrier_pins(paths["init"])}

    # DT_SECONDS is the module constant bind_mesh rebinds to the bound row's
    # declared timestep.  Reading it HERE is what closes the 2026-08-26
    # rebinding defect: before this, the config took its dataclass default
    # and the two clocks diverged silently until composite step 0.
    # The cumulus selection travels the same road as the timestep: decided
    # once at the bind from the mesh's own finest spacing, read here.  If a
    # bind happened and its request disagrees with this one, that is TWO
    # sources of the same decision -- the exact shape of the 2026-08-26 clock
    # defect -- and it is refused on the host rather than discovered in the
    # receipt of a finished run.
    from hexcore import convection_admission as _convection

    decision = dict(CONVECTION_DECISION) if CONVECTION_DECISION else None
    if decision is not None and decision.get("requested") != convection:
        raise ValueError(
            f"the bound mesh decided its convection selection under "
            f"--convection {decision.get('requested')!r} and this run was "
            f"invoked with --convection {convection!r}.  One decision, one "
            f"source: the bind's request and the driver's must be the same "
            f"string, or the receipt would name a selection the run did not "
            f"make"
        )
    if decision is None:
        decision = _convection.convection_decision(
            nominal_dx_m=float(NOMINAL_DX_M), requested=convection
        )
    convection_scheme = decision["scheme"]
    print(
        f"[convection] {decision['scheme']} ({decision['source']}): "
        f"{decision['note']}",
        flush=True,
    )

    # The surface/PBL cadence travels the same road, for the same reason and
    # with the same refusal: one decision, one source.  ``auto`` is the weld
    # (config_bldt_seconds = config_dt), which is the proven configuration
    # and the default; an explicit cadence is an A/B arm.
    from hexcore import pbl_cadence as _pbl

    pbl_decision = dict(PBL_CADENCE_DECISION) if PBL_CADENCE_DECISION else None
    if pbl_decision is not None and pbl_decision.get("requested") != pbl_cadence:
        raise ValueError(
            f"the bound mesh decided its surface/PBL cadence under "
            f"--pbl-cadence {pbl_decision.get('requested')!r} and this run "
            f"was invoked with --pbl-cadence {pbl_cadence!r}.  One decision, "
            f"one source: the bind's request and the driver's must be the "
            f"same string, or the receipt would name a cadence the run did "
            f"not use"
        )
    if pbl_decision is None:
        pbl_decision = _pbl.pbl_cadence_decision(
            dt_seconds=float(DT_SECONDS), requested=pbl_cadence
        )
    print(
        f"[pbl-cadence] {pbl_decision['label']} ({pbl_decision['source']}): "
        f"{pbl_decision['note']}",
        flush=True,
    )

    config = build_forecast_config(
        dt_seconds=float(DT_SECONDS),
        convection_scheme=convection_scheme,
        surface_pbl_seconds=pbl_decision["surface_pbl_seconds"],
        horiz_mixing=horiz_mixing,
        local_timestep=local_timestep,
        local_timestep_declared_off=local_timestep_declared_off,
        local_timestep_rates=local_timestep_rates,
        local_timestep_buffer_rings=local_timestep_buffer_rings,
        apply_lbcs=bool(lbc_paths),
    )
    config.validate()
    mesh, output_mesh, mesh_evidence = load_precision_preserving_mesh_pair(
        paths["grid"], paths["static"]
    )
    del output_mesh
    # A limited-area grid declares itself: the cull writes the
    # bdyMaskCell/Edge/Vertex triple and nothing else does.  Everything below
    # is the SAME preparation the global lane runs -- the sentinel flag says
    # "this mesh's outermost ring has one-cell edges by construction", it does
    # not select a different code path.
    is_regional = bool(getattr(mesh, "is_regional", False))
    if is_regional and not lbc_paths:
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            True,
            "this grid carries a bdyMask triple, so its outermost ring is a "
            "lateral boundary that something has to drive; integrating it with "
            "no boundary series lets the interior run against whatever the "
            "initial state left on the ring, and the domain empties from the "
            "edge inward within a few hours",
            "a --lbc-dir of files rw_mpas_lbc built from the parent forecast "
            "this mesh was cut out of",
        )
    if not is_regional and lbc_paths:
        raise ConfigurationRefusal(
            "config_apply_lbcs",
            True,
            "this grid carries no bdyMask triple, so it has no boundary zone "
            "for a lateral-boundary series to drive; the files would be read "
            "and never applied, and the receipt would name a forcing the run "
            "did not use",
            "a limited-area grid cut with rw_mpas_mesh --cull-parent, or no "
            "--lbc-dir",
        )
    reconstruction_overlay = proof.overlay_exact_init_reconstruction_coefficients(
        mesh, paths["init"]
    )
    edge_normal_overlay = proof.overlay_exact_init_edge_normal_vectors(
        mesh,
        grid_path=paths["grid"],
        static_path=paths["static"],
        init_path=paths["init"],
    )
    defc = proof.attach_inactive_zero_deformation(mesh)
    native = load_mpas_vertical_grid(
        paths["init"],
        mesh,
        config_coef_3rd_order=config.config_coef_3rd_order,
        allow_regional_sentinels=is_regional,
    )
    state, reference, saved = load_mpas_initial_state(
        paths["init"],
        mesh,
        native.vertical_grid,
        scalar_names=SOURCE_SCALAR_NAMES,
        terrain_metrics=native.terrain_metrics,
        return_saved_diagnostics=True,
        allow_regional_sentinels=is_regional,
    )
    relaxation["negative_qv"] = relax_negative_qv_pin(state)
    scalar_receipt = proof.augment_exact_wsm6_scalars(state)
    state.validate(n_cells=N_CELLS, n_edges=N_EDGES, n_vert_levels=N_LEVELS)
    saved.validate((N_LEVELS, N_CELLS), np.dtype(np.float32), N_EDGES)
    profiles = load_v841_reference_wind_profiles(paths["init"], n_vert_levels=N_LEVELS)
    prepared = PreparedCudaInputs.validated(
        config=config,
        profile=proof.PROFILE,
        target=CLAIM,
        preparation_method=(
            "precision-preserving grid/static overlay plus exact initialized "
            "reconstruction coefficients and edge-normal vectors; qv/qc/qr "
            "plus exact +0 qi/qs/qg"
        ),
        mesh=mesh,
        state=state,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        terrain_metrics=native.terrain_metrics,
        input_bytes=dict(authority_receipt["files"]),
        reference_wind_profiles=profiles,
        allow_regional_sentinels=is_regional,
    )
    f000_surface_diagnostics = proof.load_f000_initialized_surface_diagnostics(
        paths["init"]
    )
    (
        constructor_values,
        constructor_receipt,
        gwdo_host,
        classification,
    ) = build_forecast_constructor_values(
        init_path=paths["init"],
        mesh=mesh,
        vertical=native.vertical_grid,
        reference=reference,
        saved_diagnostics=saved,
        start_time_text=start_time_text,
        config=config,
    )
    if is_regional:
        # The ArWen statics move onto the PADDED extent here, BEFORE the
        # surface census is rebound, because the census counts columns and
        # the run seals nCells+1 of them.  Doing it after left the seam
        # reporting 11,021 classified columns against an expectation of
        # 11,020 and the first step receipt refused by exactly one column.
        from hexcore.cuda_regional_forecast_v841 import pad_regional_physics_host
        from hexcore.regional_v841 import derive_regional_masks, regional_bdy_checks

        regional_masks = derive_regional_masks(mesh, np.dtype(np.float32))
        # mpas_atm_bdy_checks, on the host, before a byte moves: a mesh with
        # boundary cells and no LBCs, or LBCs and no boundary cells, is
        # refused by name here rather than discovered in a finished receipt.
        regional_bdy_checks(
            regional_masks, config_apply_lbcs=True, lbc_input_interval_valid=True
        )
        constructor_values, gwdo_host, pad_receipt = pad_regional_physics_host(
            constructor_values, gwdo_host, n_cells_solve=int(N_CELLS)
        )
        constructor_receipt["regional_physics_pad"] = pad_receipt
        classification = dict(
            SealedArwenConstructorV841.from_mapping(
                constructor_values
            ).expected_surface_classification()
        )
    relaxation["surface"] = relax_surface_classification(
        classification, proof.ARWEN_GLACIER_CUDA_PROVENANCE
    )
    relaxation["p_top"] = constructor_receipt["p_top_seed"]
    sealed_constructor_audit = SealedArwenConstructorV841.from_mapping(
        constructor_values
    )
    # Belt and braces on the wiring above: the seam's clocks are DERIVED from
    # the config, so this can only fail if somebody reintroduces a second
    # source of the timestep.  It costs nothing and it is the difference
    # between a host refusal and 18,820 MiB plus 285 s on a card.
    coherence = dt_admission.require_step_clock_coherence(
        config_dt=config.config_dt,
        constructor_dt=sealed_constructor_audit.dt,
        config_radt_seconds=config.config_radt_seconds,
        constructor_radiation_seconds=constructor_values["radiation_seconds"],
        config_bldt_seconds=config.config_bldt_seconds,
        constructor_surface_pbl_seconds=constructor_values["surface_pbl_seconds"],
        config_cudt_seconds=config.config_cudt_seconds,
        constructor_cumulus_seconds=constructor_values["cumulus_seconds"],
    )
    constructor_receipt["step_clock_coherence"] = coherence
    from hexcore import convection_admission as _convection

    constructor_receipt["dt_admission"] = dt_admission.require_dt_anchor(
        config.config_dt,
        radiation_seconds=config.config_radt_seconds,
        surface_pbl_seconds=config.config_bldt_seconds,
        cumulus_seconds=config.config_cudt_seconds,
        # The anchor certifies a CONFIGURATION at a timestep.  Omitting this
        # would have admitted a convection-off run against a Grell-Freitas
        # row -- an anchor whose forecasts measured a forcing this run does
        # not apply.
        cumulus_scheme=_convection.constructor_scheme(
            config.config_convection_scheme
        ),
    ).as_dict()
    # config_bldt_seconds reaches require_dt_anchor above as the LOOKUP key
    # as well as the comparison, so a run holding the surface/PBL cadence
    # reads the row that measured that cadence and never the welded one.
    constructor_receipt["sealed_host_contract_audit"] = {
        "authority": "SealedArwenConstructorV841.from_mapping",
        "accepted": True,
        "all_required_keys_dtypes_shapes_validated": True,
        "array_fields": sorted(constructor_receipt["arrays"]),
        **dict(sealed_constructor_audit.receipt()),
    }
    constructor_receipt["convection_admission"] = decision
    constructor_receipt["pbl_cadence"] = pbl_decision
    regional: dict[str, Any] | None = None
    if is_regional:
        # The species the boundary files actually carry, intersected with
        # the model's own scalar order.  LBC_REQUIRED_VARIABLES makes
        # lbc_qv/lbc_qc/lbc_qr mandatory in every file rw_mpas_lbc writes,
        # so this is the model's leading three for a WSM6 run -- but it is
        # DERIVED from the stream rather than assumed, so a stream that
        # gains a species drives it without a code change.
        from hexcore.lbc import LBC_REQUIRED_VARIABLES

        driven = tuple(
            f"lbc_{name}"
            for name in SCALAR_NAMES
            if f"lbc_{name}" in LBC_REQUIRED_VARIABLES
        )
        regional = {
            "driven_scalars": driven,
            "lbc_paths": [str(path) for path in lbc_paths or ()],
            "start_time": datetime.strptime(
                constructor_receipt["config_start_time"], "%Y-%m-%d_%H:%M:%S"
            ),
            "n_cells_solve": int(N_CELLS),
            "boundary_zone_width": int(REGIONAL_BOUNDARY_ZONE_WIDTH),
            "free_interior_cells": int(
                np.count_nonzero(regional_masks.bdy_mask_cell == 0)
            ),
            "specified_zone_cells": int(regional_masks.spec_cells.size),
            "relaxation_zone_cells": int(regional_masks.relax_cells.size),
            "specified_zone_edges": int(regional_masks.spec_edges.size),
            "relaxation_zone_edges": int(regional_masks.relax_edges.size),
            "bdy_mask_sha256": regional_boundary_mask_digest(
                {
                    name: mesh.arrays[name]
                    for name in REGIONAL_BOUNDARY_MASK_NAMES
                    if name in mesh.arrays
                }
            ),
        }
    return {
        "config": config,
        "regional": regional,
        "convection": decision,
        "pbl_cadence": pbl_decision,
        "prepared": prepared,
        "constructor_values": constructor_values,
        "constructor_receipt": constructor_receipt,
        "gwdo_host": gwdo_host,
        "mesh_evidence": mesh_evidence,
        "f000_surface_diagnostics": f000_surface_diagnostics,
        "reconstruction_coefficients": reconstruction_overlay,
        "edge_normal_vectors": edge_normal_overlay,
        "defc": defc,
        "scalar_receipt": scalar_receipt,
        "case_pin_relaxation": relaxation,
        "start_time_text": constructor_receipt["config_start_time"],
    }


# --------------------------------------------------------------------------
# per-step health gate (kept: finite/positive laws; cheap device reductions)
# --------------------------------------------------------------------------
_STATE_HEALTH_FIELDS = ("rho", "rho_theta", "rho_u", "rho_w", "scalars")
_SAVED_HEALTH_FIELDS = (
    "theta_m",
    "exner",
    "density_perturbation",
    "rho_theta_perturbation",
    "pressure_perturbation",
    "normal_velocity",
    "vertical_velocity",
)


def step_health_gate(
    stack: Mapping[str, Any], step: int, cp: Any, *, trace_hot_cell: bool = False
) -> dict[str, Any]:
    """Refuse the moment the integration stops being finite and physical.

    Deliberately allocation-light.  The frozen Arwen phase-one seam is
    documented as sensitive to the CuPy device-pool layout, so this gate uses
    only min/max reductions (scalar results) and never materializes a
    full-size temporary such as ``isfinite(x)``.  min/max propagate NaN and
    carry +/-inf, so testing the two scalars is a complete finiteness test for
    the array.
    """

    atmosphere = stack["driver"].atmosphere
    state = atmosphere.state
    saved = atmosphere.saved
    groups = [(name, getattr(state, name)) for name in _STATE_HEALTH_FIELDS]
    groups += [(name, getattr(saved, name)) for name in _SAVED_HEALTH_FIELDS]
    # On a limited-area run every array is one element wider than the domain
    # it solves: native allocates nCells+1 (and nEdges+1) and holds pool
    # values in that element -- rho_theta, theta_m and exner are the pool
    # ZERO there by native's own rule.  This gate refuses a non-positive
    # theta_m, so reducing over the allocation instead of the domain refuses
    # every limited-area step at step 1 for a column that is not a column.
    # The bound is the model's, not a tolerance: the health of an element
    # native never solves is not a statement about the forecast.
    solve_cells = stack.get("solve_cells")
    solve_edges = stack.get("solve_edges")
    padded_extents = {
        int(value) + 1: int(value)
        for value in (solve_cells, solve_edges)
        if value is not None
    }

    def _domain(array: Any) -> Any:
        if not padded_extents:
            return array
        trimmed = padded_extents.get(int(array.shape[-1]))
        return array if trimmed is None else array[..., :trimmed]

    envelope: dict[str, list[float]] = {}
    nonfinite: list[str] = []
    for name, value in groups:
        array = cp.asarray(value)
        if array.dtype.kind != "f":
            continue
        array = _domain(array)
        low = float(cp.min(array))
        high = float(cp.max(array))
        envelope[name] = [low, high]
        if not (math.isfinite(low) and math.isfinite(high)):
            nonfinite.append(name)
    if nonfinite:
        raise FloatingPointError(
            f"step {step} produced non-finite {sorted(nonfinite)}: "
            + json.dumps({name: envelope[name] for name in sorted(nonfinite)})
        )
    rho_min = envelope["rho"][0]
    theta_min = envelope["theta_m"][0]
    exner_min = envelope["exner"][0]
    if rho_min <= 0.0 or theta_min <= 0.0 or exner_min <= 0.0:
        raise FloatingPointError(
            f"step {step} rho/theta_m/exner not strictly positive: "
            f"{(rho_min, theta_min, exner_min)}"
        )
    w_low, w_high = envelope["vertical_velocity"]
    w_abs_max = max(abs(w_low), abs(w_high))
    # WHERE, not only how big.  On a limited-area run the single most useful
    # fact about a growing vertical velocity is which boundary ring it sits
    # in: ring 0 is the free interior and the forecast owns it, rings 1-7 are
    # driven and a maximum there is the boundary treatment talking.  The two
    # readings cost one argmax on a state already resident.
    hot: dict[str, Any] | None = None
    if trace_hot_cell:
        w_field = _domain(cp.asarray(saved.vertical_velocity))
        flat = int(cp.argmax(cp.abs(w_field)))
        cell = flat % int(w_field.shape[-1])
        hot = {
            "vertical_velocity_max_abs": w_abs_max,
            "cell": int(cell),
            "level": int(flat // int(w_field.shape[-1])),
        }
        rings = stack.get("bdy_mask_cell")
        if rings is not None and cell < len(rings):
            hot["boundary_ring"] = int(rings[cell])
            hot["zone"] = (
                "free interior"
                if int(rings[cell]) == 0
                else f"driven boundary ring {int(rings[cell])} of 7"
            )
    if w_abs_max > VERTICAL_VELOCITY_REFUSAL_M_S:
        raise FloatingPointError(
            f"step {step} vertical velocity {w_abs_max} m/s exceeds the "
            f"{VERTICAL_VELOCITY_REFUSAL_M_S} m/s divergence refusal"
            + ("" if hot is None else f"; worst column {json.dumps(hot)}")
        )
    scalars = _domain(cp.asarray(state.scalars))
    qv_min = float(cp.min(scalars[0]))
    qv_max = float(cp.max(scalars[0]))
    hydrometeor_min = float(cp.min(scalars[1:]))
    if not (
        math.isfinite(qv_min) and math.isfinite(qv_max) and math.isfinite(hydrometeor_min)
    ):
        raise FloatingPointError(f"step {step} produced non-finite moisture")
    if hydrometeor_min < 0.0:
        raise FloatingPointError(
            f"step {step} left a negative hydrometeor: {hydrometeor_min}"
        )
    return {
        "step": step,
        "rho_min": rho_min,
        "theta_m_min": theta_min,
        "theta_m_max": envelope["theta_m"][1],
        "exner_min": exner_min,
        "vertical_velocity_abs_max": w_abs_max,
        "qv_min": qv_min,
        "qv_max": qv_max,
        "hydrometeor_min": hydrometeor_min,
        "hot_cell": hot,
        "finite": True,
    }


# --------------------------------------------------------------------------
# evidence writers (partition-lane compatible)
# --------------------------------------------------------------------------
class BoundaryFingerprintWriter:
    """Append-only ``step -> {atmosphere, backend}`` JSONL, partstream format."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(self.path)
        self._stream = self.path.open("w", encoding="utf-8")
        self._steps: list[int] = []

    def write(self, step: int, fingerprint: Mapping[str, Any]) -> None:
        if self._steps and step <= self._steps[-1]:
            raise ValueError(
                f"boundary fingerprints must ascend: {step} after {self._steps[-1]}"
            )
        self._steps.append(int(step))
        record = {"step": int(step), **dict(fingerprint)}
        self._stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    @property
    def steps(self) -> list[int]:
        return list(self._steps)

    def close(self) -> None:
        self._stream.close()


def _snapshot_q2_hash(snapshot: Mapping[str, Any]) -> str | None:
    value = snapshot["arrays"].get("q2")
    return None if value is None else proof.array_sha256(value)


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
def build_schedule(
    *, hours: float, history_every_minutes: int, start_text: str
) -> dict[str, Any]:
    total_seconds = float(hours) * 3600.0
    steps = total_seconds / DT_SECONDS
    if steps <= 0 or abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            f"--hours {hours} is not a whole number of {DT_SECONDS:.0f} s steps"
        )
    steps = int(round(steps))
    history_seconds = int(history_every_minutes) * 60
    if history_seconds <= 0 or history_seconds % int(DT_SECONDS) != 0:
        raise ValueError(
            f"--history-every-minutes {history_every_minutes} is not a whole "
            f"number of {DT_SECONDS:.0f} s steps"
        )
    stride = history_seconds // int(DT_SECONDS)
    if steps % stride != 0:
        raise ValueError("the history cadence does not divide the forecast length")
    capture_steps = list(range(0, steps + 1, stride))
    start = datetime.strptime(start_text, "%Y-%m-%d_%H:%M:%S")
    labels = {
        step: (start + timedelta(seconds=step * DT_SECONDS)).strftime(
            "%Y-%m-%d_%H.%M.%S"
        )
        for step in capture_steps
    }
    return {
        "start_time": start_text,
        "dt_seconds": DT_SECONDS,
        "forecast_hours": float(hours),
        "steps": steps,
        "history_every_minutes": int(history_every_minutes),
        "history_stride_steps": stride,
        "capture_steps": capture_steps,
        "labels": labels,
        "valid_times": {
            step: (start + timedelta(seconds=step * DT_SECONDS)).strftime(
                "%Y-%m-%d_%H:%M:%S"
            )
            for step in capture_steps
        },
    }


# --------------------------------------------------------------------------
# the forecast
# --------------------------------------------------------------------------
def _mixing_treatment_proof(driver: Any, executed_steps: int) -> dict[str, Any]:
    """Positive evidence the mixing treatment ran (or provably did not).

    A/B rule: exact reproduction of a mixing-off outcome without this proof
    is first evidence the treatment never ran.  Expected RK1 mixing calls =
    3 dynamics subcycles per executed step.
    """

    cfg = getattr(driver, "mixing_config_v841", None)
    calls = int(getattr(driver, "mixing_calls_v841", 0))
    expected = 3 * int(executed_steps)
    active = cfg is not None
    proof_block: dict[str, Any] = {
        "active": active,
        "lane": "v841_2d_smagorinsky" if active else "off",
        "rk1_mixing_calls": calls,
        "expected_rk1_mixing_calls": expected if active else 0,
        "calls_match_expected": (
            calls == expected if active else calls == 0
        ),
        "deformation_weights": getattr(
            driver, "deformation_weights_receipt_v841", None
        ),
    }
    if active:
        proof_block["config"] = {
            "config_horiz_mixing": cfg.config_horiz_mixing,
            "config_len_disp": float(cfg.config_len_disp),
            "config_visc4_2dsmag": float(cfg.config_visc4_2dsmag),
            "config_smagorinsky_coef": float(cfg.config_smagorinsky_coef),
            "config_del4u_div_factor": float(cfg.config_del4u_div_factor),
            "config_h_ScaleWithMesh": bool(cfg.config_h_ScaleWithMesh),
        }
        proof_block["note"] = (
            "numbers under this lane are a new sub-series; not "
            "bit-comparable to any mixing-off arm"
        )
    else:
        proof_block["note"] = (
            "REPORTED AS THE PRE-MIXING CONTROL CONFIGURATION: native "
            "itself dies in this regime on convective cases (reference-node "
            "control, case B, step 466)"
        )
    return proof_block


def execute_forecast(
    *,
    host: Mapping[str, Any],
    schedule: Mapping[str, Any],
    cache_root: Path,
    output_root: Path,
    arwen_checkout: Path,
    source_receipt: Mapping[str, Any],
    authority_receipt: Mapping[str, Any],
    fingerprint_every: int,
    stop_on_refusal: bool = False,
    grid_path: Path | None = None,
    park_physics_tier: bool = False,
    required_free_bytes: int | None = None,
) -> dict[str, Any]:
    from hexcore.cuda_arwen_physics_v841 import pin_arwen_physics_v841

    arwen_pin = dict(pin_arwen_physics_v841(arwen_checkout))
    # This must precede KernelCache's gpuwm platform-binding construction.
    from hexcore.cuda_backend import KernelCache, require_cuda

    capability = require_cuda(
        min_compute=(12, 0), required_compute=(12, 0), cache_dir=cache_root
    )
    import cupy as cp

    # ``required_free_bytes`` is the forecast door's own admission sum,
    # forwarded so this floor and the door's verdict are one number enforced
    # twice: without it, a card admitted at the door on its OWN measured row
    # would be refused here on the default model's larger fixed term, after
    # the mesh bind and the kernel compile were already paid for.  Absent,
    # the mesh-bound floor from the same admission surface applies.
    if required_free_bytes is None:
        memory = proof.gpu_memory_admission(cp)
    else:
        memory = proof.gpu_memory_admission(cp, minimum=int(required_free_bytes))
    cache = KernelCache(capability=capability, cache_dir=cache_root)
    stack = proof._construct_device_stack(
        host=host, cache=cache, arwen_checkout=arwen_checkout
    )
    # Opt-in local time stepping.  Returns None -- and rebinds nothing -- when
    # config_local_timestep is off, which is the default.
    from hexcore.cuda_driver_lts import attach_local_timestep

    lts_attachment = attach_local_timestep(
        stack["driver"], grid_path=str(grid_path) if grid_path else None
    )
    stack["local_timestep"] = lts_attachment

    physics_park = None
    if park_physics_tier:
        from hexcore.cuda_physics_tier_park_v841 import CudaPhysicsTierParkV841

        physics_park = CudaPhysicsTierParkV841(
            cp, stack["backend"]._seam, diagnose=True
        )

    capture_steps = set(schedule["capture_steps"])
    labels = schedule["labels"]
    steps = int(schedule["steps"])
    static = proof._static_output_fields(host)

    fingerprint_path = output_root / "boundary-fingerprints.jsonl"
    fingerprints = (
        BoundaryFingerprintWriter(fingerprint_path) if fingerprint_every > 0 else None
    )
    snapshot_projection: dict[str, dict[str, str]] = {}
    snapshot_q2: dict[str, str | None] = {}
    snapshot_receipts: dict[str, Any] = {}
    snapshot_files: dict[str, Any] = {}
    physical_gates: dict[str, Any] = {}
    health: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []

    capture_seconds = 0.0
    write_seconds = 0.0
    fingerprint_seconds = 0.0
    health_seconds = 0.0
    integration_seconds = 0.0
    first_step_seconds = None

    def capture(step: int) -> None:
        nonlocal capture_seconds, write_seconds
        mark = time.perf_counter()
        snapshot = proof.capture_snapshot(
            label=labels[step],
            step=step,
            driver=stack["driver"],
            backend=stack["backend"],
            prep_geometry=stack["prep_geometry"],
            kernel_cache=stack["driver"].cache,
            f000_surface_diagnostics=stack["f000_surface_diagnostics"],
            expect_refl10cm=True,
            solve_cells=stack.get("solve_cells"),
        )
        capture_seconds += time.perf_counter() - mark
        physical_gates[str(step)] = proof.physical_snapshot_gate(
            snapshot, allow_initial_negative_qv=(step == 0)
        )
        snapshot_projection[str(step)] = proof._snapshot_hash_projection(snapshot)
        snapshot_q2[str(step)] = _snapshot_q2_hash(snapshot)
        snapshot_receipts[str(step)] = snapshot["receipt"]
        mark = time.perf_counter()
        snapshot_files[str(step)] = proof.write_snapshot_netcdf(
            output_root / f"cuda-history.{labels[step]}.nc", snapshot, static
        )
        write_seconds += time.perf_counter() - mark
        del snapshot
        gc.collect()

    # ORDER MATTERS.  ``proof._run_steps`` reads the previous surface updates
    # BEFORE capturing its start-step snapshot; capture allocates device
    # memory, so keeping this order keeps the allocation history aligned with
    # the proof arm the fork-equivalence gate compares against.
    previous = proof._previous_surface_updates(stack)
    if 0 in capture_steps:
        capture(0)
    if fingerprints is not None:
        mark = time.perf_counter()
        fingerprints.write(0, proof.fingerprint_execution_boundary(stack))
        fingerprint_seconds += time.perf_counter() - mark

    refusal: dict[str, Any] | None = None
    executed_steps = steps
    loop_started = time.perf_counter()
    # GF advective forcing carried step to step; None at step 1 is native's
    # own start state (tend_physics is zero before dynamics first forms it).
    gf_dynamics_tendencies = None
    for step in range(1, steps + 1):
        mark = time.perf_counter()
        try:
            result = proof.execute_composite_step(
                driver=stack["driver"],
                backend=stack["backend"],
                scalar_names=SCALAR_NAMES,
                physics_geometry=stack["physics_geometry"],
                kernel_cache=stack["driver"].cache,
                previous_surface_updates=previous,
                dynamics_tendencies=gf_dynamics_tendencies,
                physics_park=physics_park,
                # WRF/native diagflag: the step ENDING at a history frame
                # computes refl10cm inside its own microphysics call.
                refl_10cm_due=(step in capture_steps),
            )
        except (proof.CompositeTransactionError, FloatingPointError) as error:
            if not stop_on_refusal:
                raise
            # The port refused to publish this step.  The staged two-owner
            # transaction rolled back, so the committed state is still the
            # previous step.  NO GUARD IS RELAXED: the refusal is recorded
            # verbatim, the forecast stops here, and every retained frame
            # precedes the refused step.
            refusal = {
                "refused": True,
                "step": step,
                "model_seconds": step * DT_SECONDS,
                "exception": type(error).__name__,
                "message": str(error),
                "last_committed_step": step - 1,
                "note": (
                    "the port own numeric/geometry validation refused this "
                    "step; the forecast is truncated at the last committed "
                    "step and no unpublished state was retained"
                ),
            }
            executed_steps = step - 1
            break
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - mark
        integration_seconds += elapsed
        if first_step_seconds is None:
            first_step_seconds = elapsed
        previous = result.committed.surface_updates
        gf_dynamics_tendencies = result.dynamics_tendencies
        backend_receipt = dict(result.backend_receipt)
        surface_execution = proof.require_arwen_v2_surface_execution(
            backend_receipt, executed=True, label=f"step {step} backend receipt"
        )
        step_receipts.append(
            {
                "step": step,
                "seconds": elapsed,
                "driver": asdict(result.committed.receipt),
                "backend": backend_receipt,
                "arwen_v2_surface_execution": surface_execution,
                "clamp_d2h": result.clamp_d2h.as_dict(),
                "recovery": result.recovery.receipt(),
            }
        )
        mark = time.perf_counter()
        try:
            health.append(
                step_health_gate(
                    stack,
                    step,
                    cp,
                    trace_hot_cell=stack.get("solve_cells") is not None,
                )
            )
        except FloatingPointError as error:
            if not stop_on_refusal:
                raise
            # The step was committed but the health gate refuses the state it
            # produced (non-finite, non-positive, or |w| beyond the divergence
            # refusal).  Record the refusal verbatim WITH the receipt so the
            # committed health-envelope series (including this death) is
            # preserved for signature analysis; nothing is relaxed.
            refusal = {
                "refused": True,
                "refused_in": "step health gate",
                "step": step,
                "model_seconds": step * DT_SECONDS,
                "exception": type(error).__name__,
                "message": str(error),
                "last_committed_step": step,
                "note": (
                    "the composite step committed but its state fails the "
                    "health gate; the forecast is truncated here and the "
                    "per-step health envelopes up to the previous step are "
                    "retained in this receipt"
                ),
            }
            executed_steps = step
            health_seconds += time.perf_counter() - mark
            break
        health_seconds += time.perf_counter() - mark
        if fingerprints is not None and step % fingerprint_every == 0:
            mark = time.perf_counter()
            fingerprints.write(step, proof.fingerprint_execution_boundary(stack))
            fingerprint_seconds += time.perf_counter() - mark
        if step in capture_steps:
            try:
                capture(step)
            except (
                FloatingPointError,
                RuntimeError,
                proof.CompositeTransactionError,
            ) as error:
                if not stop_on_refusal:
                    raise
                # Writing a frame re-runs the MPAS-to-physics preparation.  The
                # state immediately before the instability already fails it, so
                # this frame cannot be produced.  Stop here with the frames that
                # were captured cleanly.
                refusal = {
                    "refused": True,
                    "refused_in": "history capture",
                    "step": step,
                    "model_seconds": step * DT_SECONDS,
                    "exception": type(error).__name__,
                    "message": str(error),
                    "last_committed_step": step,
                    "note": (
                        "the step integrated and published, but the port own "
                        "numeric/geometry validation refused to prepare it for "
                        "physics, so no history frame exists for it; the "
                        "forecast is truncated at the last frame written"
                    ),
                }
                executed_steps = step
                break
    loop_seconds = time.perf_counter() - loop_started
    if refusal is not None and executed_steps > 0 and executed_steps not in capture_steps:
        # Keep the last committed state renderable even though the refusal
        # landed between scheduled captures.
        stamp = datetime.strptime(schedule["start_time"], "%Y-%m-%d_%H:%M:%S") + timedelta(
            seconds=executed_steps * DT_SECONDS
        )
        labels[executed_steps] = stamp.strftime("%Y-%m-%d_%H.%M.%S")
        schedule["valid_times"][executed_steps] = stamp.strftime("%Y-%m-%d_%H:%M:%S")
        install_capture_labels(labels)
        try:
            capture(executed_steps)
        except Exception as error:  # noqa: BLE001 - bonus capture, never fatal
            # Capturing a snapshot re-runs the MPAS-to-physics preparation, and
            # on a state the port has just refused that preparation refuses
            # again.  The last committed state is therefore not representable
            # as a history frame.  Say so; keep the scheduled frames captured
            # before the instability.
            refusal["final_state_capture_failed"] = True
            refusal["final_state_capture_error"] = str(error)
            labels.pop(executed_steps, None)
            schedule["valid_times"].pop(executed_steps, None)
        else:
            capture_steps.add(executed_steps)
    if refusal is not None:
        steps = executed_steps
    if fingerprints is not None:
        fingerprints.close()

    proof._write_exclusive_json(
        output_root / "snapshot-hashes.json",
        {"projection": snapshot_projection, "q2": snapshot_q2},
    )
    proof._write_exclusive_json(
        output_root / "step-receipts.json",
        {"schema": SCHEMA + "/step-receipts", "receipts": step_receipts},
    )

    evolution = None
    if 0 in capture_steps and steps in capture_steps:
        evolution = {
            "note": (
                "surface evolution is reported by the per-step health trace; the "
                "proof's two-snapshot evolution gate is bound to its own case"
            )
        }
    return {
        "capability": capability.as_dict(),
        "arwen_pre_kernel_cache_pin": arwen_pin,
        "memory_admission": memory,
        "physics_tier_park": (
            None
            if physics_park is None
            else {
                **physics_park.receipt(),
                "window": (
                    "held on pinned host memory from after the phase-1 "
                    "tendencies are coupled until immediately before phase 2"
                ),
                "nonclaim": (
                    "not bit-identity: restored allocations sit at different "
                    "device addresses, so the payload digest of the two arms "
                    "is the only evidence that admits or refuses this"
                ),
            }
        ),
        "source_pins": source_receipt,
        "authority": authority_receipt,
        "regional": (
            None
            if host.get("regional") is None
            else {
                **host["regional"],
                "start_time": host["regional"]["start_time"].strftime(
                    "%Y-%m-%d_%H:%M:%S"
                ),
                "config_apply_lbcs": True,
                "lbc_intervals": len(host["regional"]["lbc_paths"]),
                "anchor": (
                    None
                    if getattr(stack["driver"], "regional_v841", None) is None
                    else getattr(stack["driver"].regional_v841, "anchor", None)
                ),
                "runtime": (
                    None
                    if getattr(stack["driver"], "regional_v841", None) is None
                    else stack["driver"].regional_v841.receipt()
                ),
            }
        ),
        "host_preparation": {
            "constructor": host["constructor_receipt"],
            "scalars": host["scalar_receipt"],
            "mesh_overlay": host["mesh_evidence"],
            "reconstruction_coefficients": host["reconstruction_coefficients"],
            "edge_normal_vectors": host["edge_normal_vectors"],
            "inactive_deformation": host["defc"],
            "case_pin_relaxation": host["case_pin_relaxation"],
        },
        "schedule": {
            key: value for key, value in schedule.items() if key != "labels"
        },
        "history_labels": schedule["labels"],
        "walls": {
            "integration_seconds": integration_seconds,
            "integration_note": (
                "sum of the per-step composite transaction only, each closed by "
                "a stream synchronize; EXCLUDES host preparation, device stack "
                "construction, snapshot capture, history writing, boundary "
                "fingerprinting and the per-step health gate"
            ),
            "first_step_seconds": first_step_seconds,
            "integration_seconds_excluding_first_step": (
                integration_seconds - (first_step_seconds or 0.0)
            ),
            "first_step_note": (
                "the first step carries cold NVRTC kernel compilation for this "
                "cache root"
            ),
            "seconds_per_step_after_first": (
                (integration_seconds - (first_step_seconds or 0.0)) / (steps - 1)
                if steps > 1
                else None
            ),
            "loop_seconds_including_io": loop_seconds,
            "snapshot_capture_seconds": capture_seconds,
            "history_write_seconds": write_seconds,
            "boundary_fingerprint_seconds": fingerprint_seconds,
            "health_gate_seconds": health_seconds,
            "steps": steps,
            "forecast_seconds": steps * DT_SECONDS,
        },
        "physical_gates": physical_gates,
        "step_health": health,
        "step_receipt_count": len(step_receipts),
        "snapshot_receipts": snapshot_receipts,
        "snapshot_files": snapshot_files,
        "refusal": refusal,
        "steps_requested": int(schedule["steps"]),
        "steps_executed": executed_steps,
        "boundary_fingerprints": {
            "path": str(fingerprint_path) if fingerprints is not None else None,
            "every": fingerprint_every,
            "steps": fingerprints.steps if fingerprints is not None else [],
        },
        "surface_evolution": evolution,
        "local_timestep_treatment": (
            None
            if stack.get("local_timestep") is None
            else stack["local_timestep"].receipt()
        ),
        "horizontal_mixing": _mixing_treatment_proof(
            stack["driver"], executed_steps
        ),
        "gf_deviation": {
            "mpas_dynamics_tendencies_computed": True,
            "fa35_public_api_accepts_them": False,
            "fa35_rthften_rqvften": "zero",
            "native_gf_parity_claim": False,
        },
        "full_physics_cuda_executed": True,
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = proof.default_authority_paths()
    parser.add_argument("--grid", type=Path, default=defaults["grid"])
    parser.add_argument("--static", type=Path, default=defaults["static"])
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument(
        "--init-source",
        required=True,
        help="provenance sentence for the init (e.g. 'ERA5 2025-03-14 12Z')",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="asserted against the init's config_start_time; the init is the authority",
    )
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--history-every-minutes", type=int, required=True)
    parser.add_argument(
        "--arwen-checkout",
        type=Path,
        default=ROOT / "work" / "arwen19-mpas-column-corrected",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fingerprint-every",
        type=int,
        default=0,
        help="write a boundary fingerprint every N steps (0 disables; the "
        "fork-equivalence gate uses 1)",
    )
    parser.add_argument("--case-label", default=None)
    parser.add_argument(
        "--horiz-mixing",
        choices=("2d_smagorinsky", "off"),
        default="2d_smagorinsky",
        help=(
            "horizontal mixing lane; the default is native's Registry "
            "2d_smagorinsky (deformation-based, hexcore.mixing_v841). "
            "'off' selects the pre-mixing control lane and is reported as "
            "the configuration native itself cannot integrate on "
            "convective cases"
        ),
    )
    parser.add_argument(
        "--convection",
        choices=("auto", "off", "gf"),
        default="auto",
        help=(
            "cumulus selection.  The default APPLIES DREW'S 2026-08-26 "
            "RULING: convection is switched off below 3 km, decided from the "
            "bound mesh's own finest spacing with no flag passed.  'off' and "
            "'gf' are explicit A/B arms -- they record themselves as "
            "explicit in the receipt, and an arm that overrides the ruling "
            "says so.  See hexcore.convection_admission"
        ),
    )
    parser.add_argument(
        "--pbl-cadence",
        default="auto",
        metavar="{auto,SECONDS}",
        help=(
            "surface/PBL cadence (config_bldt_seconds).  The default 'auto' "
            "is the PROVEN CONFIGURATION: the cadence is welded to config_dt, "
            "so the surface layer, the land-surface model and the PBL run on "
            "every model step, exactly as the native x4 v8.4.1 reference "
            "ran.  An explicit number of seconds HOLDS the cadence there "
            "while config_dt moves -- an A/B instrument for measuring "
            "whether a forcing scales with call count rather than with "
            "elapsed time.  It records itself as explicit in the receipt and "
            "earns its own timestep anchor; it is never a remedy.  See "
            "hexcore.pbl_cadence"
        ),
    )
    parser.add_argument(
        "--local-timestep",
        action="store_true",
        help=(
            "OPT-IN, default off: advance coarse columns on fewer, longer "
            "acoustic sub-steps chosen from the grid file's own dcEdge. "
            "Native MPAS-A v8.4.1 has no local time stepping, so this is a "
            "DECLARED DIVERGENCE from native and the run is not bit-comparable "
            "to a default run on a variable-resolution mesh. On a quasi-uniform "
            "mesh every column lands in one class and the run is bit-identical "
            "to the default"
        ),
    )
    parser.add_argument(
        "--local-timestep-declared-off",
        action="store_true",
        help=(
            "GATE ARM, not a user feature: build the local-timestep "
            "configuration subtype with the switch OFF. The run must be "
            "bit-identical to a run with no local-timestep flag at all, which "
            "is what proves the option did not leak into the default path"
        ),
    )
    parser.add_argument(
        "--local-timestep-rates",
        default="1,3",
        help=(
            "comma-separated acoustic rate ladder; each rate must divide every "
            "RK stage's sub-step count, which for the released (1,3,6) schedule "
            "admits 1 and 3 only. Two classes by default"
        ),
    )
    parser.add_argument(
        "--local-timestep-buffer-rings",
        type=int,
        default=1,
        help="rings of cells demoted to the finer rate around a class boundary",
    )
    parser.add_argument(
        "--park-physics-tier",
        action="store_true",
        help=(
            "MEASURED NEGATIVE, kept only so the measurement reproduces: hold "
            "the frozen-Arwen physics seam's device residency in pinned host "
            "memory across the dynamics half of every step.  It moves 786.8 "
            "MiB and releases 735.4 MiB of it, and on x1.40962 it still made "
            "the allocator's reservation WORSE -- 4068.7 MiB with the park "
            "against 4055.3 MiB without it -- because the reservation, not "
            "the instantaneous peak, is what a card has to provide.  Alone it "
            "changes neither number, since the peak is inside phase-1 physics "
            "where the tier is being read.  Those absolutes were measured "
            "2026-08-20 at Arwen seam pin 629ddb6f0, BEFORE the Grell-Freitas "
            "local-memory frame cut; the A/B sign is what carries, and "
            "re-running the park at pin 0d04db712 is NOT MEASURED.  Do not "
            "reach for this as an optimisation.  Also not bit-identity: "
            "restored allocations sit at different device addresses, so "
            "arm-to-arm payload digests are the only evidence that admits a "
            "parked run"
        ),
    )
    parser.add_argument(
        "--required-free-bytes",
        type=int,
        default=None,
        help=(
            "free-device-byte requirement computed by the forecast door's "
            "admission (hexcore.device_admission.required_free_bytes over "
            "the card's resolved footprint row), forwarded so the door's "
            "verdict and this driver's floor are the same number.  Default: "
            "the mesh-bound floor from the same admission surface"
        ),
    )
    parser.add_argument(
        "--lbc-dir",
        type=Path,
        default=None,
        help=(
            "directory of lateral-boundary files (lbc.*.nc, as rw_mpas_lbc "
            "writes them) for a limited-area grid.  REQUIRED when --grid "
            "carries a bdyMask triple and refused when it does not: a "
            "limited-area domain integrated with no boundary series empties "
            "from its outer ring inward, and a global domain has no boundary "
            "zone for one to drive"
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--stop-on-refusal",
        action="store_true",
        help=(
            "when the port refuses to publish a step, stop and write the "
            "receipt for the frames already committed instead of aborting "
            "with no receipt; the refusal is recorded verbatim and no "
            "validation is relaxed"
        ),
    )
    args = parser.parse_args(argv)
    if not args.preflight_only and (args.cache_root is None or args.output is None):
        parser.error("execution requires --cache-root and --output")
    if args.fingerprint_every < 0:
        parser.error("--fingerprint-every must be >= 0")
    if args.required_free_bytes is not None and args.required_free_bytes <= 0:
        parser.error(
            "--required-free-bytes must be positive: a non-positive "
            "requirement is not a measured admission, it is the memory gate "
            "switched off, and the run it admits dies inside a CuPy "
            "allocation mid-integration"
        )
    try:
        rates = tuple(
            int(piece) for piece in str(args.local_timestep_rates).split(",") if piece
        )
    except ValueError:
        parser.error("--local-timestep-rates must be comma-separated integers")
    if not rates or rates[0] != 1 or list(rates) != sorted(set(rates)):
        parser.error(
            "--local-timestep-rates must be strictly increasing and start at 1"
        )
    args.local_timestep_rates = rates
    if args.local_timestep and args.local_timestep_declared_off:
        parser.error(
            "--local-timestep and --local-timestep-declared-off contradict: "
            "the second is the gate arm that proves the option is inert when "
            "the switch is off"
        )
    if args.local_timestep_buffer_rings < 1:
        parser.error("--local-timestep-buffer-rings must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {
        "grid": Path(args.grid).expanduser().absolute(),
        "static": Path(args.static).expanduser().absolute(),
        "init": Path(args.init).expanduser().absolute(),
    }
    arwen_checkout = proof._plain_absolute(args.arwen_checkout, "Arwen checkout")
    if not arwen_checkout.is_dir():
        raise FileNotFoundError(arwen_checkout)

    # Source pins verify first: the checkout guard imports the seam manifest
    # from a pinned module, so that module's bytes are proven before its
    # constants are trusted.
    source_before = proof.require_frozen_execution_sources()
    arwen_git_before = proof.verify_arwen_checkout_git(arwen_checkout)
    authority_before = verify_forecast_authorities(paths)
    lbc_paths: list[str] | None = None
    if args.lbc_dir is not None:
        lbc_dir = Path(args.lbc_dir).expanduser().absolute()
        lbc_paths = sorted(str(path) for path in lbc_dir.glob("lbc.*.nc"))
        if not lbc_paths:
            raise ConfigurationRefusal(
                "config_apply_lbcs",
                str(lbc_dir),
                "--lbc-dir names a directory with no lbc.*.nc in it, so the "
                "run would integrate a limited-area domain against no "
                "boundary series at all",
                "the --out-dir rw_mpas_lbc wrote its boundary files into",
            )
    host = prepare_forecast_host(
        paths,
        authority_before,
        lbc_paths=lbc_paths,
        start_time_text=args.start_time,
        horiz_mixing=args.horiz_mixing,
        convection=args.convection,
        pbl_cadence=args.pbl_cadence,
        local_timestep=args.local_timestep,
        local_timestep_declared_off=args.local_timestep_declared_off,
        local_timestep_rates=args.local_timestep_rates,
        local_timestep_buffer_rings=args.local_timestep_buffer_rings,
    )
    schedule = build_schedule(
        hours=args.hours,
        history_every_minutes=args.history_every_minutes,
        start_text=host["start_time_text"],
    )
    install_capture_labels(schedule["labels"])

    provenance = {
        "schema": SCHEMA,
        "receipt_mode": RECEIPT_MODE,
        "derived_from": DERIVED_FROM,
        "derived_from_sha256": proof.sha256_file(ROOT / DERIVED_FROM),
        "case_label": args.case_label,
        "init": {
            "path": str(paths["init"]),
            "bytes": authority_before["files"]["init"]["bytes"],
            "sha256": authority_before["files"]["init"]["sha256"],
            "source": args.init_source,
            "config_start_time": host["start_time_text"],
            "pinned": False,
        },
        "mesh": {
            role: authority_before["files"][role] for role in MESH_AUTHORITY_ROLES
        },
        "arwen_commit": proof.ARWEN_COMMIT,
        "arwen_contract_document_sha256": proof.ARWEN_CONTRACT_DOCUMENT_SHA256,
        "arwen_contract_surface_sha256": proof.ARWEN_CONTRACT_SURFACE_SHA256,
        "arwen_glacier_composed_tu_sha256": proof.ARWEN_GLACIER_COMPOSED_TU_SHA256,
        "profile": proof.PROFILE,
        "source_release": proof.SOURCE_RELEASE,
        "horiz_mixing": args.horiz_mixing,
        "convection": host["convection"],
        "local_timestep": {
            "enabled": bool(args.local_timestep),
            "declared_off_arm": bool(args.local_timestep_declared_off),
            "rates": list(args.local_timestep_rates),
            "buffer_rings": int(args.local_timestep_buffer_rings),
            "native_equivalent": False,
            "note": (
                "native MPAS-A v8.4.1 has no local time stepping; with the "
                "option on this run is a declared divergence, and with it off "
                "the executed path is the pinned one"
            ),
        },
        "config_type": type(host["config"]).__name__,
        "dropped_guarantees": list(DROPPED_GUARANTEES),
        "claim": CLAIM,
        "nonclaims": list(NONCLAIMS),
        "weather_plot_policy": "native Rust/Arwen renderer only; q2 ships in the history stream and its weather-field plots go through the same renderer",
    }

    if args.preflight_only:
        source_after = proof.require_frozen_execution_sources()
        authority_after = verify_forecast_authorities(paths)
        arwen_git_after = proof.verify_arwen_checkout_git(arwen_checkout)
        if (
            source_after != source_before
            or authority_after != authority_before
            or arwen_git_after != arwen_git_before
        ):
            raise RuntimeError(
                "source, authority, or Arwen bytes changed during preflight"
            )
        print(
            json.dumps(
                {
                    **provenance,
                    "mode": "preflight-only; CUDA not imported",
                    "sources": source_before,
                    "arwen_git": arwen_git_before,
                    "schedule": {
                        key: value
                        for key, value in schedule.items()
                        if key != "labels"
                    },
                    "constructor": host["constructor_receipt"],
                    "scalars": host["scalar_receipt"],
                    "case_pin_relaxation": host["case_pin_relaxation"],
                },
                sort_keys=True,
                default=str,
            )
        )
        return 0

    assert args.cache_root is not None and args.output is not None
    cache_root, output_root = proof.validate_destination(
        args.cache_root, args.output, tuple(paths.values())
    )
    cache_root.mkdir(parents=False)
    output_root.mkdir(parents=False)
    started = time.perf_counter()
    forecast = execute_forecast(
        host=host,
        schedule=schedule,
        cache_root=cache_root,
        output_root=output_root,
        arwen_checkout=arwen_checkout,
        source_receipt=source_before,
        authority_receipt=authority_before,
        fingerprint_every=int(args.fingerprint_every),
        stop_on_refusal=bool(args.stop_on_refusal),
        grid_path=paths["grid"],
        park_physics_tier=bool(args.park_physics_tier),
        required_free_bytes=args.required_free_bytes,
    )
    source_after = proof.require_frozen_execution_sources()
    authority_after = verify_forecast_authorities(paths)
    arwen_git_after = proof.verify_arwen_checkout_git(arwen_checkout)
    if (
        source_after != source_before
        or authority_after != authority_before
        or arwen_git_after != arwen_git_before
    ):
        raise RuntimeError("source, authority, or Arwen bytes changed during execution")
    payload = {
        **provenance,
        "status": (
            "truncated_by_model_refusal"
            if forecast.get("refusal")
            else "passed"
        ),
        "arwen_git": {"before": arwen_git_before, "after": arwen_git_after},
        "arwen_checkout_unchanged": True,
        "execution_seconds": time.perf_counter() - started,
        "forecast": forecast,
        "sources_unchanged": True,
        "authorities_unchanged": True,
    }
    payload["payload_sha256"] = proof.canonical_json_sha256(
        json.loads(json.dumps(payload, sort_keys=True, default=str))
    )
    receipt = output_root / RECEIPT_NAME
    proof._write_exclusive_json(
        receipt, json.loads(json.dumps(payload, sort_keys=True, default=str))
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt": str(receipt),
                "receipt_sha256": proof.sha256_file(receipt),
                "payload_sha256": payload["payload_sha256"],
                "integration_seconds": forecast["walls"]["integration_seconds"],
                "steps": forecast["walls"]["steps"],
                "history_frames": len(forecast["snapshot_files"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
