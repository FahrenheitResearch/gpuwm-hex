"""Resident MPAS-A v8.4.1 CUDA column-physics coupling contracts.

This module owns only the MPAS side of the physics seam.  A persistent
backend supplies uncoupled A-grid rates before RK and an in-place WSM6 state
after RK.  The functions here validate those resident buffers, convert the
held rates to MPAS conservative tendencies, and recover the post-microphysics
MPAS prognostic state.  No parameterization is implemented here.

The numerical authority is the pinned official MPAS-Model v8.4.1 source:

* ``mpas_atm_time_integration.F:2142-2151`` computes physics once before RK;
* ``mpas_atmphys_todynamics.F:385-492`` mass-couples the held rates;
* ``mpas_atmphys_todynamics.F:501-552`` projects cell vectors to edges; and
* ``mpas_atm_time_integration.F:2798-2816`` clamps water and calls
  microphysics after RK/transport, with the dry-theta recovery spelled in
  ``mpas_atmphys_interface.F:910-920``.

All public array contracts are resident ``cupy.ndarray`` values, binary32,
C-contiguous, with entity as the fastest axis.  Host arrays and strided views
are refused rather than copied implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .cuda_backend.containers import TransferStats, require_resident_array
from .cuda_backend.compile_contract import (
    CompileContractError,
    canonical_sha256,
    validate_compile_platform_fingerprint,
)
from .cuda_backend.runtime import KernelCache
from .cuda_fp32 import CUDA_FTZ_HELPERS


CUDA_PHYSICS_V841_SCHEMA = "mpas-port.cuda-physics-v841/v1"
CUDA_PHASE_ONE_EXECUTION_PROVENANCE_V841_SCHEMA = (
    "mpas-port.cuda-physics-v841/phase-one-execution-provenance/v1"
)
_PHASE_ONE_GWD_OFF_MODE = "pinned_arwen_aggregate_gwd_off"
_PHASE_ONE_EXTERNAL_GWDO_MODE = "pinned_arwen_aggregate_external_gwdo"

_AUTHORITY_DOCUMENT = {
    "schema": CUDA_PHYSICS_V841_SCHEMA,
    "mpas_version": "8.4.1",
    "precision": "resident-fp32-c-contiguous-level-entity",
    "cuda_authority": {
        "module_key": "hexcore.cuda_physics_v841",
        "compile_options": ["--std=c++17", "--fmad=false"],
        "compile_options_policy": "exact ordered base_options; no overrides",
        "arithmetic": "CUDA_FTZ_HELPERS mpas_add/mpas_mul/mpas_div",
        "implicit_cupy_authority_arithmetic": False,
    },
    "phases": {
        "pre_rk": {
            "source": "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
            "lines": "2142-2151",
            "meaning": "one physics_get_tend call held through all RK stages",
        },
        "coupling": {
            "source": "src/core_atmosphere/physics/mpas_atmphys_todynamics.F",
            "lines": "385-492,501-552",
            "meaning": "mass coupling, modified theta, and tend_toEdges",
        },
        "post_rk": {
            "source": "src/core_atmosphere/dynamics/mpas_atm_time_integration.F",
            "lines": "2798-2816",
            "meaning": "negative-water clamp followed by driver_microphysics",
        },
        "microphysics_recovery": {
            "source": "src/core_atmosphere/physics/mpas_atmphys_interface.F",
            "lines": "910-920",
            "meaning": "dry theta and qv returned as modified theta",
        },
    },
    "authority_sha256": {
        "mpas_atmphys_todynamics.F": (
            "a77b58f358991d6c18fc3718646600a9a3c9bf8db9fa24616f3eab0c2ef32b19"
        ),
        "mpas_atm_time_integration.F": (
            "937e3191a646b0f3f14aaf1678f57b0d6880574f06e402a5053ff6ed12ab706b"
        ),
        "mpas_atmphys_interface.F": (
            "165b21ecfb4e599ce512a46ebc07a38d24abe4f76e4bf47388d15706fbc49ff2"
        ),
    },
    "raw_result": ["du", "dv", "dtheta", "dscalars", "execution_provenance"],
    "held_result": [
        "rho",
        "rho_u",
        "rho_theta",
        "scalars",
        "execution_provenance",
    ],
    "execution_provenance": (
        "immutable exact pinned Arwen aggregate execution identity, with optional "
        "external YSU-GWDO execution/composition identity and four-byte validation"
    ),
    "post_rk_wsm6": [
        "theta",
        "qv",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "rainnc",
        "rainncv",
        "snownc",
        "snowncv",
        "graupelnc",
        "graupelncv",
        "sr",
        "effc",
        "effi",
        "effs",
    ],
    "wsm6_sr_roundoff": {
        "source": (
            "pinned Arwen gpuwm/core/physics.py:_wsm6_sr_roundoff_limit and "
            "gpuwm/core/kernels/wsm6.cu"
        ),
        "meaning": (
            "accept the proven positive-sum binary32 SR roundoff envelope for "
            "the mandatory exact phase-two model timestep without clipping"
        ),
        "minor_dt_seconds": 120.0,
    },
}

CUDA_PHYSICS_V841_CONTRACT_SHA256 = sha256(
    json.dumps(_AUTHORITY_DOCUMENT, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()

CUDA_PHYSICS_V841_EVIDENCE: Mapping[str, Any] = MappingProxyType(_AUTHORITY_DOCUMENT)

RV_OVER_RD_F32 = np.float32(np.float32(461.6) / np.float32(287.0))
_WSM6_MINOR_DT_SECONDS_F32 = np.float32(120.0)
_FP32_SIGNIFICAND_SCALE = 1 << 24
_FP32_ONE_BITS = 0x3F800000
WSM6_SCALAR_NAMES = ("qv", "qc", "qr", "qi", "qs", "qg")
WSM6_SURFACE_NAMES = (
    "rainnc",
    "rainncv",
    "snownc",
    "snowncv",
    "graupelnc",
    "graupelncv",
    "sr",
)
WSM6_RADIUS_NAMES = ("effc", "effi", "effs")


def physics_contract_evidence_v841() -> dict[str, Any]:
    """Return a detached JSON-ready copy of the frozen authority receipt."""

    result = json.loads(json.dumps(_AUTHORITY_DOCUMENT, sort_keys=True))
    result["contract_sha256"] = CUDA_PHYSICS_V841_CONTRACT_SHA256
    result["kernel_sha256"] = CUDA_PHYSICS_V841_KERNEL_SHA256
    return result


def _cp() -> Any:
    import cupy as cp

    return cp


def _mesh_value(mesh: object, name: str, default: Any = None) -> Any:
    if hasattr(mesh, name):
        return getattr(mesh, name)
    if isinstance(mesh, Mapping) and name in mesh:
        return mesh[name]
    arrays = getattr(mesh, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(mesh, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _host_rows(value: Any, rows: int, columns: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (rows, columns):
        return np.ascontiguousarray(array)
    if array.shape == (columns, rows):
        return np.ascontiguousarray(array.T)
    raise ValueError(
        f"{name} must have shape {(rows, columns)} or {(columns, rows)}, "
        f"got {array.shape}"
    )


def _host_vectors(value: Any, count: int, name: str) -> np.ndarray:
    result = _host_rows(value, count, 3, name).astype(np.float32, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result, dtype=np.float32)


def _scalar_names(names: Sequence[str], count: int) -> tuple[str, ...]:
    result = tuple(str(name).strip().lower() for name in names)
    if len(result) != count:
        raise ValueError(
            f"scalar_names has {len(result)} entries but state carries {count}"
        )
    if any(not name for name in result) or len(set(result)) != len(result):
        raise ValueError("scalar_names must be non-empty and unique")
    return result


def _valid_clock(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _wsm6_sr_roundoff_limit_v841(dt: float) -> tuple[np.float32, int, int]:
    """Return the pinned Arwen/WRF WSM6 positive-sum SR FP32 envelope.

    WSM6 forms SR=(SNOWNCV+GRAUPELNCV)/(RAINNCV+1e-12). The numerator
    and denominator contain the same nonnegative frozen components but
    associate their binary32 additions differently, so an all-frozen column
    may finish a few ULPs above one. This is the exact integer-rational
    construction frozen by Arwen _wsm6_sr_roundoff_limit; it admits the
    proved expression-order roundoff and does not mutate the scheme output.
    """

    if isinstance(dt, (bool, np.bool_)):
        raise TypeError("phase2_dt_seconds must be a real number, not bool")
    value = float(dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("phase2_dt_seconds must be finite and positive")
    delt = np.float32(value)
    if not np.isfinite(delt) or delt <= np.float32(0.0):
        raise ValueError("phase2_dt_seconds is outside the positive FP32 range")
    loops = max(
        int(
            np.floor(
                np.float32(
                    delt / _WSM6_MINOR_DT_SECONDS_F32 + np.float32(0.5)
                )
            )
        ),
        1,
    )
    accumulation_adds = loops - 1
    scale = _FP32_SIGNIFICAND_SCALE
    if 2 * accumulation_adds >= scale:
        raise ValueError(
            "WSM6 minor-loop count is too large for the proven FP32 SR "
            f"roundoff envelope: loops={loops}"
        )

    numerator = (scale + 1) ** 3 * (scale - 3)
    denominator = scale**2 * (scale - 6) * (scale - 2 * accumulation_adds)
    if numerator >= 2 * denominator:
        raise ValueError(
            "WSM6 FP32 SR roundoff bound reaches 2.0, where ULP(1) "
            f"linearity no longer applies: loops={loops}"
        )
    scaled_delta = (numerator - denominator) * scale
    scaled_ulp = 2 * denominator
    max_ulps = scaled_delta // scaled_ulp
    upper_bits = _FP32_ONE_BITS + max_ulps
    upper = np.asarray(upper_bits, dtype=np.uint32).view(np.float32)[()]
    return upper, int(max_ulps), loops


@dataclass(slots=True)
class CudaPhysicsGeometryV841:
    """One-time uploaded MPAS ``east/north/edgeNormalVectors`` geometry."""

    cells_on_edge: Any
    east_cell: Any
    north_cell: Any
    edge_normal_vectors: Any
    n_cells: int
    n_edges: int
    h2d: TransferStats
    #: Native's garbage element on a limited-area mesh, or ``None`` on a
    #: closed sphere.  The mesh view declares it; see the zeroing beside the
    #: edge projection for what it is used for and why.
    regional_garbage_cell: int | None = None

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    @classmethod
    def from_host(
        cls,
        mesh: object,
        *,
        east: Any | None = None,
        north: Any | None = None,
        edge_normal_vectors: Any | None = None,
    ) -> "CudaPhysicsGeometryV841":
        """Upload the exact geometry consumed by ``tend_toEdges``.

        A prepared MPAS mesh may supply its source-derived ``east``, ``north``
        and ``edgeNormalVectors`` arrays directly.  For an archive mesh that
        predates those runtime arrays, the repository's readable official-
        source vector authority reconstructs the identical fields once on the
        host before upload.
        """

        cp = _cp()
        raw_cells = np.asarray(_mesh_value(mesh, "cellsOnEdge"))
        if raw_cells.ndim != 2 or 2 not in raw_cells.shape:
            raise ValueError("cellsOnEdge must be a two-dimensional pair table")
        n_edges = raw_cells.shape[0] if raw_cells.shape[1] == 2 else raw_cells.shape[1]
        cells = _host_rows(raw_cells, n_edges, 2, "cellsOnEdge")

        east_value = _mesh_value(mesh, "east") if east is None else east
        north_value = _mesh_value(mesh, "north") if north is None else north
        normal_value = (
            _mesh_value(mesh, "edgeNormalVectors")
            if edge_normal_vectors is None
            else edge_normal_vectors
        )
        if east_value is None or north_value is None:
            lon = np.asarray(_mesh_value(mesh, "lonCell"))
            lat = np.asarray(_mesh_value(mesh, "latCell"))
            if lon.ndim != 1 or lat.shape != lon.shape:
                raise ValueError(
                    "mesh needs source-derived east/north or matching lonCell/latCell"
                )
            from .vector import zonal_meridional_vectors

            east_value, north_value, _ = zonal_meridional_vectors(lon, lat)
        east_shape = np.asarray(east_value).shape
        if len(east_shape) != 2 or 3 not in east_shape:
            raise ValueError("east must be an nCells-by-3 source vector table")
        if east_shape[1] == 3:
            n_cells = int(east_shape[0])
        elif east_shape[0] == 3:
            n_cells = int(east_shape[1])
        else:
            raise ValueError("ambiguous east source vector table")
        east_host = _host_vectors(east_value, n_cells, "east")
        north_host = _host_vectors(north_value, n_cells, "north")
        if normal_value is None:
            from .vector import initialize_vector_geometry

            normal_value = initialize_vector_geometry(mesh).edge_normal_vectors
        normal_host = _host_vectors(normal_value, n_edges, "edgeNormalVectors")
        cells = np.ascontiguousarray(cells, dtype=np.int32)
        if np.any(cells < 0) or np.any(cells >= n_cells):
            raise ValueError(
                "tend_toEdges requires two valid cell endpoints on every edge"
            )

        garbage_value = _mesh_value(mesh, "garbage_cell")
        garbage_cell = None if garbage_value is None else int(garbage_value)
        if garbage_cell is not None and garbage_cell != n_cells - 1:
            raise ValueError(
                "a limited-area mesh's garbage element is the LAST column of "
                f"its padded extent; this view declares {garbage_cell} with "
                f"{n_cells} columns"
            )
        started = time.perf_counter()
        device_cells = cp.asarray(cells)
        device_east = cp.asarray(east_host)
        device_north = cp.asarray(north_host)
        device_normal = cp.asarray(normal_host)
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        result = cls(
            cells_on_edge=device_cells,
            east_cell=device_east,
            north_cell=device_north,
            edge_normal_vectors=device_normal,
            n_cells=n_cells,
            n_edges=n_edges,
            h2d=TransferStats(
                int(
                    device_cells.nbytes
                    + device_east.nbytes
                    + device_north.nbytes
                    + device_normal.nbytes
                ),
                elapsed,
            ),
            regional_garbage_cell=garbage_cell,
        )
        result.validate()
        return result

    def validate(self) -> None:
        require_resident_array(
            "physics_geometry.cells_on_edge",
            self.cells_on_edge,
            dtype=np.int32,
            shape=(self.n_edges, 2),
        )
        for name, value, shape in (
            ("east_cell", self.east_cell, (self.n_cells, 3)),
            ("north_cell", self.north_cell, (self.n_cells, 3)),
            ("edge_normal_vectors", self.edge_normal_vectors, (self.n_edges, 3)),
        ):
            require_resident_array(
                f"physics_geometry.{name}", value, dtype=np.float32, shape=shape
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_PHYSICS_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "n_cells": self.n_cells,
            "n_edges": self.n_edges,
            "h2d": self.h2d.as_dict(),
            "geometry": "MPAS east/north/edgeNormalVectors",
        }


@dataclass(frozen=True, slots=True)
class CudaPhaseOneExecutionProvenanceV841:
    """Immutable proof carrier for the aggregate phase-one execution actually used."""

    mode: str
    arwen_commit: str
    aggregate_factory: str
    phase1_orchestration: str
    source_manifest: tuple[tuple[str, str], ...]
    aggregate_executed: bool
    h_diabatic_applied: bool
    gwd_selector: str
    gwdo_composer: str | None
    gwdo_composition_phase: str | None
    gwdo_executed: bool
    gwdo_composed_once: bool
    gwdo_result_module: str | None
    gwdo_result_class: str | None
    gwdo_contract_sha256: str | None
    gwdo_kernel_sha256: str | None
    gwdo_validation_d2h: TransferStats | None
    gwdo_input_du_is_arwen_output: bool
    gwdo_input_dv_is_arwen_output: bool
    raw_du_is_gwdo_output: bool
    raw_dv_is_gwdo_output: bool

    def __post_init__(self) -> None:
        try:
            manifest = tuple(
                (str(relative), str(digest))
                for relative, digest in self.source_manifest
            )
        except (TypeError, ValueError) as error:
            raise TypeError(
                "phase-one source_manifest must be immutable (relative path, SHA256) pairs"
            ) from error
        object.__setattr__(self, "source_manifest", manifest)
        self.validate()

    @classmethod
    def arwen_gwd_off(
        cls, *, aggregate_executed: bool
    ) -> "CudaPhaseOneExecutionProvenanceV841":
        from .config_v841 import (
            V841_ARWEN_AGGREGATE_FACTORY,
            V841_ARWEN_BUILD_COMMIT,
            V841_ARWEN_H_DIABATIC_APPLIED,
            V841_ARWEN_PHASE1_ORCHESTRATION,
            V841_ARWEN_SOURCE_MANIFEST,
        )

        return cls(
            mode=_PHASE_ONE_GWD_OFF_MODE,
            arwen_commit=V841_ARWEN_BUILD_COMMIT,
            aggregate_factory=V841_ARWEN_AGGREGATE_FACTORY,
            phase1_orchestration=V841_ARWEN_PHASE1_ORCHESTRATION,
            source_manifest=V841_ARWEN_SOURCE_MANIFEST,
            aggregate_executed=aggregate_executed,
            h_diabatic_applied=V841_ARWEN_H_DIABATIC_APPLIED,
            gwd_selector="off",
            gwdo_composer=None,
            gwdo_composition_phase=None,
            gwdo_executed=False,
            gwdo_composed_once=False,
            gwdo_result_module=None,
            gwdo_result_class=None,
            gwdo_contract_sha256=None,
            gwdo_kernel_sha256=None,
            gwdo_validation_d2h=None,
            gwdo_input_du_is_arwen_output=False,
            gwdo_input_dv_is_arwen_output=False,
            raw_du_is_gwdo_output=False,
            raw_dv_is_gwdo_output=False,
        )

    @classmethod
    def arwen_with_external_gwdo(
        cls,
        *,
        aggregate_executed: bool,
        gwdo_executed: bool,
        gwdo_composed_once: bool,
        gwdo_result_module: str,
        gwdo_result_class: str,
        gwdo_contract_sha256: str,
        gwdo_kernel_sha256: str,
        gwdo_validation_d2h: TransferStats,
        gwdo_input_du_is_arwen_output: bool,
        gwdo_input_dv_is_arwen_output: bool,
        raw_du_is_gwdo_output: bool,
        raw_dv_is_gwdo_output: bool,
    ) -> "CudaPhaseOneExecutionProvenanceV841":
        from .config_v841 import (
            V841_ARWEN_AGGREGATE_FACTORY,
            V841_ARWEN_BUILD_COMMIT,
            V841_ARWEN_H_DIABATIC_APPLIED,
            V841_ARWEN_PHASE1_ORCHESTRATION,
            V841_ARWEN_SOURCE_MANIFEST,
            V841_GWDO_COMPOSITION_PHASE,
            V841_GWDO_EXTERNAL_COMPOSER,
        )

        return cls(
            mode=_PHASE_ONE_EXTERNAL_GWDO_MODE,
            arwen_commit=V841_ARWEN_BUILD_COMMIT,
            aggregate_factory=V841_ARWEN_AGGREGATE_FACTORY,
            phase1_orchestration=V841_ARWEN_PHASE1_ORCHESTRATION,
            source_manifest=V841_ARWEN_SOURCE_MANIFEST,
            aggregate_executed=aggregate_executed,
            h_diabatic_applied=V841_ARWEN_H_DIABATIC_APPLIED,
            gwd_selector="bl_ysu_gwdo",
            gwdo_composer=V841_GWDO_EXTERNAL_COMPOSER,
            gwdo_composition_phase=V841_GWDO_COMPOSITION_PHASE,
            gwdo_executed=gwdo_executed,
            gwdo_composed_once=gwdo_composed_once,
            gwdo_result_module=gwdo_result_module,
            gwdo_result_class=gwdo_result_class,
            gwdo_contract_sha256=gwdo_contract_sha256,
            gwdo_kernel_sha256=gwdo_kernel_sha256,
            gwdo_validation_d2h=gwdo_validation_d2h,
            gwdo_input_du_is_arwen_output=gwdo_input_du_is_arwen_output,
            gwdo_input_dv_is_arwen_output=gwdo_input_dv_is_arwen_output,
            raw_du_is_gwdo_output=raw_du_is_gwdo_output,
            raw_dv_is_gwdo_output=raw_dv_is_gwdo_output,
        )

    def validate(self) -> None:
        from .config_v841 import (
            V841_ARWEN_AGGREGATE_FACTORY,
            V841_ARWEN_BUILD_COMMIT,
            V841_ARWEN_H_DIABATIC_APPLIED,
            V841_ARWEN_PHASE1_ORCHESTRATION,
            V841_ARWEN_SOURCE_MANIFEST,
            V841_GWDO_COMPOSITION_PHASE,
            V841_GWDO_CONTRACT_SHA256,
            V841_GWDO_EXTERNAL_COMPOSER,
            V841_GWDO_KERNEL_SHA256,
            V841_GWDO_RESULT_CLASS,
            V841_GWDO_RESULT_MODULE,
        )

        aggregate = {
            "arwen_commit": (self.arwen_commit, V841_ARWEN_BUILD_COMMIT),
            "aggregate_factory": (
                self.aggregate_factory,
                V841_ARWEN_AGGREGATE_FACTORY,
            ),
            "phase1_orchestration": (
                self.phase1_orchestration,
                V841_ARWEN_PHASE1_ORCHESTRATION,
            ),
            "source_manifest": (
                self.source_manifest,
                V841_ARWEN_SOURCE_MANIFEST,
            ),
            "aggregate_executed": (self.aggregate_executed, True),
            "h_diabatic_applied": (
                self.h_diabatic_applied,
                V841_ARWEN_H_DIABATIC_APPLIED,
            ),
        }
        for name, (actual, required) in aggregate.items():
            if type(required) is bool:
                matches = type(actual) is bool and actual is required
            else:
                matches = actual == required
            if not matches:
                raise ValueError(
                    f"phase-one execution provenance {name} changed: "
                    f"{actual!r} != {required!r}"
                )

        booleans = (
            "gwdo_executed",
            "gwdo_composed_once",
            "gwdo_input_du_is_arwen_output",
            "gwdo_input_dv_is_arwen_output",
            "raw_du_is_gwdo_output",
            "raw_dv_is_gwdo_output",
        )
        for name in booleans:
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"phase-one execution provenance {name} must be bool")

        if self.gwd_selector == "off":
            exact_off = {
                "mode": _PHASE_ONE_GWD_OFF_MODE,
                "gwdo_composer": None,
                "gwdo_composition_phase": None,
                "gwdo_executed": False,
                "gwdo_composed_once": False,
                "gwdo_result_module": None,
                "gwdo_result_class": None,
                "gwdo_contract_sha256": None,
                "gwdo_kernel_sha256": None,
                "gwdo_validation_d2h": None,
                "gwdo_input_du_is_arwen_output": False,
                "gwdo_input_dv_is_arwen_output": False,
                "raw_du_is_gwdo_output": False,
                "raw_dv_is_gwdo_output": False,
            }
            for name, required in exact_off.items():
                if getattr(self, name) != required:
                    raise ValueError(
                        f"GWD-off provenance {name} must be exactly {required!r}"
                    )
            return

        if self.gwd_selector != "bl_ysu_gwdo":
            raise ValueError(
                f"unsupported phase-one GWD provenance selector {self.gwd_selector!r}"
            )
        exact_on = {
            "mode": _PHASE_ONE_EXTERNAL_GWDO_MODE,
            "gwdo_composer": V841_GWDO_EXTERNAL_COMPOSER,
            "gwdo_composition_phase": V841_GWDO_COMPOSITION_PHASE,
            "gwdo_executed": True,
            "gwdo_composed_once": True,
            "gwdo_result_module": V841_GWDO_RESULT_MODULE,
            "gwdo_result_class": V841_GWDO_RESULT_CLASS,
            "gwdo_contract_sha256": V841_GWDO_CONTRACT_SHA256,
            "gwdo_kernel_sha256": V841_GWDO_KERNEL_SHA256,
            "gwdo_input_du_is_arwen_output": True,
            "gwdo_input_dv_is_arwen_output": True,
            "raw_du_is_gwdo_output": True,
            "raw_dv_is_gwdo_output": True,
        }
        for name, required in exact_on.items():
            actual = getattr(self, name)
            if type(required) is bool:
                matches = type(actual) is bool and actual is required
            else:
                matches = actual == required
            if not matches:
                raise ValueError(
                    f"external GWDO provenance {name} changed: "
                    f"{actual!r} != {required!r}"
                )
        validation = self.gwdo_validation_d2h
        if not isinstance(validation, TransferStats):
            raise TypeError("external GWDO provenance requires TransferStats validation")
        if validation.bytes != 4:
            raise ValueError("external GWDO provenance requires exact four-byte validation")
        if not math.isfinite(validation.seconds) or validation.seconds < 0.0:
            raise ValueError("external GWDO validation duration must be finite and non-negative")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CUDA_PHASE_ONE_EXECUTION_PROVENANCE_V841_SCHEMA,
            "mode": self.mode,
            "arwen_commit": self.arwen_commit,
            "aggregate_factory": self.aggregate_factory,
            "phase1_orchestration": self.phase1_orchestration,
            "source_manifest": [
                {"path": relative, "sha256": digest}
                for relative, digest in self.source_manifest
            ],
            "aggregate_executed": self.aggregate_executed,
            "h_diabatic_applied": self.h_diabatic_applied,
            "gwd_selector": self.gwd_selector,
            "gwdo_composer": self.gwdo_composer,
            "gwdo_composition_phase": self.gwdo_composition_phase,
            "gwdo_executed": self.gwdo_executed,
            "gwdo_composed_once": self.gwdo_composed_once,
            "gwdo_result_module": self.gwdo_result_module,
            "gwdo_result_class": self.gwdo_result_class,
            "gwdo_contract_sha256": self.gwdo_contract_sha256,
            "gwdo_kernel_sha256": self.gwdo_kernel_sha256,
            "gwdo_validation_d2h": (
                None
                if self.gwdo_validation_d2h is None
                else self.gwdo_validation_d2h.as_dict()
            ),
            "gwdo_input_du_is_arwen_output": self.gwdo_input_du_is_arwen_output,
            "gwdo_input_dv_is_arwen_output": self.gwdo_input_dv_is_arwen_output,
            "raw_du_is_gwdo_output": self.raw_du_is_gwdo_output,
            "raw_dv_is_gwdo_output": self.raw_dv_is_gwdo_output,
        }


@dataclass(frozen=True, slots=True)
class CudaRawColumnPhysicsV841:
    """Uncoupled Arwen A-grid rates held once per MPAS model step."""

    du: Any
    dv: Any
    dtheta: Any
    dscalars: Mapping[str, Any]
    time_seconds: float
    execution_provenance: CudaPhaseOneExecutionProvenanceV841

    def __post_init__(self) -> None:
        canonical: dict[str, Any] = {}
        for raw_name, value in self.dscalars.items():
            name = str(raw_name).strip().lower()
            if not name or name in canonical:
                raise ValueError(f"duplicate or empty scalar rate name {raw_name!r}")
            canonical[name] = value
        if set(canonical) != set(WSM6_SCALAR_NAMES):
            missing = sorted(set(WSM6_SCALAR_NAMES) - set(canonical))
            extra = sorted(set(canonical) - set(WSM6_SCALAR_NAMES))
            raise ValueError(
                f"raw physics must return exactly WSM6 scalars; missing={missing}, extra={extra}"
            )
        object.__setattr__(self, "dscalars", MappingProxyType(canonical))
        _valid_clock(self.time_seconds, "raw physics time_seconds")
        if not isinstance(
            self.execution_provenance, CudaPhaseOneExecutionProvenanceV841
        ):
            raise TypeError(
                "raw physics requires CudaPhaseOneExecutionProvenanceV841"
            )
        self.execution_provenance.validate()

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in ("du", "dv", "dtheta"):
            require_resident_array(
                f"raw_physics.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=shape,
            )
        for name, value in self.dscalars.items():
            require_resident_array(
                f"raw_physics.dscalars[{name!r}]",
                value,
                dtype=np.float32,
                shape=shape,
            )


@dataclass(frozen=True, slots=True)
class CudaHeldMpasPhysicsTendenciesV841:
    """Conservative physics tendencies retained through every RK stage."""

    rho: Any
    rho_u: Any
    rho_theta: Any
    scalars: Any
    time_seconds: float
    validation_d2h: TransferStats
    execution_provenance: CudaPhaseOneExecutionProvenanceV841

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def validate(
        self,
        *,
        n_vert_levels: int,
        n_cells: int,
        n_edges: int,
        n_scalars: int,
    ) -> None:
        _valid_clock(self.time_seconds, "held physics time_seconds")
        if not isinstance(
            self.execution_provenance, CudaPhaseOneExecutionProvenanceV841
        ):
            raise TypeError(
                "held physics requires CudaPhaseOneExecutionProvenanceV841"
            )
        self.execution_provenance.validate()
        if not isinstance(self.validation_d2h, TransferStats):
            raise TypeError("held physics validation_d2h must be TransferStats")
        if self.validation_d2h.bytes != 4:
            raise ValueError("held physics coupling validation must transfer four bytes")
        for name, value, shape in (
            ("rho", self.rho, (n_vert_levels, n_cells)),
            ("rho_u", self.rho_u, (n_vert_levels, n_edges)),
            ("rho_theta", self.rho_theta, (n_vert_levels, n_cells)),
            ("scalars", self.scalars, (n_scalars, n_vert_levels, n_cells)),
        ):
            require_resident_array(
                f"held_physics.{name}", value, dtype=np.float32, shape=shape
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_PHYSICS_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "kernel_sha256": CUDA_PHYSICS_V841_KERNEL_SHA256,
            "phase": "pre_rk_held",
            "time_seconds": float(self.time_seconds),
            "shape": {
                "rho": list(self.rho.shape),
                "rho_u": list(self.rho_u.shape),
                "rho_theta": list(self.rho_theta.shape),
                "scalars": list(self.scalars.shape),
            },
            "validation_d2h": self.validation_d2h.as_dict(),
            "execution_provenance": self.execution_provenance.receipt(),
            "authority": physics_contract_evidence_v841()["phases"]["coupling"],
        }


@dataclass(frozen=True, slots=True)
class CudaPostRkWsm6UpdateV841:
    """Direct WSM6 state returned after RK; these are not RK tendencies."""

    theta: Any
    qv: Any
    qc: Any
    qr: Any
    qi: Any
    qs: Any
    qg: Any
    rainnc: Any
    rainncv: Any
    snownc: Any
    snowncv: Any
    graupelnc: Any
    graupelncv: Any
    sr: Any
    effc: Any
    effi: Any
    effs: Any
    time_seconds: float

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        _valid_clock(self.time_seconds, "post-RK WSM6 time_seconds")
        volume_shape = (n_vert_levels, n_cells)
        for name in ("theta", *WSM6_SCALAR_NAMES, *WSM6_RADIUS_NAMES):
            require_resident_array(
                f"post_rk_wsm6.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=volume_shape,
            )
        for name in WSM6_SURFACE_NAMES:
            require_resident_array(
                f"post_rk_wsm6.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(n_cells,),
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_PHYSICS_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "phase": "post_rk_microphysics",
            "scheme": "WSM6",
            "time_seconds": float(self.time_seconds),
            "volume_shape": list(self.theta.shape),
            "surface_shape": list(self.rainnc.shape),
            "authority": physics_contract_evidence_v841()["phases"]["post_rk"],
        }


@dataclass(frozen=True, slots=True)
class CudaPostRkWsm6RecoveryV841:
    """Validated resident MPAS state plus WSM6 persistent output carriers."""

    state: Any
    surface_updates: Mapping[str, Any]
    effective_radii: Mapping[str, Any]
    time_seconds: float
    validation_d2h: TransferStats

    @property
    def contract_sha256(self) -> str:
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_PHYSICS_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "kernel_sha256": CUDA_PHYSICS_V841_KERNEL_SHA256,
            "phase": "post_rk_recovered",
            "time_seconds": float(self.time_seconds),
            "surface_fields": sorted(self.surface_updates),
            "effective_radii": sorted(self.effective_radii),
            "validation_d2h": self.validation_d2h.as_dict(),
        }


@runtime_checkable
class PersistentTwoPhaseCudaPhysicsBackendV841(Protocol):
    """Persistent aggregate backend required by the two-phase MPAS driver."""

    @property
    def contract_sha256(self) -> str: ...

    def begin_step(
        self, *, atmosphere: Any, scalar_names: Sequence[str], dt: float
    ) -> CudaRawColumnPhysicsV841: ...

    def finish_step(
        self, *, atmosphere: Any, scalar_names: Sequence[str], dt: float
    ) -> CudaPostRkWsm6UpdateV841: ...

    def restart_state(self) -> Mapping[str, Any]: ...

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None: ...

    def step_receipt(self) -> Mapping[str, Any]: ...


def couple_raw_column_physics_v841(
    raw: CudaRawColumnPhysicsV841,
    *,
    state: Any,
    scalar_names: Sequence[str],
    geometry: CudaPhysicsGeometryV841,
    rho_edge: Any,
    kernel_cache: KernelCache,
) -> CudaHeldMpasPhysicsTendenciesV841:
    """Build one validated conservative source with pinned RawKernels only."""

    cp = _cp()
    geometry.validate()
    n_vert_levels, n_cells = tuple(state.rho.shape)
    n_edges = geometry.n_edges
    if n_cells != geometry.n_cells:
        raise ValueError(
            f"state carries {n_cells} cells but geometry carries {geometry.n_cells}"
        )
    state.validate(n_vert_levels=n_vert_levels, n_cells=n_cells, n_edges=n_edges)
    if np.dtype(state.dtype) != np.dtype(np.float32):
        raise TypeError("v8.4.1 CUDA physics coupling requires FP32 state")
    names = _scalar_names(scalar_names, int(state.scalars.shape[0]))
    if names != WSM6_SCALAR_NAMES:
        raise ValueError(
            f"full WSM6 seam requires scalar order {WSM6_SCALAR_NAMES}, got {names}"
        )
    if float(raw.time_seconds) != float(state.time_seconds):
        raise ValueError(
            "raw physics time must equal the candidate step start exactly: "
            f"{raw.time_seconds} != {state.time_seconds}"
        )
    raw.validate(n_vert_levels=n_vert_levels, n_cells=n_cells)
    require_resident_array(
        "physics.rho_edge",
        rho_edge,
        dtype=np.float32,
        shape=(n_vert_levels, n_edges),
    )
    rho = cp.empty_like(state.rho)
    rho_theta = cp.empty_like(state.rho_theta)
    rho_u = cp.empty_like(state.rho_u)
    scalars = cp.empty_like(state.scalars)
    invalid = cp.zeros((1,), dtype=cp.int32)
    _launch(
        kernel_cache,
        "couple_cells_v841_f32",
        n_cells,
        (
            np.int32(n_vert_levels),
            np.int32(n_cells),
            RV_OVER_RD_F32,
            state.rho,
            state.rho_theta,
            state.scalars,
            raw.dtheta,
            raw.dscalars["qv"],
            raw.dscalars["qc"],
            raw.dscalars["qr"],
            raw.dscalars["qi"],
            raw.dscalars["qs"],
            raw.dscalars["qg"],
            rho,
            rho_theta,
            scalars,
            invalid,
        ),
    )
    if geometry.regional_garbage_cell is not None:
        # Native's physics tendency at the garbage element is ZERO, because
        # native's physics loop runs 1..nCellsSolve and never writes it.  The
        # preparation lends that column a real sounding so the schemes can
        # integrate it at all (see cuda_physics_prep_v841), which means the
        # tendency that comes back is a real column's -- and the projection
        # below is the ONE place a real element reads it: a ring-7 one-cell
        # edge has the garbage cell as its second endpoint, so it would take
        # half of some unrelated cell's momentum tendency.
        #
        # MEASURED on r4.75.11020 (2026-08-26), 556 such edges: the
        # limited-area full-physics forecast committed three steps and
        # refused the fourth with |w| = 267.4 m/s against the 200 m/s
        # divergence refusal.  Zeroing here restores native's own value at
        # that element, so a one-cell edge takes half of its real cell's
        # tendency and nothing else -- which is exactly what native computes.
        #
        # The zero is held only across the projection and then given back:
        # ``raw.du``/``raw.dv`` are the seam's own output buffers and the
        # seam carries them, so writing into them permanently would edit a
        # component this lane does not own.
        garbage_cell = int(geometry.regional_garbage_cell)
        held_du = raw.du[..., garbage_cell].copy()
        held_dv = raw.dv[..., garbage_cell].copy()
        raw.du[..., garbage_cell] = raw.du.dtype.type(0.0)
        raw.dv[..., garbage_cell] = raw.dv.dtype.type(0.0)
    else:
        garbage_cell = None
    _launch(
        kernel_cache,
        "project_edges_v841_f32",
        n_edges,
        (
            np.int32(n_vert_levels),
            np.int32(n_cells),
            np.int32(n_edges),
            geometry.cells_on_edge,
            geometry.east_cell,
            geometry.north_cell,
            geometry.edge_normal_vectors,
            raw.du,
            raw.dv,
            rho_edge,
            rho_u,
            invalid,
        ),
    )
    if garbage_cell is not None:
        raw.du[..., garbage_cell] = held_du
        raw.dv[..., garbage_cell] = held_dv
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    validation_d2h = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        # A refusal has to name what it refused.  The two kernels above set a
        # single flag between them, so the flag alone says "some field, some
        # column, some level".  This walks the inputs and the outputs on the
        # host and names the first field that breaks each kernel's law -- and
        # it runs ONLY on the failure path, so a passing step pays nothing.
        candidates: dict[str, Any] = {
            "state.rho (must be > 0)": state.rho,
            "state.rho_theta": state.rho_theta,
            "state.scalars": state.scalars,
            "raw.dtheta": raw.dtheta,
            "raw.du": raw.du,
            "raw.dv": raw.dv,
            "rho_edge": rho_edge,
            "coupled.rho_theta": rho_theta,
            "coupled.rho_u": rho_u,
            "coupled.scalars": scalars,
        }
        for species, rate in raw.dscalars.items():
            candidates[f"raw.d{species}"] = rate
        findings: list[str] = []
        for name, array in candidates.items():
            host = cp.asnumpy(array)
            bad = ~np.isfinite(host)
            if name.startswith("state.rho (") :
                bad = bad | (host <= np.float32(0.0))
            count = int(np.count_nonzero(bad))
            if count == 0:
                continue
            last = np.unique(np.argwhere(bad)[:, -1])
            findings.append(
                f"{name}: {count} value(s) over {last.size} column(s)/edge(s), "
                f"first {int(last[0])} of {int(host.shape[-1])}"
            )
        detail = "; ".join(findings) if findings else (
            "every input and output is finite on the host, so the refusal is "
            "one of the kernels' consistency laws rather than a value"
        )
        raise FloatingPointError(
            "non-finite/invalid raw physics or conservative coupling result: "
            + detail
        )
    result = CudaHeldMpasPhysicsTendenciesV841(
        rho=rho,
        rho_u=rho_u,
        rho_theta=rho_theta,
        scalars=scalars,
        time_seconds=float(raw.time_seconds),
        validation_d2h=validation_d2h,
        execution_provenance=raw.execution_provenance,
    )
    result.validate(
        n_vert_levels=n_vert_levels,
        n_cells=n_cells,
        n_edges=n_edges,
        n_scalars=len(names),
    )
    return result


def nonnegative_qv_scratch_v841(
    scalars: Any,
    *,
    scalar_names: Sequence[str],
    kernel_cache: KernelCache,
) -> tuple[Any, TransferStats]:
    """Return max(qv, +0) scratch for phase one without mutating MPAS state."""

    cp = _cp()
    names = _scalar_names(scalar_names, int(scalars.shape[0]))
    if names != WSM6_SCALAR_NAMES:
        raise ValueError(f"WSM6 scalar order must be {WSM6_SCALAR_NAMES}")
    require_resident_array("state.scalars", scalars, dtype=np.float32, ndim=3)
    qv = cp.empty(tuple(scalars.shape[1:]), dtype=cp.float32)
    invalid = cp.zeros((1,), dtype=cp.int32)
    _launch(
        kernel_cache,
        "nonnegative_qv_copy_v841_f32",
        int(qv.size),
        (np.int32(qv.size), scalars, qv, invalid),
    )
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    transfer = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        raise FloatingPointError("non-finite qv refused before aggregate phase one")
    return qv, transfer


def clamp_wsm6_scalars_in_place_v841(
    scalars: Any,
    *,
    scalar_names: Sequence[str],
    kernel_cache: KernelCache,
) -> TransferStats:
    """Apply native post-transport +0 clamp to all six WSM6 carriers."""

    cp = _cp()
    names = _scalar_names(scalar_names, int(scalars.shape[0]))
    if names != WSM6_SCALAR_NAMES:
        raise ValueError(f"WSM6 scalar order must be {WSM6_SCALAR_NAMES}")
    require_resident_array("state.scalars", scalars, dtype=np.float32, ndim=3)
    invalid = cp.zeros((1,), dtype=cp.int32)
    # Validate in a separate pass so a refusal cannot partially clamp the live
    # prognostic buffer. The second pass is safe only after the validation
    # word has crossed the transaction boundary successfully.
    _launch(
        kernel_cache,
        "clamp_wsm6_v841_f32",
        int(scalars.size),
        (np.int32(scalars.size), np.int32(0), scalars, invalid),
    )
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    transfer = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        raise FloatingPointError("non-finite WSM6 carrier refused before phase two")
    _launch(
        kernel_cache,
        "clamp_wsm6_v841_f32",
        int(scalars.size),
        (np.int32(scalars.size), np.int32(1), scalars, invalid),
    )
    return transfer


def recover_post_rk_wsm6_state_v841(
    state: Any,
    update: CudaPostRkWsm6UpdateV841,
    *,
    scalar_names: Sequence[str],
    kernel_cache: KernelCache,
    phase2_dt_seconds: float,
    previous_surface_updates: Mapping[str, Any] | None = None,
) -> CudaPostRkWsm6RecoveryV841:
    """Validate candidate buffers completely, then commit live state once."""

    cp = _cp()
    n_vert_levels, n_cells = tuple(state.rho.shape)
    n_edges = int(state.rho_u.shape[1])
    state.validate(n_vert_levels=n_vert_levels, n_cells=n_cells, n_edges=n_edges)
    if np.dtype(state.dtype) != np.dtype(np.float32):
        raise TypeError("post-RK WSM6 recovery requires FP32 MPAS state")
    sr_upper, _, _ = _wsm6_sr_roundoff_limit_v841(phase2_dt_seconds)
    names = _scalar_names(scalar_names, int(state.scalars.shape[0]))
    if names != WSM6_SCALAR_NAMES:
        raise ValueError(
            f"WSM6 recovery requires scalar order {WSM6_SCALAR_NAMES}, got {names}"
        )
    if float(update.time_seconds) != float(state.time_seconds):
        raise ValueError(
            "WSM6 update time must equal the candidate endpoint exactly: "
            f"{update.time_seconds} != {state.time_seconds}"
        )
    update.validate(n_vert_levels=n_vert_levels, n_cells=n_cells)

    # These buffers are private to this call.  No live prognostic byte is
    # written until both volume/radius and surface validation have passed.
    candidate_rho_theta = cp.empty_like(state.rho_theta)
    candidate_scalars = cp.empty_like(state.scalars)
    invalid = cp.zeros((1,), dtype=cp.int32)
    count = n_vert_levels * n_cells
    _launch(
        kernel_cache,
        "recover_wsm6_v841_f32",
        count,
        (
            np.int32(count),
            RV_OVER_RD_F32,
            state.rho,
            update.theta,
            update.qv,
            update.qc,
            update.qr,
            update.qi,
            update.qs,
            update.qg,
            update.effc,
            update.effi,
            update.effs,
            candidate_rho_theta,
            candidate_scalars,
            invalid,
        ),
    )
    previous = {} if previous_surface_updates is None else previous_surface_updates
    old_rain = previous.get("rainnc", update.rainnc)
    old_snow = previous.get("snownc", update.snownc)
    old_graupel = previous.get("graupelnc", update.graupelnc)
    for name, value in (
        ("rainnc", old_rain),
        ("snownc", old_snow),
        ("graupelnc", old_graupel),
    ):
        require_resident_array(
            f"previous_wsm6_surface.{name}",
            value,
            dtype=np.float32,
            shape=(n_cells,),
        )
    _launch(
        kernel_cache,
        "validate_wsm6_surface_v841_f32",
        n_cells,
        (
            np.int32(n_cells),
            sr_upper,
            update.rainnc,
            update.rainncv,
            update.snownc,
            update.snowncv,
            update.graupelnc,
            update.graupelncv,
            update.sr,
            old_rain,
            old_snow,
            old_graupel,
            invalid,
        ),
    )
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    transfer = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        raise FloatingPointError(
            "post-RK WSM6 numeric validation refused the candidate state"
        )

    _launch(
        kernel_cache,
        "commit_wsm6_state_v841_f32",
        count,
        (
            np.int32(count),
            candidate_rho_theta,
            candidate_scalars,
            state.rho_theta,
            state.scalars,
        ),
    )
    surface = MappingProxyType(
        {name: getattr(update, name) for name in WSM6_SURFACE_NAMES}
    )
    radii = MappingProxyType(
        {name: getattr(update, name) for name in WSM6_RADIUS_NAMES}
    )
    return CudaPostRkWsm6RecoveryV841(
        state=state,
        surface_updates=surface,
        effective_radii=radii,
        time_seconds=float(update.time_seconds),
        validation_d2h=transfer,
    )


__all__ = [
    "CUDA_PHASE_ONE_EXECUTION_PROVENANCE_V841_SCHEMA",
    "CUDA_PHYSICS_V841_CONTRACT_SHA256",
    "CUDA_PHYSICS_V841_EVIDENCE",
    "CUDA_PHYSICS_V841_KERNEL_SHA256",
    "CUDA_PHYSICS_V841_SCHEMA",
    "CudaHeldMpasPhysicsTendenciesV841",
    "CudaPhaseOneExecutionProvenanceV841",
    "CudaPhysicsGeometryV841",
    "CudaPostRkWsm6RecoveryV841",
    "CudaPostRkWsm6UpdateV841",
    "CudaRawColumnPhysicsV841",
    "PersistentTwoPhaseCudaPhysicsBackendV841",
    "RV_OVER_RD_F32",
    "WSM6_RADIUS_NAMES",
    "WSM6_SCALAR_NAMES",
    "WSM6_SURFACE_NAMES",
    "clamp_wsm6_scalars_in_place_v841",
    "couple_raw_column_physics_v841",
    "nonnegative_qv_scratch_v841",
    "physics_contract_evidence_v841",
    "recover_post_rk_wsm6_state_v841",
]
_CUDA_SOURCE = (
    CUDA_FTZ_HELPERS
    + r"""
extern "C" __global__ void couple_cells_v841_f32(
    const int nlev, const int ncells, const float rvord,
    const float *rho, const float *rho_theta, const float *q,
    const float *dtheta, const float *dqv, const float *dqc,
    const float *dqr, const float *dqi, const float *dqs,
    const float *dqg, float *out_rho, float *out_rtheta,
    float *out_q, int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    const float *rates[6] = {dqv, dqc, dqr, dqi, dqs, dqg};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float mass = rho[i];
        const float rt = rho_theta[i];
        const float qt = q[i];
        const float th_rate = dtheta[i];
        if (!isfinite(mass) || mass <= 0.0f || !isfinite(rt)
                || !isfinite(qt) || !isfinite(th_rate)) atomicExch(invalid, 1);
        out_rho[i] = 0.0f;
        for (int s = 0; s < 6; ++s) {
            const float rate = rates[s][i];
            if (!isfinite(rate)) atomicExch(invalid, 1);
            out_q[s * plane + i] = mpas_mul(mass, rate);
        }
        const float coeff = mpas_add(1.0f, mpas_mul(rvord, qt));
        const float theta_m = mpas_div(rt, mass);
        const float tend_th = mpas_mul(mass, th_rate);
        const float first = mpas_mul(coeff, tend_th);
        const float vapor = mpas_div(
            mpas_mul(mpas_mul(rvord, theta_m), out_q[i]), coeff);
        const float result = mpas_add(first, vapor);
        if (!isfinite(coeff) || coeff <= 0.0f || !isfinite(result))
            atomicExch(invalid, 1);
        out_rtheta[i] = result;
    }
}

extern "C" __global__ void project_edges_v841_f32(
    const int nlev, const int ncells, const int nedges, const int *cells,
    const float *east, const float *north, const float *normal,
    const float *du, const float *dv, const float *rho_edge,
    float *out_ru, int *invalid)
{
    const int edge = blockDim.x * blockIdx.x + threadIdx.x;
    if (edge >= nedges) return;
    const int c1 = cells[2 * edge];
    const int c2 = cells[2 * edge + 1];
    float e1 = mpas_mul(normal[3*edge], east[3*c1]);
    e1 = mpas_add(e1, mpas_mul(normal[3*edge+1], east[3*c1+1]));
    e1 = mpas_add(e1, mpas_mul(normal[3*edge+2], east[3*c1+2]));
    float n1 = mpas_mul(normal[3*edge], north[3*c1]);
    n1 = mpas_add(n1, mpas_mul(normal[3*edge+1], north[3*c1+1]));
    n1 = mpas_add(n1, mpas_mul(normal[3*edge+2], north[3*c1+2]));
    float e2 = mpas_mul(normal[3*edge], east[3*c2]);
    e2 = mpas_add(e2, mpas_mul(normal[3*edge+1], east[3*c2+1]));
    e2 = mpas_add(e2, mpas_mul(normal[3*edge+2], east[3*c2+2]));
    float n2 = mpas_mul(normal[3*edge], north[3*c2]);
    n2 = mpas_add(n2, mpas_mul(normal[3*edge+1], north[3*c2+1]));
    n2 = mpas_add(n2, mpas_mul(normal[3*edge+2], north[3*c2+2]));
    for (int k = 0; k < nlev; ++k) {
        const int ei = k * nedges + edge;
        const int c1i = k * ncells + c1;
        const int c2i = k * ncells + c2;
        float value = mpas_mul(mpas_mul(du[c1i], 0.5f), e1);
        value = mpas_add(value, mpas_mul(mpas_mul(dv[c1i], 0.5f), n1));
        value = mpas_add(value, mpas_mul(mpas_mul(du[c2i], 0.5f), e2));
        value = mpas_add(value, mpas_mul(mpas_mul(dv[c2i], 0.5f), n2));
        value = mpas_mul(value, rho_edge[ei]);
        if (!isfinite(value) || !isfinite(rho_edge[ei])) atomicExch(invalid, 1);
        out_ru[ei] = value;
    }
}

extern "C" __global__ void nonnegative_qv_copy_v841_f32(
    const int count, const float *input, float *output, int *invalid)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    const float value = input[i];
    if (!isfinite(value)) atomicExch(invalid, 1);
    output[i] = value < 0.0f ? 0.0f : value;
}

extern "C" __global__ void clamp_wsm6_v841_f32(
    const int count, const int apply_clamp, float *values, int *invalid)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    const unsigned int bits = __float_as_uint(values[i]);
    const unsigned int magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        atomicExch(invalid, 1);
        return;
    }
    // Floating comparisons are DAZ-sensitive on the certified terminal-FTZ
    // route. Inspect the stored sign bit so -minsub and -0 both become the
    // native +0 bit pattern while every positive subnormal survives exactly.
    if (apply_clamp != 0 && (bits & 0x80000000u) != 0u)
        values[i] = __uint_as_float(0u);
}

extern "C" __global__ void recover_wsm6_v841_f32(
    const int count, const float rvord, const float *rho, const float *theta,
    const float *qv, const float *qc, const float *qr, const float *qi,
    const float *qs, const float *qg, const float *effc, const float *effi,
    const float *effs, float *rho_theta, float *scalars, int *invalid)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    const float *water[6] = {qv, qc, qr, qi, qs, qg};
    const float *radius[3] = {effc, effi, effs};
    if (!isfinite(rho[i]) || rho[i] <= 0.0f
            || !isfinite(theta[i]) || theta[i] <= 0.0f) atomicExch(invalid, 1);
    for (int s = 0; s < 6; ++s) {
        const float value = water[s][i];
        // MPAS mpas_atmphys_interface.F:914 writes the post-microphysics
        // carriers back to scalars RAW -- no clamp, no refusal.  The +0 clamp
        // native applies is the POST-TRANSPORT one, reproduced by
        // clamp_wsm6_v841_f32 at the next step boundary.  Refusing a negative
        // here is stricter than the authority; only non-finite is refused.
        if (!isfinite(value)) atomicExch(invalid, 1);
        scalars[s * count + i] = value;
    }
    for (int r = 0; r < 3; ++r) {
        const float value = radius[r][i];
        if (!isfinite(value) || value < 0.0f) atomicExch(invalid, 1);
    }
    // MPAS couples theta_m with max(0.,qv) AT THE POINT OF USE, in this exact
    // expression: mpas_atmphys_interface.F:665, :778 and :1078 (the last is the
    // POST-microphysics coupling this kernel implements).  The sign-bit form is
    // used instead of fmaxf because the certified terminal-FTZ route is
    // DAZ-sensitive: it maps -0 and every negative subnormal to the native +0
    // bit pattern while leaving positive subnormals exactly alone.
    const unsigned int qv_bits = __float_as_uint(qv[i]);
    const float qv_coupled = (qv_bits & 0x80000000u) != 0u ? 0.0f : qv[i];
    const float coeff = mpas_add(1.0f, mpas_mul(rvord, qv_coupled));
    const float result = mpas_mul(rho[i], mpas_mul(theta[i], coeff));
    if (!isfinite(coeff) || coeff <= 0.0f || !isfinite(result))
        atomicExch(invalid, 1);
    rho_theta[i] = result;
}

extern "C" __global__ void commit_wsm6_state_v841_f32(
    const int count, const float *candidate_rho_theta,
    const float *candidate_scalars, float *rho_theta, float *scalars)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    rho_theta[i] = candidate_rho_theta[i];
    for (int s = 0; s < 6; ++s) {
        scalars[s * count + i] = candidate_scalars[s * count + i];
    }
}
extern "C" __global__ void validate_wsm6_surface_v841_f32(
    const int count, const float sr_upper, const float *rainnc,
    const float *rainncv, const float *snownc, const float *snowncv,
    const float *graupelnc,
    const float *graupelncv, const float *sr, const float *old_rainnc,
    const float *old_snownc, const float *old_graupelnc, int *invalid)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;
    const float acc[3] = {rainnc[i], snownc[i], graupelnc[i]};
    const float inc[3] = {rainncv[i], snowncv[i], graupelncv[i]};
    const float old[3] = {old_rainnc[i], old_snownc[i], old_graupelnc[i]};
    for (int j = 0; j < 3; ++j) {
        if (!isfinite(acc[j]) || acc[j] < 0.0f || acc[j] < old[j]
                || !isfinite(inc[j]) || inc[j] < 0.0f) atomicExch(invalid, 1);
    }
    if (!isfinite(sr[i]) || sr[i] < 0.0f || sr[i] > sr_upper)
        atomicExch(invalid, 1);
}
"""
)

CUDA_PHYSICS_V841_KERNEL_SHA256 = sha256(_CUDA_SOURCE.encode("utf-8")).hexdigest()

_KERNEL_NAMES = frozenset(
    {
        "couple_cells_v841_f32",
        "project_edges_v841_f32",
        "nonnegative_qv_copy_v841_f32",
        "clamp_wsm6_v841_f32",
        "commit_wsm6_state_v841_f32",
        "recover_wsm6_v841_f32",
        "validate_wsm6_surface_v841_f32",
    }
)
_KERNELS: dict[tuple[int, str], Any] = {}


_REQUIRED_KERNEL_CACHE_OPTIONS = ("--std=c++17", "--fmad=false")


def _require_kernel_cache_contract(cache: KernelCache) -> None:
    if not isinstance(cache, KernelCache):
        raise TypeError("kernel_cache must be a KernelCache")
    options = tuple(cache.base_options)
    if options != _REQUIRED_KERNEL_CACHE_OPTIONS:
        raise TypeError(
            "v8.4.1 physics KernelCache base_options must be exactly "
            f"{_REQUIRED_KERNEL_CACHE_OPTIONS}; got {options}"
        )
    manifest = cache.compile_manifest()
    platform = (
        manifest.get("compile_platform") if isinstance(manifest, Mapping) else None
    )
    fingerprint = platform.get("fingerprint") if isinstance(platform, Mapping) else None
    digest = platform.get("sha256") if isinstance(platform, Mapping) else None
    try:
        validated_fingerprint = validate_compile_platform_fingerprint(fingerprint)
    except CompileContractError as error:
        raise ValueError(
            "v8.4.1 physics KernelCache lacks its validated FTZ "
            "compile-platform binding"
        ) from error
    if not isinstance(digest, str) or digest != canonical_sha256(validated_fingerprint):
        raise ValueError(
            "v8.4.1 physics KernelCache lacks its validated FTZ compile-platform binding"
        )


def _kernel(cache: KernelCache, name: str) -> Any:
    _require_kernel_cache_contract(cache)
    if name not in _KERNEL_NAMES:
        raise KeyError(name)
    key = (id(cache), name)
    result = _KERNELS.get(key)
    if result is None:
        result = cache.raw_kernel(
            name, _CUDA_SOURCE, module_key="hexcore.cuda_physics_v841"
        )
        _KERNELS[key] = result
    return result


def _launch(cache: KernelCache, name: str, count: int, args: tuple[Any, ...]) -> None:
    threads = 128
    _kernel(cache, name)(((count + threads - 1) // threads,), (threads,), args)
