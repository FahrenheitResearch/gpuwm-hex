"""Resident CUDA YSU orographic gravity-wave drag for MPAS-A v8.4.1.

This module is the native ``bl_ysu_gwdo`` column operator that follows YSU in
the released MPAS physics driver. It is deliberately separate from Arwen:
the aggregate Arwen contract has no gravity-wave-drag slot. The source path
is complete and table-free, so there is no surrogate or leaf-kernel shortcut.

The numerical authority is the frozen official MPAS-Model v8.4.1 build:

* ``mpas_atmphys_driver_gwdo.F:285-552,692-834`` prepares MPAS carriers and
  couples the result;
* ``module_bl_gwdo.F:15-237`` maps the MPAS column to ``bl_gwdo_run``; and
* ``physics_mmm/bl_gwdo.F90:55-643`` is the complete Kim/YSU orographic GWD
  and flow-blocking implementation.

All forecast inputs and outputs are resident packed FP32 arrays. Layer fields
are C ``[level, column]`` and interface pressure is C ``[interface, column]``;
``level == 0`` is the lowest model layer. Authority arithmetic is confined
to one explicit RawKernel. The kernel writes candidate outputs, performs one
final four-byte validation transfer, and never mutates its inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from .cuda_backend.compile_contract import (
    CompileContractError,
    canonical_sha256,
    validate_compile_platform_fingerprint,
)
from .cuda_backend.containers import TransferStats, require_resident_array
from .cuda_backend.runtime import KernelCache
from .cuda_fp32 import CUDA_FTZ_HELPERS
from .cuda_physics_prep_v841 import (
    CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    CudaMpasToPhysColumnsV841,
)


CUDA_GWDO_V841_SCHEMA = "mpas-port.cuda-gwdo-v841/v1"
MPAS_ATMPHYS_DRIVER_GWDO_V841_SHA256 = (
    "ec76abad1cd841b3a15cb63df494060488cea1026c532a628e61e1d71871fb54"
)
MODULE_BL_GWDO_V841_SHA256 = (
    "9784f201716e026fc66c531ca0012f9dfc3098112bb7b25b3da1ec8168aa52e6"
)
BL_GWDO_V841_SHA256 = (
    "c294ef2eb9da1ee9ce6594b97adea7280eb18760e0c5576ab7149c35bb7c3c3b"
)
REGISTRY_CORE_ATMOSPHERE_V841_SHA256 = (
    "0731a09dd5c258cc510edfa90fbcf963e126e2f5ae1d09ae7cbe9e3b8eecde28"
)

GRAVITY_F32 = np.float32(9.80616)
RD_F32 = np.float32(287.0)
RV_F32 = np.float32(461.6)
CP_F32 = np.float32(np.float32(7.0) * RD_F32 / np.float32(2.0))
EP1_F32 = np.float32(RV_F32 / RD_F32 - np.float32(1.0))
PI_F32 = np.float32(3.141592653589793)
X4_N_VERT_LEVELS = 55
X4_DT_SECONDS_F32 = np.float32(120.0)
X4_NOMINAL_MIN_DC_M_F32 = np.float32(25_000.0)

_STATIC_FIELDS = (
    "mesh_density",
    "var2d",
    "con",
    "oa1",
    "oa2",
    "oa3",
    "oa4",
    "ol1",
    "ol2",
    "ol3",
    "ol4",
)

_AUTHORITY_DOCUMENT = {
    "schema": CUDA_GWDO_V841_SCHEMA,
    "mpas_version": "8.4.1",
    "selector": "bl_ysu_gwdo",
    "source": {
        "mpas_atmphys_driver_gwdo.F": {
            "lines": "285-552,692-834",
            "sha256": MPAS_ATMPHYS_DRIVER_GWDO_V841_SHA256,
        },
        "module_bl_gwdo.F": {
            "lines": "15-237",
            "sha256": MODULE_BL_GWDO_V841_SHA256,
        },
        "physics_mmm/bl_gwdo.F90": {
            "lines": "55-643",
            "sha256": BL_GWDO_V841_SHA256,
        },
        "Registry.core_atmosphere.xml": {
            "lines": "174-177,2371-2379,3045-3129,3853-3885",
            "sha256": REGISTRY_CORE_ATMOSPHERE_V841_SHA256,
        },
    },
    "layout": {
        "mass": "float32 C [level,column], column fastest",
        "interface": "float32 C [interface,column], column fastest",
        "surface": "float32 C [column]",
        "vertical_order": "surface_to_top; k=0 lowest",
        "x4_levels": X4_N_VERT_LEVELS,
    },
    "cadence_and_order": {
        "native_order": "revised-MO -> NoahMP -> YSU -> bl_ysu_gwdo -> convection",
        "cadence": "every PBL/physics step",
        "exact_x4_dt_seconds": 120.0,
        "persistent_state": "none",
        "restart_state": "none",
        "diagnostics": "overwritten on every call",
    },
    "inputs": {
        "dynamic": {
            "u_p": "model/earth-relative zonal wind, m s^-1; MPAS sina=0,cosa=1",
            "v_p": "model/earth-relative meridional wind, m s^-1",
            "t_p": "temperature, K, made with common nonnegative-qv scratch",
            "qv_p": "common nonnegative water-vapor mixing ratio, kg kg^-1",
            "pres_hydd_p": "dry hydrostatic layer pressure, Pa",
            "pres2_hydd_p": "dry hydrostatic interface pressure, Pa",
            "pi_p": "Exner function, unitless",
            "zmid_p": "layer height above sea level, m",
            "rublten": "existing YSU zonal tendency, m s^-2",
            "rvblten": "existing YSU meridional tendency, m s^-2",
        },
        "static": {
            "meshDensity": "unitless",
            "effective_config_len_disp": (
                "m; x4 resolves zero namelist to nominalMinDc=25000"
            ),
            "var2d": "subgrid orography standard deviation, m",
            "con": "orographic convexity, unitless",
            "oa1..oa4": "directional asymmetry, unitless",
            "ol1..ol4": "directional effective orographic length, unitless",
        },
        "syntactically_dead_native_wrapper_arguments": ["kpbl2d", "itimestep"],
    },
    "outputs": {
        "rublten": "YSU plus GWD zonal tendency, m s^-2",
        "rvblten": "YSU plus GWD meridional tendency, m s^-2",
        "dtaux3d": "GWD zonal acceleration, m s^-2",
        "dtauy3d": "GWD meridional acceleration, m s^-2",
        "dusfcg": "vertically integrated zonal surface stress, Pa",
        "dvsfcg": "vertically integrated meridional surface stress, Pa",
        "rubldiff": "rounded (rublten_after-rublten_before), m s^-2",
        "rvbldiff": "rounded (rvblten_after-rvblten_before), m s^-2",
        "thermodynamic_tendency": "none; native rthblten passes through unchanged",
    },
    "branch_scope": {
        "complete": [
            "Kim orographic wave stress",
            "critical-level and Richardson-number saturation",
            "directional anisotropy/asymmetry",
            "low-level flow blocking",
            "critical-line dt limiter",
        ],
        "external_tables": "none",
        "mpas_rotation_specialization": (
            "sina=+0,cosa=+1 exactly as driver lines 531-532"
        ),
        "x4_static_domain": {
            "meshDensity": "0 < value <= 1",
            "var2d": "value >= 0",
            "con": "value >= 0",
            "oa": "-1 <= value <= 1",
            "ol": "0 <= value <= 1",
        },
    },
    "cuda_authority": {
        "module_key": "mpas_port.cuda_gwdo_v841",
        "compile_options": ["--std=c++17", "--fmad=false"],
        "compile_options_policy": "exact ordered base_options; no overrides",
        "ftz_binding": (
            "validated compile-platform fingerprint plus CUDA_FTZ_HELPERS"
        ),
        "implicit_cupy_authority_arithmetic": False,
        "successful_d2h_bytes": 4,
        "transactional": True,
    },
    "sealed_native_carrier": {
        "scope": "x4.163842 native CPU F001 with NoahMP+YSU-GWDO",
        "forecast_closure_sha256": (
            "add2be79b44b7e357eeaa14e6fb943ad871b734d463dd1d048393024f930fd0e"
        ),
        "authority_table_resolution_sha256": (
            "62f7d27a54c9f17fdaa736fada78dc122fd13e2d4689443285b0ceee657e1c0b"
        ),
        "namelist_sha256": (
            "8a50e8f6bb09f0fb2a763c41bc2770795ed9a112ce255728ca88807ca929c6cc"
        ),
        "model_log_sha256": (
            "9f212050be3ed6565fc4a19cd076967b9cbfa931b31ecfae01544ee2d4b3329d"
        ),
        "facts": {
            "nCells": 163842,
            "nVertLevels": 55,
            "nominalMinDc_m": 25000.0,
            "dt_seconds": 120.0,
            "completed_steps": 30,
            "selector": "bl_ysu_gwdo",
        },
        "nonclaim": (
            "carrier proves native route/completion, not CUDA or tendency attribution"
        ),
    },
}

CUDA_GWDO_V841_CONTRACT_SHA256 = sha256(
    json.dumps(_AUTHORITY_DOCUMENT, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()
CUDA_GWDO_V841_EVIDENCE: Mapping[str, Any] = MappingProxyType(
    _AUTHORITY_DOCUMENT
)


def gwdo_contract_evidence_v841() -> dict[str, Any]:
    """Return a detached JSON-ready copy of the frozen GWD contract."""

    result = json.loads(json.dumps(_AUTHORITY_DOCUMENT, sort_keys=True))
    result["contract_sha256"] = CUDA_GWDO_V841_CONTRACT_SHA256
    result["kernel_sha256"] = CUDA_GWDO_V841_KERNEL_SHA256
    return result


def _cp() -> Any:
    import cupy as cp

    return cp


def _carrier_value(carriers: object, name: str, default: Any = None) -> Any:
    if hasattr(carriers, name):
        return getattr(carriers, name)
    if isinstance(carriers, Mapping) and name in carriers:
        return carriers[name]
    arrays = getattr(carriers, "arrays", None)
    if isinstance(arrays, Mapping) and name in arrays:
        return arrays[name]
    attrs = getattr(carriers, "attrs", None)
    if isinstance(attrs, Mapping) and name in attrs:
        return attrs[name]
    return default


def _host_f32_vector(
    value: Any, *, name: str, count: int | None = None
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1:
        raise TypeError(f"{name} must be a host FP32 vector")
    if count is not None and array.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(array)


_COLUMN_VIEW_SEAL = object()
_STATIC_SEAL = object()


@dataclass(frozen=True, slots=True)
class CudaYsuGwdoColumnViewV841:
    """Exact GWD aliases from a validated ``MPAS_to_physics`` result."""

    u_p: Any
    v_p: Any
    t_p: Any
    qv_p: Any
    pres_hydd_p: Any
    pres2_hydd_p: Any
    pi_p: Any
    zmid_p: Any
    time_seconds: float
    prep_contract_sha256: str
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def from_prepared(
        cls, prepared: CudaMpasToPhysColumnsV841
    ) -> "CudaYsuGwdoColumnViewV841":
        """Bind mechanically to dry-hydro, common-qv native field identities."""

        if not isinstance(prepared, CudaMpasToPhysColumnsV841):
            raise TypeError(
                "YSU-GWDO requires validated CudaMpasToPhysColumnsV841"
            )
        prepared.validate()
        result = cls(
            u_p=prepared.u_p,
            v_p=prepared.v_p,
            t_p=prepared.t_p,
            qv_p=prepared.qv_p,
            pres_hydd_p=prepared.pres_hydd_p,
            pres2_hydd_p=prepared.pres2_hydd_p,
            pi_p=prepared.pi_p,
            zmid_p=prepared.zmid_p,
            time_seconds=float(prepared.time_seconds),
            prep_contract_sha256=prepared.contract_sha256,
            _seal=_COLUMN_VIEW_SEAL,
        )
        result.validate()
        return result

    @property
    def n_vert_levels(self) -> int:
        return int(self.u_p.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.u_p.shape[1])

    def validate(self) -> None:
        if self._seal is not _COLUMN_VIEW_SEAL:
            raise TypeError("GWD column view must be constructed by from_prepared")
        if self.prep_contract_sha256 != CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256:
            raise ValueError("GWD column view is not bound to v8.4.1 physics prep")
        if not math.isfinite(self.time_seconds) or self.time_seconds < 0.0:
            raise ValueError("GWD time_seconds must be finite and non-negative")
        shape = tuple(self.u_p.shape)
        if shape[0] != X4_N_VERT_LEVELS or shape[1] <= 0:
            raise ValueError(
                f"exact x4 YSU-GWDO needs [{X4_N_VERT_LEVELS},nCells]"
            )
        for name in (
            "u_p",
            "v_p",
            "t_p",
            "qv_p",
            "pres_hydd_p",
            "pi_p",
            "zmid_p",
        ):
            require_resident_array(
                f"gwdo_columns.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=shape,
            )
        require_resident_array(
            "gwdo_columns.pres2_hydd_p",
            self.pres2_hydd_p,
            dtype=np.float32,
            shape=(X4_N_VERT_LEVELS + 1, shape[1]),
        )


@dataclass(frozen=True, slots=True)
class CudaYsuGwdoStaticV841:
    """Resident x4 mesh/orography carriers used by every GWD call."""

    mesh_density: Any
    var2d: Any
    con: Any
    oa1: Any
    oa2: Any
    oa3: Any
    oa4: Any
    ol1: Any
    ol2: Any
    ol3: Any
    ol4: Any
    effective_length_scale_m: float
    n_cells: int
    h2d: TransferStats
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def from_resident(
        cls,
        *,
        mesh_density: Any,
        var2d: Any,
        con: Any,
        oa1: Any,
        oa2: Any,
        oa3: Any,
        oa4: Any,
        ol1: Any,
        ol2: Any,
        ol3: Any,
        ol4: Any,
        effective_length_scale_m: float,
    ) -> "CudaYsuGwdoStaticV841":
        """Seal already resident static carriers without copying them."""

        n_cells = int(mesh_density.shape[0])
        result = cls(
            mesh_density=mesh_density,
            var2d=var2d,
            con=con,
            oa1=oa1,
            oa2=oa2,
            oa3=oa3,
            oa4=oa4,
            ol1=ol1,
            ol2=ol2,
            ol3=ol3,
            ol4=ol4,
            effective_length_scale_m=float(effective_length_scale_m),
            n_cells=n_cells,
            h2d=TransferStats(0, 0.0),
            _seal=_STATIC_SEAL,
        )
        result.validate()
        return result

    @classmethod
    def from_host(
        cls,
        carriers: object,
        *,
        config_len_disp_m: float = 0.0,
    ) -> "CudaYsuGwdoStaticV841":
        """Validate and upload official x4 static carriers exactly once.

        Native MPAS resolves a non-positive ``config_len_disp`` to the input
        file's ``nominalMinDc`` before physics. That exact resolution is
        reproduced here; the sealed x4 carrier is 25 km.
        """

        cp = _cp()
        mesh_density = _host_f32_vector(
            _carrier_value(carriers, "meshDensity"), name="meshDensity"
        )
        n_cells = int(mesh_density.size)
        host: dict[str, np.ndarray] = {"mesh_density": mesh_density}
        for name in _STATIC_FIELDS[1:]:
            host[name] = _host_f32_vector(
                _carrier_value(carriers, name), name=name, count=n_cells
            )
        configured = np.float32(config_len_disp_m)
        if configured > np.float32(0.0):
            effective = configured
        else:
            nominal = np.asarray(_carrier_value(carriers, "nominalMinDc"))
            if nominal.dtype != np.float32 or nominal.shape != ():
                raise TypeError("nominalMinDc must be a host FP32 scalar")
            effective = np.float32(nominal)
        started = time.perf_counter()
        device = {name: cp.asarray(value) for name, value in host.items()}
        cp.cuda.get_current_stream().synchronize()
        elapsed = time.perf_counter() - started
        result = cls(
            **device,
            effective_length_scale_m=float(effective),
            n_cells=n_cells,
            h2d=TransferStats(
                int(sum(int(value.nbytes) for value in device.values())), elapsed
            ),
            _seal=_STATIC_SEAL,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self._seal is not _STATIC_SEAL:
            raise TypeError("GWD static carriers require from_host/from_resident")
        if self.n_cells <= 0:
            raise ValueError("GWD static carriers require at least one cell")
        for name in _STATIC_FIELDS:
            require_resident_array(
                f"gwdo_static.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(self.n_cells,),
            )
        if np.float32(self.effective_length_scale_m).view(np.uint32) != (
            X4_NOMINAL_MIN_DC_M_F32.view(np.uint32)
        ):
            raise ValueError("exact x4 GWD requires effective len_disp=25000 m")

    def receipt(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": CUDA_GWDO_V841_SCHEMA,
            "contract_sha256": CUDA_GWDO_V841_CONTRACT_SHA256,
            "n_cells": self.n_cells,
            "effective_length_scale_m": self.effective_length_scale_m,
            "h2d": self.h2d.as_dict(),
            "fields": list(_STATIC_FIELDS),
        }


@dataclass(frozen=True, slots=True)
class CudaYsuGwdoResultV841:
    """Transactional resident GWD/PBL tendency and diagnostic candidates."""

    rublten: Any
    rvblten: Any
    dtaux3d: Any
    dtauy3d: Any
    dusfcg: Any
    dvsfcg: Any
    rubldiff: Any
    rvbldiff: Any
    time_seconds: float
    validation_d2h: TransferStats

    @property
    def contract_sha256(self) -> str:
        return CUDA_GWDO_V841_CONTRACT_SHA256

    def validate(self, *, n_vert_levels: int, n_cells: int) -> None:
        shape = (n_vert_levels, n_cells)
        for name in (
            "rublten",
            "rvblten",
            "dtaux3d",
            "dtauy3d",
            "rubldiff",
            "rvbldiff",
        ):
            require_resident_array(
                f"gwdo_result.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=shape,
            )
        for name in ("dusfcg", "dvsfcg"):
            require_resident_array(
                f"gwdo_result.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(n_cells,),
            )
        if self.validation_d2h.bytes != 4:
            raise ValueError("GWD validation must transfer exactly four bytes")

    def receipt(self) -> dict[str, Any]:
        self.validate(
            n_vert_levels=int(self.rublten.shape[0]),
            n_cells=int(self.rublten.shape[1]),
        )
        return {
            "schema": CUDA_GWDO_V841_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "kernel_sha256": CUDA_GWDO_V841_KERNEL_SHA256,
            "selector": "bl_ysu_gwdo",
            "time_seconds": self.time_seconds,
            "shape": list(self.rublten.shape),
            "validation_d2h": self.validation_d2h.as_dict(),
            "transactional": True,
            "persistent_state": "none",
        }


@dataclass(frozen=True, slots=True)
class CpuYsuGwdoResultV841:
    """Readable FP32 source-loop oracle for focused kernel verification."""

    rublten: np.ndarray
    rvblten: np.ndarray
    dtaux3d: np.ndarray
    dtauy3d: np.ndarray
    dusfcg: np.ndarray
    dvsfcg: np.ndarray
    rubldiff: np.ndarray
    rvbldiff: np.ndarray


def _f32(value: Any) -> np.float32:
    return np.float32(value)


def _add(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) + _f32(right))


def _sub(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) - _f32(right))


def _mul(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) * _f32(right))


def _div(left: Any, right: Any) -> np.float32:
    return _f32(_f32(left) / _f32(right))


def _sqrt(value: Any) -> np.float32:
    return _f32(np.sqrt(_f32(value)))


def _pow(left: Any, right: Any) -> np.float32:
    return _f32(np.power(_f32(left), _f32(right)))


def native_cell_dx_m(
    mesh_density: Any, effective_length_scale_m: float
) -> np.ndarray:
    """Native MPAS v8.4.1 per-cell dx: ``len_disp / meshDensity**0.25``.

    This is the single construction MPAS uses wherever a scheme wants a cell
    length scale on a variable-resolution mesh -- GWDO reads it per cell
    (this module's own kernel and CPU oracle) and the convection driver hands
    the same value to Grell-Freitas
    (mpas_atmphys_driver_convection.F:718).  Exposing it once keeps the two
    consumers on one construction instead of two drifting copies.

    ``effective_length_scale_m`` is native's resolved ``len_disp``: a
    non-positive ``config_len_disp`` resolves to the mesh's ``nominalMinDc``
    before physics, exactly as :meth:`CudaYsuGwdoStaticV841.from_carriers`
    resolves it.
    """

    md = np.ascontiguousarray(np.asarray(mesh_density, dtype=np.float32))
    if md.ndim != 1 or md.size == 0:
        raise ValueError("meshDensity must be a non-empty FP32 cell vector")
    if not np.all(np.isfinite(md)) or np.any(md <= np.float32(0.0)) or np.any(
        md > np.float32(1.0)
    ):
        raise ValueError("meshDensity must be finite and within (0, 1]")
    length = _f32(effective_length_scale_m)
    if not np.isfinite(length) or length <= _f32(0.0):
        raise ValueError("effective_length_scale_m must be finite and positive")
    # Element-wise in the same FP32 order the scalar helpers use, so the
    # vector and the per-cell oracle path agree bit for bit.
    out = np.empty_like(md)
    for index in range(md.size):
        out[index] = _div(length, _pow(md[index], _f32(0.25)))
    out.setflags(write=False)
    return out


def _maxf(left: Any, right: Any) -> np.float32:
    lhs = _f32(left)
    rhs = _f32(right)
    return lhs if lhs >= rhs else rhs


def _minf(left: Any, right: Any) -> np.float32:
    lhs = _f32(left)
    rhs = _f32(right)
    return lhs if lhs <= rhs else rhs


def _require_cpu_f32(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.shape != shape:
        raise ValueError(f"{name} must be FP32 {shape}, got {array.dtype} {array.shape}")
    if not array.flags.c_contiguous or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and C-contiguous")
    return array


def bl_ysu_gwdo_cpu_oracle_v841(
    *,
    u_p: np.ndarray,
    v_p: np.ndarray,
    t_p: np.ndarray,
    qv_p: np.ndarray,
    pres_hydd_p: np.ndarray,
    pres2_hydd_p: np.ndarray,
    pi_p: np.ndarray,
    zmid_p: np.ndarray,
    rublten: np.ndarray,
    rvblten: np.ndarray,
    mesh_density: np.ndarray,
    var2d: np.ndarray,
    con: np.ndarray,
    oa1: np.ndarray,
    oa2: np.ndarray,
    oa3: np.ndarray,
    oa4: np.ndarray,
    ol1: np.ndarray,
    ol2: np.ndarray,
    ol3: np.ndarray,
    ol4: np.ndarray,
    effective_length_scale_m: float = 25_000.0,
    dt_seconds: float = 120.0,
) -> CpuYsuGwdoResultV841:
    """Execute the released source loop in explicit FP32 for test authority."""

    nlev, ncells = np.asarray(u_p).shape
    if nlev != X4_N_VERT_LEVELS or ncells <= 0:
        raise ValueError("CPU GWD oracle requires exact x4 55-level columns")
    shape = (nlev, ncells)
    dynamic = {
        "u_p": u_p,
        "v_p": v_p,
        "t_p": t_p,
        "qv_p": qv_p,
        "pres_hydd_p": pres_hydd_p,
        "pi_p": pi_p,
        "zmid_p": zmid_p,
        "rublten": rublten,
        "rvblten": rvblten,
    }
    checked = {
        name: _require_cpu_f32(name, value, shape)
        for name, value in dynamic.items()
    }
    p2 = _require_cpu_f32(
        "pres2_hydd_p", pres2_hydd_p, (nlev + 1, ncells)
    )
    static_values = {
        "mesh_density": mesh_density,
        "var2d": var2d,
        "con": con,
        "oa1": oa1,
        "oa2": oa2,
        "oa3": oa3,
        "oa4": oa4,
        "ol1": ol1,
        "ol2": ol2,
        "ol3": ol3,
        "ol4": ol4,
    }
    static = {
        name: _require_cpu_f32(name, value, (ncells,))
        for name, value in static_values.items()
    }
    if _f32(effective_length_scale_m).view(np.uint32) != (
        X4_NOMINAL_MIN_DC_M_F32.view(np.uint32)
    ):
        raise ValueError("CPU oracle requires x4 effective len_disp=25000 m")
    if _f32(dt_seconds).view(np.uint32) != X4_DT_SECONDS_F32.view(np.uint32):
        raise ValueError("CPU oracle requires x4 dt=120 s")
    if np.any(static["mesh_density"] <= _f32(0.0)) or np.any(
        static["mesh_density"] > _f32(1.0)
    ):
        raise ValueError("meshDensity is outside the sealed x4 domain")
    if np.any(static["var2d"] < _f32(0.0)) or np.any(
        static["con"] < _f32(0.0)
    ):
        raise ValueError("orographic variance/convexity is outside x4 domain")
    for name in ("oa1", "oa2", "oa3", "oa4"):
        if np.any(static[name] < _f32(-1.0)) or np.any(
            static[name] > _f32(1.0)
        ):
            raise ValueError("orographic asymmetry is outside x4 domain")
    for name in ("ol1", "ol2", "ol3", "ol4"):
        if np.any(static[name] < _f32(0.0)) or np.any(
            static[name] > _f32(1.0)
        ):
            raise ValueError("orographic length is outside x4 domain")
    if np.any(checked["t_p"] <= _f32(0.0)) or np.any(
        checked["qv_p"] < _f32(0.0)
    ):
        raise ValueError("temperature/qv lies outside common physics domain")
    if np.any(checked["pres_hydd_p"] <= _f32(0.0)) or np.any(
        np.diff(checked["pres_hydd_p"], axis=0) >= _f32(0.0)
    ):
        raise ValueError("dry hydrostatic mass pressure must decrease upward")
    if np.any(p2 <= _f32(0.0)) or np.any(np.diff(p2, axis=0) >= _f32(0.0)):
        raise ValueError("dry hydrostatic interface pressure must decrease upward")
    if np.any(checked["pi_p"] <= _f32(0.0)) or np.any(
        np.diff(checked["zmid_p"], axis=0) <= _f32(0.0)
    ):
        raise ValueError("Exner/height lies outside surface-to-top domain")

    out_u = np.empty(shape, dtype=np.float32)
    out_v = np.empty(shape, dtype=np.float32)
    dtaux3d = np.empty(shape, dtype=np.float32)
    dtauy3d = np.empty(shape, dtype=np.float32)
    rubldiff = np.empty(shape, dtype=np.float32)
    rvbldiff = np.empty(shape, dtype=np.float32)
    dusfcg = np.empty((ncells,), dtype=np.float32)
    dvsfcg = np.empty((ncells,), dtype=np.float32)

    nwdir = (6, 7, 5, 8, 2, 3, 1, 4)
    fdir = _div(_f32(8.0), _mul(_f32(2.0), PI_F32))
    for cell in range(ncells):
        md = static["mesh_density"][cell]
        dx = _div(_f32(effective_length_scale_m), _pow(md, _f32(0.25)))
        diagonal = _sqrt(_add(_mul(dx, dx), _mul(dx, dx)))
        dxy4 = (dx, dx, diagonal, diagonal)
        dxy4p = (dx, dx, diagonal, diagonal)

        vtj = np.zeros((nlev,), dtype=np.float32)
        vtk = np.zeros((nlev,), dtype=np.float32)
        rho = np.zeros((nlev,), dtype=np.float32)
        delta_p = np.zeros((nlev,), dtype=np.float32)
        u1 = np.zeros((nlev,), dtype=np.float32)
        v1 = np.zeros((nlev,), dtype=np.float32)
        usqj = np.zeros((nlev,), dtype=np.float32)
        bnv2 = np.zeros((nlev,), dtype=np.float32)
        velco = np.zeros((nlev - 1,), dtype=np.float32)
        taup = np.zeros((nlev + 1,), dtype=np.float32)
        taufb = np.zeros((nlev + 1,), dtype=np.float32)
        taud = np.zeros((nlev,), dtype=np.float32)

        for level in range(nlev):
            virtual = _add(
                _f32(1.0), _mul(EP1_F32, checked["qv_p"][level, cell])
            )
            vtj[level] = _mul(checked["t_p"][level, cell], virtual)
            vtk[level] = _div(vtj[level], checked["pi_p"][level, cell])
            rho[level] = _div(
                _mul(
                    _div(_f32(1.0), RD_F32),
                    checked["pres_hydd_p"][level, cell],
                ),
                vtj[level],
            )
            delta_p[level] = _sub(p2[level, cell], p2[level + 1, cell])
            u1[level] = checked["u_p"][level, cell]
            v1[level] = checked["v_p"][level, cell]

        zlowtop = _mul(_f32(2.0), static["var2d"][cell])
        klowtop = 0
        for level in range(1, nlev):
            height = _sub(
                checked["zmid_p"][level, cell],
                checked["zmid_p"][0, cell],
            )
            if zlowtop > _f32(0.0) and klowtop == 0 and height >= zlowtop:
                klowtop = level + 2
        kbl = max(min(klowtop, nlev), 2)
        delks = _div(_f32(1.0), _sub(p2[0, cell], p2[kbl - 1, cell]))
        delks1 = _div(
            _f32(1.0),
            _sub(
                checked["pres_hydd_p"][0, cell],
                checked["pres_hydd_p"][kbl - 1, cell],
            ),
        )
        ubar = _f32(0.0)
        vbar = _f32(0.0)
        rhobar = _f32(0.0)
        for level in range(nlev):
            if level + 1 < kbl:
                weight = _mul(delta_p[level], delks)
                ubar = _add(ubar, _mul(weight, u1[level]))
                vbar = _add(vbar, _mul(weight, v1[level]))
                rhobar = _add(rhobar, _mul(weight, rho[level]))

        oa_values = tuple(static[f"oa{j}"][cell] for j in range(1, 5))
        ol_values = tuple(static[f"ol{j}"][cell] for j in range(1, 5))
        wdir = _add(_f32(np.arctan2(ubar, vbar)), PI_F32)
        nearest = int(np.floor(_add(_mul(fdir, wdir), _f32(0.5))))
        nwd = nwdir[nearest % 8]
        direction = (nwd - 1) % 4
        sign = 1 - 2 * ((nwd - 1) // 4)
        oa = _mul(_f32(sign), oa_values[direction])
        ol = ol_values[direction]
        ol_perp_values = (ol_values[1], ol_values[0], ol_values[3], ol_values[2])
        olp = ol_perp_values[direction]
        od = _div(olp, _maxf(ol, _f32(1.0e-5)))
        od = _maxf(_minf(od, _f32(10.0)), _f32(0.1))
        dxy = dxy4[direction]
        dxyp = dxy4p[direction]

        for level in range(nlev - 1):
            ti = _div(
                _f32(2.0),
                _add(
                    checked["t_p"][level, cell],
                    checked["t_p"][level + 1, cell],
                ),
            )
            rdz = _div(
                _f32(1.0),
                _sub(
                    checked["zmid_p"][level + 1, cell],
                    checked["zmid_p"][level, cell],
                ),
            )
            du = _sub(u1[level], u1[level + 1])
            dv = _sub(v1[level], v1[level + 1])
            dw2 = _add(_mul(du, du), _mul(dv, dv))
            shr2 = _mul(_mul(_maxf(dw2, _f32(1.0)), rdz), rdz)
            stability = _add(
                _div(GRAVITY_F32, CP_F32),
                _mul(rdz, _sub(vtj[level + 1], vtj[level])),
            )
            bvf2 = _mul(_mul(GRAVITY_F32, stability), ti)
            usqj[level] = _maxf(_div(bvf2, shr2), _f32(-100.0))
            bnv2[level] = _div(
                _mul(
                    _mul(_mul(_f32(2.0), GRAVITY_F32), rdz),
                    _sub(vtk[level + 1], vtk[level]),
                ),
                _add(vtk[level + 1], vtk[level]),
            )

        ulow = _maxf(
            _sqrt(_add(_mul(ubar, ubar), _mul(vbar, vbar))), _f32(1.0)
        )
        rulow = _div(_f32(1.0), ulow)
        for level in range(nlev - 1):
            aligned = _add(
                _mul(_add(u1[level], u1[level + 1]), ubar),
                _mul(_add(v1[level], v1[level + 1]), vbar),
            )
            velco[level] = _mul(_mul(_f32(0.5), aligned), rulow)
            if _f32(0.0) < velco[level] < _f32(1.0):
                velco[level] = _f32(1.0)

        ldrag = bool(velco[0] <= _f32(0.0))
        for k_fortran in range(2, nlev + 1):
            if k_fortran < kbl and velco[k_fortran - 1] <= _f32(0.0):
                ldrag = True

        weight = _mul(
            _sub(
                checked["pres_hydd_p"][0, cell],
                checked["pres_hydd_p"][1, cell],
            ),
            delks1,
        )
        bnv2[0] = _mul(weight, bnv2[0])
        usqj[0] = _mul(weight, usqj[0])
        for k_fortran in range(2, nlev + 1):
            if k_fortran < kbl:
                level = k_fortran - 1
                pressure_weight = _mul(
                    _sub(
                        checked["pres_hydd_p"][level, cell],
                        checked["pres_hydd_p"][level + 1, cell],
                    ),
                    delks1,
                )
                bnv2[0] = _add(bnv2[0], _mul(bnv2[level], pressure_weight))
                usqj[0] = _add(usqj[0], _mul(usqj[level], pressure_weight))
        ldrag = bool(
            ldrag
            or bnv2[0] <= _f32(0.0)
            or ulow == _f32(1.0)
            or static["var2d"][cell] <= _f32(0.0)
        )
        for k_fortran in range(2, nlev + 1):
            if k_fortran < kbl:
                usqj[k_fortran - 1] = usqj[0]

        bnv = _f32(0.0)
        fr = _f32(0.0)
        xn = _f32(0.0)
        yn = _f32(0.0)
        if not ldrag:
            bnv = _sqrt(bnv2[0])
            fr = _mul(_mul(_mul(bnv, rulow), static["var2d"][cell]), od)
            fr = _minf(fr, _f32(10.0))
            xn = _mul(ubar, rulow)
            yn = _mul(vbar, rulow)

        coefm = _f32(0.0)
        taub = _f32(0.0)
        if not ldrag:
            exponent = _mul(_f32(0.8), fr)
            efact = _pow(_add(oa, _f32(2.0)), exponent)
            efact = _minf(_maxf(efact, _f32(0.0)), _f32(10.0))
            coefm = _pow(_add(_f32(1.0), ol), _add(oa, _f32(1.0)))
            xlinv = _div(coefm, dx)
            tem = _mul(_mul(fr, fr), static["con"][cell])
            gfobnv = _div(tem, _mul(_add(tem, _f32(0.5)), bnv))
            taub = _mul(xlinv, rhobar)
            for _ in range(3):
                taub = _mul(taub, ulow)
            taub = _mul(_mul(taub, gfobnv), efact)
        for k_fortran in range(1, nlev + 1):
            if k_fortran <= kbl:
                taup[k_fortran - 1] = taub

        icrilv = False
        for k_fortran in range(2, nlev):
            level = k_fortran - 1
            next_level = level + 1
            brvf = _f32(0.0)
            if k_fortran >= kbl:
                icrilv = bool(
                    icrilv
                    or usqj[level] < _f32(0.25)
                    or velco[level] <= _f32(0.0)
                )
                brvf = _sqrt(_maxf(bnv2[level], _f32(1.0e-5)))
            if (
                k_fortran >= kbl
                and not ldrag
                and not icrilv
                and taup[level] > _f32(0.0)
            ):
                temv = _div(_f32(1.0), velco[level])
                tem1 = _div(coefm, dxy)
                tem1 = _mul(tem1, _add(rho[next_level], rho[level]))
                tem1 = _mul(_mul(_mul(tem1, brvf), velco[level]), _f32(0.5))
                hd = _sqrt(_div(taup[level], tem1))
                fro = _mul(_mul(brvf, hd), temv)
                tem2 = _sqrt(usqj[level])
                tem = _add(_f32(1.0), _mul(tem2, fro))
                rim = _div(
                    _mul(usqj[level], _sub(_f32(1.0), fro)),
                    _mul(tem, tem),
                )
                if rim <= _f32(0.25):
                    temc = _add(_f32(2.0), _div(_f32(1.0), tem2))
                    hd = _div(
                        _mul(
                            velco[level],
                            _sub(_mul(_f32(2.0), _sqrt(temc)), temc),
                        ),
                        brvf,
                    )
                    taup[next_level] = _mul(tem1, _mul(hd, hd))
                else:
                    taup[next_level] = taup[level]

        if not ldrag:
            kblk = 0
            fbdpe = _f32(0.0)
            zblk = _f32(0.0)
            for k_fortran in range(nlev, 1, -1):
                if kblk == 0 and k_fortran <= kbl:
                    level = k_fortran - 1
                    contribution = _mul(
                        bnv2[level],
                        _sub(
                            checked["zmid_p"][kbl - 1, cell],
                            checked["zmid_p"][level, cell],
                        ),
                    )
                    contribution = _mul(contribution, delta_p[level])
                    contribution = _div(_div(contribution, GRAVITY_F32), rho[level])
                    fbdpe = _add(fbdpe, contribution)
                    fbdke = _mul(
                        _f32(0.5),
                        _add(
                            _mul(u1[level], u1[level]),
                            _mul(v1[level], v1[level]),
                        ),
                    )
                    if fbdpe >= fbdke:
                        kblk = min(k_fortran, kbl)
                        zblk = _sub(
                            checked["zmid_p"][kblk - 1, cell],
                            checked["zmid_p"][0, cell],
                        )
            if kblk != 0:
                fbdcd = _maxf(
                    _sub(_f32(2.0), _div(_f32(1.0), od)), _f32(0.0)
                )
                value = _mul(_f32(0.5), rhobar)
                value = _div(_mul(value, coefm), _mul(dx, dx))
                for factor in (fbdcd, dxyp, olp, zblk, _mul(ulow, ulow)):
                    value = _mul(value, factor)
                taufb[0] = value
                tautem = _div(taufb[0], _f32(kblk - 1))
                for k_fortran in range(2, kblk + 1):
                    taufb[k_fortran - 1] = _sub(
                        taufb[k_fortran - 2], tautem
                    )
                for interface in range(nlev + 1):
                    taup[interface] = _add(taup[interface], taufb[interface])

        for level in range(nlev):
            taud[level] = _div(
                _mul(_sub(taup[level + 1], taup[level]), GRAVITY_F32),
                delta_p[level],
            )
        dtfac = _f32(1.0)
        for k_fortran in range(1, nlev):
            level = k_fortran - 1
            if k_fortran <= kbl and taud[level] != _f32(0.0):
                limiter = _f32(
                    abs(
                        _div(
                            velco[level],
                            _mul(_f32(dt_seconds), taud[level]),
                        )
                    )
                )
                dtfac = _minf(dtfac, limiter)

        surface_u = _f32(0.0)
        surface_v = _f32(0.0)
        for level in range(nlev):
            taud[level] = _mul(taud[level], dtfac)
            du = _mul(taud[level], xn)
            dv = _mul(taud[level], yn)
            dtaux3d[level, cell] = du
            dtauy3d[level, cell] = dv
            surface_u = _add(surface_u, _mul(du, delta_p[level]))
            surface_v = _add(surface_v, _mul(dv, delta_p[level]))
            out_u[level, cell] = _add(checked["rublten"][level, cell], du)
            out_v[level, cell] = _add(checked["rvblten"][level, cell], dv)
            rubldiff[level, cell] = _sub(
                out_u[level, cell], checked["rublten"][level, cell]
            )
            rvbldiff[level, cell] = _sub(
                out_v[level, cell], checked["rvblten"][level, cell]
            )
        dusfcg[cell] = _mul(_div(_f32(-1.0), GRAVITY_F32), surface_u)
        dvsfcg[cell] = _mul(_div(_f32(-1.0), GRAVITY_F32), surface_v)

    return CpuYsuGwdoResultV841(
        rublten=out_u,
        rvblten=out_v,
        dtaux3d=dtaux3d,
        dtauy3d=dtauy3d,
        dusfcg=dusfcg,
        dvsfcg=dvsfcg,
        rubldiff=rubldiff,
        rvbldiff=rvbldiff,
    )


_CUDA_SOURCE = (
    CUDA_FTZ_HELPERS
    + r"""
#define GWDO_NLEV 55

extern "C" __global__ void bl_ysu_gwdo_v841_f32(
    const int nlev, const int ncells, const float dt,
    const float len_disp, const float gravity, const float cp,
    const float rd, const float fv, const float pi,
    const float *u, const float *v, const float *temperature,
    const float *qv, const float *p_mass_dry_hyd,
    const float *p_interface_dry_hyd, const float *exner,
    const float *zmid, const float *rublten_in,
    const float *rvblten_in, const float *mesh_density,
    const float *var2d, const float *oc1, const float *oa1,
    const float *oa2, const float *oa3, const float *oa4,
    const float *ol1, const float *ol2, const float *ol3,
    const float *ol4, float *rublten_out, float *rvblten_out,
    float *dtaux3d, float *dtauy3d, float *dusfcg,
    float *dvsfcg, float *rubldiff, float *rvbldiff,
    int *invalid)
{
    const int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= ncells) return;
    if (nlev != GWDO_NLEV) {
        atomicExch(invalid, 1);
        return;
    }

    const float md = mesh_density[cell];
    bool bad = !isfinite(dt) || dt <= 0.0f || !isfinite(len_disp)
        || len_disp <= 0.0f || !isfinite(md) || md <= 0.0f || md > 1.0f
        || !isfinite(var2d[cell]) || var2d[cell] < 0.0f
        || !isfinite(oc1[cell]) || oc1[cell] < 0.0f;
    const float oa_input[4] = {oa1[cell], oa2[cell], oa3[cell], oa4[cell]};
    const float ol_input[4] = {ol1[cell], ol2[cell], ol3[cell], ol4[cell]};
    for (int j = 0; j < 4; ++j) {
        bad = bad || !isfinite(oa_input[j]) || oa_input[j] < -1.0f
            || oa_input[j] > 1.0f || !isfinite(ol_input[j])
            || ol_input[j] < 0.0f || ol_input[j] > 1.0f;
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        bad = bad || !isfinite(u[i]) || !isfinite(v[i])
            || !isfinite(temperature[i]) || temperature[i] <= 0.0f
            || !isfinite(qv[i]) || qv[i] < 0.0f
            || !isfinite(p_mass_dry_hyd[i]) || p_mass_dry_hyd[i] <= 0.0f
            || !isfinite(exner[i]) || exner[i] <= 0.0f
            || !isfinite(zmid[i]) || !isfinite(rublten_in[i])
            || !isfinite(rvblten_in[i]);
        const int ii = k * ncells + cell;
        const int ii_above = (k + 1) * ncells + cell;
        bad = bad || !isfinite(p_interface_dry_hyd[ii])
            || p_interface_dry_hyd[ii] <= 0.0f;
        if (k + 1 < nlev) {
            bad = bad || !(p_mass_dry_hyd[i] > p_mass_dry_hyd[i + ncells])
                || !(zmid[i + ncells] > zmid[i]);
        }
        if (k == nlev - 1) {
            bad = bad || !isfinite(p_interface_dry_hyd[ii_above])
                || p_interface_dry_hyd[ii_above] <= 0.0f;
        }
        bad = bad || !(p_interface_dry_hyd[ii]
            > p_interface_dry_hyd[ii_above]);
    }
    if (bad) {
        atomicExch(invalid, 1);
        return;
    }

    float vtj[GWDO_NLEV];
    float vtk[GWDO_NLEV];
    float rho[GWDO_NLEV];
    float delta_p[GWDO_NLEV];
    float u1[GWDO_NLEV];
    float v1[GWDO_NLEV];
    float usqj[GWDO_NLEV];
    float bnv2[GWDO_NLEV];
    float velco[GWDO_NLEV - 1];
    float taup[GWDO_NLEV + 1];
    float taufb[GWDO_NLEV + 1];
    float taud[GWDO_NLEV];
    for (int k = 0; k < nlev; ++k) {
        usqj[k] = 0.0f;
        bnv2[k] = 0.0f;
        taud[k] = 0.0f;
    }
    for (int k = 0; k <= nlev; ++k) {
        taup[k] = 0.0f;
        taufb[k] = 0.0f;
    }

    const float dx = mpas_div(len_disp, powf(md, 0.25f));
    const float diagonal = mpas_sqrt(
        mpas_add(mpas_mul(dx, dx), mpas_mul(dx, dx)));
    const float dxy4[4] = {dx, dx, diagonal, diagonal};
    const float dxy4p[4] = {dx, dx, diagonal, diagonal};
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const int ip = (k + 1) * ncells + cell;
        const float virtual_factor = mpas_add(1.0f, mpas_mul(fv, qv[i]));
        vtj[k] = mpas_mul(temperature[i], virtual_factor);
        vtk[k] = mpas_div(vtj[k], exner[i]);
        rho[k] = mpas_div(
            mpas_mul(mpas_div(1.0f, rd), p_mass_dry_hyd[i]), vtj[k]);
        delta_p[k] = mpas_sub(
            p_interface_dry_hyd[i], p_interface_dry_hyd[ip]);
        u1[k] = u[i];
        v1[k] = v[i];
    }

    const float zlowtop = mpas_mul(2.0f, var2d[cell]);
    int klowtop = 0;
    for (int k = 1; k < nlev; ++k) {
        const float height = mpas_sub(
            zmid[k * ncells + cell], zmid[cell]);
        if (zlowtop > 0.0f && klowtop == 0 && height >= zlowtop)
            klowtop = k + 2;
    }
    int kbl = klowtop;
    if (kbl > nlev) kbl = nlev;
    if (kbl < 2) kbl = 2;
    const float delks = mpas_div(
        1.0f, mpas_sub(p_interface_dry_hyd[cell],
            p_interface_dry_hyd[(kbl - 1) * ncells + cell]));
    const float delks1 = mpas_div(
        1.0f, mpas_sub(p_mass_dry_hyd[cell],
            p_mass_dry_hyd[(kbl - 1) * ncells + cell]));
    float ubar = 0.0f;
    float vbar = 0.0f;
    float rhobar = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        if (k + 1 < kbl) {
            const float weight = mpas_mul(delta_p[k], delks);
            ubar = mpas_add(ubar, mpas_mul(weight, u1[k]));
            vbar = mpas_add(vbar, mpas_mul(weight, v1[k]));
            rhobar = mpas_add(rhobar, mpas_mul(weight, rho[k]));
        }
    }

    const int nwdir[8] = {6, 7, 5, 8, 2, 3, 1, 4};
    const float fdir = mpas_div(8.0f, mpas_mul(2.0f, pi));
    const float wdir = mpas_add(atan2f(ubar, vbar), pi);
    const int nearest = (int)floorf(mpas_add(mpas_mul(fdir, wdir), 0.5f));
    const int nwd = nwdir[nearest % 8];
    const int direction = (nwd - 1) % 4;
    const int sign = 1 - 2 * ((nwd - 1) / 4);
    const float oa = mpas_mul((float)sign, oa_input[direction]);
    const float ol = ol_input[direction];
    const float ol_perp[4] = {
        ol_input[1], ol_input[0], ol_input[3], ol_input[2]};
    const float olp = ol_perp[direction];
    float od = mpas_div(olp, mpas_max(ol, 1.0e-5f));
    od = mpas_max(mpas_min(od, 10.0f), 0.1f);
    const float dxy = dxy4[direction];
    const float dxyp = dxy4p[direction];

    for (int k = 0; k < nlev - 1; ++k) {
        const int i = k * ncells + cell;
        const int ia = i + ncells;
        const float ti = mpas_div(
            2.0f, mpas_add(temperature[i], temperature[ia]));
        const float rdz = mpas_div(1.0f, mpas_sub(zmid[ia], zmid[i]));
        const float du = mpas_sub(u1[k], u1[k + 1]);
        const float dv = mpas_sub(v1[k], v1[k + 1]);
        const float dw2 = mpas_add(mpas_mul(du, du), mpas_mul(dv, dv));
        const float shr2 = mpas_mul(
            mpas_mul(mpas_max(dw2, 1.0f), rdz), rdz);
        const float stability = mpas_add(
            mpas_div(gravity, cp), mpas_mul(rdz, mpas_sub(vtj[k + 1], vtj[k])));
        const float bvf2 = mpas_mul(mpas_mul(gravity, stability), ti);
        usqj[k] = mpas_max(mpas_div(bvf2, shr2), -100.0f);
        bnv2[k] = mpas_div(
            mpas_mul(mpas_mul(mpas_mul(2.0f, gravity), rdz),
                mpas_sub(vtk[k + 1], vtk[k])),
            mpas_add(vtk[k + 1], vtk[k]));
    }

    float ulow = mpas_sqrt(mpas_add(mpas_mul(ubar, ubar), mpas_mul(vbar, vbar)));
    ulow = mpas_max(ulow, 1.0f);
    const float rulow = mpas_div(1.0f, ulow);
    for (int k = 0; k < nlev - 1; ++k) {
        float aligned = mpas_add(
            mpas_mul(mpas_add(u1[k], u1[k + 1]), ubar),
            mpas_mul(mpas_add(v1[k], v1[k + 1]), vbar));
        velco[k] = mpas_mul(mpas_mul(0.5f, aligned), rulow);
        if (velco[k] < 1.0f && velco[k] > 0.0f) velco[k] = 1.0f;
    }
    bool ldrag = velco[0] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl && velco[kf - 1] <= 0.0f) ldrag = true;
    }

    const float first_weight = mpas_mul(
        mpas_sub(p_mass_dry_hyd[cell], p_mass_dry_hyd[ncells + cell]),
        delks1);
    bnv2[0] = mpas_mul(first_weight, bnv2[0]);
    usqj[0] = mpas_mul(first_weight, usqj[0]);
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) {
            const int k = kf - 1;
            const float pressure_weight = mpas_mul(
                mpas_sub(p_mass_dry_hyd[k * ncells + cell],
                    p_mass_dry_hyd[(k + 1) * ncells + cell]), delks1);
            bnv2[0] = mpas_add(bnv2[0], mpas_mul(bnv2[k], pressure_weight));
            usqj[0] = mpas_add(usqj[0], mpas_mul(usqj[k], pressure_weight));
        }
    }
    ldrag = ldrag || bnv2[0] <= 0.0f || ulow == 1.0f || var2d[cell] <= 0.0f;
    for (int kf = 2; kf <= nlev; ++kf) {
        if (kf < kbl) usqj[kf - 1] = usqj[0];
    }

    float bnv = 0.0f;
    float fr = 0.0f;
    float xn = 0.0f;
    float yn = 0.0f;
    if (!ldrag) {
        bnv = mpas_sqrt(bnv2[0]);
        fr = mpas_mul(mpas_mul(mpas_mul(bnv, rulow), var2d[cell]), od);
        fr = mpas_min(fr, 10.0f);
        xn = mpas_mul(ubar, rulow);
        yn = mpas_mul(vbar, rulow);
    }

    float coefm = 0.0f;
    float taub = 0.0f;
    if (!ldrag) {
        float efact = powf(mpas_add(oa, 2.0f), mpas_mul(0.8f, fr));
        efact = mpas_min(mpas_max(efact, 0.0f), 10.0f);
        coefm = powf(mpas_add(1.0f, ol), mpas_add(oa, 1.0f));
        const float xlinv = mpas_div(coefm, dx);
        const float tem = mpas_mul(mpas_mul(fr, fr), oc1[cell]);
        const float gfobnv = mpas_div(
            tem, mpas_mul(mpas_add(tem, 0.5f), bnv));
        taub = mpas_mul(xlinv, rhobar);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(taub, ulow);
        taub = mpas_mul(mpas_mul(taub, gfobnv), efact);
    }
    for (int kf = 1; kf <= nlev; ++kf) {
        if (kf <= kbl) taup[kf - 1] = taub;
    }

    bool icrilv = false;
    for (int kf = 2; kf < nlev; ++kf) {
        const int k = kf - 1;
        const int kp1 = k + 1;
        float brvf = 0.0f;
        if (kf >= kbl) {
            icrilv = icrilv || usqj[k] < 0.25f || velco[k] <= 0.0f;
            brvf = mpas_sqrt(mpas_max(bnv2[k], 1.0e-5f));
        }
        if (kf >= kbl && !ldrag && !icrilv && taup[k] > 0.0f) {
            const float temv = mpas_div(1.0f, velco[k]);
            float tem1 = mpas_div(coefm, dxy);
            tem1 = mpas_mul(tem1, mpas_add(rho[kp1], rho[k]));
            tem1 = mpas_mul(tem1, brvf);
            tem1 = mpas_mul(tem1, velco[k]);
            tem1 = mpas_mul(tem1, 0.5f);
            float hd = mpas_sqrt(mpas_div(taup[k], tem1));
            const float fro = mpas_mul(mpas_mul(brvf, hd), temv);
            const float tem2 = mpas_sqrt(usqj[k]);
            const float tem = mpas_add(1.0f, mpas_mul(tem2, fro));
            const float rim = mpas_div(
                mpas_mul(usqj[k], mpas_sub(1.0f, fro)), mpas_mul(tem, tem));
            if (rim <= 0.25f) {
                const float temc = mpas_add(2.0f, mpas_div(1.0f, tem2));
                hd = mpas_div(
                    mpas_mul(velco[k],
                        mpas_sub(mpas_mul(2.0f, mpas_sqrt(temc)), temc)),
                    brvf);
                taup[kp1] = mpas_mul(tem1, mpas_mul(hd, hd));
            } else {
                taup[kp1] = taup[k];
            }
        }
    }

    if (!ldrag) {
        int kblk = 0;
        float fbdpe = 0.0f;
        float zblk = 0.0f;
        for (int kf = nlev; kf >= 2; --kf) {
            if (kblk == 0 && kf <= kbl) {
                const int k = kf - 1;
                float contribution = mpas_mul(
                    bnv2[k], mpas_sub(zmid[(kbl - 1) * ncells + cell],
                        zmid[k * ncells + cell]));
                contribution = mpas_mul(contribution, delta_p[k]);
                contribution = mpas_div(contribution, gravity);
                contribution = mpas_div(contribution, rho[k]);
                fbdpe = mpas_add(fbdpe, contribution);
                const float fbdke = mpas_mul(0.5f,
                    mpas_add(mpas_mul(u1[k], u1[k]), mpas_mul(v1[k], v1[k])));
                if (fbdpe >= fbdke) {
                    kblk = kf < kbl ? kf : kbl;
                    zblk = mpas_sub(zmid[(kblk - 1) * ncells + cell], zmid[cell]);
                }
            }
        }
        if (kblk != 0) {
            const float fbdcd = mpas_max(
                mpas_sub(2.0f, mpas_div(1.0f, od)), 0.0f);
            float value = mpas_mul(0.5f, rhobar);
            value = mpas_mul(value, coefm);
            value = mpas_div(value, mpas_mul(dx, dx));
            value = mpas_mul(value, fbdcd);
            value = mpas_mul(value, dxyp);
            value = mpas_mul(value, olp);
            value = mpas_mul(value, zblk);
            value = mpas_mul(value, mpas_mul(ulow, ulow));
            taufb[0] = value;
            const float tautem = mpas_div(taufb[0], (float)(kblk - 1));
            for (int kf = 2; kf <= kblk; ++kf)
                taufb[kf - 1] = mpas_sub(taufb[kf - 2], tautem);
            for (int k = 0; k <= nlev; ++k)
                taup[k] = mpas_add(taup[k], taufb[k]);
        }
    }

    for (int k = 0; k < nlev; ++k) {
        taud[k] = mpas_div(
            mpas_mul(mpas_sub(taup[k + 1], taup[k]), gravity), delta_p[k]);
    }
    float dtfac = 1.0f;
    for (int kf = 1; kf < nlev; ++kf) {
        const int k = kf - 1;
        if (kf <= kbl && taud[k] != 0.0f) {
            const float limiter = mpas_abs(
                mpas_div(velco[k], mpas_mul(dt, taud[k])));
            dtfac = mpas_min(dtfac, limiter);
        }
    }

    float surface_u = 0.0f;
    float surface_v = 0.0f;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        taud[k] = mpas_mul(taud[k], dtfac);
        const float du = mpas_mul(taud[k], xn);
        const float dv = mpas_mul(taud[k], yn);
        dtaux3d[i] = du;
        dtauy3d[i] = dv;
        surface_u = mpas_add(surface_u, mpas_mul(du, delta_p[k]));
        surface_v = mpas_add(surface_v, mpas_mul(dv, delta_p[k]));
        rublten_out[i] = mpas_add(rublten_in[i], du);
        rvblten_out[i] = mpas_add(rvblten_in[i], dv);
        rubldiff[i] = mpas_sub(rublten_out[i], rublten_in[i]);
        rvbldiff[i] = mpas_sub(rvblten_out[i], rvblten_in[i]);
        if (!isfinite(du) || !isfinite(dv) || !isfinite(rublten_out[i])
                || !isfinite(rvblten_out[i]) || !isfinite(rubldiff[i])
                || !isfinite(rvbldiff[i])) bad = true;
    }
    dusfcg[cell] = mpas_mul(mpas_div(-1.0f, gravity), surface_u);
    dvsfcg[cell] = mpas_mul(mpas_div(-1.0f, gravity), surface_v);
    if (!isfinite(dusfcg[cell]) || !isfinite(dvsfcg[cell]) || bad)
        atomicExch(invalid, 1);
}
"""
)

CUDA_GWDO_V841_KERNEL_SHA256 = sha256(_CUDA_SOURCE.encode("utf-8")).hexdigest()
_KERNEL_NAME = "bl_ysu_gwdo_v841_f32"
_KERNELS: dict[int, Any] = {}
_REQUIRED_KERNEL_CACHE_OPTIONS = ("--std=c++17", "--fmad=false")


def _require_kernel_cache_contract(cache: KernelCache) -> None:
    if not isinstance(cache, KernelCache):
        raise TypeError("kernel_cache must be a KernelCache")
    options = tuple(cache.base_options)
    if options != _REQUIRED_KERNEL_CACHE_OPTIONS:
        raise TypeError(
            "v8.4.1 YSU-GWDO KernelCache base_options must be exactly "
            f"{_REQUIRED_KERNEL_CACHE_OPTIONS}; got {options}"
        )
    manifest = cache.compile_manifest()
    platform = (
        manifest.get("compile_platform") if isinstance(manifest, Mapping) else None
    )
    fingerprint = platform.get("fingerprint") if isinstance(platform, Mapping) else None
    digest = platform.get("sha256") if isinstance(platform, Mapping) else None
    try:
        validated = validate_compile_platform_fingerprint(fingerprint)
    except CompileContractError as error:
        raise ValueError(
            "v8.4.1 YSU-GWDO KernelCache lacks validated FTZ binding"
        ) from error
    if not isinstance(digest, str) or digest != canonical_sha256(validated):
        raise ValueError(
            "v8.4.1 YSU-GWDO KernelCache lacks validated FTZ binding"
        )


def _kernel(cache: KernelCache) -> Any:
    _require_kernel_cache_contract(cache)
    result = _KERNELS.get(id(cache))
    if result is None:
        result = cache.raw_kernel(
            _KERNEL_NAME,
            _CUDA_SOURCE,
            module_key="mpas_port.cuda_gwdo_v841",
        )
        _KERNELS[id(cache)] = result
    return result


def run_bl_ysu_gwdo_cuda_v841(
    columns: CudaYsuGwdoColumnViewV841,
    *,
    rublten: Any,
    rvblten: Any,
    static: CudaYsuGwdoStaticV841,
    dt_seconds: float,
    kernel_cache: KernelCache,
) -> CudaYsuGwdoResultV841:
    """Run complete native YSU-GWDO into transactional resident candidates."""

    _require_kernel_cache_contract(kernel_cache)
    cp = _cp()
    if not isinstance(columns, CudaYsuGwdoColumnViewV841):
        raise TypeError("columns must be a sealed CudaYsuGwdoColumnViewV841")
    if not isinstance(static, CudaYsuGwdoStaticV841):
        raise TypeError("static must be sealed CudaYsuGwdoStaticV841")
    columns.validate()
    static.validate()
    nlev = columns.n_vert_levels
    ncells = columns.n_cells
    if static.n_cells != ncells:
        raise ValueError(
            f"GWD static carriers have {static.n_cells} cells, columns have {ncells}"
        )
    shape = (nlev, ncells)
    require_resident_array(
        "gwdo.rublten", rublten, dtype=np.float32, shape=shape
    )
    require_resident_array(
        "gwdo.rvblten", rvblten, dtype=np.float32, shape=shape
    )
    dt = np.float32(dt_seconds)
    if dt.view(np.uint32) != X4_DT_SECONDS_F32.view(np.uint32):
        raise ValueError("exact x4 YSU-GWDO requires dt_seconds=120")
    length_scale = np.float32(static.effective_length_scale_m)

    output = {
        "rublten": cp.empty(shape, dtype=cp.float32),
        "rvblten": cp.empty(shape, dtype=cp.float32),
        "dtaux3d": cp.empty(shape, dtype=cp.float32),
        "dtauy3d": cp.empty(shape, dtype=cp.float32),
        "dusfcg": cp.empty((ncells,), dtype=cp.float32),
        "dvsfcg": cp.empty((ncells,), dtype=cp.float32),
        "rubldiff": cp.empty(shape, dtype=cp.float32),
        "rvbldiff": cp.empty(shape, dtype=cp.float32),
    }
    invalid = cp.zeros((1,), dtype=cp.int32)
    threads = 64
    _kernel(kernel_cache)(
        ((ncells + threads - 1) // threads,),
        (threads,),
        (
            np.int32(nlev),
            np.int32(ncells),
            dt,
            length_scale,
            GRAVITY_F32,
            CP_F32,
            RD_F32,
            EP1_F32,
            PI_F32,
            columns.u_p,
            columns.v_p,
            columns.t_p,
            columns.qv_p,
            columns.pres_hydd_p,
            columns.pres2_hydd_p,
            columns.pi_p,
            columns.zmid_p,
            rublten,
            rvblten,
            static.mesh_density,
            static.var2d,
            static.con,
            static.oa1,
            static.oa2,
            static.oa3,
            static.oa4,
            static.ol1,
            static.ol2,
            static.ol3,
            static.ol4,
            output["rublten"],
            output["rvblten"],
            output["dtaux3d"],
            output["dtauy3d"],
            output["dusfcg"],
            output["dvsfcg"],
            output["rubldiff"],
            output["rvbldiff"],
            invalid,
        ),
    )
    started = time.perf_counter()
    invalid_value = int(cp.asnumpy(invalid)[0])
    validation = TransferStats(int(invalid.nbytes), time.perf_counter() - started)
    if invalid_value != 0:
        raise FloatingPointError(
            "bl_ysu_gwdo CUDA rejected non-finite or out-of-contract x4 input"
        )
    result = CudaYsuGwdoResultV841(
        **output,
        time_seconds=columns.time_seconds,
        validation_d2h=validation,
    )
    result.validate(n_vert_levels=nlev, n_cells=ncells)
    return result


__all__ = [
    "BL_GWDO_V841_SHA256",
    "CP_F32",
    "CUDA_GWDO_V841_CONTRACT_SHA256",
    "CUDA_GWDO_V841_EVIDENCE",
    "CUDA_GWDO_V841_KERNEL_SHA256",
    "CUDA_GWDO_V841_SCHEMA",
    "CudaYsuGwdoColumnViewV841",
    "CudaYsuGwdoResultV841",
    "CudaYsuGwdoStaticV841",
    "CpuYsuGwdoResultV841",
    "EP1_F32",
    "GRAVITY_F32",
    "MODULE_BL_GWDO_V841_SHA256",
    "MPAS_ATMPHYS_DRIVER_GWDO_V841_SHA256",
    "native_cell_dx_m",
    "PI_F32",
    "RD_F32",
    "REGISTRY_CORE_ATMOSPHERE_V841_SHA256",
    "RV_F32",
    "X4_DT_SECONDS_F32",
    "X4_NOMINAL_MIN_DC_M_F32",
    "X4_N_VERT_LEVELS",
    "bl_ysu_gwdo_cpu_oracle_v841",
    "gwdo_contract_evidence_v841",
    "run_bl_ysu_gwdo_cuda_v841",
]

