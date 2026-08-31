"""Pinned Arwen two-phase physics backend for resident MPAS-A v8.4.1.

This is the production adapter between the audited MPAS CUDA preparation
carriers and gpuwm's persistent ``run_mpas_column_batch`` seam.  It owns no
parameterization arithmetic: phase one and WSM6 remain the exact frozen-v2 Arwen
objects, while MPAS preparation, optional native YSU-GWDO, conservative
coupling, and post-RK recovery remain the separately audited MPAS objects.

The adapter is deliberately strict.  Construction requires a sealed mapping
containing every real surface, soil, land-use, solar, and cadence input; no
Arwen constructor default is admitted.  The imported Arwen tree is accepted
only when all production source bytes match
:data:`ARWEN_BUILD_COMMIT` below, which is the ONE place that commit is
spelled -- this sentence deliberately names no digest of its own.  A prose
copy of the pin is how this module came to contradict itself: it carried the
0.1.x seam-converge merge commit as present-tense fact for the whole of the
2.5.8 re-pin, while the constant below already named the 2.5.8 cut, so the
refusal a user hit and the documentation they read named different engines.
Every step is transactional:
the Arwen boundary state is exported before phase one, and an abort or any
adapter failure rebuilds a fresh seam and restores that boundary snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import inspect
import json
import math
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from .cuda_backend.containers import TransferStats, require_resident_array
from .cuda_backend.runtime import KernelCache
from .cuda_gwdo_v841 import (
    CUDA_GWDO_V841_CONTRACT_SHA256,
    CUDA_GWDO_V841_KERNEL_SHA256,
    CudaYsuGwdoColumnViewV841,
    CudaYsuGwdoResultV841,
    CudaYsuGwdoStaticV841,
    run_bl_ysu_gwdo_cuda_v841,
)
from .cuda_physics_prep_v841 import (
    CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    CUDA_PHYSICS_PREP_V841_KERNEL_SHA256,
    CudaMpasToPhysGeometryV841,
    prepare_mpas_to_phys_cuda_v841,
)
from .cuda_physics_v841 import (
    CUDA_PHYSICS_V841_CONTRACT_SHA256,
    CUDA_PHYSICS_V841_KERNEL_SHA256,
    CudaPhaseOneExecutionProvenanceV841,
    CudaPostRkWsm6UpdateV841,
    CudaRawColumnPhysicsV841,
    WSM6_SCALAR_NAMES,
)


CUDA_ARWEN_PHYSICS_V841_SCHEMA = "mpas-port.cuda-arwen-physics-v841/v2"
ARWEN_BUILD_COMMIT = "7e34a4877a8278970fcb16db3a117ef6cc5b9bbe"
MPAS_SEAM_CONTRACT_SHA256 = (
    "5c629e23be2af20c0b1660d262443c415256126b812493f6681590bf07aff92a"
)
MPAS_SEAM_CONTRACT_SURFACE_SHA256 = (
    "f83b16185a50667d65d90771e1f32942ff31fbfbd52ce4e29e09ea0cb11e1007"
)
ARWEN_GLACIER_COMPOSED_TU_SHA256 = (
    "edafcac585d4786c0cdfddf07f8e767b64d0d40b6db0e4da3dc3b2fa8c21fb59"
)

# Every byte below participates in the phase-one/phase-two execution path.
# Hashing the imported files at construction makes the private effective-
# radius access no broader than this exact, reviewed implementation.
ARWEN_SOURCE_MANIFEST: Mapping[str, str] = MappingProxyType(
    {
        "gpuwm/core/mpas_column_batch.py": (
            "f4335caa44687526089995f525c34b718465972531e75f2b8e908cc5ccbc6c7c"
        ),
        "gpuwm/core/physics.py": (
            "51b8c6067ebb27ef538dbe0291bd6a2e2d570823c78cf2e85f75e325ca8114af"
        ),
        "gpuwm/core/microphysics.py": (
            "4df5b7a3e46c348a98920c2c4b8cac46632dee955d7c8eea19136c183c8f8670"
        ),
        "gpuwm/core/gf.py": (
            "12e2c59564aef91075339707cc2aba019f892b2f1e4b930ff1d3f064bdc38ca7"
        ),
        "gpuwm/core/kernels/gf.cu": (
            "27019d7fdd4ad31aec2e2e4b21ec339dd512a449aea5e35ab2386ca5217aae5e"
        ),
        "gpuwm/core/rrtmg_legacy.py": (
            "39af610f0df41f9a3eaef601cd55b8ba27928c7e477a1d4739048094a510214d"
        ),
        "gpuwm/core/noahmp_runtime.py": (
            "990780ab87764481c463b1fa2f8c988cd9538e0833dc390203d2a0396fee66bc"
        ),
        "gpuwm/core/noahmp_glacier.py": (
            "1bb607569b5007c5d2817e879eb82112db7ed01f11931efe1802d35b3fb7b23b"
        ),
        "gpuwm/core/noahmp_glacier_gpu.py": (
            "c8409f8d0644550198b54006ab084d9bb67018037367639e60a5d8af479710fa"
        ),
        "gpuwm/core/noahmp_kernel_sources.py": (
            "4b2fab2ac92e93669491cbe526f81d23bc94e075859ae849f2b8c84827057538"
        ),
        "gpuwm/core/kernels/__init__.py": (
            "05102ff85fac24700858309f497694f9b3919b26cab3963b2cc35a4182c22919"
        ),
        "gpuwm/core/kernels/noahmp_leaves.cu": (
            "0ce9461705395dccbfebed3d9d27e87eebaeaca79896ae369eaa02ec1e77307f"
        ),
        "gpuwm/core/kernels/noahmp_glacier.cu": (
            "6a200773433a257f562f38d3e32cff13555acea1a4ce8267054b60914a6b5219"
        ),
        "gpuwm/config.py": (
            "73b91bb4f4b3de1e8521d6e932f5f7fa106e4f7efcfd400affa94effde861aa7"
        ),
        "gpuwm/io/restart.py": (
            "8482a222c0ad096400a9154e5a6a706c21704b7cc17ebbbab9d15bbe4879f91f"
        ),
        "docs/mpas-seam.md": (
            "7fe13aaa37fa944b160838c1fe5083ab79fe0d2e271da048c4dc92de8d9fc7fd"
        ),
    }
)


_GLACIER_CUDA_PROVENANCE = (
    "noahmp-glacier/cuda (gpuwm/core/kernels/noahmp_glacier.cu)"
)
_ISICE_TABLE = 15

_LIMITATIONS = (
    "fa35 exposes one phase-one pressure family: MPAS supplies moist-hydrostatic "
    "pres_hyd_p/pres2_hyd_p to all Arwen phase-one consumers; cloud fraction "
    "cannot independently receive EOS pressure through the published seam",
    "fa35 legacy RRTMG rebuilds t8w from the constructor's nominal one-dimensional "
    "vertical weights; frozen MPAS t2_p is not a published phase-one argument",
    "legacy RRTMG stages columns through the host on radiation-due calls; this "
    "occurs at the Arwen-owned radiation cadence rather than every model step",
    "fa35 Noah-MP consumes the common published phase-one atmosphere; the separate "
    "raw-qv MPAS Noah-MP sounding is prepared and validated but is not accepted by "
    "the published Arwen call signature",
    "fa35 GF convective momentum tendencies are not coupled; native MPAS v8.4.1 "
    "does not couple them either (its cu_grell_freitas call carries no "
    "rucuten/rvcuten), so this is native parity rather than a gap",
    "fa35 accepts one p_top_pa scalar derived by the runner as an area-weighted "
    "value, whereas native MPAS plrad is per-column; this adapter therefore makes "
    "no source-matched/native-parity claim for that pressure boundary",
)

_REQUIRED_ARWEN_EXPORT_FIELDS = ("tsk", "smois", "tslb", "hfx", "qfx", "lh")
_OPTIONAL_ARWEN_EXPORT_FIELDS = (
    "t2", "q2", "pblh", "u10", "v10", "psfc",
    "swdown", "glw", "olr",
)
_SOIL_DIAGNOSTIC_FIELDS = frozenset(("smois", "tslb"))
_GWDO_SURFACE_FIELDS = ("dusfcg", "dvsfcg")
_GWDO_LEVEL_FIELDS = (
    "dtaux3d",
    "dtauy3d",
    "rubldiff",
    "rvbldiff",
)
_GWDO_DIAGNOSTIC_FIELDS = (*_GWDO_SURFACE_FIELDS, *_GWDO_LEVEL_FIELDS)

_CONSTRUCTOR_ARRAY_FIELDS = (
    "latitude_deg",
    "longitude_deg",
    "terrain_height_m",
    "z_interface_nominal_m",
    "landmask",
    "xland",
    "ivgtyp",
    "isltyp",
    "vegfra",
    "tsk",
    "tmn",
    "xice",
    "snow",
    "snow_depth",
    "soil_temperature",
    "soil_moisture",
    # Native MPAS v8.4.1 hands GF a per-cell dx built from the mesh
    # (mpas_atmphys_driver_convection.F:718, len_disp/meshDensity**0.25);
    # one scalar dx is a lie on a variable-resolution mesh, so this is a
    # required sealed static, not an option.
    "dx_column_m",
)
_CONSTRUCTOR_KEYS = frozenset(
    (
        "n_levels",
        "n_columns",
        "dt",
        "radiation_seconds",
        "surface_pbl_seconds",
        "cumulus_seconds",
        "cumulus_scheme",
        "start_time",
        "p_top_pa",
        "dx_m",
        "gf_ishallow",
        "wsm6_hail_opt",
        "xice_threshold",
        *_CONSTRUCTOR_ARRAY_FIELDS,
    )
)
_SURFACE_FLOAT_FIELDS = (
    "latitude_deg",
    "longitude_deg",
    "terrain_height_m",
    "landmask",
    "xland",
    "vegfra",
    "tsk",
    "tmn",
    "xice",
    "snow",
    "snow_depth",
)
_SURFACE_INT_FIELDS = ("ivgtyp", "isltyp")
_SOIL_FIELDS = ("soil_temperature", "soil_moisture")
_CONSTRUCTOR_SEAL = object()


def _digest_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _json_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_identity(array: np.ndarray) -> Mapping[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _positive_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_interval(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _cadence_steps(name: str, seconds: float, dt: float) -> int:
    """Mirror frozen Arwen v2's pure-host exact integer cadence refusal."""

    ratio = seconds / dt
    rounded = int(round(ratio))
    if rounded < 1 or abs(ratio - rounded) > 1.0e-9 * max(ratio, 1.0):
        raise ValueError(
            f"{name}={seconds} s is not a positive integer multiple of "
            f"dt={dt} s"
        )
    return rounded


def _host_exact_array(
    mapping: Mapping[str, Any], name: str, *, dtype: np.dtype, shape: tuple[int, ...]
) -> np.ndarray:
    value = mapping[name]
    if hasattr(value, "__cuda_array_interface__"):
        raise TypeError(f"constructor {name} must be an official host array")
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"constructor {name} must be {np.dtype(dtype)}, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"constructor {name} must have shape {shape}, got {array.shape}")
    if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
        raise ValueError(f"constructor {name} contains non-finite values")
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _locate_degenerate_columns(prepared: Any) -> str:
    """Name the columns a column scheme cannot integrate, and why.

    Every input the seam is handed is finite -- the preparation refuses
    otherwise -- so when a scheme's own arithmetic produces a non-finite
    tendency, the cause is a column that is finite and DEGENERATE.  The two
    that matter to a boundary-layer scheme are a column with no wind shear
    anywhere, because the bulk Richardson number divides by it, and a column
    with no stratification.  Both are reported with column indices, so the
    reader is told which cells rather than which array.
    """

    import numpy as _np

    cp = __import__("cupy")
    findings: list[str] = []
    u = cp.asnumpy(prepared.u_p)
    v = cp.asnumpy(prepared.v_p)
    theta = cp.asnumpy(prepared.th_p)
    speed = _np.hypot(u, v)
    still = _np.argwhere(speed.max(axis=0) == _np.float32(0.0)).ravel()
    if still.size:
        findings.append(
            f"{still.size} column(s) carry zero wind at every level "
            f"(first {int(still[0])} of {speed.shape[1]})"
        )
    shear = _np.abs(_np.diff(u, axis=0)) + _np.abs(_np.diff(v, axis=0))
    flat = _np.setdiff1d(
        _np.argwhere(shear.max(axis=0) == _np.float32(0.0)).ravel(), still
    )
    if flat.size:
        findings.append(
            f"{flat.size} further column(s) carry a wind that does not change "
            f"with height (first {int(flat[0])})"
        )
    isothermal = _np.argwhere(
        _np.abs(_np.diff(theta, axis=0)).max(axis=0) == _np.float32(0.0)
    ).ravel()
    if isothermal.size:
        findings.append(
            f"{isothermal.size} column(s) carry a constant potential "
            f"temperature (first {int(isothermal[0])})"
        )
    if not findings:
        return (
            "No column is degenerate in wind shear or stratification, so the "
            "cause is inside the scheme's own carried state rather than this "
            "step's sounding."
        )
    return "Degenerate columns handed to it: " + "; ".join(findings) + "."


@dataclass(frozen=True, slots=True)
class SealedArwenConstructorV841:
    """Complete, immutable exact-real constructor mapping for frozen Arwen v2."""

    _values: Mapping[str, Any] = field(repr=False, compare=False)
    identity_sha256: str
    host_array_bytes: int
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SealedArwenConstructorV841":
        if not isinstance(values, Mapping):
            raise TypeError("Arwen constructor values must be a mapping")
        missing = sorted(_CONSTRUCTOR_KEYS - set(values))
        extra = sorted(set(values) - _CONSTRUCTOR_KEYS)
        if missing or extra:
            raise ValueError(
                "exact-real Arwen constructor mapping is not exhaustive: "
                f"missing={missing}, extra={extra}"
            )
        nlev = values["n_levels"]
        ncol = values["n_columns"]
        if isinstance(nlev, bool) or not isinstance(nlev, (int, np.integer)) or int(nlev) < 3:
            raise ValueError("n_levels must be an integer >= 3")
        if isinstance(ncol, bool) or not isinstance(ncol, (int, np.integer)) or int(ncol) < 1:
            raise ValueError("n_columns must be a positive integer")
        nlev, ncol = int(nlev), int(ncol)
        if not isinstance(values["start_time"], datetime):
            raise TypeError("start_time must be a UTC datetime")
        scheme = values["cumulus_scheme"]
        if scheme not in ("gf", "kf", None):
            raise ValueError("cumulus_scheme must be 'gf', 'kf', or None")
        if scheme is None and values["cumulus_seconds"] is not None:
            raise ValueError("cumulus_seconds requires a cumulus_scheme")
        if scheme is not None:
            _positive_real("cumulus_seconds", values["cumulus_seconds"])
        dt = _positive_real("dt", values["dt"])
        radiation_seconds = _positive_real(
            "radiation_seconds", values["radiation_seconds"]
        )
        surface_pbl_seconds = _positive_real(
            "surface_pbl_seconds", values["surface_pbl_seconds"]
        )
        _cadence_steps("radiation_seconds", radiation_seconds, dt)
        _cadence_steps("surface_pbl_seconds", surface_pbl_seconds, dt)
        cumulus_seconds = None
        if values["cumulus_seconds"] is not None:
            cumulus_seconds = _positive_real(
                "cumulus_seconds", values["cumulus_seconds"]
            )
            cumulus_steps = _cadence_steps(
                "cumulus_seconds", cumulus_seconds, dt
            )
            if scheme == "gf" and cumulus_steps != 1:
                raise ValueError(
                    "cumulus_scheme='gf' requires cumulus_seconds == dt"
                )
        hail_opt = values["wsm6_hail_opt"]
        if (
            isinstance(hail_opt, bool)
            or not isinstance(hail_opt, (int, np.integer))
            or int(hail_opt) not in (0, 1)
        ):
            raise ValueError("wsm6_hail_opt must be the integer 0 or 1")
        # GF shallow convection.  Native MPAS v8.4.1 hardwires ishallow = 1
        # (mpas_atmphys_vars.F:340); shallow OFF is reachable only as an
        # explicit A/B arm and is meaningless without GF.
        ishallow = values["gf_ishallow"]
        if (
            isinstance(ishallow, bool)
            or not isinstance(ishallow, (int, np.integer))
            or int(ishallow) not in (0, 1)
        ):
            raise ValueError("gf_ishallow must be the integer 0 or 1")
        if int(ishallow) == 1 and scheme != "gf":
            raise ValueError("gf_ishallow=1 requires cumulus_scheme='gf'")
        scalars = {
            "n_levels": nlev,
            "n_columns": ncol,
            "dt": dt,
            "radiation_seconds": radiation_seconds,
            "surface_pbl_seconds": surface_pbl_seconds,
            "cumulus_seconds": cumulus_seconds,
            "cumulus_scheme": scheme,
            "start_time": values["start_time"],
            "p_top_pa": _positive_real("p_top_pa", values["p_top_pa"]),
            "dx_m": _positive_real("dx_m", values["dx_m"]),
            "gf_ishallow": int(ishallow),
            "wsm6_hail_opt": int(hail_opt),
            "xice_threshold": _unit_interval(
                "xice_threshold", values["xice_threshold"]
            ),
        }
        arrays: dict[str, np.ndarray] = {}
        for name in _SURFACE_FLOAT_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.float32), shape=(ncol,)
            )
        for name in _SURFACE_INT_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.int32), shape=(ncol,)
            )
        for name in _SOIL_FIELDS:
            arrays[name] = _host_exact_array(
                values, name, dtype=np.dtype(np.float32), shape=(4, ncol)
            )
        # Per-column GF dx.  Native builds it per cell; the sealed mapping
        # therefore carries the whole vector, positive and finite.
        dx_column = _host_exact_array(
            values, "dx_column_m", dtype=np.dtype(np.float32), shape=(ncol,)
        )
        if not np.all(np.isfinite(dx_column)) or np.any(dx_column <= np.float32(0.0)):
            raise ValueError("dx_column_m must be finite and positive")
        arrays["dx_column_m"] = dx_column
        z = np.asarray(values["z_interface_nominal_m"])
        if z.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("z_interface_nominal_m must be host float32 or float64")
        if z.shape != (nlev + 1,) or not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0):
            raise ValueError(
                "z_interface_nominal_m must be finite, strictly increasing, and "
                f"have shape {(nlev + 1,)}"
            )
        arrays["z_interface_nominal_m"] = np.array(z, copy=True, order="C")
        arrays["z_interface_nominal_m"].setflags(write=False)
        sealed: dict[str, Any] = {**scalars, **arrays}
        identity = {
            name: (
                _array_identity(value)
                if isinstance(value, np.ndarray)
                else value.isoformat()
                if isinstance(value, datetime)
                else value
            )
            for name, value in sorted(sealed.items())
        }
        return cls(
            _values=MappingProxyType(sealed),
            identity_sha256=_json_digest(identity),
            host_array_bytes=sum(int(value.nbytes) for value in arrays.values()),
            _seal=_CONSTRUCTOR_SEAL,
        )

    @property
    def n_levels(self) -> int:
        return int(self._values["n_levels"])

    @property
    def n_columns(self) -> int:
        return int(self._values["n_columns"])

    @property
    def dt(self) -> float:
        return float(self._values["dt"])

    @property
    def xice_threshold(self) -> float:
        return float(self._values["xice_threshold"])

    def expected_surface_classification(self) -> Mapping[str, Any]:
        """Authority-only host classification for the sealed constructor."""

        xland = np.asarray(self._values["xland"], dtype=np.float32)
        xice = np.asarray(self._values["xice"], dtype=np.float32)
        ivgtyp = np.asarray(self._values["ivgtyp"], dtype=np.int32)
        threshold = np.float32(self.xice_threshold)
        sea_ice = xice >= threshold
        open_water = (xland >= np.float32(1.5)) & ~sea_ice
        land = ~(sea_ice | open_water)
        glacier = land & (ivgtyp == np.int32(_ISICE_TABLE))
        return MappingProxyType(
            {
                "xland_source": "native",
                "xland_land_columns": int(np.count_nonzero(xland < np.float32(1.5))),
                "xland_water_columns": int(np.count_nonzero(xland >= np.float32(1.5))),
                "xice_threshold": self.xice_threshold,
                "sea_ice_columns": int(np.count_nonzero(sea_ice)),
                "open_water_columns": int(np.count_nonzero(open_water)),
                "sflx_land_columns": int(np.count_nonzero(land & ~glacier)),
                "glacier_columns": int(np.count_nonzero(glacier)),
            }
        )

    def arwen_kwargs(self) -> dict[str, Any]:
        if self._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("Arwen constructor mapping is not sealed")
        # Arrays remain the sealed read-only objects.  Arwen copies/uploads
        # them during construction and never receives an adapter-owned mutable
        # static carrier.
        return dict(self._values)

    def receipt(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "identity_sha256": self.identity_sha256,
                "n_levels": self.n_levels,
                "n_columns": self.n_columns,
                "dt": self.dt,
                "host_array_bytes": self.host_array_bytes,
                "defaults_used": False,
                "surface_soil_statics": "official-exhaustive-sealed-host-mapping",
                "xland_source": "native",
                "xice_threshold": self.xice_threshold,
                "expected_surface_classification": dict(
                    self.expected_surface_classification()
                ),
            }
        )


_ADAPTER_AUTHORITY = {
    "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
    "arwen_commit": ARWEN_BUILD_COMMIT,
    "arwen_sources": dict(ARWEN_SOURCE_MANIFEST),
    "contract_document_sha256": MPAS_SEAM_CONTRACT_SHA256,
    "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
    "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
    "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
    "gwdo_contract_sha256": CUDA_GWDO_V841_CONTRACT_SHA256,
    "coupling_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
    "theta": "frozen prep th_p dry theta",
    "phase1_pressure": "pres_hyd_p/pres2_hyd_p with explicit pi_p",
    "phase2_pressure": "pres_p EOS with rho_dry and z_p",
    "h_diabatic": "explicitly declined; never replayed or folded",
    "constructor": "exhaustive sealed exact-real mapping including native xland and explicit xice_threshold; no defaults",
    "pin_order": "pin exact Arwen v2 before MPAS KernelCache construction",
    "execution_provenance": (
        "typed aggregate-executed carrier; actual GWDO result identity and 4B gate"
    ),
    "publication": (
        "begin -> finished_unpublished -> explicit commit; abort restores seam/scalars"
    ),
    "diagnostics": (
        "committed-boundary public export plus retained six-field resident GWDO copy"
    ),
    "restart": "v2 threshold/xland-source identity plus adapter-owned committed gwdo_calls counter",
    "limitations": list(_LIMITATIONS),
}
CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256 = _json_digest(_ADAPTER_AUTHORITY)


def _verify_checkout_root(root: Path) -> None:
    for relative, expected in ARWEN_SOURCE_MANIFEST.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"pinned Arwen source is missing: {path}")
        actual = _digest_file(path)
        if actual != expected:
            raise ValueError(
                f"pinned Arwen source hash mismatch for {relative}: "
                f"{actual} != {expected}"
            )


    surface = sha256()
    for relative in ("gpuwm/core/mpas_column_batch.py", "docs/mpas-seam.md"):
        surface.update((root / relative).read_bytes())
    actual_surface = surface.hexdigest()
    if actual_surface != MPAS_SEAM_CONTRACT_SURFACE_SHA256:
        raise ValueError(
            "pinned Arwen contract surface hash mismatch: "
            f"{actual_surface} != {MPAS_SEAM_CONTRACT_SURFACE_SHA256}"
        )


def _load_pinned_arwen_factory(
    checkout: str | Path | None,
) -> tuple[Callable[..., Any], Path]:
    if checkout is not None:
        requested = Path(checkout).resolve()
        _verify_checkout_root(requested)
        already = sys.modules.get("gpuwm")
        if already is not None:
            loaded = Path(inspect.getfile(already)).resolve().parent.parent
            if loaded != requested:
                raise RuntimeError(
                    "gpuwm was already imported from a different tree; exact frozen Arwen v2 "
                    "Arwen cannot replace a live package"
                )
        elif str(requested) not in sys.path:
            sys.path.insert(0, str(requested))

    import gpuwm.core.microphysics as microphysics
    import gpuwm.core.mpas_column_batch as column_batch
    import gpuwm.core.noahmp_kernel_sources as noahmp_kernel_sources
    import gpuwm.core.physics as physics

    root = Path(inspect.getfile(column_batch)).resolve().parents[2]
    _verify_checkout_root(root)
    factory = physics.run_mpas_column_batch
    if factory is not column_batch.run_mpas_column_batch:
        raise ValueError("Arwen published factory is not the pinned column-batch object")
    if column_batch.MpasColumnBatchPhysics._PHASE1_ORCHESTRATION is not physics.PhysicsDriver.compute:
        raise ValueError("Arwen phase one is no longer PhysicsDriver.compute")
    if column_batch.MpasColumnBatchPhysics._PHASE2_MICROPHYSICS is not microphysics.apply:
        raise ValueError("Arwen phase two is no longer microphysics.apply")
    composed = noahmp_kernel_sources.translation_unit_source("noahmp_glacier")
    composed_sha256 = sha256(composed.encode("ascii")).hexdigest()
    if composed_sha256 != ARWEN_GLACIER_COMPOSED_TU_SHA256:
        raise ValueError(
            "Arwen composed glacier translation unit changed: "
            f"{composed_sha256} != {ARWEN_GLACIER_COMPOSED_TU_SHA256}"
        )
    return factory, root


def pin_arwen_physics_v841(checkout: str | Path) -> Mapping[str, Any]:
    """Pin exact Arwen v2 before KernelCache imports any other gpuwm tree."""

    factory, root = _load_pinned_arwen_factory(checkout)
    return MappingProxyType(
        {
            "arwen_commit": ARWEN_BUILD_COMMIT,
            "root": str(root),
            "source_manifest": dict(ARWEN_SOURCE_MANIFEST),
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
            "factory_module": factory.__module__,
            "factory_name": factory.__name__,
            "must_precede": "MPAS KernelCache construction",
        }
    )


def _snapshot_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(_snapshot_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_snapshot_nbytes(item) for item in value)
    return 0


def _scalar_names(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(name).strip().lower() for name in names)
    if result != WSM6_SCALAR_NAMES:
        raise ValueError(f"exact WSM6 scalar order is {WSM6_SCALAR_NAMES}, got {result}")
    return result


@dataclass(frozen=True, slots=True)
class CudaArwenDiagnosticSnapshotV841:
    """Detached resident diagnostics from one committed frozen Arwen v2 boundary."""

    surface: Mapping[str, Any]
    soil: Mapping[str, Any]
    precipitation: Mapping[str, Any]
    gwdo: Mapping[str, Any]
    metadata: Mapping[str, Any]
    receipt: Mapping[str, Any]

class PersistentTwoPhaseCudaPhysicsBackendV841:
    """Transactional frozen-v2 Arwen aggregate implementing the MPAS protocol."""

    def __init__(
        self,
        *,
        constructor: SealedArwenConstructorV841,
        prep_geometry: CudaMpasToPhysGeometryV841,
        kernel_cache: KernelCache,
        gwdo_static: CudaYsuGwdoStaticV841 | None = None,
        gwdo_kernel_cache: KernelCache | None = None,
        arwen_checkout: str | Path | None = None,
    ) -> None:
        if not isinstance(constructor, SealedArwenConstructorV841) or constructor._seal is not _CONSTRUCTOR_SEAL:
            raise TypeError("constructor must be SealedArwenConstructorV841.from_mapping")
        if not isinstance(prep_geometry, CudaMpasToPhysGeometryV841):
            raise TypeError("prep_geometry must be sealed v8.4.1 MPAS preparation geometry")
        prep_geometry.validate()
        if prep_geometry.n_cells != constructor.n_columns:
            raise ValueError("preparation geometry and Arwen constructor column counts differ")
        if gwdo_static is not None:
            if not isinstance(gwdo_static, CudaYsuGwdoStaticV841):
                raise TypeError("gwdo_static must be a sealed CudaYsuGwdoStaticV841")
            gwdo_static.validate()
            if gwdo_static.n_cells != constructor.n_columns:
                raise ValueError("GWDO statics and Arwen constructor column counts differ")
            if gwdo_kernel_cache is None:
                raise ValueError("GWDO activity requires gwdo_kernel_cache")
        elif gwdo_kernel_cache is not None:
            raise ValueError("gwdo_kernel_cache requires gwdo_static")
        factory, root = _load_pinned_arwen_factory(arwen_checkout)
        self._constructor = constructor
        self._prep_geometry = prep_geometry
        self._kernel_cache = kernel_cache
        self._gwdo_static = gwdo_static
        self._gwdo_kernel_cache = gwdo_kernel_cache
        self._factory = factory
        self._arwen_root = root
        self._seam = self._new_seam()
        self._phase = "boundary"
        self._boundary_snapshot: Mapping[str, Any] | None = None
        self._step_start: float | None = None
        self._candidate_scalar_target: Any | None = None
        self._candidate_scalar_backup: Any | None = None
        self._pending_gwdo_result: CudaYsuGwdoResultV841 | None = None
        self._last_gwdo_result: CudaYsuGwdoResultV841 | None = None
        # The one-frame refl10cm handoff (WRF diagflag semantics): staged by
        # a due finish_step, published by commit_step, consumed exactly once
        # by the history capture.  Never restart state -- the field is
        # recomputed by the next due microphysics call.
        self._pending_refl10cm: Any | None = None
        self._committed_refl10cm: Any | None = None
        self._gwdo_calls = 0
        # True once a GF advective-forcing carrier has been consumed; a later
        # None then means the runner regressed to the zero lanes.
        self._gf_forcing_seen = False
        self._last_receipt: dict[str, Any] = self._base_receipt()
        self._private_binding_guard()

    @property
    def contract_sha256(self) -> str:
        # This is the contract consumed by CudaRawColumnPhysicsV841 and the
        # existing driver protocol.  The adapter-specific digest is carried in
        # step_receipt so the two independently frozen contracts remain named.
        return CUDA_PHYSICS_V841_CONTRACT_SHA256

    def _new_seam(self) -> Any:
        seam = self._factory(**self._constructor.arwen_kwargs())
        expected = {
            "run_phase1",
            "run_phase2",
            "export_state",
            "restore_state",
            "accumulated_precipitation",
        }
        missing = sorted(name for name in expected if not hasattr(seam, name))
        if missing:
            raise TypeError(f"pinned Arwen seam lacks required methods {missing}")
        public_receipts = ("surface_classification", "last_noahmp_census")
        missing_receipts = sorted(
            name for name in public_receipts if not hasattr(seam, name)
        )
        if missing_receipts:
            raise TypeError(
                f"pinned Arwen seam lacks v2 public receipts {missing_receipts}"
            )
        self._validate_surface_classification(seam)
        return seam

    def _validate_surface_classification(
        self, seam: Any | None = None
    ) -> Mapping[str, Any]:
        selected = self._seam if seam is None else seam
        actual = getattr(selected, "surface_classification", None)
        if not isinstance(actual, Mapping):
            raise TypeError("Arwen v2 surface_classification is not a mapping")
        expected = dict(self._constructor.expected_surface_classification())
        normalized = dict(actual)
        if normalized != expected:
            raise ValueError(
                "Arwen v2 surface classification differs from sealed native "
                f"constructor authority: {normalized} != {expected}"
            )
        return MappingProxyType(normalized)

    def _validate_noahmp_census(
        self, *, require: bool
    ) -> Mapping[str, Any] | None:
        raw = getattr(self._seam, "last_noahmp_census", None)
        if raw is None:
            if require:
                raise ValueError("Arwen v2 did not publish a NoahMP execution census")
            return None
        if not isinstance(raw, Mapping):
            raise TypeError("Arwen v2 last_noahmp_census is not a mapping")
        classification = dict(self._constructor.expected_surface_classification())
        expected: dict[str, Any] = {
            "land": classification["sflx_land_columns"],
            "water": classification["open_water_columns"],
            "sea_ice": classification["sea_ice_columns"],
            "glacier": classification["glacier_columns"],
        }
        if classification["glacier_columns"]:
            expected["glacier_path"] = _GLACIER_CUDA_PROVENANCE
        normalized = dict(raw)
        if normalized != expected:
            raise ValueError(
                "Arwen NoahMP census/provenance differs from sealed constructor "
                f"authority: {normalized} != {expected}"
            )
        return MappingProxyType(normalized)

    def _private_binding_guard(self) -> None:
        # Effective radii are persisted in frozen Arwen v2 but not public.  This guard
        # scopes the one private read to the exact source bytes and class.
        _verify_checkout_root(self._arwen_root)
        if type(self._seam).__module__ != "gpuwm.core.mpas_column_batch":
            raise TypeError("private radius binding requires the exact frozen Arwen v2 seam class")
        state = getattr(self._seam, "_state", None)
        for name in ("effc", "effi", "effs"):
            if state is None or not hasattr(state, name):
                raise TypeError(f"pinned fa35 private radius carrier {name!r} is missing")

    def _base_receipt(self) -> dict[str, Any]:
        return {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "adapter_contract_sha256": CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
            "coupling_contract_sha256": CUDA_PHYSICS_V841_CONTRACT_SHA256,
            "contract_document_sha256": MPAS_SEAM_CONTRACT_SHA256,
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
            "arwen_commit": ARWEN_BUILD_COMMIT,
            "arwen_source_manifest": dict(ARWEN_SOURCE_MANIFEST),
            "dependencies": {
                "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
                "prep_kernel_sha256": CUDA_PHYSICS_PREP_V841_KERNEL_SHA256,
                "gwdo_contract_sha256": CUDA_GWDO_V841_CONTRACT_SHA256,
                "gwdo_kernel_sha256": CUDA_GWDO_V841_KERNEL_SHA256,
                "coupling_kernel_sha256": CUDA_PHYSICS_V841_KERNEL_SHA256,
            },
            "constructor": dict(self._constructor.receipt()),
            "surface_classification": dict(
                self._validate_surface_classification()
            ),
            "last_noahmp_census": (
                None
                if self._validate_noahmp_census(require=False) is None
                else dict(self._validate_noahmp_census(require=False))
            ),
            "phase": self._phase if hasattr(self, "_phase") else "constructing",
            "h_diabatic": {
                "reported_by_arwen": True,
                "applied": False,
                "replayed": False,
                "policy": "MPAS explicitly declines the ARW RK replay",
            },
            "gwdo": {"enabled": self._gwdo_static is not None},
            "limitations": list(_LIMITATIONS),
            "nonclaims": [
                "no claim of separate EOS cloud-pressure routing inside frozen Arwen v2",
                "no claim of exact MPAS t2_p routing inside frozen Arwen v2 RRTMG",
                "no claim that legacy RRTMG is device-resident",
            ],
        }

    def _validate_dt(self, dt: float) -> None:
        if float(dt) != self._constructor.dt:
            raise ValueError(
                f"backend dt={dt} does not equal sealed constructor dt="
                f"{self._constructor.dt}"
            )

    def _validate_phase1_output(self, output: Any, *, time_seconds: float) -> None:
        nlev, ncol = self._constructor.n_levels, self._constructor.n_columns
        shape = (nlev, ncol)
        for name in (
            "du",
            "dv",
            "dtheta",
            "dqv",
            "dqc",
            "dqr",
            "dqi",
            "dqs",
            "dqg",
            "h_diabatic",
        ):
            require_resident_array(
                f"arwen_phase1.{name}", getattr(output, name), dtype=np.float32, shape=shape
            )
        if float(output.elapsed_seconds) != float(time_seconds):
            raise ValueError("Arwen held-output time does not equal MPAS candidate start")
        if int(output.step_index) != int(self._seam.step_index):
            raise ValueError("Arwen held-output step index changed during phase one")

    def _restore_boundary(self, snapshot: Mapping[str, Any]) -> None:
        try:
            fresh = self._new_seam()
            fresh.restore_state(snapshot)
            self._seam = fresh
            self._private_binding_guard()
        except Exception as error:
            self._phase = "broken"
            raise RuntimeError("failed to reconstruct the pinned Arwen transaction") from error
        self._phase = "boundary"
        self._boundary_snapshot = None
        self._step_start = None
        self._candidate_scalar_target = None
        self._candidate_scalar_backup = None
        self._pending_gwdo_result = None
        self._pending_refl10cm = None

    def _gf_dynamics_lanes(
        self, carrier: Any, *, start: float, dt: float
    ) -> tuple[Any, Any]:
        """Validate and unpack the previous step's GF advective forcing.

        Naming the breakage this refuses: handing GF the CURRENT step's
        rthdynten (or any other step's) silently feeds the scheme forcing it
        never saw in native, which moves the closure family that decides
        convective mass flux.  A stale or mislabelled carrier is a wrong
        answer that still runs, so the clock is checked, not assumed.
        """

        if carrier is None:
            # Native's tend_physics is zero before the first dynamics step
            # forms it, so zero lanes ARE native at a cold start, and a
            # restart resume has no retained carrier either.  What is NOT
            # allowed is a runner that fed the lane and then stopped: that
            # is a mid-run regression to the pre-parity zeros, which is the
            # exact defect this seam closes.
            if self._gf_forcing_seen:
                raise ValueError(
                    "GF dynamics forcing vanished mid-run: this adapter "
                    "already consumed a rthdynten/rqvdynten carrier, so "
                    "omitting it now would silently restore the pre-parity "
                    "zero-forcing lanes"
                )
            return None, None
        self._gf_forcing_seen = True
        # The driver stamps the carrier with the ENDPOINT time of the step
        # that produced it, which is exactly this step's start.
        held = float(getattr(carrier, "time_seconds", float("nan")))
        if not math.isfinite(held) or held != start:
            raise ValueError(
                "GF dynamics forcing must come from the PREVIOUS dynamics "
                f"step, whose endpoint is this step's start t={start} s; "
                f"got a carrier stamped t={held} s"
            )
        rthdynten = getattr(carrier, "rthdynten", None)
        rqvdynten = getattr(carrier, "rqvdynten", None)
        if rthdynten is None or rqvdynten is None:
            raise TypeError(
                "GF dynamics forcing carrier must publish rthdynten/rqvdynten"
            )
        nlev = self._constructor.n_levels
        ncol = self._constructor.n_columns
        for name, value in (("rthdynten", rthdynten), ("rqvdynten", rqvdynten)):
            if tuple(value.shape) != (nlev, ncol):
                raise ValueError(
                    f"GF {name} must be [level, cell]={(nlev, ncol)}, "
                    f"got {tuple(value.shape)}"
                )
        return rthdynten, rqvdynten

    def begin_step(
        self,
        *,
        atmosphere: Any,
        scalar_names: Sequence[str],
        dt: float,
        dynamics_tendencies: Any = None,
    ) -> CudaRawColumnPhysicsV841:
        """Prepare MPAS columns and invoke frozen Arwen v2 phase one exactly once.

        ``dynamics_tendencies`` is the PREVIOUS step's driver-owned
        :class:`CudaV841GfDynamicsTendencies` carrier -- GF's RTHFTEN/RQVFTEN
        advective forcing.  Native MPAS v8.4.1 forms rthdynten/rqvdynten at
        the end of the dynamics step and the next physics call consumes them
        (mpas_atm_time_integration.F:6936 + :2789), so the carrier handed here
        must be the one produced by the step before this one.  ``None`` is
        the first step only: native's own tend_physics starts at zero.
        """

        if self._phase != "boundary":
            raise RuntimeError("begin_step requires a clean step boundary")
        _scalar_names(scalar_names)
        self._validate_dt(dt)
        start = float(atmosphere.state.time_seconds)
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("candidate start time must be finite and non-negative")
        if float(self._seam.elapsed_seconds) != start:
            raise ValueError(
                "Arwen and MPAS clocks differ at begin_step: "
                f"{self._seam.elapsed_seconds} != {start}"
            )
        snapshot = self._seam.export_state()
        self._boundary_snapshot = snapshot
        self._step_start = start
        snapshot_bytes = _snapshot_nbytes(snapshot)
        self._pending_gwdo_result = None
        try:
            prepared = prepare_mpas_to_phys_cuda_v841(
                atmosphere,
                scalar_names=scalar_names,
                geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache,
                post_rk_wsm6=False,
            )
            if float(prepared.time_seconds) != start:
                raise ValueError("phase-one prep time changed from the exact MPAS start")
            rthdynten, rqvdynten = self._gf_dynamics_lanes(
                dynamics_tendencies, start=start, dt=dt
            )
            try:
                output = self._seam.run_phase1(
                    dt=dt,
                    u=prepared.u_p,
                    v=prepared.v_p,
                    theta=prepared.th_p,
                    pressure=prepared.pres_hyd_p,
                    pressure_interface=prepared.pres2_hyd_p,
                    z_interface=prepared.z_p,
                    w=prepared.w_p,
                    rho_dry=prepared.rho_dry,
                    qv=prepared.qv_p,
                    qc=prepared.qc_p,
                    qr=prepared.qr_p,
                    qi=prepared.qi_p,
                    qs=prepared.qs_p,
                    qg=prepared.qg_p,
                    exner=prepared.pi_p,
                    rthdynten=rthdynten,
                    rqvdynten=rqvdynten,
                )
            except FloatingPointError as error:
                # The sealed seam refuses by scheme and by field, but it has
                # no way to say WHICH column produced it: it is handed the
                # whole aggregate and it validates the aggregate.  Locating
                # the column is the difference between "the physics blew up"
                # and a sentence a reader can act on, and it costs a passing
                # step nothing because it runs only on this path.
                raise FloatingPointError(
                    f"{error}.  " + _locate_degenerate_columns(prepared)
                ) from error
            self._validate_phase1_output(output, time_seconds=start)
            surface_classification = self._validate_surface_classification()
            noahmp_census = self._validate_noahmp_census(require=True)
            du, dv = output.du, output.dv
            gwdo_receipt: Mapping[str, Any] | None = None
            gwdo_validation_bytes = 0
            gwdo = None
            if self._gwdo_static is not None:
                view = CudaYsuGwdoColumnViewV841.from_prepared(prepared)
                gwdo_input_du, gwdo_input_dv = output.du, output.dv
                gwdo = run_bl_ysu_gwdo_cuda_v841(
                    view,
                    rublten=gwdo_input_du,
                    rvblten=gwdo_input_dv,
                    static=self._gwdo_static,
                    dt_seconds=dt,
                    kernel_cache=self._gwdo_kernel_cache,
                )
                # Native GWDO returns the already-composed YSU+GWD tendency.
                # Adding dtau again here would double count the operator.
                du, dv = gwdo.rublten, gwdo.rvblten
                gwdo_receipt = gwdo.receipt()
                gwdo_validation_bytes = int(gwdo.validation_d2h.bytes)
                self._pending_gwdo_result = gwdo
            if gwdo is None:
                execution_provenance = (
                    CudaPhaseOneExecutionProvenanceV841.arwen_gwd_off(
                        aggregate_executed=True
                    )
                )
            else:
                execution_provenance = (
                    CudaPhaseOneExecutionProvenanceV841.arwen_with_external_gwdo(
                        aggregate_executed=True,
                        gwdo_executed=True,
                        gwdo_composed_once=True,
                        gwdo_result_module=type(gwdo).__module__,
                        gwdo_result_class=type(gwdo).__name__,
                        gwdo_contract_sha256=gwdo.contract_sha256,
                        gwdo_kernel_sha256=CUDA_GWDO_V841_KERNEL_SHA256,
                        gwdo_validation_d2h=gwdo.validation_d2h,
                        gwdo_input_du_is_arwen_output=(
                            gwdo_input_du is output.du
                        ),
                        gwdo_input_dv_is_arwen_output=(
                            gwdo_input_dv is output.dv
                        ),
                        raw_du_is_gwdo_output=(du is gwdo.rublten),
                        raw_dv_is_gwdo_output=(dv is gwdo.rvblten),
                    )
                )
            raw = CudaRawColumnPhysicsV841(
                du=du,
                dv=dv,
                dtheta=output.dtheta,
                dscalars={
                    "qv": output.dqv,
                    "qc": output.dqc,
                    "qr": output.dqr,
                    "qi": output.dqi,
                    "qs": output.dqs,
                    "qg": output.dqg,
                },
                time_seconds=start,
                execution_provenance=execution_provenance,
            )
            raw.validate(
                n_vert_levels=self._constructor.n_levels,
                n_cells=self._constructor.n_columns,
            )
            self._phase = "begun"
            self._last_receipt = {
                **self._base_receipt(),
                "phase": "begun",
                "start_time_seconds": start,
                "end_time_seconds": start + self._constructor.dt,
                "arwen_step_index": int(output.step_index),
                "surface_classification": dict(surface_classification),
                "noahmp_census": dict(noahmp_census),
                "cadence": {
                    "radiation_ran": bool(output.radiation_ran),
                    "surface_pbl_ran": bool(output.surface_pbl_ran),
                    "cumulus_ran": bool(output.cumulus_ran),
                    "call_counts": dict(self._seam.call_counts),
                },
                "validation_d2h_bytes": {
                    "prep": int(prepared.validation_d2h.bytes),
                    "gwdo": gwdo_validation_bytes,
                    "transaction_boundary_snapshot": snapshot_bytes,
                },
                "copies": {
                    "arwen_phase1": "published frozen Arwen v2 persistent input/output copies",
                    "transaction_boundary_snapshot_d2h_bytes": snapshot_bytes,
                    "gwdo_candidate_outputs": self._gwdo_static is not None,
                },
                "gwdo": {
                    "enabled": self._gwdo_static is not None,
                    "composed_once": self._gwdo_static is not None,
                    "receipt": None if gwdo_receipt is None else dict(gwdo_receipt),
                },
                "execution_provenance": execution_provenance.receipt(),
            }
            return raw
        except Exception:
            self._restore_boundary(snapshot)
            raise

    def _private_radii(self) -> tuple[Any, Any, Any]:
        self._private_binding_guard()
        cp = __import__("cupy")
        state = self._seam._state
        shape = (self._constructor.n_levels, self._constructor.n_columns)
        result = []
        for name in ("effc", "effi", "effs"):
            value = getattr(state, name).reshape(shape)
            require_resident_array(
                f"fa35_private.{name}", value, dtype=np.float32, shape=shape
            )
            result.append(cp.array(value, copy=True, order="C"))
        return tuple(result)

    def finish_step(
        self,
        *,
        atmosphere: Any,
        scalar_names: Sequence[str],
        dt: float,
        refl_10cm_due: bool = False,
    ) -> CudaPostRkWsm6UpdateV841:
        """Invoke frozen Arwen v2 WSM6 once on the clamped endpoint and seal its outputs.

        ``refl_10cm_due`` is WRF's history-step ``diagflag`` carried to the
        seam: the due step's microphysics computes ``refl10cm`` from its
        post-call temperature and unchanged prepared pressure (native MPAS-A
        v8.4.1 computes the history field at exactly this point), and the
        staged copy is published by ``commit_step`` for exactly one
        ``take_history_refl10cm`` consumer.
        """

        if self._phase != "begun" or self._boundary_snapshot is None:
            raise RuntimeError("finish_step requires one successful begin_step")
        snapshot = self._boundary_snapshot
        try:
            _scalar_names(scalar_names)
            self._validate_dt(dt)
            start = float(self._step_start)
            endpoint = float(atmosphere.state.time_seconds)
            if endpoint != start + self._constructor.dt:
                raise ValueError(
                    "post-RK candidate time must equal the exact step endpoint: "
                    f"{endpoint} != {start + self._constructor.dt}"
                )
        except Exception:
            prior = self._last_receipt
            self._restore_boundary(snapshot)
            self._last_receipt = {
                **prior,
                "phase": "automatic_rollback",
                "rollback": "pre-phase-two validation refused; boundary restored",
            }
            raise
        cp = __import__("cupy")
        # WSM6 receives zero-copy scalar aliases.  This device backup is the
        # transaction guard that restores the unpublished MPAS candidate if
        # adaptation/diagnostic validation fails after the in-place call.
        scalar_backup = cp.array(atmosphere.state.scalars, copy=True, order="C")
        try:
            prepared = prepare_mpas_to_phys_cuda_v841(
                atmosphere,
                scalar_names=scalar_names,
                geometry=self._prep_geometry,
                kernel_cache=self._kernel_cache,
                post_rk_wsm6=True,
            )
            if float(prepared.time_seconds) != endpoint:
                raise ValueError("phase-two prep time changed from the exact endpoint")
            wsm6 = prepared.wsm6_input_view()
            receipt = self._seam.run_phase2(
                theta=prepared.th_p,
                qv=wsm6.qv,
                qc=wsm6.qc,
                qr=wsm6.qr,
                qi=wsm6.qi,
                qs=wsm6.qs,
                qg=wsm6.qg,
                pressure=prepared.pres_p,
                rho_dry=prepared.rho_dry,
                z_interface=prepared.z_p,
                refl_10cm_due=bool(refl_10cm_due),
            )
            if float(self._seam.elapsed_seconds) != endpoint:
                raise ValueError("Arwen phase two did not advance to the MPAS endpoint")
            staged_refl = None
            if refl_10cm_due:
                staged_refl = receipt.get("refl_10cm")
                require_resident_array(
                    "history_refl10cm",
                    staged_refl,
                    dtype=np.float32,
                    shape=(
                        self._constructor.n_levels,
                        self._constructor.n_columns,
                    ),
                )
            cumulative = self._seam.accumulated_precipitation()
            required = {"RAINNC", "SNOWNC", "GRAUPELNC", "RAINC"}
            if set(cumulative) != required:
                raise ValueError(
                    f"frozen Arwen v2 cumulative precipitation keys changed: {sorted(cumulative)}"
                )
            effc, effi, effs = self._private_radii()
            update = CudaPostRkWsm6UpdateV841(
                theta=prepared.th_p,
                qv=wsm6.qv,
                qc=wsm6.qc,
                qr=wsm6.qr,
                qi=wsm6.qi,
                qs=wsm6.qs,
                qg=wsm6.qg,
                rainnc=cumulative["RAINNC"],
                rainncv=receipt["rainncv"],
                snownc=cumulative["SNOWNC"],
                snowncv=receipt["snowncv"],
                graupelnc=cumulative["GRAUPELNC"],
                graupelncv=receipt["graupelncv"],
                sr=receipt["sr"],
                effc=effc,
                effi=effi,
                effs=effs,
                time_seconds=endpoint,
            )
            update.validate(
                n_vert_levels=self._constructor.n_levels,
                n_cells=self._constructor.n_columns,
            )
            self._phase = "finished"
            self._pending_refl10cm = staged_refl
            self._candidate_scalar_target = atmosphere.state.scalars
            self._candidate_scalar_backup = scalar_backup
            prior = self._last_receipt
            self._last_receipt = {
                **self._base_receipt(),
                "phase": "finished_unpublished",
                "start_time_seconds": start,
                "end_time_seconds": endpoint,
                "publication": {
                    "state": "finished_unpublished",
                    "requires": "commit_step after MPAS recovery/driver commit",
                },
                "arwen_step_index": int(self._seam.step_index),
                "cadence": prior.get("cadence", {}),
                "gwdo": prior.get("gwdo", {"enabled": False}),
                "validation_d2h_bytes": {
                    **prior.get("validation_d2h_bytes", {}),
                    "post_rk_prep": int(prepared.validation_d2h.bytes),
                },
                "copies": {
                    **prior.get("copies", {}),
                    "candidate_scalar_transaction_backup_d2d_bytes": int(
                        scalar_backup.nbytes
                    ),
                    "effective_radius_snapshot_d2d_bytes": int(
                        effc.nbytes + effi.nbytes + effs.nbytes
                    ),
                    "cumulative_precipitation": "frozen Arwen v2 public device copies",
                },
                "post_rk": {
                    "in_place_species": list(WSM6_SCALAR_NAMES),
                    "refl_10cm_due": bool(refl_10cm_due),
                    "theta": "prepared th_p dry theta",
                    "pressure": "prepared pres_p EOS",
                    "rho": "prepared rho_dry",
                    "z_interface": "prepared z_p",
                    "cumulative_fields": ["rainnc", "snownc", "graupelnc"],
                    "increment_fields": [
                        "rainncv",
                        "snowncv",
                        "graupelncv",
                        "sr",
                    ],
                    "private_exact_hash_binding": ["effc", "effi", "effs"],
                },
            }
            return update
        except Exception:
            scalar_restore_error = None
            try:
                atmosphere.state.scalars[...] = scalar_backup
            except Exception as error:
                scalar_restore_error = error
            prior = self._last_receipt
            self._restore_boundary(snapshot)
            self._last_receipt = {
                **prior,
                "phase": "automatic_rollback",
                "rollback": "phase-two execution refused; candidate and boundary restored",
            }
            if scalar_restore_error is not None:
                raise RuntimeError(
                    "failed to restore unpublished MPAS candidate scalars"
                ) from scalar_restore_error
            raise

    def abort_step(self) -> None:
        """Rollback a begun or finished-unpublished cross-component step."""

        if (
            self._phase not in ("begun", "finished")
            or self._boundary_snapshot is None
        ):
            raise RuntimeError("abort_step requires a begun or finished transaction")
        snapshot = self._boundary_snapshot
        scalar_restore_error = None
        if self._phase == "finished":
            try:
                self._candidate_scalar_target[...] = self._candidate_scalar_backup
            except Exception as error:
                scalar_restore_error = error
        prior = self._last_receipt
        self._restore_boundary(snapshot)
        self._last_receipt = {
            **prior,
            "phase": "rolled_back",
            "rollback": "fresh frozen Arwen v2 seam reconstructed from boundary export",
        }
        if scalar_restore_error is not None:
            raise RuntimeError(
                "Arwen rolled back but unpublished MPAS scalar restoration failed"
            ) from scalar_restore_error

    def commit_step(self) -> None:
        """Publish a finished seam only after MPAS recovery/driver commit succeeds."""

        if self._phase != "finished" or self._boundary_snapshot is None:
            raise RuntimeError("commit_step requires a finished-unpublished transaction")
        if (
            self._candidate_scalar_target is None
            or self._candidate_scalar_backup is None
        ):
            snapshot = self._boundary_snapshot
            self._restore_boundary(snapshot)
            raise RuntimeError("finished transaction lost its MPAS scalar rollback guard")
        if self._gwdo_static is not None and self._pending_gwdo_result is None:
            snapshot = self._boundary_snapshot
            self._restore_boundary(snapshot)
            raise RuntimeError("finished transaction lost its validated GWDO result")
        if self._pending_gwdo_result is not None:
            self._last_gwdo_result = self._pending_gwdo_result
            self._gwdo_calls += 1
        self._pending_gwdo_result = None
        if self._pending_refl10cm is not None:
            self._committed_refl10cm = self._pending_refl10cm
        self._pending_refl10cm = None
        prior = self._last_receipt
        self._phase = "boundary"
        self._boundary_snapshot = None
        self._step_start = None
        self._candidate_scalar_target = None
        self._candidate_scalar_backup = None
        self._last_receipt = {
            **prior,
            "phase": "complete",
            "publication": {
                "state": "committed",
                "committed_after": "MPAS recovery/driver candidate commit",
            },
            "gwdo": {
                **prior.get("gwdo", {"enabled": False}),
                "committed_calls": self._gwdo_calls,
            },
        }

    def take_history_refl10cm(self) -> Any | None:
        """Consume the committed one-frame ``refl10cm`` exactly once.

        Legal only at a committed boundary, mirroring the D2 handoff rule the
        engine applies to its own output frames: the capture that writes the
        history file is the single consumer, and a second read without a new
        due step gets ``None`` rather than a stale frame.
        """

        if self._phase != "boundary":
            raise RuntimeError(
                "take_history_refl10cm is legal only at a committed boundary"
            )
        refl = self._committed_refl10cm
        self._committed_refl10cm = None
        return refl

    def diagnostic_snapshot(self) -> CudaArwenDiagnosticSnapshotV841:
        """Snapshot public frozen-v2 diagnostics at a committed boundary only."""

        if self._phase != "boundary":
            raise RuntimeError(
                "diagnostic_snapshot is legal only at a committed boundary"
            )
        self._private_binding_guard()
        exported = self._seam.export_state()
        if (
            not isinstance(exported, Mapping)
            or set(exported) != {"identity", "arrays", "scalars"}
            or not isinstance(exported["arrays"], Mapping)
        ):
            raise ValueError("frozen Arwen v2 public export_state schema changed")
        cp = __import__("cupy")
        nlev = self._constructor.n_levels
        ncol = self._constructor.n_columns
        export_arrays = exported["arrays"]
        selected: dict[str, Any] = {}
        selected_hashes: dict[str, Any] = {}
        selected_h2d_bytes = 0
        names = (*_REQUIRED_ARWEN_EXPORT_FIELDS, *_OPTIONAL_ARWEN_EXPORT_FIELDS)
        for name in names:
            key = f"fields/{name}"
            if key not in export_arrays:
                if name in _REQUIRED_ARWEN_EXPORT_FIELDS:
                    raise ValueError(f"frozen Arwen v2 public export lacks required {key!r}")
                continue
            host = np.asarray(export_arrays[key])
            if host.dtype != np.dtype(np.float32):
                raise TypeError(f"frozen Arwen v2 public export {key!r} is not FP32")
            if name in _SOIL_DIAGNOSTIC_FIELDS:
                if host.shape == (4, 1, ncol):
                    normalized = host[:, 0, :]
                elif host.shape == (4, ncol):
                    normalized = host
                else:
                    raise ValueError(
                        f"frozen Arwen v2 public export {key!r} has shape {host.shape}; "
                        f"expected {(4, 1, ncol)} or {(4, ncol)}"
                    )
                expected_shape = (4, ncol)
            else:
                if host.shape == (1, 1, ncol):
                    normalized = host[0, 0, :]
                elif host.shape == (1, ncol):
                    normalized = host[0, :]
                elif host.shape == (ncol,):
                    normalized = host
                else:
                    raise ValueError(
                        f"frozen Arwen v2 public export {key!r} has shape {host.shape}; "
                        f"expected a singleton-ny {(ncol,)} field"
                    )
                expected_shape = (ncol,)
            normalized = np.array(normalized, copy=True, order="C")
            resident = cp.asarray(normalized)
            require_resident_array(
                f"arwen_diagnostic.{name}",
                resident,
                dtype=np.float32,
                shape=expected_shape,
            )
            selected[name] = resident
            selected_hashes[key] = _array_identity(normalized)
            selected_h2d_bytes += int(resident.nbytes)

        public_precip = self._seam.accumulated_precipitation()
        precip_keys = {
            "rainc": "RAINC",
            "rainnc": "RAINNC",
            "snownc": "SNOWNC",
            "graupelnc": "GRAUPELNC",
        }
        if set(public_precip) != set(precip_keys.values()):
            raise ValueError("frozen Arwen v2 public precipitation inventory changed")
        precipitation: dict[str, Any] = {}
        for public_name, fa35_name in precip_keys.items():
            source = public_precip[fa35_name]
            require_resident_array(
                f"arwen_diagnostic.{public_name}",
                source,
                dtype=np.float32,
                shape=(ncol,),
            )
            precipitation[public_name] = cp.array(
                source, copy=True, order="C"
            )

        gwdo: dict[str, Any] = {}
        if self._last_gwdo_result is None:
            for name in _GWDO_SURFACE_FIELDS:
                gwdo[name] = cp.zeros((ncol,), dtype=cp.float32)
            for name in _GWDO_LEVEL_FIELDS:
                gwdo[name] = cp.zeros((nlev, ncol), dtype=cp.float32)
        else:
            self._last_gwdo_result.validate(
                n_vert_levels=nlev, n_cells=ncol
            )
            for name in _GWDO_DIAGNOSTIC_FIELDS:
                gwdo[name] = cp.array(
                    getattr(self._last_gwdo_result, name),
                    copy=True,
                    order="C",
                )

        surface = {
            name: selected[name]
            for name in names
            if name in selected and name not in _SOIL_DIAGNOSTIC_FIELDS
        }
        soil = {
            name: selected[name]
            for name in names
            if name in selected and name in _SOIL_DIAGNOSTIC_FIELDS
        }
        full_export_bytes = _snapshot_nbytes(exported)
        surface_classification = self._validate_surface_classification()
        noahmp_census = self._validate_noahmp_census(require=False)
        receipt = {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "boundary": "committed",
            "full_export_d2h_bytes": full_export_bytes,
            "full_export_array_inventory_sha256": _json_digest(
                sorted(export_arrays)
            ),
            "selected_export_inventory": [
                f"fields/{name}" for name in names if name in selected
            ],
            "selected_export_hashes": selected_hashes,
            "selected_export_h2d_bytes": selected_h2d_bytes,
            "precipitation_d2d_bytes": sum(
                int(value.nbytes) for value in precipitation.values()
            ),
            "gwdo_d2d_or_zero_fill_bytes": sum(
                int(value.nbytes) for value in gwdo.values()
            ),
            "surface_classification": dict(surface_classification),
            "last_noahmp_census": (
                None if noahmp_census is None else dict(noahmp_census)
            ),
            "q2_policy": "preserved bitwise; negative values are audit data",
        }
        return CudaArwenDiagnosticSnapshotV841(
            surface=MappingProxyType(surface),
            soil=MappingProxyType(soil),
            precipitation=MappingProxyType(precipitation),
            gwdo=MappingProxyType(gwdo),
            metadata=MappingProxyType(
                {
                    "step_index": int(self._seam.step_index),
                    "time_seconds": float(self._seam.elapsed_seconds),
                    "call_counts": MappingProxyType(dict(self._seam.call_counts)),
                    "surface_classification": surface_classification,
                    "last_noahmp_census": noahmp_census,
                    "gwdo_enabled": self._gwdo_static is not None,
                    "gwdo_calls": self._gwdo_calls,
                    "gwdo_has_last_result": self._last_gwdo_result is not None,
                }
            ),
            receipt=MappingProxyType(receipt),
        )

    def _restart_identity(self) -> Mapping[str, Any]:
        return {
            "adapter_contract_sha256": CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256,
            "arwen_commit": ARWEN_BUILD_COMMIT,
            "arwen_source_manifest": dict(ARWEN_SOURCE_MANIFEST),
            "contract_surface_sha256": MPAS_SEAM_CONTRACT_SURFACE_SHA256,
            "glacier_composed_tu_sha256": ARWEN_GLACIER_COMPOSED_TU_SHA256,
            "constructor_identity_sha256": self._constructor.identity_sha256,
            "prep_contract_sha256": CUDA_PHYSICS_PREP_V841_CONTRACT_SHA256,
            "gwdo_contract_sha256": (
                CUDA_GWDO_V841_CONTRACT_SHA256
                if self._gwdo_static is not None
                else None
            ),
        }

    def restart_state(self) -> Mapping[str, Any]:
        if self._phase != "boundary":
            raise RuntimeError("restart_state is legal only at a step boundary")
        return {
            "schema": CUDA_ARWEN_PHYSICS_V841_SCHEMA,
            "identity": self._restart_identity(),
            "seam": self._seam.export_state(),
            "adapter": {
                "gwdo_calls": self._gwdo_calls,
                "last_gwdo_result_persisted": False,
            },
        }

    def restore_restart_state(self, payload: Mapping[str, Any]) -> None:
        if self._phase != "boundary":
            raise RuntimeError("restore_restart_state is legal only at a step boundary")
        expected_keys = {"schema", "identity", "seam", "adapter"}
        if not isinstance(payload, Mapping) or set(payload) != expected_keys:
            raise ValueError(
                "backend restart payload must contain "
                "schema/identity/seam/adapter exactly"
            )
        expected = self._restart_identity()
        if payload["schema"] != CUDA_ARWEN_PHYSICS_V841_SCHEMA:
            raise ValueError("backend restart schema mismatch")
        if payload["identity"] != expected:
            raise ValueError("backend restart identity mismatch")
        adapter = payload["adapter"]
        if not isinstance(adapter, Mapping) or set(adapter) != {
            "gwdo_calls",
            "last_gwdo_result_persisted",
        }:
            raise ValueError("backend restart adapter metadata mismatch")
        calls = adapter["gwdo_calls"]
        if isinstance(calls, bool) or not isinstance(calls, (int, np.integer)):
            raise TypeError("restart gwdo_calls must be a non-negative integer")
        calls = int(calls)
        if calls < 0 or (self._gwdo_static is None and calls != 0):
            raise ValueError("restart gwdo_calls conflicts with GWDO identity")
        if adapter["last_gwdo_result_persisted"] is not False:
            raise ValueError("frozen Arwen v2 adapter restart does not persist trajectory-inert GWDO arrays")
        fresh = self._new_seam()
        fresh.restore_state(payload["seam"])
        self._seam = fresh
        self._gwdo_calls = calls
        self._pending_gwdo_result = None
        self._last_gwdo_result = None
        self._private_binding_guard()
        self._last_receipt = {
            **self._base_receipt(),
            "phase": "restored",
            "arwen_step_index": int(self._seam.step_index),
            "time_seconds": float(self._seam.elapsed_seconds),
            "gwdo": {
                "enabled": self._gwdo_static is not None,
                "committed_calls": self._gwdo_calls,
                "last_result_restored": False,
                "nonclaim": "trajectory-inert last diagnostics omitted; next phase1 replaces",
            },
        }

    def step_receipt(self) -> Mapping[str, Any]:
        # Detached JSON data: callers cannot mutate backend state through a
        # nested receipt mapping.
        return MappingProxyType(json.loads(json.dumps(self._last_receipt, sort_keys=True)))


__all__ = [
    "ARWEN_BUILD_COMMIT",
    "ARWEN_GLACIER_COMPOSED_TU_SHA256",
    "ARWEN_SOURCE_MANIFEST",
    "CUDA_ARWEN_PHYSICS_V841_CONTRACT_SHA256",
    "CUDA_ARWEN_PHYSICS_V841_SCHEMA",
    "MPAS_SEAM_CONTRACT_SHA256",
    "MPAS_SEAM_CONTRACT_SURFACE_SHA256",
    "CudaArwenDiagnosticSnapshotV841",
    "PersistentTwoPhaseCudaPhysicsBackendV841",
    "SealedArwenConstructorV841",
    "pin_arwen_physics_v841",
]
