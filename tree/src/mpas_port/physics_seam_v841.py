"""Additive MPAS-A v8.4.1 column-physics boundary deltas.

The established :mod:`mpas_port.physics_seam` contract is pinned to MPAS-A
v8.2.3.  This module does not relabel that contract and does not manufacture a
gpuwm runner.  It records and implements only the released v8.4.1 interface
changes that a future, real ``run_mpas_column_batch`` counterparty must honor:

* ``mp_top_level`` is initialized from ``rdzw``/``dzu``; mass-species
  tendencies and hydrometeor state are zero above that level, while the
  affected diagnostics are zeroed or reset to their released backgrounds;
* ``evapprod`` remains in MPAS ``[level, cell]`` order, matching the corrected
  ``evapprod_p(i,k,j) = evapprod(k,i)`` assignment;
* the three-dimensional ``refl10cm`` diagnostic is a required, named dBZ
  carrier for WSM6 and both Thompson variants;
* ``bl_ugwp_gwdo`` is a valid v8.4.1 MPAS selector but remains a named refusal
  because gpuwm publishes no gravity-wave physics slot; and
* ``pre_microphysics``/``post_microphysics`` are OpenACC transfer hooks only,
  not additional physical operators to reproduce in a resident backend.

Frozen source: official MPAS-Model v8.4.1 tag object
``2a934b5008a7446df96d550bf2e21466feaec686`` (commit
``91c5eac175eebeaf4206bacd5cb50c39dff3c152``), chiefly
``mpas_atmphys_init.F:276-284``, ``mpas_atmphys_interface.F:562-595,
701,857-1099,1141-1182``, ``mpas_atmphys_control.F:212-215``, and
``Registry.xml:2311-2314,2375-2380,2584-2598,2628-2634,3518-3522``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationRefusal
from .physics_seam import (
    GPUWMSchemeSelection,
    MPAS_SELECTOR_INVENTORY,
    resolve_mpas_physics,
)


FloatArray = NDArray[np.floating[Any]]
_SUPPORTED_RKINDS = (np.dtype(np.float32), np.dtype(np.float64))

V841_PHYSICS_SOURCE = (
    "MPAS-A v8.4.1 tag-object=2a934b5008a7446df96d550bf2e21466feaec686; "
    "commit=91c5eac175eebeaf4206bacd5cb50c39dff3c152; "
    "archive-sha256=772f565c2bd66999492085eff8ffa0b9aa9a2edd1e7f2c0e5d1a8bedc1160861"
)


@dataclass(frozen=True, slots=True)
class V841PhysicsFieldDelta:
    """One released field-contract change relative to the v8.2.3 seam."""

    dimensions: str
    units: str
    role: str
    authority: str


V841_PHYSICS_FIELD_DELTAS: Mapping[str, V841PhysicsFieldDelta] = MappingProxyType(
    {
        "mp_top_level": V841PhysicsFieldDelta(
            dimensions="scalar",
            units="-",
            role="highest one-based level where mass microphysics tendencies apply",
            authority="Registry.xml:3520-3521; mpas_atmphys_init.F:276-284",
        ),
        "evapprod": V841PhysicsFieldDelta(
            dimensions="nVertLevels nCells Time",
            units="s^{-1}",
            role="rain evaporation diagnostic in canonical [level,cell] order",
            authority="mpas_atmphys_interface.F:701,991; Registry.xml:2632-2634",
        ),
        "refl10cm": V841PhysicsFieldDelta(
            dimensions="nVertLevels nCells Time",
            units="dBZ",
            role="10 cm radar reflectivity for WSM6 and Thompson microphysics",
            authority="Registry.xml:2596-2598; mpas_atmphys_driver_microphysics.F:668-777",
        ),
    }
)


_v841_inventory = dict(MPAS_SELECTOR_INVENTORY)
_v841_inventory["config_gwdo_scheme"] = (
    *MPAS_SELECTOR_INVENTORY["config_gwdo_scheme"],
    "bl_ugwp_gwdo",
)
MPAS_V841_SELECTOR_INVENTORY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    _v841_inventory
)
del _v841_inventory


# These are the fields explicitly zeroed in the mass-tendency block at
# interface.F:1072-1099.  Thompson number tendencies are evaluated through
# ``kte`` at :1104-1130 and therefore must not be swept into this set.
V841_TOP_GATED_MASS_TENDENCIES = frozenset(("qv", "qc", "qr", "qi", "qs", "qg"))
V841_UNGATED_NUMBER_TENDENCIES = frozenset(("nc", "ni", "nr", "nifa", "nwfa"))
V841_TENDENCY_CARRIER_INVENTORY: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "mp_kessler": frozenset(("qv", "qc", "qr")),
        "mp_wsm6": frozenset(("qv", "qc", "qr", "qi", "qs", "qg")),
        "mp_thompson": frozenset(
            ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
        ),
        "mp_thompson_aerosols": frozenset(
            (
                "qv",
                "qc",
                "qr",
                "qi",
                "qs",
                "qg",
                "nc",
                "ni",
                "nr",
                "nifa",
                "nwfa",
            )
        ),
    }
)

# microphysics_to_MPAS zeroes these stored hydrometeors above mp_top_level,
# while deliberately leaving qv, theta_m, and the Thompson number/aerosol
# state alone.  Keeping state and tendency sets distinct captures that source
# asymmetry instead of extending the tendency rule by analogy.
V841_TOP_CLAMPED_MASS_STATE = frozenset(("qc", "qr", "qi", "qs", "qg"))
V841_TOP_UNCHANGED_STATE = frozenset(
    ("qv", "theta_m", "nc", "ni", "nr", "nifa", "nwfa")
)
V841_TOP_ZEROED_DIAGNOSTICS = frozenset(
    ("rt_diabatic_tend", "rainprod", "evapprod")
)
V841_TOP_BACKGROUND_DIAGNOSTICS = frozenset(("re_cloud", "re_ice", "re_snow"))
V841_RICH_TOP_POSTPROCESS_SCHEMES = frozenset(
    ("mp_wsm6", "mp_thompson", "mp_thompson_aerosols")
)

# mpas_atmphys_constants.F declares RKIND parameters from unsuffixed default-
# real literals.  The authority compiler's default-real mode controls literal
# interpretation: normal builds parse binary32, while ``-r8`` parses binary64
# directly.  Pin both supported RKIND builds instead of widening f32 to f64.
V841_EFFECTIVE_RADIUS_BACKGROUNDS: Mapping[
    str, Mapping[str, np.floating[Any]]
] = MappingProxyType(
    {
        "float32": MappingProxyType(
            {
                "re_cloud": np.float32(2.49e-6),
                "re_ice": np.float32(4.99e-6),
                "re_snow": np.float32(9.99e-6),
            }
        ),
        "float64": MappingProxyType(
            {
                "re_cloud": np.float64(2.49e-6),
                "re_ice": np.float64(4.99e-6),
                "re_snow": np.float64(9.99e-6),
            }
        ),
    }
)

# gpuwm/WRF uses these spellings in several leaf contracts.  An MPAS adapter
# must not be able to evade a mass-species gate merely by passing a WRF name.
V841_WRF_SCALAR_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "qvapor": "qv",
        "qcloud": "qc",
        "qrain": "qr",
        "qice": "qi",
        "qsnow": "qs",
        "qgraupel": "qg",
        "water_vapor": "qv",
        "cloud_water": "qc",
        "rain_water": "qr",
        "cloud_ice": "qi",
        "snow": "qs",
        "graupel": "qg",
    }
)


# The hooks contain no physics outside MPAS_OPENACC.  Publishing the exact
# direction and carrier list prevents an adapter from running them as a second
# pack/unpack or claiming their transfers as numerical work.
V841_OPENACC_TRANSFER_HOOKS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        "pre_microphysics": MappingProxyType(
            {
                "direction": "device_to_host",
                "transfer_only": True,
                "fields": (
                    "exner",
                    "pressure_base",
                    "pressure_p",
                    "rho_zz",
                    "theta_m",
                    "w",
                    "scalars",
                ),
            }
        ),
        "post_microphysics": MappingProxyType(
            {
                "direction": "host_to_device",
                "transfer_only": True,
                "fields": (
                    "exner",
                    "exner_base",
                    "pressure_base",
                    "pressure_p",
                    "rtheta_base",
                    "rtheta_p",
                    "rho_zz",
                    "theta_m",
                    "scalars",
                    "rt_diabatic_tend",
                ),
            }
        ),
    }
)


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def _canonical_microphysics_carrier(raw_name: object) -> str:
    name = str(raw_name).strip().lower()
    return V841_WRF_SCALAR_ALIASES.get(name, name)


def _validated_microphysics_scheme(scheme: object) -> str:
    normalized = str(scheme)
    if normalized != normalized.strip().lower():
        _refuse(
            "config_microp_scheme",
            scheme,
            "the resolved MPAS Registry spelling must be exact lowercase without surrounding whitespace",
            "an exact released lowercase scheme spelling",
        )
    if normalized not in V841_TENDENCY_CARRIER_INVENTORY:
        _refuse(
            "config_microp_scheme",
            scheme,
            "v8.4.1 microphysics top handling is pinned only for Kessler, WSM6, and Thompson branches",
            "scheme='mp_kessler', 'mp_wsm6', 'mp_thompson', or 'mp_thompson_aerosols'",
        )
    return normalized


def _validated_mp_top_level(mp_top_level: object, n_levels: int) -> int:
    if (
        isinstance(mp_top_level, (bool, np.bool_))
        or not isinstance(mp_top_level, (int, np.integer))
        or not 0 <= int(mp_top_level) <= int(n_levels)
    ):
        _refuse(
            "mp_top_level",
            mp_top_level,
            "the v8.4.1 cutoff is a one-based level count within the column",
            f"an integer from 0 through nVertLevels={n_levels}",
        )
    return int(mp_top_level)


def resolve_mpas_physics_v841(
    *,
    config_physics_suite: str = "none",
    config_microp_scheme: str = "suite",
    config_convection_scheme: str = "suite",
    config_lsm_scheme: str = "suite",
    config_pbl_scheme: str = "suite",
    config_gwdo_scheme: str = "suite",
    config_radt_cld_scheme: str = "suite",
    config_radt_lw_scheme: str = "suite",
    config_radt_sw_scheme: str = "suite",
    config_sfclayer_scheme: str = "suite",
) -> GPUWMSchemeSelection:
    """Resolve the shared routes while admitting v8.4.1's UGWP spelling.

    ``bl_ugwp_gwdo`` is intercepted before the v8.2.3 baseline resolver so
    the error says that a valid v8.4.1 scheme lacks a gpuwm gravity-wave slot,
    rather than misdiagnosing the released name as a typo.
    """

    gwdo = str(config_gwdo_scheme).strip()
    if gwdo not in MPAS_V841_SELECTOR_INVENTORY["config_gwdo_scheme"]:
        _refuse(
            "config_gwdo_scheme",
            gwdo,
            "the MPAS-A v8.4.1 Registry does not admit this gravity-wave scheme",
            "config_gwdo_scheme='off', 'bl_ysu_gwdo', or 'bl_ugwp_gwdo'",
        )
    if gwdo == "bl_ugwp_gwdo":
        _refuse(
            "config_gwdo_scheme",
            gwdo,
            "gpuwm.core.physics.PHYSICS_SLOT_DISPATCH publishes no gravity-wave driver slot",
            "config_gwdo_scheme='off' until gpuwm publishes a real UGWP column contract",
        )
    return resolve_mpas_physics(
        config_physics_suite=config_physics_suite,
        config_microp_scheme=config_microp_scheme,
        config_convection_scheme=config_convection_scheme,
        config_lsm_scheme=config_lsm_scheme,
        config_pbl_scheme=config_pbl_scheme,
        config_gwdo_scheme=gwdo,
        config_radt_cld_scheme=config_radt_cld_scheme,
        config_radt_lw_scheme=config_radt_lw_scheme,
        config_radt_sw_scheme=config_radt_sw_scheme,
        config_sfclayer_scheme=config_sfclayer_scheme,
    )


def compute_v841_mp_top_level(
    rdzw: ArrayLike,
    dzu: ArrayLike,
    *,
    config_microphysics_top: float = 45_000.0,
) -> int:
    """Return released v8.4.1 ``mp_top_level`` in Fortran one-based form.

    The return value is a count in ``[0, nVertLevels]``.  It is therefore also
    the Python slice boundary: ``array[mp_top_level:]`` lies above the admitted
    microphysics top.  Arithmetic follows ``mpas_atmphys_init.F:276-282`` in
    the input floating dtype instead of reconstructing geometric height by a
    different formula.
    """

    inverse_w_spacing = np.asarray(rdzw)
    u_spacing = np.asarray(dzu)
    if (
        inverse_w_spacing.ndim != 1
        or u_spacing.ndim != 1
        or inverse_w_spacing.shape != u_spacing.shape
        or inverse_w_spacing.size == 0
        or inverse_w_spacing.dtype not in _SUPPORTED_RKINDS
        or u_spacing.dtype not in _SUPPORTED_RKINDS
    ):
        _refuse(
            "rdzw/dzu",
            (inverse_w_spacing.shape, u_spacing.shape),
            "v8.4.1 initializes one shared vertical microphysics cutoff from equal non-empty floating profiles",
            "one-dimensional floating rdzw and dzu arrays of length nVertLevels",
        )
    if inverse_w_spacing.dtype != u_spacing.dtype:
        _refuse(
            "rdzw/dzu.dtype",
            (str(inverse_w_spacing.dtype), str(u_spacing.dtype)),
            "the released RKIND loop uses one floating kind for both vertical metrics",
            "rdzw and dzu with the same floating dtype",
        )
    if (
        not np.all(np.isfinite(inverse_w_spacing))
        or not np.all(np.isfinite(u_spacing))
        or np.any(inverse_w_spacing <= 0.0)
        or np.any(u_spacing <= 0.0)
    ):
        _refuse(
            "rdzw/dzu",
            "non-finite or non-positive metric",
            "the released height recurrence requires finite positive vertical metrics",
            "finite positive rdzw and dzu profiles",
        )

    dtype = inverse_w_spacing.dtype.type
    top = dtype(config_microphysics_top)
    if not np.isfinite(top) or top <= dtype(0.0):
        _refuse(
            "config_microphysics_top",
            config_microphysics_top,
            "the v8.4.1 Registry admits only positive finite heights",
            "config_microphysics_top=45000.0 or another positive height in metres",
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        layer_height = dtype(dtype(0.5) / inverse_w_spacing[0])
    if not np.isfinite(layer_height):
        _refuse(
            "rdzw",
            inverse_w_spacing[0],
            "the first v8.4.1 layer midpoint overflowed",
            "a vertical metric with finite 0.5/rdzw[0]",
        )
    mp_top_level = 1 if layer_height <= top else 0
    for level in range(1, inverse_w_spacing.size):
        with np.errstate(over="ignore", invalid="ignore"):
            layer_height = dtype(layer_height + u_spacing[level])
        if not np.isfinite(layer_height):
            _refuse(
                "dzu",
                u_spacing[level],
                "the v8.4.1 layer-height recurrence overflowed",
                "finite vertical metrics whose cumulative height is finite",
            )
        if layer_height <= top:
            mp_top_level = level + 1
    return mp_top_level


def gate_v841_microphysics_tendencies(
    dtheta: ArrayLike,
    dscalars: Mapping[str, ArrayLike],
    *,
    scheme: str,
    mp_top_level: int,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Copy and top-gate one *microphysics-only* tendency contribution.

    This helper must run before microphysics is composed with PBL, convection,
    or radiation.  Zeroing an already-composed :class:`ColumnTendencies` would
    erase legitimate non-microphysics heating above ``mp_top_level``.

    The released interface gates dry-theta and the mass mixing-ratio carriers
    admitted by the exact selected scheme.  Thompson number tendencies are
    copied unchanged.  Missing or extra carriers are refused so an unsupported
    scheme/alias cannot silently inherit the rich WSM6/Thompson rule.
    """

    normalized_scheme = _validated_microphysics_scheme(scheme)

    theta = np.asarray(dtheta)
    if (
        theta.ndim != 2
        or theta.dtype not in _SUPPORTED_RKINDS
        or not np.all(np.isfinite(theta))
    ):
        raise ValueError(
            "dtheta must be a finite float32/float64 RKIND [level,cell] array"
        )

    cutoff = _validated_mp_top_level(mp_top_level, theta.shape[0])
    theta_out = np.array(theta, copy=True, order="C")
    theta_out[cutoff:, :] = theta.dtype.type(0.0)
    scalar_out: dict[str, FloatArray] = {}
    for raw_name, raw_value in dscalars.items():
        name = _canonical_microphysics_carrier(raw_name)
        if name in scalar_out:
            raise ValueError(f"duplicate microphysics scalar tendency {name!r}")
        value = np.asarray(raw_value)
        if (
            value.shape != theta.shape
            or value.dtype != theta.dtype
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                f"dscalars[{raw_name!r}] must be finite {theta.dtype} RKIND "
                f"[level,cell] with shape {theta.shape}"
            )
        copied = np.array(value, copy=True, order="C")
        if name in V841_TOP_GATED_MASS_TENDENCIES:
            copied[cutoff:, :] = value.dtype.type(0.0)
        scalar_out[name] = copied
    expected_carriers = V841_TENDENCY_CARRIER_INVENTORY[normalized_scheme]
    actual_carriers = frozenset(scalar_out)
    if actual_carriers != expected_carriers:
        missing = sorted(expected_carriers - actual_carriers)
        extra = sorted(actual_carriers - expected_carriers)
        raise ValueError(
            f"{normalized_scheme} microphysics tendency carrier inventory mismatch: "
            f"missing={missing}, extra={extra}"
        )
    return theta_out, scalar_out


def postprocess_v841_microphysics_top(
    state: Mapping[str, ArrayLike],
    diagnostics: Mapping[str, ArrayLike],
    *,
    scheme: str,
    mp_top_level: int,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """Copy and apply released above-top state/diagnostic postprocessing.

    This is the second half of the v8.4.1 boundary contract.  The routine
    The rich WSM6/Thompson branch requires every state and diagnostic carrier
    it modifies.  The Kessler branch owns only ``qc``, ``qr``, and
    ``rt_diabatic_tend``.  Other schemes are a named nonclaim.  Known WRF and
    established seam aliases are canonicalized, and mixed RKIND storage is
    refused.  Extra state such as ``qv``, ``theta_m``, and Thompson
    number/aerosol fields is copied bit-for-bit and remains ungated.
    """

    normalized_scheme = _validated_microphysics_scheme(scheme)
    if normalized_scheme in V841_RICH_TOP_POSTPROCESS_SCHEMES:
        clamped_state = V841_TOP_CLAMPED_MASS_STATE
        zeroed_diagnostics = V841_TOP_ZEROED_DIAGNOSTICS
        background_diagnostics = V841_TOP_BACKGROUND_DIAGNOSTICS
    elif normalized_scheme == "mp_kessler":
        clamped_state = frozenset(("qc", "qr"))
        zeroed_diagnostics = frozenset(("rt_diabatic_tend",))
        background_diagnostics = frozenset()
    else:
        _refuse(
            "config_microp_scheme",
            scheme,
            "v8.4.1 top postprocessing is pinned only for Kessler, WSM6, and Thompson branches",
            "scheme='mp_kessler', 'mp_wsm6', 'mp_thompson', or 'mp_thompson_aerosols'",
        )

    canonical_state: dict[str, NDArray[Any]] = {}
    for raw_name, raw_value in state.items():
        name = _canonical_microphysics_carrier(raw_name)
        if name in canonical_state:
            raise ValueError(f"duplicate microphysics state carrier {name!r}")
        canonical_state[name] = np.asarray(raw_value)
    missing_state = clamped_state.difference(canonical_state)
    if missing_state:
        raise ValueError(
            "full v8.4.1 top postprocessing requires state carriers "
            f"{sorted(missing_state)}"
        )

    canonical_diagnostics: dict[str, NDArray[Any]] = {}
    for raw_name, raw_value in diagnostics.items():
        name = str(raw_name).strip().lower()
        if name in canonical_diagnostics:
            raise ValueError(f"duplicate microphysics diagnostic carrier {name!r}")
        canonical_diagnostics[name] = np.asarray(raw_value)
    required_diagnostics = zeroed_diagnostics | background_diagnostics
    missing_diagnostics = required_diagnostics.difference(canonical_diagnostics)
    if missing_diagnostics:
        raise ValueError(
            "full v8.4.1 top postprocessing requires diagnostic carriers "
            f"{sorted(missing_diagnostics)}"
        )

    authority = canonical_state["qc"]
    if authority.ndim != 2 or authority.dtype not in _SUPPORTED_RKINDS:
        raise ValueError(
            "microphysics state must use one float32/float64 RKIND [level,cell] shape"
        )
    shape = authority.shape
    dtype = authority.dtype
    cutoff = _validated_mp_top_level(mp_top_level, shape[0])

    for collection_name, collection in (
        ("state", canonical_state),
        ("diagnostics", canonical_diagnostics),
    ):
        for name, value in collection.items():
            if (
                value.shape != shape
                or value.dtype != dtype
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(
                    f"{collection_name}[{name!r}] must be finite {dtype} RKIND "
                    f"[level,cell] with shape {shape}"
                )

    backgrounds = {
        name: dtype.type(V841_EFFECTIVE_RADIUS_BACKGROUNDS[str(dtype)][name])
        for name in background_diagnostics
    }

    state_out = {
        name: np.array(value, copy=True, order="C")
        for name, value in canonical_state.items()
    }
    diagnostics_out = {
        name: np.array(value, copy=True, order="C")
        for name, value in canonical_diagnostics.items()
    }
    for name in clamped_state:
        state_out[name][cutoff:, :] = dtype.type(0.0)
    for name in zeroed_diagnostics:
        diagnostics_out[name][cutoff:, :] = dtype.type(0.0)
    for name in background_diagnostics:
        diagnostics_out[name][cutoff:, :] = backgrounds[name]

    # Keep the corrected output carrier invariant local to the postprocessor.
    if "evapprod" in zeroed_diagnostics:
        validate_v841_evapprod(
            diagnostics_out["evapprod"],
            n_levels=shape[0],
            n_columns=shape[1],
            mp_top_level=cutoff,
        )
    return state_out, diagnostics_out


def _validate_v841_level_cell_diagnostic(
    value: ArrayLike,
    *,
    name: str,
    n_levels: int,
    n_columns: int,
) -> FloatArray:
    array = np.asarray(value)
    expected = (int(n_levels), int(n_columns))
    if (
        array.shape != expected
        or array.dtype not in _SUPPORTED_RKINDS
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"{name} must be finite floating MPAS [level,cell] with shape "
            f"{expected}, got {array.shape}"
        )
    return np.array(array, copy=True, order="C")


def validate_v841_evapprod(
    value: ArrayLike, *, n_levels: int, n_columns: int, mp_top_level: int
) -> FloatArray:
    """Own corrected ``evapprod[level,cell]`` with its released top clamp."""

    array = _validate_v841_level_cell_diagnostic(
        value,
        name="evapprod",
        n_levels=n_levels,
        n_columns=n_columns,
    )
    cutoff = _validated_mp_top_level(mp_top_level, n_levels)
    above_top = np.ascontiguousarray(array[cutoff:, :])
    unsigned_dtype = np.dtype(np.uint32 if array.dtype == np.dtype(np.float32) else np.uint64)
    if not np.array_equal(
        above_top.view(unsigned_dtype),
        np.zeros_like(above_top).view(unsigned_dtype),
    ):
        raise ValueError("evapprod must be bitwise positive zero above mp_top_level")
    return array


def validate_v841_refl10cm(
    value: ArrayLike, *, n_levels: int, n_columns: int
) -> FloatArray:
    """Own the released ``refl10cm[level,cell]`` dBZ diagnostic carrier."""

    return _validate_v841_level_cell_diagnostic(
        value,
        name="refl10cm",
        n_levels=n_levels,
        n_columns=n_columns,
    )


__all__ = [
    "MPAS_V841_SELECTOR_INVENTORY",
    "V841_EFFECTIVE_RADIUS_BACKGROUNDS",
    "V841_OPENACC_TRANSFER_HOOKS",
    "V841_PHYSICS_FIELD_DELTAS",
    "V841_PHYSICS_SOURCE",
    "V841_RICH_TOP_POSTPROCESS_SCHEMES",
    "V841_TOP_BACKGROUND_DIAGNOSTICS",
    "V841_TOP_CLAMPED_MASS_STATE",
    "V841_TOP_GATED_MASS_TENDENCIES",
    "V841_TOP_UNCHANGED_STATE",
    "V841_TOP_ZEROED_DIAGNOSTICS",
    "V841_TENDENCY_CARRIER_INVENTORY",
    "V841_UNGATED_NUMBER_TENDENCIES",
    "V841_WRF_SCALAR_ALIASES",
    "V841PhysicsFieldDelta",
    "compute_v841_mp_top_level",
    "gate_v841_microphysics_tendencies",
    "postprocess_v841_microphysics_top",
    "resolve_mpas_physics_v841",
    "validate_v841_evapprod",
    "validate_v841_refl10cm",
]
