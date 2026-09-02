"""MPAS column-physics seam for the existing :mod:`gpuwm` physics stack.

The frozen authorities for this module are MPAS-A v8.2.3
``mpas_atmphys_interface.F:222-546`` (column packing),
``mpas_atmphys_driver.F:180-370`` (driver order), and
``mpas_atmphys_todynamics.F:65-456`` (dry-mass coupling and wind projection).
The gpuwm selector values are those published by ``gpuwm/config.py`` and
``gpuwm/core/physics.py::PHYSICS_SLOT_DISPATCH`` on 2026-08-09.

This module deliberately does not clone a physics parameterization.  It owns
only the unstructured-mesh seam:

* retain logical MPAS ``(level, cell)`` fields as owned, C-contiguous arrays
  with cells fastest and level zero at the surface;
* reconstruct edge-normal wind at cells before a column backend runs;
* turn uncoupled column rates back into MPAS dry-mass-coupled tendencies;
* project cell zonal/meridional acceleration back to edge normals; and
* route every MPAS scheme name either to its exact gpuwm selector or to an
  explicit refusal naming the original MPAS knob.

The default gpuwm adapter is lazy.  Importing :mod:`hexcore.physics_seam`
therefore needs neither CuPy nor CUDA.  The current gpuwm public driver takes
an ARW ``DomainState`` rather than an unstructured column batch; if a gpuwm
tree does not publish ``run_mpas_column_batch`` the adapter refuses instead of
manufacturing a structured grid or silently selecting a nearby scheme.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationRefusal
from .state import PrognosticState
from .vector import initialize_vector_geometry, reconstruct_2d, vector_lon_lat_r_to_r3


FloatArray = NDArray[np.floating[Any]]
VerticalOrder = Literal["surface_to_top", "top_to_surface"]
BackendMutation = Literal["forbid", "isolate", "allow"]

DRY_AIR_GAS_CONSTANT = 287.0
WATER_VAPOR_GAS_CONSTANT = 461.6
DRY_AIR_CP = 1004.5
REFERENCE_PRESSURE = 100_000.0
RV_OVER_RD = WATER_VAPOR_GAS_CONSTANT / DRY_AIR_GAS_CONSTANT


# Values accepted by MPAS-A v8.2.3's Registry/control module.  Keeping the
# complete inventory next to the routing rows makes an unported-but-valid name
# distinguishable from a typo.
MPAS_SELECTOR_INVENTORY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "config_physics_suite": (
            "mesoscale_reference",
            "convection_permitting",
            "none",
        ),
        "config_microp_scheme": (
            "suite",
            "off",
            "mp_kessler",
            "mp_thompson",
            "mp_thompson_aerosols",
            "mp_wsm6",
        ),
        "config_convection_scheme": (
            "suite",
            "off",
            "cu_grell_freitas",
            "cu_kain_fritsch",
            "cu_tiedtke",
            "cu_ntiedtke",
        ),
        "config_lsm_scheme": ("suite", "off", "sf_noah", "sf_noahmp"),
        "config_pbl_scheme": ("suite", "off", "bl_mynn", "bl_ysu"),
        "config_gwdo_scheme": ("suite", "off", "bl_ysu_gwdo"),
        "config_radt_cld_scheme": (
            "suite",
            "off",
            "cld_incidence",
            "cld_fraction",
            "cld_fraction_thompson",
        ),
        "config_radt_lw_scheme": ("suite", "off", "cam_lw", "rrtmg_lw"),
        "config_radt_sw_scheme": ("suite", "off", "cam_sw", "rrtmg_sw"),
        "config_sfclayer_scheme": (
            "suite",
            "off",
            "sf_mynn",
            "sf_monin_obukhov",
            "sf_monin_obukhov_rev",
        ),
    }
)


# gpuwm/config.py plus gpuwm/core/physics.py.  This is an inventory, not a
# promise that every value can run in the local CUDA environment.  The
# mp_physics row is engine MP_PHYSICS_ACCEPTED at the 2.6.4 pin line
# (gpuwm/config.py at the engine's 2.6.4 line, unchanged since 2.6.1 for
# microphysics; the cu_physics row is engine CU_SCHEMES, which gained 16
# (New Tiedtke) at the 2.6.4 cut and was re-synced here 2026-09-02 when the
# pin moved -- the inventory names it, the routes below do not):
# re-synced 2026-08-31 when the P3 row
# landed below -- the previous copy had already drifted (it lacked 9, 16
# and 50), which is exactly the second-copy hazard the ladder docstring
# names.
GPUWM_SELECTOR_INVENTORY: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "mp_physics": (0, 1, 6, 8, 9, 10, 16, 18, 28, 50),
        "cu_physics": (0, 1, 3, 16),
        "sf_surface_physics": (0, 2, 3, 4),
        "bl_pbl_physics": (0, 1, 5, 11, 900),
        "sf_sfclay_physics": (0, 1, 5, 91),
        "ra_lw_physics": (0, 1, 4, 90),
        "ra_sw_physics": (0, 1, 4, 90),
    }
)


# Post-divergence hexcore vocabulary.  The 2026-08-20 ruling pins the
# DYCORE to MPAS and sends the PHYSICS its own way; these are the scheme
# names the port admits on its own authority.  They are NOT MPAS names --
# neither the frozen v8.2.3 inventory above nor released v8.4.1 spells
# them -- and they must never migrate into MPAS_SELECTOR_INVENTORY, which
# records what MPAS itself accepts.  A name lands here only in the same
# commit as its route in ``_ROUTES`` and its scalar-requirement row in
# ``_required_scalar_names``; the retired mp=50 refusal (see
# ``_UNROUTED_MP_PHYSICS_REASONS``) is the record of why the three move
# together.
HEXCORE_SELECTOR_EXTENSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # P3 one-category two-moment ice (gpuwm mp_physics=50), admitted
        # 2026-08-31 once the seam vocabulary gained the rime pair
        # qir/qib and the engine column batch gained the eight-scalar
        # P3 transport (gpuwm/core/mpas_column_batch.py,
        # microphysics_scheme="p3").
        "config_microp_scheme": ("mp_p3",),
    }
)


_SUITE_DEFAULTS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "mesoscale_reference": MappingProxyType(
            {
                "config_microp_scheme": "mp_wsm6",
                "config_convection_scheme": "cu_ntiedtke",
                "config_lsm_scheme": "sf_noah",
                "config_pbl_scheme": "bl_ysu",
                "config_gwdo_scheme": "bl_ysu_gwdo",
                "config_radt_cld_scheme": "cld_fraction",
                "config_radt_lw_scheme": "rrtmg_lw",
                "config_radt_sw_scheme": "rrtmg_sw",
                "config_sfclayer_scheme": "sf_monin_obukhov_rev",
            }
        ),
        "convection_permitting": MappingProxyType(
            {
                "config_microp_scheme": "mp_thompson",
                "config_convection_scheme": "cu_grell_freitas",
                "config_lsm_scheme": "sf_noah",
                "config_pbl_scheme": "bl_mynn",
                "config_gwdo_scheme": "bl_ysu_gwdo",
                "config_radt_cld_scheme": "cld_fraction",
                "config_radt_lw_scheme": "rrtmg_lw",
                "config_radt_sw_scheme": "rrtmg_sw",
                "config_sfclayer_scheme": "sf_mynn",
            }
        ),
        "none": MappingProxyType(
            {
                knob: "off"
                for knob in MPAS_SELECTOR_INVENTORY
                if knob != "config_physics_suite"
            }
        ),
    }
)


_ROUTES: Mapping[str, Mapping[str, int]] = MappingProxyType(
    {
        "config_microp_scheme": MappingProxyType(
            {
                "off": 0,
                "mp_kessler": 1,
                "mp_wsm6": 6,
                "mp_thompson": 8,
                "mp_thompson_aerosols": 28,
                # HEXCORE EXTENSION, not an MPAS name (see
                # HEXCORE_SELECTOR_EXTENSIONS): P3 one-category
                # two-moment ice.  Routed 2026-08-31, the commit that
                # gave this seam qir/qib and the engine column batch the
                # eight-scalar P3 transport -- the two prerequisites the
                # retired mp=50 refusal named as its own retirement
                # condition.
                "mp_p3": 50,
            }
        ),
        "config_convection_scheme": MappingProxyType(
            {"off": 0, "cu_kain_fritsch": 1, "cu_grell_freitas": 3}
        ),
        "config_lsm_scheme": MappingProxyType({"off": 0, "sf_noah": 2, "sf_noahmp": 4}),
        "config_pbl_scheme": MappingProxyType({"off": 0, "bl_ysu": 1, "bl_mynn": 5}),
        "config_sfclayer_scheme": MappingProxyType(
            {
                "off": 0,
                "sf_monin_obukhov_rev": 1,
                "sf_mynn": 5,
                "sf_monin_obukhov": 91,
            }
        ),
        # MPAS's RRTMG names mean the legacy WRF/MPAS implementation.  The
        # gpuwm numeric selector is 4, but the variant receipt below pins
        # ``rrtmg_legacy`` so it can never fall through to RTE+RRTMGP.
        "config_radt_lw_scheme": MappingProxyType({"off": 0, "rrtmg_lw": 4}),
        "config_radt_sw_scheme": MappingProxyType({"off": 0, "rrtmg_sw": 4}),
    }
)


_UNROUTED_REASONS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        (
            "config_convection_scheme",
            "cu_tiedtke",
        ): "gpuwm has no Tiedtke column backend",
        (
            "config_convection_scheme",
            "cu_ntiedtke",
        ): "gpuwm has no new-Tiedtke column backend",
        (
            "config_radt_lw_scheme",
            "cam_lw",
        ): "gpuwm has no CAM longwave backend",
        (
            "config_radt_sw_scheme",
            "cam_sw",
        ): "gpuwm has no CAM shortwave backend",
        (
            "config_gwdo_scheme",
            "bl_ysu_gwdo",
        ): "gpuwm.core.physics publishes no gravity-wave-drag driver slot",
    }
)


_SCALAR_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "qv": "qv",
        "qvapor": "qv",
        "water_vapor": "qv",
        "qc": "qc",
        "qcloud": "qc",
        "cloud_water": "qc",
        "qr": "qr",
        "qrain": "qr",
        "rain_water": "qr",
        "qi": "qi",
        "qice": "qi",
        "cloud_ice": "qi",
        "qs": "qs",
        "qsnow": "qs",
        "snow": "qs",
        "qg": "qg",
        "qgraupel": "qg",
        "graupel": "qg",
        "qh": "qh",
        "qhail": "qh",
        "hail": "qh",
        "nc": "nc",
        "nr": "nr",
        "nrain": "nr",
        "ni": "ni",
        "nifa": "nifa",
        "nwfa": "nwfa",
        # P3's rime pair (mp_physics=50): WRF Registry names QIR (rime
        # ice mass) and QIB (rime ice volume); module_mp_p3.F spells the
        # same fields qirim/birim.  MPAS-A allocates neither -- these
        # rows are hexcore vocabulary, landed with the mp_p3 route.
        "qir": "qir",
        "qirim": "qir",
        "rime_ice_mass": "qir",
        "qib": "qib",
        "birim": "qib",
        "rime_ice_volume": "qib",
    }
)


def _canonical_scalar_name(name: str) -> str:
    normalized = str(name).strip().lower()
    return _SCALAR_ALIASES.get(normalized, normalized)


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def _resolved_name(knob: str, value: str, suite: str) -> str:
    name = str(value).strip()
    inventory = MPAS_SELECTOR_INVENTORY[knob]
    extensions = HEXCORE_SELECTOR_EXTENSIONS.get(knob, ())
    if name not in inventory and name not in extensions:
        reason = f"MPAS-A v8.2.3 accepts only {list(inventory)}"
        if extensions:
            reason += (
                f" and the post-divergence hexcore vocabulary adds "
                f"{list(extensions)}"
            )
        _refuse(knob, name, reason, f"{knob}='off'")
    return _SUITE_DEFAULTS[suite][knob] if name == "suite" else name


def _route_name(knob: str, name: str) -> int:
    route = _ROUTES[knob]
    if name in route:
        return route[name]
    reason = _UNROUTED_REASONS.get(
        (knob, name), f"no exact gpuwm selector is routed for MPAS scheme {name!r}"
    )
    _refuse(knob, name, reason, f"{knob}='off'")
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class GPUWMSchemeSelection:
    """Exact MPAS-name to gpuwm-value routing receipt."""

    mp_physics: int
    cu_physics: int
    sf_surface_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    ra_lw_physics: int
    ra_sw_physics: int
    ra_rrtmg_variant: str
    wrf_rrtmg_compatibility: str
    cloud_fraction_scheme: str
    mpas_names: Mapping[str, str]

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.mp_physics,
                self.cu_physics,
                self.sf_surface_physics,
                self.bl_pbl_physics,
                self.sf_sfclay_physics,
                self.ra_lw_physics,
                self.ra_sw_physics,
            )
        )

    def as_gpuwm_config(self) -> dict[str, int | str]:
        """Return only public gpuwm ``RunConfig`` selector spellings."""

        return {
            "mp_physics": self.mp_physics,
            "cu_physics": self.cu_physics,
            "sf_surface_physics": self.sf_surface_physics,
            "bl_pbl_physics": self.bl_pbl_physics,
            "sf_sfclay_physics": self.sf_sfclay_physics,
            "ra_physics": 0,
            "ra_lw_physics": self.ra_lw_physics,
            "ra_sw_physics": self.ra_sw_physics,
            "ra_rrtmg_variant": self.ra_rrtmg_variant,
            "wrf_rrtmg_compatibility": self.wrf_rrtmg_compatibility,
        }


def resolve_mpas_physics(
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
    """Resolve MPAS v8.2.3 physics names without fallback substitutions.

    MPAS's own ``suite`` expansion is applied first.  A valid MPAS scheme
    absent from gpuwm then raises :class:`ConfigurationRefusal` with the
    original MPAS knob and value.
    """

    suite = str(config_physics_suite).strip()
    if suite not in MPAS_SELECTOR_INVENTORY["config_physics_suite"]:
        _refuse(
            "config_physics_suite",
            suite,
            "the frozen Registry has only mesoscale_reference, convection_permitting, and none",
            "config_physics_suite='none'",
        )
    raw = {
        "config_microp_scheme": config_microp_scheme,
        "config_convection_scheme": config_convection_scheme,
        "config_lsm_scheme": config_lsm_scheme,
        "config_pbl_scheme": config_pbl_scheme,
        "config_gwdo_scheme": config_gwdo_scheme,
        "config_radt_cld_scheme": config_radt_cld_scheme,
        "config_radt_lw_scheme": config_radt_lw_scheme,
        "config_radt_sw_scheme": config_radt_sw_scheme,
        "config_sfclayer_scheme": config_sfclayer_scheme,
    }
    names = {knob: _resolved_name(knob, value, suite) for knob, value in raw.items()}

    # Fail on a known-but-unrouted slot before constructing a partial receipt.
    gwdo = names["config_gwdo_scheme"]
    if gwdo != "off":
        reason = _UNROUTED_REASONS[("config_gwdo_scheme", gwdo)]
        _refuse("config_gwdo_scheme", gwdo, reason, "config_gwdo_scheme='off'")

    mp = _route_name("config_microp_scheme", names["config_microp_scheme"])
    cu = _route_name("config_convection_scheme", names["config_convection_scheme"])
    lsm = _route_name("config_lsm_scheme", names["config_lsm_scheme"])
    pbl = _route_name("config_pbl_scheme", names["config_pbl_scheme"])
    sfclay = _route_name("config_sfclayer_scheme", names["config_sfclayer_scheme"])
    lw = _route_name("config_radt_lw_scheme", names["config_radt_lw_scheme"])
    sw = _route_name("config_radt_sw_scheme", names["config_radt_sw_scheme"])

    if (lw, sw) not in ((0, 0), (4, 4)):
        _refuse(
            "config_radt_lw_scheme/config_radt_sw_scheme",
            (names["config_radt_lw_scheme"], names["config_radt_sw_scheme"]),
            "gpuwm's exact legacy-RRTMG adapter is a coupled 4/4 LW/SW backend",
            "config_radt_lw_scheme='rrtmg_lw', config_radt_sw_scheme='rrtmg_sw'",
        )

    cloud = names["config_radt_cld_scheme"]
    if lw == 0 and sw == 0 and cloud != "off":
        _refuse(
            "config_radt_cld_scheme",
            cloud,
            "cloud-fraction physics has no radiation consumer",
            "config_radt_cld_scheme='off'",
        )
    if (lw or sw) and cloud == "off":
        _refuse(
            "config_radt_cld_scheme",
            cloud,
            "active MPAS radiation requires an explicitly named cloud-fraction producer",
            "config_radt_cld_scheme='cld_fraction'",
        )

    if lsm and not sfclay:
        _refuse(
            "config_sfclayer_scheme",
            names["config_sfclayer_scheme"],
            "an MPAS land-surface scheme consumes surface-layer exchange coefficients",
            "config_sfclayer_scheme='sf_monin_obukhov_rev'",
        )
    if pbl == 1 and sfclay not in (1, 91):
        _refuse(
            "config_sfclayer_scheme",
            names["config_sfclayer_scheme"],
            "bl_ysu requires the revised or classic Monin-Obukhov surface layer",
            "config_sfclayer_scheme='sf_monin_obukhov_rev'",
        )
    if pbl == 5 and sfclay != 5:
        # MPAS control.F silently overwrites this pairing.  The seam makes it
        # explicit because inherited mutation of a requested scheme is exactly
        # the class of substitution this port must expose.
        _refuse(
            "config_sfclayer_scheme",
            names["config_sfclayer_scheme"],
            "bl_mynn requires sf_mynn; this seam does not silently rewrite the requested surface scheme",
            "config_sfclayer_scheme='sf_mynn'",
        )
    if cu == 3 and not pbl:
        _refuse(
            "config_pbl_scheme",
            names["config_pbl_scheme"],
            "gpuwm's Grell-Freitas adapter consumes PBL-maintained trigger state",
            "config_pbl_scheme='bl_mynn'",
        )

    variant = "rrtmg_legacy" if (lw, sw) == (4, 4) else "rte-rrtmgp"
    compatibility = "wrf-rrtmg-4-4-legacy-v1" if (lw, sw) == (4, 4) else "none"
    return GPUWMSchemeSelection(
        mp_physics=mp,
        cu_physics=cu,
        sf_surface_physics=lsm,
        bl_pbl_physics=pbl,
        sf_sfclay_physics=sfclay,
        ra_lw_physics=lw,
        ra_sw_physics=sw,
        ra_rrtmg_variant=variant,
        wrf_rrtmg_compatibility=compatibility,
        cloud_fraction_scheme=cloud,
        mpas_names=MappingProxyType(dict(names)),
    )


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


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    value = _mesh_value(mesh, name)
    if value is None:
        raise AttributeError(f"mesh has no required MPAS field {name!r}")
    return np.asarray(value)


def _cells_on_edge(mesh: object, n_edges: int) -> NDArray[np.int64]:
    cells = np.asarray(_mesh_array(mesh, "cellsOnEdge"), dtype=np.int64)
    if cells.shape == (2, n_edges):
        cells = cells.T
    if cells.shape != (n_edges, 2):
        raise ValueError(f"cellsOnEdge shape {cells.shape} != ({n_edges}, 2)")
    return cells


def _source_to_batch(
    value: ArrayLike,
    shape: tuple[int, int],
    name: str,
    vertical_order: VerticalOrder,
) -> FloatArray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind != "f":
        raise ValueError(
            f"{name} must be floating with shape {shape}, got {array.shape}"
        )
    logical = array if vertical_order == "surface_to_top" else array[::-1]
    # The CUDA layout ruling is [level, entity], C contiguous: horizontal
    # owners are the fastest axis.  Do not transpose this at a host seam.
    # Own the seam buffer even when the MPAS source is already C-contiguous;
    # physics-side mutation must never alias prognostic state before unpack.
    result = np.array(logical, copy=True, order="C")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _batch_to_source(value: FloatArray, vertical_order: VerticalOrder) -> FloatArray:
    logical = np.ascontiguousarray(np.asarray(value))
    return (
        logical
        if vertical_order == "surface_to_top"
        else np.ascontiguousarray(logical[::-1])
    )


def _column_field(
    value: ArrayLike,
    n_columns: int,
    name: str,
    dtype: np.dtype[Any] | None,
) -> NDArray[Any]:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a numeric column field")
    if array.ndim == 0:
        result = np.full((n_columns,), array, dtype=array.dtype)
    elif array.shape[0] == n_columns:
        result = array.copy()
    elif array.shape[-1] == n_columns:
        result = np.moveaxis(array, -1, 0).copy()
    else:
        raise ValueError(
            f"{name} shape {array.shape} must be scalar or carry nColumns={n_columns} "
            "on its first or last axis"
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def _copy_field_mapping(
    values: Mapping[str, ArrayLike] | None,
    n_columns: int,
    prefix: str,
    dtype: np.dtype[Any] | None,
) -> dict[str, NDArray[Any]]:
    return {
        str(name): _column_field(value, n_columns, f"{prefix}.{name}", dtype)
        for name, value in ({} if values is None else values).items()
    }


@dataclass(slots=True)
class ColumnBatch:
    """Owned backend contract: contiguous ``(surface-to-top level, column)``.

    Atmospheric and scalar arrays use ``[level, cell]`` with cells fastest,
    matching the certified MPAS CUDA layout and gpuwm's level-major column
    kernels.  Surface and static fields are column-first. A scalar surface value
    has shape ``(nColumns,)``; a soil profile has shape
    ``(nColumns, nSoilLevels)``.  ``w`` is optional until a selected backend
    declares it required and then has shape ``(nLevels+1, nColumns)``.

    ``metric_density`` is MPAS ``rho_zz`` (the MPAS-A v8.2.3 Registry.xml
    state declaration), not ARW column mass. ``dry_density`` is
    ``zz*rho_zz`` and ``moist_density`` is ``zz*rho_zz*(1+qv)`` exactly as
    MPAS_to_physics in the module-level frozen interface source window.
    Pressure variants keep the source names distinct: EOS
    ``pres_p`` versus hydrostatic moist/dry mass- and interface-level
    pressures from that same frozen source window.
    """

    metric_density: FloatArray
    dry_density: FloatArray
    moist_density: FloatArray
    modified_theta: FloatArray
    theta: FloatArray
    temperature: FloatArray
    eos_pressure: FloatArray
    eos_pressure_interface: FloatArray | None
    hydrostatic_moist_pressure: FloatArray | None
    hydrostatic_moist_pressure_interface: FloatArray | None
    hydrostatic_dry_pressure: FloatArray | None
    hydrostatic_dry_pressure_interface: FloatArray | None
    height: FloatArray
    height_interface: FloatArray
    layer_thickness: FloatArray
    u: FloatArray
    v: FloatArray
    w: FloatArray | None
    scalars: MutableMapping[str, FloatArray]
    surface: MutableMapping[str, FloatArray]
    static: MutableMapping[str, FloatArray]
    time_seconds: float = 0.0
    vertical_order: str = field(default="surface_to_top", init=False)

    @property
    def n_columns(self) -> int:
        return int(self.metric_density.shape[1])

    @property
    def n_levels(self) -> int:
        return int(self.metric_density.shape[0])

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.metric_density.dtype

    def validate(self) -> None:
        shape = self.metric_density.shape
        if len(shape) != 2:
            raise ValueError(
                f"metric_density must have shape (nLevels,nColumns), got {shape}"
            )
        if self.metric_density.dtype.kind != "f":
            raise TypeError("column fields must be floating")
        for name in (
            "metric_density",
            "dry_density",
            "moist_density",
            "modified_theta",
            "theta",
            "temperature",
            "eos_pressure",
            "height",
            "layer_thickness",
            "u",
            "v",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
            if value.dtype.kind != "f" or not value.flags.c_contiguous:
                raise TypeError(f"{name} must be a C-contiguous floating array")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} contains non-finite values")
        interface_shape = (shape[0] + 1, shape[1])
        if self.height_interface.shape != interface_shape:
            raise ValueError("height_interface must have shape (nLevels+1,nColumns)")
        if (
            self.height_interface.dtype.kind != "f"
            or not self.height_interface.flags.c_contiguous
            or not np.all(np.isfinite(self.height_interface))
        ):
            raise ValueError(
                "height_interface must be floating, finite, and C-contiguous"
            )
        pressure_fields = {
            "eos_pressure": (self.eos_pressure, shape),
            "eos_pressure_interface": (
                self.eos_pressure_interface,
                interface_shape,
            ),
            "hydrostatic_moist_pressure": (
                self.hydrostatic_moist_pressure,
                shape,
            ),
            "hydrostatic_moist_pressure_interface": (
                self.hydrostatic_moist_pressure_interface,
                interface_shape,
            ),
            "hydrostatic_dry_pressure": (
                self.hydrostatic_dry_pressure,
                shape,
            ),
            "hydrostatic_dry_pressure_interface": (
                self.hydrostatic_dry_pressure_interface,
                interface_shape,
            ),
        }
        for name, (value, expected_shape) in pressure_fields.items():
            if value is None:
                continue
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape {value.shape} != {expected_shape}")
            if (
                value.dtype.kind != "f"
                or not value.flags.c_contiguous
                or not np.all(np.isfinite(value))
                or np.any(value <= 0.0)
                or not np.all(np.diff(value, axis=0) < 0.0)
            ):
                raise ValueError(
                    f"{name} must be finite, positive, C-contiguous, and "
                    "decrease from surface to top"
                )
        if self.w is not None:
            if self.w.shape != self.height_interface.shape:
                raise ValueError("w must have shape (nLevels+1,nColumns)")
            if (
                self.w.dtype.kind != "f"
                or not self.w.flags.c_contiguous
                or not np.all(np.isfinite(self.w))
            ):
                raise ValueError("w must be floating, finite, and C-contiguous")
        if (
            np.any(self.metric_density <= 0.0)
            or np.any(self.dry_density <= 0.0)
            or np.any(self.moist_density <= 0.0)
        ):
            raise ValueError("metric, dry, and moist density must be positive")
        if np.any(self.eos_pressure <= 0.0) or np.any(self.temperature <= 0.0):
            raise ValueError("pressure and temperature must be strictly positive")
        if np.any(self.layer_thickness <= 0.0):
            raise ValueError("layer thickness must be strictly positive")
        if not np.all(np.diff(self.height_interface, axis=0) > 0.0):
            raise ValueError(
                "height_interface must increase strictly from surface to top"
            )
        if not np.all(np.diff(self.eos_pressure, axis=0) < 0.0):
            raise ValueError("eos_pressure must decrease strictly from surface to top")
        for group_name in ("scalars", "surface", "static"):
            group = getattr(self, group_name)
            for name, value in group.items():
                array = np.asarray(value)
                if group_name == "scalars" and array.shape != shape:
                    raise ValueError(f"scalars.{name} shape {array.shape} != {shape}")
                if group_name != "scalars" and array.shape[0] != self.n_columns:
                    raise ValueError(
                        f"{group_name}.{name} first dimension {array.shape[0]} "
                        f"!= nColumns={self.n_columns}"
                    )
                if not array.flags.c_contiguous or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{group_name}.{name} must be finite and C-contiguous"
                    )
        qv = self.scalars.get("qv")
        if qv is not None and np.any(qv < 0.0):
            raise ValueError("scalars.qv must be non-negative")
        moisture = np.zeros_like(self.dry_density) if qv is None else qv
        expected_moist_density = self.dry_density * (1.0 + moisture)
        tolerance = 8.0 * np.finfo(self.dtype).eps
        if not np.allclose(
            self.moist_density,
            expected_moist_density,
            rtol=tolerance,
            atol=0.0,
        ):
            raise ValueError("moist_density must equal dry_density*(1+scalars.qv)")

    def copy(self) -> "ColumnBatch":
        return ColumnBatch(
            metric_density=self.metric_density.copy(),
            dry_density=self.dry_density.copy(),
            moist_density=self.moist_density.copy(),
            modified_theta=self.modified_theta.copy(),
            theta=self.theta.copy(),
            temperature=self.temperature.copy(),
            eos_pressure=self.eos_pressure.copy(),
            eos_pressure_interface=(
                None
                if self.eos_pressure_interface is None
                else self.eos_pressure_interface.copy()
            ),
            hydrostatic_moist_pressure=(
                None
                if self.hydrostatic_moist_pressure is None
                else self.hydrostatic_moist_pressure.copy()
            ),
            hydrostatic_moist_pressure_interface=(
                None
                if self.hydrostatic_moist_pressure_interface is None
                else self.hydrostatic_moist_pressure_interface.copy()
            ),
            hydrostatic_dry_pressure=(
                None
                if self.hydrostatic_dry_pressure is None
                else self.hydrostatic_dry_pressure.copy()
            ),
            hydrostatic_dry_pressure_interface=(
                None
                if self.hydrostatic_dry_pressure_interface is None
                else self.hydrostatic_dry_pressure_interface.copy()
            ),
            height=self.height.copy(),
            height_interface=self.height_interface.copy(),
            layer_thickness=self.layer_thickness.copy(),
            u=self.u.copy(),
            v=self.v.copy(),
            w=None if self.w is None else self.w.copy(),
            scalars={name: value.copy() for name, value in self.scalars.items()},
            surface={name: value.copy() for name, value in self.surface.items()},
            static={name: value.copy() for name, value in self.static.items()},
            time_seconds=self.time_seconds,
        )


@dataclass(frozen=True, slots=True)
class MpasPackingContext:
    """MPAS-only metadata intentionally hidden from a column backend."""

    mesh: object
    scalar_names: tuple[str, ...]
    metric_density_level_cell: FloatArray
    edge_metric_density_level_edge: FloatArray
    vertical_order: VerticalOrder


@dataclass(slots=True)
class PackedMpasColumns:
    batch: ColumnBatch
    context: MpasPackingContext


def pack_mpas_columns(
    state: PrognosticState,
    mesh: object,
    *,
    eos_pressure: ArrayLike,
    height_interface: ArrayLike,
    scalar_names: Sequence[str],
    eos_pressure_interface: ArrayLike | None = None,
    hydrostatic_moist_pressure: ArrayLike | None = None,
    hydrostatic_moist_pressure_interface: ArrayLike | None = None,
    hydrostatic_dry_pressure: ArrayLike | None = None,
    hydrostatic_dry_pressure_interface: ArrayLike | None = None,
    metric_zz: ArrayLike | None = None,
    edge_metric_density: ArrayLike | None = None,
    cell_wind: tuple[ArrayLike, ArrayLike] | None = None,
    vertical_velocity: ArrayLike | None = None,
    surface: Mapping[str, ArrayLike] | None = None,
    static: Mapping[str, ArrayLike] | None = None,
    coefficients: ArrayLike | None = None,
    vertical_order: VerticalOrder = "surface_to_top",
) -> PackedMpasColumns:
    """Pack one MPAS state into the stable column-batch contract.

    ``state.rho`` is MPAS metric density ``rho_zz`` and ``state.rho_theta``
    is its conservative product with modified potential temperature.
    ``metric_zz`` maps ``rho_zz`` to physical dry density. Moist density is
    then ``dry_density*(1+qv)`` exactly as the module-level frozen MPAS-A
    v8.2.3 interface source window.

    Pressure names are intentionally verbose. ``eos_pressure`` is MPAS
    ``pres_p`` (interface line 335), while the four hydrostatic arrays are
    ``pres_hyd_p``, ``pres2_hyd_p``, ``pres_hydd_p``, and
    ``pres2_hydd_p`` (interface lines 517-527). A selected backend that needs a
    missing variant refuses by that exact field name; this seam never
    substitutes one pressure family for another.
    """

    if vertical_order not in ("surface_to_top", "top_to_surface"):
        _refuse(
            "vertical_order",
            vertical_order,
            "the seam supports only the two explicit vertical orientations",
            "vertical_order='surface_to_top'",
        )
    n_levels, n_cells = np.asarray(state.rho).shape
    n_edges = np.asarray(state.rho_u).shape[1]
    state.validate(n_cells=n_cells, n_edges=n_edges, n_vert_levels=n_levels)
    if len(scalar_names) != state.scalars.shape[0]:
        raise ValueError(
            f"scalar_names has {len(scalar_names)} entries but state carries "
            f"{state.scalars.shape[0]} scalar rows"
        )
    canonical = tuple(_canonical_scalar_name(name) for name in scalar_names)
    if len(set(canonical)) != len(canonical):
        raise ValueError(
            f"scalar_names collapse to duplicate canonical names {canonical}"
        )

    shape = (n_levels, n_cells)
    metric_density = _source_to_batch(state.rho, shape, "state.rho", vertical_order)
    rho_theta = _source_to_batch(
        state.rho_theta, shape, "state.rho_theta", vertical_order
    )

    eos_pressure_batch = _source_to_batch(
        eos_pressure, shape, "eos_pressure", vertical_order
    )

    def optional_pressure(
        value: ArrayLike | None,
        expected_shape: tuple[int, int],
        name: str,
    ) -> FloatArray | None:
        return (
            None
            if value is None
            else _source_to_batch(value, expected_shape, name, vertical_order)
        )

    interface_shape = (n_levels + 1, n_cells)
    eos_pressure_interface_batch = optional_pressure(
        eos_pressure_interface,
        interface_shape,
        "eos_pressure_interface",
    )
    hydrostatic_moist_pressure_batch = optional_pressure(
        hydrostatic_moist_pressure,
        shape,
        "hydrostatic_moist_pressure",
    )
    hydrostatic_moist_pressure_interface_batch = optional_pressure(
        hydrostatic_moist_pressure_interface,
        interface_shape,
        "hydrostatic_moist_pressure_interface",
    )
    hydrostatic_dry_pressure_batch = optional_pressure(
        hydrostatic_dry_pressure,
        shape,
        "hydrostatic_dry_pressure",
    )
    hydrostatic_dry_pressure_interface_batch = optional_pressure(
        hydrostatic_dry_pressure_interface,
        interface_shape,
        "hydrostatic_dry_pressure_interface",
    )
    zz = _source_to_batch(
        np.ones(shape, dtype=metric_density.dtype) if metric_zz is None else metric_zz,
        shape,
        "metric_zz",
        vertical_order,
    )
    z_interface = _source_to_batch(
        height_interface, interface_shape, "height_interface", vertical_order
    )
    layer_thickness = np.ascontiguousarray(np.diff(z_interface, axis=0))
    height = np.ascontiguousarray(0.5 * (z_interface[:-1, :] + z_interface[1:, :]))

    scalar_batch: dict[str, FloatArray] = {}
    for index, name in enumerate(canonical):
        scalar_batch[name] = _source_to_batch(
            state.scalars[index],
            shape,
            f"state.scalars[{scalar_names[index]!r}]",
            vertical_order,
        )
    qv = scalar_batch.get("qv")
    if qv is None:
        qv = np.zeros_like(metric_density)
    if np.any(qv < 0.0):
        raise ValueError("scalars.qv must be non-negative at the physics seam")
    modified_theta = np.ascontiguousarray(rho_theta / metric_density)
    moisture_factor = 1.0 + metric_density.dtype.type(RV_OVER_RD) * qv
    theta = np.ascontiguousarray(modified_theta / moisture_factor)
    exner = (
        eos_pressure_batch / eos_pressure_batch.dtype.type(REFERENCE_PRESSURE)
    ) ** eos_pressure_batch.dtype.type(DRY_AIR_GAS_CONSTANT / DRY_AIR_CP)
    temperature = np.ascontiguousarray(theta * exner)
    dry_density = np.ascontiguousarray(metric_density * zz)
    moist_density = np.ascontiguousarray(dry_density * (1.0 + qv))

    if edge_metric_density is None:
        cells = _cells_on_edge(mesh, n_edges)
        if np.any(cells < 0) or np.any(cells >= n_cells):
            _refuse(
                "edge_metric_density",
                "missing on a regional boundary",
                "deriving edge metric density needs two valid cells on every edge",
                "an explicit edge_metric_density array",
            )
        metric_density_logical = _batch_to_source(metric_density, "surface_to_top")
        edge_metric_density_logical = 0.5 * (
            metric_density_logical[:, cells[:, 0]]
            + metric_density_logical[:, cells[:, 1]]
        )
    else:
        edge_metric_density_logical = _batch_to_source(
            _source_to_batch(
                edge_metric_density,
                (n_levels, n_edges),
                "edge_metric_density",
                vertical_order,
            ),
            "surface_to_top",
        )
    if np.any(edge_metric_density_logical <= 0.0):
        raise ValueError("edge metric density must be strictly positive")

    if cell_wind is None:
        rho_u_logical = (
            np.asarray(state.rho_u)
            if vertical_order == "surface_to_top"
            else np.asarray(state.rho_u)[::-1]
        )
        normal_velocity = np.ascontiguousarray(
            rho_u_logical / edge_metric_density_logical
        )
        reconstructed = reconstruct_2d(mesh, normal_velocity, coefficients=coefficients)
        u = np.ascontiguousarray(reconstructed.zonal)
        v = np.ascontiguousarray(reconstructed.meridional)
    else:
        u = _source_to_batch(cell_wind[0], shape, "cell_wind.u", vertical_order)
        v = _source_to_batch(cell_wind[1], shape, "cell_wind.v", vertical_order)

    w = None
    if vertical_velocity is not None:
        w = _source_to_batch(
            vertical_velocity,
            interface_shape,
            "vertical_velocity",
            vertical_order,
        )

    dtype = metric_density.dtype
    surface_fields = _copy_field_mapping(surface, n_cells, "surface", dtype)
    # Static category/index fields stay integers; casting IVGTYP/ISLTYP to a
    # thermodynamic float here would defeat the exact gpuwm launch contract.
    static_fields = _copy_field_mapping(static, n_cells, "static", None)
    # MPAS Registry.xml lines 1225-1228 declare latCell/lonCell in radians;
    # gpuwm's radiation and Noah-MP public constructors explicitly consume
    # latitude_deg/longitude_deg. Refuse an ambiguous caller override and
    # perform the one declared conversion here.
    for source_name, target_name in (
        ("latCell", "latitude_deg"),
        ("lonCell", "longitude_deg"),
    ):
        if (
            target_name in static_fields
            or target_name.removesuffix("_deg") in static_fields
        ):
            _refuse(
                f"static.{target_name}",
                "caller supplied",
                "MPAS mesh coordinates are the authority and are converted from radians",
                f"mesh.{source_name} with no static coordinate override",
            )
        value = _mesh_value(mesh, source_name)
        if value is not None:
            radians = _column_field(value, n_cells, f"mesh.{source_name}", dtype)
            static_fields[target_name] = np.ascontiguousarray(
                np.rad2deg(radians), dtype=dtype
            )
    if "terrain_height" not in static_fields:
        static_fields["terrain_height"] = np.ascontiguousarray(z_interface[0, :].copy())

    batch = ColumnBatch(
        metric_density=metric_density,
        dry_density=dry_density,
        moist_density=moist_density,
        modified_theta=modified_theta,
        theta=theta,
        temperature=temperature,
        eos_pressure=eos_pressure_batch,
        eos_pressure_interface=eos_pressure_interface_batch,
        hydrostatic_moist_pressure=hydrostatic_moist_pressure_batch,
        hydrostatic_moist_pressure_interface=(
            hydrostatic_moist_pressure_interface_batch
        ),
        hydrostatic_dry_pressure=hydrostatic_dry_pressure_batch,
        hydrostatic_dry_pressure_interface=(hydrostatic_dry_pressure_interface_batch),
        height=height,
        height_interface=z_interface,
        layer_thickness=layer_thickness,
        u=u,
        v=v,
        w=w,
        scalars=scalar_batch,
        surface=surface_fields,
        static=static_fields,
        time_seconds=float(state.time_seconds),
    )
    batch.validate()
    context = MpasPackingContext(
        mesh=mesh,
        scalar_names=canonical,
        metric_density_level_cell=_batch_to_source(metric_density, vertical_order),
        edge_metric_density_level_edge=(
            edge_metric_density_logical
            if vertical_order == "surface_to_top"
            else np.ascontiguousarray(edge_metric_density_logical[::-1])
        ),
        vertical_order=vertical_order,
    )
    return PackedMpasColumns(batch=batch, context=context)


@dataclass(frozen=True, slots=True)
class BackendRequirements:
    atmosphere: frozenset[str] = frozenset()
    scalars: frozenset[str] = frozenset()
    surface: frozenset[str] = frozenset()
    static: frozenset[str] = frozenset()


@dataclass(slots=True)
class ColumnTendencies:
    """Uncoupled physical rates returned by a column backend.

    MPAS-A v8.2.3 initializes ``tend_rho_physics`` to zero and none of its
    admitted column packages adds a dry metric-density source
    (the module-level frozen MPAS to-dynamics source window). A density tendency is
    therefore intentionally absent instead of exposing a speculative API.
    """

    du: FloatArray
    dv: FloatArray
    dtheta: FloatArray
    dscalars: MutableMapping[str, FloatArray] = field(default_factory=dict)
    surface_updates: MutableMapping[str, FloatArray] = field(default_factory=dict)
    diagnostics: MutableMapping[str, FloatArray] = field(default_factory=dict)

    @classmethod
    def zeros(cls, batch: ColumnBatch) -> "ColumnTendencies":
        return cls(
            du=np.zeros_like(batch.u),
            dv=np.zeros_like(batch.v),
            dtheta=np.zeros_like(batch.theta),
            dscalars={},
        )

    def validate(self, batch: ColumnBatch) -> None:
        shape = (batch.n_levels, batch.n_columns)
        for name in ("du", "dv", "dtheta"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind != "f":
                raise ValueError(f"backend {name} must be floating with shape {shape}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"backend {name} contains non-finite values")
        canonical: dict[str, FloatArray] = {}
        for raw_name, raw_value in self.dscalars.items():
            name = _canonical_scalar_name(raw_name)
            if name in canonical:
                raise ValueError(f"backend returned duplicate scalar tendency {name!r}")
            value = np.asarray(raw_value)
            if value.shape != shape or value.dtype.kind != "f":
                raise ValueError(
                    f"backend dscalars[{raw_name!r}] must be floating with shape {shape}"
                )
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    f"backend dscalars[{raw_name!r}] is non-finite"
                )
            canonical[name] = np.ascontiguousarray(value)
        self.dscalars = canonical
        for name, value in self.surface_updates.items():
            array = np.asarray(value)
            if array.shape[0] != batch.n_columns or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"backend surface_updates[{name!r}] must be finite and column-first"
                )


@runtime_checkable
class PhysicsBackend(Protocol):
    """Backend protocol; implementations run physics but never know the mesh."""

    name: str

    def requirements(self, selection: GPUWMSchemeSelection) -> BackendRequirements: ...

    def run(
        self,
        batch: ColumnBatch,
        selection: GPUWMSchemeSelection,
        dt: float,
    ) -> ColumnTendencies: ...


# The DOMAIN of ``_required_scalar_names``: the gpuwm microphysics selectors an
# MPAS scheme NAME can actually route to.  Derived from ``_ROUTES`` rather than
# restated, so a new route cannot drift away from the ladder that has to carry
# it, and so the ladder stays exhaustive over its own domain by construction
# rather than by anyone remembering.
_ROUTED_MP_PHYSICS: frozenset[int] = frozenset(
    _ROUTES["config_microp_scheme"].values()
)


# gpuwm microphysics selectors DELIBERATELY given no row in the ladder below,
# each with its reason, in the shape ``_UNROUTED_REASONS`` uses for MPAS scheme
# names.  An omission recorded here is a decision; an omission recorded nowhere
# reads as an oversight, and the next reader has to guess which it was.
#
# RETIRED 2026-08-31: the mp=50 (P3) reason row.  Its own text was the
# spec of what had to be true first -- "adding a P3 row to _ROUTES
# means giving this seam qir and qib first" -- and that is exactly what
# the retiring commit does: _SCALAR_ALIASES names the rime pair
# (qir/qirim, qib/birim), the mp_p3 route lands as declared hexcore
# vocabulary (HEXCORE_SELECTOR_EXTENSIONS -- the "no MPAS name to
# arrive by" half of the reason, answered by the 2026-08-20
# physics-goes-its-own-way ruling rather than by inventing an MPAS
# name), the ladder below requires all EIGHT of WRF's mp=50 scalars
# (module_microphysics_driver.F:1569-1602: qv, qc, qr, qi, ni, nr,
# qir, qib -- and neither qs nor qg, which P3 does not have), and the
# engine column batch transports them (gpuwm/core/mpas_column_batch.py
# microphysics_scheme="p3", landed on the 2.6.1 line).  The six-of-
# eight state substitution the refusal existed to stop is now
# structurally impossible: the row requires the rime pair by name.
_UNROUTED_MP_PHYSICS_REASONS: Mapping[int, str] = MappingProxyType({})


def _required_scalar_names(selection: GPUWMSchemeSelection) -> frozenset[str]:
    """Exact MPAS/gpuwm scalar state needed by the selected schemes.

    MPAS-A allocates and round-trips ``ni`` and ``nr`` for both Thompson
    variants and additionally ``nc``, ``nifa``, and ``nwfa`` for aerosol
    Thompson (mpas_atmphys_interface.F lines 653-702 and 927-970;
    mpas_atmphys_driver_microphysics.F lines 394-426). gpuwm's DomainState carries
    the same mp=8/mp=28 number fields. Omitting them is a state substitution,
    not an optional diagnostic.

    THE BREAKAGE THE REFUSAL BELOW PREVENTS, measured 2026-08-29.  This ladder
    answers "which scalars must be present", and its silence is not neutral: a
    selector with no row falls through every arm and declares ``{"qv"}`` alone,
    so ``_validate_requirements`` accepts a batch with no cloud, no rain and no
    ice state and the backend then runs on state that was substituted rather
    than supplied -- with no error anywhere.  The rows cover exactly
    ``_ROUTED_MP_PHYSICS`` today, which is why no forecast has met that; nothing
    but this refusal keeps it true.  gpuwm publishes microphysics selectors this
    seam does not route (9, 10, 16 and 18 among them, and
    ``GPUWM_SELECTOR_INVENTORY`` above is a second copy that can drift from
    ``_ROUTES``), so the arrival is a new route away, not a hypothetical.

    The mp=50 row (2026-08-31) is WRF's own P3 transport set
    (module_microphysics_driver.F:1569-1602 into mp_p3_wrapper_wrf):
    the four moist masses, both number moments, and the rime pair
    qir/qib.  P3 has ONE ice category -- it writes neither qs nor qg --
    so the (6, 8, 28) ice row deliberately does not include 50:
    requiring qs/qg would demand two species the scheme does not have,
    and the engine column batch refuses them by name on a P3 batch.
    """

    if selection.mp_physics not in _ROUTED_MP_PHYSICS:
        # Report the MPAS knob the user actually set when a name got this far
        # (the real arrival: somebody added a route).  A selection built
        # directly from selectors has no name to quote, so that case names the
        # gpuwm selector instead of putting an integer under a name knob.
        mpas_name = selection.mpas_names.get("config_microp_scheme")
        _refuse(
            "config_microp_scheme" if mpas_name is not None else "mp_physics",
            mpas_name if mpas_name is not None else selection.mp_physics,
            _UNROUTED_MP_PHYSICS_REASONS.get(
                selection.mp_physics,
                f"gpuwm mp_physics={selection.mp_physics} has no scalar "
                "requirement row in this seam, and only "
                f"{sorted(_ROUTED_MP_PHYSICS)} are reachable from an MPAS "
                "scheme name; without a row the selection would be accepted "
                "declaring qv alone, which is a state substitution",
            ),
            "config_microp_scheme='off'",
        )

    required: set[str] = set()
    if selection.mp_physics or selection.cu_physics or selection.bl_pbl_physics:
        required.add("qv")
    if selection.mp_physics in (1, 6, 8, 28, 50) or selection.bl_pbl_physics:
        required.add("qc")
    if selection.mp_physics in (1, 6, 8, 28, 50):
        required.add("qr")
    if selection.mp_physics in (6, 8, 28):
        required.update(("qi", "qs", "qg"))
    if selection.mp_physics in (8, 28):
        required.update(("nr", "ni"))
    if selection.mp_physics == 28:
        required.update(("nc", "nifa", "nwfa"))
    if selection.mp_physics == 50:
        # P3 one-category: total ice mass + number, rain number, and the
        # rime mass/volume pair.  No qs, no qg -- see the docstring.
        required.update(("qi", "ni", "nr", "qir", "qib"))
    return frozenset(required)


def _gpuwm_requirements(selection: GPUWMSchemeSelection) -> BackendRequirements:
    atmosphere: set[str] = set()
    surface: set[str] = set()
    static: set[str] = set()
    if selection.mp_physics or selection.cu_physics:
        atmosphere.add("w")
    # MPAS-A microphysics consumes EOS pres_p. Convection, surface, PBL, and
    # radiation consume hydrostatic moist pres_hyd_p/pres2_hyd_p instead
    # (driver_convection.F lines 490-491; driver_sfclayer.F lines 893-1048;
    # driver_pbl.F lines 828-915; driver_radiation_{lw,sw}.F). Never let the always
    # present EOS field satisfy this requirement.
    if (
        selection.cu_physics
        or selection.sf_sfclay_physics
        or selection.sf_surface_physics
        or selection.bl_pbl_physics
        or selection.ra_lw_physics
        or selection.ra_sw_physics
    ):
        atmosphere.update(
            (
                "hydrostatic_moist_pressure",
                "hydrostatic_moist_pressure_interface",
            )
        )
    if (
        selection.sf_sfclay_physics
        or selection.sf_surface_physics
        or selection.bl_pbl_physics
    ):
        surface.update(("surface_temperature", "landmask"))
        static.update(("latitude_deg", "longitude_deg"))
    if selection.sf_surface_physics:
        surface.update(("soil_temperature", "soil_moisture", "vegetation_fraction"))
        static.update(("landuse_category", "soil_category"))
    if selection.ra_lw_physics or selection.ra_sw_physics:
        static.update(("latitude_deg", "longitude_deg", "cloud_fraction"))
    return BackendRequirements(
        atmosphere=frozenset(atmosphere),
        scalars=_required_scalar_names(selection),
        surface=frozenset(surface),
        static=frozenset(static),
    )


def _validate_requirements(
    batch: ColumnBatch, requirements: BackendRequirements
) -> None:
    for name in sorted(requirements.atmosphere):
        if not hasattr(batch, name) or getattr(batch, name) is None:
            _refuse(
                f"atmosphere.{name}",
                None,
                "the selected physics backend declares this field required",
                f"pack_mpas_columns(..., {name if name != 'w' else 'vertical_velocity'}=...)",
            )
    for group_name in ("scalars", "surface", "static"):
        group = getattr(batch, group_name)
        for name in sorted(getattr(requirements, group_name)):
            if name not in group:
                _refuse(
                    f"{group_name}.{name}",
                    None,
                    "the selected physics backend declares this field required",
                    f"pack_mpas_columns(..., {group_name}={{'{name}': ...}})",
                )


def _batch_arrays(batch: ColumnBatch) -> dict[str, FloatArray]:
    result: dict[str, FloatArray] = {}
    for name in (
        "metric_density",
        "dry_density",
        "moist_density",
        "modified_theta",
        "theta",
        "temperature",
        "eos_pressure",
        "eos_pressure_interface",
        "hydrostatic_moist_pressure",
        "hydrostatic_moist_pressure_interface",
        "hydrostatic_dry_pressure",
        "hydrostatic_dry_pressure_interface",
        "height",
        "height_interface",
        "layer_thickness",
        "u",
        "v",
        "w",
    ):
        value = getattr(batch, name)
        if value is not None:
            result[f"atmosphere.{name}"] = value
    for group_name in ("scalars", "surface", "static"):
        result.update(
            {
                f"{group_name}.{name}": value
                for name, value in getattr(batch, group_name).items()
            }
        )
    return result


def execute_column_backend(
    batch: ColumnBatch,
    selection: GPUWMSchemeSelection,
    backend: PhysicsBackend,
    *,
    dt: float,
    mutation: BackendMutation = "isolate",
) -> ColumnTendencies:
    """Run a backend with explicit ownership/mutation semantics.

    ``isolate`` (default) gives the backend a deep copy.  ``forbid`` gives it
    the supplied batch but detects any in-place change.  ``allow`` deliberately
    exposes in-place backend updates on the supplied batch.  None of the three
    modes mutates an MPAS :class:`PrognosticState`; packing always owns copies.
    """

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if mutation not in ("forbid", "isolate", "allow"):
        _refuse(
            "backend_mutation",
            mutation,
            "allowed policies are forbid, isolate, and allow",
            "backend_mutation='isolate'",
        )
    batch.validate()
    requirements = backend.requirements(selection)
    _validate_requirements(batch, requirements)
    target = batch.copy() if mutation == "isolate" else batch
    before = (
        {name: value.copy() for name, value in _batch_arrays(target).items()}
        if mutation == "forbid"
        else None
    )
    result = backend.run(target, selection, float(dt))
    if not isinstance(result, ColumnTendencies):
        raise TypeError(
            f"physics backend {getattr(backend, 'name', type(backend).__name__)!r} "
            "must return ColumnTendencies"
        )
    if before is not None:
        after = _batch_arrays(target)
        changed = [
            name
            for name, value in before.items()
            if not np.array_equal(value, after[name])
        ]
        if changed:
            # A "forbid" policy is observationally immutable as well as
            # diagnostic: restore every mutated buffer before raising.
            for name in changed:
                after[name][...] = before[name]
            _refuse(
                "backend_mutation",
                getattr(backend, "name", type(backend).__name__),
                f"the backend changed forbidden input fields {changed}",
                "backend_mutation='isolate' or 'allow'",
            )
    result.validate(target)
    return result


@dataclass(slots=True)
class DeterministicNumpyBackend:
    """CUDA-free seam double; its constants are not a physics scheme."""

    du_rate: float = 0.25
    dv_rate: float = -0.5
    theta_rate: float = 0.01
    scalar_rates: Mapping[str, float] = field(default_factory=lambda: {"qv": 1.0e-6})
    required: BackendRequirements = field(default_factory=BackendRequirements)
    mutate: str | None = None
    name: str = field(default="numpy-deterministic-mock", init=False)
    calls: int = field(default=0, init=False)

    def requirements(self, selection: GPUWMSchemeSelection) -> BackendRequirements:
        return self.required

    def run(
        self,
        batch: ColumnBatch,
        selection: GPUWMSchemeSelection,
        dt: float,
    ) -> ColumnTendencies:
        self.calls += 1
        if self.mutate is not None:
            arrays = _batch_arrays(batch)
            if self.mutate not in arrays:
                raise KeyError(f"mock mutation field {self.mutate!r} is absent")
            arrays[self.mutate][...] += arrays[self.mutate].dtype.type(1.0)
        if not selection.enabled:
            return ColumnTendencies.zeros(batch)
        dtype = batch.dtype
        return ColumnTendencies(
            du=np.full(batch.u.shape, dtype.type(self.du_rate), dtype=dtype),
            dv=np.full(batch.v.shape, dtype.type(self.dv_rate), dtype=dtype),
            dtheta=np.full(batch.theta.shape, dtype.type(self.theta_rate), dtype=dtype),
            dscalars={
                _canonical_scalar_name(name): np.full(
                    batch.theta.shape, dtype.type(rate), dtype=dtype
                )
                for name, rate in self.scalar_rates.items()
            },
        )


def _coerce_gpuwm_result(result: object, batch: ColumnBatch) -> ColumnTendencies:
    if isinstance(result, ColumnTendencies):
        return result
    if not isinstance(result, Mapping):
        raise TypeError(
            "gpuwm run_mpas_column_batch must return a mapping or ColumnTendencies"
        )
    forbidden_density_keys = sorted(
        {"ddry_mass", "dmetric_density"}.intersection(result)
    )
    if forbidden_density_keys:
        _refuse(
            "backend.density_tendency",
            forbidden_density_keys,
            "MPAS-A v8.2.3 column physics has no metric-density source",
            "a result without ddry_mass/dmetric_density",
        )
    try:
        du = result["du"]
        dv = result["dv"]
        dtheta = result["dtheta"]
    except KeyError as exc:
        raise ValueError(
            f"gpuwm column result is missing required key {exc.args[0]!r}"
        ) from exc
    return ColumnTendencies(
        du=np.ascontiguousarray(np.asarray(du, dtype=batch.dtype)),
        dv=np.ascontiguousarray(np.asarray(dv, dtype=batch.dtype)),
        dtheta=np.ascontiguousarray(np.asarray(dtheta, dtype=batch.dtype)),
        dscalars={
            str(name): np.ascontiguousarray(np.asarray(value, dtype=batch.dtype))
            for name, value in result.get("dscalars", {}).items()
        },
        surface_updates={
            str(name): np.ascontiguousarray(np.asarray(value, dtype=batch.dtype))
            for name, value in result.get("surface_updates", {}).items()
        },
        diagnostics={
            str(name): np.ascontiguousarray(np.asarray(value))
            for name, value in result.get("diagnostics", {}).items()
        },
    )


@dataclass(slots=True)
class LazyGPUWMBackend:
    """Lazy adapter for a gpuwm tree publishing an MPAS column entry point.

    A runner may be injected for a pinned gpuwm integration build.  Otherwise
    ``gpuwm.core.physics`` is imported only on the first call and must expose
    ``run_mpas_column_batch(batch=..., selectors=..., mpas_selectors=...,
    dt=...)``.  The stock
    public ``PhysicsDriver.compute(DomainState, RunConfig)`` is intentionally
    not coerced: it owns an ARW grid and is not an honest MPAS column API.
    """

    runner: Callable[..., object] | None = None
    module_name: str = "gpuwm.core.physics"
    entrypoint: str = "run_mpas_column_batch"
    declared_requirements: BackendRequirements | None = None
    name: str = field(default="gpuwm", init=False)

    def requirements(self, selection: GPUWMSchemeSelection) -> BackendRequirements:
        return (
            _gpuwm_requirements(selection)
            if self.declared_requirements is None
            else self.declared_requirements
        )

    def _runner(self) -> Callable[..., object]:
        if self.runner is not None:
            return self.runner
        try:
            module = import_module(self.module_name)
        except Exception as exc:
            _refuse(
                "physics_backend",
                "gpuwm",
                f"lazy import of {self.module_name!r} failed ({type(exc).__name__}: {exc})",
                "an importable CUDA-capable gpuwm runtime or DeterministicNumpyBackend()",
            )
        candidate = getattr(module, self.entrypoint, None)
        if not callable(candidate):
            _refuse(
                "physics_backend",
                "gpuwm",
                f"{self.module_name} exposes PhysicsDriver.compute for ARW DomainState but not the required public {self.entrypoint}",
                f"a gpuwm build exporting {self.module_name}.{self.entrypoint}",
            )
        self.runner = candidate
        return candidate

    def run(
        self,
        batch: ColumnBatch,
        selection: GPUWMSchemeSelection,
        dt: float,
    ) -> ColumnTendencies:
        result = self._runner()(
            batch=batch,
            selectors=selection.as_gpuwm_config(),
            mpas_selectors=dict(selection.mpas_names),
            dt=float(dt),
        )
        return _coerce_gpuwm_result(result, batch)


@dataclass(slots=True)
class MpasPhysicsTendencies:
    """Dry-mass-coupled tendencies in the MPAS prognostic layouts."""

    rho: FloatArray
    rho_u: FloatArray
    rho_theta: FloatArray
    scalars: FloatArray

    def validate_like(self, state: PrognosticState) -> None:
        expected = {
            "rho": state.rho.shape,
            "rho_u": state.rho_u.shape,
            "rho_theta": state.rho_theta.shape,
            "scalars": state.scalars.shape,
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind != "f":
                raise ValueError(f"{name} tendency shape {value.shape} != {shape}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} tendency contains non-finite values")


def _cell_acceleration_to_edges(
    mesh: object, du_level_cell: FloatArray, dv_level_cell: FloatArray
) -> FloatArray:
    n_levels, n_cells = du_level_cell.shape
    if dv_level_cell.shape != (n_levels, n_cells):
        raise ValueError("du and dv must share (nLevels,nCells)")
    sphere_value = _mesh_value(mesh, "on_a_sphere", True)
    if isinstance(sphere_value, (bytes, np.bytes_)):
        sphere_value = sphere_value.decode("ascii", errors="ignore")
    sphere = (
        str(sphere_value).strip().lower() in {"yes", "true", "t", "1"}
        if isinstance(sphere_value, str)
        else bool(sphere_value)
    )
    if sphere:
        lon = _mesh_array(mesh, "lonCell").astype(du_level_cell.dtype, copy=False)
        lat = _mesh_array(mesh, "latCell").astype(du_level_cell.dtype, copy=False)
        local = np.stack(
            (du_level_cell, dv_level_cell, np.zeros_like(du_level_cell)), axis=-1
        )
        vectors = vector_lon_lat_r_to_r3(local, lon[None, :], lat[None, :])
    else:
        vectors = np.stack(
            (du_level_cell, dv_level_cell, np.zeros_like(du_level_cell)), axis=-1
        )
    geometry = initialize_vector_geometry(mesh)
    normals = geometry.edge_normal_vectors.astype(du_level_cell.dtype, copy=False)
    n_edges = normals.shape[0]
    cells = _cells_on_edge(mesh, n_edges)
    if np.any(cells < 0) or np.any(cells >= n_cells):
        _refuse(
            "regional_boundary_momentum",
            "edge without two cells",
            "MPAS tend_toEdges needs both cell-centered wind tendencies",
            "explicit boundary/halo cell tendencies",
        )
    average = du_level_cell.dtype.type(0.5) * (
        vectors[:, cells[:, 0], :] + vectors[:, cells[:, 1], :]
    )
    normal = average[..., 0] * normals[None, :, 0]
    normal += average[..., 1] * normals[None, :, 1]
    normal += average[..., 2] * normals[None, :, 2]
    return np.ascontiguousarray(normal)


def unpack_mpas_tendencies(
    packed: PackedMpasColumns,
    column_tendencies: ColumnTendencies,
    *,
    rv_over_rd: float = RV_OVER_RD,
) -> MpasPhysicsTendencies:
    """Unpack and dry-mass-couple backend rates into MPAS arrays.

    The modified-theta coupling is the frozen
    ``mpas_atmphys_todynamics.F:436-437`` expression, rearranged as
    ``d(theta_m) = (1+Rv/Rd*qv)*d(theta) + Rv/Rd*theta*d(qv)``.
    """

    batch = packed.batch
    context = packed.context
    column_tendencies.validate(batch)
    unknown = sorted(set(column_tendencies.dscalars) - set(context.scalar_names))
    if unknown:
        _refuse(
            "backend.dscalars",
            unknown,
            "the backend returned species absent from the MPAS scalar registry for this state",
            "matching scalar_names entries in pack_mpas_columns",
        )
    dtype = batch.dtype
    metric_density_batch = batch.metric_density
    qv = batch.scalars.get("qv", np.zeros_like(metric_density_batch))
    dqv = column_tendencies.dscalars.get("qv", np.zeros_like(metric_density_batch))
    moisture_factor = 1.0 + dtype.type(rv_over_rd) * qv
    dmodified_theta = (
        moisture_factor * column_tendencies.dtheta
        + dtype.type(rv_over_rd) * batch.theta * dqv
    )
    # MPAS-A column physics supplies no metric-density source. Coupling is
    # therefore exactly rho_zz*d(theta_m)/dt (to-dynamics lines 436-437).
    rho_theta_rate = metric_density_batch * dmodified_theta

    scalar_rate = np.zeros(
        (len(context.scalar_names), batch.n_levels, batch.n_columns), dtype=dtype
    )
    for index, name in enumerate(context.scalar_names):
        dq = column_tendencies.dscalars.get(name)
        if dq is not None:
            scalar_rate[index] = metric_density_batch * dq

    du_level_cell = _batch_to_source(column_tendencies.du, context.vertical_order)
    dv_level_cell = _batch_to_source(column_tendencies.dv, context.vertical_order)
    edge_rate = _cell_acceleration_to_edges(context.mesh, du_level_cell, dv_level_cell)
    rho_u_rate = edge_rate * context.edge_metric_density_level_edge

    result = MpasPhysicsTendencies(
        rho=np.zeros_like(context.metric_density_level_cell),
        rho_u=np.ascontiguousarray(rho_u_rate),
        rho_theta=_batch_to_source(rho_theta_rate, context.vertical_order),
        scalars=np.ascontiguousarray(
            scalar_rate
            if context.vertical_order == "surface_to_top"
            else scalar_rate[:, ::-1, :]
        ),
    )
    return result


def apply_mpas_tendencies(
    state: PrognosticState,
    tendencies: MpasPhysicsTendencies,
    *,
    dt: float,
    inplace: bool = False,
) -> PrognosticState:
    """Apply coupled tendencies transactionally, copying unless requested.

    MPAS scalar state is an uncoupled mixing ratio.  Application therefore
    updates ``rho*q`` and divides by the new dry mass; ``rho_theta`` and
    ``rho_u`` are already conservative prognostics.  All candidate arrays are
    validated before an in-place commit, so a failed bound check cannot leave
    a partially updated state.
    """

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    tendencies.validate_like(state)
    delta = state.rho.dtype.type(dt)
    rho = np.asarray(state.rho) + delta * tendencies.rho
    rho_theta = np.asarray(state.rho_theta) + delta * tendencies.rho_theta
    rho_u = np.asarray(state.rho_u) + delta * tendencies.rho_u
    scalar_mass = np.asarray(state.scalars) * np.asarray(state.rho)[None]
    scalar_mass = scalar_mass + delta * tendencies.scalars
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise FloatingPointError(
            "physics update produced non-positive or non-finite dry mass"
        )
    scalars = scalar_mass / rho[None]
    candidates = (rho_theta, rho_u, scalars)
    if any(np.any(~np.isfinite(value)) for value in candidates):
        raise FloatingPointError("physics update produced a non-finite prognostic")
    target = state if inplace else state.copy()
    target.rho[...] = rho
    target.rho_theta[...] = rho_theta
    target.rho_u[...] = rho_u
    target.scalars[...] = scalars
    target.time_seconds = float(state.time_seconds) + float(dt)
    return target


def apply_surface_updates(
    batch: ColumnBatch,
    column_tendencies: ColumnTendencies,
    *,
    inplace: bool = False,
) -> ColumnBatch:
    """Commit explicitly returned persistent surface state transactionally.

    Diagnostics belong in ``ColumnTendencies.diagnostics``.  A surface update
    must name an already packed persistent field; creating a new carrier on a
    later step would make restart/state inventory depend on call history.
    """

    column_tendencies.validate(batch)
    unknown = sorted(set(column_tendencies.surface_updates) - set(batch.surface))
    if unknown:
        _refuse(
            "backend.surface_updates",
            unknown,
            "surface updates must target fields present in the packed persistent state",
            "matching surface={...} entries in pack_mpas_columns",
        )
    candidates: dict[str, NDArray[Any]] = {}
    for name, value in column_tendencies.surface_updates.items():
        target_value = batch.surface[name]
        candidate = np.asarray(value)
        if candidate.shape != target_value.shape:
            raise ValueError(
                f"surface update {name!r} shape {candidate.shape} != {target_value.shape}"
            )
        if not np.can_cast(candidate.dtype, target_value.dtype, casting="same_kind"):
            raise TypeError(
                f"surface update {name!r} dtype {candidate.dtype} cannot be cast "
                f"to persistent dtype {target_value.dtype}"
            )
        candidate = np.ascontiguousarray(candidate, dtype=target_value.dtype)
        if not np.all(np.isfinite(candidate)):
            raise FloatingPointError(
                f"surface update {name!r} contains non-finite values"
            )
        candidates[name] = candidate
    target = batch if inplace else batch.copy()
    for name, value in candidates.items():
        target.surface[name][...] = value
    return target


def run_mpas_physics(
    packed: PackedMpasColumns,
    selection: GPUWMSchemeSelection,
    backend: PhysicsBackend,
    *,
    dt: float,
    backend_mutation: BackendMutation = "isolate",
    commit_surface_updates: bool = False,
) -> MpasPhysicsTendencies:
    """Pack-complete execution helper: run columns, then unpack tendencies."""

    column = execute_column_backend(
        packed.batch,
        selection,
        backend,
        dt=dt,
        mutation=backend_mutation,
    )
    if commit_surface_updates:
        apply_surface_updates(packed.batch, column, inplace=True)
    return unpack_mpas_tendencies(packed, column)


__all__ = [
    "BackendMutation",
    "BackendRequirements",
    "ColumnBatch",
    "ColumnTendencies",
    "DeterministicNumpyBackend",
    "GPUWM_SELECTOR_INVENTORY",
    "GPUWMSchemeSelection",
    "LazyGPUWMBackend",
    "MPAS_SELECTOR_INVENTORY",
    "MpasPhysicsTendencies",
    "PackedMpasColumns",
    "PhysicsBackend",
    "RV_OVER_RD",
    "apply_mpas_tendencies",
    "apply_surface_updates",
    "execute_column_backend",
    "pack_mpas_columns",
    "resolve_mpas_physics",
    "run_mpas_physics",
    "unpack_mpas_tendencies",
]
