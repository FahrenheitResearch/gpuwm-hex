"""Structured-data initialization for the frozen MPAS-A v8.2.3 target.

This module is deliberately a broad CPU authority rather than a collection of
format-specific shortcuts.  It provides one strict ingestion contract, a
reusable spherical lat/lon remap, vertical interpolation, and assembly of the
coupled prognostic fields used by :class:`~hexcore.state.PrognosticState`.

The source-backed portions follow ``init_atm_case_gfs`` in frozen
``src/core_init_atmosphere/mpas_init_atm_cases.F``:

* structured horizontal interpolation and cyclic longitude handling:
  lines 3398-4265;
* edge-normal wind projection: lines 4659-4667;
* height interpolation and logarithmic pressure interpolation: lines
  4670-4773;
* humidity, density, hydrostatic adjustment, and metric coupling: lines
  4779-4954;
* terrain-following vertical momentum and final diagnostics: lines 4955-5002.

Only rectilinear latitude/longitude sources are accepted here.  Unsupported
source layouts and behavioral knobs raise :class:`ConfigurationRefusal` with
the exact offending name; there is no nearest-field or nearest-layout guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import ConfigurationRefusal
from .state import PrognosticState
from .vertical import VerticalGrid
from .wps_intermediate import IntermediateField, WpsIntermediateReader


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]

GRAVITY = 9.80616
DRY_AIR_GAS_CONSTANT = 287.0
WATER_VAPOR_GAS_CONSTANT = 461.6
DRY_AIR_CP = 7.0 * DRY_AIR_GAS_CONSTANT / 2.0
DRY_AIR_CV = DRY_AIR_CP - DRY_AIR_GAS_CONSTANT
REFERENCE_PRESSURE = 100_000.0
MOIST_THETA_FACTOR = 1.61
REFERENCE_TEMPERATURE = 250.0


def _refuse(knob: str, value: object, reason: str, declaration: str) -> None:
    raise ConfigurationRefusal(knob, value, reason, declaration)


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


def _canonical_units(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("−", "-")
        .replace("^", "**")
        .replace(" ", "")
        .replace("degrees_", "degree_")
    )


def _variable_units(variable: object, logical_name: str) -> str:
    attrs = getattr(variable, "attrs", {})
    units = attrs.get("units") if isinstance(attrs, Mapping) else None
    if units is None:
        _refuse(
            f"source_units.{logical_name}",
            None,
            "unit-free meteorological fields cannot be converted without guessing",
            f"an explicit units attribute on {logical_name!r}",
        )
    return _canonical_units(units)


def _convert_angle(values: ArrayLike, units: str, logical_name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if units in {
        "degree",
        "degrees",
        "degree_north",
        "degree_east",
        "degreesnorth",
        "degreeseast",
    }:
        return np.deg2rad(array)
    if units in {"rad", "radian", "radians"}:
        return array
    _refuse(
        f"source_units.{logical_name}",
        units,
        "only angular degrees or radians are supported",
        f"{logical_name}.attrs['units']='degrees'",
    )


def _convert_pressure(values: ArrayLike, units: str, logical_name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if units in {"pa", "pascal", "pascals"}:
        result = array
    elif units in {"hpa", "mb", "mbar", "millibar", "millibars"}:
        result = array * 100.0
    else:
        _refuse(
            f"source_units.{logical_name}",
            units,
            "pressure conversion supports only Pa and hPa/mb",
            f"{logical_name}.attrs['units']='Pa'",
        )
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{logical_name} pressure must be finite and strictly positive")
    return result


def _convert_temperature(values: ArrayLike, units: str, logical_name: str) -> FloatArray:
    if units not in {"k", "kelvin", "kelvins"}:
        _refuse(
            f"source_units.{logical_name}",
            units,
            "the MPAS initialization authority requires absolute temperature",
            f"{logical_name}.attrs['units']='K'",
        )
    return np.asarray(values, dtype=np.float64)


def _convert_velocity(values: ArrayLike, units: str, logical_name: str) -> FloatArray:
    if units not in {
        "m/s",
        "ms-1",
        "ms**-1",
        "meterpersecond",
        "meterspersecond",
    }:
        _refuse(
            f"source_units.{logical_name}",
            units,
            "wind conversion supports only metres per second",
            f"{logical_name}.attrs['units']='m s-1'",
        )
    return np.asarray(values, dtype=np.float64)


def _convert_height(
    values: ArrayLike,
    units: str,
    logical_name: str,
    kind: Literal["geopotential_height", "geopotential"],
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if kind == "geopotential_height":
        if units not in {"m", "meter", "meters", "gpm"}:
            _refuse(
                f"source_units.{logical_name}",
                units,
                "geopotential-height input must be in metres or geopotential metres",
                f"{logical_name}.attrs['units']='m'",
            )
        return array
    if kind == "geopotential":
        if units not in {
            "m**2s**-2",
            "m2s-2",
            "m2/s2",
            "jkg-1",
            "jkg**-1",
        }:
            _refuse(
                f"source_units.{logical_name}",
                units,
                "geopotential input must have squared-velocity units",
                f"{logical_name}.attrs['units']='m**2 s**-2'",
            )
        return array / GRAVITY
    _refuse(
        f"source_kind.{logical_name}",
        kind,
        "only geopotential height and geopotential are implemented",
        f"{logical_name}_kind='geopotential_height'",
    )


def _convert_fraction(values: ArrayLike, units: str, logical_name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if units in {"1", "fraction", "unitless", "dimensionless"}:
        result = array
    elif units in {"%", "percent", "percentage"}:
        result = array * 0.01
    else:
        _refuse(
            f"source_units.{logical_name}",
            units,
            "fraction fields must be dimensionless or percent",
            f"{logical_name}.attrs['units']='1'",
        )
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{logical_name} must lie in [0, 1] after unit conversion")
    return result


def _convert_specific_humidity(
    values: ArrayLike, units: str, logical_name: str
) -> FloatArray:
    if units not in {
        "1",
        "kgkg-1",
        "kgkg**-1",
        "kg/kg",
        "fraction",
        "unitless",
    }:
        _refuse(
            f"source_units.{logical_name}",
            units,
            "specific humidity must be a mass fraction",
            f"{logical_name}.attrs['units']='kg kg-1'",
        )
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result >= 1.0)):
        raise ValueError(f"{logical_name} specific humidity must lie in [0, 1)")
    return result


@dataclass(frozen=True, slots=True)
class FieldMap:
    """Exact source names and interpretations for a structured dataset.

    Defaults match common ``cfgrib`` short names.  They are not aliases: if a
    producer uses another name, callers must declare it here.
    """

    latitude: str = "latitude"
    longitude: str = "longitude"
    pressure: str = "isobaricInhPa"
    pressure_field: str | None = None
    temperature: str = "t"
    zonal_wind: str = "u"
    meridional_wind: str = "v"
    geopotential_height: str = "gh"
    geopotential_height_kind: Literal["geopotential_height", "geopotential"] = (
        "geopotential_height"
    )
    specific_humidity: str | None = "q"
    relative_humidity: str | None = None
    surface_pressure: str = "sp"
    terrain: str = "orog"
    terrain_kind: Literal["geopotential_height", "geopotential"] = (
        "geopotential_height"
    )
    land_fraction: str = "lsm"
    skin_temperature: str = "skt"

    def requested_names(self) -> tuple[str, ...]:
        names = (
            self.latitude,
            self.longitude,
            self.pressure,
            self.pressure_field,
            self.temperature,
            self.zonal_wind,
            self.meridional_wind,
            self.geopotential_height,
            self.specific_humidity,
            self.relative_humidity,
            self.surface_pressure,
            self.terrain,
            self.land_fraction,
            self.skin_temperature,
        )
        return tuple(dict.fromkeys(name for name in names if name is not None))


def _get_variable(source: object, name: str, logical_name: str) -> object:
    try:
        return source[name]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        _refuse(
            f"source_field.{logical_name}",
            name,
            "the declared field is absent; field-name guessing is disabled",
            f"FieldMap({logical_name}='<present variable name>')",
        )


def _variable_dims(variable: object, logical_name: str) -> tuple[str, ...]:
    dims = getattr(variable, "dims", None)
    if dims is None:
        _refuse(
            f"source_layout.{logical_name}",
            type(variable).__name__,
            "the xarray adapter needs named dimensions",
            "an xarray.DataArray with named dimensions",
        )
    return tuple(str(dim) for dim in dims)


def _select_and_transpose(
    variable: object,
    *,
    logical_name: str,
    required_dims: tuple[str, ...],
    selectors: Mapping[str, int],
) -> FloatArray:
    selected = variable
    dims = _variable_dims(selected, logical_name)
    sizes = getattr(selected, "sizes", {})
    for dim in dims:
        if dim in required_dims:
            continue
        if dim in selectors:
            index = int(selectors[dim])
            try:
                selected = selected.isel({dim: index})  # type: ignore[attr-defined]
            except (AttributeError, IndexError, ValueError) as error:
                raise ValueError(
                    f"selector {dim}={index} is invalid for {logical_name}"
                ) from error
        elif int(sizes.get(dim, -1)) == 1:
            selected = selected.isel({dim: 0})  # type: ignore[attr-defined]
        else:
            _refuse(
                f"source_layout.{logical_name}.{dim}",
                sizes.get(dim),
                "non-singleton extra dimensions require an explicit integer selector",
                f"selectors={{'{dim}': 0}}",
            )

    dims = _variable_dims(selected, logical_name)
    if len(dims) != len(required_dims) or set(dims) != set(required_dims):
        _refuse(
            f"source_layout.{logical_name}",
            dims,
            f"expected exactly the declared dimensions {required_dims}",
            f"dimensions ordered as {required_dims}",
        )
    try:
        selected = selected.transpose(*required_dims)  # type: ignore[attr-defined]
    except (AttributeError, ValueError) as error:
        raise ValueError(f"could not transpose {logical_name} to {required_dims}") from error
    return np.asarray(selected.values, dtype=np.float64)  # type: ignore[attr-defined]


@dataclass(slots=True)
class StructuredAtmosphere:
    """Canonical full structured first-guess fields.

    Latitude/longitude are radians; pressure is Pa; height is metres; all
    three-dimensional arrays use ``(source_level, latitude, longitude)``.
    Exactly one or both humidity representations may be present.  A dry run
    must be requested later; missing humidity is never silently interpreted as
    zero.
    """

    latitude: FloatArray
    longitude: FloatArray
    pressure: FloatArray
    temperature: FloatArray
    zonal_wind: FloatArray
    meridional_wind: FloatArray
    geopotential_height: FloatArray
    surface_pressure: FloatArray
    terrain: FloatArray
    land_fraction: FloatArray
    skin_temperature: FloatArray
    specific_humidity: FloatArray | None = None
    relative_humidity: FloatArray | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "StructuredAtmosphere":
        self.latitude = np.asarray(self.latitude, dtype=np.float64)
        self.longitude = np.asarray(self.longitude, dtype=np.float64)
        if self.latitude.ndim != 1 or self.longitude.ndim != 1:
            _refuse(
                "source_layout.latitude_longitude",
                (self.latitude.shape, self.longitude.shape),
                "curvilinear and unstructured input coordinates are not ported",
                "one-dimensional rectilinear latitude and longitude coordinates",
            )
        if self.latitude.size < 2 or self.longitude.size < 2:
            raise ValueError("source latitude and longitude each need at least two points")
        if np.any(np.abs(self.latitude) > np.pi / 2.0 + 1.0e-12):
            raise ValueError("source latitude lies outside [-pi/2, pi/2]")

        n_level = int(np.asarray(self.temperature).shape[0])
        shape3 = (n_level, self.latitude.size, self.longitude.size)
        shape2 = (self.latitude.size, self.longitude.size)
        for name in (
            "temperature",
            "zonal_wind",
            "meridional_wind",
            "geopotential_height",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind != "f":
                value = value.astype(np.float64)
            if value.shape != shape3:
                raise ValueError(f"{name} shape {value.shape} != {shape3}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} contains non-finite values")
            setattr(self, name, value)

        pressure = np.asarray(self.pressure)
        if pressure.dtype.kind != "f":
            pressure = pressure.astype(np.float64)
        if pressure.shape == (n_level,):
            # A coordinate-level pressure source should not cost another full
            # global 3-D allocation (the real GFS intermediate file is >700 MB).
            pressure = np.broadcast_to(pressure[:, None, None], shape3)
        if pressure.shape != shape3:
            raise ValueError(f"pressure shape {pressure.shape} must be ({n_level},) or {shape3}")
        if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0.0):
            raise ValueError("pressure must be finite and strictly positive")
        self.pressure = pressure

        for name in (
            "surface_pressure",
            "terrain",
            "land_fraction",
            "skin_temperature",
        ):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind != "f":
                value = value.astype(np.float64)
            if value.shape != shape2:
                raise ValueError(f"{name} shape {value.shape} != {shape2}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} contains non-finite values")
            setattr(self, name, value)
        if np.any(self.surface_pressure <= 0.0):
            raise ValueError("surface_pressure must be strictly positive")
        if np.any((self.land_fraction < 0.0) | (self.land_fraction > 1.0)):
            raise ValueError("land_fraction must lie in [0, 1]")

        for name in ("specific_humidity", "relative_humidity"):
            raw = getattr(self, name)
            if raw is None:
                continue
            value = np.asarray(raw)
            if value.dtype.kind != "f":
                value = value.astype(np.float64)
            if value.shape != shape3:
                raise ValueError(f"{name} shape {value.shape} != {shape3}")
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"{name} contains non-finite values")
            outside = (
                (value < 0.0) | (value >= 1.0)
                if name == "specific_humidity"
                else (value < 0.0) | (value > 1.0)
            )
            if np.any(outside):
                raise ValueError(f"{name} is outside its canonical range")
            setattr(self, name, value)

        return self

    @classmethod
    def from_xarray(
        cls,
        source: object,
        *,
        fields: FieldMap = FieldMap(),
        selectors: Mapping[str, int] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "StructuredAtmosphere":
        """Materialize an xarray Dataset (or exact DataArray mapping).

        Dimension order is normalized by name, not by position.  Additional
        non-singleton dimensions such as ensemble member or time must be
        selected explicitly.
        """

        select = {} if selectors is None else dict(selectors)
        latitude_variable = _get_variable(source, fields.latitude, "latitude")
        longitude_variable = _get_variable(source, fields.longitude, "longitude")
        lat_dims = _variable_dims(latitude_variable, "latitude")
        lon_dims = _variable_dims(longitude_variable, "longitude")
        if len(lat_dims) != 1 or len(lon_dims) != 1:
            _refuse(
                "source_layout.latitude_longitude",
                (lat_dims, lon_dims),
                "only rectilinear one-dimensional coordinates are implemented",
                "1-D latitude and longitude coordinates",
            )
        lat_dim, lon_dim = lat_dims[0], lon_dims[0]
        latitude = _convert_angle(
            np.asarray(latitude_variable.values),  # type: ignore[attr-defined]
            _variable_units(latitude_variable, "latitude"),
            "latitude",
        )
        longitude = _convert_angle(
            np.asarray(longitude_variable.values),  # type: ignore[attr-defined]
            _variable_units(longitude_variable, "longitude"),
            "longitude",
        )

        pressure_coordinate = _get_variable(source, fields.pressure, "pressure")
        pressure_dims = _variable_dims(pressure_coordinate, "pressure")
        if len(pressure_dims) != 1:
            _refuse(
                "source_layout.pressure",
                pressure_dims,
                "the vertical pressure coordinate must be one-dimensional",
                "a 1-D pressure coordinate plus optional 3-D pressure_field",
            )
        level_dim = pressure_dims[0]
        required3 = (level_dim, lat_dim, lon_dim)
        required2 = (lat_dim, lon_dim)

        def field3(name: str, logical: str) -> tuple[FloatArray, object]:
            variable = _get_variable(source, name, logical)
            return (
                _select_and_transpose(
                    variable,
                    logical_name=logical,
                    required_dims=required3,
                    selectors=select,
                ),
                variable,
            )

        def field2(name: str, logical: str) -> tuple[FloatArray, object]:
            variable = _get_variable(source, name, logical)
            return (
                _select_and_transpose(
                    variable,
                    logical_name=logical,
                    required_dims=required2,
                    selectors=select,
                ),
                variable,
            )

        if fields.pressure_field is None:
            pressure = _convert_pressure(
                np.asarray(pressure_coordinate.values),  # type: ignore[attr-defined]
                _variable_units(pressure_coordinate, "pressure"),
                "pressure",
            )
        else:
            pressure_raw, pressure_variable = field3(
                fields.pressure_field, "pressure_field"
            )
            pressure = _convert_pressure(
                pressure_raw,
                _variable_units(pressure_variable, "pressure_field"),
                "pressure_field",
            )

        temperature_raw, temperature_variable = field3(fields.temperature, "temperature")
        u_raw, u_variable = field3(fields.zonal_wind, "zonal_wind")
        v_raw, v_variable = field3(fields.meridional_wind, "meridional_wind")
        height_raw, height_variable = field3(
            fields.geopotential_height, "geopotential_height"
        )
        surface_pressure_raw, surface_pressure_variable = field2(
            fields.surface_pressure, "surface_pressure"
        )
        terrain_raw, terrain_variable = field2(fields.terrain, "terrain")
        land_raw, land_variable = field2(fields.land_fraction, "land_fraction")
        skin_raw, skin_variable = field2(fields.skin_temperature, "skin_temperature")

        specific_humidity: FloatArray | None = None
        if fields.specific_humidity is not None:
            raw, variable = field3(fields.specific_humidity, "specific_humidity")
            specific_humidity = _convert_specific_humidity(
                raw,
                _variable_units(variable, "specific_humidity"),
                "specific_humidity",
            )
        relative_humidity: FloatArray | None = None
        if fields.relative_humidity is not None:
            raw, variable = field3(fields.relative_humidity, "relative_humidity")
            relative_humidity = _convert_fraction(
                raw,
                _variable_units(variable, "relative_humidity"),
                "relative_humidity",
            )

        result = cls(
            latitude=latitude,
            longitude=longitude,
            pressure=pressure,
            temperature=_convert_temperature(
                temperature_raw,
                _variable_units(temperature_variable, "temperature"),
                "temperature",
            ),
            zonal_wind=_convert_velocity(
                u_raw, _variable_units(u_variable, "zonal_wind"), "zonal_wind"
            ),
            meridional_wind=_convert_velocity(
                v_raw,
                _variable_units(v_variable, "meridional_wind"),
                "meridional_wind",
            ),
            geopotential_height=_convert_height(
                height_raw,
                _variable_units(height_variable, "geopotential_height"),
                "geopotential_height",
                fields.geopotential_height_kind,
            ),
            specific_humidity=specific_humidity,
            relative_humidity=relative_humidity,
            surface_pressure=_convert_pressure(
                surface_pressure_raw,
                _variable_units(surface_pressure_variable, "surface_pressure"),
                "surface_pressure",
            ),
            terrain=_convert_height(
                terrain_raw,
                _variable_units(terrain_variable, "terrain"),
                "terrain",
                fields.terrain_kind,
            ),
            land_fraction=_convert_fraction(
                land_raw,
                _variable_units(land_variable, "land_fraction"),
                "land_fraction",
            ),
            skin_temperature=_convert_temperature(
                skin_raw,
                _variable_units(skin_variable, "skin_temperature"),
                "skin_temperature",
            ),
            provenance={
                "adapter": "xarray",
                "field_map": fields,
                "selectors": select,
                **({} if provenance is None else dict(provenance)),
            },
        )
        return result.validate()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        fields: FieldMap = FieldMap(),
        selectors: Mapping[str, int] | None = None,
        source_format: Literal[
            "auto", "grib", "xarray", "wps_intermediate"
        ] = "auto",
        engine: str | None = None,
    ) -> "StructuredAtmosphere":
        """Read a complete GRIB collection or xarray-supported file.

        For GRIB, all cfgrib groups are inspected so surface and isobaric
        fields can coexist in one logical input.  Duplicate declared names are
        resolved only when one candidate carries the declared pressure-level
        dimension; otherwise the exact field name is refused as ambiguous.
        """

        source_path = Path(path).expanduser().resolve(strict=True)
        if source_format not in ("auto", "grib", "xarray", "wps_intermediate"):
            _refuse(
                "source_format",
                source_format,
                "only GRIB and xarray-supported sources are implemented",
                "source_format='auto'",
            )
        if source_format == "auto":
            suffix = source_path.suffix.lower()
            if _looks_like_wps_intermediate(source_path):
                source_format = "wps_intermediate"
            else:
                source_format = (
                    "grib"
                    if suffix in {".grib", ".grb", ".grib2", ".grb2"}
                    else "xarray"
                )

        if source_format == "wps_intermediate":
            if fields != FieldMap():
                _refuse(
                    "field_map",
                    fields,
                    "WPS intermediate fields have frozen exact names and are not xarray aliases",
                    "the default FieldMap() or source_format='xarray'",
                )
            if selectors:
                _refuse(
                    "selectors",
                    dict(selectors),
                    "WPS intermediate records have no unnamed ensemble/time dimensions",
                    "selectors=None",
                )
            if engine is not None:
                _refuse(
                    "source_engine",
                    engine,
                    "the WPS intermediate reader is a direct Fortran-record parser",
                    "engine=None",
                )
            return cls.from_wps_intermediate(source_path)

        if source_format == "grib":
            if engine not in (None, "cfgrib"):
                _refuse(
                    "source_engine",
                    engine,
                    "GRIB full-file grouping is implemented through cfgrib",
                    "engine='cfgrib'",
                )
            try:
                import cfgrib  # type: ignore[import-not-found]
            except ImportError:
                _refuse(
                    "cfgrib_runtime",
                    "missing",
                    "GRIB ingestion requires the optional cfgrib runtime",
                    "an environment with cfgrib and ecCodes installed",
                )
            groups = cfgrib.open_datasets(
                str(source_path), backend_kwargs={"indexpath": ""}
            )
            try:
                mapping = _select_grib_group_variables(groups, fields)
                return cls.from_xarray(
                    mapping,
                    fields=fields,
                    selectors=selectors,
                    provenance={
                        "source_path": str(source_path),
                        "source_format": "grib",
                        "group_count": len(groups),
                    },
                )
            finally:
                for group in groups:
                    group.close()

        try:
            import xarray as xr  # type: ignore[import-not-found]
        except ImportError:
            _refuse(
                "xarray_runtime",
                "missing",
                "file ingestion requires the optional xarray runtime",
                "an environment with xarray installed",
            )
        dataset = xr.open_dataset(source_path, engine=engine)
        try:
            return cls.from_xarray(
                dataset,
                fields=fields,
                selectors=selectors,
                provenance={
                    "source_path": str(source_path),
                    "source_format": "xarray",
                    "engine": engine,
                },
            )
        finally:
            dataset.close()

    @classmethod
    def from_wps_intermediate(cls, path: str | Path) -> "StructuredAtmosphere":
        """Materialize the real MPAS/WPS full-file first guess.

        The reader itself is the direct frozen-format authority in
        :mod:`hexcore.wps_intermediate`.  This adapter selects only the
        fields used by the currently supported GFS branch and seek-skips all
        other slabs.  It keeps pressure-level fields as float32, limiting the
        resident data for the 1440x721/33-level GFS files to roughly their
        useful payload rather than widening the entire file to float64.
        """

        source_path = Path(path).expanduser().resolve(strict=True)
        wanted = {
            "GHT",
            "TT",
            "UU",
            "VV",
            "RH",
            "PSFC",
            "SOILHGT",
            "LANDSEA",
            "SKINTEMP",
        }
        by_name: dict[str, dict[float, IntermediateField]] = {}
        first: IntermediateField | None = None
        endian: str
        # Pass one is metadata-only: the reader seek-skips each multi-megabyte
        # slab, letting us allocate final arrays once without retaining a list
        # of record-sized copies.
        with WpsIntermediateReader(source_path) as reader:
            endian = "big" if reader.endian == ">" else "little"
            for record in reader.iter_fields(include=wanted, load_values=False):
                if first is None:
                    first = record
                _validate_wps_record_geometry(first, record)
                level_map = by_name.setdefault(record.field, {})
                if record.level in level_map:
                    _refuse(
                        f"wps_field.{record.field}",
                        record.level,
                        "the full file contains a duplicate field/level record",
                        "one record per field and vertical level",
                    )
                level_map[record.level] = record
        if first is None:
            _refuse(
                "wps_fields",
                "empty",
                "none of the required GFS records were found",
                "a WPS intermediate file containing GHT/TT/UU/VV/RH and surface fields",
            )
        if first.projection.code != 0:
            _refuse(
                "wps_projection",
                first.projection.name,
                "the initialization remap currently admits the frozen rectilinear lat/lon branch",
                "a PROJ_LATLON WPS intermediate file",
            )
        if first.projection.start_location != "SWCORNER":
            _refuse(
                "wps_start_location",
                first.projection.start_location,
                "the rectilinear coordinate construction is authoritative for SWCORNER files",
                "a WPS file with startloc='SWCORNER'",
            )

        for required in wanted:
            if required not in by_name:
                _refuse(
                    f"wps_field.{required}",
                    None,
                    "the supported GFS initialization path requires this exact record",
                    f"a full intermediate file containing {required}",
                )
        pressure_levels = sorted(by_name["GHT"], reverse=True)
        if len(pressure_levels) < 2 or any(level <= 0.0 for level in pressure_levels):
            _refuse(
                "wps_pressure_levels",
                pressure_levels,
                "GHT must provide at least two positive isobaric levels",
                "two or more GHT pressure-level records",
            )
        for name in ("TT", "UU", "VV", "RH"):
            missing = [level for level in pressure_levels if level not in by_name[name]]
            if missing:
                _refuse(
                    f"wps_levels.{name}",
                    tuple(missing),
                    "pressure-level fields must cover every GHT level",
                    f"{name} records at all declared GHT levels",
                )
        atmospheric_names = ("TT", "UU", "VV", "RH", "GHT")
        surface_names = ("PSFC", "SOILHGT", "LANDSEA", "SKINTEMP")
        for name in atmospheric_names:
            _require_wps_units(
                [by_name[name][level] for level in pressure_levels], name
            )
        for name in surface_names:
            records = list(by_name[name].values())
            if len(records) != 1:
                _refuse(
                    f"wps_field.{name}",
                    tuple(record.level for record in records),
                    "surface authority requires exactly one record",
                    f"one {name} record",
                )
            _require_wps_units(records, name)

        n_levels = len(pressure_levels)
        shape3 = (n_levels, first.ny, first.nx)
        shape2 = (first.ny, first.nx)
        atmospheric = {
            name: np.empty(shape3, dtype=np.float32) for name in atmospheric_names
        }
        surfaces = {
            name: np.empty(shape2, dtype=np.float32) for name in surface_names
        }
        level_index = {level: index for index, level in enumerate(pressure_levels)}
        seen_levels: dict[str, set[float]] = {name: set() for name in atmospheric_names}
        seen_surfaces: set[str] = set()
        # Pass two streams each useful slab directly into its final level slot.
        with WpsIntermediateReader(source_path) as reader:
            for record in reader.iter_fields(include=wanted, load_values=True):
                _validate_wps_record_geometry(first, record)
                if record.values is None:
                    raise AssertionError("selected WPS slab was not materialized")
                if record.field in atmospheric and record.level in level_index:
                    atmospheric[record.field][level_index[record.level]] = record.values.T
                    seen_levels[record.field].add(record.level)
                elif record.field in surfaces:
                    surfaces[record.field][...] = record.values.T
                    seen_surfaces.add(record.field)
        for name in atmospheric_names:
            missing = set(pressure_levels) - seen_levels[name]
            if missing:
                raise AssertionError(f"metadata/value scans disagree for {name}: {missing}")
        if seen_surfaces != set(surface_names):
            raise AssertionError(
                f"metadata/value scans disagree for surfaces: {set(surface_names) - seen_surfaces}"
            )

        projection = first.projection
        latitude = np.deg2rad(
            projection.start_latitude
            + np.arange(first.ny, dtype=np.float64) * projection.delta_latitude
        )
        longitude = np.deg2rad(
            projection.start_longitude
            + np.arange(first.nx, dtype=np.float64) * projection.delta_longitude
        )
        relative_humidity = atmospheric["RH"]
        relative_humidity *= np.float32(0.01)
        result = cls(
            latitude=latitude,
            longitude=longitude,
            pressure=np.asarray(pressure_levels, dtype=np.float64),
            temperature=atmospheric["TT"],
            zonal_wind=atmospheric["UU"],
            meridional_wind=atmospheric["VV"],
            geopotential_height=atmospheric["GHT"],
            relative_humidity=relative_humidity,
            specific_humidity=None,
            surface_pressure=surfaces["PSFC"],
            terrain=surfaces["SOILHGT"],
            land_fraction=surfaces["LANDSEA"],
            skin_temperature=surfaces["SKINTEMP"],
            provenance={
                "adapter": "wps_intermediate",
                "source_path": str(source_path),
                "record_endian": endian,
                "valid_time": first.valid_time,
                "forecast_hour": first.forecast_hour,
                "map_source": first.map_source,
                "projection": first.projection,
                "evidence": "implemented-unverified",
            },
        )
        return result.validate()


def _looks_like_wps_intermediate(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(12)
    if len(prefix) != 12:
        return False
    for byteorder in ("big", "little"):
        if (
            int.from_bytes(prefix[0:4], byteorder=byteorder, signed=True) == 4
            and int.from_bytes(prefix[4:8], byteorder=byteorder, signed=True)
            in (3, 4, 5)
            and int.from_bytes(prefix[8:12], byteorder=byteorder, signed=True) == 4
        ):
            return True
    return False


def _validate_wps_record_geometry(
    authority: IntermediateField, candidate: IntermediateField
) -> None:
    mismatches: dict[str, object] = {}
    for name in ("version", "valid_time", "forecast_hour", "nx", "ny"):
        first_value = getattr(authority, name)
        candidate_value = getattr(candidate, name)
        if candidate_value != first_value:
            mismatches[name] = (first_value, candidate_value)
    if candidate.projection != authority.projection:
        mismatches["projection"] = (authority.projection, candidate.projection)
    if mismatches:
        _refuse(
            "wps_record_geometry",
            mismatches,
            "all fields in one full-file first guess must share grid and valid-time metadata",
            "one homogeneous WPS intermediate file",
        )


_WPS_UNITS: dict[str, set[str]] = {
    "GHT": {"m", "gpm", "meter", "meters"},
    "TT": {"k", "kelvin", "kelvins"},
    "UU": {"m/s", "ms-1", "ms**-1"},
    "VV": {"m/s", "ms-1", "ms**-1"},
    "RH": {"%", "percent", "percentage"},
    "PSFC": {"pa", "pascal", "pascals"},
    "SOILHGT": {"m", "gpm", "meter", "meters"},
    "LANDSEA": {"1", "fraction", "proprtn", "unitless", "dimensionless"},
    "SKINTEMP": {"k", "kelvin", "kelvins"},
}


def _require_wps_units(records: Sequence[IntermediateField], name: str) -> None:
    accepted = _WPS_UNITS[name]
    encountered = {_canonical_units(record.units) for record in records}
    if not encountered <= accepted:
        _refuse(
            f"wps_units.{name}",
            tuple(sorted(encountered)),
            "the WPS field units do not match the supported GFS intermediate contract",
            f"{name} units in {tuple(sorted(accepted))}",
        )


def _select_grib_group_variables(groups: Sequence[object], fields: FieldMap) -> dict[str, object]:
    """Select exact requested DataArrays across cfgrib hypercube groups."""

    pressure_candidates: list[object] = []
    for group in groups:
        if fields.pressure in group:  # type: ignore[operator]
            pressure_candidates.append(group[fields.pressure])  # type: ignore[index]
    if not pressure_candidates:
        _refuse(
            "source_field.pressure",
            fields.pressure,
            "the pressure coordinate is absent from every GRIB group",
            "FieldMap(pressure='<present pressure coordinate>')",
        )
    pressure_dims = {
        _variable_dims(candidate, "pressure") for candidate in pressure_candidates
    }
    if len(pressure_dims) != 1:
        _refuse(
            "source_layout.pressure",
            tuple(sorted(pressure_dims)),
            "GRIB groups disagree on the declared pressure coordinate layout",
            "one pressure-coordinate dimension",
        )
    pressure_dim = next(iter(pressure_dims))[0]

    atmospheric = {
        fields.temperature,
        fields.zonal_wind,
        fields.meridional_wind,
        fields.geopotential_height,
        fields.pressure_field,
        fields.specific_humidity,
        fields.relative_humidity,
    }
    result: dict[str, object] = {}
    for name in fields.requested_names():
        candidates: list[object] = []
        for group in groups:
            if name in group:  # type: ignore[operator]
                candidates.append(group[name])  # type: ignore[index]
        if name == fields.pressure:
            candidates = pressure_candidates
        if name in atmospheric:
            level_candidates = [
                candidate
                for candidate in candidates
                if pressure_dim in _variable_dims(candidate, name)
            ]
            if level_candidates:
                candidates = level_candidates
        elif name not in {fields.latitude, fields.longitude, fields.pressure}:
            surface_candidates = [
                candidate
                for candidate in candidates
                if pressure_dim not in _variable_dims(candidate, name)
            ]
            if surface_candidates:
                candidates = surface_candidates
        if not candidates:
            _refuse(
                f"source_field.{name}",
                name,
                "the declared GRIB field is absent from every compatible group",
                f"FieldMap(...={name!r}) naming a present field",
            )
        signatures = {
            (_variable_dims(candidate, name), tuple(np.asarray(candidate).shape))
            for candidate in candidates
        }
        if len(candidates) > 1 and len(signatures) > 1:
            _refuse(
                f"source_field.{name}",
                tuple(sorted(signatures)),
                "multiple incompatible GRIB groups expose the declared name",
                "a unique short name or a pre-filtered GRIB source",
            )
        result[name] = candidates[0]
    return result


@dataclass(frozen=True, slots=True)
class SphericalRemap:
    """Cached bilinear weights for a rectilinear spherical latitude/lon grid."""

    source_shape: tuple[int, int]
    latitude_lower: IntArray
    latitude_upper: IntArray
    longitude_lower: IntArray
    longitude_upper: IntArray
    latitude_fraction: FloatArray
    longitude_fraction: FloatArray
    target_shape: tuple[int, ...]

    @classmethod
    def build(
        cls,
        source_latitude: ArrayLike,
        source_longitude: ArrayLike,
        target_latitude: ArrayLike,
        target_longitude: ArrayLike,
        *,
        longitude_mode: Literal["periodic_global", "bounded"] = "periodic_global",
        latitude_boundary: Literal["refuse", "clamp"] = "refuse",
    ) -> "SphericalRemap":
        """Build source-like four-point weights with explicit sphere handling.

        ``periodic_global`` implements the cyclic longitude halo used by
        frozen-source lines 3410-3420 and 3600-3624.  Latitude clamping is
        available because the Fortran path clamps projected ``y`` at the first
        and last row, but callers must request it explicitly.
        """

        lat = np.asarray(source_latitude, dtype=np.float64)
        lon = np.asarray(source_longitude, dtype=np.float64)
        target_lat = np.asarray(target_latitude, dtype=np.float64)
        target_lon = np.asarray(target_longitude, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1:
            _refuse(
                "source_layout.latitude_longitude",
                (lat.shape, lon.shape),
                "spherical remap requires rectilinear source coordinates",
                "one-dimensional latitude and longitude",
            )
        if lat.size < 2 or lon.size < 2:
            raise ValueError("spherical remap requires at least 2x2 source points")
        if target_lat.shape != target_lon.shape:
            raise ValueError("target latitude and longitude shapes must match")
        if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
            raise FloatingPointError("source coordinates contain non-finite values")
        if not np.all(np.isfinite(target_lat)) or not np.all(np.isfinite(target_lon)):
            raise FloatingPointError("target coordinates contain non-finite values")

        lat_order = np.argsort(lat, kind="stable")
        lat_sorted = lat[lat_order]
        if np.any(np.diff(lat_sorted) <= 0.0):
            _refuse(
                "source_layout.latitude",
                "duplicate coordinate",
                "latitude coordinates must be unique",
                "strictly monotonic latitude",
            )
        if latitude_boundary not in ("refuse", "clamp"):
            _refuse(
                "latitude_boundary",
                latitude_boundary,
                "only explicit refusal or endpoint clamping is supported",
                "latitude_boundary='refuse'",
            )
        # Published MPAS coordinates may be float32 while the WPS pole is
        # reconstructed from a float32 header into float64.  Treat sub-cell
        # representation noise as the same pole, not as physical extrapolation.
        latitude_tolerance = max(
            1.0e-10, 1.0e-4 * float(np.min(np.diff(lat_sorted)))
        )
        if latitude_boundary == "refuse" and (
            np.any(target_lat < lat_sorted[0] - latitude_tolerance)
            or np.any(target_lat > lat_sorted[-1] + latitude_tolerance)
        ):
            _refuse(
                "horizontal_domain_coverage.latitude",
                (float(target_lat.min()), float(target_lat.max())),
                "target cells extend beyond the structured latitude domain",
                "a source covering all target latitudes or latitude_boundary='clamp'",
            )
        target_lat_used = np.clip(target_lat, lat_sorted[0], lat_sorted[-1])
        lat_slot = np.searchsorted(lat_sorted, target_lat_used, side="right") - 1
        lat_slot = np.clip(lat_slot, 0, lat_sorted.size - 2)
        lat_fraction = (target_lat_used - lat_sorted[lat_slot]) / (
            lat_sorted[lat_slot + 1] - lat_sorted[lat_slot]
        )
        lat_lower = lat_order[lat_slot]
        lat_upper = lat_order[lat_slot + 1]

        if longitude_mode == "periodic_global":
            normalized = np.mod(lon, 2.0 * np.pi)
            lon_order = np.argsort(normalized, kind="stable")
            lon_sorted = normalized[lon_order]
            gaps = np.diff(np.concatenate((lon_sorted, lon_sorted[:1] + 2.0 * np.pi)))
            if np.any(gaps <= 1.0e-12):
                _refuse(
                    "source_layout.longitude",
                    "duplicate cyclic coordinate",
                    "cyclic endpoint duplicates must be removed explicitly",
                    "unique longitudes over [0, 2*pi)",
                )
            typical_gap = float(np.median(gaps))
            if float(gaps.max()) > 1.5 * typical_gap:
                _refuse(
                    "longitude_mode",
                    longitude_mode,
                    "the source longitude grid does not cover the globe without a large gap",
                    "longitude_mode='bounded' or a global source grid",
                )
            target_normalized = np.mod(target_lon, 2.0 * np.pi)
            lon_slot = np.searchsorted(lon_sorted, target_normalized, side="right") - 1
            lon_slot %= lon_sorted.size
            next_slot = (lon_slot + 1) % lon_sorted.size
            x0 = lon_sorted[lon_slot]
            x1 = lon_sorted[next_slot]
            x1 = np.where(next_slot == 0, x1 + 2.0 * np.pi, x1)
            x = np.where(target_normalized < x0, target_normalized + 2.0 * np.pi, target_normalized)
            lon_fraction = (x - x0) / (x1 - x0)
            lon_lower = lon_order[lon_slot]
            lon_upper = lon_order[next_slot]
        elif longitude_mode == "bounded":
            unwrapped = np.unwrap(lon)
            if np.all(np.diff(unwrapped) < 0.0):
                lon_order = np.arange(lon.size - 1, -1, -1)
                lon_sorted = unwrapped[::-1]
            elif np.all(np.diff(unwrapped) > 0.0):
                lon_order = np.arange(lon.size)
                lon_sorted = unwrapped
            else:
                _refuse(
                    "source_layout.longitude",
                    "non-monotonic bounded coordinate",
                    "bounded longitude must be monotonic after unwrapping",
                    "monotonic source longitude",
                )
            midpoint = 0.5 * (lon_sorted[0] + lon_sorted[-1])
            target_unwrapped = target_lon + 2.0 * np.pi * np.rint(
                (midpoint - target_lon) / (2.0 * np.pi)
            )
            if np.any(target_unwrapped < lon_sorted[0] - 1.0e-12) or np.any(
                target_unwrapped > lon_sorted[-1] + 1.0e-12
            ):
                _refuse(
                    "horizontal_domain_coverage.longitude",
                    (float(target_unwrapped.min()), float(target_unwrapped.max())),
                    "target cells extend beyond the bounded longitude domain",
                    "a source covering every target longitude",
                )
            lon_slot = np.searchsorted(lon_sorted, target_unwrapped, side="right") - 1
            lon_slot = np.clip(lon_slot, 0, lon_sorted.size - 2)
            lon_fraction = (target_unwrapped - lon_sorted[lon_slot]) / (
                lon_sorted[lon_slot + 1] - lon_sorted[lon_slot]
            )
            lon_lower = lon_order[lon_slot]
            lon_upper = lon_order[lon_slot + 1]
        else:
            _refuse(
                "longitude_mode",
                longitude_mode,
                "only periodic global and bounded longitude grids are implemented",
                "longitude_mode='periodic_global'",
            )

        return cls(
            source_shape=(lat.size, lon.size),
            latitude_lower=np.asarray(lat_lower, dtype=np.int64).ravel(),
            latitude_upper=np.asarray(lat_upper, dtype=np.int64).ravel(),
            longitude_lower=np.asarray(lon_lower, dtype=np.int64).ravel(),
            longitude_upper=np.asarray(lon_upper, dtype=np.int64).ravel(),
            latitude_fraction=np.asarray(lat_fraction, dtype=np.float64).ravel(),
            longitude_fraction=np.asarray(lon_fraction, dtype=np.float64).ravel(),
            target_shape=target_lat.shape,
        )

    def apply(self, values: ArrayLike) -> FloatArray:
        """Apply weights to fields whose final dimensions are latitude/lon."""

        source = np.asarray(values)
        if source.shape[-2:] != self.source_shape:
            raise ValueError(
                f"source field trailing shape {source.shape[-2:]} != {self.source_shape}"
            )
        if source.dtype.kind != "f":
            source = source.astype(np.float64)
        fy = self.latitude_fraction
        fx = self.longitude_fraction
        v00 = source[..., self.latitude_lower, self.longitude_lower]
        v01 = source[..., self.latitude_lower, self.longitude_upper]
        v10 = source[..., self.latitude_upper, self.longitude_lower]
        v11 = source[..., self.latitude_upper, self.longitude_upper]
        result = (
            (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01)
            + fy * ((1.0 - fx) * v10 + fx * v11)
        )
        final_shape = source.shape[:-2] + self.target_shape
        result = np.asarray(result).reshape(final_shape)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("horizontal interpolation produced non-finite values")
        return result


@dataclass(frozen=True, slots=True)
class RemappedFirstGuess:
    pressure: FloatArray
    temperature: FloatArray
    zonal_wind: FloatArray
    meridional_wind: FloatArray
    geopotential_height: FloatArray
    surface_pressure: FloatArray
    terrain: FloatArray
    land_fraction: FloatArray
    skin_temperature: FloatArray
    specific_humidity: FloatArray | None
    relative_humidity: FloatArray | None


def remap_to_mpas_cells(
    source: StructuredAtmosphere,
    mesh: object,
    *,
    longitude_mode: Literal["periodic_global", "bounded"] = "periodic_global",
    latitude_boundary: Literal["refuse", "clamp"] = "refuse",
) -> RemappedFirstGuess:
    """Horizontally interpolate all structured fields to MPAS cell centers."""

    source.validate()
    remap = SphericalRemap.build(
        source.latitude,
        source.longitude,
        _mesh_array(mesh, "latCell"),
        _mesh_array(mesh, "lonCell"),
        longitude_mode=longitude_mode,
        latitude_boundary=latitude_boundary,
    )
    return RemappedFirstGuess(
        pressure=remap.apply(source.pressure),
        temperature=remap.apply(source.temperature),
        zonal_wind=remap.apply(source.zonal_wind),
        meridional_wind=remap.apply(source.meridional_wind),
        geopotential_height=remap.apply(source.geopotential_height),
        specific_humidity=(
            None
            if source.specific_humidity is None
            else remap.apply(source.specific_humidity)
        ),
        relative_humidity=(
            None
            if source.relative_humidity is None
            else remap.apply(source.relative_humidity)
        ),
        surface_pressure=remap.apply(source.surface_pressure),
        terrain=remap.apply(source.terrain),
        land_fraction=remap.apply(source.land_fraction),
        skin_temperature=remap.apply(source.skin_temperature),
    )


@dataclass(frozen=True, slots=True)
class StaticSurfaceFields:
    terrain: FloatArray
    land_fraction: FloatArray
    landmask: IntArray
    skin_temperature: FloatArray
    surface_pressure: FloatArray


def assemble_static_surface_fields(
    mesh: object,
    first_guess: RemappedFirstGuess,
    *,
    terrain_source: Literal["mesh", "structured"] = "mesh",
    land_source: Literal["mesh", "structured"] = "mesh",
    skin_temperature_source: Literal["structured", "mesh"] = "structured",
    land_fraction_threshold: float = 0.5,
) -> StaticSurfaceFields:
    """Assemble target static/surface fields with explicit source authority."""

    n_cells = int(np.asarray(first_guess.surface_pressure).size)
    if not (0.0 <= land_fraction_threshold <= 1.0):
        _refuse(
            "land_fraction_threshold",
            land_fraction_threshold,
            "the categorical land mask threshold must lie in [0, 1]",
            "land_fraction_threshold=0.5",
        )

    if terrain_source == "mesh":
        try:
            terrain = _mesh_array(mesh, "ter").astype(np.float64, copy=True)
        except AttributeError:
            _refuse(
                "terrain_source",
                terrain_source,
                "the mesh has no frozen static field 'ter'",
                "terrain_source='structured' or a static MPAS mesh containing ter",
            )
    elif terrain_source == "structured":
        terrain = np.asarray(first_guess.terrain, dtype=np.float64).copy()
    else:
        _refuse(
            "terrain_source",
            terrain_source,
            "only mesh and structured terrain authorities are implemented",
            "terrain_source='mesh'",
        )

    if land_source == "mesh":
        try:
            landmask = _mesh_array(mesh, "landmask").astype(np.int64, copy=True)
        except AttributeError:
            _refuse(
                "land_source",
                land_source,
                "the mesh has no frozen static field 'landmask'",
                "land_source='structured' or a static MPAS mesh containing landmask",
            )
        if np.any((landmask != 0) & (landmask != 1)):
            raise ValueError("mesh landmask must be categorical 0/1")
        land_fraction = landmask.astype(np.float64)
    elif land_source == "structured":
        land_fraction = np.asarray(first_guess.land_fraction, dtype=np.float64).copy()
        landmask = (land_fraction >= land_fraction_threshold).astype(np.int64)
    else:
        _refuse(
            "land_source",
            land_source,
            "only mesh and structured land authorities are implemented",
            "land_source='mesh'",
        )

    if skin_temperature_source == "structured":
        skin = np.asarray(first_guess.skin_temperature, dtype=np.float64).copy()
    elif skin_temperature_source == "mesh":
        try:
            skin = _mesh_array(mesh, "skintemp").astype(np.float64, copy=True)
        except AttributeError:
            _refuse(
                "skin_temperature_source",
                skin_temperature_source,
                "the mesh has no field 'skintemp'",
                "skin_temperature_source='structured'",
            )
    else:
        _refuse(
            "skin_temperature_source",
            skin_temperature_source,
            "only structured and mesh skin-temperature authorities are implemented",
            "skin_temperature_source='structured'",
        )

    surface_pressure = np.asarray(first_guess.surface_pressure, dtype=np.float64).copy()
    for name, value in {
        "terrain": terrain,
        "land_fraction": land_fraction,
        "landmask": landmask,
        "skin_temperature": skin,
        "surface_pressure": surface_pressure,
    }.items():
        if value.shape != (n_cells,):
            raise ValueError(f"{name} shape {value.shape} != ({n_cells},)")
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"{name} contains non-finite values")
    if np.any(surface_pressure <= 0.0):
        raise ValueError("surface_pressure must be strictly positive")
    return StaticSurfaceFields(
        terrain=terrain,
        land_fraction=land_fraction,
        landmask=landmask,
        skin_temperature=skin,
        surface_pressure=surface_pressure,
    )


def _interpolate_columns(
    source_coordinate: ArrayLike,
    source_values: ArrayLike,
    target_coordinate: ArrayLike,
    *,
    extrapolation: Literal["linear", "constant", "refuse"],
    coordinate_name: str,
) -> FloatArray:
    coordinate = np.asarray(source_coordinate, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_coordinate, dtype=np.float64)
    if coordinate.shape != values.shape:
        raise ValueError(
            f"{coordinate_name} coordinate shape {coordinate.shape} != values shape {values.shape}"
        )
    if coordinate.ndim != 2 or target.ndim != 2 or coordinate.shape[1] != target.shape[1]:
        raise ValueError(
            f"vertical interpolation requires (level,column) arrays, got {coordinate.shape} and {target.shape}"
        )
    if coordinate.shape[0] < 2:
        raise ValueError("vertical interpolation requires at least two source levels")
    if extrapolation not in ("linear", "constant", "refuse"):
        _refuse(
            "vertical_extrapolation",
            extrapolation,
            "only linear, constant, and explicit refusal are implemented",
            "vertical_extrapolation='linear'",
        )

    output = np.empty(target.shape, dtype=np.result_type(values.dtype, np.float64))
    for column in range(coordinate.shape[1]):
        order = np.argsort(coordinate[:, column], kind="stable")
        x = coordinate[order, column]
        y = values[order, column]
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise FloatingPointError(
                f"vertical source column {column} contains non-finite values"
            )
        if np.any(np.diff(x) <= 0.0):
            _refuse(
                f"source_layout.{coordinate_name}",
                f"column={column}",
                "vertical coordinates must be unique within every column",
                "strictly monotonic source vertical coordinates",
            )
        query = target[:, column]
        if extrapolation == "refuse" and (
            np.any(query < x[0]) or np.any(query > x[-1])
        ):
            _refuse(
                "vertical_domain_coverage",
                f"column={column}",
                "model levels extend outside the source vertical domain",
                "a source covering every target level or explicit extrapolation",
            )
        clipped = np.clip(query, x[0], x[-1])
        result = np.interp(clipped, x, y)
        if extrapolation == "linear":
            below = query < x[0]
            above = query > x[-1]
            result[below] = y[0] + (query[below] - x[0]) * (y[1] - y[0]) / (
                x[1] - x[0]
            )
            result[above] = y[-1] + (query[above] - x[-1]) * (
                y[-1] - y[-2]
            ) / (x[-1] - x[-2])
        output[:, column] = result
    if not np.all(np.isfinite(output)):
        raise FloatingPointError("vertical interpolation produced non-finite values")
    return output


def vertical_interpolate_log_pressure(
    source_pressure: ArrayLike,
    source_values: ArrayLike,
    target_pressure: ArrayLike,
    *,
    extrapolation: Literal["linear", "constant", "refuse"] = "refuse",
) -> FloatArray:
    """Interpolate arbitrary column fields linearly in logarithmic pressure."""

    source_p = np.asarray(source_pressure, dtype=np.float64)
    target_p = np.asarray(target_pressure, dtype=np.float64)
    if np.any(source_p <= 0.0) or np.any(target_p <= 0.0):
        raise ValueError("log-pressure interpolation requires positive pressures")
    return _interpolate_columns(
        np.log(source_p),
        source_values,
        np.log(target_p),
        extrapolation=extrapolation,
        coordinate_name="pressure",
    )


def pressure_from_height_log_interpolation(
    source_height: ArrayLike,
    source_pressure: ArrayLike,
    target_height: ArrayLike,
    *,
    extrapolation: Literal["linear", "constant", "refuse"] = "linear",
) -> FloatArray:
    """Interpolate ``log(p)`` as a function of height, then exponentiate.

    This is the pressure branch at frozen-source lines 4753-4773.
    """

    pressure = np.asarray(source_pressure, dtype=np.float64)
    if np.any(pressure <= 0.0):
        raise ValueError("source pressure must be positive before taking logarithms")
    return np.exp(
        _interpolate_columns(
            source_height,
            np.log(pressure),
            target_height,
            extrapolation=extrapolation,
            coordinate_name="geopotential_height",
        )
    )


def saturation_mixing_ratio_liquid(pressure: ArrayLike, temperature: ArrayLike) -> FloatArray:
    """Frozen MPAS ``RSLF`` polynomial saturation mixing ratio.

    Direct transcription of
    ``src/core_init_atmosphere/mpas_atmphys_functions.F:155-180``.
    """

    p = np.asarray(pressure, dtype=np.float64)
    t = np.asarray(temperature, dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError("pressure and temperature shapes must match")
    if np.any(p <= 0.0) or not np.all(np.isfinite(p + t)):
        raise ValueError("saturation inputs must be finite with positive pressure")
    coefficients = (
        0.611583699e03,
        0.444606896e02,
        0.143177157e01,
        0.264224321e-1,
        0.299291081e-3,
        0.203154182e-5,
        0.702620698e-8,
        0.379534310e-11,
        -0.321582393e-13,
    )
    x = np.maximum(-80.0, t - 273.16)
    vapor_pressure = np.full_like(x, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        vapor_pressure = coefficient + x * vapor_pressure
    vapor_pressure = np.minimum(vapor_pressure, p * 0.15)
    return 0.622 * vapor_pressure / (p - vapor_pressure)


@dataclass(frozen=True, slots=True)
class ModelLevelFields:
    pressure: FloatArray
    temperature: FloatArray
    water_vapor_mixing_ratio: FloatArray
    zonal_wind: FloatArray
    meridional_wind: FloatArray
    edge_normal_wind: FloatArray


def interpolate_to_model_levels(
    source: StructuredAtmosphere,
    first_guess: RemappedFirstGuess,
    mesh: object,
    vertical: VerticalGrid,
    *,
    humidity_mode: Literal["specific_humidity", "relative_humidity", "dry", "auto"] = "auto",
    longitude_mode: Literal["periodic_global", "bounded"] = "periodic_global",
    latitude_boundary: Literal["refuse", "clamp"] = "refuse",
) -> ModelLevelFields:
    """Vertically interpolate cell fields and source-like edge winds."""

    target_height = 0.5 * (np.asarray(vertical.zgrid[:-1]) + np.asarray(vertical.zgrid[1:]))
    n_levels, n_cells = target_height.shape
    if first_guess.temperature.shape[1] != n_cells:
        raise ValueError("first-guess and vertical-grid cell counts disagree")

    pressure = pressure_from_height_log_interpolation(
        first_guess.geopotential_height,
        first_guess.pressure,
        target_height,
        extrapolation="linear",
    )
    temperature = _interpolate_columns(
        first_guess.geopotential_height,
        first_guess.temperature,
        target_height,
        extrapolation="linear",
        coordinate_name="geopotential_height",
    )
    zonal = _interpolate_columns(
        first_guess.geopotential_height,
        first_guess.zonal_wind,
        target_height,
        extrapolation="constant",
        coordinate_name="geopotential_height",
    )
    meridional = _interpolate_columns(
        first_guess.geopotential_height,
        first_guess.meridional_wind,
        target_height,
        extrapolation="constant",
        coordinate_name="geopotential_height",
    )

    has_specific = first_guess.specific_humidity is not None
    has_relative = first_guess.relative_humidity is not None
    if humidity_mode == "auto":
        if has_specific and not has_relative:
            humidity_mode = "specific_humidity"
        elif has_relative and not has_specific:
            humidity_mode = "relative_humidity"
        elif has_specific and has_relative:
            _refuse(
                "humidity_mode",
                humidity_mode,
                "both declared humidity authorities are present",
                "humidity_mode='specific_humidity' or 'relative_humidity'",
            )
        else:
            _refuse(
                "humidity_mode",
                humidity_mode,
                "no humidity field is present and zero moisture is not assumed",
                "humidity_mode='dry'",
            )
    if humidity_mode == "specific_humidity":
        if first_guess.specific_humidity is None:
            _refuse(
                "humidity_mode",
                humidity_mode,
                "the declared specific-humidity field is absent",
                "a specific_humidity source field or humidity_mode='dry'",
            )
        specific = _interpolate_columns(
            first_guess.geopotential_height,
            first_guess.specific_humidity,
            target_height,
            extrapolation="constant",
            coordinate_name="geopotential_height",
        )
        if np.any((specific < 0.0) | (specific >= 1.0)):
            raise ValueError("interpolated specific humidity lies outside [0, 1)")
        qv = specific / (1.0 - specific)
    elif humidity_mode == "relative_humidity":
        if first_guess.relative_humidity is None:
            _refuse(
                "humidity_mode",
                humidity_mode,
                "the declared relative-humidity field is absent",
                "a relative_humidity source field or humidity_mode='dry'",
            )
        relative = _interpolate_columns(
            first_guess.geopotential_height,
            first_guess.relative_humidity,
            target_height,
            extrapolation="constant",
            coordinate_name="geopotential_height",
        )
        if np.any((relative < 0.0) | (relative > 1.0 + 1.0e-12)):
            raise ValueError("interpolated relative humidity lies outside [0, 1]")
        qv = relative * saturation_mixing_ratio_liquid(pressure, temperature)
    elif humidity_mode == "dry":
        qv = np.zeros((n_levels, n_cells), dtype=np.float64)
    else:
        _refuse(
            "humidity_mode",
            humidity_mode,
            "only specific humidity, relative humidity, dry, and auto are implemented",
            "humidity_mode='specific_humidity'",
        )

    edge_remap = SphericalRemap.build(
        source.latitude,
        source.longitude,
        _mesh_array(mesh, "latEdge"),
        _mesh_array(mesh, "lonEdge"),
        longitude_mode=longitude_mode,
        latitude_boundary=latitude_boundary,
    )
    edge_height_source = edge_remap.apply(source.geopotential_height)
    edge_u_source = edge_remap.apply(source.zonal_wind)
    edge_v_source = edge_remap.apply(source.meridional_wind)
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    if cells.ndim != 2 or cells.shape[1] != 2 or np.any(cells < 0) or np.any(cells >= n_cells):
        _refuse(
            "regional_boundary_wind",
            "missing exterior cell",
            "edge target heights require two valid MPAS cells",
            "a closed global mesh or explicit regional boundary fields",
        )
    edge_target_height = 0.5 * (
        target_height[:, cells[:, 0]] + target_height[:, cells[:, 1]]
    )
    edge_u = _interpolate_columns(
        edge_height_source,
        edge_u_source,
        edge_target_height,
        extrapolation="constant",
        coordinate_name="geopotential_height_edge",
    )
    edge_v = _interpolate_columns(
        edge_height_source,
        edge_v_source,
        edge_target_height,
        extrapolation="constant",
        coordinate_name="geopotential_height_edge",
    )
    angle = _mesh_array(mesh, "angleEdge").astype(np.float64, copy=False)
    if angle.shape != (cells.shape[0],):
        raise ValueError("angleEdge shape disagrees with cellsOnEdge")
    edge_normal = np.cos(angle)[None, :] * edge_u + np.sin(angle)[None, :] * edge_v

    return ModelLevelFields(
        pressure=pressure,
        temperature=temperature,
        water_vapor_mixing_ratio=qv,
        zonal_wind=zonal,
        meridional_wind=meridional,
        edge_normal_wind=edge_normal,
    )


@dataclass(frozen=True, slots=True)
class StateAssemblyDiagnostics:
    pressure: FloatArray
    temperature: FloatArray
    potential_temperature: FloatArray
    modified_potential_temperature: FloatArray
    dry_density: FloatArray
    water_vapor_mixing_ratio: FloatArray
    surface_pressure: FloatArray
    hydrostatic_iterations: IntArray


@dataclass(frozen=True, slots=True)
class StateAssembly:
    state: PrognosticState
    diagnostics: StateAssemblyDiagnostics


def _initial_vertical_momentum_order2(
    mesh: object,
    vertical: VerticalGrid,
    rho_u: FloatArray,
) -> FloatArray:
    """Transcribe the order-two ``zb``/``rw`` branch at lines 3337-3387/4955-4993."""

    zgrid = np.asarray(vertical.zgrid)
    zz = np.asarray(vertical.zz)
    n_levels, n_cells = zz.shape
    n_edges = rho_u.shape[1]
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    dv_edge = _mesh_array(mesh, "dvEdge").astype(np.float64, copy=False)
    area_cell = _mesh_array(mesh, "areaCell").astype(np.float64, copy=False)
    if cells_on_edge.shape != (n_edges, 2) or np.any(cells_on_edge < 0):
        _refuse(
            "regional_boundary_metrics",
            "boundary edge",
            "initial vertical momentum needs two valid cells per edge",
            "a closed global mesh",
        )

    zb = np.empty((n_levels + 1, 2, n_edges), dtype=np.result_type(zgrid, rho_u))
    for edge, (cell0, cell1) in enumerate(cells_on_edge):
        z_edge = 0.5 * (zgrid[:, cell0] + zgrid[:, cell1])
        zb[:, 0, edge] = (z_edge - zgrid[:, cell0]) * dv_edge[edge] / area_cell[cell0]
        zb[:, 1, edge] = (z_edge - zgrid[:, cell1]) * dv_edge[edge] / area_cell[cell1]

    rho_w = np.zeros((n_levels + 1, n_cells), dtype=np.result_type(zgrid, rho_u))
    for cell in range(n_cells):
        for slot in range(int(n_edges_on_cell[cell])):
            edge = int(edges_on_cell[cell, slot])
            if edge < 0 or edge >= n_edges:
                raise ValueError(f"edgesOnCell has invalid used edge {edge}")
            side = 0 if cell == cells_on_edge[edge, 0] else 1
            if cell != cells_on_edge[edge, side]:
                raise ValueError("edgesOnCell and cellsOnEdge are not reciprocal")
            for level in range(1, n_levels):
                flux = (
                    vertical.fzm[level] * rho_u[level, edge]
                    + vertical.fzp[level] * rho_u[level - 1, edge]
                )
                metric = (
                    vertical.fzm[level] * zz[level, cell]
                    + vertical.fzp[level] * zz[level - 1, cell]
                )
                contribution = metric * zb[level, side, edge] * flux
                rho_w[level, cell] += -contribution if side == 0 else contribution
    return rho_w


def assemble_prognostic_state(
    mesh: object,
    vertical: VerticalGrid,
    fields: ModelLevelFields,
    *,
    scalar_names: Sequence[str] = ("qv",),
    hydrostatic_mode: Literal["mpas_v8.2.3"] = "mpas_v8.2.3",
    config_theta_adv_order: int = 2,
    hydrostatic_tolerance: float = 1.0e-4,
    hydrostatic_max_iterations: int = 30,
) -> StateAssembly:
    """Build coupled MPAS prognostics through the frozen moist hydrostatic path."""

    if hydrostatic_mode != "mpas_v8.2.3":
        _refuse(
            "hydrostatic_mode",
            hydrostatic_mode,
            "only the frozen MPAS v8.2.3 moist hydrostatic relation is authoritative",
            "hydrostatic_mode='mpas_v8.2.3'",
        )
    if config_theta_adv_order != 2:
        _refuse(
            "config_theta_adv_order",
            config_theta_adv_order,
            "order-three/four initial omega requires deriv_two metrics not yet in this port contract",
            "config_theta_adv_order=2",
        )
    if hydrostatic_tolerance <= 0.0:
        _refuse(
            "hydrostatic_tolerance",
            hydrostatic_tolerance,
            "the convergence tolerance must be positive",
            "hydrostatic_tolerance=1.0e-4",
        )
    if hydrostatic_max_iterations < 1:
        _refuse(
            "hydrostatic_max_iterations",
            hydrostatic_max_iterations,
            "at least one pressure-density iteration is required",
            "hydrostatic_max_iterations=30",
        )

    pressure_input = np.asarray(fields.pressure, dtype=np.float64)
    temperature_input = np.asarray(fields.temperature, dtype=np.float64)
    qv = np.asarray(fields.water_vapor_mixing_ratio, dtype=np.float64)
    if pressure_input.shape != temperature_input.shape or qv.shape != pressure_input.shape:
        raise ValueError("model pressure, temperature, and qv shapes must match")
    n_levels, n_cells = pressure_input.shape
    n_edges = int(_mesh_array(mesh, "cellsOnEdge").shape[0])
    if n_levels != vertical.n_vert_levels or vertical.zz.shape != pressure_input.shape:
        raise ValueError("model fields and VerticalGrid dimensions disagree")
    if fields.edge_normal_wind.shape != (n_levels, n_edges):
        raise ValueError(
            f"edge_normal_wind shape {fields.edge_normal_wind.shape} != {(n_levels, n_edges)}"
        )
    if np.any(pressure_input <= 0.0) or np.any(temperature_input <= 0.0):
        raise ValueError("pressure and absolute temperature must be positive")
    if np.any(qv < 0.0) or not np.all(np.isfinite(qv)):
        raise ValueError("water-vapor mixing ratio must be finite and non-negative")

    scalar_names_tuple = tuple(str(name) for name in scalar_names)
    if len(set(scalar_names_tuple)) != len(scalar_names_tuple):
        raise ValueError("scalar_names must be unique")
    if "qv" not in scalar_names_tuple:
        _refuse(
            "scalar_names",
            scalar_names_tuple,
            "the moist initialization requires an exact 'qv' scalar slot",
            "scalar_names=('qv', ...)",
        )

    zz = np.asarray(vertical.zz, dtype=np.float64)
    zmid = 0.5 * (np.asarray(vertical.zgrid[:-1]) + np.asarray(vertical.zgrid[1:]))
    theta = temperature_input * (REFERENCE_PRESSURE / pressure_input) ** (
        DRY_AIR_GAS_CONSTANT / DRY_AIR_CP
    )
    exner = (pressure_input / REFERENCE_PRESSURE) ** (
        DRY_AIR_GAS_CONSTANT / DRY_AIR_CP
    )
    rho_tilde = pressure_input / (
        DRY_AIR_GAS_CONSTANT
        * exner
        * theta
        * (1.0 + (WATER_VAPOR_GAS_CONSTANT / DRY_AIR_GAS_CONSTANT - 1.0) * qv)
    )
    rho_tilde = rho_tilde / (1.0 + qv) / zz

    pressure_base = REFERENCE_PRESSURE * np.exp(
        -GRAVITY * zmid / (DRY_AIR_GAS_CONSTANT * REFERENCE_TEMPERATURE)
    )
    rho_base = pressure_base / (
        DRY_AIR_GAS_CONSTANT * REFERENCE_TEMPERATURE * zz
    )
    pressure = pressure_input.copy()
    pressure_perturbation = pressure - pressure_base
    density_perturbation = rho_tilde - rho_base
    iterations = np.zeros((n_levels, n_cells), dtype=np.int64)

    # Bottom anchoring and the level-by-level fixed-point iteration are direct
    # translations of frozen lines 4895-4943.
    rho_tilde[0] = (
        (pressure[0] / REFERENCE_PRESSURE) ** (DRY_AIR_CV / DRY_AIR_CP)
        * (REFERENCE_PRESSURE / DRY_AIR_GAS_CONSTANT)
        / (theta[0] * (1.0 + MOIST_THETA_FACTOR * qv[0]))
        / zz[0]
    )
    density_perturbation[0] = rho_tilde[0] - rho_base[0]
    for cell in range(n_cells):
        for level in range(1, n_levels):
            residual = np.inf
            for iteration in range(1, hydrostatic_max_iterations + 1):
                previous = pressure_perturbation[level, cell]
                pressure_perturbation[level, cell] = pressure_perturbation[
                    level - 1, cell
                ] - (
                    vertical.fzm[level] * density_perturbation[level, cell]
                    + vertical.fzp[level]
                    * density_perturbation[level - 1, cell]
                ) * GRAVITY * vertical.dzu[level] - (
                    vertical.fzm[level] * rho_tilde[level, cell] * qv[level, cell]
                    + vertical.fzp[level]
                    * rho_tilde[level - 1, cell]
                    * qv[level - 1, cell]
                ) * GRAVITY * vertical.dzu[level]
                pressure[level, cell] = (
                    pressure_base[level, cell]
                    + pressure_perturbation[level, cell]
                )
                if not np.isfinite(pressure[level, cell]) or pressure[level, cell] <= 0.0:
                    raise FloatingPointError(
                        f"hydrostatic pressure became invalid at cell={cell}, level={level}"
                    )
                exner_level = (
                    pressure[level, cell] / REFERENCE_PRESSURE
                ) ** (DRY_AIR_GAS_CONSTANT / DRY_AIR_CP)
                rho_tilde[level, cell] = pressure[level, cell] / (
                    DRY_AIR_GAS_CONSTANT
                    * exner_level
                    * theta[level, cell]
                    * (1.0 + MOIST_THETA_FACTOR * qv[level, cell])
                    * zz[level, cell]
                )
                density_perturbation[level, cell] = (
                    rho_tilde[level, cell] - rho_base[level, cell]
                )
                residual = abs(previous - pressure_perturbation[level, cell])
                iterations[level, cell] = iteration
                if residual <= hydrostatic_tolerance:
                    break
            if residual > hydrostatic_tolerance:
                _refuse(
                    "hydrostatic_iteration",
                    f"cell={cell},level={level},residual={residual:.6g}",
                    "the frozen fixed-point iteration did not converge",
                    "a physically consistent first guess or a larger declared iteration bound",
                )

    modified_theta = theta * (1.0 + MOIST_THETA_FACTOR * qv)
    rho_theta = rho_tilde * modified_theta
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edge_mass = 0.5 * (rho_tilde[:, cells[:, 0]] + rho_tilde[:, cells[:, 1]])
    rho_u = np.asarray(fields.edge_normal_wind, dtype=np.float64) * edge_mass
    rho_w = _initial_vertical_momentum_order2(mesh, vertical, rho_u)

    scalars = np.zeros(
        (len(scalar_names_tuple), n_levels, n_cells), dtype=np.float64
    )
    scalars[scalar_names_tuple.index("qv")] = qv
    state = PrognosticState(
        rho=np.ascontiguousarray(rho_tilde),
        rho_theta=np.ascontiguousarray(rho_theta),
        rho_u=np.ascontiguousarray(rho_u),
        rho_w=np.ascontiguousarray(rho_w),
        scalars=np.ascontiguousarray(scalars),
    )
    state.validate(n_cells=n_cells, n_edges=n_edges, n_vert_levels=n_levels)

    adjusted_exner = (pressure / REFERENCE_PRESSURE) ** (
        DRY_AIR_GAS_CONSTANT / DRY_AIR_CP
    )
    adjusted_temperature = theta * adjusted_exner
    dry_density = rho_tilde * zz
    surface_pressure = (
        0.5
        * GRAVITY
        / vertical.rdzw[0]
        * (
            1.25 * rho_tilde[0] * (1.0 + qv[0])
            - 0.25 * rho_tilde[1] * (1.0 + qv[1])
        )
        + pressure[0]
    )
    return StateAssembly(
        state=state,
        diagnostics=StateAssemblyDiagnostics(
            pressure=pressure,
            temperature=adjusted_temperature,
            potential_temperature=theta,
            modified_potential_temperature=modified_theta,
            dry_density=dry_density,
            water_vapor_mixing_ratio=qv,
            surface_pressure=surface_pressure,
            hydrostatic_iterations=iterations,
        ),
    )


@dataclass(frozen=True, slots=True)
class InitializationResult:
    state: PrognosticState
    diagnostics: StateAssemblyDiagnostics
    surface: StaticSurfaceFields
    first_guess: RemappedFirstGuess
    model_fields: ModelLevelFields
    evidence: str = "implemented-unverified"


def initialize_from_structured(
    source: StructuredAtmosphere,
    mesh: object,
    vertical: VerticalGrid,
    *,
    humidity_mode: Literal["specific_humidity", "relative_humidity", "dry", "auto"] = "auto",
    longitude_mode: Literal["periodic_global", "bounded"] = "periodic_global",
    latitude_boundary: Literal["refuse", "clamp"] = "refuse",
    terrain_source: Literal["mesh", "structured"] = "mesh",
    land_source: Literal["mesh", "structured"] = "mesh",
    skin_temperature_source: Literal["structured", "mesh"] = "structured",
    land_fraction_threshold: float = 0.5,
    scalar_names: Sequence[str] = ("qv",),
    config_theta_adv_order: int = 2,
) -> InitializationResult:
    """Run the complete supported structured-data initialization pipeline."""

    first_guess = remap_to_mpas_cells(
        source,
        mesh,
        longitude_mode=longitude_mode,
        latitude_boundary=latitude_boundary,
    )
    surface = assemble_static_surface_fields(
        mesh,
        first_guess,
        terrain_source=terrain_source,
        land_source=land_source,
        skin_temperature_source=skin_temperature_source,
        land_fraction_threshold=land_fraction_threshold,
    )
    model_fields = interpolate_to_model_levels(
        source,
        first_guess,
        mesh,
        vertical,
        humidity_mode=humidity_mode,
        longitude_mode=longitude_mode,
        latitude_boundary=latitude_boundary,
    )
    assembly = assemble_prognostic_state(
        mesh,
        vertical,
        model_fields,
        scalar_names=scalar_names,
        config_theta_adv_order=config_theta_adv_order,
    )
    return InitializationResult(
        state=assembly.state,
        diagnostics=assembly.diagnostics,
        surface=surface,
        first_guess=first_guess,
        model_fields=model_fields,
    )


def load_structured_atmosphere(
    path: str | Path,
    **kwargs: Any,
) -> StructuredAtmosphere:
    """Functional spelling of :meth:`StructuredAtmosphere.from_file`."""

    return StructuredAtmosphere.from_file(path, **kwargs)


__all__ = [
    "DRY_AIR_CP",
    "DRY_AIR_CV",
    "DRY_AIR_GAS_CONSTANT",
    "FieldMap",
    "GRAVITY",
    "InitializationResult",
    "ModelLevelFields",
    "MOIST_THETA_FACTOR",
    "REFERENCE_PRESSURE",
    "RemappedFirstGuess",
    "SphericalRemap",
    "StateAssembly",
    "StateAssemblyDiagnostics",
    "StaticSurfaceFields",
    "StructuredAtmosphere",
    "assemble_prognostic_state",
    "assemble_static_surface_fields",
    "initialize_from_structured",
    "interpolate_to_model_levels",
    "load_structured_atmosphere",
    "pressure_from_height_log_interpolation",
    "remap_to_mpas_cells",
    "saturation_mixing_ratio_liquid",
    "vertical_interpolate_log_pressure",
]
