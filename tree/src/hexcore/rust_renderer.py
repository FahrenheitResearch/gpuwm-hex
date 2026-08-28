"""Truthful MPAS regular-grid input and execution seam for ``rw_wrfbatch``.

The Rust renderer is deliberately not taught about an MPAS Voronoi mesh.  This
module writes the small post-processed-WRF/2-D dialect its existing importer
already understands, then delegates discovery, probing, catalog inspection and
rendering to the public :mod:`gpuwm.rustwx` wrapper.

The adapter's WRF-looking names are intentionally narrow:

* ``TK``, ``P`` and ``Z`` are genuine lowest-model-level temperature, pressure
  and geometric height.  Their presence (with ``PB`` absent) selects the
  renderer's post-processed 2-D importer.
* ``PSFC`` and ``HGT`` are genuine surface pressure and terrain height.
* lowest-model-level winds retain explicit descriptive names.  This module
  never writes ``T2``, ``U10`` or ``V10`` because those names mean 2 m and 10 m
  diagnostics that the current dry forecast does not compute.

``rw_wrfbatch`` has no ``--abi`` command.  The executable contract used here is
the one gpuwm itself publishes: a successful ``--help`` usage probe followed by
a real input import and store-aware ``--list-products`` catalog.  Evidence also
binds the exact executable SHA-256; no ABI-version claim is made.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
from pathlib import Path
import re
import time
from typing import Any
from uuid import uuid4
import zlib

from netCDF4 import Dataset, num2date
import numpy as np
from numpy.typing import NDArray

from .errors import MeshValidationError
from .mesh import Mesh, normalize_longitudes, validate_longitude_normalization
from .regrid import (
    REGRID_EVIDENCE,
    REGRID_WEIGHT_SCHEMA,
    RegridWeights,
    build_regrid_weights,
    load_regrid_weights,
)
from .vector import initialize_reconstruction_coefficients, reconstruct_1d
from .vertical import build_vertical_grid


RENDERER_INPUT_SCHEMA = "mpas-port.rw-wrfbatch-postprocessed-2d.v1"
RENDERER_CONTRACT = "gpuwm.rustwx --help plus real-import store catalog"
RENDERER_ABI_CLAIMED = False
INTEGRITY_SCOPE = (
    "pre/post SHA-256 observations around a trusted unchanged renderer; no claim of "
    "protection from a malicious cooperating process that mutates and restores bytes"
)
FROZEN_X1_2562_REGRID_WEIGHTS_SHA256 = (
    "6a0b38312afbac8f7eb97ce8c7d5e9ac14a7a9ecec862d44213addcdb0455d8a"
)

_TIME_DIMENSION = "Time"
_Y_DIMENSION = "south_north"
_X_DIMENSION = "west_east"
_MISLEADING_OR_ROUTE_BREAKING_VARIABLES = frozenset(
    {"PB", "T2", "U10", "V10", "SLP", "Times", "XTIME", "xtime"}
)
_REQUIRED_RENDERER_VARIABLES = frozenset(
    {
        "time",
        "XLAT",
        "XLONG",
        "TK",
        "P",
        "Z",
        "PSFC",
        "HGT",
        "surface_pressure",
        "temperature_lowest_model_level",
        "u_lowest_model_level",
        "v_lowest_model_level",
        "wind_speed_lowest_model_level",
    }
)
_PRODUCT_SLUG = re.compile(r"^[A-Za-z0-9_.:-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIME_UNITS = re.compile(
    r"^seconds since [0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$"
)
_CATALOG_KINDS = frozenset({"direct", "derived", "generic", "heavy", "windowed"})
_CATALOG_STATUSES = frozenset({"renderable", "missing-fields", "blocked", "excluded"})
_HEIGHT_TERRAIN_RELATION = (
    "Z uses the one-pass-smoothed MPAS terrain coordinate; HGT preserves "
    "the unsmoothed static-file terrain, so Z-HGT is not constrained positive"
)
_WIND_REGRID_SEMANTICS = (
    "the native-cell scalar wind speed is regridded independently of the "
    "native-cell u/v components"
)
_REGRID_ALGORITHM = "spherical inverse-distance, k=4, power=2"
_REGRID_WEIGHTS_SCHEMA = "mpas-port.saved-regrid-weights.v2"
_REGRID_WEIGHTS_BINDING = (
    "checksum-bound saved RegridWeights and reconstruction geometry; runtime "
    "neighbor search and platform libm basis generation are not used"
)
_RENDERER_MATERIALIZATION_SCHEMA = "mpas-port.renderer-materialization-authority.v1"
_RECONSTRUCTION_SEMANTICS = (
    "frozen MPAS RBF coefficients and local sin/cos basis; edge-history values "
    "are reconstructed live in source order"
)
_MATERIALIZATION_MESH_PAIR_SCHEMA = "mpas-port.materialization-mesh-pair/v1"
_MESH_TOPOLOGY_FIELDS = (
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
_MESH_COORDINATE_FIELDS = tuple(
    f"{component}{entity}"
    for entity in ("Cell", "Edge", "Vertex")
    for component in ("x", "y", "z", "lat", "lon")
)


class RustRendererError(RuntimeError):
    """A fail-closed renderer input, contract, catalog or execution refusal."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one existing regular file."""

    target = Path(path).expanduser().resolve(strict=True)
    if not target.is_file():
        raise RustRendererError(f"SHA-256 target is not a regular file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_semantic_fingerprint(mesh: Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(b"mpas-port.materialization-mesh-fingerprint/v1\0")
    for name in ("nCells", "nEdges", "nVertices", "maxEdges", "maxEdges2"):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(int(mesh.dimensions[name]).to_bytes(8, "little", signed=False))
    digest.update(np.asarray(float(mesh.attrs["sphere_radius"]), dtype="<f8").tobytes())
    for name in (*_MESH_TOPOLOGY_FIELDS, *_MESH_COORDINATE_FIELDS):
        array = np.ascontiguousarray(np.asarray(mesh.arrays[name]))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_materialization_mesh_pair(
    dynamics_mesh: Mesh,
    output_mesh: Mesh,
    *,
    grid_path: str | Path,
    static_path: str | Path,
) -> dict[str, Any]:
    """Bind a certified in-memory mesh pair to exact official source files."""

    if not isinstance(dynamics_mesh, Mesh) or not isinstance(output_mesh, Mesh):
        raise RustRendererError("materialization meshes must be Mesh instances")
    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    expected_paths = (
        (dynamics_mesh.grid_path, grid, "dynamics grid"),
        (dynamics_mesh.static_path, static, "dynamics static"),
        (output_mesh.grid_path, grid, "output grid"),
    )
    for observed, expected, label in expected_paths:
        if observed is None or Path(observed).resolve(strict=True) != expected:
            raise RustRendererError(f"materialization {label} path is not source-bound")
    if output_mesh.static_path is not None:
        raise RustRendererError("materialization output mesh must be grid-only")
    try:
        dynamics_mesh.validate()
        output_mesh.validate()
    except (KeyError, MeshValidationError, ValueError) as error:
        raise RustRendererError(f"materialization mesh pair is invalid: {error}") from error

    core_dimensions = ("nCells", "nEdges", "nVertices", "maxEdges", "maxEdges2")
    for name in core_dimensions:
        if dynamics_mesh.dimensions.get(name) != output_mesh.dimensions.get(name):
            raise RustRendererError(f"materialization mesh dimension differs at {name}")
    topology_digest = hashlib.sha256()
    for name in _MESH_TOPOLOGY_FIELDS:
        if name not in dynamics_mesh.arrays or name not in output_mesh.arrays:
            raise RustRendererError(f"materialization topology witness is absent: {name}")
        dynamics_value = np.asarray(dynamics_mesh.arrays[name])
        output_value = np.asarray(output_mesh.arrays[name])
        if not np.array_equal(dynamics_value, output_value):
            raise RustRendererError(f"materialization mesh topology differs at {name}")
        topology_digest.update(name.encode("ascii") + b"\0")
        topology_digest.update(np.ascontiguousarray(output_value).tobytes(order="C"))

    grid_radius = float(output_mesh.attrs.get("sphere_radius", np.nan))
    static_radius = float(dynamics_mesh.attrs.get("sphere_radius", np.nan))
    scale = static_radius / grid_radius
    if not np.isfinite(scale) or scale <= 0.0:
        raise RustRendererError("materialization mesh pair has invalid sphere radii")
    longitude_evidence: dict[str, Any] = {}
    with Dataset(grid) as dataset:
        dataset.set_auto_mask(False)
        for entity in ("Cell", "Edge", "Vertex"):
            for component in ("x", "y", "z"):
                name = f"{component}{entity}"
                source = np.asarray(dataset.variables[name][:])
                if not np.array_equal(np.asarray(output_mesh.arrays[name]), source):
                    raise RustRendererError(
                        f"materialization output coordinate is not exact grid {name}"
                    )
                expected_dynamics = np.ascontiguousarray(source * scale)
                if not np.array_equal(
                    np.asarray(dynamics_mesh.arrays[name]), expected_dynamics
                ):
                    raise RustRendererError(
                        f"materialization dynamics coordinate is not scaled grid {name}"
                    )
            latitude_name = f"lat{entity}"
            longitude_name = f"lon{entity}"
            source_latitude = np.asarray(dataset.variables[latitude_name][:])
            source_longitude = np.asarray(dataset.variables[longitude_name][:])
            expected_longitude = normalize_longitudes(source_longitude)
            if not np.array_equal(output_mesh.arrays[latitude_name], source_latitude):
                raise RustRendererError(
                    f"materialization output latitude changed at {latitude_name}"
                )
            if not np.array_equal(dynamics_mesh.arrays[latitude_name], source_latitude):
                raise RustRendererError(
                    f"materialization dynamics latitude changed at {latitude_name}"
                )
            if not np.array_equal(output_mesh.arrays[longitude_name], expected_longitude):
                raise RustRendererError(
                    f"materialization output longitude is not normalized at {longitude_name}"
                )
            if not np.array_equal(
                dynamics_mesh.arrays[longitude_name], expected_longitude
            ):
                raise RustRendererError(
                    f"materialization dynamics longitude is not normalized at {longitude_name}"
                )
            longitude_evidence[longitude_name] = validate_longitude_normalization(
                source_longitude, expected_longitude
            )

    return {
        "schema": _MATERIALIZATION_MESH_PAIR_SCHEMA,
        "grid_sha256": sha256_file(grid),
        "static_sha256": sha256_file(static),
        "dynamics_mesh_sha256": _mesh_semantic_fingerprint(dynamics_mesh),
        "output_mesh_sha256": _mesh_semantic_fingerprint(output_mesh),
        "topology_sha256": topology_digest.hexdigest(),
        "topology_grid_static_bit_identical": True,
        "longitude_normalization": longitude_evidence,
        "cartesian_scale": scale,
    }


def write_renderer_materialization_authority(
    path: str | Path,
    *,
    weights: RegridWeights,
    dynamics_mesh: Mesh,
    output_mesh: Mesh,
    grid_path: str | Path,
    static_path: str | Path,
    overwrite: bool = False,
) -> Path:
    """Freeze regrid weights and mesh-derived reconstruction geometry once."""

    binding = validate_materialization_mesh_pair(
        dynamics_mesh,
        output_mesh,
        grid_path=grid_path,
        static_path=static_path,
    )
    try:
        weights.validate_source(output_mesh)
    except ValueError as error:
        raise RustRendererError(
            f"renderer materialization weights bind a different output mesh: {error}"
        ) from error
    if (
        weights.method != "inverse_distance"
        or weights.n_neighbors != 4
        or weights.power != 2.0
        or weights.evidence != REGRID_EVIDENCE
    ):
        raise RustRendererError(
            "renderer materialization authority requires inverse-distance k=4 power=2"
        )
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    if destination in {grid, static}:
        raise RustRendererError("renderer authority output aliases a mesh source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    coefficients = initialize_reconstruction_coefficients(dynamics_mesh)
    payload = {
        "schema_version": np.asarray(REGRID_WEIGHT_SCHEMA, dtype=np.int64),
        "target_latitude": weights.target_latitude,
        "target_longitude": weights.target_longitude,
        "source_indices": weights.source_indices,
        "weights": weights.weights,
        "source_count": np.asarray(weights.source_count, dtype=np.int64),
        "source_fingerprint": np.asarray(weights.source_fingerprint),
        "method": np.asarray(weights.method),
        "power": np.asarray(weights.power, dtype=np.float64),
        "evidence": np.asarray(weights.evidence),
        "renderer_materialization_schema": np.asarray(
            _RENDERER_MATERIALIZATION_SCHEMA
        ),
        "renderer_grid_sha256": np.asarray(binding["grid_sha256"]),
        "renderer_static_sha256": np.asarray(binding["static_sha256"]),
        "renderer_reconstruction_semantics": np.asarray(
            _RECONSTRUCTION_SEMANTICS
        ),
        "reconstruction_coefficients": coefficients,
        "sin_lat": np.sin(np.asarray(dynamics_mesh.latCell, dtype=np.float64)),
        "cos_lat": np.cos(np.asarray(dynamics_mesh.latCell, dtype=np.float64)),
        "sin_lon": np.sin(np.asarray(dynamics_mesh.lonCell, dtype=np.float64)),
        "cos_lon": np.cos(np.asarray(dynamics_mesh.lonCell, dtype=np.float64)),
        "renderer_dynamics_mesh_sha256": np.asarray(
            binding["dynamics_mesh_sha256"]
        ),
        "renderer_output_mesh_sha256": np.asarray(binding["output_mesh_sha256"]),
        "renderer_topology_sha256": np.asarray(binding["topology_sha256"]),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(destination)
    return destination


def _load_renderer_reconstruction_authority(
    path: Path,
    *,
    grid: Path,
    static: Path,
    mesh: Mesh,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Load exact mesh-derived reconstruction geometry from the weight artifact."""

    required = {
        "renderer_materialization_schema",
        "renderer_grid_sha256",
        "renderer_static_sha256",
        "renderer_reconstruction_semantics",
        "reconstruction_coefficients",
        "sin_lat",
        "cos_lat",
        "sin_lon",
        "cos_lon",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise RustRendererError(
                    "saved renderer authority is missing: " + ", ".join(sorted(missing))
                )
            schema = str(np.asarray(archive["renderer_materialization_schema"]).item())
            grid_sha256 = str(np.asarray(archive["renderer_grid_sha256"]).item())
            static_sha256 = str(np.asarray(archive["renderer_static_sha256"]).item())
            semantics = str(
                np.asarray(archive["renderer_reconstruction_semantics"]).item()
            )
            coefficients = np.asarray(archive["reconstruction_coefficients"]).copy()
            sin_lat = np.asarray(archive["sin_lat"], dtype=np.float64).copy()
            cos_lat = np.asarray(archive["cos_lat"], dtype=np.float64).copy()
            sin_lon = np.asarray(archive["sin_lon"], dtype=np.float64).copy()
            cos_lon = np.asarray(archive["cos_lon"], dtype=np.float64).copy()
    except (OSError, ValueError) as error:
        raise RustRendererError(
            f"saved renderer reconstruction authority cannot be loaded: {error}"
        ) from error

    if schema != _RENDERER_MATERIALIZATION_SCHEMA:
        raise RustRendererError("saved renderer materialization schema is unsupported")
    if semantics != _RECONSTRUCTION_SEMANTICS:
        raise RustRendererError("saved renderer reconstruction semantics are stale")
    if grid_sha256 != sha256_file(grid) or static_sha256 != sha256_file(static):
        raise RustRendererError(
            "saved renderer reconstruction authority binds a different grid/static pair"
        )
    n_cells = int(mesh.dimensions["nCells"])
    basis = (sin_lat, cos_lat, sin_lon, cos_lon)
    if any(item.shape != (n_cells,) for item in basis):
        raise RustRendererError("saved renderer local basis shape is invalid")
    if (
        coefficients.ndim != 3
        or coefficients.shape[0] != n_cells
        or coefficients.shape[2] != 3
    ):
        raise RustRendererError(
            "saved renderer reconstruction coefficient shape is invalid"
        )
    if not np.all(np.isfinite(coefficients)) or not np.any(coefficients):
        raise RustRendererError(
            "saved renderer reconstruction coefficients are invalid"
        )
    if any(not np.all(np.isfinite(item)) for item in basis):
        raise RustRendererError("saved renderer local basis is non-finite")
    tolerance = 8.0 * np.finfo(np.float64).eps
    if not np.allclose(
        sin_lat * sin_lat + cos_lat * cos_lat, 1.0, rtol=0.0, atol=tolerance
    ):
        raise RustRendererError("saved renderer latitude basis is not normalized")
    if not np.allclose(
        sin_lon * sin_lon + cos_lon * cos_lon, 1.0, rtol=0.0, atol=tolerance
    ):
        raise RustRendererError("saved renderer longitude basis is not normalized")
    return coefficients, sin_lat, cos_lat, sin_lon, cos_lon


def _utc_naive(value: datetime | str | np.datetime64) -> datetime:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("valid time cannot be NaT")
        microseconds = value.astype("datetime64[us]").astype(np.int64)
        return datetime.fromtimestamp(
            float(microseconds) / 1.0e6, tz=timezone.utc
        ).replace(tzinfo=None)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    text = str(value).strip().replace("_", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _plain_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bool):
        return int(value)
    return value


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> NDArray[Any]:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must use a floating dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True, slots=True)
class RustWrf2dFields:
    """The honest field set used to materialize one renderer input file."""

    latitude: NDArray[Any] | Sequence[float]
    longitude: NDArray[Any] | Sequence[float]
    valid_times: Sequence[datetime | str | np.datetime64]
    temperature_lowest_model_level: NDArray[Any]
    pressure_lowest_model_level: NDArray[Any]
    height_lowest_model_level: NDArray[Any]
    surface_pressure: NDArray[Any]
    terrain_height: NDArray[Any]
    u_lowest_model_level: NDArray[Any]
    v_lowest_model_level: NDArray[Any]
    wind_speed_lowest_model_level: NDArray[Any]
    initial_time: datetime | str | np.datetime64 | None = None
    global_attrs: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> "RustWrf2dFields":
        latitude = np.asarray(self.latitude, dtype=np.float64)
        longitude = np.asarray(self.longitude, dtype=np.float64)
        if latitude.ndim != 1 or longitude.ndim != 1:
            raise ValueError("latitude and longitude must be one-dimensional axes")
        if latitude.size < 2 or longitude.size < 2:
            raise ValueError("renderer axes must each contain at least two points")
        if not np.all(np.isfinite(latitude)) or np.any(np.abs(latitude) > 90.0):
            raise ValueError("latitude must be finite and lie in [-90, 90] degrees")
        if not np.all(np.isfinite(longitude)):
            raise ValueError("longitude must be finite")
        if np.any(np.diff(latitude) <= 0.0) or np.any(np.diff(longitude) <= 0.0):
            raise ValueError("latitude and longitude axes must be strictly increasing")
        times = tuple(_utc_naive(value) for value in self.valid_times)
        if not times:
            raise ValueError("at least one valid time is required")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("valid times must be strictly increasing")
        if any(value.microsecond != 0 for value in times):
            raise ValueError("renderer valid times must fall on whole seconds")
        shape = (len(times), latitude.size, longitude.size)
        temperature = _finite_array(
            "temperature_lowest_model_level",
            self.temperature_lowest_model_level,
            shape,
        )
        pressure = _finite_array(
            "pressure_lowest_model_level", self.pressure_lowest_model_level, shape
        )
        surface_pressure = _finite_array(
            "surface_pressure", self.surface_pressure, shape
        )
        u_wind = _finite_array("u_lowest_model_level", self.u_lowest_model_level, shape)
        v_wind = _finite_array("v_lowest_model_level", self.v_lowest_model_level, shape)
        wind_speed = _finite_array(
            "wind_speed_lowest_model_level",
            self.wind_speed_lowest_model_level,
            shape,
        )
        horizontal_shape = (latitude.size, longitude.size)
        _finite_array(
            "height_lowest_model_level",
            self.height_lowest_model_level,
            horizontal_shape,
        )
        _finite_array("terrain_height", self.terrain_height, horizontal_shape)
        if np.any(temperature <= 0.0):
            raise ValueError("lowest-model-level temperature must be positive")
        if np.any(pressure <= 0.0) or np.any(surface_pressure <= 0.0):
            raise ValueError("pressure fields must be positive")
        if np.any(pressure > surface_pressure):
            raise ValueError("lowest-model pressure cannot exceed surface pressure")
        if np.any(wind_speed < 0.0):
            raise ValueError("lowest-model-level wind speed cannot be negative")
        component_speed = np.hypot(u_wind, v_wind)
        tolerance = 1.0e-5 * np.maximum(1.0, component_speed)
        if np.any(wind_speed + tolerance < component_speed):
            raise ValueError(
                "regridded scalar wind speed violates the convex component-speed bound"
            )
        initial = (
            times[0] if self.initial_time is None else _utc_naive(self.initial_time)
        )
        if initial > times[0]:
            raise ValueError("initial_time cannot follow the first valid time")
        if initial.microsecond != 0:
            raise ValueError("renderer initial_time must fall on a whole second")
        if any(not (value - initial).total_seconds().is_integer() for value in times):
            raise ValueError("renderer time offsets must be exact whole seconds")
        reserved = {
            "Conventions",
            "MAP_PROJ",
            "POLE_LAT",
            "POLE_LON",
            "grid_type",
            "renderer_adapter_schema",
            "renderer_abi_claimed",
            "renderer_contract",
            "source",
            "title",
        }
        overlap = reserved.intersection(self.global_attrs)
        if overlap:
            raise ValueError(
                "global_attrs cannot override renderer contract attributes: "
                + ", ".join(sorted(overlap))
            )
        return self


def _write_plane(
    dataset: Dataset,
    name: str,
    values: NDArray[Any],
    dimensions: tuple[str, ...],
    *,
    units: str,
    long_name: str,
) -> None:
    array = np.asarray(values)
    dtype = "f4" if array.dtype.itemsize <= 4 else "f8"
    variable = dataset.createVariable(name, dtype, dimensions)
    variable.setncatts(
        {
            "units": units,
            "long_name": long_name,
            "coordinates": "XLAT XLONG",
        }
    )
    variable[:] = array.astype(dtype, copy=False)


def write_rust_wrf2d_netcdf(
    path: str | Path,
    fields: RustWrf2dFields,
    *,
    clobber: bool = False,
) -> Path:
    """Write a renderer-importable post-processed-WRF/2-D NetCDF file."""

    fields.validate()
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not clobber:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    latitude = np.asarray(fields.latitude, dtype=np.float64)
    longitude = np.asarray(fields.longitude, dtype=np.float64)
    times = tuple(_utc_naive(value) for value in fields.valid_times)
    initial = (
        times[0] if fields.initial_time is None else _utc_naive(fields.initial_time)
    )
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    timed_dimensions = (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION)
    static_dimensions = (_Y_DIMENSION, _X_DIMENSION)

    try:
        # NETCDF4_CLASSIC plus a fixed Time dimension and a lowercase numeric
        # coordinate is material renderer compatibility, not cosmetic style.
        # The deployed netcrust reader does not discover the generic CF axis
        # when the coordinate is uppercase/unlimited, and decodes NETCDF4 S1
        # ``Times`` as String before its numeric reader rejects it.
        with Dataset(temporary, "w", format="NETCDF4_CLASSIC") as dataset:
            dataset.createDimension(_TIME_DIMENSION, len(times))
            dataset.createDimension(_Y_DIMENSION, latitude.size)
            dataset.createDimension(_X_DIMENSION, longitude.size)
            dataset.setncatts(
                {
                    "Conventions": "CF-1.10",
                    "title": "MPAS forecast fields regridded for unchanged rw_wrfbatch",
                    "source": "hexcore.rust_renderer",
                    "renderer_adapter_schema": RENDERER_INPUT_SCHEMA,
                    "renderer_contract": RENDERER_CONTRACT,
                    "renderer_abi_claimed": int(RENDERER_ABI_CLAIMED),
                    "MAP_PROJ": 6,
                    "POLE_LAT": 90.0,
                    "POLE_LON": 0.0,
                    "grid_type": "regular_ll",
                }
            )
            if fields.global_attrs:
                dataset.setncatts(
                    {
                        key: _plain_attr(value)
                        for key, value in fields.global_attrs.items()
                    }
                )

            xlat = dataset.createVariable("XLAT", "f8", static_dimensions)
            xlong = dataset.createVariable("XLONG", "f8", static_dimensions)
            xlat.setncatts(
                {
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "units": "degrees_north",
                }
            )
            xlong.setncatts(
                {
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "units": "degrees_east",
                }
            )
            xlat[:] = lat_grid
            xlong[:] = lon_grid

            time_variable = dataset.createVariable("time", "f8", (_TIME_DIMENSION,))
            time_variable.setncatts(
                {
                    "standard_name": "time",
                    "long_name": "valid time",
                    "axis": "T",
                    "calendar": "gregorian",
                    "units": f"seconds since {initial:%Y-%m-%d %H:%M:%S}",
                }
            )
            time_variable[:] = [(value - initial).total_seconds() for value in times]
            temperature = np.asarray(fields.temperature_lowest_model_level)
            pressure = np.asarray(fields.pressure_lowest_model_level)
            height = np.asarray(fields.height_lowest_model_level)
            surface_pressure = np.asarray(fields.surface_pressure)
            terrain = np.asarray(fields.terrain_height)
            u_wind = np.asarray(fields.u_lowest_model_level)
            v_wind = np.asarray(fields.v_lowest_model_level)
            wind_speed = np.asarray(fields.wind_speed_lowest_model_level)

            # Exact postprocessed-WRF/2-D gate and truthful canonical surface
            # fields.  PB is intentionally never created.
            _write_plane(
                dataset,
                "TK",
                temperature,
                timed_dimensions,
                units="K",
                long_name="temperature at lowest model level",
            )
            _write_plane(
                dataset,
                "P",
                pressure,
                timed_dimensions,
                units="Pa",
                long_name="pressure at lowest model level",
            )
            _write_plane(
                dataset,
                "Z",
                height,
                static_dimensions,
                units="m",
                long_name="geometric height of lowest model level",
            )
            _write_plane(
                dataset,
                "PSFC",
                surface_pressure,
                timed_dimensions,
                units="Pa",
                long_name="surface pressure",
            )
            _write_plane(
                dataset,
                "HGT",
                terrain,
                static_dimensions,
                units="m",
                long_name="terrain height above mean sea level",
            )

            # Raw generic variables preserve the actual vertical meaning.  In
            # particular, none borrow the WRF T2/U10/V10 names.
            _write_plane(
                dataset,
                "surface_pressure",
                surface_pressure,
                timed_dimensions,
                units="Pa",
                long_name="surface pressure",
            )
            _write_plane(
                dataset,
                "temperature_lowest_model_level",
                temperature,
                timed_dimensions,
                units="K",
                long_name="temperature at lowest model level",
            )
            _write_plane(
                dataset,
                "u_lowest_model_level",
                u_wind,
                timed_dimensions,
                units="m s-1",
                long_name="earth-relative zonal wind at lowest model level",
            )
            _write_plane(
                dataset,
                "v_lowest_model_level",
                v_wind,
                timed_dimensions,
                units="m s-1",
                long_name="earth-relative meridional wind at lowest model level",
            )
            _write_plane(
                dataset,
                "wind_speed_lowest_model_level",
                wind_speed,
                timed_dimensions,
                units="m s-1",
                long_name="horizontal wind speed at lowest model level",
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _read_times(dataset: Dataset, context: str) -> tuple[datetime, ...]:
    coordinate_name = next(
        (name for name in ("time", "Time") if name in dataset.variables), None
    )
    if coordinate_name is None:
        raise RustRendererError(f"{context} has no CF Time coordinate")
    variable = dataset.variables[coordinate_name]
    if variable.dimensions != ("Time",):
        raise RustRendererError(f"{context} Time coordinate is not one-dimensional")
    units = getattr(variable, "units", None)
    if not units:
        raise RustRendererError(f"{context} Time coordinate has no units")
    calendar = getattr(variable, "calendar", "standard")
    decoded = num2date(
        np.asarray(variable[:], dtype=np.float64),
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return tuple(
        datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
        )
        for value in np.atleast_1d(decoded)
    )


def _read_exact_field(
    dataset: Dataset,
    name: str,
    dimensions: tuple[str, ...],
    context: str,
) -> NDArray[Any]:
    if name not in dataset.variables:
        raise RustRendererError(f"{context} is missing {name}")
    variable = dataset.variables[name]
    if variable.dimensions != dimensions:
        raise RustRendererError(
            f"{context} {name} dimensions {variable.dimensions} != {dimensions}"
        )
    variable.set_auto_mask(False)
    values = np.asarray(variable[:])
    if values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise RustRendererError(f"{context} {name} is not a finite floating field")
    return values


def materialize_gfs_rust_input(
    output_path: str | Path,
    *,
    history_path: str | Path,
    latlon_path: str | Path,
    grid_path: str | Path,
    static_path: str | Path,
    dynamics_mesh: Mesh | None = None,
    output_mesh: Mesh | None = None,
    regrid_weights_path: str | Path | None = None,
    expected_regrid_weights_sha256: str | None = None,
    ztop_m: float = 30_000.0,
    clobber: bool = False,
) -> Path:
    """Build the renderer adapter from committed forecast and mesh artifacts.

    No forecast rerun is needed.  The already-written history supplies exact
    pressure records, the existing CF product supplies the already-audited
    descriptive fields and target axes, and the frozen grid/static pair
    reconstructs the same terrain-following lowest mass-level height.  A
    precision-preserving, source-bound mesh pair may be supplied for official
    meshes whose static-file coordinate overlay is lower precision than the
    grid geometry.  The two meshes are an all-or-nothing injection and are
    revalidated against the exact grid/static bytes before use.
    """

    history = Path(history_path).expanduser().resolve(strict=True)
    latlon = Path(latlon_path).expanduser().resolve(strict=True)
    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    if (dynamics_mesh is None) != (output_mesh is None):
        raise RustRendererError(
            "dynamics_mesh and output_mesh must be supplied together"
        )
    if (regrid_weights_path is None) != (expected_regrid_weights_sha256 is None):
        raise RustRendererError(
            "saved regrid weights require both a path and expected SHA-256"
        )
    saved_weights = (
        Path(regrid_weights_path).expanduser().resolve(strict=True)
        if regrid_weights_path is not None
        else None
    )
    if saved_weights is not None:
        assert expected_regrid_weights_sha256 is not None
        if _SHA256.fullmatch(expected_regrid_weights_sha256) is None:
            raise RustRendererError(
                "expected saved regrid-weight SHA-256 is not a lowercase digest"
            )
        actual_weights_sha256 = sha256_file(saved_weights)
        if actual_weights_sha256 != expected_regrid_weights_sha256:
            raise RustRendererError(
                "saved regrid-weight SHA-256 does not match the declared authority"
            )
    else:
        actual_weights_sha256 = None
    destination = Path(output_path).expanduser().resolve()
    materialization_inputs = {history, latlon, grid, static}
    if saved_weights is not None:
        materialization_inputs.add(saved_weights)
    if destination in materialization_inputs:
        raise RustRendererError(
            "renderer adapter output must be distinct from every materialization source"
        )

    with Dataset(latlon) as dataset:
        dataset.set_auto_mask(False)
        latlon_times = _read_times(dataset, "lat-lon product")
        latitude = _read_exact_field(dataset, "lat", ("lat",), "lat-lon product")
        longitude = _read_exact_field(dataset, "lon", ("lon",), "lat-lon product")
        product_dimensions = ("Time", "lat", "lon")
        surface_pressure = _read_exact_field(
            dataset, "surface_pressure", product_dimensions, "lat-lon product"
        )
        temperature = _read_exact_field(
            dataset,
            "temperature_lowest_model_level",
            product_dimensions,
            "lat-lon product",
        )
        u_wind = _read_exact_field(
            dataset, "u_lowest_model_level", product_dimensions, "lat-lon product"
        )
        v_wind = _read_exact_field(
            dataset, "v_lowest_model_level", product_dimensions, "lat-lon product"
        )
        wind_speed = _read_exact_field(
            dataset,
            "wind_speed_lowest_model_level",
            product_dimensions,
            "lat-lon product",
        )
        source_regrid_evidence = str(getattr(dataset, "regrid_evidence", ""))
        if source_regrid_evidence != REGRID_EVIDENCE:
            raise RustRendererError(
                "lat-lon product regrid_evidence is missing, forged, or stale"
            )

    with Dataset(history) as dataset:
        dataset.set_auto_mask(False)
        history_times = _read_times(dataset, "history")
        if history_times != latlon_times:
            raise RustRendererError("history and lat-lon valid times disagree")
        history_surface_pressure = _read_exact_field(
            dataset, "surface_pressure", ("Time", "nCells"), "history"
        )
        history_temperature = _read_exact_field(
            dataset,
            "temperature_lowest_model_level",
            ("Time", "nCells"),
            "history",
        )
        history_wind_speed = _read_exact_field(
            dataset,
            "wind_speed_lowest_model_level",
            ("Time", "nCells"),
            "history",
        )
        history_normal_velocity = _read_exact_field(
            dataset,
            "u",
            ("Time", "nEdges", "nVertLevels"),
            "history",
        )
        pressure = _read_exact_field(
            dataset,
            "pressure",
            ("Time", "nCells", "nVertLevels"),
            "history",
        )

    mesh_pair_binding: dict[str, Any] | None = None
    if dynamics_mesh is None:
        dynamics_mesh = Mesh.from_netcdf(grid, static)
        output_mesh = Mesh.from_netcdf(grid)
    else:
        assert output_mesh is not None
        mesh_pair_binding = validate_materialization_mesh_pair(
            dynamics_mesh,
            output_mesh,
            grid_path=grid,
            static_path=static,
        )
    assert output_mesh is not None
    if not np.array_equal(dynamics_mesh.indexToCellID, output_mesh.indexToCellID):
        raise RustRendererError("grid/static and grid-only cell identities disagree")
    n_cells = int(output_mesh.dimensions["nCells"])
    if history_surface_pressure.shape[1] != n_cells or pressure.shape[1] != n_cells:
        raise RustRendererError("history nCells does not match the frozen mesh")
    vertical = build_vertical_grid(
        dynamics_mesh,
        np.asarray(dynamics_mesh.ter, dtype=np.float64),
        n_vert_levels=pressure.shape[2],
        ztop=float(ztop_m),
        smooth_surfaces=False,
    )
    lowest_height = 0.5 * (
        np.asarray(vertical.zgrid[0], dtype=np.float64)
        + np.asarray(vertical.zgrid[1], dtype=np.float64)
    )
    if saved_weights is None:
        frozen_reconstruction = None
        weights = build_regrid_weights(
            output_mesh,
            target_latitude=latitude,
            target_longitude=longitude,
            method="inverse_distance",
            neighbors=4,
            power=2.0,
        )
    else:
        try:
            weights = load_regrid_weights(saved_weights)
            weights.validate_source(output_mesh)
        except (OSError, ValueError) as error:
            raise RustRendererError(
                f"saved regrid weights are invalid or bind a different mesh: {error}"
            ) from error
        if not np.array_equal(
            weights.target_latitude, np.asarray(latitude, dtype=np.float64)
        ) or not np.array_equal(
            weights.target_longitude, np.asarray(longitude, dtype=np.float64)
        ):
            raise RustRendererError(
                "saved regrid-weight target axes do not exactly match the lat-lon product"
            )
        if (
            weights.method != "inverse_distance"
            or weights.n_neighbors != 4
            or weights.power != 2.0
            or weights.evidence != REGRID_EVIDENCE
        ):
            raise RustRendererError(
                "saved regrid-weight algorithm/version metadata is unsupported or stale"
            )
        frozen_reconstruction = _load_renderer_reconstruction_authority(
            saved_weights,
            grid=grid,
            static=static,
            mesh=dynamics_mesh,
        )
    regridded_surface = weights.apply(history_surface_pressure, cell_axis=1)
    regridded_temperature = weights.apply(history_temperature, cell_axis=1)
    reconstruction_coefficients = (
        initialize_reconstruction_coefficients(dynamics_mesh)
        if frozen_reconstruction is None
        else frozen_reconstruction[0]
    )
    reconstructed = tuple(
        reconstruct_1d(
            dynamics_mesh,
            history_normal_velocity[record, :, 0],
            include_halos=True,
            coefficients=reconstruction_coefficients,
        )
        for record in range(history_normal_velocity.shape[0])
    )
    if frozen_reconstruction is None:
        history_u_wind = np.stack([item.zonal for item in reconstructed])
        history_v_wind = np.stack([item.meridional for item in reconstructed])
    else:
        _, sin_lat, cos_lat, sin_lon, cos_lon = frozen_reconstruction
        history_x = np.stack([item.x for item in reconstructed])
        history_y = np.stack([item.y for item in reconstructed])
        history_z = np.stack([item.z for item in reconstructed])
        history_u_wind = -history_x * sin_lon + history_y * cos_lon
        history_v_wind = (
            -(history_x * cos_lon + history_y * sin_lon) * sin_lat + history_z * cos_lat
        )
    regridded_u_wind = weights.apply(history_u_wind, cell_axis=1)
    regridded_v_wind = weights.apply(history_v_wind, cell_axis=1)
    regridded_wind_speed = weights.apply(history_wind_speed, cell_axis=1)
    if not np.array_equal(regridded_surface, surface_pressure):
        raise RustRendererError(
            "committed surface pressure is not the exact regrid of committed history"
        )
    if not np.array_equal(regridded_temperature, temperature):
        raise RustRendererError(
            "committed lowest-level temperature is not the exact regrid of history"
        )
    if not np.array_equal(regridded_u_wind, u_wind):
        raise RustRendererError(
            "committed lowest-level zonal wind is not the exact reconstructed regrid of history"
        )
    if not np.array_equal(regridded_v_wind, v_wind):
        raise RustRendererError(
            "committed lowest-level meridional wind is not the exact reconstructed regrid of history"
        )
    if not np.array_equal(regridded_wind_speed, wind_speed):
        raise RustRendererError(
            "committed lowest-level wind speed is not the exact regrid of history"
        )
    pressure_lowest = weights.apply(pressure[:, :, 0], cell_axis=1)
    height_lowest = weights.apply(lowest_height)
    terrain_height = weights.apply(np.asarray(dynamics_mesh.ter, dtype=np.float64))
    provenance = {
        "source_history_name": history.name,
        "source_history_sha256": sha256_file(history),
        "source_latlon_name": latlon.name,
        "source_latlon_sha256": sha256_file(latlon),
        "source_grid_name": grid.name,
        "source_grid_sha256": sha256_file(grid),
        "source_static_name": static.name,
        "source_static_sha256": sha256_file(static),
        "vertical_levels": int(pressure.shape[2]),
        "vertical_top_m": float(ztop_m),
        "terrain_smoothing_passes": 1,
        "vertical_surface_smoothing": 0,
        "height_terrain_relation": _HEIGHT_TERRAIN_RELATION,
        "wind_speed_regrid_semantics": _WIND_REGRID_SEMANTICS,
        "regrid_algorithm": _REGRID_ALGORITHM,
        "regrid_evidence": REGRID_EVIDENCE,
        "regrid_source_fingerprint": weights.source_fingerprint,
    }
    if mesh_pair_binding is not None:
        provenance.update(
            {
                "materialization_mesh_pair_schema": mesh_pair_binding["schema"],
                "materialization_dynamics_mesh_sha256": mesh_pair_binding[
                    "dynamics_mesh_sha256"
                ],
                "materialization_output_mesh_sha256": mesh_pair_binding[
                    "output_mesh_sha256"
                ],
                "materialization_topology_sha256": mesh_pair_binding[
                    "topology_sha256"
                ],
            }
        )
    materialized_sources: dict[str, Path] = {
        "history": history,
        "latlon": latlon,
        "grid": grid,
        "static": static,
    }
    if saved_weights is not None:
        assert actual_weights_sha256 is not None
        provenance.update(
            {
                "source_regrid_weights_name": saved_weights.name,
                "source_regrid_weights_sha256": actual_weights_sha256,
                "regrid_weights_schema": _REGRID_WEIGHTS_SCHEMA,
                "regrid_weights_method": weights.method,
                "regrid_weights_neighbor_count": weights.n_neighbors,
                "regrid_weights_power": weights.power,
                "regrid_weights_binding": _REGRID_WEIGHTS_BINDING,
                "renderer_materialization_schema": _RENDERER_MATERIALIZATION_SCHEMA,
                "renderer_reconstruction_semantics": _RECONSTRUCTION_SEMANTICS,
            }
        )
        materialized_sources["regrid_weights"] = saved_weights
    fields = RustWrf2dFields(
        latitude=latitude,
        longitude=longitude,
        valid_times=latlon_times,
        initial_time=latlon_times[0],
        temperature_lowest_model_level=temperature,
        pressure_lowest_model_level=pressure_lowest,
        height_lowest_model_level=height_lowest,
        surface_pressure=surface_pressure,
        terrain_height=terrain_height,
        u_lowest_model_level=u_wind,
        v_lowest_model_level=v_wind,
        wind_speed_lowest_model_level=wind_speed,
        global_attrs=provenance,
    )
    result = write_rust_wrf2d_netcdf(destination, fields, clobber=clobber)
    validate_rust_wrf2d_netcdf(
        result,
        materialized_sources=materialized_sources,
        require_materialized_provenance=True,
    )
    return result


def _validate_materialized_provenance(
    dataset: Dataset,
    *,
    sources: Mapping[str, str | Path] | None,
    required: bool,
) -> None:
    base_source_roles = ("history", "latlon", "grid", "static")
    saved_weights_attributes = {
        "source_regrid_weights_name",
        "source_regrid_weights_sha256",
        "regrid_weights_schema",
        "regrid_weights_method",
        "regrid_weights_neighbor_count",
        "regrid_weights_power",
        "regrid_weights_binding",
        "renderer_materialization_schema",
        "renderer_reconstruction_semantics",
    }
    mesh_pair_attributes = {
        "materialization_mesh_pair_schema",
        "materialization_dynamics_mesh_sha256",
        "materialization_output_mesh_sha256",
        "materialization_topology_sha256",
    }
    saved_weights_present = bool(
        saved_weights_attributes.intersection(dataset.ncattrs())
    )
    mesh_pair_present = bool(mesh_pair_attributes.intersection(dataset.ncattrs()))
    source_roles = (
        (*base_source_roles, "regrid_weights")
        if saved_weights_present
        else base_source_roles
    )
    required_attributes = {
        *(f"source_{role}_name" for role in source_roles),
        *(f"source_{role}_sha256" for role in source_roles),
        "vertical_levels",
        "vertical_top_m",
        "terrain_smoothing_passes",
        "vertical_surface_smoothing",
        "height_terrain_relation",
        "wind_speed_regrid_semantics",
        "regrid_algorithm",
        "regrid_evidence",
        "regrid_source_fingerprint",
    }
    if saved_weights_present:
        required_attributes.update(saved_weights_attributes)
    if mesh_pair_present:
        required_attributes.update(mesh_pair_attributes)
    present_attributes = required_attributes.intersection(dataset.ncattrs())
    for attribute in dataset.ncattrs():
        if attribute.endswith("_sha256"):
            value = str(getattr(dataset, attribute))
            if _SHA256.fullmatch(value) is None:
                raise RustRendererError(
                    f"renderer provenance {attribute} is not a lowercase SHA-256 digest"
                )
    if not required and sources is None and not present_attributes:
        return
    missing_attributes = required_attributes - set(dataset.ncattrs())
    if missing_attributes:
        raise RustRendererError(
            "renderer input has partial materialized provenance; missing: "
            + ", ".join(sorted(missing_attributes))
        )
    if sources is not None and set(sources) != set(source_roles):
        raise RustRendererError(
            "materialized source binding does not match the adapter's exact source inventory"
        )
    for role in source_roles:
        name_attr = f"source_{role}_name"
        hash_attr = f"source_{role}_sha256"
        name = str(getattr(dataset, name_attr, ""))
        digest = str(getattr(dataset, hash_attr, ""))
        if not name or Path(name).name != name or _SHA256.fullmatch(digest) is None:
            raise RustRendererError(
                f"materialized renderer provenance {role} name/hash is missing or invalid"
            )
        if sources is not None:
            source = Path(sources[role]).expanduser().resolve(strict=True)
            if name != source.name or digest != sha256_file(source):
                raise RustRendererError(
                    f"materialized renderer provenance does not bind source {role}"
                )
    exact = {
        "terrain_smoothing_passes": 1,
        "vertical_surface_smoothing": 0,
        "height_terrain_relation": _HEIGHT_TERRAIN_RELATION,
        "wind_speed_regrid_semantics": _WIND_REGRID_SEMANTICS,
        "regrid_algorithm": _REGRID_ALGORITHM,
        "regrid_evidence": REGRID_EVIDENCE,
    }
    for attribute, expected in exact.items():
        if getattr(dataset, attribute, None) != expected:
            raise RustRendererError(
                f"materialized renderer provenance {attribute} is missing or stale"
            )
    if saved_weights_present:
        saved_exact = {
            "regrid_weights_schema": _REGRID_WEIGHTS_SCHEMA,
            "regrid_weights_method": "inverse_distance",
            "regrid_weights_neighbor_count": 4,
            "regrid_weights_power": 2.0,
            "regrid_weights_binding": _REGRID_WEIGHTS_BINDING,
            "renderer_materialization_schema": _RENDERER_MATERIALIZATION_SCHEMA,
            "renderer_reconstruction_semantics": _RECONSTRUCTION_SEMANTICS,
        }
        for attribute, expected in saved_exact.items():
            if getattr(dataset, attribute, None) != expected:
                raise RustRendererError(
                    f"materialized renderer provenance {attribute} is missing or stale"
                )
    if mesh_pair_present and getattr(
        dataset, "materialization_mesh_pair_schema", None
    ) != _MATERIALIZATION_MESH_PAIR_SCHEMA:
        raise RustRendererError(
            "materialized renderer provenance mesh-pair schema is missing or stale"
        )
    vertical_top = float(getattr(dataset, "vertical_top_m", np.nan))
    if (
        int(getattr(dataset, "vertical_levels", 0)) <= 0
        or not np.isfinite(vertical_top)
        or vertical_top <= 0.0
    ):
        raise RustRendererError("materialized vertical-grid provenance is invalid")
    fingerprint = str(getattr(dataset, "regrid_source_fingerprint", ""))
    if _SHA256.fullmatch(fingerprint) is None:
        raise RustRendererError("materialized regrid source fingerprint is invalid")


def validate_rust_wrf2d_netcdf(
    path: str | Path,
    *,
    materialized_sources: Mapping[str, str | Path] | None = None,
    require_materialized_provenance: bool = False,
) -> dict[str, Any]:
    """Fail closed unless ``path`` implements the exact truthful adapter schema."""

    target = Path(path).expanduser().resolve(strict=True)
    with Dataset(target) as dataset:
        dataset.set_auto_mask(False)
        if dataset.file_format != "NETCDF4_CLASSIC":
            raise RustRendererError("renderer input must use NETCDF4_CLASSIC")
        if getattr(dataset, "renderer_adapter_schema", None) != RENDERER_INPUT_SCHEMA:
            raise RustRendererError(
                "renderer adapter schema marker is missing or stale"
            )
        if int(getattr(dataset, "renderer_abi_claimed", -1)) != 0:
            raise RustRendererError("renderer input must explicitly make no ABI claim")
        expected_globals = {
            "Conventions": "CF-1.10",
            "title": "MPAS forecast fields regridded for unchanged rw_wrfbatch",
            "source": "hexcore.rust_renderer",
            "renderer_contract": RENDERER_CONTRACT,
            "grid_type": "regular_ll",
        }
        for name, expected in expected_globals.items():
            if getattr(dataset, name, None) != expected:
                raise RustRendererError(
                    f"renderer input global {name} is missing or stale"
                )
        if int(getattr(dataset, "MAP_PROJ", -1)) != 6:
            raise RustRendererError("renderer input must declare unrotated MAP_PROJ=6")
        if (
            float(getattr(dataset, "POLE_LAT", np.nan)) != 90.0
            or float(getattr(dataset, "POLE_LON", np.nan)) != 0.0
        ):
            raise RustRendererError(
                "renderer input must declare the unrotated geographic pole"
            )
        names = set(dataset.variables)
        if set(dataset.dimensions) != {
            _TIME_DIMENSION,
            _Y_DIMENSION,
            _X_DIMENSION,
        }:
            raise RustRendererError("renderer input has undeclared dimensions")
        missing = _REQUIRED_RENDERER_VARIABLES - names
        if missing:
            raise RustRendererError(
                "renderer input is missing required variables: "
                + ", ".join(sorted(missing))
            )
        forbidden = _MISLEADING_OR_ROUTE_BREAKING_VARIABLES.intersection(names)
        if forbidden:
            raise RustRendererError(
                "renderer input contains forbidden/misleading variables: "
                + ", ".join(sorted(forbidden))
            )
        unexpected = names - _REQUIRED_RENDERER_VARIABLES
        if unexpected:
            raise RustRendererError(
                "renderer input contains undeclared variables: "
                + ", ".join(sorted(unexpected))
            )
        expected_dimensions = {
            "XLAT": (_Y_DIMENSION, _X_DIMENSION),
            "XLONG": (_Y_DIMENSION, _X_DIMENSION),
            "TK": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "P": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "Z": (_Y_DIMENSION, _X_DIMENSION),
            "PSFC": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "HGT": (_Y_DIMENSION, _X_DIMENSION),
            "surface_pressure": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "temperature_lowest_model_level": (
                _TIME_DIMENSION,
                _Y_DIMENSION,
                _X_DIMENSION,
            ),
            "u_lowest_model_level": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "v_lowest_model_level": (_TIME_DIMENSION, _Y_DIMENSION, _X_DIMENSION),
            "wind_speed_lowest_model_level": (
                _TIME_DIMENSION,
                _Y_DIMENSION,
                _X_DIMENSION,
            ),
        }
        for name, dimensions in expected_dimensions.items():
            actual = dataset.variables[name].dimensions
            if actual != dimensions:
                raise RustRendererError(
                    f"renderer variable {name} dimensions {actual} != {dimensions}"
                )
        if dataset.dimensions[_TIME_DIMENSION].isunlimited():
            raise RustRendererError(
                "renderer Time dimension must be fixed, not unlimited"
            )
        time_variable = dataset.variables["time"]
        if time_variable.dimensions != (_TIME_DIMENSION,):
            raise RustRendererError(
                "lowercase numeric time must use the Time dimension"
            )
        if np.dtype(time_variable.dtype) != np.dtype("float64"):
            raise RustRendererError(
                "lowercase time coordinate must use exact float64 storage"
            )
        time_units = str(getattr(time_variable, "units", ""))
        expected_time_attrs = {
            "standard_name": "time",
            "long_name": "valid time",
            "axis": "T",
            "calendar": "gregorian",
        }
        for name, expected in expected_time_attrs.items():
            if getattr(time_variable, name, None) != expected:
                raise RustRendererError(
                    f"renderer time coordinate {name} is missing or stale"
                )
        if _TIME_UNITS.fullmatch(time_units) is None:
            raise RustRendererError(
                "renderer time units must be whole seconds since an exact UTC-like epoch"
            )
        time_values = np.asarray(time_variable[:], dtype=np.float64)
        if (
            time_values.ndim != 1
            or time_values.size == 0
            or not np.all(np.isfinite(time_values))
            or not np.all(time_values == np.floor(time_values))
            or time_values[0] != 0.0
            or np.any(np.diff(time_values) <= 0.0)
        ):
            raise RustRendererError(
                "renderer time values must be finite, whole-second, start at zero, and increase"
            )
        valid_times = _read_times(dataset, "renderer input")
        if any(right <= left for left, right in zip(valid_times, valid_times[1:])):
            raise RustRendererError(
                "decoded renderer valid times must strictly increase"
            )
        latitude = np.asarray(dataset.variables["XLAT"][:], dtype=np.float64)
        longitude = np.asarray(dataset.variables["XLONG"][:], dtype=np.float64)
        if np.dtype(dataset.variables["XLAT"].dtype) != np.dtype("float64") or np.dtype(
            dataset.variables["XLONG"].dtype
        ) != np.dtype("float64"):
            raise RustRendererError(
                "renderer XLAT/XLONG must use exact float64 storage"
            )
        if (
            not np.all(np.isfinite(latitude))
            or not np.all(np.isfinite(longitude))
            or np.any(np.abs(latitude) > 90.0)
        ):
            raise RustRendererError("renderer XLAT/XLONG contain invalid coordinates")
        latitude_axis = latitude[:, 0]
        longitude_axis = longitude[0, :]
        expected_latitude, expected_longitude = np.meshgrid(
            latitude_axis, longitude_axis, indexing="ij"
        )
        if not np.array_equal(latitude, expected_latitude) or not np.array_equal(
            longitude, expected_longitude
        ):
            raise RustRendererError("renderer XLAT/XLONG are not regular 2-D meshgrids")
        if np.any(np.diff(latitude_axis) <= 0.0) or np.any(
            np.diff(longitude_axis) <= 0.0
        ):
            raise RustRendererError(
                "renderer coordinate axes are not strictly increasing"
            )
        coordinate_attrs = {
            "XLAT": {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
            },
            "XLONG": {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
            },
        }
        for variable_name, attributes in coordinate_attrs.items():
            variable = dataset.variables[variable_name]
            for attribute, expected in attributes.items():
                if getattr(variable, attribute, None) != expected:
                    raise RustRendererError(
                        f"renderer {variable_name} {attribute} is missing or stale"
                    )
        units = {
            "TK": "K",
            "P": "Pa",
            "Z": "m",
            "PSFC": "Pa",
            "HGT": "m",
            "surface_pressure": "Pa",
            "temperature_lowest_model_level": "K",
            "u_lowest_model_level": "m s-1",
            "v_lowest_model_level": "m s-1",
            "wind_speed_lowest_model_level": "m s-1",
        }
        long_names = {
            "TK": "temperature at lowest model level",
            "P": "pressure at lowest model level",
            "Z": "geometric height of lowest model level",
            "PSFC": "surface pressure",
            "HGT": "terrain height above mean sea level",
            "surface_pressure": "surface pressure",
            "temperature_lowest_model_level": "temperature at lowest model level",
            "u_lowest_model_level": "earth-relative zonal wind at lowest model level",
            "v_lowest_model_level": "earth-relative meridional wind at lowest model level",
            "wind_speed_lowest_model_level": "horizontal wind speed at lowest model level",
        }
        for name, expected in units.items():
            variable = dataset.variables[name]
            if np.dtype(variable.dtype).kind != "f":
                raise RustRendererError(
                    f"renderer variable {name} must use floating storage"
                )
            if getattr(variable, "units", None) != expected:
                raise RustRendererError(f"renderer variable {name} must use {expected}")
            if getattr(variable, "long_name", None) != long_names[name]:
                raise RustRendererError(f"renderer variable {name} long_name is stale")
            if getattr(variable, "coordinates", None) != "XLAT XLONG":
                raise RustRendererError(
                    f"renderer variable {name} coordinates are stale"
                )
            values = np.asarray(variable[:])
            if not np.all(np.isfinite(values)):
                raise RustRendererError(f"renderer variable {name} is not finite")
        if not np.array_equal(
            dataset.variables["TK"][:],
            dataset.variables["temperature_lowest_model_level"][:],
        ):
            raise RustRendererError(
                "TK must exactly duplicate lowest-model-level temperature"
            )
        if not np.array_equal(
            dataset.variables["PSFC"][:], dataset.variables["surface_pressure"][:]
        ):
            raise RustRendererError("PSFC must exactly duplicate surface pressure")
        temperature = np.asarray(dataset.variables["TK"][:])
        pressure = np.asarray(dataset.variables["P"][:])
        surface_pressure = np.asarray(dataset.variables["PSFC"][:])
        u_wind = np.asarray(dataset.variables["u_lowest_model_level"][:])
        v_wind = np.asarray(dataset.variables["v_lowest_model_level"][:])
        wind_speed = np.asarray(dataset.variables["wind_speed_lowest_model_level"][:])
        if np.any(temperature <= 0.0):
            raise RustRendererError("renderer temperature must be positive Kelvin")
        if np.any(pressure <= 0.0) or np.any(surface_pressure <= 0.0):
            raise RustRendererError("renderer pressure fields must be positive")
        if np.any(pressure > surface_pressure):
            raise RustRendererError(
                "renderer lowest-model pressure exceeds surface pressure"
            )
        if np.any(wind_speed < 0.0):
            raise RustRendererError("renderer wind speed cannot be negative")
        component_speed = np.hypot(u_wind, v_wind)
        tolerance = 1.0e-5 * np.maximum(1.0, component_speed)
        if np.any(wind_speed + tolerance < component_speed):
            raise RustRendererError(
                "renderer scalar wind speed violates the convex component-speed bound"
            )
        _validate_materialized_provenance(
            dataset,
            sources=materialized_sources,
            required=require_materialized_provenance,
        )
        shape = (
            len(dataset.dimensions[_Y_DIMENSION]),
            len(dataset.dimensions[_X_DIMENSION]),
        )
        return {
            "schema": RENDERER_INPUT_SCHEMA,
            "records": len(valid_times),
            "grid_shape": list(shape),
            "valid_times": [value.isoformat() + "Z" for value in valid_times],
            "variables": sorted(names),
            "forbidden_surface_aliases_absent": True,
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }


@dataclass(frozen=True, slots=True)
class RendererProbe:
    executable: Path
    executable_sha256: str
    executable_bytes: int
    probe_evidence: str
    basemap_dir: Path
    contract: str = RENDERER_CONTRACT
    abi_claimed: bool = RENDERER_ABI_CLAIMED


@dataclass(frozen=True, slots=True)
class RendererCatalogRow:
    slug: str
    kind: str
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class RendererCatalog:
    rows: tuple[RendererCatalogRow, ...]
    summary: str
    elapsed_seconds: float
    renderer_input_sha256: str
    renderer_sha256: str
    store_root: Path
    heavy: bool

    def row(self, slug: str) -> RendererCatalogRow | None:
        return next((row for row in self.rows if row.slug == slug), None)


@dataclass(frozen=True, slots=True)
class RendererRun:
    products: tuple[str, ...]
    outputs: tuple[Path, ...]
    output_sha256: tuple[str, ...]
    catalog: RendererCatalog
    elapsed_seconds: float
    renderer_input_sha256: str
    renderer_sha256: str
    frames: str
    width: int
    height: int
    source_label: str
    integrity_scope: str = INTEGRITY_SCOPE


def _load_rustwx() -> Any:
    try:
        return importlib.import_module("gpuwm.rustwx")
    except (ImportError, ModuleNotFoundError) as error:
        raise RustRendererError(
            "gpuwm.rustwx is not importable; install/activate the unchanged gpuwm runtime"
        ) from error


def _validate_probe_integrity(probe: RendererProbe) -> None:
    try:
        executable = probe.executable.resolve(strict=True)
    except OSError as error:
        raise RustRendererError(
            "renderer executable disappeared after probing"
        ) from error
    if not executable.is_file():
        raise RustRendererError("renderer executable is no longer a regular file")
    if executable.stat().st_size != probe.executable_bytes:
        raise RustRendererError("renderer executable size changed after probing")
    if sha256_file(executable) != probe.executable_sha256:
        raise RustRendererError("renderer executable SHA-256 changed after probing")
    if not probe.basemap_dir.resolve().is_dir():
        raise RustRendererError("renderer basemap disappeared after probing")


def _validate_catalog(catalog: RendererCatalog) -> None:
    if not catalog.rows:
        raise RustRendererError("renderer real-import catalog returned no product rows")
    if (
        _SHA256.fullmatch(catalog.renderer_input_sha256) is None
        or _SHA256.fullmatch(catalog.renderer_sha256) is None
    ):
        raise RustRendererError("renderer catalog has invalid input/executable binding")
    if not np.isfinite(catalog.elapsed_seconds) or catalog.elapsed_seconds < 0.0:
        raise RustRendererError("renderer catalog has invalid elapsed time")
    seen: set[str] = set()
    for row in catalog.rows:
        if _PRODUCT_SLUG.fullmatch(row.slug) is None:
            raise RustRendererError(f"renderer catalog has malformed slug {row.slug!r}")
        if row.slug in seen:
            raise RustRendererError(
                f"renderer catalog returned duplicate slug {row.slug!r}"
            )
        seen.add(row.slug)
        if row.kind not in _CATALOG_KINDS:
            raise RustRendererError(f"renderer catalog has unknown kind {row.kind!r}")
        if row.status not in _CATALOG_STATUSES:
            raise RustRendererError(
                f"renderer catalog has unknown status {row.status!r}"
            )
        if not row.detail or any(ord(character) < 32 for character in row.detail):
            raise RustRendererError(
                f"renderer catalog has malformed detail for {row.slug!r}"
            )
    tokens = catalog.summary.split()
    parsed: dict[str, int] = {}
    for token in tokens:
        match = re.fullmatch(r"([a-z][a-z-]*)=([0-9]+)", token)
        if match is None or match.group(1) in parsed:
            raise RustRendererError("renderer catalog summary is malformed")
        parsed[match.group(1)] = int(match.group(2))
    if parsed.get("total") != len(catalog.rows):
        raise RustRendererError("renderer catalog summary total disagrees with rows")
    status_counts = Counter(row.status for row in catalog.rows)
    if any(
        parsed.get(status, 0) != status_counts.get(status, 0)
        for status in _CATALOG_STATUSES
    ):
        raise RustRendererError(
            "renderer catalog summary status counts disagree with rows"
        )
    unknown_summary = set(parsed) - ({"total"} | _CATALOG_STATUSES)
    if unknown_summary:
        raise RustRendererError("renderer catalog summary has unknown counters")


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise RustRendererError(f"renderer output has no valid PNG signature: {path}")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise RustRendererError(f"renderer PNG has a truncated chunk: {path}")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise RustRendererError(
                f"renderer PNG has a truncated chunk payload: {path}"
            )
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(payload[offset + 8 + length : end], "big")
        actual_crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise RustRendererError(f"renderer PNG has a bad chunk CRC: {path}")
        chunks.append((chunk_type, data))
        offset = end
        if chunk_type == b"IEND":
            break
    if offset != len(payload) or not chunks or chunks[-1] != (b"IEND", b""):
        raise RustRendererError(f"renderer PNG has no final IEND chunk: {path}")
    if chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise RustRendererError(
            f"renderer output has no valid first IHDR chunk: {path}"
        )
    ihdr = chunks[0][1]
    width = int.from_bytes(ihdr[:4], "big")
    height = int.from_bytes(ihdr[4:8], "big")
    if width <= 0 or height <= 0 or ihdr[8:] != b"\x08\x06\x00\x00\x00":
        raise RustRendererError(
            f"renderer PNG is not positive-sized non-interlaced 8-bit RGBA: {path}"
        )
    compressed = b"".join(data for chunk_type, data in chunks if chunk_type == b"IDAT")
    if not compressed:
        raise RustRendererError(f"renderer PNG has no IDAT payload: {path}")
    try:
        pixels = zlib.decompress(compressed)
    except zlib.error as error:
        raise RustRendererError(
            f"renderer PNG IDAT stream is invalid: {path}"
        ) from error
    if len(pixels) != height * (1 + 4 * width):
        raise RustRendererError(f"renderer PNG RGBA scanline size is invalid: {path}")
    stride = 1 + 4 * width
    if any(pixels[row * stride] > 4 for row in range(height)):
        raise RustRendererError(
            f"renderer PNG uses an invalid scanline filter byte: {path}"
        )
    return width, height


def _output_snapshot(output: Path) -> dict[Path, tuple[int, int, str]]:
    snapshot: dict[Path, tuple[int, int, str]] = {}
    for candidate in output.rglob("*.png"):
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            stat = resolved.stat()
            snapshot[resolved] = (stat.st_size, stat.st_mtime_ns, sha256_file(resolved))
    return snapshot


def _bind_output_products(paths: Sequence[Path], products: Sequence[str]) -> None:
    unmatched = set(products)
    for path in paths:
        normalized = path.stem.lower().replace(":", "_").replace(".", "_")
        matches = [
            slug
            for slug in unmatched
            if slug.lower().split(":", 1)[-1].replace(".", "_") in normalized
        ]
        if len(matches) != 1:
            raise RustRendererError(
                f"renderer PNG filename does not bind one selected product: {path.name}"
            )
        unmatched.remove(matches[0])
    if unmatched:
        raise RustRendererError(
            "renderer PNG filenames omitted selected products: "
            + ", ".join(sorted(unmatched))
        )


def discover_rust_renderer(
    renderer: str | Path | None = None,
    *,
    _rustwx_module: Any | None = None,
) -> RendererProbe:
    """Resolve and launch-probe the unchanged renderer through gpuwm.rustwx."""

    rustwx = _load_rustwx() if _rustwx_module is None else _rustwx_module
    try:
        found = (
            Path(renderer).expanduser().resolve(strict=True)
            if renderer
            else rustwx.find_renderer()
        )
    except (FileNotFoundError, OSError) as error:
        raise RustRendererError(f"renderer discovery failed: {error}") from error
    if found is None:
        raise RustRendererError("gpuwm.rustwx found no rw_wrfbatch executable")
    executable = Path(found).expanduser().resolve(strict=True)
    if not executable.is_file():
        raise RustRendererError(f"renderer is not a regular file: {executable}")
    executable_bytes = executable.stat().st_size
    executable_sha256 = sha256_file(executable)
    try:
        ok, evidence = rustwx.probe_renderer(executable)
    except (OSError, RuntimeError) as error:
        raise RustRendererError(f"renderer --help probe failed: {error}") from error
    if not ok:
        raise RustRendererError(f"renderer --help probe refused executable: {evidence}")
    basemap = rustwx.resolve_basemap_dir(executable)
    if basemap is None or not Path(basemap).is_dir():
        raise RustRendererError("renderer has no resolvable basemap asset directory")
    if (
        executable.stat().st_size != executable_bytes
        or sha256_file(executable) != executable_sha256
    ):
        raise RustRendererError("renderer executable changed during discovery/probing")
    probe = RendererProbe(
        executable=executable,
        executable_sha256=executable_sha256,
        executable_bytes=executable_bytes,
        probe_evidence=str(evidence),
        basemap_dir=Path(basemap).resolve(),
    )
    _validate_probe_integrity(probe)
    return probe


def inspect_renderer_products(
    renderer_input: str | Path,
    *,
    store_root: str | Path,
    probe: RendererProbe,
    heavy: bool = False,
    _rustwx_module: Any | None = None,
) -> RendererCatalog:
    """Run the real import and store-aware catalog handshake."""

    input_path = Path(renderer_input).expanduser().resolve(strict=True)
    input_validation = validate_rust_wrf2d_netcdf(input_path)
    input_sha256 = str(input_validation["sha256"])
    _validate_probe_integrity(probe)
    store = Path(store_root).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    rustwx = _load_rustwx() if _rustwx_module is None else _rustwx_module
    started = time.perf_counter()
    try:
        rows, summary = rustwx.list_products(
            probe.executable,
            input_path,
            store_root=store,
            heavy=heavy,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RustRendererError(
            f"renderer real-import catalog failed: {error}"
        ) from error
    elapsed = time.perf_counter() - started
    _validate_probe_integrity(probe)
    post_validation = validate_rust_wrf2d_netcdf(input_path)
    if post_validation["sha256"] != input_sha256:
        raise RustRendererError("renderer input changed during real-import catalog")
    converted = tuple(RendererCatalogRow(*map(str, row)) for row in rows)
    catalog = RendererCatalog(
        rows=converted,
        summary=str(summary),
        elapsed_seconds=elapsed,
        renderer_input_sha256=input_sha256,
        renderer_sha256=probe.executable_sha256,
        store_root=store.resolve(),
        heavy=bool(heavy),
    )
    _validate_catalog(catalog)
    return catalog


def render_catalogued_products(
    renderer_input: str | Path,
    *,
    store_root: str | Path,
    out_dir: str | Path,
    products: Sequence[str],
    probe: RendererProbe,
    catalog: RendererCatalog | None = None,
    frames: str = "1",
    width: int = 1200,
    height: int = 800,
    heavy: bool = False,
    source_label: str = "MPAS-Atmosphere Python CPU port",
    _rustwx_module: Any | None = None,
) -> RendererRun:
    """Render exact catalog slugs and refuse skips, failures or missing PNGs."""

    input_path = Path(renderer_input).expanduser().resolve(strict=True)
    input_validation = validate_rust_wrf2d_netcdf(input_path)
    input_sha256 = str(input_validation["sha256"])
    _validate_probe_integrity(probe)
    selected = tuple(str(product).strip() for product in products)
    if not selected or any(not product for product in selected):
        raise ValueError("at least one non-empty renderer product is required")
    if len(set(selected)) != len(selected):
        raise ValueError("renderer products must be unique")
    if any(_PRODUCT_SLUG.fullmatch(product) is None for product in selected):
        raise ValueError("renderer product contains unsupported characters")
    if str(frames) != "1":
        raise ValueError(
            "the renderer seam requires the one explicit final frame index '1'"
        )
    if width < 64 or height < 64:
        raise ValueError("renderer output dimensions must be at least 64 pixels")
    if (
        not source_label
        or len(source_label) > 160
        or any(
            ord(character) < 32 or ord(character) > 126 for character in source_label
        )
    ):
        raise ValueError(
            "renderer source_label must be 1-160 printable ASCII characters"
        )
    store = Path(store_root).expanduser().resolve()
    output = Path(out_dir).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    actual_catalog = catalog or inspect_renderer_products(
        input_path,
        store_root=store,
        probe=probe,
        heavy=heavy,
        _rustwx_module=_rustwx_module,
    )
    _validate_catalog(actual_catalog)
    if (
        actual_catalog.renderer_input_sha256 != input_sha256
        or actual_catalog.renderer_sha256 != probe.executable_sha256
        or actual_catalog.store_root != store.resolve()
        or actual_catalog.heavy != bool(heavy)
    ):
        raise RustRendererError(
            "supplied renderer catalog is stale or bound to a different input/probe/store"
        )
    for slug in selected:
        row = actual_catalog.row(slug)
        if row is None:
            raise RustRendererError(f"renderer catalog has no requested slug {slug!r}")
        if row.status != "renderable":
            raise RustRendererError(
                f"renderer catalog refuses {slug!r}: {row.status}: {row.detail}"
            )
    rustwx = _load_rustwx() if _rustwx_module is None else _rustwx_module
    before_outputs = _output_snapshot(output)
    started = time.perf_counter()
    try:
        written, failures, skipped = rustwx.run_renderer(
            probe.executable,
            input_path,
            store_root=store,
            out_dir=output,
            products=",".join(selected),
            frames=str(frames),
            width=int(width),
            height=int(height),
            heavy=heavy,
            source_label=source_label,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RustRendererError(f"renderer execution failed: {error}") from error
    elapsed = time.perf_counter() - started
    _validate_probe_integrity(probe)
    post_validation = validate_rust_wrf2d_netcdf(input_path)
    if post_validation["sha256"] != input_sha256:
        raise RustRendererError("renderer input changed during rendering")
    if failures:
        raise RustRendererError(
            "renderer reported FAILED: " + " | ".join(map(str, failures))
        )
    if skipped:
        details = [f"{slug}: {reason}" for slug, reason in skipped]
        raise RustRendererError("renderer reported SKIPPED: " + " | ".join(details))
    output_root = output.resolve()
    resolved_outputs: list[Path] = []
    seen_outputs: set[Path] = set()
    for path in written:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = output / candidate
        candidate = candidate.resolve(strict=True)
        try:
            candidate.relative_to(output_root)
        except ValueError as error:
            raise RustRendererError(
                f"renderer reported an output outside out_dir: {candidate}"
            ) from error
        if not candidate.is_file() or candidate.suffix.lower() != ".png":
            raise RustRendererError(f"renderer output is not a PNG file: {candidate}")
        if candidate in seen_outputs:
            raise RustRendererError(
                f"renderer reported a duplicate PNG path: {candidate}"
            )
        seen_outputs.add(candidate)
        if candidate in before_outputs:
            raise RustRendererError(
                f"renderer must write into a fresh path, not reuse a pre-existing PNG: {candidate}"
            )
        actual_dimensions = _png_dimensions(candidate)
        if actual_dimensions != (int(width), int(height)):
            raise RustRendererError(
                f"renderer PNG dimensions {actual_dimensions} != {(int(width), int(height))}"
            )
        resolved_outputs.append(candidate)
    if not resolved_outputs:
        raise RustRendererError("renderer returned success but wrote zero PNGs")
    if len(resolved_outputs) != len(selected):
        raise RustRendererError(
            f"renderer wrote {len(resolved_outputs)} PNG(s) for {len(selected)} products"
        )
    post_pngs = {
        candidate.resolve(strict=True)
        for candidate in output.rglob("*.png")
        if candidate.is_file()
    }
    newly_created_pngs = post_pngs - set(before_outputs)
    if newly_created_pngs != set(resolved_outputs):
        raise RustRendererError(
            "renderer output directory contains unreported newly-created PNGs"
        )
    _bind_output_products(resolved_outputs, selected)
    output_hashes = tuple(sha256_file(path) for path in resolved_outputs)
    return RendererRun(
        products=selected,
        outputs=tuple(resolved_outputs),
        output_sha256=output_hashes,
        catalog=actual_catalog,
        elapsed_seconds=elapsed,
        renderer_input_sha256=input_sha256,
        renderer_sha256=probe.executable_sha256,
        frames=str(frames),
        width=int(width),
        height=int(height),
        source_label=str(source_label),
        integrity_scope=INTEGRITY_SCOPE,
    )


__all__ = [
    "FROZEN_X1_2562_REGRID_WEIGHTS_SHA256",
    "INTEGRITY_SCOPE",
    "RENDERER_ABI_CLAIMED",
    "RENDERER_CONTRACT",
    "RENDERER_INPUT_SCHEMA",
    "RendererCatalog",
    "RendererCatalogRow",
    "RendererProbe",
    "RendererRun",
    "RustRendererError",
    "RustWrf2dFields",
    "discover_rust_renderer",
    "inspect_renderer_products",
    "materialize_gfs_rust_input",
    "render_catalogued_products",
    "sha256_file",
    "validate_materialization_mesh_pair",
    "validate_rust_wrf2d_netcdf",
    "write_renderer_materialization_authority",
    "write_rust_wrf2d_netcdf",
]
