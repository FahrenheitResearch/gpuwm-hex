"""MPAS-compatible serial history output.

The file layout follows the frozen MPAS-A v8.2.3 ``output`` stream in
``src/core_atmosphere/Registry.xml:851-974``.  In particular, mesh geometry
and connectivity are invariant variables, while ``Time`` and model fields are
record variables.  The definitions of ``initial_time``, ``xtime``, ``Time``,
and ``u`` are at Registry lines 1520-1532.  MPAS's stream manager recognizes
single/double precision and serial NetCDF/NetCDF-4 at
``src/framework/xml_stream_parser.c:1239-1325``.

This module implements the serial, separate-file part of that contract.  It
does not pretend to schedule alarms or provide Parallel-NetCDF: every such
stream option refuses by its MPAS knob name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from netCDF4 import Dataset
from numpy.typing import NDArray

from .errors import ConfigurationRefusal


_FROZEN_OUTPUT_SOURCE = (
    "MPAS-Model-v8.2.3/src/core_atmosphere/Registry.xml:851-974,1520-1532"
)
_STR_LEN = 64

# The invariant fields named by the frozen atmosphere ``output`` stream.  A
# field is only emitted when it is present on the supplied mesh.
HISTORY_MESH_VARIABLES: tuple[str, ...] = (
    "latCell",
    "lonCell",
    "xCell",
    "yCell",
    "zCell",
    "indexToCellID",
    "latEdge",
    "lonEdge",
    "xEdge",
    "yEdge",
    "zEdge",
    "indexToEdgeID",
    "latVertex",
    "lonVertex",
    "xVertex",
    "yVertex",
    "zVertex",
    "indexToVertexID",
    "cellsOnEdge",
    "nEdgesOnCell",
    "nEdgesOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "weightsOnEdge",
    "dvEdge",
    "dcEdge",
    "angleEdge",
    "areaCell",
    "areaTriangle",
    "cellsOnCell",
    "verticesOnCell",
    "verticesOnEdge",
    "edgesOnVertex",
    "cellsOnVertex",
    "kiteAreasOnVertex",
    "meshDensity",
    "zgrid",
    "fzm",
    "fzp",
    "zz",
)

_CONNECTIVITY_VARIABLES = {
    "cellsOnEdge",
    "edgesOnCell",
    "edgesOnEdge",
    "cellsOnCell",
    "verticesOnCell",
    "verticesOnEdge",
    "edgesOnVertex",
    "cellsOnVertex",
    "advCellsForEdge",
    "edgesOnCellOne",
    "edgesOnCellTwo",
}

_ENTITY_DIMENSIONS = ("nCells", "nEdges", "nVertices")
_VERTICAL_DIMENSIONS = ("nVertLevels", "nVertLevelsP1", "nVertLevelsP2")

_FIELD_METADATA: dict[str, dict[str, str]] = {
    "u": {"units": "m s^{-1}", "long_name": "Horizontal normal velocity at edges"},
    "w": {"units": "m s^{-1}", "long_name": "Vertical velocity at vertical cell faces"},
    "pressure": {"units": "Pa", "long_name": "Pressure"},
    "surface_pressure": {"units": "Pa", "long_name": "Surface pressure"},
    "rho": {"units": "kg m^{-3}", "long_name": "Dry air density"},
    "theta": {"units": "K", "long_name": "Potential temperature"},
    "relhum": {"units": "%", "long_name": "Relative humidity"},
    "divergence": {"units": "s^{-1}", "long_name": "Horizontal divergence"},
    "vorticity": {"units": "s^{-1}", "long_name": "Vertical vorticity"},
    "ke": {"units": "m^2 s^{-2}", "long_name": "Kinetic energy"},
    "uReconstructZonal": {
        "units": "m s^{-1}",
        "long_name": "Reconstructed zonal wind",
    },
    "uReconstructMeridional": {
        "units": "m s^{-1}",
        "long_name": "Reconstructed meridional wind",
    },
}

# Frozen Registry logical (Fortran) dimensions for the core fields most often
# handed directly from this port's level-first authority arrays.  Time is
# inserted by the writer.  Unknown names still use size-based inference.
_KNOWN_FIELD_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "u": ("nVertLevels", "nEdges"),
    "w": ("nVertLevelsP1", "nCells"),
    "pressure": ("nVertLevels", "nCells"),
    "rho": ("nVertLevels", "nCells"),
    "theta": ("nVertLevels", "nCells"),
    "relhum": ("nVertLevels", "nCells"),
    "divergence": ("nVertLevels", "nCells"),
    "vorticity": ("nVertLevels", "nVertices"),
    "ke": ("nVertLevels", "nCells"),
    "uReconstructZonal": ("nVertLevels", "nCells"),
    "uReconstructMeridional": ("nVertLevels", "nCells"),
    "surface_pressure": ("nCells",),
    "rainnc": ("nCells",),
    "rainc": ("nCells",),
    "u10": ("nCells",),
    "v10": ("nCells",),
    "q2": ("nCells",),
    "t2m": ("nCells",),
}


@dataclass(frozen=True, slots=True)
class HistoryField:
    """A history field and the dimensions of its in-memory array.

    Dimension names describe the axes of ``values``.  The writer transposes
    these axes into MPAS NetCDF order: ``Time``, horizontal entity, vertical,
    then any remaining dimensions.  Thus both the port's readable logical
    layout ``(nVertLevels, nCells)`` and an already on-disk-shaped
    ``(nCells, nVertLevels)`` input are accepted without changing values.
    """

    values: NDArray[Any] | Sequence[Any]
    dimensions: tuple[str, ...] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


# Friendly short alias for callers constructing many variables.
Field = HistoryField


@dataclass(frozen=True, slots=True)
class HistoryStreamOptions:
    """The honestly supported slice of MPAS output-stream options."""

    stream_name: str = "output"
    io_type: str = "netcdf4"
    precision: str = "native"
    clobber_mode: str = "never_modify"
    runtime_format: str = "separate_file"
    output_interval: str | None = None
    filename_template: str | None = None
    filename_interval: str | None = None
    reference_time: str = "initial_time"
    record_interval: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "HistoryStreamOptions":
        if raw is None:
            return cls()
        aliases = {"name": "stream_name", "type": "stream_name"}
        allowed = set(cls.__dataclass_fields__)
        normalized: dict[str, Any] = {}
        for original, value in raw.items():
            name = aliases.get(original, original)
            if name not in allowed:
                raise ConfigurationRefusal(
                    original,
                    value,
                    "the serial history writer has no implementation for this stream option",
                    f"omit {original}",
                )
            normalized[name] = value
        return cls(**normalized)

    def validate(self) -> "HistoryStreamOptions":
        if self.stream_name not in {"output", "history"}:
            raise ConfigurationRefusal(
                "stream_name",
                self.stream_name,
                "only the frozen atmosphere output/history stream is implemented",
                "stream_name='output'",
            )
        if self.io_type not in {"netcdf", "serial_netcdf", "netcdf4"}:
            raise ConfigurationRefusal(
                "io_type",
                self.io_type,
                "Parallel-NetCDF, CDF-5, and SIONlib backends are not ported",
                "io_type='netcdf4' or io_type='netcdf'",
            )
        if self.precision not in {"native", "single", "double"}:
            raise ConfigurationRefusal(
                "precision",
                self.precision,
                "the frozen stream parser only defines native, single, and double real precision",
                "precision='native'",
            )
        if self.clobber_mode not in {"never_modify", "truncate", "replace_files"}:
            raise ConfigurationRefusal(
                "clobber_mode",
                self.clobber_mode,
                "record append/overwrite semantics are not implemented by the snapshot writer",
                "clobber_mode='never_modify' or clobber_mode='truncate'",
            )
        if self.runtime_format != "separate_file":
            raise ConfigurationRefusal(
                "runtime_format",
                self.runtime_format,
                "only one complete history file per call is implemented",
                "runtime_format='separate_file'",
            )
        for knob in (
            "output_interval",
            "filename_template",
            "filename_interval",
            "record_interval",
        ):
            value = getattr(self, knob)
            if value not in (None, "none"):
                raise ConfigurationRefusal(
                    knob,
                    value,
                    "alarm scheduling and filename rotation belong to the unported stream manager",
                    f"{knob}=None and call write_history at the desired time",
                )
        if self.reference_time != "initial_time":
            raise ConfigurationRefusal(
                "reference_time",
                self.reference_time,
                "only initial_time-relative CF time is emitted",
                "reference_time='initial_time'",
            )
        return self

    @property
    def netcdf_format(self) -> str:
        return "NETCDF4" if self.io_type == "netcdf4" else "NETCDF3_64BIT_OFFSET"


def format_mpas_filename(template: str, valid_time: datetime | str) -> str:
    """Expand the time tokens accepted by frozen MPAS filename templates."""

    stamp = _as_datetime(valid_time)
    replacements = {
        "$Y": f"{stamp.year:04d}",
        "$M": f"{stamp.month:02d}",
        "$D": f"{stamp.day:02d}",
        "$h": f"{stamp.hour:02d}",
        "$m": f"{stamp.minute:02d}",
        "$s": f"{stamp.second:02d}",
    }
    result = str(template)
    for token, replacement in replacements.items():
        result = result.replace(token, replacement)
    if "$" in result:
        token = result[result.index("$") : result.index("$") + 2]
        raise ConfigurationRefusal(
            "filename_template",
            template,
            f"unrecognized MPAS filename token {token!r}",
            "use only $Y, $M, $D, $h, $m, and $s",
        )
    return result


def _as_datetime(value: datetime | str | np.datetime64) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("valid_time must not be NaT")
        microseconds = value.astype("datetime64[us]").astype(np.int64)
        result = datetime.fromtimestamp(float(microseconds) / 1_000_000.0, timezone.utc)
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        result = datetime.fromisoformat(text.replace("_", "T", 1))
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result.replace(microsecond=0)


def _time_values(
    valid_time: datetime | str | np.datetime64 | Sequence[datetime | str | np.datetime64],
) -> tuple[datetime, ...]:
    if isinstance(valid_time, (datetime, str, np.datetime64)):
        return (_as_datetime(valid_time),)
    values = tuple(_as_datetime(value) for value in valid_time)
    if not values:
        raise ValueError("valid_time must contain at least one record")
    return values


def _mpas_time_string(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")


def _mesh_arrays(mesh: object) -> Mapping[str, Any]:
    arrays = getattr(mesh, "arrays", None)
    if arrays is not None:
        return arrays
    return {
        name: getattr(mesh, name)
        for name in HISTORY_MESH_VARIABLES
        if hasattr(mesh, name)
    }


def _mesh_dimensions(mesh: object) -> dict[str, int]:
    dimensions = getattr(mesh, "dimensions", None)
    if dimensions is not None:
        return {name: int(value) for name, value in dimensions.items()}
    result: dict[str, int] = {}
    arrays = _mesh_arrays(mesh)
    for name, variable in (
        ("nCells", "latCell"),
        ("nEdges", "latEdge"),
        ("nVertices", "latVertex"),
    ):
        if variable in arrays:
            result[name] = int(np.asarray(arrays[variable]).size)
    return result


def _mesh_variable_dimensions(mesh: object) -> Mapping[str, tuple[str, ...]]:
    return getattr(mesh, "variable_dimensions", {})


def _mesh_variable_attrs(mesh: object) -> Mapping[str, Mapping[str, Any]]:
    return getattr(mesh, "variable_attrs", {})


def _plain_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _set_attrs(target: Any, attrs: Mapping[str, Any]) -> None:
    clean = {
        str(name): _plain_attr(value)
        for name, value in attrs.items()
        if value is not None and name not in {"_FillValue", "_NCProperties"}
    }
    if clean:
        target.setncatts(clean)


def _field_parts(
    name: str,
    raw: Any,
    field_dimensions: Mapping[str, Sequence[str]] | None,
    field_attrs: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[NDArray[Any], tuple[str, ...] | None, dict[str, Any]]:
    if isinstance(raw, HistoryField):
        array = np.asarray(raw.values)
        dimensions = raw.dimensions
        attrs = dict(raw.attrs)
    else:
        array = np.asarray(raw)
        dimensions = None
        attrs = {}
    if field_dimensions is not None and name in field_dimensions:
        supplied = tuple(str(dim) for dim in field_dimensions[name])
        if dimensions is not None and tuple(dimensions) != supplied:
            raise ValueError(f"{name}: HistoryField dimensions disagree with field_dimensions")
        dimensions = supplied
    if field_attrs is not None and name in field_attrs:
        attrs.update(field_attrs[name])
    return array, dimensions, attrs


def _infer_vertical_size(
    fields: Mapping[str, Any],
    field_dimensions: Mapping[str, Sequence[str]] | None,
    explicit: int | None,
) -> int | None:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("n_vert_levels must be positive")
        return int(explicit)
    for name, raw in fields.items():
        array, dimensions, _ = _field_parts(name, raw, field_dimensions, None)
        if dimensions is None:
            continue
        for axis, dim in enumerate(dimensions):
            if dim == "nVertLevels":
                return int(array.shape[axis])
            if dim == "nVertLevelsP1":
                return int(array.shape[axis]) - 1
            if dim == "nVertLevelsP2":
                return int(array.shape[axis]) - 2
    return None


def _dimension_candidates(
    size: int,
    sizes: Mapping[str, int],
    *,
    exclude: set[str],
) -> list[str]:
    preference = [*_ENTITY_DIMENSIONS, *_VERTICAL_DIMENSIONS]
    preference.extend(name for name in sizes if name not in preference)
    return [name for name in preference if name not in exclude and sizes.get(name) == size]


def _infer_field_dimensions(
    name: str,
    array: NDArray[Any],
    sizes: dict[str, int],
    n_times: int,
) -> tuple[str, ...]:
    shape = array.shape

    known = _KNOWN_FIELD_DIMENSIONS.get(name)
    if known is not None:
        candidates = [known]
        if len(known) > 1:
            candidates.append(tuple(reversed(known)))
        for candidate in candidates:
            if len(candidate) != array.ndim:
                continue
            compatible = True
            inferred_vertical: dict[str, int] = {}
            for axis, dim in enumerate(candidate):
                size = int(shape[axis])
                if dim in _ENTITY_DIMENSIONS and sizes.get(dim) != size:
                    compatible = False
                    break
                if dim == "nVertLevels":
                    inferred_vertical[dim] = size
                elif dim == "nVertLevelsP1":
                    inferred_vertical["nVertLevels"] = size - 1
                elif dim == "nVertLevelsP2":
                    inferred_vertical["nVertLevels"] = size - 2
            if not compatible:
                continue
            if "nVertLevels" in inferred_vertical:
                nlev = inferred_vertical["nVertLevels"]
                expected = sizes.get("nVertLevels")
                if nlev <= 0 or (expected is not None and expected != nlev):
                    continue
                sizes.setdefault("nVertLevels", nlev)
                sizes.setdefault("nVertLevelsP1", nlev + 1)
                sizes.setdefault("nVertLevelsP2", nlev + 2)
            return candidate

    dimensions: list[str | None] = [None] * array.ndim
    used: set[str] = set()

    # Multiple records are unambiguous when the first axis equals Time.  A
    # single record is intentionally not guessed from an arbitrary size-one
    # axis; callers can name it explicitly.
    if n_times > 1 and shape and shape[0] == n_times:
        dimensions[0] = "Time"
        used.add("Time")

    for axis, size in enumerate(shape):
        if dimensions[axis] is not None:
            continue
        entity = [dim for dim in _ENTITY_DIMENSIONS if sizes.get(dim) == size]
        if len(entity) == 1 and entity[0] not in used:
            dimensions[axis] = entity[0]
            used.add(entity[0])

    unresolved = [axis for axis, dim in enumerate(dimensions) if dim is None]
    # A single non-entity dimension on a horizontally located field is the
    # vertical axis.  This is the common MPAS (level, entity) authority layout.
    has_entity = any(dim in _ENTITY_DIMENSIONS for dim in dimensions)
    if has_entity and len(unresolved) == 1:
        axis = unresolved.pop()
        size = shape[axis]
        vertical = [dim for dim in _VERTICAL_DIMENSIONS if sizes.get(dim) == size]
        if len(vertical) == 1:
            dimensions[axis] = vertical[0]
            used.add(vertical[0])
        elif "nVertLevels" not in sizes:
            sizes["nVertLevels"] = size
            sizes["nVertLevelsP1"] = size + 1
            sizes["nVertLevelsP2"] = size + 2
            dimensions[axis] = "nVertLevels"
            used.add("nVertLevels")

    for axis, dim in enumerate(dimensions):
        if dim is not None:
            continue
        candidates = _dimension_candidates(shape[axis], sizes, exclude=used)
        if len(candidates) == 1:
            dimensions[axis] = candidates[0]
            used.add(candidates[0])
            continue
        generated = f"{name}_dim_{axis}"
        sizes.setdefault(generated, int(shape[axis]))
        dimensions[axis] = generated
        used.add(generated)
    return tuple(str(dim) for dim in dimensions)


def _canonical_dimension_order(dimensions: Sequence[str]) -> tuple[str, ...]:
    dims = tuple(dimensions)
    result: list[str] = []
    if "Time" in dims:
        result.append("Time")
    result.extend(dim for dim in _ENTITY_DIMENSIONS if dim in dims)
    result.extend(dim for dim in _VERTICAL_DIMENSIONS if dim in dims)
    result.extend(dim for dim in dims if dim not in result)
    return tuple(result)


def _prepare_field(
    name: str,
    array: NDArray[Any],
    dimensions: tuple[str, ...] | None,
    sizes: dict[str, int],
    n_times: int,
) -> tuple[NDArray[Any], tuple[str, ...]]:
    if array.ndim == 0:
        array = array.reshape(())
    if dimensions is None:
        dimensions = _infer_field_dimensions(name, array, sizes, n_times)
    if len(dimensions) != array.ndim:
        raise ValueError(
            f"{name}: {len(dimensions)} dimension names for shape {array.shape}"
        )
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(f"{name}: repeated dimension names are not supported: {dimensions}")
    for axis, dim in enumerate(dimensions):
        size = int(array.shape[axis])
        if dim == "Time":
            if size != n_times:
                raise ValueError(f"{name}: Time axis has {size} records, expected {n_times}")
            continue
        existing = sizes.get(dim)
        if existing is not None and existing != size:
            raise ValueError(f"{name}: dimension {dim} has {size}, expected {existing}")
        sizes[dim] = size

    if "Time" not in dimensions:
        array = np.broadcast_to(array, (n_times, *array.shape))
        dimensions = ("Time", *dimensions)

    disk_dimensions = _canonical_dimension_order(dimensions)
    permutation = tuple(dimensions.index(dim) for dim in disk_dimensions)
    if permutation != tuple(range(array.ndim)):
        array = np.transpose(array, permutation)
    return np.asarray(array), disk_dimensions


def _disk_dtype(array: NDArray[Any], precision: str, netcdf_format: str) -> np.dtype[Any]:
    dtype = np.dtype(array.dtype)
    if dtype.kind == "f":
        if precision == "single":
            return np.dtype("float32")
        if precision == "double":
            return np.dtype("float64")
        if dtype.itemsize not in (4, 8):
            raise TypeError(f"only float32/float64 history fields are supported, got {dtype}")
        return dtype
    if dtype.kind == "b":
        return np.dtype("int8")
    if dtype.kind in "iu":
        if netcdf_format.startswith("NETCDF3") and dtype.itemsize > 4:
            info = np.iinfo(np.int32)
            if array.size and (np.min(array) < info.min or np.max(array) > info.max):
                raise OverflowError("integer field cannot be represented in serial NetCDF int32")
            return np.dtype("int32")
        return dtype
    raise TypeError(f"unsupported NetCDF field dtype {dtype}")


def _create_data_variable(
    dataset: Dataset,
    name: str,
    data: NDArray[Any],
    dimensions: tuple[str, ...],
    attrs: Mapping[str, Any],
    *,
    precision: str,
) -> None:
    dtype = _disk_dtype(data, precision, dataset.data_model)
    fill_value = attrs.get("_FillValue")
    kwargs: dict[str, Any] = {}
    if fill_value is not None:
        kwargs["fill_value"] = np.asarray(fill_value, dtype=dtype).item()
    variable = dataset.createVariable(name, dtype, dimensions, **kwargs)
    _set_attrs(variable, attrs)
    variable[...] = np.asarray(data, dtype=dtype)


def _mesh_dims_for_array(
    name: str,
    array: NDArray[Any],
    supplied: Mapping[str, tuple[str, ...]],
    sizes: dict[str, int],
) -> tuple[str, ...]:
    if name in supplied:
        dims = tuple(supplied[name])
        if len(dims) == array.ndim and all(sizes.get(d, array.shape[i]) == array.shape[i] for i, d in enumerate(dims)):
            return dims
    inferred: list[str] = []
    used: set[str] = set()
    for axis, size in enumerate(array.shape):
        candidates = _dimension_candidates(int(size), sizes, exclude=used)
        if not candidates:
            dim = f"{name}_dim_{axis}"
            sizes[dim] = int(size)
        else:
            dim = candidates[0]
        inferred.append(dim)
        used.add(dim)
    return tuple(inferred)


def _write_char_array(variable: Any, values: Sequence[str]) -> None:
    result = np.full((len(values), _STR_LEN), b" ", dtype="S1")
    for row, value in enumerate(values):
        encoded = value.encode("ascii")
        if len(encoded) > _STR_LEN:
            raise ValueError(f"MPAS timestamp exceeds StrLen={_STR_LEN}: {value!r}")
        result[row, : len(encoded)] = np.frombuffer(encoded, dtype="S1")
    variable[...] = result


def write_history(
    path: str | Path,
    mesh: object,
    fields: Mapping[str, Any],
    valid_time: datetime | str | np.datetime64 | Sequence[datetime | str | np.datetime64],
    *,
    initial_time: datetime | str | np.datetime64 | None = None,
    time_seconds: float | Sequence[float] | None = None,
    n_vert_levels: int | None = None,
    field_dimensions: Mapping[str, Sequence[str]] | None = None,
    field_attrs: Mapping[str, Mapping[str, Any]] | None = None,
    global_attrs: Mapping[str, Any] | None = None,
    include_mesh: bool = True,
    stream_options: HistoryStreamOptions | Mapping[str, Any] | None = None,
) -> Path:
    """Write one MPAS-compatible serial atmosphere history file.

    Float32 and float64 fields retain their dtype under the default
    ``precision='native'``.  Mesh connectivity is converted back from the
    port's zero-based/``-1`` authority form to MPAS's one-based/zero-padding
    disk convention.
    """

    options = (
        stream_options
        if isinstance(stream_options, HistoryStreamOptions)
        else HistoryStreamOptions.from_mapping(stream_options)
    ).validate()
    destination = Path(path).expanduser().resolve()
    if destination.exists() and options.clobber_mode == "never_modify":
        raise FileExistsError(
            f"clobber_mode='never_modify' refuses existing history file {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    times = _time_values(valid_time)
    reference = _as_datetime(initial_time) if initial_time is not None else times[0]
    if time_seconds is None:
        seconds = np.asarray([(value - reference).total_seconds() for value in times], dtype=np.float64)
    elif np.isscalar(time_seconds):
        if len(times) != 1:
            raise ValueError("scalar time_seconds is only valid for one history record")
        seconds = np.asarray([time_seconds], dtype=np.float64)
    else:
        seconds = np.asarray(tuple(time_seconds), dtype=np.float64)
        if seconds.shape != (len(times),):
            raise ValueError(f"time_seconds shape {seconds.shape} != ({len(times)},)")
    if not np.all(np.isfinite(seconds)):
        raise ValueError("time_seconds must be finite")

    arrays = _mesh_arrays(mesh)
    mesh_dims = _mesh_dimensions(mesh)
    variable_dims = _mesh_variable_dimensions(mesh)
    variable_attrs = _mesh_variable_attrs(mesh)
    sizes = {name: size for name, size in mesh_dims.items() if name not in {"Time", "StrLen"}}
    inferred_nlev = _infer_vertical_size(fields, field_dimensions, n_vert_levels)
    if inferred_nlev is not None:
        vertical = {
            "nVertLevels": inferred_nlev,
            "nVertLevelsP1": inferred_nlev + 1,
            "nVertLevelsP2": inferred_nlev + 2,
        }
        for name, size in vertical.items():
            if name in sizes and sizes[name] != size:
                raise ValueError(f"mesh {name}={sizes[name]} disagrees with requested {size}")
            sizes[name] = size

    prepared: dict[str, tuple[NDArray[Any], tuple[str, ...], dict[str, Any]]] = {}
    reserved = {"Time", "xtime", "initial_time"}
    for name, raw in fields.items():
        if name in reserved or (include_mesh and name in HISTORY_MESH_VARIABLES):
            raise ValueError(f"history field name {name!r} is reserved")
        array, dims, attrs = _field_parts(name, raw, field_dimensions, field_attrs)
        data, disk_dims = _prepare_field(name, array, dims, sizes, len(times))
        metadata = dict(_FIELD_METADATA.get(name, {}))
        metadata.update(attrs)
        prepared[name] = (data, disk_dims, metadata)

    selected_mesh: dict[str, tuple[NDArray[Any], tuple[str, ...], Mapping[str, Any]]] = {}
    if include_mesh:
        for name in HISTORY_MESH_VARIABLES:
            if name not in arrays:
                continue
            data = np.asarray(arrays[name])
            dims = _mesh_dims_for_array(name, data, variable_dims, sizes)
            for axis, dim in enumerate(dims):
                expected = sizes.get(dim)
                if expected is not None and expected != data.shape[axis]:
                    raise ValueError(
                        f"mesh variable {name} dimension {dim} has {data.shape[axis]}, expected {expected}"
                    )
                sizes[dim] = int(data.shape[axis])
            selected_mesh[name] = (data, dims, variable_attrs.get(name, {}))

    mode_clobber = options.clobber_mode in {"truncate", "replace_files"}
    if destination.exists() and not mode_clobber:
        raise FileExistsError(destination)

    with Dataset(destination, "w", format=options.netcdf_format) as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("StrLen", _STR_LEN)
        used_dimensions: set[str] = set()
        for _, dims, _ in selected_mesh.values():
            used_dimensions.update(dims)
        for _, dims, _ in prepared.values():
            used_dimensions.update(dim for dim in dims if dim != "Time")
        # Core horizontal and requested vertical dimensions are useful to MPAS
        # readers even if a particular snapshot has no field on one of them.
        used_dimensions.update(name for name in (*_ENTITY_DIMENSIONS, *_VERTICAL_DIMENSIONS) if name in sizes)
        for name in sizes:
            if name in used_dimensions and name not in dataset.dimensions:
                dataset.createDimension(name, sizes[name])

        inherited = dict(getattr(mesh, "attrs", {}))
        inherited.pop("_NCProperties", None)
        inherited.update(
            {
                "Conventions": "CF-1.10",
                "source": "mpas-port serial atmosphere history writer",
                "mpas_port_evidence": "implemented-unverified",
                "mpas_port_frozen_source": _FROZEN_OUTPUT_SOURCE,
                "stream_name": "output",
            }
        )
        if global_attrs:
            inherited.update(global_attrs)
        _set_attrs(dataset, inherited)

        initial = dataset.createVariable("initial_time", "S1", ("StrLen",))
        _set_attrs(
            initial,
            {"units": "YYYY-MM-DD_hh:mm:ss", "long_name": "Model initialization time"},
        )
        initial_chars = np.full(_STR_LEN, b" ", dtype="S1")
        encoded_initial = _mpas_time_string(reference).encode("ascii")
        initial_chars[: len(encoded_initial)] = np.frombuffer(encoded_initial, dtype="S1")
        initial[:] = initial_chars

        xtime = dataset.createVariable("xtime", "S1", ("Time", "StrLen"))
        _set_attrs(
            xtime,
            {"units": "YYYY-MM-DD_hh:mm:ss", "long_name": "Model valid time"},
        )
        _write_char_array(xtime, [_mpas_time_string(value) for value in times])

        time_var = dataset.createVariable("Time", "f8", ("Time",))
        _set_attrs(
            time_var,
            {
                "units": f"seconds since {reference:%Y-%m-%d %H:%M:%S}",
                "long_name": "CF-compliant valid time",
                "standard_name": "time",
                "calendar": "gregorian",
            },
        )
        time_var[:] = seconds

        converted = set(getattr(mesh, "converted_connectivity", ())) | _CONNECTIVITY_VARIABLES
        for name, (data, dims, attrs) in selected_mesh.items():
            disk_data = data
            if name in converted:
                # In-memory: 0..N-1 and -1 padding.  On disk: 1..N and 0.
                disk_data = np.asarray(data, dtype=np.int64) + 1
                if disk_data.size and (
                    np.min(disk_data) < np.iinfo(np.int32).min
                    or np.max(disk_data) > np.iinfo(np.int32).max
                ):
                    raise OverflowError(f"mesh connectivity {name} exceeds MPAS int32")
                disk_data = disk_data.astype(np.int32)
            _create_data_variable(
                dataset,
                name,
                disk_data,
                dims,
                attrs,
                precision=options.precision,
            )

        for name, (data, dims, attrs) in prepared.items():
            _create_data_variable(
                dataset,
                name,
                data,
                dims,
                attrs,
                precision=options.precision,
            )

    return destination


def write_mpas_history(*args: Any, **kwargs: Any) -> Path:
    """Compatibility spelling for :func:`write_history`."""

    return write_history(*args, **kwargs)


write_history_file = write_mpas_history


__all__ = [
    "Field",
    "HISTORY_MESH_VARIABLES",
    "HistoryField",
    "HistoryStreamOptions",
    "format_mpas_filename",
    "write_history",
    "write_history_file",
    "write_mpas_history",
]
