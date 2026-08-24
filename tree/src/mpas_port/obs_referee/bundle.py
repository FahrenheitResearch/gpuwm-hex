"""Canonical normalized observation/model bundles.

Raw MRMS GRIB2 and raw ASOS/METAR are deliberately outside this module.  A
rustwx producer (or a synthetic test producer) materializes the stable formats
described here and signs each artifact with a receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import zipfile

import numpy as np

from .canonical import (
    canonical_json_bytes,
    read_json,
    require_sha256,
    resolve_path,
    sha256_file,
    write_json,
)
from .errors import DataError, IntegrityError, MeasurementUnavailable, SchemaError
from .manifest import Manifest


GRID_SCHEMA = "gpuwm-hex.canonical-grid/v1"
STATION_SCHEMA = "gpuwm-hex.canonical-stations/v1"
RECEIPT_SCHEMA = "gpuwm-hex.normalized-artifact-receipt/v1"


@dataclass(frozen=True, slots=True)
class GridBundle:
    artifact_path: Path
    artifact_sha256: str
    producer: str
    producer_version: str
    time_unix_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    fields: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.latitude_deg.shape)


@dataclass(frozen=True, slots=True)
class StationBundle:
    artifact_path: Path
    artifact_sha256: str
    producer: str
    producer_version: str
    station_id: np.ndarray
    time_unix_s: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray
    fields: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


def load_source(
    manifest: Manifest,
    source: Mapping[str, Any],
    *,
    expected_kind: str | None = None,
) -> GridBundle | StationBundle:
    adapter = str(source["adapter"])
    path = resolve_path(
        str(source["path"]),
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    if not path.exists():
        if bool(source.get("optional", False)):
            raise MeasurementUnavailable(f"optional source is absent: {source['path']}")
        raise DataError(f"required source is absent: {source['path']}")
    explicit_digest = source.get("sha256")
    if explicit_digest:
        require_sha256(path, str(explicit_digest))

    if adapter == "canonical-grid-v1":
        if expected_kind not in (None, "grid"):
            raise SchemaError(f"source {source['path']} is a grid, expected {expected_kind}")
        return load_grid_bundle(manifest, source, path=path)
    if adapter == "canonical-stations-v1":
        if expected_kind not in (None, "stations"):
            raise SchemaError(f"source {source['path']} is stations, expected {expected_kind}")
        return load_station_bundle(manifest, source, path=path)
    if adapter == "mpas-netcdf-v1":
        if expected_kind not in (None, "grid"):
            raise SchemaError(f"source {source['path']} is a grid, expected {expected_kind}")
        return load_mpas_netcdf_bundle(manifest, source, path=path)
    raise SchemaError(f"unsupported source adapter {adapter!r}")


def load_grid_bundle(
    manifest: Manifest,
    source: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> GridBundle:
    artifact = path or resolve_path(
        str(source["path"]),
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    receipt = _load_and_validate_receipt(manifest, source, artifact, expected_kind="grid")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            names = set(archive.files)
            required = {"schema_utf8", "time_unix_s", "latitude_deg", "longitude_deg"}
            missing = required - names
            if missing:
                raise DataError(f"{artifact.name} missing canonical grid arrays {sorted(missing)}")
            schema = _decode_utf8_array(archive["schema_utf8"], name="schema_utf8")
            if schema != GRID_SCHEMA:
                raise DataError(f"{artifact.name} schema is {schema!r}, expected {GRID_SCHEMA!r}")
            times = _readonly_array(archive["time_unix_s"], np.int64, name="time_unix_s")
            lat = _readonly_array(archive["latitude_deg"], np.float64, name="latitude_deg")
            lon = _readonly_array(archive["longitude_deg"], np.float64, name="longitude_deg")
            metadata = {}
            if "metadata_utf8" in names:
                metadata_text = _decode_utf8_array(archive["metadata_utf8"], name="metadata_utf8")
                try:
                    parsed = json.loads(metadata_text)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{artifact.name} metadata_utf8 is invalid JSON: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise DataError(f"{artifact.name} metadata must be an object")
                metadata = parsed
            fields: dict[str, np.ndarray] = {}
            for key in sorted(names):
                if not key.startswith("field__"):
                    continue
                field_name = key[len("field__") :]
                _validate_field_name(field_name)
                fields[field_name] = _readonly_array(
                    archive[key], np.float64, name=key
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise DataError(f"cannot read canonical grid bundle {artifact}: {exc}") from exc

    _validate_grid_arrays(times, lat, lon, fields, artifact.name)
    return GridBundle(
        artifact_path=artifact,
        artifact_sha256=sha256_file(artifact),
        producer=str(receipt["producer"]),
        producer_version=str(receipt["producer_version"]),
        time_unix_s=times,
        latitude_deg=lat,
        longitude_deg=lon,
        fields=MappingProxyType(fields),
        metadata=MappingProxyType(metadata),
    )


def load_station_bundle(
    manifest: Manifest,
    source: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> StationBundle:
    artifact = path or resolve_path(
        str(source["path"]),
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    receipt = _load_and_validate_receipt(manifest, source, artifact, expected_kind="stations")
    station_id: list[str] = []
    times: list[int] = []
    latitudes: list[float] = []
    longitudes: list[float] = []
    rows: list[dict[str, float | None]] = []
    metadata: dict[str, Any] = {}
    field_names: set[str] = set()
    try:
        with artifact.open("r", encoding="utf-8", newline="") as handle:
            header = handle.readline()
            if not header:
                raise DataError(f"{artifact.name} is empty")
            try:
                header_value = json.loads(header)
            except json.JSONDecodeError as exc:
                raise DataError(f"{artifact.name} first line is invalid JSON: {exc}") from exc
            if not isinstance(header_value, dict) or header_value.get("schema") != STATION_SCHEMA:
                raise DataError(
                    f"{artifact.name} first line must declare schema {STATION_SCHEMA!r}"
                )
            header_metadata = header_value.get("metadata", {})
            if not isinstance(header_metadata, dict):
                raise DataError(f"{artifact.name} header metadata must be an object")
            metadata = header_metadata
            for line_number, line in enumerate(handle, 2):
                if not line.strip():
                    raise DataError(f"{artifact.name}:{line_number}: blank lines are forbidden")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(
                        f"{artifact.name}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise DataError(f"{artifact.name}:{line_number}: record must be an object")
                required = {
                    "station_id",
                    "time_unix_s",
                    "latitude_deg",
                    "longitude_deg",
                    "fields",
                }
                if set(record) != required:
                    raise DataError(
                        f"{artifact.name}:{line_number}: keys must be exactly {sorted(required)}"
                    )
                sid = record["station_id"]
                if not isinstance(sid, str) or not sid:
                    raise DataError(f"{artifact.name}:{line_number}: invalid station_id")
                field_map = record["fields"]
                if not isinstance(field_map, dict):
                    raise DataError(f"{artifact.name}:{line_number}: fields must be an object")
                normalized_fields: dict[str, float | None] = {}
                for name, value in field_map.items():
                    _validate_field_name(name)
                    field_names.add(name)
                    if value is None:
                        normalized_fields[name] = None
                    elif isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise DataError(
                            f"{artifact.name}:{line_number}: field {name!r} is not numeric/null"
                        )
                    elif not math.isfinite(float(value)):
                        raise DataError(
                            f"{artifact.name}:{line_number}: field {name!r} is non-finite"
                        )
                    else:
                        normalized_fields[name] = float(value)
                station_id.append(sid)
                times.append(_int_value(record["time_unix_s"], "time_unix_s"))
                latitudes.append(_latitude(record["latitude_deg"]))
                longitudes.append(_longitude(record["longitude_deg"]))
                rows.append(normalized_fields)
    except OSError as exc:
        raise DataError(f"cannot read canonical station bundle {artifact}: {exc}") from exc

    if not rows:
        raise DataError(f"{artifact.name} contains no station records")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            station_id[index],
            times[index],
            latitudes[index],
            longitudes[index],
            index,
        ),
    )
    fields: dict[str, np.ndarray] = {}
    for field_name in sorted(field_names):
        values = np.full(len(rows), np.nan, dtype=np.float64)
        for output_index, input_index in enumerate(order):
            value = rows[input_index].get(field_name)
            if value is not None:
                values[output_index] = value
        values.setflags(write=False)
        fields[field_name] = values
    ids = np.asarray([station_id[index] for index in order], dtype=np.str_)
    t = np.asarray([times[index] for index in order], dtype=np.int64)
    lat = np.asarray([latitudes[index] for index in order], dtype=np.float64)
    lon = np.asarray([longitudes[index] for index in order], dtype=np.float64)
    for array in (ids, t, lat, lon):
        array.setflags(write=False)
    return StationBundle(
        artifact_path=artifact,
        artifact_sha256=sha256_file(artifact),
        producer=str(receipt["producer"]),
        producer_version=str(receipt["producer_version"]),
        station_id=ids,
        time_unix_s=t,
        latitude_deg=lat,
        longitude_deg=lon,
        fields=MappingProxyType(fields),
        metadata=MappingProxyType(metadata),
    )


def load_mpas_netcdf_bundle(
    manifest: Manifest,
    source: Mapping[str, Any],
    *,
    path: Path,
) -> GridBundle:
    """Load configured MPAS/gpuwm-hex output without guessing variable names.

    All variable mappings are manifest-owned. Longitude/latitude can be radians
    or degrees. Fields may be direct variables or explicit sums. No diagnostic
    arithmetic is silently invented here.
    """

    receipt = _load_and_validate_receipt(manifest, source, path, expected_kind="grid")
    options = dict(source.get("options", {}))
    required = ("time_variable", "latitude_variable", "longitude_variable", "fields")
    missing = [name for name in required if name not in options]
    if missing:
        raise SchemaError(f"mpas-netcdf-v1 source options missing {missing}")
    field_specs = options["fields"]
    if not isinstance(field_specs, dict) or not field_specs:
        raise SchemaError("mpas-netcdf-v1 options.fields must be a non-empty object")
    try:
        from netCDF4 import Dataset, chartostring, num2date
    except ImportError as exc:
        raise MeasurementUnavailable("netCDF4 is required for mpas-netcdf-v1") from exc

    try:
        with Dataset(path, "r") as dataset:
            lat = np.asarray(dataset.variables[str(options["latitude_variable"])][:], dtype=np.float64)
            lon = np.asarray(dataset.variables[str(options["longitude_variable"])][:], dtype=np.float64)
            units = str(options.get("coordinate_units", "radians"))
            if units == "radians":
                lat = np.rad2deg(lat)
                lon = np.rad2deg(lon)
            elif units != "degrees":
                raise SchemaError("coordinate_units must be 'radians' or 'degrees'")
            lon = ((lon + 180.0) % 360.0) - 180.0
            time_variable = dataset.variables[str(options["time_variable"])]
            raw_time = time_variable[:]
            if raw_time.dtype.kind in ("S", "U") or raw_time.ndim == 2:
                text = chartostring(raw_time)
                times = np.asarray(
                    [
                        int(
                            __import__("datetime")
                            .datetime.fromisoformat(str(item).strip().replace("_", "T").replace("Z", "+00:00"))
                            .timestamp()
                        )
                        for item in np.ravel(text)
                    ],
                    dtype=np.int64,
                )
            else:
                units_attr = getattr(time_variable, "units", None)
                if not units_attr:
                    raise DataError("numeric time variable lacks CF units")
                calendar = getattr(time_variable, "calendar", "standard")
                dates = num2date(raw_time, units=units_attr, calendar=calendar)
                times = np.asarray([int(item.timestamp()) for item in np.ravel(dates)], dtype=np.int64)

            fields: dict[str, np.ndarray] = {}
            for canonical_name, spec in sorted(field_specs.items()):
                _validate_field_name(canonical_name)
                if isinstance(spec, str):
                    values = np.asarray(dataset.variables[spec][:], dtype=np.float64)
                elif isinstance(spec, dict) and set(spec) == {"sum"}:
                    names = spec["sum"]
                    if not isinstance(names, list) or not names:
                        raise SchemaError(f"field {canonical_name!r} sum must be a non-empty list")
                    values = sum(
                        (
                            np.asarray(dataset.variables[str(name)][:], dtype=np.float64)
                            for name in names
                        ),
                        start=np.zeros_like(
                            np.asarray(dataset.variables[str(names[0])][:], dtype=np.float64)
                        ),
                    )
                else:
                    raise SchemaError(
                        f"field {canonical_name!r} must be a variable name or {{'sum': [...]}}"
                    )
                fields[canonical_name] = np.asarray(values, dtype=np.float64)
    except KeyError as exc:
        raise DataError(f"configured NetCDF variable is absent in {path.name}: {exc}") from exc
    except OSError as exc:
        raise DataError(f"cannot open MPAS NetCDF {path}: {exc}") from exc

    if lat.shape != lon.shape:
        raise DataError("MPAS latitude and longitude shapes differ")
    _validate_grid_arrays(times, lat, lon, fields, path.name)
    for array in (times, lat, lon, *fields.values()):
        array.setflags(write=False)
    return GridBundle(
        artifact_path=path,
        artifact_sha256=sha256_file(path),
        producer=str(receipt["producer"]),
        producer_version=str(receipt["producer_version"]),
        time_unix_s=times,
        latitude_deg=lat,
        longitude_deg=lon,
        fields=MappingProxyType(fields),
        metadata=MappingProxyType(dict(options.get("metadata", {}))),
    )


def write_grid_bundle(
    path: str | Path,
    *,
    time_unix_s: np.ndarray,
    latitude_deg: np.ndarray,
    longitude_deg: np.ndarray,
    fields: Mapping[str, np.ndarray],
    producer: str,
    producer_version: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a byte-deterministic canonical grid bundle and receipt."""

    artifact = Path(path)
    arrays: dict[str, np.ndarray] = {
        "schema_utf8": np.frombuffer(GRID_SCHEMA.encode("utf-8"), dtype=np.uint8),
        "metadata_utf8": np.frombuffer(
            canonical_json_bytes(dict(metadata or {})).rstrip(b"\n"), dtype=np.uint8
        ),
        "time_unix_s": np.asarray(time_unix_s, dtype=np.int64),
        "latitude_deg": np.asarray(latitude_deg, dtype=np.float64),
        "longitude_deg": np.asarray(longitude_deg, dtype=np.float64),
    }
    for name, values in sorted(fields.items()):
        _validate_field_name(name)
        arrays[f"field__{name}"] = np.asarray(values, dtype=np.float64)
    _validate_grid_arrays(
        arrays["time_unix_s"],
        arrays["latitude_deg"],
        arrays["longitude_deg"],
        {key[len("field__") :]: value for key, value in arrays.items() if key.startswith("field__")},
        artifact.name,
    )
    _write_deterministic_npz(artifact, arrays)
    receipt_path = Path(f"{artifact}.receipt.json")
    write_json(
        receipt_path,
        {
            "schema": RECEIPT_SCHEMA,
            "artifact_kind": "grid",
            "artifact_name": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "producer": producer,
            "producer_version": producer_version,
            "source_artifacts": [],
            "metadata": dict(metadata or {}),
        },
    )
    return artifact, receipt_path


def write_station_bundle(
    path: str | Path,
    *,
    records: list[Mapping[str, Any]],
    producer: str,
    producer_version: str,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise SchemaError("station record must be an object")
        fields = record.get("fields")
        if not isinstance(fields, Mapping):
            raise SchemaError("station record fields must be an object")
        normalized_fields: dict[str, float | None] = {}
        for name, value in sorted(fields.items()):
            _validate_field_name(str(name))
            if value is None:
                normalized_fields[str(name)] = None
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise SchemaError("station values must be finite or null")
                normalized_fields[str(name)] = number
        normalized.append(
            {
                "station_id": str(record["station_id"]),
                "time_unix_s": _int_value(record["time_unix_s"], "time_unix_s"),
                "latitude_deg": _latitude(record["latitude_deg"]),
                "longitude_deg": _longitude(record["longitude_deg"]),
                "fields": normalized_fields,
            }
        )
    normalized.sort(
        key=lambda row: (
            row["station_id"],
            row["time_unix_s"],
            row["latitude_deg"],
            row["longitude_deg"],
        )
    )
    lines = [
        canonical_json_bytes(
            {"schema": STATION_SCHEMA, "metadata": dict(metadata or {})}
        ).rstrip(b"\n")
    ]
    lines.extend(canonical_json_bytes(row).rstrip(b"\n") for row in normalized)
    artifact.write_bytes(b"\n".join(lines) + b"\n")
    receipt_path = Path(f"{artifact}.receipt.json")
    write_json(
        receipt_path,
        {
            "schema": RECEIPT_SCHEMA,
            "artifact_kind": "stations",
            "artifact_name": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "producer": producer,
            "producer_version": producer_version,
            "source_artifacts": [],
            "metadata": dict(metadata or {}),
        },
    )
    return artifact, receipt_path


def _load_and_validate_receipt(
    manifest: Manifest,
    source: Mapping[str, Any],
    artifact: Path,
    *,
    expected_kind: str,
) -> Mapping[str, Any]:
    receipt_raw = source.get("receipt", f"{source['path']}.receipt.json")
    receipt_path = resolve_path(
        str(receipt_raw),
        manifest_dir=manifest.directory,
        variables=dict(manifest.raw.get("path_variables", {})),
    )
    if not receipt_path.exists():
        raise IntegrityError(f"normalization receipt is absent: {receipt_raw}")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict):
        raise SchemaError(f"{receipt_path.name} must contain an object")
    required = {
        "schema",
        "artifact_kind",
        "artifact_name",
        "artifact_sha256",
        "producer",
        "producer_version",
        "source_artifacts",
        "metadata",
    }
    if set(receipt) != required:
        raise SchemaError(
            f"{receipt_path.name} keys must be exactly {sorted(required)}"
        )
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise SchemaError(f"{receipt_path.name} has unsupported schema")
    if receipt["artifact_kind"] != expected_kind:
        raise SchemaError(
            f"{receipt_path.name} describes {receipt['artifact_kind']!r}, expected {expected_kind!r}"
        )
    if receipt["artifact_name"] != artifact.name:
        raise IntegrityError(
            f"{receipt_path.name} artifact_name does not match {artifact.name}"
        )
    require_sha256(artifact, str(receipt["artifact_sha256"]))
    producers = set(manifest.raw.get("producer_allowlist", ("rustwx", "gpuwm-hex")))
    producer = receipt["producer"]
    if producer not in producers:
        raise IntegrityError(
            f"producer {producer!r} is not in manifest.producer_allowlist"
        )
    if manifest.mode == "production" and str(producer).startswith("synthetic"):
        raise IntegrityError("synthetic artifact is forbidden in production mode")
    if not isinstance(receipt["producer_version"], str) or not receipt["producer_version"]:
        raise SchemaError(f"{receipt_path.name} producer_version must be non-empty")
    if not isinstance(receipt["source_artifacts"], list):
        raise SchemaError(f"{receipt_path.name} source_artifacts must be a list")
    if not isinstance(receipt["metadata"], dict):
        raise SchemaError(f"{receipt_path.name} metadata must be an object")
    return receipt


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with zipfile.ZipFile(
            raw,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for name in sorted(arrays):
                buffer = BytesIO()
                np.lib.format.write_array(
                    buffer,
                    np.asarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.create_system = 3
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _readonly_array(value: np.ndarray, dtype: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.dtype.hasobject:
        raise DataError(f"{name} may not contain Python objects")
    array.setflags(write=False)
    return array


def _decode_utf8_array(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise DataError(f"{name} must be a one-dimensional uint8 array")
    try:
        return bytes(array).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataError(f"{name} is not UTF-8") from exc


def _validate_grid_arrays(
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    fields: Mapping[str, np.ndarray],
    artifact_name: str,
) -> None:
    if times.ndim != 1 or times.size == 0:
        raise DataError(f"{artifact_name}: time_unix_s must be non-empty and one-dimensional")
    if not np.all(np.diff(times) > 0):
        raise DataError(f"{artifact_name}: time_unix_s must be strictly increasing")
    if lat.shape != lon.shape or lat.ndim not in (1, 2):
        raise DataError(f"{artifact_name}: latitude/longitude must be matching 1-D or 2-D arrays")
    if lat.size == 0:
        raise DataError(f"{artifact_name}: spatial grid is empty")
    if not np.all(np.isfinite(lat)) or np.any((lat < -90.0) | (lat > 90.0)):
        raise DataError(f"{artifact_name}: latitude is non-finite or outside [-90, 90]")
    if not np.all(np.isfinite(lon)) or np.any((lon < -180.0) | (lon > 180.0)):
        raise DataError(f"{artifact_name}: longitude is non-finite or outside [-180, 180]")
    if not fields:
        raise DataError(f"{artifact_name}: no field__ arrays are present")
    expected = (times.size, *lat.shape)
    for name, values in fields.items():
        _validate_field_name(name)
        if values.shape != expected:
            raise DataError(
                f"{artifact_name}: field {name!r} shape {values.shape} != {expected}"
            )
        if np.any(np.isinf(values)):
            raise DataError(f"{artifact_name}: field {name!r} contains infinity")


def _validate_field_name(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise SchemaError("field name must be a non-empty string")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_")
    if value.lower() != value or any(ch not in allowed for ch in value):
        raise SchemaError(
            f"field name must use lowercase ASCII letters, digits, underscore: {value!r}"
        )


def _int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"{name} must be an integer")
    return value


def _latitude(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or not -90.0 <= number <= 90.0:
        raise DataError(f"invalid latitude {value!r}")
    return number


def _longitude(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number) or not -180.0 <= number <= 180.0:
        raise DataError(f"invalid longitude {value!r}")
    return number
