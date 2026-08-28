"""Reader for the WPS ungrib intermediate format consumed by MPAS init.

This is a direct structural transcription of frozen MPAS-A v8.2.3
``src/core_init_atmosphere/mpas_init_atm_read_met.F:50-407``.  WPS
intermediate files are Fortran sequential-unformatted streams: every logical
record is enclosed by a leading and trailing byte count, and every field is a
five-record group in version 5 (four records in versions 3 and 4).

The reader streams slabs one at a time, so full GFS files do not need to fit in
memory.  Arrays retain the Fortran logical shape ``(nx, ny)`` with x as the
first index.  No interpolation, missing-value substitution, or unit guessing
occurs here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import BinaryIO, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .errors import EvidenceError


_PROJECTION_NAMES = {
    0: "latlon",
    1: "lambert_conformal",
    2: "polar_stereographic",
    3: "mercator",
    4: "gaussian",
}
_RAW_TO_MPAS_PROJECTION = {0: 0, 1: 3, 3: 1, 4: 4, 5: 2}
_LEGACY_EARTH_RADIUS_KM = 6370.0


@dataclass(frozen=True, slots=True)
class Projection:
    """Projection metadata after the frozen MPAS ``_METGRID`` transforms."""

    code: int
    name: str
    start_location: str
    start_latitude: float
    start_longitude: float
    start_i: float
    start_j: float
    delta_latitude: float = 0.0
    delta_longitude: float = 0.0
    dx_m: float = 0.0
    dy_m: float = 0.0
    central_longitude: float = 0.0
    true_latitude_1: float = 0.0
    true_latitude_2: float = 0.0
    earth_radius_km: float = _LEGACY_EARTH_RADIUS_KM


@dataclass(frozen=True, slots=True)
class IntermediateField:
    """One WPS field header and, when requested, its float32 slab."""

    version: int
    valid_time: str
    forecast_hour: float
    map_source: str
    field: str
    units: str
    description: str
    level: float
    nx: int
    ny: int
    projection: Projection
    is_wind_grid_relative: bool
    values: NDArray[np.float32] | None
    record_offset: int

    def inventory_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "version": self.version,
            "valid_time": self.valid_time,
            "forecast_hour": self.forecast_hour,
            "map_source": self.map_source,
            "field": self.field,
            "units": self.units,
            "description": self.description,
            "level": self.level,
            "nx": self.nx,
            "ny": self.ny,
            "projection": asdict(self.projection),
            "is_wind_grid_relative": self.is_wind_grid_relative,
            "record_offset": self.record_offset,
        }
        if self.values is not None:
            finite = np.isfinite(self.values)
            record["data"] = {
                "count": int(self.values.size),
                "finite": int(np.count_nonzero(finite)),
                "min": float(np.min(self.values)) if np.all(finite) else None,
                "max": float(np.max(self.values)) if np.all(finite) else None,
            }
        return record


class _FortranSequential:
    """Fail-closed 32-bit-marker sequential-unformatted stream reader."""

    def __init__(self, handle: BinaryIO, path: Path) -> None:
        self.handle = handle
        self.path = path
        start = handle.read(4)
        if not start:
            raise EvidenceError(f"empty WPS intermediate file: {path}")
        if len(start) != 4:
            raise EvidenceError(f"truncated first record marker in {path}")
        big = struct.unpack(">i", start)[0]
        little = struct.unpack("<i", start)[0]
        if big == 4:
            self.endian = ">"
        elif little == 4:
            self.endian = "<"
        else:
            raise EvidenceError(
                f"{path} is not a 32-bit-marker WPS sequential file: "
                f"first markers big={big}, little={little}"
            )
        handle.seek(0)

    def _marker(self, *, allow_eof: bool) -> int | None:
        raw = self.handle.read(4)
        if not raw and allow_eof:
            return None
        if len(raw) != 4:
            raise EvidenceError(
                f"truncated Fortran record marker at byte {self.handle.tell() - len(raw)} "
                f"in {self.path}"
            )
        length = struct.unpack(f"{self.endian}i", raw)[0]
        if length < 0:
            raise EvidenceError(
                f"negative/subrecord marker {length} at byte {self.handle.tell() - 4} "
                f"in {self.path}; split compiler subrecords are not admitted"
            )
        return length

    def read_record(self, *, allow_eof: bool = False) -> bytes | None:
        offset = self.handle.tell()
        length = self._marker(allow_eof=allow_eof)
        if length is None:
            return None
        payload = self.handle.read(length)
        if len(payload) != length:
            raise EvidenceError(
                f"truncated Fortran record payload at byte {offset} in {self.path}: "
                f"{len(payload)} != {length}"
            )
        trailing = self._marker(allow_eof=False)
        if trailing != length:
            raise EvidenceError(
                f"Fortran record marker mismatch at byte {offset} in {self.path}: "
                f"{length} != {trailing}"
            )
        return payload

    def skip_record(self) -> int:
        offset = self.handle.tell()
        length = self._marker(allow_eof=False)
        assert length is not None
        self.handle.seek(length, 1)
        trailing = self._marker(allow_eof=False)
        if trailing != length:
            raise EvidenceError(
                f"Fortran record marker mismatch at byte {offset} in {self.path}: "
                f"{length} != {trailing}"
            )
        return length


class _Payload:
    def __init__(self, payload: bytes, endian: str, context: str) -> None:
        self.payload = payload
        self.endian = endian
        self.context = context
        self.offset = 0

    def text(self, length: int) -> str:
        end = self.offset + length
        if end > len(self.payload):
            raise EvidenceError(f"truncated {self.context}")
        raw = self.payload[self.offset:end]
        self.offset = end
        try:
            return raw.decode("ascii", errors="strict").rstrip(" \x00")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"non-ASCII text in {self.context}") from error

    def real32(self) -> float:
        return float(self._unpack("f"))

    def int32(self) -> int:
        return int(self._unpack("i"))

    def _unpack(self, code: str) -> int | float:
        end = self.offset + 4
        if end > len(self.payload):
            raise EvidenceError(f"truncated {self.context}")
        value = struct.unpack_from(f"{self.endian}{code}", self.payload, self.offset)[0]
        self.offset = end
        return value

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise EvidenceError(
                f"{self.context} has {len(self.payload) - self.offset} unexpected bytes"
            )


class WpsIntermediateReader:
    """Streaming context manager for WPS versions 3, 4, and 5."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self._handle: BinaryIO | None = None
        self._records: _FortranSequential | None = None

    def __enter__(self) -> "WpsIntermediateReader":
        self._handle = self.path.open("rb")
        self._records = _FortranSequential(self._handle, self.path)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def endian(self) -> str:
        if self._records is None:
            raise RuntimeError("WPS reader is not open")
        return self._records.endian

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._records = None

    def __iter__(self) -> Iterator[IntermediateField]:
        return self.iter_fields()

    def iter_fields(
        self,
        *,
        include: set[str] | None = None,
        load_values: bool = True,
    ) -> Iterator[IntermediateField]:
        """Yield selected fields; excluded slabs are seek-skipped, not loaded."""

        if self._records is None or self._handle is None:
            raise RuntimeError("use WpsIntermediateReader as a context manager")
        selected = None if include is None else {name.strip().upper() for name in include}
        while True:
            record_offset = self._handle.tell()
            raw_version = self._records.read_record(allow_eof=True)
            if raw_version is None:
                return
            if len(raw_version) != 4:
                raise EvidenceError(
                    f"version record at byte {record_offset} in {self.path} has "
                    f"length {len(raw_version)}, expected 4"
                )
            version = struct.unpack(f"{self.endian}i", raw_version)[0]
            if version not in (3, 4, 5):
                raise EvidenceError(
                    f"WPS format version {version} at byte {record_offset} in "
                    f"{self.path}; frozen MPAS admits only 3, 4, or 5"
                )
            metadata_record = self._records.read_record()
            assert metadata_record is not None
            metadata = self._metadata(metadata_record, version, record_offset)
            projection_record = self._records.read_record()
            assert projection_record is not None
            projection = self._projection(
                projection_record,
                version=version,
                raw_code=metadata["raw_projection"],
                nx=metadata["nx"],
                ny=metadata["ny"],
                record_offset=record_offset,
            )
            if version == 5:
                wind_record = self._records.read_record()
                assert wind_record is not None
                if len(wind_record) != 4:
                    raise EvidenceError(
                        f"wind-relative logical at byte {record_offset} in {self.path} "
                        f"has length {len(wind_record)}, expected 4"
                    )
                is_wind_grid_relative = (
                    struct.unpack(f"{self.endian}i", wind_record)[0] != 0
                )
            else:
                is_wind_grid_relative = True

            name = str(metadata["field"])
            wanted = selected is None or name.upper() in selected
            nx = int(metadata["nx"])
            ny = int(metadata["ny"])
            expected_bytes = nx * ny * 4
            if wanted and load_values:
                slab_record = self._records.read_record()
                assert slab_record is not None
                if len(slab_record) != expected_bytes:
                    raise EvidenceError(
                        f"slab {name!r} at byte {record_offset} in {self.path} has "
                        f"{len(slab_record)} bytes, expected {expected_bytes}"
                    )
                values = np.frombuffer(
                    slab_record, dtype=np.dtype(f"{self.endian}f4")
                ).astype(np.float32, copy=True)
                values = values.reshape((nx, ny), order="F")
            else:
                actual_bytes = self._records.skip_record()
                if actual_bytes != expected_bytes:
                    raise EvidenceError(
                        f"slab {name!r} at byte {record_offset} in {self.path} has "
                        f"{actual_bytes} bytes, expected {expected_bytes}"
                    )
                values = None

            if wanted:
                yield IntermediateField(
                    version=version,
                    valid_time=str(metadata["valid_time"]),
                    forecast_hour=float(metadata["forecast_hour"]),
                    map_source=str(metadata["map_source"]),
                    field=name,
                    units=str(metadata["units"]),
                    description=str(metadata["description"]),
                    level=float(metadata["level"]),
                    nx=nx,
                    ny=ny,
                    projection=projection,
                    is_wind_grid_relative=is_wind_grid_relative,
                    values=values,
                    record_offset=record_offset,
                )

    def _metadata(
        self, payload: bytes, version: int, record_offset: int
    ) -> dict[str, str | float | int]:
        parser = _Payload(
            payload, self.endian, f"WPS metadata at byte {record_offset} in {self.path}"
        )
        valid_time = parser.text(24)
        forecast_hour = parser.real32()
        map_source = "" if version == 3 else parser.text(32)
        field = parser.text(9)
        if field == "HGT":
            field = "GHT"
        result: dict[str, str | float | int] = {
            "valid_time": valid_time,
            "forecast_hour": forecast_hour,
            "map_source": map_source,
            "field": field,
            "units": parser.text(25),
            "description": parser.text(46),
            "level": parser.real32(),
            "nx": parser.int32(),
            "ny": parser.int32(),
            "raw_projection": parser.int32(),
        }
        parser.finish()
        if int(result["nx"]) < 1 or int(result["ny"]) < 1:
            raise EvidenceError(
                f"non-positive WPS slab shape at byte {record_offset} in {self.path}"
            )
        return result

    def _projection(
        self,
        payload: bytes,
        *,
        version: int,
        raw_code: int | str | float,
        nx: int | str | float,
        ny: int | str | float,
        record_offset: int,
    ) -> Projection:
        raw = int(raw_code)
        if raw not in _RAW_TO_MPAS_PROJECTION:
            raise EvidenceError(
                f"unrecognized WPS projection code {raw} at byte {record_offset} "
                f"in {self.path}"
            )
        if raw == 4 and version != 5:
            raise EvidenceError(
                f"Gaussian projection code 4 is defined only for WPS version 5, "
                f"not version {version}, at byte {record_offset} in {self.path}"
            )
        code = _RAW_TO_MPAS_PROJECTION[raw]
        parser = _Payload(
            payload, self.endian, f"WPS projection at byte {record_offset} in {self.path}"
        )
        start_location = "SWCORNER" if version == 3 else parser.text(8)
        start_latitude = parser.real32()
        start_longitude = parser.real32()
        delta_latitude = delta_longitude = 0.0
        dx = dy = central_longitude = true_latitude_1 = true_latitude_2 = 0.0
        if raw in (0, 4):
            delta_latitude = parser.real32()
            delta_longitude = parser.real32()
        elif raw == 1:
            dx = parser.real32()
            dy = parser.real32()
            true_latitude_1 = parser.real32()
        elif raw == 3:
            dx = parser.real32()
            dy = parser.real32()
            central_longitude = parser.real32()
            true_latitude_1 = parser.real32()
            true_latitude_2 = parser.real32()
        elif raw == 5:
            dx = parser.real32()
            dy = parser.real32()
            central_longitude = parser.real32()
            true_latitude_1 = parser.real32()
        earth_radius = parser.real32() if version == 5 else _LEGACY_EARTH_RADIUS_KM
        parser.finish()

        if start_location == "CENTER":
            start_i = float(nx) / 2.0
            start_j = float(ny) / 2.0
        elif start_location == "SWCORNER":
            start_i = start_j = 1.0
        else:
            raise EvidenceError(
                f"unrecognized WPS start location {start_location!r} at byte "
                f"{record_offset} in {self.path}"
            )
        if start_longitude > 180.0:
            start_longitude -= 360.0
        start_latitude = min(90.0, max(-90.0, start_latitude))
        if central_longitude > 180.0:
            central_longitude -= 360.0
        return Projection(
            code=code,
            name=_PROJECTION_NAMES[code],
            start_location=start_location,
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            start_i=start_i,
            start_j=start_j,
            delta_latitude=delta_latitude,
            delta_longitude=delta_longitude,
            dx_m=dx * 1000.0,
            dy_m=dy * 1000.0,
            central_longitude=central_longitude,
            true_latitude_1=true_latitude_1,
            true_latitude_2=true_latitude_2,
            earth_radius_km=earth_radius,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(
    path: str | Path, *, include_statistics: bool = False
) -> dict[str, object]:
    """Scan an entire full-file input and return a provenance-ready inventory."""

    source = Path(path).expanduser().resolve(strict=True)
    records: list[dict[str, object]] = []
    with WpsIntermediateReader(source) as reader:
        endian = "big" if reader.endian == ">" else "little"
        for field in reader.iter_fields(load_values=include_statistics):
            records.append(field.inventory_record())
    by_name: dict[str, int] = {}
    for record in records:
        name = str(record["field"])
        by_name[name] = by_name.get(name, 0) + 1
    return {
        "schema": "mpas-port-wps-intermediate-inventory-v1",
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "record_endian": endian,
        "field_records": len(records),
        "fields": dict(sorted(by_name.items())),
        "records": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory a full WPS intermediate file")
    parser.add_argument("path", type=Path)
    parser.add_argument("--statistics", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = inventory(arguments.path, include_statistics=arguments.statistics)
    except (EvidenceError, OSError, ValueError) as error:
        parser.error(str(error))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
