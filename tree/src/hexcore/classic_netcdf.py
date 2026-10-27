"""Read whole numeric variables of a classic (CDF-1/2/5) netCDF file on several threads.

THE COST THIS REMOVES (30 km globe, x1.655362, V5090 box, 2026-10-27): ``Mesh.from_netcdf`` read every grid and
init variable through netCDF4-python one after another on the forecast's main thread (netCDF-C is not thread safe),
about 13 s of each 30 km member start while four cards waited.  A classic file stores each variable's words
big-endian at an offset its header names, so the words can be read straight from those offsets on many threads
and byte-swapped there (os.preadv and the NumPy cast release the GIL).

The arrays are the ones netCDF4-python returns for the same variable with masking off: the same native-endian
dtype, the same shape (a record variable of a one-record file keeps its leading record axis) and the same words
(the swap is exact).  Anything this reader does not handle the same way returns None for that variable, and the
caller reads it through netCDF4 as before: character data, 0-d variables, attributes netCDF4 acts on
(``scale_factor``, ``add_offset``, ``_Unsigned``), and record variables of a file with other than one record.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

import numpy as np

_TYPES = {1: "i1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8", 7: "u1", 8: ">u2", 9: ">u4", 10: ">i8", 11: ">u8"}
_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8, 7: 1, 8: 2, 9: 4, 10: 8, 11: 8}
_ACTED_ON = ("scale_factor", "add_offset", "_Unsigned")
_CHUNK = 1 << 30


@dataclass(frozen=True)
class Extent:
    """Where one variable's words lie: ``count`` values of big-endian ``dtype`` from byte ``begin``."""

    dtype: str | None
    shape: tuple[int, ...]
    begin: int
    record: bool


class _Header:
    def __init__(self, data: bytes, version: int) -> None:
        self.data, self.offset, self.version = data, 0, version

    def take(self, size: int) -> bytes:
        if self.offset + size > len(self.data):
            raise EOFError
        value = self.data[self.offset:self.offset + size]
        self.offset += size
        return value

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def count(self) -> int:
        """NON_NEG / nelems: 8 bytes in CDF-5, 4 before."""
        return struct.unpack(">Q", self.take(8))[0] if self.version == 5 else self.u32()

    def name(self) -> str:
        size = self.count()
        text = self.take(size)
        self.take(-size % 4)
        return text.decode("utf-8")

    def attributes(self) -> None:
        tag, count = self.u32(), self.count()
        if tag == 0:
            return
        for _ in range(count):
            self.name()
            kind, values = self.u32(), self.count()
            size = _SIZES[kind] * values
            self.take(size + (-size % 4))


def extents(path: str | Path) -> tuple[dict[str, Extent], int] | None:
    """Every variable's extent and the record count, or None when ``path`` is not a classic netCDF file."""

    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic[:3] != b"CDF" or magic[3] not in (1, 2, 5):
            return None
        version = magic[3]
        # The header holds names, attributes and offsets only; read it in growing pieces until it parses.
        size = 1 << 20
        while True:
            handle.seek(0)
            data = handle.read(size)
            try:
                return _parse(data, version)
            except EOFError:
                if len(data) < size:
                    return None
                size *= 4


def _parse(data: bytes, version: int) -> tuple[dict[str, Extent], int]:
    header = _Header(data, version)
    header.take(4)
    numrecs = header.count()
    dimensions: list[int] = []
    tag, count = header.u32(), header.count()
    if tag not in (0, 0x0A):
        raise ValueError("classic netCDF dimension list tag")
    for _ in range(count if tag else 0):
        header.name()
        dimensions.append(header.count())
    header.attributes()
    tag, count = header.u32(), header.count()
    if tag not in (0, 0x0B):
        raise ValueError("classic netCDF variable list tag")
    found: dict[str, Extent] = {}
    for _ in range(count if tag else 0):
        name = header.name()
        ids = [header.count() for _ in range(header.count())]
        header.attributes()
        kind = header.u32()
        header.count() if version == 5 else header.u32()  # vsize
        begin = struct.unpack(">Q" if version in (2, 5) else ">I", header.take(8 if version in (2, 5) else 4))[0]
        lengths = [dimensions[i] for i in ids]
        record = bool(lengths) and lengths[0] == 0
        shape = tuple([numrecs] + lengths[1:]) if record else tuple(lengths)
        found[name] = Extent(_TYPES.get(kind), shape, begin, record)
    return found, numrecs


def readable(extent: Extent, numrecs: int, attributes: Iterable[str]) -> bool:
    """Whether this reader returns netCDF4-python's array for the variable (see the module docstring)."""

    return (extent.dtype is not None and len(extent.shape) > 0
            and not (extent.record and numrecs != 1)
            and not any(name in _ACTED_ON for name in attributes))


def _read(path: Path, extent: Extent) -> np.ndarray:
    words = np.empty(extent.shape, dtype=extent.dtype)
    view = memoryview(words.reshape(-1).view(np.uint8))
    with path.open("rb", buffering=0) as handle:
        fd = handle.fileno()
        done = 0
        while done < len(view):
            got = os.preadv(fd, [view[done:done + _CHUNK]], extent.begin + done)
            if got <= 0:
                raise EOFError(f"{path.name}: short read of a variable")
            done += got
    return words.astype(words.dtype.newbyteorder("="))


def read_variables(path: str | Path, names: Iterable[str], table: Mapping[str, Extent], *,
                   workers: int) -> dict[str, Any]:
    """The named variables' arrays, read on ``workers`` threads, keyed by name."""

    path = Path(path)
    names = list(names)
    if workers <= 1 or len(names) <= 1:
        return {name: _read(path, table[name]) for name in names}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hex-classic-read") as pool:
        futures = {name: pool.submit(_read, path, table[name]) for name in names}
        return {name: future.result() for name, future in futures.items()}
