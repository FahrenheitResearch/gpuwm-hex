"""Reading a coarse forecast the way the detector needs it: on the mesh.

MESH-NATIVE, DELIBERATELY.  Nothing here regrids.  A detector that ran on
a regular latitude/longitude image of the forecast would inherit that
image's pole convergence and its interpolation, and the swath it placed
would sit where the PICTURE said rather than where the model said.  The
search is therefore a walk over ``cellsOnCell`` -- the connectivity the
file already carries -- and every distance is a great circle between two
cell centres.

This module is a READER.  It does not interpolate, transform, or write,
which keeps it on the orchestration side of the Python boundary: the data
path that produced this file is Rust and CUDA, and the data path that
consumes what this layer emits is Rust.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import SwathDocumentError, SwathRefusal
from .registry import FieldRow

#: MPAS writes ``cellsOnCell`` one-based with 0 meaning "no neighbour"
#: (only reachable on a culled regional mesh).  The port's own authority
#: form is zero-based with -1, and :func:`hexcore.mesh` states the same
#: convention; it is restated here because this reader deliberately does
#: not build a full ``Mesh`` -- a history file is not a grid pair and
#: validating it as one would refuse a legitimate file.
NO_NEIGHBOUR = -1

#: The mesh a mesh-native search walks.  A file that carries none of this
#: is an image of a forecast, not a forecast on its grid.
MESH_VARIABLES = ("latCell", "lonCell", "areaCell", "cellsOnCell", "nEdgesOnCell")

#: Where a run receipt written by ``gpuwm-hex forecast`` keeps the three
#: things this reader needs.  They are named as constants so a receipt
#: whose shape drifts fails here BY NAME, with the key path in the
#: refusal, rather than as a ``KeyError`` three frames later.
_RECEIPT_FRAME_FILES = ("forecast", "snapshot_files")
_RECEIPT_FRAME_LABELS = ("forecast", "history_labels")
_RECEIPT_GRID_FILE = ("forecast", "authority", "files", "grid")
#: Preferred over the grid: an MPAS static file carries the same
#: connectivity AND a real ``sphere_radius``, which the grid stream does
#: not.  See :meth:`HistoryReader.areas_km2`.
_RECEIPT_STATIC_FILE = ("forecast", "authority", "files", "static")

#: Below this, ``sphere_radius`` is a non-dimensionalisation rather than a
#: length: MPAS grid files are written on the UNIT sphere and carry
#: ``sphere_radius = 1.0``.
DIMENSIONAL_SPHERE_RADIUS_M = 1000.0

#: The two stamp spellings this project writes.  MPAS ``xtime`` uses
#: colons; the forecast door's history LABELS use dots, because a colon is
#: not a filename character on every platform the wheel installs on.
_STAMP_FORMATS = ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d_%H.%M.%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_at(receipt: Any, keys: Sequence[str], path: Path) -> Any:
    """One nested key path out of a run receipt, or a refusal naming it."""

    node = receipt
    for depth, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            raise SwathRefusal(
                f"{path.name} has no '{'.'.join(keys[: depth + 1])}', so it cannot "
                f"supply '{'.'.join(keys)}'. A '.json' "
                "history argument is read as the run receipt "
                "'gpuwm-hex forecast' writes beside its frames, and that receipt "
                "must name the frame files, their valid times and the grid the "
                "run was bound to. This file names no such thing, so it is either "
                "a different document or a receipt whose shape has moved"
            )
        node = node[key]
    return node


@dataclass(frozen=True)
class HistoryFrame:
    index: int
    time_seconds: float
    valid_time: str


class HistoryReader:
    """A coarse forecast, read on the mesh it was written on.

    TWO INPUT SHAPES, ONE READER, AND THE SECOND ONE IS WHY THIS CLASS
    EXISTS IN THIS FORM.  A self-describing history file -- an MPAS-A
    output, or this layer's fixture -- carries its own mesh and all of its
    frames on one ``Time`` axis, and is opened directly.

    THIS PROJECT'S OWN forecast door writes something else.  It writes one
    file per frame, each with ``Time`` of 1, carrying ``latCell`` and
    ``lonCell`` but no ``areaCell``, no ``cellsOnCell``, no
    ``nEdgesOnCell`` and no ``xtime``.  The shipped reader refused it by
    name -- "history file cuda-history.2026-08-12_06.00.00.nc has no
    'areaCell'" -- which meant the placement layer could not read the only
    forecast this project produces.  That refusal is correct and stays:
    the fix is NOT to add variables to the history stream, because those
    frames' digests are pinned by the execution anchors and one extra
    variable breaks every one of them.

    The fix is that the reader also accepts the RUN RECEIPT the same door
    writes beside those frames.  The receipt already names all three
    missing things: the frame sequence and its SHA-256s
    (``forecast.snapshot_files``), the valid time of each frame
    (``forecast.history_labels``), and the grid file the run was bound to
    (``forecast.authority.files.grid``), which is where the connectivity
    lives.  So the artifact the forecast door writes is exactly the
    artifact the swath door consumes, with no intermediate file, no
    filename parsing and no change to a pinned stream.

    ``grid=`` overrides the mesh source for either shape, for a file whose
    connectivity lives elsewhere.
    """

    def __init__(self, path: str | Path, *, grid: str | Path | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise SwathRefusal(
                f"history file {self.path} does not exist. The detector reads a "
                "coarse forecast this project produced; without one there is "
                "nothing to place a swath from, and a plan built on no forecast "
                "would be an operator's guess wearing a receipt"
            )
        from netCDF4 import Dataset  # noqa: PLC0415 - heavy, and only needed here

        self._Dataset = Dataset
        self._owned: list[Any] = []
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._grid_path = None if grid is None else Path(grid).expanduser().resolve()
        self._verified: dict[str, str] = {}
        if self.path.suffix.lower() == ".json":
            self._open_run_receipt()
        else:
            self._open_single_file()

    # -- opening ------------------------------------------------------------
    def _track(self, dataset: Any) -> Any:
        self._owned.append(dataset)
        return dataset

    def _open_single_file(self) -> None:
        self.kind = "history_file"
        self.sha256 = _sha256(self.path)
        dataset = self._track(self._Dataset(str(self.path), "r"))
        self._mesh = dataset
        self._mesh_path = self.path
        if self._grid_path is not None:
            self._mesh = self._track(self._Dataset(str(self._grid_path), "r"))
            self._mesh_path = self._grid_path
        # The mesh first: a file with no connectivity is the shape this
        # project's own forecast door writes, and that is the refusal an
        # operator needs to see rather than a downstream one about time.
        self._require_mesh()
        self._frames = self._frames_from_time_axis(dataset)
        self._records = tuple((dataset, frame.index) for frame in self._frames)

    def _open_run_receipt(self) -> None:
        self.kind = "forecast_run_receipt"
        self.sha256 = _sha256(self.path)
        try:
            receipt = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SwathRefusal(
                f"{self.path.name} is not readable JSON ({error}). A '.json' "
                "history argument is read as the run receipt "
                "'gpuwm-hex forecast' writes beside its frames"
            ) from error
        files = _receipt_at(receipt, _RECEIPT_FRAME_FILES, self.path)
        labels = _receipt_at(receipt, _RECEIPT_FRAME_LABELS, self.path)
        steps = sorted(files, key=lambda key: int(key))
        if len(steps) < 1:
            raise SwathRefusal(
                f"{self.path.name} records no history frames in "
                f"{'.'.join(_RECEIPT_FRAME_FILES)}, so the forecast it describes "
                "wrote nothing to place a swath from"
            )
        stamps: list[str] = []
        paths: list[Path] = []
        for step in steps:
            entry = files[step]
            frame_path = Path(str(entry["path"])).expanduser()
            frame_path = self._beside(frame_path)
            recorded = str(entry.get("sha256", ""))
            observed = _sha256(frame_path)
            if recorded and observed != recorded:
                raise SwathRefusal(
                    f"history frame {frame_path.name} does not match the SHA-256 "
                    f"its own run receipt recorded ({observed[:16]}... against "
                    f"{recorded[:16]}...). The plan would name a forecast it was "
                    "not built from, which is the one error a receipt exists to "
                    "make impossible"
                )
            self._verified[str(frame_path)] = observed
            paths.append(frame_path)
            stamps.append(str(labels.get(step, "")))
        seconds = self._seconds_from_valid_times(stamps)
        if seconds is None:
            raise SwathRefusal(
                f"{self.path.name} carries history labels this reader cannot read "
                f"as times ({stamps[:3]}...). A frame index is not a time: used as "
                "one it would scale every track speed, every flare and every "
                "ignition hour by the output interval"
            )
        datasets = [self._track(self._Dataset(str(one), "r")) for one in paths]
        self._frames = tuple(
            HistoryFrame(
                index=index,
                time_seconds=float(seconds[index] - seconds[0]),
                valid_time=stamps[index],
            )
            for index in range(len(paths))
        )
        self._records = tuple((dataset, 0) for dataset in datasets)
        self._frame_paths = tuple(paths)
        grid = self._grid_path
        if grid is None:
            # STATIC BEFORE GRID, and the reason is measured.  Both carry
            # the same connectivity, but an MPAS grid file is written on
            # the UNIT sphere: its areaCell is about 8.3e-5 where the
            # static's is 3.4e9 m^2 for the same cell.  Taking areas off
            # the grid silences the convection row's 20,000 km^2 floor
            # completely and reports a quiet world.
            entry = None
            for keys in (_RECEIPT_STATIC_FILE, _RECEIPT_GRID_FILE):
                try:
                    entry = _receipt_at(receipt, keys, self.path)
                    break
                except SwathRefusal:
                    continue
            if entry is None:
                raise SwathRefusal(
                    f"{self.path.name} names neither "
                    f"'{'.'.join(_RECEIPT_STATIC_FILE)}' nor "
                    f"'{'.'.join(_RECEIPT_GRID_FILE)}', so the mesh the forecast "
                    "was integrated on cannot be found from it"
                )
            grid = self._beside(Path(str(entry["path"])).expanduser())
            recorded = str(entry.get("sha256", ""))
            observed = _sha256(grid)
            if recorded and observed != recorded:
                raise SwathRefusal(
                    f"mesh file {grid.name} does not match the SHA-256 the run "
                    f"receipt pinned it by ({observed[:16]}... against "
                    f"{recorded[:16]}...). The detector would search a mesh the "
                    "forecast was not integrated on"
                )
            self._verified[str(grid)] = observed
        self._mesh = self._track(self._Dataset(str(grid), "r"))
        self._mesh_path = grid
        self._require_mesh()

    def _beside(self, recorded: Path) -> Path:
        """The recorded path, or the same basename beside the receipt.

        A receipt records ABSOLUTE paths on the machine that ran the
        forecast.  Copied to another machine the frames travel with the
        receipt, so a missing recorded path is retried as a sibling before
        it is refused -- and the SHA-256 check above is what makes that
        substitution safe rather than hopeful.
        """

        if recorded.exists():
            return recorded.resolve()
        for candidate in (
            self.path.parent / recorded.name,
            self.path.parent / "out" / recorded.name,
            self.path.parent.parent / recorded.name,
        ):
            if candidate.exists():
                return candidate.resolve()
        raise SwathRefusal(
            f"the run receipt {self.path.name} names {recorded}, which does not "
            f"exist here and is not beside the receipt either. The frames a plan "
            "is built from must be the frames the forecast wrote"
        )

    def _frames_from_time_axis(self, dataset: Any) -> tuple["HistoryFrame", ...]:
        variables = dataset.variables
        if "Time" not in dataset.dimensions:
            raise SwathRefusal(
                f"history file {self.path.name} has no Time dimension, so it holds "
                "no forecast sequence to project a track along"
            )
        count = int(dataset.dimensions["Time"].size)
        valid = self._valid_times(dataset, count)
        seconds: np.ndarray | None = None
        if "Time" in variables:
            candidate = np.asarray(variables["Time"][:], dtype=np.float64).ravel()
            if candidate.size == count:
                seconds = candidate
        if seconds is None and any(valid):
            seconds = self._seconds_from_valid_times(valid)
        if seconds is None:
            raise SwathRefusal(
                f"history file {self.path.name} carries neither a 'Time' coordinate "
                "in seconds nor readable 'xtime' strings. A frame index is not a "
                "time: used as one it would scale every track speed, every flare "
                "and every ignition hour by the output interval, and the plan would "
                "look reasonable while being wrong by that factor. If this is one "
                "frame of a 'gpuwm-hex forecast' run, pass that run's receipt as "
                "--history instead: it carries the whole sequence and its times"
            )
        return tuple(
            HistoryFrame(
                index=index,
                time_seconds=float(seconds[index] - seconds[0]),
                valid_time=valid[index],
            )
            for index in range(count)
        )

    def _require_mesh(self) -> None:
        self._area_scale_to_m2 = self._resolve_area_scale()
        for required in MESH_VARIABLES:
            if required not in self._mesh.variables:
                raise SwathRefusal(
                    f"{self._mesh_path.name} has no {required!r}. The detector "
                    "searches over the mesh's own connectivity rather than a "
                    "regridded image, so the mesh it was written on must be "
                    f"readable. Present variables: "
                    f"{sorted(self._mesh.variables)[:12]}... If this is one frame "
                    "of a 'gpuwm-hex forecast' run, pass that run's receipt as "
                    "--history (it names the grid file), or name the grid "
                    "directly with --grid"
                )

    def _resolve_area_scale(self) -> float:
        """What ``areaCell`` must be multiplied by to become square metres.

        A file with no ``sphere_radius`` is taken at its word: its areas
        are already physical, which is what this layer's own fixture and
        every MPAS-A output write.  A file that declares a REAL radius is
        also already physical.  A file that declares the unit sphere is
        neither, and there is no radius in it to recover one from -- so
        it is refused by name rather than scaled by a guess, because a
        guessed radius is exactly the kind of silent factor that made
        this refusal necessary.
        """

        if "sphere_radius" not in self._mesh.ncattrs():
            return 1.0
        try:
            radius = float(np.asarray(self._mesh.getncattr("sphere_radius")).ravel()[0])
        except (TypeError, ValueError):
            return 1.0
        if radius >= DIMENSIONAL_SPHERE_RADIUS_M:
            return 1.0
        raise SwathRefusal(
            f"{self._mesh_path.name} declares sphere_radius={radius:g}, so its "
            "areaCell is on the unit sphere and is not an area in square metres "
            f"(a 3.4e9 m^2 cell is stored as about 8.3e-5). Every area gate "
            "downstream -- the convection row's minimum_area_km2 above all -- "
            "would compare a number near zero against its floor, reject every "
            "region and report no error. Point --history at the run receipt "
            "(it names the STATIC file, which carries the same connectivity on "
            "the real sphere) or name a dimensional mesh with --grid"
        )

    def close(self) -> None:
        for dataset in self._owned:
            dataset.close()
        self._owned = []

    def __enter__(self) -> "HistoryReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- mesh ---------------------------------------------------------------
    @property
    def cell_count(self) -> int:
        return int(self._mesh.dimensions["nCells"].size)

    @property
    def latitudes_deg(self) -> np.ndarray:
        return np.degrees(np.asarray(self._mesh.variables["latCell"][:], dtype=np.float64))

    @property
    def longitudes_deg(self) -> np.ndarray:
        """Cell longitudes folded to ``[-180, 180)``.

        MPAS stores ``lonCell`` in ``[0, 2*pi)``; the mesh spec's
        ``vertices_deg`` rows and every figure this layer emits are in the
        half-open signed range, so the fold happens once, here.
        """

        raw = np.degrees(np.asarray(self._mesh.variables["lonCell"][:], dtype=np.float64))
        return ((raw + 180.0) % 360.0) - 180.0

    @property
    def areas_km2(self) -> np.ndarray:
        """``areaCell`` in km^2, on the sphere the file was written for.

        NOT SIMPLY A DIVISION BY 1e6, and that mattered.  An MPAS mesh
        file declares ``sphere_radius``, and a GRID file declares
        ``1.0``: it stores areas on the UNIT sphere, about 8.3e-5 for a
        cell whose real area is 3.4e9 m^2.  Divided by 1e6 that is
        8.3e-11 km^2, so every area test downstream compares a number
        near zero against its floor and fails silently -- measured on a
        real 151,649-cell global run, where the convection row's
        20,000 km^2 floor rejected all 266 connected regions of 35 dBZ
        or more at every frame while reporting no error at all.

        The scale is fixed once at open time, where a non-dimensional
        mesh with no way to recover a radius is REFUSED rather than
        guessed at.
        """

        areas = np.asarray(self._mesh.variables["areaCell"][:], dtype=np.float64)
        return areas * self._area_scale_to_m2 / 1.0e6

    def neighbours(self) -> np.ndarray:
        """``cellsOnCell`` as a zero-based array with ``-1`` for absent.

        Shape ``(nCells, maxEdges)``.  Entries beyond ``nEdgesOnCell`` are
        set to ``-1`` as well, so a caller never has to carry the valency
        alongside the table.
        """

        raw = np.asarray(self._mesh.variables["cellsOnCell"][:], dtype=np.int64)
        valency = np.asarray(self._mesh.variables["nEdgesOnCell"][:], dtype=np.int64)
        out = raw - 1
        columns = np.arange(out.shape[1])[None, :]
        out[columns >= valency[:, None]] = NO_NEIGHBOUR
        out[raw == 0] = NO_NEIGHBOUR
        return out

    # -- time ---------------------------------------------------------------
    def frames(self) -> tuple[HistoryFrame, ...]:
        """Every time record, with its offset in SECONDS from the first.

        The seconds matter: a track's speed gate, a swath's flare and a
        delayed start are all rates, and a frame INDEX used as a time
        would make every one of them wrong by the output interval.  So a
        source with no readable time axis is refused rather than counted.
        """

        return self._frames

    @staticmethod
    def _valid_times(dataset: Any, count: int) -> list[str]:
        if "xtime" not in dataset.variables:
            return [""] * count
        raw = np.asarray(dataset.variables["xtime"][:])
        out: list[str] = []
        for row in raw:
            parts = []
            for entry in np.atleast_1d(row).ravel():
                if isinstance(entry, (bytes, bytearray)):
                    parts.append(bytes(entry))
                else:
                    parts.append(str(entry).encode("ascii", "ignore"))
            out.append(b"".join(parts).decode("ascii", "ignore").strip())
        while len(out) < count:
            out.append("")
        return out[:count]

    @staticmethod
    def _seconds_from_valid_times(valid: Sequence[str]) -> np.ndarray | None:
        from datetime import datetime

        stamps = []
        for text in valid:
            parsed = None
            for shape in _STAMP_FORMATS:
                try:
                    parsed = datetime.strptime(text.strip(), shape)
                    break
                except ValueError:
                    continue
            if parsed is None:
                return None
            stamps.append(parsed)
        if not stamps:
            return None
        first = stamps[0]
        return np.asarray(
            [(stamp - first).total_seconds() for stamp in stamps], dtype=np.float64
        )

    # -- fields -------------------------------------------------------------
    def derive(
        self, row: FieldRow, frame_index: int, *, registry: Any = None
    ) -> np.ndarray:
        """The per-cell scalar one field row describes, at one frame.

        Every branch here is a DERIVATION KIND from the closed vocabulary,
        never a variable name.  There is no ``if name == "surface_pressure"``
        anywhere in this package, which is what makes a new field a row.

        A row whose operands are OTHER ROWS (``inputs``) is derived by
        recursion, so ``registry`` must be supplied to reach the rows it
        composes.  The recursion is safe because the registry refused any
        cycle in the field graph at load time, with the ring printed.
        """

        key = (row.id, frame_index)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if row.inputs:
            if registry is None:
                raise SwathRefusal(
                    f"field {row.id!r} is derived from other field rows "
                    f"{list(row.inputs)}, but no registry was supplied to reach "
                    "them. A composed field cannot be evaluated from its own row "
                    "alone; pass the metric registry that loaded it"
                )
            operands = [
                self.derive(registry.field_rows[name], frame_index, registry=registry)
                for name in row.inputs
            ]
        else:
            operands = [
                self._read(name, frame_index) for name in row.source_variables
            ]
        value = self._apply(row, operands, frame_index)
        value = np.ascontiguousarray(value, dtype=np.float64)
        self._cache[key] = value
        return value

    def _apply(
        self, row: FieldRow, operands: Sequence[np.ndarray], frame_index: int
    ) -> np.ndarray:
        kind = row.derivation_kind
        # -- leaves: the file's own raw shapes ------------------------------
        if kind == "direct":
            return self._as_cell_scalar(operands[0], row)
        if kind == "level_slice":
            return self._slice_level(operands[0], row)
        if kind == "vertical_extremum":
            return self._vertical_extremum(operands[0], row)
        if kind == "time_rate":
            return self._time_rate(row, frame_index)
        # -- operators over per-cell scalars --------------------------------
        scalars = [self._as_cell_scalar(array, row) for array in operands]
        if kind == "vector_magnitude":
            return np.hypot(scalars[0], scalars[1])
        if kind == "sea_level_reduction":
            return self._sea_level_reduction(scalars, row)
        if kind == "linear_combination":
            total = np.full(self.cell_count, float(row.offset), dtype=np.float64)
            for coefficient, array in zip(row.coefficients, scalars):
                total = total + coefficient * array
            return total
        if kind == "product":
            total = np.full(self.cell_count, float(row.coefficient), dtype=np.float64)
            for array in scalars:
                total = total * array
            return total
        if kind == "ratio":
            floor = float(row.denominator_floor or 0.0)
            denominator = scalars[1]
            # Signed floor: a denominator that crosses zero would otherwise
            # swing the ratio through infinity, and a rank score built on an
            # infinity is a candidate that wins every cycle for ever.
            safe = np.where(
                np.abs(denominator) < floor,
                np.where(denominator < 0.0, -floor, floor),
                denominator,
            )
            return float(row.coefficient) * scalars[0] / safe
        if kind == "threshold_margin":
            threshold = float(row.threshold or 0.0)
            scale = float(row.scale or 1.0)
            if row.comparison == "at_least":
                return (scalars[0] - threshold) / scale
            return (threshold - scalars[0]) / scale
        if kind == "extremum_of":
            stack = np.vstack(scalars)
            return stack.max(axis=0) if row.extremum == "maximum" else stack.min(axis=0)
        if kind == "saturation_vapour_pressure":
            return self._saturation_vapour_pressure(scalars[0], row)
        if kind == "vapour_pressure":
            return self._vapour_pressure(scalars[0], scalars[1], row)
        raise SwathDocumentError(  # pragma: no cover - the registry closes this
            f"field {row.id!r} declares derivation kind {kind!r}, "
            "which loaded but has no implementation. The vocabulary and the "
            "implementation must move together"
        )

    def _read(self, name: str, frame_index: int) -> np.ndarray:
        dataset, record = self._records[frame_index]
        if name not in dataset.variables:
            # A static that the frames do not repeat may live on the mesh
            # source instead; the grid file carries the geometry and some
            # streams keep terrain there rather than on every frame.
            if name in self._mesh.variables:
                dataset, record = self._mesh, 0
            else:
                raise SwathRefusal(
                    f"the coarse forecast at {self.path.name} does not publish "
                    f"{name!r}. The armed metric rows need it: run "
                    "'gpuwm-hex swath metrics --publication-manifest' to see the "
                    "whole list a cycle must publish, and arm the coarse run's "
                    "history stream with it. A detector that silently skipped an "
                    "unpublished field would report a quiet world on a busy day"
                )
        variable = dataset.variables[name]
        if variable.dimensions and variable.dimensions[0] == "Time":
            return np.asarray(variable[record], dtype=np.float64)
        return np.asarray(variable[:], dtype=np.float64)

    def _as_cell_scalar(self, array: np.ndarray, row: FieldRow) -> np.ndarray:
        if array.ndim == 1 and array.shape[0] == self.cell_count:
            return array
        raise SwathRefusal(
            f"field {row.id!r} derivation 'direct' read an array of shape "
            f"{array.shape}, which is not one value per cell "
            f"({self.cell_count}). Use 'level_slice' or 'vertical_extremum' for a "
            "three-dimensional variable"
        )

    def _levels_first(self, array: np.ndarray, row: FieldRow) -> np.ndarray:
        if array.ndim != 2:
            raise SwathRefusal(
                f"field {row.id!r} needs a two-dimensional (levels, cells) array; "
                f"the file gave shape {array.shape}"
            )
        if array.shape[1] == self.cell_count:
            return array
        if array.shape[0] == self.cell_count:
            return array.T
        raise SwathRefusal(
            f"field {row.id!r} read shape {array.shape}, neither axis of which is "
            f"the mesh's {self.cell_count} cells"
        )

    def _slice_level(self, array: np.ndarray, row: FieldRow) -> np.ndarray:
        levels = self._levels_first(array, row)
        index = int(row.level_index or 0)
        if not 0 <= index < levels.shape[0]:
            raise SwathRefusal(
                f"field {row.id!r} asks for level_index {index}, but the variable "
                f"has {levels.shape[0]} levels"
            )
        return levels[index]

    def _vertical_extremum(self, array: np.ndarray, row: FieldRow) -> np.ndarray:
        levels = self._levels_first(array, row)
        return levels.max(axis=0) if row.extremum == "maximum" else levels.min(axis=0)

    def _sea_level_reduction(
        self, arrays: Sequence[np.ndarray], row: FieldRow
    ) -> np.ndarray:
        """Surface pressure reduced to mean sea level, hypsometrically.

        WHY THIS EXISTS, measured rather than supposed.  A minimum search
        over RAW surface pressure on a real global forecast does not find
        cyclones, it finds ELEVATION: on the 96 km global run the twelve
        deepest cells were all on the Tibetan Plateau above 5,300 m at
        about 52,900 Pa, roughly 47 kPa below the 100,400 Pa a closed low
        is defined by, and 43 per cent of the globe sat under that
        threshold.  No threshold value fixes that, because the ordering
        itself is wrong -- the deepest "low" on Earth is a mountain.  The
        shipped metrics document said as much in its own field
        description before any of this ran; this is the field it asked
        for.

        The reduction is the standard hypsometric one,
        ``p0 = ps * exp(g z / (R Tbar))`` with ``Tbar`` the mean of the
        surface temperature and an assumed-lapse temperature at sea
        level.  It is an ASSUMPTION over terrain, exactly as it is in
        every model post-processor, and it is the assumption the 1004 hPa
        convention was written against.

        ``source_variables`` are ordered ``[pressure, height, temperature]``,
        the way ``vector_magnitude`` orders its pair.
        """

        pressure = self._as_cell_scalar(arrays[0], row)
        height = self._as_cell_scalar(arrays[1], row)
        temperature = self._as_cell_scalar(arrays[2], row)
        lapse = float(row.lapse_rate_k_per_m or 0.0065)
        gravity = 9.80665      # m s^-2, standard gravity
        gas_constant = 287.058  # J kg^-1 K^-1, dry air
        mean_temperature = temperature + 0.5 * lapse * height
        if np.any(mean_temperature <= 0.0):
            raise SwathRefusal(
                f"field {row.id!r}: the sea-level reduction formed a "
                "non-positive mean column temperature, which means the "
                "temperature source is not in kelvin. A reduction run on "
                "celsius would move every centre by hundreds of hectopascals "
                "and the plan would look ordinary while being wrong everywhere"
            )
        return pressure * np.exp(gravity * height / (gas_constant * mean_temperature))

    def _time_rate(self, row: FieldRow, frame_index: int) -> np.ndarray:
        """A published run-total accumulator, differenced into a rate per hour.

        WHY THIS IS A DERIVATION AND NOT A THRESHOLD ON THE ACCUMULATOR.
        The history stream publishes ``rainc``, ``rainnc``, ``snownc`` and
        ``graupelnc`` as totals since the run began, so a row that
        thresholded them directly would place its swath where it has
        already rained rather than where it is about to: the exceedance
        region only grows, its centroid drifts toward the start of the
        event, and by hour 20 the whole storm track is one connected
        region with a centroid in the middle of ground the weather has
        left.  Differencing turns four published accumulators into four
        rate fields with no change to the stream.

        FRAME ZERO IS ZERO, and that is a real limitation rather than a
        convenience: there is no earlier frame to difference against, so a
        rate row cannot detect at the first frame.  A row built on this
        should carry ``start_policy.kind = time_of_first_exceedance``,
        which derives its ignition hour from the forecast anyway.
        """

        name = row.source_variables[0]
        current = self._as_cell_scalar(self._read(name, frame_index), row)
        if frame_index <= 0:
            return np.zeros_like(current)
        frames = self.frames()
        hours = (
            frames[frame_index].time_seconds - frames[frame_index - 1].time_seconds
        ) / 3600.0
        if hours <= 0.0:
            raise SwathRefusal(
                f"field {row.id!r} is a 'time_rate' on {name!r}, but frame "
                f"{frame_index} is not later than frame {frame_index - 1} "
                f"({frames[frame_index].time_seconds} s against "
                f"{frames[frame_index - 1].time_seconds} s). A rate over a "
                "non-positive interval is either an infinity or a sign flip, and "
                "either one wins every ranking it enters"
            )
        previous = self._as_cell_scalar(self._read(name, frame_index - 1), row)
        return (current - previous) / hours

    @staticmethod
    def _saturation_vapour_pressure(
        temperature: np.ndarray, row: FieldRow
    ) -> np.ndarray:
        """Saturation vapour pressure over liquid water, in hPa.

        Bolton (1980), Monthly Weather Review 108, equation 10:
        ``es = 6.112 exp(17.67 Tc / (Tc + 243.5))`` with ``Tc`` in celsius.
        Quoted at better than 0.1 per cent between -35 and +35 C, which
        covers every temperature a fire-weather or heat row cares about.

        It is a PRIMITIVE rather than a whole relative humidity because
        the exponential is the only part of humidity that the row algebra
        cannot express.  With this and :meth:`_vapour_pressure`, relative
        humidity is a ``ratio`` row and vapour-pressure deficit is a
        ``linear_combination`` row -- table work, not code.
        """

        if np.any(temperature < 100.0):
            raise SwathRefusal(
                f"field {row.id!r}: 'saturation_vapour_pressure' was handed a "
                f"temperature as low as {float(np.min(temperature)):.2f}, which is "
                "not kelvin. Run on celsius the formula returns a saturation "
                "pressure near 6 hPa everywhere and every humidity built on it is "
                "wrong by a factor that changes with latitude"
            )
        celsius = temperature - 273.15
        return 6.112 * np.exp(17.67 * celsius / (celsius + 243.5))

    @staticmethod
    def _vapour_pressure(
        mixing_ratio: np.ndarray, pressure: np.ndarray, row: FieldRow
    ) -> np.ndarray:
        """Water-vapour partial pressure in hPa, from mixing ratio and pressure.

        ``e = w p / (epsilon + w)`` with ``epsilon = R_d/R_v = 0.62197``,
        converted from pascals to hectopascals so it is on the same scale
        as :meth:`_saturation_vapour_pressure`.

        THE CLAMP IS NAMED BY THE ARTIFACT, not assumed.  The history
        stream's own global attribute says so: ``q2_negative_policy =
        "preserved bitwise; native q2 itself carries negative values, so
        consumers forming dewpoint clamp at their own boundary"``.  This
        is that boundary.  Left unclamped a negative mixing ratio gives a
        negative vapour pressure and a negative relative humidity, which
        passes every ``at_most`` dryness test ever written -- so a fire
        row would fire hardest on the cells where the diagnostic is
        broken.
        """

        clamped = np.maximum(np.asarray(mixing_ratio, dtype=np.float64), 0.0)
        return clamped * pressure / (0.62197 + clamped) / 100.0

    def provenance(self) -> Mapping[str, Any]:
        out: dict[str, Any] = {
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
            "sha256": self.sha256,
            "cells": self.cell_count,
            "frames": len(self.frames()),
            "kind": self.kind,
            "mesh_source": str(self._mesh_path),
        }
        if self.kind == "forecast_run_receipt":
            out["frame_files"] = [
                {"path": str(one), "sha256": self._verified.get(str(one), "")}
                for one in self._frame_paths
            ]
            out["frames_verified_against_receipt"] = True
        return out


def ball_indices(
    seed: int,
    radius_km: float,
    *,
    neighbours: np.ndarray,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> list[int]:
    """Every cell within ``radius_km`` of ``seed``, by a walk over the mesh.

    Breadth-first over ``cellsOnCell`` rather than a k-d tree over all
    cells: the ball is a handful of cells on a coarse mesh, the walk is
    O(ball), and it needs no auxiliary structure that could disagree with
    the file's own connectivity.  The walk expands through cells that are
    themselves outside the radius only if they lead somewhere inside --
    it does not, which means a ball is CONNECTED by construction.  On a
    Voronoi mesh with spacing well under the radius that is the same set a
    distance query would return, and where it is not, a connected ball is
    the one a physical feature actually occupies.
    """

    from .geometry import great_circle_km

    seed_lat = float(latitudes_deg[seed])
    seed_lon = float(longitudes_deg[seed])
    seen = {seed}
    frontier = [seed]
    out = [seed]
    while frontier:
        nxt: list[int] = []
        for cell in frontier:
            for neighbour in neighbours[cell]:
                index = int(neighbour)
                if index == NO_NEIGHBOUR or index in seen:
                    continue
                seen.add(index)
                distance = great_circle_km(
                    seed_lat, seed_lon,
                    float(latitudes_deg[index]), float(longitudes_deg[index]),
                )
                if distance <= radius_km:
                    out.append(index)
                    nxt.append(index)
        frontier = nxt
    return out


def area_weighted_centroid(
    cells: Sequence[int],
    *,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
    areas_km2: np.ndarray,
    weights_are_absolute: bool = False,
) -> tuple[float, float]:
    """Spherical, weighted centroid of a set of cells, in degrees.

    Averaged as unit vectors and renormalised, never as latitude and
    longitude numbers: a mean of longitudes puts the centroid of a feature
    straddling the antimeridian on the opposite side of the planet.

    ``weights_are_absolute`` says ``areas_km2`` is already one weight per
    listed cell rather than a whole-mesh array to be indexed.
    """

    index = np.asarray(cells, dtype=np.int64)
    weight = np.asarray(areas_km2, dtype=np.float64) if weights_are_absolute else areas_km2[index]
    lat = np.radians(latitudes_deg[index])
    lon = np.radians(longitudes_deg[index])
    cos_lat = np.cos(lat)
    x = float(np.sum(weight * cos_lat * np.cos(lon)))
    y = float(np.sum(weight * cos_lat * np.sin(lon)))
    z = float(np.sum(weight * np.sin(lat)))
    norm = float(np.sqrt(x * x + y * y + z * z))
    if norm <= 0.0:
        raise SwathRefusal(
            "a feature's cells average to the centre of the sphere, so it has no "
            "centroid; this happens only for a feature that wraps the whole globe, "
            "which is not a placeable threat"
        )
    return (
        float(np.degrees(np.arctan2(z / norm, np.hypot(x / norm, y / norm)))),
        float(((np.degrees(np.arctan2(y / norm, x / norm)) + 180.0) % 360.0) - 180.0),
    )


__all__ = [
    "NO_NEIGHBOUR",
    "HistoryFrame",
    "HistoryReader",
    "area_weighted_centroid",
    "ball_indices",
]
