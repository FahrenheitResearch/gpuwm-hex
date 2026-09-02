"""Curated evidence panels for one finished paired run.

``gpuwm-hex pair`` runs one authored forecast twice -- a control leg, and a
treatment leg that is the same request plus a point-source table -- and writes
``pair_manifest.json`` beside both legs' history trees.  That manifest and the
text summary next to it are complete and unreadable: a summary is a column of
numbers, and the question a reader actually arrives with is *where* and *when*
the two legs parted.

This tool answers that question with a small, fixed set of pictures:

* one **column-maximum map** per declared extra scalar per selected valid
  time, on a log colour scale with zeros masked, so a field that exists only
  in the treatment leg is shown where it is rather than averaged away;
* one **map** per declared extra accumulator, on a linear scale;
* one **budget-in-time** line per declared extra -- the field summed over
  cells and levels, area-weighted when a grid is given -- so a reader can see
  whether the quantity is growing, settling or already gone;
* one **difference map** per shared weather field per selected valid time,
  treatment minus control, zero-centred and diverging;
* one **overview** page carrying the domain outline, the cell count, the mesh
  centroid, the edge-length range and the point-source table the treatment leg
  carried.

WHAT THIS TOOL IS NOT.  These are analysis charts drawn in matplotlib: a
difference, a column statistic and a budget.  The per-leg weather-field
renders are not this tool's job -- they come from ``gpuwm-hex render``, which
drives the Rust converter and renderer, and nothing here reimplements a
product plot.

Every panel drawn from a difference or from a declared extra says on its own
face that it is a model result and not evidence about the atmosphere, because
a picture travels further than the summary it was made from and arrives
without the sentence that framed it.

Usage::

    python tools/plot_pair_gallery.py --pair-out DIR --out GALLERY_DIR \\
        [--frames N] [--extras NAME,...] [--grid PATH]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import textwrap
from typing import Any, Sequence

import numpy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Read from the door rather than restated here.  The manifest's name, its
# schema string, the leg names and the record-ordering rule are the door's
# facts; a gallery that carried its own copy would keep reading the old shape
# for exactly as long as nobody noticed.
from hexcore.output import HISTORY_MESH_VARIABLES  # noqa: E402
from hexcore.pair_door import (  # noqa: E402
    LEG_NAMES,
    MANIFEST_NAME,
    PAIR_SCHEMA,
    RESULT_SENTENCE,
    _cell_major as cell_major,
)

#: Schema of the ``captions.json`` this tool writes.
GALLERY_SCHEMA = "gpuwm-hex.pair-gallery/v1"

CAPTIONS_NAME = "captions.json"
INDEX_NAME = "index.md"

#: The clause every difference panel and every declared-extra panel carries,
#: in its caption and on its own face.
MODEL_RESULT_CLAUSE = (
    "This is a model result, not evidence about the atmosphere."
)

#: What a weather-field render is, and where it comes from instead.  Stated in
#: ``--help`` so a reader who wanted product plots is redirected before they
#: mistake a difference map for one.
RENDER_SENTENCE = (
    "The per-leg weather-field renders are not this tool's job: they come "
    "from `gpuwm-hex render`, which drives the Rust converter and renderer."
)

#: The shared fields a difference panel is drawn for, in preference order,
#: each with the spellings a frame may carry.  Discovery is by name against
#: what the two legs actually hold: a field absent from the frames is simply
#: not drawn, and no more than :data:`MAX_DIFFERENCE_FIELDS` are taken.
DIFFERENCE_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("qc", ("qc", "qcloud", "qc_cloud", "scalars_qc")),
    ("qi", ("qi", "qice", "qi_ice", "scalars_qi")),
    ("qr", ("qr", "qrain", "qr_rain", "scalars_qr")),
    ("qs", ("qs", "qsnow", "qs_snow", "scalars_qs")),
    ("qg", ("qg", "qgraup", "qgraupel", "scalars_qg")),
    ("theta", ("theta", "theta_m", "th")),
    ("w", ("w",)),
)
MAX_DIFFERENCE_FIELDS = 6

#: Fields reduced down the column by MEAN rather than by maximum.  A column
#: maximum of potential temperature is the model top and says nothing about
#: where the legs differ.
COLUMN_MEAN_FIELDS = frozenset({"theta"})

_MISSING_UNITS = "units not declared in the frame"


class GalleryRefusal(RuntimeError):
    """A named gallery refusal: what breaks, then the remedy."""


def _refuse(message: str) -> GalleryRefusal:
    return GalleryRefusal(message)


# ---------------------------------------------------------------------------
# the manifest and the frames it names
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameRef:
    """One time record inside one history file, on one leg."""

    leg: str
    path: Path
    record: int
    valid_time: str


@dataclass(frozen=True)
class PairedFrame:
    """The same valid time on both legs."""

    valid_time: str
    token: str
    control: FrameRef
    treatment: FrameRef


def read_manifest(pair_out: Path) -> dict[str, Any]:
    """The pair manifest, or a refusal naming what is missing and why."""

    manifest_path = pair_out / MANIFEST_NAME
    if not manifest_path.is_file():
        raise _refuse(
            f"{pair_out} carries no {MANIFEST_NAME}.  A gallery is built from "
            f"the manifest a finished pair writes: it is the only record of "
            f"which tree was the control leg, which was the treatment leg and "
            f"which frames belong to the run.  Guessing the legs from "
            f"directory names would happily difference two halves of two "
            f"different pairs and label the result a treatment effect.  Point "
            f"--pair-out at the directory `gpuwm-hex pair --pair-out` was "
            f"given, and re-run the pair if it did not finish."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise _refuse(
            f"{manifest_path} is not readable as JSON ({error}).  A truncated "
            f"manifest is what a pair that died mid-write leaves behind; the "
            f"run it describes did not finish and there is nothing complete "
            f"to draw."
        ) from error
    schema = manifest.get("schema")
    if schema != PAIR_SCHEMA:
        raise _refuse(
            f"{manifest_path} declares schema {schema!r} and this tool reads "
            f"{PAIR_SCHEMA!r}.  The leg names, the history lists and the "
            f"point-source record are read by key; a different shape read "
            f"under these keys yields empty lists and an empty gallery that "
            f"looks like a run which produced nothing."
        )
    legs = manifest.get("legs") or {}
    for name in LEG_NAMES:
        if name not in legs:
            raise _refuse(
                f"{manifest_path} names no {name!r} leg (it has "
                f"{sorted(legs)}).  A pair is two legs; with one of them "
                f"absent every difference in this gallery would be taken "
                f"against nothing."
            )
    return manifest


def _resolve(entry: Any, leg: dict[str, Any], pair_out: Path, leg_name: str) -> Path:
    """The frame on disk, from the manifest's record or from where it sits now.

    The manifest pins absolute paths from the machine the pair ran on.  A pair
    directory that has been moved or copied still holds its frames, so the
    recorded path is tried first and the leg's own ``out`` directory second --
    and a frame that is at neither is refused by name rather than skipped,
    because a gallery quietly missing two of its six valid times is a gallery
    that says the treatment stopped acting.
    """

    recorded = Path(str(entry["path"] if isinstance(entry, dict) else entry))
    if recorded.is_file():
        return recorded
    for base in (
        Path(str(leg.get("out", ""))) if leg.get("out") else None,
        pair_out / leg_name / "out",
        pair_out / leg_name,
    ):
        if base is None:
            continue
        candidate = base / recorded.name
        if candidate.is_file():
            return candidate
    raise _refuse(
        f"the {leg_name} leg's frame {recorded.name} is named by the manifest "
        f"and is not at {recorded}, nor beside the leg's other frames.  The "
        f"gallery draws the valid times both legs carry; a frame that has "
        f"been moved or deleted since the run would silently drop a valid "
        f"time from every panel in this directory."
    )


def _time_token(valid_time: str) -> str:
    """A filename-safe stamp for one valid time.

    ``2026-01-01_00:00:00`` becomes ``2026-01-01T000000``.  The separators go
    because a colon cannot appear in a filename on every platform this tool
    runs on, and the underscore goes because it is the field separator in the
    PNG naming scheme.
    """

    token = valid_time.strip().replace(" ", "T").replace("_", "T")
    token = token.replace(":", "").replace(".", "")
    kept = [character for character in token if character.isalnum() or character in "-T"]
    return "".join(kept) or "frame"


def _xtime_strings(dataset: Any, records: int) -> list[str] | None:
    """``xtime`` as one string per record, or ``None`` when it is not there."""

    if "xtime" not in dataset.variables:
        return None
    raw = dataset.variables["xtime"][...]
    array = numpy.asarray(raw)
    if array.dtype.kind in "SU" and array.ndim <= 1 and array.dtype.itemsize > 1:
        # A string variable rather than the MPAS char array.
        values = [str(numpy.asarray(item).item()) for item in numpy.atleast_1d(array)]
    elif array.ndim == 2:
        values = [
            b"".join(numpy.asarray(row).astype("S1").ravel()).decode("ascii", "ignore")
            for row in array
        ]
    elif array.ndim == 1:
        values = [
            b"".join(numpy.asarray(array).astype("S1").ravel()).decode("ascii", "ignore")
        ]
    else:  # pragma: no cover - no other xtime shape is written
        return None
    values = [value.strip() for value in values]
    if len(values) < records or not all(values):
        return None
    return values[:records]


def leg_frames(manifest: dict[str, Any], leg_name: str, pair_out: Path) -> list[FrameRef]:
    """Every time record this leg wrote, in the order the manifest lists them."""

    from netCDF4 import Dataset

    leg = manifest["legs"][leg_name]
    history = leg.get("history") or []
    if not history:
        raise _refuse(
            f"the {leg_name} leg's manifest entry lists no history frames.  "
            f"A leg that wrote no frames ran no forecast anybody can look at, "
            f"and a gallery built from the other leg alone would be a picture "
            f"of one run presented as a comparison.  Read the leg's own log, "
            f"named in the manifest."
        )
    frames: list[FrameRef] = []
    for entry in history:
        path = _resolve(entry, leg, pair_out, leg_name)
        dataset = Dataset(str(path), "r")
        dataset.set_auto_mask(False)
        try:
            records = (
                int(dataset.dimensions["Time"].size)
                if "Time" in dataset.dimensions
                else 1
            )
            records = max(records, 1)
            stamps = _xtime_strings(dataset, records)
        finally:
            dataset.close()
        for record in range(records):
            if stamps is not None:
                valid = stamps[record]
            elif records == 1:
                # No xtime: the filename is the only stamp the frame carries.
                valid = path.stem.split(".", 1)[-1]
            else:
                valid = f"{path.stem}#{record}"
            frames.append(FrameRef(leg_name, path, record, valid))
    return frames


def pair_frames(
    control: Sequence[FrameRef], treatment: Sequence[FrameRef]
) -> tuple[list[PairedFrame], list[str], list[str]]:
    """Matched valid times, and the ones each leg holds alone.

    Matched by valid time and never by position: two legs whose history
    schedules differ would otherwise have hour 1 differenced against hour 2
    and the result read as a treatment effect.
    """

    by_time = {frame.valid_time: frame for frame in control}
    matched: list[PairedFrame] = []
    seen: set[str] = set()
    for frame in treatment:
        partner = by_time.get(frame.valid_time)
        if partner is None or frame.valid_time in seen:
            continue
        seen.add(frame.valid_time)
        matched.append(
            PairedFrame(
                valid_time=frame.valid_time,
                token=_time_token(frame.valid_time),
                control=partner,
                treatment=frame,
            )
        )
    control_only = [frame.valid_time for frame in control if frame.valid_time not in seen]
    treatment_only = [
        frame.valid_time for frame in treatment if frame.valid_time not in seen
    ]
    if not matched:
        raise _refuse(
            "the two legs share no valid time.  Control holds "
            f"{[frame.valid_time for frame in control][:4]} and treatment "
            f"holds {[frame.valid_time for frame in treatment][:4]}.  Every "
            "panel here differences one leg against the other at the SAME "
            "valid time; with no shared time there is nothing to difference, "
            "and pairing by position would subtract one hour from another."
        )
    return matched, control_only, treatment_only


def select_indices(count: int, frames: int) -> list[int]:
    """``frames`` evenly spaced positions in ``range(count)``, ends included."""

    if frames < 1:
        raise _refuse(
            f"--frames {frames} asks for no valid times.  The gallery is the "
            f"picture of the run; an empty one is a directory a reader will "
            f"take for a run that produced nothing."
        )
    if frames >= count:
        return list(range(count))
    if frames == 1:
        return [0]
    step = (count - 1) / (frames - 1)
    return sorted({int(round(index * step)) for index in range(frames)})


# ---------------------------------------------------------------------------
# what the frames carry
# ---------------------------------------------------------------------------
def _open(path: Path):
    from netCDF4 import Dataset

    dataset = Dataset(str(path), "r")
    dataset.set_auto_mask(False)
    return dataset


def field_rank(dataset: Any, name: str) -> int | None:
    """3 for a per-cell, per-level field, 2 for a per-cell field, else ``None``.

    Mesh geometry is excluded by name -- it is identical on both legs by
    construction -- and so is anything without a ``Time`` axis, which is how a
    per-cell invariant is told apart from a per-cell model field.
    """

    if name in HISTORY_MESH_VARIABLES:
        return None
    variable = dataset.variables[name]
    dimensions = tuple(variable.dimensions)
    if "Time" not in dimensions or "nCells" not in dimensions:
        return None
    if not numpy.issubdtype(numpy.dtype(variable.dtype), numpy.floating):
        return None
    others = [name_ for name_ in dimensions if name_ not in ("Time", "nCells")]
    if len(others) == 0:
        return 2
    if len(others) == 1:
        return 3
    return None


def declared_extras(
    treatment: Any, control: Any, requested: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """``(three-dimensional, two-dimensional)`` fields the treatment leg alone has.

    A DECLARED EXTRA is a per-cell field present in the treatment leg and
    absent from the control leg: the treatment leg's own account of what it
    did.  ``--extras`` replaces the discovered list outright, and a name it
    gives that the treatment frame does not carry is refused rather than
    dropped, because a misspelt field name would otherwise produce a gallery
    that is missing exactly the panel it was run for.
    """

    if requested is not None:
        three: list[str] = []
        two: list[str] = []
        for name in requested:
            if name not in treatment.variables:
                raise _refuse(
                    f"--extras names {name!r}, which the treatment leg's "
                    f"frame does not carry.  Its fields are "
                    f"{sorted(treatment.variables)[:12]}.  A named field that "
                    f"is quietly skipped leaves a gallery that looks complete "
                    f"and is missing the panel it was asked for."
                )
            rank = field_rank(treatment, name)
            if rank == 3:
                three.append(name)
            elif rank == 2:
                two.append(name)
            else:
                raise _refuse(
                    f"--extras names {name!r}, which is not a per-cell model "
                    f"field: its dimensions are "
                    f"{tuple(treatment.variables[name].dimensions)}.  This "
                    f"tool draws values on the mesh, and a field that is not "
                    f"indexed by cell cannot be drawn on one."
                )
        return three, two

    three = sorted(
        name
        for name in treatment.variables
        if name not in control.variables and field_rank(treatment, name) == 3
    )
    two = sorted(
        name
        for name in treatment.variables
        if name not in control.variables and field_rank(treatment, name) == 2
    )
    return three, two


def difference_fields(treatment: Any, control: Any) -> list[str]:
    """Up to six shared per-cell, per-level fields, in preference order."""

    lowered = {name.lower(): name for name in treatment.variables}
    chosen: list[str] = []
    for _canonical, spellings in DIFFERENCE_CANDIDATES:
        for spelling in spellings:
            name = lowered.get(spelling.lower())
            if name is None or name in chosen:
                continue
            if name not in control.variables:
                continue
            if field_rank(treatment, name) != 3 or field_rank(control, name) != 3:
                continue
            chosen.append(name)
            break
        if len(chosen) >= MAX_DIFFERENCE_FIELDS:
            break
    return chosen[:MAX_DIFFERENCE_FIELDS]


def _reduction(name: str) -> str:
    return "mean" if name.lower() in COLUMN_MEAN_FIELDS else "maximum"


def _units(dataset: Any, name: str) -> str:
    value = getattr(dataset.variables[name], "units", "")
    return str(value).strip() or _MISSING_UNITS


def _column(array: numpy.ndarray, how: str) -> numpy.ndarray:
    """One value per cell from a ``(nCells, nLevels)`` block."""

    if array.ndim == 1:
        return array
    return array.mean(axis=1) if how == "mean" else array.max(axis=1)


def read_field(dataset: Any, name: str, record: int) -> numpy.ndarray:
    """One record of one field, cell-major, however the file ordered its axes."""

    variable = dataset.variables[name]
    return cell_major(variable[...], variable.dimensions, record)


def control_presence(control: Any, name: str, record: int) -> str:
    """One measured sentence on what the control leg holds for an extra.

    A DISCOVERED extra is absent from the control leg by definition.  An
    extra named with ``--extras`` need not be: a seeded row run with no
    point-source table carries every agent bank at zero, and the reader who
    opens that panel is checking exactly whether the control was unseeded.
    A caption that asserted "the control leg does not carry this field" for
    a field the control leg does carry would be false at the one moment it
    was being relied on, so the control frame is READ and the sentence
    states what was found: absent, present and zero everywhere, or present
    with its largest magnitude.
    """

    if name not in control.variables:
        return "The control leg does not carry this field at all."
    values = numpy.asarray(read_field(control, name, record), dtype=float)
    peak = float(numpy.nanmax(numpy.abs(values))) if values.size else 0.0
    if peak == 0.0:
        return (
            "The control leg carries this field too, and it is zero in every "
            "cell at this valid time."
        )
    return (
        f"The control leg carries this field too; its largest magnitude "
        f"there at this valid time is {peak:.3g}."
    )


# ---------------------------------------------------------------------------
# the mesh the panels are drawn on
# ---------------------------------------------------------------------------
@dataclass
class MeshGeometry:
    """Cell positions in degrees, and whatever the grid file added to them."""

    latitude: numpy.ndarray
    longitude: numpy.ndarray
    area: numpy.ndarray | None
    boundary: numpy.ndarray | None
    dc_edge: tuple[float, float] | None
    weighting: str

    @property
    def cells(self) -> int:
        return int(self.latitude.size)

    @property
    def sizes(self) -> numpy.ndarray | float:
        """Marker areas: proportional to cell area when one is known."""

        if self.area is None:
            return 14.0
        peak = float(numpy.max(self.area))
        if peak <= 0.0:  # pragma: no cover - a grid of zero-area cells
            return 14.0
        return 6.0 + 42.0 * (self.area / peak)


def read_geometry(frame: Path, grid: Path | None) -> MeshGeometry:
    """Cell centres from the frame; areas, the boundary zone and edge lengths
    from ``--grid`` when it is given.

    The frame is the authority for WHERE the cells are, because coordinates
    read out of the file being drawn cannot belong to a different mesh.  Areas
    are not carried by a history frame, so an area-weighted budget needs
    ``--grid``; without it the budget is a plain sum and every caption in the
    gallery says so instead of presenting a plain sum as an area integral.
    """

    dataset = _open(frame)
    try:
        for required in ("latCell", "lonCell"):
            if required not in dataset.variables:
                raise _refuse(
                    f"{frame.name} carries no {required}.  Every panel here "
                    f"places values on the mesh by cell centre; without the "
                    f"centres there is no map to draw, only an array index."
                )
        latitude = numpy.degrees(
            numpy.asarray(dataset.variables["latCell"][...], dtype=float)
        )
        longitude = numpy.degrees(
            numpy.asarray(dataset.variables["lonCell"][...], dtype=float)
        )
    finally:
        dataset.close()
    longitude = ((longitude + 180.0) % 360.0) - 180.0

    area: numpy.ndarray | None = None
    boundary: numpy.ndarray | None = None
    dc_edge: tuple[float, float] | None = None
    weighting = (
        "UNWEIGHTED: no --grid was given and a history frame carries no "
        "areaCell, so every budget below is a plain sum over cells and "
        "levels.  On a variable-resolution mesh that over-counts the refined "
        "region"
    )
    if grid is not None:
        if not Path(grid).is_file():
            raise _refuse(
                f"--grid {grid} is not a file.  The grid is where cell areas, "
                f"the boundary zone and the edge-length range come from; a "
                f"missing one would leave the budgets unweighted while the "
                f"command line said otherwise."
            )
        dataset = _open(Path(grid))
        try:
            if "areaCell" in dataset.variables:
                values = numpy.asarray(
                    dataset.variables["areaCell"][...], dtype=float
                )
                if values.size != latitude.size:
                    raise _refuse(
                        f"--grid {grid} has {values.size} cells and the "
                        f"frames have {latitude.size}.  That is a different "
                        f"mesh, and weighting one run's fields by another "
                        f"mesh's areas produces a number with no meaning."
                    )
                area = values
                weighting = (
                    f"area-weighted by areaCell from --grid {Path(grid).name}"
                )
            if "bdyMaskCell" in dataset.variables:
                mask = numpy.asarray(
                    dataset.variables["bdyMaskCell"][...]
                ).astype(int)
                if mask.size == latitude.size:
                    boundary = numpy.flatnonzero(mask > 0)
            if "dcEdge" in dataset.variables:
                lengths = numpy.asarray(
                    dataset.variables["dcEdge"][...], dtype=float
                )
                if lengths.size:
                    dc_edge = (float(lengths.min()), float(lengths.max()))
        finally:
            dataset.close()
    return MeshGeometry(
        latitude=latitude,
        longitude=longitude,
        area=area,
        boundary=boundary if boundary is not None and boundary.size else None,
        dc_edge=dc_edge,
        weighting=weighting,
    )


def convex_hull(longitude: numpy.ndarray, latitude: numpy.ndarray) -> numpy.ndarray:
    """Monotone-chain hull of the cell centres, as an ``(n, 2)`` lon/lat ring.

    Written out rather than imported so this tool stays on numpy, netCDF4 and
    matplotlib: a gallery that needs a fourth dependency is a gallery that
    does not run where the pair ran.
    """

    points = sorted({(float(x), float(y)) for x, y in zip(longitude, latitude)})
    if len(points) < 3:
        return numpy.asarray(points, dtype=float).reshape(-1, 2)

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    ring = lower[:-1] + upper[:-1]
    if ring:
        ring = ring + [ring[0]]
    return numpy.asarray(ring, dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------
# the panels
# ---------------------------------------------------------------------------
def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _frame_axes(axes, geometry: MeshGeometry) -> None:
    axes.set_xlabel("longitude (degrees east)")
    axes.set_ylabel("latitude (degrees north)")
    axes.grid(True, linewidth=0.3, alpha=0.35)
    span_x = float(geometry.longitude.max() - geometry.longitude.min()) or 1.0
    span_y = float(geometry.latitude.max() - geometry.latitude.min()) or 1.0
    axes.set_xlim(
        float(geometry.longitude.min()) - 0.05 * span_x,
        float(geometry.longitude.max()) + 0.05 * span_x,
    )
    axes.set_ylim(
        float(geometry.latitude.min()) - 0.05 * span_y,
        float(geometry.latitude.max()) + 0.05 * span_y,
    )


def _footnote(figure, text: str) -> None:
    """The sentence under the figure.

    Every caller that puts long rotated labels under its axes must reserve
    room for this line with ``subplots_adjust``: it is drawn at a fixed figure
    coordinate, and ``bbox_inches="tight"`` grows the image around the
    artists rather than pushing them apart, so an overlap here is an overlap
    in the delivered PNG.
    """

    figure.text(0.01, 0.012, text, fontsize=7.5, color="#444444")


def _save(figure, destination: Path) -> Path:
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    figure.clf()
    _pyplot().close(figure)
    return destination


def draw_extra_map(
    values: numpy.ndarray,
    geometry: MeshGeometry,
    *,
    destination: Path,
    field: str,
    valid_time: str,
    units: str,
    logarithmic: bool,
    reduction: str | None,
) -> Path:
    """One declared extra on the mesh: log with zeros masked, or linear."""

    from matplotlib.colors import LogNorm, Normalize

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(9.0, 6.4))
    finite = numpy.where(numpy.isfinite(values), values, 0.0)
    if logarithmic:
        # Zeros are MASKED rather than clamped: a log scale has no place to
        # put them, and drawing them at the bottom of the colour bar would
        # read as "a very small amount here" where there is none.  They are
        # masked by being left out of the coloured layer entirely, so the grey
        # base layer shows the mesh and matplotlib is never handed a masked
        # array to autoscale.
        above = numpy.flatnonzero(finite > 0.0)
        sizes = geometry.sizes
        axes.scatter(
            geometry.longitude, geometry.latitude,
            s=sizes, c="#e7e7e7", edgecolors="none", zorder=1,
        )
        if above.size == 0:
            axes.text(
                0.5, 0.5,
                "no value above zero at this valid time",
                transform=axes.transAxes, ha="center", va="center",
                fontsize=11, color="#666666",
            )
            mappable = None
        else:
            values_above = finite[above]
            top = float(values_above.max())
            bottom = float(values_above.min())
            if bottom >= top:
                bottom = top / 10.0
            mappable = axes.scatter(
                geometry.longitude[above], geometry.latitude[above],
                c=values_above,
                s=sizes if numpy.isscalar(sizes) else numpy.asarray(sizes)[above],
                cmap="viridis",
                norm=LogNorm(vmin=bottom, vmax=top),
                edgecolors="none", zorder=2,
            )
    else:
        top = float(numpy.max(finite)) if finite.size else 0.0
        bottom = float(numpy.min(finite)) if finite.size else 0.0
        if bottom >= top:
            top = bottom + 1.0
        mappable = axes.scatter(
            geometry.longitude, geometry.latitude,
            c=finite, s=geometry.sizes, cmap="viridis",
            norm=Normalize(vmin=bottom, vmax=top), edgecolors="none",
        )
    if mappable is not None:
        bar = figure.colorbar(mappable, ax=axes)
        scale = "log10" if logarithmic else "linear"
        bar.set_label(f"{field} [{units}], {scale} scale")
    _frame_axes(axes, geometry)
    reduced = f"column {reduction}" if reduction else "accumulated value"
    axes.set_title(
        f"{field}: {reduced}, treatment leg\nvalid {valid_time}",
        fontsize=11,
    )
    _footnote(figure, MODEL_RESULT_CLAUSE)
    return _save(figure, destination)


def draw_difference_map(
    difference: numpy.ndarray,
    geometry: MeshGeometry,
    *,
    destination: Path,
    field: str,
    valid_time: str,
    units: str,
    reduction: str,
) -> Path:
    """Treatment minus control, zero-centred and diverging."""

    from matplotlib.colors import Normalize

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(9.0, 6.4))
    finite = numpy.where(numpy.isfinite(difference), difference, 0.0)
    peak = float(numpy.max(numpy.abs(finite))) if finite.size else 0.0
    if peak <= 0.0:
        # No colour bar at all rather than an invented +/-1 scale in the
        # field's units: a reader reads the numbers off the bar, and a bar
        # whose limits were chosen because there was nothing to plot is an
        # instrument reporting a range the run never produced.
        axes.scatter(
            geometry.longitude, geometry.latitude,
            s=geometry.sizes, c="#e7e7e7",
            edgecolors="#b9b9b9", linewidths=0.25,
        )
        axes.text(
            0.5, 0.5,
            "the two legs are identical in this field at this valid time",
            transform=axes.transAxes, ha="center", va="center",
            fontsize=10, color="#666666",
        )
    else:
        # A cell whose difference is zero draws white on a white page, so
        # every marker keeps a hairline edge: the reader sees the mesh, and
        # "no difference here" is distinguishable from "no cell here".
        mappable = axes.scatter(
            geometry.longitude, geometry.latitude,
            c=finite, s=geometry.sizes, cmap="RdBu_r",
            norm=Normalize(vmin=-peak, vmax=peak),
            edgecolors="#b9b9b9", linewidths=0.25,
        )
        bar = figure.colorbar(mappable, ax=axes)
        bar.set_label(f"treatment minus control, {field} [{units}]")
    _frame_axes(axes, geometry)
    axes.set_title(
        f"{field}: column {reduction}, treatment minus control\n"
        f"valid {valid_time}",
        fontsize=11,
    )
    _footnote(figure, MODEL_RESULT_CLAUSE)
    return _save(figure, destination)


def draw_budget(
    series: Sequence[tuple[str, float]],
    *,
    destination: Path,
    field: str,
    units: str,
    weighting: str,
) -> Path:
    """The field summed over cells and levels, one point per selected frame."""

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(9.0, 5.2))
    values = [value for _, value in series]
    axes.plot(
        range(len(values)), values,
        marker="o", linewidth=1.6, color="#1f4e79",
    )
    axes.set_xticks(range(len(series)))
    axes.set_xticklabels([label for label, _ in series], rotation=30, ha="right", fontsize=8)
    # The reduction belongs in the title, not in the axis label: an axis
    # label longer than the axes is taller than the figure once it is
    # rotated, and the saved PNG cut the units off the end of it.
    axes.set_ylabel(f"{field} [{units}]")
    axes.set_xlabel("valid time")
    axes.grid(True, linewidth=0.3, alpha=0.35)
    axes.set_title(
        f"{field}: budget in time, treatment leg\n"
        f"summed over every cell and every level",
        fontsize=11,
    )
    # Room for the rotated valid times, the axis label and the footnote, in
    # that order: without it the footnote is drawn across the tick labels.
    figure.subplots_adjust(bottom=0.36)
    _footnote(figure, f"{weighting}.  {MODEL_RESULT_CLAUSE}")
    return _save(figure, destination)


def draw_overview(
    geometry: MeshGeometry,
    *,
    destination: Path,
    facts: Sequence[tuple[str, str]],
    outline_source: str,
) -> Path:
    """The domain, the cells, and the run's own identifying facts as text."""

    plt = _pyplot()
    figure = plt.figure(figsize=(12.0, 6.6))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    axes = figure.add_subplot(grid[0, 0])
    axes.scatter(
        geometry.longitude, geometry.latitude,
        s=6.0, c="#c9d6e3", edgecolors="none", zorder=1,
    )
    if geometry.boundary is not None:
        axes.scatter(
            geometry.longitude[geometry.boundary],
            geometry.latitude[geometry.boundary],
            s=12.0, c="#b03030", edgecolors="none", zorder=3,
        )
    else:
        ring = convex_hull(geometry.longitude, geometry.latitude)
        if ring.size:
            axes.plot(ring[:, 0], ring[:, 1], color="#b03030", linewidth=1.6, zorder=3)
    centroid = (
        float(numpy.mean(geometry.longitude)),
        float(numpy.mean(geometry.latitude)),
    )
    axes.plot(
        [centroid[0]], [centroid[1]],
        marker="x", markersize=10, color="#101010", zorder=4,
    )
    _frame_axes(axes, geometry)
    axes.set_title(f"domain outline: {outline_source}", fontsize=11)

    panel = figure.add_subplot(grid[0, 1])
    panel.axis("off")
    # Wrapped here rather than by matplotlib's own ``wrap=True``, which does
    # not break a long unbroken token: a file path has no spaces in it and ran
    # straight off the right edge of the page.
    lines: list[str] = []
    for label, value in facts:
        lines.append(label)
        wrapped = textwrap.wrap(
            str(value), width=64, break_long_words=True, break_on_hyphens=False
        ) or [""]
        lines += [f"    {piece}" for piece in wrapped]
    panel.text(
        0.0, 1.0, "\n".join(lines),
        transform=panel.transAxes, va="top", ha="left",
        fontsize=8.0, family="monospace",
    )
    _footnote(
        figure,
        "Mesh geometry drawn in plain latitude and longitude, no projection "
        "and no basemap.",
    )
    return _save(figure, destination)


# ---------------------------------------------------------------------------
# the gallery
# ---------------------------------------------------------------------------
def _png(out: Path, token: str, kind: str, field: str) -> Path:
    return out / f"{token}_{kind}_{field}.png"


def _prepare_out(out: Path) -> None:
    """A fresh directory, or one this tool can prove it wrote itself.

    THE BREAKAGE THIS PREVENTS: a second run over a shorter frame selection
    leaves the first run's panels sitting in the directory, unlisted by
    ``index.md`` and indistinguishable from this run's own.  A reader opening
    the folder sees valid times that are not part of this gallery.
    """

    out.mkdir(parents=True, exist_ok=True)
    previous = out / CAPTIONS_NAME
    if previous.is_file():
        try:
            recorded = json.loads(previous.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            recorded = {}
        if recorded.get("schema") == GALLERY_SCHEMA:
            for name in recorded.get("captions", {}):
                stale = out / str(name)
                if stale.is_file() and stale.suffix == ".png":
                    stale.unlink()
    remaining = sorted(path.name for path in out.glob("*.png"))
    if remaining:
        raise _refuse(
            f"--out {out} already holds {len(remaining)} PNG file(s) this "
            f"tool did not write (first: {remaining[0]}).  They would sit "
            f"beside this run's panels, unlisted by {INDEX_NAME}, and a "
            f"reader opening the folder would take them for valid times of "
            f"this pair.  Give a fresh directory."
        )


def build_gallery(
    *,
    pair_out: Path,
    out: Path,
    frames: int = 5,
    extras: Sequence[str] | None = None,
    grid: Path | None = None,
) -> dict[str, Any]:
    """Read one finished pair, draw the panels, write the captions and index."""

    pair_out = Path(pair_out).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    manifest = read_manifest(pair_out)
    control_frames = leg_frames(manifest, "control", pair_out)
    treatment_frames = leg_frames(manifest, "treatment", pair_out)
    matched, control_only, treatment_only = pair_frames(control_frames, treatment_frames)
    picked = [matched[index] for index in select_indices(len(matched), frames)]

    _prepare_out(out)
    geometry = read_geometry(picked[0].treatment.path, grid)

    first = picked[0]
    control = _open(first.control.path)
    treatment = _open(first.treatment.path)
    try:
        three_d, two_d = declared_extras(treatment, control, extras)
        shared = difference_fields(treatment, control)
        units = {
            name: _units(treatment, name)
            for name in list(three_d) + list(two_d) + list(shared)
        }
    finally:
        control.close()
        treatment.close()

    captions: dict[str, str] = {}
    order: list[str] = []
    budgets: dict[str, dict[str, Any]] = {
        name: {
            "units": units[name],
            "rank": 3 if name in three_d else 2,
            "weighting": geometry.weighting,
            "series": [],
        }
        for name in list(three_d) + list(two_d)
    }

    def _record(path: Path, caption: str) -> None:
        order.append(path.name)
        captions[path.name] = caption

    weighted = geometry.area is not None
    weight_words = (
        "area-weighted by cell area" if weighted else "unweighted (no cell areas)"
    )

    for frame in picked:
        treatment_set = _open(frame.treatment.path)
        control_set = _open(frame.control.path)
        try:
            for name in three_d:
                if name not in treatment_set.variables:
                    continue
                block = read_field(treatment_set, name, frame.treatment.record)
                column = _column(block, "maximum")
                destination = draw_extra_map(
                    column, geometry,
                    destination=_png(out, frame.token, "colmax", name),
                    field=name, valid_time=frame.valid_time,
                    units=units.get(name, _MISSING_UNITS),
                    logarithmic=True, reduction="maximum",
                )
                _record(
                    destination,
                    f"Column maximum of the declared extra scalar {name} "
                    f"[{units.get(name, _MISSING_UNITS)}] in the treatment "
                    f"leg at valid time {frame.valid_time}, drawn on the mesh "
                    f"with a log10 colour scale and cells at zero left blank. "
                    f"{control_presence(control_set, name, frame.control.record)} "
                    f"{MODEL_RESULT_CLAUSE}",
                )
                weight = (
                    geometry.area[:, None] if weighted else numpy.ones((block.shape[0], 1))
                )
                budgets[name]["series"].append(
                    {
                        "valid_time": frame.valid_time,
                        "value": float(numpy.sum(block * weight)),
                    }
                )
            for name in two_d:
                if name not in treatment_set.variables:
                    continue
                block = read_field(treatment_set, name, frame.treatment.record)
                destination = draw_extra_map(
                    numpy.asarray(block, dtype=float).ravel(), geometry,
                    destination=_png(out, frame.token, "accum", name),
                    field=name, valid_time=frame.valid_time,
                    units=units.get(name, _MISSING_UNITS),
                    logarithmic=False, reduction=None,
                )
                _record(
                    destination,
                    f"The declared extra accumulator {name} "
                    f"[{units.get(name, _MISSING_UNITS)}] in the treatment "
                    f"leg at valid time {frame.valid_time}, one value per "
                    f"cell on a linear colour scale.  "
                    f"{control_presence(control_set, name, frame.control.record)}  "
                    f"{MODEL_RESULT_CLAUSE}",
                )
                weight = geometry.area if weighted else numpy.ones(block.shape[0])
                budgets[name]["series"].append(
                    {
                        "valid_time": frame.valid_time,
                        "value": float(
                            numpy.sum(numpy.asarray(block, dtype=float).ravel() * weight)
                        ),
                    }
                )
            for name in shared:
                if name not in treatment_set.variables or name not in control_set.variables:
                    continue
                how = _reduction(name)
                difference = _column(
                    read_field(treatment_set, name, frame.treatment.record), how
                ) - _column(read_field(control_set, name, frame.control.record), how)
                destination = draw_difference_map(
                    difference, geometry,
                    destination=_png(out, frame.token, "diff", name),
                    field=name, valid_time=frame.valid_time,
                    units=units.get(name, _MISSING_UNITS),
                    reduction=how,
                )
                _record(
                    destination,
                    f"Treatment leg minus control leg, column {how} of "
                    f"{name} [{units.get(name, _MISSING_UNITS)}] at valid "
                    f"time {frame.valid_time}, on a diverging colour scale "
                    f"centred on zero: blue where the treatment leg is lower "
                    f"and red where it is higher.  {MODEL_RESULT_CLAUSE}",
                )
        finally:
            treatment_set.close()
            control_set.close()

    for name, record in budgets.items():
        series = [(row["valid_time"], row["value"]) for row in record["series"]]
        if not series:
            continue
        destination = draw_budget(
            series,
            destination=_png(out, picked[0].token, "budget", name),
            field=name,
            units=record["units"],
            weighting=geometry.weighting,
        )
        _record(
            destination,
            f"The declared extra {name} "
            f"[{record['units']}] summed over every cell and level of the "
            f"treatment leg, {weight_words}, one point per selected valid "
            f"time from {series[0][0]} to {series[-1][0]}.  It shows whether "
            f"the quantity is growing, settling or already gone.  "
            f"{MODEL_RESULT_CLAUSE}",
        )

    source_table = (manifest.get("source_table") or {}).get("path", "not recorded")
    outline_source = (
        "boundary-zone cells from bdyMaskCell in --grid"
        if geometry.boundary is not None
        else "convex hull of the cell centres"
    )
    facts: list[tuple[str, str]] = [
        ("pair directory", str(pair_out)),
        ("cells", f"{geometry.cells}"),
        (
            "mesh centroid",
            f"{numpy.mean(geometry.latitude):.4f} N, "
            f"{numpy.mean(geometry.longitude):.4f} E",
        ),
        (
            "latitude range",
            f"{geometry.latitude.min():.4f} to {geometry.latitude.max():.4f} N",
        ),
        (
            "longitude range",
            f"{geometry.longitude.min():.4f} to {geometry.longitude.max():.4f} E",
        ),
    ]
    if geometry.dc_edge is not None:
        facts.append(
            (
                "dcEdge range",
                f"{geometry.dc_edge[0]:.1f} to {geometry.dc_edge[1]:.1f} m",
            )
        )
    facts += [
        ("point-source table", str(source_table)),
        ("valid times matched", f"{len(matched)}"),
        ("valid times drawn", f"{len(picked)}"),
        ("declared extra scalars", ", ".join(three_d) or "none"),
        ("declared extra accumulators", ", ".join(two_d) or "none"),
        ("shared fields differenced", ", ".join(shared) or "none"),
        ("budget weighting", geometry.weighting),
    ]
    overview = draw_overview(
        geometry,
        destination=_png(out, picked[0].token, "overview", "domain"),
        facts=facts,
        outline_source=outline_source,
    )
    _record(
        overview,
        f"Where this pair ran: {geometry.cells} cells drawn at their centres "
        f"in plain latitude and longitude, the {outline_source} in red, the "
        f"mesh centroid marked, and beside them the run's own facts including "
        f"the point-source table the treatment leg carried.  Geometry only; "
        f"no model field is shown.",
    )

    result: dict[str, Any] = {
        "schema": GALLERY_SCHEMA,
        "tool": "plot_pair_gallery.py",
        "pair_out": str(pair_out),
        "out": str(out),
        "grid": str(grid) if grid is not None else None,
        "pair_schema": manifest.get("schema"),
        "source_table": source_table,
        "result_sentence": RESULT_SENTENCE,
        "render_note": RENDER_SENTENCE,
        "cells": geometry.cells,
        "weighting": geometry.weighting,
        "valid_times_matched": [frame.valid_time for frame in matched],
        "valid_times_drawn": [frame.valid_time for frame in picked],
        "valid_times_control_only": control_only,
        "valid_times_treatment_only": treatment_only,
        "declared_extras": {
            "three_dimensional": list(three_d),
            "two_dimensional": list(two_d),
            "source": "--extras" if extras is not None else "discovered",
        },
        "difference_fields": list(shared),
        "units": units,
        "budgets": budgets,
        "pngs": list(order),
        "captions": captions,
    }
    (out / CAPTIONS_NAME).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_index(out, result)
    return result


def write_index(out: Path, result: dict[str, Any]) -> Path:
    """``index.md``: every PNG in the order it was drawn, with its caption."""

    lines = [
        "# Paired-run evidence gallery",
        "",
        RESULT_SENTENCE,
        "",
        RENDER_SENTENCE,
        "",
        f"- pair directory: `{result['pair_out']}`",
        f"- point-source table: `{result['source_table']}`",
        f"- cells: {result['cells']}",
        f"- valid times matched: {len(result['valid_times_matched'])}, "
        f"drawn: {len(result['valid_times_drawn'])}",
        f"- budget weighting: {result['weighting']}",
        f"- declared extra scalars: "
        f"{', '.join(result['declared_extras']['three_dimensional']) or 'none'}",
        f"- declared extra accumulators: "
        f"{', '.join(result['declared_extras']['two_dimensional']) or 'none'}",
        f"- shared fields differenced: "
        f"{', '.join(result['difference_fields']) or 'none'}",
        "",
    ]
    for name in result["pngs"]:
        lines += [
            f"## {name}",
            "",
            f"![{name}]({name})",
            "",
            result["captions"][name],
            "",
        ]
    destination = out / INDEX_NAME
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return destination


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_pair_gallery.py",
        description=(
            "Turn a finished paired run into a small curated evidence set: "
            "column-maximum maps of every declared extra scalar, maps of "
            "every declared extra accumulator, a budget-in-time line per "
            "extra, treatment-minus-control difference maps of the shared "
            "weather fields, and a one-page overview of the domain.  Every "
            "panel is an analysis chart of a model result, and every caption "
            "says so."
        ),
        epilog=(
            RENDER_SENTENCE
            + "  PNG names are <valid-time>_<kind>_<field>.png, where kind is "
            "one of colmax, accum, budget, diff or overview."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pair-out", type=Path, required=True, metavar="DIR",
        help="the finished pair directory: the one holding "
             f"{MANIFEST_NAME} and both legs' trees",
    )
    parser.add_argument(
        "--out", type=Path, required=True, metavar="DIR",
        help=f"where the PNGs, {CAPTIONS_NAME} and {INDEX_NAME} are written",
    )
    parser.add_argument(
        "--frames", type=int, default=5, metavar="N",
        help="how many valid times to draw: N evenly spaced ones including "
             "the first and the last",
    )
    parser.add_argument(
        "--extras", type=str, default=None, metavar="NAME,...",
        help="comma-separated field names to treat as the declared extras, "
             "replacing the discovered list (which is every per-cell field "
             "the treatment leg carries and the control leg does not)",
    )
    parser.add_argument(
        "--grid", type=Path, default=None, metavar="PATH",
        help="the mesh file the run was bound to.  It supplies areaCell for "
             "the budgets and the marker sizes, bdyMaskCell for the domain "
             "outline and dcEdge for the edge-length range; without it the "
             "budgets are plain sums and every caption says so",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    extras = (
        [name.strip() for name in str(arguments.extras).split(",") if name.strip()]
        if arguments.extras
        else None
    )
    try:
        result = build_gallery(
            pair_out=arguments.pair_out,
            out=arguments.out,
            frames=int(arguments.frames),
            extras=extras,
            grid=arguments.grid,
        )
    except GalleryRefusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr, flush=True)
        return 2
    print(
        f"GALLERY {result['out']} "
        f"{len(result['pngs'])} panel(s) over "
        f"{len(result['valid_times_drawn'])} valid time(s)",
        flush=True,
    )
    print(f"CAPTIONS {Path(result['out']) / CAPTIONS_NAME}", flush=True)
    print(f"INDEX {Path(result['out']) / INDEX_NAME}", flush=True)
    print(RESULT_SENTENCE, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command line
    raise SystemExit(main())
