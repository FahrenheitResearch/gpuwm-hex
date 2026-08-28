"""Published MPAS mesh loading and topology/geometry validation.

MPAS writes connectivity as one-based indices with zero denoting a missing
entry.  The authority representation in this package is zero-based with
``-1`` padding.  Padding is canonicalized from the corresponding count field
because some otherwise-valid MPAS mesh files repeat the final used entry in
unused ``cellsOnCell`` and ``edgesOnCell`` slots.

The topology conventions checked here are the conventions consumed by
``atm_compute_signs`` in frozen MPAS-A v8.2.3:
``src/core_atmosphere/mpas_atm_core.F:1137-1224``.  The metric relationships
are the MPAS C-grid discretization represented by the mesh-spec fields.

A REGIONAL (limited-area) mesh -- a bounded disk culled from a parent sphere,
carrying the ``bdyMaskCell/Edge/Vertex`` triple WITH A NONEMPTY BOUNDARY ZONE
-- is validated against the conventions measured on native
MPAS-Limited-Area culls of two published meshes (2026-08-25, real bytes,
both culls 100 percent).  The zone, not the schema, is what makes a mesh
regional: native MPAS-A writes the same triple all-zero into a GLOBAL mesh's
static file and the unified ``rw_mpas_static`` follows that convention on
every static this project generates, so classifying on presence alone refused
both the published x1.40962 and every generated global mesh as a corrupt cull
(measured 2026-08-26; see :meth:`Mesh.validate`).  The conventions:

* masks are integers 0 (interior) through 7 (outermost ring), edge/vertex
  masks are the MINIMUM of their present cells' masks, neighbouring cell
  masks never differ by more than 1, and every ring-``k`` cell touches at
  least one ring-``k-1`` cell.  That last one used to be "ring populations
  grow outward", which holds on a uniform mesh and fails on a variable-
  resolution one -- see :func:`regional_ring_shell_errors` for the three
  culls that measured it;
* absent-element sentinels (file zeros, canonical ``-1``) appear only on
  ring-7 elements and only in the five arrays of
  :data:`REGIONAL_SENTINEL_ARRAYS`; every other connectivity array is
  complete because the cull keeps every edge and vertex of a kept cell
  (``nEdgesOnEdge`` is NOT shrunk -- sentinels sit inside the declared row);
* the Euler characteristic ``nCells - nEdges + nVertices`` is 1 (a disk),
  not the closed sphere's 2, and the whole-sphere area identities do not
  apply -- the local kite/area identities and the arc-length checks still do.

Carrying the triple is NOT what makes a mesh regional; carrying a boundary
ZONE is.  MPAS writes an all-zero ``bdyMask`` triple on a global mesh (every
element interior, no ring at all), and the unified ``rw_mpas_static`` writer
follows that convention, so every static this project GENERATES ships the
triple.  Presence alone therefore classified a closed sphere as a disk and
refused it for having Euler characteristic 2 and empty rings 1..7 -- which is
exactly what a global mesh is.  MEASURED 2026-08-26 on the generated graded
mesh ``v20.80.151649``: bound clean, then refused at load by that rule, as
would every generated-static row (``u96.64002``, every graded row).  The test
is a NONZERO mask value, and it is the whole difference; an INCOMPLETE triple
stays refused on presence, because a half-written triple cannot say which
sentinels are boundary and which are corruption.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator, Mapping

import numpy as np
from netCDF4 import Dataset
from numpy.typing import NDArray

from .errors import MeshValidationError


_COUNTED_CONNECTIVITY = {
    "cellsOnCell": "nEdgesOnCell",
    "edgesOnCell": "nEdgesOnCell",
    "verticesOnCell": "nEdgesOnCell",
    "edgesOnEdge": "nEdgesOnEdge",
    "advCellsForEdge": "nAdvCellsForEdge",
}

_CORE_CONNECTIVITY = {
    "cellsOnCell",
    "edgesOnCell",
    "verticesOnCell",
    "cellsOnEdge",
    "verticesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
    "edgesOnEdge",
}

#: The mask triple a regional (limited-area) cull adds.  All three or none:
#: a partial triple cannot say which sentinels are boundary and which are
#: corruption, so it is refused rather than guessed at.
REGIONAL_BOUNDARY_MASK_NAMES = ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex")

#: Rings 1..7, outermost ring 7 touching the cut, 0 interior.  Measured on
#: the native culls; MPAS-A v8.4.1 stages its specified/relaxation zones on
#: exactly these seven rings.
REGIONAL_BOUNDARY_ZONE_WIDTH = 7


def _regional_boundary_zone_present(arrays: Mapping[str, Any]) -> bool:
    """True when the complete mask triple carries at least one ring element.

    THE BREAKAGE THIS PREVENTS (measured 2026-08-26, ``v20.80.151649`` on
    the proving RTX 5090): an all-zero triple is MPAS's global convention and the
    unified ``rw_mpas_static`` writes it on every generated static, so
    classifying on PRESENCE refused every generated global mesh at load --
    "not a bounded disk: nCells-nEdges+nVertices = 2, not 1" and "rings
    [1..7] are empty", which is the correct description of a sphere.
    """

    if not all(name in arrays for name in REGIONAL_BOUNDARY_MASK_NAMES):
        return False
    return any(
        bool(np.any(np.asarray(arrays[name]) != 0))
        for name in REGIONAL_BOUNDARY_MASK_NAMES
    )

#: The only connectivity arrays in which a native cull stores an
#: absent-element sentinel (file zero, canonical -1), and only on ring-7
#: rows.  ``edgesOnCell``, ``verticesOnCell`` and ``verticesOnEdge`` are
#: complete by construction -- the cull keeps every edge and vertex of a
#: kept cell -- so a sentinel there is a lost element, never a boundary.
REGIONAL_SENTINEL_ARRAYS = (
    "cellsOnCell",
    "cellsOnEdge",
    "edgesOnEdge",
    "cellsOnVertex",
    "edgesOnVertex",
)

#: Ceiling on the RELATIVE half of the arc-length tolerance.  Eight binary32
#: epsilons: a stored binary32 length rounds by at most half an epsilon, so
#: this carries sixteen such roundings and nothing wider.  It is a ceiling
#: rather than the value, so a binary64 mesh keeps its own tighter bound.
_SPHERICAL_ARC_MAX_RTOL = 8.0 * float(np.finfo(np.float32).eps)

LONGITUDE_TRIG_EQUIVALENCE_ATOL = 8.0 * np.finfo(np.float64).eps
GRID_ROTATE_SOURCE_SHA256 = (
    "2be1c67cd2700ffd65b41f241c8858c0e24ca2b67bc7655465ef3807ab654d36"
)
GRID_ROTATE_SOURCE_HEAD = "4b5c11b4be471498da36a2637ad1cf49962b3d05"
GRID_ROTATE_DEFAULT_REAL_PI = float(
    np.float32(2.0) * np.arcsin(np.float32(1.0))
)
GRID_ROTATE_PI_DELTA = GRID_ROTATE_DEFAULT_REAL_PI - float(np.pi)
GRID_ROTATE_CONVERT_XL_EPS = float(np.float32(1.0e-10))
GRID_ROTATE_PRODUCER_MAX_ULP_ERROR = 2.0
GRID_ROTATE_LATITUDE_MAX_ULP_ERROR = 256.0
GRID_ROTATE_CORRECTED_ATAN2_ATOL = 16.0 * np.finfo(np.float64).eps
_BINARY64_MESH_RTOL = 5.0e-10
_GEOMETRY_COORDINATES = tuple(
    f"{component}{entity}"
    for entity in ("Cell", "Edge", "Vertex")
    for component in ("x", "y", "z", "lat", "lon")
)
_TOPOLOGY_WITNESSES = (
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


def _camel_to_snake(name: str) -> str:
    first = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first).lower()


def _plain_attribute(value: Any) -> Any:
    """Detach a netCDF attribute from its open Dataset."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_netcdf(
    path: Path,
) -> tuple[
    dict[str, int],
    dict[str, Any],
    dict[str, NDArray[Any]],
    dict[str, tuple[str, ...]],
    dict[str, dict[str, Any]],
]:
    dimensions: dict[str, int] = {}
    attributes: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {}
    variable_dimensions: dict[str, tuple[str, ...]] = {}
    variable_attributes: dict[str, dict[str, Any]] = {}

    with Dataset(path, mode="r") as dataset:
        dataset.set_auto_mask(False)
        dimensions.update({name: len(dim) for name, dim in dataset.dimensions.items()})
        attributes.update(
            {name: _plain_attribute(getattr(dataset, name)) for name in dataset.ncattrs()}
        )
        for name, variable in dataset.variables.items():
            arrays[name] = np.array(variable[...], copy=True)
            variable_dimensions[name] = tuple(variable.dimensions)
            variable_attributes[name] = {
                attr: _plain_attribute(getattr(variable, attr))
                for attr in variable.ncattrs()
            }

    return dimensions, attributes, arrays, variable_dimensions, variable_attributes


def _looks_like_connectivity(name: str, array: NDArray[Any]) -> bool:
    if name in _CORE_CONNECTIVITY:
        return True
    if array.dtype.kind not in "iu" or name.startswith("indexTo"):
        return False
    if re.match(r"^n[A-Z]", name):
        return False
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "cellson",
            "cellsfor",
            "edgeson",
            "edgesfor",
            "verticeson",
            "verticesfor",
        )
    )


def _canonicalize_connectivity(arrays: dict[str, NDArray[Any]]) -> tuple[str, ...]:
    converted: list[str] = []
    for name, source in tuple(arrays.items()):
        if not _looks_like_connectivity(name, source):
            continue
        arrays[name] = np.asarray(source, dtype=np.int64) - 1
        converted.append(name)

    for name, count_name in _COUNTED_CONNECTIVITY.items():
        if name not in arrays or count_name not in arrays:
            continue
        array = arrays[name]
        counts = np.asarray(arrays[count_name], dtype=np.int64)
        if array.ndim != 2 or counts.shape != (array.shape[0],):
            continue
        padding = np.arange(array.shape[1])[None, :] >= counts[:, None]
        array[padding] = -1

    return tuple(sorted(converted))


def _bad_examples(indices: NDArray[Any], limit: int = 5) -> str:
    flat = np.asarray(indices).ravel()
    return ", ".join(str(int(index)) for index in flat[:limit])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    """Hash an array with its dtype and shape, not just its raw payload."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def regional_boundary_mask_digest(masks: Mapping[str, Any]) -> str:
    """SHA-256 of the bdyMask triple a regional registry row pins.

    Fixed encoding -- each name in :data:`REGIONAL_BOUNDARY_MASK_NAMES`
    order, a NUL, then the little-endian int32 payload -- so the digest a
    ``mesh-check`` receipt prints is byte-for-byte the digest a registry row
    declares and the bind verifies.  Two definitions would eventually
    disagree and admit a mesh whose rings nobody checked.
    """

    digest = hashlib.sha256()
    for name in REGIONAL_BOUNDARY_MASK_NAMES:
        if name not in masks:
            raise MeshValidationError(
                f"regional mask digest requires {name}; a partial triple "
                "cannot identify the boundary rings"
            )
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            np.ascontiguousarray(np.asarray(masks[name], dtype="<i4")).tobytes(
                order="C"
            )
        )
    return digest.hexdigest()


def regional_ring_shell_errors(
    mask_cell: NDArray[np.int64],
    source_cells: NDArray[np.int64],
    neighbour_cells: NDArray[np.int64],
    zone_width: int,
) -> list[str]:
    """Refuse a boundary zone whose rings do not sit on the ring inside them.

    THE BREAKAGE THIS PREVENTS: a torn or renumbered relaxation zone.  Ring
    ``k`` is staged against ring ``k-1`` every step; a ring-``k`` cell with no
    ring-``k-1`` neighbour is forced from a ring that does not touch it, and
    the boundary blend has a hole nothing announces.

    THIS REPLACES A TEST THAT DID NOT IMPLY ITS OWN CLAIM, and the counter-
    example is measured (2026-08-27, the proving RTX 5090, evidence/nest-ratio-20260827/).
    The check here used to be that ring POPULATIONS grow outward, on the
    stated reasoning that "each ring wraps the previous in a growing shell".
    Populations are a proxy for wrapping on a UNIFORM mesh only, and the only
    meshes this program culls are variable resolution.  Three culls of one
    parent, one binary, one run -- one of them byte-identical to the
    registered ``r4.75.11020`` -- were handed to the old check and two were
    refused:

    ======  ==============================  ==============================
    cull    ring cell counts 1..7           ring mean cell width, km
    ======  ==============================  ==============================
    d045    207 209 212 215 218 217 225     5.03 -> 5.49
    d070    254 251 249 242 241 237 232     6.31 -> 7.71
    d100    240 245 251 257 263 269 275     8.99 -> 9.65
    ======  ==============================  ==============================

    ``d070``'s rings grow outward into the parent's coarsening ramp: each ring
    is 2.9 per cent wider than the one inside it, so the shell's PERIMETER
    grows 11.6 per cent over the seven rings while its POPULATION falls 8.7
    per cent.  Fewer, larger cells wrapping a larger circumference is not a
    tear.  The arithmetic is exact -- with a perimeter of ``P`` cells, one ring
    outward gives ``(P + 2*pi) / (1 + g)`` for a fractional width growth ``g``,
    which shrinks whenever ``g > 2*pi/P`` -- so on any cull big enough to
    matter, a few per cent of grading per ring inverts the old test.

    Every one of the three satisfies the property asserted here, and so does
    every native cull the conventions were measured on.  The claims the old
    test was reaching for are kept and are checked elsewhere: no empty ring,
    neighbouring masks never differing by more than one, and Euler
    characteristic 1.
    """

    errors: list[str] = []
    inner = mask_cell[neighbour_cells] == mask_cell[source_cells] - 1
    touches_inner = np.zeros(mask_cell.size, dtype=bool)
    touches_inner[source_cells[inner]] = True
    for ring in range(1, zone_width + 1):
        members = np.flatnonzero(mask_cell == ring)
        if members.size == 0:
            continue  # empty rings are reported by their own check
        orphans = members[~touches_inner[members]]
        if orphans.size:
            errors.append(
                f"bdyMaskCell ring {ring} has cells with no ring-{ring - 1} "
                "neighbour at "
                + _bad_examples(orphans)
                + f": ring {ring} is staged against ring {ring - 1} every "
                "step, so a cell that touches no cell of the ring inside it "
                "is blended from a ring that does not reach it -- a torn or "
                "renumbered zone"
            )
    return errors


def spherical_arc_tolerance(
    radius: float, coordinate_dtype: Any, metric_rtol: float = _SPHERICAL_ARC_MAX_RTOL
) -> tuple[float, float]:
    """Tolerance for comparing a stored edge length against a recomputed arc.

    The comparison has two independent error sources and a purely relative
    bound can only express one of them.

    * The stored length rounds to its own storage precision.  That is a
      RELATIVE error and ``metric_rtol`` carries it.
    * The Cartesian coordinates the arc is recomputed FROM are quantized at
      the sphere radius, and that error does not shrink with the edge.  At
      binary32 and Earth radius the coordinate spacing is
      ``np.spacing(float32(6371220.0)) == 0.5`` metres whether the edge is
      8 km or 800 km long, so it is an ABSOLUTE floor.

    Deriving the floor: each of an endpoint's three components rounds by at
    most half a coordinate spacing, bounding the endpoint's displacement by
    ``sqrt(3)/2`` spacings; the two endpoints of an edge can move in opposite
    directions along it, so the arc carries at most ``sqrt(3)`` spacings.  The
    factor of two on top carries a mesh whose coordinates were rounded a
    second time -- :func:`load_precision_preserving_mesh_pair` rescales them
    between two sphere radii, which is exactly that second rounding.

    Without the floor a defect-free mesh is refused for its own storage
    precision: at 8.7 km vertex spacing the quantization alone is ~1e-5 of the
    edge, which sits on top of a 2.0e-5 relative bound and makes admission a
    coin flip.  With the floor named, the relative half no longer has to
    stand in for it, so it is capped at what binary32 storage actually needs
    (``8 * eps`` ~ 9.5e-7) rather than the 2.0e-5 that let metre-scale
    corruption hide at coarse spacing.
    """

    spacing = float(np.spacing(np.asarray(abs(radius), dtype=coordinate_dtype)))
    return min(float(metric_rtol), _SPHERICAL_ARC_MAX_RTOL), 2.0 * math.sqrt(3.0) * spacing


def _wrapped_angle_delta(left: Any, right: Any) -> NDArray[np.float64]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    return np.asarray((first - second + np.pi) % (2.0 * np.pi) - np.pi)


def _maximum_ulp_error(actual: Any, expected: Any) -> float:
    candidate = np.asarray(actual, dtype=np.float64)
    authority = np.asarray(expected, dtype=np.float64)
    if candidate.shape != authority.shape:
        return float("inf")
    gaps = np.abs(candidate - authority)
    spacing = np.abs(np.spacing(authority))
    with np.errstate(over="ignore", invalid="ignore"):
        ratios = np.divide(
            gaps,
            spacing,
            out=np.full(gaps.shape, np.inf, dtype=np.float64),
            where=spacing > 0.0,
        )
    ratios[gaps == 0.0] = 0.0
    return float(np.max(ratios, initial=0.0))


def _grid_rotate_convert_xl_longitude(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the pinned Fortran ``convert_xl`` longitude arithmetic.

    The producer stores its result in binary64, but ``pii = 2.*asin(1.0)`` is
    evaluated in default-real binary32.  ``multiplicity`` records how many
    copies of the resulting pi error each literal quadrant branch introduces.
    """

    longitude = np.zeros(x.shape, dtype=np.float64)
    multiplicity = np.zeros(x.shape, dtype=np.float64)
    x_active = np.abs(x) > GRID_ROTATE_CONVERT_XL_EPS
    y_active = np.abs(y) > GRID_ROTATE_CONVERT_XL_EPS
    both_active = x_active & y_active
    both_indices = np.flatnonzero(both_active)
    base = np.arctan(np.abs(y[both_active] / x[both_active]))
    longitude[both_active] = base

    second_quadrant = (x[both_indices] <= 0.0) & (y[both_indices] >= 0.0)
    indices = both_indices[second_quadrant]
    longitude[indices] = GRID_ROTATE_DEFAULT_REAL_PI - base[second_quadrant]
    multiplicity[indices] = 1.0

    third_quadrant = (x[both_indices] <= 0.0) & (y[both_indices] < 0.0)
    indices = both_indices[third_quadrant]
    longitude[indices] = base[third_quadrant] + GRID_ROTATE_DEFAULT_REAL_PI
    multiplicity[indices] = 1.0

    fourth_quadrant = (x[both_indices] >= 0.0) & (y[both_indices] <= 0.0)
    indices = both_indices[fourth_quadrant]
    longitude[indices] = 2.0 * GRID_ROTATE_DEFAULT_REAL_PI - base[fourth_quadrant]
    multiplicity[indices] = 2.0

    x_axis = x_active & ~y_active
    negative_x_axis = x_axis & (x <= 0.0)
    longitude[negative_x_axis] = GRID_ROTATE_DEFAULT_REAL_PI
    multiplicity[negative_x_axis] = 1.0

    y_axis = ~x_active & y_active
    positive_y_axis = y_axis & (y > 0.0)
    longitude[positive_y_axis] = GRID_ROTATE_DEFAULT_REAL_PI / 2.0
    multiplicity[positive_y_axis] = 0.5
    negative_y_axis = y_axis & (y <= 0.0)
    longitude[negative_y_axis] = 3.0 * GRID_ROTATE_DEFAULT_REAL_PI / 2.0
    multiplicity[negative_y_axis] = 1.5
    return longitude, multiplicity


def validate_longitude_normalization(
    source: Any, normalized: Any
) -> dict[str, float | str]:
    """Prove a ``[-pi, pi)`` longitude normalization preserves sin/cos."""

    original = np.asarray(source, dtype=np.float64)
    candidate = np.asarray(normalized, dtype=np.float64)
    if (
        original.shape != candidate.shape
        or not np.all(np.isfinite(original))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("longitude normalization requires same-shaped finite arrays")
    expected = (original + np.pi) % (2.0 * np.pi) - np.pi
    if not np.array_equal(candidate, expected):
        raise ValueError("longitude normalization is not the exact modulo mapping")
    if np.any(candidate < -np.pi) or np.any(candidate >= np.pi):
        raise ValueError("normalized longitude lies outside [-pi, pi)")
    sin_gap = float(np.max(np.abs(np.sin(candidate) - np.sin(original)), initial=0.0))
    cos_gap = float(np.max(np.abs(np.cos(candidate) - np.cos(original)), initial=0.0))
    if (
        sin_gap > LONGITUDE_TRIG_EQUIVALENCE_ATOL
        or cos_gap > LONGITUDE_TRIG_EQUIVALENCE_ATOL
    ):
        raise ValueError("longitude modulo mapping is not trig-equivalent")
    return {
        "method": "(longitude + pi) modulo 2pi minus pi",
        "sin_max_abs_gap": sin_gap,
        "cos_max_abs_gap": cos_gap,
        "trig_equivalence_atol": LONGITUDE_TRIG_EQUIVALENCE_ATOL,
    }


def normalize_longitudes(source: Any) -> NDArray[np.float64]:
    """Return the exact evidenced longitude normalization in binary64."""

    original = np.asarray(source, dtype=np.float64)
    normalized = np.ascontiguousarray((original + np.pi) % (2.0 * np.pi) - np.pi)
    validate_longitude_normalization(original, normalized)
    return normalized


class Mesh:
    """In-memory MPAS mesh with canonical Python connectivity.

    Arrays retain their official MPAS camel-case names in :attr:`arrays` and
    are also accessible as attributes in either camel case (``cellsOnEdge``)
    or snake case (``cells_on_edge``).  The same applies to dimensions and
    global attributes.
    """

    def __init__(
        self,
        *,
        arrays: Mapping[str, NDArray[Any]],
        dimensions: Mapping[str, int],
        attrs: Mapping[str, Any],
        grid_attrs: Mapping[str, Any] | None = None,
        static_attrs: Mapping[str, Any] | None = None,
        variable_dimensions: Mapping[str, tuple[str, ...]] | None = None,
        variable_attrs: Mapping[str, Mapping[str, Any]] | None = None,
        variable_sources: Mapping[str, str] | None = None,
        grid_path: str | Path | None = None,
        static_path: str | Path | None = None,
        converted_connectivity: tuple[str, ...] = (),
    ) -> None:
        self.arrays = {name: np.asarray(value) for name, value in arrays.items()}
        self.variables = self.arrays
        self.dimensions = dict(dimensions)
        self.attrs = dict(attrs)
        self.grid_attrs = dict(grid_attrs or attrs)
        self.static_attrs = dict(static_attrs or {})
        self.variable_dimensions = dict(variable_dimensions or {})
        self.variable_attrs = {
            name: dict(value) for name, value in (variable_attrs or {}).items()
        }
        self.variable_sources = dict(variable_sources or {})
        self.grid_path = None if grid_path is None else str(Path(grid_path).resolve())
        self.static_path = None if static_path is None else str(Path(static_path).resolve())
        self.converted_connectivity = tuple(converted_connectivity)
        self.provenance = {
            "grid_path": self.grid_path,
            "static_path": self.static_path,
            "grid_attrs": self.grid_attrs,
            "static_attrs": self.static_attrs,
            "variable_sources": self.variable_sources,
            "connectivity_indexing": "zero-based with -1 padding",
        }
        self._aliases: dict[str, str] = {}
        for name in (*self.arrays, *self.dimensions, *self.attrs):
            self._aliases.setdefault(_camel_to_snake(name), name)

    @classmethod
    def from_netcdf(
        cls,
        grid_path: str | Path,
        static_path: str | Path | None = None,
        *,
        validate: bool = True,
    ) -> "Mesh":
        """Load an official MPAS grid and optionally overlay a static file.

        All variables from the grid are retained.  A static file contributes
        all of its variables and replaces same-named grid variables, which is
        important because published grid files commonly use a unit sphere
        while initialized static files contain Earth-scaled metrics and useful
        fields such as ``ter``, ``landmask``, and ``coeffs_reconstruct``.
        Per-file attributes and per-variable source labels remain available in
        :attr:`provenance`.
        """

        grid = Path(grid_path).expanduser().resolve(strict=True)
        (
            dimensions,
            grid_attrs,
            arrays,
            variable_dimensions,
            variable_attrs,
        ) = _read_netcdf(grid)
        sources = {name: "grid" for name in arrays}
        effective_attrs = dict(grid_attrs)
        static_attrs: dict[str, Any] = {}
        static: Path | None = None

        if static_path is not None:
            static = Path(static_path).expanduser().resolve(strict=True)
            (
                static_dimensions,
                static_attrs,
                static_arrays,
                static_variable_dimensions,
                static_variable_attrs,
            ) = _read_netcdf(static)

            incompatible = {
                name: (dimensions[name], size)
                for name, size in static_dimensions.items()
                if name in dimensions and dimensions[name] != size
            }
            if incompatible:
                raise MeshValidationError(
                    f"static/grid dimensions disagree: {incompatible}"
                )

            grid_id = str(grid_attrs.get("file_id", "")).strip()
            parent_id = str(static_attrs.get("parent_id", "")).strip()
            if grid_id and parent_id and grid_id != parent_id:
                raise MeshValidationError(
                    "static parent_id does not match grid file_id: "
                    f"{parent_id!r} != {grid_id!r}"
                )

            dimensions.update(static_dimensions)
            effective_attrs.update(static_attrs)
            arrays.update(static_arrays)
            variable_dimensions.update(static_variable_dimensions)
            variable_attrs.update(static_variable_attrs)
            sources.update({name: "static" for name in static_arrays})

        converted = _canonicalize_connectivity(arrays)
        mesh = cls(
            arrays=arrays,
            dimensions=dimensions,
            attrs=effective_attrs,
            grid_attrs=grid_attrs,
            static_attrs=static_attrs,
            variable_dimensions=variable_dimensions,
            variable_attrs=variable_attrs,
            variable_sources=sources,
            grid_path=grid,
            static_path=static,
            converted_connectivity=converted,
        )
        if validate:
            mesh.validate()
        return mesh

    def __getattr__(self, name: str) -> Any:
        arrays = self.__dict__.get("arrays", {})
        if name in arrays:
            return arrays[name]
        dimensions = self.__dict__.get("dimensions", {})
        if name in dimensions:
            return dimensions[name]
        attrs = self.__dict__.get("attrs", {})
        if name in attrs:
            return attrs[name]
        official = self.__dict__.get("_aliases", {}).get(name)
        if official in arrays:
            return arrays[official]
        if official in dimensions:
            return dimensions[official]
        if official in attrs:
            return attrs[official]
        raise AttributeError(f"Mesh has no MPAS field, dimension, or attribute {name!r}")

    def __getitem__(self, name: str) -> Any:
        try:
            return getattr(self, name)
        except AttributeError as error:
            raise KeyError(name) from error

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        try:
            getattr(self, name)
        except AttributeError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def copy(self, *, deep: bool = True) -> "Mesh":
        """Copy the mesh; deep copies are convenient for mutation controls."""

        arrays = {
            name: value.copy() if deep else value for name, value in self.arrays.items()
        }
        return type(self)(
            arrays=arrays,
            dimensions=self.dimensions,
            attrs=self.attrs,
            grid_attrs=self.grid_attrs,
            static_attrs=self.static_attrs,
            variable_dimensions=self.variable_dimensions,
            variable_attrs=self.variable_attrs,
            variable_sources=self.variable_sources,
            grid_path=self.grid_path,
            static_path=self.static_path,
            converted_connectivity=self.converted_connectivity,
        )

    @property
    def is_regional(self) -> bool:
        """True when the complete ``bdyMask`` triple carries a boundary ZONE.

        A global mesh may carry the triple all-zero -- that is MPAS's own
        convention for "every element interior", and the unified
        ``rw_mpas_static`` writer emits it on every static this project
        generates.  Classifying on presence alone made a closed sphere a
        disk; see the module docstring for the measured refusal.
        """

        return _regional_boundary_zone_present(self.arrays)

    def validate(self) -> "Mesh":
        """Validate the complete MPAS C-grid contract, global or regional.

        Validation deliberately includes both local reciprocity and global
        metric identities.  A corrupt mesh therefore cannot pass merely
        because every integer happens to be in range.

        A mesh carrying the full ``bdyMaskCell/Edge/Vertex`` triple is a
        regional cull and is validated against the measured regional
        conventions (module docstring): sentinel-tolerant ranges keyed to
        :data:`REGIONAL_SENTINEL_ARRAYS` with a ring-7 placement rule,
        reciprocity exempted only where an absent neighbour makes it
        undefined, Euler characteristic 1 instead of 2, the mask-derivation
        rules, and rings that each sit on the ring inside them.  Every refusal names
        the concrete breakage instead of reporting the disk as a corrupt
        sphere.
        """

        errors: list[str] = []

        required_dimensions = (
            "nCells",
            "nEdges",
            "nVertices",
            "maxEdges",
            "maxEdges2",
            "vertexDegree",
        )
        missing_dimensions = [
            name for name in required_dimensions if name not in self.dimensions
        ]
        if missing_dimensions:
            raise MeshValidationError(
                f"missing required dimensions: {', '.join(missing_dimensions)}"
            )

        n_cells = int(self.dimensions["nCells"])
        n_edges = int(self.dimensions["nEdges"])
        n_vertices = int(self.dimensions["nVertices"])
        max_edges = int(self.dimensions["maxEdges"])
        max_edges2 = int(self.dimensions["maxEdges2"])
        vertex_degree = int(self.dimensions["vertexDegree"])
        if min(n_cells, n_edges, n_vertices, max_edges, max_edges2, vertex_degree) <= 0:
            errors.append("all core mesh dimensions must be positive")

        bdy_present = [
            name for name in REGIONAL_BOUNDARY_MASK_NAMES if name in self.arrays
        ]
        triple_complete = len(bdy_present) == len(REGIONAL_BOUNDARY_MASK_NAMES)
        # THE BREAKAGE THIS PREVENTS, MEASURED (2026-08-26, all three fleet
        # cards on the published x1.40962 at the anchored 120 s timestep, and
        # v20.80.151649 on the proving RTX 5090): carrying the triple is NECESSARY
        # for a mesh to be regional and it is not SUFFICIENT.  Native MPAS-A
        # writes bdyMaskCell/Edge/Vertex into the static file of a GLOBAL mesh
        # too, all zero -- the published x1.40962 static (NCAR,
        # init_atmosphere v8.2.0) carries all three, 0/40962, 0/122880,
        # 0/81920 nonzero -- and the unified rw_mpas_static follows that
        # convention on every static this project generates.  So presence
        # alone classified the published global mesh AND every generated one
        # as a regional cull, and refused them with four checks a sphere
        # cannot pass.  Both of those refusals were themselves the proof it is
        # global: "nCells-nEdges+nVertices = 2, not 1" is the Euler
        # characteristic of a closed sphere, and "bdyMask rings [1..7] are
        # empty" is what an all-zero mask means -- every element interior, no
        # boundary zone at all.  The registry agreed the row was global
        # (boundary_zone_width and bdy_mask_sha256 both unset); this validator
        # was the second, disagreeing classifier.
        #
        # A cull HAS a boundary zone: that is what culling produces.  So the
        # rule is the triple plus a nonempty zone, which is content, not
        # schema, and which cannot be satisfied by a global mesh.  The
        # incompleteness refusal below stays keyed to PRESENCE -- a
        # half-written triple is refused whatever its values.
        regional = triple_complete and _regional_boundary_zone_present(self.arrays)
        if triple_complete and not regional and n_cells - n_edges + n_vertices != 2:
            errors.append(
                "the boundary-mask triple is present and entirely zero, but "
                f"nCells-nEdges+nVertices = {n_cells - n_edges + n_vertices}, "
                "not the closed sphere's 2.  An all-zero triple means every "
                "element is interior, which only a global mesh can be, and a "
                "global mesh is a closed sphere -- so this file is neither a "
                "sphere nor a cull with a boundary zone, and no convention "
                "this validator holds describes it"
            )
        if bdy_present and not triple_complete:
            missing_masks = [
                name for name in REGIONAL_BOUNDARY_MASK_NAMES if name not in self.arrays
            ]
            raise MeshValidationError(
                "regional boundary masks are incomplete: "
                f"{', '.join(bdy_present)} present without {', '.join(missing_masks)}. "
                "Without the full triple the validator cannot tell an "
                "absent-neighbour sentinel from index corruption, so the mesh "
                "can be neither admitted nor refused by name"
            )
        zone_width = REGIONAL_BOUNDARY_ZONE_WIDTH
        mask_cell = mask_edge = mask_vertex = None
        if regional:
            for name, count in (
                ("bdyMaskCell", n_cells),
                ("bdyMaskEdge", n_edges),
                ("bdyMaskVertex", n_vertices),
            ):
                mask = np.asarray(self.arrays[name])
                if mask.dtype.kind not in "iu":
                    raise MeshValidationError(
                        f"{name} has dtype {mask.dtype}, not an integer ring "
                        "number: a non-integer ring stages an element in no "
                        "boundary ring at all"
                    )
                if mask.shape != (count,):
                    raise MeshValidationError(
                        f"{name} has shape {mask.shape}, expected ({count},): "
                        "a mask that does not cover every element leaves "
                        "elements with no ring assignment"
                    )
                bad_mask = np.flatnonzero((mask < 0) | (mask > zone_width))
                if bad_mask.size:
                    raise MeshValidationError(
                        f"{name} carries ring numbers outside [0,{zone_width}] "
                        f"at indices {_bad_examples(bad_mask)}: the boundary "
                        f"zone is {zone_width} rings, so a value outside them "
                        "belongs to no ring and the runtime would neither "
                        "force nor freely integrate the element"
                    )
            mask_cell = np.asarray(self.arrays["bdyMaskCell"], dtype=np.int64)
            mask_edge = np.asarray(self.arrays["bdyMaskEdge"], dtype=np.int64)
            mask_vertex = np.asarray(self.arrays["bdyMaskVertex"], dtype=np.int64)

        expected_shapes = {
            "nEdgesOnCell": (n_cells,),
            "cellsOnCell": (n_cells, max_edges),
            "edgesOnCell": (n_cells, max_edges),
            "verticesOnCell": (n_cells, max_edges),
            "cellsOnEdge": (n_edges, 2),
            "verticesOnEdge": (n_edges, 2),
            "cellsOnVertex": (n_vertices, vertex_degree),
            "edgesOnVertex": (n_vertices, vertex_degree),
            "nEdgesOnEdge": (n_edges,),
            "edgesOnEdge": (n_edges, max_edges2),
            "weightsOnEdge": (n_edges, max_edges2),
            "dcEdge": (n_edges,),
            "dvEdge": (n_edges,),
            "areaCell": (n_cells,),
            "areaTriangle": (n_vertices,),
            "kiteAreasOnVertex": (n_vertices, vertex_degree),
        }
        missing_arrays = [name for name in expected_shapes if name not in self.arrays]
        if missing_arrays:
            raise MeshValidationError(
                f"missing required mesh fields: {', '.join(missing_arrays)}"
            )
        wrong_shapes = {
            name: (self.arrays[name].shape, expected)
            for name, expected in expected_shapes.items()
            if self.arrays[name].shape != expected
        }
        if wrong_shapes:
            raise MeshValidationError(f"mesh field shapes disagree: {wrong_shapes}")

        n_edges_on_cell = np.asarray(self.nEdgesOnCell, dtype=np.int64)
        n_edges_on_edge = np.asarray(self.nEdgesOnEdge, dtype=np.int64)
        cells_on_cell = np.asarray(self.cellsOnCell, dtype=np.int64)
        edges_on_cell = np.asarray(self.edgesOnCell, dtype=np.int64)
        vertices_on_cell = np.asarray(self.verticesOnCell, dtype=np.int64)
        cells_on_edge = np.asarray(self.cellsOnEdge, dtype=np.int64)
        vertices_on_edge = np.asarray(self.verticesOnEdge, dtype=np.int64)
        cells_on_vertex = np.asarray(self.cellsOnVertex, dtype=np.int64)
        edges_on_vertex = np.asarray(self.edgesOnVertex, dtype=np.int64)
        edges_on_edge = np.asarray(self.edgesOnEdge, dtype=np.int64)

        bad_counts = np.flatnonzero(
            (n_edges_on_cell < 3) | (n_edges_on_cell > max_edges)
        )
        if bad_counts.size:
            errors.append(
                "nEdgesOnCell outside [3,maxEdges] at cells "
                + _bad_examples(bad_counts)
            )
        bad_stencil_counts = np.flatnonzero(
            (n_edges_on_edge < 0) | (n_edges_on_edge > max_edges2)
        )
        if bad_stencil_counts.size:
            errors.append(
                "nEdgesOnEdge outside [0,maxEdges2] at edges "
                + _bad_examples(bad_stencil_counts)
            )

        counted_ranges = (
            ("cellsOnCell", cells_on_cell, n_edges_on_cell, n_cells),
            ("edgesOnCell", edges_on_cell, n_edges_on_cell, n_edges),
            ("verticesOnCell", vertices_on_cell, n_edges_on_cell, n_vertices),
            ("edgesOnEdge", edges_on_edge, n_edges_on_edge, n_edges),
        )
        # A sentinel in an array the cull keeps complete is a LOST element,
        # not a boundary; each message names what the loss breaks at runtime.
        regional_complete_breakage = {
            "edgesOnCell": (
                "the cull keeps every edge of a kept cell, so the cell's "
                "divergence/KE stencils would read a nonexistent edge"
            ),
            "verticesOnCell": (
                "the cull keeps every vertex of a kept cell, so the cell's "
                "vorticity gather would read a nonexistent vertex"
            ),
            "verticesOnEdge": (
                "every kept edge keeps both endpoints, so the tangential and "
                "kite terms at this edge would read a nonexistent vertex"
            ),
        }
        row_mask_for = {
            "cellsOnCell": mask_cell,
            "edgesOnCell": mask_cell,
            "verticesOnCell": mask_cell,
            "edgesOnEdge": mask_edge,
        }
        counts_safe = not bad_counts.size and not bad_stencil_counts.size
        if counts_safe:
            for name, array, counts, upper in counted_ranges:
                used_mask = np.arange(array.shape[1])[None, :] < counts[:, None]
                used = array[used_mask]
                sentinel_allowed = regional and name in REGIONAL_SENTINEL_ARRAYS
                lower = -1 if sentinel_allowed else 0
                bad_used = np.flatnonzero((used < lower) | (used >= upper))
                if bad_used.size:
                    errors.append(
                        f"{name} has used entries outside [0,{upper})"
                    )
                if regional:
                    sentinel_rows = np.flatnonzero(
                        ((array == -1) & used_mask).any(axis=1)
                    )
                    if sentinel_allowed:
                        misplaced = sentinel_rows[
                            row_mask_for[name][sentinel_rows] != zone_width
                        ]
                        if misplaced.size:
                            errors.append(
                                f"{name} marks a neighbour absent below the "
                                "outermost ring at rows "
                                + _bad_examples(misplaced)
                                + f": only ring-{zone_width} elements border "
                                "the cut, so an absent reference deeper in "
                                "the zone means the cull cut inside the "
                                "relaxation rings"
                            )
                    elif sentinel_rows.size:
                        lost = "an edge" if name == "edgesOnCell" else "a corner"
                        errors.append(
                            f"{name} lost {lost} at cells "
                            + _bad_examples(sentinel_rows)
                            + ": "
                            + regional_complete_breakage[name]
                        )
                padding = array[~used_mask]
                if np.any(padding != -1):
                    errors.append(f"{name} padding is not canonical -1")

        fixed_ranges = (
            ("cellsOnEdge", cells_on_edge, n_cells, mask_edge),
            ("verticesOnEdge", vertices_on_edge, n_vertices, mask_edge),
            ("cellsOnVertex", cells_on_vertex, n_cells, mask_vertex),
            ("edgesOnVertex", edges_on_vertex, n_edges, mask_vertex),
        )
        fixed_valid = True
        for name, array, upper, row_mask in fixed_ranges:
            sentinel_allowed = regional and name in REGIONAL_SENTINEL_ARRAYS
            lower = -1 if sentinel_allowed else 0
            bad = np.argwhere((array < lower) | (array >= upper))
            if bad.size:
                fixed_valid = False
                errors.append(f"{name} contains entries outside [0,{upper})")
            if regional:
                sentinel_rows = np.flatnonzero((array == -1).any(axis=1))
                if sentinel_allowed:
                    misplaced = sentinel_rows[row_mask[sentinel_rows] != zone_width]
                    if misplaced.size:
                        fixed_valid = False
                        errors.append(
                            f"{name} marks a neighbour absent below the "
                            "outermost ring at rows "
                            + _bad_examples(misplaced)
                            + f": only ring-{zone_width} elements border the "
                            "cut, so an absent reference deeper in the zone "
                            "means the cull cut inside the relaxation rings"
                        )
                elif sentinel_rows.size:
                    fixed_valid = False
                    errors.append(
                        "verticesOnEdge lost an endpoint at edges "
                        + _bad_examples(sentinel_rows)
                        + ": "
                        + regional_complete_breakage["verticesOnEdge"]
                    )

        if regional:
            euler = n_cells - n_edges + n_vertices
            if euler != 1:
                errors.append(
                    f"regional mesh is not a bounded disk: "
                    f"nCells-nEdges+nVertices = {euler}, not 1; the cull kept "
                    "or dropped elements inconsistently (a hole, a tear, or a "
                    "duplicate), so the boundary rings do not enclose the "
                    "interior"
                )
            # The closed-sphere incidence identities, corrected by exactly the
            # measured sentinel counts: an edge with one absent cell appears
            # once in edgesOnCell instead of twice, and a culled edge leaves
            # one empty slot per kept endpoint in edgesOnVertex.
            coe_sentinels = int(np.count_nonzero(cells_on_edge == -1))
            eov_sentinels = int(np.count_nonzero(edges_on_vertex == -1))
            if (
                not bad_counts.size
                and int(n_edges_on_cell.sum()) != 2 * n_edges - coe_sentinels
            ):
                errors.append(
                    f"sum(nEdgesOnCell) = {int(n_edges_on_cell.sum())} does "
                    f"not equal 2*nEdges minus cut-adjacent slots "
                    f"({2 * n_edges - coe_sentinels}): cells reference edges "
                    "the file does not carry, or the file carries edges no "
                    "kept cell owns"
                )
            if vertex_degree * n_vertices != 2 * n_edges + eov_sentinels:
                errors.append(
                    f"vertexDegree*nVertices = {vertex_degree * n_vertices} "
                    f"does not equal 2*nEdges plus culled edge slots "
                    f"({2 * n_edges + eov_sentinels}): vertices carry more or "
                    "fewer edge slots than the kept edges provide"
                )
        else:
            if n_cells - n_edges + n_vertices != 2:
                errors.append(
                    "closed-sphere Euler characteristic is not nCells-nEdges+nVertices=2"
                )
            if not bad_counts.size and int(n_edges_on_cell.sum()) != 2 * n_edges:
                errors.append("sum(nEdgesOnCell) does not equal 2*nEdges")
            if vertex_degree * n_vertices != 2 * n_edges:
                errors.append("vertexDegree*nVertices does not equal 2*nEdges")

        if regional and counts_safe and fixed_valid:
            huge = np.iinfo(np.int64).max
            orphan_edges = np.flatnonzero((cells_on_edge < 0).all(axis=1))
            if orphan_edges.size:
                errors.append(
                    "edges with no present cell at "
                    + _bad_examples(orphan_edges)
                    + ": every kept edge borders at least one kept cell; an "
                    "orphan edge has no column to take a tendency from"
                )
            else:
                adjacent_masks = np.where(
                    cells_on_edge >= 0,
                    mask_cell[np.clip(cells_on_edge, 0, None)],
                    huge,
                )
                wrong_edges = np.flatnonzero(
                    adjacent_masks.min(axis=1) != mask_edge
                )
                if wrong_edges.size:
                    errors.append(
                        "bdyMaskEdge is not the minimum of its present cells' "
                        "masks at edges "
                        + _bad_examples(wrong_edges)
                        + ": zone-staged edge tendencies key on this ring "
                        "number, so a wrong value forces or frees the edge in "
                        "the wrong ring"
                    )
            orphan_vertices = np.flatnonzero((cells_on_vertex < 0).all(axis=1))
            if orphan_vertices.size:
                errors.append(
                    "vertices with no present cell at "
                    + _bad_examples(orphan_vertices)
                    + ": every kept vertex is a corner of at least one kept "
                    "cell; an orphan vertex has no kite to weight"
                )
            else:
                vertex_masks = np.where(
                    cells_on_vertex >= 0,
                    mask_cell[np.clip(cells_on_vertex, 0, None)],
                    huge,
                )
                wrong_vertices = np.flatnonzero(
                    vertex_masks.min(axis=1) != mask_vertex
                )
                if wrong_vertices.size:
                    errors.append(
                        "bdyMaskVertex is not the minimum of its present "
                        "cells' masks at vertices "
                        + _bad_examples(wrong_vertices)
                        + ": zone-staged vorticity terms key on this ring "
                        "number, so a wrong value stages the vertex in the "
                        "wrong ring"
                    )
            used_slots = (
                np.arange(max_edges)[None, :] < n_edges_on_cell[:, None]
            ) & (cells_on_cell >= 0)
            neighbour_cells = cells_on_cell[used_slots]
            source_cells = np.broadcast_to(
                np.arange(n_cells)[:, None], cells_on_cell.shape
            )[used_slots]
            jumps = (
                np.abs(mask_cell[source_cells] - mask_cell[neighbour_cells]) > 1
            )
            if np.any(jumps):
                errors.append(
                    "bdyMaskCell jumps by more than 1 between neighbouring "
                    "cells at "
                    + _bad_examples(np.unique(source_cells[jumps]))
                    + ": rings advance one cell per ring from the cut, so a "
                    "jump means renumbered or torn rings"
                )
            for label, mask in (
                ("bdyMaskCell", mask_cell),
                ("bdyMaskEdge", mask_edge),
                ("bdyMaskVertex", mask_vertex),
            ):
                ring_counts = np.bincount(mask, minlength=zone_width + 1)
                if ring_counts[0] == 0:
                    errors.append(
                        f"{label} has no interior (ring-0) elements: the "
                        "subset is all boundary zone, leaving nothing for the "
                        "relaxation rings to protect"
                    )
                empty_rings = [
                    ring
                    for ring in range(1, zone_width + 1)
                    if ring_counts[ring] == 0
                ]
                if empty_rings:
                    errors.append(
                        f"{label} rings {empty_rings} are empty: the "
                        f"{zone_width}-ring zone is a contiguous shell, so an "
                        "empty ring means renumbered masks and zone-staged "
                        "tendencies skipping a ring"
                    )
            errors.extend(
                regional_ring_shell_errors(
                    mask_cell, source_cells, neighbour_cells, zone_width
                )
            )

        topology_ranges_valid = (
            counts_safe
            and fixed_valid
            and not any("used entries outside" in error for error in errors)
        )
        if topology_ranges_valid:
            bad_cell_rows: list[int] = []
            for cell in range(n_cells):
                count = int(n_edges_on_cell[cell])
                row_edges = edges_on_cell[cell, :count]
                row_cells = cells_on_cell[cell, :count]
                row_vertices = vertices_on_cell[cell, :count]
                # On a regional ring-7 cell several neighbour slots hold the
                # -1 sentinel; uniqueness applies to the present ones.  On a
                # global mesh every used slot is present (range-checked
                # above), so this is the original exact check.
                present_cells = row_cells[row_cells >= 0]
                if (
                    np.unique(row_edges).size != count
                    or np.unique(present_cells).size != present_cells.size
                    or np.unique(row_vertices).size != count
                ):
                    bad_cell_rows.append(cell)
                    continue
                for slot, edge in enumerate(row_edges):
                    edge_cells = cells_on_edge[edge]
                    if cell not in edge_cells:
                        bad_cell_rows.append(cell)
                        break
                    other = edge_cells[1] if edge_cells[0] == cell else edge_cells[0]
                    if row_cells[slot] != other:
                        bad_cell_rows.append(cell)
                        break
                    endpoints = vertices_on_edge[edge]
                    next_slot = (slot + 1) % count
                    if row_vertices[slot] not in endpoints or row_vertices[next_slot] not in endpoints:
                        bad_cell_rows.append(cell)
                        break
            if bad_cell_rows:
                errors.append(
                    "cell edge/neighbor/vertex slot reciprocity fails at cells "
                    + ", ".join(str(cell) for cell in bad_cell_rows[:5])
                )

            bad_edges: list[int] = []
            for edge in range(n_edges):
                edge_cells = cells_on_edge[edge]
                endpoints = vertices_on_edge[edge]
                if edge_cells[0] == edge_cells[1] or endpoints[0] == endpoints[1]:
                    bad_edges.append(edge)
                    continue
                # A regional ring-7 edge may have one absent cell (-1); the
                # reciprocity a missing cell cannot express is exempt, the
                # present side is still required to list the edge.
                if any(
                    edge not in edges_on_cell[cell, : n_edges_on_cell[cell]]
                    for cell in edge_cells
                    if cell >= 0
                ):
                    bad_edges.append(edge)
                    continue
                if any(edge not in edges_on_vertex[vertex] for vertex in endpoints):
                    bad_edges.append(edge)
            if bad_edges:
                errors.append(
                    "edge-to-cell/vertex reciprocity fails at edges "
                    + ", ".join(str(edge) for edge in bad_edges[:5])
                )

            bad_vertices: list[int] = []
            for vertex in range(n_vertices):
                row_cells = cells_on_vertex[vertex]
                row_edges = edges_on_vertex[vertex]
                # Regional ring-7 vertices hold -1 for culled cells/edges;
                # uniqueness and reciprocity apply to the present entries.
                # Global rows are fully present, so this is the original
                # exact check there.
                present_cells = row_cells[row_cells >= 0]
                present_edges = row_edges[row_edges >= 0]
                if (
                    np.unique(present_cells).size != present_cells.size
                    or np.unique(present_edges).size != present_edges.size
                ):
                    bad_vertices.append(vertex)
                    continue
                if any(
                    vertex
                    not in vertices_on_cell[cell, : n_edges_on_cell[cell]]
                    for cell in present_cells
                ) or any(vertex not in vertices_on_edge[edge] for edge in present_edges):
                    bad_vertices.append(vertex)
            if bad_vertices:
                errors.append(
                    "vertex-to-cell/edge reciprocity fails at vertices "
                    + ", ".join(str(vertex) for vertex in bad_vertices[:5])
                )

            bad_stencils: list[int] = []
            for edge in range(n_edges):
                actual = edges_on_edge[edge, : n_edges_on_edge[edge]]
                if regional and (
                    np.any(actual == -1) or np.any(cells_on_edge[edge] == -1)
                ):
                    # A ring-7 edge bordering the cut: its parent stencil
                    # reached culled edges (now -1 inside the unshrunk row)
                    # and possibly retained edges of the culled cell, so the
                    # exact-set identity is undefined.  What stays required:
                    # no self-reference and no duplicate present entry.
                    present = actual[actual >= 0]
                    if edge in present or np.unique(present).size != present.size:
                        bad_stencils.append(edge)
                    continue
                expected: set[int] = set()
                for cell in cells_on_edge[edge]:
                    expected.update(
                        int(value)
                        for value in edges_on_cell[cell, : n_edges_on_cell[cell]]
                    )
                expected.discard(edge)
                if (
                    edge in actual
                    or np.unique(actual).size != actual.size
                    or set(int(value) for value in actual) != expected
                ):
                    bad_stencils.append(edge)
            if bad_stencils:
                errors.append(
                    "edgesOnEdge tangential stencil is invalid at edges "
                    + ", ".join(str(edge) for edge in bad_stencils[:5])
                )

            visited = np.zeros(n_cells, dtype=bool)
            stack = [0]
            visited[0] = True
            while stack:
                cell = stack.pop()
                for neighbor in cells_on_cell[cell, : n_edges_on_cell[cell]]:
                    # The >= 0 guard is the same wrap family as the kite fix
                    # below: visited[-1] silently reads/marks the LAST cell.
                    if neighbor >= 0 and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(int(neighbor))
            if not np.all(visited):
                errors.append(
                    f"cell graph is disconnected ({np.count_nonzero(~visited)} unvisited cells)"
                )

        weights = np.asarray(self.weightsOnEdge)
        if not np.all(np.isfinite(weights)):
            errors.append("weightsOnEdge contains non-finite values")
        if not bad_stencil_counts.size:
            padding = np.arange(max_edges2)[None, :] >= n_edges_on_edge[:, None]
            if np.any(weights[padding] != 0):
                errors.append("weightsOnEdge padding is not zero")

        positive_metrics = (
            "dcEdge",
            "dvEdge",
            "areaCell",
            "areaTriangle",
            "kiteAreasOnVertex",
        )
        for name in positive_metrics:
            metric = np.asarray(self.arrays[name])
            if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
                errors.append(f"{name} must be finite and strictly positive")
        for optional in ("meshDensity", "nominalMinDc"):
            if optional in self.arrays:
                metric = np.asarray(self.arrays[optional])
                if not np.all(np.isfinite(metric)) or np.any(metric <= 0):
                    errors.append(f"{optional} must be finite and strictly positive")
        if "angleEdge" in self.arrays:
            angles = np.asarray(self.angleEdge)
            angle_tol = 1.0e-5 if angles.dtype.itemsize <= 4 else 1.0e-12
            if (
                not np.all(np.isfinite(angles))
                or np.any(np.abs(angles) > np.pi + angle_tol)
            ):
                errors.append("angleEdge must be finite and in [-pi,pi]")

        metric_arrays = [
            np.asarray(self.arrays[name])
            for name in positive_metrics
            if name in self.arrays
        ]
        has_float32 = any(array.dtype.itemsize <= 4 for array in metric_arrays)
        metric_rtol = 2.0e-5 if has_float32 else 5.0e-10

        area_cell = np.asarray(self.areaCell, dtype=np.float64)
        area_triangle = np.asarray(self.areaTriangle, dtype=np.float64)
        kite_areas = np.asarray(self.kiteAreasOnVertex, dtype=np.float64)
        if not np.allclose(
            kite_areas.sum(axis=1), area_triangle, rtol=metric_rtol, atol=0.0
        ):
            errors.append("kiteAreasOnVertex rows do not sum to areaTriangle")
        if fixed_valid:
            cell_kite_sum = np.zeros(n_cells, dtype=np.float64)
            # np.add.at WRAPS a negative index: a -1 sentinel folds the
            # absent neighbour's kite into cell nCells-1 and fabricates a
            # failure there against a defect-free mesh.  Measured on the
            # native regional culls: relative error 100.46 (quick cull) and
            # 394.39 (reference cull) at the wrap target.  Present slots
            # only; on a global mesh no sentinel reaches here, so the sum is
            # unchanged.
            flat_cells = cells_on_vertex.ravel()
            flat_kites = kite_areas.ravel()
            present_slots = flat_cells >= 0
            np.add.at(
                cell_kite_sum, flat_cells[present_slots], flat_kites[present_slots]
            )
            if not np.allclose(
                cell_kite_sum, area_cell, rtol=metric_rtol, atol=0.0
            ):
                errors.append("kites belonging to each cell do not sum to areaCell")

        on_sphere = str(self.attrs.get("on_a_sphere", "NO")).strip().upper() == "YES"
        if on_sphere:
            try:
                radius = float(self.attrs["sphere_radius"])
            except (KeyError, TypeError, ValueError):
                errors.append("spherical mesh lacks a positive sphere_radius attribute")
                radius = -1.0
            if not np.isfinite(radius) or radius <= 0:
                errors.append("sphere_radius must be finite and strictly positive")
            elif not regional:
                # A regional disk covers a fraction of the sphere (7.25
                # percent on the measured quick cull), so the whole-sphere
                # closure identities apply only to the global contract.  The
                # local kite/area identities above and the arc-length checks
                # below still bind every regional element.
                if not np.isclose(
                    area_cell.sum(), 4.0 * np.pi * radius * radius, rtol=metric_rtol
                ):
                    errors.append("sum(areaCell) does not equal 4*pi*sphere_radius**2")
                elif not np.isclose(
                    area_triangle.sum(), 4.0 * np.pi * radius * radius, rtol=metric_rtol
                ):
                    errors.append("sum(areaTriangle) does not equal 4*pi*sphere_radius**2")

            if radius > 0:
                coordinates: dict[str, NDArray[np.float64]] = {}
                coordinate_shapes = {
                    "Cell": n_cells,
                    "Edge": n_edges,
                    "Vertex": n_vertices,
                }
                for suffix, count in coordinate_shapes.items():
                    names = tuple(axis + suffix for axis in "xyz")
                    spherical_names = ("lat" + suffix, "lon" + suffix)
                    missing = [
                        name for name in (*names, *spherical_names) if name not in self.arrays
                    ]
                    if missing:
                        errors.append(
                            f"spherical mesh lacks coordinate fields: {', '.join(missing)}"
                        )
                        continue
                    xyz = np.stack(
                        [np.asarray(self.arrays[name], dtype=np.float64) for name in names],
                        axis=1,
                    )
                    lat = np.asarray(self.arrays[spherical_names[0]], dtype=np.float64)
                    lon = np.asarray(self.arrays[spherical_names[1]], dtype=np.float64)
                    if xyz.shape != (count, 3) or lat.shape != (count,) or lon.shape != (count,):
                        errors.append(f"{suffix} spherical coordinate shapes are invalid")
                        continue
                    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(lat + lon)):
                        errors.append(f"{suffix} coordinates contain non-finite values")
                        continue
                    norms = np.linalg.norm(xyz, axis=1)
                    if not np.allclose(norms, radius, rtol=metric_rtol, atol=0.0):
                        errors.append(f"{suffix} Cartesian coordinates are not on sphere_radius")
                    expected_xyz = radius * np.stack(
                        (
                            np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat),
                        ),
                        axis=1,
                    )
                    if not np.allclose(xyz, expected_xyz, rtol=metric_rtol, atol=radius * metric_rtol):
                        errors.append(f"{suffix} Cartesian and lat/lon coordinates disagree")
                    coordinates[suffix] = xyz / norms[:, None]

                if topology_ranges_valid and all(
                    suffix in coordinates for suffix in ("Cell", "Vertex")
                ):
                    def arc_distance(first: NDArray[np.float64], second: NDArray[np.float64]) -> NDArray[np.float64]:
                        return radius * np.arctan2(
                            np.linalg.norm(np.cross(first, second), axis=1),
                            np.sum(first * second, axis=1),
                        )

                    # Indexing coordinates with a regional -1 sentinel would
                    # WRAP to the last cell and fabricate an arc-length
                    # disagreement on every cut-adjacent edge.  The culled
                    # neighbour's coordinates are not in the file, so its
                    # dcEdge (copied from the parent) is unverifiable here
                    # and exempt; both dv endpoints are always present.
                    dc_verifiable = (cells_on_edge >= 0).all(axis=1)
                    expected_dc = arc_distance(
                        coordinates["Cell"][np.clip(cells_on_edge[:, 0], 0, None)],
                        coordinates["Cell"][np.clip(cells_on_edge[:, 1], 0, None)],
                    )
                    expected_dv = arc_distance(
                        coordinates["Vertex"][vertices_on_edge[:, 0]],
                        coordinates["Vertex"][vertices_on_edge[:, 1]],
                    )
                    # The floor comes from the dtype the COORDINATES are
                    # stored in, not from the metric arrays: the quantization
                    # being tolerated is theirs.  A file may carry binary64
                    # metrics over binary32 coordinates, and reading the floor
                    # off the metrics would leave that file with no floor at
                    # all.
                    for family, stored_name, expected, verifiable, message in (
                        (
                            "Cell",
                            "dcEdge",
                            expected_dc,
                            dc_verifiable,
                            "dcEdge disagrees with spherical cell-center arc length",
                        ),
                        (
                            "Vertex",
                            "dvEdge",
                            expected_dv,
                            None,
                            "dvEdge disagrees with spherical vertex arc length",
                        ),
                    ):
                        coordinate_dtype = min(
                            (
                                np.asarray(self.arrays[axis + family]).dtype
                                for axis in "xyz"
                            ),
                            key=lambda dtype: dtype.itemsize,
                        )
                        arc_rtol, arc_atol = spherical_arc_tolerance(
                            radius, coordinate_dtype, metric_rtol
                        )
                        stored = np.asarray(self.arrays[stored_name], dtype=np.float64)
                        compared = expected
                        if verifiable is not None:
                            stored = stored[verifiable]
                            compared = expected[verifiable]
                        if not np.allclose(
                            stored,
                            compared,
                            rtol=arc_rtol,
                            atol=arc_atol,
                        ):
                            errors.append(message)

        if errors:
            raise MeshValidationError(
                "MPAS mesh validation failed:\n - " + "\n - ".join(errors)
            )
        return self


def reconcile_grid_rotate_longitudes(mesh: Mesh) -> dict[str, Any]:
    """Reconcile only the proven MPAS-Tools ``grid_rotate`` pi defect.

    A mesh that is already Cartesian/lat-lon consistent is only mapped to the
    established ``[-pi, pi)`` representation.  An inconsistent mesh is
    admitted only when all three entity families reproduce the pinned
    ``grid_rotate.f90`` ``convert_xl`` formula to at most two binary64 ULP.
    The branch-specific binary32-pi error is then removed in memory and the
    unchanged strict :meth:`Mesh.validate` contract must pass.

    No input file, Cartesian coordinate, latitude, topology, metric, or
    validation tolerance is modified.
    """

    if str(mesh.attrs.get("on_a_sphere", "NO")).strip().upper() != "YES":
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires a spherical mesh"
        )
    try:
        radius = float(mesh.attrs["sphere_radius"])
    except (KeyError, TypeError, ValueError) as error:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires sphere_radius"
        ) from error
    if not np.isfinite(radius) or radius <= 0.0:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation requires positive sphere_radius"
        )

    staged: dict[str, NDArray[np.float64]] = {}
    originals: dict[str, NDArray[Any]] = {}
    original_sources: dict[str, str | None] = {}
    entity_evidence: dict[str, Any] = {}
    modes: list[str] = []
    changed_fields: list[str] = []

    for entity in ("Cell", "Edge", "Vertex"):
        coordinate_names = tuple(f"{axis}{entity}" for axis in "xyz")
        latitude_name = f"lat{entity}"
        longitude_name = f"lon{entity}"
        required = (*coordinate_names, latitude_name, longitude_name)
        missing = [name for name in required if name not in mesh.arrays]
        if missing:
            raise MeshValidationError(
                "grid_rotate longitude reconciliation lacks fields: "
                + ", ".join(missing)
            )
        values = {name: np.asarray(mesh.arrays[name]) for name in required}
        if any(value.dtype != np.dtype(np.float64) for value in values.values()):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation requires binary64 {entity} "
                "coordinates"
            )
        shapes = {value.shape for value in values.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1:
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation found invalid {entity} shapes"
            )
        if any(not np.all(np.isfinite(value)) for value in values.values()):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation found non-finite {entity} values"
            )

        x, y, z = (np.asarray(values[name], dtype=np.float64) for name in coordinate_names)
        latitude = np.asarray(values[latitude_name], dtype=np.float64)
        longitude = np.asarray(values[longitude_name], dtype=np.float64)
        xyz = np.stack((x, y, z), axis=1)
        norms = np.linalg.norm(xyz, axis=1)
        if not np.allclose(
            norms,
            radius,
            rtol=_BINARY64_MESH_RTOL,
            atol=0.0,
        ):
            raise MeshValidationError(
                f"grid_rotate longitude reconciliation refuses off-sphere {entity} xyz"
            )

        producer_latitude = np.arcsin(z / norms)
        latitude_ulp_error = _maximum_ulp_error(latitude, producer_latitude)
        authoritative_longitude = np.arctan2(y, x)
        expected_pre = radius * np.stack(
            (
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ),
            axis=1,
        )
        pre_max_xyz_gap = float(np.max(np.abs(xyz - expected_pre), initial=0.0))
        cartesian_consistent = bool(
            np.allclose(
                xyz,
                expected_pre,
                rtol=_BINARY64_MESH_RTOL,
                atol=radius * _BINARY64_MESH_RTOL,
            )
        )

        producer_gap: float | None = None
        producer_ulp_error: float | None = None
        correction_counts: dict[str, int] | None = None
        class_counts: list[int] | None = None
        if cartesian_consistent:
            mode = "already_cartesian_consistent"
            corrected_unwrapped = longitude
            candidate = normalize_longitudes(corrected_unwrapped)
        else:
            mode = "grid_rotate_default_real_pi_reconciled"
            producer_longitude, multiplicity = _grid_rotate_convert_xl_longitude(x, y)
            producer_gap = float(
                np.max(np.abs(longitude - producer_longitude), initial=0.0)
            )
            producer_ulp_error = _maximum_ulp_error(longitude, producer_longitude)
            if producer_ulp_error > GRID_ROTATE_PRODUCER_MAX_ULP_ERROR:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: raw "
                    "longitude does not reproduce the pinned producer formula "
                    f"within {GRID_ROTATE_PRODUCER_MAX_ULP_ERROR:g} ULP "
                    f"(observed {producer_ulp_error:g})"
                )
            if latitude_ulp_error > GRID_ROTATE_LATITUDE_MAX_ULP_ERROR:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: latitude "
                    "does not reproduce convert_xl "
                    f"(observed {latitude_ulp_error:g} ULP)"
                )
            corrected_unwrapped = longitude - multiplicity * GRID_ROTATE_PI_DELTA
            candidate = normalize_longitudes(corrected_unwrapped)
            authoritative_normalized = normalize_longitudes(authoritative_longitude)
            corrected_gap = float(
                np.max(
                    np.abs(_wrapped_angle_delta(candidate, authoritative_normalized)),
                    initial=0.0,
                )
            )
            if corrected_gap > GRID_ROTATE_CORRECTED_ATAN2_ATOL:
                raise MeshValidationError(
                    f"grid_rotate longitude reconciliation refuses {entity}: branch "
                    f"correction misses Cartesian atan2 by {corrected_gap:.17g} radians"
                )
            correction_counts = {
                label: int(np.count_nonzero(multiplicity == value))
                for label, value in (
                    ("0", 0.0),
                    ("0.5", 0.5),
                    ("1", 1.0),
                    ("1.5", 1.5),
                    ("2", 2.0),
                )
            }
            class_counts = [
                correction_counts["0"],
                correction_counts["1"],
                correction_counts["2"],
            ]

        canonicalization = validate_longitude_normalization(
            corrected_unwrapped, candidate
        )
        expected_post = radius * np.stack(
            (
                np.cos(latitude) * np.cos(candidate),
                np.cos(latitude) * np.sin(candidate),
                np.sin(latitude),
            ),
            axis=1,
        )
        post_max_xyz_gap = float(np.max(np.abs(xyz - expected_post), initial=0.0))
        post_atan2_gap = float(
            np.max(
                np.abs(_wrapped_angle_delta(candidate, authoritative_longitude)),
                initial=0.0,
            )
        )
        pre_hashes = {name: _sha256_array(values[name]) for name in required}
        staged[longitude_name] = np.ascontiguousarray(candidate)
        originals[longitude_name] = mesh.arrays[longitude_name]
        original_sources[longitude_name] = mesh.variable_sources.get(longitude_name)
        modes.append(mode)
        if not np.array_equal(longitude, candidate):
            changed_fields.append(longitude_name)
        entity_evidence[entity] = {
            "mode": mode,
            "count": int(longitude.size),
            "dtype": "float64",
            "producer_formula_max_abs_gap_radians": producer_gap,
            "producer_formula_max_ulp_error": producer_ulp_error,
            "producer_latitude_max_ulp_error": latitude_ulp_error,
            "pi_error_multiplicity_counts": correction_counts,
            "delta_class_counts_0_1_2": class_counts,
            "pre_max_cartesian_component_gap": pre_max_xyz_gap,
            "post_max_cartesian_component_gap": post_max_xyz_gap,
            "pre_max_atan2_angular_gap_radians": float(
                np.max(
                    np.abs(_wrapped_angle_delta(authoritative_longitude, longitude)),
                    initial=0.0,
                )
            ),
            "post_max_atan2_angular_gap_radians": post_atan2_gap,
            "longitude_sha256_pre": pre_hashes[longitude_name],
            "longitude_sha256_post": _sha256_array(candidate),
            "latitude_sha256_pre": pre_hashes[latitude_name],
            "latitude_sha256_post": pre_hashes[latitude_name],
            "cartesian_sha256_pre": {
                name: pre_hashes[name] for name in coordinate_names
            },
            "cartesian_sha256_post": {
                name: pre_hashes[name] for name in coordinate_names
            },
            "canonicalization": canonicalization,
        }

    if len(set(modes)) != 1:
        raise MeshValidationError(
            "grid_rotate longitude reconciliation refuses mixed producer signatures "
            f"across Cell/Edge/Vertex: {modes}"
        )

    try:
        for name, candidate in staged.items():
            mesh.arrays[name] = candidate
            mesh.variable_sources[name] = (
                "grid_binary64_grid_rotate_default_real_pi_reconciled_from_cartesian"
                if modes[0] == "grid_rotate_default_real_pi_reconciled"
                else "grid_binary64_equivalent_longitude_range"
            )
        mesh.validate()
        for entity, details in entity_evidence.items():
            for name, expected_hash in details["cartesian_sha256_pre"].items():
                if _sha256_array(mesh.arrays[name]) != expected_hash:
                    raise RuntimeError(
                        f"longitude reconciliation changed Cartesian field {name}"
                    )
            latitude_name = f"lat{entity}"
            if (
                _sha256_array(mesh.arrays[latitude_name])
                != details["latitude_sha256_pre"]
            ):
                raise RuntimeError(
                    f"longitude reconciliation changed latitude field {latitude_name}"
                )
    except Exception:
        for name, original in originals.items():
            mesh.arrays[name] = original
            source = original_sources[name]
            if source is None:
                mesh.variable_sources.pop(name, None)
            else:
                mesh.variable_sources[name] = source
        raise

    grid_path = None if mesh.grid_path is None else Path(mesh.grid_path)
    evidence = {
        "schema": "mpas-port.grid-rotate-longitude-reconciliation/v1",
        "mode": modes[0],
        "in_memory_only": True,
        "input_grid_name": None if grid_path is None else grid_path.name,
        "input_grid_sha256": (
            None if grid_path is None or not grid_path.is_file() else _sha256_file(grid_path)
        ),
        "producer_authority": {
            "repository": "MPAS-Dev/MPAS-Tools",
            "head": GRID_ROTATE_SOURCE_HEAD,
            "relative_path": "mesh_tools/grid_rotate/grid_rotate.f90",
            "sha256": GRID_ROTATE_SOURCE_SHA256,
            "source_lines": "15,25,447-501",
            "pi_expression": "pii = 2.*asin(1.0)",
            "storage_kind": "RKIND=8",
            "expression_kind": "unsuffixed default real (binary32)",
        },
        "producer_default_real_pi": GRID_ROTATE_DEFAULT_REAL_PI,
        "true_binary64_pi": float(np.pi),
        "default_real_pi_delta_radians": GRID_ROTATE_PI_DELTA,
        "producer_match_max_ulp": GRID_ROTATE_PRODUCER_MAX_ULP_ERROR,
        "latitude_match_max_ulp": GRID_ROTATE_LATITUDE_MAX_ULP_ERROR,
        "corrected_atan2_atol_radians": GRID_ROTATE_CORRECTED_ATAN2_ATOL,
        "validation_tolerance_changed": False,
        "strict_mesh_validation_after_reconciliation": "passed",
        "changed_longitude_fields": changed_fields,
        "cartesian_latitude_topology_metrics_changed": False,
        "entities": entity_evidence,
    }
    mesh.provenance["grid_rotate_longitude_reconciliation"] = evidence
    return evidence


def load_precision_preserving_mesh_pair(
    grid_path: str | Path,
    static_path: str | Path,
) -> tuple[Mesh, Mesh, dict[str, Any]]:
    """Merge static fields while retaining binary64 official grid geometry.

    The returned dynamics mesh uses the static file's Earth radius, metrics,
    terrain and other initialized fields.  Its Cartesian coordinates are the
    official grid coordinates scaled to that radius.  Both it and the
    grid-only output mesh use the exact trig-equivalent ``[-pi, pi)``
    longitude representation.  No mesh-validation tolerance is changed.
    """

    grid = Path(grid_path).expanduser().resolve(strict=True)
    static = Path(static_path).expanduser().resolve(strict=True)
    output_mesh = Mesh.from_netcdf(grid, validate=False)
    longitude_reconciliation = reconcile_grid_rotate_longitudes(output_mesh)
    dynamics_mesh = Mesh.from_netcdf(grid, static, validate=False)

    raw_overlay: dict[str, Any]
    try:
        dynamics_mesh.validate()
    except MeshValidationError as error:
        raw_overlay = {"status": "failed", "error": str(error)}
    else:
        raw_overlay = {"status": "passed", "error": None}

    for name in _TOPOLOGY_WITNESSES:
        if name not in output_mesh.arrays or name not in dynamics_mesh.arrays:
            raise MeshValidationError(f"precision overlay lacks topology witness {name}")
        if not np.array_equal(output_mesh.arrays[name], dynamics_mesh.arrays[name]):
            raise MeshValidationError(
                f"precision overlay grid/static topology differs at {name}"
            )

    grid_radius = float(output_mesh.attrs.get("sphere_radius", np.nan))
    static_radius = float(dynamics_mesh.attrs.get("sphere_radius", np.nan))
    if (
        not np.isfinite(grid_radius)
        or not np.isfinite(static_radius)
        or grid_radius <= 0.0
        or static_radius <= 0.0
    ):
        raise MeshValidationError("precision overlay requires positive finite sphere radii")
    scale = static_radius / grid_radius
    for entity in ("Cell", "Edge", "Vertex"):
        for component in ("x", "y", "z"):
            name = f"{component}{entity}"
            source = np.asarray(output_mesh.arrays[name])
            if source.dtype != np.dtype(np.float64):
                raise MeshValidationError(f"official grid {name} is not binary64")
            dynamics_mesh.arrays[name] = np.ascontiguousarray(source * scale)
            dynamics_mesh.variable_sources[name] = (
                "grid_binary64_scaled_to_static_sphere_radius"
            )

        latitude_name = f"lat{entity}"
        longitude_name = f"lon{entity}"
        latitude = np.asarray(output_mesh.arrays[latitude_name])
        longitude = np.asarray(output_mesh.arrays[longitude_name])
        if latitude.dtype != np.dtype(np.float64) or longitude.dtype != np.dtype(np.float64):
            raise MeshValidationError(
                f"official grid {latitude_name}/{longitude_name} is not binary64"
            )
        output_mesh.arrays[latitude_name] = np.ascontiguousarray(latitude.copy())
        output_mesh.arrays[longitude_name] = np.ascontiguousarray(longitude.copy())
        dynamics_mesh.arrays[latitude_name] = np.ascontiguousarray(latitude.copy())
        dynamics_mesh.arrays[longitude_name] = np.ascontiguousarray(longitude.copy())
        dynamics_mesh.variable_sources[latitude_name] = "grid_binary64"
        dynamics_mesh.variable_sources[longitude_name] = output_mesh.variable_sources[
            longitude_name
        ]

    dynamics_mesh.validate()
    output_mesh.validate()
    evidence = {
        "schema": "mpas-port.precision-preserving-mesh-pair/v1",
        "grid_name": grid.name,
        "grid_sha256": longitude_reconciliation["input_grid_sha256"],
        "static_name": static.name,
        "static_sha256": _sha256_file(static),
        "raw_static_overlay": raw_overlay,
        "topology_grid_static_bit_identical": True,
        "grid_sphere_radius": grid_radius,
        "static_sphere_radius": static_radius,
        "cartesian_scale": scale,
        "longitude_reconciliation": longitude_reconciliation,
        "longitude_normalization": {
            f"lon{entity}": longitude_reconciliation["entities"][entity][
                "canonicalization"
            ]
            for entity in ("Cell", "Edge", "Vertex")
        },
        "changed_array_inventory": list(_GEOMETRY_COORDINATES),
        "merged_validation_without_tolerance_change": "passed",
    }
    dynamics_mesh.provenance["precision_preserving_mesh_pair"] = evidence
    output_mesh.provenance["precision_preserving_mesh_pair"] = evidence
    return dynamics_mesh, output_mesh, evidence


__all__ = [
    "GRID_ROTATE_CONVERT_XL_EPS",
    "GRID_ROTATE_CORRECTED_ATAN2_ATOL",
    "GRID_ROTATE_DEFAULT_REAL_PI",
    "GRID_ROTATE_LATITUDE_MAX_ULP_ERROR",
    "GRID_ROTATE_PI_DELTA",
    "GRID_ROTATE_PRODUCER_MAX_ULP_ERROR",
    "GRID_ROTATE_SOURCE_HEAD",
    "GRID_ROTATE_SOURCE_SHA256",
    "LONGITUDE_TRIG_EQUIVALENCE_ATOL",
    "Mesh",
    "MeshValidationError",
    "REGIONAL_BOUNDARY_MASK_NAMES",
    "REGIONAL_BOUNDARY_ZONE_WIDTH",
    "REGIONAL_SENTINEL_ARRAYS",
    "load_precision_preserving_mesh_pair",
    "normalize_longitudes",
    "reconcile_grid_rotate_longitudes",
    "regional_boundary_mask_digest",
    "spherical_arc_tolerance",
    "validate_longitude_normalization",
]
