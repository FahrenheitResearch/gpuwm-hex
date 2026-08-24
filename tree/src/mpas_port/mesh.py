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

    def validate(self) -> "Mesh":
        """Validate the complete closed-sphere MPAS C-grid contract.

        Validation deliberately includes both local reciprocity and global
        metric identities.  A corrupt mesh therefore cannot pass merely
        because every integer happens to be in range.
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
        counts_safe = not bad_counts.size and not bad_stencil_counts.size
        if counts_safe:
            for name, array, counts, upper in counted_ranges:
                used_mask = np.arange(array.shape[1])[None, :] < counts[:, None]
                used = array[used_mask]
                bad_used = np.flatnonzero((used < 0) | (used >= upper))
                if bad_used.size:
                    errors.append(
                        f"{name} has used entries outside [0,{upper})"
                    )
                padding = array[~used_mask]
                if np.any(padding != -1):
                    errors.append(f"{name} padding is not canonical -1")

        fixed_ranges = (
            ("cellsOnEdge", cells_on_edge, n_cells),
            ("verticesOnEdge", vertices_on_edge, n_vertices),
            ("cellsOnVertex", cells_on_vertex, n_cells),
            ("edgesOnVertex", edges_on_vertex, n_edges),
        )
        fixed_valid = True
        for name, array, upper in fixed_ranges:
            bad = np.argwhere((array < 0) | (array >= upper))
            if bad.size:
                fixed_valid = False
                errors.append(f"{name} contains entries outside [0,{upper})")

        if n_cells - n_edges + n_vertices != 2:
            errors.append(
                "closed-sphere Euler characteristic is not nCells-nEdges+nVertices=2"
            )
        if not bad_counts.size and int(n_edges_on_cell.sum()) != 2 * n_edges:
            errors.append("sum(nEdgesOnCell) does not equal 2*nEdges")
        if vertex_degree * n_vertices != 2 * n_edges:
            errors.append("vertexDegree*nVertices does not equal 2*nEdges")

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
                if (
                    np.unique(row_edges).size != count
                    or np.unique(row_cells).size != count
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
                if any(
                    edge not in edges_on_cell[cell, : n_edges_on_cell[cell]]
                    for cell in edge_cells
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
                if (
                    np.unique(row_cells).size != vertex_degree
                    or np.unique(row_edges).size != vertex_degree
                ):
                    bad_vertices.append(vertex)
                    continue
                if any(
                    vertex
                    not in vertices_on_cell[cell, : n_edges_on_cell[cell]]
                    for cell in row_cells
                ) or any(vertex not in vertices_on_edge[edge] for edge in row_edges):
                    bad_vertices.append(vertex)
            if bad_vertices:
                errors.append(
                    "vertex-to-cell/edge reciprocity fails at vertices "
                    + ", ".join(str(vertex) for vertex in bad_vertices[:5])
                )

            bad_stencils: list[int] = []
            for edge in range(n_edges):
                expected: set[int] = set()
                for cell in cells_on_edge[edge]:
                    expected.update(
                        int(value)
                        for value in edges_on_cell[cell, : n_edges_on_cell[cell]]
                    )
                expected.discard(edge)
                actual = edges_on_edge[edge, : n_edges_on_edge[edge]]
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
                    if not visited[neighbor]:
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
            np.add.at(cell_kite_sum, cells_on_vertex.ravel(), kite_areas.ravel())
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
            elif not np.isclose(
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

                    expected_dc = arc_distance(
                        coordinates["Cell"][cells_on_edge[:, 0]],
                        coordinates["Cell"][cells_on_edge[:, 1]],
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
                    for family, stored_name, expected, message in (
                        (
                            "Cell",
                            "dcEdge",
                            expected_dc,
                            "dcEdge disagrees with spherical cell-center arc length",
                        ),
                        (
                            "Vertex",
                            "dvEdge",
                            expected_dv,
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
                        if not np.allclose(
                            np.asarray(self.arrays[stored_name], dtype=np.float64),
                            expected,
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
    "load_precision_preserving_mesh_pair",
    "normalize_longitudes",
    "reconcile_grid_rotate_longitudes",
    "spherical_arc_tolerance",
    "validate_longitude_normalization",
]
