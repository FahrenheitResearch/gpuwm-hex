"""Mesh validation on a synthetic, defect-free, variable-resolution sphere.

The port's :meth:`Mesh.validate` recomputes ``dcEdge`` and ``dvEdge`` from the
stored Cartesian coordinates and compares.  The comparison used to be purely
relative, and a purely relative bound cannot express the dominant error in it:
the coordinates are quantized at the sphere radius, so the recomputed arc
carries an ABSOLUTE error of a metre or so no matter how short the edge is.
On a coarse mesh that hides under the relative bound; on a fine one it does
not, and a defect-free mesh is refused for its own storage precision.

These tests need a mesh, and the byte-pinned authority meshes ship with no
fetch path (tier 2).  So the mesh is built here: a genuine spherical
Delaunay/Voronoi pair, exact in binary64, then stored in binary32 exactly as a
mesh file stores it.  ``test_fixture_*`` below is the instrument check -- it
proves the synthetic mesh is defect-free by construction before any test uses
it as evidence that a defect-free mesh is admitted.

Measured on the default fixture -- 10,242 cells, 30,720 edges, 20,480
vertices, spacing graded 13.78 km to 3,044.1 km:

    kite tiling error   6.4e-12 relative      (every circumcentre is inside)
    sphere closure      0.0e+00 relative      (the kites tile the sphere)
    worst |dvEdge - recomputed|   0.532 m     (bound sqrt(3)*ulp = 0.866 m)
    worst relative disagreement   4.9e-05     at the 4.95 km vertex spacing
    edges over the shipped 2.0e-5 bound       393 of 30,720 dv, 2 of 30,720 dc
    edges over the derived floor 1.7321 m     0 of 30,720, both metrics
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from hexcore.mesh import Mesh, MeshValidationError, spherical_arc_tolerance


#: The radius the published meshes carry.
EARTH_RADIUS = 6371220.0


# ---------------------------------------------------------------------------
# the instrument: a defect-free variable-resolution spherical mesh
# ---------------------------------------------------------------------------
def _icosahedral_points(level: int) -> np.ndarray:
    """Unit-sphere points of the geodesic icosahedron at ``level``."""

    golden = (1.0 + 5.0**0.5) / 2.0
    corners = []
    for first, second in ((1.0, golden), (-1.0, golden), (1.0, -golden), (-1.0, -golden)):
        corners += [(0.0, first, second), (first, second, 0.0), (second, 0.0, first)]
    base = np.array(corners, dtype=np.float64)
    base /= np.linalg.norm(base, axis=1)[:, None]

    steps = 1 << level
    points = []
    for face in ConvexHull(base).simplices:
        first, second, third = base[face]
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                k = steps - i - j
                points.append((i * first + j * second + k * third) / steps)
    grid = np.array(points, dtype=np.float64)
    grid /= np.linalg.norm(grid, axis=1)[:, None]
    _, unique = np.unique(np.round(grid, 9), axis=0, return_index=True)
    return grid[np.sort(unique)]


def _conformal_zoom(points: np.ndarray, factor: float) -> np.ndarray:
    """Grade the point set by a Möbius zoom -- a conformal sphere map.

    Conformal matters: the map rescales lengths locally without shearing, so a
    near-equilateral triangulation stays near-equilateral and every triangle
    keeps its circumcentre inside itself.  Spacing runs from ``1/factor`` of
    the uniform spacing at one pole to ``factor`` times it at the other, which
    is how a real variable-resolution mesh carries fine and coarse edges in one
    file.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    denominator = np.where(1.0 - z < 1.0e-15, 1.0e-15, 1.0 - z)
    real = factor * x / denominator
    imaginary = factor * y / denominator
    square = real * real + imaginary * imaginary
    zoomed = np.stack(
        (
            2.0 * real / (1.0 + square),
            2.0 * imaginary / (1.0 + square),
            (square - 1.0) / (1.0 + square),
        ),
        axis=1,
    )
    return zoomed / np.linalg.norm(zoomed, axis=1)[:, None]


def _generic_rotation() -> np.ndarray:
    """A fixed rotation that keeps the refined region off the axes.

    On a coordinate pole two of the three binary32 components are near zero
    and quantize hundreds of times finer than they do anywhere else, and the
    third one's error is radial -- which normalization removes.  A mesh refined
    there would understate its own storage noise by orders of magnitude and the
    instrument would report a defect-free-looking mesh for the wrong reason.
    """

    matrix = np.eye(3)
    for axis, angle in enumerate((0.6435011087932844, 0.9272952180016122, 0.4636476090008061)):
        turn = np.eye(3)
        first, second = [index for index in range(3) if index != axis]
        turn[first, first] = math.cos(angle)
        turn[second, second] = math.cos(angle)
        turn[first, second] = -math.sin(angle)
        turn[second, first] = math.sin(angle)
        matrix = matrix @ turn
    return matrix


def _spherical_triangle_area(first, second, third):
    """Van Oosterom & Strackee excess, on the unit sphere."""

    triple = np.abs(np.einsum("ij,ij->i", first, np.cross(second, third)))
    denominator = (
        1.0
        + np.einsum("ij,ij->i", first, second)
        + np.einsum("ij,ij->i", second, third)
        + np.einsum("ij,ij->i", third, first)
    )
    return 2.0 * np.arctan2(triple, denominator)


def _arc_length(first, second, radius):
    return radius * np.arctan2(
        np.linalg.norm(np.cross(first, second), axis=1),
        np.einsum("ij,ij->i", first, second),
    )


def _angles_and_positions(unit: np.ndarray, radius: float, dtype) -> dict[str, np.ndarray]:
    scaled = unit * radius
    latitude = np.arcsin(np.clip(unit[:, 2], -1.0, 1.0))
    longitude = np.arctan2(unit[:, 1], unit[:, 0])
    return {
        "x": scaled[:, 0].astype(dtype),
        "y": scaled[:, 1].astype(dtype),
        "z": scaled[:, 2].astype(dtype),
        "lat": latitude.astype(dtype),
        "lon": longitude.astype(dtype),
    }


def graded_sphere_mesh(
    *,
    level: int = 5,
    zoom: float = 12.0,
    radius: float = EARTH_RADIUS,
    dtype=np.float32,
) -> Mesh:
    """A defect-free variable-resolution MPAS mesh, stored at ``dtype``.

    Cells are the points, mesh vertices are the Delaunay circumcentres, and
    every metric is the exact binary64 spherical measure of the figure the
    connectivity describes.  Nothing is approximated except by the final cast
    to ``dtype``, which is the point: that cast is what the arc-length check
    has to tolerate.
    """

    base = _icosahedral_points(level)
    # Exact icosahedral symmetry puts cocircular quadruples on the sphere, and
    # a cocircular quadruple gives two triangles one shared circumcentre -- a
    # zero-length dvEdge that is a degeneracy of the construction rather than
    # anything a generator would emit.  A deterministic jitter at a few per
    # cent of the (uniform) base spacing removes it, and the conformal map
    # carries it as the same few per cent of the local spacing everywhere.
    base = base + np.random.default_rng(20260823).normal(scale=1.0e-3, size=base.shape)
    base /= np.linalg.norm(base, axis=1)[:, None]
    unit = _conformal_zoom(base, zoom) @ _generic_rotation().T
    unit /= np.linalg.norm(unit, axis=1)[:, None]

    triangles = ConvexHull(unit, qhull_options="Qt").simplices.astype(np.int64)
    n_cells = unit.shape[0]
    n_vertices = triangles.shape[0]

    corner = [unit[triangles[:, index]] for index in range(3)]
    normal = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    outward = np.where(np.einsum("ij,ij->i", normal, corner[0]) < 0.0, -1.0, 1.0)
    vertex_unit = normal * outward[:, None]

    pairs = np.sort(
        np.concatenate(
            (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    cells_on_edge, inverse = np.unique(pairs, axis=0, return_inverse=True)
    n_edges = cells_on_edge.shape[0]
    edge_of_pair = {
        (int(low), int(high)): index for index, (low, high) in enumerate(cells_on_edge)
    }

    vertices_on_edge = np.full((n_edges, 2), -1, dtype=np.int64)
    filled = np.zeros(n_edges, dtype=np.int64)
    for position, edge in enumerate(np.asarray(inverse).ravel()):
        vertices_on_edge[edge, filled[edge]] = position % n_vertices
        filled[edge] += 1
    if not np.all(filled == 2):
        raise AssertionError("every Delaunay edge must carry exactly two triangles")

    triangles_on_cell: list[list[int]] = [[] for _ in range(n_cells)]
    for triangle, row in enumerate(triangles):
        for cell in row:
            triangles_on_cell[int(cell)].append(triangle)

    n_edges_on_cell = np.array(
        [len(entry) for entry in triangles_on_cell], dtype=np.int64
    )
    max_edges = int(n_edges_on_cell.max())
    vertices_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)
    edges_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)
    cells_on_cell = np.full((n_cells, max_edges), -1, dtype=np.int64)

    for cell in range(n_cells):
        centre = unit[cell]
        seed = np.array([1.0, 0.0, 0.0]) if abs(centre[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        first_axis = np.cross(centre, seed)
        first_axis /= np.linalg.norm(first_axis)
        second_axis = np.cross(centre, first_axis)
        incident = np.array(triangles_on_cell[cell], dtype=np.int64)
        offsets = vertex_unit[incident]
        order = np.argsort(
            np.arctan2(offsets @ second_axis, offsets @ first_axis), kind="stable"
        )
        ring = incident[order]
        count = ring.size
        vertices_on_cell[cell, :count] = ring
        for slot in range(count):
            here = set(int(value) for value in triangles[ring[slot]])
            there = set(int(value) for value in triangles[ring[(slot + 1) % count]])
            shared = (here & there) - {cell}
            if len(shared) != 1:
                raise AssertionError(
                    f"cell {cell} slot {slot} does not have one shared neighbour"
                )
            neighbour = shared.pop()
            cells_on_cell[cell, slot] = neighbour
            edges_on_cell[cell, slot] = edge_of_pair[
                (min(cell, neighbour), max(cell, neighbour))
            ]

    edges_on_vertex = np.array(
        [
            [
                edge_of_pair[
                    (
                        min(int(row[index]), int(row[(index + 1) % 3])),
                        max(int(row[index]), int(row[(index + 1) % 3])),
                    )
                ]
                for index in range(3)
            ]
            for row in triangles
        ],
        dtype=np.int64,
    )

    max_edges2 = 2 * max_edges
    n_edges_on_edge = (
        n_edges_on_cell[cells_on_edge[:, 0]] + n_edges_on_cell[cells_on_edge[:, 1]] - 2
    )
    edges_on_edge = np.full((n_edges, max_edges2), -1, dtype=np.int64)
    for edge in range(n_edges):
        stencil = []
        for cell in cells_on_edge[edge]:
            stencil += [
                int(value)
                for value in edges_on_cell[cell, : n_edges_on_cell[cell]]
                if int(value) != edge
            ]
        edges_on_edge[edge, : len(stencil)] = stencil
        if len(stencil) != int(n_edges_on_edge[edge]):
            raise AssertionError("tangential stencil length disagrees with its count")

    # Kites: the quad (cell centre, arc midpoint, circumcentre, arc midpoint).
    # They tile the Delaunay triangle and the Voronoi cell at once, so summing
    # them one way gives areaTriangle and the other way gives areaCell, with no
    # second formula that could disagree with the first.
    kite_areas = np.zeros((n_vertices, 3), dtype=np.float64)
    for index in range(3):
        here = unit[triangles[:, index]]
        left = unit[triangles[:, (index + 1) % 3]]
        right = unit[triangles[:, (index + 2) % 3]]
        first_middle = here + left
        first_middle /= np.linalg.norm(first_middle, axis=1)[:, None]
        second_middle = here + right
        second_middle /= np.linalg.norm(second_middle, axis=1)[:, None]
        kite_areas[:, index] = radius * radius * (
            _spherical_triangle_area(here, first_middle, vertex_unit)
            + _spherical_triangle_area(here, vertex_unit, second_middle)
        )

    area_triangle = kite_areas.sum(axis=1)
    area_cell = np.zeros(n_cells, dtype=np.float64)
    np.add.at(area_cell, triangles.ravel(), kite_areas.ravel())

    # Construction diagnostics, measured in binary64 on the exact figure.
    # They are what proves the instrument: the tests must not re-derive them
    # from the binary32 arrays, because binary32 is the thing under test.
    direct_triangle = radius * radius * _spherical_triangle_area(
        corner[0], corner[1], corner[2]
    )
    exact_dv = _arc_length(
        vertex_unit[vertices_on_edge[:, 0]], vertex_unit[vertices_on_edge[:, 1]], radius
    )
    exact_dc = _arc_length(unit[cells_on_edge[:, 0]], unit[cells_on_edge[:, 1]], radius)
    diagnostics = {
        # A circumcentre outside its triangle would make the three unsigned
        # kites overlap and overshoot the triangle they are meant to tile.
        "kite_tiling_max_relative_error": float(
            np.max(np.abs(kite_areas.sum(axis=1) - direct_triangle) / direct_triangle)
        ),
        "sphere_closure_relative_error": float(
            abs(area_cell.sum() / (4.0 * math.pi * radius * radius) - 1.0)
        ),
        "min_dc_metres": float(exact_dc.min()),
        "max_dc_metres": float(exact_dc.max()),
        "min_dv_metres": float(exact_dv.min()),
        "max_dv_metres": float(exact_dv.max()),
    }

    cell_positions = _angles_and_positions(unit, radius, dtype)
    vertex_positions = _angles_and_positions(vertex_unit, radius, dtype)
    edge_unit = unit[cells_on_edge[:, 0]] + unit[cells_on_edge[:, 1]]
    edge_unit /= np.linalg.norm(edge_unit, axis=1)[:, None]
    edge_positions = _angles_and_positions(edge_unit, radius, dtype)

    arrays: dict[str, np.ndarray] = {
        "nEdgesOnCell": n_edges_on_cell.astype(np.int32),
        "cellsOnCell": cells_on_cell,
        "edgesOnCell": edges_on_cell,
        "verticesOnCell": vertices_on_cell,
        "cellsOnEdge": cells_on_edge,
        "verticesOnEdge": vertices_on_edge,
        "cellsOnVertex": triangles,
        "edgesOnVertex": edges_on_vertex,
        "nEdgesOnEdge": n_edges_on_edge.astype(np.int32),
        "edgesOnEdge": edges_on_edge,
        "weightsOnEdge": np.zeros((n_edges, max_edges2), dtype=dtype),
        "dcEdge": _arc_length(
            unit[cells_on_edge[:, 0]], unit[cells_on_edge[:, 1]], radius
        ).astype(dtype),
        "dvEdge": _arc_length(
            vertex_unit[vertices_on_edge[:, 0]], vertex_unit[vertices_on_edge[:, 1]], radius
        ).astype(dtype),
        "areaCell": area_cell.astype(dtype),
        "areaTriangle": area_triangle.astype(dtype),
        "kiteAreasOnVertex": kite_areas.astype(dtype),
    }
    for suffix, positions in (
        ("Cell", cell_positions),
        ("Edge", edge_positions),
        ("Vertex", vertex_positions),
    ):
        for axis in ("x", "y", "z"):
            arrays[f"{axis}{suffix}"] = positions[axis]
        arrays[f"lat{suffix}"] = positions["lat"]
        arrays[f"lon{suffix}"] = positions["lon"]

    mesh = Mesh(
        arrays=arrays,
        dimensions={
            "nCells": n_cells,
            "nEdges": n_edges,
            "nVertices": n_vertices,
            "maxEdges": max_edges,
            "maxEdges2": max_edges2,
            "vertexDegree": 3,
        },
        attrs={"on_a_sphere": "YES", "sphere_radius": radius},
    )
    mesh.provenance["synthetic_construction"] = diagnostics
    return mesh


@pytest.fixture(scope="module")
def fine_mesh() -> Mesh:
    return graded_sphere_mesh()


# ---------------------------------------------------------------------------
# instrument check: the synthetic mesh really is defect-free
# ---------------------------------------------------------------------------
def test_fixture_is_a_closed_sphere_with_positive_metrics(fine_mesh: Mesh) -> None:
    dimensions = fine_mesh.dimensions
    assert (
        dimensions["nCells"] - dimensions["nEdges"] + dimensions["nVertices"] == 2
    )
    assert 3 * dimensions["nVertices"] == 2 * dimensions["nEdges"]
    for name in ("dcEdge", "dvEdge", "areaCell", "areaTriangle", "kiteAreasOnVertex"):
        metric = np.asarray(fine_mesh.arrays[name])
        assert np.all(np.isfinite(metric)) and np.all(metric > 0)
    construction = fine_mesh.provenance["synthetic_construction"]
    assert construction["sphere_closure_relative_error"] < 1.0e-12


def test_fixture_kites_prove_every_circumcentre_lies_inside_its_triangle(
    fine_mesh: Mesh,
) -> None:
    # If a circumcentre fell outside its triangle the three unsigned kites
    # would overlap and their sum would overshoot the triangle they tile.  So
    # this is the well-centredness check, measured rather than asserted -- and
    # measured on the binary64 figure, because the binary32 arrays carry the
    # quantization that is itself under test.
    construction = fine_mesh.provenance["synthetic_construction"]
    assert construction["kite_tiling_max_relative_error"] < 1.0e-9


def test_fixture_spans_fine_and_coarse_spacing_in_one_mesh(fine_mesh: Mesh) -> None:
    # Fine enough that the shipped purely relative bound is exceeded by
    # binary32 storage alone -- the published 15 km mesh's scale -- and coarse
    # enough that the same mesh also exercises the relative half.
    construction = fine_mesh.provenance["synthetic_construction"]
    assert construction["min_dc_metres"] < 20_000.0
    assert construction["max_dc_metres"] > 1_000_000.0


# ---------------------------------------------------------------------------
# the defect: a defect-free fine mesh must be admitted
# ---------------------------------------------------------------------------
def test_defect_free_binary32_fine_mesh_is_admitted(fine_mesh: Mesh) -> None:
    assert fine_mesh.validate() is fine_mesh


def test_binary32_storage_noise_alone_breaks_a_purely_relative_bound(
    fine_mesh: Mesh,
) -> None:
    # The measurement behind the fix: on this defect-free mesh the recomputed
    # arc disagrees with the stored one by up to about a metre -- an absolute
    # quantity -- and on the fine edges that is more than 2.0e-5 of the edge.
    unit = np.stack(
        [
            np.asarray(fine_mesh.arrays[f"{axis}Vertex"], dtype=np.float64)
            for axis in "xyz"
        ],
        axis=1,
    )
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    ends = np.asarray(fine_mesh.verticesOnEdge, dtype=np.int64)
    recomputed = _arc_length(unit[ends[:, 0]], unit[ends[:, 1]], EARTH_RADIUS)
    stored = np.asarray(fine_mesh.dvEdge, dtype=np.float64)
    error = np.abs(stored - recomputed)

    coordinate_ulp = float(np.spacing(np.float32(EARTH_RADIUS)))
    assert error.max() <= math.sqrt(3.0) * coordinate_ulp
    assert np.any(error > 2.0e-5 * recomputed)

    _, atol = spherical_arc_tolerance(EARTH_RADIUS, np.dtype(np.float32))
    assert np.all(error <= atol)


# ---------------------------------------------------------------------------
# the other direction: a real corruption is still named and still refused
# ---------------------------------------------------------------------------
def _coarsest_edge(mesh: Mesh) -> int:
    return int(np.argmax(np.asarray(mesh.dcEdge, dtype=np.float64)))


def test_corrupt_dv_edge_is_still_refused_by_name(fine_mesh: Mesh) -> None:
    corrupt = fine_mesh.copy()
    edge = _coarsest_edge(corrupt)
    spacing = float(np.asarray(corrupt.dvEdge)[edge])
    corrupt.arrays["dvEdge"][edge] = np.float32(spacing + 5.0)
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "dvEdge disagrees with spherical vertex arc length" in str(refusal.value)


def test_corrupt_dc_edge_is_still_refused_by_name(fine_mesh: Mesh) -> None:
    corrupt = fine_mesh.copy()
    edge = _coarsest_edge(corrupt)
    spacing = float(np.asarray(corrupt.dcEdge)[edge])
    corrupt.arrays["dcEdge"][edge] = np.float32(spacing + 5.0)
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "dcEdge disagrees with spherical cell-center arc length" in str(refusal.value)


def test_a_displaced_vertex_is_still_refused(fine_mesh: Mesh) -> None:
    # The corruption a stale or mis-scaled coordinate array produces: the
    # metrics are untouched and one vertex has moved.  Ten metres is far below
    # anything the relative bound alone would catch on a coarse edge.
    corrupt = fine_mesh.copy()
    edge = _coarsest_edge(corrupt)
    vertex = int(np.asarray(corrupt.verticesOnEdge)[edge, 0])
    original = float(np.asarray(corrupt.arrays["xVertex"])[vertex])
    corrupt.arrays["xVertex"][vertex] = np.float32(original + 10.0)
    with pytest.raises(MeshValidationError) as refusal:
        corrupt.validate()
    assert "dvEdge disagrees with spherical vertex arc length" in str(refusal.value)


# ---------------------------------------------------------------------------
# the tolerance itself: derived, not a magic number
# ---------------------------------------------------------------------------
def test_arc_tolerance_floor_tracks_the_radius_it_is_derived_from() -> None:
    _, earth = spherical_arc_tolerance(EARTH_RADIUS, np.dtype(np.float32))
    _, half = spherical_arc_tolerance(EARTH_RADIUS / 2.0, np.dtype(np.float32))
    _, unit = spherical_arc_tolerance(1.0, np.dtype(np.float32))

    assert earth == pytest.approx(2.0 * math.sqrt(3.0) * 0.5)
    assert half == pytest.approx(earth / 2.0)
    # A unit-sphere mesh quantizes at 1.2e-7 rather than half a metre, and the
    # floor has to follow it down or it becomes a licence rather than a budget.
    assert unit == pytest.approx(2.0 * math.sqrt(3.0) * float(np.spacing(np.float32(1.0))))


def test_arc_tolerance_floor_is_negligible_for_binary64_coordinates() -> None:
    _, floor = spherical_arc_tolerance(EARTH_RADIUS, np.dtype(np.float64))
    assert floor < 1.0e-8


def test_arc_tolerance_relative_half_is_never_looser_than_binary32_storage() -> None:
    coarse, _ = spherical_arc_tolerance(EARTH_RADIUS, np.dtype(np.float32))
    assert coarse <= 8.0 * float(np.finfo(np.float32).eps)
    # Metre-scale corruption at coarse spacing must not fit inside the relative
    # half: at 200 km that bound is under a quarter of a metre.
    assert coarse * 200_000.0 < 0.25


# ---------------------------------------------------------------------------
# a global mesh that carries the boundary-mask triple
#
# MEASURED (2026-08-26, all three fleet cards, x1.40962 at the anchored 120 s
# timestep): every forecast on the published global mesh was refused with
# "regional mesh is not a bounded disk: nCells-nEdges+nVertices = 2, not 1"
# and three "bdyMask rings [1..7] are empty" findings.  Both are the proof the
# mesh is GLOBAL -- 2 is a closed sphere's Euler characteristic, and empty
# rings are what an all-zero mask means -- and they were being read as proof
# of a corrupt cull.
#
# The cause is that native MPAS-A writes bdyMaskCell/Edge/Vertex into a GLOBAL
# mesh's static file as well, all zero.  Read from the real published bytes:
# x1.40962.static.nc (NCAR, init_atmosphere v8.2.0) carries all three, with
# 0 of 40,962 / 0 of 122,880 / 0 of 81,920 entries nonzero.  Classifying on
# the triple's PRESENCE therefore refuses the published family outright, while
# the mesh registry's own row for the same mesh says global (its
# boundary_zone_width and bdy_mask_sha256 are both unset).
# ---------------------------------------------------------------------------
def _with_zero_boundary_masks(mesh: Mesh) -> Mesh:
    """The fixture as native MPAS-A would write it: the triple, all zero."""

    arrays = dict(mesh.arrays)
    for name, count in (
        ("bdyMaskCell", mesh.dimensions["nCells"]),
        ("bdyMaskEdge", mesh.dimensions["nEdges"]),
        ("bdyMaskVertex", mesh.dimensions["nVertices"]),
    ):
        arrays[name] = np.zeros(int(count), dtype=np.int32)
    return Mesh(
        arrays=arrays, dimensions=dict(mesh.dimensions), attrs=dict(mesh.attrs)
    )


def test_a_global_mesh_carrying_an_all_zero_mask_triple_is_still_global(
    fine_mesh: Mesh,
) -> None:
    """The regression: presence of the triple is necessary, not sufficient."""

    mesh = _with_zero_boundary_masks(fine_mesh)
    # every mask is present, so the schema alone says "regional"
    assert all(
        name in mesh.arrays
        for name in ("bdyMaskCell", "bdyMaskEdge", "bdyMaskVertex")
    )
    # and the mesh is a closed sphere, so it is not one
    assert (
        mesh.dimensions["nCells"]
        - mesh.dimensions["nEdges"]
        + mesh.dimensions["nVertices"]
        == 2
    )
    assert mesh.validate() is mesh


def test_a_cull_with_a_real_zone_is_still_classified_regional(
    fine_mesh: Mesh,
) -> None:
    """The other direction: the fix must not stop seeing genuine culls.

    A nonempty zone is what makes a mesh regional, so a sphere carrying one is
    a contradiction and must be refused -- by the disk check, exactly as
    before.  This is the arm that would go quiet if the new rule were simply
    "never regional".
    """

    arrays = dict(fine_mesh.arrays)
    n_cells = int(fine_mesh.dimensions["nCells"])
    n_edges = int(fine_mesh.dimensions["nEdges"])
    n_vertices = int(fine_mesh.dimensions["nVertices"])
    mask_cell = np.zeros(n_cells, dtype=np.int32)
    mask_cell[:64] = 7
    arrays["bdyMaskCell"] = mask_cell
    arrays["bdyMaskEdge"] = np.zeros(n_edges, dtype=np.int32)
    arrays["bdyMaskVertex"] = np.zeros(n_vertices, dtype=np.int32)
    mesh = Mesh(
        arrays=arrays,
        dimensions=dict(fine_mesh.dimensions),
        attrs=dict(fine_mesh.attrs),
    )
    with pytest.raises(MeshValidationError) as caught:
        mesh.validate()
    assert "bounded disk" in str(caught.value)


def test_an_all_zero_triple_on_a_mesh_that_is_no_sphere_is_refused_by_name(
    fine_mesh: Mesh,
) -> None:
    """The gap the new rule opens is closed where it opens.

    An all-zero triple now means "global", and global means "closed sphere".
    A file that claims neither is described by no convention this validator
    holds, and says so rather than being waved through as global.
    """

    mesh = _with_zero_boundary_masks(fine_mesh)
    n_vertices = int(mesh.dimensions["nVertices"])
    # drop one vertex consistently -- dimension AND every array indexed by it
    # -- so the file is shape-coherent and its Euler characteristic is 1
    arrays = {
        name: (
            value[: n_vertices - 1]
            if getattr(value, "shape", (None,))[0] == n_vertices
            else value
        )
        for name, value in mesh.arrays.items()
    }
    dimensions = dict(mesh.dimensions)
    dimensions["nVertices"] = n_vertices - 1
    broken = Mesh(arrays=arrays, dimensions=dimensions, attrs=dict(mesh.attrs))
    with pytest.raises(MeshValidationError) as caught:
        broken.validate()
    text = str(caught.value)
    assert "entirely zero" in text
    assert "closed sphere's 2" in text
