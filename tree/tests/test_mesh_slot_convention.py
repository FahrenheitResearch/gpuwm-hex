"""verticesOnCell slot convention and measured area-closure tolerance.

THE BREAKAGE (measured 2026-10-05 on RRFS v2's public conus3km grid): the
file stores every one of its 2,341,818 cells with vertex ``j`` at the END of
edge ``j`` (edge ``j`` joins vertices ``j-1`` and ``j``), where conus12km,
NCAR's x1 meshes and the rw-mpas generator put it at the START (edge ``j``
joins ``j`` and ``j+1``).  The rings are complete and reciprocal either way,
but the validator assumed the start convention and refused the mesh at cells
0..4.  Its binary64 kites also close only to 8.1e-10 (triangles) and 5.9e-10
(cells): above the fixed 5e-10 bound, inside what a binary64 producer can
reach on 3 km elements (``16 u / theta_min**2`` ~ 2.4e-8).

MPAS consumes the stored order as-is, so nothing here reorders the array: the
convention is detected per mesh and reciprocity is checked under it.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.mesh import (
    Mesh,
    MeshValidationError,
    closure_tolerance,
    vertex_slot_convention,
)
from hexcore.mesh import _stored_significant_bits
from test_mesh_validation import graded_sphere_mesh


def _rows(mesh: Mesh):
    return (
        np.asarray(mesh.nEdgesOnCell, dtype=np.int64),
        np.asarray(mesh.edgesOnCell, dtype=np.int64),
        np.asarray(mesh.verticesOnCell, dtype=np.int64),
        np.asarray(mesh.verticesOnEdge, dtype=np.int64),
    )


def _to_edge_end(mesh: Mesh, cells=None) -> Mesh:
    """Store the same rings starting one slot later: edge j = (v[j-1], v[j])."""
    rotated = mesh.copy()
    counts = np.asarray(rotated.nEdgesOnCell, dtype=np.int64)
    vertices = np.array(rotated.arrays["verticesOnCell"], copy=True)
    for cell in range(counts.size) if cells is None else cells:
        n = int(counts[cell])
        vertices[cell, :n] = np.roll(vertices[cell, :n], -1)
    rotated.arrays["verticesOnCell"] = vertices
    return rotated


@pytest.fixture(scope="module")
def start_mesh() -> Mesh:
    return graded_sphere_mesh()


@pytest.fixture(scope="module")
def end_mesh(start_mesh: Mesh) -> Mesh:
    return _to_edge_end(start_mesh)


@pytest.fixture(scope="module")
def binary64_mesh() -> Mesh:
    return graded_sphere_mesh(dtype=np.float64)


# --------------------------------------------------------------- convention
def test_the_fixture_is_edge_start_and_admitted(start_mesh: Mesh) -> None:
    verdict = vertex_slot_convention(*_rows(start_mesh))
    assert verdict["convention"] == "edge-start"
    assert verdict["edge_start_cells"] == start_mesh.dimensions["nCells"]
    assert verdict["edge_end_cells"] == 0
    assert start_mesh.validate() is start_mesh


def test_a_consistently_edge_end_mesh_is_admitted_without_reordering(
    end_mesh: Mesh,
) -> None:
    verdict = vertex_slot_convention(*_rows(end_mesh))
    assert verdict["convention"] == "edge-end"
    assert verdict["edge_end_cells"] == end_mesh.dimensions["nCells"]
    before = np.array(end_mesh.arrays["verticesOnCell"], copy=True)
    assert end_mesh.validate() is end_mesh
    # MPAS reads the stored order; validation must leave it byte-identical.
    assert end_mesh.arrays["verticesOnCell"].tobytes() == before.tobytes()


def test_the_validation_view_maps_edge_end_back_onto_edge_start(
    start_mesh: Mesh, end_mesh: Mesh
) -> None:
    view = vertex_slot_convention(*_rows(end_mesh))["view"]
    assert np.array_equal(view, np.asarray(start_mesh.verticesOnCell))


def test_a_mixed_mesh_is_refused_by_name(start_mesh: Mesh) -> None:
    half = range(0, start_mesh.dimensions["nCells"], 2)
    mixed = _to_edge_end(start_mesh, cells=half)
    assert vertex_slot_convention(*_rows(mixed))["convention"] == "mixed"
    with pytest.raises(MeshValidationError) as refusal:
        mixed.validate()
    assert "mixes slot conventions" in str(refusal.value)


@pytest.mark.parametrize("which", ["start", "end"])
def test_a_broken_ring_is_refused_at_that_cell_under_either_convention(
    which: str, start_mesh: Mesh, end_mesh: Mesh
) -> None:
    broken = (start_mesh if which == "start" else end_mesh).copy()
    vertices = np.array(broken.arrays["verticesOnCell"], copy=True)
    cell = 4321
    vertices[cell, [1, 3]] = vertices[cell, [3, 1]]
    broken.arrays["verticesOnCell"] = vertices
    verdict = vertex_slot_convention(*_rows(broken))
    assert verdict["first_neither_cells"] == [cell]
    with pytest.raises(MeshValidationError) as refusal:
        broken.validate()
    text = str(refusal.value)
    assert f"slot reciprocity fails at cells {cell}" in text
    # Only the broken cell is named, not every cell of an edge-end mesh.
    assert "fails at cells 0," not in text


def test_reversed_rings_satisfy_neither_convention_and_are_refused(
    start_mesh: Mesh,
) -> None:
    reversed_mesh = start_mesh.copy()
    counts = np.asarray(reversed_mesh.nEdgesOnCell, dtype=np.int64)
    vertices = np.array(reversed_mesh.arrays["verticesOnCell"], copy=True)
    for cell in range(counts.size):
        n = int(counts[cell])
        vertices[cell, :n] = vertices[cell, :n][::-1]
    reversed_mesh.arrays["verticesOnCell"] = vertices
    assert vertex_slot_convention(*_rows(reversed_mesh))["convention"] == "none"
    with pytest.raises(MeshValidationError) as refusal:
        reversed_mesh.validate()
    assert "slot reciprocity fails" in str(refusal.value)


# ------------------------------------------------------------------ closure
def test_storage_precision_is_measured_from_the_values() -> None:
    values = np.linspace(1.0, 2.0, 1001)
    assert _stored_significant_bits(values) == 53
    assert _stored_significant_bits(values.astype(np.float32)) == 24
    # binary64 dtype holding binary32 numbers stores binary32 precision
    assert _stored_significant_bits(values.astype(np.float32).astype(np.float64)) == 24


def test_binary32_valued_binary64_metrics_get_the_binary32_bound() -> None:
    widened = np.linspace(1.0, 2.0, 11).astype(np.float32).astype(np.float64)
    bound = closure_tolerance(
        {"areaCell": widened}, radius=1.0, edge_lengths=np.array([1.0e-3])
    )
    assert bound["stored_significant_bits"] == 24
    assert bound["rtol"] == 2.0e-5


def test_binary64_bound_follows_the_mesh_resolution(binary64_mesh: Mesh) -> None:
    radius = float(binary64_mesh.attrs["sphere_radius"])
    lengths = np.concatenate(
        (np.asarray(binary64_mesh.dcEdge), np.asarray(binary64_mesh.dvEdge))
    )
    bound = closure_tolerance(
        {"areaCell": binary64_mesh.areaCell}, radius=radius, edge_lengths=lengths
    )
    theta = lengths.min() / radius
    assert bound["stored_significant_bits"] == 53
    assert bound["rtol"] == pytest.approx(16.0 * 2.0**-53 / theta**2)
    assert bound["rtol"] > 5.0e-10
    # A coarse mesh keeps the old bound: nothing measured loosens it.
    coarse = closure_tolerance(
        {"areaCell": binary64_mesh.areaCell},
        radius=radius,
        edge_lengths=np.array([0.05 * radius]),
    )
    assert coarse["rtol"] == 5.0e-10
    # conus3km scale: shortest edge ~1.7 km on the Earth sphere.
    conus3km = closure_tolerance(
        {"areaCell": binary64_mesh.areaCell},
        radius=1.0,
        edge_lengths=np.array([1700.0 / 6371229.0]),
    )
    assert 8.1e-10 < conus3km["rtol"] < 5.0e-8


def _kite_scaled(mesh: Mesh, relative: float, *, coarsest: bool = False) -> tuple[Mesh, int]:
    changed = mesh.copy()
    kites = np.array(changed.arrays["kiteAreasOnVertex"], copy=True)
    pick = np.argmax if coarsest else np.argmin
    vertex = int(pick(np.asarray(changed.areaTriangle)))
    kites[vertex, 0] *= 1.0 + relative
    changed.arrays["kiteAreasOnVertex"] = kites
    return changed, vertex


def test_binary64_closure_gap_at_producer_precision_is_admitted(
    binary64_mesh: Mesh,
) -> None:
    # 3e-9 on one kite (~1e-9 of its triangle): refused by the old fixed
    # 5e-10, inside the 16 u / theta_min**2 (~2.9e-9) that the fixture's
    # 4.9 km elements admit.
    changed, _ = _kite_scaled(binary64_mesh, 3.0e-9)
    assert changed.validate() is changed


def test_binary64_kite_corruption_is_refused_with_the_measured_bound(
    binary64_mesh: Mesh,
) -> None:
    changed, vertex = _kite_scaled(binary64_mesh, 1.0e-6)
    with pytest.raises(MeshValidationError) as refusal:
        changed.validate()
    text = str(refusal.value)
    assert "kiteAreasOnVertex rows do not sum to areaTriangle" in text
    assert f"at index {vertex}" in text
    assert "53 significand bits" in text
    assert "move circulation between cells" in text


def test_the_bound_is_per_element_so_coarse_triangles_stay_tight(
    binary64_mesh: Mesh,
) -> None:
    # The same 3e-9 that the finest triangle admits is refused on the
    # coarsest one (~3,000 km spacing), whose own bound is still 5e-10: one
    # fine element does not loosen the rest of a graded mesh.
    changed, vertex = _kite_scaled(binary64_mesh, 3.0e-9, coarsest=True)
    with pytest.raises(MeshValidationError) as refusal:
        changed.validate()
    text = str(refusal.value)
    assert f"at index {vertex}" in text
    assert "element bound 5.000e-10" in text
