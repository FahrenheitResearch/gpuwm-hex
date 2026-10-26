"""Rate classes as sub-meshes for active full-step local time stepping.

CPU only, integer and geometry bookkeeping; nothing here advances weather.

A rate class is treated like a partition of the same-host executor: it owns a
set of cells, carries a halo of cells owned by other classes, and runs the
unchanged whole-local-mesh kernels on that sub-mesh.  What differs from a
partition is that the classes do not share a clock.  The law built here is
FINE FIRST with recorded trajectories:

* the class of every cell (index 0 is the COARSEST rate, the last index the
  finest), graded so two neighbouring cells never differ by more than one
  dyadic level;
* toward its COARSER neighbours a class carries a SCRATCH ZONE: coarser cells
  it advances with its own kernels at its own (shorter, always stable) step
  and then discards.  A zone cell is reset to its owner's accepted state
  whenever the owner has one at the step's start time, i.e. at the owner's own
  step boundaries.  Beyond the zone sits the plain three-ring halo of coarser
  cells, FROZEN at the owner's latest accepted state.  The zone is deep enough
  that nothing travelling at ``signal_speed`` from that frozen ring reaches an
  owned cell before the zone is reset (``scratch_zones``), so no owned cell
  ever reads a value its neighbour had to guess;
* toward its FINER neighbours a class carries only the plain halo.  The finer
  class has already covered the whole interval.  With RK2, three dynamics
  splits and N acoustic sub-steps (N even) every stage and acoustic time of a
  class of step H is a multiple of H/(3N): a PLANE of that step.  The finer
  classes record their ACCEPTED states (split ends) at every plane they land
  on.  A plane no recorded state lands on is the linear-in-time interpolation
  of the two recorded states around it -- never an extrapolation.  For N = 2
  every plane is a split end of every strictly finer dyadic class, so the halo
  holds values the model actually computed, and only two classes that share a
  step (an event tail) interpolate, at the odd sixths, with weights 1/2.  For
  N = 4 the odd planes of a class fall in the middle of the next finer class's
  splits and are interpolated with weights 1/2 too; a class two levels finer
  supplies every plane exactly (``plane_weights``);
* the interface register: every edge between cells of different classes.  The
  finer class owns the edge and its integrated mass and theta fluxes are the
  accepted ones; the coarser cell, which moved its own estimate through that
  edge, is refluxed with the difference.  That is what makes the dry mass
  telescope across the interface.

Edge ownership is the partition law with one change: an edge belongs to the
FINER of its two cells' classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .partition_assets_v841 import (
    PartitionLayout,
    _UNREACHED,
    _cell_rings,
    _expand_frontier,
    _order_secondary,
    assert_exact_cover,
)

# Four halo rings past a class's computed cells (its own and its scratch zone).  A row of the outermost ring has
# an incomplete stencil and reads clamped references (partition_local_mesh_v841: an absent neighbour is local index
# 0), and between two fills that garbage travelled three rings inward, into the computed cells.  With three rings
# the result depended on which cell a layout numbers 0: a class cut into blocks across cards (fullstep_lts_blocks)
# parted from the one-card class at the class interface (MPAS route, 240/60 km globe: 0 differing words at the
# first fine steps, 90 w words at 30 min, u off by 1.3 m/s at 3 h), and one card's own forecast moved with the halo
# depth on both physics routes (3 against 4 rings: 3 % of the words at 30 min).  Four rings and five give the same
# bytes on both routes, and the blocks reproduce the one-card class (2026-10-25).
HALO_RINGS = 4
SCRATCH_FLAG = 64   # carried coarser entity: advanced here, reset at its owner's step boundaries
CLASS_MASK = 31
DYADIC = (1, 2, 4, 8, 16, 32)
# Fastest signal the scratch zone must outrun (sound plus wind), and the margin
# on the distance it covers between two resets of the zone.
SIGNAL_SPEED = 400.0
ZONE_FACTOR = 1.5
MIN_ZONE_RINGS = 3


def grade_rates(rates: np.ndarray, cells_on_edge: np.ndarray) -> np.ndarray:
    """Lower rates until no two edge neighbours differ by more than a factor 2.

    Breakage this prevents: an edge between a rate-1 and a rate-4 column would
    put two populated classes next to each other that are not parent and
    child in the recursion, so its register would close over an interval the
    schedule never treats as one unit.
    """

    result = np.array(rates, dtype=np.int32, copy=True)
    coe = inner_edges(cells_on_edge)
    if np.any(~np.isin(result, DYADIC)):
        raise ValueError("rates must be dyadic 1,2,4,8,16,32; other ladders have no nested interval coverage")
    while True:
        previous = result.copy()
        np.minimum.at(result, coe[:, 0], 2 * previous[coe[:, 1]])
        np.minimum.at(result, coe[:, 1], 2 * previous[coe[:, 0]])
        if np.array_equal(result, previous):
            return result


def inner_edges(cells_on_edge: np.ndarray) -> np.ndarray:
    """The edges with a cell on both sides (all of them on a closed mesh).

    A limited-area mesh's outermost edges have one cell; the other side is
    MPAS's absent-cell sentinel (-1).  Such an edge joins no two cells, so it
    takes no part in grading, buffering or the rate interface, and indexing
    with its sentinel would silently read the LAST cell of the mesh.
    """

    coe = np.asarray(cells_on_edge, dtype=np.int64)
    return coe[(coe >= 0).all(axis=1)]


def _with_absence(values: np.ndarray, fill: Any) -> np.ndarray:
    """``values`` with one trailing slot that an absent-cell sentinel indexes."""

    values = np.asarray(values)
    return np.concatenate((values, np.asarray([fill], dtype=values.dtype)))


def _present(cells_on_entity: np.ndarray, n_cells: int) -> np.ndarray:
    """Incidence with the absent-cell sentinel mapped to the one-past index ``n_cells``."""

    incidence = np.asarray(cells_on_entity, dtype=np.int64)
    return np.where(incidence < 0, n_cells, incidence)


def class_indices(rates: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    """Return (rates of the populated classes, coarsest first; class of each cell)."""

    populated = tuple(sorted({int(r) for r in np.unique(rates)}, reverse=True))
    lookup = {rate: index for index, rate in enumerate(populated)}
    cls = np.fromiter((lookup[int(r)] for r in rates), dtype=np.int32, count=len(rates))
    return populated, cls


def _csr(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adjacency = np.asarray(cells_on_cell)
    counts = np.asarray(n_edges_on_cell).astype(np.int64)
    n_cells = adjacency.shape[0]
    live = (np.arange(adjacency.shape[1])[None, :] < counts[:, None]) & (adjacency >= 0)
    if np.any(adjacency[live] >= n_cells):
        raise ValueError("cellsOnCell references a cell outside the mesh")
    # Sorted, deduplicated rows, as the METIS graph file the partition builder
    # reads lists them.
    rows = np.where(live, adjacency, n_cells).astype(np.int64)
    rows.sort(axis=1)
    keep = rows < n_cells
    keep[:, 1:] &= rows[:, 1:] != rows[:, :-1]
    per_row = keep.sum(axis=1)
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(per_row, out=offsets[1:])
    return offsets, rows[keep].astype(np.int32)


def cell_spacing(dc_edge: np.ndarray, cells_on_edge: np.ndarray, n_cells: int) -> np.ndarray:
    """Shortest incident dcEdge of every cell, in the units of ``dc_edge``."""

    dc = np.asarray(dc_edge, dtype=np.float64)
    coe = _present(cells_on_edge, n_cells)
    spacing = np.full(n_cells + 1, np.inf)
    np.minimum.at(spacing, coe[:, 0], dc)
    np.minimum.at(spacing, coe[:, 1], dc)
    return spacing[:n_cells]


def scratch_zones(
    cls: np.ndarray,
    rates: Sequence[int],
    offsets: np.ndarray,
    flat: np.ndarray,
    spacing_m: np.ndarray,
    *,
    base_dt: float,
    signal_speed: float = SIGNAL_SPEED,
    zone_factor: float = ZONE_FACTOR,
    min_rings: int = MIN_ZONE_RINGS,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Per class, the mask of coarser cells it carries as scratch.

    A zone cell of class J is reset every ``rates[J] * base_dt`` seconds.  In
    between, the frozen ring beyond the zone feeds an error inward at no more
    than ``signal_speed``.  The zone therefore reaches out, through coarser
    cells only, until crossing it takes longer than ``zone_factor`` times the
    longest reset period of any class it contains (and at least
    ``min_rings`` rings).  Breakage a shallower zone causes: an owned cell
    next to the interface reads, inside one coarse interval, the wave that a
    frozen boundary value launches, i.e. a neighbour state nobody computed.
    """

    n_classes = len(rates)
    n_cells = int(cls.size)
    zones: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for index in range(n_classes):
        zone = np.zeros(n_cells, dtype=bool)
        info: dict[str, Any] = {"rings": 0, "cells": 0, "crossing_seconds_required": 0.0,
                                "coarsest_rate_inside": None}
        if index == 0:
            zones.append(zone)
            receipts.append(info)
            continue
        allowed = cls < index
        required = zone_factor * base_dt * rates[index - 1]
        while True:
            zone[:] = False
            frontier = np.flatnonzero(cls == index).astype(np.int32)
            far_side = np.zeros(n_cells)            # owned cells: zero
            rings = 0
            while frontier.size:
                candidates = np.unique(_expand_frontier(frontier, offsets, flat))
                fresh = candidates[allowed[candidates] & ~zone[candidates]]
                if fresh.size == 0:
                    break
                # nearest already-reached neighbour decides how far the signal still has to go
                near = np.full(fresh.size, np.inf)
                for position, cell in enumerate(fresh):
                    row = flat[offsets[cell]:offsets[cell + 1]]
                    reached = row[(cls[row] == index) | zone[row]]
                    near[position] = far_side[reached].min()
                keep = (near < required) | (rings < min_rings)
                fresh = fresh[keep]
                if fresh.size == 0:
                    break
                far_side[fresh] = near[keep] + spacing_m[fresh] / signal_speed
                zone[fresh] = True
                frontier = fresh.astype(np.int32)
                rings += 1
            coarsest = int(cls[zone].min()) if zone.any() else index - 1
            needed = zone_factor * base_dt * rates[coarsest]
            if needed <= required * (1.0 + 1e-12):
                break
            required = needed
        info.update(rings=rings, cells=int(zone.sum()), crossing_seconds_required=float(required),
                    coarsest_rate_inside=int(rates[coarsest]))
        zones.append(zone)
        receipts.append(info)
    return zones, receipts


def build_class_layouts(
    *,
    cells_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    cells_on_vertex: np.ndarray,
    cell_class: np.ndarray,
    zones: Sequence[np.ndarray] | None = None,
    edges_on_edge: np.ndarray | None = None,
    n_edges_on_edge: np.ndarray | None = None,
    halo_rings: int = HALO_RINGS,
    replicated_cells: Sequence[int] = (),
) -> list[PartitionLayout]:
    """One layout per class; the partition builder's laws with fine-side edges and carried zones.

    ``replicated_cells`` are global cells every class must hold locally
    whoever owns them (the limited-area garbage donor: the regional driver
    lends that real column's physics inputs to native's garbage column, and
    refuses a sub-mesh that lacks it).  In a class that does not own or
    carry one it is a plain halo cell with its own halo rings, filled from
    its owner like any other; it is never advanced as scratch.
    """

    cls = np.asarray(cell_class, dtype=np.int32)
    n_cells = int(cls.size)
    raw_coe = np.asarray(cells_on_edge, dtype=np.int64)
    raw_cov = np.asarray(cells_on_vertex, dtype=np.int64)
    # A limited-area mesh carries the absent-cell sentinel (-1) on its outer
    # edges and vertices.  It maps to the one-past index, and every table
    # below carries one trailing "absent" slot (the partition builder's law):
    # never local-missing, never complete-blocking, never an owner.
    if (raw_coe.ndim != 2 or raw_coe.shape[1] != 2 or raw_coe.min() < -1 or raw_coe.max() >= n_cells
            or np.any((raw_coe < 0).all(axis=1))):
        raise ValueError("cellsOnEdge references a cell outside the mesh, or an edge has no cell")
    if raw_cov.min() < -1 or raw_cov.max() >= n_cells or np.any((raw_cov < 0).all(axis=1)):
        raise ValueError("cellsOnVertex references a cell outside the mesh, or a vertex has no cell")
    coe = _present(raw_coe, n_cells)
    cov = _present(raw_cov, n_cells)
    cls_a = _with_absence(cls, -1)
    n_edges, n_vertices = int(coe.shape[0]), int(cov.shape[0])
    n_classes = int(cls.max()) + 1
    offsets, flat = _csr(cells_on_cell, n_edges_on_cell)
    # The finer class (larger index) owns a rate-interface edge.
    edge_owner = np.maximum(cls_a[coe[:, 0]], cls_a[coe[:, 1]])
    vertex_owner = cls_a[cov].max(axis=1)
    eoe_active = eoe_safe = None
    if edges_on_edge is not None:
        eoe = np.asarray(edges_on_edge)
        counts = np.asarray(n_edges_on_edge)
        eoe_active = (np.arange(eoe.shape[1])[None, :] < counts[:, None]) & (eoe >= 0)
        eoe_safe = np.where(eoe >= 0, eoe, 0).astype(np.int64)
    neighbour_of = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(offsets))
    layouts: list[PartitionLayout] = []
    for partition in range(n_classes):
        owned = np.flatnonzero(cls == partition).astype(np.int32)
        if owned.size == 0:
            raise ValueError(f"class {partition} owns no cells")
        carried = np.empty(0, dtype=np.int32) if zones is None else np.flatnonzero(zones[partition]).astype(np.int32)
        seeds = np.union1d(carried, np.asarray(tuple(replicated_cells), dtype=np.int32)).astype(np.int32)
        cell_l2g, cell_ring = _cell_rings(owned, offsets, flat, n_cells=n_cells, halo_rings=halo_rings,
                                          replicated_cells=seeds if seeds.size else None)
        ring_global = np.full(n_cells + 1, _UNREACHED, dtype=np.int32)
        ring_global[cell_l2g.astype(np.int64)] = cell_ring
        local = ring_global < _UNREACHED
        local[n_cells] = True                       # the absent cell is never missing
        incomplete = np.bincount(neighbour_of[~local[flat.astype(np.int64)]], minlength=n_cells)
        complete = _with_absence(incomplete == 0, True)
        cell_row_complete = complete[cell_l2g.astype(np.int64)] & local[cell_l2g.astype(np.int64)]
        edge_ring = ring_global[coe].min(axis=1)
        edge_l2g, n_owned_edges = _order_secondary(edge_owner, edge_ring, partition)
        edge_complete = local[coe].all(axis=1) & complete[coe].all(axis=1)
        if eoe_active is not None:
            edge_local = np.zeros(n_edges, dtype=bool)
            edge_local[edge_l2g.astype(np.int64)] = True
            edge_complete &= np.where(eoe_active, edge_local[eoe_safe], True).all(axis=1)
        vertex_ring = ring_global[cov].min(axis=1)
        vertex_l2g, n_owned_vertices = _order_secondary(vertex_owner, vertex_ring, partition)
        vertex_complete = local[cov].all(axis=1)
        layout = PartitionLayout(
            partition=partition,
            n_owned_cells=int(owned.size),
            n_owned_edges=int(n_owned_edges),
            n_owned_vertices=int(n_owned_vertices),
            cell_l2g=cell_l2g,
            edge_l2g=edge_l2g,
            vertex_l2g=vertex_l2g,
            cell_ring=cell_ring,
            cell_row_complete=np.ascontiguousarray(cell_row_complete, dtype=bool),
            edge_row_complete=np.ascontiguousarray(edge_complete[edge_l2g.astype(np.int64)], dtype=bool),
            vertex_row_complete=np.ascontiguousarray(vertex_complete[vertex_l2g.astype(np.int64)], dtype=bool),
        )
        object.__setattr__(layout, "_global_cells", n_cells)
        object.__setattr__(layout, "_global_edges", n_edges)
        object.__setattr__(layout, "_global_vertices", n_vertices)
        object.__setattr__(layout, "_replicated_cells", tuple(int(v) for v in carried))
        layouts.append(layout)
    assert_exact_cover(layouts, n_cells=n_cells, n_edges=n_edges, n_vertices=n_vertices)
    return layouts


@dataclass(frozen=True)
class ClassTables:
    """Index law of one class against the mailbox and the register."""

    index: int
    rate: int
    layout: PartitionLayout
    # halo = the non-owned tail of the local ordering
    halo_cell_mail: np.ndarray      # mailbox id of local cell n_owned+i
    halo_cell_class: np.ndarray     # owning class, | SCRATCH_FLAG when carried
    halo_edge_mail: np.ndarray
    halo_edge_class: np.ndarray
    # published = owned entities some other class holds in its halo
    pub_cell_local: np.ndarray
    pub_cell_mail: np.ndarray
    pub_edge_local: np.ndarray
    pub_edge_mail: np.ndarray
    # register edges this class sees: local edge, slot, the edge's two local cells
    fine_edge_local: np.ndarray     # interface edges with the next coarser class (owned here)
    fine_slot: np.ndarray
    fine_cells: np.ndarray          # (n, 2)
    coarse_edge_local: np.ndarray   # interface edges with the next finer class (halo here)
    coarse_slot: np.ndarray
    coarse_cells: np.ndarray
    child: int                      # index of the next finer class, or -1
    reflux_cell_local: np.ndarray   # owned cells on a coarse-side register edge
    reflux_slot: np.ndarray         # (n_cells, width) register slots, padded with 0
    reflux_coef: np.ndarray         # (n_cells, width) float64  -sign/area, padded with 0
    reflux_edge_local: np.ndarray   # owned edges whose velocity reads a refluxed density
    reflux_edge_cells: np.ndarray   # (n, 2) their local cells
    finest_halo_distance: int       # largest (owner class - this class) in the halo


@dataclass(frozen=True)
class ClassPlan:
    rates: tuple[int, ...]          # coarsest first
    cell_class: np.ndarray
    classes: tuple[ClassTables, ...]
    n_mail_cells: int
    n_mail_edges: int
    n_register_edges: int
    register_edge_global: np.ndarray
    register_coarse_class: np.ndarray
    level_depth: int                # trajectory plane groups the mailbox needs
    zones: tuple[dict[str, Any], ...]
    zone_policy: dict[str, float]

    def receipt(self) -> dict[str, Any]:
        return {
            "rates_coarsest_first": list(self.rates),
            "cells_per_class": [int(t.layout.n_owned_cells) for t in self.classes],
            "local_cells_per_class": [int(t.layout.n_local_cells) for t in self.classes],
            "edges_per_class": [int(t.layout.n_owned_edges) for t in self.classes],
            "local_edges_per_class": [int(t.layout.n_local_edges) for t in self.classes],
            "scratch_cells_per_class": [int(np.count_nonzero(t.halo_cell_class & SCRATCH_FLAG)) for t in self.classes],
            "scratch_zone": list(self.zones),
            "zone_policy": dict(self.zone_policy),
            "mailbox_cells": int(self.n_mail_cells),
            "mailbox_edges": int(self.n_mail_edges),
            "mailbox_level_depth": int(self.level_depth),
            "register_edges": int(self.n_register_edges),
            "refluxed_cells_per_class": [int(t.reflux_cell_local.size) for t in self.classes],
            "cell_work_per_base_step": float(sum(t.layout.n_owned_cells / t.rate for t in self.classes)),
            "local_cell_work_per_base_step": float(sum(t.layout.n_local_cells / t.rate for t in self.classes)),
        }


def build_class_plan(
    *,
    rates: np.ndarray,
    cells_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    cells_on_vertex: np.ndarray,
    area_cell: np.ndarray,
    dc_edge_m: np.ndarray,
    base_dt: float,
    edges_on_edge: np.ndarray | None = None,
    n_edges_on_edge: np.ndarray | None = None,
    halo_rings: int = HALO_RINGS,
    signal_speed: float = SIGNAL_SPEED,
    zone_factor: float = ZONE_FACTOR,
    min_zone_rings: int = MIN_ZONE_RINGS,
    replicated_cells: Sequence[int] = (),
) -> ClassPlan:
    raw_coe = np.asarray(cells_on_edge, dtype=np.int64)
    graded = grade_rates(rates, raw_coe)
    if not np.array_equal(graded, np.asarray(rates)):
        raise ValueError("the rate map is not graded; call grade_rates first so the plan and the caller agree")
    populated, cls = class_indices(graded)
    n_cells, n_edges = int(cls.size), int(raw_coe.shape[0])
    coe = _present(raw_coe, n_cells)
    cls_a = _with_absence(cls, -1)
    both = (raw_coe >= 0).all(axis=1)
    offsets, flat = _csr(cells_on_cell, n_edges_on_cell)
    spacing = cell_spacing(dc_edge_m, raw_coe, n_cells)
    zones, zone_receipts = scratch_zones(
        cls, populated, offsets, flat, spacing, base_dt=float(base_dt), signal_speed=float(signal_speed),
        zone_factor=float(zone_factor), min_rings=int(min_zone_rings))
    layouts = build_class_layouts(
        cells_on_cell=cells_on_cell, n_edges_on_cell=n_edges_on_cell, cells_on_edge=raw_coe,
        cells_on_vertex=cells_on_vertex, cell_class=cls, zones=zones, edges_on_edge=edges_on_edge,
        n_edges_on_edge=n_edges_on_edge, halo_rings=halo_rings, replicated_cells=replicated_cells)
    edge_owner = np.maximum(cls_a[coe[:, 0]], cls_a[coe[:, 1]])
    # Mailbox: every entity that is non-owned somewhere.
    cell_shared = np.zeros(n_cells, dtype=bool)
    edge_shared = np.zeros(n_edges, dtype=bool)
    for layout in layouts:
        cell_shared[layout.cell_l2g[layout.n_owned_cells:].astype(np.int64)] = True
        edge_shared[layout.edge_l2g[layout.n_owned_edges:].astype(np.int64)] = True
    cell_mail = np.full(n_cells, -1, dtype=np.int32)
    cell_mail[cell_shared] = np.arange(int(cell_shared.sum()), dtype=np.int32)
    edge_mail = np.full(n_edges, -1, dtype=np.int32)
    edge_mail[edge_shared] = np.arange(int(edge_shared.sum()), dtype=np.int32)
    # Register: every edge between two classes.  Grading makes them parent and child.
    # (an outer edge of a limited-area mesh has one cell and is never one)
    register = np.flatnonzero(both & (cls_a[coe[:, 0]] != cls_a[coe[:, 1]]))
    fine_side = np.maximum(cls[coe[register, 0]], cls[coe[register, 1]])
    coarse_side = np.minimum(cls[coe[register, 0]], cls[coe[register, 1]])
    if np.any(fine_side - coarse_side != 1):
        raise AssertionError("a register edge joins classes that are not parent and child")
    area = np.asarray(area_cell, dtype=np.float64)
    tables = []
    level_depth = 1
    for layout in layouts:
        index = int(layout.partition)
        c_l2g = layout.cell_l2g.astype(np.int64)
        e_l2g = layout.edge_l2g.astype(np.int64)
        c_g2l = layout.cell_g2l().astype(np.int64)
        e_g2l = layout.edge_g2l().astype(np.int64)
        halo_cells = c_l2g[layout.n_owned_cells:]
        halo_edges = e_l2g[layout.n_owned_edges:]
        owned_cells = c_l2g[:layout.n_owned_cells]
        owned_edges = e_l2g[:layout.n_owned_edges]
        zone = _with_absence(zones[index], False)
        cell_flag = np.where(zone[halo_cells], SCRATCH_FLAG, 0)
        # A non-owned edge of a coarser class is advanced here when it touches
        # a zone cell; one between two frozen cells stays frozen.
        edge_flag = np.where((edge_owner[halo_edges] < index) & zone[coe[halo_edges]].any(axis=1), SCRATCH_FLAG, 0)
        if np.any((cls[halo_cells] == index)) or np.any(edge_owner[halo_edges] == index):
            raise AssertionError(f"class {index}: an owned entity sits in its own halo")
        if np.any(zone[halo_cells] & (cls[halo_cells] > index)):
            raise AssertionError(f"class {index}: a finer cell entered the scratch zone")
        finer = cls[halo_cells][cls[halo_cells] > index]
        distance = int(finer.max() - index) if finer.size else 0
        finer_edges = edge_owner[halo_edges][edge_owner[halo_edges] > index]
        if finer_edges.size:
            distance = max(distance, int(finer_edges.max() - index))
        level_depth = max(level_depth, distance)
        pub_c = np.flatnonzero(cell_shared[owned_cells])
        pub_e = np.flatnonzero(edge_shared[owned_edges])
        fine_slots = np.flatnonzero(fine_side == index).astype(np.int32)
        coarse_slots = np.flatnonzero(coarse_side == index).astype(np.int32)
        fine_edges_g = register[fine_slots]
        coarse_edges_g = register[coarse_slots]
        for name, edges_g in (("fine", fine_edges_g), ("coarse", coarse_edges_g)):
            if np.any(e_g2l[edges_g] < 0) or np.any(c_g2l[coe[edges_g]] < 0):
                raise AssertionError(f"class {index}: a {name}-side register edge or its cells are not local")
        if fine_edges_g.size and (np.any(e_g2l[fine_edges_g] >= layout.n_owned_edges)
                                  or not np.all(layout.edge_row_complete[e_g2l[fine_edges_g]])):
            raise AssertionError(f"class {index}: a fine-side register edge is not owned with a complete row")
        # Reflux table: one row per owned cell on a coarse-side register edge,
        # its register edges in ascending slot order.
        first_is_mine = cls[coe[coarse_edges_g, 0]] == index
        coarse_cell_g = np.where(first_is_mine, coe[coarse_edges_g, 0], coe[coarse_edges_g, 1])
        sign = np.where(first_is_mine, 1.0, -1.0)
        reflux_cells_g = np.unique(coarse_cell_g)
        row_of = np.searchsorted(reflux_cells_g, coarse_cell_g)
        width = int(np.bincount(row_of, minlength=reflux_cells_g.size).max()) if reflux_cells_g.size else 1
        reflux_slot = np.zeros((reflux_cells_g.size, width), dtype=np.int32)
        reflux_coef = np.zeros((reflux_cells_g.size, width), dtype=np.float64)
        fill = np.zeros(reflux_cells_g.size, dtype=np.int64)
        for position in np.argsort(coarse_slots, kind="stable"):
            row = row_of[position]
            reflux_slot[row, fill[row]] = coarse_slots[position]
            # density_tendency = -(sum sign*dv*ru)/area; the register already holds dv*ru*dt.
            reflux_coef[row, fill[row]] = -sign[position] / area[coarse_cell_g[position]]
            fill[row] += 1
        touched = np.zeros(n_cells + 1, dtype=bool)
        touched[reflux_cells_g] = True
        reflux_edge = np.flatnonzero(touched[coe[owned_edges]].any(axis=1) & both[owned_edges])
        tables.append(ClassTables(
            index=index, rate=int(populated[index]), layout=layout,
            halo_cell_mail=cell_mail[halo_cells],
            halo_cell_class=(cls[halo_cells] | cell_flag).astype(np.int32),
            halo_edge_mail=edge_mail[halo_edges],
            halo_edge_class=(edge_owner[halo_edges] | edge_flag).astype(np.int32),
            pub_cell_local=pub_c.astype(np.int32), pub_cell_mail=cell_mail[owned_cells[pub_c]],
            pub_edge_local=pub_e.astype(np.int32), pub_edge_mail=edge_mail[owned_edges[pub_e]],
            fine_edge_local=e_g2l[fine_edges_g].astype(np.int32), fine_slot=fine_slots,
            fine_cells=c_g2l[coe[fine_edges_g]].astype(np.int32).reshape(-1, 2),
            coarse_edge_local=e_g2l[coarse_edges_g].astype(np.int32), coarse_slot=coarse_slots,
            coarse_cells=c_g2l[coe[coarse_edges_g]].astype(np.int32).reshape(-1, 2),
            child=index + 1 if index + 1 < len(layouts) else -1,
            reflux_cell_local=c_g2l[reflux_cells_g].astype(np.int32),
            reflux_slot=reflux_slot, reflux_coef=reflux_coef,
            reflux_edge_local=reflux_edge.astype(np.int32),
            reflux_edge_cells=c_g2l[coe[owned_edges[reflux_edge]]].astype(np.int32).reshape(-1, 2),
            finest_halo_distance=distance,
        ))
        last = tables[-1]
        if np.any(last.halo_cell_mail < 0) or np.any(last.halo_edge_mail < 0):
            raise AssertionError("a halo entity has no mailbox slot")
        if np.any(last.reflux_cell_local >= layout.n_owned_cells):
            raise AssertionError("a refluxed cell is not owned by the coarse class")
    return ClassPlan(
        rates=populated, cell_class=cls, classes=tuple(tables),
        n_mail_cells=int(cell_shared.sum()), n_mail_edges=int(edge_shared.sum()),
        n_register_edges=int(register.size), register_edge_global=register.astype(np.int64),
        register_coarse_class=coarse_side.astype(np.int32), level_depth=int(level_depth),
        zones=tuple(zone_receipts),
        zone_policy={"signal_speed_m_per_s": float(signal_speed), "zone_factor": float(zone_factor),
                     "min_rings": int(min_zone_rings), "base_dt": float(base_dt)})


def macro_schedule(rates: Sequence[int], macro_rate: int) -> tuple[tuple[str, int, int, int], ...]:
    """The recursive event order of one macro of ``macro_rate`` base steps: FINE FIRST.

    The order does not depend on the acoustic sub-step count.

    Rows are ``(event, class, start, length)`` in base steps from the macro
    start.  Before a class steps over an interval, every finer class has
    covered that interval in its own steps and recorded its accepted states;
    ``step`` then advances the class once with those states in its halo, and
    ``sync`` refluxes it at its finer interface and publishes its accepted
    state.  A class whose rate exceeds the macro takes the macro's rate (an
    event tail).
    """

    if macro_rate not in DYADIC:
        raise ValueError("a macro spans a dyadic number of base steps")
    effective = [min(int(rate), int(macro_rate)) for rate in rates]
    if any(effective[i] < effective[i + 1] for i in range(len(effective) - 1)):
        raise ValueError("classes must be ordered coarsest first")
    events: list[tuple[str, int, int, int]] = []

    def run(index: int, start: int, span: int) -> None:
        if index == len(effective):
            return
        length = effective[index]
        for part in range(span // length):
            begin = start + part * length
            run(index + 1, begin, length)
            events.append(("step", index, begin, length))
            events.append(("sync", index, begin, length))

    run(0, 0, int(macro_rate))
    return tuple(events)


def lts_acoustic_substeps(count: Any) -> int:
    """The acoustic sub-step count the class clock is built for, or ValueError.

    The clock counts ticks of ``base_dt / (3 N)``: with RK2 and three dynamics
    splits, every stage and acoustic sub-step of a class of rate r ends on a
    multiple of r ticks (a plane).  Breakage the refusal prevents: with an odd
    N the middle of a coarser class's step (3N/2 planes) is not a plane, so
    the scalar interface could not publish the finer class's mid-step accepted
    scalars there and the scalar register would close over an interval no
    class recorded; with N < 2 the first two RK2 stages have no sub-step of
    their own (MPAS takes max(1, N/2)) and would end at a whole split.
    """

    value = int(count)
    if value != count or value < 2 or value % 2:
        raise ValueError(
            f"full-step LTS needs an even acoustic sub-step count of at least 2, not {count!r}: with an "
            "odd count the middle of a coarser class's step is not a plane of its clock, so the finer "
            "class's mid-step scalars could not be published there and the scalar register would close "
            "over an interval no class recorded")
    return value


def planes_per_step(acoustic_substeps: int) -> int:
    """Planes (stage and acoustic times) of one class step: 3 splits times N sub-steps."""

    return 3 * lts_acoustic_substeps(acoustic_substeps)


def trajectory_plane(consumer_start: int, consumer_length: int, time: int, planes: int) -> int | None:
    """Which plane of the consumer's step ``time`` is, or None.

    Ticks are ``base_dt / planes`` with ``planes = 3 N`` (``planes_per_step``):
    a class of effective rate r steps ``planes * r`` ticks, and its stages and
    acoustic sub-steps end on multiples of ``r`` ticks from its step start.
    """

    offset = int(time) - int(consumer_start)
    if offset <= 0 or offset > int(planes) * consumer_length or offset % consumer_length:
        return None
    return offset // consumer_length


def plane_weights(plane: int, consumer_length: int, provider_length: int,
                  acoustic_substeps: int) -> tuple[int, int, float, float]:
    """``(a, b, wa, wb)``: the provider's halo value at a plane of the consumer's step.

    The provider (a finer class, or one that shares the step in an event tail)
    records its accepted states -- split ends, every ``N * provider_length``
    ticks -- on the planes they land on (``trajectory_plane``).  Recorded
    planes are the multiples of ``stride = lcm(L, N P) / L`` (L the consumer's
    and P the provider's effective length).  A recorded plane is read as it is
    (``b == a``, ``wb == 0``); any other plane is the linear-in-time
    interpolation of the recorded planes around it.  For N = 2 and a strictly
    finer dyadic provider the stride is 1: no interpolation at all.
    """

    import math

    count = lts_acoustic_substeps(acoustic_substeps)
    split = count * int(provider_length)
    stride = (int(consumer_length) * split // math.gcd(int(consumer_length), split)) // int(consumer_length)
    plane = int(plane)
    offset = plane % stride
    if offset == 0:
        return plane, plane, 1.0, 0.0
    a = plane - offset
    wb = offset / stride
    return a, a + stride, 1.0 - wb, wb


def child_flux_intervals(consumer_start: int, consumer_length: int, split_end: int, split_ticks: int,
                         planes: int) -> range:
    """The planes of the parent's step a child's split spans, in ticks of ``base_dt / planes``.

    ``planes = 3 N`` (``planes_per_step``).  The parent's acoustic sub-steps
    span exactly one plane each (``N`` per split, ``N / 2`` in each of the
    first two RK stages), so the matched interface flux reads, for the
    sub-step over plane interval ``q``, the child's accepted mean over the
    child split that contains it.  A dyadic child (half the parent's step)
    ends a split every ``N / 2`` planes; a child that shares the parent's step
    (an event tail) every ``N`` planes.  Every plane interval the child's
    split covers takes that split's mean, so the parent moves through the
    edge exactly the integral the child moved (with N = 2: one plane per
    dyadic split, the sixths of the original fix).  Breakage a misfit would
    cause: the parent would move mass through the edge over one interval with
    the child's flux of another, which the reflux would then have to repair
    after the fact.
    """

    if consumer_length <= 0 or split_ticks <= 0 or split_ticks % consumer_length:
        raise ValueError("a child's split does not tile its parent's planes")
    count = split_ticks // consumer_length
    plane = trajectory_plane(consumer_start, consumer_length, split_end, planes)
    if plane is None or count < 1 or count > int(planes) // 3 or plane - count < 0:
        raise ValueError(f"a split ending at tick {split_end} is not on a plane of the parent's step "
                         f"starting at {consumer_start} with planes of {consumer_length} ticks")
    return range(plane - count, plane)


def plan_macros(n_base_steps: int, max_rate: int, *, warmup_steps: int = 0) -> tuple[int, ...]:
    """Macro lengths covering exactly ``n_base_steps``: optional base-step warm-up, then greedy dyadic."""

    if n_base_steps < 0 or max_rate not in DYADIC or warmup_steps < 0:
        raise ValueError("invalid macro plan request")
    macros: list[int] = []
    remaining = int(n_base_steps)
    for _ in range(min(int(warmup_steps), remaining)):
        macros.append(1)
        remaining -= 1
    while remaining:
        length = max(rate for rate in DYADIC if rate <= max_rate and rate <= remaining)
        macros.append(length)
        remaining -= length
    return tuple(macros)
