"""Rate classes split into blocks across the cards of one host (active full-step LTS).

CPU only, integer and geometry bookkeeping; nothing here advances weather.

The one-device plan (``fullstep_lts_classes``) gives every rate class one
sub-mesh: its owned cells, a SCRATCH ZONE of coarser cells it advances itself,
and the halo rings beyond.  Call the owned cells and the zone together the
class's COMPUTED SET.  This module cuts each class's computed set into blocks,
one per card that class uses, exactly as the same-host executor cuts a global
mesh into partitions:

* a block owns some of the class's cells, carries some of its zone cells, and
  holds a three-ring halo around both;
* a halo entity that another block of the SAME class computes is a PEER
  entity: it is refreshed at every native round (A, B, C, D, E, P) by the
  same-host byte transport, so the block computes its own entities exactly as
  the one-device class sub-mesh does (the partition-invariance law of the
  same-host executor, applied to a class instead of a mesh);
* every other halo entity belongs to another class and is filled from the
  mailbox exactly as on one device (finer owners' accepted trajectory,
  coarser owners' frozen or reset state).

Ownership of an edge inside a class follows the partition law (the block of
its lower-numbered cell) with one exception that the scalar interface needs:
an edge between the class and the next COARSER class belongs to the block that
carries that edge's coarse cell.  All interface edges of one coarse column
therefore sit in one block, which holds the column's donor reserve and caps
the finer class's packets against it with every flux in hand.

Every limiter adjustment a block makes to an edge it does not own (the donor
guard of a coarse column, the reserve cap of a finer one) is a per-column
factor; the factors are exchanged between the class's blocks and applied by
every block that holds the column (``cuda_fullstep_lts_multi``), so each copy
of an edge flux carries the same bytes.

Rate classes keep their order and their schedule: every block of a class steps
at the same time on its own card, and the classes step one after another, fine
first, as on one device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .fullstep_lts_classes import SCRATCH_FLAG, ClassPlan, _csr
from .partition_assets_v841 import (
    PartitionLayout,
    _UNREACHED,
    _cell_rings,
    _expand_frontier,
    assert_exact_cover,
)

HALO_RINGS = 3


def rcb_parts(points: np.ndarray, n_parts: int) -> np.ndarray:
    """Recursive coordinate bisection: compact parts of (nearly) equal count.

    Deterministic (stable sorts); the longest extent is cut at each level.
    """

    points = np.asarray(points, dtype=np.float64)
    labels = np.zeros(points.shape[0], dtype=np.int32)
    if n_parts < 1 or n_parts > max(points.shape[0], 1):
        raise ValueError(f"cannot cut {points.shape[0]} cells into {n_parts} parts")

    def split(index: np.ndarray, first: int, count: int) -> None:
        if count == 1:
            labels[index] = first
            return
        left = count // 2
        sub = points[index]
        axis = int(np.argmax(sub.max(axis=0) - sub.min(axis=0)))
        order = np.argsort(sub[:, axis], kind="stable")
        cut = int(round(index.size * left / count))
        split(index[order[:cut]], first, left)
        split(index[order[cut:]], first + left, count - left)

    split(np.arange(points.shape[0]), 0, int(n_parts))
    return labels


def _assign_zone(part: np.ndarray, zone: np.ndarray, offsets: np.ndarray, flat: np.ndarray) -> None:
    """Give every zone cell the smallest part among its already assigned neighbours, ring by ring."""

    neighbour_of = np.repeat(np.arange(offsets.size - 1, dtype=np.int64), np.diff(offsets))
    frontier = np.flatnonzero(part >= 0).astype(np.int32)
    big = np.iinfo(np.int32).max
    while frontier.size:
        candidates = np.unique(_expand_frontier(frontier, offsets, flat))
        candidates = candidates[zone[candidates] & (part[candidates] < 0)]
        if candidates.size == 0:
            break
        is_candidate = np.zeros(part.size, dtype=bool)
        is_candidate[candidates] = True
        slots = np.flatnonzero(is_candidate[neighbour_of])
        owners = part[flat[slots]]
        keep = owners >= 0
        best = np.full(part.size, big, dtype=np.int64)
        np.minimum.at(best, neighbour_of[slots[keep]], owners[keep])
        part[candidates] = best[candidates].astype(np.int32)
        frontier = candidates.astype(np.int32)
    if np.any(zone & (part < 0)):
        raise AssertionError("a scratch-zone cell is not connected to its class through the zone")


@dataclass(frozen=True)
class BlockTables:
    """Index law of one block against the transport, the mailbox and the registers."""

    index: int                  # class index (0 = coarsest)
    rate: int
    block: int                  # block number within the class
    gid: int                    # global block number
    device: int
    rank: int                   # position in the plan's device list (a card may hold several ranks)
    layout: PartitionLayout     # owned = the class cells this block owns
    # exchange ownership: owned + carried zone cells this block computes
    n_xown_cells: int
    n_xlocal_cells: int         # end of the peer halo
    n_xown_edges: int
    n_xlocal_edges: int
    halo_cell_mail: np.ndarray
    halo_cell_class: np.ndarray
    halo_edge_mail: np.ndarray
    halo_edge_class: np.ndarray
    pub_cell_local: np.ndarray
    pub_cell_mail: np.ndarray
    pub_edge_local: np.ndarray
    pub_edge_mail: np.ndarray
    # fine side (interface with the next coarser class): owned edges here
    fine_edge_local: np.ndarray
    fine_slot: np.ndarray
    fine_cells: np.ndarray
    # every local interface edge with the next coarser class (velocity recovery after a reset)
    up_edge_local: np.ndarray
    up_edge_cells: np.ndarray
    # coarse side (interface with the next finer class): edges of this block's owned cells
    coarse_edge_local: np.ndarray
    coarse_slot: np.ndarray
    coarse_cells: np.ndarray
    reflux_cell_local: np.ndarray
    reflux_slot: np.ndarray
    reflux_coef: np.ndarray
    reflux_edge_local: np.ndarray
    reflux_edge_cells: np.ndarray
    # every local packet edge (coarse side) with both cells local
    packet_edge_local: np.ndarray
    packet_slot: np.ndarray
    # donor-guard rows: owned rows (== reflux rows) first, then halo rows with complete stencils
    guard_cell_local: np.ndarray
    n_guard_owned: int
    # reserve-cap rows (coarse columns of the next coarser class): owned first
    cap_cell_local: np.ndarray
    n_cap_owned: int
    cap_slot: np.ndarray
    cap_coef: np.ndarray
    cap_area: np.ndarray        # area of the owned cap rows
    edge_of_slot: np.ndarray    # (n_slots,) local edge of every local fine-side interface edge, else -1
    child: int
    finest_halo_distance: int


@dataclass(frozen=True)
class PeerTables:
    peer: int
    cell_send: np.ndarray
    cell_recv: np.ndarray
    edge_send: np.ndarray
    edge_recv: np.ndarray


@dataclass(frozen=True)
class TransportTables:
    """The same-host transport's table contract over one block's exchange-owned/peer extents."""

    rank: int
    n_ranks: int
    n_owned_cells: int
    n_local_cells: int
    n_owned_edges: int
    n_local_edges: int
    peers: tuple[PeerTables, ...]


@dataclass(frozen=True)
class BlockPlan:
    plan: ClassPlan
    devices: tuple[int, ...]
    blocks: tuple[BlockTables, ...]             # in class order, then block order
    by_class: tuple[tuple[int, ...], ...]       # global block numbers of each class
    transport: tuple[tuple[TransportTables, ...] | None, ...]   # per class; None for one block
    # mailbox routes: (source block gid, device) -> mail ids the device reads
    route_cells: dict[tuple[int, int], np.ndarray]
    route_edges: dict[tuple[int, int], np.ndarray]
    # union of each class's published mail ids (for the per-device plane rotation)
    class_pub_cells: tuple[np.ndarray, ...]
    class_pub_edges: tuple[np.ndarray, ...]
    # register pulls before a class steps: class -> [(writer device, reader device, slots)]
    register_pulls: tuple[tuple[tuple[int, int, np.ndarray], ...], ...]
    # class -> [(writer device, slots zeroed there after the pulls)]
    register_zero_after_pull: tuple[tuple[tuple[int, np.ndarray], ...], ...]

    def receipt(self) -> dict[str, Any]:
        rows = []
        for t in self.blocks:
            rows.append({
                "class": t.index, "rate": t.rate, "block": t.block, "device": t.device, "rank": t.rank,
                "owned_cells": int(t.layout.n_owned_cells), "carried_cells": int(t.n_xown_cells - t.layout.n_owned_cells),
                "peer_cells": int(t.n_xlocal_cells - t.n_xown_cells),
                "mail_cells": int(t.layout.n_local_cells - t.n_xlocal_cells),
                "local_cells": int(t.layout.n_local_cells), "owned_edges": int(t.layout.n_owned_edges),
                "local_edges": int(t.layout.n_local_edges)})
        per_device: dict[int, dict[str, float]] = {}
        for t in self.blocks:
            row = per_device.setdefault(t.device, {"owned_cell_work": 0.0, "local_cell_work": 0.0, "blocks": 0})
            row["owned_cell_work"] += t.layout.n_owned_cells / t.rate
            row["local_cell_work"] += t.layout.n_local_cells / t.rate
            row["blocks"] += 1
        return {"schema": "gpuwm-hex.fullstep-lts-blocks/v1", "devices": list(self.devices),
                "blocks_per_class": [len(g) for g in self.by_class], "blocks": rows,
                "per_device": {str(k): v for k, v in sorted(per_device.items())},
                "mail_routes": {"cell_ids": int(sum(v.size for v in self.route_cells.values())),
                                "edge_ids": int(sum(v.size for v in self.route_edges.values()))}}


#: Owned cells per block below which a class is not cut further.  It was 50,000, which kept every class
#: under 100,000 owned cells whole on ONE card of two while the other card idled (classes step one after
#: another).  THE BREAKAGE THAT CAUSED (measured 2026-10-15 on 2 x RTX PRO 6000, Python 3.14t, a 60/30/15 km
#: variable mesh of 222,042 cells classed 144,039 / 36,093 / 41,910 at rates 4 / 2 / 1, full ArWen physics):
#: the rate-2 and rate-1 classes, 6 of every 7 class steps, ran on card 0 alone, and two-card LTS took
#: 21.2 s per forecast hour against 26.9 on one card.  With every class cut in two the same run takes 16.9.
DEFAULT_MIN_BLOCK_CELLS = 16_000


def blocks_per_class(plan: ClassPlan, n_devices: int, *,
                     min_block_cells: int = DEFAULT_MIN_BLOCK_CELLS) -> tuple[int, ...]:
    """How many cards each class is cut across: one block per ``min_block_cells`` owned cells, at most one per card."""

    counts = []
    for tables in plan.classes:
        owned = int(tables.layout.n_owned_cells)
        counts.append(int(max(1, min(int(n_devices), owned // max(int(min_block_cells), 1)))))
    return tuple(counts)


def build_block_plan(
    plan: ClassPlan,
    *,
    devices: Sequence[int],
    n_blocks: Sequence[int],
    cells_on_cell: np.ndarray,
    n_edges_on_cell: np.ndarray,
    cells_on_edge: np.ndarray,
    cells_on_vertex: np.ndarray,
    cell_xyz: np.ndarray,
    area_cell: np.ndarray,
    edges_on_edge: np.ndarray | None = None,
    n_edges_on_edge: np.ndarray | None = None,
) -> BlockPlan:
    devices = tuple(int(d) for d in devices)
    n_classes = len(plan.classes)
    if len(n_blocks) != n_classes:
        raise ValueError("one block count per class")
    if not devices or min(devices) < 0:
        raise ValueError("at least one non-negative card ordinal")
    cls = np.asarray(plan.cell_class, dtype=np.int32)
    n_cells = int(cls.size)
    coe = np.asarray(cells_on_edge, dtype=np.int64)
    cov = np.asarray(cells_on_vertex, dtype=np.int64)
    n_edges, n_vertices = int(coe.shape[0]), int(cov.shape[0])
    offsets, flat = _csr(cells_on_cell, n_edges_on_cell)
    neighbour_of = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(offsets))
    area = np.asarray(area_cell, dtype=np.float64)
    eoe_active = eoe_safe = None
    if edges_on_edge is not None:
        eoe = np.asarray(edges_on_edge)
        counts = np.asarray(n_edges_on_edge)
        eoe_active = (np.arange(eoe.shape[1])[None, :] < counts[:, None]) & (eoe >= 0)
        eoe_safe = np.where(eoe >= 0, eoe, 0).astype(np.int64)
    edge_class = np.maximum(cls[coe[:, 0]], cls[coe[:, 1]])
    vertex_class = cls[cov].max(axis=1)

    # global mailbox ids and zones, recovered from the one-device class tables
    cell_mail = np.full(n_cells, -1, dtype=np.int64)
    edge_mail = np.full(n_edges, -1, dtype=np.int64)
    zones = []
    for t in plan.classes:
        lay = t.layout
        halo_c = lay.cell_l2g[lay.n_owned_cells:].astype(np.int64)
        halo_e = lay.edge_l2g[lay.n_owned_edges:].astype(np.int64)
        cell_mail[halo_c] = t.halo_cell_mail
        edge_mail[halo_e] = t.halo_edge_mail
        zone = np.zeros(n_cells, dtype=bool)
        zone[halo_c[(np.asarray(t.halo_cell_class) & SCRATCH_FLAG) != 0]] = True
        zones.append(zone)
    register = np.asarray(plan.register_edge_global, dtype=np.int64)
    n_slots = max(int(register.size), 1)
    slot_of_edge = np.full(n_edges, -1, dtype=np.int64)
    slot_of_edge[register] = np.arange(register.size)
    reg_fine = np.maximum(cls[coe[register, 0]], cls[coe[register, 1]])
    reg_coarse = np.minimum(cls[coe[register, 0]], cls[coe[register, 1]])
    reg_coarse_cell = np.where(cls[coe[register, 0]] == reg_coarse, coe[register, 0], coe[register, 1])

    # partitions of every class's computed set
    parts: list[np.ndarray] = []
    for c in range(n_classes):
        part = np.full(n_cells, -1, dtype=np.int32)
        owned = np.flatnonzero(cls == c)
        if n_blocks[c] < 1 or n_blocks[c] > len(devices):
            raise ValueError(f"class {c}: {n_blocks[c]} blocks on {len(devices)} cards")
        part[owned] = rcb_parts(np.asarray(cell_xyz)[owned], int(n_blocks[c]))
        _assign_zone(part, zones[c], offsets, flat)
        parts.append(part)

    # exchange owner of every edge a class computes: owned edges and carried zone edges
    edge_xpart: list[np.ndarray] = []
    edge_carried: list[np.ndarray] = []
    for c in range(n_classes):
        part, zone = parts[c], zones[c]
        xp = np.full(n_edges, -1, dtype=np.int32)
        own = edge_class == c
        interface = own & (cls[coe[:, 0]] != cls[coe[:, 1]])
        inner = own & ~interface
        low = coe.min(axis=1)
        xp[inner] = part[low[inner]]
        coarse_end = np.where(cls[coe[:, 0]] < cls[coe[:, 1]], coe[:, 0], coe[:, 1])
        if np.any(~zone[coarse_end[interface]]):
            raise AssertionError(f"class {c}: an interface edge's coarse cell is outside the class's scratch zone")
        xp[interface] = part[coarse_end[interface]]
        carried = (edge_class < c) & zone[coe].any(axis=1)
        zone_low = np.where(zone[coe[:, 0]] & (~zone[coe[:, 1]] | (coe[:, 0] < coe[:, 1])), coe[:, 0], coe[:, 1])
        xp[carried] = part[zone_low[carried]]
        if np.any(xp[own | carried] < 0):
            raise AssertionError(f"class {c}: an edge it computes has no block")
        edge_xpart.append(xp)
        edge_carried.append(carried)
    vertex_part: list[np.ndarray] = []
    for c in range(n_classes):
        masked = np.where(cls[cov] == c, cov, n_cells)
        lowest = masked.min(axis=1)
        vp = np.full(n_vertices, -1, dtype=np.int32)
        sel = vertex_class == c
        vp[sel] = parts[c][lowest[sel]]
        vertex_part.append(vp)

    # rank of every block: a cut class takes ranks cursor..cursor+n-1, a one-block class the next
    # rank, so the blocks of one class always sit on distinct ranks
    block_rank: list[list[int]] = []
    cursor = 0
    for c in range(n_classes):
        row = [(cursor + j) % len(devices) for j in range(int(n_blocks[c]))]
        cursor = (cursor + int(n_blocks[c])) % len(devices)
        block_rank.append(row)

    layouts: list[PartitionLayout] = []
    infos: list[dict[str, Any]] = []
    gid = 0
    by_class: list[tuple[int, ...]] = []
    for c in range(n_classes):
        part, zone = parts[c], zones[c]
        in_set = (cls == c) | zone
        members = []
        for j in range(int(n_blocks[c])):
            owned = np.flatnonzero((cls == c) & (part == j)).astype(np.int32)
            carried = np.flatnonzero(zone & (part == j)).astype(np.int32)
            if owned.size == 0:
                raise ValueError(f"class {c} block {j} owns no cells; use fewer blocks for this class")
            l2g, ring = _cell_rings(owned, offsets, flat, n_cells=n_cells, halo_rings=HALO_RINGS,
                                    replicated_cells=carried if carried.size else None)
            head = owned.size + carried.size
            tail, tail_ring = l2g[head:], ring[head:]
            peer = in_set[tail]
            l2g = np.concatenate((l2g[:head], tail[peer], tail[~peer])).astype(np.int32)
            ring = np.concatenate((ring[:head], tail_ring[peer], tail_ring[~peer])).astype(np.int16)
            n_peer_cells = int(peer.sum())
            ring_global = np.full(n_cells, _UNREACHED, dtype=np.int32)
            ring_global[l2g.astype(np.int64)] = ring
            local = ring_global < _UNREACHED
            incomplete = np.bincount(neighbour_of[~local[flat.astype(np.int64)]], minlength=n_cells)
            complete = incomplete == 0
            # edges
            edge_ring = ring_global[coe].min(axis=1)
            elocal = edge_ring < _UNREACHED
            xp = edge_xpart[c]
            e_owned = np.flatnonzero((edge_class == c) & (xp == j))
            e_carried = np.flatnonzero(edge_carried[c] & (xp == j))
            if np.any(~elocal[e_owned]) or np.any(~elocal[e_carried]):
                raise AssertionError(f"class {c} block {j}: an edge it computes is not local")
            rest_peer = np.flatnonzero(elocal & (xp >= 0) & (xp != j))
            rest_mail = np.flatnonzero(elocal & (xp < 0))
            rest_peer = rest_peer[np.lexsort((rest_peer, edge_ring[rest_peer]))]
            rest_mail = rest_mail[np.lexsort((rest_mail, edge_ring[rest_mail]))]
            e_l2g = np.concatenate((e_owned, e_carried, rest_peer, rest_mail)).astype(np.int32)
            edge_complete = local[coe].all(axis=1) & complete[coe].all(axis=1)
            if eoe_active is not None:
                edge_local = np.zeros(n_edges, dtype=bool)
                edge_local[e_l2g.astype(np.int64)] = True
                edge_complete &= np.where(eoe_active, edge_local[eoe_safe], True).all(axis=1)
            # vertices
            vertex_ring = ring_global[cov].min(axis=1)
            vlocal = vertex_ring < _UNREACHED
            v_owned = np.flatnonzero((vertex_class == c) & (vertex_part[c] == j))
            if np.any(~vlocal[v_owned]):
                raise AssertionError(f"class {c} block {j}: an owned vertex is not local")
            v_rest = np.flatnonzero(vlocal & ~((vertex_class == c) & (vertex_part[c] == j)))
            v_rest = v_rest[np.lexsort((v_rest, vertex_ring[v_rest]))]
            v_l2g = np.concatenate((v_owned, v_rest)).astype(np.int32)
            vertex_complete = local[cov].all(axis=1)
            layout = PartitionLayout(
                partition=gid, n_owned_cells=int(owned.size), n_owned_edges=int(e_owned.size),
                n_owned_vertices=int(v_owned.size), cell_l2g=l2g, edge_l2g=e_l2g, vertex_l2g=v_l2g,
                cell_ring=ring,
                cell_row_complete=np.ascontiguousarray(complete[l2g.astype(np.int64)] & local[l2g.astype(np.int64)]),
                edge_row_complete=np.ascontiguousarray(edge_complete[e_l2g.astype(np.int64)]),
                vertex_row_complete=np.ascontiguousarray(vertex_complete[v_l2g.astype(np.int64)]))
            object.__setattr__(layout, "_global_cells", n_cells)
            object.__setattr__(layout, "_global_edges", n_edges)
            object.__setattr__(layout, "_global_vertices", n_vertices)
            object.__setattr__(layout, "_replicated_cells", tuple(int(v) for v in carried))
            layouts.append(layout)
            infos.append({"class": c, "block": j, "gid": gid, "rank": block_rank[c][j],
                          "device": devices[block_rank[c][j]],
                          "n_carried": int(carried.size), "n_peer_cells": n_peer_cells,
                          "n_carried_edges": int(e_carried.size), "n_peer_edges": int(rest_peer.size),
                          "complete": complete})
            members.append(gid)
            gid += 1
        by_class.append(tuple(members))
    assert_exact_cover(layouts, n_cells=n_cells, n_edges=n_edges, n_vertices=n_vertices)

    # per-block tables
    blocks: list[BlockTables] = []
    interface_cells = [np.zeros(n_cells, dtype=bool) for _ in range(n_classes)]   # coarse side of (c, c+1)
    for s in range(register.size):
        interface_cells[int(reg_coarse[s])][reg_coarse_cell[s]] = True

    def rows_of(cells_g: np.ndarray, slot_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Register rows of coarse cells: their slots ascending, coefficient -sign/area."""

        chosen = np.flatnonzero(slot_mask)
        row_index = {int(cell): r for r, cell in enumerate(cells_g)}
        counts = np.zeros(cells_g.size, dtype=np.int64)
        for s in chosen:
            r = row_index.get(int(reg_coarse_cell[s]))
            if r is not None:
                counts[r] += 1
        width = int(counts.max()) if cells_g.size and counts.size else 1
        width = max(width, 1)
        slots = np.zeros((cells_g.size, width), dtype=np.int32)
        coef = np.zeros((cells_g.size, width), dtype=np.float64)
        fill = np.zeros(cells_g.size, dtype=np.int64)
        for s in chosen:          # ascending slot order
            r = row_index.get(int(reg_coarse_cell[s]))
            if r is None:
                continue
            cell = int(reg_coarse_cell[s])
            sign = 1.0 if coe[register[s], 0] == cell else -1.0
            slots[r, fill[r]] = s
            coef[r, fill[r]] = -sign / area[cell]
            fill[r] += 1
        return slots, coef

    for c in range(n_classes):
        for j, g in enumerate(by_class[c]):
            layout, info = layouts[g], infos[g]
            c_l2g = layout.cell_l2g.astype(np.int64)
            e_l2g = layout.edge_l2g.astype(np.int64)
            c_g2l = layout.cell_g2l().astype(np.int64)
            e_g2l = layout.edge_g2l().astype(np.int64)
            n_own_c, n_own_e = int(layout.n_owned_cells), int(layout.n_owned_edges)
            n_xown_c = n_own_c + info["n_carried"]
            n_xloc_c = n_xown_c + info["n_peer_cells"]
            n_xown_e = n_own_e + info["n_carried_edges"]
            n_xloc_e = n_xown_e + info["n_peer_edges"]
            zone = zones[c]
            halo_c = c_l2g[n_own_c:]
            halo_e = e_l2g[n_own_e:]
            cflag = np.where(zone[halo_c], SCRATCH_FLAG, 0)
            ecarried = edge_carried[c][halo_e]
            eflag = np.where(ecarried, SCRATCH_FLAG, 0)
            halo_cell_class = (cls[halo_c] | cflag).astype(np.int32)
            halo_edge_class = (edge_class[halo_e] | eflag).astype(np.int32)
            hm = cell_mail[halo_c]
            em = edge_mail[halo_e]
            # every halo entity another class owns is in the one-device class halo, hence in the mailbox
            needs_c = (cls[halo_c] != c)
            needs_e = (edge_class[halo_e] != c)
            if np.any(hm[needs_c] < 0) or np.any(em[needs_e] < 0):
                raise AssertionError(f"class {c} block {j}: a halo entity of another class has no mailbox slot")
            hm = np.where(hm < 0, 0, hm).astype(np.int32)
            em = np.where(em < 0, 0, em).astype(np.int32)
            owned_c = c_l2g[:n_own_c]
            owned_e = e_l2g[:n_own_e]
            pub_c = np.flatnonzero(cell_mail[owned_c] >= 0)
            pub_e = np.flatnonzero(edge_mail[owned_e] >= 0)
            complete = info["complete"]
            both_local = (c_g2l[coe[:, 0]] >= 0) & (c_g2l[coe[:, 1]] >= 0)
            # fine side: owned interface edges with c-1
            fine_slots = np.flatnonzero((reg_fine == c) & (edge_xpart[c][register] == j)).astype(np.int32)
            fine_e = register[fine_slots]
            # every local (c, c-1) interface edge with both cells local
            up_slots = np.flatnonzero((reg_fine == c) & both_local[register] & (e_g2l[register] >= 0))
            up_e = register[up_slots]
            # coarse side: edges whose coarse cell is owned here
            coarse_slots = np.flatnonzero((reg_coarse == c) & (c_g2l[reg_coarse_cell] >= 0)
                                          & (c_g2l[reg_coarse_cell] < n_own_c)).astype(np.int32)
            coarse_e = register[coarse_slots]
            reflux_cells_g = np.unique(reg_coarse_cell[coarse_slots]).astype(np.int64)
            reflux_slot, reflux_coef = rows_of(reflux_cells_g, (reg_coarse == c) & np.isin(reg_coarse_cell, reflux_cells_g))
            refluxed_class_cells = interface_cells[c]
            reflux_edge = np.flatnonzero(refluxed_class_cells[coe[owned_e]].any(axis=1))
            packet_slots = np.flatnonzero((reg_coarse == c) & both_local[register] & (e_g2l[register] >= 0)).astype(np.int32)
            packet_e = register[packet_slots]
            # guard rows: owned (== reflux rows, same order) then local halo rows with complete stencils
            halo_rows = np.flatnonzero(refluxed_class_cells[c_l2g] & complete[c_l2g])
            halo_rows = halo_rows[halo_rows >= n_own_c]
            guard_rows = np.concatenate((c_g2l[reflux_cells_g], halo_rows)).astype(np.int32)
            # cap rows: coarse cells (class c-1) of the (c-1, c) interface
            if c > 0:
                cap_mask = interface_cells[c - 1]
                cap_owned_g = np.flatnonzero(cap_mask & (parts[c] == j) & zone)
                cap_halo = np.flatnonzero(cap_mask[c_l2g] & complete[c_l2g])
                cap_halo_g = c_l2g[cap_halo]
                cap_halo_g = cap_halo_g[~np.isin(cap_halo_g, cap_owned_g)]
                cap_g = np.concatenate((cap_owned_g, np.sort(cap_halo_g))).astype(np.int64)
                if np.any(c_g2l[cap_g] < 0):
                    raise AssertionError(f"class {c} block {j}: a reserve row is not local")
                cap_slot, cap_coef = rows_of(cap_g, reg_coarse == c - 1)
                n_cap_owned = int(cap_owned_g.size)
                eos = np.full(n_slots, -1, dtype=np.int32)
                eos[up_slots] = e_g2l[up_e]
                # every slot of an owned row is an edge this block owns
                owned_rows_slots = cap_slot[:n_cap_owned][cap_coef[:n_cap_owned] != 0.0]
                if owned_rows_slots.size and np.any(e_g2l[register[owned_rows_slots]] >= n_own_e):
                    raise AssertionError(f"class {c} block {j}: an interface edge of a reserve row is owned elsewhere")
            else:
                cap_g = np.zeros(0, dtype=np.int64)
                cap_slot = np.zeros((0, 1), dtype=np.int32)
                cap_coef = np.zeros((0, 1), dtype=np.float64)
                n_cap_owned = 0
                eos = np.full(n_slots, -1, dtype=np.int32)
            blocks.append(BlockTables(
                index=c, rate=int(plan.rates[c]), block=j, gid=g, device=int(info["device"]),
                rank=int(info["rank"]), layout=layout,
                n_xown_cells=n_xown_c, n_xlocal_cells=n_xloc_c, n_xown_edges=n_xown_e, n_xlocal_edges=n_xloc_e,
                halo_cell_mail=hm, halo_cell_class=halo_cell_class, halo_edge_mail=em, halo_edge_class=halo_edge_class,
                pub_cell_local=pub_c.astype(np.int32), pub_cell_mail=cell_mail[owned_c[pub_c]].astype(np.int32),
                pub_edge_local=pub_e.astype(np.int32), pub_edge_mail=edge_mail[owned_e[pub_e]].astype(np.int32),
                fine_edge_local=e_g2l[fine_e].astype(np.int32), fine_slot=fine_slots,
                fine_cells=c_g2l[coe[fine_e]].astype(np.int32).reshape(-1, 2),
                up_edge_local=e_g2l[up_e].astype(np.int32),
                up_edge_cells=c_g2l[coe[up_e]].astype(np.int32).reshape(-1, 2),
                coarse_edge_local=e_g2l[coarse_e].astype(np.int32), coarse_slot=coarse_slots,
                coarse_cells=c_g2l[coe[coarse_e]].astype(np.int32).reshape(-1, 2),
                reflux_cell_local=c_g2l[reflux_cells_g].astype(np.int32),
                reflux_slot=reflux_slot, reflux_coef=reflux_coef,
                reflux_edge_local=reflux_edge.astype(np.int32),
                reflux_edge_cells=c_g2l[coe[owned_e[reflux_edge]]].astype(np.int32).reshape(-1, 2),
                packet_edge_local=e_g2l[packet_e].astype(np.int32), packet_slot=packet_slots,
                guard_cell_local=guard_rows, n_guard_owned=int(reflux_cells_g.size),
                cap_cell_local=c_g2l[cap_g].astype(np.int32), n_cap_owned=n_cap_owned,
                cap_slot=cap_slot, cap_coef=cap_coef,
                cap_area=area[cap_g[:n_cap_owned]].astype(np.float64),
                edge_of_slot=eos,
                child=c + 1 if c + 1 < n_classes else -1,
                finest_halo_distance=int(plan.classes[c].finest_halo_distance)))
            last = blocks[-1]
            if np.any(last.reflux_cell_local >= n_own_c):
                raise AssertionError("a refluxed cell is not owned by its block")
            if c + 1 < n_classes and np.any(~np.isin(coarse_slots, packet_slots)):
                raise AssertionError("an owned row's interface edge is not a local packet edge")

    # transport tables, per class with more than one block
    transport: list[tuple[TransportTables, ...] | None] = []
    for c in range(n_classes):
        members = by_class[c]
        if len(members) == 1:
            transport.append(None)
            continue
        part = parts[c]
        cell_owner = np.full(n_cells, -1, dtype=np.int32)
        sel = (cls == c) | zones[c]
        cell_owner[sel] = part[sel]
        edge_owner_j = edge_xpart[c]
        # owner-local index of every exchange-owned entity
        owner_local_c = np.full(n_cells, -1, dtype=np.int64)
        owner_local_e = np.full(n_edges, -1, dtype=np.int64)
        for j, g in enumerate(members):
            t = blocks[g]
            lay = t.layout
            owner_local_c[lay.cell_l2g[:t.n_xown_cells].astype(np.int64)] = np.arange(t.n_xown_cells)
            owner_local_e[lay.edge_l2g[:t.n_xown_edges].astype(np.int64)] = np.arange(t.n_xown_edges)
        rows = []
        for j, g in enumerate(members):
            t = blocks[g]
            lay = t.layout
            peers = []
            for k, h in enumerate(members):
                if k == j:
                    continue
                other = blocks[h]
                olay = other.layout
                o_peer_c = olay.cell_l2g[other.n_xown_cells:other.n_xlocal_cells].astype(np.int64)
                o_peer_e = olay.edge_l2g[other.n_xown_edges:other.n_xlocal_edges].astype(np.int64)
                send_c = owner_local_c[o_peer_c[cell_owner[o_peer_c] == j]]
                send_e = owner_local_e[o_peer_e[edge_owner_j[o_peer_e] == j]]
                m_peer_c = lay.cell_l2g[t.n_xown_cells:t.n_xlocal_cells].astype(np.int64)
                m_peer_e = lay.edge_l2g[t.n_xown_edges:t.n_xlocal_edges].astype(np.int64)
                recv_c = np.flatnonzero(cell_owner[m_peer_c] == k) + t.n_xown_cells
                recv_e = np.flatnonzero(edge_owner_j[m_peer_e] == k) + t.n_xown_edges
                peers.append(PeerTables(k, np.ascontiguousarray(send_c, dtype=np.int64),
                                        np.ascontiguousarray(recv_c, dtype=np.int64),
                                        np.ascontiguousarray(send_e, dtype=np.int64),
                                        np.ascontiguousarray(recv_e, dtype=np.int64)))
            rows.append(TransportTables(rank=j, n_ranks=len(members), n_owned_cells=t.n_xown_cells,
                                        n_local_cells=t.n_xlocal_cells, n_owned_edges=t.n_xown_edges,
                                        n_local_edges=t.n_xlocal_edges, peers=tuple(peers)))
        transport.append(tuple(rows))

    # mailbox routes
    need_c: dict[int, np.ndarray] = {}
    need_e: dict[int, np.ndarray] = {}
    for t in blocks:
        lay = t.layout
        halo_c = lay.cell_l2g[lay.n_owned_cells:].astype(np.int64)
        halo_e = lay.edge_l2g[lay.n_owned_edges:].astype(np.int64)
        mc = cell_mail[halo_c][(cls[halo_c] != t.index)]
        me = edge_mail[halo_e][(edge_class[halo_e] != t.index)]
        need_c[t.device] = np.union1d(need_c.get(t.device, np.zeros(0, np.int64)), mc)
        need_e[t.device] = np.union1d(need_e.get(t.device, np.zeros(0, np.int64)), me)
    route_cells: dict[tuple[int, int], np.ndarray] = {}
    route_edges: dict[tuple[int, int], np.ndarray] = {}
    for t in blocks:
        for d in devices:
            if d == t.device:
                continue
            rc = t.pub_cell_mail[np.isin(t.pub_cell_mail, need_c.get(d, np.zeros(0)))]
            re = t.pub_edge_mail[np.isin(t.pub_edge_mail, need_e.get(d, np.zeros(0)))]
            if rc.size:
                route_cells[(t.gid, d)] = np.ascontiguousarray(rc, dtype=np.int32)
            if re.size:
                route_edges[(t.gid, d)] = np.ascontiguousarray(re, dtype=np.int32)
    class_pub_cells = tuple(np.unique(np.concatenate([blocks[g].pub_cell_mail for g in by_class[c]])).astype(np.int32)
                            for c in range(n_classes))
    class_pub_edges = tuple(np.unique(np.concatenate([blocks[g].pub_edge_mail for g in by_class[c]])).astype(np.int32)
                            for c in range(n_classes))

    # register pulls before a class steps (coarse side c reads what class c+1 wrote)
    writer_device = np.full(n_slots, -1, dtype=np.int64)
    for s in range(register.size):
        f = int(reg_fine[s])
        j = int(edge_xpart[f][register[s]])
        writer_device[s] = blocks[by_class[f][j]].device
    pulls: list[tuple[tuple[int, int, np.ndarray], ...]] = []
    zeros: list[tuple[tuple[int, np.ndarray], ...]] = []
    for c in range(n_classes):
        rows_p = []
        consumed_on: dict[int, set[int]] = {}
        for g in by_class[c]:
            t = blocks[g]
            consumed_on.setdefault(t.device, set()).update(int(s) for s in t.packet_slot)
        for d, slots in sorted(consumed_on.items()):
            arr = np.array(sorted(slots), dtype=np.int64)
            if arr.size == 0:
                continue
            for w in sorted(set(int(v) for v in writer_device[arr])):
                if w == d:
                    continue
                sel = arr[writer_device[arr] == w]
                rows_p.append((w, d, np.ascontiguousarray(sel, dtype=np.int32)))
        pulls.append(tuple(rows_p))
        rows_z = []
        all_slots = np.flatnonzero(reg_coarse == c)
        for w in sorted(set(int(v) for v in writer_device[all_slots])) if all_slots.size else []:
            mine = all_slots[writer_device[all_slots] == w]
            local_consumed = consumed_on.get(w, set())
            remote_only = np.array([s for s in mine if int(s) not in local_consumed], dtype=np.int32)
            if remote_only.size:
                rows_z.append((w, remote_only))
        zeros.append(tuple(rows_z))
    return BlockPlan(plan=plan, devices=devices, blocks=tuple(blocks), by_class=tuple(by_class),
                     transport=tuple(transport), route_cells=route_cells, route_edges=route_edges,
                     class_pub_cells=class_pub_cells, class_pub_edges=class_pub_edges,
                     register_pulls=tuple(pulls), register_zero_after_pull=tuple(zeros))
