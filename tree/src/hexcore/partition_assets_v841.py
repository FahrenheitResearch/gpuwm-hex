"""Partition assets for the v8.4.1 CUDA lane (brief-1 foundation).

This module turns an official METIS ``.part.P`` file plus the mesh adjacency
into per-partition *layouts*: the ordered local->global index vectors for
cells, edges and vertices, the halo ring index of every local cell, and the
"adjacency row is complete" masks that tell the driver validator which rows
may legally reference a clamped neighbour.

Every ordering law in here is bitwise-critical; see brief-1 "Ownership and
ordering laws".  Nothing in this module performs floating point arithmetic.

Ownership laws
--------------
1. cell owner   = the part file, verbatim.
2. edge owner   = ``part[min(cellsOnEdge[e, 0], cellsOnEdge[e, 1])]``.
3. vertex owner = ``part[min(cellsOnVertex[v])]``.

Local ordering law
------------------
``[owned ascending global id] ++ [ring 1 ascending] ++ ... ++ [ring K ascending]``
for cells; for edges and vertices ``[owned ascending] ++ [non-owned ascending
by (ring, global id)]``.

Row-completeness law
--------------------
A local entity's adjacency row is *complete* when every **active** slot of
every adjacency array of that entity references an entity that is present in
the local set.  Padding slots (negative sentinels, MPAS "-1 padding") are not
references and never make a row incomplete.  The predicates below are exact
for a closed MPAS mesh and are re-verified against the real arrays by
:mod:`hexcore.partition_local_mesh_v841`:

* cell   -- complete iff every ``cellsOnCell`` neighbour is local.  (The
  ``edgesOnCell``/``verticesOnCell`` rows of a local cell are always local:
  an edge is local when either endpoint cell is local, and a vertex is local
  when any of its ``cellsOnVertex`` is local.)
* edge   -- complete iff both endpoint cells are local *and* both endpoint
  cells have complete ``cellsOnCell`` rows.  The second clause is what
  ``advCellsForEdge`` (the transport two-ring stencil: the endpoint cells plus
  their neighbours) and ``edgesOnEdge`` (the union of both endpoint cells'
  edges) require.
* vertex -- complete iff all three ``cellsOnVertex`` cells are local.  The
  three ``edgesOnVertex`` edges then each join two local cells.

Note (brief ambiguity, simplest-correct choice): the brief says a referenced
id absent from the local set is clamped to local index 0.  Padding entries are
*not* references, so a negative padding sentinel is copied verbatim rather
than clamped; this keeps inactive slots byte-identical to the resident mesh.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BUILDER_VERSION = "partition-assets-v841/1"

# Ring sentinel for "not in the local set".  Larger than any legal ring depth.
_UNREACHED = np.int32(1 << 30)


@dataclass(frozen=True)
class PartitionLayout:
    """Local<->global index law for one partition."""

    partition: int
    n_owned_cells: int
    n_owned_edges: int
    n_owned_vertices: int
    cell_l2g: np.ndarray  # int32 (n_local_cells,)
    edge_l2g: np.ndarray  # int32 (n_local_edges,)
    vertex_l2g: np.ndarray  # int32 (n_local_vertices,)
    cell_ring: np.ndarray  # int16 (n_local_cells,), 0 == owned
    cell_row_complete: np.ndarray  # bool (n_local_cells,)
    edge_row_complete: np.ndarray  # bool (n_local_edges,)
    vertex_row_complete: np.ndarray  # bool (n_local_vertices,)

    @property
    def n_local_cells(self) -> int:
        return int(self.cell_l2g.size)

    @property
    def n_local_edges(self) -> int:
        return int(self.edge_l2g.size)

    @property
    def n_local_vertices(self) -> int:
        return int(self.vertex_l2g.size)

    def cell_g2l(self) -> np.ndarray:
        """Dense global->local cell map; ``-1`` where the cell is not local."""

        return _dense_inverse(self.cell_l2g, self._global_cells)

    def edge_g2l(self) -> np.ndarray:
        return _dense_inverse(self.edge_l2g, self._global_edges)

    def vertex_g2l(self) -> np.ndarray:
        return _dense_inverse(self.vertex_l2g, self._global_vertices)

    # Global extents are needed to size the dense inverses.  They are stored
    # as plain object attributes by the builder/loader (the dataclass is
    # frozen, so object.__setattr__ is used).
    _global_cells: int = 0
    _global_edges: int = 0
    _global_vertices: int = 0

    def receipt(self) -> dict[str, Any]:
        return {
            "partition": int(self.partition),
            "n_owned_cells": int(self.n_owned_cells),
            "n_local_cells": self.n_local_cells,
            "n_owned_edges": int(self.n_owned_edges),
            "n_local_edges": self.n_local_edges,
            "n_owned_vertices": int(self.n_owned_vertices),
            "n_local_vertices": self.n_local_vertices,
            "max_ring": int(self.cell_ring.max()) if self.cell_ring.size else 0,
            "ring_sizes": [
                int(np.count_nonzero(self.cell_ring == ring))
                for ring in range(int(self.cell_ring.max()) + 1)
            ]
            if self.cell_ring.size
            else [],
            "cell_rows_complete": int(np.count_nonzero(self.cell_row_complete)),
            "edge_rows_complete": int(np.count_nonzero(self.edge_row_complete)),
            "vertex_rows_complete": int(np.count_nonzero(self.vertex_row_complete)),
        }


def _dense_inverse(l2g: np.ndarray, n_global: int) -> np.ndarray:
    if n_global <= 0:
        n_global = int(l2g.max()) + 1 if l2g.size else 0
    inverse = np.full(int(n_global), -1, dtype=np.int32)
    inverse[l2g.astype(np.int64, copy=False)] = np.arange(l2g.size, dtype=np.int32)
    return inverse


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def read_graph_info(path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse a METIS ``graph.info`` into CSR adjacency.

    Returns ``(offsets, flat, n_graph_edges)`` with ``offsets`` of shape
    ``(nCells + 1,)`` int64 and ``flat`` int32 zero-based neighbour ids.
    Line ``i`` (1-based, after the header) lists the 1-based ``cellsOnCell``
    of cell ``i - 1``.
    """

    path = Path(path)
    with open(path, "r") as handle:
        header = handle.readline().split()
        if len(header) < 2:
            raise ValueError(f"{path}: graph.info header must be 'nCells nEdges'")
        n_cells = int(header[0])
        n_graph_edges = int(header[1])
        counts = np.zeros(n_cells, dtype=np.int64)
        tokens: list[str] = []
        for cell in range(n_cells):
            line = handle.readline()
            if line == "":
                raise ValueError(f"{path}: truncated at cell {cell}")
            row = line.split()
            counts[cell] = len(row)
            tokens.extend(row)
        remainder = handle.readline()
        while remainder != "":
            if remainder.strip():
                raise ValueError(f"{path}: trailing content after {n_cells} cells")
            remainder = handle.readline()
    flat64 = np.array(tokens, dtype=np.int64) if tokens else np.empty(0, np.int64)
    if flat64.size and (flat64.min() < 1 or flat64.max() > n_cells):
        raise ValueError(f"{path}: neighbour id outside [1,{n_cells}]")
    offsets = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    flat = (flat64 - 1).astype(np.int32, copy=False)
    return offsets, flat, n_graph_edges


def read_part_file(path: str | Path, *, n_cells: int | None = None) -> np.ndarray:
    """Parse a METIS ``.part.P`` file into an int32 partition id per cell."""

    part = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if part.ndim != 1:
        raise ValueError(f"{path}: part file must be one column")
    if n_cells is not None and part.size != n_cells:
        raise ValueError(f"{path}: {part.size} lines != nCells {n_cells}")
    if part.size == 0 or part.min() < 0:
        raise ValueError(f"{path}: partition ids must be non-negative")
    return part.astype(np.int32, copy=False)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _expand_frontier(
    frontier: np.ndarray, offsets: np.ndarray, flat: np.ndarray
) -> np.ndarray:
    """Vectorised CSR neighbour gather for a frontier of cells."""

    if frontier.size == 0:
        return np.empty(0, dtype=np.int32)
    starts = offsets[frontier.astype(np.int64, copy=False)]
    counts = offsets[frontier.astype(np.int64, copy=False) + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int32)
    row_start = np.repeat(starts, counts)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate(([0], np.cumsum(counts)[:-1])), counts
    )
    return flat[row_start + within]


def _cell_rings(
    owned: np.ndarray,
    offsets: np.ndarray,
    flat: np.ndarray,
    *,
    n_cells: int,
    halo_rings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cell_l2g, cell_ring)`` for one partition.

    Vectorised BFS: the frontier is expanded through the CSR rows in one shot
    and deduped against a global boolean visited mask.
    """

    visited = np.zeros(n_cells, dtype=bool)
    visited[owned] = True
    order: list[np.ndarray] = [owned.astype(np.int32, copy=False)]
    rings: list[np.ndarray] = [np.zeros(owned.size, dtype=np.int16)]
    frontier = owned.astype(np.int32, copy=False)
    for ring in range(1, int(halo_rings) + 1):
        candidates = _expand_frontier(frontier, offsets, flat)
        if candidates.size == 0:
            break
        fresh = candidates[~visited[candidates.astype(np.int64, copy=False)]]
        if fresh.size == 0:
            break
        fresh = np.unique(fresh).astype(np.int32, copy=False)
        visited[fresh.astype(np.int64, copy=False)] = True
        order.append(fresh)
        rings.append(np.full(fresh.size, ring, dtype=np.int16))
        frontier = fresh
    return np.concatenate(order).astype(np.int32, copy=False), np.concatenate(
        rings
    ).astype(np.int16, copy=False)


def _order_secondary(
    owner: np.ndarray,
    ring: np.ndarray,
    partition: int,
) -> tuple[np.ndarray, int]:
    """Order an edge/vertex class: owned ascending, then (ring, gid) ascending.

    ``ring`` carries ``_UNREACHED`` for entities outside the local set.
    """

    is_owned = owner == partition
    in_set = ring < _UNREACHED
    if not np.all(in_set[is_owned]):
        raise AssertionError("an owned entity fell outside the local halo set")
    owned = np.flatnonzero(is_owned).astype(np.int32, copy=False)
    other = np.flatnonzero(in_set & ~is_owned).astype(np.int32, copy=False)
    if other.size:
        # lexsort: last key is primary -> sort by ring, then global id.
        other = other[np.lexsort((other.astype(np.int64), ring[other]))]
    return np.concatenate((owned, other)).astype(np.int32, copy=False), int(owned.size)


def build_partition_layouts(
    graph_info: Path,
    part_file: Path,
    *,
    halo_rings: int,
    cells_on_vertex: np.ndarray,
    cells_on_edge: np.ndarray,
    partitions: Iterable[int] | None = None,
) -> list[PartitionLayout]:
    """Build every partition's layout from an official part file."""

    if int(halo_rings) < 0:
        raise ValueError("halo_rings must be non-negative")
    offsets, flat, _ = read_graph_info(graph_info)
    n_cells = int(offsets.size - 1)
    part = read_part_file(part_file, n_cells=n_cells)
    n_parts = int(part.max()) + 1

    coe = np.asarray(cells_on_edge)
    cov = np.asarray(cells_on_vertex)
    if coe.ndim != 2 or coe.shape[1] != 2:
        raise ValueError("cells_on_edge must have shape (nEdges,2)")
    if cov.ndim != 2:
        raise ValueError("cells_on_vertex must have shape (nVertices,vertexDegree)")
    if coe.min() < 0 or coe.max() >= n_cells:
        raise ValueError("cells_on_edge references a cell outside the graph")
    if cov.min() < 0 or cov.max() >= n_cells:
        raise ValueError("cells_on_vertex references a cell outside the graph")
    n_edges = int(coe.shape[0])
    n_vertices = int(cov.shape[0])

    # Law 2 / law 3: owner of the smallest-global-index incident cell.
    edge_min_cell = np.minimum(coe[:, 0], coe[:, 1]).astype(np.int64, copy=False)
    edge_owner = part[edge_min_cell]
    vertex_min_cell = cov.min(axis=1).astype(np.int64, copy=False)
    vertex_owner = part[vertex_min_cell]

    # Row-completeness needs, per global cell, "are all my graph neighbours
    # inside this partition's local set".  Computed per partition below.
    neighbour_of = np.repeat(
        np.arange(n_cells, dtype=np.int64), np.diff(offsets)
    )  # owner cell of every CSR slot

    wanted = sorted(set(range(n_parts) if partitions is None else partitions))
    layouts: list[PartitionLayout] = []
    for partition in wanted:
        owned_cells = np.flatnonzero(part == partition).astype(np.int32, copy=False)
        if owned_cells.size == 0:
            raise ValueError(f"partition {partition} owns no cells")
        cell_l2g, cell_ring = _cell_rings(
            owned_cells, offsets, flat, n_cells=n_cells, halo_rings=halo_rings
        )

        cell_ring_global = np.full(n_cells, _UNREACHED, dtype=np.int32)
        cell_ring_global[cell_l2g.astype(np.int64, copy=False)] = cell_ring
        cell_local = cell_ring_global < _UNREACHED

        # cellsOnCell completeness per GLOBAL cell (True also for non-local
        # cells that happen to have all neighbours local; only local rows are
        # read out below).
        neighbour_local = cell_local[flat.astype(np.int64, copy=False)]
        incomplete_count = np.bincount(
            neighbour_of[~neighbour_local], minlength=n_cells
        )
        cell_complete_global = incomplete_count == 0
        cell_row_complete = cell_complete_global[cell_l2g.astype(np.int64, copy=False)] & cell_local[
            cell_l2g.astype(np.int64, copy=False)
        ]

        edge_ring = np.minimum(
            cell_ring_global[coe[:, 0].astype(np.int64, copy=False)],
            cell_ring_global[coe[:, 1].astype(np.int64, copy=False)],
        )
        edge_l2g, n_owned_edges = _order_secondary(edge_owner, edge_ring, partition)
        both_cells_local = cell_local[coe[:, 0].astype(np.int64, copy=False)] & cell_local[
            coe[:, 1].astype(np.int64, copy=False)
        ]
        both_cells_complete = cell_complete_global[
            coe[:, 0].astype(np.int64, copy=False)
        ] & cell_complete_global[coe[:, 1].astype(np.int64, copy=False)]
        edge_complete_global = both_cells_local & both_cells_complete
        edge_row_complete = edge_complete_global[edge_l2g.astype(np.int64, copy=False)]

        vertex_ring = cell_ring_global[cov.astype(np.int64, copy=False)].min(axis=1)
        vertex_l2g, n_owned_vertices = _order_secondary(
            vertex_owner, vertex_ring, partition
        )
        vertex_complete_global = cell_local[cov.astype(np.int64, copy=False)].all(axis=1)
        vertex_row_complete = vertex_complete_global[
            vertex_l2g.astype(np.int64, copy=False)
        ]

        layout = PartitionLayout(
            partition=int(partition),
            n_owned_cells=int(owned_cells.size),
            n_owned_edges=int(n_owned_edges),
            n_owned_vertices=int(n_owned_vertices),
            cell_l2g=cell_l2g,
            edge_l2g=edge_l2g,
            vertex_l2g=vertex_l2g,
            cell_ring=cell_ring,
            cell_row_complete=np.ascontiguousarray(cell_row_complete, dtype=bool),
            edge_row_complete=np.ascontiguousarray(edge_row_complete, dtype=bool),
            vertex_row_complete=np.ascontiguousarray(vertex_row_complete, dtype=bool),
        )
        object.__setattr__(layout, "_global_cells", n_cells)
        object.__setattr__(layout, "_global_edges", n_edges)
        object.__setattr__(layout, "_global_vertices", n_vertices)
        layouts.append(layout)

    if partitions is None:
        assert_exact_cover(layouts, n_cells=n_cells, n_edges=n_edges, n_vertices=n_vertices)
    return layouts


def assert_exact_cover(
    layouts: Sequence[PartitionLayout],
    *,
    n_cells: int,
    n_edges: int,
    n_vertices: int,
) -> dict[str, int]:
    """Law 6: owned locals are a disjoint exact cover of the global space."""

    checked: dict[str, int] = {}
    for name, extent, accessor in (
        ("cells", n_cells, lambda l: l.cell_l2g[: l.n_owned_cells]),
        ("edges", n_edges, lambda l: l.edge_l2g[: l.n_owned_edges]),
        ("vertices", n_vertices, lambda l: l.vertex_l2g[: l.n_owned_vertices]),
    ):
        hits = np.zeros(extent, dtype=np.int32)
        for layout in layouts:
            owned = accessor(layout).astype(np.int64, copy=False)
            if owned.size and (
                np.any(np.diff(owned) <= 0)
            ):
                raise AssertionError(f"{name}: owned ids are not strictly ascending")
            hits[owned] += 1
        if not np.all(hits == 1):
            missing = int(np.count_nonzero(hits == 0))
            doubled = int(np.count_nonzero(hits > 1))
            raise AssertionError(
                f"{name}: ownership is not an exact cover "
                f"({missing} unowned, {doubled} multiply owned of {extent})"
            )
        checked[name] = int(extent)
    return checked


# ---------------------------------------------------------------------------
# npz cache
# ---------------------------------------------------------------------------

_PER_LAYOUT_ARRAYS = (
    "cell_l2g",
    "edge_l2g",
    "vertex_l2g",
    "cell_ring",
    "cell_row_complete",
    "edge_row_complete",
    "vertex_row_complete",
)
_PER_LAYOUT_SCALARS = ("n_owned_cells", "n_owned_edges", "n_owned_vertices")


def save_layouts_npz(
    path: str | Path,
    layouts: Sequence[PartitionLayout],
    *,
    mesh_sha256: str,
    part_sha256: str,
    halo_rings: int,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "mesh_sha256": np.array(str(mesh_sha256)),
        "part_sha256": np.array(str(part_sha256)),
        "halo_rings": np.array(int(halo_rings), dtype=np.int64),
        "builder_version": np.array(BUILDER_VERSION),
        "n_partitions": np.array(len(layouts), dtype=np.int64),
        "global_cells": np.array(layouts[0]._global_cells, dtype=np.int64),
        "global_edges": np.array(layouts[0]._global_edges, dtype=np.int64),
        "global_vertices": np.array(layouts[0]._global_vertices, dtype=np.int64),
    }
    for index, layout in enumerate(layouts):
        payload[f"p{index}.partition"] = np.array(layout.partition, dtype=np.int64)
        for name in _PER_LAYOUT_SCALARS:
            payload[f"p{index}.{name}"] = np.array(getattr(layout, name), dtype=np.int64)
        for name in _PER_LAYOUT_ARRAYS:
            payload[f"p{index}.{name}"] = getattr(layout, name)
    np.savez(path, **payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n")
    return path


def load_layouts_npz(
    path: str | Path,
    *,
    expect_mesh_sha256: str,
    expect_part_sha256: str,
    expect_halo_rings: int,
) -> list[PartitionLayout]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        header = {
            "mesh_sha256": str(data["mesh_sha256"]),
            "part_sha256": str(data["part_sha256"]),
            "halo_rings": int(data["halo_rings"]),
            "builder_version": str(data["builder_version"]),
        }
        for key, expected in (
            ("mesh_sha256", str(expect_mesh_sha256)),
            ("part_sha256", str(expect_part_sha256)),
            ("halo_rings", int(expect_halo_rings)),
            ("builder_version", BUILDER_VERSION),
        ):
            if header[key] != expected:
                raise ValueError(
                    f"{path}: cached {key}={header[key]!r} != expected {expected!r}"
                )
        n_partitions = int(data["n_partitions"])
        globals_ = (
            int(data["global_cells"]),
            int(data["global_edges"]),
            int(data["global_vertices"]),
        )
        layouts: list[PartitionLayout] = []
        for index in range(n_partitions):
            kwargs: dict[str, Any] = {
                "partition": int(data[f"p{index}.partition"]),
            }
            for name in _PER_LAYOUT_SCALARS:
                kwargs[name] = int(data[f"p{index}.{name}"])
            for name in _PER_LAYOUT_ARRAYS:
                kwargs[name] = np.ascontiguousarray(data[f"p{index}.{name}"])
            layout = PartitionLayout(**kwargs)
            object.__setattr__(layout, "_global_cells", globals_[0])
            object.__setattr__(layout, "_global_edges", globals_[1])
            object.__setattr__(layout, "_global_vertices", globals_[2])
            layouts.append(layout)
    return layouts


def build_or_load_layouts(
    cache_path: str | Path,
    *,
    graph_info: Path,
    part_file: Path,
    halo_rings: int,
    cells_on_vertex: np.ndarray,
    cells_on_edge: np.ndarray,
    mesh_sha256: str,
) -> list[PartitionLayout]:
    """Cache-first entry point; refuses on any header mismatch."""

    cache_path = Path(cache_path)
    part_sha = sha256_file(part_file)
    if cache_path.exists():
        return load_layouts_npz(
            cache_path,
            expect_mesh_sha256=mesh_sha256,
            expect_part_sha256=part_sha,
            expect_halo_rings=halo_rings,
        )
    layouts = build_partition_layouts(
        Path(graph_info),
        Path(part_file),
        halo_rings=halo_rings,
        cells_on_vertex=cells_on_vertex,
        cells_on_edge=cells_on_edge,
    )
    save_layouts_npz(
        cache_path,
        layouts,
        mesh_sha256=mesh_sha256,
        part_sha256=part_sha,
        halo_rings=halo_rings,
    )
    return layouts
