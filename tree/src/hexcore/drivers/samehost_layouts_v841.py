#!/usr/bin/env python3
"""Build contiguous METIS partitions and same-host layout assets.

Only integer connectivity is processed.  Mesh loading, connectivity
zero-basing, layout ownership and cache serialization use existing helpers.

Examples:
    python tools/build_samehost_layouts_v841.py --grid mesh.grid.nc --ranks 2 --output-dir assets2
    python tools/build_samehost_layouts_v841.py --grid mesh.grid.nc --ranks 4 --output-dir assets4
    python tools/build_samehost_layouts_v841.py --grid mesh.grid.nc --part mesh.graph.info.part.2 --ranks 2 --output-dir assets2
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from hexcore.file_digest import sha256_file as file_version_sha256
from hexcore.mesh import Mesh
from hexcore.partition_assets_v841 import (
    BUILDER_VERSION,
    assert_exact_cover,
    build_partition_layouts_from_csr,
    load_layouts_npz,
    read_part_file,
    save_layouts_npz,
    sha256_file,
)

SCHEMA = "mpas-port.samehost-layout-assets/v1"

#: The METIS 5.1.0 build this wheel carries in hexcore/libexec (tools/build_gpmetis.py).
BUNDLED_METIS = "hex_gpmetis"
#: What a host without a bundled partitioner runs to get one, named in the refusal.
METIS_INSTALL = ("pip install --force-reinstall --no-deps gpuwm-hex (the Linux wheel carries hex_gpmetis); "
                 "or install METIS's gpmetis: sudo apt-get install metis (Debian/Ubuntu), "
                 "sudo dnf install metis (RHEL/Rocky/Alma 8-9 with EPEL), conda install -c conda-forge metis")


def metis_program(gpmetis: str | None = None) -> str:
    """The gpmetis executable to run: an explicit one, else the bundled build, else PATH.

    THE BREAKAGE THIS PREVENTS (hex global, 2x RTX PRO 6000, 2026-10-16,
    RESULTS.md defect 4): the door looked on PATH only, a pip install has no
    gpmetis there, and every automatic two-card run refused until a JEDI
    spack view was put on PATH by hand.  The wheel's hex_gpmetis is METIS 5.1.0
    (idx_t and real_t 64-bit) and writes partition files byte-identical to a
    spack metis@5.1.0+int64+real64 gpmetis (third_party/metis-5.1.0/
    HEX-VENDORING.txt), so it is preferred over whichever gpmetis PATH holds.
    """

    if gpmetis is not None:
        found = shutil.which(gpmetis)
        if found is None:
            raise ValueError(f"the named gpmetis {gpmetis!r} is not an executable")
        return found
    from hexcore.engines import bundled_program

    found = bundled_program(BUNDLED_METIS) or shutil.which("gpmetis")
    if found is None:
        raise ValueError("no METIS partitioner: this install has no bundled hex_gpmetis and no gpmetis is on "
                         "PATH; automatic multi-card execution needs contiguous METIS partitions to preserve halo "
                         f"ownership and avoid missing or duplicated cells. Install one: {METIS_INSTALL}; "
                         "or provide an existing --part file (at the forecast door: --samehost-part and "
                         "--samehost-layouts)")
    return found


def _canonical_graph(mesh: Any) -> np.ndarray:
    """Read the undirected cell graph, excluding canonical missing slots."""

    n_cells = int(mesh.dimensions["nCells"])
    adjacency = np.asarray(mesh.arrays["cellsOnCell"])
    counts = np.asarray(mesh.arrays["nEdgesOnCell"])
    if adjacency.ndim != 2 or adjacency.shape[0] != n_cells or counts.shape != (n_cells,):
        raise ValueError("cell adjacency/count shapes do not match nCells")
    if adjacency.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
        raise ValueError("cell adjacency and counts must be integer metadata")
    from hexcore.host_parallel import run_blocks

    # Sort/check independent rows in NumPy, which releases the GIL. Avoid a
    # Python set and min/max dispatch for every cell of a multi-million-cell
    # mesh. n_cells is a private sentinel, never serialized into graph.info.
    canonical = np.empty(adjacency.shape, dtype=np.int64)
    slots = np.arange(adjacency.shape[1])
    failures = []

    def normalize(block):
        # Bound masks and neighbor gathers even on the one-worker path.
        for start in range(block.start, block.stop, 32768):
            stop = min(start + 32768, block.stop)
            values = adjacency[start:stop]
            count = counts[start:stop]
            cells = np.arange(start, stop)
            active = slots[None, :] < count[:, None]
            bad_count = (count < 0) | (count > adjacency.shape[1])
            bad_bounds = np.any(active & ((values < -1) | (values >= n_cells)), axis=1)
            bad_self = np.any(active & (values == cells[:, None]), axis=1)
            bad = np.flatnonzero(bad_count | bad_bounds | bad_self)
            if bad.size:
                offset = int(bad[0])
                reason = ("nEdgesOnCell is outside its adjacency row" if bad_count[offset]
                          else "adjacency is outside nCells or the -1 sentinel" if bad_bounds[offset]
                          else "a self edge cannot be passed to METIS")
                failures.append((start + offset, reason))
                continue
            out = canonical[start:stop]
            out.fill(n_cells)
            np.copyto(out, values, where=active & (values >= 0), casting="unsafe")
            out.sort(axis=1)
            # Keep the first occurrence of each neighbor, as sorted(set(row)).
            duplicate = out[:, 1:] == out[:, :-1]
            out[:, 1:][duplicate] = n_cells
            out.sort(axis=1)

    run_blocks(normalize, n_cells, elements=adjacency.size)
    if failures:
        cell, reason = min(failures)
        raise ValueError(f"cell {cell}: {reason}")

    missing = []

    def check_reciprocal(block):
        for start in range(block.start, block.stop, 32768):
            stop = min(start + 32768, block.stop)
            cells = np.arange(start, stop)
            for slot in range(adjacency.shape[1]):
                neighbors = canonical[start:stop, slot]
                valid = neighbors != n_cells
                source = cells[valid]
                target = neighbors[valid]
                found = np.any(canonical[target] == source[:, None], axis=1)
                bad = np.flatnonzero(~found)
                if bad.size:
                    offset = int(bad[0])
                    missing.append((int(source[offset]), int(target[offset])))

    run_blocks(check_reciprocal, n_cells, elements=adjacency.size)
    if missing:
        cell, neighbor = min(missing)
        raise ValueError(f"cell graph is not reciprocal at {cell}->{neighbor}")
    return canonical


def graph_rows(mesh: Any) -> tuple[tuple[int, ...], ...]:
    """Compatibility view; production retains packed integer connectivity."""
    canonical = _canonical_graph(mesh)
    n_cells = canonical.shape[0]
    return tuple(tuple(value for value in row if value != n_cells)
                 for row in canonical.tolist())


class PackedGraph:
    """Validated CSR plus parallel exact METIS text, without per-cell objects.

    Connectivity checks stay in _canonical_graph. SciPy's compiled integer
    traversal replaces the Python set walk; independent partition checks and
    decimal encoding run in host blocks. No graph text is parsed back into
    integers during construction. METIS itself keeps its pinned recipe.
    """

    def __init__(self, mesh: Any):
        from scipy.sparse import csr_matrix

        self.canonical = _canonical_graph(mesh)
        self.n_cells = self.canonical.shape[0]
        active = self.canonical != self.n_cells
        counts = np.count_nonzero(active, axis=1)
        self.offsets = np.empty(self.n_cells + 1, dtype=np.int64)
        self.offsets[0] = 0
        np.cumsum(counts, out=self.offsets[1:])
        self.flat = self.canonical[active].astype(np.int32)
        self.csr = csr_matrix((np.ones(self.flat.size, dtype=np.int8),
                               self.flat, self.offsets),
                              shape=(self.n_cells, self.n_cells))

    def components(self, members: np.ndarray | None = None) -> int:
        from scipy.sparse.csgraph import connected_components

        graph = self.csr if members is None else self.csr[members][:, members]
        return int(connected_components(graph, directed=False, return_labels=False))

    def partition_components(self, assignments: np.ndarray, ranks: int) -> list[int]:
        from concurrent.futures import ThreadPoolExecutor
        from hexcore.host_parallel import threads

        def check(rank):
            return self.components(np.flatnonzero(assignments == rank))

        if ranks == 1:
            return [1]  # The whole graph has already passed the connected check.
        with ThreadPoolExecutor(max_workers=min(threads(), ranks)) as pool:
            return list(pool.map(check, range(ranks)))

    def cut_edges(self, assignments: np.ndarray) -> int:
        from hexcore.host_parallel import run_blocks

        totals = []
        def count(block):
            start, stop = self.offsets[block.start], self.offsets[block.stop]
            owners = np.repeat(assignments[block], np.diff(self.offsets[block.start:block.stop + 1]))
            totals.append(int(np.count_nonzero(owners != assignments[self.flat[start:stop]])))
        run_blocks(count, self.n_cells, elements=self.flat.size)
        return sum(totals) // 2

    def metis_bytes(self) -> bytes:
        from hexcore.host_parallel import run_blocks

        chunks = []
        width = self.canonical.shape[1]
        digits = len(str(max(1, self.n_cells)))
        def encode(block):
            # Bounded buffers, including on the one-worker path. Integer
            # decimal extraction is exact and each ufunc releases the GIL.
            for start in range(block.start, block.stop, 8192):
                stop = min(start + 8192, block.stop)
                if width == 0:
                    chunks.append((start, b"\n" * (stop - start)))
                    continue
                values = self.canonical[start:stop] + 1
                present = values <= self.n_cells
                counts = np.count_nonzero(present, axis=1)
                data = np.empty((*values.shape, digits + 1), dtype=np.uint8)
                keep = np.zeros(data.shape, dtype=bool)
                for column in range(digits):
                    power = 10 ** (digits - column - 1)
                    data[:, :, column] = (values // power) % 10 + 48
                    keep[:, :, column] = present & (values >= power)
                data[:, :, -1] = 32
                keep[:, :, -1] = present
                last = np.maximum(counts - 1, 0)
                rows = np.arange(stop - start)
                data[rows, last, -1] = 10
                keep[rows, last, -1] = True
                chunks.append((start, data[keep].tobytes()))
        run_blocks(encode, self.n_cells, elements=self.canonical.size)
        return (f"{self.n_cells} {self.flat.size // 2}\n".encode("ascii")
                + b"".join(value for _, value in sorted(chunks)))


def component_count(rows: tuple[tuple[int, ...], ...], members: np.ndarray | None = None) -> int:
    """Count connected components using only integer graph membership."""

    remaining = set(range(len(rows))) if members is None else set(int(cell) for cell in members)
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            cell = frontier.pop()
            for neighbor in rows[cell]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return components


def _write_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"{path}: existing asset has different bytes; use a fresh output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def reuse_verified_receipt(
    grid: Path,
    *,
    ranks: int,
    output_dir: Path,
    part: Path | None = None,
    halo_rings: int = 3,
) -> dict[str, Any] | None:
    """The receipt an earlier :func:`build_assets` wrote for these exact bytes, or None.

    ``build_assets`` writes its receipt only after every check has passed: a
    reciprocal cell graph, contiguous partitions covering exactly ranks
    0..N-1, an exact owned cover, and layouts keyed by the mesh and part
    digests.  A warm run of the same mesh recomputed all of it in pure Python
    over every cell -- 21 s of the 20 km door on a rented box (g20-C, 2026-10-13,
    1,472,532 cells, 4 ranks) with every card idle -- to rewrite the same
    receipt.  The receipt is reused only when the grid, graph, part and
    layouts files it names still carry the SHA-256 it recorded (the grid's
    from :mod:`hexcore.file_digest`, which is the file's real digest, reused
    only for the same file version), at the same paths, under the same
    builder version and schema.  Any difference returns None and the caller
    rebuilds and re-verifies everything, exactly as before.
    """

    grid = Path(grid).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    tag = grid.stem.removesuffix(".grid")
    receipt_path = output_dir / f"{tag}.part{ranks}.K{halo_rings}.json"
    graph = output_dir / f"{tag}.graph.info"
    target_part = Path(f"{graph}.part.{ranks}")
    part_source = Path(part).resolve() if part is not None else target_part
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict):
        return None
    expected = {
        "schema": SCHEMA, "builder_version": BUILDER_VERSION,
        "grid": str(grid), "graph": str(graph), "part": str(target_part),
        "part_source": str(part_source),
        "layouts": str(output_dir / f"{tag}.part{ranks}.K{halo_rings}.npz"),
        "ranks": ranks, "halo_rings": halo_rings, "receipt": str(receipt_path),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        return None
    try:
        recorded = (
            (grid, receipt["grid_sha256"]),
            (graph, receipt["graph_sha256"]),
            (target_part, receipt["part_sha256"]),
            (part_source, receipt["part_sha256"]),
            (Path(receipt["layouts"]), receipt["layouts_sha256"]),
        )
        for path, digest in recorded:
            if not isinstance(digest, str) or file_version_sha256(path) != digest:
                return None
    except (OSError, KeyError, TypeError):
        return None
    return receipt


def _load_mesh_and_digest(grid: Path) -> tuple[str, Mesh]:
    """Overlap a large grid's unchanged SHA stream with its mesh read.

    Only plain-file hashing runs on the worker. NetCDF and connectivity
    admission stay on the caller, and hash errors retain their precedence
    over mesh-read errors. The builder still rechecks the digest before
    publishing its receipt. Small files and one-worker hosts stay inline.
    """
    from hexcore.file_digest import MIN_CACHED_BYTES
    from hexcore.host_parallel import threads

    if threads() == 1 or grid.stat().st_size < MIN_CACHED_BYTES:
        return file_version_sha256(grid), Mesh.from_netcdf(grid, validate=False)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hex-partition-digest") as pool:
        pending = pool.submit(file_version_sha256, grid)
        try:
            mesh = Mesh.from_netcdf(grid, validate=False)
        except BaseException:
            pending.result()
            raise
        return pending.result(), mesh


def build_assets(
    grid: Path,
    *,
    ranks: int,
    output_dir: Path,
    part: Path | None = None,
    halo_rings: int = 3,
    gpmetis: str | None = None,
) -> dict[str, Any]:
    """Build or verify reusable graph, part, layout and receipt assets."""

    grid = Path(grid).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if ranks < 1 or halo_rings < 0:
        raise ValueError("ranks must be positive and halo_rings non-negative")
    reused = reuse_verified_receipt(grid, ranks=ranks, output_dir=output_dir, part=part,
                                    halo_rings=halo_rings)
    if reused is not None:
        return reused
    # Cold construction used to read the entire grid twice, then hash it
    # again on first receipt reuse. Share the existing version-keyed digest
    # cache with the warm path. A changed inode, size, mtime or ctime still
    # forces a real hash at the final check; cache=off retains direct reads.
    mesh_sha, mesh = _load_mesh_and_digest(grid)
    regional = bool(mesh.is_regional) or any(
        np.any(np.asarray(mesh.arrays[name]) < 0)
        for name in ("cellsOnEdge", "cellsOnVertex")
    )
    replicated_cells = (int(mesh.dimensions["nCells"]) - 1,) if regional else ()
    packed = PackedGraph(mesh)
    if ranks > packed.n_cells:
        raise ValueError("more ranks than cells would create a rank with no owned cells")
    if packed.components() != 1:
        raise ValueError("contiguous METIS partitions need a connected cell graph")
    tag = grid.stem.removesuffix(".grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = output_dir / f"{tag}.graph.info"
    graph_edges = packed.flat.size // 2
    graph_bytes = packed.metis_bytes()
    _write_or_verify(graph, graph_bytes)
    target_part = Path(f"{graph}.part.{ranks}")
    part_source = Path(part).resolve() if part is not None else target_part
    metis_command = None
    metis_log = output_dir / f"{tag}.metis{ranks}.log"
    if part_source.exists():
        _write_or_verify(target_part, part_source.read_bytes())
    elif ranks == 1:
        _write_or_verify(target_part, ("0\n" * packed.n_cells).encode("ascii"))
    else:
        executable = metis_program(gpmetis)
        if target_part.exists():
            raise ValueError("a generated part file already exists; select it with --part or use a fresh directory")
        metis_command = [executable, "-ptype=kway", "-objtype=cut", "-contig", "-seed=841", "-ncuts=4", "-niter=20", str(graph), str(ranks)]
        completed = subprocess.run(metis_command, cwd=output_dir, check=False, capture_output=True, text=True, timeout=600)
        metis_log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0 or not target_part.is_file():
            raise RuntimeError(f"gpmetis did not produce a partition file; exit={completed.returncode}; log={metis_log}")
    if part is not None and not part_source.exists():
        _write_or_verify(part_source, target_part.read_bytes())
    part_sha = sha256_file(target_part)
    assignments = read_part_file(target_part, n_cells=packed.n_cells)
    if not np.array_equal(np.unique(assignments), np.arange(ranks)):
        raise ValueError(f"the part file does not cover exactly ranks 0..{ranks - 1}")
    components = packed.partition_components(assignments, ranks)
    if any(count != 1 for count in components):
        raise ValueError(f"partitions are not contiguous in the cell graph: {components}")
    cut_edges = packed.cut_edges(assignments)
    target = output_dir / f"{tag}.part{ranks}.K{halo_rings}.npz"
    if target.exists():
        layouts = load_layouts_npz(target, expect_mesh_sha256=mesh_sha, expect_part_sha256=part_sha, expect_halo_rings=halo_rings, expect_replicated_cells=replicated_cells)
    else:
        layouts = build_partition_layouts_from_csr(packed.offsets, packed.flat, assignments, halo_rings=halo_rings,
                                         cells_on_edge=mesh.arrays["cellsOnEdge"],
                                         cells_on_vertex=mesh.arrays["cellsOnVertex"],
                                         edges_on_edge=mesh.arrays.get("edgesOnEdge"),
                                         n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge"),
                                         replicated_cells=replicated_cells)
        save_layouts_npz(target, layouts, mesh_sha256=mesh_sha, part_sha256=part_sha, halo_rings=halo_rings)
    extents = {"n_cells": int(mesh.dimensions["nCells"]), "n_edges": int(mesh.dimensions["nEdges"]), "n_vertices": int(mesh.dimensions["nVertices"])}
    cover = assert_exact_cover(layouts, **extents)
    if len(layouts) != ranks:
        raise ValueError("cached layouts do not match the requested rank count")
    if file_version_sha256(grid) != mesh_sha or sha256_file(target_part) != part_sha or sha256_file(part_source) != part_sha:
        raise ValueError("the mesh or part file changed while layouts were being built")
    receipt_path = output_dir / f"{tag}.part{ranks}.K{halo_rings}.json"
    receipt = {
        "schema": SCHEMA,
        "builder_version": BUILDER_VERSION,
        "grid": str(grid), "grid_sha256": mesh_sha,
        "graph": str(graph), "graph_sha256": sha256_file(graph),
        "part_source": str(part_source), "part": str(target_part), "part_sha256": part_sha,
        "layouts": str(target), "layouts_sha256": sha256_file(target),
        "ranks": ranks, "halo_rings": halo_rings,
        "replicated_cells": list(replicated_cells),
        "graph_edges": graph_edges, "cut_edges": int(cut_edges),
        "owned_connected_components": components,
        "exact_cover": cover, "partitions": [layout.receipt() for layout in layouts],
        "metis_command": metis_command,
        "metis_log": str(metis_log) if metis_log.exists() else None,
        "receipt": str(receipt_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--part", type=Path, help="existing exact-N part file, or a destination for generated bytes")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--halo-rings", type=int, default=3)
    parser.add_argument("--gpmetis", default=None,
                        help="a gpmetis to run (default: the bundled hex_gpmetis, then PATH)")
    args = parser.parse_args(argv)
    receipt = build_assets(args.grid, ranks=args.ranks, output_dir=args.output_dir, part=args.part, halo_rings=args.halo_rings, gpmetis=args.gpmetis)
    print(json.dumps({key: receipt[key] for key in ("ranks", "cut_edges", "layouts", "layouts_sha256", "receipt")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
