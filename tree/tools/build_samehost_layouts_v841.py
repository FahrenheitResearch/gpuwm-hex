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
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.mesh import Mesh
from hexcore.partition_assets_v841 import (
    BUILDER_VERSION,
    assert_exact_cover,
    build_partition_layouts,
    load_layouts_npz,
    read_part_file,
    save_layouts_npz,
    sha256_file,
)

SCHEMA = "mpas-port.samehost-layout-assets/v1"


def graph_rows(mesh: Any) -> tuple[tuple[int, ...], ...]:
    """Read the undirected cell graph, excluding canonical missing slots."""

    n_cells = int(mesh.dimensions["nCells"])
    adjacency = np.asarray(mesh.arrays["cellsOnCell"])
    counts = np.asarray(mesh.arrays["nEdgesOnCell"])
    if adjacency.ndim != 2 or adjacency.shape[0] != n_cells or counts.shape != (n_cells,):
        raise ValueError("cell adjacency/count shapes do not match nCells")
    if adjacency.dtype.kind not in "iu" or counts.dtype.kind not in "iu":
        raise ValueError("cell adjacency and counts must be integer metadata")
    rows = []
    for cell in range(n_cells):
        count = int(counts[cell])
        if not 0 <= count <= adjacency.shape[1]:
            raise ValueError(f"cell {cell}: nEdgesOnCell is outside its adjacency row")
        active = adjacency[cell, :count]
        if active.size and (active.min() < -1 or active.max() >= n_cells):
            raise ValueError(f"cell {cell}: adjacency is outside nCells or the -1 sentinel")
        row = tuple(sorted({int(value) for value in active if value >= 0}))
        if cell in row:
            raise ValueError(f"cell {cell}: a self edge cannot be passed to METIS")
        rows.append(row)
    sets = [set(row) for row in rows]
    for cell, row in enumerate(rows):
        for neighbor in row:
            if cell not in sets[neighbor]:
                raise ValueError(f"cell graph is not reciprocal at {cell}->{neighbor}")
    return tuple(rows)


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
    mesh_sha = sha256_file(grid)
    mesh = Mesh.from_netcdf(grid, validate=False)
    regional = bool(mesh.is_regional) or any(
        np.any(np.asarray(mesh.arrays[name]) < 0)
        for name in ("cellsOnEdge", "cellsOnVertex")
    )
    from hexcore.cuda_regional_forecast_v841 import regional_garbage_donor
    replicated_cells = (regional_garbage_donor(mesh.attrs, int(mesh.dimensions["nCells"])),) if regional else ()
    rows = graph_rows(mesh)
    if ranks > len(rows):
        raise ValueError("more ranks than cells would create a rank with no owned cells")
    if component_count(rows) != 1:
        raise ValueError("contiguous METIS partitions need a connected cell graph")
    tag = grid.stem.removesuffix(".grid")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = output_dir / f"{tag}.graph.info"
    graph_edges = sum(len(row) for row in rows) // 2
    graph_bytes = ("\n".join([f"{len(rows)} {graph_edges}"] + [" ".join(str(cell + 1) for cell in row) for row in rows]) + "\n").encode("ascii")
    _write_or_verify(graph, graph_bytes)
    target_part = Path(f"{graph}.part.{ranks}")
    part_source = Path(part).resolve() if part is not None else target_part
    metis_command = None
    metis_log = output_dir / f"{tag}.metis{ranks}.log"
    if part_source.exists():
        _write_or_verify(target_part, part_source.read_bytes())
    elif ranks == 1:
        _write_or_verify(target_part, ("0\n" * len(rows)).encode("ascii"))
    else:
        from hexcore.drivers.samehost_layouts_v841 import metis_program

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
    assignments = read_part_file(target_part, n_cells=len(rows))
    if set(int(value) for value in assignments) != set(range(ranks)):
        raise ValueError(f"the part file does not cover exactly ranks 0..{ranks - 1}")
    components = [component_count(rows, np.flatnonzero(assignments == rank)) for rank in range(ranks)]
    if any(count != 1 for count in components):
        raise ValueError(f"partitions are not contiguous in the cell graph: {components}")
    cut_edges = sum(assignments[cell] != assignments[neighbor] for cell, row in enumerate(rows) for neighbor in row) // 2
    target = output_dir / f"{tag}.part{ranks}.K{halo_rings}.npz"
    if target.exists():
        layouts = load_layouts_npz(target, expect_mesh_sha256=mesh_sha, expect_part_sha256=part_sha, expect_halo_rings=halo_rings, expect_replicated_cells=replicated_cells)
    else:
        layouts = build_partition_layouts(graph, target_part, halo_rings=halo_rings, cells_on_edge=mesh.arrays["cellsOnEdge"], cells_on_vertex=mesh.arrays["cellsOnVertex"], replicated_cells=replicated_cells, edges_on_edge=mesh.arrays.get("edgesOnEdge"), n_edges_on_edge=mesh.arrays.get("nEdgesOnEdge") if mesh.arrays.get("edgesOnEdge") is not None else None)
        save_layouts_npz(target, layouts, mesh_sha256=mesh_sha, part_sha256=part_sha, halo_rings=halo_rings)
    extents = {"n_cells": int(mesh.dimensions["nCells"]), "n_edges": int(mesh.dimensions["nEdges"]), "n_vertices": int(mesh.dimensions["nVertices"])}
    cover = assert_exact_cover(layouts, **extents)
    if len(layouts) != ranks:
        raise ValueError("cached layouts do not match the requested rank count")
    if sha256_file(grid) != mesh_sha or sha256_file(target_part) != part_sha or sha256_file(part_source) != part_sha:
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
