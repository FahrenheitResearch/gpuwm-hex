#!/usr/bin/env python3
"""Diagnostic: localize a 2-GPU vs 1-GPU step divergence on the mesh.

For every differing element of every dumped field, report its graph distance
from the partition cut (BFS over cellsOnCell).  Near-cut-only differences
implicate the exchange/stencil law; global differences implicate a shared
input.  CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexcore.partition_assets_v841 import (  # noqa: E402
    load_layouts_npz,
    read_graph_info,
    read_part_file,
    sha256_file,
)


def cut_distances(graph: Path, part: Path) -> np.ndarray:
    offsets, flat, _ = read_graph_info(graph)
    n_cells = offsets.size - 1
    parts = read_part_file(part, n_cells=n_cells)
    distance = np.full(n_cells, -1, dtype=np.int32)
    frontier = []
    for cell in range(n_cells):
        row = flat[offsets[cell] : offsets[cell + 1]]
        if np.any(parts[row] != parts[cell]):
            distance[cell] = 0
            frontier.append(cell)
    ring = 0
    frontier = np.asarray(frontier, dtype=np.int64)
    while frontier.size:
        ring += 1
        candidates = np.concatenate(
            [flat[offsets[c] : offsets[c + 1]] for c in frontier]
        ).astype(np.int64)
        fresh = np.unique(candidates[distance[candidates] < 0])
        distance[fresh] = ring
        frontier = fresh
    return distance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembled", type=Path, required=True)
    parser.add_argument("--whole", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--part", type=Path, required=True)
    parser.add_argument("--grid-cells-on-edge", type=Path, default=None,
                        help="npy of zero-based cellsOnEdge; else derived from layouts")
    parser.add_argument("--layouts", type=Path, default=None)
    args = parser.parse_args()

    assembled = np.load(args.assembled)
    whole = np.load(args.whole)
    distance = cut_distances(args.graph, args.part)

    edge_distance = None
    if args.layouts is not None:
        # derive an edge distance via the grid file is heavy; approximate via
        # layouts' edge rings is partition-relative -- prefer cellsOnEdge npy
        pass
    if args.grid_cells_on_edge is not None:
        coe = np.load(args.grid_cells_on_edge)
        edge_distance = np.minimum(distance[coe[:, 0]], distance[coe[:, 1]])

    report: dict[str, object] = {}
    for key in sorted(assembled.files):
        mine = assembled[key]
        ref_key = key if key in whole.files else key.replace("__", ".")
        theirs = whole[ref_key]
        if mine.shape != theirs.shape:
            report[key] = {"shape_mismatch": [list(mine.shape), list(theirs.shape)]}
            continue
        diff = mine != theirs
        count = int(diff.sum())
        entry: dict[str, object] = {"differing_values": count, "total": int(diff.size)}
        if count:
            axis_extent = mine.shape[-1]
            columns = np.unique(np.nonzero(diff)[-1])
            entry["differing_entities"] = int(columns.size)
            if axis_extent == distance.size:
                hist: dict[str, int] = {}
                for cell in columns:
                    hist[str(int(distance[cell]))] = hist.get(str(int(distance[cell])), 0) + 1
                entry["cells_by_cut_distance"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
                entry["max_cut_distance"] = int(distance[columns].max())
            elif edge_distance is not None and axis_extent == edge_distance.size:
                hist = {}
                for edge in columns:
                    hist[str(int(edge_distance[edge]))] = hist.get(str(int(edge_distance[edge])), 0) + 1
                entry["edges_by_cut_distance"] = dict(sorted(hist.items(), key=lambda kv: int(kv[0])))
                entry["max_cut_distance"] = int(edge_distance[columns].max())
        report[key] = entry
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
