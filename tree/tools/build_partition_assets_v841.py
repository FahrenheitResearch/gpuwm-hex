#!/usr/bin/env python
"""Build the cached partition-layout npz for a mesh + official part file.

CPU only.  Reads ``cellsOnEdge``/``cellsOnVertex`` straight out of the grid
netCDF (raw MPAS files are 1-based; they are shifted to the port's zero-based
convention here) so the builder never has to materialise the whole mesh.

    python tools/build_partition_assets_v841.py \
        --grid $MPAS_ASSET_ROOT/meshes/official-vr-92to25/x4.163842.grid.nc \
        --graph .../x4.163842.graph.info --part .../x4.163842.graph.info.part.16 \
        --halo-rings 60 --out $MPAS_ASSET_ROOT/partitions
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hexcore.partition_assets_v841 import (  # noqa: E402
    BUILDER_VERSION,
    assert_exact_cover,
    build_partition_layouts,
    load_layouts_npz,
    save_layouts_npz,
    sha256_file,
)

# The per-partition memory sanity line answers from the ONE admission
# surface (rerouted 2026-08-25, stale-guard audit #347 finding 5): the
# retired 22 GiB whole-mesh constant scaled linearly with no fixed term,
# the shape the L6 floor re-derivation retired.  The projection below is
# the measured converged row applied to the largest partition's local cell
# count -- a DECLARED DERIVATION (no 2-GPU #264 row exists at the
# converged pin), printed with that label.
from hexcore.device_admission import (  # noqa: E402
    MIB,
    required_free_bytes,
)


def read_zero_based(grid: Path) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    from netCDF4 import Dataset

    with Dataset(grid) as data:
        cells_on_edge = np.asarray(data.variables["cellsOnEdge"][:], dtype=np.int32) - 1
        cells_on_vertex = (
            np.asarray(data.variables["cellsOnVertex"][:], dtype=np.int32) - 1
        )
        n_cells = int(len(data.dimensions["nCells"]))
        n_edges = int(len(data.dimensions["nEdges"]))
        n_vertices = int(len(data.dimensions["nVertices"]))
    if cells_on_edge.min() < 0 or cells_on_vertex.min() < 0:
        raise ValueError(
            "the grid carries padded (0) cell references; this builder assumes "
            "a closed global mesh with every reference valid"
        )
    return cells_on_edge, cells_on_vertex, n_cells, n_edges, n_vertices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--part", required=True, type=Path)
    parser.add_argument("--halo-rings", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mesh-tag", default=None)
    parser.add_argument("--receipt", default=None, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    tag = args.mesh_tag or args.grid.name.split(".grid.nc")[0]
    n_parts = int(args.part.name.rsplit(".", 1)[-1])
    target = Path(args.out) / f"{tag}.part{n_parts}.K{args.halo_rings}.npz"

    started = time.time()
    mesh_sha = sha256_file(args.grid)
    part_sha = sha256_file(args.part)
    hashed = time.time()

    coe, cov, n_cells, n_edges, n_vertices = read_zero_based(args.grid)
    read_done = time.time()

    if target.exists() and not args.force:
        layouts = load_layouts_npz(
            target,
            expect_mesh_sha256=mesh_sha,
            expect_part_sha256=part_sha,
            expect_halo_rings=args.halo_rings,
        )
        built = "loaded-from-cache"
    else:
        layouts = build_partition_layouts(
            args.graph,
            args.part,
            halo_rings=args.halo_rings,
            cells_on_vertex=cov,
            cells_on_edge=coe,
        )
        save_layouts_npz(
            target,
            layouts,
            mesh_sha256=mesh_sha,
            part_sha256=part_sha,
            halo_rings=args.halo_rings,
        )
        built = "built"
    finished = time.time()

    covered = assert_exact_cover(
        layouts, n_cells=n_cells, n_edges=n_edges, n_vertices=n_vertices
    )
    rows = [layout.receipt() for layout in layouts]
    max_local = max(row["n_local_cells"] for row in rows)
    projected_gib = required_free_bytes(max_local) / (1024 * MIB)

    header = (
        f"{'part':>4} {'owned':>8} {'local':>8} {'ratio':>6} "
        f"{'ownedE':>8} {'localE':>8} {'ownedV':>8} {'localV':>8} "
        f"{'ring' + str(args.halo_rings):>8} {'maxring':>7}"
    )
    print(f"mesh={tag} parts={n_parts} K={args.halo_rings} {built}")
    print(f"nCells={n_cells} nEdges={n_edges} nVertices={n_vertices}")
    print(header)
    for row in rows:
        ring_k = row["ring_sizes"][args.halo_rings] if len(
            row["ring_sizes"]
        ) > args.halo_rings else 0
        print(
            f"{row['partition']:>4} {row['n_owned_cells']:>8} {row['n_local_cells']:>8} "
            f"{row['n_local_cells'] / row['n_owned_cells']:>6.2f} "
            f"{row['n_owned_edges']:>8} {row['n_local_edges']:>8} "
            f"{row['n_owned_vertices']:>8} {row['n_local_vertices']:>8} "
            f"{ring_k:>8} {row['max_ring']:>7}"
        )
    print(f"exact cover: {covered}")
    print(
        f"memory sanity: max_local_cells={max_local} -> "
        f"{projected_gib:.3f} GiB required free per partition device "
        "(device_admission.required_free_bytes over local cells; per-"
        "partition application of the measured row is DERIVED, NOT MEASURED)"
    )
    print(f"npz: {target}  sha256={sha256_file(target)}")

    receipt = {
        "builder_version": BUILDER_VERSION,
        "mesh": str(args.grid),
        "mesh_sha256": mesh_sha,
        "graph_info": str(args.graph),
        "part_file": str(args.part),
        "part_sha256": part_sha,
        "halo_rings": int(args.halo_rings),
        "n_partitions": n_parts,
        "n_cells": n_cells,
        "n_edges": n_edges,
        "n_vertices": n_vertices,
        "exact_cover": covered,
        "max_local_cells": max_local,
        "projected_gib_per_partition": projected_gib,
        "npz": str(target),
        "npz_sha256": sha256_file(target),
        "source": built,
        "seconds": {
            "hash": hashed - started,
            "read_grid": read_done - hashed,
            "build": finished - read_done,
        },
        "partitions": rows,
    }
    if args.receipt is not None:
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
