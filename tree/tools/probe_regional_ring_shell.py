#!/usr/bin/env python3
"""What a limited-area mesh's boundary rings are, and what they are not.

THE MEASUREMENT THIS EXISTS FOR.  ``Mesh.validate`` refused two culls that
``rw_mpas_mesh --cull-parent`` produced from the same parent, with the same
binary, in the same run as a third cull it accepted -- and the third is
byte-identical to the registered ``r4.75.11020``.  The refusal was that ring
populations shrink outward, which the message reads as "a torn or renumbered
zone".

They shrink for a reason that has nothing to do with tearing.  The rings grow
OUTWARD from the requested polygon into the parent's coarsening ramp, and a
ring made of larger cells needs fewer of them to wrap the same interior.
Population is a proxy for wrapping only on a uniform mesh, and the only meshes
this program culls are variable resolution.

So this prints the three numbers together -- population, mean cell width, and
whether every ring-k cell actually touches a ring-(k-1) cell.  The last one is
what "each ring wraps the previous" means, it is what a torn zone would break,
and it is independent of how big the cells are.

    python tools/probe_regional_ring_shell.py GRID.nc [GRID.nc ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ZONE_RINGS = 8

#: A grid file stores dcEdge on the UNIT sphere, in radians, not in metres --
#: the sphere radius lives in the file's own ``sphere_radius`` attribute and
#: the arrays are normalised by it.  Reading these as metres gives cell widths
#: of about seven ten-thousandths of a kilometre, which is small enough to look
#: like a zero and be reported as one.
SPHERE_RADIUS_KM = 6_371.229


def shell_report(path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as grid:
        mask = np.asarray(grid.variables["bdyMaskCell"][:], np.int64)
        cells_on_cell = np.asarray(grid.variables["cellsOnCell"][:], np.int64) - 1
        n_edges_on_cell = np.asarray(grid.variables["nEdgesOnCell"][:], np.int64)
        edges_on_cell = np.asarray(grid.variables["edgesOnCell"][:], np.int64) - 1
        dc_edge = np.asarray(grid.variables["dcEdge"][:], np.float64)

    n_cells = mask.size
    used = (
        np.arange(cells_on_cell.shape[1])[None, :] < n_edges_on_cell[:, None]
    ) & (cells_on_cell >= 0)
    source = np.broadcast_to(np.arange(n_cells)[:, None], cells_on_cell.shape)[used]
    neighbour = cells_on_cell[used]

    # Containment, ring by ring: the property the population test was standing
    # in for.  A ring-k cell with no ring-(k-1) neighbour is a ring that does
    # not sit on the one inside it, which IS a torn or renumbered zone.
    orphans: dict[str, int] = {}
    for ring in range(1, ZONE_RINGS):
        members = np.flatnonzero(mask == ring)
        if members.size == 0:
            continue
        touches_inner = np.zeros(n_cells, bool)
        pair = mask[neighbour] == ring - 1
        touches_inner[source[pair]] = True
        missing = members[~touches_inner[members]]
        if missing.size:
            orphans[str(ring)] = int(missing.size)

    edge_slots = (
        np.arange(edges_on_cell.shape[1])[None, :] < n_edges_on_cell[:, None]
    ) & (edges_on_cell >= 0)
    safe = np.where(edge_slots, edges_on_cell, 0)
    spacing = np.where(edge_slots, dc_edge[safe], 0.0).sum(axis=1) / np.maximum(
        1, edge_slots.sum(axis=1)
    )

    counts = np.bincount(mask, minlength=ZONE_RINGS)
    return {
        "grid": str(path),
        "cells": int(n_cells),
        "ring_cell_counts": counts[:ZONE_RINGS].tolist(),
        "ring_mean_cell_width_km": [
            (
                round(float(spacing[mask == ring].mean()) * SPHERE_RADIUS_KM, 3)
                if int(counts[ring]) else None
            )
            for ring in range(ZONE_RINGS)
        ],
        "populations_grow_outward": bool(
            all(
                counts[ring] <= counts[ring + 1]
                for ring in range(1, ZONE_RINGS - 1)
            )
        ),
        "ring_cells_with_no_inner_neighbour": orphans,
        "every_ring_wraps_the_one_inside_it": not orphans,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grid", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    rows = [shell_report(path) for path in arguments.grid]
    report = {"schema": "gpuwm-hex.regional-ring-shell/v1", "grids": rows}
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for row in rows:
        print(Path(row["grid"]).name)
        print(f"  populations   {row['ring_cell_counts']}")
        print(f"  width km      {row['ring_mean_cell_width_km']}")
        print(f"  grow outward  {row['populations_grow_outward']}")
        print(f"  wraps inside  {row['every_ring_wraps_the_one_inside_it']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
