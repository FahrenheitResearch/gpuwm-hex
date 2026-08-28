#!/usr/bin/env python3
"""Does the limited-area run hold the same weather as the uncut parent?

THE COMPARISON IS EXACT ON BOTH SIDES, and that is the point.  A cull moves no
cell centre and invents no field: the child's cells ARE a subset of the
parent's, in parent order, and its initial state is the parent's initial state
sliced.  So the two runs start bit-identical on every cell they share, run the
same dycore at the same timestep, and differ in exactly two things -- the child
has a lateral boundary where the parent has more atmosphere, and the child has
no background outside it.

Everything here is therefore a difference between two model runs on the SAME
cells, reported by boundary ring so a reader can see whether a disagreement
grows inward from the boundary (a boundary-treatment effect) or is flat across
the interior (something else).  A limited-area forecast that agrees with its
parent only in the driven rings has proved nothing; those rings are overwritten
from the boundary every step.

    python tools/measure_lam_interior_agreement.py \
        --lam-grid CULL.nc --global-grid PARENT.nc \n        --lam-frames DIR --global-frames DIR --out J.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

#: The dry-dycore fields both lanes have always published, then the ones a
#: FULL-PHYSICS run adds.  The second list is the reason this comparison is
#: worth more than it was: before 2026-08-26 a limited-area run published
#: none of them, so "does the swath hold the same weather" could only be
#: asked of the dynamics.  A field absent from either side is skipped and
#: named in the receipt rather than silently dropped.
FIELDS = ("theta", "rho", "qv", "w", "pressure")
PHYSICS_FIELDS = (
    "refl10cm", "rainnc", "rainc", "u10", "v10", "t2", "qc", "qr", "qi",
    "hfx", "lh", "tsk",
)


def _frames(directory: Path) -> dict[str, Path]:
    """Frames by valid time, under either writer's file-name prefix.

    The regional instrument writes ``history.<stamp>.nc`` and the production
    forecast door writes ``cuda-history.<stamp>.nc``.  Keying on the stamp
    rather than the whole name is what lets one arm of a comparison come from
    each, which is exactly the comparison a limited-area full-physics run
    wants: its own door's output against the global door's.
    """

    rows: dict[str, Path] = {}
    for path in sorted(directory.glob("*history.*.nc")):
        stamp = path.name.split("history.", 1)[1][: -len(".nc")]
        if stamp in rows:
            raise SystemExit(
                f"{directory} holds two frames for {stamp}: {rows[stamp].name} "
                f"and {path.name}; one valid time cannot have two answers"
            )
        rows[stamp] = path
    return rows


def main() -> int:
    from netCDF4 import Dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lam-grid", type=Path, required=True)
    parser.add_argument("--global-grid", type=Path, required=True)
    parser.add_argument("--lam-frames", type=Path, required=True)
    parser.add_argument("--global-frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    # NOT indexToCellID.  The cull reindexes the child contiguously (1..nCells),
    # exactly as MPAS-Limited-Area v2.2 does, so the child's cell IDs say
    # nothing about which parent cells they were.  The map is recovered from
    # the coordinates instead, which the cull copies byte for byte: a child
    # cell and its parent cell carry the SAME float64 bits in latCell/lonCell,
    # so the lookup is exact and a miss is a refusal rather than a nearest
    # neighbour.
    with Dataset(str(arguments.lam_grid)) as grid:
        child_lat = np.asarray(grid.variables["latCell"][:], np.float64)
        child_lon = np.asarray(grid.variables["lonCell"][:], np.float64)
        bdy = np.asarray(grid.variables["bdyMaskCell"][:], np.int64)
    with Dataset(str(arguments.global_grid)) as grid:
        parent_lat = np.asarray(grid.variables["latCell"][:], np.float64)
        parent_lon = np.asarray(grid.variables["lonCell"][:], np.float64)
    lookup = {
        (int(a), int(b)): index
        for index, (a, b) in enumerate(
            zip(parent_lat.view(np.int64), parent_lon.view(np.int64))
        )
    }
    parent_index = np.empty(child_lat.size, np.int64)
    for index, (a, b) in enumerate(
        zip(child_lat.view(np.int64), child_lon.view(np.int64))
    ):
        found = lookup.get((int(a), int(b)))
        if found is None:
            raise SystemExit(
                f"child cell {index} has no parent cell with the same "
                f"coordinate bits; these two files are not a cull and its parent"
            )
        parent_index[index] = found
    if len(set(parent_index.tolist())) != parent_index.size:
        raise SystemExit("two child cells matched the same parent cell")
    interior = bdy == 0

    lam_frames = _frames(arguments.lam_frames)
    global_frames = _frames(arguments.global_frames)
    shared = sorted(set(lam_frames) & set(global_frames))
    if not shared:
        raise SystemExit("the two runs publish no valid time in common")

    report: dict[str, Any] = {
        "schema": "gpuwm-hex.lam-interior-agreement/v1",
        "lam_grid": str(arguments.lam_grid),
        "lam_cells": int(parent_index.size),
        "interior_cells": int(interior.sum()),
        "ring_cells": {str(r): int((bdy == r).sum()) for r in range(8)},
        "frames": [],
    }

    for valid in shared:
        with Dataset(str(lam_frames[valid])) as a, Dataset(str(global_frames[valid])) as b:
            row: dict[str, Any] = {"xtime": valid, "fields": {}, "absent": []}
            for name in (*FIELDS, *PHYSICS_FIELDS):
                if name not in a.variables or name not in b.variables:
                    row["absent"].append(name)
                    continue
                child = np.asarray(a.variables[name][0], np.float64)
                parent = np.asarray(b.variables[name][0], np.float64)
                # Some frames carry a field on (cells,) and some on
                # (levels, cells); the cell axis is whichever matches.
                if child.ndim == 2 and child.shape[0] != parent_index.size:
                    child = child.T
                    parent = parent.T
                if child.shape[0] != parent_index.size:
                    row["absent"].append(f"{name} (shape {child.shape})")
                    continue
                parent = parent[parent_index]
                delta = child - parent
                scale = float(np.std(parent[interior])) or 1.0
                # A field that is identically zero on both sides -- an ice
                # species neither run made, say -- correlates as NaN and says
                # nothing.  Report it as agreeing exactly rather than as a
                # correlation nobody can read.
                degenerate = bool(
                    np.std(child[interior]) == 0.0 and np.std(parent[interior]) == 0.0
                )
                by_ring = {}
                for ring in range(8):
                    selected = bdy == ring
                    if not selected.any():
                        continue
                    block = np.abs(delta[selected])
                    by_ring[str(ring)] = {
                        "cells": int(selected.sum()),
                        "rms": float(np.sqrt(np.mean(block ** 2))),
                        "max_abs": float(block.max()),
                        "bitwise_equal": bool(
                            np.array_equal(
                                child[selected].astype(np.float32).view(np.uint32),
                                parent[selected].astype(np.float32).view(np.uint32),
                            )
                        ),
                    }
                row["fields"][name] = {
                    "interior_rms": float(
                        np.sqrt(np.mean(delta[interior] ** 2))
                    ),
                    "interior_max_abs": float(np.abs(delta[interior]).max()),
                    "interior_std_of_parent": scale,
                    "interior_rms_over_parent_std": float(
                        np.sqrt(np.mean(delta[interior] ** 2)) / scale
                    ),
                    "correlation_interior": (
                        None
                        if degenerate
                        else float(
                            np.corrcoef(
                                child[interior].ravel(), parent[interior].ravel()
                            )[0, 1]
                        )
                    ),
                    "both_constant": degenerate,
                    "by_ring": by_ring,
                }
            report["frames"].append(row)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    for row in report["frames"]:
        theta = row["fields"].get("theta", {})
        print(
            f"{row['xtime']}  theta interior rms "
            f"{theta.get('interior_rms', float('nan')):.5f} K  "
            f"r={theta.get('correlation_interior', float('nan')):.6f}"
        )
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
