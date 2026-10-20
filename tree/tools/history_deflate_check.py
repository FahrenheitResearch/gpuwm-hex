"""Write one history frame both ways and compare the files byte for byte.

    python tools/history_deflate_check.py FRAME.nc --out DIR [--cells N]

FRAME.nc is a hex history file (``cuda-history.*.nc``).  Its arrays are read
back into the snapshot ``write_snapshot_netcdf`` takes, optionally tiled along
the cell axis to N cells (``--cells 2621442`` makes a 15 km-sized frame from a
coarser one), and written twice: the single-thread HDF5 way, and through
:mod:`hexcore.history_deflate`.  Prints the seconds of each, the sha256 of
each, and the deflate counts.  Exit 0 only when the two files are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def tile(array: np.ndarray, size: int, axis: int) -> np.ndarray:
    reps = -(-size // array.shape[axis])
    tiled = np.concatenate([array] * reps, axis=axis) if reps > 1 else array
    return np.ascontiguousarray(np.take(tiled, np.arange(size), axis=axis))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("frame", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=None)
    args = parser.parse_args(argv)
    from netCDF4 import Dataset

    from hexcore import history_deflate
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof

    with Dataset(args.frame) as source:
        cells = len(source.dimensions["nCells"])
        edges = len(source.dimensions["nEdges"])
        levels = len(source.dimensions["nVertLevels"])
        interfaces = len(source.dimensions["nVertLevelsP1"])
        soil = len(source.dimensions["nSoilLevels"]) if "nSoilLevels" in source.dimensions else 0
        static = {name: np.asarray(source[name][:]) for name in
                  ("indexToCellID", "latCell", "lonCell", "ter", "indexToEdgeID", "latEdge", "lonEdge")}
        arrays = {}
        for name, variable in source.variables.items():
            if variable.dimensions[:1] != ("Time",):
                continue
            value = np.asarray(variable[:], dtype=np.float32)[0]
            arrays[name] = np.ascontiguousarray(value.T) if value.ndim == 2 else value
    scale = 1
    if args.cells:
        scale = args.cells / cells
        new_cells, new_edges = args.cells, int(round(edges * scale))
        for name, value in list(arrays.items()):
            axis = value.ndim - 1
            size = new_edges if value.shape[axis] == edges else new_cells
            arrays[name] = tile(value, size, axis)
        for name in list(static):
            size = new_edges if static[name].shape[0] == edges else new_cells
            static[name] = tile(static[name], size, 0)
        cells, edges = new_cells, new_edges
    proof.N_CELLS, proof.N_EDGES, proof.N_LEVELS = cells, edges, levels
    proof.N_INTERFACES, proof.N_SOIL_LEVELS = interfaces, soil
    snapshot = {"arrays": arrays, "receipt": {"step": 6}}
    raw = sum(int(v.nbytes) for v in arrays.values())
    args.out.mkdir(parents=True, exist_ok=True)
    plain, fast = args.out / "plain.nc", args.out / "fast.nc"
    for path in (plain, fast):
        path.unlink(missing_ok=True)
    started = time.perf_counter()
    proof.write_snapshot_netcdf(plain, snapshot, static)
    plain_seconds = time.perf_counter() - started
    ready = history_deflate.prepare()
    if not ready["engaged"]:
        print(json.dumps({"refused": ready}, indent=2))
        return 2
    started = time.perf_counter()
    with history_deflate.frame(proof.write_snapshot_netcdf, (fast, snapshot, static)) as stats:
        proof.write_snapshot_netcdf(fast, snapshot, static)
    fast_seconds = time.perf_counter() - started
    result = {"frame": str(args.frame), "cells": cells, "edges": edges, "raw_bytes": raw,
              "file_bytes": fast.stat().st_size,
              "single_thread_seconds": round(plain_seconds, 2), "parallel_seconds": round(fast_seconds, 2),
              "plain_sha256": sha256(plain), "fast_sha256": sha256(fast), "deflate": stats}
    result["identical"] = result["plain_sha256"] == result["fast_sha256"]
    print(json.dumps(result, indent=2))
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
