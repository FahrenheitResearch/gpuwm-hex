#!/usr/bin/env python3
"""Run the port's complete structured-data initialization and emit a receipt.

This is deliberately a port execution harness, not a stock-MPAS wrapper.  The
input may be a WPS intermediate GFS file or another format accepted by
``StructuredAtmosphere.from_file``.  The receipt labels numerical equivalence
honestly: a finite initialized state is execution evidence, not yet a matched
Fortran initialization oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

from mpas_port.initialization import initialize_from_structured, load_structured_atmosphere
from mpas_port.mesh import Mesh
from mpas_port.vertical import build_vertical_grid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.grid.nc"
DEFAULT_STATIC = ROOT / "data" / "meshes" / "x1.2562" / "x1.2562.static.nc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_receipt(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    selected = values[finite]
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "count": int(values.size),
        "finite_count": int(np.count_nonzero(finite)),
        "min": float(np.min(selected)) if selected.size else None,
        "max": float(np.max(selected)) if selected.size else None,
        "mean_float64": float(np.mean(selected, dtype=np.float64)) if selected.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--levels", type=int, default=55)
    parser.add_argument("--ztop", type=float, default=30_000.0)
    parser.add_argument(
        "--smooth-surfaces",
        action="store_true",
        help="enable the expensive frozen vertical-surface smoother",
    )
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve(strict=True)
    started = time.perf_counter()
    source = load_structured_atmosphere(source_path)
    loaded = time.perf_counter()
    mesh = Mesh.from_netcdf(args.grid, args.static)
    meshed = time.perf_counter()
    vertical = build_vertical_grid(
        mesh,
        np.asarray(mesh.ter, dtype=np.float64),
        n_vert_levels=args.levels,
        ztop=args.ztop,
        smooth_surfaces=args.smooth_surfaces,
    )
    verticalized = time.perf_counter()
    initialized = initialize_from_structured(source, mesh, vertical)
    finished = time.perf_counter()

    state = initialized.state
    payload = {
        "schema": "mpas-port.gfs-initialization-receipt.v1",
        "evidence": {
            "status": initialized.evidence,
            "claim": "real source ingested and finite MPAS state assembled",
            "non_claim": "a frozen-Fortran initialization match has not yet been established",
        },
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "adapter": source.provenance.get("adapter"),
            "valid_time": source.provenance.get("valid_time"),
            "forecast_hour": source.provenance.get("forecast_hour"),
        },
        "mesh": {
            "grid": str(Path(args.grid).resolve()),
            "grid_sha256": sha256_file(Path(args.grid)),
            "static": str(Path(args.static).resolve()),
            "static_sha256": sha256_file(Path(args.static)),
            "nCells": int(mesh.dimensions["nCells"]),
            "nEdges": int(mesh.dimensions["nEdges"]),
        },
        "vertical_grid": {
            "levels": args.levels,
            "ztop": args.ztop,
            "smooth_surfaces": args.smooth_surfaces,
        },
        "timing_seconds": {
            "source_load": loaded - started,
            "mesh_load": meshed - loaded,
            "vertical_grid": verticalized - meshed,
            "initialization": finished - verticalized,
            "total_before_hashing": finished - started,
        },
        "state": {
            "rho": array_receipt(state.rho),
            "rho_theta": array_receipt(state.rho_theta),
            "rho_u": array_receipt(state.rho_u),
            "rho_w": array_receipt(state.rho_w),
            "scalars": array_receipt(state.scalars),
        },
        "diagnostics": {
            "pressure": array_receipt(initialized.diagnostics.pressure),
            "temperature": array_receipt(initialized.diagnostics.temperature),
            "qv": array_receipt(initialized.diagnostics.water_vapor_mixing_ratio),
            "hydrostatic_iterations_max": int(
                np.max(initialized.diagnostics.hydrostatic_iterations)
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.receipt is not None:
        receipt = args.receipt.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
