#!/usr/bin/env python
"""Write an explicit local-timestep classing for an A/B arm.

The shipped option classes cells from the grid file's own ``dcEdge`` and holds
every driven boundary cell of a limited-area cull at rate 1.  On the culls the
doors make (a fine core cut at 1.35 times its radius, seven driven rings) that
leaves ONE class: every cell whose spacing would earn rate 3 sits in the
driven rings or one cell inside them, so the option is inert there and the run
is bit-identical to the default.  That answers "does it pay" (it cannot) but
not "what does a class interface do to the fields", which needs an interface
in the interior.

This instrument writes ``cell_rate`` for ``--local-timestep-classing``: rate 3
for every INTERIOR cell whose spacing ratio ``h_c / h_min`` is at least
``--ratio`` (below the ladder's own 3, deliberately -- see
``lts_v841.classify_from_cell_rates`` for why the acoustic Courant number
keeps that stable), rate 1 elsewhere.  The driven-zone hold and the buffer
ring are applied by the classing itself when the file is used, and the
receipt written beside the file says what the rule produced: cells per rate,
interface edges, the coarse cells' distance band from the point, and the
acoustic Courant number the coarse class will run at.

Usage::

    PYTHONPATH=src python tools/lts_forced_classing.py \\
        --grid <cull>.grid.nc --ratio 2.0 --out forced.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hexcore.lts_v841 import (  # noqa: E402
    cell_min_spacing,
    classify_from_cell_rates,
    classify_local_timestep,
)

EARTH_RADIUS_M = 6_371_229.0
SOUND_SPEED_M_S = 340.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grid", type=Path, required=True, help="the cull's grid file")
    parser.add_argument(
        "--ratio",
        type=float,
        default=2.0,
        help="interior cells with h_c/h_min >= RATIO take rate 3 (default 2.0)",
    )
    parser.add_argument("--rates", default="1,3", help="the ladder (default 1,3)")
    parser.add_argument("--buffer-rings", type=int, default=1)
    parser.add_argument(
        "--dts",
        type=float,
        default=None,
        help="the fine acoustic sub-step in seconds, for the Courant line "
             "(default: read dt from --dt and the (3, 6) schedule)",
    )
    parser.add_argument("--dt", type=float, default=5.0, help="model step (default 5 s)")
    parser.add_argument("--out", type=Path, required=True, help="the .npz to write")
    args = parser.parse_args(argv)

    from netCDF4 import Dataset

    rates = tuple(int(v) for v in args.rates.split(","))
    with Dataset(str(args.grid), "r") as dataset:
        dc_edge = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges_on_cell = np.asarray(dataset.variables["edgesOnCell"][:])
        n_edges_on_cell = np.asarray(dataset.variables["nEdgesOnCell"][:])
        cells_on_edge = np.asarray(dataset.variables["cellsOnEdge"][:])
        driven = (
            np.asarray(dataset.variables["bdyMaskCell"][:])
            if "bdyMaskCell" in dataset.variables
            else np.zeros(edges_on_cell.shape[0], dtype=np.int32)
        )
        lat = np.asarray(dataset.variables["latCell"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lonCell"][:], dtype=np.float64)
        on_sphere = str(getattr(dataset, "on_a_sphere", "YES")).strip().upper().startswith("Y")
        sphere_radius = float(getattr(dataset, "sphere_radius", 1.0))

    h_cell = cell_min_spacing(dc_edge, edges_on_cell, n_edges_on_cell)
    h_min = float(h_cell.min())
    ratio = h_cell / h_min
    interior = driven == 0
    coarse = interior & (ratio >= float(args.ratio))
    cell_rate = np.where(coarse, max(rates), 1).astype(np.int32)

    shipped = classify_local_timestep(
        dc_edge=dc_edge,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        cells_on_edge=cells_on_edge,
        rates=rates,
        buffer_rings=int(args.buffer_rings),
        driven_ring=driven,
        driven_zone_source="bdyMaskCell",
    )
    forced = classify_from_cell_rates(
        cell_rate=cell_rate,
        dc_edge=dc_edge,
        edges_on_cell=edges_on_cell,
        n_edges_on_cell=n_edges_on_cell,
        cells_on_edge=cells_on_edge,
        rates=rates,
        buffer_rings=int(args.buffer_rings),
        driven_ring=driven,
        driven_zone_source="bdyMaskCell",
    )

    # The coarse class's distance band from the fine core's centre (the
    # centroid of the cells within 10 % of the minimum spacing), in km, so a
    # reader can place the interface.
    core = ratio < 1.1
    if on_sphere:
        x = np.cos(lat[core]) * np.cos(lon[core])
        y = np.cos(lat[core]) * np.sin(lon[core])
        z = np.sin(lat[core])
        centre_lat = float(np.arctan2(z.mean(), np.hypot(x.mean(), y.mean())))
        centre_lon = float(np.arctan2(y.mean(), x.mean()))
        cos_d = np.sin(centre_lat) * np.sin(lat) + np.cos(centre_lat) * np.cos(lat) * np.cos(
            lon - centre_lon
        )
        distance_km = np.arccos(np.clip(cos_d, -1.0, 1.0)) * EARTH_RADIUS_M / 1000.0
        # dcEdge in radians on the unit sphere -> metres.
        metres_per_unit = EARTH_RADIUS_M if sphere_radius == 1.0 else 1.0
    else:
        centre_lat = float(lat[core].mean())
        centre_lon = float(lon[core].mean())
        distance_km = np.hypot(lat - centre_lat, lon - centre_lon) / 1000.0
        metres_per_unit = 1.0
    h_min_m = h_min * metres_per_unit
    dts = float(args.dts) if args.dts is not None else float(args.dt) / 3.0 / 6.0
    coarse_final = forced.cell_rate == max(rates)
    coarse_h_min_m = float(h_cell[coarse_final].min() * metres_per_unit) if coarse_final.any() else None
    receipt = {
        "schema": "gpuwm-hex.lts-forced-classing/v1",
        "grid": str(args.grid),
        "rule": {
            "interior_ratio_at_least": float(args.ratio),
            "rates": list(rates),
            "buffer_rings": int(args.buffer_rings),
            "driven_zone": "bdyMaskCell > 0 held at rate 1",
        },
        "h_min_m": h_min_m,
        "centre_deg": [float(np.degrees(centre_lat)), float(np.degrees(centre_lon))] if on_sphere else [centre_lat, centre_lon],
        "shipped_classing": shipped.summary(),
        "forced_classing": forced.summary(),
        "coarse_class": {
            "cells_requested": int(coarse.sum()),
            "cells_after_buffer": int(coarse_final.sum()),
            "distance_from_centre_km": (
                None
                if not coarse_final.any()
                else {
                    "min": float(distance_km[coarse_final].min()),
                    "median": float(np.median(distance_km[coarse_final])),
                    "max": float(distance_km[coarse_final].max()),
                }
            ),
            "spacing_m": (
                None
                if not coarse_final.any()
                else {
                    "min": coarse_h_min_m,
                    "max": float(h_cell[coarse_final].max() * metres_per_unit),
                }
            ),
            "acoustic_courant": {
                "fine_class_at_h_min": SOUND_SPEED_M_S * dts / h_min_m,
                "coarse_class_at_its_finest_cell": (
                    None
                    if coarse_h_min_m is None
                    else SOUND_SPEED_M_S * dts * max(rates) / coarse_h_min_m
                ),
                "dts_seconds": dts,
                "sound_speed_m_s": SOUND_SPEED_M_S,
            },
        },
        "interface": {
            "edges": int(forced.interface_edges.size),
            "distance_from_centre_km": (
                None
                if forced.interface_edges.size == 0
                else {
                    "min": float(distance_km[forced.interface_cells].min()),
                    "max": float(distance_km[forced.interface_cells].max()),
                }
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), cell_rate=cell_rate)
    receipt_path = args.out.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("coarse_class", "interface")}, indent=2))
    print(f"wrote {args.out} and {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
