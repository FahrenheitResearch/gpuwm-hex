"""The placement layer against the artifacts, end to end, with no card.

WHAT THIS PROVES.  Every leg below runs a SHIPPED binary or a shipped door
on a REAL published mesh; none of it is a mock and none of it opens a
device:

  1. ``hexcore.output.write_history`` writes a history on the real
     ``x1.40962`` cell graph, so the detector reads the port's own on-disk
     conventions rather than a file this lane invented.
  2. ``gpuwm-hex swath plan`` places the swaths.
  3. ``rw_mpas_mesh --spec ... --dry-run`` sizes each placement -- the
     generator's own sizing integral, not this layer's arithmetic.
  4. ``rw_mpas_mesh --cull-parent ... --region <the row this layer
     emitted>`` culls a real limited-area mesh out of the real parent.
     This is the leg that proves the emitted shape needs no new grammar:
     the culler is unmodified and has never heard of a swath.
  5. ``gpuwm-hex mesh-check --grid-only`` validates each culled grid --
     Euler characteristic, boundary rings, dual-edge admission.
  6. The area integral this layer prices with is CALIBRATED against the
     culler's own ``region_cells`` on those same culls, which is what
     turns ``predicted_cells_in`` from arithmetic into a measured
     estimator with a stated error.

WHAT IT DOES NOT PROVE.  The FIELDS in leg 1 are analytic (see
``build_swath_fixture_history``).  Nothing here says the thresholds in the
shipped metrics document find real tropical cyclones in a real forecast;
that needs a real coarse global run, which needs a card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT / "tools")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import build_swath_fixture_history as fixture  # noqa: E402
from hexcore.swath import registry  # noqa: E402
from hexcore.swath.geometry import (  # noqa: E402
    destination,
    polygon_area_km2,
    predicted_cells_in,
)
from hexcore.swath.sizing import resolve_engine  # noqa: E402


def _run(argv: Sequence[str], *, label: str) -> dict[str, Any]:
    started = time.time()
    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return {
        "label": label,
        "argv": [str(item) for item in argv],
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_swath_placement_chain",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grid", type=Path, required=True,
                        help="a real MPAS grid file to place swaths on and cull from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mesh-exe", type=Path, default=None)
    parser.add_argument("--card", default="rtx-5070-ti")
    arguments = parser.parse_args(argv)

    out = arguments.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    engine = Path(arguments.mesh_exe) if arguments.mesh_exe else resolve_engine(None)
    legs: list[dict[str, Any]] = []

    # -- 1. a history on the real mesh, through the port's own writer ------
    started = time.time()
    history = fixture.build(out / "coarse-history.nc", grid=arguments.grid)
    legs.append({
        "label": "history (hexcore.output.write_history on the real grid)",
        "returncode": 0,
        "seconds": round(time.time() - started, 3),
        "output": str(history),
        "bytes": history.stat().st_size,
    })

    # -- 2/3. plan, priced through the real generator ----------------------
    plan_dir = out / "plan"
    legs.append(_run(
        [sys.executable, "-m", "hexcore", "swath", "plan",
         "--history", str(history), "--out", str(plan_dir),
         "--card", arguments.card, "--mesh-exe", str(engine)],
        label="gpuwm-hex swath plan (prices through rw_mpas_mesh --dry-run)",
    ))
    if legs[-1]["returncode"] != 0:
        _write(out / "CHAIN.json", {"legs": legs, "failed_at": legs[-1]["label"]})
        print(json.dumps(legs[-1], indent=2))
        return 1
    plan = json.loads((plan_dir / "swath-plan.json").read_text(encoding="utf-8"))

    # -- 4/5/6. cull each emitted row with the unmodified culler ----------
    parent_cells = _parent_cells(arguments.grid)
    parent_spacing_km = _spacing_km(parent_cells)
    calibration: list[dict[str, Any]] = []
    for row in plan["admitted"]:
        slot = row["slot_id"]
        region_path = plan_dir / "specs" / f"{slot}.cull-region.json"
        grid_path = out / "culls" / f"{slot}.region.nc"
        receipt_path = out / "culls" / f"{slot}.cull-receipt.json"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        legs.append(_run(
            [str(engine), "--cull-parent", str(arguments.grid),
             "--region", str(region_path), "--out", str(grid_path),
             "--receipt", str(receipt_path), "--clobber"],
            label=f"rw_mpas_mesh --cull-parent ({slot}, the row this layer emitted)",
        ))
        if legs[-1]["returncode"] != 0:
            continue
        cull = json.loads(receipt_path.read_text(encoding="utf-8"))
        legs.append(_run(
            [sys.executable, "-m", "hexcore", "mesh-check",
             "--grid", str(grid_path), "--grid-only"],
            label=f"gpuwm-hex mesh-check --grid-only ({slot})",
        ))
        measured = int(cull["region_cells"])
        predicted = predicted_cells_in(
            [tuple(vertex) for vertex in row["ring_deg"]], spacing_km=parent_spacing_km
        )
        calibration.append({
            "slot_id": slot,
            "shape_kind": row["cull_region"]["kind"],
            "parent_cells": int(cull["parent_cells"]),
            "measured_region_cells": measured,
            "predicted_cells_in": round(predicted, 2),
            "relative_error": round((predicted - measured) / measured, 5),
            "euler_v_minus_e_plus_f": int(cull["euler_v_minus_e_plus_f"]),
            "ring_edge_counts": list(cull["ring_edge_counts"]),
            "output_sha256": cull["output_sha256"],
            "mesh_check_returncode": legs[-1]["returncode"],
        })

    # -- the estimator's own error curve, measured against the culler ----
    sweep = _calibration_sweep(engine, arguments.grid, out, parent_spacing_km, legs)

    errors = [abs(entry["relative_error"]) for entry in calibration]
    summary = {
        "schema": "gpuwm-hex.swath-chain.v1",
        "parent_grid": str(arguments.grid),
        "parent_cells": parent_cells,
        "parent_spacing_km": round(parent_spacing_km, 4),
        "engine": str(engine),
        "admitted": len(plan["admitted"]),
        "declined": len(plan["declined"]),
        "calibration": calibration,
        "predicted_cells_in_worst_relative_error": (
            round(max(errors), 5) if errors else None
        ),
        "all_culls_are_bounded_disks": all(
            entry["euler_v_minus_e_plus_f"] == 1 for entry in calibration
        ),
        "all_mesh_checks_passed": all(
            entry["mesh_check_returncode"] == 0 for entry in calibration
        ),
        "estimator_error_curve": sweep,
        "legs": legs,
    }
    _write(out / "CHAIN.json", summary)
    print(json.dumps({
        key: summary[key] for key in (
            "parent_cells", "parent_spacing_km", "admitted", "declined",
            "predicted_cells_in_worst_relative_error",
            "all_culls_are_bounded_disks", "all_mesh_checks_passed",
        )
    }, indent=2))
    print("  estimator error against the shipped culler, by region width in cells:")
    for entry in sweep:
        print(
            f"    r={entry['radius_km']:>6.0f} km  {entry['cells_across_radius']:>6.1f} "
            f"cells  culled {entry['measured_region_cells']:>6}  predicted "
            f"{entry['predicted_cells_in']:>8.1f}  error "
            f"{entry['relative_error'] * 100:+7.2f}%"
        )
    for entry in calibration:
        print(
            f"  {entry['slot_id']}  {entry['shape_kind']:<8} culled "
            f"{entry['measured_region_cells']:>6} cells, predicted "
            f"{entry['predicted_cells_in']:>9.1f}  error "
            f"{entry['relative_error'] * 100:+.2f}%  euler "
            f"{entry['euler_v_minus_e_plus_f']}"
        )
    return 0 if summary["all_culls_are_bounded_disks"] and summary["all_mesh_checks_passed"] else 1


def _calibration_sweep(
    engine: Path, grid: Path, out: Path, spacing_km: float, legs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cull a family of caps of growing radius and record this layer's error.

    Caps rather than swaths because a cap's exact spherical area is a
    closed form, so the only thing under test is the estimator's treatment
    of the boundary halo and of discretization -- not the ring geometry.
    The independent variable is the RATIO of the region's radius to the
    mesh spacing, because that is what decides both error terms: a region
    three cells across is dominated by which cell centres happen to fall
    inside it, and a region a hundred cells across is not.
    """

    rows: list[dict[str, Any]] = []
    for radius_km in (300.0, 600.0, 1200.0, 2400.0, 4800.0):
        region = {
            "kind": "cap",
            "center_deg": [18.0, -55.0],
            "radius_km": radius_km,
        }
        region_path = out / "calibration" / f"cap-{int(radius_km)}.region.json"
        grid_path = out / "calibration" / f"cap-{int(radius_km)}.region.nc"
        receipt_path = out / "calibration" / f"cap-{int(radius_km)}.cull-receipt.json"
        _write(region_path, region)
        legs.append(_run(
            [str(engine), "--cull-parent", str(grid), "--region", str(region_path),
             "--out", str(grid_path), "--receipt", str(receipt_path), "--clobber"],
            label=f"rw_mpas_mesh --cull-parent (calibration cap {radius_km:.0f} km)",
        ))
        if legs[-1]["returncode"] != 0:
            continue
        cull = json.loads(receipt_path.read_text(encoding="utf-8"))
        ring = [destination(18.0, -55.0, bearing, radius_km) for bearing in range(0, 360, 2)]
        predicted = predicted_cells_in(ring, spacing_km=spacing_km)
        measured = int(cull["region_cells"])
        rows.append({
            "radius_km": radius_km,
            "cells_across_radius": round(radius_km / spacing_km, 2),
            "measured_region_cells": measured,
            "measured_interior_cells": int(cull["mark"]["ring_cell_counts"][0]),
            "predicted_cells_in": round(predicted, 2),
            "relative_error": round((predicted - measured) / measured, 5),
            "ring_area_km2": round(polygon_area_km2(ring), 1),
        })
    return rows


def _write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parent_cells(grid: Path) -> int:
    from netCDF4 import Dataset

    with Dataset(str(grid), "r") as source:
        return int(source.dimensions["nCells"].size)


def _spacing_km(cells: int) -> float:
    """Across-flats spacing of a uniform mesh of this cell count."""

    import math

    from hexcore.swath.geometry import EARTH_RADIUS_KM, HEXAGON_AREA_FACTOR

    area = 4.0 * math.pi * EARTH_RADIUS_KM**2 / cells
    return math.sqrt(area / HEXAGON_AREA_FACTOR)


if __name__ == "__main__":
    raise SystemExit(main())
