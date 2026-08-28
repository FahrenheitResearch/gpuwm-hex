#!/usr/bin/env python3
"""Does a limited-area interior change when you move its boundary further away?

THE QUESTION THIS SETTLES, AND WHY THE OBVIOUS COMPARISON CANNOT.  A 4.6 km
limited-area forecast driven from a 96 km parent agrees with a global run over
the same ground almost exactly in temperature and poorly in vertical velocity.
Two explanations predict that same gradient and they lead to opposite
decisions:

  1. the boundary is starving the interior, because a parent fifteen times
     coarser cannot hand the child the scales it needs -- in which case
     intermediate resolution levels are worth building; or
  2. the child is correctly resolving structure the parent cannot represent at
     all -- in which case a low correlation on the smallest, fastest field is
     the cascade WORKING, and a ladder buys nothing.

Fine-against-coarse cannot separate them, in either direction, and neither can
any amount of extra fields: both explanations predict a disagreement that is
worst exactly where the resolution gap is widest.

WHAT SEPARATES THEM IS DOMAIN SIZE AT FIXED RESOLUTION.  Run the same spacing
over the same ground with the boundary at different distances and compare the
arms over the SMALLEST arm's interior.  Explanation 1 says the arms disagree
there, and that the disagreement grows as you approach the small arm's edge.
Explanation 2 says the arms agree there, because whatever structure the fine
grid is making, it makes the same structure whether its edge is 200 km away or
600 km away.  No external truth is needed and no ladder has to be built first.

THE COMPARISON IS EXACT.  A cull moves no cell centre, so nested culls of one
parent carry identical float64 bits in latCell/lonCell and a cell is matched to
a cell rather than interpolated to one.  A miss is a refusal.

THE X AXIS IS ONE GEOMETRY FOR EVERY ARM.  Each cell of the patch is placed by
its great-circle distance to the PATCH arm's own driven zone -- the tightest
boundary in the comparison.  Every arm is then plotted against that same
coordinate, so the curves lie on top of each other if the boundary does
nothing and separate near zero if it does.  Binning each arm by its own
boundary distance instead would give every arm a different axis and the one
picture that matters could not be drawn.

    python tools/measure_domain_size_agreement.py \\
        --patch d1 \\
        --reference global \\
        --arm d1=D1/grid.nc:D1/out \\
        --arm d2=D2/grid.nc:D2/out \\
        --arm d3=D3/grid.nc:D3/out \\
        --arm global=PARENT/grid.nc:PARENT/out \\
        --out agreement.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: MPAS's own sphere radius, the one every grid file declares.  Distances are
#: reported in kilometres on it rather than in cells, because a reader
#: comparing this against a WRF nesting rule of thumb thinks in kilometres and
#: because two arms of this comparison have different cell counts.
SPHERE_RADIUS_KM = 6_371.229

FIELDS = ("theta", "rho", "qv", "w", "pressure")
PHYSICS_FIELDS = (
    "refl10cm", "rainnc", "rainc", "u10", "v10", "t2", "qc", "qr", "qi",
    "hfx", "lh", "tsk",
)


def unit_vectors(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """Cell centres as 3-D unit vectors, one row per cell."""

    cos_lat = np.cos(lat_rad)
    return np.stack(
        (cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)),
        axis=1,
    )


def boundary_distance_km(
    patch_vectors: np.ndarray, driven_vectors: np.ndarray
) -> np.ndarray:
    """Great-circle distance from each patch cell to the nearest driven cell.

    Brute force on purpose.  The patch is a few thousand cells and the driven
    zone under two thousand, so the whole matrix is a few million dot products
    -- less time than importing a spatial index, and with no tree-construction
    tolerance to be wrong about.  Chunked so the matrix never has to be
    resident all at once for a larger domain.
    """

    if driven_vectors.size == 0:
        raise SystemExit(
            "the patch arm declares no driven cells (bdyMaskCell is zero "
            "everywhere), so it has no boundary to measure distance from; a "
            "global run cannot be the patch arm"
        )
    out = np.empty(patch_vectors.shape[0], np.float64)
    step = 4096
    for start in range(0, patch_vectors.shape[0], step):
        block = patch_vectors[start : start + step]
        dots = np.clip(block @ driven_vectors.T, -1.0, 1.0)
        out[start : start + step] = np.arccos(dots.max(axis=1))
    return out * SPHERE_RADIUS_KM


def bin_edges_for(distance_km: np.ndarray, width_km: float) -> np.ndarray:
    """Fixed-width bins from zero to just past the furthest patch cell.

    Fixed width rather than quantiles: the claim under test is about a
    PHYSICAL distance -- how far a boundary's influence reaches -- so bins of
    equal kilometres are the ones a reader can compare against a nesting rule
    of thumb.  Quantile bins would make every arm's axis depend on its own
    cell distribution, which is the mistake this module exists to avoid.
    """

    top = float(distance_km.max())
    count = max(1, int(math.ceil(top / width_km)))
    return np.arange(count + 1, dtype=np.float64) * width_km


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def profile_by_distance(
    arm_values: np.ndarray,
    reference_values: np.ndarray,
    distance_km: np.ndarray,
    edges: np.ndarray,
) -> list[dict[str, Any]]:
    """RMS difference and correlation, per distance-from-boundary bin.

    ``arm_values`` and ``reference_values`` are (patch cells,) or
    (patch cells, levels); ``distance_km`` is per patch cell and selects whole
    columns, because a boundary acts on a column, not on a level.
    """

    if arm_values.shape != reference_values.shape:
        raise SystemExit(
            f"arm and reference disagree on shape: {arm_values.shape} against "
            f"{reference_values.shape}"
        )
    if arm_values.shape[0] != distance_km.size:
        raise SystemExit(
            f"the value array has {arm_values.shape[0]} columns and the "
            f"distance array has {distance_km.size}"
        )
    rows: list[dict[str, Any]] = []
    for index in range(edges.size - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        last = index == edges.size - 2
        selected = (distance_km >= low) & (
            (distance_km <= high) if last else (distance_km < high)
        )
        count = int(selected.sum())
        if count == 0:
            continue
        arm = np.asarray(arm_values[selected], np.float64).ravel()
        reference = np.asarray(reference_values[selected], np.float64).ravel()
        delta = arm - reference
        spread = float(np.std(reference))
        both_flat = float(np.std(arm)) == 0.0 and spread == 0.0
        rows.append(
            {
                "from_km": low,
                "to_km": high,
                "cells": count,
                "rms": _rms(delta),
                "max_abs": float(np.abs(delta).max()),
                "reference_std": spread,
                "rms_over_reference_std": (
                    None if spread == 0.0 else _rms(delta) / spread
                ),
                "correlation": (
                    None
                    if both_flat or spread == 0.0 or float(np.std(arm)) == 0.0
                    else float(np.corrcoef(arm, reference)[0, 1])
                ),
            }
        )
    return rows


def _frames(directory: Path) -> dict[str, Path]:
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


def _coordinate_keys(lat: np.ndarray, lon: np.ndarray) -> list[tuple[int, int]]:
    """The exact bits of each cell centre, as a (lat, lon) integer pair.

    Bit equality, not a tolerance, and a PAIR rather than a mixed hash.  Two
    culls of one parent copy these values byte for byte, so the lookup is
    exact and a miss is a refusal instead of a nearest neighbour.  Folding the
    two words into one integer would be faster and would admit a collision --
    two different cells reading as the same cell is precisely the failure this
    comparison cannot survive, since it would report agreement between a cell
    and some other cell entirely.
    """

    return list(
        zip(
            lat.view(np.int64).tolist(),
            lon.view(np.int64).tolist(),
        )
    )


def _read_grid(path: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    with Dataset(str(path)) as grid:
        lat = np.asarray(grid.variables["latCell"][:], np.float64)
        lon = np.asarray(grid.variables["lonCell"][:], np.float64)
        if "bdyMaskCell" in grid.variables:
            bdy = np.asarray(grid.variables["bdyMaskCell"][:], np.int64)
        else:
            bdy = np.zeros(lat.size, np.int64)
        # dcEdge is stored on the UNIT sphere, in radians; the file's own
        # sphere_radius carries the scale.  Read as metres it is about seven
        # ten-thousandths of a kilometre, which is small enough to look like a
        # zero and be reported as one.
        spacing = np.zeros(lat.size, np.float64)
        if {"dcEdge", "edgesOnCell", "nEdgesOnCell"} <= set(grid.variables):
            dc = np.asarray(grid.variables["dcEdge"][:], np.float64)
            eoc = np.asarray(grid.variables["edgesOnCell"][:], np.int64) - 1
            nec = np.asarray(grid.variables["nEdgesOnCell"][:], np.int64)
            used = (np.arange(eoc.shape[1])[None, :] < nec[:, None]) & (eoc >= 0)
            safe = np.where(used, eoc, 0)
            spacing = (
                np.where(used, dc[safe], 0.0).sum(axis=1)
                / np.maximum(1, used.sum(axis=1))
            ) * SPHERE_RADIUS_KM
    return {"lat": lat, "lon": lon, "bdy": bdy, "spacing_km": spacing}


def arm_geometry(arm: dict[str, Any]) -> dict[str, Any]:
    """The two numbers a reader needs to place an arm on the headline axis.

    MEASURED OFF THE ARM'S OWN GRID rather than transcribed into the plotting
    code.  A chart whose x axis is a hand-copied radius is a chart that can
    quietly disagree with the run it describes, and this axis is what the whole
    decision is read along.
    """

    driven = arm["bdy"] > 0
    if not driven.any():
        return {"mean_radius_km": None, "driven_ring_mean_width_km": None}
    # latCell/lonCell are RADIANS on disk (the files declare units "rad"),
    # so they go into unit_vectors as they are.  Converting again shrinks
    # every angle by 57.3 and produced a mean radius of 4.2 km for a domain
    # whose boundary is 135 km out -- small enough to look like a plausible
    # cell width rather than an obvious error.
    centre = unit_vectors(arm["lat"], arm["lon"]).mean(axis=0)
    centre = centre / np.linalg.norm(centre)
    edge = unit_vectors(arm["lat"][driven], arm["lon"][driven])
    radius = np.arccos(np.clip(edge @ centre, -1.0, 1.0)) * SPHERE_RADIUS_KM
    width = arm["spacing_km"][driven]
    return {
        "mean_radius_km": float(radius.mean()),
        "driven_ring_mean_width_km": (
            float(width.mean()) if float(width.mean()) > 0.0 else None
        ),
    }


def _parse_arm(text: str) -> tuple[str, Path, Path]:
    name, _, rest = text.partition("=")
    grid, sep, frames = rest.rpartition(":")
    if not name or not sep or not grid or not frames:
        raise SystemExit(
            f"--arm {text!r} is not NAME=GRID.nc:FRAMES_DIR; every arm needs a "
            "name so the receipt and the chart can say which domain a curve is"
        )
    return name, Path(grid), Path(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=GRID:DIR")
    parser.add_argument(
        "--patch",
        required=True,
        help="the arm whose free interior is the comparison patch, and whose "
        "boundary sets the distance axis; normally the smallest domain",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="the arm every other arm is differenced against; normally the "
        "run with no lateral boundary at all",
    )
    parser.add_argument("--bin-width-km", type=float, default=25.0)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    arms = {}
    for text in arguments.arm:
        name, grid, frames = _parse_arm(text)
        if name in arms:
            raise SystemExit(f"--arm {name!r} was given twice")
        arms[name] = {"grid": grid, "frames_dir": frames}
    for role in ("patch", "reference"):
        chosen = getattr(arguments, role)
        if chosen not in arms:
            raise SystemExit(
                f"--{role} {chosen!r} names no arm; the arms are "
                f"{sorted(arms)}"
            )

    for name, arm in arms.items():
        arm.update(_read_grid(arm["grid"]))
        arm["key"] = _coordinate_keys(arm["lat"], arm["lon"])
        arm["index"] = {key: i for i, key in enumerate(arm["key"])}
        if len(arm["index"]) != len(arm["key"]):
            raise SystemExit(
                f"arm {name!r} carries two cells with the same coordinate bits"
            )

    patch_arm = arms[arguments.patch]
    patch_local = np.flatnonzero(patch_arm["bdy"] == 0)
    if patch_local.size == 0:
        raise SystemExit(
            f"--patch {arguments.patch!r} has no free interior: every cell is "
            "in a driven ring, so there is nothing the forecast owns to compare"
        )
    patch_keys = [patch_arm["key"][index] for index in patch_local.tolist()]

    # Every arm must hold every patch cell.  A missing cell means the arms are
    # not nested and the comparison would quietly become "wherever they happen
    # to overlap", which is a different and much weaker claim.
    selection: dict[str, np.ndarray] = {}
    for name, arm in arms.items():
        rows = np.empty(len(patch_keys), np.int64)
        for position, key in enumerate(patch_keys):
            found = arm["index"].get(key)
            if found is None:
                raise SystemExit(
                    f"arm {name!r} does not contain patch cell {position} of "
                    f"{len(patch_keys)}; the arms are not nested culls of one "
                    "parent, so no patch is shared by all of them"
                )
            rows[position] = found
        selection[name] = rows

    driven_local = np.flatnonzero(patch_arm["bdy"] > 0)
    distance_km = boundary_distance_km(
        unit_vectors(patch_arm["lat"][patch_local], patch_arm["lon"][patch_local]),
        unit_vectors(patch_arm["lat"][driven_local], patch_arm["lon"][driven_local]),
    )
    edges = bin_edges_for(distance_km, arguments.bin_width_km)

    frames = {name: _frames(arm["frames_dir"]) for name, arm in arms.items()}
    shared = sorted(set.intersection(*(set(f) for f in frames.values())))
    if not shared:
        raise SystemExit("the arms publish no valid time in common")

    report: dict[str, Any] = {
        "schema": "gpuwm-hex.domain-size-agreement/v1",
        "patch_arm": arguments.patch,
        "reference_arm": arguments.reference,
        "patch_cells": len(patch_keys),
        "bin_width_km": arguments.bin_width_km,
        "bin_edges_km": [float(v) for v in edges],
        "boundary_distance_km": {
            "min": float(distance_km.min()),
            "max": float(distance_km.max()),
            "mean": float(distance_km.mean()),
        },
        "arms": {
            name: {
                "grid": str(arm["grid"]),
                "frames_dir": str(arm["frames_dir"]),
                "cells": int(arm["lat"].size),
                "driven_cells": int((arm["bdy"] > 0).sum()),
                "free_cells": int((arm["bdy"] == 0).sum()),
                **arm_geometry(arm),
            }
            for name, arm in arms.items()
        },
        "frames": [],
    }

    from netCDF4 import Dataset

    for valid in shared:
        row: dict[str, Any] = {"xtime": valid, "fields": {}, "absent": []}
        handles = {
            name: Dataset(str(frames[name][valid])) for name in arms
        }
        try:
            for field in (*FIELDS, *PHYSICS_FIELDS):
                if any(field not in h.variables for h in handles.values()):
                    row["absent"].append(field)
                    continue
                values: dict[str, np.ndarray] = {}
                bad = False
                for name in arms:
                    raw = np.asarray(handles[name].variables[field][0], np.float64)
                    if raw.ndim == 2 and raw.shape[0] != arms[name]["lat"].size:
                        raw = raw.T
                    if raw.shape[0] != arms[name]["lat"].size:
                        row["absent"].append(f"{field} (shape {raw.shape})")
                        bad = True
                        break
                    values[name] = raw[selection[name]]
                if bad:
                    continue
                reference = values[arguments.reference]
                row["fields"][field] = {
                    name: {
                        "patch_rms": _rms(values[name] - reference),
                        "patch_max_abs": float(
                            np.abs(values[name] - reference).max()
                        ),
                        "patch_correlation": (
                            None
                            if float(np.std(reference)) == 0.0
                            or float(np.std(values[name])) == 0.0
                            else float(
                                np.corrcoef(
                                    values[name].ravel(), reference.ravel()
                                )[0, 1]
                            )
                        ),
                        "by_distance": profile_by_distance(
                            values[name], reference, distance_km, edges
                        ),
                    }
                    for name in arms
                }
        finally:
            for handle in handles.values():
                handle.close()
        report["frames"].append(row)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    last = report["frames"][-1]
    print(f"patch {len(patch_keys)} cells, {len(shared)} shared frames")
    for field in ("theta", "w", "refl10cm", "rainnc"):
        entry = last["fields"].get(field)
        if not entry:
            continue
        parts = " ".join(
            f"{name}={entry[name]['patch_rms']:.5g}"
            for name in sorted(entry)
            if name != arguments.reference
        )
        print(f"  {last['xtime']} {field:10s} rms vs {arguments.reference}: {parts}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
