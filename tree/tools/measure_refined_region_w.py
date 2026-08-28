"""What the refined region resolves that its own background does not.

WHY THIS EXISTS, MEASURED.  ``tools/measure_w_band.py`` reads the model's own
per-step ``vertical_velocity_abs_max``, which is a maximum over **every cell on
the globe**.  On a placed mesh that is almost entirely background: two runs on
two different placed meshes, over two different storms, from the same GFS
2026-08-12 06Z init and the same 75 km background, produced half-hour band
means agreeing to better than 0.5 per cent in all twelve windows
(``evidence/swath-first-forecast-20260826/armA.w-band.json`` against
``evidence/swath-real-cascade-20260826/w-band.json``).  That statistic settles
whether a runaway appears; it says nothing about what the fine core resolved,
because the fine core is 15 per cent of the cells and is not where the global
maximum lives.

This reads the history frames instead and splits the SAME field by where the
cell is:

* the **refined core** -- cells whose spacing is within twice the mesh's finest
  spacing, which is the same rule ``rw_mpas_convert``'s derived render window
  uses to decide what to render;
* the **background** -- every other cell of the same global mesh, at the same
  timestep, in the same run.

and, when a coarse run over the same ground is given, the same statistic on the
coarse mesh restricted to the refined core's own footprint.  That last column
is the comparison the cascade exists to make: the same storm, the same hour,
the same field, resolved two ways.

The field defaults to ``w`` and is SETTABLE, which is not cosmetic.  Measured
2026-08-27 over four placed grids on four different kinds of weather: |w| is
what an extratropical cyclone's fine core buys, and it is close to the wrong
question for an atmospheric river, whose fine grid buys moisture transport and
rainfall rate rather than ascent.  A hard-wired field reports "unimpressive"
about a grid that was doing its job in a variable the instrument refused to
read.  ``--field u10,v10`` takes a vector magnitude, so a wind speed is a
measurement here rather than a variable the history writer has to grow.

It is an analysis instrument over one field.  It plots nothing and it renders
no weather field.

Usage::

    python tools/measure_refined_region_w.py --grid FINE.grid.nc \\
        --history FINE_FRAME.nc [...] [--coarse-grid C.nc --coarse-history F.nc ...] \\
        [--field w|rainnc|u10,v10] [--out W-BY-REGION.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EARTH_RADIUS_M = 6_371_229.0


def _cell_geometry(grid: Path) -> dict[str, np.ndarray]:
    import netCDF4

    with netCDF4.Dataset(str(grid)) as dataset:
        lat = np.degrees(np.asarray(dataset.variables["latCell"][:], dtype=np.float64))
        lon = np.degrees(np.asarray(dataset.variables["lonCell"][:], dtype=np.float64))
        dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
        edges = np.asarray(dataset.variables["edgesOnCell"][:])
        counts = np.asarray(dataset.variables["nEdgesOnCell"][:])
        radius = float(getattr(dataset, "sphere_radius", 1.0) or 1.0)
    if radius < 1000.0:
        dc = dc * EARTH_RADIUS_M
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    one_based = int(edges.max()) >= dc.size
    spacing = np.zeros(lat.size, dtype=np.float64)
    for cell in range(lat.size):
        n = int(counts[cell])
        idx = np.asarray(edges[cell, :n], dtype=np.int64)
        idx = idx - 1 if one_based else idx
        idx = idx[(idx >= 0) & (idx < dc.size)]
        spacing[cell] = dc[idx].mean() if idx.size else np.nan
    return {"lat": lat, "lon": lon, "spacing_m": spacing}


def _read(dataset, name: str, source: Path) -> np.ndarray:
    if name not in dataset.variables:
        raise SystemExit(
            f"{source.name} carries no {name!r}. This instrument reads a field the "
            "model itself published; substituting another would answer a different "
            f"question under the same name. The frame carries: "
            f"{', '.join(sorted(dataset.variables))}"
        )
    return np.asarray(dataset.variables[name][:], dtype=np.float64)


def _collapse(values: np.ndarray, cells: int) -> np.ndarray:
    """One number per cell: the extremum over every other axis.

    A published field is per-cell (``t2``), per-cell-per-level (``w``, ``qv``)
    or per-cell-per-level with a leading time axis of one.  The cell axis is
    identified by LENGTH against the mesh, never by position, because the two
    orderings both occur in this history stream and guessing wrong silently
    reports the maximum over a column as the maximum over the globe.
    """

    while values.ndim > 1 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 1:
        return values
    axes = [axis for axis, size in enumerate(values.shape) if size == cells]
    if not axes:
        raise SystemExit(
            f"no axis of a {values.shape} field has the mesh's {cells} cells, so "
            "which axis is the cell axis cannot be established"
        )
    keep = axes[0]
    return values.max(axis=tuple(a for a in range(values.ndim) if a != keep))


def _frame_field(path: Path, field: str, cells: int) -> np.ndarray:
    """One value per cell for one history frame.

    ``field`` is one published name, or two comma-separated names whose
    VECTOR MAGNITUDE is taken -- ``u10,v10`` is a wind speed, and computing it
    here rather than asking the caller for a third published variable is the
    difference between a measurement and a field the history writer would have
    to grow.  A single field is taken in absolute value, because every
    question this instrument answers is about size, not sign.
    """

    import netCDF4

    names = [part.strip() for part in field.split(",") if part.strip()]
    if not names or len(names) > 2:
        raise SystemExit(
            f"--field takes one published name, or two for a vector magnitude; "
            f"got {field!r}"
        )
    with netCDF4.Dataset(str(path)) as dataset:
        parts = [_collapse(_read(dataset, name, path), cells) for name in names]
    if len(parts) == 1:
        return np.abs(parts[0])
    return np.hypot(parts[0], parts[1])


def _stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"cells": 0, "max": math.nan, "p99_9": math.nan, "mean": math.nan}
    return {
        "cells": int(finite.size),
        "max": float(finite.max()),
        "p99_9": float(np.percentile(finite, 99.9)),
        "mean": float(finite.mean()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--coarse-grid", type=Path, default=None)
    parser.add_argument("--coarse-history", type=Path, nargs="*", default=())
    parser.add_argument("--refined-multiple", type=float, default=2.0,
                        help="a cell is 'refined' at or below this multiple of the "
                             "mesh's finest spacing; 2.0 is the rule the derived "
                             "render window uses")
    parser.add_argument("--field", default="w",
                        help="the published field to split, or two "
                             "comma-separated names for a vector magnitude "
                             "('u10,v10' is a 10 m wind speed). Default 'w'. "
                             "WHY IT IS SETTABLE, MEASURED: |w| is the right "
                             "question for a cyclone and the wrong one for an "
                             "atmospheric river, where the fine grid buys "
                             "moisture transport and rainfall rather than "
                             "ascent -- a fixed field would have reported "
                             "'unimpressive' about a grid that was doing its "
                             "job in a variable this tool refused to read")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)

    fine = _cell_geometry(arguments.grid)
    finest = float(np.nanmin(fine["spacing_m"]))
    core = fine["spacing_m"] <= arguments.refined_multiple * finest
    document: dict[str, Any] = {
        "schema": "gpuwm-hex.refined-region-w/v1",
        "field": arguments.field,
        "grid": str(arguments.grid),
        "cells": int(fine["lat"].size),
        "finest_spacing_m": finest,
        "refined_multiple": arguments.refined_multiple,
        "refined_cells": int(core.sum()),
        "refined_fraction": float(core.mean()),
        "refined_bounds_deg": {
            "south": float(fine["lat"][core].min()), "north": float(fine["lat"][core].max()),
            "west": float(fine["lon"][core].min()), "east": float(fine["lon"][core].max()),
        },
        "frames": [],
    }

    bounds = document["refined_bounds_deg"]
    coarse_mask = None
    coarse_geometry = None
    if arguments.coarse_grid is not None:
        coarse_geometry = _cell_geometry(arguments.coarse_grid)
        coarse_mask = (
            (coarse_geometry["lat"] >= bounds["south"]) & (coarse_geometry["lat"] <= bounds["north"])
            & (coarse_geometry["lon"] >= bounds["west"]) & (coarse_geometry["lon"] <= bounds["east"])
        )
        document["coarse_grid"] = str(arguments.coarse_grid)
        document["coarse_cells_in_footprint"] = int(coarse_mask.sum())
        document["coarse_finest_spacing_m"] = float(np.nanmin(coarse_geometry["spacing_m"]))

    coarse_by_label = {p.name.split("cuda-history.")[-1]: p for p in arguments.coarse_history}
    cells = int(fine["lat"].size)
    coarse_cells = int(coarse_geometry["lat"].size) if coarse_geometry is not None else 0
    for path in arguments.history:
        w = _frame_field(path, arguments.field, cells)
        row: dict[str, Any] = {
            "frame": path.name,
            "refined_core": _stats(w[core]),
            "background": _stats(w[~core]),
        }
        label = path.name.split("cuda-history.")[-1]
        if coarse_mask is not None and label in coarse_by_label:
            cw = _frame_field(coarse_by_label[label], arguments.field, coarse_cells)
            row["coarse_same_footprint"] = _stats(cw[coarse_mask])
        document["frames"].append(row)

    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
