#!/usr/bin/env python3
"""Admit or refuse an init's surface columns the way the forecast door does, from the init alone.

Reads the four native fields the door partitions on (``xland``, ``xice``,
``ivgtyp``, ``landmask``), applies the door's lake fold (ivgtyp 21 becomes
open water, as the door does before it classifies), and runs
``hexcore.surface_signature`` over them, so
the admission rule can be checked on a real init in well under a minute on a
CPU, without the door's full host preparation (about 30 min at 3 km).

Prints the surface-classification receipt block as JSON, and for every column
the old "category signature" used to refuse on (a glacier on landmask 0, sea
ice on landmask 1, a glacier with sub-threshold xice) the first cell indices
with their latitude and longitude when the init carries ``latCell`` and
``lonCell``.  Exit 0 when the door would admit the init; exit 2 with the
refusal text when it would not.

Example (RRFS v2 conus3km, the init that was refused on 2026-10-05)::

    python tools/check_surface_signature.py /path/to/conus3km.init.nc

The expected result there: admitted, ``glacier_columns_on_landmask_0`` = 2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from hexcore.surface_signature import SurfaceSignatureRefusal, classify_native_surface  # noqa: E402

#: How many cells each measured category names.
_NAMED = 8


def _cell_rows(indices: np.ndarray, lat: np.ndarray | None, lon: np.ndarray | None) -> list[dict]:
    rows = []
    for index in indices[:_NAMED]:
        row = {"cell": int(index)}
        if lat is not None and lon is not None:
            row["lat_deg"] = round(float(math.degrees(lat[index])), 3)
            row["lon_deg"] = round(float(math.degrees(lon[index])), 3)
        rows.append(row)
    return rows


def check_init(init_path: Path) -> tuple[dict, str | None]:
    """Return ``(report, refusal)``; ``refusal`` is the door's message or None."""

    from netCDF4 import Dataset

    from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof

    with Dataset(init_path, "r") as dataset:
        variables = dataset.variables
        missing = [name for name in ("xland", "xice", "ivgtyp", "landmask") if name not in variables]
        if missing:
            raise SystemExit(f"{init_path}: not a native init, missing {missing}")
        xland = np.asarray(variables["xland"][0, :], dtype=np.float32)
        xice = np.asarray(variables["xice"][0, :], dtype=np.float32)
        ivgtyp = np.asarray(variables["ivgtyp"][:], dtype=np.int32)
        landmask = np.asarray(variables["landmask"][:], dtype=np.int32)
        lat = np.asarray(variables["latCell"][:], dtype=np.float64) if "latCell" in variables else None
        lon = np.asarray(variables["lonCell"][:], dtype=np.float64) if "lonCell" in variables else None
        threshold_attr = (
            float(dataset.getncattr("config_tsk_seaice_threshold"))
            if "config_tsk_seaice_threshold" in dataset.ncattrs()
            else None
        )

    threshold = float(proof.ARWEN_XICE_THRESHOLD)
    # The door folds lake columns (ivgtyp 21) into open water BEFORE it
    # classifies (run_cuda_v841_forecast.build_forecast_constructor_values:
    # ivgtyp 21 -> 17, landmask -> 0, xland -> 2.0, because the frozen
    # v8.4.1 physics has no lake model).  Without the same fold here, an
    # init with lake cells would count them as vegetated land and this
    # report would disagree with the door's receipt by exactly that count.
    lake_mask = ivgtyp == np.int32(21)
    lake_columns = int(np.count_nonzero(lake_mask))
    if lake_columns:
        ivgtyp = np.where(lake_mask, np.int32(17), ivgtyp).astype(np.int32)
        landmask = np.where(lake_mask, np.int32(0), landmask).astype(np.int32)
        xland = np.where(lake_mask, np.float32(2.0), xland).astype(np.float32)
    report: dict = {
        "init": str(init_path),
        "n_cells": int(xland.size),
        "config_tsk_seaice_threshold": threshold_attr,
        "lake_fold": {"lake_category": 21, "lake_columns": lake_columns},
    }
    try:
        counts, masks = classify_native_surface(
            xland=xland, xice=xice, ivgtyp=ivgtyp, landmask=landmask, xice_threshold=threshold
        )
    except SurfaceSignatureRefusal as refusal:
        report["admitted"] = False
        return report, str(refusal)

    report["admitted"] = True
    report["surface_classification"] = dict(counts)
    glacier = masks["glacier"]
    sea_ice = masks["sea_ice"]
    report["measured_cells"] = {
        "glacier_on_landmask_0": _cell_rows(np.flatnonzero(glacier & (landmask == 0)), lat, lon),
        "sea_ice_on_landmask_1": _cell_rows(np.flatnonzero(sea_ice & (landmask == 1)), lat, lon),
        "glacier_with_subthreshold_xice": _cell_rows(np.flatnonzero(glacier & (xice > 0.0)), lat, lon),
    }
    return report, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("init", type=Path, help="native init.nc to check")
    arguments = parser.parse_args(argv)
    report, refusal = check_init(arguments.init)
    print(json.dumps(report, indent=2))
    if refusal is not None:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
