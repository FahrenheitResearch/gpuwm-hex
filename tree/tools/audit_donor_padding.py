#!/usr/bin/env python
"""Donor-readability audit for the #333 padding defect.

A graded global mesh is the LAM donor, and an outside reader of a CULLED
donor walks ``edgesOnCell``/``cellsOnCell`` rows to their declared padding.
Native ``init_atmosphere`` statics pad those rows with 0 (one-based files),
so a reader that overruns ``nEdgesOnCell`` gathers a sentinel; a file that
pads with a REAL index hands that reader a real edge to gather into a
pentagon.  This tool measures which convention a file actually carries --
it changes nothing and fixes nothing (task #333 owns the fix).

MEASURED 2026-08-25 (this lane), for the record the tool exists to extend:

| file | edgesOnCell padding | cellsOnCell | verticesOnCell |
| --- | --- | --- | --- |
| published `x1.40962` GRID | real indices (163,860) | real indices | zeros |
| published `x1.40962` STATIC (native-made) | zeros (163,860) | zeros | zeros |
| registered `v15.150.38857` GRID (`rw_mpas_mesh`) | real indices (155,440) | real indices | zeros |
| `v15.150.38857` unified STATIC (`rw_mpas_static`) | real indices (155,440) | real indices | zeros |

So the GRID convention (real-index padding) is the published family's own
and our grid writer matches it; the divergence is in STATICS: native-made
statics pad zeros, `rw_mpas_static` carries the grid convention through.
A donor cull that reads the static is the exposed surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(path: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(str(path)) as ds:
        ne_oc = np.asarray(ds.variables["nEdgesOnCell"][:], dtype=np.int64)
        out: dict = {"file": str(path), "nCells": int(ne_oc.size), "fields": {}}
        for name in ("edgesOnCell", "cellsOnCell", "verticesOnCell"):
            var = ds.variables.get(name)
            if var is None:
                out["fields"][name] = None
                continue
            arr = np.asarray(var[:], dtype=np.int64)
            pad_mask = np.arange(arr.shape[1])[None, :] >= ne_oc[:, None]
            pad = arr[pad_mask]
            out["fields"][name] = {
                "pad_slots": int(pad.size),
                "zeros": int((pad == 0).sum()),
                "real_indices": int((pad >= 1).sum()),
                "negatives": int((pad < 0).sum()),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, help="also write the readings here")
    args = parser.parse_args()
    rows = []
    for path in args.files:
        row = audit(path)
        rows.append(row)
        print(path)
        for name, reading in row["fields"].items():
            if reading is None:
                print(f"  {name}: absent")
                continue
            verdict = (
                "zeros (native static convention)"
                if reading["real_indices"] == 0 and reading["negatives"] == 0
                else "REAL INDICES (published grid convention; #333 surface on a static)"
                if reading["zeros"] == 0
                else "MIXED"
            )
            print(f"  {name}: {reading} -> {verdict}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
