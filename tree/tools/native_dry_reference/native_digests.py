#!/usr/bin/env python3
"""SHA-256 of every native output field in hex layout, for remote comparison.

Lets identity_compare.py run on a GPU box that does not hold the native run:
it needs only the init, the native time-zero file and this JSON. The same
existing hex loader (load_mpas_initial_state) builds the 12 compared fields
from each native tracer file; a field is identical exactly when its float32
bytes hash the same. Word counts for a differing field are taken afterwards
from the saved hex arrays against the native file (compare_saved.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--scalars", default="qv")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    tree = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(tree / "src"), str(Path(__file__).resolve().parent)]
    from compare_dry_identity import SIDECAR_FIELDS, STATE_FIELDS
    from hexcore.driver import load_mpas_initial_state, load_mpas_vertical_grid
    from hexcore.mesh import Mesh
    names = tuple(n.strip() for n in a.scalars.split(",") if n.strip())
    mesh = Mesh.from_netcdf(a.init)
    grid = load_mpas_vertical_grid(a.init, mesh, config_coef_3rd_order=.25)
    files = sorted(a.native.glob("tracer.*.nc"))
    t0 = datetime.strptime(files[0].name[7:-3], "%Y-%m-%d_%H.%M.%S")
    rows = []
    for f in files:
        state, _, saved = load_mpas_initial_state(f, mesh, grid.vertical_grid, scalar_names=names,
                                                  reference_path=f, terrain_metrics=grid.terrain_metrics,
                                                  return_saved_diagnostics=True)
        t = datetime.strptime(f.name[7:-3], "%Y-%m-%d_%H.%M.%S")
        fields = {n: sha(getattr(state, n)) for n in STATE_FIELDS}
        fields.update({n: sha(getattr(saved, n)) for n in SIDECAR_FIELDS})
        species = {n: sha(np.asarray(state.scalars)[i]) for i, n in enumerate(names)}
        h = hashlib.sha256()
        with open(f, "rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                h.update(block)
        rows.append({"name": f.name, "seconds": (t - t0).total_seconds(), "fields": fields,
                     "species": species, "file_sha256": h.hexdigest()})
        print(f.name, flush=True)
    a.out.write_text(json.dumps({"schema": "gpuwm-hex.native-digests/v1", "native": str(a.native),
                                 "scalars": names, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
