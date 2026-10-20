"""Check the bit identity promised by an unchanged-centres rebuild."""

import json, subprocess
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

subprocess.run(
    [
        "bridges/rw_mpas_mesh",
        "--from-centres",
        "experiments/grid.nc",
        "--out",
        "experiments/identity-rebuilt.nc",
        "--clobber",
        "--receipt",
        "evidence/identity-rebuild.json",
    ],
    check=True,
    stdout=subprocess.DEVNULL,
)
with (
    Dataset("experiments/grid.nc") as a,
    Dataset("experiments/identity-rebuilt.nc") as b,
):
    rows = {}
    for name in [
        "xCell",
        "yCell",
        "zCell",
        "areaCell",
        "dcEdge",
        "dvEdge",
        "latCell",
        "lonCell",
    ]:
        x = np.asarray(a[name][:])
        y = np.asarray(b[name][:])
        rows[name] = dict(
            changed=int(np.count_nonzero(x != y)),
            entries=x.size,
            max_abs_difference=float(abs(x - y).max()),
        )
Path("evidence/identity-rebuild-differences.json").write_text(
    json.dumps(rows, indent=2) + "\n"
)
