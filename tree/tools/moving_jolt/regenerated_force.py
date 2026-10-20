"""Complete resting-state remap between rw-mpas regenerated meshes."""

import os, json, argparse
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from dycore_fixture import *
import force_fixture as ff

parser = argparse.ArgumentParser()
parser.add_argument("--refined", action="store_true")
args = parser.parse_args()
m, cfg, _, _ = setup(
    "experiments/refined-0.nc" if args.refined else "experiments/grid.nc"
)
terrain = lambda m: (
    100 * (np.sin(m.arrays["latCell"]) * np.cos(m.arrays["lonCell"])) ** 2
)
s, saved, p = fixture(m, terrain(m))
tm, _, _, _ = setup(
    "experiments/refined-1.nc" if args.refined else "experiments/move1.nc"
)
_, _, tp = fixture(tm, terrain(tm))
a = np.load(
    "experiments/refined-areas1.npz" if args.refined else "experiments/areas1.npz"
)
scale = 6371229.0**2
h = ConservativeMap(
    load_npz(
        "experiments/refined-overlap1.npz"
        if args.refined
        else "experiments/overlap1.npz"
    )
    * scale,
    a["source"] * scale,
    a["target"] * scale,
    {},
).with_physical_areas(m.arrays["areaCell"], tm.arrays["areaCell"])
ff.mesh = m
ff.edges = m.arrays["cellsOnEdge"]
ff.ne = len(ff.edges)
rows = [dict(arm="source_rest", forces=ff.forces(s, p))]
ff.mesh = tm
ff.edges = tm.arrays["cellsOnEdge"]
ff.ne = len(ff.edges)
for balance in ("0", "1"):
    os.environ["GPUWM_HEX_MOVING_BALANCE"] = balance
    w = HeightConservativeMap(
        h,
        p.vertical.zgrid,
        tp.vertical.zgrid,
        p.vertical.dzw,
        tp.vertical.dzw,
        endpoint_alignment=False,
    )
    out, diag, receipt = remap_atmosphere(
        payload_of(s, saved, p), w, source_mesh=m, target_prepared=tp, config=cfg
    )
    rows.append(
        dict(
            arm="balanced" if balance == "1" else "baseline",
            forces=ff.forces(out, tp),
            receipt=receipt,
        )
    )
print(json.dumps(rows, indent=2))
Path(
    "evidence/refined-force.json" if args.refined else "evidence/regenerated-force.json"
).write_text(json.dumps(rows, indent=2) + "\n")
