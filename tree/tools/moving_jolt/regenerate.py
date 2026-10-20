"""Local centre displacements, with all derived geometry rebuilt by rw-mpas."""

import shutil, subprocess, json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from hexcore.moving_remap import SphericalCells, ConservativeMap
from dycore_fixture import setup
from scipy.sparse import save_npz

base = Path("experiments/grid.nc")
receipts = []
for i in range(1, 4):
    centres = Path(f"experiments/centres{i}.nc")
    output = Path(f"experiments/move{i}.nc")
    shutil.copyfile(base, centres)
    with Dataset(centres, "r+") as ds:
        xyz = np.stack([ds[n + "Cell"][:] for n in "xyz"], axis=-1)
        radius = np.linalg.norm(xyz, axis=1)
        unit = xyz / radius[:, None]
        taper = np.maximum(0, (unit[:, 0] - 0.5) / 0.5) ** 2
        angle = 0.025 * i * taper
        x = unit[:, 0] * np.cos(angle) - unit[:, 1] * np.sin(angle)
        y = unit[:, 0] * np.sin(angle) + unit[:, 1] * np.cos(angle)
        unit[:, 0] = x
        unit[:, 1] = y
        for j, n in enumerate("xyz"):
            ds[n + "Cell"][:] = unit[:, j] * radius
    subprocess.run(
        [
            "bridges/rw_mpas_mesh",
            "--from-centres",
            str(centres),
            "--out",
            str(output),
            "--clobber",
            "--receipt",
            f"evidence/regenerate{i}.json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    source = SphericalCells.read(base if i == 1 else f"experiments/move{i - 1}.nc")
    target = SphericalCells.read(output)
    w = ConservativeMap.build(source, target)
    # build() already uses physical areas and common radius.
    save_npz(f"experiments/overlap{i}.npz", w.overlap)
    np.savez(f"experiments/areas{i}.npz", source=w.source_area, target=w.target_area)
    receipts.append(dict(move=i, receipt=w.receipt))
Path("evidence/regenerated-overlaps.json").write_text(
    json.dumps(receipts, indent=2) + "\n"
)
