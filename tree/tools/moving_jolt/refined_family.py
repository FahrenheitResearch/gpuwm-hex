"""Reduced moving refinement cap built through rw-mpas --from-centres.

Contract centres toward a cap centre by a smooth monotone angular map.
This is a synthetic 1.4:1 centre-spacing request, not a fitted 48/3 km mesh.
"""

import json, subprocess, shutil
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
from scipy.sparse import save_npz
from hexcore.moving_remap import SphericalCells, ConservativeMap, sphere_xyz

seed = Path("experiments/refined-seed.nc")
subprocess.run(
    [
        "bridges/rw_mpas_mesh",
        "--background-km",
        "1500",
        "--out",
        str(seed),
        "--cells",
        "512",
        "--fit-spacing",
        "yes",
        "--sweeps",
        "5",
        "--tolerance",
        ".1",
        "--clobber",
    ],
    check=True,
    stdout=subprocess.DEVNULL,
)
for i in range(4):
    path = Path(f"experiments/refined-centres{i}.nc")
    shutil.copyfile(seed, path)
    with Dataset(path, "r+") as ds:
        xyz = np.stack([ds[n + "Cell"][:] for n in "xyz"], axis=-1)
        radius = np.linalg.norm(xyz, axis=1)
        unit = xyz / radius[:, None]
        center = sphere_xyz(np.deg2rad(15), np.deg2rad(10.0 * i))
        cosine = np.clip(unit @ center, -1.0, 1.0)
        angle = np.arccos(cosine)
        tangent = unit - cosine[:, None] * center
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        mapped = angle * (1.0 - 0.3 * np.exp(-0.5 * (angle / 0.3) ** 2))
        new = np.cos(mapped)[:, None] * center + np.sin(mapped)[:, None] * tangent
        for j, n in enumerate("xyz"):
            ds[n + "Cell"][:] = new[:, j] * radius
    out = f"experiments/refined-{i}.nc"
    subprocess.run(
        [
            "bridges/rw_mpas_mesh",
            "--from-centres",
            str(path),
            "--out",
            out,
            "--clobber",
            "--receipt",
            f"evidence/refined-mesh{i}.json",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if i:
        w = ConservativeMap.build(
            SphericalCells.read(f"experiments/refined-{i - 1}.nc"),
            SphericalCells.read(out),
        )
        save_npz(f"experiments/refined-overlap{i}.npz", w.overlap)
        np.savez(
            f"experiments/refined-areas{i}.npz",
            source=w.source_area,
            target=w.target_area,
        )
        Path(f"evidence/refined-overlap{i}.json").write_text(
            json.dumps(w.receipt, indent=2) + "\n"
        )
