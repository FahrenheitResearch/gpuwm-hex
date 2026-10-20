"""Deterministic synthetic inputs for pristine native MPAS; no production physics."""
from pathlib import Path
import itertools
import json
import subprocess
import numpy as np


def sw_columns(directory, variations=4):
    directory = Path(directory)
    nz = 12
    cases = []
    for overlap, flags, ozone, variant in itertools.product(
        range(1, 6), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), (0, 2), range(variations)):
        cases.append((overlap, flags, ozone, 12., 1, variant, 0))
    cases += [(2, (0, 0, 0), 2, 0., 1, 0, 0), (2, (0, 0, 0), 2, 12., 0, 0, 0),
              (0, (1, 1, 1), 2, 12., 1, 0, 0)]
    cases += [(o, (1, 1, 1), 2, 12., 1, 0, a) for o, a in itertools.product(range(6), (1, 2, 3))]
    with (directory / "columns.txt").open("w") as f:
        f.write(f"{len(cases)} {nz}\n")
        for overlap, flags, ozone, gmt, icloud, variant, aerosol in cases:
            f.write(" ".join(map(str, (icloud, overlap, variant % 2, ozone, *flags, aerosol, 93))) + "\n")
            f.write(f"0 {gmt} 30 0.01745329238474369 0.09 1367\n")
            surface = ([35., -5., .18, 285., 1., .1, 23.],
                       [-65., 17., .8, 260., 2., .9, 700.],
                       [10., -10., 0., 300., 2., 0., 0.],
                       [50., 170., 1., 270., 1., 1., 100.])[variant % 4]
            f.write(" ".join(map(str, surface)) + "\n")
            interfaces = np.geomspace(100021.125 + variant * 419, 720.5 + variant * 170, nz + 1).astype(np.float32)
            for k in range(nz):
                p = (interfaces[k] + interfaces[k + 1]) * np.float32(.5)
                t = np.float32(291. - k * 8.25 + variant * 2)
                cf = np.float32((0., .7, .5, 1., 1.e-21, .33)[(k + variant) % 6])
                row = (p, t, 500. + k * 111.25, (p / 1.e5) ** (287. / 1004.5),
                       max(1.e-12, .012 * float(p / 1.e5) ** 2), 1.e-5 * (k % 4),
                       1.e-5 * ((k + 1) % 3), 2.e-5 * (k % 3), cf,
                       0. if k % 2 else 12.e-6, 0. if k % 3 else 38.e-6,
                       0. if k % 2 else 180.e-6)
                f.write(" ".join(f"{np.float32(v):.9e}" for v in row) + "\n")
            for k, p in enumerate(interfaces):
                f.write(f"{p:.9e} {np.float32(295. - k * 8.25 + variant * 2):.9e}\n")
    return {"cases": len(cases), "levels": nz}


def aerosol_columns(directory):
    directory = Path(directory)
    nc, nz = 32, 12
    col = np.arange(nc)[:, None]
    lev = np.arange(nz)[None, :]
    p = np.broadcast_to(98000. * np.exp(-lev / 3.2), (nc, nz)).astype(np.float32).copy()
    t = np.broadcast_to(295. - lev * 9. + (col % 4) * 2, (nc, nz)).astype(np.float32).copy()
    dz = np.broadcast_to(123.5 + lev * 205.25 + col * .75, (nc, nz)).astype(np.float32).copy()
    # Explicitly place RH interpolation at nodes, below/above nodes and clamps.
    humidity = np.array([-10., 0., 10., 49.999, 50., 60., 70., 80., 85., 90., 95., 98., 99., 120.], np.float32)
    rh = humidity[(np.arange(nc)[:, None] + np.arange(nz)[None, :]) % humidity.size]
    es = 6.112 * np.exp((17.6 * (t.astype(np.float64) - 273.15)) / (t.astype(np.float64) - 273.15 + 243.5))
    e = rh * es / 100
    qv = (.62197 * e / (.01 * p - e)).astype(np.float32)
    nwfa = np.broadcast_to(1.e8 * (1 + col % 8), (nc, nz)).astype(np.float32).copy()
    nifa = np.broadcast_to(2.e5 * (1 + col % 3), (nc, nz)).astype(np.float32).copy()
    nwfa[0] = -10.; nwfa[1] = 1.e20; nifa[2] = -1.; nifa[3] = 1.e20
    xland = np.where(np.arange(nc) % 2, 2., 1.).astype(np.float32)
    ht = (np.arange(nc) * 121.25).astype(np.float32)
    arrays = dict(p=p, t=t, qv=qv, dz=dz, nwfa=nwfa, nifa=nifa, xland=xland, ht=ht)
    with (directory / "aer3_mpas_in.bin").open("wb") as f:
        f.write(np.asarray((nc, nz), dtype=">i4").tobytes())
        for a in arrays.values():
            f.write(a.astype(">f4").tobytes(order="F"))
    np.savez(directory / "aerosol_inputs.npz", **arrays)
    return nc, nz


def run_native_aerosols(directory):
    directory = Path(directory).resolve()
    aerosol_columns(directory)
    subprocess.run(["./oracle_aer3_mpas"], cwd=directory, check=True)
    rows = []
    index = 0
    # All scalar/gridded/spectral options and all three table types.
    for has3d, typ in itertools.product((0, 1), (1, 2, 3)):
        for ang, ssa, asy in itertools.product((1, 2, 3), repeat=3):
            if has3d and ang != 3:
                continue  # Native reads undefined aod550spc3d in these combinations.
            for aod in (1, 2):
                for exponent in ((1., 1.3) if ang == 1 else (1.3,)):
                    control = (has3d, aod, ang, ssa, asy, typ, .12, exponent, .85, .9)
                    (directory / "aerosol_control.txt").write_text(" ".join(map(str, control)) + "\n")
                    subprocess.run(["./oracle_aerosol_options"], cwd=directory, check=True, stdout=subprocess.DEVNULL)
                    output = directory / f"aerosol-{index:04}.bin"
                    (directory / "aerosol_options_out.bin").rename(output)
                    rows.append({"control": control, "output": output.name})
                    index += 1
    rows = aerosol_exponent_boundaries(directory, rows)
    (directory / "aerosol_cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def aerosol_exponent_boundaries(directory, rows):
    """MPAS aerosol module:158-176, warnings, clamps and near-one switch."""
    directory = Path(directory).resolve()
    for exponent in (-.5, 3., .999, 1.001):
        control = (0, 1, 1, 3, 3, 2, .12, exponent, .85, .9)
        if any(tuple(row["control"]) == control for row in rows):
            continue
        (directory / "aerosol_control.txt").write_text(" ".join(map(str, control)) + "\n")
        subprocess.run(["./oracle_aerosol_options"], cwd=directory, check=True)
        output = directory / f"aerosol-{len(rows):04}.bin"
        (directory / "aerosol_options_out.bin").rename(output)
        rows.append({"control": control, "output": output.name})
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    args = p.parse_args()
    print(sw_columns(args.directory))
    print({"aerosol_cases": len(run_native_aerosols(args.directory))})
