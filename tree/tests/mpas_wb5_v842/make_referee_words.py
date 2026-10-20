"""Native referee words for correct-mode finding WB5 (cluster mpas-water-budget).

The cluster's native correct-mode build (MPAS-A v8.4.2 + MMM-physics 585bcc6, native binary a75371a6...)
books WSM6's below-ground rain-out with the remap's piecewise-linear profile when the module switch
wsm6_plm_surface is .true. (mp_wsm6.F90 rain-out of nislfv_rain_plm and nislfv_rain_plm6). This script
builds that module VERBATIM from the referee's source tree with the WSM6 oracle's strict flags
(tests/wsm6_oracle/prepare.py, without coverage), next to the oracle's unchanged column bridge
(tests/wsm6_oracle/oracle.F90) and a one-line switch setter, then runs whole WSM6 columns with the switch
off (exact, as released) and on (correct) and stores every output word.

The five other sources must equal the pinned v8.4.2 authority (tests/wsm6_oracle/source_pins.json); the
referee's mp_wsm6.F90 may differ from the pinned one only by the WB5 switch (its diff is recorded).

Usage:  python make_referee_words.py <referee MPAS root> <pinned MPAS root> <fixtures dir>
"""
from __future__ import annotations

import argparse
import ctypes as ct
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE = HERE.parent / "wsm6_oracle"
FLAGS = ("-std=f2008 -fimplicit-none -O3 -ffree-line-length-none -fconvert=big-endian -ffree-form "
         "-ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations -fPIC").split()
SOURCES = ("ccpp_kind_types.F", "module_libmassv.F90", "mp_radar.F90", "mp_wsm6.F90", "mp_wsm6_effectRad.F90")
#: tests/wsm6_oracle/replay.py CONSTANTS (MPAS's compiled values for the WSM6 interface).
CONSTANTS = np.array([9.80616, 1004.5, 1846.4, 287, 461.6, 273.15, np.float32(461.6) / np.float32(287) - np.float32(1),
                      np.float32(287) / np.float32(461.6), 0, 0, 0, 1e-15, 2.85e6, 2.5e6, 3.5e5, 1.28, 1000, 4190,
                      2106, 610.78, 2.49e-6, 4.99e-6, 9.99e-6, 50e-6, 125e-6, 999e-6], np.float32)
SWITCH = """subroutine wsm6_set_plm_surface(flag) bind(C)
 use iso_c_binding
 use mp_wsm6, only: wsm6_plm_surface
 implicit none
 integer(c_int),value:: flag
 wsm6_plm_surface = (flag /= 0)
end subroutine wsm6_set_plm_surface
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    """Raining columns whose sedimentation crosses the ground in one step, plus dry controls."""
    rng = np.random.default_rng(20261014)
    rows = []
    for n in range(48):
        nz = (8, 24, 55)[n % 3]
        dt = (120.0, 300.0, 600.0, 60.0)[(n // 3) % 4]
        v = np.zeros((10, nz), np.float32)
        v[0] = rng.uniform(255, 305, nz)
        v[1] = rng.uniform(1e-4, 0.025, nz)
        v[2:7] = rng.uniform(0, 0.008, (5, nz))
        if n % 8 == 7:
            v[2:7] = 0                                  # dry control column: nothing to rain out
        v[8] = np.linspace(96000, 22000, nz, dtype=np.float32)
        v[7] = v[8] / (CONSTANTS[3] * v[0])
        v[9] = rng.uniform(80, 900, nz)
        rows.append(dict(nz=nz, dt=dt, hail=(0, 1)[n % 2], v=v))
    return rows


def build(referee: Path, pinned: Path, work: Path) -> dict:
    physics = referee / "src/core_atmosphere/physics"
    pins = json.loads((ORACLE / "source_pins.json").read_text())
    record = {}
    for name in SOURCES:
        source = next(physics.rglob(name))
        shutil.copyfile(source, work / name)
        record[name] = sha(source)
        if name != "mp_wsm6.F90" and record[name] != pins[name]:
            raise SystemExit(f"{source}: differs from the pinned v8.4.2 authority")
    pristine = next((pinned / "src/core_atmosphere/physics").rglob("mp_wsm6.F90"))
    if sha(pristine) != pins["mp_wsm6.F90"]:
        raise SystemExit(f"{pristine}: not the pinned v8.4.2 mp_wsm6.F90")
    diff = list(difflib.unified_diff(pristine.read_text().splitlines(), (work / "mp_wsm6.F90").read_text().splitlines(),
                                     "pinned/mp_wsm6.F90", "referee/mp_wsm6.F90", n=0, lineterm=""))
    changed = [line for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if not changed or any("wsm6_plm_surface" not in line and not any(token in line for token in (
            "th = (0.0-za(k))/dza(k)", "dza(k)*(qmi(k)*th + 0.5*(qpi(k)-qmi(k))*th*th)",
            "precip(i) = precip(i) + qa(k)*(0.0-za(k))", "else", "endif", "!")) and line.strip("+- ")
            for line in changed):
        raise SystemExit("the referee's mp_wsm6.F90 differs from v8.4.2 by more than the WB5 switch")
    for name in ("stubs.F90", "oracle.F90"):
        shutil.copyfile(ORACLE / name, work / name)
    (work / "wb5_switch.F90").write_text(SWITCH)
    objects = []
    for name in ("stubs.F90", *SOURCES):
        subprocess.run(["gfortran", *FLAGS, "-c", name], cwd=work, check=True)
        objects.append(Path(name).with_suffix(".o").name)
    subprocess.run(["gfortran", *FLAGS, "-shared", "oracle.F90", "wb5_switch.F90", *objects, "-o", "libwb5.so"],
                   cwd=work, check=True)
    return {"sources_sha256": record, "pinned_mp_wsm6_sha256": sha(pristine), "mp_wsm6_diff": diff}


def run(library: Path, rows):
    lib = ct.CDLL(str(library))
    fp = ct.POINTER(ct.c_float)
    lib.wsm6_oracle.argtypes = [ct.c_int, ct.c_float, ct.c_int] + [fp] * 6
    lib.wsm6_set_plm_surface.argtypes = [ct.c_int]
    out = {}
    for flag, mode in ((0, "exact"), (1, "correct")):
        lib.wsm6_set_plm_surface(flag)
        results = []
        for row in rows:
            nz = row["nz"]
            arrays = [row["v"].copy(), np.full(7, 1.25, np.float32), np.full((3, nz), 1.7e-5, np.float32),
                      np.full(nz, -35, np.float32), np.full((2, nz), .0027, np.float32)]
            c = CONSTANTS.copy()
            lib.wsm6_oracle(nz, row["dt"], row["hail"], *(a.ctypes.data_as(fp) for a in (c, *arrays)))
            results.append(arrays)
        out[mode] = results
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("referee", type=Path)
    parser.add_argument("pinned", type=Path)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    rows = cases()
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        manifest = build(args.referee, args.pinned, work)
        words = run(work / "libwb5.so", rows)
    names = ("volume", "surface", "radii", "dbz", "production")
    arrays = {}
    for n, row in enumerate(rows):
        arrays[f"case{n}_v"] = row["v"]
        for mode in ("exact", "correct"):
            for name, a in zip(names, words[mode][n]):
                arrays[f"case{n}_{mode}_{name}"] = a
    acted = [n for n in range(len(rows)) if not np.array_equal(words["exact"][n][1].view(np.uint32),
                                                                words["correct"][n][1].view(np.uint32))]
    others = [n for n in range(len(rows)) for i in (0, 2, 3, 4)
              if not np.array_equal(words["exact"][n][i].view(np.uint32), words["correct"][n][i].view(np.uint32))]
    args.fixtures.mkdir(parents=True, exist_ok=True)
    np.savez(args.fixtures / "wb5_native_columns.npz", **arrays)
    version = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    manifest.update(schema="hex.correct-mode-referee-words/v1", finding="WB5", cluster="mpas-water-budget",
                    referee_root=args.referee.resolve().name, flags=FLAGS, compiler=version, constants=CONSTANTS.tolist(),
                    cases=[dict(nz=r["nz"], dt=r["dt"], hail=r["hail"]) for r in rows],
                    surface_changed_cases=acted, non_surface_changed_cases=sorted(set(others)),
                    words_sha256=sha(args.fixtures / "wb5_native_columns.npz"))
    (args.fixtures / "wb5_native_columns.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("sources_sha256", "surface_changed_cases",
                                                "non_surface_changed_cases")}, indent=2))


if __name__ == "__main__":
    main()
