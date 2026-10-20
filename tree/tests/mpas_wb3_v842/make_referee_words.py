"""Native referee words for correct-mode finding WB3 (cluster mpas-water-budget).

The cluster's native correct-mode build (MPAS-A v8.4.2 f34984b2 plus the gated water-budget patch,
branch correct/mpas-water-budget 15a5ce3; atmosphere_model sha256 a75371a6...) computes precipitable
water in precip_to_MPAS with a dry_density switch.  This script copies that loop VERBATIM out of the
referee's own source file (it refuses a file whose text is not the expected loop), compiles it with the
pinned strict GNU recipe's flags (the referee's build.log), runs it on the input columns in
fixtures/wb3_inputs.bin, and writes precipw for dry_density = .false. (exact, as released) and .true.
(correct) as big-endian binary32 words, plus a manifest naming the source sha256, flags and compiler.

Usage:  python make_referee_words.py <referee MPAS root> <fixtures dir> [--make-inputs]

The GPU test tests/mpas_wb3_v842/test_mpasphys_wb3_kernels_gpu_v842.py holds every hex-cuda
microphysics precip kernel to these words in both modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

NZ, NC = 24, 64
#: The referee build's FFLAGS_OPT (build.log of v8.4.2-gnu-strict-atmosphere-correct-mode-v3).
FLAGS = ("-O3", "-ffp-contract=off", "-fno-tree-vectorize", "-fno-unsafe-math-optimizations",
         "-ffree-line-length-none", "-fconvert=big-endian", "-ffree-form")
SOURCE = "src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F"
EXPECTED = ("do k = kts,kte", "if(dry_density) then", "rho_a = rho_p(i,k,j)", "else",
            "rho_a = rho_p(i,k,j) / (1._RKIND + qv_p(i,k,j))", "endif",
            "precipw(i) = precipw(i) + qv_p(i,k,j) * rho_a * dz_p(i,k,j)", "enddo")


def make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columns covering moist and dry air, a vapour-free column and tiny vapour, all binary32."""
    rng = np.random.default_rng(20261014)
    qv = rng.uniform(0.0, 0.025, (NZ, NC)).astype(np.float32)
    qv *= np.exp(-np.arange(NZ, dtype=np.float32) / np.float32(6.0))[:, None].astype(np.float32)
    qv[:, 0] = 0.0                                   # vapour-free column
    qv[:, 1] = np.float32(1.0e-12)                   # tiny vapour
    qv[:, 2] = np.float32(0.03)                      # saturated tropics
    rho = (np.float32(1.25) * np.exp(-np.arange(NZ, dtype=np.float32) / np.float32(9.0))[:, None]
           * rng.uniform(0.9, 1.1, (1, NC))).astype(np.float32)
    dz = rng.uniform(40.0, 1200.0, (NZ, NC)).astype(np.float32)
    return qv, rho, dz


def extract_loop(path: Path) -> tuple[list[str], int]:
    lines = path.read_text().splitlines()
    start = next(n for n, line in enumerate(lines) if line.strip() == "!precipitable water:") + 1
    block = lines[start:start + len(EXPECTED)]
    if [line.strip() for line in block] != list(EXPECTED):
        raise SystemExit(f"{path}: the precipitable-water loop is not the referee's dry_density loop")
    return block, start + 1


PROGRAM = """program wb3_referee
 implicit none
 integer,parameter:: RKIND = selected_real_kind(6)
 integer,parameter:: nc = {nc}, nz = {nz}
 integer:: i,j,k,its,ite,jts,jte,kts,kte,mode
 logical:: dry_density
 real(kind=RKIND):: rho_a
 real(kind=RKIND):: qv_p(nc,nz,1),rho_p(nc,nz,1),dz_p(nc,nz,1),precipw(nc)
 open(10,file='wb3_inputs.bin',access='stream',form='unformatted',status='old')
 read(10) qv_p,rho_p,dz_p
 close(10)
 its = 1; ite = nc; jts = 1; jte = 1; kts = 1; kte = nz
 open(11,file='wb3_native_words.bin',access='stream',form='unformatted',status='replace')
 do mode = 0,1
 dry_density = (mode == 1)
 do i = its,ite
    precipw(i) = 0._RKIND
 enddo
 do j = jts,jte
 do i = its,ite
{loop}
 enddo
 enddo
 write(11) precipw
 enddo
 close(11)
end program wb3_referee
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("referee", type=Path, help="root of the referee's MPAS source tree")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--make-inputs", action="store_true")
    args = parser.parse_args()
    source = args.referee / SOURCE
    loop, first = extract_loop(source)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    inputs = args.fixtures / "wb3_inputs.bin"
    if args.make_inputs:
        qv, rho, dz = make_inputs()
        inputs.write_bytes(b"".join(a.astype(">f4").tobytes() for a in (qv, rho, dz)))
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        (work / "wb3_referee.f90").write_text(PROGRAM.format(nc=NC, nz=NZ, loop="\n".join(loop)))
        (work / "wb3_inputs.bin").write_bytes(inputs.read_bytes())
        subprocess.run(["gfortran", *FLAGS, "wb3_referee.f90", "-o", "wb3_referee"], cwd=work, check=True)
        subprocess.run(["./wb3_referee"], cwd=work, check=True)
        words = (work / "wb3_native_words.bin").read_bytes()
    (args.fixtures / "wb3_native_words.bin").write_bytes(words)
    version = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    manifest = {
        "schema": "hex.correct-mode-referee-words/v1", "finding": "WB3", "cluster": "mpas-water-budget",
        "referee_source": f"{args.referee.resolve().name}/{SOURCE}", "referee_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "loop_first_line": first, "loop": [line.strip() for line in loop],
        "flags": list(FLAGS), "compiler": version, "nz": NZ, "nc": NC,
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "words_sha256": hashlib.sha256(words).hexdigest(),
        "layout": "big-endian binary32; inputs qv,rho,dz each [level][cell]; words precipw[cell] exact then correct",
    }
    (args.fixtures / "wb3_native_words.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
