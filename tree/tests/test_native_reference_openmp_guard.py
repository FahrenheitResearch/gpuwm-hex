"""DY1 guard on the native reference recipe: a threaded native build is no referee.

tools/native_dry_reference/check_single_threaded.sh must refuse (exit 2) every
OpenMP build the recipe can produce and pass the pinned single-threaded one.
Two holes in the first guard (verifier, 2026-10-13) are pinned here:
  * Intel's -qopenmp (the flag of the script's own ifx-o3 and ifx-valuesafe
    recipes) did not match the compile-line pattern -q?fopenmp;
  * `ldd BIN | grep -q libgomp` under pipefail fails open when grep exits early
    and ldd takes SIGPIPE.
Linux only: the recipe is a Linux build script and the checks use ldd.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "tools" / "native_dry_reference" / "check_single_threaded.sh"

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("bash") is None,
                                reason="Linux build-recipe guard (bash, ldd)")

# Shapes of real MPAS make output: the sub-make echo carries every target's OpenMP
# flag in quotes whether or not OPENMP=true; compile lines carry " -c ".
_ECHO = '( cd src; make all "FC = mpif90" "FFLAGS_OMP = -fopenmp" "CFLAGS_OMP = -fopenmp" "OPENMP = " )'
_GNU = "mpif90 -D_MPI -DSINGLE_PRECISION -O3 -ffp-contract=off -ffree-form {omp}-c mpas_atm_time_integration.F90"
_IFX = "mpiifort -D_MPI -O3 -convert big_endian -free -align array64byte {omp}-c mpas_atm_time_integration.F90"


def _run(tmp_path, log_lines, binary=None, ldd=None):
    log = tmp_path / "build.log"
    log.write_text("\n".join(log_lines) + "\n")
    if binary is None:
        binary = tmp_path / "atmosphere_model"
        binary.write_bytes(b"\x7fELF single-threaded stand-in\n")
    env = dict(os.environ)
    if ldd is not None:
        bindir = tmp_path / "bin"
        bindir.mkdir(exist_ok=True)
        fake = bindir / "ldd"
        fake.write_text("#!/usr/bin/env bash\n" + ldd)
        fake.chmod(0o755)
        env["PATH"] = f"{bindir}:{env['PATH']}"
    return subprocess.run(["bash", str(GUARD), str(log), str(binary)], env=env,
                          capture_output=True, text=True)


def test_single_threaded_gnu_build_passes(tmp_path):
    r = _run(tmp_path, [_ECHO, _GNU.format(omp="")], ldd="echo '\tlibmpi.so.12 => /opt/x'\n")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("OK single-threaded")


@pytest.mark.parametrize("line", [_GNU.format(omp="-fopenmp "), _IFX.format(omp="-qopenmp "),
                                  _IFX.format(omp="-fiopenmp "), _IFX.format(omp="-openmp ")],
                         ids=["gnu-fopenmp", "ifx-qopenmp", "ifx-fiopenmp", "intel-openmp"])
def test_openmp_compile_line_is_refused(tmp_path, line):
    r = _run(tmp_path, [_ECHO, line], ldd="echo '\tlibmpi.so.12 => /opt/x'\n")
    assert r.returncode == 2 and "DY1" in r.stderr, (r.returncode, r.stderr)


def test_openmp_simd_alone_is_not_threading(tmp_path):
    r = _run(tmp_path, [_IFX.format(omp="-qopenmp-simd ")], ldd="echo '\tlibmpi.so.12 => /opt/x'\n")
    assert r.returncode == 0, r.stderr


def test_linked_runtime_is_refused_even_when_ldd_would_take_sigpipe(tmp_path):
    """libgomp on the first line, then enough output that an early `grep -q`
    exit SIGPIPEs ldd: the old `ldd | grep -q` test came out false here."""
    ldd = ("echo '\tlibgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1'\n"
           "for i in $(seq 1 200000); do echo \"\tlibpad$i.so => /nowhere\"; done\n")
    r = _run(tmp_path, [_ECHO, _GNU.format(omp="")], ldd=ldd)
    assert r.returncode == 2 and "libgomp" in r.stderr, (r.returncode, r.stderr)


def test_static_runtime_entry_points_are_refused(tmp_path):
    binary = tmp_path / "atmosphere_model"
    binary.write_bytes(b"\x7fELF\x00\x01GOMP_parallel\x00__kmpc_fork_call\x00")
    r = _run(tmp_path, [_ECHO, _GNU.format(omp="")], binary=binary,
             ldd="echo '\tnot a dynamic executable'; exit 1\n")
    assert r.returncode == 2 and "entry point" in r.stderr, (r.returncode, r.stderr)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
@pytest.mark.parametrize("openmp", [False, True])
def test_real_binaries(tmp_path, openmp):
    src = tmp_path / "p.c"
    src.write_text("#include <stdio.h>\nint main(void){int s=0;\n#pragma omp parallel for reduction(+:s)\n"
                   "for(int i=0;i<8;i++) s+=i;\nprintf(\"%d\\n\",s);return 0;}\n")
    exe = tmp_path / "atmosphere_model"
    flags = ["-fopenmp"] if openmp else []
    subprocess.run(["gcc", "-O1", *flags, str(src), "-o", str(exe)], check=True)
    r = _run(tmp_path, [_ECHO, _GNU.format(omp="")], binary=exe)
    assert r.returncode == (2 if openmp else 0), (r.returncode, r.stderr)
