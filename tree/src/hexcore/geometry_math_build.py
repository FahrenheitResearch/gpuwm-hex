"""One build recipe for the wheel and source-tree strict geometry authority.

Only standard-library imports: setup.py loads this file without importing the
package or requiring its runtime dependencies in the isolated build environment.
"""
from pathlib import Path
import hashlib
import os
import platform
import shutil
import subprocess
import tempfile

LIBRARY = "libhex_geometry_math.so"
FLAGS = ("-O2", "-fPIC", "-ffp-contract=off", "-fno-fast-math")
SOURCES = ("geometry_host_math.cpp", "cuda_host_shim.h", "glibc_trig_flt32.cuh",
           "core-math/atan2f.c")


def source_digest(source: Path) -> str:
    """Key every translation unit, included header and the shared build recipe."""
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for name in SOURCES:
        digest.update(name.encode() + b"\0" + (source / name).read_bytes())
    digest.update((platform.system() + "/" + platform.machine()).encode())
    return digest.hexdigest()


def build_geometry_math(source: Path, out: Path) -> Path:
    """Compile the exact GNU 2.43 / CORE-MATH sources; never use host substitutes."""
    compilers = [shutil.which(name) for name in ("cc", "c++")]
    if not all(compilers):
        raise RuntimeError("strict deformation geometry requires libhex_geometry_math.so; "
                           "install C and C++ compilers (cc and c++) to build the shipped "
                           "sources, or install a Linux wheel containing the library; "
                           "host libm is not a strict substitute")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([compilers[0], *FLAGS, "-c", str(source / "core-math/atan2f.c"),
                    "-o", str(out / "atan2f.o")], check=True, capture_output=True)
    library = out / LIBRARY
    subprocess.run([compilers[1], "-std=c++17", *FLAGS, "-shared", "-I", str(source),
                    str(source / "geometry_host_math.cpp"), str(out / "atan2f.o"),
                    "-o", str(library), "-lm"], check=True, capture_output=True)
    return library


def cached_geometry_math(package: Path) -> Path:
    """Build once across processes, publishing a complete source-addressed library."""
    # Strict host geometry and this ELF build are Linux-only, like the wheel.
    import fcntl

    source = package / "data/mpasphys-maths"
    try:
        key = source_digest(source)
    except FileNotFoundError as error:
        raise RuntimeError("strict deformation geometry requires libhex_geometry_math.so "
                           "or its shipped data/mpasphys-maths sources; reinstall a complete "
                           "sdist or Linux wheel; host libm is not a strict substitute") from error
    base = os.environ.get("HEXCORE_NATIVE_CPU_CACHE")
    if base is None:
        base = str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                   / "gpuwm-hex" / "native")
    root = Path(base).expanduser() / "geometry-math" / key
    target = root / LIBRARY
    root.mkdir(parents=True, exist_ok=True)
    # Each process opens its own descriptor. flock also serializes threads with
    # distinct descriptors, so no process-local lock or partial .so can escape.
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.is_file():
            try:
                with tempfile.TemporaryDirectory(prefix="build-", dir=root) as temporary:
                    library = build_geometry_math(source, Path(temporary))
                    os.replace(library, target)
            except subprocess.CalledProcessError as error:
                raise RuntimeError("cannot build strict geometry libhex_geometry_math.so; "
                                   "install working C/C++ compilers or a Linux wheel containing "
                                   "the library; host libm is not a strict substitute: "
                                   + error.stderr.decode(errors="replace")) from error
    return target
