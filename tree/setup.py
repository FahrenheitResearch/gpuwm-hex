"""Bind the wheel's regional kernel set once, while building the wheel."""
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py

try:  # setuptools >= 70.1 vendors bdist_wheel; older trees use the wheel package
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel

HOSTPREP = "hex_contract_hostprep" + (".exe" if sys.platform == "win32" else "")
_EXE = ".exe" if sys.platform == "win32" else ""
# The MPAS physics slots' table decoders (rust/physics-tables/Cargo.toml says
# what breaks without them); hexcore.mpasphys.resources_build binds these names.
PHYSICS_TABLES = ({"win32": "lookup_decode.dll", "darwin": "liblookup_decode.dylib"}.get(
                      sys.platform, "liblookup_decode.so"),
                  "table_decode" + _EXE, "rw_mpas_rrtmg_lw_tables" + _EXE, "rw_mpas_rrtmg_sw_tables" + _EXE)
# The MPAS route's Registry pool reader (rust/physics-pool/Cargo.toml says what
# breaks without it); hexcore.mpasphys.resources_build binds this name.
POOL_READER = "rw_mpas_physics_pool" + _EXE
OBS_SCORER = "hex_obs_scorecard" + (".exe" if sys.platform == "win32" else "")
CHEM_SOURCES = "rw_hex_chem_sources" + (".exe" if sys.platform == "win32" else "")
# The GSL drag's orographic statistics for any mesh (hexcore.ugwp_oro binds this name).
UGWP_ORO = "rw_hex_ugwp_oro" + (".exe" if sys.platform == "win32" else "")
S2S_ENERGY = "rw_s2s_energy" + (".exe" if sys.platform == "win32" else "")
NETCDF_COPY = "gpuwm-hex-netcdf-copy" + (".exe" if sys.platform == "win32" else "")
# The history writer's parallel deflate (hexcore.history_deflate binds this name).
HISTORY_DEFLATE = "libhex_history_deflate.so"
GEOMETRY_MATH = "libhex_geometry_math.so"
SURFACE_READER = {"win32": "rw_surface.dll", "darwin": "librw_surface.dylib"}.get(
    sys.platform, "librw_surface.so")
# The three engine bridges this wheel bundles because no released engine carries
# the versions this release needs (src/hexcore/data/engine-bridges.json pins
# their bytes and their gpuwm source; tools/build_engine_bridges.sh builds them).
ENGINE_BRIDGES = ("rw_mpas_geometry" + _EXE, "rw_mpas_stream" + _EXE, "rw_mpas_static" + _EXE,
                  # The RRFS fork dycore's three Rust programs (hexcore.core_rrfs, damping_rrfs,
                  # deformation_rrfs): without them every RRFS forecast from an install refused
                  # ("Rust bridge is not installed"; RRFS-MPAS from a pip install, 2026-10-15).
                  "rw_mpas_rrfs_connectivity" + _EXE, "rw_mpas_rrfs_damping" + _EXE,
                  "rw_mpas_rrfs_deformation" + _EXE,
                  # The case-9 boundary producer that matches native init_atmosphere case 9: no released
                  # gpuwm carries it, so a plain install could not build native-identical boundary files
                  # (2.9.0's refuses a grid with no model_name and was never proven against case 9).
                  "rw_mpas_lbc" + _EXE,
                  # The IGRA2 radiosonde door gpuwm-hex score reads a past case's sondes through
                  # (hexcore.observation_scorecard): the published gpuwm 2.9.0 wheel does not install it,
                  # so every case NOMADS no longer serves scored no sondes (daily rehearsal HX7, 2026-10-15).
                  "rw_igra2" + _EXE,
                  # The mesh generator and region culler (hexcore.engines MESH) with its triangulation and
                  # derive on every core: the 2.9.0 bridge rebuilt the whole Delaunay on ONE thread about
                  # 1,000 times per graded mesh (123 s on the 1 km-core spec, now 11 s; 2026-10-16: no
                  # single-threaded stages). hexcore.engines reads hexcore/libexec before gpuwm's bridges.
                  "rw_mpas_mesh" + _EXE)
# Built from crate rw-obs in a cargo call of their own, so the rw-mpas programs keep their bytes.
OBS_BRIDGES = ("rw_igra2" + _EXE,)
RRFS_BRIDGES = ("rw_mpas_rrfs_connectivity", "rw_mpas_rrfs_damping", "rw_mpas_rrfs_deformation")
# What marks a gpuwm source whose rw_mpas_static reads a binary64 grid at
# binary64 (gpuwm 46c8a71513e, Mesh::read_as); older sources build the static
# builder that broke the point door (ARWEN-VS-HEX-TIMING-2026-10-15).
STATIC_BUILDER_MARK = b"fn read_as("
# What marks a gpuwm source carrying the native case-9 LBC port (gpuwm e72f3809).
LBC_CASE9_MARK = b"fn read_lbc_met_file("
# Explicit source builds use these recipes. The default MPAS route fetches
# pinned native framework assets through hexcore.mpasphys.resources_fetch.
NATIVE_BACKEND = ("build.py", "generate.py", "framework.py", "tables.py", "noahmp_io.py", "runtime.f90",
                  "licence_stub_noahmp.F90")
# The RRFS fork's library builder (pin set rrfs), shipped the same way: without it no install could build
# libmpasphys_native_rrfs.so, so no pip user could run RRFS-MPAS (2026-10-15).
NATIVE_BACKEND_RRFS = ("build.py", "generate.py", "framework.py", "sources.py", "tables.py", "noahmp_io.py",
                       "runtime.f90", "licence_stub_noahmp.F90")


def engine_source() -> Path | None:
    """A gpuwm source checkout whose Rust workspace can build the scorer, or None.

    $GPUWM_HEX_ENGINE_SOURCE names one explicitly; otherwise an importable
    gpuwm is used only when it is itself a source checkout (an installed
    wheel's crate_dir points at a tools/rustwx that does not exist).
    """

    import os

    candidates = []
    if os.environ.get("GPUWM_HEX_ENGINE_SOURCE"):
        named = Path(os.environ["GPUWM_HEX_ENGINE_SOURCE"])
        if not (named / "tools/rustwx/vendor/netcrust/Cargo.toml").is_file():
            # Asked for explicitly: a wrong path is an error, not a quiet skip.
            raise SystemExit(f"GPUWM_HEX_ENGINE_SOURCE={named} is not a gpuwm source checkout "
                             "(no tools/rustwx/vendor/netcrust/Cargo.toml)")
        candidates.append(named)
    else:
        try:
            from gpuwm.rustwx import crate_dir

            candidates.append(crate_dir().parents[1])
        except Exception:  # no gpuwm in this build environment
            pass
    return next((engine for engine in candidates
                 if (engine / "tools/rustwx/vendor/netcrust/Cargo.toml").is_file()), None)


class BuildPy(build_py):

    @staticmethod
    def netcdf_environment(root: Path, *, full: bool = False):
        """Use the pinned static format-specific archive in Linux wheel payloads."""
        import os
        env = os.environ.copy()
        if not sys.platform.startswith("linux"):
            return env
        kind = "NETCDF4" if full else "NETCDF_CLASSIC"
        suffix = "netcdf4" if full else "netcdf-classic"
        prefix = Path(env.get(f"GPUWM_HEX_{kind}_PREFIX", str(root / "rust/netcdf-copy/target" / suffix))).resolve()
        if not (prefix / "lib/libnetcdf.a").is_file():
            script = "build_netcdf4_static.sh" if full else "build_netcdf_classic_static.sh"
            subprocess.run(["bash", str(root / "tools" / script), str(prefix)], check=True, env=env)
        env[f"GPUWM_HEX_{kind}_PREFIX"] = str(prefix)
        # An explicit link-search argument also invalidates stale cargo outputs
        # built against the host shared library. LIBRARY_PATH alone does not.
        flags = env.get("CARGO_ENCODED_RUSTFLAGS")
        if flags is None:
            import shlex
            flags = "\x1f".join(shlex.split(env.get("RUSTFLAGS", "")))
        env["CARGO_ENCODED_RUSTFLAGS"] = flags + ("\x1f" if flags else "") + "-Lnative=" + str(prefix / "lib")
        return env
    def run(self):
        super().run()
        root = Path(__file__).parent
        module = ast.parse((root / "src/hexcore/cuda_backend/regional_admission.py").read_text())
        assignment = next(node for node in module.body if
                          (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                           and node.target.id == "REGIONAL_KERNEL_SOURCES") or
                          (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                           and target.id == "REGIONAL_KERNEL_SOURCES" for target in node.targets)))
        digest = hashlib.sha256()
        for name in ast.literal_eval(assignment.value):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256((root / name).read_bytes()).digest())
        path = Path(self.build_lib) / "hexcore/kernel-set.json"
        path.write_text(json.dumps({"schema": "gpuwm-hex.kernel-set/v1", "kernel_set_sha256": digest.hexdigest()},
                                   sort_keys=True) + "\n")
        # The default deck needs this exact native reference engine after install.
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(root / "rust/host-prep/Cargo.toml"), "--target-dir",
                        str(root / "rust/host-prep/target")], check=True)
        destination = Path(self.build_lib) / "hexcore/libexec" / HOSTPREP
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "rust/host-prep/target/release" / HOSTPREP, destination)
        # The MPAS physics suites' table decoders, for `gpuwm-hex physics-resources`.
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(root / "rust/physics-tables/Cargo.toml"), "--target-dir",
                        str(root / "rust/physics-tables/target")], check=True)
        for name in PHYSICS_TABLES:
            shutil.copy2(root / "rust/physics-tables/target/release" / name, destination.parent / name)
        # The MPAS route's Registry pool reader.  It links netCDF-C; a build
        # host without the development files still installs everything else,
        # and the MPAS route then refuses by name unless --native-pool-reader
        # names one (hexcore.mpasphys.resources_build.pool_reader).
        reader = self.build_pool_reader(root)
        if reader is not None:
            shutil.copy2(reader, destination.parent / POOL_READER)
        scorer = root / "src/hexcore/libexec" / OBS_SCORER
        if not scorer.is_file():
            engine = engine_source()
            if engine is None:
                # THE BREAKAGE THIS PREVENTS: the scorer links the engine's
                # Rust readers (netcrust, rw-obs, rustwx-io), which exist only
                # in a gpuwm SOURCE checkout.  An installed gpuwm wheel has no
                # tools/rustwx, and an isolated build has no gpuwm at all, so
                # demanding the scorer made `pip install gpuwm-hex` from the
                # sdist fail outright on every host without the manylinux
                # wheel (glibc < 2.35, Windows).  Such a build now installs
                # everything else; `gpuwm-hex score` and the forecast's
                # automatic scorecard refuse by name (hexcore.observation_
                # scorecard.binary) and `gpuwm-hex doctor` reports the gap.
                print("gpuwm-hex: hex_obs_scorecard NOT BUILT: no gpuwm source checkout "
                      "(set GPUWM_HEX_ENGINE_SOURCE to one, with cargo on PATH, to build it); "
                      "this install's `gpuwm-hex score` will refuse by name", file=sys.stderr)
            else:
                subprocess.run([sys.executable, str(root / "tools/build_obs_scorecard.py"),
                                "--engine", str(engine), "--out", str(root / "rust/obs-scorecard/build"),
                                "--install"], check=True)
        if scorer.is_file():
            shutil.copy2(scorer, destination.parent / OBS_SCORER)
        energy = s2s_energy_bridge(root)
        if energy is not None:
            shutil.copy2(energy, destination.parent / S2S_ENERGY)
        # THE BREAKAGE THIS PREVENTS (verify-final-hex item 4, 2026-10-12):
        # the vertical door's CDF-5 template copier was looked up only beside
        # a source checkout, so no pip install had it.  Every install then
        # copied the static template as-is (about an hour of single-core
        # header rewrites on a 1.5 M-cell mesh), and on a mesh whose zb/zb3
        # exceed the CDF-2 variable limit the door refused with "Build
        # rust/netcdf-copy", which no pip user can do.  The copier links
        # libnetcdf; a build host without its development files still installs
        # everything else, and the door says what it costs (vertical_spec).
        # Chemistry source preparation is required by the chemistry public door.
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(root / "rust/chem-sources/Cargo.toml"), "--target-dir",
                        str(root / "rust/chem-sources/target")], check=True)
        shutil.copy2(root / "rust/chem-sources/target/release" / ("rw_hex_chem_sources" + _EXE),
                     destination.parent / ("rw_hex_chem_sources" + _EXE))
        # THE BREAKAGE THIS PREVENTS (2026-10-24): the RRFS-shape suite's GSL drag reads
        # <mesh>.ugwp_oro_data.nc, which RRFS publishes for its own meshes only; without
        # this generator every other mesh refused the suite before its first step.
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(root / "rust/ugwp-oro/Cargo.toml"), "--target-dir",
                        str(root / "rust/ugwp-oro/target")], check=True)
        shutil.copy2(root / "rust/ugwp-oro/target/release" / UGWP_ORO, destination.parent / UGWP_ORO)
        copier = self.build_netcdf_copy(root)
        if copier is not None:
            shutil.copy2(copier, destination.parent / NETCDF_COPY)
        # THE BREAKAGE THIS PREVENTS (B4, 2026-10-16): without the library in
        # the wheel every install compressed history frames on one thread (a
        # 15 km frame: 143 s against 53 s on 16 CPUs, measured, same bytes).
        # Linux only (it hooks the HDF5 the netCDF4 wheel loads); elsewhere the
        # writer says why it is not engaged and writes the single-thread way.
        # Host glibc 2.39 and 2.43 differ in geometry asinf/atan2f words.
        # Strict geometry must carry its arithmetic authority in the wheel.
        geometry = self.build_geometry_math(root)
        if geometry is not None:
            shutil.copy2(geometry, destination.parent / GEOMETRY_MATH)
        deflate = self.build_history_deflate(root)
        if deflate is not None:
            shutil.copy2(deflate, destination.parent / HISTORY_DEFLATE)
        if sys.platform.startswith("linux"):
            crate = root / "rust/s2s-zarr"
            subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                            str(crate / "Cargo.toml"), "--target-dir", str(crate / "target")], check=True)
            shutil.copy2(crate / "target/release/libhex_s2s_zarr.so", destination.parent / "libhex_s2s_zarr.so")
        # Both physics routes decode their timed surface stream through this
        # source-built bridge. A lab-only reader.so is never a package input.
        surface = self.build_surface_reader(root)
        if surface is not None:
            shutil.copy2(surface, destination.parent / SURFACE_READER)
        # THE BREAKAGE THIS PREVENTS (release readiness B5, 2026-10-14): the
        # init door needs rw_mpas_geometry speaking hex-vertical-v1 and the
        # stream I/O needs rw_mpas_stream, and no released engine carries
        # either (the frozen 2.9.0 bridge lacks hex-vertical-v1), so a clean
        # install refused both doors.  hexcore.engines reads hexcore/libexec
        # before gpuwm's bridges.  rw_mpas_static joined them
        # (ARWEN-VS-HEX-TIMING-2026-10-15): 2.9.0's static builder gives a
        # generated grid a static that declares binary64 coordinates it does
        # not hold, so the point door could not build a regional mesh.
        for bridge in engine_bridges(root):
            shutil.copy2(bridge, destination.parent / bridge.name)
        # The surface-series producer is owned by this source distribution.
        # Its source fingerprint is separate from the frozen engine bridge pins.
        series = surface_series_bridge(root)
        if series is not None:
            shutil.copy2(series, destination.parent / series.name)
        renderer = s2s_renderer_bridge(root)
        if renderer is not None:
            shutil.copy2(renderer, destination.parent / renderer.name)
            sibling = renderer.with_name("rw_ensbatch" + _EXE)
            shutil.copy2(sibling, destination.parent / sibling.name)
            reducer = renderer.with_name("rw_s2s" + _EXE)
            shutil.copy2(reducer, destination.parent / reducer.name)
        converter = chemistry_converter_bridge(root)
        if converter is not None:
            shutil.copy2(converter, destination.parent / converter.name)
        # THE BREAKAGE THIS PREVENTS (hex global on 2x RTX PRO 6000, 2026-10-16,
        # defect 4): automatic multi-card runs need a METIS partition and a pip
        # install had no gpmetis, so every two-card run refused.  The wheel now
        # carries METIS 5.1.0 built from third_party/ (Apache-2.0); a host
        # without a C compiler still installs everything else, and the door
        # then uses PATH's gpmetis or refuses with the install command.
        partitioner = self.build_gpmetis(root)
        if partitioner is not None:
            shutil.copy2(partitioner, destination.parent / partitioner.name)

    @staticmethod
    def build_gpmetis(root: Path) -> Path | None:
        sys.path.insert(0, str(root / "tools"))
        try:
            import build_gpmetis

            return build_gpmetis.build(gpmetis_build_dir(root))
        except Exception as error:  # no C compiler, or Windows
            print(f"gpuwm-hex: hex_gpmetis NOT BUILT ({error}); automatic multi-card runs will use a "
                  "gpmetis on PATH or refuse with the install command", file=sys.stderr)
            return None
        finally:
            sys.path.pop(0)

    @staticmethod
    def build_netcdf_copy(root: Path) -> Path | None:
        crate = root / "rust/netcdf-copy"
        env = BuildPy.netcdf_environment(root)
        built = subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                                str(crate / "Cargo.toml"), "--target-dir", str(crate / "target")], env=env)
        binary = crate / "target/release" / NETCDF_COPY
        if built.returncode == 0 and binary.is_file():
            return binary
        print("gpuwm-hex: gpuwm-hex-netcdf-copy NOT BUILT: cargo could not link it against "
              "libnetcdf (install the netCDF-C development files, e.g. libnetcdf-dev or "
              "netcdf-devel); the init door will copy vertical templates the slow way and "
              "refuse meshes past the CDF-2 limit by name", file=sys.stderr)
        return None

    @staticmethod
    def build_geometry_math(root: Path) -> Path | None:
        if not sys.platform.startswith("linux"):
            return None
        recipe = runpy.run_path(str(root / "src/hexcore/geometry_math_build.py"))
        return recipe["build_geometry_math"](
            root / "src/hexcore/data/mpasphys-maths", root / "build/geometry-math")

    @staticmethod
    def build_history_deflate(root: Path) -> Path | None:
        if not sys.platform.startswith("linux"):
            return None
        crate = root / "rust/history-deflate"
        built = subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                                str(crate / "Cargo.toml"), "--target-dir", str(crate / "target")])
        library = crate / "target/release" / HISTORY_DEFLATE
        if built.returncode == 0 and library.is_file():
            return library
        print("gpuwm-hex: libhex_history_deflate NOT BUILT: cargo could not build rust/history-deflate; "
              "history frames are compressed on one thread (hexcore.history_deflate says so per frame)",
              file=sys.stderr)
        return None

    @staticmethod
    def build_surface_reader(root: Path) -> Path | None:
        import os
        prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
        if prebuilt:
            library = Path(prebuilt) / SURFACE_READER
            record = json.loads((Path(prebuilt) / "provenance.json").read_text())["surface_reader"]
            for path, expected in ((library, record["sha256"]),
                                   (root / "tools/rustwx/crates/rw-surface/src/lib.rs",
                                    record["source_sha256"])):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise SystemExit(f"surface reader build provenance differs: {path}")
            if not sys.platform.startswith("linux"):
                return library
            # ldd exits 1 on a file with no dynamic section (a fully static reader):
            # that has no host netCDF/HDF5 dependency, so it is kept, not a crash.
            dependencies = subprocess.run(["ldd", str(library)], check=False, text=True, capture_output=True).stdout
            if "libnetcdf.so" not in dependencies and "libhdf5" not in dependencies:
                return library
            print("gpuwm-hex: rebuilding verified surface reader source with static netCDF; "
                  "the supplied reader depends on a host netCDF/HDF5 shared library", file=sys.stderr)
        # THE BREAKAGE THIS PREVENTS (hex 0.4.1 doors, Windows desktop,
        # 2026-10-20): this step ran `bash tools/build_surface_reader.sh`, and
        # on Windows `bash` is WSL's launcher or absent, so `pip install
        # gpuwm-hex` from the sdist died with exit 127 on every Windows host.
        # The same recipe in Python needs only cargo, on every platform; the
        # shell script stays for the release container.
        out = root / "rust/surface-reader-build"
        crate = out / "crate"
        if crate.exists():
            shutil.rmtree(crate)
        shutil.copytree(root / "tools/rustwx/crates/rw-surface", crate)
        workspace = (root / "tools/rustwx/Cargo.toml").read_text(encoding="utf-8")
        profile = re.search(r"^\[profile\.release\]", workspace, re.MULTILINE)
        manifest = crate / "Cargo.toml"
        with manifest.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n[workspace]\n")
            if profile is not None:
                handle.write(workspace[profile.start():])
        jobs = os.environ.get("CARGO_BUILD_JOBS")
        env = BuildPy.netcdf_environment(root, full=True)
        locked = subprocess.run(["cargo", "generate-lockfile", "--offline", "--manifest-path", str(manifest)])
        built = locked.returncode == 0 and subprocess.run(
            ["cargo", os.environ.get("RW_SURFACE_CARGO_ACTION", "build"), "--release", "--locked", "--offline",
             "--manifest-path", str(manifest), "--target-dir", str(out / "target")]
            + (["--features", "static-netcdf"] if sys.platform.startswith("linux") else [])
            + (["-j", jobs] if jobs else []), env=env).returncode == 0
        library = out / "target/release" / SURFACE_READER
        if built and library.is_file():
            return library
        # It links libnetcdf, like the pool reader and the template copier: a
        # host without the netCDF-C development files (every stock Windows
        # host) still installs everything else, and a run that has a surface
        # series refuses by name (hexcore.surface_series_reader.reader_library).
        print("gpuwm-hex: rw_surface NOT BUILT: cargo could not link it against libnetcdf "
              "(install the netCDF-C development files, e.g. libnetcdf-dev or netcdf-devel); "
              "runs with a surface series will refuse by name", file=sys.stderr)
        return None

    @staticmethod
    def build_pool_reader(root: Path) -> Path | None:
        crate = root / "rust/physics-pool"
        built = subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                                str(crate / "Cargo.toml"), "--target-dir", str(crate / "target")],
                               env=BuildPy.netcdf_environment(root))
        binary = crate / "target/release" / POOL_READER
        if built.returncode == 0 and binary.is_file():
            return binary
        print("gpuwm-hex: rw_mpas_physics_pool NOT BUILT: cargo could not link it against libnetcdf "
              "(install the netCDF-C development files, e.g. libnetcdf-dev or netcdf-devel); the MPAS "
              "physics route will refuse by name unless --native-pool-reader names one", file=sys.stderr)
        return None




def s2s_energy_bridge(root: Path) -> Path | None:
    """The S2S energy tables' reducer, built like the scorer from the gpuwm source checkout.

    THE BREAKAGE THIS PREVENTS (hex 0.4.2, 2026-10-23): the s2s products stage draws
    the hub-height wind and solar tables with rw_s2s_energy, which no wheel carried, so
    every default `gpuwm-hex s2s` run from pip stopped at products ("rw_s2s_energy is
    missing").  It links the engine's netcrust and rw-store, which exist only in a gpuwm
    SOURCE checkout; a build without one installs everything else and the energy step
    refuses by name (hexcore.s2s_energy).
    """
    installed = root / "src/hexcore/libexec" / S2S_ENERGY
    if installed.is_file():
        return installed
    engine = engine_source()
    if engine is None:
        print("gpuwm-hex: rw_s2s_energy NOT BUILT: set GPUWM_HEX_ENGINE_SOURCE to a gpuwm source checkout; "
              "the s2s energy tables will refuse by name", file=sys.stderr)
        return None
    output = root / "rust/s2s-energy-build"
    subprocess.run(["bash", str(root / "tools/build_s2s_energy.sh"), str(engine), str(output)], check=True)
    return output / S2S_ENERGY


def surface_series_bridge(root: Path) -> Path | None:
    """Bundle the source-owned producer, checking prebuilt source fingerprints."""
    import os
    name = "rw_sfc_series" + _EXE
    crate = root / "tools/rustwx/crates/rw-sfc-series"
    sources = {str(path.relative_to(crate)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(crate.rglob("*")) if path.is_file()}
    installed = root / "src/hexcore/libexec" / name
    if installed.is_file():
        return installed
    prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
    if prebuilt and (Path(prebuilt) / name).is_file():
        directory = Path(prebuilt)
        provenance = json.loads((directory / "provenance.json").read_text())
        receipt = provenance.get("bridges", {}).get("rw_sfc_series", {})
        binary = directory / name
        if receipt.get("sources") != sources or receipt.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
            raise SystemExit("rw_sfc_series prebuilt does not match this source tree; rebuild tools/build_engine_bridges.sh")
        return binary
    engine = engine_source()
    if engine is None:
        print("gpuwm-hex: rw_sfc_series NOT BUILT: set GPUWM_HEX_ENGINE_SOURCE to a gpuwm source checkout; "
              "surface-series will refuse by name", file=sys.stderr)
        return None
    output = root / "rust/sfc-series-build"
    subprocess.run(["bash", str(root / "tools/build_sfc_series.sh"), str(engine), str(output)], check=True)
    return output / name


def s2s_renderer_bridge(root: Path) -> Path | None:
    """Ship the source-owned renderer capabilities through the ordinary engine ladder."""
    import os
    name = "rw_wrfbatch" + _EXE
    patches = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / "docs/s2s/engine-patches").glob("*.patch"))}
    recipe = hashlib.sha256((root / "tools/build_s2s_diagnostics.sh").read_bytes()).hexdigest()
    prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
    if prebuilt:
        directory = Path(prebuilt)
        binary = directory / name
        receipt = json.loads((directory / "provenance.json").read_text()).get("bridges", {}).get("rw_wrfbatch", {})
        integrated = (receipt.get("patches") == {} and
                      receipt.get("source_copy") == "unchanged integrated engine source; no patches applied")
        if (not binary.is_file() or
                not (receipt.get("patches") == patches or integrated) or
                receipt.get("recipe_sha256") != recipe):
            raise SystemExit("rw_wrfbatch prebuilt lacks the current S2S fixes; rebuild tools/build_engine_bridges.sh")
        if receipt.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
            raise SystemExit("rw_wrfbatch prebuilt hash mismatch; rebuild tools/build_engine_bridges.sh")
        sibling = directory / ("rw_ensbatch" + _EXE)
        if (not sibling.is_file() or receipt.get("ensemble_renderer", {}).get("sha256")
                != hashlib.sha256(sibling.read_bytes()).hexdigest()):
            raise SystemExit("rw_ensbatch prebuilt lacks the matching S2S renderer; rebuild tools/build_engine_bridges.sh")
        reducer = directory / ("rw_s2s" + _EXE)
        if (not reducer.is_file() or receipt.get("diagnostics", {}).get("rw_s2s_sha256")
                != hashlib.sha256(reducer.read_bytes()).hexdigest()
                or receipt.get("diagnostics", {}).get("reducer_lock_sha256")
                != hashlib.sha256((root / "tools/rustwx/Cargo.lock").read_bytes()).hexdigest()
                or receipt.get("diagnostics", {}).get("reducer_sources")
                != {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root / "tools/rustwx/crates/rw-s2s").rglob('*.rs'))}
                or receipt.get("diagnostics", {}).get("renderer_source_sha256")
                != hashlib.sha256((root / "tools/s2s_renderer/s2s_diagnostics.rs").read_bytes()).hexdigest()):
            raise SystemExit("rw_s2s prebuilt lacks the matching S2S diagnostics; rebuild tools/build_engine_bridges.sh")
        return binary
    engine = engine_source()
    if engine is None:
        print("gpuwm-hex: S2S rw_wrfbatch NOT BUILT: set GPUWM_HEX_ENGINE_SOURCE to a gpuwm source checkout", file=sys.stderr)
        return None
    output = root / "rust/s2s-renderer-build"
    subprocess.run(["bash", str(root / "tools/build_s2s_diagnostics.sh"), str(engine), str(output)], check=True)
    return output / name


def chemistry_converter_bridge(root: Path) -> Path | None:
    """Bundle the chemistry-aware converter on the ordinary engine ladder."""
    name = "rw_mpas_convert" + _EXE
    patch = hashlib.sha256((root / "engine-patches/chemistry-products.patch").read_bytes()).hexdigest()
    recipe = hashlib.sha256((root / "tools/build_chemistry_converter.sh").read_bytes()).hexdigest()
    prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
    if prebuilt and (Path(prebuilt) / name).is_file():
        directory = Path(prebuilt)
        provenance = json.loads((directory / "provenance.json").read_text())
        # publish.yml stages each source prebuilt's pinned provenance under "bridges" by file name.
        record = provenance.get("bridges", {}).get(name) or provenance.get("chemistry_converter", {})
        binary = directory / name
        if (record.get("patch_sha256") != patch or record.get("recipe_sha256") != recipe
                or record.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest()):
            raise SystemExit("rw_mpas_convert prebuilt lacks current chemistry product provenance")
        return binary
    engine = engine_source()
    if engine is None:
        print("gpuwm-hex: chemistry rw_mpas_convert NOT BUILT: set GPUWM_HEX_ENGINE_SOURCE "
              "to a gpuwm source checkout; chemistry products require this converter", file=sys.stderr)
        return None
    output = root / "rust/chemistry-converter-build"
    subprocess.run(["bash", str(root / "tools/build_chemistry_converter.sh"), str(engine), str(output)], check=True)
    return output / name


def engine_bridges(root: Path) -> list[Path]:
    """The pinned engine bridges to bundle: prebuilt by hash, else built from a gpuwm source.

    1. ``$GPUWM_HEX_ENGINE_BRIDGES`` names a directory holding the bytes
       ``tools/build_engine_bridges.sh`` produced (publish.yml downloads them
       from the release assets ``engine-bridges.json`` names).  Every file must
       equal its pinned SHA-256, or the build stops: a bridge whose bytes are
       not the proven ones is not shipped.
    2. Otherwise a gpuwm source checkout (``engine_source``) that carries all
       three sources at the versions this release needs builds them with
       cargo (the stream bridge links libnetcdf).
    3. Otherwise nothing is bundled and the build says so; the doors then
       resolve down the ladder and refuse by name, and the release's platform
       check (--require-libexec) refuses the wheel.
    """

    import os

    pins = json.loads((root / "src/hexcore/data/engine-bridges.json").read_text(encoding="utf-8"))
    prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
    if prebuilt:
        directory, found = Path(prebuilt), []
        for name, row in sorted(pins["bridges"].items()):
            path = directory / name
            if not path.is_file():
                raise SystemExit(f"GPUWM_HEX_ENGINE_BRIDGES={directory} has no {name}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != row["sha256"]:
                raise SystemExit(f"{path}: sha256 {actual} is not the pinned {row['sha256']} "
                                 "(src/hexcore/data/engine-bridges.json)")
            for marker in row.get("required_markers", ()):
                if marker.encode("ascii") not in path.read_bytes():
                    raise SystemExit(f"{path}: missing required bridge capability {marker}")
            found.append(path)
        return found
    engine = engine_source()
    bins = engine / "tools/rustwx/crates/rw-mpas/src/bin" if engine is not None else None
    builder = bins.parent / "static_builder.rs" if bins is not None else None
    if (bins is not None and (bins / "rw_mpas_stream.rs").is_file()
            and b"hex-vertical-v1" in (bins / "rw_mpas_geometry.rs").read_bytes()
            and builder.is_file() and STATIC_BUILDER_MARK in builder.read_bytes()
            and all((bins / f"{name}.rs").is_file() for name in RRFS_BRIDGES)
            and LBC_CASE9_MARK in (bins.parent / "init/metfile.rs").read_bytes()
            and (engine / "tools/rustwx/crates/rw-obs/src/bin/igra2.rs").is_file()):
        workspace = engine / "tools/rustwx"
        mpas = [name for name in ENGINE_BRIDGES if name not in OBS_BRIDGES]
        built = subprocess.run(["cargo", "build", "--release", "--locked", "--offline", "-p", "rw-mpas",
                                "--features", "system-netcdf",
                                *[part for name in mpas for part in ("--bin", name.removesuffix(_EXE))]],
                               cwd=workspace)
        if built.returncode == 0:
            built = subprocess.run(["cargo", "build", "--release", "--locked", "--offline", "-p", "rw-obs",
                                    *[part for name in OBS_BRIDGES for part in ("--bin", name.removesuffix(_EXE))]],
                                   cwd=workspace)
        target_dir = Path(os.environ.get("CARGO_TARGET_DIR", workspace / "target"))
        if not target_dir.is_absolute():
            target_dir = workspace / target_dir
        paths = [target_dir / "release" / name for name in ENGINE_BRIDGES]
        if built.returncode == 0 and all(path.is_file() for path in paths):
            # Source builds must obey the same byte pins as prebuilt assets.
            for path in paths:
                expected = pins["bridges"][path.name]["sha256"]
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise SystemExit(f"{path}: sha256 {actual} is not the pinned {expected} "
                                     "(src/hexcore/data/engine-bridges.json); use pinned "
                                     "GPUWM_HEX_ENGINE_BRIDGES or qualify and update the pins")
                for marker in pins["bridges"][path.name].get("required_markers", ()):
                    if marker.encode("ascii") not in path.read_bytes():
                        raise SystemExit(f"{path}: missing required bridge capability {marker}")
            return paths
    print("gpuwm-hex: rw_mpas_geometry, rw_mpas_stream, rw_mpas_static, the three RRFS programs, rw_mpas_lbc "
          "and rw_igra2 NOT BUNDLED: set "
          "GPUWM_HEX_ENGINE_BRIDGES to the pinned prebuilt bridges (src/hexcore/data/engine-bridges.json), "
          "or GPUWM_HEX_ENGINE_SOURCE to a gpuwm checkout carrying all of them (gpuwm 0aa356bf or newer); "
          "the init door, the stream I/O and the point door will look for them on gpuwm's bridge ladder "
          "and refuse by name", file=sys.stderr)
    return []


class BuildHex(BuildPy):
    """Keep release kernel packs out of the size-limited distribution."""

    def run(self):
        super().run()
        packs = sorted((Path(self.build_lib) / "hexcore/data").glob("kernel-pack*.json"))
        if packs:
            raise RuntimeError("kernel packs must ship as release assets, not wheel data: "
                               + ", ".join(str(path) for path in packs))


class NativeReferenceDistribution(Distribution):
    def has_ext_modules(self):
        # The Rust executable must never receive a platform-independent wheel tag.
        return True


def glibc_floor(binary: Path) -> tuple[int, int] | None:
    """The newest GLIBC symbol version an ELF executable requires, or None.

    Read from the version-requirement names the linker wrote (GLIBC_2.34 and
    the like); a binary that names none needs no particular glibc.
    """

    found = [(int(major), int(minor))
             for major, minor in re.findall(rb"GLIBC_(\d+)\.(\d+)", binary.read_bytes())]
    return max(found) if found else None


def wheel_platform_tag(binary: Path, *others: Path) -> str:
    """The platform tag the wheel may claim for the bundled helpers.

    THE BREAKAGE THIS PREVENTS: the wheel was tagged cp3X-cp3X-linux_x86_64.
    PyPI rejects a bare linux_x86_64 tag outright, so the 0.3.4 upload would
    have failed, and the cp3X tag bound the wheel to the build's Python minor
    although the helper is a subprocess executable that links no libpython.
    On Linux the tag is manylinux_<glibc floor> measured from the binary
    itself, so it never claims an older glibc than the helper needs.
    """

    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if not sys.platform.startswith("linux"):
        return plat
    # Every bundled ELF counts: the physics-table decoders ship beside the
    # helper, and the newest glibc any of them needs is the wheel's floor.
    floors = [floor for path in (binary, *others) if path.is_file()
              for floor in (glibc_floor(path),) if floor is not None]
    major, minor = max(floors) if floors else (2, 17)
    return f"manylinux_{major}_{minor}_{platform.machine()}"


class HexWheel(bdist_wheel):
    """py3-none-<platform>: one wheel for every Python the package supports."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        binary = Path(__file__).parent / "rust/host-prep/target/release" / HOSTPREP
        if not binary.is_file():
            # bdist_wheel asks for tags before build_py has built the helper.
            return super().get_tag()
        return "py3", "none", wheel_platform_tag(binary, *bundled_helpers(Path(__file__).parent))


def gpmetis_build_dir(root: Path) -> Path:
    return root / "third_party/metis-5.1.0/build"


def bundled_engine_bridges(root: Path) -> tuple[Path, ...]:
    """The engine bridges as get_tag sees them (the pinned prebuilt directory, when named)."""
    import os

    prebuilt = os.environ.get("GPUWM_HEX_ENGINE_BRIDGES")
    return tuple(Path(prebuilt) / name for name in ENGINE_BRIDGES) if prebuilt else ()


def bundled_helpers(root: Path) -> tuple[Path, ...]:
    """Every ELF the wheel carries beside hex_contract_hostprep (absent ones are skipped by the tag)."""
    import os
    tables = root / "rust/physics-tables/target/release"
    return (root / "src/hexcore/libexec" / OBS_SCORER,
            root / "rust/chemistry-converter-build" / ("rw_mpas_convert" + _EXE),
            root / "rust/chem-sources/target/release" / CHEM_SOURCES,
            root / "rust/ugwp-oro/target/release" / UGWP_ORO,
            root / "rust/s2s-energy-build" / S2S_ENERGY,
            root / "rust/netcdf-copy/target/release" / NETCDF_COPY,
            (Path(os.environ["GPUWM_HEX_ENGINE_BRIDGES"]) / ("rw_wrfbatch" + _EXE)
             if os.environ.get("GPUWM_HEX_ENGINE_BRIDGES") else
             root / "rust/s2s-renderer-build" / ("rw_wrfbatch" + _EXE)),
            (Path(os.environ["GPUWM_HEX_ENGINE_BRIDGES"]) / ("rw_ensbatch" + _EXE)
             if os.environ.get("GPUWM_HEX_ENGINE_BRIDGES") else
             root / "rust/s2s-renderer-build" / ("rw_ensbatch" + _EXE)),
            (Path(os.environ["GPUWM_HEX_ENGINE_BRIDGES"]) / ("rw_s2s" + _EXE)
             if os.environ.get("GPUWM_HEX_ENGINE_BRIDGES") else
             root / "rust/s2s-renderer-build" / ("rw_s2s" + _EXE)),
            root / "rust/s2s-zarr/target/release/libhex_s2s_zarr.so",
            *bundled_engine_bridges(root),
            *(tables / name for name in PHYSICS_TABLES),
            root / "rust/physics-pool/target/release" / POOL_READER,
            (Path(os.environ["GPUWM_HEX_ENGINE_BRIDGES"]) / SURFACE_READER
             if os.environ.get("GPUWM_HEX_ENGINE_BRIDGES") else
             root / "rust/surface-reader-build/target/release" / SURFACE_READER),
            gpmetis_build_dir(root) / ("hex_gpmetis" + _EXE))


setup(cmdclass={"build_py": BuildHex, "bdist_wheel": HexWheel},
      distclass=NativeReferenceDistribution)
