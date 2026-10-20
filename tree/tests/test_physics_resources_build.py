"""`gpuwm-hex physics-resources build`: the MPAS hex-cuda packet from an install.

THE BREAKAGE THIS PREVENTS
--------------------------
Every MPAS hex-cuda slot that reads tables or shared device maths refused to
run without a ``--physics-resources`` packet, and the only packets named lane
scratch files on one build host, so a clean install could not reach the MPAS
physics suites.  These CPU tests hold the parts of the builder that need no
network, compiler or device: the device maths reproduce the certified digests
from the vendored sources, every pin agrees with the scheme modules that
enforce it, and each missing prerequisite refuses by name.  The decoders'
decoded-word check runs when a certified packet is supplied
(``HEX_CERTIFIED_PHYSICS_RESOURCES``) and the wheel's decoders are present.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

import pytest

from hexcore.mpasphys import resources_build as rb

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("legacy", [False, True])
def test_packet_resources_follow_cache_move_without_original(tmp_path, legacy):
    original = tmp_path / "build-cache"
    (original / "packets").mkdir(parents=True)
    table = original / "tables" / "data.bin"
    table.parent.mkdir()
    table.write_bytes(b"certified data")
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    packet = {"scheme": {"table": {"path": str(table), "sha256": digest},
                         "table_directory": str(table.parent), "external": "/opt/explicit.so",
                         "landuse": "MODIFIED_IGBP_MODIS_NOAH"}}
    file = original / "packets" / "physics-resources-test.json"
    file.write_text(json.dumps(packet if legacy else rb.portable_packet(packet, original)))
    file.with_suffix(".receipt.json").write_text(json.dumps({"cache": str(original)}))
    moved = tmp_path / "runtime-cache"
    original.rename(moved)
    loaded = rb.load_packet(moved / "packets" / file.name)["scheme"]
    assert not original.exists()
    assert loaded["external"] == "/opt/explicit.so"
    assert loaded["landuse"] == "MODIFIED_IGBP_MODIS_NOAH"
    assert Path(loaded["table_directory"]) == moved / "tables"
    assert hashlib.sha256(Path(loaded["table"]["path"]).read_bytes()).hexdigest() == digest


def test_device_maths_reproduce_the_certified_headers(tmp_path):
    out = rb.build_maths(tmp_path)
    assert set(out) == {"bl_mynn", "mp_thompson", "mp_tempo"}
    for scheme, (filename, expected) in rb.MATHS.items():
        path = Path(out[scheme]["path"])
        assert path.name == filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected == out[scheme]["sha256"]
    sources = json.loads((ROOT / "src/hexcore/data/mpasphys-maths/SOURCES.json").read_text(encoding="utf-8"))
    assert sorted(sources["outputs"].values()) == sorted(sha for _, sha in rb.MATHS.values())


def test_a_changed_vendored_source_refuses_by_name(tmp_path, monkeypatch):
    copy = tmp_path / "data" / "mpasphys-maths"
    shutil.copytree(ROOT / "src/hexcore/data/mpasphys-maths", copy)
    target = copy / "core-math" / "tanhf.c"
    target.write_bytes(target.read_bytes() + b"\n")
    monkeypatch.setattr(rb, "_data_root", lambda name: tmp_path / "data" / name)
    with pytest.raises(rb.PhysicsResourceError, match="core-math/tanhf.c"):
        rb.build_maths(tmp_path / "cache")


def test_vendored_notices_ship_beside_the_maths():
    root = ROOT / "src/hexcore/data/mpasphys-maths"
    sources = json.loads((root / "SOURCES.json").read_text(encoding="utf-8"))
    for name, sha in sources["notices"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha
    for name, row in sources["files"].items():
        assert row["licence"] and row["origin"]
        if name.startswith("core-math/"):
            assert "Permission is hereby granted" in (root / name).read_text(encoding="utf-8")


def test_pins_agree_with_the_scheme_modules_that_enforce_them():
    from hexcore.mpasphys.noah_v842 import TABLE_FILE_SHA256 as NOAH
    from hexcore.mpasphys.table_resources_v842 import THOMPSON_AUXILIARY_SHA256
    from hexcore.mpasphys.thompson_v842 import TABLE_FILE_SHA256 as THOMPSON
    assert {name: rb.TABLE_FILES[name][2] for name in NOAH} == NOAH
    expected = rb._thompson_expected()
    assert {name: expected[name] for name in THOMPSON} == THOMPSON
    assert expected["auxiliary.bin"] == THOMPSON_AUXILIARY_SHA256
    for value in rb.DECODED_SHA256.values():
        assert re.fullmatch(r"[0-9a-f]{64}", value)


def test_the_aerosol_slot_binds_its_own_loader_and_auxiliary():
    # Breakage: the aerosol entry was the shared Thompson one (unit CCN table) and the descriptor
    # bound the classic loader, so aerosol-aware Thompson never reached the GPU from an install.
    from hexcore.physics_suites import load_table
    from hexcore.mpasphys import thompson_aero_v842
    row = load_table()["mpasphys_v842"]["slots"]["microphysics"]["cuda_entries"]["mp_thompson_aerosols"]
    assert row["resources"]["module"] == "thompson_aero_v842" and row["resources"]["call"] == "load_resources"
    assert "mp_thompson_aerosols" not in rb.THOMPSON_SHARING
    assert re.fullmatch(r"[0-9a-f]{64}", thompson_aero_v842.AUXILIARY_SHA256)
    assert rb.TABLE_FILES["CCN_ACTIVATE_DATA"][2] == "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd"
    from hexcore.mpasphys.native_library_build import referee_builder
    tool = referee_builder("thompson_tables")
    assert (Path(tool.__file__).parent / "table_sources/thompson_aero_auxiliary_export.F90").is_file()


def test_every_resource_slot_is_built_or_named():
    from hexcore.physics_suites import load_table
    built = {"bl_mynn", "cu_kain_fritsch", "mp_thompson", "mp_thompson_aerosols", "rrtmg_lw", "rrtmg_sw",
             "sf_noah", *rb.THOMPSON_SHARING}
    for row in load_table()["mpasphys_v842"]["slots"].values():
        for scheme, descriptor in row.get("cuda_entries", {}).items():
            if descriptor.get("resources"):
                assert scheme in built or scheme in rb.NOT_BUILT, scheme
    # The RRFS stage defaults its ported slots to hex-cuda (driver_v842.default_slot_backends), so
    # each resource those slots bind is built by the install, or the scheme is held with its reason.
    # Breakage prevented: a bare hrrrv5 run refusing at slot registration for a packet entry no
    # install can make.
    rrfs = load_table()["mpasphys_rrfs"]
    for row in rrfs["slots"].values():
        for scheme, descriptor in row.get("cuda_entries", {}).items():
            if descriptor.get("resources"):
                assert (scheme in built | {"mp_tempo", "bl_mynnedmf"}
                        or scheme in rb.NOT_BUILT), scheme


def test_fetch_reads_a_local_source_tree_and_refuses_other_bytes(tmp_path):
    payload = b"pinned bytes\n"
    sha = hashlib.sha256(payload).hexdigest()
    tree = tmp_path / "mpas"
    local = tree / rb._local_path("mpas-data", "atmosphere/physics_wrf/files/EXAMPLE.TBL")
    local.parent.mkdir(parents=True)
    local.write_bytes(payload)
    got = rb.fetch("EXAMPLE.TBL", "mpas-data", "atmosphere/physics_wrf/files/EXAMPLE.TBL", sha,
                   tmp_path / "cache" / "EXAMPLE.TBL", mpas_source=tree, offline=True)
    assert got.read_bytes() == payload
    local.write_bytes(b"other\n")
    (tmp_path / "cache" / "EXAMPLE.TBL").unlink()
    with pytest.raises(rb.PhysicsResourceError, match="not the pinned file"):
        rb.fetch("EXAMPLE.TBL", "mpas-data", "atmosphere/physics_wrf/files/EXAMPLE.TBL", sha,
                 tmp_path / "cache" / "EXAMPLE.TBL", mpas_source=tree, offline=True)


def test_offline_without_a_local_copy_names_the_url(tmp_path):
    with pytest.raises(rb.PhysicsResourceError, match="raw.githubusercontent.com/MPAS-Dev/MPAS-Data/c57dbc7"):
        rb.fetch("RRTMG_LW_DATA", *rb.TABLE_FILES["RRTMG_LW_DATA"], tmp_path / "RRTMG_LW_DATA", offline=True)


def test_thompson_strict_tables_without_gfortran_name_both_remedies(tmp_path, monkeypatch):
    monkeypatch.setattr(rb.shutil, "which", lambda name: None)
    with pytest.raises(rb.PhysicsResourceError, match="gfortran.*--thompson-tables"):
        rb.build_thompson_strict(tmp_path)


def test_thompson_generators_link_the_pinned_log10f(tmp_path):
    """Breakage: on glibc 2.39 (Ubuntu 24.04) MP_THOMPSON_freezeH2O_DATA.DBL followed the host's log10f, the
    build refused, and no MPAS suite ran from a normal install.  Every gate host has glibc 2.43, whose log10f
    already is CORE-MATH's, so only this test sees the pin go missing."""
    commands = rb.thompson_generator_commands("gfortran")
    links = [command for command in commands if "-o" in command and command[command.index("-o") + 1]
             in ("build_tables", "thompson_auxiliary_export")]
    assert len(links) == 2 and all("cr_log10f.o" in command for command in links)
    shim = next(command for command in commands if "cr_log10f.c" in command)
    assert "-Dcr_log10f=log10f" in shim and commands.index(shim) < commands.index(links[0])
    staged = rb._stage_pinned_log10f(tmp_path)
    sources = json.loads((ROOT / "src/hexcore/data/mpasphys-maths/SOURCES.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == sources["files"][rb.PINNED_LOG10F]["sha256"]
    assert "float cr_log10f(float x)" in staged.read_text(encoding="utf-8")
    # The release image has glibc 2.28. Its other libm functions also differ
    # from the pinned Thompson maths and changed all five strict table hashes.
    assert all("thompson_host_math.o" in command for command in links)
    assert all(all(f"-Wl,--wrap={name}" in command for name in rb.HOST_MATH_FUNCTIONS) for command in links)
    rb._stage_pinned_host_maths(tmp_path)
    for name in ("cuda_host_shim.h", "thompson_host_math.cpp", "glibc_flt32.cuh",
                 "glibc_flt64.cuh", "thompson_aerosol_libm.cuh"):
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==sources["files"][name]["sha256"]


def test_thompson_tables_directory_must_hold_the_strict_bytes(tmp_path):
    supplied = tmp_path / "tables"
    supplied.mkdir()
    for name in rb._thompson_expected():
        (supplied / name).write_bytes(b"not the strict table")
    with pytest.raises(rb.PhysicsResourceError, match="strict native reference"):
        rb.build_thompson_strict(tmp_path / "cache", thompson_tables=supplied)


def test_cache_root_follows_the_package_cache_convention(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_HEX_CACHE_ROOT", str(tmp_path))
    assert rb.default_cache() == tmp_path / "physics-resources"
    monkeypatch.delenv("GPUWM_HEX_CACHE_ROOT")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert rb.default_cache() == tmp_path / "xdg" / "gpuwm-hex" / "physics-resources"


def test_the_cli_names_the_door():
    from hexcore.cli import build_parser
    parser = build_parser()
    arguments = parser.parse_args(["physics-resources", "build", "--offline", "--landuse", "USGS"])
    assert arguments.handler is rb._build_command
    assert arguments.landuse == "USGS" and arguments.offline


def test_first_use_reads_the_session_landuse():
    from hexcore.mpasphys.device_pools_v842 import _session_landuse

    class Native:
        def __init__(self, value):
            self.value = value

        def get_text_field(self, pool, name):
            assert (pool, name) == ("sfc_input", "mminlu")
            if isinstance(self.value, Exception):
                raise self.value
            return self.value

    class Provider:
        def __init__(self, value):
            self.native = Native(value)

    assert _session_landuse(Provider("USGS    ")) == "USGS"
    assert _session_landuse(Provider("   ")) is None
    assert _session_landuse(Provider(KeyError("sfc_input.mminlu"))) is None


def test_the_crate_builds_every_decoder_the_builder_binds():
    cargo = (ROOT / "rust/physics-tables/Cargo.toml").read_text(encoding="utf-8")
    names = set(re.findall(r'^name = "([^"]+)"', cargo, re.M)) - {"hex-physics-tables"}
    assert names == set(rb.DECODERS)
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "graft rust/physics-tables" in manifest and "prune rust/physics-tables/target" in manifest
    setup = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "rust/physics-tables/Cargo.toml" in setup
    for name in rb.DECODERS:
        assert (f'"{name}" + _EXE' in setup) or (name == "lookup_decode" and '"liblookup_decode.so"' in setup), name


@pytest.mark.skipif(not os.environ.get("HEX_CERTIFIED_PHYSICS_RESOURCES"),
                    reason="set HEX_CERTIFIED_PHYSICS_RESOURCES to a certified packet to compare decoded words")
def test_decoders_decode_like_the_certified_packet(tmp_path):
    certified = json.loads(Path(os.environ["HEX_CERTIFIED_PHYSICS_RESOURCES"]).read_text(encoding="utf-8"))
    reference = rb.decoded_digests(certified)
    for key, value in reference.items():
        if key in rb.DECODED_SHA256:
            assert rb.DECODED_SHA256[key] == value, key
    decoders = rb.stage_decoders(tmp_path)
    rebuilt = json.loads(json.dumps(certified))
    rebuilt["mp_thompson"]["rust_decoder"] = decoders["lookup_decode"]
    rebuilt["rrtmg_lw"]["decoder"] = decoders["rw_mpas_rrtmg_lw_tables"]
    rebuilt["rrtmg_sw"]["decoder"] = decoders["rw_mpas_rrtmg_sw_tables"]
    rebuilt["sf_noah"]["rust_decoder"] = decoders["table_decode"]
    assert rb.decoded_digests(rebuilt) == reference


def test_first_use_is_off_in_the_battery_and_names_the_command(tmp_path):
    assert os.environ.get("GPUWM_HEX_PHYSICS_RESOURCES_FIRST_USE") == "off"
    with pytest.raises(rb.PhysicsResourceError, match="gpuwm-hex physics-resources build"):
        rb.ensure_packet(cache=tmp_path)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize('operation', ['classic', 'aerosol', 'commands'])
def test_installed_table_build_refuses_even_with_a_cached_checkout(tmp_path, monkeypatch, operation):
    from hexcore.mpasphys import native_library_build as metadata
    tool = metadata.referee_builder('thompson_tables')
    assert (Path(tool.__file__).parent / 'table_sources/framework_stubs.F90').is_file()
    installed = tmp_path / 'site-packages/hexcore/mpasphys/native_library_build.py'
    installed.parent.mkdir(parents=True)
    # An old wheel's Fortran files never make an installed compiler valid.
    old_sources = installed.parents[1] / 'data/mpasphys-native'
    old_sources.mkdir(parents=True)
    (old_sources / 'framework_stubs.F90').write_text('old packaged source\n')
    monkeypatch.setattr(metadata, '__file__', str(installed))
    with pytest.raises(rb.PhysicsResourceError, match='source-checkout referee tool'):
        if operation == 'classic':
            rb.build_thompson_strict(tmp_path / 'cache')
        elif operation == 'aerosol':
            rb.build_thompson_aero_auxiliary(tmp_path / 'cache', tmp_path, tmp_path / 'ccn')
        else:
            rb.thompson_generator_commands('gfortran')


@pytest.mark.parametrize('existing_cache', [False, True])
def test_pinned_table_bytes_can_be_imported_without_a_referee(tmp_path, monkeypatch, existing_cache):
    # An installed CUDA resource packet may import known table bytes. Only
    # native generation needs the checkout; validation and reuse do not.
    data = {'table.DBL': b'\x00\x00\x80?\x00\x00\x00\x80', 'auxiliary.bin': b'captured table words'}
    expected = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
    monkeypatch.setattr(rb, '_thompson_expected', lambda: expected)
    def forbidden():
        raise AssertionError('pinned byte import reached native compiler tooling')
    monkeypatch.setattr(rb, '_thompson_referee', forbidden)
    cache = tmp_path / 'cache'
    target = cache / 'thompson-strict/tables'
    supplied = target if existing_cache else tmp_path / 'supplied'
    supplied.mkdir(parents=True)
    for name, raw in data.items():
        (supplied / name).write_bytes(raw)
    assert rb.build_thompson_strict(cache, thompson_tables=None if existing_cache else supplied) == target
    assert {name: (target / name).read_bytes() for name in data} == data
