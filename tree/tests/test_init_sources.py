"""The init-source table and its one path.

THE BREAKAGES THESE PREVENT.  (1) A per-model branch creeping into the path:
the arbitrary acceptance test (a project ruling, 2026-08-16) says a new source is
table work, so the module must not name any source.  (2) A row whose Vtable
or first-guess declaration is missing reaches met_intermediate or the init
door and fails after the download was paid for, or silently drops fields.
(3) A member request that addresses the wrong member's bytes: the receipt
would name member 7 while the state came from member 1.  (4) Two rows
claiming one name, so a request picks one silently.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from hexcore import init_sources as src

TABLE = src.load_table()


def test_the_table_loads_and_every_runnable_row_has_its_vtable():
    runnable = [row for row in TABLE["sources"] if row.get("status") != "refused"]
    assert {row["id"] for row in runnable} >= {"gfs", "ifs", "aifs", "ifs-ens", "gefs", "era5", "icon"}
    for row in runnable:
        assert src.resolve_vtable(row).is_file(), row["id"]


# ---------------------------------------------------------------------------
# an installed wheel can run the door (g16 fresh-install QA, 2026-10-12)
# ---------------------------------------------------------------------------
def test_every_vtable_ships_inside_the_package():
    # THE BREAKAGE: the Vtables lived at the repository root, outside tree/,
    # so neither the wheel nor the sdist carried them and an installed door
    # refused "Vtable <venv>/lib/vtables/Vtable.GFS does not exist" for every
    # row, after paying for the download.  data/** is package data, so a
    # Vtable under the package directory is in every artifact.
    import hexcore

    package = Path(hexcore.__file__).resolve().parent
    for row in TABLE["sources"]:
        if row.get("status") == "refused":
            continue
        vtable = src.resolve_vtable(row)
        assert vtable.is_relative_to(package / "data"), (row["id"], vtable)


def _no_engine_estate(monkeypatch, tmp_path, name):
    from hexcore import engines

    for spec in (engines.MET_INTERMEDIATE, engines.GDT101_REMAP):
        for env in (*spec.env_names, spec.gpuwm_env):
            monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(engines, "_gpuwm_module_candidates", lambda spec: None)
    monkeypatch.setattr(engines, "_gpuwm_package_dir", lambda: tmp_path / "site" / "gpuwm")
    monkeypatch.setattr(engines, "USER_BRIDGE_DIR", tmp_path / "home" / ".gpuwm" / "bridges")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    return tmp_path / "site" / "gpuwm" / "libexec" / "bridges" / engines.executable_name(name)


@pytest.mark.parametrize("resolver,name", [(src.resolve_writer, "met_intermediate"),
                                           (src.resolve_remapper, "gdt101_remap")])
def test_the_writers_are_found_where_the_gpuwm_wheel_installs_them(tmp_path, monkeypatch, resolver, name):
    # THE BREAKAGE: the door read only its flag, one variable and PATH, so a
    # clean `pip install gpuwm gpuwm-hex` refused "met_intermediate was not
    # found" while gpuwm's wheel carried the binary in libexec/bridges.
    import os

    bundled = _no_engine_estate(monkeypatch, tmp_path, name)
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"#!/bin/sh\n")
    bundled.chmod(0o755)
    assert os.path.samefile(resolver(None), bundled)


@pytest.mark.parametrize("resolver,name", [(src.resolve_writer, "met_intermediate"),
                                           (src.resolve_remapper, "gdt101_remap")])
def test_an_absent_writer_refuses_by_name_with_the_staging_command(tmp_path, monkeypatch, resolver, name):
    from hexcore import engines

    _no_engine_estate(monkeypatch, tmp_path, name)
    with pytest.raises(engines.EngineRefusal, match=f"{name} not found"):
        resolver(None)
    with pytest.raises(engines.EngineRefusal, match="not a file"):
        resolver(tmp_path / "named-but-missing")


def test_the_writer_is_resolved_before_the_download(tmp_path, monkeypatch):
    # A missing writer found after the acquire stage costs the whole transfer.
    import argparse

    from hexcore import engines

    _no_engine_estate(monkeypatch, tmp_path, "met_intermediate")

    def _no_download(*args, **kwargs):
        raise AssertionError("acquire ran before the writer was resolved")

    monkeypatch.setattr(src, "acquire", _no_download)
    arguments = argparse.Namespace(
        table=None, list=False, source="gfs", cycle="2026-10-11T12", member=None,
        out_dir=tmp_path / "out", grid=None, static=None, vertical_spec=None, staged=[],
        writer=None, remapper=None, jobs=1, stop_after=None, order_map=None, hex_order=None,
        case_template=None, keep_unordered=False)
    with pytest.raises(engines.EngineRefusal, match="met_intermediate not found"):
        src.run_init_source_command(arguments)


def test_the_writer_is_a_doctor_row():
    from hexcore import engines

    assert engines.MET_INTERMEDIATE in engines.ENGINES
    # The build remedy names the workspace the bridge actually builds in.
    assert "run at tools/grib1_bridge" in engines.remedy(engines.MET_INTERMEDIATE)
    assert "run at tools/rustwx" in engines.remedy(engines.INIT)


def test_the_path_names_no_source():
    tree = ast.parse(Path(src.__file__).read_text(encoding="utf-8"))
    literals = {node.value.lower() for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    names = set()
    for row in TABLE["sources"]:
        names.update(n.lower() for n in (row["id"], *row.get("aliases", ())))
    assert not (names & literals), sorted(names & literals)


def test_map_source_labels_are_the_writers_closed_vocabulary():
    # met_intermediate refuses an unknown label; the rows must use its labels.
    known = {"ncep-gfs", "ncep-gefs", "ncep-cdas-cfsv2", "ecmwf", "dwd"}
    for row in TABLE["sources"]:
        if row.get("status") != "refused":
            assert row["decode"]["map_source"] in known, row["id"]


def test_aliases_resolve_and_unknown_names_refuse_with_the_list():
    assert src.resolve_source(TABLE, "ECMWF")["id"] == "ifs"
    assert src.resolve_source(TABLE, "aifs_single")["id"] == "aifs"
    with pytest.raises(src.InitSourceRefusal, match="known: gfs"):
        src.resolve_source(TABLE, "nope")


def test_a_refused_row_refuses_with_its_named_breakage(tmp_path):
    table = json.loads(src.TABLE_PATH.read_text(encoding="utf-8"))
    table["sources"].append({"id": "nowhere", "display_name": "x", "status": "refused",
                             "refusal": "its bytes are on a grid no writer reads"})
    path = tmp_path / "t.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(src.InitSourceRefusal, match="no writer reads"):
        src.resolve_source(src.load_table(path), "nowhere")


def test_an_unstructured_source_is_rows_sets_and_a_remap_stage():
    # ICON global: 2 coordinate records, 5 pressure fields x 18 levels, 12
    # surface/invariant fields, 8 soil-temperature nodes, 8 soil-water layers.
    icon = src.resolve_source(TABLE, "icon")
    plans = src.plan_products(icon, src.Cycle.parse("2026-10-11T12"), None)
    assert len(plans) == 2 + 90 + 12 + 8 + 8
    t850 = [p for p in plans if p.filename.endswith("_000_850_T.grib2")]
    assert len(t850) == 1 and t850[0].remap == {"selector": "0,0,0,100,85000,255,0", "method": "idw4"}
    assert t850[0].url.startswith("https://opendata.dwd.de/weather/nwp/icon/grib/12/t/")
    assert t850[0].codec == "bz2" and t850[0].url.endswith(".grib2.bz2")
    layer = [p for p in plans if p.filename.endswith("_000_9_W_SO.grib2")][0]
    assert layer.remap["selector"] == "2,3,20,106,0.09,106,0.27"
    assert sum(1 for p in plans if p.remap is None) == 2   # CLAT and CLON build the plan only


def test_member_addressing_selects_that_members_records():
    cycle = src.Cycle.parse("2026-10-03T12")
    ens = src.resolve_source(TABLE, "ifs-ens")
    plans = src.plan_products(ens, cycle, 7)
    assert plans[0].index_select == {"type": "pf", "number": "7"}
    assert plans[0].filename.endswith("-m07.grib2")
    assert plans[1].index_select == {"param": "z", "levtype": "sfc"}
    # The enfo publication has no control record; member 0 refuses by name.
    with pytest.raises(src.InitSourceRefusal, match="no control member is published"):
        src.plan_products(ens, cycle, 0)
    gefs = src.resolve_source(TABLE, "gefs")
    urls = [p.url for p in src.plan_products(gefs, cycle, 3)]
    assert all("/gep03.t12z." in url for url in urls) and "gefs.20261003/12/" in urls[0]
    assert "/gec00.t12z." in src.plan_products(gefs, cycle, 0)[0].url
    with pytest.raises(src.InitSourceRefusal, match="pass --member"):
        src.plan_products(gefs, cycle, None)
    with pytest.raises(src.InitSourceRefusal, match="outside"):
        src.plan_products(gefs, cycle, 31)
    with pytest.raises(src.InitSourceRefusal, match="declares no members"):
        src.plan_products(src.resolve_source(TABLE, "gfs"), cycle, 1)


def test_index_selection_matches_every_declared_key():
    index = "\n".join(json.dumps(r) for r in (
        {"type": "pf", "number": "1", "param": "t", "_offset": 0, "_length": 5},
        {"type": "pf", "number": "10", "param": "t", "_offset": 5, "_length": 5},
        {"type": "cf", "param": "t", "_offset": 10, "_length": 5},
    ))
    assert [r["_offset"] for r in src.select_index_records(index, {"type": "pf", "number": "1"})] == [0]
    assert [r["_offset"] for r in src.select_index_records(index, {"type": "cf"})] == [10]


def test_every_row_initialises_under_the_same_run_switches(tmp_path):
    cycle = src.Cycle.parse("2026-10-03T12")
    tails = set()
    for row in TABLE["sources"]:
        if row.get("status") == "refused":
            continue
        argv = src.init_argv(TABLE, row, met=tmp_path / "m", grid=tmp_path / "g", static=tmp_path / "s",
                             vertical_spec=tmp_path / "v", out=tmp_path / "o", cycle=cycle)
        tails.add(tuple(argv[argv.index("--extrap-airtemp"):]))
        assert argv[argv.index("--start-time") + 1] == "2026-10-03_12:00:00"
    assert len(tails) == 1


def test_staged_products_are_used_without_transport(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "gfs.t12z.pgrb2.0p25.f000").write_bytes(b"GRIB-bytes")
    cycle = src.Cycle.parse("2026-10-03T12")
    plans = src.plan_products(src.resolve_source(TABLE, "gfs"), cycle, None)
    out = src.acquire(plans, tmp_path / "products", cycle, staged=[staged])
    assert out[0]["staged_from"].endswith("gfs.t12z.pgrb2.0p25.f000") and out[0]["bytes"] == 10


def test_a_row_with_a_misspelt_key_refuses(tmp_path):
    table = json.loads(src.TABLE_PATH.read_text(encoding="utf-8"))
    table["sources"][0]["frist_guess"] = {}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(src.InitSourceRefusal, match="unknown keys"):
        src.load_table(path)


# ---------------------------------------------------------------------------
# one vertical artifact per mesh, whatever the source's first-guess ladder
# ---------------------------------------------------------------------------
def _door_arguments(tmp_path, nfglevels, nfgsoillevels, use_spechumd="no"):
    import argparse

    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    return argparse.Namespace(
        grid=tmp_path / "grid.nc", static=tmp_path / "static.nc", vertical_spec=spec,
        start_time="2026-10-03_12:00:00", nfglevels=nfglevels, nfgsoillevels=nfgsoillevels,
        use_spechumd=use_spechumd, frac_seaice="yes", extrap_airtemp="constant",
        tsk_seaice_threshold=100.0, landuse_table="MODIFIED_IGBP_MODIS_NOAH")


def test_the_first_guess_ladder_does_not_enter_the_vertical_cache_key(tmp_path, monkeypatch):
    # THE BREAKAGE: with nfglevels in the key, GFS (34), IFS (15), GEFS (32),
    # ERA5 (38) and ICON (19, 8 soil) each rebuilt the same vertical arrays
    # into a separate 5.9 GB file (measured: 35 GB for one 30 km mesh).
    import hexcore.init_door as door
    import hexcore.rust_geometry as rust_geometry
    import hexcore.vertical_spec as vertical_spec

    class _Spec:
        def sha256(self):
            return "spec"

    monkeypatch.setattr(door, "_sha256", lambda path: "digest")
    monkeypatch.setattr(vertical_spec.VerticalSpec, "from_file", classmethod(lambda cls, path: _Spec()))
    monkeypatch.setattr(rust_geometry, "geometry_engine", lambda: (None, "geometry"))
    gfs, _ = door._vertical_cache_key(_door_arguments(tmp_path, 34, 4))
    icon, _ = door._vertical_cache_key(_door_arguments(tmp_path, 19, 8))
    aifs, material = door._vertical_cache_key(_door_arguments(tmp_path, 15, 2, use_spechumd="yes"))
    assert gfs == icon
    assert gfs != aifs  # a YES/NO declaration changes width, so it stays in the key
    assert not set(door.STAMPED_DECLARATIONS) & set(material["declarations"])


def test_the_cached_artifact_is_stamped_with_this_runs_ladder(tmp_path):
    import numpy as np
    from netCDF4 import Dataset

    import hexcore.init_door as door

    artifact = tmp_path / "v.nc"
    with Dataset(artifact, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.setncattr("config_start_time", "2026-10-03_12:00:00")
        dataset.setncattr("config_nfglevels", np.int32(34))
        dataset.setncattr("config_nfgsoillevels", np.int32(4))
    declarations = {"config_start_time": "2026-10-11_12:00:00", "config_nfglevels": 19,
                    "config_nfgsoillevels": 8, "config_use_spechumd": False}
    stamp = door._stamp_run_declarations(artifact, declarations)
    assert stamp["config_nfglevels_before"] == 34 and stamp["config_nfglevels"] == 19
    with Dataset(artifact) as dataset:
        assert dataset.getncattr("config_start_time") == "2026-10-11_12:00:00"
        assert int(dataset.getncattr("config_nfglevels")) == 19
        assert dataset.getncattr("config_nfglevels").dtype == np.int32
        assert int(dataset.getncattr("config_nfgsoillevels")) == 8


# ---------------------------------------------------------------------------
# byte-range subset (fresh-box start timing, 2026-10-16: the whole 512 MB GFS
# object on one HTTPS stream was 73 % of a hex global row's start)
# ---------------------------------------------------------------------------
NCEP_IDX = "\n".join((
    "1:0:d=2026101612:PRMSL:mean sea level:anl:",
    "2:100:d=2026101612:CLMR:1 hybrid level:anl:",
    "3:180:d=2026101612:TMP:500 mb:anl:",
    "4:260:d=2026101612:RH:500 mb:anl:",
    "5:300:d=2026101612:VVEL:500 mb:anl:",
    "6:350:d=2026101612:TMP:80 m above ground:anl:",
    "7:400:d=2026101612:HGT:surface:anl:",
))


def test_ncep_idx_selection_takes_full_matches_and_runs_the_last_record_to_the_end():
    # THE BREAKAGE: a selector that matched by prefix ("TMP:5" taking TMP:50 mb
    # and TMP:500 mb) or that cut the last record of the file short would decode
    # to wrong or truncated fields.
    select = {"records": [["TMP|RH", "[0-9.]+ mb"], ["PRMSL", "mean sea level"], ["HGT", "surface"]]}
    chosen = src.select_ncep_idx_records(NCEP_IDX, select)
    assert [r["record"] for r in chosen] == ["PRMSL:mean sea level", "TMP:500 mb", "RH:500 mb", "HGT:surface"]
    assert [(r["_offset"], r["_length"]) for r in chosen] == [(0, 100), (180, 80), (260, 40), (400, None)]


def test_coalesced_pieces_are_exactly_the_selected_bytes_in_order():
    blob = bytes(range(256)) * 4
    spans = [(0, 99), (180, 259), (260, 299), (400, None)]
    pieces = src.coalesce_spans(spans, piece=32)
    assert all(last is None or last - first + 1 <= 32 for first, last in pieces)
    joined = b"".join(blob[first:(None if last is None else last + 1)] for first, last in pieces)
    wanted = b"".join(blob[first:(None if last is None else last + 1)] for first, last in spans)
    assert joined == wanted
    # touching records collapse into one run before splitting: 180-259 and 260-299 are one run
    assert src.coalesce_spans(spans, piece=1 << 20) == [(0, 99), (180, 299), (400, None)]


def test_an_ncep_idx_index_without_records_refuses():
    row = json.loads(json.dumps(src.resolve_source(TABLE, "gfs")))
    row["products"][0]["index"]["select"] = {"records": []}
    with pytest.raises(src.InitSourceRefusal, match="select.records"):
        src.validate_row(row)


def test_the_gfs_row_fetches_its_records_by_range_on_many_streams():
    plan = src.plan_products(src.resolve_source(TABLE, "gfs"), src.Cycle.parse("2026-10-16T12"), None)[0]
    assert plan.index_format == "ncep-idx" and plan.index_url == plan.url + ".idx"
    assert plan.streams >= 8
