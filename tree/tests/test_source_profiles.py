"""The source profile table holds to its schema and to the tree.

THE BREAKAGE THESE PREVENT: a profile row that says a dycore setting is
refused after the refusal was lifted (or admitted while the refusal still
stands) would let a receipt claim a configuration hex does not run, and a
pin with a short or wrong commit would look like a pin while pinning
nothing.  Each test below holds one of those links.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import re
import warnings
from pathlib import Path

import pytest

from hexcore.source_profiles import (
    AUDIT_CLASSES,
    SourceProfileError,
    load_source_profiles,
    source_profile,
)
from hexcore.vertical_spec import VerticalSpec

TREE = Path(__file__).resolve().parents[1]
TABLE = TREE / "src" / "hexcore" / "data" / "profiles" / "source-profiles.v1.json"


def test_the_packaged_table_loads_and_names_its_rows():
    profiles = load_source_profiles()
    assert set(profiles) == {"mpas-a-v8.4.2", "rrfs-v2", "gfs-public-quarter-degree"}
    assert profiles["mpas-a-v8.4.2"]["model_source"]["commit"] == (
        "f34984b2c89353cf31c235b08d3b4acd96b37bb3"
    )


def test_both_model_candidates_are_pinned_with_their_physics():
    row = source_profile("rrfs-v2")
    by_id = {c["id"]: c for c in row["model_candidates"]}
    assert set(by_id) == {"workflow-pin", "fork-tip"}
    assert by_id["workflow-pin"]["registry_version"] == "8.3.1-noaa"
    assert by_id["fork-tip"]["registry_version"] == "8.4.1-noaa"
    pin = {s["path"]: s["commit"] for s in by_id["workflow-pin"]["physics_submodules"]}
    tip = {s["path"]: s["commit"] for s in by_id["fork-tip"]["physics_submodules"]}
    # The fork tip adds SPPT and moves exactly two physics pins.
    assert set(tip) - set(pin) == {"src/core_atmosphere/stochastic_physics"}
    moved = {path for path in pin if pin[path] != tip[path]}
    assert moved == {
        "src/core_atmosphere/physics/physics_noaa/MYNN-EDMF",
        "src/core_atmosphere/physics/physics_noaa/SMOKE",
    }


def test_the_vertical_declaration_is_the_level_file_the_row_names():
    row = source_profile("rrfs-v2")["vertical"]
    spec = VerticalSpec.from_file(TREE / row["hex_vertical_spec"])
    assert spec.sha256() == row["hex_vertical_spec_sha256"]
    assert spec.scheme == "specified"
    assert spec.n_vert_levels == row["layers"] == 59
    assert len(spec.specified_interfaces_m) == row["interfaces"] == 60
    assert spec.ztop_m == row["top_m"] == spec.specified_interfaces_m[-1]


def test_the_audit_covers_both_candidates_with_known_classes():
    audit = source_profile("rrfs-v2")["dycore_audit"]
    assert {row["candidate"] for row in audit["rows"]} == {"workflow-pin", "fork-tip"}
    assert {row["class"] for row in audit["rows"]} <= AUDIT_CLASSES
    for row in audit["rows"]:
        assert row["routines"] and row["difference"] and row["at_rrfs_settings"]


def test_the_audit_source_sites_link_to_the_pinned_candidate_and_upstream():
    profile = source_profile("rrfs-v2")
    upstream = source_profile("mpas-a-v8.4.2")["model_source"]
    sources = {"upstream": upstream}
    sources.update({candidate["id"]: candidate for candidate in profile["model_candidates"]})
    for row in profile["dycore_audit"]["rows"]:
        assert row["source_sites"], row["item"]
        for reference in row["source_sites"]:
            assert reference["routine"] in row["routines"]
            assert reference["source"] in ("upstream", row["candidate"])
            source = sources[reference["source"]]
            assert reference["commit"] == source["commit"]
            path, line = reference["site"].rsplit(":", 1)
            assert path.startswith("src/") and int(line) > 0
            assert reference["url"] == (
                f"{source['repository']}/blob/{source['commit']}/{path}#L{line}"
            )


def test_a_short_commit_is_refused_by_name(tmp_path):
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    table["profiles"][1]["model_candidates"][0]["commit"] = "1d6e370e"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(SourceProfileError, match="40-character"):
        load_source_profiles(bad)


def test_an_unknown_profile_is_refused_by_name():
    with pytest.raises(SourceProfileError, match="no source profile"):
        source_profile("no-such-profile")


def test_shipped_table_text_has_no_dashes_of_the_banned_kind():
    text = TABLE.read_text(encoding="utf-8")
    assert "—" not in text and "–" not in text


def test_the_workflow_pin_carries_the_da_cycling_contract():
    # Both candidates start warm cycles from the da_state stream when
    # config_do_DAcycling is set, and the public workflow sets it on warm
    # cycles; an audit that recorded it for one candidate only would let a
    # stage H6 claim skip the pin the public build actually compiles.
    audit = source_profile("rrfs-v2")["dycore_audit"]["rows"]
    da = {row["candidate"]: row for row in audit if row["item"] == "DA cycling read and write"}
    assert set(da) == {"workflow-pin", "fork-tip"}
    assert all(row["class"] == "port" for row in da.values())


def _broken_table(tmp_path, edit):
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    edit(table["profiles"][1])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(table), encoding="utf-8")
    return bad


def test_a_refused_lane_without_its_reason_is_refused_by_name(tmp_path):
    def edit(row):
        setting = row["hex_dycore_settings"][0]
        setting["hex_today"] = "refused"
        setting["lanes"]["namelist-dry"] = {
            "state": "refused", "site": "src/hexcore/driver.py",
            "function": "DryDycoreConfig.validate",
        }

    with pytest.raises(SourceProfileError, match="a refusal needs 'reason'"):
        load_source_profiles(_broken_table(tmp_path, edit))


def test_a_summary_that_disagrees_with_its_lanes_is_refused_by_name(tmp_path):
    def edit(row):
        setting = row["hex_dycore_settings"][0]
        setting["hex_today"] = "refused" if setting["hex_today"] == "admitted" else "admitted"

    with pytest.raises(SourceProfileError, match="disagrees with its lanes"):
        load_source_profiles(_broken_table(tmp_path, edit))


# ---------------------------------------------------------------------------
# The doors.  Each setting row names, per hex lane, the refusal that fires
# first (knob, reason, file and function).  The tests below make the lanes
# say it: the namelist door reads the public lines 1-26 as written, and the
# forecast-column lane takes each knob alone.  When a port lands, its rows and
# these tests move together; a lifted refusal with an unmoved row fails here.
# ---------------------------------------------------------------------------

_KEY = re.compile(r"^\s*(config_\w+)\s*=", re.MULTILINE)


def _profile():
    return source_profile("rrfs-v2")


def _fragment() -> str:
    record = _profile()["namelist"]["lines_1_26"]
    return (TREE / record["hex_copy"]).read_text(encoding="utf-8")


def _fragment_for(mesh: str) -> str:
    holes = _profile()["namelist"]["lines_1_26"]["placeholders"]
    return (
        _fragment()
        .replace("${dt}", holes[mesh]["dt"])
        .replace("${substeps}", holes[mesh]["substeps"])
        .replace("${start_time}", holes["start_time"])
        .replace("${run_duration}", holes["run_duration"])
    )


def _rows_for(mesh: str):
    return [
        row
        for row in _profile()["hex_dycore_settings"]
        if row.get("kind") != "declaration" and row.get("mesh", mesh) == mesh
    ]


def _raised_at(error: BaseException) -> tuple[str, str]:
    """(file relative to the tree, function qualname) that raised the refusal."""

    frames = []
    tb = error.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code)
        tb = tb.tb_next
    code = frames[-1]
    if code.co_name == "_refuse" and len(frames) > 1:
        code = frames[-2]
    path = Path(code.co_filename).resolve()
    return path.relative_to(TREE).as_posix(), code.co_qualname


def _reason_matches(entry, reason: str) -> bool:
    if entry.get("match", "exact") == "prefix":
        return reason.startswith(entry["reason"])
    return reason == entry["reason"]


def _config_type(lane: str):
    dotted = _profile()["hex_lanes"][lane]["config_type"]
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def test_the_namelist_copy_is_the_pinned_bytes():
    record = _profile()["namelist"]["lines_1_26"]
    data = (TREE / record["hex_copy"]).read_bytes()
    assert len(data) == record["bytes"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_every_key_of_the_public_lines_has_a_row():
    keys = set(_KEY.findall(_fragment()))
    rows = {row["knob"] for row in _profile()["hex_dycore_settings"]}
    assert len(keys) == 20
    assert keys <= rows, f"keys with no setting row: {sorted(keys - rows)}"
    for mesh in ("conus3km", "conus12km"):
        assert {row["knob"] for row in _rows_for(mesh)} >= keys


@pytest.mark.parametrize("mesh", ["conus3km", "conus12km"])
def test_the_namelist_door_meets_exactly_the_refusals_the_rows_name(mesh):
    from hexcore.config import dry_dycore_config_from_namelist
    from hexcore.config_v841 import LegacyEpssmIgnored
    from hexcore.errors import ConfigurationRefusal

    text = _fragment_for(mesh)
    met: dict[str, tuple[str, tuple[str, str]]] = {}
    for _ in range(len(_KEY.findall(text)) + 1):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                dry_dycore_config_from_namelist(text, source_release="v8.4.1")
            break
        except ConfigurationRefusal as refusal:
            assert refusal.knob not in met, f"{refusal.knob} refused twice"
            met[refusal.knob] = (refusal.reason, _raised_at(refusal))
            line = re.compile(
                r"^\s*" + re.escape(refusal.knob) + r"\s*=.*\n", re.MULTILINE | re.IGNORECASE
            )
            text, removed = line.subn("", text)
            assert removed == 1, f"the door refused {refusal.knob!r}, which is not a key of the lines"
    else:
        pytest.fail("the namelist door never admitted the remaining lines")

    expected = {
        row["knob"]: row["lanes"]["namelist-dry"]
        for row in _rows_for(mesh)
        if row["lanes"]["namelist-dry"]["state"] == "refused"
    }
    assert set(met) == set(expected)
    for knob, (reason, (site, function)) in met.items():
        entry = expected[knob]
        assert _reason_matches(entry, reason), f"{knob}: door says {reason!r}, row says {entry['reason']!r}"
        assert (site, function) == (entry["site"], entry["function"]), knob

    warned = {type(w.message).__name__ for w in caught}
    for row in _rows_for(mesh):
        warning = row["lanes"]["namelist-dry"].get("warning")
        if warning:
            assert warning in warned, f"{row['knob']}: no {warning} on the admitted read"
    assert LegacyEpssmIgnored.__name__ in warned


def _column_cases():
    cases = []
    for row in _profile()["hex_dycore_settings"]:
        entry = row["lanes"].get("forecast-column")
        if entry is None:
            continue
        requested = row["requested"]
        if row.get("mesh"):
            values = {row["mesh"]: requested}
        elif isinstance(requested, dict):
            values = dict(requested)
        else:
            values = {"both": requested}
        for mesh, value in values.items():
            resolved = dict(entry)
            resolved["context"] = {
                **entry.get("context", {}),
                **entry.get("context_by_mesh", {}).get(mesh, {}),
            }
            cases.append(pytest.param(row, resolved, value, id=f"{row['knob']}-{mesh}"))
    return cases


@pytest.mark.parametrize("row,entry,value", _column_cases())
def test_the_forecast_column_lane_meets_exactly_the_state_each_row_names(row, entry, value):
    import dataclasses

    from hexcore.errors import ConfigurationRefusal

    config_type = _config_type("forecast-column")
    knob = row["knob"]
    if entry["state"] == "no_field":
        assert knob not in {field.name for field in dataclasses.fields(config_type)}
        return
    groups = _profile()["namelist"]
    for group in ("nhyd_model", "damping", "limited_area"):
        if knob in groups[group]:
            assert groups[group][knob] == value, f"{knob}: row and namelist record disagree"
    values = {**entry.get("context", {}), knob: value}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry["state"] == "refused":
            with pytest.raises(ConfigurationRefusal) as raised:
                config_type(**values).validate()
            refusal = raised.value
            assert refusal.knob == knob
            assert _reason_matches(entry, refusal.reason), (
                f"{knob}: lane says {refusal.reason!r}, row says {entry['reason']!r}"
            )
            assert _raised_at(refusal) == (entry["site"], entry["function"])
        else:
            config_type(**values).validate()
    if entry.get("warning"):
        assert entry["warning"] in {type(w.message).__name__ for w in caught}


def _function_node(tree: ast.AST, qualname: str):
    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{prefix}{child.name}"
                if name == qualname and not isinstance(child, ast.ClassDef):
                    return child
                found = walk(child, f"{name}.")
                if found is not None:
                    return found
        return None

    return walk(tree, "")


def _strings(node: ast.AST) -> list[str]:
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _layers():
    for row in _profile()["hex_dycore_settings"]:
        for lane, entry in row["lanes"].items():
            if entry["state"] == "refused":
                yield pytest.param(row["knob"], entry, True, id=f"{row['knob']}-{row.get('mesh', '')}-{lane}-first")
        for index, layer in enumerate(row.get("further_layers", [])):
            yield pytest.param(
                layer.get("knob", row["knob"]),
                layer,
                False,
                id=f"{row['knob']}-{row.get('mesh', '')}-{layer['lane']}-{index}",
            )


@pytest.mark.parametrize("knob,layer,first", list(_layers()))
def test_every_refusal_layer_is_in_its_file_and_function(knob, layer, first):
    # The CUDA layers cannot be reached without a card, and a layer behind
    # the first refusal cannot be reached at all until that one is lifted, so
    # each is held by source: its function exists, names the knob, and holds
    # the reason text.  A shared reason ("...lane is exact") is therefore held
    # per knob, inside the one function that refuses it.
    tree = ast.parse((TREE / layer["site"]).read_text(encoding="utf-8"))
    function = _function_node(tree, layer["function"])
    assert function is not None, f"{layer['site']} has no {layer['function']}"
    if first:
        # The first refusal on a lane is reached by the door tests above,
        # which assert its knob, reason and raising function from the live
        # refusal; here it only has to name a function that exists.
        return
    assert knob in _strings(function), f"{layer['function']} no longer names {knob}"
    if layer.get("match") == "prefix":
        # A computed reason (the timestep anchor registry builds it); the
        # dynamic door test above holds its words, and the function must
        # still hand the registry's message to the refusal.
        assert "dt_admission" in ast.unparse(function)
        return
    assert any(layer["reason"] in text for text in _strings(function)), (
        f"{layer['function']} no longer refuses with {layer['reason']!r}"
    )
