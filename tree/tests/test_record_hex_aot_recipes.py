"""Source export must follow final plan options and retain cached metadata."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest
from hexcore import kernel_pack, kernel_plan

path = Path(__file__).resolve().parents[1] / "tools/record_hex_aot_recipes.py"
spec = importlib.util.spec_from_file_location("record_hex_aot_recipes", path)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def documents():
    identity = {"cupy": "stub", "nvrtc": [13, 4], "cache_key": "stub"}
    cached = dict(source="cached source", options=["--std=c++17", "--fmad=false"],
                  name_expressions=None, extra_source="header bytes", backend="nvrtc")
    final = dict(source="cached source", options=["--std=c++17", "-ftz=true"],
                 name_expressions=None, filename="a123.cubin.cu", method=None,
                 arch="device", cache_in_memory=False)
    direct = dict(source="direct source", options=["--ftz=false", "--fmad=false"],
                  name_expressions=None, filename="land.cu", method=None,
                  arch="device", cache_in_memory=True)
    plan = dict(schema=kernel_plan.SCHEMA, compiler=identity,
                entries=[dict(arguments=a, sha256=kernel_plan._key(a)) for a in [final, direct]])
    manifest = dict(schema=kernel_pack.SCHEMA, compiler=identity,
                    entries=[dict(arguments=cached, sha256=kernel_pack.digest(cached))])
    return plan, manifest


def normalize(options):
    return [o for o in options if o != "--fmad=false"]


def test_export_final_cached_options_and_exact_direct_options():
    plan, manifest = documents()
    rows, bindings = export.plan_entries(plan, manifest, normalize)
    assert rows[0]["options"] == ["--std=c++17"]
    assert rows[0]["extra_source"].startswith("sha256:")
    assert rows[1]["options"] == ["--ftz=false", "--fmad=false"]
    assert rows[1]["filename"] == "land.cu"
    assert rows[1]["cache_in_memory"] is True
    assert [x["route"] for x in bindings] == ["cached", "direct-nvrtc"]
    assert [x["request_sha256"] for x in bindings] == [x["sha256"] for x in plan["entries"]]


@pytest.mark.parametrize("which", ["plan", "manifest"])
def test_changed_request_without_digest_update_refused(which):
    plan, manifest = documents()
    doc = plan if which == "plan" else manifest
    doc["entries"][0]["arguments"]["source"] += " changed"
    with pytest.raises(ValueError, match="digest mismatch"):
        export.plan_entries(plan, manifest, normalize)


def test_mixed_compiler_manifest_refused():
    plan, manifest = documents()
    manifest["compiler"] = dict(manifest["compiler"], nvrtc=[12, 9])
    with pytest.raises(ValueError, match="compiler identities differ"):
        export.plan_entries(plan, manifest, normalize)


def test_no_guess_for_missing_or_wrong_numerical_companion():
    plan, manifest = documents()
    with pytest.raises(ValueError, match="found 0"):
        export.plan_entries(plan, manifest, lambda options: options)
    manifest["entries"] = []
    with pytest.raises(ValueError, match="found 0"):
        export.plan_entries(plan, manifest, normalize)


def test_ambiguous_extra_source_headers_refused():
    plan, manifest = documents()
    other = copy.deepcopy(manifest["entries"][0]["arguments"])
    other["extra_source"] = "different headers"
    manifest["entries"].append(dict(arguments=other, sha256=kernel_pack.digest(other)))
    with pytest.raises(ValueError, match="found 2"):
        export.plan_entries(plan, manifest, normalize)


@pytest.mark.parametrize("field,value,message", [("method", "ptx", "unsupported compiler method"),
                                                 ("name_expressions", ["kernel"], "name expressions")])
def test_unsupported_direct_route_refused(field, value, message):
    plan, manifest = documents()
    row = plan["entries"][1]
    row["arguments"][field] = value
    row["sha256"] = kernel_plan._key(row["arguments"])
    with pytest.raises(ValueError, match=message):
        export.plan_entries(plan, manifest, normalize)


def test_environment_requires_recorded_controls_and_refuses_debug():
    plan, _ = documents()
    with pytest.raises(ValueError, match="lacks a recorded"):
        export.plan_environment(plan)
    env = {k: "" for k in ("GPUWM_HEX_NUMERICS", "GPUWM_NUMERICS", "GPUWM_WRF_EXACT", "CUPY_CUDA_COMPILE_WITH_DEBUG")}
    env["GPUWM_NUMERICS"] = "strict"
    plan["tag_text"] = json.dumps(dict(environment=env))
    assert export.plan_environment(plan) == env
    env["CUPY_CUDA_COMPILE_WITH_DEBUG"] = "1"
    plan["tag_text"] = json.dumps(dict(environment=env))
    with pytest.raises(ValueError, match="debug compilation"):
        export.plan_environment(plan)
