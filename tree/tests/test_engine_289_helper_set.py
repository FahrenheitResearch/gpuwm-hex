"""The 2.8.9 source row retains the whole helper boundary and mixed-set refusal."""
import json
from pathlib import Path
import pytest
from hexcore import cuda_arwen_helper_scope_v2 as scope


def test_289_source_set_is_complete_and_uses_ordinary_route():
    mode = "engine-2.8.9-source"
    assert set(scope.ADMITTED_SOURCE_SETS[mode]) == set(scope.PUBLISHED_SOURCES)
    assert mode in scope.PUBLISHED_MODES  # ordinary route, no overlay event provider
    assert "source identity only" in scope.RUNTIME_DESCRIPTIONS[mode]
    rows = json.loads(Path(scope.__file__).with_name("data").joinpath("profiles", "engine-helper-sets.v1.json").read_text())["sets"]
    row = next(row for row in rows if row["id"] == mode)
    assert row["engine_commit"] == "7133bb1e2a3c2b3f0058fe2871b36be60f16e2aa"
    assert all(len(digest) == 64 for digest in row["sources"].values())


def test_289_row_does_not_accept_unknown_or_mixed_helper_bytes(tmp_path):
    for name in scope.PUBLISHED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unmeasured helper bytes\n")
    with pytest.raises(RuntimeError, match="unknown or mixed gpuwm execution source set"):
        scope.bind_arwen_helpers_v2(tmp_path)


def test_the_refusal_names_the_engine_and_both_remedies(tmp_path):
    """Plan call 14: the range admits the whole minor of the floor, so the door names the engine it refuses."""
    from hexcore import engine_pin

    for name in scope.PUBLISHED_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unmeasured helper bytes\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = \"gpuwm\"\nversion = \"2.9.1\"\n")
    with pytest.raises(RuntimeError) as refused:
        scope.bind_arwen_helpers_v2(tmp_path)
    message = str(refused.value)
    assert "the installed engine is gpuwm 2.9.1" in message
    assert f"measured on gpuwm {engine_pin.wanted_version()}" in message
    assert f"pip install \"gpuwm=={engine_pin.wanted_version()}\"" in message
    assert "gpuwm-hex release that admits gpuwm 2.9.1" in message
