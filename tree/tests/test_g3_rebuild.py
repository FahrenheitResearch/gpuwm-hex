"""G3 rebuilds the native reference and compares it word for word; its table is never a default.

THE BREAKAGE (plan gate g): G1/G2 compare hex with outputs one strict native
binary wrote once.  G3 (tools/mpas_nightly.sh, tools/native_physics_reference/
g3_rebuild.py) rebuilds that reference from the recipe in this tree; these
host-only checks hold the parts that need no compiler.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

TREE = Path(__file__).resolve().parents[1]
TOOL = TREE / "tools" / "native_physics_reference" / "g3_rebuild.py"


@pytest.fixture(scope="module")
def g3():
    spec = importlib.util.spec_from_file_location("_g3_rebuild_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_an_index_of_another_schema_or_with_no_decks_cannot_pass(g3, tmp_path):
    index = tmp_path / "g3.json"
    index.write_text(json.dumps({"schema": "mpas-gate-index-v1", "decks": [{}]}))
    with pytest.raises(SystemExit, match="not an mpas-g3-index-v1"):
        g3.read_index(index)
    index.write_text(json.dumps({"schema": "mpas-g3-index-v1", "decks": []}))
    with pytest.raises(SystemExit, match="empty gate cannot pass"):
        g3.read_index(index)


def test_the_rebuild_uses_the_recipe_the_gates_binary_came_from(g3):
    text = TOOL.read_text(encoding="utf-8")
    assert '"build_reference.sh"' in text and '"run_forecast.py"' in text
    assert "pinned_tables_manifest" in text
    assert g3.build_dir({"recipe": "gnu-strict"}, Path("/w")) == Path("/w/native/v8.4.2-gnu-strict-atmosphere")


def test_compare_needs_the_same_contract_inputs_and_every_word(g3, tmp_path, monkeypatch):
    reference, work = tmp_path / "reference", tmp_path / "work"
    other = work / "forecasts" / "row"
    for folder in (reference, other):
        folder.mkdir(parents=True)
        (folder / "MANIFEST.json").write_text(json.dumps({
            "deck": "suite-none", "ranks": 1, "namelist_sha256": "n", "streams_sha256": "s",
            "pin_set_hash": "p", "inputs": {"atmosphere_model": str(folder), "x.init.nc": "i"}}))
    (work / "build-record.json").write_text(json.dumps({"tables_identical": True, "publication_differs": []}))
    index_path = tmp_path / "g3.json"
    index_path.write_text("{}")
    index = {"_path": str(index_path), "decks": [{"id": "row", "reference": str(reference)}]}
    import compare_runs

    def fake(reference, other, comparator, pattern, differing=0):
        return {"all_identical": differing == 0, "all_file_bytes_identical": False, "errors": [],
                "files": {"a.nc": {"words": 10, "differing_words": differing, "all_identical": differing == 0}}}

    monkeypatch.setattr(compare_runs, "compare", fake)
    receipt = tmp_path / "G3.json"
    assert g3.compare(index, work, tmp_path / "cmp", receipt) == 0
    assert json.loads(receipt.read_text())["passed"] is True
    monkeypatch.setattr(compare_runs, "compare", lambda *a: fake(*a, differing=1))
    assert g3.compare(index, work, tmp_path / "cmp", receipt) == 1
    monkeypatch.setattr(compare_runs, "compare", fake)
    manifest = json.loads((other / "MANIFEST.json").read_text())
    manifest["inputs"]["x.init.nc"] = "another init"
    (other / "MANIFEST.json").write_text(json.dumps(manifest))
    assert g3.compare(index, work, tmp_path / "cmp", receipt) == 1
    assert json.loads(receipt.read_text())["decks"][0]["inputs_differ"] == ["x.init.nc"]


def test_g3_is_the_nightly_and_runs_the_rebuild_before_g1_g2():
    text = (TREE / "tools" / "mpas_nightly.sh").read_text(encoding="utf-8")
    steps = [text.index(f'"$g3" {step}') for step in ("build", "run", "compare")]
    assert steps == sorted(steps)
    assert steps[-1] < text.index("--gates G1,G2")
