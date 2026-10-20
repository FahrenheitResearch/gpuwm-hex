"""The public gate index ships complete, relative and consistent.

THE BREAKAGE THIS PREVENTS: the only gate index used to name every deck and
input by an absolute path on the maintainers' hosts, so no public checkout had
a gate index at all.  verification/gates/ carries the same decks, pins and
ratchet with relative paths; this keeps it that way on every push.  The data
files themselves are rebuilt by the reader (data-manifest.json), so only the
structure is checked here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "verification" / "gates"


def _checker():
    spec = importlib.util.spec_from_file_location("check_gate_index", ROOT / "tools" / "check_gate_index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_gate_index_is_relative_and_consistent():
    assert _checker().check(GATES, with_data=False) == []


def test_public_gate_index_names_g1_and_g2_with_the_landed_decks():
    index = json.loads((GATES / "index.json").read_text())
    assert set(index["gates"]) >= {"G1", "G2"}
    assert index["landed_decks"]
    roles = set()
    for entry in index["gates"]["G1"]:
        roles.add(json.loads((GATES / entry["path"]).read_text()).get("role"))
    assert {"dycore", "none-physics"} <= roles


def test_checker_refuses_an_absolute_path_and_a_moved_pin(tmp_path):
    checker = _checker()
    deck = {"schema": "mpas-verification-deck-v1", "id": "d", "process_count": 1,
            "inputs": {"init": {"path": "/somewhere/x.init.nc", "sha256": "a" * 64}},
            "hex_command": ["gpuwm-hex", "forecast", "--x", "/abs/lib.so"]}
    (tmp_path / "decks").mkdir()
    text = json.dumps(deck)
    (tmp_path / "decks" / "d.json").write_text(text)
    (tmp_path / "index.json").write_text(json.dumps({
        "schema": "mpas-gate-index-v1", "landed_decks": ["d", "e"],
        "gates": {"G2": [{"path": "decks/d.json", "sha256": "0" * 64}]}}))
    (tmp_path / "data-manifest.json").write_text(json.dumps({"files": {}}))
    (tmp_path / "ratchet.json").write_text(json.dumps({"schema": "mpas-ratchet-v1", "entries": []}))
    findings = "\n".join(checker.check(tmp_path, with_data=False))
    assert "does not match its pin" in findings
    assert "absolute path /somewhere/x.init.nc" in findings
    assert "model command names an absolute path /abs/lib.so" in findings
    assert "G2 omits landed deck(s) ['e']" in findings
