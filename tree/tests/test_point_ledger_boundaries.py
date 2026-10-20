"""The timing hook retains a real child's bytes and original refusal."""
import json
import sys
import pytest
from hexcore import mesh_point


def test_real_child_output_and_boundaries(monkeypatch, tmp_path):
    pytest.importorskip("gpuwm.timeline.ledger", reason="the engine ships no gpuwm.timeline (gpuwm 2.8.7 and earlier)")
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(ledger))
    result = mesh_point._run([sys.executable, "-c", "print('unchanged output')"],
                             what="rw_mpas_mesh")
    assert result.stdout == "unchanged output\n"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [(row["stage"], row["phase"]) for row in rows] == [
        ("prep.place.mesh", "begin"), ("prep.place.mesh", "end")]
    # The shared emitter may add timing metadata such as cpu_s. Check the
    # attributes owned by this boundary without rejecting those measurements.
    expected = {"engine": "rw_mpas_mesh", "terrain_only": False}
    assert all(expected.items() <= row["attrs"].items() for row in rows)


def test_real_child_refusal_survives_logging_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("GPUWM_TIME_LEDGER", str(tmp_path))
    with pytest.raises(mesh_point.PointPlanRefusal, match="rw_mpas_static exited 7"):
        mesh_point._run([sys.executable, "-c", "raise SystemExit(7)"],
                        what="rw_mpas_static")
