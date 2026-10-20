"""Byte receipt comparison keeps missing boundaries and metadata visible."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from hexcore.drivers.samehost_checks_v841 import (
    compare_boundary_files, compare_boundary_records, read_boundary_jsonl,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from compare_2gpu_invariance_v841 import compare_boundary_records as previous_compare  # noqa: E402


def _record():
    return {"atmosphere": {"model_time_seconds": 60., "state": {"fields": {"rho": {"sha256": "a"}}},
                           "saved_diagnostics": {"fields": {"theta_m": {"sha256": "b"}}}},
            "backend": {"arrays": {"soil": {"sha256": "c"}},
                        "scalars": {"identity/n_columns": 12, "schema": "v1"}}}


@pytest.mark.parametrize("mutation", [None, "atmosphere", "array", "metadata", "expected_metadata", "artifact"])
def test_packaged_comparator_is_exactly_the_previous_byte_gate(mutation):
    reference = _record()
    candidate = deepcopy(reference)
    if mutation == "atmosphere":
        candidate["atmosphere"]["state"]["fields"]["rho"]["sha256"] = "changed"
    elif mutation == "array":
        candidate["backend"]["arrays"]["soil"]["sha256"] = "changed"
    elif mutation == "metadata":
        candidate["backend"]["scalars"]["schema"] = "v2"
    elif mutation == "expected_metadata":
        candidate["backend"]["scalars"]["identity/n_columns"] = 6
    elif mutation == "artifact":
        candidate["batch_order_artifacts"] = {"soil": {"rank0_sha256": "artifact"}}
        candidate["backend"]["arrays"].pop("soil")
    assert compare_boundary_records(candidate, reference) == previous_compare(candidate, reference)


def _write(path, steps):
    path.write_text("".join(json.dumps({"step": step, **_record()}) + "\n" for step in steps))


def test_all_boundaries_and_final_state_are_required(tmp_path):
    reference, candidate = tmp_path / "ref.jsonl", tmp_path / "candidate.jsonl"
    _write(reference, [0, 1, 2])
    _write(candidate, [0, 1, 2])
    assert compare_boundary_files(reference, candidate, require_steps=3)["status"] == "PASS"
    _write(candidate, [0, 1])
    result = compare_boundary_files(reference, candidate)
    assert result["status"] == "INSUFFICIENT"
    assert result["reference_steps_missing_in_candidate"] == [2]
    assert not result["final_state_identical"]


def test_jsonl_reader_refuses_duplicate_boundary(tmp_path):
    path = tmp_path / "duplicated.jsonl"
    _write(path, [0, 0])
    with pytest.raises(ValueError, match="duplicate boundary for step 0"):
        read_boundary_jsonl(path)


def _partitioned_record():
    record=_record()
    record['backend']['arrays'].pop('soil')
    record['batch_order_artifacts']={'soil':{'dtype':'<f4','kind':'cell','law':'unconsumed staggered carrier',
                                           'rank0_shape':[2,1,7],'rank0_sha256':'words'}}
    return record


def test_partitioned_reference_compares_its_own_complete_artifact_words():
    record=_partitioned_record()
    assert compare_boundary_records(record,deepcopy(record))['state_invariant']


@pytest.mark.parametrize('mutation',['missing','extra','words','shape','law','overlap'])
def test_partitioned_reference_refuses_missing_or_mutated_artifact(mutation):
    reference=_partitioned_record();candidate=deepcopy(reference)
    if mutation=='missing':candidate['batch_order_artifacts']={}
    elif mutation=='extra':candidate['batch_order_artifacts']['extra']=deepcopy(candidate['batch_order_artifacts']['soil'])
    elif mutation=='words':candidate['batch_order_artifacts']['soil']['rank0_sha256']='changed'
    elif mutation=='shape':candidate['batch_order_artifacts']['soil']['rank0_shape']=[2,1,8]
    elif mutation=='law':candidate['batch_order_artifacts']['soil']['law']='changed'
    elif mutation=='overlap':candidate['backend']['arrays']['soil']={'sha256':'words'}
    verdict=compare_boundary_records(candidate,reference)
    assert not verdict['state_invariant']
    assert any(diff['group']=='backend/artifacts' for diff in verdict['state_diffs'])
