"""Every committed subcycle must replay, including the first."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_cuda_graph_replay import (GraphReplayEvidenceError,
                                     validate_graph_replay_receipt,
                                     validate_identity_pair)


def _receipt(steps=15):
    return {"status": "passed", "forecast": {
        "steps_requested": steps, "steps_executed": steps,
        "step_receipt_count": steps, "schedule": {"steps": steps},
        "cuda_graphs": {
            "subcycles": {"captured_subcycles": 1, "replayed_subcycles": steps * 3,
                          "eager_subcycles": 0, "fallback_reasons": {}, "fallbacks": []},
            "fully_replayed_committed_steps": steps,
            "partly_replayed_committed_steps": 0,
        }}}


@pytest.mark.parametrize("steps", [2, 15])
def test_complete_forecast_requires_all_replay_including_first(steps):
    result = validate_graph_replay_receipt(_receipt(steps))
    assert result["replayed_subcycles"] == steps * 3
    assert result["fully_replayed_committed_steps"] == steps


@pytest.mark.parametrize("defect", ["missing_graphs", "missing_replay_count", "all_eager_h2d",
                                   "extra_fallback", "extra_warmup", "partial_run", "one_step",
                                   "wrong_step_partition", "refusal", "boolean_count", "first_warmup"])
def test_incomplete_treatment_cannot_qualify(defect):
    receipt = _receipt()
    forecast = receipt["forecast"]
    graph = forecast["cuda_graphs"]
    subcycles = graph["subcycles"]
    if defect == "missing_graphs":
        del forecast["cuda_graphs"]
    elif defect == "missing_replay_count":
        del subcycles["replayed_subcycles"]
    elif defect == "all_eager_h2d":
        subcycles.update(captured_subcycles=0, replayed_subcycles=0, eager_subcycles=45,
                         fallback_reasons={"capture_failed:cudaErrorStreamCaptureUnsupported:H2D": 1,
                                           "capture_unavailable": 44})
    elif defect == "extra_fallback":
        subcycles["fallback_reasons"]["scratch_memory_exhausted"] = 1
    elif defect == "extra_warmup":
        subcycles.update(replayed_subcycles=43, eager_subcycles=2,
                         fallback_reasons={"layout_warmup": 2})
    elif defect == "first_warmup":
        subcycles.update(replayed_subcycles=44, eager_subcycles=1,
                         fallback_reasons={"layout_warmup": 1})
        graph.update(fully_replayed_committed_steps=14, partly_replayed_committed_steps=1)
    elif defect == "partial_run":
        forecast["steps_executed"] = 14
    elif defect == "one_step":
        receipt = _receipt(1)
    elif defect == "wrong_step_partition":
        graph["fully_replayed_committed_steps"] = 13
    elif defect == "refusal":
        forecast["refusal"] = {"step": 15}
    elif defect == "boolean_count":
        subcycles["captured_subcycles"] = True
    with pytest.raises(GraphReplayEvidenceError):
        validate_graph_replay_receipt(receipt)


def _identity(steps=15):
    digest = "a" * 64
    state = {name: {"sha256": digest} for name in
             ("rho", "rho_theta", "rho_u", "rho_w", "scalars")}
    saved = {name: {"sha256": digest} for name in
             ("theta_m", "exner", "density_perturbation", "rho_theta_perturbation",
              "pressure_perturbation", "normal_velocity", "vertical_velocity")}
    return {"rc": 0, "complete_run_identity_eligible": True,
            "completed_composite_steps": steps,
            "execution_boundary": {
                "atmosphere": {"sha256": digest, "state": {"fields": state},
                               "saved_diagnostics": {"fields": saved}},
                "backend": {"sha256": digest, "arrays": {"state/tsk": {"sha256": digest}}}},
            "dynamics_tendencies": {"sha256": digest, "arrays": {
                "rthdynten": {"sha256": digest}, "rqvdynten": {"sha256": digest}}}}


def test_final_state_and_every_history_hash_are_required():
    snapshots = {"projection": {"0": {"theta": "a" * 64}, "15": {"theta": "b" * 64}},
                 "q2": {"0": "c" * 64, "15": "d" * 64}}
    identity = _identity()
    assert validate_identity_pair(snapshots, deepcopy(snapshots), identity,
                                  deepcopy(identity), steps=15)["final_execution_boundary_identical"]
    changed = deepcopy(snapshots)
    changed["projection"]["15"]["theta"] = "c" * 64
    with pytest.raises(GraphReplayEvidenceError, match="history"):
        validate_identity_pair(snapshots, changed, identity, identity, steps=15)
    changed_identity = deepcopy(identity)
    changed_identity["dynamics_tendencies"]["sha256"] = "b" * 64
    with pytest.raises(GraphReplayEvidenceError, match="dynamics_tendencies"):
        validate_identity_pair(snapshots, snapshots, identity, changed_identity, steps=15)


def test_equal_partial_or_empty_identity_does_not_qualify():
    snapshots = {"projection": {"0": {"theta": "a" * 64}, "15": {"theta": "b" * 64}}, "q2": {}}
    identity = _identity()
    identity["complete_run_identity_eligible"] = False
    with pytest.raises(GraphReplayEvidenceError, match="complete-run"):
        validate_identity_pair(snapshots, snapshots, identity, identity, steps=15)
    identity = _identity()
    identity["execution_boundary"]["backend"] = {}
    with pytest.raises(GraphReplayEvidenceError):
        validate_identity_pair(snapshots, snapshots, identity, identity, steps=15)


def test_explicit_conus_evidence_path_is_checked_without_skip(monkeypatch, tmp_path):
    from test_cuda_subcycle_graphs_conus import test_real_conus_forecast_replays_and_matches_control

    monkeypatch.setenv("GPUWM_HEX_GRAPH_CONUS_RECEIPT", str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError):
        test_real_conus_forecast_replays_and_matches_control()
