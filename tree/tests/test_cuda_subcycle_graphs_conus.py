"""Validate retained real CONUS graph-run evidence, without opening CUDA.

An explicitly configured evidence run fails on fallback or missing artifacts.
The forecast itself is run separately through its real driver or CLI.
"""

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_cuda_graph_replay import (read_json, validate_graph_replay_receipt,
                                     validate_identity_pair)


def test_real_conus_forecast_replays_and_matches_control():
    path = os.environ.get("GPUWM_HEX_GRAPH_CONUS_RECEIPT")
    if not path:
        pytest.skip("set GPUWM_HEX_GRAPH_CONUS_RECEIPT to validate a completed real CONUS run")
    document = read_json(path)
    assert "conus12km" in str(document.get("profile", "")), "receipt does not identify conus12km"
    replay = validate_graph_replay_receipt(document)
    names = ("GPUWM_HEX_GRAPH_CONUS_BASE_SNAPSHOTS", "GPUWM_HEX_GRAPH_CONUS_SNAPSHOTS",
             "GPUWM_HEX_GRAPH_CONUS_BASE_IDENTITY", "GPUWM_HEX_GRAPH_CONUS_IDENTITY")
    missing = [name for name in names if not os.environ.get(name)]
    assert not missing, f"configured CONUS proof lacks paired evidence paths: {missing}"
    identity = validate_identity_pair(*(read_json(os.environ[name]) for name in names),
                                      steps=replay["steps"])
    assert identity["history_hash_maps_identical"]
    assert identity["final_execution_boundary_identical"]
    assert identity["final_dynamics_tendencies_identical"]
