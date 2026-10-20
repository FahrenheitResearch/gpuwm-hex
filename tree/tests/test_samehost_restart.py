import copy
import pickle
from types import SimpleNamespace
from dataclasses import dataclass

import pytest

from hexcore.drivers import samehost_restart_v841 as restart


@dataclass
class Config:
    dt: float = 30.0


def example():
    expected = restart.identity(host={"config": Config(), "scalar_names": ("qv",)},
                                grid_sha256="grid", layouts_sha256="layouts", part_sha256="part")
    schedule = {"capture_steps": [0, 1, 2], "steps": 2,
                "valid_times": {0: "zero", 1: "one", 2: "two"}}
    rank = {"state": {"time_seconds": 30.0}, "backend_state": {"rain": 1.0},
            "forcing": {"time_seconds": 30.0}}
    checkpoint = {"schema": restart.SCHEMA, "identity": expected, "ranks": [rank, copy.deepcopy(rank)],
                  "step": 1, "valid_time": "one", "dt_seconds": 30.0}
    return expected, schedule, checkpoint


def test_shared_committed_boundary_roundtrip(tmp_path):
    expected, schedule, checkpoint = example()
    receipt = restart.write(tmp_path / "checkpoint.pkl", entries=checkpoint["ranks"],
                            expected=expected, step=1, schedule=schedule, dt=30.0)
    with (tmp_path / "checkpoint.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert receipt["bytes"] > 0
    assert restart.validate(restored, expected=expected, schedule=schedule, rank_count=2) == 1
    assert restored["ranks"][1]["backend_state"] == {"rain": 1.0}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("field,value", [("physics_backend", "other"),
    ("seam_options", {"radiation": "other"}), ("constructor_receipt", {"init": "other"})])
def test_restart_identity_includes_physics_and_constructor(field, value):
    host = {"config": Config(), "scalar_names": ("qv",)}
    kwargs = dict(grid_sha256="grid", layouts_sha256="layouts", part_sha256="part")
    before = restart.identity(host=host, **kwargs)
    host[field] = value
    assert restart.identity(host=host, **kwargs) != before


def test_restart_binds_stochastic_seed_but_allows_restart_workflow_flag():
    from hexcore.stochastic import StochasticConfig
    host = {"config": Config(), "scalar_names": ("qv",)}
    kwargs = dict(grid_sha256="grid", layouts_sha256="layouts", part_sha256="part")
    disabled = restart.identity(host=host, **kwargs)
    assert 'stochastic' not in disabled  # Preserve existing off checkpoints.
    host['stochastic'] = StochasticConfig(sppt=True, seed=711).json()
    enabled = restart.identity(host=host, **kwargs)
    assert enabled != disabled
    host['stochastic'] = StochasticConfig(sppt=True, seed=711, restart_required=True).json()
    assert restart.identity(host=host, **kwargs) == enabled
    host['stochastic'] = StochasticConfig(sppt=True, seed=712).json()
    assert restart.identity(host=host, **kwargs) != enabled


@pytest.mark.parametrize("change,match", [
    (lambda c: c.update(schema="single-card"), "rank-local"),
    (lambda c: c["identity"].update(grid_sha256="other"), "mesh, partition"),
    (lambda c: c["ranks"].pop(), "rank count"),
    (lambda c: c.update(step=2), "before the end"),
    (lambda c: c.update(valid_time="other"), "valid time"),
    (lambda c: c["ranks"][1]["state"].update(time_seconds=0), "committed clock"),
    (lambda c: c["ranks"][1]["forcing"].update(time_seconds=0), "forcing"),
])
def test_reject_incompatible_or_uncommitted_checkpoint(change, match):
    expected, schedule, checkpoint = example()
    checkpoint = copy.deepcopy(checkpoint)
    change(checkpoint)
    with pytest.raises(ValueError, match=match):
        restart.validate(checkpoint, expected=expected, schedule=schedule, rank_count=2)
