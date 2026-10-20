"""Class checkpoint law of active full-step LTS (CPU): what a resume refuses, how class leaves assemble."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.drivers import fullstep_lts_restart_v841 as restart


def layout(owned_cells, halo_cells, n_global):
    l2g = np.asarray(list(owned_cells) + list(halo_cells), dtype=np.int64)
    return SimpleNamespace(n_owned_cells=len(owned_cells), n_local_cells=l2g.size, cell_l2g=l2g,
                           n_owned_edges=0, n_local_edges=0, edge_l2g=np.zeros(0, dtype=np.int64),
                           _global_cells=n_global)


def plan(*layouts_):
    return SimpleNamespace(classes=[SimpleNamespace(index=i, rate=2 ** (len(layouts_) - 1 - i), layout=item)
                                    for i, item in enumerate(layouts_)])


def test_class_seam_leaves_assemble_owned_columns_in_global_order():
    a = layout([0, 2, 4], [1], 6)
    b = layout([1, 3, 5], [0, 2], 6)

    class Backend:
        def __init__(self, value):
            self.value = value

        def restart_state(self):
            return self.value

    def leaves(state):
        return state

    # a [level, column] leaf, a column leaf with native's garbage column, and a metadata leaf
    sa = {"t": np.array([[0, 2, 4, 99]] * 2, dtype=np.float32), "g": np.array([0, 2, 4, 98, -1.0]),
          "meta": np.array([7, 8])}
    sb = {"t": np.array([[1, 3, 5, 99, 99]] * 2, dtype=np.float32), "g": np.array([1, 3, 5, 98, 98, -1.0]),
          "meta": np.array([7, 8])}
    merged = restart.assemble_seam_leaves([{"backend": Backend(sa)}, {"backend": Backend(sb)}], [a, b],
                                          n_global_cells=6, seam_leaf_fields=leaves)
    assert merged["t"].tolist() == [[0, 1, 2, 3, 4, 5]] * 2 and merged["t"].dtype == np.float32
    assert merged["g"].tolist() == [0, 1, 2, 3, 4, 5]
    assert merged["meta"].tolist() == [7, 8]
    sb["meta"] = np.array([7, 9])
    with pytest.raises(ValueError, match="differs between classes"):
        restart.assemble_seam_leaves([{"backend": Backend(sa)}, {"backend": Backend(sb)}], [a, b],
                                     n_global_cells=6, seam_leaf_fields=leaves)


def test_a_resume_refuses_what_would_put_a_class_on_the_wrong_columns():
    a, b = layout([0, 2, 4], [1], 6), layout([1, 3, 5], [0, 2], 6)
    rates = np.array([2, 1, 2, 1, 2, 1], dtype=np.int32)
    schedule = {"capture_steps": {0, 4, 8}, "steps": 8, "valid_times": {4: "T4", 8: "T8", 0: "T0"}}
    good = {"schema": restart.CHECKPOINT_SCHEMA, "dt_seconds": 10.0, "n_cells": 6, "top_rate": 2, "step": 4,
            "valid_time": "T4", "plan_sha256": restart.plan_digest(plan(a, b)),
            "rates_sha256": __import__("hashlib").sha256(rates.tobytes()).hexdigest()}
    kwargs = dict(base_dt=10.0, top=2, rates=rates, plan=plan(a, b), n_cells=6, schedule=schedule)
    assert restart.validate_checkpoint(good, **kwargs) == 4
    with pytest.raises(ValueError, match="single-stack checkpoint"):
        restart.validate_checkpoint({**good, "schema": "gpuwm-hex.forecast-checkpoint/v1"}, **kwargs)
    with pytest.raises(ValueError, match="wrong columns"):
        restart.validate_checkpoint(good, **{**kwargs, "plan": plan(b, a)})
    with pytest.raises(ValueError, match="wrong columns"):
        restart.validate_checkpoint(good, **{**kwargs, "rates": rates[::-1].copy()})
    with pytest.raises(ValueError, match="ORIGINAL run"):
        restart.validate_checkpoint({**good, "step": 8, "valid_time": "T8"}, **kwargs)
    with pytest.raises(ValueError, match="valid time"):
        restart.validate_checkpoint({**good, "valid_time": "T0"}, **kwargs)


def test_a_global_analysis_lands_on_every_local_column_of_a_class():
    from hexcore.driver import DrySavedDiagnostics
    from hexcore.state import PrognosticState

    lay = SimpleNamespace(cell_l2g=np.array([4, 1, 0]), edge_l2g=np.array([2, 0]))
    cells = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    edges = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
    faces = np.arange(3 * 5, dtype=np.float32).reshape(3, 5)
    state = PrognosticState(rho=cells, rho_theta=cells + 100, rho_u=edges, rho_w=faces,
                            scalars=np.stack([cells, cells + 1]), time_seconds=60.0)
    saved = DrySavedDiagnostics(theta_m=cells, exner=cells, density_perturbation=cells,
                                rho_theta_perturbation=cells, pressure_perturbation=cells,
                                normal_velocity=edges, vertical_velocity=faces)
    local, local_saved = restart.slice_global_state(state, saved, lay)
    assert local.rho.tolist() == cells[:, [4, 1, 0]].tolist()
    assert local.rho_u.tolist() == edges[:, [2, 0]].tolist()
    assert local.scalars.shape == (2, 2, 3) and local.time_seconds == 60.0
    assert local_saved.normal_velocity.tolist() == edges[:, [2, 0]].tolist()


def test_the_door_forwards_the_cycling_selections_to_the_driver(tmp_path):
    from hexcore.forecast_door import _cycling_argv

    assert _cycling_argv(SimpleNamespace()) == ()
    checkpoint = tmp_path / "checkpoint.pkl"
    argv = _cycling_argv(SimpleNamespace(checkpoint_hours="1,2", restart_from=checkpoint, da_analysis=None,
                                         jedi_background=True))
    assert argv == ("--checkpoint-hours", "1,2", "--restart-from", str(checkpoint.absolute()), "--jedi-background")
