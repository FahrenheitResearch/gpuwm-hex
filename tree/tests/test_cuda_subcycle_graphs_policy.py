"""The graph selection contract used by the forecast owner."""
from hexcore.cuda_subcycle_graphs import GRAPH_EAGER_FROM_CELLS, graph_policy, graphs_enabled


def test_default_replays_and_explicit_zero_selects_eager(monkeypatch):
    monkeypatch.delenv("GPUWM_HEX_CUDA_GRAPHS", raising=False)
    assert graphs_enabled()
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "0")
    assert not graphs_enabled()
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "1")
    assert graphs_enabled()


def test_auto_runs_global_meshes_eager_and_explicit_settings_win(monkeypatch):
    # Census items 5 and 11: the graph's staged inputs and retained scratch
    # are not held where replay buys nothing (x1.40962 and up).
    monkeypatch.delenv("GPUWM_HEX_CUDA_GRAPHS", raising=False)
    assert GRAPH_EAGER_FROM_CELLS == 40_962
    assert graphs_enabled(15_755)
    assert graphs_enabled(10_242)
    assert not graphs_enabled(40_962)
    assert not graphs_enabled(655_362)
    assert graph_policy(655_362) == {"setting": "auto", "cells": 655_362,
                                     "eager_from_cells": 40_962, "replay": False}
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "auto")
    assert not graphs_enabled(163_842)
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "1")
    assert graphs_enabled(655_362)
    monkeypatch.setenv("GPUWM_HEX_CUDA_GRAPHS", "0")
    assert not graphs_enabled(10_242)
