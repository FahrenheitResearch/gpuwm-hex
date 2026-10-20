"""The whole-array edgesOnCell_sign is the per-slot loop's array, refusals included."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.acoustic import edge_signs_on_cells


def _loop(counts, edges, cells):
    signs = np.zeros(edges.shape, dtype=np.float64)
    for cell in range(counts.size):
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            if edge >= 0:
                signs[cell, slot] = 1.0 if cells[edge, 0] == cell else -1.0
    return signs


def _mesh(counts, edges, cells):
    return SimpleNamespace(arrays={"nEdgesOnCell": counts, "edgesOnCell": edges, "cellsOnEdge": cells})


@pytest.mark.parametrize("seed", range(6))
def test_matches_the_per_slot_loop(seed):
    rng = np.random.default_rng(seed)
    n_cells, n_edges, width = 300, 700, 7
    counts = rng.integers(0, width + 1, size=n_cells).astype(np.int32)
    edges = rng.integers(-1, n_edges, size=(n_cells, width)).astype(np.int32)
    cells = rng.integers(-1, n_cells, size=(n_edges, 2)).astype(np.int32)
    cells[rng.integers(0, n_edges, size=200), 0] = rng.integers(0, n_cells, size=200)
    expected = _loop(counts.astype(np.int64), edges.astype(np.int64), cells.astype(np.int64))
    actual = edge_signs_on_cells(_mesh(counts, edges, cells))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_a_count_past_the_row_refuses_like_the_loop():
    counts = np.array([3], dtype=np.int32)
    edges = np.zeros((1, 2), dtype=np.int32)
    cells = np.zeros((1, 2), dtype=np.int32)
    with pytest.raises(IndexError):
        _loop(counts, edges, cells)
    with pytest.raises(IndexError):
        edge_signs_on_cells(_mesh(counts, edges, cells))


def test_an_edge_past_cells_on_edge_refuses_like_the_loop():
    counts = np.array([1], dtype=np.int32)
    edges = np.array([[5]], dtype=np.int32)
    cells = np.zeros((2, 2), dtype=np.int32)
    with pytest.raises(IndexError):
        _loop(counts, edges, cells)
    with pytest.raises(IndexError):
        edge_signs_on_cells(_mesh(counts, edges, cells))
