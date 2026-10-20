"""Compact slot layouts for padded mesh tables (hexcore.cuda_backend.compaction).

THE BREAKAGE THIS PREVENTS
--------------------------
The device mesh tables are stored with the slot stride the mesh actually uses
(6 cell slots instead of maxEdges=10, 10 edge-stencil slots instead of
maxEdges2=20, 10 advection cells instead of FIFTEEN).  That is bitwise only
while two things hold, and these tests pin both on host arrays:

1. the kept stride is never below the largest count, so every
   ``slot < count`` loop still reads the same words;
2. a table is compacted only when every dropped slot holds ONE bit pattern,
   so the host layout (what the dual-run execution fingerprints hash) is
   rebuilt exactly.  A tail of -0.0 or a stray nonzero must refuse.

The knobs that turn each compaction off for A/B are pinned as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from hexcore.cuda_backend import compaction


def test_used_width_is_the_largest_count_clipped_to_the_stored_width():
    assert compaction.used_width(np.array([5, 6, 6, 0]), 10) == 6
    assert compaction.used_width(np.array([12, 3]), 10) == 10
    assert compaction.used_width(np.array([0, 0]), 10) == 1
    assert compaction.used_width(np.array([], dtype=np.int32), 10) == 10


def test_uniform_tail_accepts_one_constant_and_refuses_anything_else():
    index = np.array([[3, 4, -1, -1], [5, -1, -1, -1]], dtype=np.int32)
    assert compaction.uniform_tail(index, 2, axis=1) == -1
    weights = np.zeros((2, 4), dtype=np.float32)
    weights[:, :2] = 0.25
    assert compaction.uniform_tail(weights, 2, axis=1) == 0.0
    weights[1, 3] = 1.0e-30
    assert compaction.uniform_tail(weights, 2, axis=1) is None


def test_a_negative_zero_tail_is_not_taken_for_zeros():
    values = np.zeros((1, 4), dtype=np.float32)
    values[0, 3] = -0.0
    assert compaction.uniform_tail(values, 2, axis=1) is None


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.dtype(">f4")])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_parallel_tail_preserves_words_on_strided_axes(monkeypatch, dtype, axis):
    from hexcore import host_parallel

    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 1)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 2)
    array = np.full((13, 17, 19), -0.0 if np.dtype(dtype).kind == "f" else -1,
                    dtype=dtype)
    for value in (array, np.asfortranarray(array), array[::-1, ::2, ::-1]):
        tail = np.take(value, np.arange(2, value.shape[axis]), axis=axis)
        expected = tail.flat[0]
        actual = compaction.uniform_tail(value, 2, axis=axis)
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        index = [-1] * value.ndim
        value[tuple(index)] = 1
        assert compaction.uniform_tail(value, 2, axis=axis) is None
        value[tuple(index)] = expected


def test_parallel_tail_keeps_nan_payloads_and_does_not_copy_tail(monkeypatch):
    from hexcore import host_parallel

    monkeypatch.setenv(host_parallel.ENVIRONMENT, "4")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 1)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 2)
    words = np.full((13, 17, 10), 0x7fc01234, dtype=np.uint32)
    array = words.view(np.float32)
    with monkeypatch.context() as guard:
        guard.setattr(np, "take", lambda *a, **k: pytest.fail("padding copy"))
        guard.setattr(np, "ascontiguousarray", lambda *a, **k: pytest.fail("padding pack"))
        actual = compaction.uniform_tail(array, 6, axis=2)
    assert np.asarray(actual).view(np.uint32) == 0x7fc01234
    words[-1, -1, -1] = 0x7fc01235
    assert compaction.uniform_tail(array, 6, axis=2) is None
    assert compaction.uniform_tail(array, 10, axis=2) == 0
    # Retain np.take's wrapped-index behavior for a negative keep value.
    words.fill(0x80000000)
    assert np.signbit(compaction.uniform_tail(array, -1, axis=2))


def test_compact_host_keeps_the_leading_slots_on_any_axis_contiguously():
    metrics = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    kept = compaction.compact_host(metrics, 6, axis=2)
    assert kept.shape == (2, 3, 6) and kept.flags.c_contiguous
    np.testing.assert_array_equal(kept, metrics[:, :, :6])
    coeffs = np.arange(4 * 10 * 3, dtype=np.float32).reshape(4, 10, 3)
    kept = compaction.compact_host(coeffs, 6, axis=1)
    assert kept.shape == (4, 6, 3)
    assert kept.tobytes() == np.ascontiguousarray(coeffs[:, :6, :]).tobytes()


@pytest.mark.parametrize("table", compaction.TABLES)
def test_each_table_has_its_own_off_knob_and_the_master_knob_turns_all_off(
    monkeypatch, table
):
    monkeypatch.delenv("GPUWM_HEX_MESH_COMPACT", raising=False)
    monkeypatch.delenv(f"GPUWM_HEX_MESH_COMPACT_{table}", raising=False)
    assert compaction.compaction_enabled(table)
    monkeypatch.setenv(f"GPUWM_HEX_MESH_COMPACT_{table}", "0")
    assert not compaction.compaction_enabled(table)
    monkeypatch.delenv(f"GPUWM_HEX_MESH_COMPACT_{table}")
    monkeypatch.setenv("GPUWM_HEX_MESH_COMPACT", "0")
    assert not compaction.compaction_enabled(table)


def test_an_unknown_table_name_is_refused():
    with pytest.raises(ValueError):
        compaction.compaction_enabled("ZB")
