"""Sealing a prepared execution costs whole-array work, with unchanged records.

Breakage this guards (G3, 2026-10-12, conus3km host preparation): the v8.4.1
seal built the v8.2.3 damping profile in a Python loop over every cell and
level (209.2 s) only to replace its record, and rebuilt the edge signs in a
per-slot Python loop (20.1 s).  Both are host-only; the sealed records must
be the bytes the loops produced.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore import cuda_driver, cuda_dualrun


def _slot_loop(edges_on_cell, cells_on_edge, counts):
    signs = np.zeros(edges_on_cell.shape, dtype=np.float32)
    for cell in range(edges_on_cell.shape[0]):
        for slot in range(int(counts[cell])):
            edge = int(edges_on_cell[cell, slot])
            signs[cell, slot] = np.float32(1.0) if int(cells_on_edge[edge, 0]) == cell else np.float32(-1.0)
    return cuda_dualrun._authority_array_record(signs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_edge_sign_record_is_the_slot_loop_record(seed):
    rng = np.random.default_rng(seed)
    cells, edges, max_edges = 257, 801, 7
    counts = rng.integers(0, max_edges + 1, size=cells).astype(np.int32)
    edges_on_cell = rng.integers(-3, edges, size=(cells, max_edges)).astype(np.int32)
    cells_on_edge = rng.integers(0, cells, size=(edges, 2)).astype(np.int32)
    # Make about half the slots owned by their cell, as on a real mesh.
    for cell in range(cells):
        for slot in range(0, int(counts[cell]), 2):
            cells_on_edge[edges_on_cell[cell, slot], 0] = cell
    mesh = SimpleNamespace(edgesOnCell=edges_on_cell, cellsOnEdge=cells_on_edge, nEdgesOnCell=counts)
    assert cuda_dualrun._host_edge_sign_record(mesh) == _slot_loop(edges_on_cell, cells_on_edge, counts)


def test_edge_sign_record_refuses_an_out_of_range_edge_like_the_loop():
    mesh = SimpleNamespace(edgesOnCell=np.array([[0, 5]], dtype=np.int32),
                           cellsOnEdge=np.zeros((3, 2), dtype=np.int32),
                           nEdgesOnCell=np.array([2], dtype=np.int32))
    with pytest.raises(IndexError):
        cuda_dualrun._host_edge_sign_record(mesh)


def test_the_group_record_of_a_supplied_profile_is_the_record_the_v841_seal_writes():
    """The v8.4.1 seal overwrites the vertical group's dss with
    ``_authority_array_record(dss, float32)``, whatever profile the group
    recorded first (since 2026-10-12 the all-zero one), and that record is
    the group's own ``_authority_array_fingerprint(dss, float32)`` form."""
    rng = np.random.default_rng(7)
    for dtype in (np.float32, np.float64):
        dss = rng.random((59, 1031)).astype(dtype)
        assert (cuda_driver._authority_array_fingerprint(dss, dtype=np.float32)
                == cuda_dualrun._authority_array_record(dss, dtype=np.float32))



def _copy_fingerprint(value, dtype=None):
    import hashlib
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    raw = array.tobytes(order="C")
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                c_bytes=len(raw), c_bytes_sha256=hashlib.sha256(raw).hexdigest())


def test_authority_byte_view_matches_copied_bytes_for_layouts_and_dtypes():
    bits = np.array([0, 0x80000000, 1, 0x7fc00001, 0x7f800000], dtype=np.uint32)
    values = [bits.view(np.float32), bits[::-1], np.arange(12).reshape(3, 4).T,
              np.empty((2, 0, 3)), np.array(3.0), np.array([], dtype="V0"),
              np.array(["abc", "def"]), np.array([1, 2], dtype="datetime64[ns]"),
              np.array([(1, 2.)], dtype=[("a", ">i4"), ("b", "<f8")]),
              np.arange(15, dtype=">i4").reshape(3, 5)[:, ::2]]
    for value in values:
        assert cuda_driver._authority_array_fingerprint(value) == _copy_fingerprint(value)
    with pytest.raises(TypeError, match="cannot use object dtype"):
        cuda_driver._authority_array_fingerprint(np.array([object()]))


def test_parallel_authority_records_preserve_order_bytes_and_error_policy(monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "4")
    bits = np.tile(np.array([0, 0x80000000, 1, 0x7fc00001], dtype=np.uint32), 600_001)
    arrays = {"signed_nan": (bits.view(np.float32), None),
              "strided": (bits[::-2], np.float64),
              "big_endian": (bits, ">u4"), "empty": (np.empty((0, 2)), None)}
    expected = {name: _copy_fingerprint(value, dtype) for name, (value, dtype) in arrays.items()}
    actual = cuda_driver._authority_array_fingerprints(arrays)
    assert list(actual) == list(expected)
    assert actual == expected
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "1")
    assert cuda_driver._authority_array_fingerprints(arrays) == expected
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "4")
    arrays["overflow"] = (np.full(2, np.finfo(np.float64).max), np.float32)
    arrays["later_bad_object"] = (np.array([object()]), None)
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError, match="overflow"):
            cuda_driver._authority_array_fingerprints(arrays)
