"""Descriptor retention beyond the normal cache and capture-only misses."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.cuda_graph_tables import GraphTables


@dataclass
class _Table:
    nbytes: int


class _Owner:
    def __init__(self, cuda):
        self.cuda = cuda
        self.table_uploads = 0
        self.allocators = []

    def table(self, rows):
        self.table_uploads += 1
        self.allocators.append(self.cuda.allocator)
        return _Table(len(rows) * 40)


class _Cuda:
    def __init__(self):
        self.allocator = "scratch"

    @contextmanager
    def using_allocator(self, allocator):
        before = self.allocator
        self.allocator = allocator
        try:
            yield
        finally:
            self.allocator = before


def test_prepare_uses_original_allocator_and_deduplicates_owners():
    cuda = _Cuda()
    owner = _Owner(cuda)
    other = _Owner(cuda)
    tables = GraphTables(SimpleNamespace(cuda=cuda), [owner, owner, other], "original")
    rows = [(1, 2, 3, 4, 5)]
    with tables.scope():
        first = owner.table(rows)
        assert owner.table(rows) is first
        assert other.table(rows) is not first
    assert owner.allocators == ["original"]
    assert other.allocators == ["original"]
    assert cuda.allocator == "scratch"
    assert "table" not in vars(owner)
    assert "table" not in vars(other)
    assert tables.receipt() == {
        "tables": 2,
        "uploads_during_prepare": 2,
        "retained_table_bytes": 80,
    }


def test_capture_miss_restores_existing_instance_method():
    cuda = _Cuda()
    owner = _Owner(cuda)
    previous = lambda rows: _Table(40)
    owner.table = previous
    tables = GraphTables(SimpleNamespace(cuda=cuda), [owner], "original")
    tables.capturing = True
    with pytest.raises(RuntimeError, match="host-to-device transfer"):
        with tables.scope():
            owner.table([(1, 2, 3, 4, 5)])
    assert owner.table is previous
    assert owner.table_uploads == 0
    assert cuda.allocator == "scratch"


@pytest.mark.gpu
def test_capture_retains_more_than_256_descriptor_tables():
    if any(
        os.environ.get(name, "") not in ("", "0")
        for name in ("GPUWM_HEX_NO_LOCAL_GPU", "GPUWM_NO_LOCAL_GPU")
    ):
        pytest.skip("local GPU access is disabled")
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    if not available:
        pytest.skip("no CUDA device")

    from hexcore.cuda_solve_region_v841 import SolveRegionKernels

    owner = SolveRegionKernels.__new__(SolveRegionKernels)
    owner._tables = {}
    owner.table_uploads = 0
    original_allocator = cp.cuda.get_allocator()
    tables = GraphTables(cp, [owner, owner], original_allocator)
    rows = [((0, 0, 0, 0, index),) for index in range(300)]
    scratch_requests = []

    def scratch_allocator(size):
        scratch_requests.append(size)
        return original_allocator(size)

    with cp.cuda.using_allocator(scratch_allocator), tables.scope():
        for descriptor in rows:
            owner.table(descriptor)
    assert scratch_requests == []
    assert owner.table_uploads == 300
    assert len(owner._tables) < 256
    assert "table" not in vars(owner)
    assert tables.receipt() == {
        "tables": 300,
        "uploads_during_prepare": 300,
        "retained_table_bytes": 300 * 40,
    }

    kernel = cp.RawKernel(
        r'''
        extern "C" __global__ void graph_table_read(
            const long long *table, long long *output, const int index)
        {
            if (threadIdx.x == 0) output[index] = table[4];
        }
        ''',
        "graph_table_read",
    )
    kernel.compile()
    output = cp.empty(300, dtype=cp.int64)
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    tables.capturing = True
    with stream, tables.scope():
        stream.begin_capture()
        try:
            for index, descriptor in enumerate(rows):
                table = owner.table(descriptor)
                kernel((1,), (1,), (table, output, np.int32(index)))
        finally:
            graph = stream.end_capture()
        graph.launch(stream)
    stream.synchronize()
    assert cp.asnumpy(output).tobytes() == np.arange(300, dtype=np.int64).tobytes()
    assert owner.table_uploads == 300
    assert "table" not in vars(owner)
    with pytest.raises(RuntimeError, match="descriptor table was not prepared"):
        with tables.scope():
            owner.table(((0, 0, 0, 0, 301),))
    assert owner.table_uploads == 300
    assert "table" not in vars(owner)
