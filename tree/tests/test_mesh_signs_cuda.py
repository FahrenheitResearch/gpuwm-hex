"""The production sign kernel must match the former scalar loop word for word."""
import numpy as np
import pytest

from hexcore.cuda_backend.containers import _device_edge_signs


@pytest.mark.gpu
@pytest.mark.parametrize('index_dtype,float_dtype', [(np.int32, np.float32), (np.int64, np.float64)])
def test_device_signs_preserve_active_signs_and_inactive_sentinels(index_dtype, float_dtype):
    cp = pytest.importorskip('cupy')
    edges = np.array([[0, 2, -1, -99], [0, 1, -1, -99], [1, 2, -1, -99]], dtype=index_dtype)
    endpoints = np.array([[0, 1], [1, 2], [2, 0]], dtype=index_dtype)
    counts = np.array([2, 2, 0], dtype=index_dtype)
    expected = np.zeros(edges.shape, dtype=float_dtype)
    for cell in range(len(counts)):
        for slot in range(int(counts[cell])):
            expected[cell, slot] = 1 if endpoints[edges[cell, slot], 0] == cell else -1
    actual = cp.asnumpy(_device_edge_signs(cp, cp.asarray(edges), cp.asarray(endpoints), cp.asarray(counts), float_dtype))
    assert actual.tobytes() == expected.tobytes()
