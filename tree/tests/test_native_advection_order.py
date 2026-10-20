"""Native stencil order changes reductions, including regional garbage edges."""
import numpy as np

from hexcore.transport import _native_advection_order


def test_native_insertion_order_preserves_coefficients_and_unsolved_edges():
    counts = np.array([2, 2, 1])
    neighbors = np.array([[1, 2], [0, 2], [0, -1]])
    owners = np.array([[0, 1], [0, 3], [3, 3]])
    cells = np.array([[0, 1, 2, -1], [0, 1, 2, 3], [3, -1, -1, -1]])
    counts_adv = np.array([3, 4, 1])
    adv = np.array([[100, 101, 102, 0], [200, 201, 202, 203], [300, 0, 0, 0]], dtype=np.float32)
    adv3 = -adv
    a, b, c = _native_advection_order(counts, neighbors, owners, adv, adv3, counts_adv, cells, garbage=3)
    np.testing.assert_array_equal(c, [[0, 1, 2, -1], [0, 3, 1, 2], [3, -1, -1, -1]])
    np.testing.assert_array_equal(a, [[100, 101, 102, 0], [200, 203, 201, 202], [300, 0, 0, 0]])
    np.testing.assert_array_equal(b.view(np.uint32), (-a).view(np.uint32))
    np.testing.assert_array_equal(adv, [[100, 101, 102, 0], [200, 201, 202, 203], [300, 0, 0, 0]])
