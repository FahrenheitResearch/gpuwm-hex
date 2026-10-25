"""Weakly connected overlaps must close unchanged conservation gates quickly."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix, block_diag
from hexcore.moving_remap import reconcile_area_marginals


def weak_ring(n=64):
    ids = np.arange(n)
    return csr_matrix((np.r_[np.full(n, 1.-1e-6), np.full(n, 1e-6)],
                       (np.r_[ids, ids], np.r_[ids, (ids+1)%n])), shape=(n, n))


@pytest.mark.parametrize('legacy_scipy', [False, True])
def test_weak_overlap_closes_without_relaxing_integrals_or_support(monkeypatch, legacy_scipy):
    if legacy_scipy:
        import scipy.sparse.linalg as linear
        current = linear.cg
        def old_cg(A, b, *, tol, **kwargs):
            return current(A, b, rtol=tol, **kwargs)
        monkeypatch.setattr(linear, 'cg', old_cg)
    overlap = weak_ring()
    original = overlap.copy()
    source = np.ones(64)
    target = source.copy()
    target[0] += 2e-9
    target[32] -= 2e-9
    with pytest.raises(ValueError, match='failed to converge'):
        reconcile_area_marginals(overlap, source, target, limit=8)
    balanced, receipt = reconcile_area_marginals(overlap, source, target)
    assert receipt['reconciliation_newton']
    assert receipt['reconciliation_iterations'] <= 16
    assert receipt['reconciled_relative_max'] <= 1e-10
    assert receipt['donor_area_relative_max'] <= 2e-14
    assert np.all(balanced.data > 0)
    for key in ('indices', 'indptr'):
        np.testing.assert_array_equal(getattr(balanced, key), getattr(original, key))
    np.testing.assert_array_equal(overlap.data, original.data)
    np.testing.assert_array_equal((np.asarray(balanced.sum(1)).ravel()/target).astype(np.float32),
                                  np.ones(64, np.float32))
    fields = np.stack([np.arange(64)-30., np.eye(1, 64, 13)[0]])
    mapped = (balanced @ fields.T).T / target
    np.testing.assert_allclose(mapped @ target, fields @ source, rtol=4e-15, atol=3e-14)


def test_component_rounding_keeps_original_limits():
    overlap = block_diag([csr_matrix(np.eye(2)), weak_ring()], format='csr')
    source = np.ones(66)
    target = source.copy()
    target[0] += 5e-11
    target[1] -= 5e-11
    target[2] += 2e-9
    target[34] -= 2e-9
    balanced, receipt = reconcile_area_marginals(overlap, source, target)
    assert receipt['components'] == 3
    assert receipt['reconciled_relative_max'] <= 1e-10
    assert receipt['donor_area_relative_max'] <= 2e-14
    np.testing.assert_array_equal(balanced.toarray()[:2, :2], np.eye(2))
    target[0] = 1 + 5e-10
    target[1] = 1 - 5e-10
    with pytest.raises(ValueError, match='failed to converge.*component floor='):
        reconcile_area_marginals(overlap, source, target)
