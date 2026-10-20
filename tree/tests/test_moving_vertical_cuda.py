"""GPU height integration preserves sparse, signed and sub-ground donor arithmetic."""
import os
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from hexcore.moving_remap import ConservativeMap
from hexcore.moving_vertical import HeightConservativeMap


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('aligned', [False, True])
@pytest.mark.parametrize('kind', ['positive', 'signed', 'sparse'])
def test_ordered_height_integrals_match_host(monkeypatch, aligned, kind):
    pytest.importorskip('cupy')
    rng=np.random.default_rng(117)
    n=19; levels=9
    overlap=np.eye(n)*.75+np.roll(np.eye(n),1,axis=1)*.25
    horizontal=ConservativeMap(csr_matrix(overlap),np.ones(n),np.ones(n),{})
    z=np.linspace(0.,10000.,levels+1)[:,None]*np.ones((1,n))
    target=z+(1-z/10000.)*rng.uniform(-10.,10.,n)
    w=HeightConservativeMap(horizontal,z,target,np.diff(z[:,0]),np.diff(z[:,0]),endpoint_alignment=aligned)
    assert w.backend == 'cuda'
    density=rng.uniform(.1,1.1,(levels,n))
    value=rng.uniform(0.,2.,(levels,n))
    if kind=='signed': value-=1.
    if kind=='sparse':value[value<1.8]=0.
    if not aligned:
        w.extend_below_ground(np.full_like(density,300.),np.ones_like(density))
    w.backend='cpu'
    expected=w.mass_intensive(value,density=density)
    receipts=(w.last_domain_reconciliation,w.last_positivity_roundoff)
    w.backend='cuda'
    actual=w.mass_intensive(value,density=density)
    assert actual.tobytes()==expected.tobytes()
    assert (w.last_domain_reconciliation,w.last_positivity_roundoff)==receipts
    # Reusing geometry must not retain the previous product or extension.
    if not aligned: w.extend_below_ground(np.full_like(density,310.),np.full_like(density,.98))
    actual=w.mass_intensive(value*.75,density=density)
    w.backend='cpu'
    expected=w.mass_intensive(value*.75,density=density)
    assert actual.tobytes()==expected.tobytes()


@pytest.mark.skipif(os.environ.get('HEX_TEST_CUDA') != '1', reason='requires an owned CUDA card')
@pytest.mark.parametrize('source_levels,target_levels', [(1, 1), (9, 5), (55, 55)])
@pytest.mark.parametrize('middle', [(), (2, 3)])
def test_layer_rates_match_host_and_refresh_values(monkeypatch, source_levels, target_levels, middle):
    pytest.importorskip('cupy')
    rng = np.random.default_rng(461)
    n = 19
    overlap = np.eye(n)*.75 + np.roll(np.eye(n), 1, axis=1)*.25
    horizontal = ConservativeMap(csr_matrix(overlap), np.ones(n), np.ones(n), {})
    source = np.linspace(0., 10000., source_levels+1)[:, None]*np.ones((1, n))
    target = np.linspace(0., 10000., target_levels+1)[:, None]*np.ones((1, n))
    target += (1-target/10000.)*rng.uniform(-500., 500., n)
    w = HeightConservativeMap(horizontal, source, target, np.diff(source[:, 0]),
                             np.diff(np.linspace(0., 10000., target_levels+1)), endpoint_alignment=False)
    assert w.backend == 'cuda'
    values = rng.normal(size=(source_levels, *middle, n))
    for product in (values, values*.25):
        w.backend='cpu'
        expected = w.layer_intensive(product)
        w.backend='cuda'
        actual = w.layer_intensive(product)
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
