import numpy as np
import pytest
from types import SimpleNamespace
from hexcore.nest_handoff import updraft_helicity, strongest_storm


def test_helicity_integrates_only_updraft_and_partial_layers_above_terrain():
    z = np.array([0, 1500, 3500, 6000.])[:, None] + np.array([0, 800.])[None, :]
    vort = np.full((3, 2), .01)
    w = np.broadcast_to([10., -10.], (4, 2))
    np.testing.assert_allclose(updraft_helicity(vort, w, z), [300., 0.])
    with pytest.raises(ValueError, match='model top'):
        updraft_helicity(vort, w, z, top_m=7000)


def test_diagnosed_reflectivity_selects_native_peak_without_guess_position():
    mesh = SimpleNamespace(arrays=dict(areaCell=np.ones(2), latCell=np.deg2rad([35, 41]),
                                      lonCell=np.deg2rad([260, 272])))
    result = strongest_storm(mesh, None, None, None, reflectivity=np.array([[38, 45], [51, 60]]))
    assert result['cell'] == 1
    assert result['latitude'] == 41
    assert result['longitude'] == -88


def test_continuity_projection_preserves_parent_convergence_and_vertical_flux():
    from hexcore.nest_handoff import balance_regional_mass_flux
    from hexcore.moving_state import mass_flux_divergence
    arrays = dict(cellsOnEdge=np.array([[-1, 0], [0, 1], [1, -1]]),
                  edgesOnCell=np.array([[0, 1], [1, 2]]), nEdgesOnCell=np.array([2, 2]),
                  areaCell=np.array([2., 3.]), dcEdge=np.array([1., 2., 1.]), dvEdge=np.array([2., 1., 2.]))
    ru = np.array([[1., 2., 4.], [2., 5., 3.]], np.float32)
    rw = np.array([[0., 0.], [2., -3.], [0., 0.]], np.float32)
    before_w = rw.copy()
    dz = np.array([100., 200.])
    # A deliberately nonzero density tendency, different at each level/cell.
    expected = np.array([[.04, -.02], [.1, -.2]])
    fixed, receipt = balance_regional_mass_flux(ru, rw, expected, SimpleNamespace(dzw=dz), arrays)
    actual = mass_flux_divergence(fixed, arrays) + np.diff(rw, axis=0) / dz[:, None]
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(rw, before_w)
    assert receipt['fp64_solve_residual_abs_max'] < 1e-12
    assert receipt['total_divergence_error_after_rms'] < 1e-7
