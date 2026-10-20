"""IAU window arithmetic (hexcore.iau_forcing.IauForcing) on host arrays.

The forcing must hand every class exactly the whole increment over the window whatever its step
length, add nothing outside the window, and add nothing at all for a zero increment.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.iau_forcing import IauForcing


class _HostCp:
    """The two CuPy calls IauForcing makes, on numpy."""

    @staticmethod
    def asarray(a):
        return np.asarray(a)

    @staticmethod
    def ElementwiseKernel(_inputs, _outputs, _body, _name):
        def axpy(x, a, y):
            nz = x != 0
            y[nz] += np.float32(a) * x[nz]
        return axpy


def _increment(n_cells=5, n_edges=7, n_lev=3, n_scalars=2, seed=0):
    rng = np.random.default_rng(seed)
    return {"rho": rng.standard_normal((n_lev, n_cells)).astype(np.float32),
            "rho_theta": rng.standard_normal((n_lev, n_cells)).astype(np.float32),
            "rho_u": rng.standard_normal((n_lev, n_edges)).astype(np.float32),
            "scalars": rng.standard_normal((n_scalars, n_lev, n_cells)).astype(np.float32)}


def _held(n_cells, n_edges, n_lev=3, n_scalars=2):
    return SimpleNamespace(rho=np.zeros((n_lev, n_cells), np.float32),
                           rho_theta=np.zeros((n_lev, n_cells), np.float32),
                           rho_u=np.zeros((n_lev, n_edges), np.float32),
                           scalars=np.zeros((n_scalars, n_lev, n_cells), np.float32))


def _layouts():
    # two classes with overlapping local sets (halo cells/edges belong to both)
    return [SimpleNamespace(cell_l2g=np.array([0, 1, 2]), edge_l2g=np.array([0, 1, 2, 3])),
            SimpleNamespace(cell_l2g=np.array([2, 3, 4]), edge_l2g=np.array([3, 4, 5, 6]))]


@pytest.mark.parametrize("window", [600.0, 450.0])
def test_every_class_receives_the_whole_increment(window):
    inc = _increment()
    layouts = _layouts()
    forcing = IauForcing(inc, layouts, window_seconds=window, cp=_HostCp)
    for index, (layout, dt) in enumerate(zip(layouts, (18.75, 150.0))):
        total = _held(3, 4)
        t = 0.0
        while t < 1200.0:
            held = _held(3, 4)
            forcing.add(index, held, t, dt)
            for name in ("rho", "rho_theta", "rho_u", "scalars"):
                getattr(total, name)[...] += getattr(held, name) * np.float32(dt)
            t += dt
        cells, edges = layout.cell_l2g, layout.edge_l2g
        np.testing.assert_allclose(total.rho, inc["rho"][:, cells], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(total.rho_u, inc["rho_u"][:, edges], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(total.scalars, inc["scalars"][:, :, cells], rtol=1e-5, atol=1e-6)
        assert forcing.applied_seconds[index] == pytest.approx(window)
    assert forcing.receipt()["complete"]


def test_zero_increment_adds_nothing_bitwise():
    inc = {k: np.zeros_like(v) for k, v in _increment().items()}
    forcing = IauForcing(inc, _layouts(), window_seconds=600.0, cp=_HostCp)
    rng = np.random.default_rng(1)
    held = _held(3, 4)
    for name in ("rho", "rho_theta", "rho_u", "scalars"):
        getattr(held, name)[...] = rng.standard_normal(getattr(held, name).shape)
    before = {n: getattr(held, n).copy() for n in ("rho", "rho_theta", "rho_u", "scalars")}
    forcing.add(0, held, 0.0, 150.0)
    for name, value in before.items():
        assert np.array_equal(getattr(held, name), value)


def test_zero_window_is_refused():
    with pytest.raises(ValueError, match="direct insertion"):
        IauForcing(_increment(), _layouts(), window_seconds=0.0, cp=_HostCp)
