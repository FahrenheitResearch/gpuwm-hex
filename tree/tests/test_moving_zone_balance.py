"""D4 column balance: reproduces a column's native vertical residual while holding its mass."""
from types import SimpleNamespace

import numpy as np

from hexcore.moving_zone_state import _balance_columns, _vertical_residual_from_saved


def _column(nl=30, ncol=4):
    dz = 400.0
    z = (np.arange(nl) + 0.5) * dz
    theta = 300.0 + 0.004 * z[:, None] + np.linspace(0, 2, ncol)[None, :]
    rho = 1.15 * np.exp(-z / 8500.0)[:, None] * np.linspace(1.0, 1.02, ncol)[None, :]
    vertical = SimpleNamespace(zz=np.ones((nl, ncol)), rdzu=np.full(nl, 1 / dz), fzm=np.full(nl, 0.5),
                               fzp=np.full(nl, 0.5), dzw=np.full(nl, dz))
    rb = rho * 0.98
    rtb = rb * (theta - 1.0)
    exb = (287.0 * rtb / 1e5) ** (287.0 / (1004.5 - 287.0))
    reference = SimpleNamespace(rho_base=rb, rho_theta_base=rtb, exner_base=exb)
    return rho, theta, vertical, reference


def _saved(rho, theta, vertical, reference, q):
    p = 1e5 * (287.0 * vertical.zz * rho * theta / 1e5) ** (1004.5 / (1004.5 - 287.0))
    pb = vertical.zz * 287.0 * reference.rho_theta_base * reference.exner_base
    return dict(pressure_perturbation=(p - pb).astype(np.float32),
                density_perturbation=(rho - reference.rho_base).astype(np.float32),
                rho=rho.astype(np.float32), scalars=q[None].astype(np.float32))


def test_balance_recovers_a_perturbed_column_with_its_mass():
    rho, theta, vertical, reference = _column()
    q = 0.012 * np.exp(-np.arange(rho.shape[0]) / 6.0)[:, None] * np.ones(rho.shape[1])
    old = _saved(rho, theta, vertical, reference, q)
    columns = np.arange(rho.shape[1])
    target = _vertical_residual_from_saved(old, vertical, columns, [0])
    # Recompute the exact (FP64) residual of the float32-stored truth for a clean comparison.
    rho0 = old['rho'].astype(np.float64)
    tracer = (rho0 * old['scalars'][0]).astype(np.float64)[None]
    noisy = rho0 * (1 + 0.01 * np.sin(np.arange(rho.shape[0]))[:, None])
    noisy *= (rho0.sum(axis=0) / noisy.sum(axis=0))[None, :]
    result, receipt = _balance_columns(noisy, theta, tracer, [0], target, vertical, reference, columns,
                                       vertical.dzw)
    assert receipt['column_mass_relative_abs_max'] < 1e-12
    assert np.allclose((result * vertical.dzw[:, None]).sum(axis=0), (noisy * vertical.dzw[:, None]).sum(axis=0),
                       rtol=1e-12)
    assert np.max(np.abs(result / rho0 - 1)) < 2e-5
