"""Convection-permitting cells stay at the base rate in the LTS class map.

Breakage prevented: classes are admitted once from the initial state, and the
horizontal bounds scale with spacing, so a 6 km ring was promoted to 2 dt; once
deep convection grew there its vertical Courant number (w dt / dz) passed 1 and
the 48 h moving typhoon runs of 2026-10-12 refused the outer step (start member
at 4.7 h, its fallback at 7.1 h).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hexcore.fullstep_lts_admission import admit_mesh_rates


def _chain(spacings_m):
    """A line of cells; edge i joins cell i and i+1, dcEdge the coarser neighbour's spacing."""
    n = len(spacings_m)
    coe = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    dc = np.array([max(spacings_m[i], spacings_m[i + 1]) for i in range(n - 1)], dtype=np.float64)
    levels = 4
    mesh = SimpleNamespace(arrays={"dcEdge": dc, "cellsOnEdge": coe, "dvEdge": dc * 0.6},
                           area_cell=np.array(spacings_m, dtype=np.float64) ** 2)
    state = SimpleNamespace(rho=np.ones((levels, n)), rho_u=np.zeros((levels, n - 1)),
                            rho_w=np.zeros((levels + 1, n)))
    saved = SimpleNamespace(normal_velocity=np.full((levels, n - 1), 1.0),
                            theta_m=np.full((levels, n), 300.0), exner=np.full((levels, n), 1.0))
    vertical = SimpleNamespace(dzw=np.full(levels, 500.0))
    return mesh, state, saved, vertical


def _rates(monkeypatch, spacing_env=None):
    if spacing_env is None:
        monkeypatch.delenv("GPUWM_HEX_LTS_CONVECTIVE_SPACING_M", raising=False)
    else:
        monkeypatch.setenv("GPUWM_HEX_LTS_CONVECTIVE_SPACING_M", spacing_env)
    spacings = [3000.0] * 12 + [6000.0] * 24 + [12000.0] * 24
    mesh, state, saved, vertical = _chain(spacings)
    rates, info = admit_mesh_rates(mesh, state, saved, vertical, dt=18.75, max_rate=8, acoustic_substeps=6,
                                   target_transport=None)
    return np.array(spacings), rates, info


def test_six_km_ring_is_held_at_the_base_rate(monkeypatch):
    spacings, rates, info = _rates(monkeypatch)
    assert np.all(rates[spacings < 8000.0] == 1)
    assert rates[spacings >= 12000.0].max() > 1          # coarse cells still take larger steps
    assert info["convective_cells_held_at_base_rate"] > 0


def test_hold_off_restores_the_geometric_promotion(monkeypatch):
    spacings, rates, info = _rates(monkeypatch, "0")
    assert rates[(spacings == 6000.0)].max() == 2
    assert "convective_cells_held_at_base_rate" not in info
