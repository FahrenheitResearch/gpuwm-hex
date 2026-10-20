"""The two transport entry points that read ``allow_regional_sentinels`` take it.

THE BREAKAGE THIS PREVENTS, found by ``ruff --select F821`` on 2026-10-09:
``initialize_deriv_two`` and ``monotonic_tracer_advection_tendency`` both
passed ``allow_regional_sentinels`` to ``_topology`` without declaring it, so
every call raised ``NameError`` before touching the mesh.  The second sits
behind the public exports ``monotonic_tracer_tendency`` and
``mpas_tracer_advection_mono_tend``.  Both now declare the keyword with the
same default ``build_advection_coefficients`` uses (a global mesh), and this
file calls each one for real on a closed synthetic sphere.
"""

from __future__ import annotations

import numpy as np

from hexcore import transport
from test_mesh_validation import graded_sphere_mesh


def _mesh():
    """A closed sphere plus the ``angleEdge`` the synthetic fixture omits.

    ``angleEdge`` is MPAS's angle from local east to the edge normal (cell 1
    to cell 2), measured at the edge point.
    """

    mesh = graded_sphere_mesh(level=2, zoom=2.0, dtype=np.float64)
    a = mesh.arrays
    cells = np.asarray(a["cellsOnEdge"], dtype=np.int64)
    if cells.min() >= 1:  # one-based NetCDF connectivity
        cells = cells - 1
    xyz = np.stack((a["xCell"], a["yCell"], a["zCell"]), axis=-1)
    normal = xyz[cells[:, 1]] - xyz[cells[:, 0]]
    lat, lon = np.asarray(a["latEdge"]), np.asarray(a["lonEdge"])
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon),
                      np.cos(lat)), axis=-1)
    a["angleEdge"] = np.arctan2((normal * north).sum(-1), (normal * east).sum(-1))
    return mesh


def test_initialize_deriv_two_runs_and_the_default_is_the_global_mesh():
    mesh = _mesh()
    n_edges = int(np.asarray(mesh.arrays["cellsOnEdge"]).shape[0])
    default = transport.initialize_deriv_two(mesh)
    explicit = transport.initialize_deriv_two(mesh, allow_regional_sentinels=False)
    assert default.shape[:2] == (n_edges, 2)
    assert np.all(np.isfinite(default))
    np.testing.assert_array_equal(default, explicit)


def test_monotonic_tendency_runs_and_a_still_uniform_tracer_has_no_tendency():
    mesh = _mesh()
    n_cells = int(np.asarray(mesh.arrays["nEdgesOnCell"]).size)
    n_edges = int(np.asarray(mesh.arrays["cellsOnEdge"]).shape[0])
    n_levels = 2
    coefficients = transport.build_advection_coefficients(
        mesh, transport.initialize_deriv_two(mesh))
    tracers = np.ones((1, n_levels, n_cells))
    still_edges = np.zeros((n_levels, n_edges))
    still_w = np.zeros((n_levels + 1, n_cells))
    thickness = np.full((n_levels, n_cells), 100.0)
    no_change = np.zeros((n_levels, n_cells))
    for function in (transport.monotonic_tracer_advection_tendency,
                     transport.monotonic_tracer_tendency):
        for flag in ({}, {"allow_regional_sentinels": False}):
            tendency = function(
                mesh, tracers, still_edges, still_w, thickness, thickness,
                no_change, 10.0, coefficients=coefficients, **flag)
            assert tendency.shape == tracers.shape
            np.testing.assert_array_equal(tendency, np.zeros_like(tracers))
