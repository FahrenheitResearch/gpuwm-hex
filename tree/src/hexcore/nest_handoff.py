"""Storm selection from the native C-grid state for nested forecast placement."""
from __future__ import annotations

import numpy as np


def balance_regional_mass_flux(rho_u, rho_w, expected_total_divergence, vertical, arrays):
    """Match the parent's mapped density tendency with minimum-gradient edge correction.

    The native continuity equation is D(ru) + diff(rw)/dzw. The provisional
    terrain-coupled vertical flux is held. A dual-length graph projection
    supplies the horizontal correction, with zero potential outside the open
    regional boundary. It preserves resolved parent convergence, not zero
    divergence. Physical w is recovered from the final pair by the caller.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import splu
    from scipy.sparse.csgraph import connected_components
    from .moving_state import mass_flux_divergence
    cells = np.asarray(arrays['cellsOnEdge'])
    area = np.asarray(arrays['areaCell'], dtype=np.float64)
    nc, ne = len(area), len(cells)
    if not np.any(cells < 0):
        raise ValueError('nested mass-flux balance requires an open regional boundary')
    if np.any(np.all(cells < 0, axis=1)):
        raise ValueError('regional edge has no adjacent cell')
    rows, columns, signs = [], [], []
    for side, sign in ((0, 1.), (1, -1.)):
        valid = cells[:, side] >= 0
        rows.extend(cells[valid, side]); columns.extend(np.flatnonzero(valid)); signs.extend(np.full(valid.sum(), sign))
    incidence = coo_matrix((signs, (rows, columns)), shape=(nc, ne)).tocsr()
    dc = np.asarray(arrays['dcEdge'], dtype=np.float64)
    dv = np.asarray(arrays['dvEdge'], dtype=np.float64)
    if np.any(dc <= 0) or np.any(dv <= 0) or np.any(area <= 0):
        raise ValueError('mass-flux projection requires positive native geometry')
    laplacian = incidence @ diags(dv / dc) @ incidence.T
    count, labels = connected_components(laplacian, directed=False)
    boundary = np.unique(cells[np.any(cells < 0, axis=1)])
    boundary = boundary[boundary >= 0]
    if len(np.unique(labels[boundary])) != count:
        raise ValueError('a regional component has no open boundary anchor')
    current = mass_flux_divergence(rho_u, arrays) + np.diff(rho_w.astype(np.float64), axis=0) / np.asarray(vertical.dzw)[:, None]
    expected = np.asarray(expected_total_divergence, dtype=np.float64)
    if expected.shape != current.shape or not np.isfinite(expected).all():
        raise ValueError('mapped parent continuity field is invalid')
    rhs = ((expected - current) * area).T
    factor = splu(laplacian.tocsc())
    potential = factor.solve(np.ascontiguousarray(rhs))
    correction = ((incidence.T @ potential) / dc[:, None]).T
    exact = rho_u.astype(np.float64) + correction
    rounded = exact.astype(rho_u.dtype)
    after_exact = mass_flux_divergence(exact, arrays) + np.diff(rho_w.astype(np.float64), axis=0) / np.asarray(vertical.dzw)[:, None]
    residual = after_exact - expected
    if np.max(np.abs(residual)) > 1e-9 * max(1., float(np.max(np.abs(expected)))):
        raise ValueError('regional continuity projection did not converge to its mapped parent tendency')
    after = mass_flux_divergence(rounded, arrays) + np.diff(rho_w.astype(np.float64), axis=0) / np.asarray(vertical.dzw)[:, None]
    return rounded, dict(method='open-boundary graph projection matching mapped native mass continuity',
        horizontal_flux_correction_rms=float(np.sqrt(np.mean(correction**2))),
        horizontal_flux_correction_abs_max=float(abs(correction).max()),
        total_divergence_error_before_rms=float(np.sqrt(np.mean((current-expected)**2))),
        total_divergence_error_after_rms=float(np.sqrt(np.mean((after-expected)**2))),
        total_divergence_error_after_abs_max=float(abs(after-expected).max()),
        fp64_solve_residual_abs_max=float(abs(residual).max()),
        vertical_mass_flux_changed=False, boundary_potential=0.)


def cell_vertical_vorticity(mesh, normal_velocity):
    """MPAS dual-cell circulation, kite-area averaged onto primal cells."""
    a = mesh.arrays
    u = np.asarray(normal_velocity)
    edges = np.asarray(a['edgesOnVertex'])
    vertices = np.asarray(a['verticesOnEdge'])
    valid = edges >= 0
    safe = np.maximum(edges, 0)
    signs = np.where(vertices[safe, 1] == np.arange(len(edges))[:, None], 1, -1)
    curl = np.zeros((len(u), len(edges)), dtype=u.dtype)
    for slot in range(edges.shape[1]):
        e = safe[:, slot]
        curl += u[:, e] * (signs[:, slot] * valid[:, slot] * a['dcEdge'][e]).astype(u.dtype)
    curl /= a['areaTriangle'][None, :]
    vc = a['verticesOnCell']; cv = a['cellsOnVertex']
    result = np.zeros((len(u), len(vc)), dtype=u.dtype)
    cells = np.arange(len(vc))
    for slot in range(vc.shape[1]):
        active = slot < a['nEdgesOnCell']
        vertex = np.maximum(vc[:, slot], 0)
        matches = cv[vertex] == cells[:, None]
        if np.any(active & (matches.sum(axis=1) != 1)):
            raise ValueError('native kite connectivity is inconsistent')
        kite = a['kiteAreasOnVertex'][vertex, matches.argmax(axis=1)]
        result += curl[:, vertex] * np.where(active, kite, 0).astype(u.dtype)
    result /= a['areaCell'][None, :]
    return result


def updraft_helicity(vorticity, w, zgrid, *, bottom_m=2000., top_m=5000.):
    """Layer-integrated signed w*zeta over the requested AGL slab."""
    z = np.asarray(zgrid, dtype=np.float64)
    z = z - z[0]
    width = np.maximum(0., np.minimum(z[1:], top_m) - np.maximum(z[:-1], bottom_m))
    if not 0 <= bottom_m < top_m:
        raise ValueError('helicity layer bounds must increase above ground')
    if np.any(z[-1] < top_m):
        raise ValueError('helicity layer exceeds the model top')
    mid_w = .5 * (np.asarray(w)[1:] + np.asarray(w)[:-1])
    if mid_w.shape != np.asarray(vorticity).shape or mid_w.shape != width.shape:
        raise ValueError('helicity state staggering differs from the vertical grid')
    return np.sum(np.maximum(mid_w, 0.) * vorticity * width, axis=0)


def strongest_storm(mesh, normal_velocity, w, zgrid, *, reflectivity=None, eligible=None):
    """Use a diagnosed reflectivity maximum, falling back to native 2-5 km UH."""
    eligible = np.ones(len(mesh.arrays['areaCell']), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if eligible.shape != (len(mesh.arrays['areaCell']),) or not eligible.any():
        raise ValueError('storm search footprint is empty or differs from native cells')
    if reflectivity is not None:
        values = np.asarray(reflectivity)
        if values.ndim == 2:
            values = values.max(axis=0)
        if values.shape != (len(mesh.arrays['areaCell']),):
            raise ValueError('reflectivity must have one column per native cell')
        if not np.isfinite(values).all():
            raise ValueError('reflectivity contains nonfinite values')
        usable = float(values[eligible].max()) >= 35.
    else:
        usable = False
    if usable:
        metric, units = 'composite_reflectivity', 'dBZ'
    else:
        values = updraft_helicity(cell_vertical_vorticity(mesh, normal_velocity), w, zgrid)
        metric, units = 'updraft_helicity_2_5km', 'm2 s-2'
    if not np.isfinite(values).all() or values[eligible].max() <= 0:
        raise ValueError('no positive resolved storm metric is present in the parent state')
    cell = int(np.argmax(np.where(eligible, values, -np.inf)))
    return dict(metric=metric, units=units, value=float(values[cell]), cell=cell,
                latitude=float(np.rad2deg(mesh.arrays['latCell'][cell])),
                longitude=float((np.rad2deg(mesh.arrays['lonCell'][cell]) + 180) % 360 - 180))
