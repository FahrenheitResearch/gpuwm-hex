"""Independent FP64 donor reference in the native transport coordinate.

Arrays are (level, entity), edges_on_cell is zero based, and acoustic_sign
is the native sign: positive denotes an outward contribution. Fluxes are
the actual split-averaged uhAvg/wwAvg, not primitive velocity.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ColumnTransportRates:
    combined: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    level: np.ndarray


def column_transport_rates(*, rho_old, uh_avg, ww_avg, rdzw,
                           edges_on_cell, n_edges_on_cell, acoustic_sign,
                           dv_edge, area_cell):
    rho, uh, ww, rz, dv, area = [np.asarray(x, dtype=np.float64) for x in
        (rho_old, uh_avg, ww_avg, rdzw, dv_edge, area_cell)]
    edges, counts = np.asarray(edges_on_cell), np.asarray(n_edges_on_cell)
    signs = np.asarray(acoustic_sign, dtype=np.float64)
    if rho.ndim != 2 or uh.ndim != 2:
        raise ValueError('density and horizontal flux must be level/entity matrices')
    nk, nc = rho.shape
    ne = uh.shape[1]
    if (uh.shape[0] != nk or ww.shape != (nk+1, nc) or rz.shape != (nk,)
        or dv.shape != (ne,) or area.shape != (nc,) or counts.shape != (nc,)
        or edges.ndim != 2 or edges.shape[0] != nc or signs.shape != edges.shape):
        raise ValueError('transport input extents disagree')
    for name, value in [('rho_old', rho), ('uh_avg', uh), ('ww_avg', ww),
                        ('rdzw', rz), ('dv_edge', dv), ('area_cell', area)]:
        invalid = ~np.isfinite(value)
        if name in ('rho_old', 'rdzw', 'dv_edge', 'area_cell'):
            invalid |= value <= 0
        if np.any(invalid):
            location = tuple(int(i) for i in np.argwhere(invalid)[0])
            raise ValueError(f'{name}{location}: a finite local bound or conservative flux cannot be formed')
    if counts.dtype.kind not in 'iu' or edges.dtype.kind not in 'iu' or np.any((counts < 1) | (counts > edges.shape[1])):
        raise ValueError('invalid incident edge counts or integer connectivity')
    horizontal = np.zeros_like(rho)
    for slot in range(edges.shape[1]):
        cells = np.flatnonzero(counts > slot)
        incident = edges[cells, slot]
        sg = signs[cells, slot]
        if np.any((incident < 0) | (incident >= ne)) or np.any(~np.isin(sg, (-1, 1))):
            raise ValueError('invalid active incident edge or acoustic sign')
        horizontal[:, cells] += np.maximum(uh[:, incident] * sg, 0) * dv[incident] / area[cells]
    vertical = (np.maximum(ww[1:], 0) + np.maximum(-ww[:-1], 0)) * rz[:, None]
    horizontal /= rho
    vertical /= rho
    combined = horizontal + vertical
    if not np.all(np.isfinite(combined)):
        raise ValueError('transport rate overflow: a finite local bound cannot be formed')
    levels = np.argmax(combined, axis=0)
    columns = np.arange(nc)
    return ColumnTransportRates(combined[levels, columns],
        np.max(horizontal, axis=0), np.max(vertical, axis=0), levels)
