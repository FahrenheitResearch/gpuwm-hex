"""The zone divergence projection reaches the requested divergence and leaves other faces alone."""
import numpy as np

from hexcore.moving_state import mass_flux_divergence
from hexcore.moving_zone_state import _project_zone_divergence


def _hex_patch(n=12):
    """A small planar Voronoi mesh (regular triangular lattice of generators) as MPAS-style arrays."""
    from scipy.spatial import Delaunay
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    points = np.stack((xs + 0.5 * (ys % 2), ys * np.sqrt(3) / 2), axis=-1).reshape(-1, 2)
    tri = Delaunay(points)
    edges = np.unique(np.sort(np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]],
                                              tri.simplices[:, [2, 0]]]), axis=1), axis=0)
    dc = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    dv = np.full(len(edges), 1 / np.sqrt(3))
    cells = len(points)
    edges_on_cell = -np.ones((cells, 8), dtype=np.int64)
    counts = np.zeros(cells, dtype=np.int64)
    for e, (a, b) in enumerate(edges):
        for c in (a, b):
            edges_on_cell[c, counts[c]] = e
            counts[c] += 1
    area = np.full(cells, np.sqrt(3) / 2)
    return dict(cellsOnEdge=edges, dvEdge=dv, dcEdge=dc, areaCell=area, edgesOnCell=edges_on_cell,
                nEdgesOnCell=counts)


def test_projection_matches_expected_divergence_on_zone():
    arrays = _hex_patch()
    rng = np.random.default_rng(3)
    nl, ne, nc = 3, len(arrays['dvEdge']), len(arrays['areaCell'])
    rho_u = rng.normal(size=(nl, ne)).astype(np.float32)
    zone = np.arange(nc)[(np.arange(nc) % 12 > 2) & (np.arange(nc) % 12 < 9) & (np.arange(nc) // 12 > 2)
                         & (np.arange(nc) // 12 < 9)]
    in_zone = np.zeros(nc, dtype=bool)
    in_zone[zone] = True
    cells = arrays['cellsOnEdge']
    free = in_zone[cells[:, 0]] & in_zone[cells[:, 1]]
    divergence = mass_flux_divergence(rho_u, arrays)
    # A compatible target: same zone total as the current divergence (boundary faces are fixed).
    expected = divergence[:, zone] + rng.normal(size=(nl, len(zone)))
    expected -= ((expected - divergence[:, zone]) * arrays['areaCell'][zone]).sum(axis=1, keepdims=True) / \
        arrays['areaCell'][zone].sum()
    correction, receipt = _project_zone_divergence(rho_u, divergence, expected, zone, free, arrays,
                                                   arrays['areaCell'])
    corrected = rho_u.astype(np.float64)
    corrected[:, receipt['edges']] += correction
    after = mass_flux_divergence(corrected, arrays)
    assert np.allclose(after[:, zone], expected, atol=1e-8)
    outside = np.setdiff1d(np.arange(ne), receipt['edges'])
    assert np.array_equal(corrected[:, outside], rho_u[:, outside].astype(np.float64))
    assert np.allclose(after[:, ~in_zone], divergence[:, ~in_zone], atol=1e-12)
