"""Large threaded rows must match the independent scalar compression authority."""
import numpy as np
import pytest

from hexcore.errors import ConfigurationRefusal
from hexcore.transport import (
    _compress_advection_coefficients_serial, _native_advection_compression,
    _native_advection_order, _topology,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("regional", [False, True])
def test_large_edge_slices_keep_scalar_words(dtype, regional, monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "4")
    rng = np.random.default_rng(729)
    counts = np.full(7, 2, np.int64)
    neighbors = np.array([[(i + 1) % 7, (i + 3) % 7] for i in range(7)])
    owners = np.array([[i % 7, (i + 2) % 7] for i in range(11)])
    if regional:
        owners[-1, 1] = 7
        neighbors[-1, 1] = 7
    ids = np.array([5, 2, 7, 1, 4, 6, 3])
    deriv = rng.normal(size=(11, 2, 15)).astype(dtype)
    deriv[0, 0, :3] = [dtype(-0.0), np.nextafter(dtype(0), dtype(1)), dtype(0)]
    dc = rng.uniform(1000, 12000, 11).astype(dtype)
    dv = rng.uniform(500, 13000, 11).astype(dtype)
    arguments = dict(source_order_v841=True, allow_regional_sentinels=regional)
    expected = _compress_advection_coefficients_serial(
        counts, neighbors, owners, ids, deriv, dc, dv, **arguments)
    bridge = _compress_advection_coefficients_serial(
        counts, neighbors, owners, ids, deriv, dc, dv,
        source_order_v841=False, allow_regional_sentinels=regional)
    # 66,011 edges, including unequal slice lengths; each is an independent
    # repeat of the small scalar authority, not of the new vector code.
    repeat = 6001
    tile = lambda a: np.tile(a, (repeat,) + (1,) * (a.ndim - 1))
    adv, adv3, n_adv, cells = map(tile, bridge)
    _, _, ordered = _native_advection_order(counts, neighbors, tile(owners),
        adv, adv3, n_adv, cells, garbage=7 if regional else -1)
    a, b = _native_advection_compression(counts, neighbors, tile(owners),
        tile(deriv), tile(dc), tile(dv), n_adv, ordered)
    for actual, authority in zip((a, b, n_adv, ordered), expected):
        assert actual.tobytes() == tile(authority).tobytes()


def test_active_slot_validation_keeps_first_row_and_error_priority(monkeypatch):
    monkeypatch.setenv("GPUWM_HEX_HOSTPREP_THREADS", "4")
    n = 65539
    mesh = dict(nEdgesOnCell=np.ones(n, np.int64),
        edgesOnCell=np.tile([0, -999], (n, 1)),
        cellsOnCell=np.tile([0, -999], (n, 1)),
        cellsOnEdge=np.array([[0, 1]]))
    _topology(mesh)  # Negative inactive padding must stay ignored.
    mesh['cellsOnCell'][17000, 0] = -1
    mesh['edgesOnCell'][50000, 0] = -1
    with pytest.raises(ConfigurationRefusal, match="active regional neighbor"):
        _topology(mesh)
    mesh['edgesOnCell'][17000, 0] = -1
    with pytest.raises(ValueError, match="active edgesOnCell entry"):
        _topology(mesh)
    mesh['edgesOnCell'][:, 0] = 0
    actual = _topology(mesh, allow_regional_sentinels=True)
    assert actual[4][17000, 0] == n
