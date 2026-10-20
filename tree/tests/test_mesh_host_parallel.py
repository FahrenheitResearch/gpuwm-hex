"""Cold-loader blocks retain stored connectivity and arc-validation words."""
import numpy as np
import pytest

from hexcore import host_parallel
from hexcore.mesh import _canonicalize_connectivity, _spherical_arc_lengths


@pytest.fixture
def blocked(monkeypatch):
    monkeypatch.setenv(host_parallel.ENVIRONMENT, "2")
    monkeypatch.setattr(host_parallel, "MIN_PARALLEL_ELEMENTS", 0)
    monkeypatch.setattr(host_parallel, "MIN_BLOCK_COLUMNS", 19)


@pytest.mark.parametrize("dtype", ["i4", "i8", "u8", ">i4"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_connectivity_matches_cast_then_subtract_and_padding(blocked, dtype, order):
    rng = np.random.default_rng(841)
    raw = np.array(rng.integers(0, 100, size=(1003, 6)), dtype=dtype, order=order)
    counts = rng.integers(0, 7, size=1003)
    if raw.dtype.kind == "u":
        raw[0, :3] = [np.iinfo(np.uint64).max, 1 << 63, 0]
        counts[0] = 6
    expected = np.asarray(raw, dtype=np.int64) - 1
    expected[np.arange(6)[None, :] >= counts[:, None]] = -1
    untouched = raw.copy()
    arrays = {"cellsOnCell": raw, "nEdgesOnCell": counts, "indexToCellID": counts}
    assert _canonicalize_connectivity(arrays) == ("cellsOnCell",)
    assert arrays["cellsOnCell"].dtype == expected.dtype
    assert arrays["cellsOnCell"].tobytes() == expected.tobytes()
    assert arrays["cellsOnCell"].flags.f_contiguous == expected.flags.f_contiguous
    assert raw.tobytes() == untouched.tobytes()
    assert arrays["indexToCellID"] is counts


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("rows", [0, 1, 1003])
def test_arc_words_match_original_full_gathers(blocked, clip, rows):
    rng = np.random.default_rng(842)
    coordinates = rng.normal(size=(501, 3))
    coordinates /= np.linalg.norm(coordinates, axis=1)[:, None]
    pairs = rng.integers(-1 if clip else 0, 501, size=(rows, 2))
    # Include coincident and antipodal endpoints, and missing regional slots.
    if rows > 2:
        coordinates[1] = -coordinates[0]
        pairs[:3] = [[0, 0], [0, 1], [-1 if clip else 0, 1]]
    used = np.clip(pairs, 0, None) if clip else pairs
    first, second = coordinates[used[:, 0]], coordinates[used[:, 1]]
    expected = 6371229.0 * np.arctan2(
        np.linalg.norm(np.cross(first, second), axis=1),
        np.sum(first * second, axis=1),
    )
    actual = _spherical_arc_lengths(coordinates, pairs, 6371229.0, clip_missing=clip)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("mutation", [None, "radius", "longitude", "both", "nonfinite"])
def test_coordinate_words_and_error_precedence_match_original(blocked, order, mutation):
    from hexcore.mesh import _spherical_coordinate_checks
    rng = np.random.default_rng(843)
    radius, tolerance = 6371229.0, 1.0e-12
    lat = rng.uniform(-1.5, 1.5, 1003)
    lon = rng.uniform(-np.pi, np.pi, 1003)
    xyz = np.array(radius * np.stack((np.cos(lat) * np.cos(lon),
                                    np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1),
                   order=order)
    if mutation in ("radius", "both", "nonfinite"):
        xyz[20] *= 1.1
    if mutation in ("longitude", "both", "nonfinite"):
        lon[-1] += 0.2
    if mutation == "nonfinite":
        xyz[600, 0] = np.nan
    # Independent original whole-array expressions, including error priority.
    expected_errors = []
    expected = None
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(lat + lon)):
        expected_errors.append("Cell coordinates contain non-finite values")
    else:
        norms = np.linalg.norm(xyz, axis=1)
        if not np.allclose(norms, radius, rtol=tolerance, atol=0.0):
            expected_errors.append("Cell Cartesian coordinates are not on sphere_radius")
        points = radius * np.stack((np.cos(lat) * np.cos(lon),
                                    np.cos(lat) * np.sin(lon), np.sin(lat)), axis=1)
        if not np.allclose(xyz, points, rtol=tolerance, atol=radius * tolerance):
            expected_errors.append("Cell Cartesian and lat/lon coordinates disagree")
        expected = xyz / norms[:, None]
    actual, errors = _spherical_coordinate_checks(xyz, lat, lon, radius, tolerance, "Cell")
    assert errors == expected_errors
    if expected is None:
        assert actual is None
    else:
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dtype", ["f4", "f8", ">f8"])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("mutation", [None, "active_nan", "padding_nan", "padding_inf", "padding_value", "both"])
def test_edge_weight_refusals_and_words_match_original(blocked, dtype, order, mutation):
    from hexcore.mesh import _edge_weight_errors
    rows, width = 1003, 10
    counts = np.arange(rows) % (width + 1)
    slots = np.arange(width)[None, :]
    weights = np.where(slots < counts[:, None], -1.25, -0.0).astype(dtype)
    if order == "strided":
        backing = np.zeros((rows * 2, width * 2), dtype=dtype)
        backing[::2, ::2] = weights
        weights = backing[::2, ::2]
    else:
        weights = np.array(weights, order=order)
    # The failures fall in different worker blocks, including the last row.
    if mutation in ("active_nan", "both"):
        weights[20, 0] = np.nan  # count[20] = 9
    if mutation == "padding_nan":
        weights[-1, -1] = np.nan
    if mutation == "padding_inf":
        weights[-1, -1] = np.inf
    if mutation in ("padding_value", "both"):
        weights[-1, -1] = 3.0
    original = weights.tobytes()
    weights.flags.writeable = False
    expected = []
    if not np.all(np.isfinite(weights)):
        expected.append("weightsOnEdge contains non-finite values")
    if np.any(weights[slots >= counts[:, None]] != 0):
        expected.append("weightsOnEdge padding is not zero")
    assert _edge_weight_errors(weights, counts, check_padding=True) == expected
    assert weights.tobytes() == original


def test_edge_weight_invalid_counts_keep_padding_check_suppressed(blocked):
    from hexcore.mesh import _edge_weight_errors
    weights = np.ones((1003, 10))
    weights[-1, -1] = np.nan
    # Already rejected by Mesh.validate; the original code does not use them.
    counts = np.full(1003, -999)
    assert _edge_weight_errors(weights, counts, check_padding=False) == [
        "weightsOnEdge contains non-finite values",
    ]


@pytest.mark.parametrize("dtype", ["i4", "i8", ">i8"])
@pytest.mark.parametrize("order", ["C", "F", "strided"])
@pytest.mark.parametrize("mode", ["start", "end", "mixed", "end_majority", "broken", "reversed"])
def test_vertex_slot_global_verdict_and_view_preserved(blocked, dtype, order, mode):
    from hexcore.mesh import vertex_slot_convention
    rows, width = 1003, 6
    counts = 3 + np.arange(rows) % 4
    slots = np.arange(width)[None, :]
    vertices = np.where(slots < counts[:, None], np.arange(rows)[:, None] * width + slots, -1)
    edges = np.where(slots < counts[:, None], np.arange(rows)[:, None] * width + slots, -1)
    following = np.where(slots + 1 < counts[:, None], slots + 1, 0)
    endpoints = np.stack((np.arange(rows)[:, None] * width + np.broadcast_to(slots, edges.shape),
                          np.arange(rows)[:, None] * width + following), axis=-1).reshape(-1, 2)
    selected = np.zeros(rows, dtype=bool)
    if mode == "end": selected[:] = True
    if mode == "mixed": selected[::2] = True
    if mode == "end_majority": selected[:800] = True
    for cell in np.flatnonzero(selected):
        vertices[cell, :counts[cell]] = np.roll(vertices[cell, :counts[cell]], -1)
    if mode == "broken":
        for cell in [1, 20, 499, 500, 700, 999, 1002]:
            vertices[cell, [0, 1]] = -99
    if mode == "reversed":
        for cell in range(rows):
            vertices[cell, :counts[cell]] = vertices[cell, :counts[cell]][::-1]
    def stored(a):
        if order != "strided": return np.array(a, dtype=dtype, order=order)
        backing = np.empty(tuple(n * 2 for n in a.shape), dtype=dtype)
        index = tuple(slice(None, None, 2) for _ in a.shape)
        backing[index] = a
        return backing[index]
    inputs = [stored(a) for a in (counts, edges, vertices, endpoints)]
    originals = [a.tobytes() for a in inputs]
    for a in inputs: a.flags.writeable = False
    c, e, v, ep = [np.asarray(a, dtype=np.int64) for a in inputs]
    # Independent original whole-array expressions and global verdict.
    used = slots < c[:, None]
    next_slot = np.where(slots + 1 < c[:, None], slots + 1, 0)
    previous = np.where(slots >= 1, slots - 1, c[:, None] - 1)
    first, second = ep[np.where(used, e, 0), 0], ep[np.where(used, e, 0), 1]
    own = (v == first) | (v == second)
    nxt, prv = v[np.arange(rows)[:, None], next_slot], v[np.arange(rows)[:, None], previous]
    starts = ((own & ((nxt == first) | (nxt == second))) | ~used).all(axis=1)
    ends = ((own & ((prv == first) | (prv == second))) | ~used).all(axis=1)
    ns, ne = int(starts.sum()), int(ends.sum())
    neither = np.flatnonzero(~starts & ~ends)
    convention = "edge-start" if ns == rows else "edge-end" if ne == rows else "mixed" if ns and ne else "none"
    expected_view = np.where(used, prv, -1) if ne > ns else v
    actual = vertex_slot_convention(*inputs)
    view = actual.pop("view")
    assert actual == dict(convention=convention, edge_start_cells=ns, edge_end_cells=ne,
                          neither_cells=len(neither), first_neither_cells=neither[:5].tolist())
    assert view.dtype == expected_view.dtype
    assert view.tobytes() == expected_view.tobytes()
    assert view.flags.c_contiguous == expected_view.flags.c_contiguous
    assert view.flags.f_contiguous == expected_view.flags.f_contiguous
    assert view.flags.writeable == expected_view.flags.writeable
    assert np.shares_memory(view, inputs[2]) == np.shares_memory(expected_view, inputs[2])
    assert [a.tobytes() for a in inputs] == originals


@pytest.mark.parametrize("width", [0, 6])
def test_vertex_slot_empty_cells_preserve_alias(blocked, width):
    from hexcore.mesh import vertex_slot_convention
    vertices = np.empty((0, width), dtype=np.int64)
    actual = vertex_slot_convention(np.empty(0, dtype=np.int64), vertices, vertices,
                                    np.zeros((1, 2), dtype=np.int64))
    assert actual.pop("view") is vertices
    assert actual == dict(convention="edge-start", edge_start_cells=0, edge_end_cells=0,
                          neither_cells=0, first_neither_cells=[])
