"""Exact placement objective, grant remapping, and ownership preservation."""
from types import SimpleNamespace
import itertools

import pytest

from hexcore.partition_topology_v841 import parse_topology, place_ranks


def _links(n=8):
    return tuple(tuple("X" if i == j else "PIX" if i // 2 == j // 2
                       else "NODE" if i // 4 == j // 4 else "SYS"
                       for j in range(n)) for i in range(n))


def _tables(n, edges):
    return [SimpleNamespace(rank=i, peers=tuple(
        SimpleNamespace(peer=j, cell_send=tuple(range(weight)), edge_send=())
        for a, b, weight in edges for j in ([b] if a == i else [a] if b == i else [])
    )) for i in range(n)]


def test_eight_card_placement_joins_heavy_neighbors_on_switch_pairs():
    edges = [(0, 4, 10), (1, 5, 9), (2, 6, 8), (3, 7, 7),
             (0, 1, 2), (2, 3, 2), (1, 2, 1)]
    tables = _tables(8, edges)
    before = repr(tables)
    receipt = place_ranks(tables, range(8), _links())
    order = receipt["rank_to_slot"]
    assert sorted(order) == list(range(8))
    assert receipt["objective"][0] == 2
    for a, b, weight in edges[:4]:
        assert order[a] // 2 == order[b] // 2
    assert receipt["objective"] < receipt["baseline_objective"]
    assert repr(tables) == before


def test_placement_matches_independent_exhaustive_objective_and_visible_ordinals():
    edges = [(0, 2, 7), (1, 3, 6), (0, 1, 1), (2, 3, 2)]
    links = _links(4)
    def score(order):
        return sum(2 * w for a, b, w in edges if links[order[a]][order[b]] != "PIX")
    receipt = place_ranks(_tables(4, edges), (3, 0, 7, 2), links)
    assert receipt["objective"] == [0, min(map(score, itertools.permutations(range(4))))]
    assert receipt["devices"] == [((3, 0, 7, 2)[i]) for i in receipt["rank_to_slot"]]


def test_parse_topology_resolves_cuda_mask_by_bus_and_ignores_cpu_columns():
    inventory = "0, 00000000:01:00.0\n1, 00000000:02:00.0\n2, 00000000:03:00.0\n"
    topology = "GPU0 GPU1 GPU2 CPU Affinity NUMA Affinity\nGPU0 X PIX SYS 0-63 0\nGPU1 PIX X SYS 0-63 0\nGPU2 SYS SYS X 64-127 1\n"
    assert parse_topology(topology, inventory, ("0000:03:00.0", "0000:01:00.0")) == (("X", "SYS"), ("SYS", "X"))


def test_unknown_topology_keeps_order_and_names_missing_evidence(monkeypatch):
    from hexcore import partition_topology_v841 as module
    monkeypatch.setattr(module, "discover_topology", lambda devices: (None, "unavailable"))
    receipt = place_ranks(_tables(2, [(0, 1, 1)]), (5, 3))
    assert receipt["devices"] == [5, 3] and receipt["objective"] is None


def test_equal_objective_keeps_identity_and_one_card_is_valid():
    assert place_ranks(_tables(4, []), range(4), _links(4))["rank_to_slot"] == list(range(4))
    assert place_ranks(_tables(1, []), (6,), (("X",),))["devices"] == [6]


@pytest.mark.parametrize("devices,links,match", [
    ((0, 0), _links(2), "distinct cards"),
    ((0, 1), (("X",),), "dimensions"),
    ((0, 1), (("X", "PIX"), ("SYS", "X")), "symmetric"),
    ((0, 1), (("X", "bad"), ("bad", "X")), "unknown topology"),
])
def test_invalid_topology_refuses_concrete_misrouting(devices, links, match):
    with pytest.raises(ValueError, match=match):
        place_ranks(_tables(2, [(0, 1, 1)]), devices, links)
