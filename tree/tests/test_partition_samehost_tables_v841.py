"""CPU identity checks for N-rank owner-to-halo index plans."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hexcore.partition_assets_v841 import build_partition_layouts
from hexcore.partition_executor_v841 import ExecutorError, HaloExchangeTables
from hexcore.partition_samehost_tables_v841 import (
    SameHostHaloExchangeTables,
    build_all,
)


def _cycle_layouts(tmp_path, n_ranks, *, halo_rings=2):
    n_cells = 12
    graph = tmp_path / "cycle.graph.info"
    graph.write_text(
        "\n".join(
            [f"{n_cells} {n_cells}"]
            + [
                f"{(cell - 1) % n_cells + 1} {(cell + 1) % n_cells + 1}"
                for cell in range(n_cells)
            ]
        )
        + "\n"
    )
    part = tmp_path / f"cycle.graph.info.part.{n_ranks}"
    part.write_text(
        "\n".join(str(cell * n_ranks // n_cells) for cell in range(n_cells)) + "\n"
    )
    adjacency = np.array(
        [[cell, (cell + 1) % n_cells] for cell in range(n_cells)], dtype=np.int32
    )
    return build_partition_layouts(
        graph,
        part,
        halo_rings=halo_rings,
        cells_on_vertex=adjacency,
        cells_on_edge=adjacency,
    )


def test_two_rank_plan_retains_existing_receiver_order(tmp_path):
    layouts = _cycle_layouts(tmp_path, 2)
    for rank, tables in enumerate(build_all(layouts)):
        old = HaloExchangeTables.build(layouts, rank)
        assert len(tables.peers) == 1
        peer = tables.peers[0]
        assert peer.peer == 1 - rank
        np.testing.assert_array_equal(peer.cell_send, old.cell_send)
        np.testing.assert_array_equal(peer.edge_send, old.edge_send)
        np.testing.assert_array_equal(
            peer.cell_recv, np.arange(old.n_owned_cells, old.n_local_cells)
        )
        np.testing.assert_array_equal(
            peer.edge_recv, np.arange(old.n_owned_edges, old.n_local_edges)
        )


def test_three_rank_interleaved_halo_follows_receiver_local_order(tmp_path):
    layouts = _cycle_layouts(tmp_path, 3)
    tables = build_all(layouts)
    # Rank 0 owns 0..3, followed by ring 1 [4,11], then ring 2 [5,10].
    assert layouts[0].cell_l2g.tolist() == [0, 1, 2, 3, 4, 11, 5, 10]
    assert tables[0].peers[0].cell_recv.tolist() == [4, 6]
    assert tables[0].peers[1].cell_recv.tolist() == [5, 7]
    # Rank 1 requests rank 0 globals [3,2]; rank 2 requests [0,1].
    assert tables[0].peers[0].cell_send.tolist() == [3, 2]
    assert tables[0].peers[1].cell_send.tolist() == [0, 1]


@pytest.mark.parametrize("n_ranks", [1, 2, 3, 4, 6])
def test_owner_gather_restores_all_halos_byte_for_byte(tmp_path, n_ranks):
    layouts = _cycle_layouts(tmp_path, n_ranks)
    tables = build_all(layouts)
    # Include signed zero, infinities and two distinct NaN payloads.  The
    # exchange must retain all bits, including those ordinary equality loses.
    truth = np.array(
        [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7FC00002,
         0x3F800000, 0xBF800000, 0x00000001, 0x807FFFFF, 0x7F7FFFFF, 0xFF7FFFFF],
        dtype=np.uint32,
    ).view(np.float32)
    truth = np.ascontiguousarray(np.broadcast_to(truth, (2, 3, truth.size)))
    for kind, plural in (("cell", "cells"), ("edge", "edges")):
        fields = []
        owned_before = []
        for layout in layouts:
            n_owned = getattr(layout, f"n_owned_{plural}")
            field = truth[..., getattr(layout, f"{kind}_l2g")].copy()
            field[..., n_owned:] = np.float32(42)
            fields.append(field)
            owned_before.append(field[..., :n_owned].tobytes())
        for rank, table in enumerate(tables):
            coverage = np.zeros(getattr(layouts[rank], f"n_local_{plural}"), dtype=np.int32)
            for peer in table.peers:
                reverse = next(p for p in tables[peer.peer].peers if p.peer == rank)
                recv = getattr(peer, f"{kind}_recv")
                send = getattr(reverse, f"{kind}_send")
                fields[rank][..., recv] = fields[peer.peer][..., send]
                coverage[recv] += 1
                assert np.all(recv >= getattr(layouts[rank], f"n_owned_{plural}"))
                assert np.all(send < getattr(layouts[peer.peer], f"n_owned_{plural}"))
                assert recv.dtype == send.dtype == np.int64
                assert recv.flags.c_contiguous and send.flags.c_contiguous
                assert not recv.flags.writeable and not send.flags.writeable
            n_owned = getattr(layouts[rank], f"n_owned_{plural}")
            assert np.all(coverage[:n_owned] == 0)
            assert np.all(coverage[n_owned:] == 1)
        for rank, layout in enumerate(layouts):
            expected = truth[..., getattr(layout, f"{kind}_l2g")]
            assert fields[rank].tobytes() == expected.tobytes()
            n_owned = getattr(layout, f"n_owned_{plural}")
            assert fields[rank][..., :n_owned].tobytes() == owned_before[rank]


def test_empty_peer_links_are_present_and_empty(tmp_path):
    layouts = _cycle_layouts(tmp_path, 4, halo_rings=0)
    tables = build_all(layouts)
    assert [peer.peer for peer in tables[0].peers] == [1, 2, 3]
    # No cell halos at K=0.  Edges incident on owned cells may still be halos.
    assert all(peer.cell_recv.size == peer.cell_send.size == 0 for peer in tables[0].peers)
    assert tables[0].peers[1].edge_recv.size == 0


@pytest.mark.parametrize("plural, kind", [("cells", "cell"), ("edges", "edge"), ("vertices", "vertex")])
def test_missing_owner_refuses_incomplete_owned_union(tmp_path, plural, kind):
    layouts = _cycle_layouts(tmp_path, 3)
    layouts[0] = replace(layouts[0], **{f"n_owned_{plural}": 0})
    with pytest.raises(ExecutorError, match=f"{plural}: global index .* has no owner"):
        build_all(layouts)


@pytest.mark.parametrize("plural, kind", [("cells", "cell"), ("edges", "edge"), ("vertices", "vertex")])
def test_duplicate_owner_refuses_overlapping_owned_union(tmp_path, plural, kind):
    layouts = _cycle_layouts(tmp_path, 3)
    vector = getattr(layouts[0], f"{kind}_l2g")
    layouts[0] = replace(layouts[0], **{f"n_owned_{plural}": vector.size})
    with pytest.raises(ExecutorError, match=f"{plural}: global index .* has more than one owner"):
        build_all(layouts)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"_global_cells": 13}, "disagree on the global extent"),
        ({"n_owned_cells": -1}, "owned count -1"),
        ({"n_owned_cells": 100}, "owned count 100"),
        ({"cell_l2g": np.array([0, 0], dtype=np.int32)}, "owned count"),
        ({"cell_l2g": np.array([0, 1, 2, 3, 3], dtype=np.int32)}, "duplicate local"),
        ({"cell_l2g": np.array([0, 1, 2, 12], dtype=np.int32)}, "outside global extent"),
        ({"cell_l2g": np.array([0, 1, 2, 3], dtype=np.float32)}, "integer vector"),
        ({"cell_l2g": np.array([[0, 1, 2, 3]], dtype=np.int32)}, "integer vector"),
        ({"partition": 1}, "rank order must match"),
    ],
)
def test_malformed_layout_refuses_before_planning(tmp_path, change, reason):
    layouts = _cycle_layouts(tmp_path, 3)
    layouts[0] = replace(layouts[0], **change)
    with pytest.raises(ExecutorError, match=reason):
        build_all(layouts)


def test_empty_layouts_and_out_of_range_rank_refuse(tmp_path):
    with pytest.raises(ExecutorError, match="at least one layout"):
        build_all([])
    layouts = _cycle_layouts(tmp_path, 3)
    for rank in (-1, 3):
        with pytest.raises(ExecutorError, match="outside 3 layouts"):
            SameHostHaloExchangeTables.build(layouts, rank)
    with pytest.raises(ExecutorError, match="rank must be an integer"):
        SameHostHaloExchangeTables.build(layouts, 1.5)


def test_receipt_has_index_hashes_and_stable_rank_order(tmp_path):
    layouts = _cycle_layouts(tmp_path, 3)
    tables = SameHostHaloExchangeTables.build(layouts, 0)
    receipt = tables.receipt()
    assert receipt["n_ranks"] == 3
    assert receipt["cells_received_per_field"] == 4
    assert [peer["peer"] for peer in receipt["peers"]] == [1, 2]
    assert receipt == build_all(layouts)[0].receipt()
    for peer in receipt["peers"]:
        for key in ("cell_send", "cell_recv", "edge_send", "edge_recv"):
            assert len(peer[key]["sha256_int64_le"]) == 64
