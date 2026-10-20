"""Index law of rate classes cut into blocks across cards (active full-step LTS, CPU)."""
from __future__ import annotations

import numpy as np
import pytest

from hexcore.fullstep_lts_blocks import blocks_per_class, build_block_plan, rcb_parts
from hexcore.fullstep_lts_classes import SCRATCH_FLAG, build_class_plan, grade_rates


def ring(n):
    cells = np.arange(n)
    coc = np.stack([(cells - 1) % n, (cells + 1) % n], axis=1).astype(np.int32)
    counts = np.full(n, 2, dtype=np.int32)
    coe = np.stack([cells, (cells + 1) % n], axis=1).astype(np.int32)
    cov = coe.copy()
    angle = 2 * np.pi * cells / n
    xyz = np.stack([np.cos(angle), np.sin(angle), np.zeros(n)], axis=1)
    return coc, counts, coe, cov, xyz


def block_plan(rates, devices, blocks):
    coc, counts, coe, cov, xyz = ring(len(rates))
    rates = grade_rates(np.asarray(rates, dtype=np.int32), coe)
    area = np.linspace(1.0, 2.0, len(rates))
    plan = build_class_plan(rates=rates, cells_on_cell=coc, n_edges_on_cell=counts, cells_on_edge=coe,
                            cells_on_vertex=cov, area_cell=area, dc_edge_m=np.ones(len(rates)), base_dt=1.0,
                            signal_speed=1.0, zone_factor=1.5, min_zone_rings=1)
    bplan = build_block_plan(plan, devices=devices, n_blocks=blocks, cells_on_cell=coc, n_edges_on_cell=counts,
                             cells_on_edge=coe, cells_on_vertex=cov, cell_xyz=xyz, area_cell=area)
    return plan, bplan, coe


RATES = [1] * 40 + [2] * 30 + [4] * 60 + [2] * 20 + [1] * 10 + [2] * 40


@pytest.mark.parametrize("devices,blocks", [((0, 1), (2, 2, 2)), ((0, 1, 2), (1, 3, 2)), ((5,), (1, 1, 1)),
                                            ((0, 1, 2, 3), (4, 4, 4))])
def test_blocks_cover_every_entity_once_and_each_class_keeps_its_cells(devices, blocks):
    plan, bplan, coe = block_plan(RATES, devices, blocks)
    cls = plan.cell_class
    seen_cells = np.zeros(cls.size, dtype=int)
    for t in bplan.blocks:
        lay = t.layout
        owned = lay.cell_l2g[:lay.n_owned_cells]
        assert np.all(cls[owned] == t.index)
        seen_cells[owned] += 1
        # carried cells are coarser; peer cells are the class's computed set; the rest another class's
        carried = lay.cell_l2g[lay.n_owned_cells:t.n_xown_cells]
        assert np.all(cls[carried] < t.index)
        assert np.all(t.halo_cell_class[:t.n_xown_cells - lay.n_owned_cells] & SCRATCH_FLAG)
        rest = lay.cell_l2g[t.n_xlocal_cells:]
        assert np.all(cls[rest] != t.index)
        assert t.device in devices
    assert np.all(seen_cells == 1)
    assert [len(g) for g in bplan.by_class] == list(blocks)


@pytest.mark.parametrize("devices,blocks", [((0, 1), (2, 2, 2)), ((0, 1, 2, 3), (4, 4, 4))])
def test_transport_tables_pair_the_same_global_entities(devices, blocks):
    plan, bplan, coe = block_plan(RATES, devices, blocks)
    for c, members in enumerate(bplan.by_class):
        tables = bplan.transport[c]
        if tables is None:
            continue
        for j, g in enumerate(members):
            mine = bplan.blocks[g]
            covered_c = []
            covered_e = []
            for peer in tables[j].peers:
                other = bplan.blocks[members[peer.peer]]
                reverse = next(p for p in tables[peer.peer].peers if p.peer == j)
                # what j receives from k is what k sends to j, entity for entity
                np.testing.assert_array_equal(mine.layout.cell_l2g[peer.cell_recv],
                                              other.layout.cell_l2g[reverse.cell_send])
                np.testing.assert_array_equal(mine.layout.edge_l2g[peer.edge_recv],
                                              other.layout.edge_l2g[reverse.edge_send])
                assert np.all(reverse.cell_send < other.n_xown_cells)
                covered_c.extend(peer.cell_recv.tolist())
                covered_e.extend(peer.edge_recv.tolist())
            assert sorted(covered_c) == list(range(mine.n_xown_cells, mine.n_xlocal_cells))
            assert sorted(covered_e) == list(range(mine.n_xown_edges, mine.n_xlocal_edges))


def test_a_coarse_columns_interface_edges_sit_in_one_block_with_its_reserve():
    plan, bplan, coe = block_plan(RATES, (0, 1, 2), (3, 3, 3))
    cls = plan.cell_class
    register = plan.register_edge_global
    fine_owner = {}
    for t in bplan.blocks:
        for s, e in zip(t.fine_slot, t.layout.edge_l2g[t.fine_edge_local]):
            assert register[s] == e
            assert s not in fine_owner, "an interface edge has two fine owners"
            fine_owner[int(s)] = t.gid
        # every slot of an owned reserve row is an edge this block owns
        rows = t.cap_slot[:t.n_cap_owned][t.cap_coef[:t.n_cap_owned] != 0]
        assert set(rows.tolist()) <= set(t.fine_slot.tolist())
    assert sorted(fine_owner) == list(range(register.size))
    # and every coarse cell's slots share one fine owner
    coarse_cell = np.where(cls[coe[register, 0]] < cls[coe[register, 1]], coe[register, 0], coe[register, 1])
    for cell in np.unique(coarse_cell):
        owners = {fine_owner[int(s)] for s in np.flatnonzero(coarse_cell == cell)}
        assert len(owners) == 1


def test_every_mailbox_read_is_on_the_card_or_routed_there():
    plan, bplan, coe = block_plan(RATES, (0, 1, 2, 3), (2, 4, 3))
    cls = plan.cell_class
    published_on = {}
    for t in bplan.blocks:
        for m in t.pub_cell_mail.tolist():
            published_on[m] = t
    for t in bplan.blocks:
        lay = t.layout
        halo = lay.cell_l2g[lay.n_owned_cells:]
        for position, cell in enumerate(halo):
            if cls[cell] == t.index:
                continue
            mail = int(t.halo_cell_mail[position])
            source = published_on[mail]
            if source.device == t.device:
                continue
            routed = bplan.route_cells.get((source.gid, t.device))
            assert routed is not None and mail in routed.tolist()


def test_every_register_read_on_another_card_is_pulled_there():
    plan, bplan, coe = block_plan(RATES, (0, 1, 2, 3), (2, 4, 3))
    writer = {}
    for t in bplan.blocks:
        for s in t.fine_slot.tolist():
            writer[s] = t.device
    for c, members in enumerate(bplan.by_class):
        pulled = {(w, d): set(s.tolist()) for w, d, s in bplan.register_pulls[c]}
        for g in members:
            t = bplan.blocks[g]
            for s in t.packet_slot.tolist():
                if writer[s] != t.device:
                    assert s in pulled[(writer[s], t.device)]


def test_block_counts_follow_the_owned_cells_and_the_cards():
    plan, _, _ = block_plan(RATES, (0,), (1, 1, 1))
    owned = [t.layout.n_owned_cells for t in plan.classes]
    counts = blocks_per_class(plan, 8, min_block_cells=20)
    assert counts == tuple(max(1, min(8, n // 20)) for n in owned)


def test_default_block_size_cuts_the_finer_classes_of_the_measured_mesh_on_two_cards():
    # 2026-10-15: at the old 50,000 default the 36,093- and 41,910-cell classes stayed whole on one
    # card of two (counts 2, 1, 1); at the default every class is cut in two.
    from types import SimpleNamespace

    from hexcore.fullstep_lts_blocks import DEFAULT_MIN_BLOCK_CELLS

    plan = SimpleNamespace(classes=[SimpleNamespace(layout=SimpleNamespace(n_owned_cells=n))
                                    for n in (144_039, 36_093, 41_910)])
    assert blocks_per_class(plan, 2, min_block_cells=50_000) == (2, 1, 1)
    assert blocks_per_class(plan, 2) == (2, 2, 2)
    assert blocks_per_class(plan, 2, min_block_cells=DEFAULT_MIN_BLOCK_CELLS) == (2, 2, 2)
    assert blocks_per_class(plan, 1) == (1, 1, 1)


def test_coordinate_bisection_is_balanced_and_deterministic():
    points = np.random.default_rng(1).normal(size=(1001, 3))
    labels = rcb_parts(points, 5)
    assert np.array_equal(labels, rcb_parts(points, 5))
    sizes = np.bincount(labels)
    assert sizes.max() - sizes.min() <= 2
    with pytest.raises(ValueError):
        rcb_parts(points[:3], 4)


def test_ranks_may_share_a_card_but_one_class_never_repeats_a_rank():
    plan, bplan, _ = block_plan(RATES, (0, 0, 1), (3, 2, 3))
    for members in bplan.by_class:
        ranks = [bplan.blocks[g].rank for g in members]
        assert len(set(ranks)) == len(ranks)
        assert all(bplan.blocks[g].device == (0, 0, 1)[bplan.blocks[g].rank] for g in members)
