"""Index law, event order and register arithmetic of active full-step LTS (CPU)."""
from __future__ import annotations

import numpy as np
import pytest

from hexcore.fullstep_lts_classes import (
    CLASS_MASK,
    SCRATCH_FLAG,
    build_class_layouts,
    build_class_plan,
    class_indices,
    grade_rates,
    macro_schedule,
    plan_macros,
    trajectory_plane,
    lts_acoustic_substeps,
    plane_weights,
    planes_per_step,
)
from hexcore.partition_assets_v841 import build_partition_layouts


def ring(n):
    """A closed periodic chain: edge i joins cells i and i+1, vertex i sits on edge i."""

    cells = np.arange(n)
    coc = np.stack([(cells - 1) % n, (cells + 1) % n], axis=1).astype(np.int32)
    counts = np.full(n, 2, dtype=np.int32)
    coe = np.stack([cells, (cells + 1) % n], axis=1).astype(np.int32)
    cov = coe.copy()
    return coc, counts, coe, cov


def plan_for(rates, area=None, *, spacing=None, base_dt=1.0, speed=1.0, factor=1.5, min_rings=1):
    """Unit spacing and unit signal speed: a zone is ``factor * rate`` cells deep unless spacing says otherwise."""

    coc, counts, coe, cov = ring(len(rates))
    rates = np.asarray(rates, dtype=np.int32)
    area = np.ones(len(rates)) if area is None else np.asarray(area, dtype=np.float64)
    dc = np.ones(len(rates)) if spacing is None else np.asarray(spacing, dtype=np.float64)
    return build_class_plan(rates=rates, cells_on_cell=coc, n_edges_on_cell=counts,
                            cells_on_edge=coe, cells_on_vertex=cov, area_cell=area,
                            dc_edge_m=dc, base_dt=base_dt, signal_speed=speed, zone_factor=factor,
                            min_zone_rings=min_rings), coe


def test_grading_never_leaves_more_than_one_dyadic_level_across_an_edge():
    _, _, coe, _ = ring(12)
    rates = np.array([1, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], dtype=np.int32)
    graded = grade_rates(rates, coe)
    assert graded.tolist() == [1, 2, 4, 8, 16, 16, 16, 16, 16, 8, 4, 2]
    assert np.all(np.maximum(graded[coe[:, 0]], graded[coe[:, 1]])
                  <= 2 * np.minimum(graded[coe[:, 0]], graded[coe[:, 1]]))
    # already graded maps are returned unchanged, and odd ladders are refused
    np.testing.assert_array_equal(grade_rates(graded, coe), graded)
    with pytest.raises(ValueError, match="dyadic"):
        grade_rates(np.array([1, 3] * 6), coe)


def test_classes_are_indexed_coarsest_first():
    rates, cls = class_indices(np.array([1, 4, 4, 2, 1]))
    assert rates == (4, 2, 1)
    assert cls.tolist() == [2, 0, 0, 1, 2]


def test_class_layouts_obey_the_partition_builders_cell_laws(tmp_path):
    """Same rings, order and completeness as the partition builder; only edge ownership differs."""

    n = 24
    coc, counts, coe, cov = ring(n)
    cls = np.array([0] * 8 + [1] * 4 + [2] * 6 + [1] * 6, dtype=np.int32)
    graph = tmp_path / "ring.graph.info"
    graph.write_text("\n".join([f"{n} {n}"] + [" ".join(str(v + 1) for v in sorted(row)) for row in coc]) + "\n")
    part = tmp_path / "ring.graph.info.part.3"
    part.write_text("\n".join(str(v) for v in cls) + "\n")
    reference = build_partition_layouts(graph, part, halo_rings=3, cells_on_edge=coe, cells_on_vertex=cov)
    mine = build_class_layouts(cells_on_cell=coc, n_edges_on_cell=counts, cells_on_edge=coe,
                               cells_on_vertex=cov, cell_class=cls)
    for a, b in zip(reference, mine):
        np.testing.assert_array_equal(a.cell_l2g, b.cell_l2g)
        np.testing.assert_array_equal(a.cell_ring, b.cell_ring)
        np.testing.assert_array_equal(a.cell_row_complete, b.cell_row_complete)
        assert set(a.edge_l2g.tolist()) == set(b.edge_l2g.tolist())
        assert set(a.vertex_l2g.tolist()) == set(b.vertex_l2g.tolist())
        complete_a = dict(zip(a.edge_l2g.tolist(), a.edge_row_complete.tolist()))
        complete_b = dict(zip(b.edge_l2g.tolist(), b.edge_row_complete.tolist()))
        assert complete_a == complete_b
    # the finer class (larger index) owns every interface edge
    for layout in mine:
        owned = layout.edge_l2g[:layout.n_owned_edges]
        np.testing.assert_array_equal(np.maximum(cls[coe[owned, 0]], cls[coe[owned, 1]]), layout.partition)


def test_the_register_is_the_rate_interface_and_the_finer_class_owns_it():
    rates = [1] * 6 + [2] * 6 + [4] * 12 + [2] * 6
    plan, coe = plan_for(rates)
    assert plan.rates == (4, 2, 1)
    rates = np.asarray(rates)
    interface = np.flatnonzero(rates[coe[:, 0]] != rates[coe[:, 1]])
    np.testing.assert_array_equal(plan.register_edge_global, interface)
    seen_fine = np.zeros(interface.size, dtype=int)
    seen_coarse = np.zeros(interface.size, dtype=int)
    for tables in plan.classes:
        layout = tables.layout
        seen_fine[tables.fine_slot] += 1
        seen_coarse[tables.coarse_slot] += 1
        np.testing.assert_array_equal(layout.edge_l2g[tables.fine_edge_local], interface[tables.fine_slot])
        np.testing.assert_array_equal(layout.edge_l2g[tables.coarse_edge_local], interface[tables.coarse_slot])
        np.testing.assert_array_equal(layout.cell_l2g[tables.fine_cells], coe[interface[tables.fine_slot]])
        np.testing.assert_array_equal(layout.cell_l2g[tables.coarse_cells], coe[interface[tables.coarse_slot]])
        # the fine side owns the edge, the coarse side holds it in its halo
        assert np.all(tables.fine_edge_local < layout.n_owned_edges)
        assert np.all(tables.coarse_edge_local >= layout.n_owned_edges)
        assert tables.halo_cell_mail.size == layout.n_local_cells - layout.n_owned_cells
        assert np.all((tables.halo_cell_class & CLASS_MASK) != tables.index)
    np.testing.assert_array_equal(seen_fine, 1)
    np.testing.assert_array_equal(seen_coarse, 1)
    published_cells = np.concatenate([t.pub_cell_mail for t in plan.classes])
    published_edges = np.concatenate([t.pub_edge_mail for t in plan.classes])
    assert sorted(published_cells.tolist()) == list(range(plan.n_mail_cells))
    assert sorted(published_edges.tolist()) == list(range(plan.n_mail_edges))
    assert plan.classes[-1].reflux_cell_local.size == 0
    assert [t.child for t in plan.classes] == [1, 2, -1]


def test_a_class_carries_only_coarser_cells_and_the_coarsest_carries_none():
    # cells 0..19 rate 1, cells 20..59 rate 2: the fine zone needs 1.5 * 2 = 3 unit cells
    rates = np.array([1] * 20 + [2] * 40)
    plan, coe = plan_for(rates)
    coarse, fine = plan.classes
    assert not np.any(coarse.halo_cell_class & SCRATCH_FLAG)
    coarse_halo = coarse.layout.cell_l2g[coarse.layout.n_owned_cells:]
    assert sorted(coarse_halo.tolist()) == [0, 1, 2, 17, 18, 19]          # three plain rings of finer cells
    fine_halo = fine.layout.cell_l2g[fine.layout.n_owned_cells:]
    carried = fine_halo[(fine.halo_cell_class & SCRATCH_FLAG) != 0]
    assert sorted(carried.tolist()) == [20, 21, 22, 57, 58, 59]
    frozen = fine_halo[(fine.halo_cell_class & SCRATCH_FLAG) == 0]
    assert sorted(frozen.tolist()) == [23, 24, 25, 54, 55, 56]            # three frozen rings beyond the zone
    assert fine.layout.cell_row_complete[fine.layout.cell_g2l()[carried]].all()
    # carried edges: those of the coarser class that touch a zone cell
    fine_edges = fine.layout.edge_l2g[fine.layout.n_owned_edges:]
    free = set(fine_edges[(fine.halo_edge_class & SCRATCH_FLAG) != 0].tolist())
    assert free == {20, 21, 22, 56, 57, 58}
    assert plan.receipt()["scratch_cells_per_class"] == [0, 6]
    assert plan.level_depth == 1
    assert plan.zones[1]["rings"] == 3 and plan.zones[1]["coarsest_rate_inside"] == 2


def test_the_zone_outruns_the_signal_between_two_resets_of_its_coarsest_class():
    # twice the spacing halves the rings; a faster signal or a larger margin deepens the zone
    rates = np.array([1] * 20 + [2] * 60)
    assert plan_for(rates, spacing=np.full(80, 2.0))[0].zones[1]["rings"] == 2
    assert plan_for(rates, speed=2.0)[0].zones[1]["rings"] == 6
    assert plan_for(rates, factor=3.0)[0].zones[1]["rings"] == 6
    assert plan_for(rates, base_dt=4.0)[0].zones[1]["rings"] == 12
    assert plan_for(rates, factor=0.1, min_rings=4)[0].zones[1]["rings"] == 4
    # a thin rate-2 band: the finest class's zone reaches rate-4 cells, which are
    # reset only every 4 base steps, so the zone must be 1.5 * 4 = 6 cells deep
    rates = np.array([1] * 10 + [2] * 2 + [4] * 30 + [2] * 2)
    plan, _ = plan_for(rates)
    assert plan.zones[2] == {"rings": 6, "cells": 12, "crossing_seconds_required": 6.0,
                             "coarsest_rate_inside": 4}
    finest = plan.classes[2]
    halo = finest.layout.cell_l2g[finest.layout.n_owned_cells:]
    carried = halo[(finest.halo_cell_class & SCRATCH_FLAG) != 0]
    assert sorted(carried.tolist()) == [10, 11, 12, 13, 14, 15, 38, 39, 40, 41, 42, 43]
    # ... and the coarsest class then reads finest cells two levels away
    assert plan.level_depth == 2 and plan.classes[0].finest_halo_distance == 2


def test_reflux_coefficients_are_minus_incidence_over_area():
    rates = [1] * 4 + [2] * 8
    area = np.linspace(2.0, 3.0, 12)
    plan, coe = plan_for(rates, area)
    coarse = plan.classes[0]
    cells = coarse.layout.cell_l2g[coarse.reflux_cell_local]
    assert sorted(cells.tolist()) == [4, 11]
    for row, cell in enumerate(cells):
        slots = coarse.reflux_slot[row][coarse.reflux_coef[row] != 0.0]
        assert slots.size == 1
        edge = plan.register_edge_global[slots[0]]
        sign = 1.0 if coe[edge, 0] == cell else -1.0
        assert coarse.reflux_coef[row][coarse.reflux_coef[row] != 0.0][0] == -sign / area[cell]


def chain(n):
    """An OPEN chain, as a limited-area mesh is open: the two outer edges have one cell (-1 sentinel)."""

    cells = np.arange(n)
    coc = np.stack([cells - 1, cells + 1], axis=1).astype(np.int32)
    coc[coc >= n] = -1
    counts = np.full(n, 2, dtype=np.int32)
    coe = np.stack([np.arange(-1, n), np.arange(0, n + 1)], axis=1).astype(np.int32)
    coe[coe >= n] = -1
    cov = coe.copy()
    return coc, counts, coe, cov


def test_an_ungraded_map_is_refused_and_an_open_mesh_is_planned():
    # Retired 2026-10-12: an open (limited-area) mesh was refused here.  It is
    # now planned: its outer edges have one cell, take no part in grading or
    # the rate interface, and are owned by the class of their one cell.
    with pytest.raises(ValueError, match="not graded"):
        plan_for([1] * 6 + [4] * 6)
    n = 24
    coc, counts, coe, cov = chain(n)
    rates = grade_rates(np.array([1] + [16] * (n - 1), dtype=np.int32), coe)
    # the absent-cell sentinel must not couple the first cell to the last
    assert rates.tolist() == [1, 2, 4, 8] + [16] * (n - 4)
    plan = build_class_plan(rates=rates, cells_on_cell=coc, n_edges_on_cell=counts, cells_on_edge=coe,
                            cells_on_vertex=cov, area_cell=np.ones(n), dc_edge_m=np.ones(n + 1),
                            base_dt=1.0, signal_speed=1.0, zone_factor=1.5, min_zone_rings=1)
    outer = [0, n]
    assert not np.isin(plan.register_edge_global, outer).any()
    owners = {int(e): t.index for t in plan.classes for e in t.layout.edge_l2g[:t.layout.n_owned_edges]}
    assert owners[0] == plan.cell_class[0] and owners[n] == plan.cell_class[n - 1]
    assert sorted(owners) == list(range(n + 1))


def test_the_lateral_boundary_zone_and_its_buffer_rings_form_one_mask():
    from types import SimpleNamespace

    from hexcore.fullstep_lts_admission import BOUNDARY_BUFFER_RINGS, boundary_zone

    n = 20
    _, _, coe, _ = chain(n)
    mask = np.zeros(n, dtype=np.int32)
    mask[[0, n - 1]], mask[[1, n - 2]] = 2, 1
    zone = boundary_zone(SimpleNamespace(arrays={"bdyMaskCell": mask}), coe, n)
    depth = 2 + BOUNDARY_BUFFER_RINGS
    assert zone.tolist() == [True] * depth + [False] * (n - 2 * depth) + [True] * depth
    with pytest.raises(ValueError, match="bdyMaskCell"):
        boundary_zone(SimpleNamespace(arrays={}), coe, n)


def test_a_replicated_cell_is_local_to_every_class_and_carried_by_none():
    # the limited-area garbage donor: a real column every class sub-mesh must hold
    n = 40
    coc, counts, coe, cov = chain(n)
    rates = grade_rates(np.array([1] * 6 + [16] * (n - 6), dtype=np.int32), coe)
    donor = n - 1
    kwargs = dict(rates=rates, cells_on_cell=coc, n_edges_on_cell=counts, cells_on_edge=coe, cells_on_vertex=cov,
                  area_cell=np.ones(n), dc_edge_m=np.ones(n + 1), base_dt=1.0, signal_speed=1.0,
                  zone_factor=1.5, min_zone_rings=1)
    without = build_class_plan(**kwargs)
    plan = build_class_plan(**kwargs, replicated_cells=(donor,))
    assert any(donor not in t.layout.cell_l2g for t in without.classes)
    for t in plan.classes:
        local = t.layout.cell_g2l()[donor]
        assert local >= 0
        if plan.cell_class[donor] != t.index:
            halo = local - t.layout.n_owned_cells
            assert halo >= 0 and not (t.halo_cell_class[halo] & SCRATCH_FLAG)
            assert (t.halo_cell_class[halo] & CLASS_MASK) == plan.cell_class[donor]


def covered(events, index):
    return sorted((begin, begin + length) for kind, i, begin, length in events if kind == "step" and i == index)


@pytest.mark.parametrize("rates, macro", [((4, 2, 1), 4), ((16, 8, 4, 2, 1), 16), ((16, 4, 1), 16),
                                          ((16, 8, 4, 2, 1), 4), ((4, 2, 1), 1), ((2, 1), 2)])
def test_every_class_covers_the_macro_once_after_its_finer_classes(rates, macro):
    events = macro_schedule(rates, macro)
    for index, rate in enumerate(rates):
        length = min(rate, macro)
        assert covered(events, index) == [(b, b + length) for b in range(0, macro, length)]
    position = {event: at for at, event in enumerate(events)}
    for kind, index, begin, length in events:
        if kind != "step":
            continue
        step = position[(kind, index, begin, length)]
        assert position[("sync", index, begin, length)] == step + 1
        for finer in range(index + 1, len(rates)):
            inside = [position[e] for e in events
                      if e[0] in ("step", "sync") and e[1] == finer and begin <= e[2] < begin + length]
            # fine first: every finer step and sync of the interval precedes this step
            assert inside and max(inside) < step
            later = [position[e] for e in events if e[1] == finer and e[2] >= begin + length]
            assert not later or min(later) > step + 1


def test_every_sixth_of_a_step_is_a_split_end_of_every_finer_dyadic_class():
    # N = 2, ticks of base_dt/6: a class of effective rate r ends a split every 2r ticks
    assert planes_per_step(2) == 6
    for consumer in (2, 4, 8, 16):
        for provider in (1, 2, 4, 8):
            if provider >= consumer:
                continue
            ends = range(2 * provider, 6 * consumer + 1, 2 * provider)
            seen = sorted(p for p in (trajectory_plane(0, consumer, t, 6) for t in ends) if p is not None)
            assert seen == [1, 2, 3, 4, 5, 6]
            # so no plane of a strictly finer provider is interpolated
            assert all(plane_weights(p, consumer, provider, 2)[2:] == (1.0, 0.0) for p in range(7))
    # two classes sharing a step (an event tail) only give the even sixths
    assert [trajectory_plane(12, 2, 12 + t, 6) for t in (4, 8, 12)] == [2, 4, 6]
    assert trajectory_plane(12, 2, 12, 6) is None and trajectory_plane(12, 2, 26, 6) is None
    assert trajectory_plane(0, 2, 3, 6) is None
    # and the odd sixths of a tail are the mean of the neighbouring accepted states
    assert [plane_weights(p, 2, 2, 2) for p in (1, 2, 3)] == [(0, 2, 0.5, 0.5), (2, 2, 1.0, 0.0), (2, 4, 0.5, 0.5)]


def test_four_substeps_put_every_other_plane_mid_split_of_the_next_finer_class():
    # N = 4 (MPAS-A v8.4.2's default): ticks of base_dt/12, planes every H/12
    assert planes_per_step(4) == 12
    for consumer in (2, 4, 8, 16):
        for provider in (1, 2, 4, 8):
            if provider >= consumer:
                continue
            ends = range(4 * provider, 12 * consumer + 1, 4 * provider)
            seen = sorted(p for p in (trajectory_plane(0, consumer, t, 12) for t in ends) if p is not None)
            if 2 * provider == consumer:
                assert seen == list(range(2, 13, 2))
                assert plane_weights(3, consumer, provider, 4) == (2, 4, 0.5, 0.5)
                assert plane_weights(4, consumer, provider, 4) == (4, 4, 1.0, 0.0)
            else:   # two or more levels finer: every plane is a split end
                assert seen == list(range(1, 13))
                assert all(plane_weights(p, consumer, provider, 4)[2:] == (1.0, 0.0) for p in range(13))
    # an event tail records only the split ends, every fourth plane
    assert plane_weights(1, 4, 4, 4) == (0, 4, 0.75, 0.25)
    assert plane_weights(6, 4, 4, 4) == (4, 8, 0.5, 0.5)


def test_interpolation_reads_only_planes_the_provider_records():
    # Whatever even N: both ends of an interpolation are planes a provider split end lands on,
    # they bracket the plane, and the weights sum to one.
    for count in (2, 4, 6, 8, 10):
        planes = planes_per_step(count)
        for consumer in (1, 2, 4, 8, 16):
            for provider in (1, 2, 4, 8, 16):
                if provider > consumer:
                    continue
                ends = range(count * provider, planes * consumer + 1, count * provider)
                recorded = {0} | {p for p in (trajectory_plane(0, consumer, t, planes) for t in ends)
                                  if p is not None}
                assert planes in recorded
                for plane in range(planes + 1):
                    a, b, wa, wb = plane_weights(plane, consumer, provider, count)
                    assert a in recorded and b in recorded and a <= plane <= b
                    assert abs(wa + wb - 1.0) < 1e-15 and (wb == 0.0) == (a == plane == b)
                    if wb:
                        assert abs(wa * a + wb * b - plane) < 1e-12


def test_odd_or_too_few_substeps_are_refused_with_the_breakage_named():
    for good in (2, 4, 6, 8):
        assert lts_acoustic_substeps(good) == good
    for bad in (0, 1, 3, 5, 2.5):
        with pytest.raises(ValueError, match="not a plane of its clock"):
            lts_acoustic_substeps(bad)


def test_the_macro_plan_spends_exactly_the_requested_base_steps():
    assert plan_macros(1200, 16) == (16,) * 75
    assert plan_macros(1200, 16, warmup_steps=2) == (1, 1) + (16,) * 74 + (8, 4, 2)
    assert plan_macros(1, 16) == (1,)
    assert plan_macros(0, 16) == ()
    assert plan_macros(40, 4) == (4,) * 10
    for steps in range(0, 70):
        assert sum(plan_macros(steps, 16)) == steps
    with pytest.raises(ValueError):
        plan_macros(10, 3)


def lts_upwind(rates, steps, *, reflux=True, seed=3, factor=1.5, constant=False):
    """Upwind transport on the ring through the SAME tables, event order and register law.

    Fine first: a class advances its owned cells and its scratch zone of
    coarser cells; the zone is reset wherever its owner stands at the step's
    start, the ring beyond it is frozen at the owner's accepted value, and a
    finer neighbour is read at its accepted start-of-step value.  After its
    step the class is refluxed with the finer class's accumulated flux.  This
    is the conservation mechanism of the CUDA route in plain arithmetic, so a
    wrong table, sign or event order shows up here.
    """

    rates = np.asarray(rates, dtype=np.int32)
    plan, coe = plan_for(rates, factor=factor)
    n = rates.size
    rng = np.random.default_rng(seed)
    truth = np.ones(n) if constant else 1.0 + rng.random(n)   # accepted value at the owner's time
    start_of = truth.copy()                        # a finer owner's value at the reader's step start
    velocity = 0.2                                 # Courant number at rate 1; flow is toward +index
    cls = plan.cell_class
    local = [truth[t.layout.cell_l2g].copy() for t in plan.classes]
    stands = np.zeros(len(plan.classes), dtype=int)
    reg_fine = np.zeros(plan.n_register_edges)
    reg_coarse = np.zeros(plan.n_register_edges)
    initial = float(truth.sum())
    clock = 0

    def owned_total():
        return float(sum(q[:t.layout.n_owned_cells].sum() for q, t in zip(local, plan.classes)))

    for macro in plan_macros(steps, int(rates.max())):
        for kind, index, begin, length in macro_schedule(plan.rates, macro):
            tables = plan.classes[index]
            layout = tables.layout
            q = local[index]
            n_owned = layout.n_owned_cells
            g2l = layout.cell_g2l()
            owned = layout.cell_l2g[:n_owned]
            if kind == "step":
                t0, dt = clock + begin, float(length)
                assert stands[index] == t0
                halo = layout.cell_l2g[n_owned:]
                owner = tables.halo_cell_class & CLASS_MASK
                carried = (tables.halo_cell_class & SCRATCH_FLAG) != 0
                finer = owner > index
                tail = q[n_owned:]
                tail[finer] = start_of[halo[finer]]
                at_truth = stands[owner] == t0
                write = ~finer & (~carried | at_truth)
                tail[write] = truth[halo[write]]
                advance = np.ones(layout.n_local_cells, dtype=bool)
                advance[n_owned:] = carried
                start = q.copy()
                flux = {}                           # through global edge e (cells e, e+1), leaving cell e
                for cell_local in np.flatnonzero(advance):
                    cell = int(layout.cell_l2g[cell_local])
                    for edge in ((cell - 1) % n, cell):
                        donor = g2l[coe[edge, 0]]
                        assert donor >= 0, "a carried cell read a cell that is not local"
                        flux[edge] = velocity * dt * start[donor]
                    q[cell_local] += flux[(cell - 1) % n] - flux[cell]
                for edges, slots, target in ((tables.fine_edge_local, tables.fine_slot, reg_fine),
                                             (tables.coarse_edge_local, tables.coarse_slot, reg_coarse)):
                    for edge_local, slot in zip(edges, slots):
                        edge = int(layout.edge_l2g[edge_local])
                        if edge not in flux:
                            # the coarse side's own estimate through an edge it does not advance a cell beyond
                            flux[edge] = velocity * dt * start[g2l[coe[edge, 0]]]
                        target[slot] += flux[edge]
                stands[index] = t0 + length
            else:
                if reflux:
                    for row, cell_local in enumerate(tables.reflux_cell_local):
                        for slot, coef in zip(tables.reflux_slot[row], tables.reflux_coef[row]):
                            q[cell_local] += coef * (reg_fine[slot] - reg_coarse[slot])
                reg_fine[tables.coarse_slot] = reg_coarse[tables.coarse_slot] = 0.0
                truth[owned] = q[:n_owned]
                # every finer class stands at the same instant: their values start this class's next step
                finer_cells = cls > index
                start_of[finer_cells] = truth[finer_cells]
        clock += macro
    final = np.empty(n)
    for q, t in zip(local, plan.classes):
        final[t.layout.cell_l2g[:t.layout.n_owned_cells]] = q[:t.layout.n_owned_cells]
    return initial, owned_total(), final, reg_fine, reg_coarse


@pytest.mark.parametrize("factor", [0.5, 1.5, 4.0])
@pytest.mark.parametrize("rates", [[1] * 16 + [2] * 32,
                                   [1] * 12 + [2] * 12 + [4] * 24 + [2] * 12,
                                   [1] * 10 + [2] * 2 + [4] * 30 + [2] * 2])
def test_the_register_law_conserves_and_its_absence_is_seen(rates, factor):
    initial, final, q, fine, coarse = lts_upwind(rates, 48, factor=factor)
    assert abs(final - initial) <= 64 * np.finfo(np.float64).eps * initial
    assert np.all(np.isfinite(q)) and q.min() > 0.0
    assert not fine.any() and not coarse.any()
    # the instrument must see a planted defect: without reflux the total drifts
    initial, broken, _, _, _ = lts_upwind(rates, 48, reflux=False, factor=factor)
    assert abs(broken - initial) > 1e-6 * initial


def test_a_constant_stays_constant_through_every_interface():
    _, _, q, _, _ = lts_upwind([1] * 12 + [2] * 12 + [4] * 24 + [2] * 12, 48, constant=True)
    np.testing.assert_allclose(q, 1.0, rtol=0, atol=64 * np.finfo(np.float64).eps)


def test_active_requests_validate_and_still_refuse_the_other_schedules():
    from types import SimpleNamespace

    from hexcore.config_fullstep_lts import FullStepLtsConfig, validate_fullstep_lts_request
    from hexcore.errors import ConfigurationRefusal

    assert validate_fullstep_lts_request(FullStepLtsConfig(True, fullstep_lts_mode="active"))
    request = dict(fullstep_local_timestep=True, fullstep_lts_mode="active")
    with pytest.raises(ConfigurationRefusal, match="same accepted interval"):
        validate_fullstep_lts_request(SimpleNamespace(**request, local_timestep=True))
    # Retired 2026-10-12: a boundary series (--lbc-dir) was refused here; a
    # limited-area request now validates and its zone takes one class.
    assert validate_fullstep_lts_request(SimpleNamespace(**request, lbc_dir="lbc"))
    with pytest.raises(ConfigurationRefusal, match="RK2/split3"):
        validate_fullstep_lts_request(SimpleNamespace(**request, time_integration_order=3))
    # the MPAS-A v8.4.2 default of four sub-steps, an explicit two, and any even count are taken
    for substeps in (None, 2, 4, 6):
        assert validate_fullstep_lts_request(SimpleNamespace(**request, acoustic_substeps=substeps))
    with pytest.raises(ConfigurationRefusal, match="odd count"):
        validate_fullstep_lts_request(SimpleNamespace(**request, acoustic_substeps=3))
