"""Coordinator failure, clock and existing driver round protocol checks."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from hexcore.partition_executor_v841 import ExecutorError
from hexcore.partition_samehost_v841 import SameHostHaloExchanger, SameHostRanks


class Link:
    def __init__(self):
        self.aborted = []
        self.tags = []

    def abort(self, reason):
        self.aborted.append(reason)

    def exchange(self, rank, tag, fields):
        self.tags.append((rank, tag, fields))


def test_all_ranks_step_concurrently_on_one_clock():
    link = Link()
    clocks = [0., 0., 0., 0.]
    rendezvous = threading.Barrier(4, timeout=2)
    def advance(rank):
        rendezvous.wait()
        clocks[rank] += 120.
        return rank
    with SameHostRanks([0, 0, 0, 0], link) as ranks:
        assert ranks.step(advance, clocks.__getitem__, dt=120.) == [0, 1, 2, 3]
        assert ranks.time_seconds == 120.
        ranks.step(advance, clocks.__getitem__, dt=120.)
        assert ranks.time_seconds == 240.
    assert not link.aborted


def test_failed_rank_aborts_peers_and_cannot_resume():
    link = Link()
    def operation(rank):
        if rank == 1:
            raise RuntimeError("test failure")
    with SameHostRanks([0, 0], link) as ranks:
        with pytest.raises(RuntimeError, match="test failure"):
            ranks.step(operation, lambda rank: 0., dt=120.)
        assert ranks.time_seconds is None
        with pytest.raises(ExecutorError, match="cannot resume"):
            ranks.call(operation)
    assert link.aborted


@pytest.mark.parametrize("clocks,after", [([0., 1.], False), ([0., 0.], True)])
def test_clock_disagreement_aborts(clocks, after):
    link = Link()
    def advance(rank):
        clocks[rank] += 120. if rank else 121.
    with SameHostRanks([0, 0], link) as ranks:
        with pytest.raises(ExecutorError, match="clocks"):
            ranks.step(advance, clocks.__getitem__, dt=120.)
        assert ranks.time_seconds is None
    assert link.aborted


def test_tags_frame_step_and_round_position():
    link = Link()
    exchanger = SameHostHaloExchanger(SimpleNamespace(rank=2, n_local_cells=3), link, expected_sequence=("D", "D", "A"))
    exchanger.begin_step(4)
    exchanger.round_transport(np.zeros((1, 3)))
    exchanger.round_transport(np.zeros((1, 4)))
    exchanger._round("A", [])
    exchanger.end_step()
    assert [entry[:2] for entry in link.tags] == [(2, "4:0:D"), (2, "4:1:D"), (2, "4:2:A")]
    assert link.tags[1][2][0][1].shape[-1] == 3


def test_bad_round_aborts_before_transport():
    link = Link()
    exchanger = SameHostHaloExchanger(SimpleNamespace(rank=0), link, expected_sequence=("C",))
    exchanger.begin_step(1)
    with pytest.raises(ExecutorError, match="expected C"):
        exchanger.round_transport("wrong")
    assert link.aborted and not link.tags


def test_physics_packets_refresh_complete_halos_without_widening_dynamics():
    from hexcore.partition_samehost_transport_v841 import SameHostTransport
    link = Link()
    table = SimpleNamespace(rank=0, n_local_cells=3, n_local_edges=3)
    ex = SameHostHaloExchanger(table, link, expected_sequence=("B", "P", "A"))
    state = SimpleNamespace(**{n: np.array([[10., 20., -1.]], dtype=np.float32)
        for n in ('rho', 'rho_theta', 'rho_u', 'rho_w', 'scalars')})
    saved = SimpleNamespace(**{n: np.array([[10., 20., -1.]], dtype=np.float32)
        for n in ('theta_m', 'exner', 'density_perturbation', 'rho_theta_perturbation',
                  'pressure_perturbation', 'normal_velocity', 'vertical_velocity')})
    ex.begin_step(1)
    ex.round_stage_entry(state, saved)
    ex.round_physics_candidate(SimpleNamespace(state=state, saved=saved))
    ex.round_step_boundary(state, saved)
    ex.end_step()
    native = link.tags[0][2]
    phase_two, phase_one = [r[2] for r in link.tags[1:]]
    assert all(not n.startswith('physics/') for n, _, _ in native)
    assert [n for n, _, _ in phase_two] == [n for n, _, _ in phase_one]
    assert len(phase_two) == 12 and all(n.startswith('physics/') for n, _, _ in phase_two)
    # Exercise the actual packet selector: native rho uses its narrow plan,
    # while physics rho copies the complete local halo, without modifying
    # either the values or the native plan.
    transport = object.__new__(SameHostTransport)
    transport.xp = np
    transport._cpu = True
    narrow = np.array([0], np.int64)
    full = np.array([0, 1], np.int64)
    transport.field_halo_indices = {(0, 1, 'rho', 'send'): narrow}
    peer = SimpleNamespace(peer=1, cell_send=full, edge_send=full)
    a = transport._make_packet([native[0]], peer, 'send', rank=0)
    b = transport._make_packet([phase_two[0]], peer, 'send', rank=0)
    assert np.array_equal(a.indices[0], narrow)
    assert np.array_equal(b.indices[0], full)
    assert np.array_equal(state.rho, [[10., 20., -1.]])


def test_caller_streams_are_the_rank_streams_and_must_match_the_ranks():
    # The same-host driver builds each rank on the stream it then steps on
    # (one CuPy pool arena per rank). Breakage the check prevents: a rank
    # running on another rank's queue, or a CPU group holding CUDA streams.
    class Device:
        def __init__(self, device):
            self.device = device

        def use(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device))
    with pytest.raises(ValueError, match="one stream per rank"):
        SameHostRanks([0, 0], None, cp=cp, streams=[object()])
    with pytest.raises(ValueError, match="one stream per rank"):
        SameHostRanks([0], None, streams=[object()])
    mine = [object(), object()]
    ranks = SameHostRanks([0, 0], None, cp=cp, streams=mine)
    try:
        assert ranks.streams == mine and ranks.streams is not mine
    finally:
        ranks.close()
