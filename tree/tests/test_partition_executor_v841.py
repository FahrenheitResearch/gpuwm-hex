"""CPU tests for the two-rank executor, net peer and 2-way part merge.

Synthetic mesh: a 12-cell cycle (cell i neighbours i-1 and i+1 mod 12), split
6/6.  Every table below is hand-derived in the comments; the exchange tests
then prove the round machinery restores owner truth over a REAL socket pair
(``PeerLink`` itself under test) with numpy standing in for cupy -- the
pack/unpack law is array-module agnostic by construction.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from hexcore.partition_assets_v841 import build_partition_layouts  # noqa: E402
from hexcore.partition_executor_v841 import (  # noqa: E402
    ExecutorError,
    HaloExchangeTables,
    HaloExchanger,
    expected_round_sequence,
)
from hexcore.partition_net_v841 import NetPeerError, PeerLink, connect_peer  # noqa: E402

N_CELLS = 12
NLEV = 3


@pytest.fixture()
def cycle_layouts(tmp_path):
    graph = tmp_path / "cycle.graph.info"
    lines = [f"{N_CELLS} {N_CELLS}"]
    for cell in range(N_CELLS):
        lines.append(f"{(cell - 1) % N_CELLS + 1} {(cell + 1) % N_CELLS + 1}")
    graph.write_text("\n".join(lines) + "\n")
    part = tmp_path / "cycle.graph.info.part.2"
    part.write_text("\n".join(["0"] * 6 + ["1"] * 6) + "\n")
    cells_on_edge = np.array(
        [[edge, (edge + 1) % N_CELLS] for edge in range(N_CELLS)], dtype=np.int32
    )
    cells_on_vertex = np.array(
        [[vertex, (vertex + 1) % N_CELLS] for vertex in range(N_CELLS)], dtype=np.int32
    )
    return build_partition_layouts(
        graph,
        part,
        halo_rings=2,
        cells_on_vertex=cells_on_vertex,
        cells_on_edge=cells_on_edge,
    )


def test_expected_round_sequence_matches_the_step_structure():
    # design 44-round table + the final scalar publish round (the measured
    # step-12 phase-2 halo hole) = 45 with the limiter, 44 without
    sequence = expected_round_sequence()
    assert len(sequence) == 45
    assert "".join(sequence) == (
        ("CB" + "C" * 3 + "B" + "C" * 6 + "B") * 3 + "DDDE" + "D" + "A"
    )
    assert sequence.count("A") == 1
    assert sequence.count("B") == 9
    assert sequence.count("C") == 30
    assert sequence.count("D") == 4
    assert sequence.count("E") == 1
    off = expected_round_sequence(monotonic=False)
    assert "".join(off) == (
        ("CB" + "C" * 3 + "B" + "C" * 6 + "B") * 3 + "DDD" + "D" + "A"
    )


def test_tables_match_hand_derivation(cycle_layouts):
    # P0: owned cells 0..5; halo ring1 [6,11], ring2 [7,10].
    # P1: owned cells 6..11; halo ring1 [0,5], ring2 [1,4].
    p0 = HaloExchangeTables.build(cycle_layouts, 0)
    p1 = HaloExchangeTables.build(cycle_layouts, 1)
    assert list(cycle_layouts[0].cell_l2g) == [0, 1, 2, 3, 4, 5, 6, 11, 7, 10]
    assert list(cycle_layouts[1].cell_l2g) == [6, 7, 8, 9, 10, 11, 0, 5, 1, 4]
    # P0 sends what P1's halo order [0,5,1,4] demands, as P0-local indices.
    assert list(p0.cell_send) == [0, 5, 1, 4]
    # P1 sends what P0's halo order [6,11,7,10] demands, as P1-local indices.
    assert list(p1.cell_send) == [0, 5, 1, 4]
    # Edge ownership: min-endpoint law puts edges {0..5,11} on P0, {6..10} on P1.
    assert list(cycle_layouts[0].edge_l2g) == [0, 1, 2, 3, 4, 5, 11, 6, 10, 7, 9]
    assert list(cycle_layouts[1].edge_l2g) == [6, 7, 8, 9, 10, 5, 11, 0, 4, 1, 3]
    assert list(p0.edge_send) == [5, 6, 0, 4, 1, 3]
    assert list(p1.edge_send) == [0, 4, 1, 3]
    # every send index addresses an OWNED entity (owner-computes law)
    assert p0.cell_send.max() < p0.n_owned_cells
    assert p1.edge_send.max() < p1.n_owned_edges
    assert p0.cell_recv_count == 4 and p0.edge_recv_count == 4
    assert p1.cell_recv_count == 4 and p1.edge_recv_count == 6


def _global_cell_field(seed: float) -> np.ndarray:
    return (
        seed
        + 100.0 * np.arange(NLEV, dtype=np.float32)[:, None]
        + np.arange(N_CELLS, dtype=np.float32)[None, :]
    ).astype(np.float32)


def _global_edge_field(seed: float) -> np.ndarray:
    return _global_cell_field(seed + 0.5)


def _local_with_poisoned_halo(global_arr, l2g, n_owned):
    local = np.ascontiguousarray(global_arr[..., l2g])
    local[..., n_owned:] = np.float32(-9.999e9)
    return local


def _link_pair(timeout=30.0):
    result = {}
    import socket

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    def serve_on_port():
        result["server"] = connect_peer(
            role="server", host="127.0.0.1", port=port, rendezvous_seconds=timeout
        )

    thread = threading.Thread(target=serve_on_port, daemon=True)
    thread.start()
    client = connect_peer(
        role="client", host="127.0.0.1", port=port, rendezvous_seconds=timeout
    )
    thread.join(timeout=timeout)
    return result["server"], client


def test_full_step_schedule_restores_owner_truth_end_to_end(cycle_layouts):
    """Both ranks run the whole 44-round schedule; every halo value must end
    bitwise equal to the owner's value and every owned value must be untouched."""

    globals_ = {
        "rho": _global_cell_field(1000.0),
        "rho_theta": _global_cell_field(2000.0),
        "rho_u": _global_edge_field(3000.0),
        "rho_w": _global_cell_field(4000.0),
        "scalars": np.stack([_global_cell_field(5000.0 + 10 * s) for s in range(6)]),
        "theta_m": _global_cell_field(6000.0),
        "exner": _global_cell_field(7000.0),
        "density_perturbation": _global_cell_field(8000.0),
        "rho_theta_perturbation": _global_cell_field(9000.0),
        "pressure_perturbation": _global_cell_field(10000.0),
        "normal_velocity": _global_edge_field(11000.0),
        "vertical_velocity": _global_cell_field(12000.0),
        "rho_pp": _global_cell_field(13000.0),
        "rtheta_pp": _global_cell_field(14000.0),
        "ru_p": _global_edge_field(15000.0),
        "scale_in": np.stack([_global_cell_field(16000.0 + s) for s in range(6)]),
        "scale_out": np.stack([_global_cell_field(17000.0 + s) for s in range(6)]),
    }
    edge_fields = {"rho_u", "normal_velocity", "ru_p"}
    server, client = _link_pair()
    links = {0: server, 1: client}
    failures: list[BaseException] = []
    locals_by_rank: dict[int, dict[str, np.ndarray]] = {}

    def run_rank(rank: int) -> None:
        try:
            layout = cycle_layouts[rank]
            tables = HaloExchangeTables.build(cycle_layouts, rank)
            fields = {}
            for name, value in globals_.items():
                if name in edge_fields:
                    fields[name] = _local_with_poisoned_halo(
                        value, layout.edge_l2g, layout.n_owned_edges
                    )
                else:
                    fields[name] = _local_with_poisoned_halo(
                        value, layout.cell_l2g, layout.n_owned_cells
                    )
            locals_by_rank[rank] = fields
            state = SimpleNamespace(
                rho=fields["rho"],
                rho_theta=fields["rho_theta"],
                rho_u=fields["rho_u"],
                rho_w=fields["rho_w"],
                scalars=fields["scalars"],
            )
            saved = SimpleNamespace(
                theta_m=fields["theta_m"],
                exner=fields["exner"],
                density_perturbation=fields["density_perturbation"],
                rho_theta_perturbation=fields["rho_theta_perturbation"],
                pressure_perturbation=fields["pressure_perturbation"],
                normal_velocity=fields["normal_velocity"],
                vertical_velocity=fields["vertical_velocity"],
            )
            acoustic = SimpleNamespace(
                rho_pp=fields["rho_pp"],
                rtheta_pp=fields["rtheta_pp"],
                ru_p=fields["ru_p"],
            )
            exchanger = HaloExchanger(tables, links[rank], xp=np)
            exchanger.begin_step(1)
            for _subcycle in range(3):
                for acoustic_steps in (1, 3, 6):
                    for _small in range(acoustic_steps):
                        exchanger.round_acoustic(acoustic)
                    exchanger.round_stage_entry(state, saved)
            for stage in (1, 2, 3):
                exchanger.round_transport(fields["scalars"])
                if stage == 3:
                    exchanger.round_fct_scale(fields["scale_in"], fields["scale_out"])
            exchanger.round_transport(fields["scalars"])  # final publish round
            exchanger.round_step_boundary(state, saved)
            exchanger.end_step()
            locals_by_rank[rank + 10] = exchanger.receipt()  # keep for assertions
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)

    thread = threading.Thread(target=run_rank, args=(1,), daemon=True)
    thread.start()
    run_rank(0)
    thread.join(timeout=60)
    server.close()
    client.close()
    assert not failures, failures

    for rank in (0, 1):
        layout = cycle_layouts[rank]
        for name, value in globals_.items():
            local = locals_by_rank[rank][name]
            if name in edge_fields:
                l2g = layout.edge_l2g
            else:
                l2g = layout.cell_l2g
            expected = value[..., l2g]
            np.testing.assert_array_equal(
                local, expected, err_msg=f"rank {rank} field {name}"
            )
    receipt = locals_by_rank[10]
    assert receipt["rounds"]["A"]["rounds"] == 1
    assert receipt["rounds"]["B"]["rounds"] == 9
    assert receipt["rounds"]["C"]["rounds"] == 30
    assert receipt["rounds"]["D"]["rounds"] == 4
    assert receipt["rounds"]["E"]["rounds"] == 1


def test_out_of_schedule_round_refuses(cycle_layouts):
    tables = HaloExchangeTables.build(cycle_layouts, 0)

    class DeadLink:
        def exchange(self, tag, payload):  # pragma: no cover - must not be reached
            raise AssertionError("the schedule law must refuse before any wire use")

    exchanger = HaloExchanger(tables, DeadLink(), xp=np)
    exchanger.begin_step(1)
    acoustic = SimpleNamespace(
        rho_pp=np.zeros((NLEV, tables.n_local_cells), np.float32),
        rtheta_pp=np.zeros((NLEV, tables.n_local_cells), np.float32),
        ru_p=np.zeros((NLEV, tables.n_local_edges), np.float32),
    )
    state = SimpleNamespace(
        rho=np.zeros((NLEV, tables.n_local_cells), np.float32),
        rho_theta=np.zeros((NLEV, tables.n_local_cells), np.float32),
        rho_u=np.zeros((NLEV, tables.n_local_edges), np.float32),
        rho_w=np.zeros((NLEV, tables.n_local_cells), np.float32),
        scalars=np.zeros((6, NLEV, tables.n_local_cells), np.float32),
    )
    saved = SimpleNamespace(
        theta_m=state.rho, exner=state.rho, density_perturbation=state.rho,
        rho_theta_perturbation=state.rho, pressure_perturbation=state.rho,
        normal_velocity=state.rho_u, vertical_velocity=state.rho,
    )
    with pytest.raises(ExecutorError, match="expects 'C'"):
        exchanger.round_step_boundary(state, saved)
    with pytest.raises(ExecutorError, match="round sequence"):
        exchanger.end_step()


def test_peer_link_tag_divergence_refuses():
    server, client = _link_pair()
    errors: list[BaseException] = []

    def client_side():
        try:
            client.exchange("B", b"client-payload")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=client_side, daemon=True)
    thread.start()
    with pytest.raises(NetPeerError, match="diverged"):
        server.exchange("A", b"server-payload")
    thread.join(timeout=30)
    server.close()
    client.close()
    assert len(errors) == 1 and isinstance(errors[0], NetPeerError)


def test_owned_union_reconstruction_is_bytewise_exact(cycle_layouts):
    """The invariance-gate law: scattering each rank's owned region rebuilds
    the global array byte-for-byte (exact-cover), for cells and edges."""

    from hexcore.partition_state_v841 import (
        scatter_owned_cells,
        scatter_owned_edges,
    )

    cell_truth = _global_cell_field(42.0)
    edge_truth = _global_edge_field(43.0)
    cell_rebuilt = np.full_like(cell_truth, np.nan)
    edge_rebuilt = np.full_like(edge_truth, np.nan)
    for layout in cycle_layouts:
        local_cells = np.ascontiguousarray(cell_truth[..., layout.cell_l2g])
        local_edges = np.ascontiguousarray(edge_truth[..., layout.edge_l2g])
        # poison the halo: reconstruction must never read it
        local_cells[..., layout.n_owned_cells :] = np.float32(-1e30)
        local_edges[..., layout.n_owned_edges :] = np.float32(-1e30)
        scatter_owned_cells(cell_rebuilt, local_cells, layout)
        scatter_owned_edges(edge_rebuilt, local_edges, layout)
    assert cell_rebuilt.tobytes() == cell_truth.tobytes()
    assert edge_rebuilt.tobytes() == edge_truth.tobytes()


def test_axis_classification_and_axis0_reconstruction(cycle_layouts):
    from hexcore.partition_executor_v841 import (
        classify_partition_axis,
        owned_block,
        scatter_owned_axis,
    )

    layout = cycle_layouts[0]
    kwargs = dict(
        n_local_cells=layout.n_local_cells,
        n_local_edges=layout.n_local_edges,
        n_local_vertices=layout.n_local_vertices,
    )
    assert classify_partition_axis((NLEV, layout.n_local_cells), **kwargs) == (
        "cell",
        1,
    )
    # leading-axis (Arwen seam convention) classification
    assert classify_partition_axis((layout.n_local_cells, 4), **kwargs)[1] == 0
    assert classify_partition_axis((NLEV, NLEV), **kwargs) == ("none", None)

    # axis-0 reconstruction round-trip across both ranks
    truth = np.arange(N_CELLS * 4, dtype=np.float32).reshape(N_CELLS, 4)
    rebuilt = np.full_like(truth, np.nan)
    for candidate in cycle_layouts:
        local = np.ascontiguousarray(truth[candidate.cell_l2g, :])
        local[candidate.n_owned_cells :, :] = np.float32(-1e30)
        block = owned_block(local, 0, candidate.n_owned_cells)
        scatter_owned_axis(
            rebuilt, block, candidate.cell_l2g, candidate.n_owned_cells, 0
        )
    assert rebuilt.tobytes() == truth.tobytes()


def test_invariance_comparator_law():
    import json

    from compare_2gpu_invariance_v841 import compare_boundary_records

    atmosphere = {
        "model_time_seconds": 120.0,
        "state": {"fields": {"rho": {"bytes_sha256": "aa"}}, "sha256": "s1"},
        "saved_diagnostics": {"fields": {}, "sha256": "s2"},
        "sha256": "top",
    }
    backend = {
        "arrays": {"seam/soil": {"bytes_sha256": "bb", "shape": [12]}},
        "scalars": {
            "identity/constructor_identity_sha256": "sliced",
            "adapter/gwdo_calls": 4,
        },
        "sha256": "b-top",
    }
    reference = {
        "atmosphere": json.loads(json.dumps(atmosphere)),
        "backend": {
            "arrays": json.loads(json.dumps(backend["arrays"])),
            "scalars": {
                "identity/constructor_identity_sha256": "whole-mesh",
                "adapter/gwdo_calls": 4,
            },
            "sha256": "different-top",
        },
    }
    import json as _json

    outcome = compare_boundary_records({"atmosphere": atmosphere, "backend": backend}, reference)
    # identical state, expected-divergent constructor identity only -> invariant
    assert outcome["state_invariant"] is True
    assert outcome["unexpected_metadata_diffs"] == []
    assert len(outcome["metadata_diffs"]) == 1
    assert outcome["metadata_diffs"][0]["expected_divergence"] is True

    # flip one backend array byte -> state divergence
    broken = _json.loads(_json.dumps({"atmosphere": atmosphere, "backend": backend}))
    broken["backend"]["arrays"]["seam/soil"]["bytes_sha256"] = "cc"
    outcome = compare_boundary_records(broken, reference)
    assert outcome["state_invariant"] is False
    assert outcome["state_diffs"][0]["group"] == "backend/arrays"

    # an unexpected scalar diff is not state divergence but must be surfaced
    drifted = _json.loads(_json.dumps({"atmosphere": atmosphere, "backend": backend}))
    drifted["backend"]["scalars"]["adapter/gwdo_calls"] = 5
    outcome = compare_boundary_records(drifted, reference)
    assert outcome["state_invariant"] is True
    assert len(outcome["unexpected_metadata_diffs"]) == 1


def test_build_2way_part_merge(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    from build_2way_part_v841 import merge_to_two_way

    part12 = np.array([0, 4, 8, 9, 1, 2, 3, 5, 6, 7, 10, 11, 0, 4], dtype=np.int64)
    source = tmp_path / "g.graph.info.part.12"
    source.write_text("\n".join(str(v) for v in part12) + "\n")
    out = tmp_path / "g.graph.info.part.2"
    receipt = merge_to_two_way(source, out, small_parts=(0, 4, 8, 9))
    merged = np.loadtxt(out, dtype=np.int64)
    expected = np.where(np.isin(part12, (0, 4, 8, 9)), 1, 0)
    np.testing.assert_array_equal(merged, expected)
    assert receipt["small_side_cells"] == int(expected.sum())
    assert receipt["big_side_cells"] == int((expected == 0).sum())
