"""Packets follow each field's receive layer, preserving receiver order."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest

from test_build_samehost_layouts_v841 import _grid
from hexcore.mesh import Mesh
from hexcore.drivers.samehost_layouts_v841 import build_assets
from hexcore.partition_assets_v841 import load_layouts_npz
from hexcore.partition_samehost_tables_v841 import build_all
from hexcore.partition_field_halos_v841 import build_field_indices, FIELD_WIDTHS, FIELD_KINDS
from hexcore.partition_samehost_transport_v841 import SameHostTransport, SameHostTransportError, choose_copy_paths


def _assets(tmp_path):
    grid = _grid(tmp_path)
    part = tmp_path / "part2"
    part.write_text("0\n" * 6 + "1\n" * 6)
    receipt = build_assets(grid, ranks=2, part=part, output_dir=tmp_path / "assets")
    layouts = load_layouts_npz(receipt["layouts"], expect_mesh_sha256=receipt["grid_sha256"],
                              expect_part_sha256=receipt["part_sha256"], expect_halo_rings=3)
    mesh = Mesh.from_netcdf(grid, validate=False)
    return layouts, mesh.arrays["cellsOnEdge"]


def test_reciprocal_field_indices_select_the_receiver_layer_and_global_ids(tmp_path):
    layouts, edges = _assets(tmp_path)
    indexes = build_field_indices(layouts, edges)
    for (rank, peer, name, direction), index in indexes.items():
        if direction != "recv":
            continue
        kind = FIELD_KINDS[name]
        send = indexes[peer, rank, name, "send"]
        assert np.array_equal(getattr(layouts[rank], kind + "_l2g")[index],
                              getattr(layouts[peer], kind + "_l2g")[send])
        if kind == "cell":
            assert np.all(layouts[rank].cell_ring[index] <= FIELD_WIDTHS[name])
        assert not index.flags.writeable
    assert len(indexes[0, 1, "rho_pp", "recv"]) == len(indexes[0, 1, "scalars", "recv"])
    assert FIELD_WIDTHS["rho_pp"] == FIELD_WIDTHS["rtheta_pp"] == 2


def test_native_u_layer_three_includes_edges_bounding_cell_layer_three(tmp_path):
    layouts, edges = _assets(tmp_path)
    indexes = build_field_indices(layouts, edges)
    # Edge 3 joins global cells 2 and 3. On rank 1, cell 3 is layer 3
    # and cell 2 is absent. This outer native u layer must still be received.
    local = int(np.flatnonzero(layouts[1].edge_l2g == 3)[0])
    assert local in indexes[1, 0, "normal_velocity", "recv"]
    assert local in indexes[1, 0, "rho_u", "recv"]
    assert local not in indexes[1, 0, "ru_p", "recv"]


def test_real_cpu_packets_preserve_unselected_halos_and_all_owned_storage(tmp_path):
    layouts, edges = _assets(tmp_path)
    tables = build_all(layouts)
    indexes = build_field_indices(layouts, edges)
    transport = SameHostTransport(tables, (0, 0), xp=np, field_halo_indices=indexes)
    fields = []
    for layout in layouts:
        fields.append(tuple((name, np.full((2, layout.n_local_cells), -999, np.int64), "cell")
                            for name in ("rho_pp", "scalars")))
        for _, array, _ in fields[-1]:
            array[:, :layout.n_owned_cells] = layout.cell_l2g[:layout.n_owned_cells]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda rank: transport.exchange(rank, "C", fields[rank]), range(2)))
    for rank, layout in enumerate(layouts):
        for name, array, _ in fields[rank]:
            expected = np.full(layout.n_local_cells, -999, np.int64)
            expected[:layout.n_owned_cells] = layout.cell_l2g[:layout.n_owned_cells]
            receive = indexes[rank, 1 - rank, name, "recv"]
            expected[receive] = layout.cell_l2g[receive]
            assert np.array_equal(array, np.broadcast_to(expected, array.shape))
    transport.close()


def test_invalid_field_indices_refuse_before_cuda_packet_allocation(tmp_path):
    layouts, edges = _assets(tmp_path)
    indexes = build_field_indices(layouts, edges)
    indexes[0, 1, "rho_pp", "recv"] = np.array([0], np.int64)
    with pytest.raises(SameHostTransportError, match="ordered subset"):
        SameHostTransport(build_all(layouts), (0, 0), xp=np, field_halo_indices=indexes)


def test_mixed_socket_auto_keeps_local_peers_and_stages_only_cross_socket():
    nodes = {device: device // 4 for device in range(8)}
    peers = {(a, b): True for a in range(8) for b in range(8) if a != b}
    routes = choose_copy_paths(range(8), peers, numa_nodes=nodes)
    assert all(route == ("local" if a == b else "peer" if a // 4 == b // 4 else "staged")
               for (a, b), route in routes.items())


@pytest.mark.gpu
def test_rank_health_checks_every_owned_value_and_ignores_disposable_halo_values():
    import cupy as cp
    from test_step_gate_batch import health_stack
    from hexcore.drivers import run_cuda_v841_forecast as forecast
    stack = health_stack(cp)
    stack["owned_health_extents"] = {5: 2}
    atmosphere = stack["driver"].atmosphere
    for owner in (atmosphere.state, atmosphere.saved):
        for value in vars(owner).values():
            value[..., 2:] = cp.nan
    # A precomputed whole-local envelope must not override owned bounds.
    stack["driver"]._step_gate_envelope = (atmosphere, {"invalid": "whole local halo"})
    receipt = forecast.step_health_gate(stack, 3, cp)
    assert receipt["finite"] and receipt["entity_scope"] == "owned partition union"
    atmosphere.saved.vertical_velocity[0, 0] = 201
    with pytest.raises(FloatingPointError, match="200.0"):
        forecast.step_health_gate(stack, 4, cp)
