"""Native rank boundary contracts, without a CUDA context or MPI install."""
from dataclasses import dataclass, field
from types import SimpleNamespace
import pickle
import json

import numpy as np
import pytest

from hexcore.mpasphys.mpi_v842 import MpiHaloTransport, placement, round_sequence, local_inputs
from hexcore.mpasphys.moving_mpi_v842 import assemble_columns, prepare_moving
from hexcore.mpasphys.features_v842 import background_due
from hexcore.mpasphys.cycling_v842 import background_fields, recouple_analysis
from hexcore.partition_executor_v841 import ExecutorError


def partitions():
    result = []
    for rank, ids in enumerate(([2, 0, 3, 1], [3, 1, 2, 0])):
        ids = np.array(ids, np.int32)
        result.append(SimpleNamespace(partition=rank, cell_l2g=ids, edge_l2g=ids,
            vertex_l2g=ids, n_owned_cells=2, n_owned_edges=2, n_owned_vertices=2,
            n_local_cells=4, n_local_edges=4, n_local_vertices=4,
            _global_cells=4, _global_edges=4, _global_vertices=4))
    return result


class Comm:
    def __init__(self, packet=None, hosts=None):
        self.packet, self.hosts = packet, hosts
        self.freed = False
        self.sent = None
    def Get_size(self): return 2
    def Get_rank(self): return 0
    def Dup(self): return self
    def Free(self): self.freed = True
    def allgather(self, value): return self.hosts or [value, value]
    def alltoall(self, values):
        self.sent = values
        return [None, self.packet]


def packet(name='scalars', words=None, shape=None, tag='1:0:D'):
    if words is None:
        words = np.array([[0x80000000, 0x7fc12345]], np.uint32).view(np.float32)
    return tag, [(name, 'cell', words.dtype.str, shape or words.shape, words.tobytes())]


def transport(comm, indices=None):
    from hexcore.partition_samehost_tables_v841 import build_all
    return MpiHaloTransport(comm, build_all(partitions()), xp=np, field_indices=indices)


def test_mpi_words_send_receiver_order_and_leave_owners_untouched():
    comm = Comm(packet())
    link = transport(comm)
    words = np.array([[0x7fc54321, 0x80000000, 0xdeadbeef, 0xdeadbeef]], np.uint32)
    owners = words[..., :2].copy()
    stats = link.exchange(0, '1:0:D', [('scalars', words.view(np.float32), 'cell')])
    np.testing.assert_array_equal(words[..., :2], owners)
    np.testing.assert_array_equal(words[..., 2:], [[0x80000000, 0x7fc12345]])
    # Peer order [3,1,2,0] wants our owned globals [2,0] in that order.
    sent = comm.sent[1][1][0]
    np.testing.assert_array_equal(np.frombuffer(sent[-1], np.uint32), owners.ravel())
    assert stats['bytes_sent'] == stats['bytes_received'] == 8
    link.close()
    assert comm.freed
    with pytest.raises(ExecutorError, match='closed'):
        link.exchange(0, '1:0:D', [])


@pytest.mark.parametrize('bad', [
    packet(tag='2:0:D'), packet(name='rho'), packet(shape=(2, 1)),
    ('1:0:D', [('scalars', 'cell', '<f4', (1, 2), b'bad')]),
])
def test_malformed_mpi_packet_cannot_partially_publish(bad):
    comm = Comm(bad)
    link = transport(comm)
    words = np.arange(4, dtype=np.float32).reshape(1, 4)
    original = words.copy()
    with pytest.raises(ExecutorError):
        link.exchange(0, '1:0:D', [('scalars', words, 'cell')])
    np.testing.assert_array_equal(words, original)


def test_native_widths_and_full_physics_publication_are_distinct():
    selected = {(0, 1, 'rho', 'send'): np.array([1]),
                (0, 1, 'rho', 'recv'): np.array([3])}
    comm = Comm(packet('rho', np.array([[42]], np.float32)))
    link = transport(comm, selected)
    field = np.array([[1, 2, -100, -200]], np.float32)
    link.exchange(0, '1:0:D', [('rho', field, 'cell')])
    np.testing.assert_array_equal(field, [[1, 2, -100, 42]])
    comm.packet = packet('physics/rho', np.array([[13, 14]], np.float32), tag='1:1:P')
    link.exchange(0, '1:1:P', [('physics/rho', field, 'cell')])
    np.testing.assert_array_equal(field, [[1, 2, 13, 14]])


def test_samehost_admission_refuses_duplicate_cards_and_other_hosts():
    assert placement(Comm(), (0, 3)) == 0
    with pytest.raises(ValueError, match='distinct'):
        placement(Comm(), (3, 3))
    with pytest.raises(ValueError, match='one host'):
        placement(Comm(hosts=['node-a', 'node-b']), (0, 1))


def test_schedule_uses_rk2_and_places_native_post_transport_before_commit(monkeypatch):
    monkeypatch.setenv('GPUWM_HEX_LIMITER_SPECIES_PER_PASS', '2')
    config = SimpleNamespace(config_dt=60, config_time_integration_order=2,
        config_number_of_sub_steps=4, config_dynamics_split_steps=3,
        config_monotonic=True, config_positive_definite=False)
    sequence = round_sequence(config, 7)
    assert sequence.count('B') == 9
    assert sequence.count('C') == 24
    assert sequence[-2:] == ('P', 'A')
    assert sequence.count('P') == sequence.count('A') == 1
    regional = round_sequence(config, 7, regional=True)
    assert regional.count('D') == sequence.count('D') + 3


def test_moving_assembly_discards_all_halo_and_dummy_words():
    parts = partitions()
    expected = np.arange(12, dtype=np.uint32).reshape(3, 4) + np.uint32(0x7fc00000)
    values = []
    for part in parts:
        buffer = np.full((3, 5), 0xdeadbeef, np.uint32)
        buffer[..., :2] = expected[..., part.cell_l2g[:2]]
        values.append(buffer.view(np.float32))
    actual = assemble_columns(values, parts, entity='cell')
    np.testing.assert_array_equal(actual.view(np.uint32), expected)
    values[1] = values[1].astype(np.float64)
    with pytest.raises(ValueError, match='shape/dtype'):
        assemble_columns(values, parts, entity='cell')


def test_moving_window_is_owned_only():
    parts = partitions()
    expected = np.arange(4, dtype=np.int32)
    values = [expected[p.cell_l2g[:2]] for p in parts]
    np.testing.assert_array_equal(assemble_columns(values, parts, entity='cell'), expected)
    with pytest.raises(ValueError, match='one buffer per rank'):
        assemble_columns(values[:1], parts, entity='cell')


def test_rank_projection_cannot_mutate_global_derived_dataclass():
    @dataclass
    class Derived:
        size: int
        rho: np.ndarray = field(init=False)
        def __post_init__(self):
            self.rho = np.arange(self.size, dtype=np.float32)
    source = Derived(4)
    original = source.rho.copy()
    parts = partitions()
    for part in parts:
        part._global_edges, part._global_vertices = 6, 5
    inputs = (None, SimpleNamespace(derived=source), None, None, None, None, ('qv', 'tke'))
    first = local_inputs(inputs, parts[0])
    second = local_inputs(inputs, parts[1])
    np.testing.assert_array_equal(source.rho, original)
    np.testing.assert_array_equal(first[1].derived.rho, original[parts[0].cell_l2g])
    np.testing.assert_array_equal(second[1].derived.rho, original[parts[1].cell_l2g])
    assert first[1].derived is not second[1].derived


def test_moving_control_is_reconstructed_instead_of_pickling_its_lock(monkeypatch, tmp_path):
    from hexcore.mpasphys import moving_mpi_v842 as moving
    from hexcore.mpasphys.moving_v842 import MovingRoute
    from hexcore.moving_control import MovingControl
    monkeypatch.setattr(moving, 'NativePoolHydration', lambda *args: None)
    route = MovingRoute(control=MovingControl(tmp_path/'control', seal_every_seconds=600))
    layout = SimpleNamespace(configs={'config_start_time': '2026-10-17_00:00:00'}, fields=[])
    inputs = (None, SimpleNamespace(vertical_grid=None), None, None, None, None, ('qv',))
    _, prepared = prepare_moving(route, inputs, None, layout, tmp_path, SimpleNamespace())
    loaded = pickle.loads(pickle.dumps(prepared, protocol=5))
    assert loaded['route'].control is None
    assert loaded['control']['seal_every_seconds'] == 600
    assert route.control is not None


@pytest.mark.parametrize('launcher_failure', [False, True])
def test_mpi_launcher_records_then_discards_its_private_inputs(monkeypatch, tmp_path, launcher_failure):
    from hexcore.mpasphys import launch_v842 as launch
    job = tmp_path/'native-ranks'
    job.mkdir()
    (job/'request.pickle').write_bytes(b'private prepared words')
    path = tmp_path/'native-rank-launch.json'
    path.write_text(json.dumps(dict(completed=False, command=['test-mpi'], visible_devices=[0, 1])))
    for rank in range(2):
        (tmp_path/f'native-physics-receipt.rank{rank}.json').write_text(json.dumps(dict(completed=True, steps_completed=1)))
    def run(*args, **kwargs):
        if launcher_failure:
            raise OSError('MPI launcher unavailable')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launch.subprocess, 'run', run)
    if launcher_failure:
        with pytest.raises(OSError, match='unavailable'):
            launch.execute_native_ranks(job)
    else:
        assert launch.execute_native_ranks(job) == 0
    receipt = json.loads(path.read_text())
    assert receipt['discarded_input_bytes'] == len(b'private prepared words')
    assert receipt['completed'] is not launcher_failure
    assert receipt['steps_completed'] == [1, 1]
    assert not job.exists()
    with pytest.raises(ValueError, match='only once'):
        launch.execute_native_ranks(job)


def test_native_config_survives_process_request_serialization(tmp_path):
    from hexcore.mpasphys.launch_v842 import _dump
    from hexcore.mpasphys.driver_v842 import resolve_namelist
    config = resolve_namelist("&physics\n config_physics_suite='convection_permitting'\n/\n")
    path = tmp_path/'request.pickle'
    _dump(path, config)
    with path.open('rb') as handle:
        restored = pickle.load(handle)
    assert restored.receipt() == config.receipt()
    assert dict(restored.execution_options) == dict(config.execution_options)
    with pytest.raises(TypeError):
        restored.options['config_physics_suite'] = 'none'


def test_preparation_failure_records_and_discards_partial_rank_inputs(monkeypatch, tmp_path):
    from hexcore.mpasphys import launch_v842 as launch, runtime_v842 as runtime
    from hexcore.drivers import samehost_forecast_v841 as samehost
    def runtime_with_seam(*args, execution=None, **kwargs):
        pass
    def failed_partition(arguments, request):
        (request.scratch/'partial-pool.bin').write_bytes(b'partial native input')
        raise ValueError('partition preparation failed')
    monkeypatch.setattr(runtime, 'run_native_dynamics', runtime_with_seam)
    monkeypatch.setattr(samehost, 'request_options', failed_partition)
    launcher = tmp_path/'mpiexec'
    launcher.write_text('unused')
    monkeypatch.setenv('MPAS_NATIVE_MPIEXEC', str(launcher))
    arguments = SimpleNamespace(samehost_devices='0,1', init=tmp_path/'init.nc')
    layout = SimpleNamespace(dimensions={'nCells': 4})
    with pytest.raises(ValueError, match='preparation failed'):
        launch.prepare_native_ranks(arguments, None, layout, tmp_path/'source', [],
                                    None, None, None, 60, 1, tmp_path/'out')
    receipt = json.loads((tmp_path/'out/native-rank-launch.json').read_text())
    assert not receipt['completed']
    assert receipt['discarded_input_bytes'] == len(b'partial native input')
    assert receipt['preparation_error'] == 'partition preparation failed'
    assert not (tmp_path/'out/native-ranks').exists()


def test_rrfs_mpi_refuses_missing_ring_projection_before_preparing_inputs(monkeypatch, tmp_path):
    from hexcore.mpasphys import launch_v842 as launch, runtime_v842 as runtime
    from hexcore.drivers import samehost_forecast_v841 as samehost

    def forbidden(*args, **kwargs):
        raise AssertionError('unsupported rank preparation must not start')

    monkeypatch.setattr(runtime, 'load_native_dynamics_inputs', forbidden)
    monkeypatch.setattr(samehost, 'request_options', forbidden)
    monkeypatch.delenv('MPAS_NATIVE_MPIEXEC', raising=False)
    config = SimpleNamespace(native_layout_prepare=
        'hexcore.mpasphys.rrfs.driver_rrfs:prepare_native_layout')
    with pytest.raises(ValueError, match='rank-local halo-ring ends'):
        launch.prepare_native_ranks(SimpleNamespace(samehost_devices='0,1'), config,
            None, tmp_path/'source', [], None, None, None, 60, 1, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_jedi_output_follows_history_boundary_and_always_final():
    args = SimpleNamespace(jedi_background=True, history_every_minutes=5)
    assert background_due(args, 0, 11, 60)
    assert background_due(args, 5, 11, 60)
    assert not background_due(args, 4, 11, 60)
    assert background_due(args, 11, 11, 60)
    args.jedi_background = False
    assert not background_due(args, 11, 11, 60)


def test_ordinary_native_dispatch_works_with_the_original_runtime_signature(monkeypatch):
    from hexcore.mpasphys import features_v842 as features, runtime_v842 as runtime
    calls = []
    def original(*args, moving=None):
        calls.append((args, moving))
        return 17
    monkeypatch.setattr(runtime, 'run_native_dynamics', original)
    values = (SimpleNamespace(), *[object() for _ in range(10)])
    assert features.run_native_dynamics(*values) == 17
    assert calls == [(values, None)]


def test_native_feature_refuses_missing_runtime_seam_before_input_preparation(monkeypatch):
    from hexcore.mpasphys import features_v842 as features, runtime_v842 as runtime
    monkeypatch.setattr(runtime, 'run_native_dynamics', lambda *args, moving=None: 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('large input preparation must not run')
    monkeypatch.setattr(runtime, 'load_native_dynamics_inputs', forbidden)
    with pytest.raises(RuntimeError, match='RankExecution extension seam'):
        features.run_native_dynamics(SimpleNamespace(jedi_background=True),
            *[object() for _ in range(10)])


def test_jedi_boundary_requests_publication_even_without_a_native_stream_alarm():
    from hexcore.mpasphys.features_v842 import CyclingExecution
    execution = CyclingExecution(SimpleNamespace(jedi_background=True,
        history_every_minutes=5), None, None)
    assert execution.publication_required(0, 11, 60)
    assert execution.publication_required(5, 11, 60)
    assert not execution.publication_required(4, 11, 60)
    assert execution.publication_required(11, 11, 60)


def test_da_recoupling_preserves_native_les_carriers(monkeypatch, tmp_path):
    from hexcore.mpasphys import cycling_v842 as cycling
    original = np.arange(4 * 2 * 4, dtype=np.float32).reshape(4, 2, 4)
    original[3].view(np.uint32)[0, 0] = 0x80000000
    old_state = SimpleNamespace(scalars=original, time_seconds=300)
    mesh, grid, reference, saved = object(), SimpleNamespace(vertical_grid=object(), terrain_metrics=object()), object(), object()
    inputs = (mesh, grid, None, old_state, reference, saved, ('qv', 'qc', 'qr', 'tke'))
    args = SimpleNamespace(da_analysis=tmp_path/'analysis.nc', init=tmp_path/'init.nc')
    layout = SimpleNamespace(configs=dict(config_do_DAcycling=True, config_do_restart=True,
        config_start_time='2026-10-17_00:05:00'), dimensions=dict(nCells=4, nEdges=4, nVertLevels=2))
    calls = []
    def recouple(path, **options):
        calls.append(options)
        return SimpleNamespace(scalars=np.full((3, 2, 4), 17, np.float32)), saved, {}
    monkeypatch.setattr(cycling.contract, 'recouple_da_cycling', recouple)
    result, receipt = recouple_analysis(args, layout, inputs, work_dir=tmp_path)
    np.testing.assert_array_equal(result[3].scalars[:3], 17)
    np.testing.assert_array_equal(result[3].scalars[3].view(np.uint32), original[3].view(np.uint32))
    assert calls[0]['scalar_names'] == ('qv', 'qc', 'qr')
    assert calls[0]['time_seconds'] == 300
    assert receipt['native_species_retained'] == ['tke']
    layout.configs['config_do_restart'] = False
    with pytest.raises(ValueError, match='restart'):
        recouple_analysis(args, layout, inputs, work_dir=tmp_path)


def test_background_copies_native_soil_and_all_species_without_dummy():
    from hexcore.jedi_contract import BACKGROUND_FIELDS
    dims = dict(nCells=4, nEdges=6, nVertices=5, nVertLevels=2, nSoilLevels=4, num_scalars=3)
    rows, data = [], {}
    for name in BACKGROUND_FIELDS:
        entity = 'nEdges' if name == 'u' else 'nCells'
        leading = ('num_scalars', 'nVertLevels') if name == 'scalars' else ('nSoilLevels',) if name in ('smois', 'tslb') else ()
        dimensions = [*leading, entity]
        shape = [dims[d] + int(d == entity) for d in dimensions]
        rows.append(dict(name=name, pool='pool', pool_name=name, dimensions=dimensions,
                         components=['qv', 'qc', 'tke'] if name == 'scalars' else []))
        data[name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        data[name][..., -1] = np.nan
    session = SimpleNamespace(native=SimpleNamespace(get_field=lambda pool, name: data[name]))
    fields, provenance = background_fields(session, SimpleNamespace(fields=rows, dimensions=dims), ('qv', 'qc', 'tke'))
    assert fields['smois'].shape == (4, 4)
    assert fields['u'].shape == (6,)
    assert fields['tke'].shape == (2, 4)
    assert 'native soil column' in provenance['tslb']
    data['rho'][0] = np.nan
    with pytest.raises(ValueError, match='non-finite'):
        background_fields(session, SimpleNamespace(fields=rows, dimensions=dims), ('qv', 'qc', 'tke'))
