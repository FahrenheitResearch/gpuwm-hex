"""Byte identity and ownership contracts for deferred history frames."""
import hashlib
import numpy as np
import pytest
from types import SimpleNamespace
from hexcore.snapshot_transfer import SnapshotTransfer, array_records


@pytest.fixture
def fake_cp():
    """Queued copies become readable only at an explicit event/stream wait."""
    class Stream:
        def __init__(self):
            self.pending = []
            self.waits = 0

        def synchronize(self):
            self.waits += 1
            for source, target in self.pending:
                target[:] = source.values
            self.pending.clear()

    stream = Stream()

    class Device:
        id = 0

        def __init__(self, device=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Event:
        def __init__(self, **kwargs):
            self.stream = None
            self.waits = 0

        def record(self, producer):
            self.stream = producer

        def synchronize(self):
            self.waits += 1
            self.stream.synchronize()

    class Array:
        def __init__(self):
            self.values = np.arange(8, dtype='f4')
            self.shape, self.dtype, self.nbytes = self.values.shape, self.values.dtype, self.values.nbytes

        def get(self, *, out, stream, blocking):
            assert not blocking
            stream.pending.append((self, out))

    return SimpleNamespace(ndarray=Array, ascontiguousarray=lambda value: value,
                           cuda=SimpleNamespace(Device=Device, Event=Event,
                                                get_current_stream=lambda: stream,
                                                alloc_pinned_memory=bytearray))


@pytest.mark.parametrize('sealed', [False, True])
def test_failed_capture_drains_queued_copies_before_releasing_owners(fake_cp, sealed):
    stream = fake_cp.cuda.get_current_stream()
    transfer = SnapshotTransfer(fake_cp)
    source = fake_cp.ndarray()
    with pytest.raises(ValueError, match='diagnostic refused'):
        with transfer:
            target = transfer.download(source, fake_cp)
            if sealed:
                transfer.seal()
            assert transfer.sources and stream.waits == 0
            raise ValueError('diagnostic refused')
    assert stream.waits == 1 and not stream.pending and not transfer.sources
    assert target.tobytes() == source.values.tobytes()
    if sealed:
        assert transfer.event.waits == 1


def test_event_record_failure_drains_producer_stream(fake_cp, monkeypatch):
    def fail_record(event, stream):
        raise RuntimeError('record failed')

    monkeypatch.setattr(fake_cp.cuda.Event, 'record', fail_record)
    transfer = SnapshotTransfer(fake_cp)
    with pytest.raises(RuntimeError, match='record failed'):
        with transfer:
            transfer.download(fake_cp.ndarray(), fake_cp)
            transfer.seal()
    assert transfer.event is None
    assert transfer.stream.waits == 1 and not transfer.sources


def test_successful_construction_leaves_wait_to_consumer(fake_cp):
    with SnapshotTransfer(fake_cp) as transfer:
        transfer.download(fake_cp.ndarray(), fake_cp)
        transfer.seal()
    assert transfer.stream.waits == 0 and transfer.sources
    transfer.wait()
    assert transfer.event.waits == 1 and not transfer.sources


def test_snapshot_diagnostic_refusal_drains_already_queued_fields(fake_cp, monkeypatch):
    import sys
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof
    from hexcore import cuda_physics_prep_v841 as prep

    monkeypatch.setitem(sys.modules, 'cupy', fake_cp)
    monkeypatch.setattr(proof, '_phase_from_receipt', lambda backend: 'boundary')
    # execute_forecast rebinds proof.SNAPSHOT_LABELS to its own schedule (install_capture_labels)
    # and an earlier forecast test in the same process can leave it keyed without step 0,
    # which refused this capture before any field was queued (G0 batch-96: KeyError 0).
    # The capture under test is step 0; pin the label it is checked against.
    monkeypatch.setattr(proof, 'SNAPSHOT_LABELS', {0: 'F000'})
    prepared = SimpleNamespace(**{name: fake_cp.ndarray() for name in
        ('u_p', 'v_p', 'rho_dry', 'th_p', 'pres_p', 'psfc_p')})
    monkeypatch.setattr(prep, 'prepare_mpas_to_phys_cuda_v841', lambda *a, **kw: prepared)
    driver = SimpleNamespace(atmosphere=SimpleNamespace(
        state=SimpleNamespace(time_seconds=0, scalars=[fake_cp.ndarray()]),
        saved=SimpleNamespace(normal_velocity=fake_cp.ndarray(), vertical_velocity=fake_cp.ndarray())))

    def refuse():
        raise RuntimeError('diagnostic refused after state copies')

    with pytest.raises(RuntimeError, match='diagnostic refused after state copies'):
        proof.capture_snapshot(label='F000', step=0, driver=driver,
            backend=SimpleNamespace(diagnostic_snapshot=refuse), prep_geometry=None,
            kernel_cache=None, f000_surface_diagnostics={}, scalar_names=('qv',), deferred=True)
    stream = fake_cp.cuda.get_current_stream()
    assert stream.waits == 1 and not stream.pending


def test_parallel_array_records_keep_nan_signed_zero_and_empty_hashes():
    arrays = {'a': np.array([0x80000000,0x7fc00001,0x3f800000], dtype='u4').view('f4'),
              'b': np.empty((0, 3), dtype='f4')}
    digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
    rows = array_records(arrays, digest)
    assert rows['a']['sha256'] == digest(arrays['a'])
    assert rows['a']['nonzero'] == 2
    assert rows['b']['minimum'] is None

@pytest.mark.gpu
def test_deferred_download_owns_the_boundary():
    from concurrent.futures import ThreadPoolExecutor
    import cupy as cp
    host = {'x': np.arange(2**21, dtype='f4'), 'y': np.arange(2**20, dtype='i4')}
    with cp.cuda.Stream(non_blocking=True):
        device = {name: cp.asarray(value) for name, value in host.items()}
        frame = SnapshotTransfer(cp)
        outputs = {n: frame.download(v, cp) for n,v in device.items()}
        frame.seal()
        for value in device.values():
            value.fill(-4)  # queued after capture; must never alter captured bytes
        with ThreadPoolExecutor(max_workers=1) as consumer:
            consumer.submit(frame.wait).result()
    for n in host:
        assert outputs[n].tobytes() == host[n].tobytes()
    assert not frame.sources

    with cp.cuda.Stream(non_blocking=True):
        source = cp.asarray(host['x'])
        failed = SnapshotTransfer(cp)
        with pytest.raises(ValueError, match='failed before seal'):
            with failed:
                captured = failed.download(source, cp)
                raise ValueError('failed before seal')
        source.fill(-4)
        assert captured.tobytes() == host['x'].tobytes()
        assert not failed.sources


def test_samehost_actual_rank_capture_owns_host_diagnostics_and_matches_sync(fake_cp, monkeypatch):
    import sys
    from hexcore import cuda_physics_prep_v841 as prep
    from hexcore.drivers import samehost_forecast_v841 as adapter
    from hexcore.drivers import run_cuda_v841_full_physics_x4 as proof

    monkeypatch.setitem(sys.modules, 'cupy', fake_cp)
    monkeypatch.setattr(proof, '_phase_from_receipt', lambda backend: 'boundary')
    monkeypatch.setattr(proof, '_host_array', lambda value, cp: np.array(
        value.values if isinstance(value, cp.ndarray) else value, copy=True))
    classification = dict(sflx_land_columns=2, open_water_columns=0,
                          sea_ice_columns=0, glacier_columns=0)
    metadata = dict(surface_classification=classification, last_noahmp_census=None)
    diagnostic = np.array([0x80000000, 0x7fc01234, 0x3f800000], dtype='u4').view('f4')
    snapshot = dict(surface={'t2m': diagnostic}, metadata=metadata, receipt=metadata)
    calls = []
    def public():
        calls.append('public')
        return snapshot
    def resident():
        calls.append('resident')
        return snapshot
    backend = SimpleNamespace(diagnostic_snapshot=public, history_diagnostic_snapshot=resident,
                              step_receipt=lambda: metadata, take_history_refl10cm=fake_cp.ndarray)
    prepared = SimpleNamespace(**{name: fake_cp.ndarray() for name in
        ('u_p', 'v_p', 'rho_dry', 'th_p', 'pres_p', 'psfc_p')})
    monkeypatch.setattr(prep, 'prepare_mpas_to_phys_cuda_v841', lambda *a, **kw: prepared)
    driver = SimpleNamespace(atmosphere=SimpleNamespace(
        state=SimpleNamespace(time_seconds=0, scalars=[fake_cp.ndarray()]),
        saved=SimpleNamespace(normal_velocity=fake_cp.ndarray(), vertical_velocity=fake_cp.ndarray())),
        cache=None)
    host = dict(config=SimpleNamespace(config_dt=360), scalar_names=('qv',))
    stack = dict(driver=driver, backend=backend, prep_geometry=None,
                 constructor=SimpleNamespace(expected_surface_classification=lambda: classification))
    reference = adapter._capture_rank(proof, host, stack, 0, 'F000')
    pending = adapter._capture_rank(proof, host, stack, 0, 'F000', deferred=True)
    assert fake_cp.cuda.get_current_stream().waits == 0
    diagnostic[:] = 42  # live backend host storage can change after capture
    candidate = pending.result()
    assert calls == ['public', 'resident']
    assert reference.keys() == candidate.keys()
    assert all(reference[name].tobytes() == candidate[name].tobytes() for name in reference)
    assert pending.transfer.sources == []
