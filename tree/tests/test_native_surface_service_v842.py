"""Real f3 pool ABI/CUDA service replay; no forecast acceptance claim.

mpas_atmphys_update_surface.F:54-568; mpas_atmphys_update.F:72-254;
mpas_atmphys_driver_convection.F:1060-1100.
Run only inside the GPU and build queues with P25_SURFACE_GPU=1 and an authenticated
P25_NATIVE_PHYSICS_LIBRARY. All arrays and both arithmetic paths are real.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os

import numpy as np
import pytest

from hexcore.mpasphys.surface_update_v842 import NativeSurfaceService, CUDA_OPTIONS


def test_service_operations_and_calendar_snapshot():
    """Metadata only: MPAS callback ownership and saved DATA lifetime."""
    service = NativeSurfaceService(SimpleNamespace(native=None), None)
    assert set(service.callbacks()) == set(service.OPERATIONS)
    assert 'driver_microphysics' not in service.callbacks()
    assert service.receipt()['executed'] is False
    service.calendar.weights('2000-02-29_00:00:00')
    state = service.restart_state()
    restored = NativeSurfaceService(SimpleNamespace(native=None), None)
    restored.restore_restart_state(state)
    assert restored.calendar.february == 29
    assert restored.receipt()['executed'] is False
    assert NativeSurfaceService(SimpleNamespace(native=None), None).calendar.february == 28
    with pytest.raises(ValueError, match='February'):
        restored.restore_restart_state({'february': 30})


@pytest.mark.parametrize('operation', NativeSurfaceService.OPERATIONS)
def test_actual_native_pool_service_words(operation):
    """Actual native pools, pristine f3 entry, CUDA kernels, every output word."""
    library = os.environ.get('P25_NATIVE_PHYSICS_LIBRARY')
    if os.environ.get('P25_SURFACE_GPU') != '1' or not library:
        pytest.skip('real pool replay requires a slot and authenticated f3 library')
    import cupy as cp
    from hexcore.mpasphys.native_backend import NativePhysics
    from hexcore.mpasphys.driver_v842 import Backend, NativePoolArguments, SlotCall, StepFlags
    from test_surface_update_v842 import initial
    from surface_native_v842 import replay_fields, RAD

    fields = replay_fields(initial())
    surface_names = set(('landmask albedo12m greenfrac vegfra shdmin shdmax sfc_albbck '
        'ivgtyp isltyp sst tmn skintemp tslb sh2o smois snow snowc snowh seaice xice xland').split())
    pools = {pool: {key: np.array(value, order='F', copy=True)
                   for key, value in fields.items() if (key in surface_names) == (pool == 'sfc_input')}
             for pool in ('sfc_input', 'diag_physics')}
    pools['sfc_input'].update(isice=np.array(24, dtype='i4'), iswater=np.array(16, dtype='i4'))
    options = dict(config_sfc_albedo=True, config_frac_seaice=True,
                   config_bucket_radt=100., config_bucket_rainc=100.)
    mesh = dict(nCells=255, nCellsSolve=251, nSoilLevels=4, nLags=30)
    date = '2000-02-29_12:00:00'
    with NativePhysics(library, test_only=True) as native:
        for key, value in mesh.items(): native.dimension('mesh', key, value)
        for key, value in options.items(): native.config(key, value)
        native.put_fields(pools)
        for key, value in dict(dt_dyn=np.array(120., dtype='f4'),
                xice_threshold=np.array(.02, dtype='f4'),
                l_acradt=np.array(True), l_acrain=np.array(True)).items():
            native.put_variable(key, value)
        arguments = NativePoolArguments(native, pools={p:p for p in ('configs','mesh','sfc_input','diag_physics')},
                                        its=1, ite=251, synchronize=lambda entry, call: None)
        arguments.update_calendar(dict(current_date=date, leapyear=True, julian_in=365., julian=365.))
        def put(pool, field, value):
            native.put_field(pool, field, value, nlevels=native._fields[pool, field][2])
        session = SimpleNamespace(native=native, layout=SimpleNamespace(dimensions=mesh),
                                  arguments=arguments, put=put)
        streams_read = []
        def install(stamp, device):
            streams_read.append(stamp)
            for key in ('sst','xice'): device[key].set(pools['sfc_input'][key])
        service = NativeSurfaceService(session, cp, read_surface_record=install)
        call = SlotCall(operation, 'driver', Backend.HEX_CUDA, 1, 0., 120.,
                        StepFlags(rain_bucket=True, radiation_bucket=True), session, options)
        native.call(operation, **arguments(operation, call))
        expected = {p:{k:native.get_field(p,k) for k in f} for p,f in pools.items()}
        native.put_fields(pools)
        actual_receipt = service.invoke(call)
        differences, compared = {}, 0
        halo_differences = 0
        for pool, items in expected.items():
            for key, want in items.items():
                got = native.get_field(pool, key)
                differing = int(np.count_nonzero(got.view('u4') != want.view('u4')))
                compared += want.size
                if differing: differences[pool+'.'+key] = differing
                if want.ndim:
                    halo_differences += int(np.count_nonzero(got[...,251:].view('u4') != pools[pool][key][...,251:].view('u4')))
        destination = Path(os.environ['P25_RECEIPTS'])
        destination.mkdir(parents=True, exist_ok=True)
        record = dict(tier='actual native pool ABI/CUDA service; no whole-forecast claim',
                      process_count=1, operation=operation, words=compared,
                      differing_words=sum(differences.values()), failures=differences,
                      halo_differing_words=halo_differences, streams_read=streams_read,
                      cuda_compiler='direct NVRTC', cuda_options=list(CUDA_OPTIONS),
                      native_library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                      service=actual_receipt)
        (destination/(operation+'.json')).write_text(json.dumps(record,indent=2)+'\n')
        assert not differences, differences
        assert halo_differences == 0
        assert streams_read == ([date] if operation == 'physics_update_sst' else [])
        assert actual_receipt['executed'] is True
        assert actual_receipt['operations'][operation] == 1
        assert sum(actual_receipt['operations'].values()) == 1


def test_rust_surface_clock_selection_and_halos():
    """Actual native stream/Rust decoder; latest-before metadata and halo contract."""
    from datetime import timedelta
    from hexcore.mpasphys.surface_update_v842 import RustSurfaceRecordReader
    library, path = os.environ.get('P25_MONTHLY_READER_LIB'), os.environ.get('P25_REAL_STREAM_FILE')
    if not library or not path:
        pytest.skip('published surface file and built Rust bridge are required')
    class Buffer:
        def __init__(self): self.words = np.full(261, -123., dtype='f4')
        def get(self): return self.words.copy()
        def set(self, value): self.words[...] = value
    reader = RustSurfaceRecordReader(library, path, n_cells=256,
                                     first_cell=int(os.environ.get('P25_FIRST_CELL', '0')))
    pool = {key: Buffer() for key in ('sst', 'xice')}
    assert len(reader.times) == int(os.environ.get('P25_REAL_STREAM_RECORDS', '21'))
    for record, now in enumerate(reader.times):
        if record + 1 < len(reader.times):
            now = now + (reader.times[record+1]-now)/2
        reader(now.isoformat(), pool)
        assert reader.last_record == record
        for buffer in pool.values():
            assert np.all(buffer.words[256:] == np.float32(-123.))
    with pytest.raises(ValueError, match='at or before'):
        reader((reader.times[0]-timedelta(seconds=1)).isoformat(), pool)
    with pytest.raises(ValueError, match='cell extent'):
        reader(reader.times[0].isoformat(), {'sst':SimpleNamespace(get=lambda:np.empty(1,'f4'))})
