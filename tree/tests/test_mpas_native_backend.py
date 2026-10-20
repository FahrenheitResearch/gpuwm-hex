"""Exercise the built MPAS artifact, never substitute mocks for native calls.

MPAS source: src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:189-372.
Set MPAS_NATIVE_LIBRARY to the private strict build's shared-library path.
"""
import os
import hashlib
import json
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

from hexcore.mpasphys.native_backend import NativePhysics
_replay_spec = importlib.util.spec_from_file_location(
    'native_oracle_replay', Path(__file__).parents[1] / 'tools/native_physics_reference/backend/oracle_replay.py')
_replay_module = importlib.util.module_from_spec(_replay_spec)
_replay_spec.loader.exec_module(_replay_module)
Capture, replay, transport = _replay_module.Capture, _replay_module.replay, _replay_module.transport


@pytest.fixture
def library():
    path = os.environ.get('MPAS_NATIVE_LIBRARY')
    if not path:
        pytest.skip('requires the actual strict MPAS shared library')
    assert Path(path).is_file()
    return path


def test_native_context_matches_strict_pool_error_policy(library):
    # MPAS source: src/framework/mpas_pool_routines.F:108-155.
    # MPAS source: src/framework/mpas_framework.F:107-109.
    # Query actual native state before any session can override it: FATAL
    # incorrectly aborts valid land-use initialization on absent mavail.
    import ctypes as ct
    with NativePhysics(library, test_only=True) as native:
        get_level = native.lib.__mpas_pool_routines_MOD_mpas_pool_get_error_level
        get_level.argtypes = []
        get_level.restype = ct.c_int
        assert get_level() == 1001  # Native MPAS_POOL_SILENT.


@pytest.mark.parametrize('date', [
    '2000-01-01_00:00:00', '2000-02-29_12:00:00',
    '2001-03-15_00:00:00', '2001-04-30_23:59:59.9999',
    '2001-05-15_12:00:00', '2001-06-01_00:00:00',
    '2001-07-31_23:59:59', '2001-08-15_00:00:00',
    '2001-09-01_00:00:00', '2001-10-15_00:00:00',
    '2001-11-30_12:00:00', '2001-12-31_23:59:59',
])
@pytest.mark.parametrize('albedo', [False, True])
def test_native_surface_date_storage(library, date, albedo):
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_update_surface.F:54-111.
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_date_time.F:126-219.
    # The direct native caller has StrKIND storage. A trimmed ABI temporary
    # breaks monthly interpolation's fixed-length date read at columns21:24.
    probe = Path(os.environ.get('MPAS_NATIVE_SURFACE_PROBE',
                                Path(library).with_name('probe_surface')))
    if not probe.is_file():
        pytest.skip('requires independently compiled direct native surface probe')
    direct = subprocess.check_output([str(probe), date, str(int(albedo))], text=True)
    expected = np.asarray([int(word, 16) for word in direct.splitlines()], dtype=np.uint32)
    with NativePhysics(library, test_only=True) as native:
        native.dimension('mesh', 'nCellsSolve', 4)
        native.put_field('sfc_input', 'landmask', np.asarray([0, 1, 1, 0, 1], dtype=np.int32))
        # Exactly representable test inputs shared with probe_surface.f90.
        native.put_field('sfc_input', 'albedo12m', np.asarray(
            [[month + cell for cell in range(1, 6)] for month in range(1, 13)], dtype=np.float32))
        native.put_field('sfc_input', 'greenfrac', np.asarray(
            [[13 - month + cell for cell in range(1, 6)] for month in range(1, 13)], dtype=np.float32))
        fields = ('sfc_albbck', 'vegfra', 'shdmin', 'shdmax')
        for name in fields:
            native.put_field('sfc_input', name, np.full(5, -999., dtype=np.float32))
        native.call('physics_update_surface', current_date=date, config_sfc_albedo=albedo,
                    mesh='mesh', sfc_input='sfc_input')
        actual = np.concatenate([native.get_field('sfc_input', name).view(np.uint32) for name in fields])
        np.testing.assert_array_equal(actual, expected)


def test_captured_character_levels_in_prepared_replay(library, tmp_path):
    # MPAS source: src/framework/mpas_pool_routines.F:4056-4116.
    # Use actual f1 driver storage to detect losing level metadata in replay.
    prefix = os.environ.get('MPAS_NATIVE_CHARACTER_CAPTURE_PREFIX')
    if not prefix:
        pytest.skip('requires actual f1 scalar character time-level captures')
    mappings, expected = [], {}
    for level in (1, 2):
        path = Path(prefix + f'.t{level}.bin')
        with Capture(transport(library), path) as record:
            expected[level] = record.scalar()
        mappings.append(dict(role='text_field', pool='state', name='xtime',
                             capture=str(path), nlevels=2, level=level))
    manifest = tmp_path / 'character-levels.json'
    manifest.write_text(json.dumps(dict(schema='mpas-native-replay-v1',
        input_completeness='actual captured xtime only; no forecast claim',
        calls=[dict(entry='block', arguments={}, inputs=mappings, outputs=mappings)])))
    with NativePhysics(library, test_only=True) as native:
        receipt = replay(library, manifest, native=native)
        assert native._text_fields['state', 'xtime'] == 2
        assert {level: native.get_text_field('state', 'xtime', level=level)
                for level in (1, 2)} == expected
        assert len(receipt['input_captures']) == 2
        assert len(receipt['comparisons']) == 2
        assert receipt['different_words'] == 0
        assert sum(row['words'] for row in receipt['comparisons']) == 1024
        # The comparison must detect native character changes, including level2.
        native.put_text_field('state', 'xtime', 'changed timestamp', nlevels=2, level=2)
        changed = []
        for mapping in mappings:
            with Capture(transport(library), mapping['capture']) as record:
                changed.append(record.compare(native.get_text_field_storage(
                    'state', 'xtime', level=mapping['level'])))
        assert changed[0]['different_words'] == 0
        assert changed[1]['different_words'] > 0
        # The caller's initialized context remains live after replay.
        native.framework_object('block')


def test_native_real_config_reads_live_storage(library):
    # MPAS source: src/framework/mpas_pool_routines.F:4761-4791.
    # Mutate through the Fortran ABI directly so a Python cached config fails.
    with NativePhysics(library, test_only=True) as native:
        native.config('config_len_disp', 0., pool='configs')
        native.lib.native_config_real(native.pool('configs'), b'config_len_disp', 480000., 0)
        value = native.get_real_config('config_len_disp')
        assert value.dtype == np.dtype('float32')
        assert value.view(np.uint32) == np.float32(480000.).view(np.uint32)
        native.config('config_len_disp', -0., pool='other')
        assert native.get_real_config('config_len_disp', pool='other').view(np.uint32) == 0x80000000
        assert native.get_real_config('config_len_disp') == np.float32(480000.)
        native.config('integer_config', 1)
        with pytest.raises(TypeError, match='not declared real'):
            native.get_real_config('integer_config')


def test_native_monthly_surface_dates_preserve_fortran_padding(library):
    # update_surface.F passes dates to monthly_interp_to_date's fixed StrKIND
    # argument. A trimmed temporary can expose garbage as fractional seconds.
    # Exercise the actual native routine for both campaign dates, including
    # the standard 19-character form and its explicit fractional equivalent.
    with NativePhysics(library, test_only=True) as native:
        native.dimension('mesh', 'nCellsSolve', 2)
        native.put_field('sfc_input', 'landmask', np.array([1, 0], dtype='int32'))
        native.put_field('sfc_input', 'albedo12m', np.full((12, 2), 25., dtype='float32'))
        native.put_field('sfc_input', 'greenfrac', np.full((12, 2), .25, dtype='float32'))
        for field in ('sfc_albbck', 'vegfra', 'shdmin', 'shdmax'):
            native.put_field('sfc_input', field, np.zeros(2, dtype='float32'))
        for date in ('2022-12-22_12:00:00', '2023-03-31_12:00:00'):
            results = []
            for stamp in (date, date + '.0000'):
                native.call('physics_update_surface', current_date=stamp,
                            config_sfc_albedo=True, mesh='mesh', sfc_input='sfc_input')
                results.append({field: native.get_field('sfc_input', field).copy()
                                for field in ('sfc_albbck', 'vegfra', 'shdmin', 'shdmax')})
            for field in results[0]:
                assert np.array_equal(results[0][field].view('uint32'),
                                      results[1][field].view('uint32'))
                expected = np.array([.25, .08] if field == 'sfc_albbck'
                                    else [.25, .25], dtype='float32')
                assert np.array_equal(results[0][field].view('uint32'), expected.view('uint32'))


def test_boundary_manifest_requires_associated_capture(library, tmp_path, monkeypatch):
    # MPAS source: src/framework/mpas_pool_types.inc:7-11.
    # Remove a real tapped payload: capture loss must fail, whereas native
    # unassociated storage must remain explicitly inventoried.
    source = os.environ.get('MPAS_NATIVE_BOUNDARY_CAPTURE_ROOT')
    prefix = os.environ.get('MPAS_NATIVE_BOUNDARY_CAPTURE_PREFIX')
    if not source or not prefix:
        pytest.skip('requires actual f1 pool captures with unassociated storage')
    backend = Path(__file__).parents[1] / 'tools/native_physics_reference/backend'
    monkeypatch.syspath_prepend(str(backend))
    from boundary_manifest import build_manifest
    for phase in ('pre', 'post'):
        for label in ('state.xtime', 'diag_physics.cldfrac_bl'):
            for path in Path(source).glob(prefix + phase + '.' + label + '.*.bin'):
                (tmp_path / path.name).symlink_to(path.resolve())
    specification = dict(capture_root=str(tmp_path), boundary_prefix=prefix,
                         entry='block', arguments={}, process_count=1,
                         input_completeness='actual tapped storage validation only')
    manifest = build_manifest(library, specification)
    assert {(m['name'], m['level']) for m in manifest['calls'][0]['outputs']} == {
        ('xtime', 1), ('xtime', 2)}
    assert {m['name'] for m in manifest['unassociated_storage']} == {'cldfrac_bl'}
    sealed = {p.name: {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in tmp_path.iterdir()}
    assert build_manifest(library, specification, sealed_files=sealed) == manifest
    payload = prefix + 'post.state.xtime.t2.bin'
    sealed[payload] = {'sha256': '0' * 64}
    with pytest.raises(ValueError, match='pool capture differs from f1 seal'):
        build_manifest(library, specification, sealed_files=sealed)
    (tmp_path / payload).unlink()
    with pytest.raises(ValueError, match='missing associated post field capture'):
        build_manifest(library, specification)


@pytest.mark.parametrize('start,next_day,julday,leap', [
    ('2024-02-28_23:59:59', '2024-02-29', 60, True),
    ('1900-02-28_23:59:59', '1900-03-01', 60, False),
    ('2000-02-28_23:59:59', '2000-02-29', 60, True),
    ('2023-12-31_23:59:59', '2024-01-01', 1, True),
])
@pytest.mark.parametrize('dt', [1., 700., .125])
def test_production_calendar_dispatch(library, start, next_day, julday, leap, dt):
    # Exercise the actual merged library export through f4's argument provider.
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_manager.F:204-213,263-375.
    from hexcore.mpasphys import (Backend, NativeCalendarState, NativePoolArguments,
                                 SlotCall, StepFlags, resolve_namelist)
    config = resolve_namelist('&physics\nconfig_physics_suite="none"\n/')
    with NativePhysics(library, test_only=True) as native:
        native.clock(start=start, stop='2100-01-01_00:00:00', dt=dt)
        arguments = NativePoolArguments(native, pools={name: name for name in
            ['configs', 'mesh', 'state', 'diag', 'diag_physics', 'sfc_input',
             'tend', 'tend_physics']}, its=1, ite=5,
            synchronize=lambda entry, call: None)
        update = NativeCalendarState(native, native.lib, arguments)
        flags = StepFlags(radiation_lw=True, radiation_sw=False, convection=True,
                          cam_abs=False, cam_restart=True, rain_bucket=False,
                          radiation_bucket=True, diagnostics=False)
        call = SlotCall('driver', 'driver', Backend.NATIVE_CPU, 1, 0, dt, flags,
                        None, config.execution_options)
        assert update(call)['current_date'].startswith(start)
        steps = 8 if dt < 1 else 1
        for _ in range(steps):
            native.advance_clock()
        result = update(call)
        assert result['current_date'].startswith(next_day)
        assert result['leapyear'] == leap
        minutes, seconds = (11, 39) if dt == 700 else (0, 0)
        utc_h = np.float32(np.float32(minutes) / np.float32(60) +
                           np.float32(seconds) / np.float32(3600))
        expected = np.float32(np.float32(julday - 1) + utc_h / np.float32(24))
        current = native.get_table('mpas_atmphys_manager.curr_julday')[0].item()
        assert np.float32(current).view(np.uint32) == expected.view(np.uint32)
        assert result['julian'] == current
        for name, bit in zip(['l_radtlw', 'l_radtsw', 'l_conv', 'doabsems',
                              'l_camlw', 'l_acrain', 'l_acradt', 'l_diags'],
                             [True, False, True, False, True, False, True, False]):
            assert bool(native.get_variable(name)[0].item()) == bit
        mapped = arguments('physics_update_deepsoiltemp', call)
        assert mapped['leapyear'] == leap
        assert mapped['julian_in'] == current
        assert mapped['dt'] == dt
        assert arguments('driver_microphysics', call)['time_lev'] == 2
        assert arguments('mpas_to_physics', call)['time_lev'] == 1
        assert arguments('physics_update_surface', call)['current_date'] == result['current_date']


def test_test_only_selection():
    with pytest.raises(ValueError, match='test-only'):
        NativePhysics('unused')


def test_native_pool_ownership_and_levels(library):
    # Caller mutation/destruction must never invalidate native storage.
    for _ in range(2):
        with NativePhysics(library, test_only=True) as backend:
            source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            backend.put_field('state', 'scalars', source, nlevels=2)
            backend.put_field('state', 'scalars', source + np.float32(40), level=2, nlevels=2)
            source[:] = -100
            expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            np.testing.assert_array_equal(backend.get_field('state', 'scalars'), expected)
            np.testing.assert_array_equal(backend.get_field('state', 'scalars', level=2), expected + 40)
            for dtype in [np.int32, np.float32]:
                backend.put_field('diag', str(dtype), np.asarray(13, dtype=dtype))
                assert backend.get_field('diag', str(dtype)) == 13
            backend.dimension('mesh', 'nCells', 4)
            backend.dimension('mesh', 'nCellsSolve', 3)
            backend.dimension('mesh', 'nCellsSolve', 2)
            backend.dimension('mesh', 'nCellsHalo', [3, 4])
            backend.config('logical', True)
            backend.config('logical', False)
            backend.config('integer', 4)
            backend.config('real', 1.25)
            backend.config('string', '')
            backend.config('string', 'cld_fraction')
            backend.put_variable('qv_p', np.ones((4, 3, 1), dtype=np.float32), lower=(-1, 0, 1))
            value, lower = backend.get_variable('qv_p')
            assert lower == (-1, 0, 1)
            np.testing.assert_array_equal(value, 1)
            backend.clock(start='2000-01-01_00:00:00', stop='2000-01-01_01:00:00', dt=30)
            backend.streams()
            assert all(backend.framework_object(name) for name in ['dminfo', 'clock', 'stream_manager'])
        with pytest.raises(RuntimeError, match='closed'):
            backend.pool('mesh')


TEMPERATURE = [290., 260., 240., 280., 275., 255.]
VAPOR = [.015, .001, .00001, .000001, .010, .002]
CLOUD = [.00001, .000001, .0000005, 0., .00002, .00003]
ICE = [0., .00002, .0000002, 0., .00001, .00005]
SNOW = [0., 0., .000001, 0., 0., .00001]


def cloudiness(backend, scheme, cells, *, flags=(True, True), owned=None):
    nc, nz = len(cells), 3
    owned = nc if owned is None else owned
    for prefix, size in [('i', nc), ('j', 1), ('k', nz)]:
        for scope in ['d', 'm', 't']:
            backend.put_variable(prefix + scope + 's', np.asarray(1, dtype=np.int32))
            backend.put_variable(prefix + scope + 'e', np.asarray(size, dtype=np.int32))
    backend.put_variable('f_qc', np.asarray(flags[0]))
    backend.put_variable('f_qi', np.asarray(flags[1]))
    backend.config('config_radt_cld_scheme', scheme)
    backend.config('config_len_disp', 480000.)
    backend.put_field('mesh', 'areaCell', np.full(nc, 1.e10, dtype=np.float32))
    backend.put_field('mesh', 'meshDensity', np.ones(nc, dtype=np.float32))
    backend.put_field('sfc_input', 'xland', np.asarray([1 if i % 2 else 2 for i in cells], dtype=np.float32))
    backend.put_field('diag_physics', 'cldfrac', np.full((nz, nc), -999., dtype=np.float32))
    for name, values in [('t_p', TEMPERATURE), ('qv_p', VAPOR), ('qc_p', CLOUD), ('qi_p', ICE), ('qs_p', SNOW)]:
        data = np.empty((nc, nz, 1), dtype=np.float32, order='F')
        data[:, :, 0] = np.asarray(values, dtype=np.float32)[cells, None]
        backend.put_variable(name, data)
    for name, value in [('pres_p', 80000.), ('pres_hyd_p', 80000.), ('rho_p', 1.)]:
        backend.put_variable(name, np.full((nc, nz, 1), value, dtype=np.float32))
    backend.allocate_cloudiness()
    backend.driver_cloudiness(configs='configs', mesh='mesh', diag_physics='diag_physics',
                               sfc_input='sfc_input', its=1, ite=owned)
    result = backend.get_field('diag_physics', 'cldfrac')
    backend.deallocate_cloudiness()
    return result


@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
def test_column_partition_words(library, scheme):
    with NativePhysics(library, test_only=True) as backend:
        full = cloudiness(backend, scheme, list(range(6)), owned=5)
    assert np.all(full[:, 5] == np.float32(-999.))
    np.testing.assert_array_equal(full[:, 3], 0.)
    joined = np.empty_like(full[:, :5])
    for cells in [[4, 0], [2, 1, 3]]:
        with NativePhysics(library, test_only=True) as backend:
            joined[:, cells] = cloudiness(backend, scheme, cells)
    np.testing.assert_array_equal(full[:, :5].view(np.uint32), joined.view(np.uint32))


@pytest.mark.parametrize('flags', [(True, True), (True, False), (False, True), (False, False)])
def test_incidence_thresholds(library, flags):
    with NativePhysics(library, test_only=True) as backend:
        result = cloudiness(backend, 'cld_incidence', list(range(6)), flags=flags)
    expected = np.asarray([1, 1 if flags[1] else 0, 0, 0, 1, 1], dtype=np.float32)
    if not flags[0]:
        expected[:] = 0
    np.testing.assert_array_equal(result, np.broadcast_to(expected, result.shape))


@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
@pytest.mark.parametrize('flags', [(True, True), (True, False), (False, True), (False, False)])
def test_direct_native_driver_words(library, scheme, flags):
    probe = Path(library).with_name('probe_cloudiness')
    if not probe.is_file():
        pytest.skip('requires the separately compiled direct Fortran driver probe')
    direct = subprocess.check_output([str(probe), scheme, *[str(int(v)) for v in flags]], text=True)
    words = np.asarray([int(v, 16) for v in direct.splitlines()], dtype=np.uint32)
    with NativePhysics(library, test_only=True) as backend:
        result = cloudiness(backend, scheme, list(range(6)), flags=flags, owned=5)
    np.testing.assert_array_equal(result.ravel(order='F').view(np.uint32), words)


def test_native_physics_init_all_off(library, tmp_path, monkeypatch):
    # Exercise the full init entry and pristine land-use table reader. This deck
    # intentionally enables no parameterization and is not suite acceptance.
    tables = os.environ.get('MPAS_NATIVE_TABLES')
    if not tables:
        pytest.skip('requires the pinned native LANDUSE.TBL directory')
    (tmp_path / 'LANDUSE.TBL').symlink_to(Path(tables) / 'LANDUSE.TBL')
    monkeypatch.chdir(tmp_path)
    with NativePhysics(library, test_only=True) as b:
        for key in ['do_restart', 'o3climatology', 'oml1d', 'ngw_scheme', 'frac_seaice', 'sfc_albedo',
                    'sstdiurn_update', 'deepsoiltemp_update']:
            b.config('config_' + key, False)
        for key in ['convection', 'lsm', 'microp', 'pbl', 'sfclayer', 'radt_lw', 'radt_sw', 'gwdo']:
            b.config('config_' + key + '_scheme', 'off')
        for key in ['conv_interval', 'pbl_interval', 'radtlw_interval', 'radtsw_interval', 'bucket_update', 'camrad_abs_update']:
            b.config('config_' + key, 'none')
        b.config('config_greeness_update', '01_00:00:00')
        b.config('config_sst_update', False)
        b.config('config_microp_re', False)
        for key, value in [('oml_hml0', 50.), ('dt', 30.), ('microphysics_top', 0.)]:
            b.config('config_' + key, value)
        nc, nz = 6, 3
        for name, value in [('nCells', nc), ('nCellsSolve', 5), ('nLags', 2), ('nVertLevels', nz)]:
            b.dimension('mesh', name, value)
        for name, value in [('cam_dim1', 3), ('nMonths', 12), ('nAerLevels', 3), ('nOznLevels', 3), ('nSoilLevels', 4)]:
            b.dimension('mesh', name, value)
        b.dimension('state', 'num_aerosols', 0)
        for name in ['qc', 'qr', 'qi', 'qs', 'qg', 'nc', 'ni', 'nifa', 'nwfa']:
            b.dimension('state', 'index_' + name, -1)
        for name in ['diag', 'tend', 'atm_input', 'ngw_input', 'diag_physics_noahmp', 'output_noahmp']:
            b.pool(name)
        for name in ['latCell', 'lonCell']:
            b.put_field('mesh', name, np.zeros(nc, dtype=np.float32))
        for name in ['east', 'north']:
            b.put_field('mesh', name, np.zeros((3, nc), dtype=np.float32))
        for name in ['rdzw', 'dzu']:
            b.put_field('mesh', name, np.ones(nz, dtype=np.float32))
        b.put_field('tend_physics', 'mp_top_level', np.asarray(1, dtype=np.int32))
        accumulated = ['acswdnb', 'acswdnbc', 'acswdnt', 'acswdntc', 'acswupb', 'acswupbc', 'acswupt', 'acswuptc',
                       'aclwdnb', 'aclwdnbc', 'aclwdnt', 'aclwdntc', 'aclwupb', 'aclwupbc', 'aclwupt', 'aclwuptc']
        for name in accumulated:
            b.put_field('diag_physics', name, np.full(nc, -99., dtype=np.float32))
            b.put_field('diag_physics', 'i_' + name, np.full(nc, -99, dtype=np.int32))
        for name in ['nsteps_accum', 'ndays_accum', 'tday_accum', 'tyear_accum', 'tyear_mean', 'sstsk', 'xicem',
                     't_oml', 't_oml_initial', 't_oml_200m_initial', 'h_oml', 'h_oml_initial', 'hu_oml', 'hv_oml',
                     'sfc_emibck', 'mavail', 'sfc_albedo', 'sfc_emiss', 'thc', 'ust', 'z0', 'znt']:
            b.put_field('diag_physics', name, np.full(nc, -99., dtype=np.float32))
        b.put_field('diag_physics', 'tlag', np.full((2, nc), -99., dtype=np.float32))
        for name, value in [('sst', 289.), ('tmn', 285.), ('xice', 0.), ('xland', 1.),
                            ('snoalb', .7), ('snowc', 0.), ('sfc_albbck', .2)]:
            b.put_field('sfc_input', name, np.full(nc, value, dtype=np.float32))
        b.put_field('sfc_input', 'landmask', np.ones(nc, dtype=np.int32))
        b.put_field('sfc_input', 'ivgtyp', np.full(nc, 7, dtype=np.int32))
        b.put_field('sfc_input', 'isice', np.asarray(24, dtype=np.int32))
        b.put_field('sfc_input', 'iswater', np.asarray(16, dtype=np.int32))
        b.put_field('sfc_input', 'dzs', np.zeros((4, nc), dtype=np.float32))
        b.put_field('sfc_input', 'albedo12m', np.full((12, nc), 20., dtype=np.float32))
        b.put_field('sfc_input', 'greenfrac', np.full((12, nc), .5, dtype=np.float32))
        for name in ['vegfra', 'shdmin', 'shdmax']:
            b.put_field('sfc_input', name, np.full(nc, -99., dtype=np.float32))
        b.put_text_field('sfc_input', 'mminlu', 'USGS')
        assert b.get_text_field('sfc_input', 'mminlu') == 'USGS'
        b.clock(start='2000-01-01_00:00:00', stop='2000-01-01_01:00:00', dt=30)
        b.streams()
        b.stream('output', filename='output.nc', direction='output', interval='00:10:00')
        b.stream('diagnostics', filename='diagnostics.nc', direction='output')
        b.block()
        b.physics_tables_init(dminfo=b.framework_object('dminfo'), configs='configs')
        b.physics_registry_init(mesh='mesh', configs='configs', sfc_input='sfc_input')
        b.physics_run_init(configs='configs', mesh='mesh', state='state',
                           clock=b.framework_object('clock'), stream_manager=b.framework_object('stream_manager'))
        assert b.get_table('mpas_atmphys_manager.year')[0] == 2000
        assert b.get_variable('dt_pbl')[0] == np.float32(30.)
        b.allocate_forall_physics(configs='configs')
        arguments = {name: name for name in b.abi['entries']['physics_init']['arguments']}
        arguments.update(dminfo=b.framework_object('dminfo'), clock=b.framework_object('clock'),
                         stream_manager=b.framework_object('stream_manager'), time_lev=1)
        b.physics_init(**arguments)
        for name in accumulated:
            expected = np.asarray([0., 0., 0., 0., 0., -99.], dtype=np.float32)
            np.testing.assert_array_equal(b.get_field('diag_physics', name), expected)
        np.testing.assert_array_equal(b.get_field('diag_physics', 'sstsk')[:5], 289.)
        np.testing.assert_array_equal(b.get_field('diag_physics', 'tyear_mean')[:5], 285.)
        np.testing.assert_array_equal(b.get_field('diag_physics', 'tlag')[:, :5], 285.)
        assert np.all(np.isfinite(b.get_field('diag_physics', 'sfc_albedo')))
        assert np.all(b.get_field('diag_physics', 'sfc_emiss') > 0)
        assert b.get_variable('f_qc')[0] == 0
        b.atmphys_finalize(configs='configs')
        b.deallocate_forall_physics(configs='configs')
        b.advance_clock()
        b.physics_timetracker(domain=b.framework_object('domain'), dt=30.,
                             clock=b.framework_object('clock'), itimestep=1, xtime_s=30.)
        assert b.get_table('mpas_atmphys_manager.curr_julday')[0] > 0.
        np.testing.assert_array_equal(b.get_field('sfc_input', 'vegfra')[:5], .5)
        assert b.get_field('sfc_input', 'vegfra')[-1] == -99.


def test_rrtmg_native_init_tables(library, tmp_path, monkeypatch):
    tables = os.environ.get('MPAS_NATIVE_TABLES')
    probe = Path(library).with_name('probe_tables')
    if not tables or not probe.is_file():
        pytest.skip('requires native RRTMG data and the direct initialization probe')
    for name in ['RRTMG_LW_DATA', 'RRTMG_SW_DATA']:
        (tmp_path / name).symlink_to(Path(tables) / name)
    monkeypatch.chdir(tmp_path)
    output = subprocess.check_output([str(probe)], text=True)
    text = output.split('TABLE_WORDS_BEGIN\n', 1)[1].split('TABLE_WORDS_END', 1)[0]
    words = np.asarray([int(v, 16) for v in text.splitlines()], dtype=np.uint32)
    names = ['rrlw_ref.pref', 'rrsw_ref.pref', 'rrlw_tbl.tau_tbl', 'rrsw_tbl.tau_tbl', 'rrlw_kg01.kao', 'rrsw_cld.extliq1']
    with NativePhysics(library, test_only=True) as b:
        b.rrtmg_initlw_formpas(dminfo=b.framework_object('dminfo'))
        b.rrtmg_initsw_formpas(dminfo=b.framework_object('dminfo'))
        actual = [b.get_table(name)[0] for name in names]
        assert all(v.dtype == np.dtype('float32') for v in actual)
        assert b.get_table('rrlw_tbl.tau_tbl')[1] == (0,)
        assert b.get_table('rrsw_cld.extliq1')[1] == (1, 16)
        packed = np.concatenate([v.ravel(order='F').view(np.uint32) for v in actual])
    np.testing.assert_array_equal(packed, words)


def test_rrtmg_published_tables(library, tmp_path, monkeypatch):
    # Published pre-radiation state is independent of this library's initialization.
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_rrtmg_lwinit.F:40-105;
    # mpas_atmphys_rrtmg_swinit.F:34-62.
    root = os.environ.get('MPAS_NATIVE_ORACLE')
    tables = os.environ.get('MPAS_NATIVE_TABLES')
    if not root or not tables:
        pytest.skip('requires an explicit published radiation capture and native tables')
    root = Path(root)
    published = json.loads((root / 'manifest.json').read_text())
    with NativePhysics(library, test_only=True) as native:
        available = set(native.abi['tables'])
    outputs = []
    for name, metadata in sorted(published['files'].items()):
        if not name.startswith('rrtmg_lwrad__') or '.pre.state.' not in name:
            continue
        table = name.split('.pre.state.', 1)[1].removesuffix('.bin')
        if table not in available or not table.startswith(('rrlw_', 'rrsw_')):
            continue
        capture = root / name
        assert hashlib.sha256(capture.read_bytes()).hexdigest() == metadata['sha256']
        outputs.append({'role': 'table', 'name': table, 'capture': str(capture.resolve())})
    assert outputs, 'publication contains no exposed native RRTMG tables'
    for name in ['RRTMG_LW_DATA', 'RRTMG_SW_DATA']:
        (tmp_path / name).symlink_to(Path(tables) / name)
    monkeypatch.chdir(tmp_path)
    manifest = {'schema': 'mpas-native-replay-v1',
                'input_completeness': 'Published pre-radiation public primitive tables; '
                                     'not complete level-c suite initialization',
                'calls': [
                    {'entry': 'rrtmg_initlw_formpas',
                     'arguments': {'dminfo': {'framework': 'dminfo'}}},
                    {'entry': 'rrtmg_initsw_formpas',
                     'arguments': {'dminfo': {'framework': 'dminfo'}}, 'outputs': outputs}]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    receipt = replay(library, manifest_path)
    assert receipt['different_words'] == 0
    receipt['publication_manifest_sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    receipt['publication_sha256'] = hashlib.sha256((root / 'PUBLICATION.json').read_bytes()).hexdigest()
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def prepare_stream_block(b, ids, owned, *, clock_options=None):
    # All three index fields are required by pristine postread_reindex, even
    # when the stream only contains a cell field.
    b.block()
    b.dimension('mesh', 'nCells', len(ids))
    b.dimension('mesh', 'nCellsSolve', owned)
    b.put_field('mesh', 'indexToCellID', np.asarray(ids, dtype=np.int32))
    b.register_field('mesh', 'indexToCellID', dimensions=('nCells',))
    for dimension, value in [('maxEdges', 1), ('maxEdges2', 1), ('vertexDegree', 1)]:
        b.dimension('mesh', dimension, value)
    for name, size, dimension in [('nEdgesOnCell', len(ids), 'nCells'), ('nEdgesOnEdge', 1, 'nEdges')]:
        b.put_field('mesh', name, np.zeros(size, dtype=np.int32))
        b.register_field('mesh', name, dimensions=(dimension,))
    for dimension, name in [('nEdges', 'indexToEdgeID'), ('nVertices', 'indexToVertexID')]:
        b.dimension('mesh', dimension, 1)
        b.dimension('mesh', dimension + 'Solve', 1)
        b.put_field('mesh', name, np.asarray([1], dtype=np.int32))
        b.register_field('mesh', name, dimensions=(dimension,))
    b.clock(**(clock_options or dict(start='2000-01-01_00:00:00', stop='2000-01-01_01:00:00', dt=30.)))
    b.streams()


def test_native_compiled_constants(library):
    """Compare constant-expression words with independent compiled Fortran uses.

    MPAS source: src/core_atmosphere/physics/mpas_atmphys_constants.F:8-103;
    src/framework/mpas_constants.F:43-56.
    """
    probe = Path(library).with_name('probe_constants')
    words = dict(line.split() for line in subprocess.check_output([str(probe)], text=True).splitlines())
    with NativePhysics(library, test_only=True) as b:
        for name, word in words.items():
            if name == 'rounded_time':
                continue
            value = b.get_constant(name)
            assert value.dtype == np.dtype('float32')
            assert int.from_bytes(value.tobytes(), 'little') == int(word, 16)
        with pytest.raises(KeyError):
            b.get_constant('missing_constant')


@pytest.mark.parametrize('clock_advanced', [False, True])
def test_native_post_dynamics_time(library, tmp_path, clock_advanced):
    """Native calendar and RKIND time addition feed actual CF stream metadata.

    MPAS source: src/core_atmosphere/mpas_atm_core.F:264-277;
    src/core_atmosphere/dynamics/mpas_atm_time_integration.F:1835-1852.
    """
    probe = Path(library).with_name('probe_constants')
    words = dict(line.split() for line in subprocess.check_output([str(probe)], text=True).splitlines())
    with NativePhysics(library, test_only=True) as b:
        prepare_stream_block(b, [1], 1, clock_options=dict(
            start='2024-02-28_23:59:59', stop='2024-02-29_00:00:01', dt=1.))
        b.put_text_field('state', 'xtime', 'uninitialized', nlevels=2)
        b.put_text_field('state', 'initial_time', '2024-02-28_23:59:59', nlevels=2)
        b.put_text_field('state', 'initial_time', 'uninitialized', nlevels=2, level=2)
        b.put_field('state', 'Time', np.asarray(16777216., dtype=np.float32), nlevels=2)
        b.put_field('state', 'Time', np.asarray(-99., dtype=np.float32), nlevels=2, level=2)
        b.register_field('state', 'Time', has_time=True)
        b.register_field('state', 'xtime', has_time=True)
        b.register_field('state', 'initial_time', has_time=False)
        b.initialize_state_time()
        assert b.get_text_field('state', 'initial_time', level=2) == '2024-02-28_23:59:59'
        assert b.get_text_field('state', 'xtime', level=2) == '2024-02-28_23:59:59'
        if clock_advanced:
            b.advance_clock()
        b.update_state_time(1., clock_advanced=clock_advanced)
        assert b.get_text_field('state', 'xtime') == '2024-02-28_23:59:59'
        assert b.get_text_field('state', 'xtime', level=2) == '2024-02-29_00:00:00'
        assert int.from_bytes(b.get_field('state', 'Time', level=2).tobytes(), 'little') == int(words['rounded_time'], 16)
        assert b.get_field('state', 'Time', level=1) == np.float32(16777216.)
        filename = tmp_path / 'time.nc'
        b.stream('time', filename=str(filename), direction='output')
        for name in ('Time', 'xtime', 'initial_time'):
            b.stream_field('time', name)
        b.transfer_stream('time', read=False, level=2)
    header = subprocess.check_output(['ncdump', '-h', str(filename)], text=True)
    assert 'seconds since 2024-02-28 23:59:59' in header


def test_native_stream_filename_rotation(library, tmp_path):
    """Native filename intervals preserve one file per stream output time.

    MPAS source: src/framework/mpas_stream_manager.F:1833-1905,3422-3579.
    """
    with NativePhysics(library, test_only=True) as b:
        prepare_stream_block(b, [1], 1, clock_options=dict(
            start='2000-01-01_00:00:00', stop='2000-01-01_02:00:00', dt=3600.))
        b.put_field('state', 'sample', np.asarray([3.], dtype=np.float32))
        b.register_field('state', 'sample', dimensions=('nCells',), has_time=True)
        b.stream('history', filename=str(tmp_path / 'history.$Y-$M-$D_$h.$m.$s.nc'),
                 direction='output', output_interval='01:00:00', filename_interval='01:00:00')
        b.stream_field('history', 'sample')
        for value in range(3):
            b.put_field('state', 'sample', np.asarray([value + 3.], dtype=np.float32))
            b.transfer_stream('history', read=False)
            if value < 2:
                b.advance_clock()
    files = sorted(tmp_path.glob('history.*.nc'))
    assert [p.name for p in files] == [f'history.2000-01-01_0{i}.00.00.nc' for i in range(3)]
    for value, path in enumerate(files):
        output = subprocess.check_output(['ncdump', '-v', 'sample', str(path)], text=True)
        assert 'Time = UNLIMITED ; // (1 currently)' in output
        assert f'sample =\n  {value + 3} ;' in output


def test_native_stream_roundtrip(library, tmp_path):
    filename = str(tmp_path / 'native.nc')
    value = np.asarray([[1., -0., .5, -10.], [2., 3., 4., 5.]], dtype=np.float32)
    with NativePhysics(library, test_only=True) as b:
        prepare_stream_block(b, [1, 2, 3, 4], 4)
        b.put_field('state', 'theta', np.full((2, 4), -99., dtype=np.float32), nlevels=2)
        b.put_field('state', 'theta', value, nlevels=2, level=2)
        b.register_field('state', 'theta', dimensions=('nVertLevels', 'nCells'))
        b.stream('write', filename=filename, direction='output')
        b.stream_field('write', 'theta')
        b.transfer_stream('write', read=False, level=2)
    # The read uses actual native decomposition, not a Python file decoder.
    with NativePhysics(library, test_only=True) as b:
        prepare_stream_block(b, [1, 2, 3, 4], 4)
        b.decomposition('nCells', global_ids=np.asarray([1, 2, 3, 4], dtype=np.int32), owned=4)
        b.put_field('state', 'theta', np.full((2, 4), -99., dtype=np.float32), nlevels=2)
        b.put_field('state', 'theta', np.full((2, 4), -99., dtype=np.float32), nlevels=2, level=2)
        b.register_field('state', 'theta', dimensions=('nVertLevels', 'nCells'), decomposed=True)
        b.stream('read', filename=filename, direction='input')
        b.stream_field('read', 'theta')
        b.transfer_stream('read', read=True, level=2)
        np.testing.assert_array_equal(b.get_field('state', 'theta', level=2).view(np.uint32), value.view(np.uint32))
        np.testing.assert_array_equal(b.get_field('state', 'theta', level=1), -99.)


def test_native_registry_alias_constituent_roundtrip(library, tmp_path):
    """Actual native I/O must preserve aliases, constituent axes and time levels.

    MPAS source: src/framework/mpas_io_streams.F:1027-1092;
    src/framework/mpas_stream_manager.F:834-930,1158-1331.
    """
    filename = str(tmp_path / 'registry.nc')
    state = np.asarray([1., -0., 3., 4.], dtype=np.float32)
    tendency = np.asarray([5., 6., -7., 8.], dtype=np.float32)
    scalars = np.asarray([[.1, .2, .3, .4], [.5, .6, .7, .8]], dtype=np.float32)
    for reading in (False, True):
        with NativePhysics(library, test_only=True) as b:
            prepare_stream_block(b, [1, 2, 3, 4], 4)
            b.decomposition('nCells', global_ids=np.asarray([1, 2, 3, 4], dtype=np.int32), owned=4)
            b.dimension('mesh', 'num_scalars', 2)
            for pool, registry, value in [('state', 'u', state), ('tend', 'u_tend', tendency)]:
                b.put_field(pool, 'u', np.full_like(value, -99.) if reading else value)
                b.register_field(pool, 'u', registry_name=registry,
                                 dimensions=('nCells',), decomposed=True, has_time=True)
            for level in (1, 2):
                b.put_field('state', 'scalars', np.full_like(scalars, -99.) if reading or level == 1 else scalars,
                            nlevels=2, level=level)
            b.register_field('state', 'scalars', dimensions=('num_scalars', 'nCells'),
                             components=('qv', 'qc'), has_time=True, decomposed=True)
            b.stream('restart', filename=filename, direction='input;output',
                     input_interval='00:00:02', output_interval='00:00:03')
            for field in ('u', 'u_tend', 'scalars'):
                b.stream_field('restart', field)
            # Scalar fields have one time level; write all fields at level one,
            # then the constituent field separately at its second level.
            b.transfer_stream('restart', read=reading, level=1)
            b.stream('constituents', filename=str(tmp_path / 'scalars.nc'),
                     direction='input' if reading else 'output')
            b.stream_field('constituents', 'scalars')
            b.transfer_stream('constituents', read=reading, level=2)
            if reading:
                for pool, value in [('state', state), ('tend', tendency)]:
                    np.testing.assert_array_equal(b.get_field(pool, 'u').view(np.uint32), value.view(np.uint32))
                np.testing.assert_array_equal(b.get_field('state', 'scalars', level=2).view(np.uint32), scalars.view(np.uint32))
                np.testing.assert_array_equal(b.get_field('state', 'scalars', level=1), -99.)


def test_native_independent_stream_alarms(library, tmp_path):
    """MPAS source: src/framework/mpas_timekeeping.F:490-561,814-842."""
    with NativePhysics(library, test_only=True) as b:
        prepare_stream_block(b, [1, 2, 3, 4], 4)
        b.stream('surface', filename=str(tmp_path / 'surface.nc'), direction='input;output',
                 input_interval='00:01:00', output_interval='00:01:30')
        b.stream('initial', filename=str(tmp_path / 'initial.nc'), direction='input',
                 input_interval='initial_only')
        b.stream('final', filename=str(tmp_path / 'final.nc'), direction='output',
                 output_interval='final_only')
        for seconds in range(0, 3601, 30):
            assert b.alarm_ringing('surface:in') == (seconds % 60 == 0)
            assert b.alarm_ringing('surface:out') == (seconds % 90 == 0)
            assert b.alarm_ringing('initial:in') == (seconds == 0)
            assert b.alarm_ringing('final:out') == (seconds == 3600)
            for alarm in ('surface:in', 'surface:out', 'initial:in', 'final:out'):
                if b.alarm_ringing(alarm):
                    b.reset_alarm(alarm)
            b.advance_clock()


def test_native_two_rank_halo_words(library, tmp_path):
    receipt = json.loads(Path(library).with_name('build-receipt.json').read_text())
    mpiexec = Path(os.environ.get('MPAS_NATIVE_MPIEXEC', str(Path(receipt['mpif90']).with_name('mpiexec'))))
    assert mpiexec.is_file(), 'use the same MPI runtime as the compiled native library'
    worker = Path(__file__).parents[1] / 'tools/native_physics_reference/backend/probe_mpi.py'
    result = subprocess.run([str(mpiexec), '-n', '2', os.sys.executable,
                             str(worker), library, str(tmp_path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        check = json.loads((tmp_path / f'mpi-rank-{rank}.json').read_text())
        assert check['checked_words'] == 152
        assert check['different_words'] == 0


@pytest.mark.parametrize('lw,sw,expected', [(False, False, 0.), (True, False, .25),
                                          (False, True, .5), (True, True, .75)])
@pytest.mark.parametrize('wind', [False, True])
def test_native_rk_tendency_array_abi(library, lw, sw, expected, wind):
    # Exact powers-of-two fixture checks native output buffers and pool time levels.
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_todynamics.F:98-294,471-498.
    with NativePhysics(library, test_only=True) as b:
        b.block()
        b.mpas_atm_set_dims(nvertlevels_val=2, maxedges_val=1, maxedges2_val=1, num_scalars_val=9)
        for name, value in [('nCells', 2), ('nCellsSolve', 2), ('nEdges', 1), ('nEdgesSolve', 1)]:
            b.dimension('mesh', name, value)
        for index, name in enumerate(['qv', 'qc', 'qr', 'qi', 'qs', 'nc', 'ni', 'nifa', 'nwfa'], 1):
            b.dimension('state', 'index_' + name, index)
        for option in ['convection', 'microp', 'pbl', 'radt_lw', 'radt_sw']:
            b.config('config_' + option + '_scheme',
                     'rrtmg' if (option == 'radt_lw' and lw) or (option == 'radt_sw' and sw) else 'off')
        if wind:
            b.config('config_pbl_scheme', 'bl_ysu')
            b.config('config_convection_scheme', 'cu_ntiedtke')
        cell = np.full((2, 3), 2., dtype=np.float32)
        edge = np.ones((2, 2), dtype=np.float32)
        b.put_field('state', 'theta_m', cell, nlevels=2)
        b.put_field('state', 'rho_zz', cell, nlevels=2)
        b.put_field('state', 'rho_zz', cell, nlevels=2, level=2)
        b.put_field('state', 'scalars', np.zeros((9, 2, 3), dtype=np.float32), nlevels=2)
        b.put_field('diag', 'rho_edge', edge)
        b.put_field('diag', 'tend_u_phys', edge)
        for name in ['rublten', 'rvblten', 'rthblten', 'rqvblten', 'rqcblten', 'rqiblten', 'rqsblten',
                     'rncblten', 'rniblten', 'rnifablten', 'rnwfablten', 'rucuten', 'rvcuten', 'rthcuten',
                     'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']:
            b.put_field('tend_physics', name, np.zeros_like(cell))
        for name in ['rublten_Edge', 'rucuten_Edge']:
            b.put_field('tend_physics', name, np.zeros_like(edge))
        if wind:
            b.decomposition('nCells', global_ids=np.asarray([1, 2], dtype=np.int32), owned=2)
            b.put_field('mesh', 'cellsOnEdge', np.asarray([[1, 1], [2, 2]], dtype=np.int32))
            b.put_field('mesh', 'east', np.asarray([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]], dtype=np.float32))
            b.put_field('mesh', 'north', np.asarray([[0., 0., 0.], [1., 1., 1.], [0., 0., 0.]], dtype=np.float32))
            b.put_field('mesh', 'edgeNormalVectors', np.asarray([[1., 1.], [0., 0.], [0., 0.]], dtype=np.float32))
            for name, value in [('rublten', 2.), ('rvblten', 0.), ('rucuten', 4.), ('rvcuten', 0.)]:
                b.put_field('tend_physics', name, np.full_like(cell, value))
                b.register_field('tend_physics', name, dimensions=('nVertLevels', 'nCells'), decomposed=True)
            b.exchange_group('physics:blten', ['rublten', 'rvblten'])
            b.exchange_group('physics:cuten', ['rucuten', 'rvcuten'])
        b.put_field('tend_physics', 'rthratenlw', np.full_like(cell, .125))
        b.put_field('tend_physics', 'rthratensw', np.full_like(cell, .25))
        b.put_field('tend', 'scalars_tend', np.full((9, 2, 3), -99., dtype=np.float32))
        ru, rtheta, rho = np.full_like(edge, -99.), np.full_like(cell, -99.), np.full_like(cell, -99.)
        b.pre_physics_get_tend(configs='configs', state='state', diag='diag', tend='tend')
        arguments = dict(block=b.framework_object('block'), mesh='mesh', state='state',
                         diag='diag', tend='tend', tend_physics='tend_physics', configs='configs',
                         rk_step=1, dynamics_substep=1, tend_ru_physics=ru,
                         tend_rtheta_physics=rtheta, tend_rho_physics=rho)
        ru.flags.writeable = False
        with pytest.raises(ValueError, match='writable output storage'):
            b.physics_get_tend(**arguments)
        np.testing.assert_array_equal(rtheta, -99.)
        ru.flags.writeable = True
        b.physics_get_tend(**arguments)
        b.post_physics_get_tend(configs='configs', state='state', diag='diag', tend='tend')
        np.testing.assert_array_equal(rtheta[:, :2], np.float32(expected))
        np.testing.assert_array_equal(rtheta[:, 2], np.float32(0.))
        np.testing.assert_array_equal(ru[:, 0], 6. if wind else 0.)
        np.testing.assert_array_equal(ru[:, 1], 0.)
        np.testing.assert_array_equal(rho, 0.)
        np.testing.assert_array_equal(b.get_field('tend', 'scalars_tend'), 0.)
        if wind:
            # RK stages/substeps must reuse the first-stage edge projection.
            # Changing cell tendencies distinguishes reuse from a fresh projection.
            b.put_field('tend_physics', 'rublten', np.full_like(cell, 20.))
            b.put_field('tend_physics', 'rucuten', np.full_like(cell, 40.))
            for rk_step, substep in [(2, 1), (3, 1), (1, 2)]:
                arguments.update(rk_step=rk_step, dynamics_substep=substep)
                b.physics_get_tend(**arguments)
                np.testing.assert_array_equal(ru[:, 0], 6.)
            arguments.update(rk_step=1, dynamics_substep=1)
            b.physics_get_tend(**arguments)
            np.testing.assert_array_equal(ru[:, 0], 60.)


def _cloudiness_manifest(tmp_path, owned):
    inputs = []
    pools = {'areaCell': 'mesh', 'meshDensity': 'mesh', 'xland': 'sfc_input', 'cldfrac': 'diag_physics'}
    variables = ['ids', 'ide', 'ims', 'ime', 'jds', 'jde', 'jms', 'jme', 'jts', 'jte',
                 'kds', 'kde', 'kms', 'kme', 'kts', 'kte', 'f_qc', 'f_qi',
                 't_p', 'qv_p', 'qc_p', 'qi_p', 'qs_p', 'pres_p', 'pres_hyd_p', 'rho_p']
    for name in ['config_radt_cld_scheme', 'config_len_disp', *pools, *variables]:
        mapping = {'capture': f'capture.pre.{name}.bin', 'name': name}
        if name in pools:
            mapping.update(role='field', pool=pools[name])
        elif name.startswith('config_'):
            mapping.update(role='config', pool='configs')
        else:
            mapping.update(role='variable')
        inputs.append(mapping)
    manifest = {'schema': 'mpas-native-replay-v1', 'input_completeness': 'synthetic cloudiness fixture only',
                'calls': [{'entry': 'allocate_cloudiness', 'arguments': {}, 'inputs': inputs},
                          {'entry': 'driver_cloudiness',
                           'arguments': {'configs': 'configs', 'mesh': 'mesh', 'diag_physics': 'diag_physics',
                                         'sfc_input': 'sfc_input', 'its': 1, 'ite': owned},
                           'outputs': [{'role': 'field', 'pool': 'diag_physics', 'name': 'cldfrac',
                                        'capture': 'capture.post.cldfrac.bin'}]}]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
@pytest.mark.parametrize('flags', [(True, True), (True, False), (False, True), (False, False)])
def test_f1_format_driver_replay(library, tmp_path, scheme, flags):
    probe = Path(library).with_name('probe_cloudiness_capture')
    if not probe.is_file():
        pytest.skip('build with --tap-transport pointing to f1 native capture writer')
    subprocess.run([str(probe), scheme, *[str(int(v)) for v in flags], str(tmp_path / 'capture')],
                   check=True, capture_output=True)
    manifest_path = _cloudiness_manifest(tmp_path, 5)
    receipt = replay(library, manifest_path)
    assert receipt['different_words'] == 0
    assert receipt['comparisons'][0]['words'] == 18
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with Capture(transport(library), tmp_path / 'capture.post.cldfrac.bin') as captured:
        perturbed = captured.array().copy()
        perturbed.view(np.uint32)[0, 0] ^= np.uint32(1)
        assert captured.compare(perturbed)['different_words'] == 1
    malformed = tmp_path / 'malformed.bin'
    malformed.write_bytes(b'bad')
    with pytest.raises(ValueError, match='Rust rejected'):
        Capture(transport(library), malformed)


@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
def test_native_real_mesh_partition(library, tmp_path, scheme):
    _native_mesh_partition(library, tmp_path, scheme)


@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
def test_native_real_case_columns(library, tmp_path, scheme):
    init = os.environ.get('MPAS_NATIVE_INIT')
    state = os.environ.get('MPAS_NATIVE_STATE')
    if not init or not state:
        pytest.skip('requires published initialization and strict native initial diagnostic output')
    _native_mesh_partition(library, tmp_path, scheme, case_files=(init, state))


@pytest.mark.gpu
@pytest.mark.parametrize('scheme', ['cld_incidence', 'cld_fraction', 'cld_fraction_thompson'])
def test_native_device_driver_seam(library, tmp_path, scheme):
    # GPU use requires explicit opt-in under the GPU queue; importing tests opens no GPU.
    # MPAS source: src/core_atmosphere/physics/mpas_atmphys_driver_cloudiness.F:175-372.
    if os.environ.get('MPAS_NATIVE_DEVICE_TEST') != '1':
        pytest.skip('requires an explicit the GPU queue job')
    import cupy as cp
    mesh = os.environ['MPAS_NATIVE_MESH']
    graph = os.environ['MPAS_NATIVE_GRAPH']
    probe = Path(library).with_name('probe_cloudiness_capture')
    subprocess.run([str(probe), scheme, '1', '1', str(tmp_path / 'capture'),
                    mesh, f'{graph}.part.2', '-1'], check=True, capture_output=True)
    lib = transport(library)
    with Capture(lib, tmp_path / 'capture.post.cldfrac.bin') as reference:
        owned = reference.shape[1] - 1
    manifest = json.loads(_cloudiness_manifest(tmp_path, owned).read_text())
    with NativePhysics(library, test_only=True) as native:
        for mapping in manifest['calls'][0]['inputs']:
            with Capture(lib, tmp_path / mapping['capture']) as record:
                role, name = mapping['role'], mapping['name']
                if role == 'config':
                    native.config(name, record.scalar(), pool=mapping['pool'])
                elif role == 'field':
                    native.put_field(mapping['pool'], name, cp.asarray(record.array()))
                elif role == 'variable':
                    native.put_variable(name, cp.asarray(record.array()), lower=record.lower)
                else:
                    raise AssertionError(f'unexpected seam input role: {role}')
        native.allocate_cloudiness()
        native.driver_cloudiness(**manifest['calls'][1]['arguments'])
        with Capture(lib, tmp_path / 'capture.post.cldfrac.bin') as reference:
            output = cp.empty(reference.shape, dtype=cp.float32, order='F')
            assert native.get_field('diag_physics', 'cldfrac', out=output) is output
            compared = reference.compare(output.get(), (1,) * output.ndim)
        assert compared['different_words'] == 0
        native.deallocate_cloudiness()
        receipt = {'schema': 'mpas-native-device-seam-v1', 'forecast_proof': False,
                   'column_origin': 'synthetic atmospheric columns on published native mesh',
                   'mesh': mesh, 'mesh_sha256': hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),
                   'scheme': scheme, 'library_sha256': native.build['library_sha256'],
                   'comparison': compared}
        (tmp_path / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def _native_mesh_partition(library, tmp_path, scheme, case_files=()):
    mesh = os.environ.get('MPAS_NATIVE_MESH')
    graph = os.environ.get('MPAS_NATIVE_GRAPH')
    probe = Path(library).with_name('probe_cloudiness_capture')
    if not mesh or not graph or not probe.is_file():
        pytest.skip('requires published mesh, graph partitions and f1 native capture writer')
    receipts = []
    baseline = None
    for parts in (1, 2, 4):
        partition_receipts = []
        for rank in range(parts):
            folder = tmp_path / f'{parts}-{rank}'
            folder.mkdir()
            subprocess.run([str(probe), scheme, '1', '0' if case_files else '1', str(folder / 'capture'),
                            mesh, f'{graph}.part.{parts}', str(rank if parts > 1 else -1), *case_files],
                           check=True, capture_output=True)
            post = folder / 'capture.post.cldfrac.bin'
            # Scalar returned by the Rust decoder; no Python field decoding.
            with Capture(transport(library), post) as capture:
                stored_cells = capture.header[16]
                levels = capture.shape[0]
            if parts == 1:
                baseline = post
                owned = stored_cells - 1
                total_cells = owned
            else:
                owned_comparison = _replay_module.compare_owned(library, baseline, post)
                owned = owned_comparison['cells']
                assert owned_comparison['different_words'] == 0
                partition_receipts.append(owned_comparison)
            manifest = _cloudiness_manifest(folder, owned)
            receipt = replay(library, manifest)
            assert receipt['different_words'] == 0
            receipts.append({'parts': parts, 'rank': rank, 'stored_cells': stored_cells,
                             'owned_cells': owned, 'backend_replay': receipt})
        if parts > 1:
            assert sum(v['cells'] for v in partition_receipts) == total_cells
            assert sum(v['words'] for v in partition_receipts) == total_cells * levels
        receipts.append({'parts': parts, 'global_id_comparisons': partition_receipts})
    report = {'schema': 'mpas-native-column-partition-v1', 'forecast_proof': False,
              'weather_columns': ('native initialization and initial diagnostics through pristine MPAS_to_physics'
                                  if case_files else 'synthetic; mesh areas, density, IDs and adjacency are native'),
              'column_origin': os.environ.get('MPAS_NATIVE_COLUMN_ORIGIN', 'published-real-analysis') if case_files else 'synthetic',
              'case_files': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case_files},
              'mesh': mesh, 'mesh_sha256': hashlib.sha256(Path(mesh).read_bytes()).hexdigest(),
              'partition_hashes': {str(n): hashlib.sha256(Path(f'{graph}.part.{n}').read_bytes()).hexdigest()
                                   for n in (2, 4)},
              'scheme': scheme, 'cells': total_cells, 'levels': levels,
              'different_words': 0, 'runs': receipts}
    (tmp_path / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')


def test_native_complete_noahmp_io_transport(library):
    # MPAS source: src/core_atmosphere/physics/physics_noahmp/drivers/mpas/NoahmpIOVarType.F90:18-938.
    import ctypes as ct
    with NativePhysics(library, test_only=True) as native:
        fields = native.abi.get('noahmp_io')
        if fields is None:
            pytest.skip('historical immutable provider predates complete derived IO export')
        assert len(fields) == 808
        assert 'infxsrt' not in fields  # WRF-only member must not shift the MPAS ABI.
        direct = native.lib.native_noahmp_io_copy
        direct.argtypes = [ct.c_int, ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                           ct.POINTER(ct.c_int), ct.c_void_p, ct.POINTER(ct.c_int)]
        direct.restype = None
        for name, field in fields.items():
            dims, lower = (ct.c_int * 5)(), (ct.c_int * 5)()
            width, status = ct.c_int(), ct.c_int()
            direct(field['index'], dims, lower, ct.byref(width), None, ct.byref(status))
            actual = native.get_noahmp_io(name)
            if status.value == 1:
                assert actual is None and field['rank'] > 0
            else:
                assert status.value == 0
                value, bounds = actual
                assert value.shape == tuple(dims[:field['rank']])
                assert bounds == tuple(lower[:field['rank']])
                assert value.dtype.itemsize == width.value
                if field['kind'] == 'logical':
                    assert value.dtype == np.dtype('int32')
                if field['kind'] == 'character':
                    assert value.dtype.itemsize == field['character_length']
        with pytest.raises(KeyError):
            native.get_noahmp_io('unknown_native_component')


@pytest.mark.parametrize('dtype,rank', [('float32', r) for r in range(6)] + [('int32', r) for r in range(4)])
def test_field_input_storage_is_not_retained(library, dtype, rank):
    # MPAS source: src/framework/mpas_pool_routines.F:2108-3211,3436-4240.
    # Verify the real ABI copies on creation and update, including scalar ranks,
    # distinct time levels, subnormal/signed-zero/NaN payload bits and disposal.
    shape = (2,) * rank
    words = np.resize(np.array([0x80000000, 0x00000001, 0x7fc12345, 0x3f800000],
                               dtype=np.uint32), shape)
    value = np.array(words.view(dtype), order='F', copy=True)
    expected = value.tobytes(order='F')
    with NativePhysics(library, test_only=True) as native:
        native.put_field('diag_physics', 'owned_words', value, nlevels=2)
        value[...] = 0
        assert native.get_field('diag_physics', 'owned_words').tobytes(order='F') == expected
        native.put_field('diag_physics', 'owned_words', value, level=2, nlevels=2)
        value[...] = 1
        assert native.get_field('diag_physics', 'owned_words', level=2).tobytes(order='F') == bytes(value.nbytes)
        native.put_field('diag_physics', 'owned_words', value, nlevels=2)
        expected_update = value.tobytes(order='F')
        value[...] = 2
        assert native.get_field('diag_physics', 'owned_words').tobytes(order='F') == expected_update
        assert native.get_field('diag_physics', 'owned_words', level=2).tobytes(order='F') == bytes(value.nbytes)
    # Destruction must not free or overwrite the caller's surviving storage.
    assert np.all(value == 2)


@pytest.mark.parametrize('layout', ['c', 'f', 'strided', 'reversed', 'readonly', 'unaligned', 'empty'])
def test_field_input_layout_and_payload(library, layout):
    # MPAS source: src/framework/mpas_pool_routines.F:2108-3211,3436-4240.
    words = np.resize(np.array([0x80000000, 0x00000001, 0x7fc12345, 0xff800000],
                               dtype=np.uint32), (4, 6))
    value = np.array(words.view(np.float32), order='F' if layout != 'c' else 'C')
    if layout == 'strided': value = value[:, ::2]
    if layout == 'reversed': value = value[::-1, :]
    if layout == 'readonly': value.flags.writeable = False
    if layout == 'unaligned':
        unaligned = np.ndarray(value.shape, dtype=value.dtype,
                               buffer=bytearray(value.nbytes + 1), offset=1, order='F')
        unaligned[...] = value
        value = unaligned
        assert not value.flags.aligned
    if layout == 'empty': value = value[:, :0]
    expected = value.tobytes(order='F')
    with NativePhysics(library, test_only=True) as native:
        native.put_field('diag_physics', 'layout_words', value)
        native.put_field('diag_physics', 'layout_words', value)
        assert native.get_field('diag_physics', 'layout_words').tobytes(order='F') == expected
        assert value.tobytes(order='F') == expected
