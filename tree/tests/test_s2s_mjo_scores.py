"""Regression tests, including a whole Rust reduction of analytical fixtures.

Fixture fields are not an ERA5 hindcast and do not establish forecast skill.
"""
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hexcore.s2s_door import S2SError
from hexcore import s2s_mjo


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_binding_restart_overlap_and_member_export_init(tmp_path, monkeypatch):
    start = datetime(2020, 1, 1)
    first, second = tmp_path / 'a1', tmp_path / 'a2'
    members = [{'run_id': name, 'init': str(tmp_path / (name + '.init.nc')),
                'start_time': '2019-12-31_18:00:00'} for name in ('mem001', 'mem002')]
    for member in members:
        for day in range(4):
            stamp = start + timedelta(days=day)
            path = first / member['run_id'] / stamp.strftime('cuda-history.%Y-%m-%d_%H.%M.%S.nc')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'original')
        newer = second / member['run_id'] / 'cuda-history.2020-01-03_00.00.00.nc'
        newer.parent.mkdir(parents=True)
        newer.write_bytes(b'restart overlap')
        newer.with_suffix('.nc.part').write_bytes(b'unfinished')
    value = {'start_time': start.strftime(s2s_mjo.TIME_FORMAT), 'signature': 'run-signature',
             'paths': {'work': str(tmp_path), 'grid': str(tmp_path / 'grid.nc')}, 'members': members}
    state = {'completed': ['mem001', 'mem002'], 'attempts': [
        {'out': str(first), 'members': ['mem001', 'mem002']},
        {'out': str(second), 'members': ['mem001', 'mem002']}]}
    args = SimpleNamespace(days=3, history_hours=24, s2s_export_window='global')
    binding, path = s2s_mjo.bind_histories(value, state, args)
    assert path.is_file()
    assert all(len(row['frames']) == 4 for row in binding['members'])
    assert all('/a2/' in row['frames'][2]['path'] for row in binding['members'])
    calls = []
    def export(argv, log, env):
        calls.append(argv)
        out = Path(argv[argv.index('--out') + 1])
        write(out / 'export-receipt.json', {'schema': 'gpuwm-hex.export/v1', 'frames': [{}] * 4,
            'histories': argv[argv.index('--history') + 1:argv.index('--mesh')],
            'init': argv[argv.index('--init') + 1],
            'simulation_start': argv[argv.index('--simulation-start') + 1], 'field_set': 'full'})
        return 0
    monkeypatch.setattr('hexcore.global_ensemble._cli', export)
    receipt = s2s_mjo.export_members(value, state, args, {})
    assert receipt['status'] == 'complete'
    assert {argv[argv.index('--init') + 1] for argv in calls} == {m['init'] for m in members}
    assert all(argv[argv.index('--simulation-start') + 1] == '2019-12-31_18:00:00' for argv in calls)
    assert all(argv[argv.index('--field-set') + 1] == 'full' for argv in calls)
    state['completed'].pop()
    with pytest.raises(S2SError, match='incomplete S2S member mem002'):
        s2s_mjo.bind_histories(value, state, args)
    state['completed'].append('mem002')
    (first / 'mem001' / 'cuda-history.2020-01-04_00.00.00.nc').unlink()
    with pytest.raises(S2SError, match='missing 1 scheduled histories'):
        s2s_mjo.bind_histories(value, state, args)


def test_missing_reference_is_explicit(tmp_path):
    args = SimpleNamespace(mjo_reference=None, stage='diagnostics')
    with pytest.raises(S2SError, match='fixed observed EOFs'):
        s2s_mjo.build_mjo({'paths': {'work': str(tmp_path)}}, {}, args, {})
    assert json.loads((tmp_path / 's2s-mjo/receipt.json').read_text())['status'] == 'unavailable'


def test_failed_export_retry_invalidates_completed_receipt(tmp_path):
    receipt = tmp_path / 's2s-export/receipt.json'
    write(receipt, {'status': 'complete', 'members': ['old-member']})
    with pytest.raises(S2SError, match='incomplete S2S member'):
        s2s_mjo.export_members({'start_time': '2020-01-01_00:00:00',
            'paths': {'work': str(tmp_path)}, 'members': [{'run_id': 'mem001'}]},
            {'completed': []}, SimpleNamespace(), {})
    value = json.loads(receipt.read_text())
    assert value['status'] == 'failed' and 'old-member' not in str(value)
    assert 'incomplete S2S member' in value['error']


def test_public_door_failed_export_replaces_old_stage_success(tmp_path):
    from hexcore.cli import build_parser
    from hexcore import s2s_door
    args = build_parser().parse_args(['s2s', '--row', 'global-120km', '--source', 'gfs',
        '--cycle', '2020-01-01T00', '--days', '30', '--members', '2',
        '--work-dir', str(tmp_path), '--stage', 'export'])
    value = s2s_door.plan(args)
    write(tmp_path / 's2s-state.json', {'schema': s2s_door.SCHEMA,
        'signature': value['signature'], 'attempts': [], 'completed': [],
        'stages': {'export': {'status': 'complete', 'members': ['old-member']}}})
    with pytest.raises(S2SError, match='incomplete S2S member'):
        s2s_door.run(args)
    for name in ['s2s-state.json', 's2s-receipt.json']:
        stage = json.loads((tmp_path / name).read_text())['stages']['export']
        assert stage['status'] == 'failed' and 'old-member' not in str(stage)
        assert 'incomplete S2S member' in stage['error']


@pytest.mark.parametrize('native_receipt', [{'status': 'complete'},
    {'status': 'complete', 'schema': 'gpuwm-hex.s2s-scores/v1'}])
def test_backend_rejects_wrong_schema_and_invalidates_success(tmp_path, monkeypatch, native_receipt):
    output = tmp_path / 'native'
    write(output / 'receipt.json', {'status': 'complete', 'schema': 'gpuwm-hex.s2s-mjo/v1'})
    monkeypatch.setattr(s2s_mjo, 'backend', lambda: sys.executable)
    def invoke(argv, **kwargs):
        assert json.loads((output / 'receipt.json').read_text())['status'] == 'running'
        write(output / 'receipt.json', native_receipt)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(s2s_mjo.subprocess, 'run', invoke)
    with pytest.raises(S2SError, match='did not complete'):
        s2s_mjo.run_backend('mjo', {}, output, {})
    assert json.loads((output / 'receipt.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('render_result', ['failed', 'silent', 'partial', 'complete'])
def test_render_retry_requires_acknowledged_complete_figure_set(tmp_path, monkeypatch, render_result):
    from types import ModuleType
    native = {'schema': 'gpuwm-hex.s2s-mjo/v1', 'status': 'complete',
        'receipt': str(tmp_path / 's2s-mjo/receipt.json'), 'series': [
        {'diagnostic': 'rmm', 'member': 'mem001', 'fields': ['olr', 'u850', 'u200'], 'latitude_bins': 1}]}
    monkeypatch.setattr(s2s_mjo, 'bind_histories', lambda *a: ({}, tmp_path / 'binding.json'))
    monkeypatch.setattr(s2s_mjo, 'run_backend', lambda *a: native)
    bridge = ModuleType('gpuwm.rustwx')
    bridge.find_renderer = lambda: Path('/real/rw_wrfbatch')
    bridge.renderer_env = lambda: {}
    engine = ModuleType('gpuwm')
    engine.rustwx = bridge
    monkeypatch.setitem(sys.modules, 'gpuwm', engine)
    monkeypatch.setitem(sys.modules, 'gpuwm.rustwx', bridge)
    monkeypatch.setattr('hexcore.engines.resolve', lambda _: tmp_path / 'rw_wrfbatch')
    directory = tmp_path / 's2s-mjo/figures'
    directory.mkdir(parents=True)
    names = ['rmm-mem001-phase.png'] + [f'rmm-mem001-{f}-longitude-time.png' for f in ['olr', 'u850', 'u200']]
    # Every old picture exists, including an extra member from a previous run.
    for name in names + ['rmm-old-member-phase.png']:
        (directory / name).write_bytes(b'old picture')
    def render(argv, stdout, **kwargs):
        assert json.loads((directory.parent / 'stage-receipt.json').read_text())['status'] == 'running'
        acknowledged = names if render_result == 'complete' else names[:1] if render_result == 'partial' else []
        for name in acknowledged:
            (directory / name).write_bytes(b'new picture')
            stdout.write('RENDERED\t' + str(directory / name) + '\n')
        return SimpleNamespace(returncode=2 if render_result == 'failed' else 0)
    monkeypatch.setattr(s2s_mjo.subprocess, 'run', render)
    args = SimpleNamespace(mjo_reference=tmp_path / 'basis.json', s2s_reference_license=None)
    call = lambda: s2s_mjo.build_mjo({'paths': {'work': str(tmp_path)}}, {}, args, {'RAYON_NUM_THREADS': '3'})
    if render_result == 'complete':
        value = call()
        assert len(value['images']) == 4 and len(value['figure_sha256']) == 4
        assert value['series_rows'] == 1 and 'series' not in value
        assert not any('old-member' in p for p in value['images'])
        assert value['render_seconds'] >= 0
        assert value['parallelism']['threads'] == 3
    else:
        with pytest.raises(S2SError):
            call()
    assert json.loads((directory.parent / 'stage-receipt.json').read_text())['status'] == (
        'complete' if render_result == 'complete' else 'failed')


@pytest.fixture
def backend():
    path = os.environ.get('RW_S2S_TEST_BINARY')
    if not path:
        pytest.skip('set RW_S2S_TEST_BINARY to the built native reducer for NetCDF integration tests')
    assert Path(path).is_file()
    return path


@pytest.fixture
def analytical_case(tmp_path):
    from netCDF4 import Dataset
    start = datetime(2020, 1, 1)
    days = 35
    latitude, longitude = np.meshgrid(np.array([-15., -5., 5., 15., 25., 35.]),
                                     np.arange(2.5, 360., 5.), indexing='ij')
    lat, lon = np.deg2rad(latitude.ravel()), np.deg2rad(longitude.ravel())
    n = lat.size
    grid = tmp_path / 'mesh.nc'
    with Dataset(grid, 'w') as ds:
        ds.createDimension('nCells', n)
        # Native global meshes use the standard SI abbreviation, while older
        # fixtures used the expanded spelling. Exercise both in the whole case.
        for name, values, units in [('latCell', lat, 'rad'), ('lonCell', lon, 'radians'),
                                    ('areaCell', np.cos(lat), 'm2')]:
            var = ds.createVariable(name, 'f8', ('nCells',)); var[:] = values; var.units = units
    members = []
    for name in ['mem001', 'mem002']:
        frames = []
        for day in range(days + 1):
            stamp = start + timedelta(days=day)
            path = tmp_path / name / stamp.strftime('cuda-history.%Y-%m-%d_%H.%M.%S.nc')
            path.parent.mkdir(exist_ok=True)
            with Dataset(path, 'w') as ds:
                for dim, size in [('Time', 1), ('nCells', n), ('nVertLevels', 4), ('StrLen', 64)]:
                    ds.createDimension(dim, size)
                clock = ds.createVariable('xtime', 'S1', ('Time', 'StrLen'))
                clock[0] = np.frombuffer(stamp.strftime(s2s_mjo.TIME_FORMAT).ljust(64).encode(), dtype='S1')
                for label, values in [('latCell', lat), ('lonCell', lon)]:
                    # The real forecast stores signed FP32 longitudes, while
                    # the mesh above retains unsigned double coordinates.
                    carried = np.where(values > np.pi, values - 2 * np.pi, values) if label == 'lonCell' else values
                    ds.createVariable(label, 'f4', ('nCells',))[:] = carried
                var = ds.createVariable('olr', 'f8', ('Time', 'nCells'))
                var[:] = 240. + day + np.sin(lon) * 2.; var.units = 'W m-2'
                var = ds.createVariable('pressure', 'f8', ('Time', 'nCells', 'nVertLevels'))
                var[:] = np.broadcast_to([100000., 85000., 50000., 20000.], (1, n, 4)); var.units = 'Pa'
                var = ds.createVariable('u_zonal', 'f8', ('Time', 'nCells', 'nVertLevels'))
                var[:] = np.broadcast_to([4., 3., 2., 1.], (1, n, 4)) + day / 10.; var.units = 'm s-1'
            frames.append({'valid_time': stamp.strftime(s2s_mjo.TIME_FORMAT), 'path': str(path), 'bytes': path.stat().st_size})
        members.append({'member': name, 'frames': frames})
    write(tmp_path / 'binding.json', {'schema': 'gpuwm-hex.s2s-history-binding/v1',
        'start': start.strftime(s2s_mjo.TIME_FORMAT), 'days': days, 'cadence_hours': 24,
        'grid': str(grid), 'members': members, 'binding_sha256': 'analytical-regression-only'})
    definitions = []
    for name, fields, domain, ny in [('rmm', ['olr', 'u850', 'u200'], [-15., 15., 0., 360.], 1),
                                   ('bsiso1', ['olr', 'u850'], [-10., 40., 40., 160.], 2),
                                   ('bsiso2', ['olr', 'u850'], [-10., 40., 40., 160.], 2)]:
        size = len(fields) * 4 * ny
        normal = np.repeat([240., 3., 1.][:len(fields)], 4 * ny).tolist()
        definitions.append({'id': name, 'fields': fields, 'domain': domain, 'nx': 4, 'ny': ny,
            'standard_deviation': [1.] * len(fields), 'eofs': np.eye(size)[:2].tolist(),
            'pc_standard_deviation': [1., 1.], 'phase_order': [5, 6, 7, 8, 1, 2, 3, 4],
            'eof_modes': [3, 4] if name == 'bsiso2' else [1, 2],
            'seasonal': {(start + timedelta(days=d)).strftime('%Y-%m-%d'): normal for d in range(days)},
            'prior_anomalies': {(start - timedelta(days=d)).strftime('%Y-%m-%d'): [0.] * size for d in range(1, 121)}})
    write(tmp_path / 'basis.json', {'schema': 'gpuwm-hex.s2s-mjo-reference/v1',
        'source': 'analytical regression fixture, not observations', 'license': 'CC0-1.0',
        'training_period': 'analytical fixture', 'daily_interval_start_hour': 0, 'basis_id': 'test-only-basis', 'method': 'fixed-observed-eofs',
        'raw_fields': {'olr': 'olr', 'pressure': 'pressure', 'u': 'u_zonal'}, 'definitions': definitions})
    plan = {'binding': str(tmp_path / 'binding.json'), 'reference': str(tmp_path / 'basis.json'),
            'output': str(tmp_path / 'mjo')}
    write(tmp_path / 'mjo-plan.json', plan)
    return tmp_path, start, days, plan


def invoke(backend, mode, plan):
    return subprocess.run([backend, mode, str(plan)], capture_output=True, text=True,
                          env={**os.environ, 'RAYON_NUM_THREADS': '2'})


def test_native_transport_cf_unpack_and_bounded_time_slice(backend, tmp_path):
    from netCDF4 import Dataset
    source = tmp_path / 'packed.nc'
    with Dataset(source, 'w') as ds:
        ds.createDimension('time', 3)
        ds.createDimension('x', 2)
        var = ds.createVariable('wind', 'i2', ('time', 'x'), fill_value=-32767)
        var.set_auto_maskandscale(False)
        var[:] = [[2, 4], [6, -32767], [10, 12]]
        var.scale_factor = .5
        var.add_offset = 100.
        var.units = 'm s-1'
    plan = tmp_path / 'extract.json'
    write(plan, {'input': str(source), 'output': str(tmp_path / 'arrays'),
                 'variables': [{'name': 'wind', 'start': 1, 'count': 1}]})
    result = invoke(backend, 'extract', plan)
    assert result.returncode == 0, result.stderr
    row = json.loads((tmp_path / 'arrays/arrays.json').read_text())['arrays']['wind']
    assert row['shape'] == [1, 2] and row['units'] == 'm s-1'
    np.testing.assert_equal(np.fromfile(row['path'], dtype='<f8'), [103., np.nan])
    write(plan, {'input': str(source), 'output': str(tmp_path / 'arrays'),
                 'variables': [{'name': 'wind', 'start': 2, 'count': 2}]})
    result = invoke(backend, 'extract', plan)
    assert result.returncode == 2 and 'selection' in result.stderr


def test_fixed_observation_mask_excludes_unsupported_pressure_surface(backend, analytical_case):
    from netCDF4 import Dataset
    root, _, _, _ = analytical_case
    basis = json.loads((root / 'basis.json').read_text())
    # Every u850 feature is excluded by an independently specified fixed mask.
    # Other pressure surfaces and OLR remain fully checked and projected.
    for d in basis['definitions']:
        size = d['nx'] * d['ny']
        d['feature_mask'] = [field != 'u850' for field in d['fields'] for _ in range(size)]
        for table in ('seasonal', 'prior_anomalies'):
            for values in d[table].values():
                values[size:2 * size] = [0.] * size
    basis['license'] = 'CC-BY-4.0'
    write(root / 'basis.json', basis)
    binding = json.loads((root / 'binding.json').read_text())
    for member in binding['members']:
        for frame in member['frames']:
            with Dataset(frame['path'], 'a') as ds:
                # Only 850 hPa lies below the first model level.
                ds.variables['pressure'][:, :, 0] = 80000.
                ds.variables['pressure'][:, :, 1] = 70000.
            stat = Path(frame['path']).stat()
            frame.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write(root / 'binding.json', binding)
    result = invoke(backend, 'mjo', root / 'mjo-plan.json')
    assert result.returncode == 0, result.stderr
    rows = json.loads((root / 'mjo/receipt.json').read_text())['series']
    assert all(all(value == 0. for value in r['hovmoller'][1]) for r in rows)


def test_rust_daily_mjo_bsiso_and_weekly_scores(backend, analytical_case):
    from netCDF4 import Dataset
    root, start, days, plan = analytical_case
    process = invoke(backend, 'mjo', root / 'mjo-plan.json')
    assert process.returncode == 0, process.stderr
    mjo = json.loads((root / 'mjo/receipt.json').read_text())
    assert len(mjo['series']) == days * 2 * 3
    rmm = [r for r in mjo['series'] if r['diagnostic'] == 'rmm' and r['member'] == 'mem001']
    # Daily integration is exact for a linearly increasing field. Removing a
    # causal 120-day mean must use only completed earlier forecast intervals.
    longitude_centres = np.deg2rad(np.arange(2.5, 360., 5.))
    initial = 0.5 + 2. * np.sin(longitude_centres).reshape(4, 18).mean(axis=1)
    for day, row in enumerate(rmm):
        expected = initial + day - (day * initial + day * (day - 1) / 2) / 120
        np.testing.assert_allclose(row['hovmoller'][0], expected, atol=1e-11)
        np.testing.assert_allclose(row['pc'], expected[:2], atol=1e-11)
        wind = 0.05 + 0.1 * day - (0.05 * day + 0.1 * day * (day - 1) / 2) / 120
        np.testing.assert_allclose(row['hovmoller'][1:], wind, atol=1e-11)
        assert row['date'] == (start + timedelta(days=day)).strftime('%Y-%m-%d')
    monsoon_longitudes = np.deg2rad(np.arange(42.5, 160., 5.))
    bsiso_initial = 0.5 + 2. * np.sin(monsoon_longitudes).reshape(4, 6).mean(axis=1)
    for row in mjo['series']:
        if row['diagnostic'] in ('bsiso1', 'bsiso2'):
            day = (datetime.strptime(row['date'], '%Y-%m-%d') - start).days
            expected = bsiso_initial + day - (day * bsiso_initial + day * (day - 1) / 2) / 120
            np.testing.assert_allclose(row['hovmoller'][0], expected, atol=1e-11)
            np.testing.assert_allclose(row['latitude_hovmoller'][0], expected.mean(), atol=1e-11)
            np.testing.assert_allclose(row['pc'], expected[:2], atol=1e-11)
    latitude, longitude = np.meshgrid(np.linspace(-90, 90, 5), np.arange(-180, 180, 90), indexing='ij')
    weeks = []
    for week in range(1, 6):
        begin = (start + timedelta(days=(week - 1) * 7)).strftime(s2s_mjo.TIME_FORMAT)
        end = (start + timedelta(days=week * 7)).strftime(s2s_mjo.TIME_FORMAT)
        for kind in ['forecast', 'observations', 'climatology']:
            path = root / ('products' if kind == 'forecast' else 'reference') / (f'week-{week}.nc' if kind == 'forecast' else f'{kind}-{week}.nc')
            path.parent.mkdir(exist_ok=True)
            with Dataset(path, 'w') as ds:
                ds.createDimension('south_north', 5); ds.createDimension('west_east', 4)
                for name, values, units in [('XLAT', latitude, 'degrees_north'), ('XLONG', longitude, 'degrees_east'),
                                             ('model_orography', np.zeros((5, 4)), 'm')]:
                    var = ds.createVariable(name, 'f8', ('south_north', 'west_east')); var[:] = values; var.units = units
                if kind == 'forecast':
                    ds.s2s_member_ids = json.dumps(['mem001', 'mem002']); ds.s2s_window = f'week-{week}'; ds.valid_time = end
                else:
                    ds.s2s_reference_source = 'ERA5'; ds.s2s_reference_kind = kind
                    ds.s2s_window_start = begin; ds.s2s_window_end = end
                for field, units in [('temperature2', 'K'), ('precipitation_rate', 'mm day-1'), ('z500', 'm'), ('mslp', 'hPa')]:
                    pattern = np.arange(20).reshape(5, 4) + 1.
                    if kind == 'forecast':
                        ds.createDimension(field + '_member', 2)
                        var = ds.createVariable(field + '_members', 'f8', (field + '_member', 'south_north', 'west_east'))
                        var[:] = np.stack([pattern + 1., pattern + 3.])
                    else:
                        var = ds.createVariable(field, 'f8', ('south_north', 'west_east'))
                        var[:] = pattern if kind == 'observations' else np.zeros_like(pattern)
                    var.units = units
        weeks.append({'week': week, 'start': begin, 'end': end, 'observations': f'reference/observations-{week}.nc',
                      'climatology': f'reference/climatology-{week}.nc'})
    obs = {'schema': 'gpuwm-hex.s2s-observations/v1', 'source': 'ERA5', 'license': 'CC0-1.0',
           'provenance': 'analytical regression fixture only; not ERA5 evidence', 'weeks': weeks,
           'rmm': {r['date']: r['pc'] for r in rmm}, 'minimum_area_fraction': 0.95,
           'rmm_basis_id': 'test-only-basis', 'rmm_sampling': 'daily-interval-start',
           'rmm_interval_start': start.strftime(s2s_mjo.TIME_FORMAT)}
    write(root / 'obs.json', obs)
    score_plan = {'observation_manifest': str(root / 'obs.json'), 'products': str(root / 'products'),
                  'mjo': str(root / 'mjo/receipt.json'), 'output': str(root / 'scores'), 'days': days,
                  'start': start.strftime(s2s_mjo.TIME_FORMAT), 'members': ['mem001', 'mem002']}
    write(root / 'score-plan.json', score_plan)
    process = invoke(backend, 'score', root / 'score-plan.json')
    assert process.returncode == 0, process.stderr
    scores = json.loads((root / 'scores/receipt.json').read_text())
    write(root / 'successful-score-receipt.json', scores)
    assert len(scores['weekly']) == 5 * 4 * 3
    assert len(scores['rmm']) == 3 * 6
    assert all(r['scores']['rmse'] == pytest.approx(2.) for r in scores['weekly'] if r['member'] == 'ensemble-mean')
    assert all(r['scores']['bivariate_correlation'] == pytest.approx(1.) for r in scores['rmm'])
    obs['rmm_basis_id'] = 'different-basis'
    write(root / 'obs.json', obs)
    assert 'basis or daily sampling differs' in invoke(backend, 'score', root / 'score-plan.json').stderr
    assert json.loads((root / 'scores/receipt.json').read_text())['status'] == 'failed'
    obs['rmm_basis_id'] = 'test-only-basis'
    obs['rmm_interval_start'] = '2020-01-01_12:00:00'
    write(root / 'obs.json', obs)
    assert 'basis or daily sampling differs' in invoke(backend, 'score', root / 'score-plan.json').stderr
    obs['rmm_interval_start'] = start.strftime(s2s_mjo.TIME_FORMAT)
    obs['weeks'].pop()
    write(root / 'obs.json', obs)
    assert 'week-5 missing' in invoke(backend, 'score', root / 'score-plan.json').stderr
    obs['weeks'] = weeks
    # Restore from the recorded table, whose list was popped above.
    obs['weeks'].append({'week': 5, 'start': (start + timedelta(days=28)).strftime(s2s_mjo.TIME_FORMAT),
        'end': (start + timedelta(days=35)).strftime(s2s_mjo.TIME_FORMAT),
        'observations': 'reference/observations-5.nc', 'climatology': 'reference/climatology-5.nc'})
    write(root / 'obs.json', obs)
    with Dataset(root / 'products/week-1.nc', 'a') as ds:
        ds.variables['XLAT'][1, :] = -40.
    assert 'not regular global' in invoke(backend, 'score', root / 'score-plan.json').stderr
    with Dataset(root / 'products/week-1.nc', 'a') as ds:
        ds.variables['XLAT'][:] = latitude
    with Dataset(root / 'reference/observations-1.nc', 'a') as ds:
        ds.variables['temperature2'].units = 'degC'
    assert 'units' in invoke(backend, 'score', root / 'score-plan.json').stderr
    with Dataset(root / 'reference/observations-1.nc', 'a') as ds:
        ds.variables['temperature2'].units = 'K'
        ds.variables['temperature2'][1:4, :] = np.nan
    assert 'insufficient paired area' in invoke(backend, 'score', root / 'score-plan.json').stderr


@pytest.mark.parametrize('defect', ['prior', 'seasonal', 'eof', 'license', 'reference-hour', 'clock', 'native-clock', 'mtime', 'roster', 'feature-mask'])
def test_rust_rejects_unqualified_inputs(backend, analytical_case, defect):
    root, start, days, plan = analytical_case
    basis = json.loads((root / 'basis.json').read_text())
    if defect == 'prior':
        basis['definitions'][0]['prior_anomalies'].pop('2019-12-31')
    elif defect == 'seasonal':
        basis['definitions'][0]['seasonal'].pop('2020-01-15')
    elif defect == 'eof':
        basis['definitions'][0]['eofs'][0] = basis['definitions'][0]['eofs'][0][:-1]
    elif defect == 'license':
        basis['license'] = 'CC-BY-NC-4.0'
    elif defect == 'feature-mask':
        basis['definitions'][0]['feature_mask'] = [True]
    elif defect == 'reference-hour':
        basis['daily_interval_start_hour'] = 12
    elif defect in ['clock', 'native-clock', 'mtime', 'roster']:
        binding = json.loads((root / 'binding.json').read_text())
        if defect == 'clock':
            binding['members'][0]['frames'][0]['valid_time'] = '2020-01-02_00:00:00'
        elif defect == 'native-clock':
            from netCDF4 import Dataset
            frame = binding['members'][0]['frames'][0]
            with Dataset(frame['path'], 'a') as ds:
                ds.variables['xtime'][0] = np.frombuffer(b'2020-01-02_00:00:00'.ljust(64), dtype='S1')
            frame['bytes'] = Path(frame['path']).stat().st_size
        elif defect == 'mtime':
            binding['members'][0]['frames'][0]['mtime_ns'] = 0
        else:
            binding['members'][0]['frames'].pop()
        write(root / 'binding.json', binding)
    write(root / 'basis.json', basis)
    process = invoke(backend, 'mjo', root / 'mjo-plan.json')
    assert process.returncode == 2
    assert json.loads((root / 'mjo/receipt.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('zero_day', [0, 1])
def test_cold_start_olr_is_unavailable_not_a_physical_zero(backend, analytical_case, zero_day):
    from netCDF4 import Dataset
    root, start, days, plan = analytical_case
    binding = json.loads((root / 'binding.json').read_text())
    frame = binding['members'][0]['frames'][zero_day]
    with Dataset(frame['path'], 'a') as ds:
        ds.variables['olr'][:] = 0.
    frame['bytes'] = Path(frame['path']).stat().st_size
    write(root / 'binding.json', binding)
    process = invoke(backend, 'mjo', root / 'mjo-plan.json')
    if zero_day:
        assert process.returncode != 0
        assert 'globally zero OLR after initialization' in process.stderr
        return
    assert process.returncode == 0, process.stderr
    receipt = json.loads((root / 'mjo/receipt.json').read_text())
    assert len(receipt['cold_start_olr_intervals']) == 1
    assert receipt['cold_start_olr_intervals'][0]['member'] == 'mem001'
    assert receipt['cold_start_olr_intervals'][0]['hours'] == 24
    rows = [r for r in receipt['series'] if r['diagnostic'] == 'rmm']
    first = {r['member']: r for r in rows if r['date'] == start.strftime('%Y-%m-%d')}
    # First valid ending flux replaces the unavailable F000 endpoint; winds
    # retain their exact trapezoid. The first anomaly then enters the causal
    # background once, so the correction on the next day is -0.5/120.
    np.testing.assert_allclose(np.array(first['mem001']['hovmoller'][0]) - first['mem002']['hovmoller'][0], .5, atol=1e-11)
    np.testing.assert_allclose(first['mem001']['hovmoller'][1:], first['mem002']['hovmoller'][1:], atol=1e-11)
    second = {r['member']: r for r in rows if r['date'] == (start + timedelta(days=1)).strftime('%Y-%m-%d')}
    np.testing.assert_allclose(np.array(second['mem001']['hovmoller'][0]) - second['mem002']['hovmoller'][0], -.5/120, atol=1e-11)
