"""Generated forecasts must preserve licences, variables and the hex valid clock."""
import json
from types import SimpleNamespace
import sys

import pytest

from hexcore import init_sources as src
from hexcore.cli import build_parser


def row(name='earth2-fcn3'):
    return src.resolve_source(src.load_table(), name)


def test_bindings_cover_every_declared_channel_and_complete_pressure_column():
    table = src.load_table()
    generated = [r for r in table['sources'] if r.get('generation')]
    assert len(generated) == 34
    for r in generated:
        g = r['generation']
        assert [b['channel'] for b in g['bindings']] == g['channels']
        assert all(b.get('field') or b.get('reason') for b in g['bindings'])
        if r['status'] == 'refused':
            assert r['refusal']
            continue
        levels = [{b['pressure_pa'] for b in g['bindings'] if b.get('field') == field and 'pressure_pa' in b}
                  for field in ('TT', 'UU', 'VV', 'GHT', 'SPECHUMD')]
        assert all(values == levels[0] for values in levels), r['id']
        assert len(levels[0]) == 13, r['id']
    assert table['_extensions'][0]['sha256'] == src._sha256_file(src.GENERATED_TABLE_PATH)


def test_default_licences_and_research_opt_in_are_enforced_before_generation():
    table = src.load_table()
    defaults = [r for r in table['sources'] if r.get('generation') and r['license']['default_enabled']]
    assert {r['id'] for r in defaults} == {'earth2-fcn3', 'earth2-sfno', 'earth2-aurora'}
    for r in defaults:
        assert src.resolve_source(table, r['id']) is r
        assert all(r['license'][kind]['source'].startswith('https://') for kind in ('code','weights'))
    with pytest.raises(src.InitSourceRefusal, match='CC BY-NC-SA 4.0'):
        src.resolve_source(table, 'earth2-pangu')
    with pytest.raises(src.InitSourceRefusal, match='research-only'):
        src.resolve_source(table, 'earth2-pangu', ['MIT'])
    assert src.resolve_source(table, 'earth2-pangu', ['Apache-2.0','CC BY-NC-SA 4.0'])['status'] == 'research-only'
    with pytest.raises(src.InitSourceRefusal, match='daily-mean'):
        src.resolve_source(table, 'earth2-fuxi-s2s')


def test_one_inference_path_uses_positive_forecast_lead_and_common_valid_time(monkeypatch, tmp_path):
    from contextlib import nullcontext
    calls = []
    def run_model(**kwargs):
        calls.append(kwargs)
        return {'forecast': ['real.nc']}
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_run', SimpleNamespace(run_model=run_model))
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_licensing', SimpleNamespace(licence_scope=lambda _: nullcontext()))
    for name in ('earth2-fcn3','earth2-sfno','earth2-aurora'):
        result = src.generate(row(name), src.Cycle.parse('2024-10-08T06'), 12, None, tmp_path)
        assert calls[-1]['cycle'].isoformat() == '2024-10-08T00:00:00+00:00'
        assert calls[-1]['hours'] == 18
        assert calls[-1]['area'] is None
        assert result['hex_binding']['valid_time'] == '2024-10-08_18:00:00'
        assert result['hex_binding']['forecast_lead_hours'] > 0


def test_global_and_s2s_front_doors_accept_the_same_model_row(tmp_path):
    from hexcore import global_door, s2s_door
    parser = build_parser()
    common = ['--row','global-120km','--source','earth2-fcn3','--cycle','2024-10-08T06','--work-dir',str(tmp_path)]
    global_plan = global_door.plan(parser.parse_args(['global',*common]))
    assert global_plan['commands']['init'][:3] == ['init-source','--source','earth2-fcn3']
    s2s = s2s_door.plan(parser.parse_args(['s2s',*common,'--days','30','--members','2','--lagged','0,6']))
    assert {m['start_time'] for m in s2s['members']} == {'2024-10-08_06:00:00'}
    assert [m['lead_hours'] for m in s2s['members']] == [0,6]
    assert s2s['members'][1]['init_argv'][-2:] == ['--lead-hours','6']
    restricted = [*common,'--source','earth2-pangu']
    with pytest.raises(src.InitSourceRefusal, match='research-only'):
        global_door.plan(parser.parse_args(['global',*restricted]))
    accepted = ['--accept-licence','Apache-2.0','--accept-licence','CC BY-NC-SA 4.0']
    planned = global_door.plan(parser.parse_args(['global',*restricted,*accepted]))
    assert '--accept-licence' in planned['commands']['init']


def test_rust_export_counts_drive_init_not_nominal_model_levels(monkeypatch, tmp_path):
    args = build_parser().parse_args(['init-source','--source','earth2-fcn3','--cycle','2024-10-08T06',
                                     '--out-dir',str(tmp_path),'--grid','g','--static','s','--vertical-spec','v'])
    monkeypatch.setattr(src,'generation_preflight',lambda *a: None)
    monkeypatch.setattr(src,'generate',lambda *a: {'runner':{'files':[]}})
    monkeypatch.setattr(src,'decode_generated',lambda *a: {'met':'actual.met','export':{'nfglevels':14,'nfgsoillevels':4}})
    seen = []
    monkeypatch.setattr(src,'run_init',lambda argv: seen.extend(argv) or 0)
    assert src.run_init_source_command(args) == 0
    assert seen[seen.index('--nfglevels')+1] == '14'
    receipt=json.loads((tmp_path/'init-source-receipt.json').read_text())
    assert receipt['valid_time'] == '2024-10-08_06:00:00'
    assert receipt['effective_first_guess']['nfglevels'] == 14


def test_installed_weight_revision_mismatch_refuses_before_runtime_probe(monkeypatch):
    declared = row()
    g = declared['generation']
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_run', SimpleNamespace(
        model_declaration=lambda _: (None, {'channels':g['channels'], 'profile':g['profile'],
                                           'weights':{'revision':'different checkpoint'}}),
        earth2_python=lambda: pytest.fail('runtime probe ran before weight contract check')))
    monkeypatch.setitem(sys.modules, 'gpuwm.mapped_engine_bridge', SimpleNamespace(require_engine=None))
    monkeypatch.setitem(sys.modules, 'gpuwm.source_authorities', SimpleNamespace(packaged_authorities=None))
    with pytest.raises(src.InitSourceRefusal, match='weights_revision differs'):
        src.generation_preflight(declared)


def test_native_earth2_members_come_from_the_row(tmp_path):
    from hexcore import s2s_door
    parser = build_parser()
    args = ['s2s','--row','global-120km','--source','earth2-fcn3','--cycle','2024-10-08T06',
            '--work-dir',str(tmp_path),'--days','30','--members','2','--lagged','0']
    planned = s2s_door.plan(parser.parse_args(args))
    assert [m['source_member'] for m in planned['members']] == [1,2]
    assert len({tuple(m['init_argv']) for m in planned['members']}) == 2


def test_preflight_retains_model_and_rust_checks_without_inference_runtime(monkeypatch, tmp_path):
    declared = row()
    g = declared['generation']
    fields = {}
    for binding in g['bindings']:
        if binding.get('field') is not None:
            fields.setdefault(binding['source_field'], {'selectors': []})['selectors'].append({
                'name': [binding['channel']], 'level_value': binding.get('pressure_pa', 0) / 100})
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'fields': fields}))
    # The inference environment may be gone after a real forecast has been
    # produced. Preflight still checks its declared provenance and Rust reader.
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_run', SimpleNamespace(
        model_declaration=lambda _: (None, {'channels': g['channels'], 'profile': g['profile'],
            'weights': {'revision': g['weights_revision'], 'licence': declared['license']['weights']['name']}})))
    monkeypatch.setitem(sys.modules, 'gpuwm.source_authorities', SimpleNamespace(
        packaged_authorities=lambda _: {'mapping': mapping}))
    monkeypatch.setitem(sys.modules, 'gpuwm.mapped_engine_bridge', SimpleNamespace(require_engine=lambda: 'rust-reader'))
    calls = []
    def probe(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout='{"wps_intermediate_v5": true}')
    monkeypatch.setattr(src.subprocess, 'run', probe)
    src.generation_preflight(declared)
    assert calls == [['rust-reader', 'capabilities']]


def test_loader_inherits_worker_budget_and_affinity_is_restored_on_failure(monkeypatch, tmp_path):
    from contextlib import nullcontext
    affinity = set(range(168))
    def set_affinity(pid, cpus):
        nonlocal affinity
        affinity = set(cpus)
    monkeypatch.setattr(src.os, 'sched_getaffinity', lambda _: set(affinity), raising=False)
    monkeypatch.setattr(src.os, 'sched_setaffinity', set_affinity, raising=False)
    def fail(**kwargs):
        assert affinity == set(range(14))
        raise RuntimeError('loader failed')
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_run', SimpleNamespace(run_model=fail))
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_licensing', SimpleNamespace(licence_scope=lambda _: nullcontext()))
    with pytest.raises(RuntimeError, match='loader failed'):
        src.generate(row(), src.Cycle.parse('2024-10-08T06'), 0, None, tmp_path, workers=14)
    assert affinity == set(range(168))


def test_verified_forecast_reuse_checks_bytes_and_model_contract(monkeypatch, tmp_path):
    r = row()
    g = r['generation']
    frame = tmp_path / 'forecast.nc'
    frame.write_bytes(b'forecast receipt fixture')
    prior = {'model':g['model'], 'cycle':'2024-10-08T00:00:00+00:00', 'hours':6,
             'members':[0], 'initial_condition_source':g['ic_source'], 'forecast_start_hour':0,
             'model_row':{'channels':g['channels'], 'profile':g['profile'],
                          'weights':{'revision':g['weights_revision'], 'licence':r['license']['weights']['name']}},
             'forecast':[str(frame)], 'runner':{'files':[{'path':str(frame), 'bytes':frame.stat().st_size,
                 'sha256':src._sha256_file(frame), 'member':0, 'lead_hours':6}]}}
    cached = tmp_path / 'earth2-run.json'
    cached.write_text(json.dumps(prior))
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_run', SimpleNamespace(
        run_model=lambda **_: pytest.fail('repeated inference')))
    monkeypatch.setitem(sys.modules, 'gpuwm.earth2_licensing', SimpleNamespace(licence_scope=None))
    result = src.generate(r, src.Cycle.parse('2024-10-08T06'), 0, None, tmp_path)
    assert result['hex_binding']['reused_verified_forecast'] is True
    prior['model_row']['weights']['revision'] = 'different'
    cached.write_text(json.dumps(prior))
    with pytest.raises(src.InitSourceRefusal, match='different request'):
        src.generate(r, src.Cycle.parse('2024-10-08T06'), 0, None, tmp_path)
    prior['model_row']['weights']['revision'] = g['weights_revision']
    cached.write_text(json.dumps(prior))
    frame.write_bytes(b'changed forecast bytes')
    with pytest.raises(src.InitSourceRefusal, match='forecast bytes changed'):
        src.generate(r, src.Cycle.parse('2024-10-08T06'), 0, None, tmp_path)


def test_global_donor_bypasses_dateline_corner_parser_and_reuses_verified_bytes(monkeypatch, tmp_path):
    calls = []
    def extract(request, *, request_path, output):
        calls.append(request)
        output.write_bytes(b'Rust output fixture')
        return {'schema':'arwen.regular-forcing.v1'}
    monkeypatch.setitem(sys.modules, 'gpuwm.zarr_bridge', SimpleNamespace(extract_regular_zarr=extract))
    native = {'area':[-90,179.75,90,-180],
              'fields':[{'output':v,'pressure':False} for v in ('D2','T2','PSFC')]
                       + [{'output':'T','pressure':True}],
              'derived':[{'kind':'pressure_from_levels','output':'PRES'},
                         {'inputs':['D2','T2','PSFC'],'output':'Q2'}]}
    (tmp_path / 'era5-arco-acquisition.json').write_text(json.dumps({'request':{'native_request':native}}))
    generated = {'hex_binding':{}, 'donor':{'provider':'arco','analysis':str(tmp_path/'analysis.nc'),
                 'supplements':{'physical_analysis_surface_data':'old.nc'}}}
    src.global_surface_donor(row(), generated, tmp_path)
    assert calls[0]['area'] == [-90,-180,90,180]
    assert all(not f['pressure'] for f in calls[0]['fields'])
    assert [f['output'] for f in calls[0]['derived']] == ['Q2']
    assert generated['donor']['supplements']['physical_analysis_surface_data'].endswith('/global-donor/surface.nc')
    src.global_surface_donor(row(), generated, tmp_path)
    assert len(calls) == 1
    (tmp_path/'global-donor/surface.nc').write_bytes(b'changed')
    with pytest.raises(src.InitSourceRefusal, match='requested bytes'):
        src.global_surface_donor(row(), generated, tmp_path)


def test_generated_first_guess_commands_parse_with_the_real_init_door(tmp_path):
    table = src.load_table()
    parser = build_parser()
    for declared in table['sources']:
        if not declared.get('generation') or declared['status'] == 'refused':
            continue
        argv = src.init_argv(table, declared, met=tmp_path/'MET', grid=tmp_path/'grid',
                             static=tmp_path/'static', vertical_spec=tmp_path/'vertical',
                             out=tmp_path/'init.nc', cycle=src.Cycle.parse('2024-10-08T06'))
        args = parser.parse_args(argv)
        assert args.use_spechumd == 'yes', declared['id']
    invalid = row()
    invalid['first_guess'] = {**invalid['first_guess'], 'use_spechumd': 'true'}
    with pytest.raises(src.InitSourceRefusal, match='use_spechumd must be yes or no'):
        src.validate_row(invalid)
