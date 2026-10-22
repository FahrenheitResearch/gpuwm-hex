"""Hold the first model fix, and release owned children before the card grant."""
from types import SimpleNamespace
import pytest
from hexcore import moving_run

@pytest.mark.parametrize('offsets,passed', [([9.9, 86.469241315], False),
                                        ([9.9, None], False), ([9.9, 40], True)])
def test_completed_forecast_requires_every_fix_inside_fine_interior(offsets, passed):
    state = dict(t=21600, hours=6, wall_seconds=1500, moves=[],
                 fixes_source='run-time tracker on model state',
                 frames=[dict(offset_km=offset, fix=dict(ok=True)) for offset in offsets],
                 segments=[dict(rc=0, integration_seconds=1000, wall_seconds=1400)])
    summary = moving_run.summarize(state, 82)
    assert (summary['status'] == 'passed') == passed
    assert summary['a2_storm_inside_interior_every_frame'] == passed

@pytest.mark.parametrize('error',[KeyboardInterrupt,RuntimeError,SystemExit])
def test_failed_controller_stops_all_owned_child_groups(monkeypatch,error):
    prep,forecast=object(),object()
    def run():raise error('interrupted')
    controller=SimpleNamespace(run=run,procs={'prep:x':prep},forecast_children=[forecast])
    monkeypatch.setattr(moving_run,'Controller',lambda _:controller)
    stopped=[]
    monkeypatch.setattr(moving_run,'stop_process_tree',stopped.append)
    with pytest.raises(error):moving_run.run_controller(None)
    assert stopped==[prep,forecast]


def test_initial_fix_can_hold_before_any_member_is_preparing(tmp_path):
    control=tmp_path/'control';control.mkdir()
    (control/'progress.jsonl').write_text('{"model_seconds":0,"history":"initial.nc"}\n')
    observed=[]
    def on_frame(*args):observed.append((control/'await-verdict.json').exists())
    fake=SimpleNamespace(poll_background=lambda:None,ready_members=lambda:{},
        a=SimpleNamespace(hold_for_member_seconds=10,verdict_timeout=1,await_within_minutes=5,segment_timeout=60),
        state={'frames':[],'members':{'start':{'status':'ready'}},'current':'start'},on_frame=on_frame)
    proc=SimpleNamespace(poll=lambda:0,returncode=0)
    assert moving_run.Controller.monitor(fake,proc,tmp_path,control)==0
    assert observed==[True]


@pytest.mark.parametrize('hold', [0, 10])
def test_initial_hold_exists_before_forecast_can_publish(monkeypatch, tmp_path, hold):
    member = {'name': 'start', 'center_deg': [0, 0]}
    controller = SimpleNamespace(
        out=tmp_path, duration=60, case={}, init_source='fixture', forecast_children=[],
        state={'segments': [], 'members': {'start': member}, 'current': 'start',
               't': 0, 'boundary': None, 'operator': None, 'source_grid': None},
        a=SimpleNamespace(hold_for_member_seconds=hold, verdict_timeout=1,
                          check_minutes=1, engine='arwen', forecast_arg=[]),
        segment_env=lambda *args: {}, save=lambda: None, log=lambda message: None)
    proc = SimpleNamespace()

    def spawn(command, *, stdout, env, cwd):
        assert (tmp_path / 'segment-000/control/await-verdict.json').exists() == bool(hold)
        stdout.close()
        return proc

    monkeypatch.setattr(moving_run, 'segment_command', lambda *args, **kwargs: ['forecast'])
    monkeypatch.setattr(moving_run, 'start_child', spawn)
    moving_run.Controller.launch(controller, kind='start')
    assert controller.forecast_children == [proc]


def _anticipation_controller():
    from hexcore import storm_tracker as st
    controller = object.__new__(moving_run.Controller)
    controller.st = st
    controller.duration = 1800
    controller.stitch = False
    controller.a = SimpleNamespace(anticipate_moves=2, zone_carry=False, anticipate_hours=24)
    controller.session = st.TrackSession(st.TrackerConfig(initial_center_deg=[0, 0], placement=st.PlacementConfig(
        interior_radius_km=81, core_margin_km=10, min_move_km=30,
        follow='continuous', check_minutes=5, snap_deg=0.05,
        max_move_km=60, continuous_lead_minutes=120)))
    controller.session.fixes = [st.Fix(seconds=0, ok=True, lat=0, lon=0),
                                st.Fix(seconds=300, ok=True, lat=0, lon=-0.02)]
    return controller


@pytest.mark.parametrize('anticipated_seconds', [1800, 2700])
def test_no_preparation_for_a_move_after_forecast_ends(anticipated_seconds):
    controller = _anticipation_controller()
    controller.session.policy.anticipate = lambda *a, **kw: {'at_seconds': anticipated_seconds}
    decision = controller.st.Decision(300, False, [0, 0], 600, 0, 'test')
    assert controller.anticipate_chain(decision, {'center_deg': [0, 0]}) == []


def test_anticipation_chain_stops_at_forecast_end():
    controller = _anticipation_controller()
    controller.session.policy.anticipate = lambda *a, **kw: {'at_seconds': 900, 'centre': [0, 0]}
    controller.session.policy.dwell = lambda *a: 20
    decision = controller.st.Decision(300, False, [0, 0], 600, 0, 'test')
    chain = controller.anticipate_chain(decision, {'center_deg': [0, 0]})
    assert [row['at_seconds'] for row in chain] == [900]


def test_zero_anticipated_moves_does_not_prepare_next_member():
    controller = SimpleNamespace(a=SimpleNamespace(anticipate_moves=0))
    assert moving_run.Controller.anticipate_chain(controller, None, None) == []


def test_decided_move_waits_for_current_centre_not_next_anticipated_move(tmp_path):
    from hexcore.storm_tracker import Fix, Decision
    controller = object.__new__(moving_run.Controller)
    current = [24.4, -83.0]
    chosen = [24.75, -83.0]  # emergency move39km is below the usual41km minimum
    future = [25.25, -83.0]  # ready but outside the fix's safe core radius
    fix = Fix(seconds=7200, ok=True, lat=24.733039782250614, lon=-83.00713393179386)
    decision = Decision(7200, True, chosen, 10800, 60, 'current region unsafe')
    controller.state = dict(current='start', members={
        'start': dict(status='ready', center_deg=current),
        'future': dict(status='ready', center_deg=future)}, frames=[], segments=[{}])
    controller.interior_km = 82
    controller.session = SimpleNamespace(
        policy=SimpleNamespace(c=SimpleNamespace(safe_km=lambda: 82-27.333333333333332)),
        decide=lambda *args, **kwargs: decision)
    controller.a = SimpleNamespace(hold_for_member_seconds=1800)
    controller.observe = lambda *args: (fix, 0.1)
    controller.anticipate_chain = lambda *args: [dict(centre=future, at_seconds=9000)]
    requested = []
    def request(centre, why):
        requested.append(centre)
        controller.state['members']['chosen'] = dict(status='prepared', center_deg=centre)
        return 'chosen'
    controller.request_member = request
    def poll():
        if 'chosen' in controller.state['members']:
            controller.state['members']['chosen']['status'] = 'ready'
    controller.poll_background = poll
    controller.event = lambda *args, **kwargs: None
    controller.save = lambda: None
    controller.log = lambda message: None
    (tmp_path/'await-verdict.json').write_text('{}')
    verdict = controller.on_frame(7200, 'history.nc', tmp_path)
    assert requested == [chosen]
    assert verdict == dict(stop=True, target='chosen', reason=decision.reason)
    assert controller.state['pending_move']['target'] == 'chosen'


def test_resume_redraws_products_that_died_with_the_controller(tmp_path):
    def segment(name, products_tail, rc=0):
        root = tmp_path / name
        fc = root / 'fc'
        (fc / 'products.products-scratch' / 'render-global').mkdir(parents=True)
        (fc / 'products.log').write_text('PRODUCTS view=global window=global\n' + products_tail)
        (root / 'forecast.log').write_text(
            'x\nPRODUCTS $ gpuwm-hex products --forecast ' + str(fc) + ' --out ' + str(fc / 'products') + '\n')
        return dict(path=str(root), rc=rc)
    killed = segment('segment-000', 'DOOR rendered=84 skipped=2 failed=0 out=x\n')
    finished = segment('segment-001', 'PRODUCTS done maps=307 point_pictures=1 failed_views=none manifest=m\n')
    failed = segment('segment-002', '', rc=2)
    started, logged = [], []

    def popen(argv, **kw):
        started.append((argv, kw['env']['CUDA_VISIBLE_DEVICES']))
        return SimpleNamespace(pid=7)
    redrawn = moving_run.restart_dead_products([killed, finished, failed], logged.append,
                                               popen=popen, alive=lambda fc: False)
    fc = tmp_path / 'segment-000' / 'fc'
    assert redrawn == [str(fc)]
    assert started == [([moving_run.sys.executable, '-m', 'hexcore.cli', 'products', '--forecast', str(fc),
                         '--out', str(fc / 'products')], '')]
    assert not (fc / 'products.products-scratch').exists()
    assert (tmp_path / 'segment-001' / 'fc' / 'products.products-scratch').exists()
    assert moving_run.restart_dead_products([killed], logged.append, popen=popen,
                                            alive=lambda fc: True) == []
