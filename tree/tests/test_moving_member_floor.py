"""A 1 km case's 863 m edges must not fail the former 48/3 km cutoff."""
import json
import time
from types import SimpleNamespace

import netCDF4
import pytest

from hexcore import moving_run


@pytest.mark.parametrize('edge', [863.4994506835938, 216.0])
def test_start_and_completed_runtime_member_share_admission(tmp_path, edge):
    static = tmp_path / 'member.static.nc'
    with netCDF4.Dataset(static, 'w') as d:
        d.createDimension('nEdges', 2)
        d.createVariable('dcEdge', 'f4', ('nEdges',))[:] = [edge, edge * 2]
    for name in ('start', 'next'):
        (tmp_path / (name + '-case.json')).write_text(json.dumps(
            dict(name=name, static=str(static), center_deg=[13, 128])))
    controller = object.__new__(moving_run.Controller)
    controller.members_dir = tmp_path
    controller.state = dict(current='start', members={
        'start': controller.read_member('start', 'ready'),
        'next': dict(status='preparing', requested_wall=time.time(), why='tracker move')})
    controller.procs = {'prep:next': SimpleNamespace(poll=lambda: 0)}
    controller.ensure_operator = lambda name: True
    controller.save = lambda: None
    events = []
    controller.event = lambda kind, **data: events.append(kind)
    controller.poll_background()
    assert not controller.procs
    assert events == ['member-prep-done']
    for member in controller.state['members'].values():
        assert member['status'] == 'ready'
        assert member['min_dc_edge_m'] == pytest.approx(edge)
    assert controller.ready_members() == {'next': [13, 128]}


def test_failed_preparation_is_still_refused(tmp_path):
    controller = object.__new__(moving_run.Controller)
    controller.members_dir = tmp_path
    controller.state = dict(current='start', members={
        'start': dict(status='ready'),
        'next': dict(status='preparing', requested_wall=time.time())})
    controller.procs = {'prep:next': SimpleNamespace(poll=lambda: 1)}
    controller.save = lambda: None
    controller.event = lambda *args, **kwargs: None
    controller.poll_background()
    assert controller.state['members']['next']['status'] == 'failed'


def test_recovery_can_select_an_admitted_fine_member(tmp_path):
    init = tmp_path / 'member.init.nc'
    init.touch()
    operator = tmp_path / 'operator.npz'
    operator.touch()
    controller = object.__new__(moving_run.Controller)
    controller.state = dict(current='bad', t=300, moves=[], members={
        'bad': dict(status='ready', center_deg=[13, 128], grid='source.grid.nc'),
        'fine': dict(status='ready', center_deg=[13.05, 127.85],
                     init=str(init), min_dc_edge_m=863.4994506835938)})
    controller.a = SimpleNamespace(zone_carry=False)
    controller.stitch = False
    controller.carrier_member = lambda: 'bad'
    controller.forget_after = lambda t: 0
    controller.operator_path = lambda source, target: operator
    controller.event = lambda *args, **kwargs: None
    controller.session = SimpleNamespace(
        fixes=[SimpleNamespace(ok=True, lat=13.05, lon=127.85, seconds=300)],
        motion=SimpleNamespace(velocity=lambda fixes: ([0, 0], None)),
        policy=SimpleNamespace(dwell=lambda *args: 60,
                               c=SimpleNamespace(max_lead_minutes=900)))
    assert controller.fallback({}) == 'move'
    assert controller.state['current'] == 'fine'
    assert controller.state['members']['bad']['status'] == 'unstable'
    assert len(controller.state['moves']) == 1


def _tracking_controller():
    from hexcore import storm_tracker as st
    controller = object.__new__(moving_run.Controller)
    controller.st = st
    controller.duration = 10800
    controller.stitch = False
    controller.a = SimpleNamespace(anticipate_moves=1, zone_carry=False, anticipate_hours=3)
    controller.session = st.TrackSession(st.TrackerConfig(initial_center_deg=[0, 0], placement=st.PlacementConfig(
        interior_radius_km=81, core_margin_km=25, min_move_km=30,
        follow='continuous', check_minutes=5, snap_deg=0.05,
        max_move_km=60, continuous_lead_minutes=120, prepared_tolerance_km=5)))
    controller.state = dict(current='start', members={})
    return controller


def test_anticipation_uses_last_two_fixes_not_steering_or_unexecuted_decision():
    c = _tracking_controller()
    st = c.st
    # Eastward steering contradicts the model's latest westward displacement.
    c.session.fixes = [st.Fix(seconds=0, ok=True, lat=0, lon=0, steering_ms=[20, 0])]
    decision = st.Decision(0, True, [0, 0.5], 300, 0, 'unexecuted eastward move')
    assert c.anticipate_chain(decision, {'center_deg': [0, 0]}) == []
    c.session.fixes += [st.Fix(seconds=300, ok=True, lat=0, lon=-0.02, steering_ms=[20, 0])]
    decision.seconds = 300
    velocity = c.fix_velocity()
    assert velocity[0] < -20 and abs(velocity[1]) < 0.01
    chain = c.anticipate_chain(decision, {'center_deg': [0, 0]})
    assert chain and chain[0]['centre'][1] < 0
    # A newer northward fix replaces that direction, rather than fitting all history.
    c.session.fixes += [st.Fix(seconds=600, ok=True, lat=0.02, lon=-0.02, steering_ms=[20, 0])]
    velocity = c.fix_velocity()
    assert abs(velocity[0]) < 0.01 and velocity[1] > 20


def test_refresh_keeps_nearer_work_and_bounds_preparation_to_two(tmp_path, monkeypatch):
    c = _tracking_controller()
    c.members_dir = tmp_path
    c.a = SimpleNamespace(zone_carry=False, max_preparing=2, reuse_km=40,
                          case='case.json', casedir=str(tmp_path), prep_threads=48,
                          prep_tile_workers=48, geog=None)
    c.state['members'] = {
        'start': dict(status='ready', center_deg=[0, 0]),
        'near': dict(status='preparing', center_deg=[0, -0.2], order=1, requested_wall=1)}
    near = SimpleNamespace(poll=lambda: None)
    c.procs = {'prep:near': near}
    stopped, events, launched = [], [], []
    c.stop_child = stopped.append
    c.event = lambda event, **kw: events.append((event, kw))
    c.save = lambda: None
    def spawn(command, stdout):
        stdout.close()
        proc = SimpleNamespace(poll=lambda: None)
        launched.append(proc)
        return proc
    monkeypatch.setattr(moving_run, 'start_child', spawn)
    # Small snapped changes reuse work; a changed direction fills the free slot
    # without throwing away a partially built member that might still serve.
    assert c.request_member([0, -0.3], 'nearby revised target') == 'near'
    far = c.request_member([0, 1], 'changed direction')
    assert len(launched) == 1 and not stopped
    assert c.state['members']['near']['status'] == 'preparing'
    # Speculation never evicts either preparation. A next-move refresh replaces
    # only the furthest one, even when both were requested at order 1.
    assert c.request_member([0, -0.6], 'later move', order=2) is None
    fresh = c.request_member([0, -0.6], 'next move refresh')
    assert stopped == [launched[0]]
    assert c.state['members'][far]['status'] == 'cancelled'
    assert c.state['members']['near']['status'] == 'preparing'
    assert c.state['members'][fresh]['status'] == 'preparing'
    assert len(c.procs) == 2 and len(launched) == 2
    assert [e[0] for e in events].count('member-prep-cancelled') == 1
    assert c.request_member([0, -0.65], 'next check') == fresh
    assert len(launched) == 2


def test_due_move_stops_on_safe_ready_member_without_new_preparation(tmp_path):
    c = _tracking_controller()
    fix = c.st.Fix(seconds=5700, ok=True, lat=12.95, lon=128.30)
    decision = c.st.Decision(5700, True, [13.10, 127.85], 6000, 0, 'move')
    c.state.update(members={'start': dict(status='ready', center_deg=[13.3, 128.3])},
                   frames=[], segments=[{}], pending_move=None)
    c.interior_km = 81
    c.observe = lambda *args: (fix, 0.01)
    c.ready_members = lambda: {'anticipated': [13.05, 127.85]}
    c.session.decide = lambda *args, **kwargs: decision
    c.save = lambda: None
    c.log = lambda *args: None
    c.request_member = lambda *args, **kwargs: pytest.fail('safe ready member must serve immediately')
    control = tmp_path / 'control'
    control.mkdir()
    verdict = c.on_frame(5700, tmp_path / 'history.nc', control)
    assert verdict == dict(stop=True, target='anticipated', reason='move')
    assert c.state['pending_move']['target'] == 'anticipated'
    assert json.loads((control / 'stop-request.json').read_text()) == dict(
        stop_at_seconds=5700, target='anticipated')


def test_ready_member_admission_uses_current_fix_and_core_margin():
    c = _tracking_controller()
    fix = c.st.Fix(seconds=5700, ok=True, lat=12.95, lon=128.30)
    decision = c.st.Decision(5700, True, [13.10, 127.85], 6000, 0, 'move')
    current = {'center_deg': [13.3, 128.3]}
    ready = {'anticipated': [13.05, 127.85], 'nearer': [13.0, 128.0], 'unsafe': [14, 128]}
    assert c.pick_target(decision, current, 5700, ready, fix) == 'nearer'
    assert c.pick_target(decision, current, 5700, {'anticipated': ready['anticipated']}, fix) == 'anticipated'
    # Even an exact match to the decision must satisfy current-fix containment.
    decision.centre = ready['unsafe']
    assert c.pick_target(decision, current, 5700, {'unsafe': ready['unsafe']}, fix) is None
    fix.ok = False
    assert c.pick_target(decision, current, 5700, ready, fix) is None


def test_preparation_defaults_use_all_available_cores_and_propagate(monkeypatch):
    from hexcore import moving_door, moving_members
    from hexcore.cli import build_parser
    monkeypatch.setattr(moving_door.os, 'sched_getaffinity', lambda _: set(range(48)))
    parser = build_parser()
    run = parser.parse_args(['moving', 'run', '--case', 'c', '--casedir', 'd', '--out', 'r', '--hours', '3'])
    prep = parser.parse_args(['moving', 'prepare', '--case', 'c', '--casedir', 'd'])
    assert (run.prep_threads, run.prep_tile_workers, prep.threads, prep.tile_workers) == (48, 48, 48, 48)
    assert run.max_preparing == 2
    with pytest.raises(SystemExit):
        parser.parse_args(['moving', 'run', '--case', 'c', '--casedir', 'd', '--out', 'r',
                           '--hours', '3', '--max-preparing', '3'])
    env = moving_members._tool_env(run.prep_threads, run.prep_tile_workers)
    assert [env[k] for k in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'RW_MPAS_TILE_WORKERS')] == ['48'] * 3
    limited = parser.parse_args(['moving', 'run', '--case', 'c', '--casedir', 'd', '--out', 'r', '--hours', '3',
                                '--prep-threads', '8', '--prep-tile-workers', '4'])
    assert (limited.prep_threads, limited.prep_tile_workers) == (8, 4)


@pytest.mark.parametrize('completed', [True, False])
def test_due_move_holds_until_preparation_ready_or_stops(tmp_path, monkeypatch, completed):
    c = _tracking_controller()
    fix = c.st.Fix(seconds=7200, ok=True, lat=0, lon=0.3)
    decision = c.st.Decision(7200, True, [0, 0.5], 9000, 0, 'move due')
    c.state.update(members={'start': dict(status='ready', center_deg=[0, 0]),
                            'next': dict(status='preparing', center_deg=[0, 0.5])},
                   frames=[], segments=[{}], pending_move=None)
    c.a.hold_for_member_seconds = 1800
    c.interior_km = 82
    c.observe = lambda *args: (fix, 0.01)
    c.ready_members = lambda: ({'next': [0, 0.5]}
        if c.state['members']['next']['status'] == 'ready' else {})
    c.session.decide = lambda *args, **kwargs: decision
    c.anticipate_chain = lambda *args: [{'centre': [0, 0.5]}]
    c.save = lambda: None
    c.log = lambda *args: None
    c.event = lambda *args, **kwargs: None
    c.request_member = lambda *args, **kwargs: 'next'
    c.poll_background = lambda: None
    waits = []
    def wait(seconds):
        waits.append(seconds)
        c.state['members']['next']['status'] = 'ready' if completed else 'failed'
    monkeypatch.setattr(moving_run.time, 'sleep', wait)
    control = tmp_path / 'control'; control.mkdir()
    (control / 'await-verdict.json').write_text('{}')
    if completed:
        assert c.on_frame(7200, tmp_path / 'history.nc', control)['stop'] is True
        assert c.state['pending_move']['target'] == 'next'
    else:
        with pytest.raises(RuntimeError, match='continuing would let the storm leave'):
            c.on_frame(7200, tmp_path / 'history.nc', control)
        assert not (control / 'verdict-00007200.json').exists()
    assert waits == [2.0]
