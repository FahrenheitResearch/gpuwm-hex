"""Hold the first model fix, and release owned children before the card grant."""
from types import SimpleNamespace
import pytest
from hexcore import moving_run

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
