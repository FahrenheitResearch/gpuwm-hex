"""Storm-following admission: a due move never reuses a member its own admission rule refuses, and a
case with no safe radius is refused when it is loaded, before any member preparation."""
import json
from types import SimpleNamespace

import pytest

from hexcore import moving_run
from hexcore import storm_tracker as st


def _controller(tmp_path, monkeypatch):
    controller = object.__new__(moving_run.Controller)
    controller.st = st
    controller.duration = 1800
    controller.stitch = False
    controller.session = st.TrackSession(st.TrackerConfig(initial_center_deg=[0, 0], placement=st.PlacementConfig(
        interior_radius_km=81, core_margin_km=10, min_move_km=30,
        follow='continuous', check_minutes=5, snap_deg=0.05,
        max_move_km=60, continuous_lead_minutes=120)))
    controller.a = SimpleNamespace(zone_carry=False, reuse_km=25, max_preparing=1,
                                   case=tmp_path / 'case.json', casedir=tmp_path, prep_threads=2,
                                   prep_tile_workers=2, geog=None, rotate_template=False)
    controller.members_dir = tmp_path
    controller.state = dict(current='start', members={}, frames=[], t_seen=0)
    controller.procs = {}
    controller.save = lambda: None
    controller.event = lambda *a, **k: None
    launched = []

    def spawn(argv, *, stdout):
        stdout.close()
        launched.append(argv)
        return SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(moving_run, 'start_child', spawn)
    # An earlier member of this case on disk, ~17 km from the requested centre (inside --reuse-km 25).
    row = dict(name='neighbour', center_deg=[0.0, 0.15])
    for key in ('grid', 'static', 'init'):
        path = tmp_path / f'neighbour.{key}.nc'
        path.write_bytes(b'prepared earlier')
        row[key] = str(path)
    (tmp_path / 'neighbour-case.json').write_text(json.dumps(row))
    return controller, launched


@pytest.mark.parametrize('admit,reused', [(None, True), (lambda centre: True, True), (lambda centre: False, False)])
def test_a_due_move_reuses_an_on_disk_member_only_when_admission_takes_it(tmp_path, monkeypatch, admit, reused):
    """hex-day2 storm-following runs died twice on 2026-10-27: the move reused an on-disk member off the
    policy's centre, pick_target refused it once READY (its core was not safe there for one check), and the
    door stopped with 'moving member ... is ready after holding the forecast'."""
    controller, launched = _controller(tmp_path, monkeypatch)
    name = controller.request_member([0.0, 0.0], 'move decided, no ready member', admit=admit)
    if reused:
        assert name == 'neighbour' and launched == []
    else:
        assert name not in (None, 'neighbour')
        assert len(launched) == 1 and launched[0][launched[0].index('--name') + 1] == name


def test_the_admission_rule_is_pick_targets(tmp_path, monkeypatch):
    controller, _ = _controller(tmp_path, monkeypatch)
    fix = st.Fix(seconds=0, ok=True, lat=0.0, lon=0.0)
    controller.session.fixes = [fix]
    near, far = [0.0, 0.05], [0.0, 0.6]       # ~6 km and ~67 km from the core (safe radius 71 km)
    assert controller.admits(near, 0, fix)
    assert not controller.admits(far, 0, fix)
    controller.state['members'] = {'start': dict(status='ready', center_deg=[0, 0])}
    ready = {'near': near, 'far': far}
    assert controller.pick_target(None, None, 0, ready, fix) == 'near'
    assert controller.pick_target(None, None, 0, {'far': far}, fix) is None


def test_a_case_with_no_safe_radius_is_refused_at_load():
    """TC-Milton case 2026-10-26: a 146 km interior under the default 150 km core margin was accepted and
    failed only after its first member preparation."""
    with pytest.raises(ValueError, match='no safe radius'):
        st.PlacementConfig(interior_radius_km=146.0)
    with pytest.raises(ValueError, match='core_margin_km'):
        st.TrackerConfig.from_dict({'initial_center_deg': [0, 0], 'placement': {'interior_radius_km': 146.0}})
    assert st.PlacementConfig(interior_radius_km=146.0, core_margin_km=48.0).safe_km() == 98.0
