"""CPU tests for the predictive storm tracker and placement policy (synthetic vortices only)."""
import math

import numpy as np
import pytest

from hexcore import storm_tracker as st


def cloud_about(lat0, lon0, n=60000, radius_km=1500.0, seed=1):
    rng = np.random.default_rng(seed)
    r = radius_km * np.sqrt(rng.random(n))
    a = 2 * np.pi * rng.random(n)
    lat, lon = st.unproject(lat0, lon0, r * np.sin(a), r * np.cos(a))
    return lat, lon


def vortex_state(lat, lon, centre, *, depth_hpa=40.0, rmw_km=40.0, vmax=45.0, background=(-5.0, 2.0),
                 hemisphere=1.0, terrain=None):
    x, y = st.project(centre[0], centre[1], lat, lon)
    r = np.hypot(x, y)
    mslp = (1010.0 - depth_hpa * np.exp(-(r / 120.0) ** 1.5)) * 100.0
    vt = np.where(r < rmw_km, vmax * r / rmw_km, vmax * (rmw_km / np.maximum(r, 1e-6)) ** 0.6)
    # Counter-clockwise in the northern hemisphere: velocity = vt * (-y, x)/r.
    ux = -hemisphere * vt * y / np.maximum(r, 1e-6)
    vy = hemisphere * vt * x / np.maximum(r, 1e-6)
    u_p, v_p = {}, {}
    for level in (85000.0, 70000.0, 50000.0, 30000.0, 20000.0):
        decay = {85000.0: 1.0, 70000.0: 0.9, 50000.0: 0.6, 30000.0: 0.3, 20000.0: 0.1}[level]
        u_p[level] = background[0] + decay * ux
        v_p[level] = background[1] + decay * vy
    return st.StateFields(lat=lat, lon=lon, mslp_pa=mslp, u_p=u_p, v_p=v_p,
                          wind10=np.hypot(0.8 * ux, 0.8 * vy))


def test_projection_round_trip():
    lat, lon = st.unproject(15.0, 130.0, np.array([0.0, 300.0, -250.0]), np.array([0.0, 100.0, -400.0]))
    x, y = st.project(15.0, 130.0, lat, lon)
    assert np.allclose(x, [0, 300, -250], atol=1e-6) and np.allclose(y, [0, 100, -400], atol=1e-6)
    assert st.distance_km(15.0, 130.0, lat[1], lon[1]) == pytest.approx(math.hypot(300, 100), rel=1e-9)


def test_pressure_interpolation():
    p = np.array([[100000.0, 90000.0, 80000.0, 70000.0]])
    f = np.array([[0.0, 10.0, 20.0, 30.0]])
    out = st.interpolate_to_pressure(p, f, 85000.0)
    w = (math.log(85000) - math.log(90000)) / (math.log(80000) - math.log(90000))
    assert out[0] == pytest.approx(10 + 10 * w)
    assert np.isnan(st.interpolate_to_pressure(p, f, 105000.0)[0])


@pytest.mark.parametrize('hemisphere,centre', [(1.0, (12.3, 127.6)), (-1.0, (-16.2, 60.4))])
def test_tc_fix_finds_offset_vortex(hemisphere, centre):
    guess = (centre[0] + 0.6, centre[1] - 0.7)  # ~100 km off
    lat, lon = cloud_about(*guess)
    fields = vortex_state(lat, lon, centre, hemisphere=hemisphere)
    fix = st.TCFixer().fix(0.0, fields, guess, 150.0)
    assert fix.ok, fix.reason
    assert st.distance_km(fix.lat, fix.lon, *centre) < 8.0
    assert set(fix.components) >= {'mslp', 'vort850'}
    assert fix.depression_hpa > 20
    # Steering = background flow: the disc mean of an axisymmetric vortex vanishes.
    assert fix.steering_ms == pytest.approx([-5.0, 2.0], abs=0.6)


def test_tc_fix_rejects_weak_or_far():
    centre = (14.0, 125.0)
    lat, lon = cloud_about(*centre)
    weak = vortex_state(lat, lon, centre, depth_hpa=0.2, vmax=1.0)
    fix = st.TCFixer().fix(0.0, weak, centre, 150.0)
    assert not fix.ok
    far = vortex_state(lat, lon, st.destination(*centre, 90.0, 400.0))
    fix = st.TCFixer().fix(0.0, far, centre, 150.0)
    assert not fix.ok


def test_object_fixer_picks_dominant_near_object():
    centre = (35.0, -95.0)
    lat, lon = cloud_about(*centre, radius_km=600)
    x, y = st.project(*centre, lat, lon)
    near = 50 * np.exp(-((x - 40) ** 2 + (y + 20) ** 2) / 30 ** 2)
    far = 55 * np.exp(-((x + 300) ** 2 + y ** 2) / 30 ** 2)
    fields = st.StateFields(lat=lat, lon=lon, object_field=near + far)
    fix = st.ObjectFixer().fix(0.0, fields, centre, 150.0)
    assert fix.ok and st.distance_km(fix.lat, fix.lon, *st.unproject(*centre, 40.0, -20.0)) < 10


def moving_fixes(start, velocity_kmh, hours, every_h=1.0, noise_km=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t in np.arange(0, hours + 1e-9, every_h):
        lat, lon = st.unproject(start[0], start[1], velocity_kmh[0] * t + rng.normal(0, noise_km),
                                velocity_kmh[1] * t + rng.normal(0, noise_km))
        out.append(st.Fix(seconds=t * 3600, ok=True, lat=float(lat), lon=float(lon),
                          steering_ms=[velocity_kmh[0] / 3.6, velocity_kmh[1] / 3.6]))
    return out


def test_motion_fit_recovers_velocity():
    fixes = moving_fixes((12.0, 130.0), (-20.0, 6.0), 6.0, noise_km=3.0)
    v, alpha = st.MotionModel().velocity(fixes)
    assert alpha == 1.0 and v == pytest.approx([-20.0, 6.0], abs=2.5)
    first = st.MotionModel().velocity(fixes[:1])[0]
    assert first == pytest.approx([-20.0, 6.0], abs=1e-6)  # steering only


def policy(**kw):
    return st.PlacementPolicy(st.PlacementConfig(interior_radius_km=396.0, **kw))


def test_policy_stays_then_moves_ahead_of_storm():
    p = policy()
    fixes = moving_fixes((12.0, 130.0), (-20.0, 0.0), 3.0)
    region = (12.0, 130.0)
    d = p.decide(3 * 3600, fixes, region)
    assert not d.move and d.dwell_minutes >= 30
    # Storm already 220 km west of the region centre: must move, and the new centre leads the storm.
    fixes = moving_fixes(st.destination(12.0, 130.0, 270.0, 160.0), (-20.0, 0.0), 3.0)
    d = p.decide(3 * 3600, fixes, region)
    assert d.move
    storm = (fixes[-1].lat, fixes[-1].lon)
    x, _ = st.project(*storm, *d.centre)
    assert x < 0, 'new centre is placed ahead (west) of a westward-moving storm'
    assert d.dwell_minutes >= 120
    assert st.distance_km(*d.centre, *storm) + 25 <= 396 - 150


def test_policy_holds_on_miss_and_freezes_when_lost():
    p = policy(max_misses=2)
    fixes = moving_fixes((12.0, 130.0), (-20.0, 0.0), 2.0) + [st.Fix(seconds=3 * 3600, ok=False)]
    d = p.decide(3 * 3600, fixes, (12.0, 131.5), misses=1)
    assert not d.move and d.status == 'holding'
    d = p.decide(5 * 3600, fixes, (12.0, 131.5), misses=3)
    assert not d.move and d.status == 'lost'


def test_session_end_to_end_synthetic_track():
    """Closed loop on synthetic model states: the storm is followed for 48 h and never leaves the safe core."""
    start = (12.0, 132.0)
    vel = (-22.0, 5.0)
    cfg = st.TrackerConfig(mode='tc', initial_center_deg=list(start),
                           placement=st.PlacementConfig(interior_radius_km=396.0, check_minutes=60.0))
    s = st.TrackSession(cfg)
    region = list(start)
    moves, worst = 0, 0.0
    lat, lon = None, None
    t = 0.0
    while t <= 48 * 3600:
        truth = st.unproject(*start, vel[0] * t / 3600, vel[1] * t / 3600)
        truth = (float(truth[0]), float(truth[1]))
        if lat is None or st.distance_km(truth[0], truth[1], lat0, lon0) > 400:
            lat0, lon0 = truth
            lat, lon = cloud_about(lat0, lon0, n=30000, radius_km=1300, seed=int(t))
        fields = vortex_state(lat, lon, truth, background=(vel[0] / 3.6, vel[1] / 3.6))
        fix = s.observe(t, fields)
        assert fix.ok, (t, fix.reason)
        worst = max(worst, float(st.distance_km(*region, *truth)))
        d = s.decide(t, region)
        if d.move:
            moves += 1
            region = d.centre
            worst = max(worst, float(st.distance_km(*region, *truth)))
        t = d.next_check_seconds if d.next_check_seconds > t else t + 3600
        t = math.ceil(t / 3600) * 3600
    assert worst <= 396 - 150 + 1e-6
    assert 2 <= moves <= 8
    restored = st.TrackSession.restore(s.to_json())
    assert len(restored.fixes) == len(s.fixes) and restored.decisions[-1].centre == s.decisions[-1].centre


def test_policy_max_move_limits_step():
    p = policy(max_move_km=120.0)
    fixes = moving_fixes(st.destination(12.0, 130.0, 270.0, 200.0), (-20.0, 0.0), 3.0)
    d = p.decide(3 * 3600, fixes, (12.0, 130.0))
    assert d.move and st.distance_km(12.0, 130.0, *d.centre) <= 120.0 + 1e-6


def test_anticipate_looks_past_the_check_cap():
    p = policy(max_dwell_minutes=120.0)
    fixes = moving_fixes((12.0, 130.0), (-20.0, 0.0), 3.0)
    region = (12.0, 130.0)
    d = p.decide(3 * 3600, fixes, region)
    assert not d.move and d.next_check_seconds == 3 * 3600 + 120 * 60
    a = p.anticipate(d, fixes, region)
    assert a['move_in_minutes'] > 120 and a['at_seconds'] > d.next_check_seconds
    storm_then = p.motion.predict(fixes, a['at_seconds'] - fixes[-1].seconds)
    x, _ = st.project(*storm_then, *a['centre'])
    assert x < 0  # placed ahead of the storm's predicted position at the move time


def test_policy_max_move_smaller_than_lattice_overshoot():
    """Storm 194 km from the region; an 80 km cap must still produce a move of <= 80 km."""
    p = policy(max_move_km=80.0, check_minutes=60.0)
    fixes = moving_fixes(st.destination(10.5, 128.8, 271.0, 150.0), (-21.7, 4.7), 11.0)
    d = p.decide(11 * 3600, fixes, (10.5, 128.8))
    assert d.move and st.distance_km(10.5, 128.8, *d.centre) <= 80.0 + 1e-6


def test_continuous_follow_makes_small_frequent_moves():
    p = policy(max_move_km=80.0, min_move_km=40.0, follow='continuous', check_minutes=60.0)
    region = [12.0, 130.0]
    start = (12.0, 130.0)
    moves = []
    for hour in range(1, 25):
        fixes = [st.Fix(seconds=h * 3600.0, ok=True, lat=float(a), lon=float(b), steering_ms=[-6.0, 0.0])
                 for h in range(hour + 1) for a, b in [st.unproject(*start, -22.0 * h, 0.0)]]
        d = p.decide(hour * 3600.0, fixes, region)
        storm = (fixes[-1].lat, fixes[-1].lon)
        if d.move:
            moves.append(st.distance_km(*region, *d.centre))
            region = d.centre
        assert st.distance_km(*region, *storm) <= 246.0
    assert len(moves) >= 4 and max(moves) <= 80.0 + 1e-6


def test_continuous_anticipate_predicts_next_step():
    p = policy(max_move_km=120.0, min_move_km=60.0, follow='continuous', check_minutes=60.0)
    fixes = moving_fixes((12.0, 130.0), (-22.0, 0.0), 2.0)
    d = p.decide(2 * 3600, fixes, (12.0, 129.75))
    a = p.anticipate(d, fixes, d.centre)
    assert a is not None and 60.0 <= st.distance_km(*d.centre, *a['centre']) <= 120.0 + 1e-6
    assert a['at_seconds'] > d.seconds


def test_continuous_uses_prepared_member_and_chain():
    p = policy(max_move_km=120.0, min_move_km=60.0, follow='continuous', check_minutes=60.0)
    fixes = moving_fixes((12.0, 130.0), (-22.0, 0.0), 3.0)
    region = (12.0, 130.25)
    d0 = p.decide(3 * 3600, fixes, region)
    assert d0.move
    near = st.destination(*d0.centre, 0.0, 25.0)
    d1 = p.decide(3 * 3600, fixes, region, available=[[float(near[0]), float(near[1])]])
    assert d1.move and d1.used_prepared
    chain = p.anticipate_chain(d0, fixes, d0.centre, 2)
    assert len(chain) == 2 and chain[1]['at_seconds'] > chain[0]['at_seconds']


def test_continuous_never_alternates_between_prepared_members():
    """Regression (live12-c2): two prepared members 61 km apart were alternated every check."""
    p = policy(max_move_km=120.0, min_move_km=60.0, follow='continuous', check_minutes=60.0)
    a, b = [10.25, 128.25], [10.0, 127.75]
    region = list(b)
    start = (10.37, 128.9)
    moves = 0
    for hour in range(3, 7):
        fixes = [st.Fix(seconds=h * 3600.0, ok=True, lat=float(x), lon=float(y), steering_ms=[-5.0, -0.3])
                 for h in range(hour + 1) for x, y in [st.unproject(*start, -19.0 * h, -1.0 * h)]]
        d = p.decide(hour * 3600.0, fixes, region, available=[a, b])
        if d.move:
            moves += 1
            assert st.distance_km(*d.centre, *a) > 1 or st.distance_km(*region, *a) > 1
            region = d.centre
    assert moves <= 1


def test_policy_avoids_excluded_centre():
    p = policy()
    fixes = moving_fixes(st.destination(12.0, 130.0, 270.0, 160.0), (-20.0, 0.0), 3.0)
    d = p.decide(3 * 3600, fixes, (12.0, 130.0))
    assert d.move
    d2 = p.decide(3 * 3600, fixes, (12.0, 130.0), excluded=[d.centre])
    assert d2.move and st.distance_km(*d2.centre, *d.centre) >= 1.0 and d2.dwell_minutes >= 60
