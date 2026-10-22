"""Moving cases from track metadata. Runtime following uses model state only."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from .moving_case import MovingCaseError, great_circle_km, load_case, resolve_vertical_spec


def timestamp(value):
    try:
        result = dt.datetime.fromisoformat(str(value).replace('_', 'T').replace('Z', '+00:00'))
        if result.tzinfo is None:
            result = result.replace(tzinfo=dt.timezone.utc)
        return result.astimezone(dt.timezone.utc)
    except (ValueError, TypeError):
        raise MovingCaseError(f'invalid UTC track time: {value!r}') from None


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MovingCaseError(f'{label} must be a finite number') from None
    if not math.isfinite(result):
        raise MovingCaseError(f'{label} must be a finite number')
    return result


def read_track(path, *, storm=None, best_track=False):
    """Read tc-tracks tracks/fixes, moving best-track rows, or IBTrACS CSV."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == '.csv':
        if not best_track or not storm:
            raise MovingCaseError('IBTrACS CSV requires --best-track and --storm SID')
        from .moving_case import best_track_from_ibtracs
        document = best_track_from_ibtracs(path, storm)
    else:
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise MovingCaseError(f'cannot read track {path}: {error}') from None
    if not isinstance(document, dict):
        raise MovingCaseError('track JSON must be an object containing tracks, fixes or rows')
    if 'tracks' in document:
        candidates = document['tracks']
        if not isinstance(candidates, list) or any(not isinstance(t, dict) for t in candidates):
            raise MovingCaseError('tracks must be an array of track objects')
        if storm is not None:
            candidates = [t for t in candidates if str(t.get('track_id')) == storm]
        if len(candidates) != 1:
            raise MovingCaseError('select exactly one track with --storm TRACK_ID; '
                                  f'found {len(candidates)} matching tracks')
        selected = candidates[0]
        identifier = selected.get('track_id')
        rows = selected.get('fixes', [])
    else:
        identifier = document.get('sid', document.get('track_id'))
        if storm and identifier and str(identifier) != storm:
            raise MovingCaseError(f'track identifier {identifier!r} does not match --storm {storm!r}')
        rows = document.get('rows', document.get('fixes'))
        if rows is None and ('time' in document or 'valid' in document):
            rows = [document]
    if not isinstance(rows, list) or not rows:
        raise MovingCaseError('selected track has no rows or fixes')
    fixes = {}
    for index, row in enumerate(rows):
        if row is None:
            continue
        if not isinstance(row, dict):
            raise MovingCaseError(f'row {index} must be a track-fix object or null')
        if row.get('lat') is None or row.get('lon') is None:
            continue
        time = timestamp(row.get('valid', row.get('time')))
        lat = _number(row['lat'], f'row {index} latitude')
        lon = _number(row['lon'], f'row {index} longitude')
        if abs(lat) > 90 or not -180 <= lon <= 360:
            raise MovingCaseError(f'row {index} coordinates outside latitude/longitude bounds')
        point = [lat, (lon + 180) % 360 - 180]
        if time in fixes and fixes[time] != point:
            raise MovingCaseError(f'conflicting positions at {time.isoformat()}')
        fixes[time] = point
    if not fixes:
        raise MovingCaseError('selected track contains no valid positions')
    return document, identifier, sorted(fixes.items()), path


def author_case(path, *, track, cycle, storm=None, best_track=False, template=None,
                fine_km=3.0, background_km=None, fine_radius_km=200.0,
                hours=12.0, source='gfs', dt_seconds=None, physics_args=None):
    """Seed from an exact analysis position and earlier fixes, never future observations."""
    from .init_sources import load_table, resolve_source
    source_row = resolve_source(load_table(), source)
    source = source_row['id']
    _, identifier, fixes, track_path = read_track(track, storm=storm, best_track=best_track)
    analysis = timestamp(cycle)
    if analysis.minute or analysis.second or analysis.microsecond:
        raise MovingCaseError('--cycle must be a whole UTC analysis hour')
    at_analysis = [p for time, p in fixes if time == analysis]
    if not at_analysis:
        raise MovingCaseError('track has no valid position exactly at --cycle; '
                              'an interpolated future best-track position cannot seed an analysis')
    center = at_analysis[0]
    past = [(time, p) for time, p in fixes if dt.timedelta(0) < analysis-time <= dt.timedelta(hours=6)]
    heading, speed = 0.0, 0.0
    motion_times = []
    if past:
        previous_time, previous = past[-1]
        elapsed = (analysis-previous_time).total_seconds()/3600
        speed = great_circle_km(previous, center)/elapsed
        lat1, lat2 = map(math.radians, (previous[0], center[0]))
        delta = math.radians((center[1]-previous[1]+180)%360-180)
        heading = math.degrees(math.atan2(math.sin(delta)*math.cos(lat2),
            math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(delta)))%360
        motion_times = [previous_time.isoformat(), analysis.isoformat()]
    if template:
        case = load_case(template)
        vertical = resolve_vertical_spec(case)
        packaged = Path(__file__).with_name('data') / 'countries' / vertical.name
        case['vertical_spec'] = vertical.name if packaged.is_file() and packaged.read_bytes() == vertical.read_bytes() else str(vertical)
        case.pop('_path', None)
        case['tracker'] = dict(case.get('tracker') or {})
        case['tracker'].pop('initial_center_deg', None)
        case.pop('gfs_cycle', None)
    else:
        fine = _number(fine_km, '--fine-km')
        background = (_number(background_km, '--background-km') if background_km is not None
                      else fine * 2**max(1, math.ceil(math.log2(48/fine))) if fine > 0 else 48)
        radius = _number(fine_radius_km, '--fine-radius-km')
        if not 0 < fine < background or radius <= 18*fine:
            raise MovingCaseError('require 0 < fine spacing < background and fine radius > 18 fine cells')
        ratio = math.log2(background/fine)
        if not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise MovingCaseError('background/fine spacing must be a power of two for the mesh generator')
        rings = [dict(spacing_km=fine*2**i, extra_radius_km=54*fine*(2**i-1),
                      transition_km=18*fine*2**i) for i in range(1, round(ratio))]
        case = dict(n_levels=55, vertical_spec='tc55-v1.json', dt_seconds=6.25*fine,
                    mesh=dict(background_km=background, fine_km=fine, fine_radius_km=radius,
                              fine_transition_km=18*fine, rings=rings),
                    tracker=dict(mode='tc', placement=dict(
                        core_margin_km=min(50.0, (radius-18*fine)/3), check_minutes=30.0,
                        min_move_km=min(60.0, (radius-18*fine)/2), max_move_km=2*(radius-18*fine),
                        max_dwell_minutes=180.0, follow='continuous')), physics_args=[])
    if dt_seconds is not None:
        case['dt_seconds'] = _number(dt_seconds, '--dt-seconds')
    step = _number(case['dt_seconds'], 'dt_seconds')
    duration = _number(hours, '--hours')
    check_minutes = float((case.get('tracker', {}).get('placement') or {}).get('check_minutes', 60))
    macro = 8*step
    for label, seconds in [('radiation interval', 1800), ('tracker interval', 60*check_minutes),
                           ('duration', 3600*duration)]:
        if step <= 0 or seconds <= 0 or not math.isclose(seconds/macro, round(seconds/macro), abs_tol=1e-8):
            raise MovingCaseError(f'{label} must be a positive whole multiple of 8 * dt_seconds ({macro:g} s)')
    if not math.isclose(duration*60/check_minutes, round(duration*60/check_minutes), abs_tol=1e-8):
        raise MovingCaseError('--hours must contain a whole number of tracker intervals')
    if physics_args is not None:
        case['physics_args'] = list(physics_args)
    case.update(case=Path(path).stem, description='Track-authored moving fine region; model-state following',
                start_time=analysis.strftime('%Y-%m-%d_%H:%M:%S'), hours=duration, init_source=source)
    case['storm'] = dict(initial_center_deg=center,
        initial_center_source='selected track position at the analysis time',
        initial_motion=dict(heading_deg=heading, speed_kmh=speed,
                            source='positions at and before analysis' if past else 'no prior fix within 6 h; stationary seed'))
    if best_track and identifier:
        case['storm']['ibtracs_sid'] = identifier
    case['authoring'] = dict(schema='gpuwm-hex.moving-author/v1', track=str(track_path),
        track_sha256=hashlib.sha256(track_path.read_bytes()).hexdigest(), track_id=identifier,
        kind='best-track analysis seed' if best_track else 'tracks analysis seed',
        analysis_utc=analysis.isoformat(), motion_times_utc=motion_times,
        future_positions_used=False, runtime_fixes='model state',
        template=str(Path(template).resolve()) if template else None)
    path = Path(path).expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(case, indent=2, allow_nan=False)+'\n')
    return case
