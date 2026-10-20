"""A2 check for any moving run (CPU only): is the model storm inside the fine interior at every output time?

usage: a2_check.py RUNDIR --case CASE.json [--out A2.json] [--interior-radius-km R]

RUNDIR holds segment-*/ directories, each with result.json or command.json naming its region centre
(selected_center_deg, or center_deg) and an fc/ folder of history files. The tracker (the same predictive
tracker used at run time, seeded only with the analysis-time position) is re-run over every history in time
order. Each output is checked against the region of the segment that wrote it; an output at a move time is
also checked against the region that takes over. Pass: every fix succeeds and every offset <= the interior
radius (cap radius minus its ramp, from the case mesh block). The safe radius (interior minus the policy's
core margin) is reported alongside.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hexcore import storm_tracker as st  # noqa: E402
from track_history import HistoryReader, history_time  # noqa: E402


def segment_centre(folder: Path):
    for name in ('result.json', 'command.json'):
        f = folder / name
        if f.is_file():
            d = json.loads(f.read_text())
            c = d.get('selected_center_deg') or d.get('center_deg')
            if c:
                return [float(c[0]), float(c[1])], d
    return None, {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--interior-radius-km', type=float)
    a = p.parse_args(argv)
    case = json.loads(a.case.read_text())
    tcfg = dict(case.get('tracker', {}))
    tcfg.setdefault('initial_center_deg', case['storm']['initial_center_deg'])
    place = dict(tcfg.pop('placement', {}) or {})
    interior = a.interior_radius_km or st.interior_radius_from_mesh(case['mesh'])
    safe = interior - float(place.get('core_margin_km', st.PlacementConfig(interior_radius_km=interior).core_margin_km))
    session = st.TrackSession(st.TrackerConfig.from_dict(tcfg))
    start = datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    outputs = []  # (seconds, path, centre, segment)
    segments = sorted(a.run.glob('segment-*'))
    centres = []
    for seg in segments:
        centre, info = segment_centre(seg)
        if centre is None:
            continue
        centres.append((float(info.get('begin_seconds', 0.0)), centre, seg.name))
        for h in (seg / 'fc').glob('cuda-history.*.nc'):
            outputs.append(((history_time(h) - start).total_seconds(), h, centre, seg.name))
    outputs.sort(key=lambda r: r[0])
    reader = HistoryReader(object_field=(session.config.object.field_name if session.config.mode == 'object' else None))
    rows, seen = [], set()
    for seconds, path, centre, seg in outputs:
        if seconds in seen:
            continue
        seen.add(seconds)
        guess, _ = session.predicted(seconds)
        fix = session.observe(seconds, reader(path, guess))
        check = [dict(segment=seg, centre=centre)]
        taking_over = [c for c in centres if abs(c[0] - seconds) < 1 and c[1] != centre]
        check += [dict(segment=c[2], centre=c[1], at_move=True) for c in taking_over]
        for c in check:
            c['offset_km'] = None if not fix.ok else float(st.distance_km(*c['centre'], fix.lat, fix.lon))
        rows.append(dict(seconds=seconds, lead_h=seconds / 3600, fix_ok=fix.ok, fix=[fix.lat, fix.lon],
                         mslp_min_hpa=fix.mslp_min_hpa, reason=fix.reason, regions=check))
        worst = max((c['offset_km'] for c in check if c['offset_km'] is not None), default=float('nan'))
        print(f'+{seconds / 3600:6.2f} h {"ok " if fix.ok else "MISS"} offset {worst:6.1f} km  ({seg})', flush=True)
    offsets = [c['offset_km'] for r in rows for c in r['regions'] if c['offset_km'] is not None]
    result = dict(run=str(a.run), interior_radius_km=interior, safe_radius_km=safe, outputs=len(rows),
                  fixes_ok=sum(r['fix_ok'] for r in rows), max_offset_km=max(offsets) if offsets else None,
                  a2_pass=bool(rows) and all(r['fix_ok'] for r in rows) and max(offsets) <= interior,
                  core_inside_safe_every_output=bool(offsets) and max(offsets) <= safe, rows=rows,
                  tracker='hexcore.storm_tracker (model state only; seeded with the analysis-time position)')
    print(f'A2 {"PASS" if result["a2_pass"] else "FAIL"}: {result["fixes_ok"]}/{len(rows)} fixes, max offset '
          f'{result["max_offset_km"]:.1f} km (interior {interior:.0f}, safe {safe:.0f})')
    if a.out:
        a.out.write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
