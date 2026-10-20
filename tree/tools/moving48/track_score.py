"""Score storm tracks against a best track (verification only; never read at run time).

usage: track_score.py --best BEST.json --start-time YYYY-mm-dd_HH:MM:SS --track LABEL=TRACK.json [...]
                      [--leads 0,12,24,36,48] --out SCORE.json [--markdown SCORE.md]

TRACK.json: output of track_history.py (rows with seconds and fix) or a live run's tracker session
(fixes with seconds/lat/lon/ok). BEST.json: `gpuwm-hex moving stage --best-track` output (IBTrACS rows).
Best-track positions are interpolated linearly in time between records (records are 3- or 6-hourly).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from hexcore import storm_tracker as st


def best_rows(path: Path, start: datetime):
    data = json.loads(path.read_text())
    rows = []
    for r in data['rows']:
        t = datetime.strptime(r['time'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
        if r.get('lat') is None:
            continue
        pres = r.get('wmo_pres_hpa') or r.get('tokyo_pres_hpa') or r.get('usa_pres_hpa')
        rows.append(((t - start).total_seconds(), r['lat'], r['lon'], pres))
    return sorted(rows), data.get('name'), data.get('sid')


def best_at(rows, seconds):
    t = np.array([r[0] for r in rows])
    if not t[0] <= seconds <= t[-1]:
        return None
    i = min(np.searchsorted(t, seconds, side='right') - 1, len(t) - 2)
    f = (seconds - t[i]) / (t[i + 1] - t[i])
    a, b = rows[i], rows[i + 1]
    lon_b = a[2] + ((b[2] - a[2] + 180) % 360 - 180)
    pres = None if a[3] is None or b[3] is None else a[3] + f * (b[3] - a[3])
    return a[1] + f * (b[1] - a[1]), st.wrap_lon(a[2] + f * (lon_b - a[2])), pres


def track_fixes(path: Path):
    data = json.loads(path.read_text())
    if 'track' in data:
        return [(r['seconds'], r['fix']) for r in data['track']]
    if 'fixes' in data:
        return [(f['seconds'], f) for f in data['fixes']]
    raise ValueError(f'{path}: neither a track_history output nor a tracker session')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best', type=Path, required=True)
    p.add_argument('--start-time', required=True)
    p.add_argument('--track', action='append', required=True, metavar='LABEL=PATH')
    p.add_argument('--leads', default='0,12,24,36,48')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--markdown', type=Path)
    a = p.parse_args(argv)
    start = datetime.strptime(a.start_time, '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    best, name, sid = best_rows(a.best, start)
    leads = [float(x) for x in a.leads.split(',')]
    result = dict(best_track=str(a.best), sid=sid, start_time=a.start_time, leads_h=leads, tracks={})
    for item in a.track:
        label, path = item.split('=', 1)
        fixes = {round(s): f for s, f in track_fixes(Path(path))}
        rows, all_errors = [], []
        for s, f in sorted(fixes.items()):
            b = best_at(best, s)
            if b is None or not f.get('ok'):
                continue
            all_errors.append(float(st.distance_km(f['lat'], f['lon'], b[0], b[1])))
        for lead in leads:
            s = round(lead * 3600)
            f, b = fixes.get(s), best_at(best, s)
            row = dict(lead_h=lead, available=bool(f and f.get('ok') and b))
            if row['available']:
                row.update(lat=f['lat'], lon=f['lon'], best_lat=b[0], best_lon=b[1],
                           error_km=float(st.distance_km(f['lat'], f['lon'], b[0], b[1])),
                           min_mslp_hpa=f.get('mslp_min_hpa'), best_pres_hpa=b[2],
                           pressure_error_hpa=(None if f.get('mslp_min_hpa') is None or b[2] is None
                                               else f['mslp_min_hpa'] - b[2]))
            elif f is not None and not f.get('ok'):
                row['reason'] = f.get('reason')
            rows.append(row)
        scored = [r['error_km'] for r in rows if r['available'] and r['lead_h'] > 0]
        result['tracks'][label] = dict(path=path, rows=rows, mean_error_km_leads_gt0=(float(np.mean(scored)) if scored else None),
                                       mean_error_km_all_outputs=(float(np.mean(all_errors)) if all_errors else None),
                                       n_outputs=len(all_errors))
    a.out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    lines = [f'Track error (km) against IBTrACS {sid} ({name}), start {a.start_time}', '',
             '| track | ' + ' | '.join(f'{l:g} h' for l in leads) + ' | mean (leads > 0) | mean (all outputs) |',
             '|---|' + '---:|' * (len(leads) + 2)]
    for label, t in result['tracks'].items():
        cells = [f"{r['error_km']:.0f}" if r['available'] else '-' for r in t['rows']]
        m = t['mean_error_km_leads_gt0']; ma = t['mean_error_km_all_outputs']
        lines.append(f'| {label} | ' + ' | '.join(cells) + f" | {'-' if m is None else f'{m:.0f}'} | "
                     f"{'-' if ma is None else f'{ma:.0f}'} (n={t['n_outputs']}) |")
    text = '\n'.join(lines) + '\n'
    print(text)
    if a.markdown:
        a.markdown.write_text(text)


if __name__ == '__main__':
    main()
