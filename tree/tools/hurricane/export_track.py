"""Export collected model fixes and hourly storm-to-member distances.

No interpolation or future best track enters this exporter. Feed fixes.json to
the hurricane benchmark harness's ATCF export command. Failed tracker fixes and
noninteger forecast hours are retained in the audit but excluded from ATCF.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path


def distance_km(lat, lon, center):
    a, b = map(math.radians, (lat, center[0]))
    delta = math.radians(lon-center[1])
    h = math.sin((a-b)/2)**2 + math.cos(a)*math.cos(b)*math.sin(delta/2)**2
    return 12742 * math.asin(math.sqrt(min(1, max(0, h))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--pressure-evidence', type=Path, help='Unsmoothed intensity sidecar keyed by history hash and time')
    a = p.parse_args()
    pressure_rows = {} if a.pressure_evidence is None else {r['model_seconds']: r for r in json.loads(a.pressure_evidence.read_text())['records']}
    a.out.mkdir(parents=True, exist_ok=True)
    case = json.loads(a.case.read_text())
    initial = datetime.strptime(case['start_time'], '%Y-%m-%d_%H:%M:%S').replace(tzinfo=timezone.utc)
    # Match moving_case.interior_km: the transition annulus is not counted as
    # the guaranteed fine interior used by the placement policy.
    radius = float(case['mesh']['fine_radius_km']) - float(case['mesh'].get('fine_transition_km', 0))
    rows, fixes, audit, seen = [], [], [], set()
    for path in sorted(a.evidence.glob('s*-t*/receipt.json')):
        raw = path.read_bytes()
        record = json.loads(raw)
        frame = record['frame']
        seconds = float(frame['model_seconds'])
        if seconds in seen:
            raise ValueError(f'duplicate model time: {seconds}')
        seen.add(seconds)
        fix = frame.get('fix', {})
        valid = (initial+timedelta(seconds=seconds)).strftime('%Y-%m-%dT%H:%M:%SZ')
        accepted = fix.get('ok', False) and all(fix.get(k) is not None and math.isfinite(fix[k]) for k in ('lat', 'lon'))
        distance = distance_km(fix['lat'], fix['lon'], frame['member_center']) if accepted else None
        rows.append(dict(model_seconds=seconds, valid=valid, member=frame['member'],
                         center_distance_km=distance, fine_radius_km=radius,
                         inside_fine_region=None if distance is None else distance <= radius,
                         tracker_ok=accepted))
        audit.append(dict(receipt=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                          model_seconds=seconds, tracker_ok=accepted,
                          atcf_eligible=accepted and seconds % 3600 == 0,
                          reason=fix.get('reason')))
        if accepted and seconds % 3600 == 0:
            wind, pressure = fix.get('vmax10_ms'), None
            intensity = pressure_rows.get(seconds)
            if intensity is not None:
                if intensity['history_sha256'] != record['history_sha256']:
                    raise ValueError(f'pressure evidence does not match history: {path}')
                pressure = intensity.get('mslp_hpa')
            fixes.append(dict(valid=valid, lat=fix['lat'], lon=fix['lon'],
                vmax_kt=wind*1.9438444924406 if wind is not None and math.isfinite(wind) else None,
                mslp_hpa=pressure if pressure is not None and math.isfinite(pressure) else None))
    rows.sort(key=lambda r:r['model_seconds'])
    fixes.sort(key=lambda r:r['valid'])
    with (a.out / 'centering.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model_seconds', 'valid', 'member', 'center_distance_km',
                                                   'fine_radius_km', 'inside_fine_region', 'tracker_ok'])
        writer.writeheader()
        writer.writerows(rows)
    (a.out / 'fixes.json').write_text(json.dumps(dict(fixes=fixes), indent=2)+'\n')
    collection_path = a.evidence / 'collection.json'
    collection = json.loads(collection_path.read_text()) if collection_path.exists() else None
    complete = bool(collection and not collection.get('error') and collection['producer']['returncode'] == 0
                    and not collection['producer'].get('reason'))
    receipt = dict(complete_producer=complete, records=audit, frames=len(rows), hourly_fixes=len(fixes),
        maximum_center_distance_km=max((r['center_distance_km'] for r in rows if r['tracker_ok']), default=None),
        outside_fine_region=sum(r['inside_fine_region'] is False for r in rows),
        missing_tracker=sum(not r['tracker_ok'] for r in rows),
        pressure_note='Smoothed tracker pressure is excluded; intensity requires separately hash-bound native evidence.',
        note='Fine-region test is distance to the configured circular interior; mesh geometry is shown separately.')
    (a.out / 'export.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return 0 if complete and rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
