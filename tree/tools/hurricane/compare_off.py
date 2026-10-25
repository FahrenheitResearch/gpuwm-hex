"""Compare every recorded history variable of completed OFF and base forecasts.

Hashes come from collect_moving.py before raw-history retirement. Matching
subsets, failed forecasts and missing output times never establish identity.
"""
import argparse
import json
from pathlib import Path


def read(root):
    collection = json.loads((root / 'collection.json').read_text())
    if collection.get('error') or collection['producer']['returncode'] or collection['producer'].get('reason'):
        raise ValueError(f'forecast or collection did not complete: {root}')
    rows = {}
    for path in sorted(root.glob('s*-t*/receipt.json')):
        row = json.loads(path.read_text())
        t = float(row['frame']['model_seconds'])
        if t in rows:
            raise ValueError(f'duplicate output time {t}: {root}')
        rows[t] = row['fields']
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--off', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--cadence-minutes', type=float, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    duration, cadence = a.hours * 3600, a.cadence_minutes * 60
    if duration <= 0 or cadence <= 0 or duration % cadence:
        p.error('positive cadence must divide the whole forecast duration')
    expected = {i * cadence for i in range(int(duration / cadence) + 1)}
    result = dict(status='unverified', off=str(a.off), base=str(a.base),
                  expected_frames=len(expected), includes_initial_frame=True,
                  scope='bitwise identity of every published history variable, including coordinate fields')
    try:
        off, base = read(a.off), read(a.base)
        result.update(off_frames=len(off), base_frames=len(base),
                      off_missing_times=sorted(expected - off.keys()),
                      base_missing_times=sorted(expected - base.keys()))
        if set(off) != expected or set(base) != expected:
            raise ValueError('output times do not exactly cover the requested whole case')
        differences, compared = [], 0
        for t in sorted(expected):
            for name in sorted(off[t].keys() | base[t].keys()):
                left, right = off[t].get(name), base[t].get(name)
                keys = ('shape', 'dtype', 'sha256')
                if left is None or right is None or any(left[k] != right[k] for k in keys):
                    differences.append(dict(model_seconds=t, variable=name))
                compared += 1
        result.update(status='different' if differences else 'bitwise-identical',
                      compared_variables=compared, differences=differences)
    except (ValueError, KeyError, OSError) as error:
        result['reason'] = str(error)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['status'] == 'bitwise-identical' else 1


if __name__ == '__main__':
    raise SystemExit(main())
