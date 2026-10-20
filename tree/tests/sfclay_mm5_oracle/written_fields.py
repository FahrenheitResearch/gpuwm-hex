"""Check the integrator's recorded-omission rule without changing the ratchet.

The original Rust/verify-mpas receipt remains authoritative. A written-field
claim is distinct from complete native BITWISE identity. Baselines are unchanged
published foundation inventories, never omissions learned from the candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def assess(rows, baseline, stored):
    failures, seen = [], set()
    for row in rows:
        key = '|'.join(map(str, (row['stream'], row['field'], row['step'])))
        if key in seen:
            failures.append('duplicate comparison sample ' + key)
        seen.add(key)
        if row['status'] == 'compared':
            if row['differing_words'] != 0:
                failures.append('written words differ ' + key)
        elif (row['status'] != 'field missing in hex' or
              baseline.get(key) != dict(status=row['status'], words=row['compared_words'])):
            failures.append('new omission or structural failure ' + key)
    if not rows or not any(r['status'] == 'compared' and r['compared_words'] > 0 for r in rows):
        failures.append('no written native words compared')
    files = stored.get('files', [])
    if not files or stored.get('words', 0) <= 0 or stored.get('differing_words') != 0:
        failures.append('stored-word comparison absent or nonzero')
    for row in files:
        if row.get('error'):
            failures.append(row['file'] + ': ' + row['error'])
        stream = row['file'].split('.')[0]
        permitted = {key.split('|')[1] for key, value in baseline.items()
                     if key.split('|')[0] == stream and value['status'] == 'field missing in hex'}
        for error in row.get('errors', []):
            if (not error.startswith('variable set differs: ') or
                    error.removeprefix('variable set differs: ') not in permitted):
                failures.append(row['file'] + ': ' + error)
    return dict(scope='09:22Z ruling: every written word identical; unchanged recorded native-only inventory',
                passed=not failures, acceptance_tier='BITWISE-WRITTEN-FIELDS' if not failures else None,
                all_native_fields_bitwise=not failures and all(r['status'] == 'compared' for r in rows),
                computed_words=sum(r['compared_words'] for r in rows if r['status'] == 'compared'),
                computed_differing_words=sum(r['differing_words'] for r in rows if r['status'] == 'compared'),
                recorded_missing_samples=sum(r['status'] == 'field missing in hex' for r in rows),
                stored_words=stored.get('words'), stored_differing_words=stored.get('differing_words'),
                failures=failures, ratchet_written=False)


def compare_stored(root):
    metadata = json.loads((root / 'written-proof.json').read_text())
    binary = root / 'stored-word-comparator'
    if hashlib.sha256(binary.read_bytes()).hexdigest() != metadata['comparator_sha256']:
        raise ValueError('stored-word comparator differs from its published hash')
    rows = []
    native = root / 'native'
    output = root / 'hex-output'
    manifest = json.loads((native / 'MANIFEST.json').read_text())
    for candidate in sorted(output.glob('*.nc')):
        if candidate.name not in manifest['outputs']:
            rows.append(dict(file=candidate.name, error='written file absent from native manifest'))
            continue
        process = subprocess.run([str(binary), str(native / candidate.name), str(candidate)],
                                 capture_output=True, text=True)
        parsed = [json.loads(line) for line in process.stdout.splitlines() if line.startswith('{')]
        if len(parsed) != 1:
            rows.append(dict(file=candidate.name, error=process.stderr or 'no Rust stored-word result',
                             exit_code=process.returncode))
        else:
            rows.append(dict(**parsed[0], file=candidate.name, exit_code=process.returncode))
    return dict(schema='mpas-mm5.stored-forecast-words/v1', process_count=1, files=rows,
                words=sum(r.get('words', 0) for r in rows),
                differing_words=sum(r.get('differing_words', 0) for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare_stored(args.bundle), indent=2) + '\n')
