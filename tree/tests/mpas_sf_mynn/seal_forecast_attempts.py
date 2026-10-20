"""Bind all eight actual verifier receipts, preserving structural failures.

Only reads JSON metadata emitted by the Rust comparator and real CLI. It
does not decode weather fields or replace forecast/gate acceptance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(index_path, attempts):
    index = json.loads(index_path.read_text())
    wanted = set(itertools.product(('x1.2562', 'x1.10242'),
                                  ('2023-03-31', '2022-12-22'),
                                  ('single_sf_mynn', 'convection_permitting')))
    rows = index['decks']
    if (len(rows) != 8 or {(r['mesh'], r['date'], r['suite']) for r in rows} != wanted
            or index['backends'] != {'surface_layer': 'hex-cuda',
                                    'all_other_active_slots': 'native-cpu'}
            or index['surface_layer_scheme'] != 'sf_mynn'):
        raise ValueError('index does not bind the complete MYNN forecast matrix')
    decks = {}
    for row in rows:
        path = Path(row['path'])
        deck = json.loads(path.read_text())
        if (digest(path) != row['sha256'] or deck['process_count'] != 1
                or row['ranks'] != 1 or row['hours'] != 24 or not row['all_streams']):
            raise ValueError(f'changed or incomplete native deck: {path}')
        decks[deck['id']] = (row, deck)
    results, seen = [], set()
    for path in attempts:
        receipt = json.loads(path.read_text())
        name = receipt['deck']
        if name in seen or name not in decks:
            raise ValueError(f'duplicate or unindexed forecast: {path}')
        row, deck = decks[name]
        if (receipt['process_count'] != 1
                or receipt['provenance']['deck_sha256'] != row['sha256']
                or receipt['pin_set'] != deck['pin_set']):
            raise ValueError(f'forecast does not bind its original native deck: {path}')
        if receipt['evidence_kind'] not in ('FORECAST', 'FORECAST-FAILED'):
            raise ValueError(f'comparison-only evidence cannot seal an attempted forecast: {path}')
        fields = receipt['comparison']['fields']
        streams, first = {}, None
        for field in sorted(fields, key=lambda f: (f['step'], f['stream'], f['field'])):
            stream = streams.setdefault(field['stream'], {})
            counts = stream.setdefault(field['field'], dict(
                samples=0, compared_words=0, computed_differing_words=0,
                missing_words=0, structural_statuses=[]))
            counts['samples'] += 1
            if field['status'] == 'compared':
                counts['compared_words'] += field['compared_words']
                counts['computed_differing_words'] += field['differing_words']
                if field['differing_words'] and first is None:
                    first = dict(field)
            else:
                if 'missing' in field['status']:
                    counts['missing_words'] += field['compared_words']
                if field['status'] not in counts['structural_statuses']:
                    counts['structural_statuses'].append(field['status'])
        passed = bool(fields) and all(
            f['status'] == 'compared' and f['differing_words'] == 0 for f in fields)
        passed = bool(passed and receipt['passed'] and receipt['model_execution']
                      and receipt['evidence_kind'] == 'FORECAST'
                      and receipt['tier'] == 'BITWISE' and not receipt['ratchet_regressions'])
        results.append(dict(
            mesh=row['mesh'], date=row['date'], suite=row['suite'], process_count=1,
            receipt=str(path.resolve()), sha256=digest(path),
            evidence_kind=receipt['evidence_kind'], model_execution=receipt['model_execution'],
            failures=receipt['failures'], passed=passed, identity_tier=receipt['tier'],
            first_computed_difference=first, streams=streams))
        seen.add(name)
    if seen != set(decks):
        raise ValueError('a native deck lacks an actual forecast attempt receipt')
    return dict(schema='mynn-whole-forecast-matrix-v1',
                index=str(index_path.resolve()), index_sha256=digest(index_path),
                process_count=1, attempts=results,
                passed=all(r['passed'] for r in results),
                note='Missing/structural samples are separate from computed word differences; '
                     'this matrix does not certify G0-G2 or landing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.index, args.attempt)
    payload = json.dumps(result, indent=2) + '\n'
    if len(payload.encode()) >= 10_000_000:
        raise ValueError('matrix receipt exceeds delivery size limit')
    args.receipt.write_text(payload)
    print(json.dumps(dict(attempts=len(result['attempts']), passed=result['passed'])))


if __name__ == '__main__':
    main()
