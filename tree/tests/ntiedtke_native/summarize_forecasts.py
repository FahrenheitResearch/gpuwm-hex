"""Seal full forecast attempts without counting absent fields as numeric differences."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    decks = json.loads(args.decks.read_text())
    expected = {row['key']: row for row in decks['decks']}
    if (decks.get('complete') is not True or decks.get('pending_deck_keys')
            or len(expected) != decks['requested_deck_count']):
        raise ValueError('the complete native-reference deck matrix is required')
    for row in expected.values():
        backends = row['backends']
        if (row['process_count'] != 1 or row['scheme'] != 'cu_ntiedtke'
                or backends.get('convection') != 'hex-cuda'
                or any(value != 'native-cpu' for slot, value in backends.items()
                       if slot != 'convection')):
            raise ValueError('this proof binds one process and only convection on CUDA')
    rows = {}
    for path in args.attempts:
        matrix = json.loads(path.read_text())
        if matrix.get('complete') is not True:
            raise ValueError(f'{path}: attempt matrix is incomplete')
        for row in matrix['runs']:
            if row['key'] in rows:
                raise ValueError('duplicate forecast attempt')
            rows[row['key']] = dict(row)
    if set(rows) != set(expected):
        raise ValueError('attempt keys differ from the full native-reference matrix')
    for key, row in rows.items():
        if row['deck_sha256'] != expected[key]['sha256']:
            raise ValueError(f'{key}: attempt used a different deck')
        path = Path(row['receipt'])
        result = json.loads(path.read_text())
        if result.get('process_count') != expected[key]['process_count']:
            raise ValueError(f'{key}: process count differs')
        fields = result.get('comparison', {}).get('fields', [])
        compared = [field for field in fields if field.get('status') == 'compared']
        streams = {}
        for field in fields:
            stream = streams.setdefault(field['stream'], dict(
                compared_field_samples=0, missing_field_samples=0,
                differing_words=None, missing_words=0))
            if field.get('status') == 'compared':
                stream['compared_field_samples'] += 1
                stream['differing_words'] = (stream['differing_words'] or 0) + field['differing_words']
            else:
                stream['missing_field_samples'] += 1
                stream['missing_words'] += field['compared_words']
        row.update(full_receipt_sha256=digest(path), streams=streams,
                   passed=result.get('passed') is True,
                   evidence_kind=result.get('evidence_kind'),
                   tier=result.get('tier'), compared_fields=len(compared),
                   unrepresented_fields=len(fields)-len(compared),
                   differing_words=(sum(field['differing_words'] for field in compared)
                                    if compared else None),
                   first_computed_differing_field=next(
                       (field for field in compared if field['differing_words']), None))
        command = row['argv']
        log = Path(command[command.index('--work-root')+1])/'hex.log'
        if log.is_file():
            row['hex_log_sha256'] = digest(log)
            row['actual_hex_failure'] = next(
                (line for line in reversed(log.read_text().splitlines())
                 if line.startswith('gpuwm-hex:')), None)
    ordered = [rows[key] for key in expected]
    compared_count = sum(row['compared_fields'] for row in ordered)
    identity = all(row['passed'] and row['tier'] == 'BITWISE'
                   and row['evidence_kind'] == 'FORECAST'
                   and row['compared_fields'] > 0
                   and row['unrepresented_fields'] == 0
                   and row['differing_words'] == 0 for row in ordered)
    summary = dict(schema='mpas-ntiedtke-whole-forecast-attempts/v2',
                   complete=True, requested_decks=len(ordered),
                   native_process_count=1, all_forecasts_passed=all(row['passed'] for row in ordered),
                   identity_proven=identity, compared_fields=compared_count,
                   differing_words=(sum(row['differing_words'] or 0 for row in ordered)
                                    if compared_count else None),
                   decks=dict(path=str(args.decks), sha256=digest(args.decks)),
                   runs=ordered)
    args.receipt.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'runs'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decks', type=Path, required=True)
    parser.add_argument('--attempts', type=Path, action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    main(parser.parse_args())
