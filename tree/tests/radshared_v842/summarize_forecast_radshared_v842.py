"""Summarize actual verifier metadata without counting absent words as errors.

The Rust verifier owns field reads and word comparison. This helper aggregates
its original receipt and retains every stream/field, incompatible status and
first computed difference. It cannot change acceptance or the ratchet.
"""
import argparse
import hashlib
import json
from pathlib import Path


COUNTERS = ('compared_records', 'compared_words', 'differing_words',
            'missing_records', 'missing_words', 'incompatible_records')


def summarize(path):
    data = json.loads(path.read_text())
    groups, steps, incompatible, first = {}, {}, [], None
    for row in sorted(data['comparison']['fields'],
                      key=lambda item: (item['step'], item['stream'], item['field'])):
        key = row['stream'], row['field']
        group = groups.setdefault(key, dict(stream=key[0], field=key[1],
            **dict.fromkeys(COUNTERS, 0), first_difference=None, max_ulp=None))
        step = steps.setdefault(row['step'], dict(step=row['step'], time=row['time'],
                                                  **dict.fromkeys(COUNTERS, 0)))
        status = row['status']
        if status == 'compared':
            counts = dict(compared_records=1, compared_words=row['compared_words'],
                          differing_words=row['differing_words'])
            if row['differing_words']:
                if first is None:
                    first = row
                if group['first_difference'] is None:
                    group['first_difference'] = {name: row[name] for name in
                        ('step', 'time', 'differing_words', 'first_differing_index')}
            if row.get('max_ulp') is not None:
                group['max_ulp'] = max(group['max_ulp'] or 0, row['max_ulp'])
        elif status == 'field missing in hex':
            counts = dict(missing_records=1, missing_words=row['compared_words'])
        else:
            counts = dict(incompatible_records=1)
            incompatible.append(row)
        for name, count in counts.items():
            group[name] += count
            step[name] += count
    streams = []
    for name in sorted({key[0] for key in groups}):
        rows = [group for (stream, _), group in groups.items() if stream == name]
        streams.append(dict(stream=name, **{counter: sum(row[counter] for row in rows)
                                           for counter in COUNTERS}))
    totals = {counter: sum(stream[counter] for stream in streams) for counter in COUNTERS}
    return dict(schema='p21-whole-proof-field-summary-v2', source_receipt=str(path.resolve()),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_execution=data['model_execution'], evidence_kind=data['evidence_kind'],
        failed_model_exit=data.get('failed_model_exit'), passed=data['passed'],
        process_count=data['process_count'], deck=data['deck'],
        totals=totals, numeric_difference_defined=bool(totals['compared_records']),
        first_computed_difference=first, streams=streams, fields=list(groups.values()),
        steps=list(steps.values()), incompatible_records=incompatible,
        missing_words_are_not_numerical_differences=True, ratchet_applied=False,
        acceptance_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.receipt)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
        ('passed', 'model_execution', 'evidence_kind', 'process_count', 'totals',
         'first_computed_difference')}, indent=2))


if __name__ == '__main__':
    main()
