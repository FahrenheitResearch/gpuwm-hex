"""Summarize every Rust-compared forecast field without processing weather data.

MPAS mpas_atmphys_driver_seaice.F:160-558 defines the lane's driver boundary.
Missing/incompatible samples never become a zero-word numerical comparison.
"""
import argparse
import json
from pathlib import Path


def summarize(receipt):
    streams = {}
    for row in receipt.get('comparison', {}).get('fields', []):
        stream = streams.setdefault(row['stream'], dict(
            numerical_compared_words=0, numerical_differing_words=0,
            missing_samples=0, missing_or_incompatible_words=0,
            first_computed_difference=None, fields={}))
        field = stream['fields'].setdefault(row['field'], dict(
            numerical_compared_words=0, numerical_differing_words=0,
            missing_samples=0, missing_or_incompatible_words=0,
            first_computed_step=None, statuses=[]))
        if row['status'] == 'compared':
            for key, source in (('numerical_compared_words', 'compared_words'),
                                ('numerical_differing_words', 'differing_words')):
                stream[key] += row[source]
                field[key] += row[source]
            if row['differing_words']:
                step = field['first_computed_step']
                if step is None or row['step'] < step:
                    field['first_computed_step'] = row['step']
                first = stream['first_computed_difference']
                if first is None or row['step'] < first['step']:
                    stream['first_computed_difference'] = dict(
                        field=row['field'], step=row['step'],
                        differing_words=row['differing_words'])
        else:
            for record in (stream, field):
                record['missing_samples'] += 1
                record['missing_or_incompatible_words'] += row['differing_words']
        if row['status'] not in field['statuses']:
            field['statuses'].append(row['status'])
    for stream in streams.values():
        for record in (stream, *stream['fields'].values()):
            if not record['numerical_compared_words']:
                record['numerical_differing_words'] = None
    return dict(kind='every-field forecast comparison metadata; no replacement identity grade',
                deck=receipt['deck'], process_count=receipt['process_count'],
                model_execution=receipt.get('model_execution'),
                evidence_kind=receipt['evidence_kind'],
                passed=receipt['passed'], acceptance_tier=receipt.get('acceptance_tier'),
                failed_model_exit=receipt.get('failed_model_exit'), streams=streams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.receipt.read_text()))
    result['full_receipt'] = str(args.receipt)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    for name, record in result['streams'].items():
        print(name, json.dumps({k: v for k, v in record.items() if k != 'fields'}))
