"""Summarize authentic verify-mpas receipts without treating absent words as diffs.

This is JSON receipt orchestration, not weather-field decoding. The underlying
Rust comparator receipts remain the evidence for every field and timestep.
"""
import argparse
import hashlib
import json
from pathlib import Path


def summary(rows):
    computed = [row for row in rows if row['status'] == 'compared']
    return dict(samples=len(rows), computed_samples=len(computed),
                actually_compared_words=sum(row['compared_words'] for row in computed),
                numerical_differing_words=(sum(row['differing_words'] for row in computed)
                                           if computed else None),
                missing_or_incompatible_samples=len(rows)-len(computed),
                missing_or_incompatible_words=sum(row.get('compared_words', 0)
                                                 for row in rows if row['status'] != 'compared'),
                first_computed_differing_step=next((row['step'] for row in computed
                                                   if row['differing_words']), None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipts', nargs='+', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in args.receipts:
        receipt = json.loads(path.read_text())
        rows = receipt.get('comparison', {}).get('fields', [])
        fields = {}
        streams = {}
        for row in rows:
            fields.setdefault((row['stream'], row['field']), []).append(row)
            streams.setdefault(row['stream'], []).append(row)
        results.append(dict(receipt=str(path.resolve()),
                            receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            deck=receipt['deck'], process_count=receipt['process_count'],
                            evidence_kind=receipt['evidence_kind'], passed=receipt['passed'],
                            acceptance_tier=receipt.get('acceptance_tier'),
                            model_execution=receipt.get('model_execution', False),
                            failed_model_exit=receipt.get('failed_model_exit'),
                            model_failure=(receipt.get('failures') or [None])[0],
                            total=summary(rows),
                            streams={stream: summary(group) for stream, group in streams.items()},
                            fields=[dict(stream=stream, field=field,
                                         steps=[row['step'] for row in group],
                                         statuses=sorted({row['status'] for row in group}),
                                         **summary(group))
                                    for (stream, field), group in fields.items()]))
    args.out.write_text(json.dumps(dict(schema='d2-les-whole-field-accounting-v1',
                                       numerical_diff_requires_computed_words=True,
                                       forecasts=results), indent=2)+'\n')
    for result in results:
        print(result['deck'], json.dumps(result['total']))


if __name__ == '__main__':
    main()
