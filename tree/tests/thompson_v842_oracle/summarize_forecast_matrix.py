"""Account for every row in actual Rust whole-forecast comparison receipts."""
import argparse
import hashlib
import json
from pathlib import Path


LABELS = ('small-march', 'small-december', 'medium-march', 'medium-december',
          'cp-small-march', 'cp-small-december', 'cp-medium-march', 'cp-medium-december')


def summarize(directory, *, require_complete=True):
    rows = []
    for label in LABELS:
        path = directory / (label + '.json')
        if not path.is_file():
            if require_complete:
                raise FileNotFoundError(path)
            continue
        data = json.loads(path.read_text())
        assert data['process_count'] == 1
        fields = data.get('comparison', {}).get('fields', [])
        groups = {}
        for field in fields:
            key = field['stream'], field['field']
            group = groups.setdefault(key, dict(stream=key[0], field=key[1],
                records=0, computed_words=0, computed_differing_words=0,
                missing_or_incompatible_words=0, first_computed_difference=None,
                structural_status_counts={}))
            group['records'] += 1
            if field['status'] == 'compared':
                group['computed_words'] += field['compared_words']
                group['computed_differing_words'] += field['differing_words']
                if field['differing_words'] and group['first_computed_difference'] is None:
                    group['first_computed_difference'] = field
            else:
                group['missing_or_incompatible_words'] += field['differing_words']
                counts = group['structural_status_counts']
                counts[field['status']] = counts.get(field['status'], 0) + 1
        groups = [groups[key] for key in sorted(groups)]
        assert sum(group['records'] for group in groups) == len(fields)
        counts = {name: sum(group[name] for group in groups) for name in
                  ('computed_words', 'computed_differing_words', 'missing_or_incompatible_words')}
        assert counts['computed_differing_words'] + counts['missing_or_incompatible_words'] == sum(
            field['differing_words'] for field in fields)
        rows.append(dict(label=label, receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            passed=data['passed'], tier=data['tier'], evidence_kind=data['evidence_kind'],
            model_execution=data.get('model_execution', False), field_rows=len(fields),
            failure_count=len(data.get('failures', [])),
            model_failure=next((failure for failure in data.get('failures', [])
                                if failure.startswith('hex model door failed')), None),
            per_stream_field=groups, **counts))
    result = dict(acceptance=False, process_count=1, duration_hours=24,
                  complete_attempt_matrix=len(rows) == len(LABELS),
                  complete_matrix=len(rows) == len(LABELS) and all(
                      row['model_execution'] and row['field_rows'] for row in rows),
                  rows=rows)
    text = json.dumps(result, indent=2) + '\n'
    assert len(text.encode()) < 10_000_000
    destination = directory / ('summary.json' if require_complete else 'available-receipts-summary.json')
    destination.write_text(text)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--available', action='store_true')
    args = parser.parse_args()
    summarize(args.directory, require_complete=not args.available)
