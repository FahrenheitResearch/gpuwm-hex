"""Summarize actual forecast/comparator receipts without dropping missing fields.

The Rust comparator owns weather data access and word comparisons. Python
aggregates its receipt metadata; missing native words are reported separately
from words actually compared. This diagnostic collector never grants acceptance.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summarize(root):
    receipt = root / 'receipts'
    job = json.loads((receipt / 'whole-door-job.json').read_text())
    output = Path(job['output'])
    comparison_path = receipt / 'whole-door-verify.json'
    comparison = json.loads(comparison_path.read_text())
    runtime_path = output / 'native-physics-receipt.json'
    runtime = json.loads(runtime_path.read_text())
    if job.get('native_services') is not True or job.get('p25_cuda_executed') is not False:
        raise SystemExit('This collector requires an explicitly submitted native baseline.')
    if set(runtime['physics']['backends'].values()) != {'native-cpu'} or 'surface_service' in runtime:
        raise SystemExit('Actual runtime is not the all-native baseline this collector describes.')
    model_exit = int((receipt / 'whole-door-forecast.exit').read_text())
    verify_exit = int((receipt / 'whole-door-verify.exit').read_text())
    if model_exit != 0 or not runtime.get('completed'):
        raise SystemExit('Actual model completion is required, not inferred from comparison.')
    if runtime['steps_completed'] != runtime['steps_requested']:
        raise SystemExit('Actual forecast did not complete its requested steps.')
    if comparison['process_count'] != job['process_count']:
        raise SystemExit('Actual comparator and submitted process counts disagree.')
    fields = comparison['comparison']['fields']
    aggregates = defaultdict(lambda: dict(samples=0, compared_samples=0,
                                         compared_words=0, differing_words=0,
                                         missing_samples=0, missing_native_words=0,
                                         statuses=Counter()))
    for row in fields:
        target = aggregates[(row['stream'], row['field'])]
        target['samples'] += 1
        target['statuses'][row['status']] += 1
        if row['status'] == 'compared':
            target['compared_samples'] += 1
            target['compared_words'] += row['compared_words']
            target['differing_words'] += row['differing_words']
        else:
            # The comparator's compared_words counts the native field size even
            # for missing fields. Those words were not numerically compared.
            target['missing_samples'] += 1
            target['missing_native_words'] += row['compared_words']
    keys = ('samples', 'compared_samples', 'compared_words', 'differing_words',
            'missing_samples', 'missing_native_words')
    by_stream = defaultdict(lambda: {key: 0 for key in keys})
    for (stream, _field), target in aggregates.items():
        for key in keys:
            by_stream[stream][key] += target[key]
    mismatches = sorted((row for row in fields if row['status'] == 'compared'
                         and row['differing_words'] > 0),
                        key=lambda row: (row['step'], row['stream'], row['field']))
    retained = receipt / 'whole-door-runtime'
    retained.mkdir(exist_ok=True)
    pins = {}
    for name in ('native-physics-receipt.json', 'native-physics-request.json',
                 'physics-calls.jsonl', 'timeline.jsonl'):
        path = output / name
        if path.is_file():
            if path.stat().st_size >= 10_000_000:
                raise SystemExit('Runtime receipt exceeds the delivery size limit: ' + str(path))
            shutil.copy2(path, retained / name)
            pins[name] = digest(path)
    summary = dict(schema='p25-whole-door-summary/v1', diagnostic_only=True,
                   process_count=job['process_count'], model_exit=model_exit,
                   steps_completed=runtime['steps_completed'], verify_exit=verify_exit,
                   comparison_passed=comparison['passed'], acceptance_tier=comparison['acceptance_tier'],
                   evidence_kind=comparison['evidence_kind'],
                   p25_cuda_executed=False,
                   service_execution_reason='Native baseline; no p25 service selector registration.',
                   comparison_sha256=digest(comparison_path), runtime_sha256=pins,
                   statuses=dict(Counter(row['status'] for row in fields)),
                   totals={key: sum(stream[key] for stream in by_stream.values()) for key in keys},
                   by_stream=dict(sorted(by_stream.items())),
                   first_computed_difference=mismatches[0] if mismatches else None,
                   acceptance_failures=comparison['acceptance_failures'],
                   fields=[dict(stream=stream, field=field, **target)
                           for (stream, field), target in sorted(aggregates.items())])
    (receipt / 'whole-door-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                      ('model_exit', 'steps_completed', 'verify_exit', 'totals',
                       'by_stream', 'first_computed_difference', 'p25_cuda_executed')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    arguments = parser.parse_args()
    summarize(arguments.directory.resolve())
