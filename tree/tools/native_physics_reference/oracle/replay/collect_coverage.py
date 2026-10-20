#!/usr/bin/env python3
"""Collect real gcov counters and map coverage to the pinned native source.

Counters can contain several replay executions. Their cumulative counts are
reported as such; the receipt never asserts branch or forecast completeness.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

if __package__:
    from .coverage import summarize
else:
    from coverage import summarize


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execution_counts(counts):
    if not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError('coverage needs positive integer native replay process counts')
    if len(set(counts)) != 1:
        raise ValueError('coverage cannot combine different native replay process counts')
    return list(counts)


def collect(counter, source, build, destination, compiled_build_root=None,
            replay_process_counts=None):
    counts = execution_counts(replay_process_counts)
    counter = counter.resolve()
    counter.relative_to(build.resolve())
    note = counter.with_suffix('.gcno')
    before = sha(counter)
    destination.mkdir(parents=True)
    run = subprocess.run(['gcov', '-j', '-b', '-c', str(counter)],
                         cwd=destination, capture_output=True, text=True)
    (destination / 'gcov.log').write_text(run.stdout + run.stderr)
    run.check_returncode()
    artifact = destination / (counter.stem + '.gcov.json.gz')
    if sha(counter) != before:
        raise ValueError(f'coverage counter changed during collection: {counter}')
    report = summarize(json.loads(gzip.decompress(artifact.read_bytes())), source, build,
                       compiled_build_root, retain_unmapped_sources=True)
    if any(row.get('native_branches_counted') != 0 for row in report['unmapped_files']):
        raise ValueError(f'unmapped native coverage: {report["unmapped_files"]}')
    report.update(schema='mpas-oracle-gcov-v2', counter=str(counter),
                  counter_sha256=before, notes_sha256=sha(note),
                  gcov_sha256=sha(artifact), counters_cumulative=True,
                  compiled_build_root=str((compiled_build_root or build).resolve()),
                  collected_build_root=str(build.resolve()),
                  process_count=counts[0], replay_process_counts=counts,
                  coverage_complete=False, forecast_proven=False)
    path = destination / 'coverage.json'
    payload = (json.dumps(report, separators=(',', ':')) + '\n').encode()
    if len(payload) >= 10_000_000:
        raise ValueError('native coverage receipt exceeds delivery limit')
    path.write_bytes(payload)
    return {'receipt': str(path), 'sha256': sha(path),
            'process_count': counts[0], 'replay_process_counts': counts,
            'native_sources': [{key: row[key] for key in (
                'native_source', 'source_sha256', 'branches_total', 'branches_taken',
                'lines_total', 'lines_executed')} for row in report['files']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counter', type=Path, action='append', required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    parser.add_argument('--compiled-build-root', type=Path,
                        help='original compiler directory for relocated notes, counters and sources')
    parser.add_argument('--replay-receipt', type=Path, action='append', required=True)
    parser.add_argument('--matrix-receipt', type=Path,
                        help='authenticated matrix supplying legacy batch process-count witnesses')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('coverage output must be fresh')
    witnesses = []
    matrix = json.loads(args.matrix_receipt.read_text()) if args.matrix_receipt else None
    for path in args.replay_receipt:
        receipt = json.loads(path.read_text())
        if receipt.get('replay_recipe') != 'coverage' or not receipt.get('comparison_complete'):
            raise ValueError(f'completed coverage replay required: {path}')
        count = receipt.get('replay_process_count')
        if count is None and matrix:
            matched = [s['recipes']['coverage'] for s in matrix['sites']
                       if s['recipes']['coverage']['sha256'] == sha(path)]
            if (len(matched) != 1 or matrix.get('process_count') != 1 or
                    matched[0]['executable_sha256'] != receipt['executable_sha256']):
                raise ValueError('matrix does not bind legacy coverage replay')
            count = matrix['process_count']
        if count is None:
            raise ValueError('coverage replay lacks a process-count witness')
        witnesses.append({'receipt': str(path), 'sha256': sha(path),
                          'dump_root_sha256': receipt['dump_root_sha256'],
                          'replay_process_count': count,
                          'executable_sha256': receipt['executable_sha256']})
    counts = execution_counts([row['replay_process_count'] for row in witnesses])
    args.out.mkdir(parents=True)
    rows = []
    for counter in sorted(set(args.counter)):
        relative = counter.resolve().relative_to(args.build_root.resolve())
        rows.append(collect(counter, args.source_root, args.build_root,
                            args.out / relative.with_suffix(''), args.compiled_build_root,
                            counts))
    result = {'schema': 'mpas-native-coverage-collection-v1', 'files': rows,
              'executions': witnesses, 'counters_cumulative': True,
              'process_count': counts[0], 'replay_process_counts': counts,
              'coverage_complete': False, 'forecast_proven': False, 'proof_bar': False,
              'matrix_receipt_sha256': sha(args.matrix_receipt) if args.matrix_receipt else None}
    (args.out / 'index.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
