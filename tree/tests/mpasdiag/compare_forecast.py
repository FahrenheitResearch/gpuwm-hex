"""Compare every stream from a witnessed forecast using the real Rust door."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('deck', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('comparator', type=Path)
    parser.add_argument('--transport-deck', type=Path)
    args = parser.parse_args()
    attempt = json.loads(args.attempt.read_text())
    if attempt['deck_sha256'] != hashlib.sha256(args.deck.read_bytes()).hexdigest():
        raise ValueError('comparison deck differs from executed deck')
    command = [sys.executable, '-m', 'hexcore.cli', 'verify-mpas',
               '--deck', str(args.transport_deck or args.deck), '--hex-output', str(args.output),
               '--receipt', str(args.receipt), '--work-root', str(args.work),
               '--comparator', str(args.comparator)]
    if attempt['returncode']:
        command += ['--failed-model-exit', str(attempt['returncode'])]
    with args.receipt.with_suffix('.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    receipt = json.loads(args.receipt.read_text())
    rows = receipt.get('comparison', {}).get('fields', [])
    streams = {}
    for row in rows:
        stream = streams.setdefault(row['stream'], dict(compared_words=0,
            numeric_differing_words=0, missing_or_incompatible_samples=0,
            missing_or_incompatible_words=0))
        if row['status'] == 'compared':
            stream['compared_words'] += row['compared_words']
            stream['numeric_differing_words'] += row['differing_words']
        else:
            stream['missing_or_incompatible_samples'] += 1
            stream['missing_or_incompatible_words'] += row['differing_words']
    differing = sorted((row for row in rows if row['status'] == 'compared'
                        and row['differing_words']), key=lambda row:
                       (row['step'], row['stream'], row['field']))
    summary = dict(schema='mpas-o2.whole-stream-comparison/v1', process_count=1,
        completed=attempt['completed'], steps_completed=attempt['steps_completed'],
        model_returncode=attempt['returncode'], comparison_returncode=result.returncode,
        passed=False, full_proof_accepted=False, streams=streams,
        first_computed_difference=differing[0] if differing else None,
        field_records=len(rows), comparison=str(args.receipt),
        comparison_sha256=hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        actual_attempt=str(args.attempt),
        actual_attempt_sha256=hashlib.sha256(args.attempt.read_bytes()).hexdigest())
    args.receipt.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    return result.returncode or (0 if receipt.get('passed') else 1)


if __name__ == '__main__':
    raise SystemExit(main())
