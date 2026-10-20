"""Retire owned graded forecasts under the integrator's disk ruling.

Every complete grade, including numerical failures, remains sealed. Only the
candidate NetCDF files authenticated by that full grade are removed. Shared
native references and unfinished forecast outputs are never selected.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(args):
    args.receipts.mkdir(parents=True, exist_ok=False)
    summary = dict(scope=__doc__, process_count=1, forecasts=[], bytes_retired=0,
                   excluded_words=[], changes_to_proof_grade=False)
    for root in args.grades:
        for report in sorted(root.rglob('complete-attempt.json')):
            grade = json.loads(report.read_text())
            if not grade.get('words') or not grade.get('streams'):
                continue
            full = report.parent / 'all-field-word-grade.json'
            if sha(full) != grade['full_receipt_sha256']:
                raise ValueError('Complete stored-word grade changed')
            command = grade['execution']['command']
            output = Path(command[command.index('--out') + 1]).resolve()
            if not output.is_relative_to(args.owned_outputs.resolve()):
                continue
            existing = [output / name for name in grade['candidate_hashes']
                        if (output / name).exists()]
            if not existing:
                continue
            if len(existing) != len(grade['candidate_hashes']):
                raise ValueError('Partially retired bank needs its original receipt')
            destination = args.receipts / f'{len(summary["forecasts"]):04}'
            subprocess.run([sys.executable,
                str(Path(__file__).with_name('seal_complete_evidence.py')),
                '--source', str(report.parent), '--output', str(destination)],
                check=True)
            retirement = destination / 'RETIREMENT.json'
            subprocess.run([sys.executable,
                str(Path(__file__).with_name('retire_committed_outputs.py')),
                '--output', str(output), '--grade', str(report),
                '--receipt', str(retirement)], check=True)
            retired = json.loads(retirement.read_text())['files']
            summary['forecasts'].append(dict(output=str(output),
                original_grade=str(report), original_grade_sha256=sha(report),
                complete_words=grade['words'],
                differing_words=grade['differing_words'],
                grade_seal_sha256=sha(destination / 'SEAL.json'),
                retirement_sha256=sha(retirement),
                bytes_retired=sum(row['bytes'] for row in retired)))
            summary['bytes_retired'] += summary['forecasts'][-1]['bytes_retired']
            (args.receipts / 'summary.json').write_text(
                json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grades', nargs='+', type=Path, required=True)
    parser.add_argument('--owned-outputs', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    main(parser.parse_args())
