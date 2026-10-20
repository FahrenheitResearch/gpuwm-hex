"""Retire disposable native-pool scratch after the full forecast word grade.

Candidate scientific output and comparison receipts are handled separately.
This removes only the named completed forecast's owned scratch directory after
recording every regular file hash and owned symlink. Shared targets are untouched.
Reproduction regenerates these pools from the original pinned forecast inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main(a):
    a.receipts.mkdir(parents=True, exist_ok=False)
    summary = dict(scope=__doc__, retired=[], bytes_retired=0, excluded_words=[],
        scientific_grade_changed=False, shared_targets_changed=False)
    for root in a.grades:
        for report in sorted(root.rglob('complete-attempt.json')):
            grade = json.loads(report.read_text())
            if not grade.get('words') or not grade.get('streams'):
                continue
            full = report.parent / 'all-field-word-grade.json'
            if sha(full) != grade['full_receipt_sha256']:
                raise ValueError('Complete all-field grade changed')
            command = grade['execution']['command']
            output = Path(command[command.index('--out') + 1]).resolve()
            if not output.is_relative_to(a.owned_outputs.resolve()):
                continue
            scratch = output / 'scratch'
            if not scratch.exists():
                continue
            if scratch.is_symlink():
                raise ValueError('Scratch must be an owned directory')
            files = {}
            for path in sorted(scratch.rglob('*')):
                name = str(path.relative_to(scratch))
                if path.is_symlink():
                    files[name] = dict(symlink=str(path.readlink()), target_deleted=False)
                elif path.is_file():
                    files[name] = dict(bytes=path.stat().st_size, sha256=sha(path))
            record = dict(output=str(output), retired_scratch=str(scratch), files=files,
                original_grade=str(report), original_grade_sha256=sha(report),
                complete_word_grade_sha256=sha(full), words=grade['words'],
                differing_words=grade['differing_words'], process_count=1,
                excluded_words=[], bytes_retired=sum(r.get('bytes', 0) for r in files.values()))
            receipt = a.receipts / (f'{len(summary["retired"]):04}' + '.json')
            receipt.write_text(json.dumps(record, indent=2) + '\n')
            if receipt.stat().st_size >= 10000000:
                raise ValueError('Retirement receipt exceeds delivery size')
            shutil.rmtree(scratch)
            summary['retired'].append(dict(receipt=receipt.name, sha256=sha(receipt),
                scratch=str(scratch), bytes_retired=record['bytes_retired']))
            summary['bytes_retired'] += record['bytes_retired']
            (a.receipts / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grades', type=Path, nargs='+', required=True)
    for name in ('owned-outputs', 'receipts'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())
