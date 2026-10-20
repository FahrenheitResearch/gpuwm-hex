"""Bind diagnostic replay to every plain-replay output hash and difference mask.

Rust produces the scientific comparisons and payload hashes. This auditor
authenticates their metadata; equal aggregate counts are never an identity
witness. Both complete ledgers, including failures, are mandatory.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ledger(path):
    summary = json.loads(path.read_text())
    assert summary['excluded_words'] == []
    rows = {}
    for chunk in summary['grade_chunks']:
        stored = Path(chunk['path'])
        local = path.parent / stored.parent.name / stored.name
        selected = local if local.is_file() else stored
        assert selected.stat().st_size == chunk['bytes']
        assert sha(selected) == chunk['sha256']
        entries = json.loads(selected.read_text())
        assert len(entries) == chunk['records']
        for row in entries:
            key = (row['deck'], row['call'], row['field'])
            assert key not in rows and row['excluded_words'] == []
            rows[key] = row
    assert len(rows) == summary['grade_records']
    assert sum(row['words'] for row in rows.values()) == summary['words']
    assert sum(row['differing_words'] for row in rows.values()) == summary['differing_words']
    return summary, rows


def run(args):
    left, a = ledger(args.plain)
    right, b = ledger(args.diagnostic)
    differences = []
    columns = ('words', 'reference_sha256', 'candidate_sha256',
               'difference_mask_sha256', 'differing_words', 'excluded_words')
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            differences.append(dict(key=key, missing='plain' if key not in a else 'diagnostic'))
        else:
            changed = [field for field in columns if a[key][field] != b[key][field]]
            if changed:
                differences.append(dict(key=key, changed=changed))
    equal_scope = all(left[field] == right[field] for field in
                      ('fixture_sha256', 'process_count', 'words', 'grade_records'))
    result = dict(scope=__doc__, process_count=left['process_count'],
                  plain_summary_sha256=sha(args.plain),
                  diagnostic_summary_sha256=sha(args.diagnostic),
                  same_complete_scope=equal_scope,
                  all_final_payloads_and_masks_identical=equal_scope and not differences,
                  grade_records=len(a), words=left['words'],
                  differing_ledger_records=differences, excluded_words=[],
                  diagnostic_only=True, forecast_proven=False)
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    return int(not result['all_final_payloads_and_masks_identical'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plain', 'diagnostic', 'receipt'):
        parser.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
