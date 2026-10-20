"""Authenticate complete native checkpoints into one whole-deck catalog.

Original source-plan identities and closed reference receipts are retained.
No native numerical run, missing allowance or comparison grade is invented.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path, action='append', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['host'] == 'v35' and plan['process_count'] == 1
    expected = {entry['request_index']: entry for entry in plan['entries']}
    owners = {}
    source_pins = {}
    for path in args.source_plan:
        source = json.loads(path.read_text())
        digest = sha(path)
        source_pins[str(path)] = digest
        for entry in source['entries']:
            index = entry['request_index']
            assert index not in owners and entry == expected[index]
            owners[index] = digest
    assert owners.keys() == expected.keys()
    candidates = {}
    checkpoints = [args.root / 'matrix.json', *sorted(args.root.glob('request-*.json'))]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            continue
        native = json.loads(checkpoint.read_text())
        assert native['host'] == 'v35' and native['process_count'] == 1
        for row in native['rows']:
            index = row['request_index']
            assert native['plan_sha256'] == owners[index]
            entry = expected[index]
            assert row['name'] == entry['name'] and row['complete']
            assert row['binary_sha256'] == plan['binary_sha256']
            assert row['namelist_sha256'] == entry['namelist_sha256']
            assert row['streams_sha256'] == entry['streams_sha256']
            root = args.root / row['name']
            for filename, key in (('MANIFEST.json', 'manifest_sha256'),
                                  ('CLOCKS.json', 'clocks_sha256'),
                                  ('FIELD-DIGESTS.json', 'field_digest_sha256')):
                assert sha(root / filename) == row[key]
            assert json.loads((root / 'CLOCKS.json').read_text())['complete']
            if index in candidates:
                assert candidates[index]['manifest_sha256'] == row['manifest_sha256']
            candidates[index] = dict(row, source_plan_sha256=owners[index],
                                     source_checkpoint_sha256=sha(checkpoint))
    rows = []
    for entry in plan['entries']:
        if entry['request_index'] not in candidates:
            break
        rows.append(candidates[entry['request_index']])
    result = dict(schema='p21-v35-native-option-reference-matrix/v1', host='v35',
        process_count=1, plan_sha256=sha(args.plan), source_plans=source_pins,
        rows=rows, available_request_indices=sorted(candidates), required=len(expected),
        complete=len(rows) == len(expected), cuda_forecast_identity_claim=False)
    temporary = args.output.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(dict(available=len(candidates), contiguous=len(rows), required=len(expected))))


if __name__ == '__main__':
    main()
