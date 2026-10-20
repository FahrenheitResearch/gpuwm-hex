"""Seal returned whole native forecasts against their original request plans.

This records custody, not CUDA parity. Only complete original byte returns
are admitted; prior immutable custody receipts keep their original identities.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path, action='append', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior-custody', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    expected = {r['request_index']: r for r in plan['entries']}
    assert len(expected) == len(plan['entries'])
    owners = {}
    source_pins = {}
    for path in args.source_plan:
        source = load(path)
        digest = sha(path)
        source_pins[str(path.resolve())] = digest
        for entry in source['entries']:
            index = entry['request_index']
            assert index not in owners and entry == expected[index]
            owners[index] = digest
    assert owners.keys() == expected.keys()
    known = {}
    for path in args.prior_custody:
        prior = load(path)
        assert prior['host'] == plan['host'] and prior['process_count'] == plan['process_count']
        for row in prior.get('new_rows', prior.get('rows', [])):
            assert row['returned_originals_authenticated'] and row['native_only']
            index = row['request_index']
            assert row['name'] == expected[index]['name'] and index not in known
            known[index] = row
    candidates = {}
    checkpoint_pins = {}
    for path in [args.root / 'matrix.json', *sorted(args.root.glob('request-*.json'))]:
        if not path.exists():
            continue
        checkpoint = load(path)
        assert checkpoint['host'] == plan['host'] and checkpoint['process_count'] == plan['process_count']
        checkpoint_pins[str(path.resolve())] = sha(path)
        for row in checkpoint['rows']:
            index = row['request_index']
            assert checkpoint['plan_sha256'] == owners[index]
            assert row['name'] == expected[index]['name'] and row['complete']
            assert row['binary_sha256'] == plan['binary_sha256']
            for key in ('namelist_sha256', 'streams_sha256'):
                assert row[key] == expected[index][key]
            if index in candidates:
                assert candidates[index]['manifest_sha256'] == row['manifest_sha256']
            candidates[index] = row
    args.out.mkdir(exist_ok=False)
    rows, pending = [], []
    for index, row in sorted(candidates.items()):
        if index in known:
            assert known[index]['manifest_sha256'] == row['manifest_sha256']
            continue
        directory = args.root / row['name']
        metadata = ('MANIFEST.json', 'CLOCKS.json', 'FIELD-DIGESTS.json', 'PROVENANCE.json')
        if not all((directory / name).is_file() for name in metadata):
            pending.append(index)
            continue
        manifest = load(directory / 'MANIFEST.json')
        assert sha(directory / 'MANIFEST.json') == row['manifest_sha256']
        assert manifest['ranks'] == plan['process_count']
        assert manifest['binary_sha256'] == plan['binary_sha256']
        assert manifest['inputs']['namelist.atmosphere'] == row['namelist_sha256']
        assert manifest['inputs']['streams.atmosphere'] == row['streams_sha256']
        if not all((directory / name).is_file() for name in manifest['outputs']):
            pending.append(index)
            continue
        for name, digest in manifest['outputs'].items():
            assert sha(directory / name) == digest, (index, name)
        assert len(manifest['outputs']) == row['output_files']
        assert sum((directory / name).stat().st_size for name in manifest['outputs']) == row['original_bytes']
        assert sha(directory / 'CLOCKS.json') == row['clocks_sha256']
        assert load(directory / 'CLOCKS.json')['complete']
        assert sha(directory / 'FIELD-DIGESTS.json') == row['field_digest_sha256']
        provenance = load(directory / 'PROVENANCE.json')
        assert provenance['host'] == plan['host'] and provenance['process_count'] == plan['process_count']
        assert provenance['plan_sha256'] == owners[index]
        assert provenance['native_binary_sha256'] == plan['binary_sha256']
        assert provenance['full_original_outputs_retained'] and not provenance['native_library_rebuilt']
        pins = {}
        for name in metadata:
            source, target = directory / name, args.out / (row['name'] + '-' + name)
            assert source.stat().st_size < 10_000_000
            shutil.copy2(source, target)
            pins[name] = sha(target)
            assert pins[name] == sha(source)
        rows.append(dict(row, durable_originals_host_b=str(directory.resolve()),
            returned_originals_authenticated=True, native_only=True,
            source_plan_sha256=owners[index], original_metadata_sha256=pins))
        print(json.dumps(dict(whole_native_request_closed=index,
            original_output_files=len(manifest['outputs']))), flush=True)
    result = dict(schema='p21-v35-native-return-custody/v1', host=plan['host'],
        process_count=plan['process_count'], plan_sha256=sha(args.plan), source_plans=source_pins,
        source_checkpoints=checkpoint_pins,
        prior_custody=[dict(path=str(p.resolve()), sha256=sha(p)) for p in args.prior_custody],
        new_rows=rows, pending_originals=pending, required=len(expected),
        locally_authenticated_request_indices=sorted(set(known) | {r['request_index'] for r in rows}),
        cuda_forecast_identity_claim=False)
    (args.out / 'custody.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(new_native_references=len(rows),
        durable_native_references=len(result['locally_authenticated_request_indices']),
        pending_originals=pending)), flush=True)


if __name__ == '__main__':
    main()
