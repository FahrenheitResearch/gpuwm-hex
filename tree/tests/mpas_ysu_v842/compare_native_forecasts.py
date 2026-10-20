"""Compare complete native forecast inventories with the Rust word reader.

Python authenticates metadata and schedules files; NetCDF decoding and every
numeric/character comparison remain in native_word_compare.rs.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess

from audit_native_references import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--other', type=Path, required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    args = parser.parse_args()
    reference, other = args.reference.resolve(), args.other.resolve()
    helper, receipts = args.helper.resolve(), args.receipts.resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    manifests = [root / 'MANIFEST.json' for root in (reference, other)]
    records = [json.loads(path.read_text()) for path in manifests]
    for key in ('ranks', 'binary_sha256', 'pin_set_hash'):
        assert records[0][key] == records[1][key], f'Native metadata differs: {key}'
    for name, digest in records[0]['inputs'].items():
        assert records[1]['inputs'].get(name) == digest, f'Pinned input differs: {name}'
    extra_inputs = {name: digest for name, digest in records[1]['inputs'].items()
                    if name not in records[0]['inputs']}
    for name, digest in extra_inputs.items():
        assert sha(other / name) == digest, f'Additional input SHA differs: {name}'
    assert records[0]['outputs'] and set(records[0]['outputs']) == set(records[1]['outputs'])
    for root, record in zip((reference, other), records):
        assert {p.name for p in root.glob('*.nc') if p.is_file() and not p.is_symlink()} == set(record['outputs'])
        assert 'Finished running the atmosphere core' in (root / 'log.atmosphere.0000.out').read_text()
    summary = {'schema': 'p11-native-forecast-comparison/v1',
               'scope': 'native repeat witness, not hex forecast acceptance',
               'host': platform.node(), 'ranks': records[0]['ranks'],
               'binary_sha256': records[0]['binary_sha256'],
               'pin_set_hash': records[0]['pin_set_hash'],
               'manifest_sha256': [sha(path) for path in manifests],
               'helper_sha256': sha(helper), 'excluded_words': 0,
               'additional_repeat_inputs': extra_inputs,
               'files': [], 'all_identical': True, 'words': 0,
               'differing_words': 0, 'floating_words': 0,
               'differing_floating_words': 0, 'field_samples': 0}
    for filename in sorted(records[0]['outputs']):
        assert Path(filename).name == filename
        paths = [root / filename for root in (reference, other)]
        for path, record in zip(paths, records):
            assert sha(path) == record['outputs'][filename], f'Output SHA differs: {path}'
        result = subprocess.run([str(helper), *(str(path) for path in paths)],
                                capture_output=True, text=True)
        comparison = json.loads(result.stdout)
        assert result.returncode in (0, 1) and not comparison['errors'], comparison
        write(receipts / (filename + '.json'), comparison)
        summary['files'].append({'name': filename, 'receipt': filename + '.json',
                                 'sha256': [record['outputs'][filename] for record in records],
                                 'all_identical': comparison['all_identical'],
                                 'words': comparison['words'],
                                 'differing_words': comparison['differing_words']})
        summary['all_identical'] &= comparison['all_identical']
        summary['field_samples'] += len(comparison['fields'])
        for key in ('words', 'differing_words', 'floating_words', 'differing_floating_words'):
            summary[key] += comparison[key]
        write(receipts / 'SUMMARY.json', summary)
        print(json.dumps(summary['files'][-1]), flush=True)
    print(json.dumps({key: value for key, value in summary.items() if key != 'files'}), flush=True)
    raise SystemExit(0 if summary['all_identical'] else 1)


if __name__ == '__main__':
    main()
