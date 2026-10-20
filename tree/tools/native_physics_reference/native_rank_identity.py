#!/usr/bin/env python3
"""Compare every native physics field/stream across ranks using raw Rust words.

Extends native_dry_reference/native_rank_identity.py's tracer-only scope without
loading weather fields in Python. Integer and character fields are included.
Missing output files, variables, or unsupported field types fail closed.
"""
import argparse
import json
from pathlib import Path
import sys
from compare_runs import compare


def witness(reference, other, comparator):
    """Bind the comparison to completed runs of the same deck and inputs."""
    manifests = [json.loads((run/'MANIFEST.json').read_text()) for run in (reference, other)]
    for key in ('schema', 'pin_set_hash', 'deck', 'runtime', 'binary_sha256',
                'namelist_sha256', 'streams_sha256'):
        if key not in manifests[0] or manifests[0][key] != manifests[1].get(key):
            raise ValueError(f'rank comparison run contract differs: {key}')
    inputs = [{name: digest for name, digest in manifest['inputs'].items()
               if not name.endswith('.graph.info.part.'+str(manifest['ranks']))}
              for manifest in manifests]
    if inputs[0] != inputs[1]:
        raise ValueError('rank comparison native inputs differ')
    if manifests[0]['ranks'] == manifests[1]['ranks']:
        raise ValueError('rank comparison requires distinct native rank counts')
    result = compare(reference, other, comparator, '*.nc')
    if any(set(result['files']) != set(manifest['outputs']) for manifest in manifests):
        raise ValueError('rank comparison output manifests differ from actual file set')
    for name, entry in result['files'].items():
        if entry['sha256'] != [manifest['outputs'][name] for manifest in manifests]:
            raise ValueError(f'rank comparison output differs from native manifest: {name}')
    result['ranks'] = [manifest['ranks'] for manifest in manifests]
    result['runtime'] = manifests[0]['runtime']
    result['first_difference'] = next((
        {'file': name, 'field': field['field'], 'word': field['first_difference']}
        for name, entry in result['files'].items() for field in entry['fields']
        if field['differing_words']), None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--comparator', type=Path, required=True)
    args = parser.parse_args()
    if len(args.runs) < 2:
        raise ValueError('at least two completed native runs required')
    comparisons = [witness(args.runs[0], other, args.comparator) for other in args.runs[1:]]
    result = {'schema': 'gpuwm-hex.native-physics-rank-identity/v1',
              'runs': [str(run) for run in args.runs], 'comparisons': comparisons,
              'all_identical': all(row['all_identical'] for row in comparisons),
              'all_file_bytes_identical': all(row['all_file_bytes_identical'] for row in comparisons)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'all_identical': result['all_identical'], 'runs': len(args.runs)}))
    return int(not result['all_identical'])


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f'rank witness error: {error}', file=sys.stderr)
        raise SystemExit(2)
