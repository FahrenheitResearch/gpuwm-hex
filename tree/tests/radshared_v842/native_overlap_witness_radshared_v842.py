"""Authenticate the whole native overlap witness using Rust field digests.

Comparisons stay at a fixed MPI rank count. Different hashes identify changed
fields; they never invent a numerical differing-word count.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory(directory, row):
    name = row['name']
    manifest_path = directory / (name + '-MANIFEST.json')
    clocks_path = directory / (name + '-CLOCKS.json')
    assert sha(manifest_path) == row['manifest_sha256']
    assert sha(clocks_path) == row['field_digests_sha256']
    manifest = json.loads(manifest_path.read_text())
    clocks = json.loads(clocks_path.read_text())
    assert clocks['complete'] and manifest['outputs_retired']
    assert manifest['ranks'] == row['process_count']
    assert manifest['namelist_sha256'] == row['namelist_sha256']
    assert manifest['streams_sha256'] == row['streams_sha256']
    assert manifest['retired_clock_receipt_sha256'] == sha(clocks_path)
    assert manifest['output_digest_tool_sha256'] == clocks['comparator_sha256']
    fields = {}
    for filename, entry in clocks['files'].items():
        assert manifest['outputs'][filename] == entry['sha256']
        for variable in entry['inventory']['variables']:
            fields[filename, variable['field']] = {
                key: variable[key] for key in ('netcdf_type', 'shape', 'words', 'word_sha256')}
    return fields, manifest['binary_sha256'], clocks['comparator_sha256']


def compare(left, right):
    assert left.keys() == right.keys(), 'native output inventory differs'
    changed = [dict(file=key[0], field=key[1], reference=left[key], actual=right[key])
               for key in sorted(left) if left[key] != right[key]]
    return dict(fields=len(left), words=sum(v['words'] for v in left.values()),
                differing_fields=len(changed),
                differing_words=0 if not changed else None, changed_fields=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', type=Path, action='append', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert len(args.host) == 2
    datasets = []
    pins = set()
    hosts = set()
    for directory in args.host:
        matrix = json.loads((directory / 'matrix.json').read_text())
        assert matrix['complete'] and len(matrix['rows']) == 10
        data = {}
        for row in matrix['rows']:
            assert row['complete'] and row['native_exit'] == row['clock_exit'] == 0
            fields, binary, reader = inventory(directory, row)
            pins.add((binary, reader, matrix['source_namelist_sha256'],
                      matrix['source_streams_sha256']))
            hosts.add(row['host'])
            data[row['name']] = fields
        datasets.append((directory, data))
    assert len(pins) == 1 and len(hosts) == 2, 'two-host native pins differ'
    comparisons = []
    for ranks in (1, 2):
        baseline = datasets[0][1][f'none-rank{ranks}-repeat1']
        for directory, data in datasets:
            for option in ('none', 'unknown'):
                for repeat in (1, 2):
                    name = f'{option}-rank{ranks}-repeat{repeat}'
                    result = compare(baseline, data[name])
                    comparisons.append(dict(host_receipts=str(directory), name=name,
                                            process_count=ranks, **result))
            result = compare(baseline, data[f'maximum_random-rank{ranks}-repeat1'])
            assert result['differing_fields'] > 0, 'overlap sensitivity control did not differ'
            comparisons.append(dict(host_receipts=str(directory),
                                    name=f'maximum_random-rank{ranks}-repeat1',
                                    sensitivity_control=True, process_count=ranks, **result))
    stable = all(r['differing_fields'] == 0 for r in comparisons
                 if not r.get('sensitivity_control'))
    binary, reader, namelist, streams = next(iter(pins))
    receipt = dict(purpose='two-host repeated full-RRTMG native overlap defect witness',
                   native_binary_sha256=binary, rust_reader_sha256=reader,
                   source_namelist_sha256=namelist, source_streams_sha256=streams,
                   host_matrices={str(p / 'matrix.json'): sha(p / 'matrix.json') for p in args.host},
                   duration='0_01:00:00', process_counts=[1, 2],
                   unknown_reproduces_none=stable, comparisons=comparisons,
                   cuda_forecast_identity_claim=False)
    encoded = json.dumps(receipt, indent=2) + '\n'
    assert len(encoded.encode()) < 10_000_000
    args.receipt.write_text(encoded)
    print(json.dumps(dict(unknown_reproduces_none=stable,
                          fixed_rank_comparisons=len(comparisons),
                          compared_words=sum(r['words'] for r in comparisons
                                             if not r.get('sensitivity_control')))))
    return 0 if stable else 1


if __name__ == '__main__':
    raise SystemExit(main())
