"""Measure a complete matching CAM native deck across two hosts with Rust.

The publications, scientific inputs, binary and process count must match.
This is a native host-scope witness, never a CUDA forecast acceptance claim.
Python handles metadata and subprocesses; Rust reads every stored field word.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(path):
    return dict(path=str(path.resolve()), sha256=digest(path))


def compare(args):
    from hexcore.verify_mpas import published_native_manifest, stream_inventory
    from archive_native_proof import archive, restore
    args.output.mkdir(parents=True, exist_ok=False)
    roots = [args.first.resolve(), args.second.resolve()]
    summaries = [json.loads(path.read_text()) for path in
                 (args.first_summary, args.second_summary)]
    rows = []
    for summary in summaries:
        if (not summary['complete'] or summary['process_count'] != 1
                or len(summary['cases']) != 1
                or summary['required_plan_sha256'] != digest(args.plan)):
            raise ValueError('Cross-host witness requires a complete one-process deck')
        rows.append(summary['cases'][0])
    if (rows[0]['id'] != rows[1]['id'] or rows[0]['host'] == rows[1]['host']
            or summaries[0]['binary_sha256'] != summaries[1]['binary_sha256']
            or rows[0]['runtime'] != rows[1]['runtime']):
        raise ValueError('Native cross-host binary, deck, host or clock differs')
    plan = json.loads(args.plan.read_text())
    required = next(row for row in plan['required_cases'] if row['id'] == rows[0]['id'])
    restored = []
    for root in roots:
        if (root/'ARCHIVE.json').exists():
            restored.append((root, restore(root)))
    manifests = [json.loads((root/'MANIFEST.json').read_text()) for root in roots]
    if (manifests[0]['inputs'] != manifests[1]['inputs']
            or set(manifests[0]['outputs']) != set(manifests[1]['outputs'])
            or any(manifest['ranks'] != 1 or manifest['runtime']['duration'] != '1_00:00:00'
                   or manifest['binary_sha256'] != summaries[0]['binary_sha256']
                   or manifest['namelist_sha256'] != required['namelist_sha256']
                   or manifest['streams_sha256'] != required['streams_sha256']
                   for manifest in manifests)):
        raise ValueError('Native cross-host scientific inputs or full stream inventory differs')
    normalized = []
    shared_input_copies = []

    def input_reference(root, name, expected):
        # Host transport can omit large shared tables. Both completed native
        # manifests must already bind the same bytes; use an existing pinned
        # copy, never invent data or change a scientific input.
        path = root/name
        if not path.is_file():
            path = next((other/name for other in roots if (other/name).is_file()), path)
            shared_input_copies.append(dict(host_root=str(root), name=name,
                                           pinned_copy=str(path.resolve()), sha256=expected))
        result = reference(path)
        if result['sha256'] != expected:
            raise ValueError('Native cross-host input copy differs: '+name)
        return result

    for root, manifest, row in zip(roots, manifests, rows):
        publication_path = root/'PUBLICATION.json'
        if digest(publication_path) != row['publication_sha256']:
            raise ValueError('Native cross-host publication changed after capture')
        inputs = {name: input_reference(root, filename, manifest['inputs'][filename]) for name, filename in
                  (('namelist', 'namelist.atmosphere'), ('streams', 'streams.atmosphere'),
                   ('init', row['runtime']['init']))}
        auxiliary = {name: input_reference(root, name, manifest['inputs'][name]) for name in manifest['inputs']
                     if name not in ('atmosphere_model', 'namelist.atmosphere',
                                     'streams.atmosphere', row['runtime']['init'])}
        deck = dict(inputs=inputs, auxiliary_inputs=auxiliary,
                    process_count=1, duration_hours=24)
        normalized.append(published_native_manifest(publication_path,
            json.loads(publication_path.read_text()), root, deck, args.comparator.resolve()))
        expected = {root/path['path'] for path in normalized[-1]['outputs']}
        excluded = {Path(ref['path']) for ref in (*inputs.values(), *auxiliary.values())}
        if stream_inventory(args.comparator.resolve(), root) - excluded != expected:
            raise ValueError('Native cross-host publication omits an output file')
    second = {row['path']: row for row in normalized[1]['outputs']}
    samples = []
    for first in normalized[0]['outputs']:
        other = second[first['path']]
        if any(first[name] != other[name] for name in ('stream', 'steps', 'times')):
            raise ValueError('Native cross-host stored output clocks differ')
        sample = dict(stream=first['stream'], steps=first['steps'], times=first['times'],
                      strict=str(roots[0]/first['path']), hex=str(roots[1]/other['path']))
        if first.get('clock_witness'):
            sample['clock_witness'] = first['clock_witness']
        samples.append(sample)
    request = args.output/'request.json'
    request.write_text(json.dumps(dict(schema='mpas-stream-compare-v1', samples=samples), indent=2)+'\n')
    result = args.output/'comparison.json'
    with (args.output/'comparator.log').open('w') as log:
        executed = subprocess.run([str(args.comparator.resolve()), '--request', str(request),
                                   '--out', str(result)], stdout=log, stderr=subprocess.STDOUT)
    comparison = json.loads(result.read_text()) if result.exists() else {}
    fields = comparison.get('fields', [])
    complete = bool(fields) and all(row['status'] == 'compared' for row in fields)
    differences = sum(row.get('differing_words', 0) for row in fields)
    accepted = executed.returncode == 0 and complete and differences == 0
    receipt = dict(scope='complete strict native CAM deck cross-host comparison',
        cuda_forecast_acceptance=False, shared_host_scope_extended=False,
        passed=accepted, complete=complete, process_count=1, duration_hours=24,
        id=rows[0]['id'], hosts=[row['host'] for row in rows],
        binary_sha256=summaries[0]['binary_sha256'], required_plan_sha256=digest(args.plan),
        summary_sha256=[digest(p) for p in (args.first_summary, args.second_summary)],
        publication_sha256=[row['publication_sha256'] for row in rows],
        shared_input_copies=shared_input_copies,
        comparator_sha256=digest(args.comparator), request_sha256=digest(request),
        comparison_sha256=digest(result) if result.exists() else None,
        files=len(samples), field_samples=len(fields), excluded_words=0,
        compared_words=sum(row.get('compared_words', 0) for row in fields),
        differing_words=differences, returncode=executed.returncode,
        archives=[proof['archive_sha256'] for _, proof in restored])
    (args.output/'summary.json').write_text(json.dumps(receipt, indent=2)+'\n')
    # The original archive survives unchanged; only restored owned products
    # retire after the complete comparator receipt and its hashes are written.
    if complete:
        for root, _ in restored:
            archive(root, retire=True)
    return 0 if accepted else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('first', 'second', 'first-summary', 'second-summary',
                 'plan', 'comparator', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(compare(parser.parse_args()))
