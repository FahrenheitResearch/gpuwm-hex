"""Bind complete audited native runs to the actual verify-mpas forecast door.

This prepares metadata only. It never certifies a CUDA forecast. NetCDF
clocks and field inventories come from the existing Rust audit receipts.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def reference(path, expected=None):
    path = path.resolve(strict=True)
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Native input changed: {path}')
    return {'path': str(path), 'sha256': actual}


def write(path, value):
    encoded = json.dumps(value, indent=2) + '\n'
    if len(encoded.encode()) >= 10_000_000:
        raise ValueError(f'Receipt exceeds delivery limit: {path}')
    path.write_text(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'outputs', 'inventory', 'target', 'library', 'reader', 'source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--reference-outputs', type=Path)
    args = parser.parse_args()
    root = args.work.resolve(strict=True)
    target = args.target.resolve()
    target.mkdir(exist_ok=False, parents=True)
    inventory = json.loads((args.inventory/'SUMMARY.json').read_text())
    if inventory['pending']:
        raise ValueError('Every native reference must finish its Rust inventory audit')
    index = args.index.resolve() if args.index else root/'native-acceptance/index.json'
    assert digest(index) == inventory['index_sha256']
    entries = json.loads(index.read_text())['entries']
    audited = {entry['name']: entry for entry in inventory['entries']}
    assert set(audited) == {entry['name'] for entry in entries}
    summary = {'schema': 'p11-verification-deck-preparation/v1',
               'forecast_proven': False, 'process_count': 1, 'entries': []}
    for entry in entries:
        name = entry['name']
        output = (args.outputs/name).resolve(strict=True)
        # verify-mpas's output_root is an output-only artifact directory.
        # Native run directories also contain linked initialization/table
        # NetCDF inputs. Keep those pinned as inputs, and hard-link EVERY
        # produced output into a distinct root without re-encoding data.
        reference_base = args.reference_outputs or args.outputs.parent/'verification-outputs'
        reference_output = reference_base/name
        reference_output.mkdir(parents=True, exist_ok=False)
        original_path = output/'MANIFEST.json'
        original = json.loads(original_path.read_text())
        assert digest(original_path) == audited[name]['native_manifest_sha256']
        assert original['ranks'] == entry['ranks'] == 1
        assert original['binary_sha256'] == audited[name]['binary_sha256']
        log = output/'log.atmosphere.0000.out'
        assert 'Finished running the atmosphere core' in log.read_text()
        if any(row.startswith('ERROR:') for row in log.read_text().splitlines()):
            raise ValueError('Native reference reported errors; do not bind failed output as an oracle')
        init_name = entry['mesh'] + '.init.nc'
        inputs = {key: reference(output/filename, original[sha_key])
                  for key, filename, sha_key in (
                      ('namelist', 'namelist.atmosphere', 'namelist_sha256'),
                      ('streams', 'streams.atmosphere', 'streams_sha256'))}
        inputs['init'] = reference(root/entry['case']/init_name, original['inputs'][init_name])
        # Primary namelist/streams/init are already pinned above. The actual
        # verifier forbids auxiliaries that would replace their staged files.
        primary_names = {'namelist.atmosphere', 'streams.atmosphere', init_name}
        auxiliary = {filename: reference(output/filename, sha)
                     for filename, sha in original['inputs'].items()
                     if filename not in primary_names}
        declarations = ET.parse(output/'streams.atmosphere').getroot()
        by_prefix = {stream.attrib['filename_template'].split('.', 1)[0]: stream
                     for stream in declarations if 'output' in stream.get('type', '')}
        # Native clock arithmetic is metadata: no weather values are decoded here.
        namelist = (output/'namelist.atmosphere').read_text()
        start = datetime.strptime(re.search(r"config_start_time\s*=\s*'([^']+)'", namelist).group(1), '%Y-%m-%d_%H:%M:%S')
        dt = float(re.search(r'config_dt\s*=\s*([0-9.eEdD+-]+)', namelist).group(1).replace('d', 'e').replace('D', 'e'))
        duration = 86400 / dt
        assert duration.is_integer() and entry['hours'] == 24
        outputs = []
        expected_samples = {}
        for filename, sha in sorted(original['outputs'].items()):
            (reference_output/filename).hardlink_to(output/filename)
            receipt_path = args.inventory/(name + '--' + filename + '.json')
            receipt = json.loads(receipt_path.read_text())
            assert receipt['file_sha256'] == sha
            assert receipt['native_manifest_sha256'] == digest(original_path)
            prefix, stamp = filename[:-3].split('.', 1)
            declaration = by_prefix[prefix]
            times = receipt['xtime'] or [datetime.strptime(stamp, '%Y-%m-%d_%H.%M.%S').strftime('%Y-%m-%d_%H:%M:%S')]
            steps = []
            for clock in times:
                step = (datetime.strptime(clock, '%Y-%m-%d_%H:%M:%S') - start).total_seconds()/dt
                assert step.is_integer() and 0 <= step <= duration
                steps.append(int(step))
            item = dict(stream=declaration.attrib['name'], path=filename, sha256=sha,
                        steps=steps, times=times)
            if not receipt['xtime']:
                item['clock_witness'] = {'filename_template': declaration.attrib['filename_template']}
            outputs.append(item)
            expected_samples.setdefault(item['stream'], []).extend(steps)
        assert len(outputs) == audited[name]['output_files']
        for samples in expected_samples.values():
            assert max(samples) == int(duration) and len(samples) == len(set(samples))
        native_path = target/(name + '-native.json')
        write(native_path, dict(schema='mpas-native-run-v1', pin_set=original['pin_set_hash'],
                                process_count=1, complete=True, output_root=str(reference_output.resolve()),
                                inputs={key: ref['sha256'] for key, ref in inputs.items()},
                                auxiliary_inputs={key: ref['sha256'] for key, ref in auxiliary.items()},
                                outputs=outputs, completion_witness=reference(log),
                                source_manifest=reference(original_path),
                                rust_inventory=reference(args.inventory/'SUMMARY.json')))
        command = ['gpuwm-hex', 'forecast', '--mesh', entry['mesh'],
                   '--grid', str((root/entry['case']/(entry['mesh']+'.grid.nc')).resolve()),
                   '--init', '{init}', '--physics-backend', 'mpasphys',
                   '--physics-namelist', '{namelist}', '--streams', '{streams}',
                   '--physics', "config_pbl_backend='hex-cuda'",
                   '--native-physics-library', str(args.library.resolve(strict=True)),
                   '--native-pool-reader', str(args.reader.resolve(strict=True)),
                   '--mpas-source', str(args.source.resolve(strict=True)), '--out', '{out}']
        orographic = entry['mesh'] + '.ugwp_oro_data.nc'
        if orographic in auxiliary:
            command.extend(['--orographic-statistics', auxiliary[orographic]['path']])
        deck_path = target/(name + '-deck.json')
        write(deck_path, dict(schema='mpas-verification-deck-v1', pin_set=original['pin_set_hash'],
                              mesh=entry['mesh'], date=start.strftime('%Y-%m-%d_%H:%M:%S'),
                              id='p11-' + name, suite=entry['suite'], process_count=1,
                              inputs=inputs, auxiliary_inputs=auxiliary,
                              native=reference(native_path), duration_steps=int(duration),
                              expected_samples=expected_samples, hex_command=command,
                              hex_support_inputs={'native-library': reference(args.library)}))
        summary['entries'].append(dict(name=name, deck=reference(deck_path),
                                       native=reference(native_path), output_files=len(outputs),
                                       option=(('single-ysu-topdown' if entry['ysu_pblmix'] else 'single-ysu')
                                               if entry['suite'] == 'none' else entry['suite']),
                                       mesh=entry['mesh'], date=entry['date'], ranks=1, hours=24))
        write(target/'SUMMARY.json', summary)
    print(json.dumps({'sealed_verification_decks': len(entries), 'forecast_proven': False}))


if __name__ == '__main__':
    main()
