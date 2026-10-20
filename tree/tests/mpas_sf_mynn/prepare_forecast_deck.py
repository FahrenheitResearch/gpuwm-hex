"""Seal an owned strict-native run and bind the actual MYNN forecast door.

Consumes f0 run_forecast.py metadata. It never decodes field payloads, changes
the native namelist, or certifies a forecast from column comparisons.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--strict-binary', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--suite', choices=('none', 'convection_permitting'), default='none')
    args = parser.parse_args()
    root = args.native_run.resolve(strict=True)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'mpas.native-forecast-cache/v1' or manifest['ranks'] != 1:
        raise ValueError('requires a completed fixed-count strict native run')
    binary_sha = digest(args.strict_binary)
    if (manifest['binary_sha256'] != binary_sha
            or manifest['inputs']['atmosphere_model'] != binary_sha):
        raise ValueError('native run did not use the supplied pinned strict executable')
    template = json.loads(args.template.read_text())
    if manifest['pin_set_hash'] != template['pin_set']:
        raise ValueError('native and forecast template pin sets differ')
    if manifest['runtime']['mesh'] != template['mesh']:
        raise ValueError('native and template meshes differ')
    if manifest['runtime']['start'] != template['date']:
        raise ValueError('native and template dates differ')
    # Exact output set only: input symlinks must not become native outputs.
    reference = args.reference.resolve()
    reference.mkdir(parents=True, exist_ok=False)
    names = [*manifest['outputs'], 'MANIFEST.json', 'namelist.atmosphere',
             'streams.atmosphere', 'log.atmosphere.0000.out']
    files = {}
    for name in names:
        source = root / name
        sha = digest(source)
        if name in manifest['outputs'] and sha != manifest['outputs'][name]:
            raise ValueError(f'native output changed after run: {name}')
        target = reference / name
        # Both paths are owned scratch. Hard links retain the exact bytes
        # without duplicating multi-gigabyte output on the shared disk.
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
        files[name] = sha
    publication = dict(schema='mpas.native-cache-publication/v1',
                       description=f'owned pristine strict-native MYNN {args.suite} forecast, one rank',
                       source=str(root), files=files)
    publication_path = reference / 'PUBLICATION.json'
    publication_path.write_text(json.dumps(publication, indent=2) + '\n')
    # Existing f4 template supplies all stream clocks and the real door.
    deck = template
    deck.update(id='p10-sf-mynn-' + args.suite + '-' + root.parent.name + '-' + root.name,
                role='scheme', suite=args.suite, process_count=1, ranks=1)
    init_name = manifest['runtime']['init']
    deck['inputs'] = {
        'namelist': {'path': str(reference / 'namelist.atmosphere'),
                     'sha256': manifest['namelist_sha256']},
        'streams': {'path': str(reference / 'streams.atmosphere'),
                    'sha256': manifest['streams_sha256']},
        'init': {'path': str((root / init_name).resolve(strict=True)),
                 'sha256': manifest['inputs'][init_name]},
    }
    excluded = {'atmosphere_model', 'namelist.atmosphere', 'streams.atmosphere', init_name}
    deck['auxiliary_inputs'] = {
        name: {'path': str((root / name).resolve(strict=True)), 'sha256': sha}
        for name, sha in manifest['inputs'].items() if name not in excluded
    }
    deck['native'] = {'path': str(publication_path), 'sha256': digest(publication_path)}
    command = deck['hex_command']
    if '--physics' not in command:
        command += ['--physics', "config_sfclayer_backend='hex-cuda'"]
    # Rust reads every native record clock; templates cannot silently truncate
    # or substitute the cadence of a different scheme's native reference.
    from hexcore.verify_mpas import published_native_manifest
    native = published_native_manifest(publication_path, publication,
                                       args.deck.parent, deck, args.comparator)
    samples = {}
    for output in native['outputs']:
        samples.setdefault(output['stream'], []).extend(output['steps'])
    deck['expected_samples'] = {key: sorted(values) for key, values in samples.items()}
    deck['duration_steps'] = max(step for values in samples.values() for step in values)
    args.deck.write_text(json.dumps(deck, indent=2) + '\n')
    print(json.dumps({'deck': str(args.deck.resolve()), 'deck_sha256': digest(args.deck),
                      'native_binary_sha256': manifest['binary_sha256'],
                      'process_count': 1, 'native_outputs': len(manifest['outputs']),
                      'publication_sha256': digest(publication_path),
                      'forecast_identity_checked': False}))


if __name__ == '__main__':
    main()
