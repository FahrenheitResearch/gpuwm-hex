"""Seal an owned completed strict run for the real whole-stream referee.

Uses f2's published metadata adapter and Rust record inventory. No weather
payload is read or decoded in Python; every output remains in the inventory.
"""
import argparse
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path.resolve()), sha256=digest(path))


def validate_output_inventory(root, entries, inputs, comparator):
    """Check the original run before separating its inputs and outputs."""
    inventory = subprocess.run([str(comparator), '--inventory-root', str(root)],
                               capture_output=True, text=True, check=True)
    observed = {Path(p).resolve() for p in json.loads(inventory.stdout)}
    expected = {(root/entry['path']).resolve() for entry in entries}
    input_paths = {(root/name).resolve() for name in inputs
                   if (root/name).is_file()}
    if observed-input_paths != expected:
        raise ValueError('original native run contains omitted or unexpected output files')


def stage_output_files(root, entries, destination):
    """Keep every sealed output in a directory without native input links."""
    destination.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        relative = Path(entry['path'])
        source = root / relative
        if (relative.is_absolute() or '..' in relative.parts
                or source.is_symlink() or not source.is_file()):
            raise ValueError('sealed native output must be a regular file inside its run')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, target)
    return destination.resolve()


def prepare(args):
    root, output = args.native.resolve(), args.output.resolve()
    input_root = args.inputs_root.resolve() if args.inputs_root else root
    case = args.case.resolve() if args.case else root
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((root/'MANIFEST.json').read_text())
    assert source['ranks'] == 1
    assert source['runtime']['duration'] == '1_00:00:00'
    log = root/'log.atmosphere.0000.out'
    assert 'Finished running the atmosphere core' in log.read_text()
    files = {name: sha for name, sha in source['outputs'].items()}
    files.update({p.name: digest(p) for p in (root/'MANIFEST.json', log)})
    publication = dict(schema='mpas.native-cache-publication/v1', files=files,
                       provenance='owned measurement of unchanged f0 strict executable; not f0 publication')
    if args.publication is not None:
        publication_path = args.publication.resolve()
        assert publication_path.parent == root
        publication = json.loads(publication_path.read_text())
    else:
        publication_path = root/'p7-PUBLICATION.json'
        publication_path.write_text(json.dumps(publication, indent=2)+'\n')
    inputs = dict(namelist=ref(input_root/'namelist.atmosphere'),
                  streams=ref(input_root/'streams.atmosphere'),
                  init=ref(case/source['runtime']['init']))
    auxiliary = {name: ref(input_root/name) for name in source['inputs']
                 if name.startswith('stream_list.')}
    deck = dict(schema='mpas-verification-deck-v1', id=source['deck'],
                mesh=source['runtime']['mesh'], date=source['runtime']['start'],
                pin_set=source['pin_set_hash'], process_count=1, inputs=inputs,
                auxiliary_inputs=auxiliary)
    spec = importlib.util.spec_from_file_location('ntiedtke_f2_metadata_witness', args.referee)
    referee = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(referee)
    manifest = referee.published_native_manifest(publication_path, publication, output, deck, args.comparator)
    validate_output_inventory(root, manifest['outputs'], source['inputs'], args.comparator)
    manifest['output_root'] = str(stage_output_files(root, manifest['outputs'], output/'native-output'))
    manifest_path = output/'native.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    deck['native'] = ref(manifest_path)
    expected = {}
    for entry in manifest['outputs']:
        expected.setdefault(entry['stream'], []).extend(entry['steps'])
    deck.update(expected_samples={key: sorted(value) for key, value in expected.items()},
                duration_steps=max(step for steps in expected.values() for step in steps),
                slot_backends={'convection':'hex-cuda', 'other_active_slots':'native-cpu'})
    mesh = deck['mesh']
    # The verifier validates this public door spelling, then invokes this
    # checkout's CLI module rather than trusting a console script on PATH.
    deck['hex_command'] = ['gpuwm-hex', 'forecast', '--mesh', mesh, '--grid', str(case/(mesh+'.grid.nc')),
                           '--init', '{init}', '--physics-namelist', '{namelist}', '--streams', '{streams}',
                           '--physics-backend', 'mpasphys', '--physics', "config_convection_backend='hex-cuda'",
                           '--hours', '24', '--out', '{out}']
    providers = {name: ref(getattr(args, name)) for name in ('library', 'pool_reader')}
    deck['native_providers'] = providers
    deck['hex_command'] += ['--native-physics-library', providers['library']['path'],
                            '--native-pool-reader', providers['pool_reader']['path'],
                            '--mpas-source', str(args.mpas_source.resolve())]
    for name in ('stream_list_directory', 'native_physics_tables'):
        path = getattr(args, name, None)
        if path is not None:
            deck['hex_command'] += ['--'+name.replace('_', '-'), str(path.resolve())]
    (output/'deck.json').write_text(json.dumps(deck, indent=2)+'\n')
    receipt = dict(schema='mpas-ntiedtke-whole-reference-preparation/v1', acceptance=False,
                   native_process_count=1, native_output_files=len(manifest['outputs']),
                   native_binary_sha256=source['binary_sha256'], expected_samples=deck['expected_samples'],
                   duration_steps=deck['duration_steps'], deck=ref(output/'deck.json'),
                   metadata_adapter=ref(args.referee), comparator=ref(args.comparator),
                   native_providers=providers)
    args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('native', 'output', 'referee', 'comparator', 'receipt',
                 'library', 'pool-reader', 'mpas-source'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--publication', type=Path)
    parser.add_argument('--inputs-root', type=Path)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--stream-list-directory', type=Path)
    parser.add_argument('--native-physics-tables', type=Path)
    prepare(parser.parse_args())
