"""Restore authentic native bytes and check the actual f2 reference protocol.

This creates a native-only diagnostic publication, never a CUDA comparison or
forecast acceptance receipt. Restored output can later supply the actual Rust
word comparator instead of the unusable digest-only baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archives', 'native-directory', 'output', 'template-deck', 'comparator', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reuse-restored', action='store_true',
                        help='authenticate an existing owned restoration without duplicating output bytes')
    parser.add_argument('--case', type=Path,
                        help='bind this exact published case instead of the template case')
    args = parser.parse_args()
    archive = json.loads(args.archives.read_text())
    source = args.native_directory.resolve()
    manifest = json.loads((source / 'MANIFEST.json').read_text())
    assert archive['complete'] and archive['process_count'] == manifest['ranks'] == 1
    assert sha(source / 'MANIFEST.json') == archive['native_manifest_sha256']
    assert {k: v['sha256'] for k, v in archive['files'].items()} == manifest['outputs']
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.reuse_restored)
    for name, row in archive['files'].items():
        assert Path(name).name == name and name.endswith('.nc')
        blob = Path(row['archive'])
        assert sha(blob) == row['archive_sha256'] and row['restored_all_bytes_verified']
        if not args.reuse_restored or not (output / name).exists():
            subprocess.run(['tar', '-I', 'zstd', '-xf', str(blob), '-C', str(output), name], check=True)
        assert sha(output / name) == row['sha256']
    for name in ('MANIFEST.json', 'CLOCKS.json', 'namelist.atmosphere',
                 'streams.atmosphere', 'log.atmosphere.0000.out'):
        shutil.copy2(source / name, output / name)
    assert sha(output / 'namelist.atmosphere') == manifest['namelist_sha256']
    assert sha(output / 'streams.atmosphere') == manifest['streams_sha256']
    publication = dict(schema='mpas.native-cache-publication/v1',
                       description='p21 restored strict native option; native-only diagnostic',
                       source=str(source), selection=['actual archived native outputs and metadata'],
                       files={p.name: sha(p) for p in sorted(output.iterdir())
                              if p.name != 'PUBLICATION.json'})
    publication_path = output / 'PUBLICATION.json'
    publication_path.write_text(json.dumps(publication, indent=2) + '\n')
    deck = json.loads(args.template_deck.read_text())
    deck['id'] = 'p21-native-byte-archive-reference-diagnostic'
    deck['inputs']['namelist'] = dict(path=str(output / 'namelist.atmosphere'),
                                    sha256=manifest['namelist_sha256'])
    deck['inputs']['streams'] = dict(path=str(output / 'streams.atmosphere'),
                                   sha256=manifest['streams_sha256'])
    if args.case:
        case = args.case.resolve(strict=True)
        case_info = json.loads((case / 'MANIFEST.json').read_text())['case']
        inputs = [s for s in ET.parse(output / 'streams.atmosphere').getroot()
                  if s.get('name') == 'input']
        assert len(inputs) == 1
        init = case / inputs[0].get('filename_template')
        assert sha(init) == manifest['inputs'][init.name]
        deck['inputs']['init'] = dict(path=str(init), sha256=sha(init))
        deck.update(mesh=case_info['mesh'], date=case_info['start'],
                    process_count=manifest['ranks'], pin_set=manifest['pin_set_hash'])
        # Bind actual native run inputs, never reuse another mesh's graph or
        # static table references from the diagnostic template.
        excluded = {'namelist.atmosphere', 'streams.atmosphere',
                    'atmosphere_model', init.name}
        deck['auxiliary_inputs'] = {}
        for name, digest in manifest['inputs'].items():
            if name in excluded:
                continue
            original = (source / name).resolve(strict=True)
            assert sha(original) == digest
            deck['auxiliary_inputs'][name] = dict(path=str(original), sha256=digest)
    # Native reference validation only. No model is called and no native
    # output is passed off as a hex forecast.
    from hexcore.verify_mpas import native_manifest
    deck['native'] = dict(path=str(publication_path), sha256=sha(publication_path))
    _, native = native_manifest(deck['native'], args.template_deck.parent,
                                deck, args.comparator)
    result = dict(purpose='native byte-archive/f2 reference protocol diagnostic',
                  process_count=1, native_binary_sha256=manifest['binary_sha256'],
                  archive_receipt_sha256=sha(args.archives),
                  publication_path=str(publication_path), publication_sha256=sha(publication_path),
                  actual_f2_module_sha256=sha(Path(__import__('hexcore.verify_mpas', fromlist=['']).__file__)),
                  comparator_sha256=sha(args.comparator),
                  native_output_files=len(native['outputs']),
                  native_output_records=sum(len(r['steps']) for r in native['outputs']),
                  restored_all_bytes_verified=True, actual_f2_reference_validation=True,
                  validation_entry='native_manifest (publication, every byte, Rust clocks and complete stream inventory)',
                  helper_sha256=sha(Path(__file__)),
                  input_identity={key: ref['sha256'] for key, ref in deck['inputs'].items()},
                  auxiliary_input_identity={key: ref['sha256'] for key, ref in deck.get('auxiliary_inputs', {}).items()},
                  mesh=deck['mesh'], date=deck['date'],
                  model_execution=False, cuda_forecast_identity_claim=False)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
