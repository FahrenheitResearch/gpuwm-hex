"""Bind original strict suite publications to seaice-only CUDA forecasts.

MPAS mpas_atmphys_driver_seaice.F:160-558 is the port boundary.
All NetCDF inventory/clock decoding uses the foundation Rust comparator.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def seal(path):
    return dict(path=str(path.resolve()), sha256=sha(path))


def prepare(args):
    from hexcore.verify_mpas import published_native_manifest
    template = json.loads(args.template.read_text())
    rows = []
    args.out.mkdir(exist_ok=True, parents=True)
    for item in json.loads(args.inventory.read_text())['references']:
        original = Path(item['root'])
        root = args.restored / original.name if item['missing_native_output_files'] else original
        root = root.resolve()
        publication_path = root / 'PUBLICATION.json'
        if not publication_path.is_file():
            continue
        publication = json.loads(publication_path.read_text())
        source = json.loads((root / 'MANIFEST.json').read_text())
        runtime = source['runtime']
        mesh, date = runtime['mesh'], runtime['start']
        case = args.cases / mesh / date.replace(':', '.') / 'init-modis20-v2'
        init = case / runtime['init']
        assert sha(init) == source['inputs'][init.name]
        auxiliary = {}
        for name, digest in source['inputs'].items():
            if name in ('namelist.atmosphere', 'streams.atmosphere', 'atmosphere_model', init.name):
                continue
            candidates = (original / name, root / name, case / name,
                          *(directory / name for directory in args.runtime))
            candidate = next((p for p in candidates if p.is_file() and sha(p) == digest), None)
            assert candidate is not None, name
            auxiliary[name] = seal(candidate)
        deck = dict(template, id='p18-suite-' + original.name,
                    suite=source['deck'], mesh=mesh, date=date,
                    inputs=dict(namelist=seal(root / 'namelist.atmosphere'),
                                streams=seal(root / 'streams.atmosphere'), init=seal(init)),
                    auxiliary_inputs=auxiliary, native=seal(publication_path))
        argv = list(template['hex_command'])
        argv[argv.index('--mesh') + 1] = mesh
        argv[argv.index('--grid') + 1] = str((case / (mesh + '.grid.nc')).resolve())
        deck['hex_command'] = argv
        native = published_native_manifest(publication_path, publication,
                                           args.out.resolve(), deck, args.comparator.resolve())
        expected = {}
        for stream in native['outputs']:
            expected.setdefault(stream['stream'], []).extend(stream['steps'])
        deck['expected_samples'] = {k: sorted(v) for k, v in expected.items()}
        deck['duration_steps'] = max(step for steps in expected.values() for step in steps)
        deck['process_count'] = deck['ranks'] = 1
        path = args.out / (original.name + '.deck.json')
        path.write_text(json.dumps(deck, indent=2) + '\n')
        rows.append(dict(deck=seal(path), native=seal(publication_path),
                         native_complete=True, process_count=1,
                         samples=sum(map(len, expected.values())), cuda_forecast_proven=False))
        args.receipt.write_text(json.dumps(dict(
            kind='authenticated suite deck metadata; not CUDA acceptance',
            process_count=1, decks=rows), indent=2) + '\n')
        print(json.dumps(dict(deck=deck['id'], samples=rows[-1]['samples'])), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('template', 'inventory', 'restored', 'cases', 'comparator', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--runtime', type=Path, action='append', default=[])
    raise SystemExit(prepare(parser.parse_args()))
