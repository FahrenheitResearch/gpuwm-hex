"""Check sealed decks through the staged verifier, without running forecasts.

This validates real stored-word references and input bindings only; it never
certifies model execution or a column result as whole-model acceptance.
"""
import argparse
import json
from pathlib import Path
from hexcore import verify_mpas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    args = parser.parse_args()
    comparator = verify_mpas.comparator_path(args.comparator.resolve())
    rows = []
    for path in sorted(args.directory.glob('*-deck.json')):
        deck = verify_mpas.read_json(path)
        verify_mpas.identity(deck)
        verify_mpas.process_count(deck['process_count'])
        assert deck['process_count'] == 1
        assert set(deck['inputs']) == {'namelist', 'streams', 'init'}
        primary_names = {'namelist.atmosphere', 'streams.atmosphere',
                         Path(deck['inputs']['init']['path']).name}
        for name in deck['auxiliary_inputs']:
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert name not in primary_names, f'Auxiliary replaces a primary input: {name}'
        for ref in (*deck['inputs'].values(), *deck['auxiliary_inputs'].values()):
            verify_mpas.file_ref(ref, path.parent)
        native_path, native = verify_mpas.native_manifest(deck['native'], path.parent, deck, comparator)
        verify_mpas.validate_model_command(deck['hex_command'])
        rows.append(dict(deck=path.name, deck_sha256=verify_mpas.digest(path),
                         output_files=len(native['outputs']), process_count=1,
                         native_manifest_sha256=verify_mpas.digest(native_path),
                         forecast_executed=False))
        print(json.dumps(rows[-1]), flush=True)
    assert rows, 'No sealed forecast decks supplied'
    (args.directory/'VALIDATION.json').write_text(json.dumps(dict(
        schema='p11-verification-deck-validation/v1', all_decks_valid=True,
        forecast_executed=False, comparator_sha256=verify_mpas.digest(comparator),
        entries=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
