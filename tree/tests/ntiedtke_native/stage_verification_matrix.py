"""Stage complete native output inventories for an existing forecast matrix.

This changes only the reference layout. Every recorded native output is retained
with its original digest, stream, record clock and completion provenance.
"""
import argparse
import json
from pathlib import Path

from prepare_verification import digest, ref, stage_output_files, validate_output_inventory


def main(args):
    requested = json.loads(args.matrix.read_text())
    rows = []
    for number, row in enumerate(requested):
        source = Path(row['path']).resolve()
        deck = json.loads(source.read_text())
        native_path = (source.parent / deck['native']['path']).resolve()
        if digest(native_path) != deck['native']['sha256']:
            raise ValueError('source verification native manifest digest differs')
        native = json.loads(native_path.read_text())
        output = args.output / str(number)
        output.mkdir(parents=True, exist_ok=False)
        root = (native_path.parent / native['output_root']).resolve()
        validate_output_inventory(root, native['outputs'],
                                  native['native_input_inventory'], args.comparator)
        native['output_root'] = str(stage_output_files(root, native['outputs'], output/'native-output'))
        target_manifest = output/'native.json'
        target_manifest.write_text(json.dumps(native, indent=2)+'\n')
        deck['native'] = ref(target_manifest)
        for group in ('inputs', 'auxiliary_inputs', 'native_providers'):
            for item in deck.get(group, {}).values():
                item['path'] = str((source.parent/item['path']).resolve())
        target_deck = output/'deck.json'
        target_deck.write_text(json.dumps(deck, indent=2)+'\n')
        rows.append(dict(key=row['key'], path=str(target_deck.resolve()),
                         source=ref(source), source_native=ref(native_path),
                         deck_sha256=digest(target_deck),
                         native_output_files=len(native['outputs'])))
    args.receipt.write_text(json.dumps(dict(
        schema='mpas-ntiedtke-native-output-layout/v1', acceptance=False,
        outputs_omitted=0, original_run_inventory_checked=True,
        native_words_changed=False, rows=rows), indent=2)+'\n')
    args.output_matrix.write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'output', 'output-matrix', 'receipt', 'comparator'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
