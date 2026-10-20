"""Seal native sample coverage and the independent foundation omission grant.

Native keys and word counts come from prior Rust reads of pinned native files.
No omission status from a lane's candidate is used to grant missing coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--native-index', type=Path, required=True)
    parser.add_argument('--native-prefix', required=True)
    parser.add_argument('--foundation', type=Path, required=True)
    parser.add_argument('--out-pins', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    old = json.loads(args.native_index.read_text())
    foundation = json.loads(args.foundation.read_text())
    assert foundation['shared_stage_acceptance'] is True
    assert foundation['all_written_numerical_words_identical'] is True
    pins = dict(schema='p10-native-inventory-pins-v1',
                index_sha256=digest(args.index),
                foundation={'path': str(args.foundation), 'sha256': digest(args.foundation)},
                decks={})
    provider_path = args.index.parent / index['native_provider_receipt']
    pins['provider_receipt_sha256'] = digest(provider_path)
    pins['providers'] = json.loads(provider_path.read_text())['providers']
    key = lambda r: (r['stream'], r['field'], r['step'], r['time'])
    for row in index['decks']:
        deck_path = Path(row['path'])
        assert digest(deck_path) == row['sha256']
        deck = json.loads(deck_path.read_text())
        previous = next(r for r in old['decks'] if
                        (r['mesh'], r['date'], r['suite']) ==
                        (row['mesh'], row['date'], row['suite']))
        previous_path = Path(previous['path'])
        native_path = args.index.parent / (args.native_prefix + '-' + previous_path.stem + '.json')
        native = json.loads(native_path.read_text())
        comparator_sha = native['provenance']['comparator_sha256']
        assert pins.setdefault('comparator_sha256', comparator_sha) == comparator_sha
        # This is native inventory authority, never candidate forecast identity.
        assert native['provenance']['deck_sha256'] == digest(previous_path)
        assert native['provenance']['native_manifest_sha256'] == deck['native']['sha256']
        grant = next(r for r in foundation['decks'] if
                     r['mesh'] == deck['mesh'] and r['date'] == deck['date'] and
                     r['suite'] == 'convection_permitting')
        grant_path = args.foundation.parent.parent / grant['receipt']['path']
        assert digest(grant_path) == grant['receipt']['sha256']
        published = json.loads(grant_path.read_text())
        assert published['completed'] is True and published['numeric_differing_words'] == 0
        baseline = {key(r): r for r in published['comparison']['comparison']['fields']}
        rows = native['comparison']['fields']
        assert len({key(r) for r in rows}) == len(rows)
        samples = []
        for sample in rows:
            reference = baseline[key(sample)]
            assert sample['compared_words'] == reference['compared_words']
            assert reference['status'] in ('compared', 'field missing in hex')
            samples.append({**{name: sample[name] for name in ('stream','field','step','time')},
                            'words': sample['compared_words'],
                            'missing_allowed': reference['status'] == 'field missing in hex'})
        inventory = {name: deck[name] for name in
                     ('pin_set','mesh','date','suite','duration_steps','process_count')}
        inventory.update(schema='p10-native-written-inventory-v1', samples=samples,
                         native_manifest_sha256=deck['native']['sha256'],
                         native_coverage_source={'path':str(native_path), 'sha256':digest(native_path),
                            'scope':'native sample keys/counts only; candidate status is unused'},
                         missing_authority={'path':str(grant_path),'sha256':digest(grant_path),
                            'scope':'independently sealed f4 foundation omissions'})
        path = args.index.parent / (deck_path.stem + '-native-inventory.json')
        path.write_text(json.dumps(inventory, indent=2) + '\n')
        pins['decks'][digest(deck_path)] = {'inventory':path.name, 'sha256':digest(path)}
    args.out_pins.write_text(json.dumps(pins, indent=2) + '\n')


if __name__ == '__main__':
    main()
