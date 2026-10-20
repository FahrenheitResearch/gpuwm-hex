"""Isolate authenticated original forecast packets from interrupted copies.

MPAS mpas_atmphys_driver_seaice.F:160-558 supplies the lane boundary.
The original publication and every selected file keep their bytes. Only
publication-listed files are copied into this private packet; a temporary
rsync file is not a native output. Python does no weather decoding.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sealed(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.decks.glob('*.deck.json')):
        deck = json.loads(path.read_text())
        publication = Path(deck['native']['path'])
        assert sealed(publication)['sha256'] == deck['native']['sha256']
        original = publication.parent
        destination = (args.references / original.name).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        names = list(json.loads(publication.read_text())['files']) + ['PUBLICATION.json']
        for name in names:
            source, target = original / name, destination / name
            assert source.is_file() and not source.is_symlink()
            assert Path(name).name == name
            if not target.exists():
                os.link(source, target)
            # Identity of the existing immutable byte packet needs no decode
            # and no second multi-gigabyte copy. Verification checks all hashes.
            assert source.stat().st_dev == target.stat().st_dev
            assert source.stat().st_ino == target.stat().st_ino
        assert {p.name for p in destination.iterdir()} == set(names)
        deck['native']['path'] = str(destination / 'PUBLICATION.json')
        for key, name in (('namelist', 'namelist.atmosphere'), ('streams', 'streams.atmosphere')):
            deck['inputs'][key]['path'] = str(destination / name)
        target = args.out / path.name
        target.write_text(json.dumps(deck, indent=2) + '\n')
        rows.append(dict(deck=sealed(target), native=deck['native'],
                         source_publication=sealed(publication),
                         publication_selected_files=len(names), same_file_inodes=True,
                         process_count=deck['process_count'], acceptance_tier=None))
    args.receipt.write_text(json.dumps(dict(
        kind='isolated authentic native reference bytes; no forecast grade',
        references=rows), indent=2) + '\n')
    print(len(rows), 'original native packets isolated; original hashes unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('decks', 'references', 'out', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
