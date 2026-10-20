"""Restore authenticated f0 forecast bytes and prepare the full proof matrix.

Only opaque copies/hashes and subprocess orchestration occur in Python.
f0's archive restorer verifies every file; f2's Rust tool inventories records.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def prepare(args):
    for row in json.loads(args.matrix.read_text()):
        root = args.output / row['key']
        native = root / 'native'
        native.mkdir(parents=True, exist_ok=False)
        for source in Path(row['metadata']).iterdir():
            shutil.copy2(source, native / source.name)
        subprocess.run([sys.executable, str(args.restorer), 'extract',
                        str(native / 'ARCHIVE.json'), '--out', str(native)], check=True)
        deck = args.decks / row['mesh'] / row['date'] / 'suite-mesoscale_reference'
        case = args.cases / row['mesh'] / row['date'] / 'init-modis20-v2'
        subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_verification.py')),
                        '--native', str(native), '--publication', str(native / 'PUBLICATION.json'),
                        '--inputs-root', str(deck), '--case', str(case),
                        '--output', str(root / 'verification'), '--receipt', str(root / 'receipt.json'),
                        '--referee', str(args.referee), '--comparator', str(args.comparator),
                        '--library', str(args.library), '--pool-reader', str(args.pool_reader),
                        '--mpas-source', str(args.mpas_source)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'output', 'restorer', 'decks', 'cases', 'referee',
                 'comparator', 'library', 'pool-reader', 'mpas-source'):
        parser.add_argument('--'+name, type=Path, required=True)
    prepare(parser.parse_args())
