"""Retain actual output inodes until the required Rust difference sheets exist.

This watches filenames only; MPAS decoding and rendering remain Rust work.
"""
import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path


def retain(deck, source, target, stopped, parent):
    data = json.loads(deck.read_text())
    start = datetime.datetime.strptime(data['date'], '%Y-%m-%d_%H:%M:%S')
    names = ['output.' + (start + datetime.timedelta(hours=lead)).strftime(
        '%Y-%m-%d_%H.%M.%S') + '.nc' for lead in (6, 12, 24)]
    target.mkdir(parents=True, exist_ok=False)
    while True:
        for name in names:
            if (target / name).exists():
                continue
            try:
                os.link(source / name, target / name)
            except FileNotFoundError:
                pass
        if stopped.exists():
            break
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    # The run door has closed its files before the caller signals stopped.
    files = {}
    for name in names:
        path = target / name
        if path.exists():
            with path.open('rb') as stream:
                files[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (target / 'retention.json').write_text(json.dumps(dict(
        schema='p3-sheet-output-retention/v1', acceptance=False,
        deck_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),
        source=str(source), files=files, missing=sorted(set(names) - set(files)),
        scope='Actual run output inodes; no weather decoding or modification.'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('deck', 'source', 'target', 'stopped'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--parent', type=int, required=True)
    retain(**vars(parser.parse_args()))
