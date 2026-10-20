"""Check this lane's frozen citation windows against pinned MPAS v8.4.2.

This supplements the base's v8.2.3-only checker; it changes no shared pins.
SOURCE is the read-only src/core_atmosphere/physics directory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

RANGE = r'\d+(?:\s*-\s*\d+)?'
CITATION = re.compile(rf'(?P<file>(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:F90|F)):'
                      rf'(?P<ranges>{RANGE}(?:\s*,\s*{RANGE})*)')
# Corrected referee windows belong to our committed derivative, not pristine MPAS.
REFEREE_CITATION = re.compile(
    rf'(?P<file>correct_referee/[A-Za-z0-9_.-]+\.(?:F90|F)) lines '
    rf'(?P<ranges>{RANGE}(?:\s*,\s*{RANGE})*)')


def occurrences(text):
    rows = Counter()
    for match in (*CITATION.finditer(text), *REFEREE_CITATION.finditer(text)):
        for window in match['ranges'].split(','):
            parts = [int(v) for v in window.split('-')]
            first, last = parts[0], parts[-1]
            if first < 1 or last < first:
                raise ValueError('invalid citation range')
            rows[(match['file'], first, last)] += 1
    return rows


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('receipt', type=Path)
    a = p.parse_args()
    support = Path(__file__).resolve().parent
    module = support.parents[1] / 'src/hexcore/mpasphys/sfclay_mm5_v842.py'
    pins = json.loads((support / 'citation_pins_v842.json').read_text())
    expected = Counter({(v['citation_file'], v['first'], v['last']): v['occurrences']
                        for v in pins['windows']})
    if occurrences(module.read_text()) != expected:
        raise ValueError('module citations differ from frozen windows')
    for path, digest in pins['source_sha256'].items():
        if sha((a.source / path).read_bytes()) != digest:
            raise ValueError('pinned MPAS source changed: ' + path)
    for path, digest in pins.get('referee_sha256', {}).items():
        if sha((support / 'rp3' / path).read_bytes()) != digest:
            raise ValueError('pinned RP3 referee source changed: ' + path)
    for v in pins['windows']:
        root = support / 'rp3' if v['source_path'].startswith('correct_referee/') else a.source
        lines = (root / v['source_path']).read_bytes().splitlines(keepends=True)
        if v['last'] > len(lines) or sha(b''.join(lines[v['first']-1:v['last']])) != v['window_sha256']:
            raise ValueError('cited source window changed: ' + v['citation_file'])
    receipt = dict(status='PASS', release='v8.4.2', windows=len(pins['windows']),
                   occurrences=sum(expected.values()), module_sha256=sha(module.read_bytes()),
                   pins_sha256=sha((support / 'citation_pins_v842.json').read_bytes()),
                   source_sha256=pins['source_sha256'],
                   scope='This lane only; shared v8.2.3 citation checker unchanged')
    a.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print('PASS', receipt['windows'], 'frozen source windows', receipt['occurrences'], 'citations')


if __name__ == '__main__':
    main()
