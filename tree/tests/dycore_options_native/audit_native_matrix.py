"""Audit complete option-deck metadata; this never claims Hex forecast identity.

Native sealing already read every output with f2's actual Rust comparator.
Archives carry measured complete extraction witnesses; reproduction checks their
bytes again before comparison. This audit checks pins, option selection, clocks
and matrix completeness without repeating scientific data processing.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from hexcore.config import parse_fortran_namelist


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned(base, row):
    path = base / row['path']
    if sha(path) != row['sha256']:
        raise ValueError(f'changed metadata artifact: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--scope', choices=('one_step', '24h'), required=True)
    parser.add_argument('--options', default='',
                        help='comma-separated complete option groups; empty audits the full matrix')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    base = args.receipts.resolve()
    index_path = base / 'forecast-reproduction-index.json'
    index = json.loads(index_path.read_text())
    requested = json.loads((base / 'forecast-proof-plan.json').read_text())['decks']
    if len(index['cases']) != len(requested):
        raise ValueError('matrix no longer names every requested deck')
    selected = set(args.options.split(',')) if args.options else {c['name'] for c in requested}
    if selected - {c['name'] for c in requested}:
        raise ValueError('requested option group is absent from the sealed matrix')
    rows = []
    coverage = defaultdict(list)
    for i, entry in enumerate(index['cases']):
        case = entry['case']
        if any(case[k] != requested[i][k] for k in case):
            raise ValueError('case changed from requested matrix')
        if case['name'] not in selected:
            continue
        proof, = [p for p in entry['proofs'] if p['scope'] == args.scope]
        deck_path = pinned(base, proof['deck'])
        deck = json.loads(deck_path.read_text())
        native_path = pinned(base, deck['native'])
        native = json.loads(native_path.read_text())
        if (native['schema'] != 'mpas-native-run-v1' or native['complete'] is not True
                or native['process_count'] != 1 or native['ranks'] != 1
                or native['pin_set'] != deck['pin_set']):
            raise ValueError('native proof scope/pin/process count changed')
        if native['inputs'] != {k: v['sha256'] for k, v in deck['inputs'].items()}:
            raise ValueError('native input pins differ from the deck')
        for k in ('mesh', 'date'):
            if deck[k] != case[k]:
                raise ValueError('deck case identity changed')
        namelist_path = pinned(base, deck['inputs']['namelist'])
        groups = parse_fortran_namelist(namelist_path.read_text())
        values = {k: v for g in groups.values() for k, v in g.items()}
        if values['config_physics_suite'] != 'none':
            raise ValueError('option proof must have physics off')
        if any(values[k] != v for k, v in case['parameters'].items()):
            raise ValueError('native namelist does not select requested option')
        if deck['duration_steps'] * int(values['config_dt']) != deck['duration_seconds']:
            raise ValueError('native clock and declared duration disagree')
        if args.scope == 'one_step' and deck['duration_steps'] != 1:
            raise ValueError('one-step scope changed')
        if args.scope == '24h' and deck['duration_seconds'] != 86400:
            raise ValueError('24-hour scope changed')
        actual = defaultdict(list)
        for output in native['outputs']:
            actual[output['stream']].extend(output['steps'])
        if {k: sorted(v) for k, v in actual.items()} != deck['expected_samples']:
            raise ValueError('native sealed output samples differ from declared streams')
        archive_path = native_path.with_name(native_path.name.replace('-run.json', '-archive.json'))
        archive = json.loads(archive_path.read_text())
        if (archive['schema'] != 'd1-native-byte-archive-v1'
                or archive['native_manifest_sha256'] != sha(native_path)
                or archive['restored_all_bytes_verified'] is not True
                or archive['files'] != {r['path']: r['sha256'] for r in native['outputs']}):
            raise ValueError('archive witness differs from sealed native inventory')
        coverage[case['name']].append((case['mesh'], case['date'], case['state']))
        rows.append(dict(index=i, case=case, deck_sha256=sha(deck_path),
                         native_manifest_sha256=sha(native_path),
                         archive_receipt_sha256=sha(archive_path),
                         native_output_files=len(native['outputs']),
                         native_binary_sha256=native['binary_sha256']))
    requested_coverage = defaultdict(list)
    for case in requested:
        if case['name'] in selected:
            requested_coverage[case['name']].append((case['mesh'], case['date'], case['state']))
    if dict(coverage) != dict(requested_coverage):
        raise ValueError('an option or case is absent')
    result = dict(schema='d1-native-matrix-metadata-v1', scope=args.scope,
                  status='PASS', evidence='authenticated native forecasts only; Hex identity unmeasured',
                  native_process_count=1, cases=len(rows),
                  native_output_files=sum(r['native_output_files'] for r in rows),
                  option_coverage={k: len(v) for k, v in coverage.items()},
                  binary_counts=dict(Counter(r['native_binary_sha256'] for r in rows)),
                  index_snapshot_sha256=sha(index_path), rows=rows)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'scope', 'cases', 'native_output_files')}))


if __name__ == '__main__':
    main()
