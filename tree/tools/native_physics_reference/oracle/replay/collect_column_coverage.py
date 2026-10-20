#!/usr/bin/env python3
"""Collect native coverage from complete, manifest-bound column-bank replays.

This consumes sealed comparison metadata, not a forecast capture index. Packet
bytes were authenticated by the replay; every present POST record must still
appear in its comparison receipt. Coverage is never a producer passivity claim.
"""
import argparse
import json
from pathlib import Path

from collect_archived_coverage import collect_verified
from reproduce_archived_replay import authority, sha
from run_capture_matrix import harness


def validate_column(replay, index, manifest, dependency_root=None):
    data = authority(index, dependency_root)
    measured = json.loads((replay / 'summary.json').read_text())
    bank = json.loads(manifest.read_text())
    native = json.loads((Path(data['build']) / 'oracle-inventory.json').read_text())
    if (native['reference'] != bank['reference']
            or native['pin_set_sha256'] != bank['pin_set_sha256']
            or any(native['source_sha256'].get(name) != expected
                   for name, expected in bank['source_sha256'].items())):
        raise ValueError('column bank native source authority differs')
    if (data.get('recipe') != 'coverage' or measured.get('recipe') != 'coverage'
            or not measured.get('comparison_complete') or measured['process_count'] != 1
            or measured['differing_words'] or measured['input_index_sha256'] != sha(index)
            or measured['manifest_sha256'] != sha(manifest)
            or measured['reference'] != bank['reference']
            or measured['calls'] != len(bank['captures'])
            or measured['audit_sha256'] != sha(data['audit'])):
        raise ValueError('complete manifest-bound coverage replay required')
    captures = [row['capture'] for row in measured['cases']]
    if len(set(captures)) != len(captures) or set(captures) != set(bank['captures']):
        raise ValueError('column coverage call inventory differs')
    words = arrays = 0
    for row in measured['cases']:
        capture = row['capture']
        if Path(capture).name != capture or capture in ('.', '..'):
            raise ValueError('column coverage capture escapes replay directory')
        path = replay / (capture + '.json')
        if row['exit'] or row['differing_words'] or sha(path) != row['receipt_sha256']:
            raise ValueError('column coverage comparison changed or differs')
        receipt = harness.read_replay_receipt(path)
        if (not receipt['comparison_complete'] or not receipt['all_outputs_bitwise_equal']
                or receipt['reference'] != bank['reference']
                or receipt['dump_root_sha256'] != bank['dump_root_sha256']
                or receipt['pin_set_sha256'] != bank['pin_set_sha256']
                or receipt['source_sha256'] != bank['source_sha256']
                or receipt['replay_recipe'] != 'coverage'
                or receipt['replay_process_count'] != 1
                or receipt['backend'] != 'native-replay'
                or receipt['resolved_scheme'] != bank['captures'][capture]['site']
                or len(receipt['captures']) != 1
                or receipt['captures'][0]['capture'] != capture):
            raise ValueError('column coverage comparison authority differs')
        expected = {name[len(capture + '.post.'):-4] for name in bank['files']
                    if name.startswith(capture + '.post.') and name.endswith('.bin')}
        values = receipt['captures'][0]['arrays']
        if not expected or set(values) != expected:
            raise ValueError('column coverage POST inventory differs')
        if any(v['different_words'] or not v['bitwise_equal'] for v in values.values()):
            raise ValueError('column coverage output words differ')
        count = sum(v['words'] for v in values.values())
        if count != row['compared_words']:
            raise ValueError('column coverage word total differs')
        words += count
        arrays += len(values)
    if words != measured['compared_words'] or arrays != measured['arrays']:
        raise ValueError('column coverage aggregate differs')
    counters = measured.get('coverage_counters_sha256', {})
    if not counters:
        raise ValueError('column coverage has no runtime counters')
    for relative, expected in counters.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'profiles':
            raise ValueError('column coverage counter escapes replay directory')
        if sha(replay / path) != expected:
            raise ValueError('column coverage counter changed')
    if set(counters) != {str(p.relative_to(replay)) for p in (replay / 'profiles').rglob('*.gcda')}:
        raise ValueError('column coverage runtime counter inventory differs')
    # Same native source mapping; these are actual column calls, not relabelled
    # archive cases or forecast producers.
    data = dict(data, sites=sorted({r['site'] for r in bank['captures'].values()}))
    measured = dict(measured, cases=[dict(id=c) for c in captures])
    return data, measured


def collect(replay, index, manifest, source, out, dependency_root=None):
    data, measured = validate_column(replay, index, manifest, dependency_root)
    collect_verified(data, measured, replay, index, source, out, replay / 'summary.json')
    result = json.loads((out / 'summary.json').read_text())
    result.update(schema='real-native-column-replay-coverage-v1', manifest_sha256=sha(manifest),
                  column_collector_sha256=sha(Path(__file__)),
                  producer_passivity_proven=False, native_forecast_crosshost_identity_proven=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'index', 'manifest', 'source', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path)
    raise SystemExit(collect(**vars(parser.parse_args())))
