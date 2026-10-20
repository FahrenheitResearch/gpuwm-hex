#!/usr/bin/env python3
"""Authenticate complete native recipe witnesses for every generated interface.

This is an evidence inventory, not forecast or hidden-state acceptance. It
keeps cumulative native coverage and compiler differences explicitly separate.
"""
import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from column_oracle_replay import read_replay_receipt

RECIPES = {'gnu-strict', 'gnu-o3', 'ifx-o3', 'ifx-valuesafe'}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def arrays(receipt):
    result = {}
    for capture in receipt['captures']:
        for name, value in capture['arrays'].items():
            key = capture['capture'] + ':' + name
            if key in result:
                raise ValueError('duplicate capture/array in replay')
            result[key] = value
    if not result:
        raise ValueError('empty native replay evidence')
    return result


def audit_site(paths, spread_path, process_witness=None):
    receipts = {recipe: read_replay_receipt(path) for recipe, path in paths.items()}
    if set(receipts) != RECIPES | {'coverage'}:
        raise ValueError('five actual native recipe receipts required')
    witness = None
    if process_witness:
        path = Path(process_witness['path'])
        if sha(path) != process_witness['sha256']:
            raise ValueError('legacy process-count witness changed')
        witness = read(path)
        if witness.get('replay_process_count') != 1:
            raise ValueError('legacy batch does not witness one process')
    for recipe, receipt in receipts.items():
        count = receipt.get('replay_process_count')
        if count is None and witness:
            matched = [member for site in witness['sites'] if site['site'] == receipt['scheme']
                       for member in site['replays'] if member['recipe'] == recipe]
            if (len(matched) != 1 or
                    Path(matched[0].get('receipt', matched[0].get('reused_receipt', ''))).resolve()
                    != paths[recipe].resolve()):
                raise ValueError('legacy process witness does not bind this replay')
            count = witness['replay_process_count']
        if (receipt.get('replay_recipe') != recipe or
                receipt.get('backend') != 'native-replay' or
                not receipt.get('comparison_complete') or
                count != 1):
            raise ValueError('incomplete or differently ranked native replay')
    for key in ['scheme', 'dump_root_sha256', 'pin_set_sha256', 'source_sha256']:
        if len({json.dumps(r[key], sort_keys=True) for r in receipts.values()}) != 1:
            raise ValueError('native recipe receipts disagree: ' + key)
    data = {recipe: arrays(receipt) for recipe, receipt in receipts.items()}
    if any(set(a) != set(data['gnu-strict']) for a in data.values()):
        raise ValueError('native recipe array inventories disagree')
    for recipe in ['gnu-strict', 'coverage']:
        if (not receipts[recipe].get('all_outputs_bitwise_equal') or
                any(a['different_words'] or not a['bitwise_equal']
                    for a in data[recipe].values())):
            raise ValueError('strict/coverage native outputs differ')
    spread = read(spread_path)
    for key in ['scheme', 'dump_root_sha256', 'pin_set_sha256', 'source_sha256']:
        if spread[key] != receipts['gnu-strict'][key]:
            raise ValueError('spread belongs to different native inputs: ' + key)
    members = spread['receipts']
    if len(members) != 4 or {m['recipe'] for m in members} != RECIPES:
        raise ValueError('four native spread members required')
    for member in members:
        recipe = member['recipe']
        if (sha(paths[recipe]) != member['sha256'] or
                receipts[recipe]['executable_sha256'] != member['executable_sha256']):
            raise ValueError('spread replay hash changed')
    combined = dict(spread['arrays'])
    shards = []
    for shard in spread.get('array_shards', []):
        path = spread_path.parent / shard['file']
        if path.resolve().parent != spread_path.parent.resolve() or sha(path) != shard['sha256']:
            raise ValueError('spread shard missing, changed, or outside receipt directory')
        payload = read(path)['arrays']
        if set(payload) & set(combined):
            raise ValueError('duplicate spread shard arrays')
        combined.update(payload)
        shards.append(dict(path=str(path), sha256=sha(path)))
    if (set(combined) != set(data['gnu-strict']) or
            len(combined) != spread.get('array_count', len(combined))):
        raise ValueError('spread omits native arrays')
    required_pairs = {frozenset(p) for p in itertools.combinations(RECIPES, 2)}
    for value in combined.values():
        pairs = value['pairs']
        if len(pairs) != 6 or {frozenset(p['recipes']) for p in pairs} != required_pairs:
            raise ValueError('spread does not witness all six native recipe pairs')
    original = receipts['gnu-strict']
    return dict(site=original['scheme'], dump_root_sha256=original['dump_root_sha256'],
                pin_set_sha256=original['pin_set_sha256'],
                source_sha256=original['source_sha256'], array_records=len(combined),
                output_words=sum(a['words'] for a in data['gnu-strict'].values()),
                strict_differing_words=0, coverage_differing_words=0,
                replay_process_count=1,
                recipes={recipe: dict(receipt=str(paths[recipe]), sha256=sha(paths[recipe]),
                    executable_sha256=r['executable_sha256'],
                    different_words=sum(a['different_words'] for a in data[recipe].values()),
                    max_ulp=max(a['max_ulp'] or 0 for a in data[recipe].values()))
                    for recipe, r in receipts.items()},
                process_count_witness=process_witness,
                spread=dict(receipt=str(spread_path), sha256=sha(spread_path), shards=shards,
                    purpose='debugging-only',proof_bar=False,replay_process_count=1,
                    largest_pairwise_differing_words=max(v['largest_pairwise_differing_words'] for v in combined.values()),
                    largest_pairwise_max_ulp=max(v['largest_pairwise_max_ulp'] for v in combined.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True,
                        help='generated build manifest plus per-site five-recipe/spread paths')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    build_path = Path(plan['generated_build_manifest'])
    expected = {name.removeprefix('replay_').removesuffix('.F90')
                for name in read(build_path)['sources'] if name.startswith('replay_')}
    if not expected or {s['site'] for s in plan['sites']} != expected:
        raise ValueError('matrix does not cover every generated native interface')
    rows = []
    for site in plan['sites']:
        row = audit_site({r: Path(p) for r, p in site['recipes'].items()}, Path(site['spread']),
                         site.get('process_count_witness'))
        if row['site'] != site['site']:
            raise ValueError('matrix site is mislabeled')
        rows.append(row)
    coverage = []
    for path in map(Path, plan.get('coverage_indexes', [])):
        index = read(path)
        for entry in index['files']:
            receipt_path = Path(entry['receipt'])
            if sha(receipt_path) != entry['sha256']:
                raise ValueError('native coverage receipt hash changed')
        coverage.append(dict(receipt=str(path), sha256=sha(path)))
    result = dict(schema='mpas-native-interface-replay-matrix-v1',
                  generated_build_manifest_sha256=sha(build_path), plan_sha256=sha(args.plan),
                  process_count=1, interfaces=len(rows), sites=rows, coverage=coverage,
                  coverage_complete=False, hidden_state_complete=False,
                  forecast_proven=False, proof_bar=False)
    payload = (json.dumps(result, indent=2) + '\n').encode()
    if len(payload) >= 10_000_000:
        raise ValueError('matrix receipt exceeds delivery size limit')
    args.out.write_bytes(payload)


if __name__ == '__main__':
    main()
