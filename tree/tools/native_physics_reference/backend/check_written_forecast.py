#!/usr/bin/env python3
"""Check the integrator's M0 written-field scope without changing verify-mpas.

MPAS authority: mpas_atmphys_driver.F:202-373. This is receipt orchestration,
not a weather decoder or numerical implementation. Missing samples remain in
the original verifier and ratchet. A sealed inventory permits only the missing
fields specifically deferred by the integrator; every computed word must match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sample_key(row):
    return row['stream'], row['field'], row['step'], row['time']


def check(receipt, completion, deck, inventory, *, deck_sha256):
    """Return scoped evidence; reject incomplete or incompatible comparisons."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(receipt.get('model_execution') is True and
            receipt.get('evidence_kind') == 'FORECAST' and
            not receipt.get('failed_model_exit'), 'actual forecast execution failed')
    require(all(type(x) is int for x in (receipt.get('process_count'),
                                       deck['process_count'], inventory['process_count'])) and
            receipt.get('process_count') == deck['process_count'] ==
            inventory['process_count'] == 1, 'process count must be fixed at one')
    for name in ('pin_set', 'mesh', 'date'):
        require(receipt.get(name) == deck[name] == inventory[name],
                f'{name} differs from sealed native inventory')
    require(deck['suite'] == inventory['suite'] == completion['physics']['suite'],
            'suite differs from sealed native inventory')
    require(completion.get('completed') is True and
            completion.get('steps_requested') == completion.get('steps_completed') ==
            deck['duration_steps'] == inventory['duration_steps'],
            'complete forecast duration not witnessed')
    require(completion['native_library_sha256'] ==
            deck['hex_support_inputs']['native-library']['sha256'],
            'completion uses a different native library')
    require(receipt['provenance']['deck_sha256'] == deck_sha256,
            'verifier did not execute this deck')
    require(receipt['provenance']['native_manifest_sha256'] ==
            inventory['native_manifest_sha256'], 'native manifest differs')
    require(deck['native']['sha256'] == inventory['native_publication_sha256'],
            'native publication differs')
    rows = receipt['comparison']['fields']
    expected = {sample_key(row): row for row in inventory['samples']}
    actual = {sample_key(row): row for row in rows}
    require(len(expected) == len(inventory['samples']) and len(actual) == len(rows),
            'duplicate comparison samples')
    require(actual.keys() == expected.keys(), 'comparison sample coverage differs')
    computed_words = missing_words = missing_samples = computed_samples = 0
    for key, row in actual.items():
        reference = expected[key]
        require(row['compared_words'] == reference['words'],
                f'{key}: native word count differs')
        require(row['step'] in deck['expected_samples'][row['stream']],
                f'{key}: unexpected output step')
        if row['status'] == 'compared':
            require(row['differing_words'] == 0, f'{key}: computed words differ')
            computed_samples += 1
            computed_words += row['compared_words']
        else:
            require(row['status'] == 'field missing in hex' and
                    reference['missing_allowed'] is True,
                    f'{key}: newly missing or incompatible field')
            missing_samples += 1
            missing_words += row['compared_words']
    require(computed_samples > 0 and computed_words > 0, 'no computed evidence')
    expected_failures = {
        f"{r['stream']}/{r['field']} step {r['step']}: field missing in hex"
        for r in rows if r['status'] != 'compared'
    }
    require(set(receipt.get('failures', [])) == expected_failures,
            'verifier has a failure outside deferred missing fields')
    return dict(schema='mpas-written-forecast-scope-v1', passed=True,
                scope='integrator M0: every written field bitwise; deferred fields inventoried',
                process_count=1, duration_steps=deck['duration_steps'],
                computed_samples=computed_samples, computed_words=computed_words,
                computed_differing_words=0, missing_samples=missing_samples,
                missing_words=missing_words, official_passed=receipt['passed'],
                official_tier=receipt['tier'], official_receipt_modified=False,
                full_field_bitwise=missing_samples == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        if digest(args.inventory) != args.inventory_sha256:
            raise ValueError('sealed inventory hash differs')
        result = check(*(json.loads(p.read_text()) for p in
                         (args.receipt, args.completion, args.deck, args.inventory)),
                       deck_sha256=digest(args.deck))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(schema='mpas-written-forecast-scope-v1', passed=False,
                      failure=str(error), official_receipt_modified=False)
    result['inputs'] = {name: {'path': str(getattr(args, name)),
                              'sha256': digest(getattr(args, name))}
                       for name in ('receipt', 'completion', 'deck', 'inventory')
                       if getattr(args, name).is_file()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
