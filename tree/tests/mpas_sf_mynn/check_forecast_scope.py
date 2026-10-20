"""Validate written-word identity against independently recorded omissions.

The integrator's 2026-10-13 09:22Z instruction permits only the foundation's
sealed missing inventory. This reads Rust comparison metadata, never fields.
"""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_key(row):
    return row['stream'], row['field'], row['step'], row['time']


def check(receipt, completion, deck, inventory, deck_sha256):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(receipt.get('model_execution') is True and
            receipt.get('evidence_kind') == 'FORECAST' and
            not receipt.get('failed_model_exit'), 'forecast execution failed')
    require(receipt.get('process_count') == deck['process_count'] ==
            inventory['process_count'] == 1, 'process count differs')
    for name in ('pin_set', 'mesh', 'date', 'suite', 'duration_steps'):
        require(deck[name] == inventory[name], f'changed native {name}')
    for name in ('pin_set', 'mesh', 'date'):
        require(receipt[name] == inventory[name], f'changed receipt {name}')
    require(completion.get('completed') is True and
            completion.get('steps_requested') == completion.get('steps_completed') ==
            inventory['duration_steps'], 'full forecast not completed')
    physics = completion['physics']
    require(physics['suite'] == deck['suite'] and
            physics['selectors']['surface_layer'] == 'sf_mynn',
            'wrong surface scheme or suite')
    backends = physics['backends']
    require(backends.get('surface_layer') == 'hex-cuda' and
            all(value == 'native-cpu' for key, value in backends.items()
                if key != 'surface_layer'), 'wrong slot backend selection')
    command = deck['hex_command']
    library = Path(command[command.index('--native-physics-library') + 1])
    require(completion['native_library_sha256'] == digest(library),
            'native provider differs from completion')
    require(receipt['provenance']['deck_sha256'] == deck_sha256,
            'verifier executed a different deck')
    require(receipt['provenance']['native_manifest_sha256'] ==
            inventory['native_manifest_sha256'] == deck['native']['sha256'],
            'native publication differs')
    rows = receipt['comparison']['fields']
    actual = {sample_key(row): row for row in rows}
    expected = {sample_key(row): row for row in inventory['samples']}
    require(len(actual) == len(rows) and len(expected) == len(inventory['samples']),
            'duplicate field-time samples')
    require(actual.keys() == expected.keys(), 'native sample coverage differs')
    computed_words = missing_words = missing_samples = computed_samples = 0
    for key, row in actual.items():
        reference = expected[key]
        require(row['compared_words'] == reference['words'],
                f'{key}: native word count differs')
        require(row['step'] in deck['expected_samples'][row['stream']],
                f'{key}: unexpected output step')
        if row['status'] == 'compared':
            require(row['differing_words'] == 0, f'{key}: written words differ')
            computed_words += row['compared_words']
            computed_samples += 1
        else:
            require(row['status'] == 'field missing in hex' and
                    reference['missing_allowed'] is True,
                    f'{key}: new omission or incompatible field')
            missing_words += row['compared_words']
            missing_samples += 1
    require(computed_words > 0, 'no written word evidence')
    expected_failures = {
        f"{r['stream']}/{r['field']} step {r['step']}: field missing in hex"
        for r in rows if r['status'] != 'compared'}
    require(set(receipt.get('failures', [])) == expected_failures,
            'failure outside recorded omissions')
    return dict(schema='p10-written-forecast-scope-v1', passed=True,
                scope='integrator 09:22Z: written fields bitwise; sealed omissions may not grow',
                process_count=1, duration_steps=deck['duration_steps'],
                computed_samples=computed_samples, computed_words=computed_words,
                computed_differing_words=0, missing_samples=missing_samples,
                missing_words=missing_words, full_field_bitwise=missing_samples == 0,
                official_passed=receipt['passed'], official_tier=receipt['tier'],
                official_receipt_modified=False)


def check_sealed(receipt, completion, deck_path, receipt_dir):
    pins = json.loads(Path(__file__).with_name('forecast_scope_pins.json').read_text())
    pin = pins['decks'][digest(deck_path)]
    if receipt['provenance']['comparator_sha256'] != pins['comparator_sha256']:
        raise ValueError('Rust comparator differs from pinned authority')
    for provider in pins['providers']:
        if digest(provider['path']) != provider['sha256']:
            raise ValueError('native provider or ABI sidecar changed')
    inventory_path = Path(receipt_dir) / pin['inventory']
    if digest(inventory_path) != pin['sha256']:
        raise ValueError('recorded native inventory changed')
    result = check(receipt, completion, json.loads(Path(deck_path).read_text()),
                   json.loads(inventory_path.read_text()), digest(deck_path))
    result['inventory'] = pin
    return result
