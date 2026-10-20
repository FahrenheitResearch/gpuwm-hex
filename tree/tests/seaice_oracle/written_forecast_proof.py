"""Enforce the integrator's written-field whole-forecast landing scope.

MPAS mpas_atmphys_driver_seaice.F:160-558 defines the subsystem boundary.
Weather comparisons are Rust receipts. The existing foundation missing-field
inventory is immutable; numerical differences and new omissions always fail.
"""
from collections import defaultdict


def check_stored(files, expected_files, inventory):
    """Reject differences in every stored type, incomplete files and new gaps."""
    failures = []
    if not files or {row.get('file') for row in files} != set(expected_files):
        failures.append('stored-word file inventory incomplete')
    if len({row.get('file') for row in files}) != len(files):
        failures.append('duplicate stored-word file')
    permitted = {(row['stream'], row['field']) for row in inventory
                 if row['status'] == 'field missing in hex'}
    for row in files:
        if (row.get('exit_code') not in (0, 1) or row.get('differing_words') != 0 or
                row.get('words', 0) <= 0):
            failures.append('stored word differs or empty file: ' + row.get('file', '?'))
        fields = row.get('fields', [])
        if (not fields or sum(field.get('words', 0) for field in fields) != row.get('words') or
                any(field.get('differing_words') != 0 for field in fields)):
            failures.append('stored field inventory absent or inconsistent')
        for error in row.get('errors', []):
            field = error.removeprefix('variable set differs: ')
            if not error.startswith('variable set differs: ') or (row['stream'], field) not in permitted:
                failures.append('stored-word comparison: ' + error)
    return failures


def check(receipt, deck, model, foundation, inventory=None):
    failures = []
    context = ('pin_set', 'mesh', 'date', 'process_count')
    for key in context:
        if receipt.get(key) != deck.get(key):
            failures.append('comparison context differs: ' + key)
    if receipt.get('deck') != deck['id']:
        failures.append('comparison deck differs')
    if receipt.get('model_execution') is not True or receipt.get('evidence_kind') != 'FORECAST':
        failures.append('no complete real forecast comparison')
    if receipt.get('failed_model_exit') is not None:
        failures.append('model failed')
    if not model or not model.get('completed') or (
            model.get('steps_completed') != deck['duration_steps'] or
            model.get('steps_requested') != deck['duration_steps']):
        failures.append('model did not complete the original deck')
    if model:
        backends = model.get('physics', {}).get('backends', {})
        if backends.get('seaice') != 'hex-cuda' or any(
                value != 'native-cpu' for key, value in backends.items() if key != 'seaice'):
            failures.append('seaice-only backend selection changed')
    allowed = {}
    for entry in foundation['entries']:
        if (entry['mesh'] != deck['mesh'] or entry['pin_set'] != deck['pin_set'] or
                entry['process_count'] != deck['process_count'] or
                entry.get('statuses') != ['field missing in hex']):
            continue
        key = (entry['stream'], entry['field'])
        bound = (entry['missing_samples'], entry['differing_words'])
        if key in allowed and allowed[key] != bound:
            failures.append('ambiguous foundation missing bound: ' + repr(key))
        allowed[key] = bound
    fields = receipt.get('comparison', {}).get('fields', [])
    if not fields:
        failures.append('empty native comparison inventory')
    seen, missing, schedule = {}, defaultdict(list), defaultdict(set)
    for row in fields:
        key = (row['stream'], row['field'], row['step'])
        if key in seen:
            failures.append('duplicate native field-time sample: ' + repr(key))
        seen[key] = row
        schedule[row['stream']].add(row['step'])
        if row['status'] == 'compared':
            if row['differing_words'] != 0 or row['compared_words'] <= 0:
                failures.append('nonzero or empty written field: ' + repr(key))
        elif row['status'] == 'field missing in hex':
            missing[key[:2]].append(row)
        else:
            failures.append('incompatible or corrupt field: ' + repr(key))
    for key, rows in missing.items():
        bound = allowed.get(key)
        if bound is None or len(rows) > bound[0] or sum(
                row['differing_words'] for row in rows) > bound[1]:
            failures.append('new or growing native-only gap: ' + repr(key))
    if dict(schedule) != {key: set(value) for key, value in deck['expected_samples'].items()}:
        failures.append('complete native output schedule changed')
    if not any(row['status'] == 'compared' for row in fields):
        failures.append('no written native field was compared')
    if inventory is not None:
        expected = {(row['stream'], row['field'], row['step']): row for row in inventory}
        if set(expected) != set(seen):
            failures.append('native field-time inventory changed')
        for key, old in expected.items():
            new = seen.get(key)
            if new is None:
                continue
            if old['status'] == 'compared' and (
                    new['status'] != 'compared' or new['compared_words'] != old['compared_words']):
                failures.append('previously written field disappeared or changed shape: ' + repr(key))
    return failures
