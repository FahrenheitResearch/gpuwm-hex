"""Audit whole-forecast evidence; never accept column results as RRFS-M0.

This is delivery tooling, not a forecast guard. The supplemental witness bundle
must be produced in the fresh gate output directory. Its producer is still an
integration requirement; missing runtime evidence deliberately fails the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hexcore.verify_mpas import file_ref, gate_receipt_failures
from hexcore.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clean_source(receipt, commit):
    source = receipt['provenance']['hex_source']
    require(source['commit'] == commit and source['dirty'] == '',
            'forecast must use this clean committed checkout')


def audit_gate(folder, index_path, ratchet_path, stages, commit):
    """Require every indexed deck, authentic context, final stream sample and word."""
    folder, index_path, ratchet_path = map(Path, (folder, index_path, ratchet_path))
    gate, index, ratchet = read(folder/'gate.json'), read(index_path), read(ratchet_path)
    require(gate['schema'] == 'mpas-gate-v1' and gate['passed'] is True
            and not gate['failures'], 'whole forecast gate failed')
    require(gate['inputs'] == {'index_sha256': digest(index_path),
                              'ratchet_sha256': digest(ratchet_path)},
            'gate index or immutable ratchet changed')
    require(set(gate['stages']) == set(stages), 'gate stages incomplete')
    receipts = {}
    for stage in stages:
        if stage == 'G0':
            require(gate['stages'][stage], 'G0 result absent')
            continue
        expected = index['gates'][stage]
        rows = gate['stages'][stage]['decks']
        require(expected and len(rows) == len(expected), 'forecast deck count incomplete')
        for number, (ref, row) in enumerate(zip(expected, rows)):
            deck_path = file_ref(ref, index_path.parent)
            deck = read(deck_path)
            receipt_path = folder/f'{stage}-{number}.json'
            require(Path(row['receipt']).resolve() == receipt_path.resolve(),
                    'forecast receipt outside fresh proof directory')
            require(row['deck'] == deck['id'] and row['exit_code'] == 0
                    and row['process_count'] == 1 and not row['failures']
                    and row.get('recovery_exit_code') is None,
                    'forecast command failed or deck identity changed')
            receipt = read(receipt_path)
            require(not gate_receipt_failures(receipt, deck, stage, ratchet),
                    'forecast fields, schedule or immutable missing inventory failed')
            require(receipt.get('acceptance_tier') in ('BITWISE', 'BITWISE-WRITTEN-FIELDS'),
                    'RRFS driver proof requires zero words')
            fields = receipt['comparison']['fields']
            require(any(f['status'] == 'compared' and f['compared_words'] > 0 for f in fields),
                    'forecast compared no stored words')
            require(all(f['differing_words'] == 0 for f in fields if f['status'] == 'compared'),
                    'forecast has differing stored words')
            require(receipt['provenance']['deck_sha256'] == digest(deck_path)
                    and Path(receipt['provenance']['deck']).resolve() == deck_path.resolve(),
                    'forecast receipt belongs to another deck')
            clean_source(receipt, commit)
            require(deck['id'] not in receipts, 'duplicate forecast deck identity')
            receipts[deck['id']] = (deck, receipt, receipt_path)
    return receipts


def audit_witnesses(folder, receipts, commit):
    """Bind resolved extensions, full alarm traces and a localized broken slot.

    The sidecars must have hashes captured by the forecast verifier before its
    disposable output is removed. Normalized native traces must name the exact
    strict native manifest used in the corresponding field comparison.
    """
    folder = Path(folder)
    bundle = read(folder/'driver-witnesses.json')
    require(bundle['schema'] == 'rrfs-driver-forecast-witnesses-v1', 'unknown driver witness schema')
    require(bundle['source_commit'] == commit, 'driver witnesses use another commit')
    require(len(bundle['decks']) == len(receipts), 'driver witnesses omit a forecast deck')
    seen, operations = set(), set()
    for row in bundle['decks']:
        identity = row['deck']
        require(identity in receipts and identity not in seen, 'unknown or duplicate witness deck')
        seen.add(identity)
        deck, baseline, baseline_path = receipts[identity]
        require(row['forecast_sha256'] == digest(baseline_path), 'driver witness baseline changed')
        runtime_path = file_ref(row['runtime'], folder)
        runtime = read(runtime_path)
        hashes = baseline['provenance']['hex_output_sha256']
        require(hashes.get('native-physics-receipt.json') == digest(runtime_path),
                'resolved configuration was not captured from this forecast')
        require(runtime['schema'] == 'mpas-native-forecast-v1' and runtime['completed'] is True
                and runtime['steps_requested'] == runtime['steps_completed'] == deck['duration_steps'],
                'runtime did not complete the whole forecast')
        physics = runtime['physics']
        spec = specification()
        config = resolve_namelist(file_ref(deck['inputs']['namelist'],
                                  Path(baseline['provenance']['deck']).parent).read_text())
        resolved = config.receipt()
        require(all(physics.get(key) == resolved[key] for key in
                    ('options', 'execution_options', 'selectors', 'backends')),
                'runtime physics differs from the original resolved namelist')
        require(physics['model_commit'] == spec['model_commit']
                and physics['dycore_path'] == spec['dycore_path'], 'runtime selected another core')
        require(runtime['resolved_extensions'] == spec['matched_extensions'],
                'actual matched extensions differ from the fork configuration')
        require(physics['backends'] and all(value in ('native-cpu', 'off')
                                          for value in physics['backends'].values()),
                'baseline contains a CUDA physics slot')
        native_path = file_ref(row['native_alarm_trace'], folder)
        hex_path = file_ref(row['hex_alarm_trace'], folder)
        native, candidate = read(native_path), read(hex_path)
        require(hashes.get('physics-alarm-trace.json') == digest(hex_path),
                'hex alarm trace was not captured from this forecast')
        for trace in (native, candidate):
            require(trace['evidence_kind'] == 'FORECAST' and trace['process_count'] == 1
                    and trace['duration_steps'] == deck['duration_steps']
                    and trace['deck'] == identity, 'alarm evidence is not the complete same forecast')
            require(trace['native_manifest_sha256'] == baseline['provenance']['native_manifest_sha256'],
                    'alarm evidence belongs to another strict native reference')
        require(native['events'] == candidate['events'], 'native and hex alarm/reset schedules differ')
        measured = {event['operation'] for event in native['events']}
        require(all(slot in measured for slot in
                    ('radiation_sw', 'radiation_lw', 'pbl', 'land_surface', 'convection')
                    if config.active(slot)), 'active forecast alarm branch unmeasured')
        for event in native['events']:
            step = event['step']
            require(type(step) is int and 0 <= step <= deck['duration_steps'], 'invalid alarm step')
            operations.add(event['operation'])
    required = {'radiation_sw', 'radiation_lw', 'pbl', 'land_surface', 'convection',
                'max_hail_diameter_sfc_acc', 'max_hail_diameter_column_acc'}
    require(required <= operations, 'forecast alarm or hail reset branch unmeasured')
    fault = bundle['broken_slot']
    require(fault['baseline_deck'] in receipts, 'broken-slot baseline absent')
    _, baseline, _ = receipts[fault['baseline_deck']]
    candidate = read(file_ref(fault['receipt'], folder))
    clean_source(candidate, commit)
    require(candidate['evidence_kind'] == 'FORECAST' and candidate['model_execution'] is True
            and candidate['passed'] is False and candidate['failed_model_exit'] is None
            and candidate['process_count'] == 1, 'broken slot did not execute a real forecast')
    require(candidate['provenance']['native_manifest_sha256'] ==
            baseline['provenance']['native_manifest_sha256'], 'broken slot uses another reference')
    runtime_path = file_ref(fault['runtime'], folder)
    runtime = read(runtime_path)
    require(candidate['provenance']['hex_output_sha256'].get('native-physics-receipt.json')
            == digest(runtime_path), 'broken-slot runtime was not captured from this forecast')
    base_row = next(row for row in bundle['decks'] if row['deck'] == fault['baseline_deck'])
    base_runtime = read(file_ref(base_row['runtime'], folder))
    slot = fault['slot']
    require(slot in specification()['slots'] and base_runtime['physics']['backends'].get(slot)
            == 'native-cpu', 'broken slot is not active on the baseline native path')
    expected_physics = dict(base_runtime['physics'])
    expected_physics['backends'] = dict(expected_physics['backends'], **{slot: 'hex-cuda'})
    require(runtime['physics'] == expected_physics
            and runtime['resolved_extensions'] == base_runtime['resolved_extensions']
            and runtime['completed'] is True
            and runtime['steps_completed'] == base_runtime['steps_completed'],
            'broken forecast must flip exactly its own slot and finish')
    key = lambda f: (f['stream'], f['field'], f['step'])
    base_fields = {key(f): f for f in baseline['comparison']['fields']}
    fields = candidate['comparison']['fields']
    require(len(fields) == len(base_fields) and {key(f) for f in fields} == set(base_fields)
            and all(f['status'] == base_fields[key(f)]['status']
                    and f['compared_words'] == base_fields[key(f)]['compared_words']
                    for f in fields),
            'broken slot changed the output or missing inventory')
    require(all(f['differing_words'] == base_fields[key(f)]['differing_words']
                for f in fields if f['status'] != 'compared'),
            'broken slot changed the native-only missing inventory')
    changed = [f for f in fields if f['status'] == 'compared' and f['differing_words'] > 0]
    require(changed, 'intentionally broken slot did not change a stored word')
    first = min(f['step'] for f in changed)
    expected = fault['expected_first_difference']
    require(type(expected['step']) is int and first == expected['step']
            and any(f['step'] == first and f['field'] == expected['field']
                    and f['stream'] == expected['stream'] for f in changed),
            'broken slot failed at another field or step')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'rrfs_index', 'mpas_index', 'ratchet', 'commit'):
        parser.add_argument(name)
    args = parser.parse_args()
    run = Path(args.run)
    rrfs = audit_gate(run/'rrfs', args.rrfs_index, args.ratchet, ('G2',), args.commit)
    audit_gate(run/'mpas', args.mpas_index, args.ratchet, ('G0', 'G1', 'G2'), args.commit)
    audit_witnesses(run/'rrfs', rrfs, args.commit)
    from rrfs_sheet_contract import audit_sheets
    audit_sheets(run/'rrfs', rrfs, args.commit)
    print('Complete bitwise RRFS forecast and driver witnesses passed')


if __name__ == '__main__':
    main()
