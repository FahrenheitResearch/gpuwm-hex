"""Adversarial proof-contract fixtures, never scientific forecast evidence."""
import copy
import json
from pathlib import Path

import pytest

from rrfs_proof_contract import audit_gate, audit_witnesses, digest
from hexcore.mpasphys.rrfs.driver_rrfs import resolve_namelist, specification


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': digest(path)}


@pytest.fixture
def proof(tmp_path):
    # Deliberately small contract fixtures. They cannot satisfy reproduce.sh's
    # 18-case matrix or the authenticated r0 publication inventory.
    folder = tmp_path/'rrfs'
    namelist = "&physics\n config_physics_suite='hrrrv5'\n/\n"
    namelist_path = tmp_path/'namelist.atmosphere'
    namelist_path.write_text(namelist)
    deck = dict(id='fixture', pin_set='fixture-pin', mesh='fixture-mesh', date='fixture-date',
                process_count=1, duration_steps=3, expected_samples={'diagnostics': [0, 3]},
                inputs={'namelist': {'path': str(namelist_path), 'sha256': digest(namelist_path)}})
    ref = write(tmp_path/'deck.json', deck)
    index = tmp_path/'index.json'
    write(index, {'gates': {'G2': [ref]}})
    ratchet = tmp_path/'ratchet.json'
    write(ratchet, {'schema': 'mpas-ratchet-v1', 'entries': []})
    fields = [dict(stream='diagnostics', field='olrtoa', step=step, status='compared',
                   compared_words=24, differing_words=0, max_ulp=0) for step in (0, 3)]
    receipt = dict(deck='fixture', pin_set='fixture-pin', mesh='fixture-mesh', date='fixture-date',
                   process_count=1, passed=True, evidence_kind='FORECAST', model_execution=True,
                   failed_model_exit=None, acceptance_tier='BITWISE', tier='BITWISE',
                   comparison={'fields': fields}, ratchet_regressions=[],
                   provenance={'hex_source': {'commit': 'fixture-commit', 'dirty': ''},
                               'deck': str(tmp_path/'deck.json'),
                               'deck_sha256': ref['sha256'], 'native_manifest_sha256': 'fixture-native',
                               'hex_output_sha256': {}})
    spec = specification()
    runtime = dict(schema='mpas-native-forecast-v1', completed=True,
                   steps_requested=3, steps_completed=3, resolved_extensions=dict(spec['matched_extensions']),
                   physics=resolve_namelist(namelist).receipt())
    runtime_ref = write(folder/'runtime.json', runtime)
    operations = ('radiation_sw', 'radiation_lw', 'pbl', 'land_surface', 'convection',
                  'max_hail_diameter_sfc_acc', 'max_hail_diameter_column_acc')
    trace = dict(evidence_kind='FORECAST', process_count=1, duration_steps=3, deck='fixture',
                 native_manifest_sha256='fixture-native',
                 events=[{'operation': op, 'step': 3} for op in operations])
    native_ref = write(folder/'native-trace.json', trace)
    hex_ref = write(folder/'hex-trace.json', trace)
    receipt['provenance']['hex_output_sha256'] = {
        'native-physics-receipt.json': runtime_ref['sha256'],
        'physics-alarm-trace.json': hex_ref['sha256']}
    receipt_ref = write(folder/'G2-0.json', receipt)
    gate = dict(schema='mpas-gate-v1', passed=True, failures=[],
                inputs={'index_sha256': digest(index), 'ratchet_sha256': digest(ratchet)},
                stages={'G2': {'decks': [dict(deck='fixture', process_count=1, exit_code=0,
                                            recovery_exit_code=None, failures=[],
                                            receipt=str(folder/'G2-0.json'))]}})
    write(folder/'gate.json', gate)
    broken_runtime = copy.deepcopy(runtime)
    broken_runtime['physics']['backends']['radiation_lw'] = 'hex-cuda'
    broken_runtime_ref = write(folder/'broken-runtime.json', broken_runtime)
    broken = copy.deepcopy(receipt)
    broken['passed'] = False
    broken['comparison']['fields'][1]['differing_words'] = 24
    broken['provenance']['hex_output_sha256']['native-physics-receipt.json'] = broken_runtime_ref['sha256']
    broken_ref = write(folder/'broken.json', broken)
    witnesses = dict(schema='rrfs-driver-forecast-witnesses-v1', source_commit='fixture-commit',
                     decks=[dict(deck='fixture', forecast_sha256=receipt_ref['sha256'],
                                 runtime=runtime_ref, native_alarm_trace=native_ref, hex_alarm_trace=hex_ref)],
                     broken_slot=dict(baseline_deck='fixture', slot='radiation_lw', receipt=broken_ref,
                                      runtime=broken_runtime_ref,
                                      expected_first_difference={'stream': 'diagnostics', 'field': 'olrtoa', 'step': 3}))
    write(folder/'driver-witnesses.json', witnesses)
    return folder, index, ratchet, gate, receipt, witnesses


def audit(proof):
    folder, index, ratchet, *_ = proof
    receipts = audit_gate(folder, index, ratchet, ('G2',), 'fixture-commit')
    audit_witnesses(folder, receipts, 'fixture-commit')


def test_complete_contract_fixture(proof):
    audit(proof)


@pytest.mark.parametrize('defect', ['no-decks', 'duplicate-deck', 'other-index', 'other-ratchet',
                                  'missing-stage', 'outside-receipt', 'failed-exit'])
def test_gate_cannot_certify_incomplete_or_rebound_runs(proof, defect):
    folder, index, ratchet, gate, *_ = proof
    if defect == 'no-decks': gate['stages']['G2']['decks'] = []
    elif defect == 'duplicate-deck': gate['stages']['G2']['decks'] *= 2
    elif defect == 'other-index': gate['inputs']['index_sha256'] = 'wrong'
    elif defect == 'other-ratchet': gate['inputs']['ratchet_sha256'] = 'wrong'
    elif defect == 'missing-stage': gate['stages'] = {}
    elif defect == 'outside-receipt': gate['stages']['G2']['decks'][0]['receipt'] = str(index)
    elif defect == 'failed-exit': gate['stages']['G2']['decks'][0]['exit_code'] = 143
    write(folder/'gate.json', gate)
    with pytest.raises(ValueError): audit(proof)


@pytest.mark.parametrize('defect', ['wrong-commit', 'dirty', 'no-fields', 'wrong-deck', 'short-run',
                                  'differing-word', 'column', 'zero-words', 'missing-output'])
def test_passing_label_cannot_hide_unproven_forecast(proof, defect):
    folder, _, _, _, receipt, _ = proof
    if defect == 'wrong-commit': receipt['provenance']['hex_source']['commit'] = 'old'
    elif defect == 'dirty': receipt['provenance']['hex_source']['dirty'] = 'M module.py'
    elif defect == 'wrong-deck': receipt['provenance']['deck_sha256'] = 'other'
    elif defect == 'no-fields': receipt['comparison']['fields'] = []
    elif defect == 'short-run': receipt['comparison']['fields'].pop()
    elif defect == 'differing-word': receipt['comparison']['fields'][0]['differing_words'] = 1
    elif defect == 'column': receipt['evidence_kind'] = 'COLUMN'
    elif defect == 'zero-words':
        for field in receipt['comparison']['fields']: field['compared_words'] = 0
    elif defect == 'missing-output': receipt['comparison']['fields'][0]['status'] = 'field missing in hex'
    write(folder/'G2-0.json', receipt)
    with pytest.raises(ValueError): audit(proof)


@pytest.mark.parametrize('defect', ['no-witness', 'old-commit', 'no-decks', 'wrong-baseline',
                                  'alarm-changed', 'extensions-on', 'no-stub-diff', 'wrong-stub-step',
                                  'wrong-stub-field', 'stub-column', 'stub-missing', 'stub-unswitched'])
def test_forecasts_alone_cannot_certify_driver_acceptance(proof, defect):
    folder, _, _, _, _, witnesses = proof
    row, fault = witnesses['decks'][0], witnesses['broken_slot']
    if defect == 'no-witness':
        (folder/'driver-witnesses.json').unlink()
        with pytest.raises(FileNotFoundError): audit(proof)
        return
    if defect == 'old-commit': witnesses['source_commit'] = 'old'
    elif defect == 'no-decks': witnesses['decks'] = []
    elif defect == 'wrong-baseline': row['forecast_sha256'] = 'other'
    elif defect == 'wrong-stub-step': fault['expected_first_difference']['step'] = 2
    elif defect == 'wrong-stub-field': fault['expected_first_difference']['field'] = 'wrong'
    else:
        name = ('hex_alarm_trace' if defect == 'alarm-changed' else
                'runtime' if defect == 'extensions-on' else 'receipt')
        ref = row[name] if defect in ('alarm-changed', 'extensions-on') else fault[name]
        path = Path(ref['path'])
        data = json.loads(path.read_text())
        if defect == 'alarm-changed': data['events'][0]['step'] = 2
        elif defect == 'extensions-on': data['resolved_extensions'] = {'woof': True}
        elif defect == 'no-stub-diff': data['comparison']['fields'][1]['differing_words'] = 0
        elif defect == 'stub-column': data['evidence_kind'] = 'COLUMN'
        elif defect == 'stub-missing': data['comparison']['fields'][1]['status'] = 'field missing in hex'
        elif defect == 'stub-unswitched':
            ref = fault['runtime']; path = Path(ref['path'])
            data = json.loads(path.read_text()); data['physics']['backends']['radiation_lw'] = 'native-cpu'
        ref.update(write(path, data))
        # Rebind the synthetic fixture's recorded hashes so these cases reach
        # the scientific-contract check rather than only failing authentication.
        if defect in ('alarm-changed', 'extensions-on'):
            baseline_path = folder/'G2-0.json'
            baseline = json.loads(baseline_path.read_text())
            artifact = 'physics-alarm-trace.json' if defect == 'alarm-changed' else 'native-physics-receipt.json'
            baseline['provenance']['hex_output_sha256'][artifact] = ref['sha256']
            row['forecast_sha256'] = write(baseline_path, baseline)['sha256']
        elif defect == 'stub-unswitched':
            candidate_path = Path(fault['receipt']['path'])
            candidate = json.loads(candidate_path.read_text())
            candidate['provenance']['hex_output_sha256']['native-physics-receipt.json'] = ref['sha256']
            fault['receipt'].update(write(candidate_path, candidate))
    write(folder/'driver-witnesses.json', witnesses)
    with pytest.raises(ValueError): audit(proof)
