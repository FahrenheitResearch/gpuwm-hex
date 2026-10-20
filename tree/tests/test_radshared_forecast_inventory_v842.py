"""Metadata refusals stay failures; these fixtures never represent a forecast."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def refused_request(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')
        return path

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    context = dict(deck='synthetic-refusal', pin_set='fixture-pin',
                   mesh='fixture-mesh', date='2000-01-01_00:00:00', process_count=1)
    overlay = "config_radt_cld_backend='hex-cuda'"
    deck = write('deck.json', dict(context, id=context['deck'],
        inputs={'namelist': {'sha256': 'fixture-namelist'},
                'streams': {'sha256': 'fixture-streams'}},
        hex_command=['forecast', '--physics-backend', 'mpasphys', '--physics', overlay]))
    source = write('checkout/fixture-source.json', {})
    plan = write('plan.json', dict(process_count=1, source_private_patch=False,
        clean_checkout_at_start=True, source_commit='fixture-commit',
        source_sha256={'fixture-source.json': sha(source)},
        decks=[{'path': str(deck), 'sha256': sha(deck)}]))
    receipt = write('refusal.json', dict(context, evidence_kind='REFUSAL',
        passed=False, tier=None, failures=['fixture interrupted scratch directory']))
    matrix = write('matrix.json', dict(process_count=1, plan_sha256=sha(plan),
        rows=[dict(deck=context['deck'], receipt=str(receipt),
                   receipt_sha256=sha(receipt), exit=1)]))
    required = write('required.json', dict(process_count=1, required_requests=1,
        requests=[dict(mesh=context['mesh'], date=context['date'], process_count=1,
            namelist_sha256='fixture-namelist', streams_sha256='fixture-streams',
            hex_backend_override=overlay)]))
    comparator = write('comparator-identity.json', {})
    config = write('config.json', dict(checkout=str(source.parent),
        required=str(required), comparator=str(comparator),
        plans=[{'plan': str(plan), 'matrix': str(matrix)}]))
    return config, receipt, matrix, source, sha


def collect(monkeypatch, config, output):
    path = Path(__file__).parent / 'radshared_v842/collect_whole_forecasts_radshared_v842.py'
    spec = importlib.util.spec_from_file_location('radshared_inventory_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, 'argv', [str(path), '--config', str(config), '--out', str(output)])
    module.main()


@pytest.fixture
def trace_selection(tmp_path):
    path = Path(__file__).parent / 'radshared_v842/retain_forecast_traces.py'
    spec = importlib.util.spec_from_file_location('radshared_trace_selection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entries = []
    for name in ('previous-failure', 'completed', 'retry'):
        deck = tmp_path / (name + '.json')
        deck.write_text(json.dumps({'id': name}))
        entries.append({'path': str(deck), 'sha256': module.pin(deck)})
    return module.selected_rows, {'decks': entries}


def test_trace_collection_closes_selected_completed_requests(trace_selection):
    select, plan = trace_selection
    rows, complete = select({'only_ids': ['completed']}, plan, {'rows': [
        {'deck': 'previous-failure'}, {'deck': 'completed'}]})
    assert complete and rows == [{'deck': 'completed'}]


def test_old_attempt_cannot_satisfy_pending_trace_retry(trace_selection):
    select, plan = trace_selection
    rows, complete = select({'only_ids': ['retry']}, plan, {'rows': [
        {'deck': 'previous-failure'}, {'deck': 'completed'}]})
    assert rows == [] and not complete


@pytest.mark.parametrize('fault', ['unknown_selection', 'duplicate_selection',
                                 'unknown_result', 'duplicate_result', 'changed_deck'])
def test_trace_selection_rejects_ambiguous_or_unpinned_requests(trace_selection, fault):
    select, plan = trace_selection
    packet = {'only_ids': ['completed']}
    actual = {'rows': [{'deck': 'completed'}]}
    if fault == 'unknown_selection':
        packet['only_ids'] = ['unknown']
    elif fault == 'duplicate_selection':
        packet['only_ids'].append('completed')
    elif fault == 'unknown_result':
        actual['rows'].append({'deck': 'unknown'})
    elif fault == 'duplicate_result':
        actual['rows'].append({'deck': 'completed'})
    else:
        Path(plan['decks'][0]['path']).write_text('{"id":"changed"}')
    with pytest.raises(AssertionError):
        select(packet, plan, actual)


def test_preflight_refusal_survives_without_model_provenance(refused_request, monkeypatch, tmp_path):
    config, *_ = refused_request
    output = tmp_path / 'inventory.json'
    collect(monkeypatch, config, output)
    inventory = json.loads(output.read_text())
    assert inventory['ready'] is False and inventory['matrix_complete'] is False
    assert inventory['unbound_requests'] == []
    row, = inventory['decks']
    assert row['status'] == 'request refused; no model execution'
    assert row['accepted'] is False and row['model_execution'] is False
    assert row['counters'] is None and row['acceptance_tier'] is None
    assert row['refusal_failures'] == ['fixture interrupted scratch directory']


@pytest.mark.parametrize('fault', ['passed', 'model_execution', 'tier', 'deck', 'process_count', 'receipt_hash', 'source_hash'])
def test_refusal_cannot_hide_wrong_context_or_claim_execution(refused_request, monkeypatch, tmp_path, fault):
    config, receipt, matrix, source, sha = refused_request
    value = json.loads(receipt.read_text())
    if fault in ('passed', 'model_execution'):
        value[fault] = True
    elif fault == 'tier':
        value[fault] = 'BITWISE'
    elif fault == 'deck':
        value[fault] += '-unrelated'
    elif fault == 'process_count':
        value[fault] = 2
    elif fault == 'receipt_hash':
        value['failures'].append('unauthenticated mutation')
    else:
        source.write_text('changed source\n')
    receipt.write_text(json.dumps(value) + '\n')
    if fault != 'receipt_hash':
        actual = json.loads(matrix.read_text())
        actual['rows'][0]['receipt_sha256'] = sha(receipt)
        matrix.write_text(json.dumps(actual) + '\n')
    output = tmp_path / 'inventory.json'
    with pytest.raises(AssertionError):
        collect(monkeypatch, config, output)
    assert not output.exists()
