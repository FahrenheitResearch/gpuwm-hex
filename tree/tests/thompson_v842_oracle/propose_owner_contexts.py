"""Prepare an unapplied owner review of existing M0 native-only inventories.

No scientific data path: this authenticates Rust comparison receipts, compares
their records, and proposes context bindings. It never supplies a baseline to
a forecast, changes a canonical file, or declares forecast acceptance.
"""
import argparse
import copy
import difflib
import hashlib
import json
from pathlib import Path

from hexcore.verify_mpas import aggregate, recorded_missing_failures
from summarize_forecast_matrix import LABELS


CONTEXT = ('pin_set', 'mesh', 'date', 'deck', 'process_count')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def authenticated(root, member):
    path = root / member['path']
    assert sha(path) == member['sha256'], path
    return path, read(path)


def missing(fields):
    return [row for row in fields if row['status'] != 'compared']


def qualify(actual, authority, context, proposal):
    fields = actual['comparison']['fields']
    assert fields and any(row['status'] == 'compared' for row in fields)
    assert all(row['differing_words'] == 0 for row in fields if row['status'] == 'compared')
    assert missing(fields) == missing(authority), 'M0 inventory records changed'
    assert not recorded_missing_failures(actual['comparison'], context, proposal)
    # Exercise the production policy with real full-deck records, retaining all
    # unrelated records. No fabricated receipt is presented as a model run.
    controls = []
    computed = next(i for i, row in enumerate(fields) if row['status'] == 'compared')
    absent = next(i for i, row in enumerate(fields) if row['status'] != 'compared')
    for name in ('computed-word', 'missing-word-growth', 'new-missing-field',
                 'structural-error', 'previously-written-field', 'removed-record',
                 'wrong-process-count'):
        mutated = copy.deepcopy(actual['comparison'])
        changed = mutated['fields']
        ctx = dict(context)
        if name == 'computed-word':
            changed[computed]['differing_words'] = 1
            changed[computed]['max_ulp'] = 1
        elif name == 'missing-word-growth':
            changed[absent]['differing_words'] += 1
        elif name == 'new-missing-field':
            changed[absent]['field'] += '_unregistered'
        elif name == 'structural-error':
            changed[absent]['status'] = 'shape differs'
        elif name == 'previously-written-field':
            changed[computed]['status'] = 'field missing in hex'
            changed[computed]['differing_words'] = max(1, changed[computed]['compared_words'])
        elif name == 'removed-record':
            del changed[absent]
        else:
            ctx['process_count'] = 2
        failures = recorded_missing_failures(mutated, ctx, proposal)
        assert failures, name
        controls.append(dict(name=name, rejected=True, failure_count=len(failures)))
    return controls


def propose(directory, decks, foundation, canonical, output):
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = foundation / 'receipts/M0-clean-eight-deck-matrix-v842.json'
    matrix = read(matrix_path)
    assert matrix['source_clean'] and not matrix['source_patches_applied']
    assert matrix['all_written_numerical_words_identical']
    baseline = read(canonical)
    proposal = copy.deepcopy(baseline)
    original_count = len(baseline['entries'])
    rows, authorities = [], {}
    for label in LABELS:
        deck_path = decks / label / 'deck.json'
        deck = read(deck_path)
        context = {key: deck[key] for key in CONTEXT if key != 'deck'}
        context['deck'] = deck['id']
        assert context['process_count'] == 1
        matches = [d for d in matrix['decks'] if d['mesh'] == context['mesh']
                   and d['date'] == context['date'] and d['suite'] == 'convection_permitting']
        assert len(matches) == 1
        owner = matches[0]
        owner_path, owner_receipt = authenticated(foundation, owner['receipt'])
        assert owner_receipt['completed'] and owner_receipt['process_count'] == 1
        assert owner_receipt['forecast_progress']['completed']
        comparison = owner_receipt['comparison']
        fields = comparison['comparison']['fields']
        assert all(row['differing_words'] == 0 for row in fields if row['status'] == 'compared')
        omitted = missing(fields)
        assert omitted and all(row['status'] == 'field missing in hex' for row in omitted)
        owner_context = {key: comparison[key] for key in CONTEXT}
        assert owner_context['mesh'] == context['mesh'] and owner_context['date'] == context['date']
        additions = aggregate(omitted, context)['entries']
        proposal['entries'].extend(additions)
        authorities[label] = fields
        rows.append(dict(label=label, requested_context=context, original_owner_context=owner_context,
            owner_receipt=str(owner_path), owner_receipt_sha256=sha(owner_path),
            owner_commit=owner_receipt['hex_commit'], deck=str(deck_path), deck_sha256=sha(deck_path),
            missing_entries=len(additions), missing_samples=len(omitted),
            missing_words=sum(row['differing_words'] for row in omitted),
            complete_cuda_forecast_available=False))
    assert proposal['entries'][:original_count] == baseline['entries']
    keys = [tuple(row[k] for k in CONTEXT[:-1]) + (row['stream'], row['field'])
            for row in proposal['entries']]
    assert len(keys) == len(set(keys)), 'context already registered; request must be reconciled'
    for row in rows:
        label = row['label']
        path = directory / (label + '.json')
        if not path.is_file():
            continue
        actual = read(path)
        assert actual['model_execution'] and actual['evidence_kind'] == 'FORECAST'
        assert all(actual[k] == v for k, v in row['requested_context'].items())
        assert not actual['provenance']['hex_source']['dirty']
        attempts = sorted((directory / 'metadata').glob(label + '-attempt-*'),
                          key=lambda p: int(p.name.rsplit('-', 1)[1]))
        metadata = attempts[-1]
        preservation = read(metadata / 'preservation.json')
        for name, expected in preservation['observed_sha256'].items():
            assert sha(metadata / name) == expected
        progress = read(metadata / 'native-physics-receipt.json')
        request = read(metadata / 'native-physics-request.json')
        assert progress['completed'] and progress['steps_completed'] == progress['steps_requested'] == request['steps']
        assert progress['physics'] == request['physics']
        assert request['physics']['backends']['microphysics'] == 'hex-cuda'
        assert request['physics']['selectors']['microphysics'] == 'mp_thompson'
        assert all(v == 'native-cpu' for k, v in request['physics']['backends'].items() if k != 'microphysics')
        controls = qualify(actual, authorities[label], row['requested_context'], proposal)
        row.update(complete_cuda_forecast_available=True, actual_verifier_passed=actual['passed'],
            actual_receipt=str(path), actual_receipt_sha256=sha(path),
            source_commit=actual['provenance']['hex_source']['commit'],
            progress_sha256=sha(metadata / 'native-physics-receipt.json'),
            computed_words=sum(r['compared_words'] for r in actual['comparison']['fields'] if r['status'] == 'compared'),
            computed_differing_words=0, exact_missing_records_match=True, negative_controls=controls)
    proposed_text = json.dumps(proposal, indent=2) + '\n'
    diff = ''.join(difflib.unified_diff(canonical.read_text().splitlines(True),
        proposed_text.splitlines(True), fromfile='a/receipts/ratchet.json', tofile='b/receipts/ratchet.json'))
    diff_path = output / 'owner-contexts-proposed.diff'
    diff_path.write_text(diff)
    result = dict(schema='p3-thompson-owner-context-proposal/v1', proposal_only=True,
        owner_approved=False, usable_as_baseline=False, supplied_to_forecast=False,
        canonical_accepted=False, process_count=1, duration_hours=24,
        canonical_ratchet_sha256=sha(canonical), canonical_entries_unchanged=original_count,
        source_owner_matrix=str(matrix_path), source_owner_matrix_sha256=sha(matrix_path),
        diff=str(diff_path), diff_sha256=sha(diff_path), proposed_missing_entries=len(proposal['entries'])-original_count,
        complete_cuda_forecasts=sum(r['complete_cuda_forecast_available'] for r in rows),
        full_eight_cuda_matrix_qualified=all(r['complete_cuda_forecast_available'] for r in rows), rows=rows,
        scope='Owner review required for explicit pin-set/deck context binding. Only original M0 missing records are proposed; no canonical policy or forecast is changed.')
    (output / 'proposal.json').write_text(json.dumps(result, indent=2) + '\n')
    assert sha(canonical) == result['canonical_ratchet_sha256']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'decks', 'foundation', 'canonical', 'output'):
        parser.add_argument(name, type=Path)
    result = propose(**vars(parser.parse_args()))
    print('Unapplied owner proposal; completed CUDA contexts qualified:', result['complete_cuda_forecasts'])
