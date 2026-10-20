"""Assemble an owner-review request from complete original forecast evidence.

This consumes comparator and custody metadata only. It neither runs physics,
installs a baseline nor changes any original forecast acceptance result.
"""
import argparse
import hashlib
import json
from pathlib import Path
from owner_context_packets import load_packet, write_packet, LIMIT


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def read_pin(item):
    path = Path(item['path'])
    assert sha(path) == item['sha256']
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-proposal', 'inventory', 'custody', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nongrowth', type=Path, nargs='*', default=[])
    parser.add_argument('--split-receipts', action='store_true',
                        help='Write a hash-bound packet set, each receipt under 10 MB.')
    parser.add_argument('--receipt-limit', type=int, default=LIMIT)
    args = parser.parse_args()
    prior = load_packet(args.prior_proposal)
    assert prior['proposal_only'] and not prior['owner_approved']
    assert not prior['usable_as_baseline'] and prior['process_count'] == 1
    read_pin(prior['original_forecast_inventory'])
    read_pin(prior['completion_log_custody'])
    inventory = json.loads(args.inventory.read_text())
    custody = json.loads(args.custody.read_text())
    assert inventory['process_count'] == 1
    assert custody['inventory_sha256'] == sha(args.inventory)
    assert custody['original_acceptance_unchanged']
    logs = {row['request_index']: row for row in custody['rows']}
    assert len(logs) == len(custody['rows']) == custody['complete_forecasts']

    witnesses = list(prior['missing_inventory_nongrowth_witnesses'])
    witnesses.extend(pin(path) for path in args.nongrowth)
    audits = {}
    for item in witnesses:
        audit = read_pin(item)
        assert audit['process_count'] == 1 and not audit['private_ratchet_created']
        assert audit['acceptance_unchanged'] and audit['new_missing_samples'] == 0
        assert audit['missing_inventory_exactly_matches']
        assert sha(audit['owner_inventory']) == audit['owner_inventory_sha256']
        key = audit['verify_sha256']
        assert key not in audits, 'Each complete forecast needs one original audit'
        audits[key] = (item, audit)

    contexts = []
    for row in inventory['decks']:
        if not row.get('model_execution') or row.get('evidence_kind') != 'FORECAST':
            continue
        receipt_pin = dict(path=row['receipt'], sha256=row['receipt_sha256'])
        receipt = read_pin(receipt_pin)
        deck_pin = dict(path=row['path'], sha256=row['deck_sha256'])
        deck = read_pin(deck_pin)
        assert receipt['provenance']['deck_sha256'] == row['deck_sha256']
        assert receipt['process_count'] == deck['process_count'] == deck['ranks'] == 1
        assert receipt['model_execution'] and receipt['evidence_kind'] == 'FORECAST'
        assert receipt['passed'] is False and receipt['acceptance_tier'] is None
        assert not receipt['provenance']['hex_source']['dirty']
        assert receipt['provenance']['hex_source']['commit'] == row['source_commit']
        model = logs[row['request_index']]
        assert model['original_receipt_sha256'] == receipt_pin['sha256']
        assert model['deck_sha256'] == deck_pin['sha256']
        assert model['steps_completed'] == deck['duration_steps']
        assert sha(model['model_log']) == model['model_log_sha256']
        _, audit = audits[receipt_pin['sha256']]
        assert audit['deck_sha256'] == deck_pin['sha256']
        assert audit['missing_records'] == row['counters']['missing_records']
        assert audit['missing_words'] == row['counters']['missing_words']
        contexts.append(dict(request_index=row['request_index'],
            context={key: receipt[key] for key in
                     ('pin_set', 'mesh', 'date', 'deck', 'process_count')},
            receipt=receipt_pin, deck=deck_pin, original_passed=receipt['passed'],
            original_acceptance_tier=receipt['acceptance_tier'],
            counters=row['counters'],
            all_actual_aggregate_entries=receipt['candidate_ratchet']['entries']))
    contexts.sort(key=lambda row: row['request_index'])
    old = {row['request_index']: row for row in prior['completed_contexts']}
    current = {row['request_index']: row for row in contexts}
    assert len(current) == len(contexts) == len(logs)
    assert old.keys() <= current.keys()
    assert all(current[index] == row for index, row in old.items())
    assert len(audits) == len(contexts), 'No absent or unrelated audit is allowed'
    packet = dict(schema=prior['schema'], process_count=1, proposal_only=True,
        owner_approved=False, usable_as_baseline=False,
        original_forecast_inventory=pin(args.inventory), completed_contexts=contexts,
        missing_inventory_nongrowth_witnesses=witnesses,
        completion_log_custody=pin(args.custody))
    write_packet(packet, args.out, args.split_receipts, args.receipt_limit)
    print(json.dumps(dict(completed_contexts=len(contexts),
        new_contexts=len(current) - len(old), owner_approved=False,
        original_acceptance_unchanged=True, packet_sha256=sha(args.out))))


if __name__ == '__main__':
    main()
