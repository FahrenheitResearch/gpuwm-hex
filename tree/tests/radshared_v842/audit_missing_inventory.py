"""Check an actual forecast's missing samples against a frozen owner inventory.

The audit never promotes the original acceptance result or enrolls a ratchet.
It compares metadata produced by the real Rust stored-word comparator.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory', 'forecast', 'deck', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    forecast = json.loads(args.forecast.read_text())
    deck = json.loads(args.deck.read_text())
    assert forecast['model_execution'] and forecast['evidence_kind'] == 'FORECAST'
    assert forecast['process_count'] == deck['process_count'] == deck['ranks'] == 1
    assert forecast['provenance']['deck_sha256'] == sha(args.deck)
    assert forecast['deck'] == deck['id']
    assert forecast['mesh'] == deck['mesh'] and forecast['date'] == deck['date']
    missing = {}
    for field in forecast['comparison']['fields']:
        if field['status'] != 'field missing in hex':
            continue
        key = '|'.join(str(field[k]) for k in ('stream', 'field', 'step'))
        assert key not in missing
        missing[key] = dict(status=field['status'], words=field['compared_words'])
    expected = {key: value for key, value in inventory.items()
                if value['status'] == 'field missing in hex'}
    new_samples = {key: value for key, value in missing.items()
                   if expected.get(key) != value}
    receipt = dict(schema='p21-owner-inventory-nongrowth-audit/v1', process_count=1,
                   owner_inventory=str(args.inventory.resolve()),
                   owner_inventory_sha256=sha(args.inventory),
                   verify_receipt=str(args.forecast.resolve()),
                   verify_sha256=sha(args.forecast), missing_records=len(missing),
                   deck=str(args.deck.resolve()), deck_sha256=sha(args.deck),
                   missing_words=sum(v['words'] for v in missing.values()),
                   new_missing_samples=len(new_samples),
                   missing_inventory_exactly_matches=missing == expected,
                   private_ratchet_created=False, acceptance_unchanged=True)
    assert not args.receipt.exists()
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0 if not new_samples else 1


if __name__ == '__main__':
    raise SystemExit(main())
