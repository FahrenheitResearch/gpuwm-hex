"""Bind a whole proof inventory to unchanged native option deck identities.

This is metadata provisioning, never an acceptance decision. Actual verifier
receipts retain their numerical and missing-field evidence. The reproducibility
door remains closed until complete forecasts and framework closure are proven.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(deck):
    return (deck['mesh'], deck['date'], deck['process_count'],
            deck['inputs']['namelist']['sha256'], deck['inputs']['streams']['sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--required', type=Path, required=True)
    parser.add_argument('--plan', type=Path, action='append', default=[])
    parser.add_argument('--deck', type=Path, action='append', default=[])
    parser.add_argument('--verifier-receipt', type=Path, action='append', default=[])
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    required = json.loads(args.required.read_text())
    assert required['process_count'] == 1
    assert len(required['requests']) == required['required_requests']
    candidates = []
    plan_hashes = {}
    for path in args.plan:
        plan_hashes[str(path)] = sha(path)
        for entry in json.loads(path.read_text())['decks']:
            deck = Path(entry['path'])
            assert sha(deck) == entry['sha256']
            candidates.append(deck)
    candidates.extend(args.deck)
    by_identity = {}
    for path in candidates:
        deck = json.loads(path.read_text())
        assert deck['process_count'] == deck['ranks'] == 1
        command = deck['hex_command']
        assert command[command.index('--physics-backend') + 1] == 'mpasphys'
        overlays = [command[i + 1] for i, item in enumerate(command[:-1]) if item == '--physics']
        assert overlays == ["config_radt_cld_backend='hex-cuda'"]
        for name in ('namelist', 'streams', 'init'):
            packet = deck['inputs'][name]
            assert sha(Path(packet['path'])) == packet['sha256']
        assert sha(Path(deck['native']['path'])) == deck['native']['sha256']
        key = identity(deck)
        assert key not in by_identity, 'duplicate native option identity'
        by_identity[key] = (path, deck)
    receipts = {}
    for path in args.verifier_receipt:
        data = json.loads(path.read_text())
        assert data['process_count'] == 1
        assert data['deck'] not in receipts, 'duplicate verifier receipt for a selected deck'
        receipts[data['deck']] = (path, data)
    bound, missing = [], []
    seen = set()
    for number, request in enumerate(required['requests']):
        key = (request['mesh'], request['date'], request['process_count'],
               request['namelist_sha256'], request['streams_sha256'])
        assert key not in seen, 'required requests must have distinct native identities'
        seen.add(key)
        if key not in by_identity:
            missing.append(number)
            continue
        path, deck = by_identity.pop(key)
        row = dict(path=str(path.resolve().relative_to(args.output.parent.resolve())),
                   request_index=number, deck_sha256=sha(path),
                   status='native bytes bound; whole forecast proof pending')
        if deck['id'] in receipts:
            proof_path, proof = receipts[deck['id']]
            assert proof['provenance']['deck_sha256'] == row['deck_sha256']
            row.update(receipt=str(proof_path), receipt_sha256=sha(proof_path),
                       evidence_kind=proof['evidence_kind'], model_execution=proof['model_execution'],
                       verifier_passed=proof['passed'], acceptance_tier=proof['acceptance_tier'],
                       status='whole attempt recorded; framework and complete output acceptance pending')
        bound.append(row)
    assert not by_identity, 'a supplied deck has no required native option identity'
    index = dict(schema='p21-whole-model-matrix/v1', process_count=1,
                 ready=False, matrix_complete=False, decks=bound,
                 required_requests=required['required_requests'], unbound_requests=missing,
                 required_matrix_sha256=sha(args.required), plan_sha256=plan_hashes,
                 comparator=str(args.comparator.resolve()), comparator_sha256=sha(args.comparator),
                 pending_reason='Complete every-stream BITWISE forecasts, native host authority, '
                 'ozone nested/time CUDA hooks and radiation radius consumers remain unproven.',
                 inventory_only=True, missing_fields_are_not_numerical_differences=True)
    temporary = args.output.with_suffix('.partial')
    temporary.write_text(json.dumps(index, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(dict(bound=len(bound), required=required['required_requests'],
                          whole_attempts=sum('receipt' in row for row in bound), ready=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
