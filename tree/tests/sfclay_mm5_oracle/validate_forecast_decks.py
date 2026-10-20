"""Authenticate prepared references with f2's real Rust stream reader.

This exercises verification admission only; it never certifies a hex forecast.
No weather field decoding or numerical processing occurs in Python.
"""
import argparse, hashlib, json
from pathlib import Path
from hexcore.verify_mpas import native_manifest, comparator_path, file_ref, validate_model_command, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('forecast_proof_plan.json'))
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--comparator', type=Path, required=True)
args = parser.parse_args()
plan = json.loads(args.plan.read_text())
binary = comparator_path(args.comparator)
rows = []
result = dict(scope='native publication and real Rust stream admission; no hex forecast',
              status='PASS', forecast_proven=False, process_count=1, excluded_words=[],
              comparator_sha256=hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
              plan_sha256=hashlib.file_digest(args.plan.open('rb'), 'sha256').hexdigest(), decks=rows)
for row in plan['decks']:
    path = args.plan.parent/row['path']
    try:
        file_ref(dict(path=str(path.resolve()), sha256=row['sha256']), args.plan.parent)
        deck = json.loads(path.read_text())
        validate_model_command(deck['hex_command'])
        for provider in deck['providers'].values():
            file_ref(provider, path.parent)
        _, native = native_manifest(deck['native'], path.parent, deck, binary)
        clocks = {}
        for output in native['outputs']:
            clocks.setdefault(output['stream'], []).extend(output['steps'])
        clocks = {stream: sorted(set(steps)) for stream, steps in clocks.items()}
        assert clocks == deck['expected_samples'], 'Prepared clock does not match Rust record clocks'
        assert native['process_count'] == 1
        measured = dict(status='PASS', id=deck['id'], output_files=len(native['outputs']),
                        records=sum(o['records'] for o in native['outputs']), streams=clocks,
                        native_publication_sha256=deck['native']['sha256'])
    except Exception as error:
        measured = dict(status='FAIL', path=str(path), error=f'{type(error).__name__}: {error}')
        result['status'] = 'FAIL'
    rows.append(measured)
    write_json(args.receipt, result)
    print(measured.get('id', measured.get('path')), measured['status'], flush=True)
raise SystemExit(result['status'] != 'PASS')
