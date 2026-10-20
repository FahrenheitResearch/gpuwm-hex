"""Authenticate and summarize every complete whole-forecast attempt matrix.

MPAS mpas_atmphys_driver_seaice.F:160-558 defines the subsystem boundary.
This reads comparison metadata only; Rust computes all weather word diffs.
Failed attempts and missing samples remain explicit, without a new grade.
"""
import argparse
import hashlib
import json
from pathlib import Path

from forecast_summary import summarize


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    attempts, sources = [], []
    for matrix_path in args.matrix:
        payload = matrix_path.read_bytes()
        matrix = json.loads(payload)
        assert matrix['process_count'] == 1
        assert len(matrix['runs']) == matrix['expected_decks']
        sources.append(dict(path=str(matrix_path), sha256=digest(payload)))
        for index, record in enumerate(matrix['runs']):
            assert record['process_count'] == 1
            sealed = record['verification_receipt']
            assert sealed is not None, 'a missing actual receipt cannot be summarized'
            original = Path(sealed['path'])
            local = args.receipts / original.parent.name / original.name
            payload = local.read_bytes()
            assert digest(payload) == sealed['sha256']
            receipt = json.loads(payload)
            assert receipt['process_count'] == 1
            fields = receipt.get('comparison', {}).get('fields', [])
            compared = [field for field in fields if field['status'] == 'compared']
            assert record['numerical_compared_words'] == sum(
                field['compared_words'] for field in compared)
            assert record['numerical_differing_words'] == (
                sum(field['differing_words'] for field in compared) if compared else None)
            assert record['missing_or_incompatible_samples'] == len(fields) - len(compared)
            summary = summarize(receipt)
            target = args.out / f'{matrix_path.stem}-{index:02d}-fields.json'
            target.write_text(json.dumps(summary, indent=2) + '\n')
            model = record.get('model_progress')
            if model:
                backends = model['physics']['backends']
                assert backends['seaice'] == 'hex-cuda'
                assert all(value == 'native-cpu' for key, value in backends.items()
                           if key != 'seaice')
                assert model['physics']['options']['config_frac_seaice'] is True
            attempts.append(dict(
                deck=receipt['deck'], receipt=sealed, source_matrix=str(matrix_path),
                field_summary=dict(path=str(target), sha256=digest(target.read_bytes())),
                evidence_kind=receipt['evidence_kind'], passed=receipt['passed'],
                acceptance_tier=receipt.get('acceptance_tier'),
                completed=False if model is None else model['completed'],
                steps_requested=None if model is None else model['steps_requested'],
                steps_completed=None if model is None else model['steps_completed'],
                numerical_compared_words=record['numerical_compared_words'],
                numerical_differing_words=record['numerical_differing_words'],
                missing_or_incompatible_samples=record['missing_or_incompatible_samples'],
                native_library_sha256=None if model is None else model['native_library_sha256'],
                backends=None if model is None else model['physics']['backends'],
                selectors=None if model is None else model['physics']['selectors']))
    unique = {item['deck'] for item in attempts}
    assert len(unique) == args.expected_unique_decks
    full = [item for item in attempts if item['completed'] and
            item['steps_completed'] == item['steps_requested']]
    result = dict(kind='authenticated complete forecast-attempt matrices; no replacement grade',
                  process_count=1, matrices=sources, attempts=attempts,
                  unique_decks=len(unique), completed_attempts=len(full),
                  startup_or_incomplete_attempts=len(attempts) - len(full),
                  computed_words=sum(item['numerical_compared_words'] for item in full),
                  numerical_differing_words=sum(item['numerical_differing_words'] or 0
                                                for item in full),
                  missing_or_incompatible_samples=sum(item['missing_or_incompatible_samples']
                                                      for item in full),
                  exclusions=[], acceptance_tier=None)
    (args.out / 'complete-attempt-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('matrices', 'attempts')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, action='append', required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-unique-decks', type=int, required=True)
    run(parser.parse_args())
