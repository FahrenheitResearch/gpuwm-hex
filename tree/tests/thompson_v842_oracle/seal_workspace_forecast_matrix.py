"""Seal every real forecast and charged word in a complete Thompson attempt.

Scientific comparisons come from the Rust verifier. Native progress is required
separately; complete execution alone never supplies an acceptance tier.
"""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_forecast_matrix import LABELS, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def seal(directory, preparation, receipt_path):
    prepared = load(preparation)
    summary = summarize(directory)
    assert summary['complete_attempt_matrix'] and summary['complete_matrix']
    rows = []
    for row in summary['rows']:
        label = row['label']
        actual_path = directory / (label + '.json')
        actual = load(actual_path)
        assert actual['process_count'] == 1 and actual['model_execution']
        assert actual['provenance']['hex_source']['commit'] == prepared['commit']
        assert not actual['provenance']['hex_source']['dirty']
        attempts = sorted((directory / 'metadata').glob(label + '-attempt-*'),
                          key=lambda path: int(path.name.rsplit('-', 1)[1]))
        metadata = attempts[-1]
        preservation = load(metadata / 'preservation.json')
        for name, expected in preservation['observed_sha256'].items():
            assert sha(metadata / name) == expected
        request = load(metadata / 'native-physics-request.json')
        progress = load(metadata / 'native-physics-receipt.json')
        assert progress['steps_requested'] == request['steps']
        assert progress['native_library_sha256'] == prepared['provider_sha256']
        assert progress['physics'] == request['physics']
        backends = request['physics']['backends']
        assert backends['microphysics'] == 'hex-cuda'
        assert all(v == 'native-cpu' for k, v in backends.items() if k != 'microphysics')
        assert request['physics']['selectors']['microphysics'] == 'mp_thompson'
        rows.append(dict(label=label, receipt_sha256=sha(actual_path),
            metadata=str(metadata), preservation_sha256=sha(metadata / 'preservation.json'),
            native_progress_sha256=sha(metadata / 'native-physics-receipt.json'),
            request_sha256=sha(metadata / 'native-physics-request.json'),
            completed=progress['completed'], steps_completed=progress['steps_completed'],
            steps_requested=progress['steps_requested'], verifier_passed=actual['passed'],
            acceptance_tier=actual['acceptance_tier'], model_failure=row['model_failure'],
            computed_words=row['computed_words'], computed_differing_words=row['computed_differing_words'],
            missing_or_incompatible_words=row['missing_or_incompatible_words']))
    assert tuple(row['label'] for row in rows) == LABELS
    result = dict(schema='p3-native-workspace-eight-forecast-attempt/v1',
        acceptance=False, complete_attempt_matrix=True, process_count=1, duration_hours=24,
        source_commit=prepared['commit'], production_sha256=prepared['source_sha256'],
        provider_sha256=prepared['provider_sha256'], preparation_sha256=sha(preparation),
        summary_sha256=sha(directory / 'summary.json'), rows=rows,
        all_forecasts_completed=all(r['completed'] and r['steps_completed'] == r['steps_requested'] for r in rows),
        all_verifier_receipts_passed=all(r['verifier_passed'] for r in rows),
        scope='Complete original eight-deck actual CUDA attempt; all missing words charged. Final trial gates and reproducer still required.')
    for name in ('computed_words', 'computed_differing_words', 'missing_or_incompatible_words'):
        result[name] = sum(row[name] for row in rows)
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'preparation', 'receipt_path'):
        parser.add_argument(name, type=Path)
    result = seal(**vars(parser.parse_args()))
    print('Eight actual Thompson CUDA attempts sealed; completed:', result['all_forecasts_completed'],
          'computed differing words:', result['computed_differing_words'],
          'missing/incompatible words:', result['missing_or_incompatible_words'])
