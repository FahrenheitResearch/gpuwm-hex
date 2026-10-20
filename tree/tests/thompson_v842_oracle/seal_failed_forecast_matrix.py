"""Seal every actual failed forecast in the complete Thompson test matrix.

Orchestration only. Field words come from Rust comparison receipts. Startup
word equality does not certify a forecast; preserved native progress is required.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from summarize_forecast_matrix import LABELS, summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('preparation', type=Path)
    parser.add_argument('patch', type=Path)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    prepared = load(args.preparation)
    summary = summarize(args.directory)
    assert summary['complete_attempt_matrix'] and summary['complete_matrix']
    assert tuple(row['label'] for row in summary['rows']) == LABELS
    logs = args.directory / 'sealed-model-logs'
    logs.mkdir(exist_ok=True)
    rows = []
    for row in summary['rows']:
        label = row['label']
        receipt_path = args.directory / (label + '.json')
        receipt = load(receipt_path)
        assert not receipt['passed'] and receipt['model_execution']
        assert receipt['acceptance_tier'] is None and receipt['process_count'] == 1
        assert receipt['provenance']['hex_source']['commit'] == prepared['commit']
        assert not receipt['provenance']['hex_source']['dirty']
        failure = row['model_failure']
        assert failure and '; see ' in failure
        original_log = Path(failure.split('; see ', 1)[1])
        destination = logs / (label + '.log')
        shutil.copyfile(original_log, destination)
        assert sha(destination) == sha(original_log)
        assert 'native prepared array qv_p is not allocated before CUDA slot' in destination.read_text()
        attempts = sorted((args.directory / 'metadata').glob(label + '-attempt-*'),
                          key=lambda path: int(path.name.rsplit('-', 1)[1]))
        metadata = attempts[-1]
        seal = load(metadata / 'preservation.json')
        for name, expected in seal['observed_sha256'].items():
            assert sha(metadata / name) == expected
        request = load(metadata / 'native-physics-request.json')
        progress = load(metadata / 'native-physics-receipt.json')
        assert not progress['completed'] and progress['steps_completed'] == 0
        assert progress['steps_requested'] == request['steps']
        backends = request['physics']['backends']
        assert backends['microphysics'] == 'hex-cuda'
        assert all(value == 'native-cpu' for key, value in backends.items() if key != 'microphysics')
        assert request['physics']['selectors']['microphysics'] == 'mp_thompson'
        rows.append(dict(label=label, receipt_sha256=sha(receipt_path),
                         model_log=str(destination), model_log_sha256=sha(destination),
                         metadata=str(metadata), preservation_sha256=sha(metadata / 'preservation.json'),
                         native_progress_sha256=sha(metadata / 'native-physics-receipt.json'),
                         request_sha256=sha(metadata / 'native-physics-request.json'),
                         steps_completed=0, steps_requested=request['steps'],
                         computed_words=row['computed_words'],
                         computed_differing_words=row['computed_differing_words'],
                         missing_words=row['missing_or_incompatible_words']))
    result = dict(acceptance=False, forecast_proven=False, complete_attempt_matrix=True,
                  source_commit=prepared['commit'], production_sha256=prepared['source_sha256'],
                  provider_sha256=prepared['provider_sha256'], process_count=1, duration_hours=24,
                  preparation_sha256=sha(args.preparation),
                  summary_sha256=sha(args.directory / 'summary.json'), rows=rows,
                  reconcile_patch_sha256=sha(args.patch),
                  reason='All eight actual forecasts stop before the Thompson CUDA callback: native qv_p is unallocated.',
                  scope='Failed whole-model attempts; startup equality is not forecast identity.')
    for name in ('computed_words', 'computed_differing_words', 'missing_words'):
        result[name] = sum(row[name] for row in rows)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print('Eight failed whole forecasts sealed; no acceptance tier.')


if __name__ == '__main__':
    main()
