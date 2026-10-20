"""Bind completed real-door receipts to their retained model completion logs.

This metadata audit preserves the verifier's original acceptance result.
It neither launches a model nor enrolls a missing-field allowance.
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
    for name in ('inventory', 'work-root', 'receipt'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--provider-sha256', required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    assert inventory['process_count'] == 1
    rows = []
    for row in inventory['decks']:
        if not row.get('model_execution') or row['evidence_kind'] != 'FORECAST':
            continue
        original_path = Path(row['receipt'])
        assert sha(original_path) == row['receipt_sha256']
        original = json.loads(original_path.read_text())
        deck_path = Path(row['path'])
        assert sha(deck_path) == row['deck_sha256'] == original['provenance']['deck_sha256']
        deck = json.loads(deck_path.read_text())
        assert original['process_count'] == deck['process_count'] == deck['ranks'] == 1
        directory = (Path(row['work_directory']) if 'work_directory' in row else
                     args.work_root / original_path.parent.name / deck['id'])
        assert directory.name == deck['id']
        log = directory / 'hex.log'
        progress = []
        for line in log.read_text().splitlines():
            if not line.startswith('{'):
                continue
            value = json.loads(line)
            if value.get('schema') == 'mpas-native-forecast-v1':
                progress.append(value)
        assert len(progress) == 1
        model = progress[0]
        assert model['completed']
        assert model['steps_requested'] == model['steps_completed'] == deck['duration_steps']
        assert model['native_library_sha256'] == args.provider_sha256
        assert model['physics']['suite'] == deck['suite']
        backends = model['physics']['backends']
        assert backends['cloudiness'] == 'hex-cuda'
        assert all(value == 'native-cpu' for key, value in backends.items() if key != 'cloudiness')
        written = {}
        for sample in model['streams']:
            written.setdefault(sample['name'], []).append(sample['step'])
        for name, steps in written.items():
            assert steps == deck['expected_samples'][name]
        assert written
        rows.append(dict(request_index=row['request_index'],
            original_receipt=str(original_path.resolve()), original_receipt_sha256=sha(original_path),
            deck=str(deck_path.resolve()), deck_sha256=sha(deck_path),
            model_log=str(log.resolve()), model_log_sha256=sha(log),
            process_count=original['process_count'], steps_completed=model['steps_completed'],
            native_library_sha256=model['native_library_sha256'],
            selectors=model['physics']['selectors'], options=model['physics']['options'],
            backends=backends, written_stream_schedules=written,
            compile_modules=list(model['physics_compile_manifest']['modules']),
            model_identity_checked=model['identity_checked'],
            original_passed=original['passed'], original_acceptance_tier=original['acceptance_tier']))
    assert rows and not args.receipt.exists()
    args.receipt.write_text(json.dumps(dict(schema='p21-whole-model-log-custody/v1',
        inventory_sha256=sha(args.inventory), complete_forecasts=len(rows), rows=rows,
        original_acceptance_unchanged=True, private_ratchet_created=False), indent=2) + '\n')
    print(json.dumps(dict(complete_forecasts=len(rows), completion_logs_authenticated=True)))


if __name__ == '__main__':
    main()
