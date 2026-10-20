"""Publish a proven forecast plan only from twelve authenticated actual runs."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--exit-receipt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.attempt.startswith('reproduce-noah-') and '/' not in args.attempt
    assert int(args.exit_receipt.read_text()) == 0, 'actual forecast controller did not exit successfully'
    plan = json.loads(args.plan.read_text())
    assert plan['process_count'] == 1 and plan['duration_hours'] == 24
    assert len(plan['rows']) == 12 and len({r['label'] for r in plan['rows']}) == 12
    receipts = args.plan.resolve().parent / args.attempt
    summary_path = receipts / 'summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['passed'] and not summary['all_native_fields_bitwise']
    assert summary['process_count'] == 1 and summary['duration_hours'] == 24
    assert summary['commit'] == plan['initial_checkout_commit']
    assert len(summary['cases']) == 12
    cases = {row['label']: row for row in summary['cases']}
    assert len(cases) == 12 and cases.keys() == {row['label'] for row in plan['rows']}
    checker = runpy.run_path(str(Path(__file__).with_name('reproduce_written_forecasts.py')))['written_failures']
    seals = []
    for row in plan['rows']:
        label = row['label']; case = cases[label]
        assert case['accepted'] and not case['failures']
        assert case['differing_words'] == case['stored_differing_words'] == 0
        model_path = receipts / (label + '-model.json')
        verify_path = receipts / (label + '-verify.json')
        stored_path = receipts / (label + '-stored.json')
        execution_path = receipts / (label + '-comparison-execution.json')
        model = json.loads(model_path.read_text())
        assert sha(model_path) == case['model_receipt_sha256']
        assert sha(verify_path) == case['verify_sha256']
        assert model['exit_code'] == 0 and model['completed'] and model['source_clean_after']
        assert model['process_count'] == 1 and model['commit'] == summary['commit']
        assert model['production_sha256'] == plan['production_sha256']
        assert sha(row['deck']) == row['deck_sha256'] == model['deck_sha256']
        deck = json.loads(Path(row['deck']).read_text())
        assert model['steps_completed'] == model['steps_requested'] == deck['duration_steps']
        assert model['backends']['land_surface'] == 'hex-cuda'
        assert all(value == 'native-cpu' for key, value in model['backends'].items() if key != 'land_surface')
        execution = json.loads(execution_path.read_text())
        assert execution['model'] == model
        assert execution['verify_sha256'] == sha(verify_path)
        assert execution['stored_sha256'] == sha(stored_path)
        fields = json.loads(verify_path.read_text())['comparison']['fields']
        stored = json.loads(stored_path.read_text())
        inventory = row['foundation_inventory']
        assert sha(inventory['path']) == inventory['sha256'] == case['foundation_inventory_sha256']
        baseline = json.loads(Path(inventory['path']).read_text())['fields']
        assert not checker(fields, baseline, stored, row.get('native_inventory_subset', False))
        clocks = {}
        for field in fields:
            clocks.setdefault(field['stream'], set()).add(field['step'])
        assert clocks == {name: set(steps) for name, steps in deck['expected_samples'].items()}
        assert max(field['step'] for field in fields) == deck['duration_steps']
        assert case['computed_words'] == sum(f['compared_words'] for f in fields if f['status'] == 'compared')
        assert case['stored_words'] == stored['words'] and case['missing_samples'] == sum(f['status'] != 'compared' for f in fields)
        seals.append(dict(label=label, model_sha256=sha(model_path),
                          execution_sha256=sha(execution_path), verify_sha256=sha(verify_path),
                          stored_sha256=sha(stored_path), inventory_sha256=inventory['sha256']))
    assert not args.out.exists(), 'never overwrite a previously sealed proof plan'
    plan['forecast_proven'] = True
    plan['all_native_fields_bitwise'] = False
    plan['proven_identity_tier'] = 'BITWISE-WRITTEN-FIELDS'
    plan['dependencies']['accepted-forecast-matrix'] = dict(path=str(summary_path), sha256=sha(summary_path))
    plan['accepted_execution'] = dict(exit_receipt_sha256=sha(args.exit_receipt), cases=seals)
    args.out.write_text(json.dumps(plan, indent=2) + '\n')
    print('Authenticated twelve-deck forecast plan:', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
