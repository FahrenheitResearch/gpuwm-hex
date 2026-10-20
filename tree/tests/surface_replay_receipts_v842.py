"""Audit fresh direct-NVRTC replay evidence without accepting a forecast claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re

CUDA_OPTIONS = ['--std=c++17', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']


def audit(root):
    science = root/'hex/tree/src/hexcore/mpasphys/surface_update_v842.py'
    source_sha = hashlib.sha256(science.read_bytes()).hexdigest()
    fresh = root/'receipts/direct-nvrtc'
    groups = {
        'synthetic': (list(fresh.glob('m*-*.json')), 45),
        'persistent': ([fresh/'repeated-restart.json'], 1),
        'monthly': (list(fresh.glob('real-monthly-*.json')), 3),
        'land-use': (list(fresh.glob('land-use-*.json')), 2),
        'complete-March': (list(fresh.glob('complete-real-mode*.json')), 7),
        'native-stream': ([fresh/'real-surface-stream.json'], 1),
        'subnormal': (list(fresh.glob('subnormal-mode*.json')), 3),
        'complete-winter': (list((root/'receipts/direct-nvrtc-winter').glob('complete-real-mode*.json')), 7),
    }
    errors = []
    records = {}
    exit_path = root/'receipts/direct-nvrtc-gpu.exit'
    if not exit_path.is_file() or exit_path.read_text().strip() != '0':
        errors.append('Fresh slot-backed batch lacks actual exit0')
    for name, (paths, count) in groups.items():
        if len(paths) != count or any(not p.is_file() for p in paths):
            errors.append(f'{name} lacks all {count} comparison receipts')
            continue
        rows = [json.loads(p.read_text()) for p in paths]
        differing = 0
        for path, row in zip(paths, rows):
            if (row.get('process_count') != 1 or row.get('cuda_compiler') != 'direct NVRTC'
                    or row.get('cuda_options') != CUDA_OPTIONS
                    or row.get('science_source_sha256') != source_sha):
                errors.append(f'Compiler/source/process provenance mismatch: {path}')
            differences = row.get('differing_words')
            if differences is None:
                differences = sum(r['differing_words'] for r in row['records'])
            differing += differences
            if differences:
                errors.append(f'Differing words: {path}')
        records[name] = dict(cases=count, words=sum(row['words'] for row in rows),
            differing_words=differing,
            real_carrier_words=sum(row.get('real_carrier_words', 0) for row in rows),
            receipts=[str(p.relative_to(root)) for p in sorted(paths)])
    witness_path = root/'receipts/ftz-witness/summary.json'
    if not witness_path.is_file() or json.loads(witness_path.read_text()).get('regression_detects_legacy_ftz') is not True:
        errors.append('Legacy compiler subnormal failure witness is absent')
    for folder, expected in [('direct-nvrtc',67), ('direct-nvrtc-winter',7)]:
        log_path = root/'receipts'/folder/'run.log'
        log = log_path.read_text() if log_path.exists() else ''
        if not re.search(rf'\b{expected} passed\b', log) or re.search(r'\b(?:skipped|failed|errors?)\b', log):
            errors.append(f'{folder} lacks all {expected} passing/no-skip tests')
    result = dict(tier='subsystem CUDA/strict Fortran; no whole-forecast claim',
        process_count=1, cuda_compiler='direct NVRTC', cuda_options=CUDA_OPTIONS,
        science_source_sha256=source_sha, groups=records, errors=errors,
        all_new_replays_passed=not errors,
        words=sum(group['words'] for group in records.values()),
        differing_words=sum(group['differing_words'] for group in records.values()),
        comparison_cases=sum(group['cases'] for group in records.values()))
    (root/'receipts/direct-nvrtc-replay-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['all_new_replays_passed'] else 1)
