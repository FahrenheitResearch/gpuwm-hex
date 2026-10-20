"""Authenticate complete native/host/CUDA driver replays and their raw words."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    settings_path = root / 'settings.json'
    settings = json.loads(settings_path.read_text())
    if settings.get('complete') is not True or not settings.get('settings'):
        raise ValueError('a nonempty complete input matrix is required')
    expected = settings['settings']
    matrices, pins, totals = {}, {}, {}
    for mode, execution in (('host', 'host-glibc-witness'),
                            ('cuda', 'hex-cuda'),
                            ('native-repeat', 'strict-native-repeat')):
        path = root / mode / 'matrix.json'
        matrix = json.loads(path.read_text())
        if (matrix.get('complete') is not True or matrix.get('passed') is not True
                or matrix.get('native_process_count') != 1
                or matrix.get('execution') != execution
                or matrix.get('settings_sha256') != digest(settings_path)
                or matrix.get('requested') != len(expected)
                or len(matrix.get('runs', [])) != len(expected)):
            raise ValueError(f'{mode}: incomplete or unauthenticated matrix')
        totals[mode] = 0
        for index, row in enumerate(matrix['runs']):
            if row.get('index') != index or row.get('setting') != expected[index]:
                raise ValueError(f'{mode}: input order or setting differs')
            hashes = row.get('native_csv_sha256', {})
            if set(hashes) != {'nt-levels.csv', 'nt-surface.csv', 'nt-isolation.csv'}:
                raise ValueError(f'{mode}: all three native CSV hashes required')
            if mode == 'native-repeat':
                if row.get('identical') is not True:
                    raise ValueError('native repeat is not identical')
                continue
            receipt = root / mode / f'{index}.json'
            if digest(receipt) != row.get('sha256'):
                raise ValueError(f'{mode}: replay receipt hash differs')
            result = json.loads(receipt.read_text())
            if (result.get('execution') != execution
                    or result.get('differing_words') != 0
                    or result.get('repeat_differing_words') != 0
                    or result.get('restart_counter_one_checked') is not
                       (expected[index].get('initflag', 0) == 0)
                    or result.get('native_spread_or_forecast_proof') is not False):
                raise ValueError(f'{mode} setting {index}: solve, repeat or restart failed')
            arrays = [array for group in result['results']
                      for array in group['arrays'].values()]
            words = sum(array['words'] for array in arrays)
            if (not arrays or words <= 0
                    or any(array['differing_words'] != 0 for array in arrays)
                    or words != row.get('compared_words')
                    or row.get('differing_words') != 0):
                raise ValueError(f'{mode}: declared output words differ')
            totals[mode] += words
        matrices[mode] = matrix
        pins[mode] = dict(path=str(path), sha256=digest(path))
    for index in range(len(expected)):
        if not (matrices['host']['runs'][index]['native_csv_sha256']
                == matrices['cuda']['runs'][index]['native_csv_sha256']
                == matrices['native-repeat']['runs'][index]['native_csv_sha256']):
            raise ValueError('host, CUDA and strict repeat use different native words')
    if totals['host'] != totals['cuda']:
        raise ValueError('host and CUDA declared output counts differ')
    strict_hash = matrices['native-repeat'].get('native_binary_sha256')
    if not strict_hash or len(strict_hash) != 64:
        raise ValueError('strict uninstrumented native binary pin required')
    return dict(schema='ntiedtke-authenticated-whole-driver-replay/v1',
                complete=True, passed=True, acceptance=False,
                native_process_count=1, settings=len(expected),
                host_words=totals['host'], cuda_words=totals['cuda'],
                differing_words=0, strict_plain_sha256=strict_hash,
                settings_sha256=digest(settings_path), matrices=pins,
                every_branch_covered=False,
                forecast_proof=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
