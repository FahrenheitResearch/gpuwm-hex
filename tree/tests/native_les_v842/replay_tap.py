"""Replay full native LES entry storage, compare every native output word.

MPAS mpas_atm_dissipation_models.F:208-1606. Rust owns record decode/encode
and word comparison; Python transfers the native arrays to/from CUDA only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema') != 'd2-les-native-tap-capture-v1' or manifest.get('process_count') != 1:
        raise ValueError('unqualified native capture manifest')
    expected_banks = {case['model'] for case in manifest['cases']}
    if expected_banks != {'3d_smagorinsky', 'prognostic_1.5_order'}:
        raise ValueError('both native LES closure captures are required')
    for case in manifest['cases']:
        if case['exit_code'] != 0:
            raise ValueError('native capture did not complete')
        bank = args.captures / case['model']
        if {p.name for p in bank.glob('*.bin')} != {r['name'] for r in case['records']}:
            raise ValueError('native record inventory changed')
        for record in case['records']:
            path = bank / record['name']
            if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('native capture record hash mismatch: ' + str(path))
    import cupy as cp
    from hexcore.mixing_v842_les import LesDissipationV842, ROUTINES
    helper = str(args.records.resolve())
    launch = LesDissipationV842()
    result = dict(schema='d2-les-native-tap-replay-v1', calls=[], words=0,
                  differing_words=0, passed=False, capture_banks=[])
    args.out.mkdir(parents=True, exist_ok=True)
    for bank in sorted(path for path in args.captures.iterdir() if path.is_dir()):
        if bank.name not in expected_banks:
            raise ValueError('unenrolled capture bank')
        files = sorted(bank.glob('*.pre.*.bin'))
        if not files:
            continue
        result['capture_banks'].append(bank.name)
        prefixes = sorted({str(path).split('.pre.')[0] for path in files})
        observed = set()
        for prefix in prefixes:
            routine = Path(prefix).name.split('.s')[0]
            spec = ROUTINES[routine]
            observed.add(routine)
            values, headers = {}, {}
            for name in spec['args']:
                path = prefix + '.pre.' + name + '.bin'
                header = json.loads(subprocess.check_output([helper, 'header', path, '-']))
                headers[name] = header
                dtype = '<f4' if spec['fields'][name]['type'] == 'float' else '<i4'
                raw = subprocess.check_output([helper, 'payload', path, '-'])
                host = np.frombuffer(raw, dtype=dtype).reshape(tuple(header['shape']), order='F')
                values[name] = (cp.array(host, order='F') if header['shape'] or dtype == '<f4'
                                else int(host.item()))
            launch.call(routine, **values)
            cp.cuda.get_current_stream().synchronize()
            row = dict(bank=bank.name, call=Path(prefix).name, routine=routine,
                       words=0, differing_words=0, fields=[])
            for name in spec['args']:
                # Includes untouched inputs and sentinel storage, as well as every output.
                host = cp.asnumpy(values[name]) if isinstance(values[name], cp.ndarray) else np.array(values[name], dtype='<i4')
                actual = args.out / (bank.name + '-' + Path(prefix).name + '.post.' + name + '.bin')
                subprocess.run([helper, 'wrap', prefix + '.pre.' + name + '.bin', str(actual)],
                               input=host.tobytes(order='F'), check=True)
                expected = prefix + '.post.' + name + '.bin'
                comparison = json.loads(subprocess.check_output([helper, 'compare', expected, str(actual)]))
                row['fields'].append(dict(name=name, **comparison))
                if comparison.get('schema_mismatch') or type(comparison.get('different_words')) is not int:
                    raise ValueError('native/CUDA record schema mismatch')
                row['words'] += comparison['words']
                row['differing_words'] += comparison['different_words']
                actual.unlink()
            result['calls'].append(row)
            result['words'] += row['words']
            result['differing_words'] += row['differing_words']
            print(f"{bank.name} {row['call']}: {row['differing_words']}/{row['words']} different", flush=True)
        if observed != set(ROUTINES):
            raise ValueError('capture bank does not cover the complete LES module')
    if not result['calls']:
        raise ValueError('no real native calls found')
    result['module_sha256'] = hashlib.sha256(Path(__import__('hexcore.mixing_v842_les', fromlist=['x']).__file__).read_bytes()).hexdigest()
    result['passed'] = result['differing_words'] == 0
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
