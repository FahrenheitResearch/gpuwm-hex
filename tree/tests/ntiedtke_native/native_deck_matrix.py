"""Prepare every published single-option deck using the unchanged strict model.

Only executable orchestration, opaque copies and metadata are processed here.
Native MPAS writes the fields; the published Rust referee inventories them.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main(args):
    binary = args.build / 'atmosphere_model'
    manifest = json.loads((args.build / 'MANIFEST.json').read_text())
    audit = json.loads((args.build / 'flag-audit.json').read_text())
    if (audit.get('recipe') != 'gnu-strict' or audit.get('all_ok') is not True
            or manifest.get('recipe') != 'gnu-strict'
            or hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['binary_sha256']):
        raise ValueError('published strict native executable/audit differs')
    args.work.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=True)
    requested = json.loads(args.matrix.read_text())
    if not requested or len({row['key'] for row in requested}) != len(requested):
        raise ValueError('native matrix requires distinct, nonempty deck rows')
    runs = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for row in requested:
        root = args.work / row['key']
        native = root / 'native'
        root.mkdir(exist_ok=True)
        command = [sys.executable, str(args.runner), '--build', str(args.build),
                   '--deck', row['deck'], '--case', row['case'],
                   '--out', str(native), '--tables', str(args.tables),
                   '--mpi', str(args.mpi), '--ranks', '1',
                   '--pin-set-hash', args.pin_set]
        status = 0
        if not (native / 'MANIFEST.json').is_file():
            with (args.receipts / (row['key'] + '.native.log')).open('w') as log:
                status = subprocess.run(command, env=env, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
        if status == 0:
            recorded = json.loads((native / 'MANIFEST.json').read_text())
            if (recorded['ranks'] != 1 or recorded['pin_set_hash'] != args.pin_set
                    or recorded['binary_sha256'] != manifest['binary_sha256']):
                raise ValueError('completed native run does not match the strict matrix')
            for filename in ('namelist.atmosphere', 'streams.atmosphere'):
                if hashlib.sha256((Path(row['deck']) / filename).read_bytes()).hexdigest() != recorded['inputs'][filename]:
                    raise ValueError('completed native run belongs to different deck bytes')
        reference = root / 'verification'
        if status == 0 and not (reference / 'deck.json').is_file():
            prepare = [sys.executable, str(Path(__file__).with_name('prepare_verification.py')),
                       '--native', str(native), '--inputs-root', row['deck'],
                       '--case', row['case'], '--output', str(reference),
                       '--receipt', str(args.receipts / (row['key'] + '.reference.json'))]
            for name in ('referee', 'comparator', 'library', 'pool_reader', 'mpas_source'):
                prepare += ['--' + name.replace('_', '-'), str(getattr(args, name))]
            prepare += ['--stream-list-directory', str(args.build),
                        '--native-physics-tables', str(args.tables)]
            with (args.receipts / (row['key'] + '.prepare.log')).open('w') as log:
                status = subprocess.run(prepare, env=env, stdout=log,
                                        stderr=subprocess.STDOUT).returncode
        runs.append(dict(key=row['key'], argv=command, exit_code=status,
                         native_process_count=1, native_host=os.uname().nodename,
                         deck=str(reference / 'deck.json'),
                         completed_reference=status == 0 and (reference / 'deck.json').is_file()))
        (args.receipts / 'matrix.json').write_text(json.dumps(dict(
            schema='mpas-ntiedtke-single-option-native-matrix/v1',
            acceptance=False, requested=requested, runs=runs,
            complete=len(runs) == len(requested),
            all_references_prepared=(len(runs) == len(requested)
                                     and all(r['completed_reference'] for r in runs))),
            indent=2) + '\n')
    return 0 if all(r['completed_reference'] for r in runs) else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix', 'work', 'receipts', 'runner', 'build', 'tables', 'mpi',
                 'referee', 'comparator', 'library', 'pool-reader', 'mpas-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pin-set', required=True)
    raise SystemExit(main(parser.parse_args()))
