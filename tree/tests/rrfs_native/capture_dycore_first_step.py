"""Run real-input initial-step captures with unchanged/disabled/active controls.

The full-module overlay is prepared by prepare_dycore_tap.py. All source changes
are observers; raw captured words remain diagnostic until passivity is measured.
This launcher performs only filesystem/process orchestration, never field I/O.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess


def digest(path):
    with path.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256')
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--instrumented', type=Path, required=True)
    parser.add_argument('--mpi', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1,
                        help='number of original native timesteps to capture')
    parser.add_argument('--keys', nargs='+', default=['dycore__' + name for name in (
        'atm_compute_vert_imp_coefs_work', 'atm_advance_acoustic_step_work',
        'atm_advance_scalars_work', 'atm_compute_dyn_tend_work', 'atm_zero_gradient_w_bdy_work')])
    args = parser.parse_args()
    case = args.case.resolve()
    manifest = json.loads(args.inputs.read_text())
    for row in manifest['input_rows']:
        path = case / row['name']
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError(f"native input mismatch: {path}")
    strict = case / 'strict-atmosphere_model'
    if digest(strict) != manifest['native_binary_sha256']:
        raise ValueError('unchanged strict executable mismatch')
    namelist = (case / 'namelist.atmosphere').read_text()
    duration = re.search(r"config_run_duration\s*=\s*'(\d+)_(\d+):(\d+):(\d+)'", namelist)
    timestep = re.search(r'config_dt\s*=\s*([\d.]+)', namelist)
    seconds = sum(int(part)*factor for part,factor in zip(duration.groups(),(86400,3600,60,1))) if duration else None
    if args.steps < 1 or not timestep or seconds != args.steps * float(timestep[1]):
        raise ValueError('diagnostic capture duration must equal the stated number of original native timesteps')
    allowed = {'dycore__' + name for name in (
        'atm_compute_vert_imp_coefs_work', 'atm_advance_acoustic_step_work',
        'atm_advance_scalars_work', 'atm_compute_dyn_tend_work', 'atm_zero_gradient_w_bdy_work', 'atm_advance_scalars_mono_work')}
    cold_keys = {'core__atm_compute_damping_coefs', 'core__atm_compute_cellsoncellsoncell'}
    allowed |= cold_keys
    if not args.keys or not set(args.keys) <= allowed:
        raise ValueError('unsupported native dycore capture boundary')
    root = args.work_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema='rrfs-dycore-initial-steps-capture-v2',
                   capture_host=socket.gethostname(),
                   inputs_manifest_sha256=digest(args.inputs),
                   instrumented_binary_sha256=digest(args.instrumented),
                   strict_binary_sha256=digest(strict), process_count=1,
                   steps=args.steps, native_timestep_seconds=float(timestep[1]),
                   scope=f'real initialized fork input, first {args.steps} steps/{seconds}s; passivity not yet inferred',
                   captured_boundaries=args.keys,
                   runs=[])
    if 'source_authority' in manifest:
        receipt['source_authority'] = manifest['source_authority']
    for kind in ('strict', 'disabled', 'active'):
        run = root / kind
        run.mkdir()
        for source in case.iterdir():
            if source.name == 'strict-atmosphere_model':
                continue
            if source.name in ('namelist.atmosphere', 'streams.atmosphere'):
                (run / source.name).write_bytes(source.read_bytes())
            else:
                (run / source.name).symlink_to(source, target_is_directory=source.is_dir())
        env = os.environ.copy()
        env.pop('MPAS_ORACLE_DIR', None)
        env.pop('MPAS_ORACLE_KEYS', None)
        env.pop('MPAS_ORACLE_STEPS', None)
        env['LD_LIBRARY_PATH'] = str(args.runtime.resolve()) + ':' + env.get('LD_LIBRARY_PATH', '')
        if kind == 'active':
            captures = run / 'captures'
            captures.mkdir()
            env['MPAS_ORACLE_DIR'] = str(captures)
            env['MPAS_ORACLE_KEYS'] = ','.join(args.keys)
            first = 0 if set(args.keys) & cold_keys else 1
            env['MPAS_ORACLE_STEPS'] = ','.join(str(step) for step in range(first, args.steps + 1))
        binary = strict if kind == 'strict' else args.instrumented.resolve()
        command = [str(args.mpi.resolve()), '-n', '1', str(binary)]
        with (run / 'run.log').open('wb') as log:
            result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
        outputs = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
                   for p in sorted(run.glob('*.nc')) if not p.is_symlink()]
        records = list((run / 'captures').glob('*.bin')) if kind == 'active' else []
        native_log = run / 'log.atmosphere.0000.out'
        completed = native_log.exists() and 'Finished running the atmosphere core' in native_log.read_text()
        receipt['runs'].append(dict(kind=kind, command=command, exit_code=result.returncode,
                                    completed=completed,
                                    outputs=outputs, capture_records=len(records),
                                    captured_bytes=sum(p.stat().st_size for p in records)))
        (root / 'capture-runs.json').write_text(json.dumps(receipt, indent=2) + '\n')
    for row in manifest['input_rows']:
        if digest(case / row['name']) != row['sha256']:
            raise ValueError(f"native input changed during capture: {row['name']}")
    if any(r['exit_code'] != 0 or not r['completed'] for r in receipt['runs']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
