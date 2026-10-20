"""Measure the unchanged strict fork's scalar-filter caller on original inputs.

MPAS-Model-rrfs/src/core_atmosphere/dynamics/mpas_atm_time_integration.F:
3496,3548-3574,3783-3834 at 1d6e370e. No native source or state is repaired.
This measures a first-step option attempt, never a whole-forecast proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'inputs', 'mpi', 'runtime', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--observer', type=Path,
        help='Optional read-only SIGSEGV observer; never alters native code/data/signals')
    args = parser.parse_args()
    manifest = json.loads(args.inputs.read_text())
    case = args.case.resolve()
    for row in manifest['input_rows']:
        if digest(case/row['name']) != row['sha256']:
            raise ValueError('native option input hash mismatch: '+row['name'])
    binary = case/'strict-atmosphere_model'
    if digest(binary) != manifest['native_binary_sha256']:
        raise ValueError('unchanged strict binary authentication failed')
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='rrfs-native-scalar-filter-attempt/v1',
        scope='unchanged strict original real initialization, GSL eddy/filter on, first720s step only; failures are evidence, not output exclusions',
        host=os.uname().nodename, source_commit='1d6e370e7b43a332074809f170532a216c2212fb',
        binary_sha256=digest(binary), inputs_manifest_sha256=digest(args.inputs),
        graph_sha256=digest(case/'x1.2562.graph.info.part.2'),
        options=manifest['options'], runs=[])
    if args.observer:
        receipt['observer_sha256'] = digest(args.observer)
        receipt['observer_descriptor_stack_offset'] = '0x548'
    env = os.environ.copy()
    for name in ('MPAS_ORACLE_DIR', 'MPAS_ORACLE_KEYS', 'MPAS_ORACLE_STEPS'):
        env.pop(name, None)
    env['LD_LIBRARY_PATH'] = str(args.runtime.resolve())+':'+env.get('LD_LIBRARY_PATH', '')
    for ranks in (1, 2):
        for repetition in (1, 2):
            run = args.out/f'ranks-{ranks}-repeat-{repetition}'
            run.mkdir()
            for p in case.iterdir():
                if p.name == 'strict-atmosphere_model':
                    continue
                if p.name in ('namelist.atmosphere', 'streams.atmosphere'):
                    (run/p.name).write_bytes(p.read_bytes())
                else:
                    (run/p.name).symlink_to(p, target_is_directory=p.is_dir())
            command = [str(args.mpi.resolve()), '-n', str(ranks), str(binary)]
            if args.observer:
                # Pinned x86_64 ELF: six pushes +0x488-byte frame; argument24
                # (meshScalingDel4 descriptor) is at rsp+0x4b8+0x90=0x548.
                command = command[:-1]+[str(args.observer.resolve()), str(binary),
                    str((run/'faults').resolve()), '0x548']
            with (run/'run.log').open('wb') as log:
                result = subprocess.run(command, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
            logs = sorted(run.glob('log.atmosphere.*.out'))
            completed = len(logs)==ranks and all('Finished running the atmosphere core' in p.read_text() for p in logs)
            outputs = [dict(path=p.name, bytes=p.stat().st_size, sha256=digest(p))
                       for p in sorted(run.glob('*.nc')) if not p.is_symlink()]
            receipt['runs'].append(dict(process_count=ranks, repetition=repetition,
                command=command, exit_code=result.returncode, completed=completed,
                outputs=outputs, log_sha256=digest(run/'run.log'),
                faults=[json.loads(p.read_text()) for p in sorted((run/'faults').glob('*.json'))]))
            (args.out/'attempts.json').write_text(json.dumps(receipt, indent=2)+'\n')
    for row in manifest['input_rows']:
        if digest(case/row['name']) != row['sha256']:
            raise ValueError('native option input modified by attempt')


if __name__ == '__main__':
    main()
