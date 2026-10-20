#!/usr/bin/env python3
"""Rebuild trial-checkout instrumentation and compare a real dump-ON forecast.

The profile supplies case/deck choices as metadata. Every output is compared by
the Rust native verifier; a scheme replay cannot substitute for this self-check.
"""
from hexcore import site_runner
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import sysconfig
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(arguments, log, environment):
    with log.open('w') as stream:
        subprocess.run(arguments, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def library_directory(prefix, library):
    """Find the installed library without assuming a distro's lib layout."""
    root = Path(prefix)
    multiarch = sysconfig.get_config_var('MULTIARCH')
    candidates = ['lib64', 'lib']
    if multiarch:
        candidates.append('lib/' + multiarch)
    for relative in candidates:
        if any(path.is_file() for path in (root / relative).glob('lib' + library + '.*')):
            return relative
    raise ValueError(f'installed {library} library missing under {root}')


def forecast_cases(profile):
    """An explicit profile matrix shares one audited native executable."""
    if 'cases' not in profile:
        return [profile]
    selected, names = [], set()
    if not profile['cases']:
        raise ValueError('forecast matrix has no cases')
    allowed = {'name', 'deck', 'case', 'steps', 'keys'}
    for item in profile['cases']:
        name = item.get('name', '')
        if (not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name)
                or name in names or set(item) - allowed):
            raise ValueError('forecast case name or profile override is invalid')
        names.add(name)
        selected.append(dict(profile, **item))
    return selected


def retire_case_outputs(work, dump_members):
    """Retire only output bytes authenticated by the completed comparison."""
    retired, size = 0, 0
    for variant in ('plain', 'oracle'):
        root = work / variant
        manifest = json.loads((root / 'MANIFEST.json').read_text())
        for name, record in manifest['outputs'].items():
            path = root / name
            if not path.is_file() or path.is_symlink():
                continue
            expected = record['sha256'] if isinstance(record, dict) else record
            if digest(path) != expected:
                raise ValueError('completed forecast output changed before retirement')
            size += path.stat().st_size
            path.unlink()
            retired += 1
    for name, record in dump_members.items():
        path = work / 'dumps' / name
        if path.is_symlink() or digest(path) != record['sha256']:
            raise ValueError('completed oracle record changed before retirement')
        size += path.stat().st_size
        path.unlink()
        retired += 1
    (work / 'output-retirement.json').write_text(json.dumps(dict(
        process_count=json.loads((work / 'plain/MANIFEST.json').read_text())['ranks'],
        retired_files=retired, retired_logical_bytes=size,
        all_identical=True, dump_member_manifest_sha256=digest(work / 'dump-members.json'),
        plain_manifest_sha256=digest(work / 'plain/MANIFEST.json'),
        oracle_manifest_sha256=digest(work / 'oracle/MANIFEST.json')), indent=2) + '\n')


def compare_case(profile, work, build, build_work, tools, environment, native, commit, profile_path):
    dump = work / 'dumps'
    dump.mkdir(parents=True)
    reference = Path(profile['reference_build'])
    mpi = Path(profile['mpi_root'])
    common = ['python', str(tools / 'run_forecast.py'), '--deck', profile['deck'],
              '--case', profile['case'], '--tables', profile['tables'],
              '--mpi', str(mpi / 'bin/mpiexec'), '--ranks', str(profile['process_count']),
              '--pin-set-hash', profile['pin_set_sha256']]
    for variant, model in (('plain', reference), ('oracle', build)):
        selected = environment.copy()
        if variant == 'oracle':
            selected.update(MPAS_ORACLE_DIR=str(dump),
                            MPAS_ORACLE_STEPS=profile['steps'], MPAS_ORACLE_KEYS=profile['keys'])
        run(common + ['--build', str(model), '--out', str(work / variant)],
            work / (variant + '.log'), selected)
    if not any(dump.glob('*.pre.*.bin')) or not any(dump.glob('*.post.*.bin')):
        raise ValueError('selected dump-ON run emitted no scheme PRE/POST records')
    from replay.framework_state_check import check as check_framework_state
    framework_receipt = check_framework_state(build, build_work / 'build.log', dump,
        work / 'framework-state-replay', environment,
        capture_process_count=profile['process_count'])
    dump_members = {member.name: dict(sha256=digest(member), bytes=member.stat().st_size)
                    for member in sorted(dump.glob('*.bin'))}
    dump_manifest = work / 'dump-members.json'
    dump_manifest.write_text(json.dumps(dump_members, separators=(',', ':')) + '\n')
    receipt = work / 'dump-on-identity.json'
    run(['python', str(tools / 'compare_runs.py'), str(work / 'plain'), str(work / 'oracle'),
         '--comparator', profile['comparator'], '--out', str(receipt)],
        work / 'compare.log', environment)
    proof = json.loads(receipt.read_text())
    proof.update(reference_process_count=profile['process_count'],
                 oracle_process_count=profile['process_count'], execution_host=profile['host'],
                 checkout_commit=commit, profile_sha256=digest(profile_path),
                 object_audit_sha256=digest(build_work / 'object-audit.json'),
                 oracle_binary_sha256=digest(build / 'atmosphere_model'),
                 reference_binary_sha256=native['binary_sha256'],
                 dump_member_manifest_sha256=digest(dump_manifest),
                 framework_state_restore_sha256=digest(framework_receipt),
                 library_layout_sha256=digest(build_work / 'library-layout.json'),
                 plain_manifest_sha256=digest(work / 'plain/MANIFEST.json'),
                 oracle_manifest_sha256=digest(work / 'oracle/MANIFEST.json'),
                 dump_records=len(dump_members),
                 scope='Dump-ON versus plain strict forecast self-check; other f1 requirements remain separate')
    text = json.dumps(proof, indent=2) + '\n'
    if len(text.encode()) >= 10_000_000:
        raise ValueError('identity receipt exceeds size limit')
    receipt.write_text(text)
    print(json.dumps({'receipt': str(receipt), 'all_identical': proof['all_identical']}), flush=True)
    if not proof['all_identical']:
        raise SystemExit(1)
    # Each complete case is sealed and retired before the next case starts.
    # This keeps a multi-suite reproduction from accumulating raw forecasts.
    retire_case_outputs(work, dump_members)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'profile', 'work-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    profile = json.loads(args.profile.read_text())
    cases = forecast_cases(profile)
    if socket.gethostname() != profile['host']:
        raise ValueError('self-check must execute on the stated proof host')
    commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    work = args.work_root.resolve() / uuid.uuid4().hex
    work.mkdir(parents=True)
    (work / '.keep').touch()
    tools = checkout / 'tree/tools/native_physics_reference'
    oracle = tools / 'oracle'
    environment = os.environ.copy()
    environment.update(MPAS_UNIT='f1-native-tap', MPICH_FC='gfortran',
                       MPICH_CC='gcc', MPICH_CXX='g++')
    mpi = Path(profile['mpi_root'])
    environment['PATH'] = str(mpi / 'bin') + ':' + environment['PATH']
    environment['LD_LIBRARY_PATH'] = str(mpi / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
    for name in ('MPAS_ORACLE_DIR', 'MPAS_ORACLE_STEPS', 'MPAS_ORACLE_KEYS'):
        environment.pop(name, None)
    # Bind metadata inputs before executing either model.
    for name, expected in profile['input_sha256'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'profile input changed: {name}')
    reference = Path(profile['reference_build'])
    native = json.loads((reference / 'MANIFEST.json').read_text())
    if native['recipe'] != 'gnu-strict' or digest(reference / 'atmosphere_model') != native['binary_sha256']:
        raise ValueError('plain model is not the pinned strict reference')
    build = work / 'oracle-build'
    pnetcdf_lib = library_directory(profile['pnetcdf'], 'pnetcdf')
    netcdf_lib = library_directory(profile['netcdf'], 'netcdf')
    if not (Path(profile['pnetcdf']) / 'include/pnetcdf.h').is_file():
        raise ValueError('installed PnetCDF header missing')
    (work / 'library-layout.json').write_text(json.dumps(dict(
        pnetcdf_prefix=profile['pnetcdf'], pnetcdf_library_directory=pnetcdf_lib,
        netcdf_prefix=profile['netcdf'], netcdf_library_directory=netcdf_lib), indent=2) + '\n')
    run(['python', str(oracle / 'prepare_build.py'), '--source', profile['source'],
         '--out', str(build), '--reference', str(reference)], work / 'prepare.log', environment)
    run(site_runner.build_command(['make', '-j4', '-C', str(build), 'gfortran', 'CORE=atmosphere',
         'PRECISION=single', 'PNETCDF=' + profile['pnetcdf'], 'NETCDF=' + profile['netcdf'],
         'PNETCDFLIBLOC=' + pnetcdf_lib, 'NETCDFLIBLOC=' + netcdf_lib]),
        work / 'build.log', environment)
    run(['python', str(oracle / 'audit_build.py'), '--build', str(build),
         '--log', str(work / 'build.log'), '--receipt', str(work / 'object-audit.json')],
        work / 'audit.log', environment)
    rows = []
    for selected in cases:
        case_work = work / 'cases' / selected['name'] if 'cases' in profile else work
        proof = compare_case(selected, case_work, build, work, tools, environment, native, commit, args.profile)
        if 'cases' in profile:
            rows.append(dict(name=selected['name'], deck=selected['deck'], case=selected['case'],
                process_count=profile['process_count'], all_identical=proof['all_identical'],
                outputs=len(proof['files']), words=sum(v['words'] for v in proof['files'].values()),
                different_words=sum(v['differing_words'] for v in proof['files'].values()),
                receipt_sha256=digest(case_work / 'dump-on-identity.json')))
    if subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() != commit:
        raise ValueError('trial checkout changed during self-check')
    if rows:
        summary = dict(schema='f1-native-dump-on-forecast-matrix-v1', cases=rows,
            all_identical=all(r['all_identical'] for r in rows),
            reference_process_count=profile['process_count'], oracle_process_count=profile['process_count'],
            execution_host=profile['host'], checkout_commit=commit,
            profile_sha256=digest(args.profile), object_audit_sha256=digest(work / 'object-audit.json'),
            oracle_binary_sha256=digest(build / 'atmosphere_model'), reference_binary_sha256=native['binary_sha256'],
            outputs=sum(r['outputs'] for r in rows), words=sum(r['words'] for r in rows),
            different_words=sum(r['different_words'] for r in rows),
            scope='Dump-ON versus plain strict forecast matrix; other f1 requirements remain separate')
        (work / 'dump-on-identity.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
