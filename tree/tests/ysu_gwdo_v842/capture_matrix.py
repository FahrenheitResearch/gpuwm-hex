#!/usr/bin/env python3
"""Capture first native YSU call on published f0 cases using f1's frozen tap.

No native build or numerical preprocessing. Case names are explicit test data.
Run under the lane cpuset; CPU capture is independent of the CUDA queue.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f1', type=Path, required=True)
    parser.add_argument('--ref', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--suite', choices=['none', 'mesoscale_reference', 'convection_permitting'],
                        default='mesoscale_reference')
    args = parser.parse_args()
    args.scratch = args.scratch.resolve()
    args.receipts = args.receipts.resolve()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.receipts.mkdir(parents=True, exist_ok=True)
    frozen = args.f1/'scratch/oracle-v10-frozen-run-build'
    binary = args.scratch/'atmosphere_model'
    if not binary.exists():
        shutil.copyfile(frozen/'atmosphere_model', binary)
    elif digest(binary) != digest(frozen/'atmosphere_model'):
        raise ValueError('resume binary differs from frozen tap')
    binary.chmod(0o755)
    template = json.loads((args.f1/'scratch/mesoscale-real-v10-dumps/manifest.json').read_text())
    site = next(s for s in template['sites'] if s['site'] == 'gwdo__driver_gwdo__L759')
    site['scheme'] = 'ysu_gwdo'
    for mesh in ('x1.2562', 'x1.10242', 'x1.40962'):
        for date in ('2023-03-31_12.00.00', '2022-12-22_12.00.00'):
            key = mesh+'-'+date
            run = args.scratch/key
            if (run/'dump/manifest.json').exists():
                print(json.dumps(dict(case=key, resumed='completed capture retained')), flush=True)
                continue
            run.mkdir(exist_ok=True)
            dump = run/'dump'
            dump.mkdir(exist_ok=True)
            if any(dump.iterdir()):
                raise ValueError('incomplete capture needs a fresh scratch directory: '+key)
            source = args.ref/'cases'/mesh/date/'init-modis20-v2'
            selector = 'gwdo_scheme-bl_ysu_gwdo' if args.suite == 'none' else 'suite-'+args.suite
            deck = args.ref/'decks/native-full-physics-v2/G2'/mesh/date/selector
            if not deck.exists():
                deck = args.ref/'decks/native-full-physics-v2/G3'/mesh/date/selector
            namelist = (deck/'namelist.atmosphere').read_text()
            timestep = int(re.search(r'config_dt\s*=\s*(\d+)', namelist)[1])
            duration = f'0_00:{timestep//60:02}:{timestep%60:02}'
            namelist = re.sub(r"config_run_duration\s*=\s*'[^']+'",
                              f"config_run_duration = '{duration}'", namelist)
            (run/'namelist.atmosphere').write_text(namelist)
            shutil.copyfile(deck/'streams.atmosphere', run/'streams.atmosphere')
            for path in frozen.iterdir():
                if path.name != 'atmosphere_model':
                    (run/path.name).symlink_to(path.resolve())
            table_hashes = {}
            if args.suite == 'convection_permitting':
                tables = next(p for p in (args.ref/'native').iterdir()
                              if (p/'MP_THOMPSON_QRacrQG_DATA.DBL').is_file())
                table_manifest = json.loads((tables/'MANIFEST.json').read_text())
                for name, expected in table_manifest['files'].items():
                    path = tables/name
                    if digest(path) != expected:
                        raise ValueError('published Thompson table changed: '+str(path))
                    (run/name).symlink_to(path.resolve())
                    table_hashes[name] = expected
            (run/'atmosphere_model').symlink_to(binary)
            publication = json.loads((source/'PUBLICATION.json').read_text())
            case_hashes = {}
            for path in source.iterdir():
                if path.suffix == '.nc' or '.graph.info' in path.name:
                    expected = publication['files'][path.name]
                    if digest(path) != expected:
                        raise ValueError('published case hash changed: '+str(path))
                    case_hashes[path.name] = expected
                    (run/path.name).symlink_to(path.resolve())
            provenance = dict(schema='ysu-native-case-capture-v1', mesh=mesh, date=date,
                              suite=args.suite, option='bl_ysu_gwdo', timestep=timestep,
                              process_count=1, launch_command=[str(binary)],
                              binary_sha256=digest(binary), case_publication_sha256=digest(source/'PUBLICATION.json'),
                              case_inputs_sha256=case_hashes, thompson_tables_sha256=table_hashes, namelist_sha256=digest(run/'namelist.atmosphere'),
                              streams_sha256=digest(run/'streams.atmosphere'),
                              native_build_log_sha256=digest(args.f1/'receipts/oracle-state-v10-build.log'),
                              native_tap_on_identity_receipt_sha256=digest(args.f1/'receipts/mesoscale-real-v10-dump-on-identity.json'))
            environment = dict(os.environ, MPAS_ORACLE_DIR=str(dump), MPAS_ORACLE_STEPS='1',
                               MPAS_ORACLE_KEYS=site['site'])
            with (args.receipts/(key+'-native.log')).open('w') as log:
                result = subprocess.run([str(binary)], cwd=run, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            provenance['returncode'] = result.returncode
            # Preserve short model log; forecast output is disposable.
            if (run/'log.atmosphere.0000.out').exists():
                shutil.copyfile(run/'log.atmosphere.0000.out', args.receipts/(key+'-model.log'))
            bins = sorted(dump.glob('*.bin'))
            if result.returncode or not bins:
                (args.receipts/(key+'-capture.json')).write_text(json.dumps(provenance, indent=2)+'\n')
                raise RuntimeError('native first-call capture failed: '+key)
            files = {p.name: dict(sha256=digest(p), bytes=p.stat().st_size) for p in bins}
            captures = {p.name.split('.pre.')[0]: dict(site=site['site']) for p in bins if '.pre.' in p.name}
            manifest = {**template, 'files':files, 'sites':[site], 'captures':captures,
                        'capture_provenance':provenance,
                        'alias_note':'YSU native gwdo call; only the scheme inventory alias changed'}
            del manifest['dump_root_sha256']
            manifest['dump_root_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (dump/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            provenance['dump_root_sha256'] = manifest['dump_root_sha256']
            provenance['files'] = files
            (args.receipts/(key+'-capture.json')).write_text(json.dumps(provenance, indent=2)+'\n')
            for path in run.iterdir():
                if path.is_file() and not path.is_symlink() and path.suffix == '.nc':
                    path.unlink()
            print(json.dumps(dict(case=key, captures=len(captures), returncode=result.returncode)), flush=True)


if __name__ == '__main__':
    main()
