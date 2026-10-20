#!/usr/bin/env python3
"""Rerun the native/hex forecast proof against an explicit trial checkout.

All case choices live in the sealed receipt configuration. This orchestrates the
pristine native executable and the checkout's real dycore comparator, with no
weather-field decoding or approximate comparison in this runner.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from hexcore import site_runner
from publish_cache import sha
from native_completion import completed
from verify_acceptance import validate


def validate_reuse_contract(current, executed):
    """A cancelled GPU queue may reuse completed native science, never a changed deck."""
    for key in ('binary_sha256', 'initial_sha256', 'ranks', 'cells', 'levels', 'cores'):
        if current[key] != executed[key]:
            raise ValueError('native reuse contract differs: '+key)
    if sha(Path(current['mpi'])) != sha(Path(executed['mpi'])):
        raise ValueError('native reuse MPI executable differs')
    def science(config):
        return [(r['name'], r['hours'], {n:v['sha256'] for n,v in r['files'].items()})
                for r in config['runs']]
    if science(current) != science(executed):
        raise ValueError('native reuse deck/input/sample contract differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkout', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--reuse-native', type=Path)
    p.add_argument('--executed-config', type=Path)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    if bool(a.reuse_native) != bool(a.executed_config):
        raise ValueError('native reuse requires its original executed configuration')
    if a.reuse_native:
        validate_reuse_contract(cfg, json.loads(a.executed_config.read_text()))
    checkout = a.checkout.resolve(strict=True)
    binary = Path(cfg['binary'])
    initial = Path(cfg['initial'])
    for file, digest in [(binary, cfg['binary_sha256']), (initial, cfg['initial_sha256'])]:
        if sha(file) != digest:
            raise ValueError(f'proof input differs from its pin: {file}')
    tool = checkout/'tree/tools/native_dry_reference/identity_compare.py'
    if not tool.is_file():
        raise ValueError('trial checkout has no real native dycore comparator')
    a.work.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='proof-', dir=a.work))
    result = {'schema': 'mpas.native-acceptance-rerun/v1',
              'checkout': str(checkout), 'checkout_commit': subprocess.check_output(
                  ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip(),
              'native_ranks': cfg['ranks'], 'hex_devices': 1,
              'config_sha256': sha(a.config), 'binary_sha256': sha(binary),
              'initial_sha256': sha(initial), 'comparator_sha256': sha(tool), 'runs': []}
    for row in cfg['runs']:
        previous = a.reuse_native/row['name'] if a.reuse_native else None
        reused = bool(previous and list(previous.glob('tracer.*.nc')))
        native = previous if reused else root/row['name']
        if not reused:
            native.mkdir()
        for name, item in row['files'].items():
            source = Path(item['path']).resolve(strict=True)
            if sha(source) != item['sha256']:
                raise ValueError(f'proof deck/input differs: {source}')
            if reused:
                if sha(native/name) != item['sha256']:
                    raise ValueError('completed native input changed: '+name)
            elif name in ('namelist.atmosphere', 'streams.atmosphere'):
                shutil.copy2(source, native/name)
            else:
                (native/name).symlink_to(source)
        if reused:
            if sha(native/'atmosphere_model') != cfg['binary_sha256']:
                raise ValueError('completed native executable changed')
        else:
            (native/'atmosphere_model').symlink_to(binary.resolve())
            with (native/'native.stdout.log').open('w') as log:
                subprocess.run(['taskset', '-c', cfg['cores'], cfg['mpi'], '-np', str(cfg['ranks']),
                                str(binary)], cwd=native, stdout=log, stderr=subprocess.STDOUT, check=True)
        model = native/'log.atmosphere.0000.out'
        if not completed(model.read_text(), 'atmosphere'):
            raise ValueError('fresh native forecast failed to complete')
        output_hashes = {file.name: sha(file) for file in native.glob('tracer.*.nc')}
        if not output_hashes:
            raise ValueError('completed native forecast has no original outputs')
        checkpoint = {'native': str(native), 'native_execution_reused': reused,
                      'native_output_sha256': output_hashes, 'native_log_sha256': sha(model),
                      'native_ranks': cfg['ranks'], 'binary_sha256': cfg['binary_sha256'],
                      'executed_config_sha256': sha(a.executed_config) if reused else sha(a.config)}
        (root/(row['name']+'-native-checkpoint.json')).write_text(json.dumps(checkpoint, indent=2)+'\n')
        receipt = root/(row['name']+'-identity.json')
        with (root/(row['name']+'-hex.log')).open('w') as log:
            subprocess.run(site_runner.gpu_command(['taskset', '-c', cfg['cores'], 'python', str(tool),
                            '--hex-tree', str(checkout/'tree'), '--init', str(initial),
                            '--native', str(native), '--out', str(receipt),
                            '--cache-root', str(root/'cache')], 10, tag='f0-native-ref-reproduce'), stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        proof = json.loads(receipt.read_text())
        errors = validate(proof, row['hours'], cfg['cells'], cfg['levels'], cfg['initial_sha256'])
        if errors:
            raise ValueError('; '.join(errors))
        samples = proof['rows']
        if (not proof.get('all_identical') or not proof.get('native_finished')
                or proof.get('partial') or [r['hours'] for r in samples] != row['hours']):
            raise ValueError('fresh whole-model identity proof is incomplete or differs')
        for sample in samples:
            for field in list(sample['fields'].values()) + list(sample['species'].values()):
                if not field.get('identical') or field.get('different_words') != 0:
                    raise ValueError('fresh forecast contains differing words')
        if output_hashes != {file.name: sha(file) for file in native.glob('tracer.*.nc')}:
            raise ValueError('native output changed during actual GPU comparison')
        result['runs'].append({'name': row['name'], 'hours': row['hours'],
                               'identity_receipt_sha256': sha(receipt),
                               'native_output_sha256': output_hashes, 'all_identical': True,
                               'native_execution_reused': reused, 'native_path': str(native)})
        # Original bytes are disposable after the freshly executed comparison.
        for name in output_hashes:
            (native/name).unlink()
        if (root/'cache').exists():
            shutil.rmtree(root/'cache')
    result['all_identical'] = True
    (root/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'all_identical': True, 'receipt': str(root/'RESULT.json'),
                      'native_ranks': cfg['ranks'], 'hex_devices': 1}), flush=True)


if __name__ == '__main__':
    main()
