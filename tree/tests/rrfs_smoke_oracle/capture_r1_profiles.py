"""Regenerate r1's complete native SMOKE fixtures as diagnostic transport inputs.

These source-derived synthetic inputs never qualify as real forecast captures.
Only the existing native executables run; no native reference is rebuilt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(root, out):
    native = root / 'n1out/smoke-native-v169/gnu-strict/standalone'
    site = 'mpas_smoke_driver__driver_smoke__L1159'
    source_manifest = native / site / 'synthetic-manifest.json'
    fixtures = json.loads(source_manifest.read_text())
    original = root / 'receipts/smoke-native-v173/gnu-strict/synthetic-manifest.json'
    if sha(source_manifest) != sha(original):
        raise ValueError('r1 compiled fixture metadata differs from its recorded receipt')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['LD_LIBRARY_PATH'] = str(root / 'runner-inputs/native-dependencies-v32/mpich/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    inventory = root / 'n1out/full-build-v108-gnu-strict/builds/gnu-strict/oracle-inventory.json'
    if sha(inventory) != fixtures['inventory_sha256']:
        raise ValueError('r1 native inventory changed')
    native_site = next(s for s in json.loads(inventory.read_text())['sites'] if s['site'] == site)
    result = dict(schema='rrfs-smoke-synthetic-transport-v1', real_inputs=False,
                  forecast_proven=False, process_count=1, recipe='gnu-strict',
                  source_manifest_sha256=sha(original), inventory=str(inventory),
                  inventory_sha256=sha(inventory), native_site=native_site, cases=[])
    for name, definition in sorted(fixtures['profiles'].items()):
        executable = native / site / 'gnu-strict' / ('capture_' + name)
        source = executable.with_suffix('.F90')
        if sha(source) != definition['generated_source_sha256']:
            raise ValueError('r1 native profile source changed: ' + name)
        dump = out / name
        dump.mkdir()
        env.update(MPAS_ORACLE_DIR=str(dump.resolve()), MPAS_ORACLE_KEYS=site)
        run = subprocess.run([str(executable.resolve())], env=env, cwd=dump,
                             capture_output=True, text=True)
        (out / (name + '.log')).write_text(run.stdout + run.stderr)
        result['cases'].append(dict(name=name, dump=str(dump.resolve()),
                                   executable_sha256=sha(executable), native_exit=run.returncode,
                                   files={p.name: sha(p) for p in sorted(dump.glob('*.bin'))}))
        (out / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    return 1 if any(c['native_exit'] or not c['files'] for c in result['cases']) else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--r1-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(capture(args.r1_root, args.out))
