"""Capture complete seaice_noah calls using f1's frozen native executable.

Runs MPAS's driver_seaice calls at mpas_atmphys_driver_seaice.F:488-549.
The external native forecast runner and f1 sealing door are used unchanged.
A failed forecast may contain a complete scheme call; --seal-after-native-failure
records that distinction explicitly, without altering the executable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture(a):
    a.out.mkdir(parents=True, exist_ok=False)
    deck = a.out/'deck'
    deck.mkdir()
    text = (a.deck/'namelist.atmosphere').read_text()
    dt = int(re.search(r'config_dt\s*=\s*(\d+)', text)[1])
    duration = f'{dt//86400}_{dt//3600%24:02}:{dt//60%60:02}:{dt%60:02}'
    text = re.sub(r"config_run_duration\s*=\s*'[^']+'",
                  f"config_run_duration = '{duration}'", text)
    (deck/'namelist.atmosphere').write_text(text)
    (deck/'streams.atmosphere').write_bytes((a.deck/'streams.atmosphere').read_bytes())
    dumps = a.out/'dumps'
    dumps.mkdir()
    env = dict(os.environ, MPAS_ORACLE_DIR=str(dumps.resolve()),
               MPAS_ORACLE_STEPS='1', MPAS_ORACLE_KEYS=','.join(a.site))
    command = [sys.executable, str(a.runner), '--build', str(a.build),
               '--deck', str(deck), '--case', str(a.case), '--out', str(a.out/'run'),
               '--tables', str(a.tables), '--mpi', str(a.mpi.resolve()),
               '--ranks', str(a.ranks), '--pin-set-hash', sha(a.pins)]
    result = subprocess.run(command, env=env)
    record = {'kind': 'native one-step scheme capture; not CUDA forecast proof',
              'runner_status': result.returncode, 'forecast_completed': result.returncode == 0,
              'binary_sha256': sha(a.build/'atmosphere_model'),
              'runner_sha256': sha(a.runner), 'seal_door_sha256': sha(a.seal_door),
              'inventory_sha256': sha(a.inventory), 'pin_set_sha256': sha(a.pins),
              'source_deck': str(a.deck), 'source_case': str(a.case),
              'namelist_sha256': sha(deck/'namelist.atmosphere'),
              'streams_sha256': sha(deck/'streams.atmosphere'),
              'ranks': a.ranks, 'sites': a.site, 'forecast_outputs': {}}
    for payload in (a.out/'run').glob('*.nc'):
        if not payload.is_symlink():
            record['forecast_outputs'][payload.name] = {
                'bytes': payload.stat().st_size, 'sha256': sha(payload)}
    log = a.out/'run/native.stdout.log'
    if log.exists():
        record['native_stdout'] = log.read_text()
    (a.out/'capture_run.json').write_text(json.dumps(record, indent=2)+'\n')
    # Only owned output payloads are retired, after their hashes were recorded.
    for name in record['forecast_outputs']:
        (a.out/'run'/name).unlink()
    if result.returncode and not a.seal_after_native_failure:
        return result.returncode
    subprocess.run([sys.executable, str(a.seal_door), '--scheme', 'seaice_noah',
                    '--dump', str(dumps), '--seal', '--inventory', str(a.inventory),
                    '--pins', str(a.pins), '--words', str(a.words),
                    '--recipe', a.recipe], check=True)
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('deck', 'case', 'out', 'runner', 'build', 'tables', 'mpi',
                 'seal-door', 'inventory', 'pins', 'words'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--site', action='append', required=True)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--recipe', default='gnu-strict')
    p.add_argument('--seal-after-native-failure', action='store_true')
    raise SystemExit(capture(p.parse_args()))
