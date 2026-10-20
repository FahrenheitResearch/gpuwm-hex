"""Capture the first IAU boundary using the current whole-deck inputs.

The pristine f1 instrumented executable owns all data and calculations. This
one-step capture supports debugging a complete forecast; it is not acceptance.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--base-run', type=Path, required=True)
p.add_argument('--oracle', type=Path, required=True)
p.add_argument('--reader', type=Path, required=True)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
run = a.run.resolve()
run.mkdir(parents=True, exist_ok=False)
for source in a.base_run.resolve().iterdir():
    if source.is_symlink() and source.exists():
        (run/source.name).symlink_to(source.resolve())
for name in ('namelist.atmosphere', 'streams.atmosphere'):
    text = (a.base_run/name).read_text()
    if name == 'namelist.atmosphere':
        text = text.replace("config_run_duration = '1_00:00:00'",
                            "config_run_duration = '00:20:00'")
    (run/name).write_text(text)
(run/'dumps').mkdir()
env = dict(os.environ, OMP_NUM_THREADS='1', MPAS_ORACLE_KEYS='boundary__atm_add_tend_anal_incr',
           MPAS_ORACLE_STEPS='1', MPAS_ORACLE_DIR=str(run/'dumps'))
with (run/'native.stdout.log').open('w') as log:
    subprocess.run(['taskset', '-c', '8-23', str(a.oracle.resolve())],
                   cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
(run/'capture-provenance.json').write_text(json.dumps(dict(
    scope='first boundary of controlled whole-deck input; diagnostic only',
    process_count=1, oracle_sha256=hashlib.sha256(a.oracle.read_bytes()).hexdigest(),
    namelist_sha256=hashlib.sha256((run/'namelist.atmosphere').read_bytes()).hexdigest(),
    streams_sha256=hashlib.sha256((run/'streams.atmosphere').read_bytes()).hexdigest(),
    base=str(a.base_run.resolve()), host=os.uname().nodename), indent=2)+'\n')
subprocess.run([os.sys.executable, str(Path(__file__).with_name('collect_capture.py')),
                '--run', str(run), '--reader', str(a.reader.resolve()),
                '--out', str(a.out.resolve())], check=True)
