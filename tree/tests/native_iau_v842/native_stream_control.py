"""Run a complete strict MPAS control with an actual selected input stream.

This is metadata orchestration around the published immutable executable.
Controlled increment inputs never qualify as analysis-minus-background proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
for name in ('native', 'tables', 'case', 'request', 'increment', 'mpi', 'out', 'receipt'):
    parser.add_argument('--' + name, type=Path, required=True)
args = parser.parse_args()
work = Path(__file__).resolve().parents[4]
run = args.out.resolve()
if not run.is_relative_to(work.resolve()):
    parser.error('native controls must remain inside this lane work directory')
if args.receipt.exists() or not args.receipt.resolve().is_relative_to(work.resolve()):
    parser.error('receipt must be a new file inside this lane work directory')
run.mkdir(parents=True, exist_ok=False)
shutil.copy2(args.request / 'namelist.atmosphere', run / 'namelist.atmosphere')
streams = ET.parse(args.request / 'streams.atmosphere')
iau = [element for element in streams.getroot() if element.get('name') == 'iau']
if len(iau) != 1 or 'input' not in iau[0].get('type', '').split(';'):
    parser.error('native selected input requires the actual iau input stream')
iau[0].set('filename_template', str(args.increment.resolve()))
streams.write(run / 'streams.atmosphere', encoding='unicode')
for path in (*args.case.glob('*.nc'), *args.case.glob('*.graph.info*')):
    (run / path.name).symlink_to(path.resolve())
for path in args.native.iterdir():
    if path.is_file() and (path.suffix in ('.TBL', '.DBL') or
            path.name.startswith(('stream_list.', 'OZONE_')) or
            path.name in ('CCN_ACTIVATE_DATA', 'RRTMG_LW_DATA', 'RRTMG_SW_DATA')):
        (run / path.name).symlink_to(path.resolve())
for path in args.tables.glob('*.DBL'):
    if not (run / path.name).exists():
        (run / path.name).symlink_to(path.resolve())
argv = ['taskset', '-c', '8-23', str((args.mpi / 'bin/mpirun').resolve()), '-n', '1',
        str((args.native / 'atmosphere_model').resolve())]
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
       'LD_LIBRARY_PATH': str((args.mpi / 'lib').resolve()) + ':' + os.environ.get('LD_LIBRARY_PATH', '')}
with (run / 'stdout.log').open('w') as log:
    result = subprocess.run(argv, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT)
receipt = dict(schema='mpas.iau-native-selected-stream-control/v1', acceptance=False,
               scope='actual whole native controlled input selection, not analysis proof',
               host=os.uname().nodename, process_count=1, openmp_threads=1,
               argv=argv, exit=result.returncode, input_stream=str(args.increment.resolve()),
               inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (args.increment, run / 'namelist.atmosphere',
                             run / 'streams.atmosphere', args.native / 'atmosphere_model')},
               output_files=[dict(file=p.name, bytes=p.stat().st_size)
                   for p in sorted(run.glob('*.nc')) if not p.is_symlink()],
               logs=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(set(run.glob('*.log')) | set(run.glob('log.atmosphere*')))])
args.receipt.parent.mkdir(parents=True, exist_ok=True)
args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
