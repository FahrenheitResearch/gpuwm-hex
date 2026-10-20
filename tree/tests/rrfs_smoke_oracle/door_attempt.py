"""Run an unchanged original deck through the committed forecast door.

This records a routing attempt, never an identity result. Use forecast_proof.py
and the certified comparator when matching native chemistry outputs exist.
The request is a JSON argument vector with authenticated external inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from proof_mode import exact_forecast_arguments
from hexcore.mpasphys.forecast_v842 import split_physics_mode

p = argparse.ArgumentParser()
p.add_argument('--checkout', type=Path, required=True)
p.add_argument('--request', type=Path, required=True)
p.add_argument('--receipt', type=Path, required=True)
p.add_argument('--log', type=Path, required=True)
a = p.parse_args()
checkout = a.checkout.resolve()
assert not subprocess.check_output(['git', '-C', str(checkout), 'status',
                                    '--porcelain', '--untracked-files=no']).strip()
request = json.loads(a.request.read_text())
assert request['schema'] == 'r14-smoke-door-attempt-v1'
for row in request['inputs']:
    assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
namelist_mode = None
if '--physics-namelist' in request['arguments']:
    path = Path(request['arguments'][request['arguments'].index('--physics-namelist') + 1])
    _, namelist_mode, _ = split_physics_mode(path.read_text())
command = [sys.executable, '-m', 'hexcore.cli', 'forecast',
           *exact_forecast_arguments(request['arguments'], namelist_mode=namelist_mode)]
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([str(checkout / 'tree/src'),
                                    str(checkout / 'tree/tools'),
                                    env.get('PYTHONPATH', '')])
a.log.parent.mkdir(parents=True, exist_ok=True)
with a.log.open('w') as log:
    completed = subprocess.run(command, cwd=checkout / 'tree', env=env,
                               stdout=log, stderr=subprocess.STDOUT)
streams = Path(command[command.index('--streams') + 1])
out = Path(command[command.index('--out') + 1])
result = dict(scope='real forecast routing attempt; no identity claim',
              hex_commit=subprocess.check_output(['git', '-C', str(checkout),
                                                  'rev-parse', 'HEAD'], text=True).strip(),
              request_sha256=hashlib.sha256(a.request.read_bytes()).hexdigest(),
              host=os.uname().nodename, process_count=request['process_count'],
              inputs=request['inputs'], limitations=request['limitations'],
              command=command, exit_code=completed.returncode,
              physics_mode=dict(mode='exact', source='explicit forecast CLI'),
              output_created=out.exists(), output=str(out), log=str(a.log),
              differing_words=None, compared_fields=[], whole_model_proven=False,
              stream_comparisons=[dict(stream=v.get('name'), schedule=dict(v.attrib),
                                       differing_words=None, compared_fields=[],
                                       status='uncompared: matching chemistry reference unavailable')
                                  for v in ET.parse(streams).getroot()
                                  if 'output' in v.get('type', '').split(';')])
a.receipt.write_text(json.dumps(result, indent=2) + '\n')
raise SystemExit(completed.returncode)
