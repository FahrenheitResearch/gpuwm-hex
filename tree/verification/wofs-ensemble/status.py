"""One-shot observation of this lane's processes and receipts."""
import os
from pathlib import Path
import json, subprocess
W = Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
for name in ('cpu-fetch-mesh-ready', 'mesh', 'init-progress', 'inputs-ready', 'cpu-admission'):
    p = W/'evidence'/f'{name}.json'
    if p.is_file():
        value=json.loads(p.read_text())
        print(name, json.dumps(value['emitted'] if name=='mesh' else value)[:4000])
print('own processes:')
for line in subprocess.check_output(['ps', '-eo', 'pid,etime,pcpu,args'], text=True).splitlines():
    if str(W) in line and 'ps -eo' not in line: print(line)
for p in sorted((W/'logs').glob('*.log')):
    if p.name in ('mesh.log', 'static.log', 'initialize-controller.log', 'prepare-controller.log') or '-init' in p.name:
        with p.open(errors='replace') as f: lines = f.readlines()
        print(p.name, ''.join(lines[-4:])[-1500:])
