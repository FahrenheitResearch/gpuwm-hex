"""Detach an owned preparation or queue controller; record its explicit PID."""
import json, os, subprocess, sys
from pathlib import Path
W = Path(os.environ.get('HEX_WOFS_ROOT', 'hex-wofs-ensemble'))
phase = sys.argv[1]
if phase not in ('initialize', 'pilot', 'showcase', 'products'):
    raise SystemExit('unknown phase')
command = sys.argv[2:]
if not command: raise SystemExit('command required')
with (W/'logs'/f'{phase}-controller.log').open('a') as log:
    child = subprocess.Popen(command, cwd=W, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
(W/f'{phase}.pid').write_text(str(child.pid)+'\n')
print(json.dumps(dict(pid=child.pid, command=command)))
