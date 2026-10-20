"""Receipt-only observation of a forecast, including native dlopen rejection."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

p=argparse.ArgumentParser()
p.add_argument('--receipt',type=Path,required=True)
p.add_argument('arguments',nargs=argparse.REMAINDER)
a=p.parse_args()
started=time.monotonic(); samples=[]; loads=[]; done=threading.Event()
def audit(event,args):
    if event=='ctypes.dlopen':
        name=str(args[0]);loads.append(name)
        if 'mpasphys_native' in name or 'libgfortran' in name:
            raise RuntimeError('native forecast library load forbidden: '+name)
sys.addaudithook(audit)
def observe():
    previous={};last=time.monotonic();hz=os.sysconf('SC_CLK_TCK')
    while not done.wait(1):
        now=time.monotonic();current={}
        for path in Path('/proc/self/task').glob('*/stat'):
            try:
                words=path.read_text().rsplit(')',1)[1].split()
                current[path.parent.name]=(int(words[11])+int(words[12]))/hz
            except (FileNotFoundError,ProcessLookupError):continue
        usage={key:(value-previous[key])/(now-last) for key,value in current.items() if key in previous}
        samples.append(dict(seconds=now-started,cpu_cores=sum(usage.values()),
                            busy_threads=sum(x>=.3 for x in usage.values())))
        previous=current;last=now
thread=threading.Thread(target=observe,daemon=True);thread.start()
code=0
try:
    sys.argv=['gpuwm-hex',*a.arguments]
    runpy.run_module('hexcore.cli',run_name='__main__')
except SystemExit as error:
    code=error.code or 0
    raise
except BaseException:
    code=1
    raise
finally:
    done.set();thread.join()
    maps=Path('/proc/self/maps').read_text()
    row=dict(command=sys.argv,exit_code=code,wall_seconds=time.monotonic()-started,
             ctypes_dlopen=loads,fortran_runtime_mapped='libgfortran' in maps,
             fortran_runtime_mappings=sorted({line.split()[-1] for line in maps.splitlines()
                                              if 'libgfortran' in line}),samples=samples)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(row,indent=2)+'\n')
