"""Authenticate MMM/driver citations, including names omitted by older checkers."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(reference: Path, receipt: Path):
    here=Path(__file__).resolve().parent
    authority=reference/'src/core_atmosphere/physics'
    pinned=json.loads((here/'source_pins.json').read_text())
    files={name:next(authority.rglob(name)) for name in pinned}
    for name,path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=pinned[name]:
            raise RuntimeError(f'pinned native source changed: {name}')
    source=(here.parents[1]/'src/hexcore/mpasphys/wsm6_v842.py').read_text()
    rows=[]
    pattern=r'([A-Za-z0-9_]+\.(?:F90|F)):((?:\d+(?:-\d+)?)(?:,\s*\d+(?:-\d+)?)*)'
    for m in re.finditer(pattern,source):
        name,ranges=m.groups()
        path=files[name];lines=path.read_bytes().splitlines(keepends=True)
        for part in ranges.split(','):
            endpoints=part.strip().split('-');lo=int(endpoints[0]);hi=int(endpoints[-1])
            if not 1<=lo<=hi<=len(lines):raise RuntimeError(f'invalid citation {name}:{part}')
            rows.append(dict(file=name,start=lo,end=hi,sha256=hashlib.sha256(b''.join(lines[lo-1:hi])).hexdigest()))
    routines=[]
    for name in ('mp_wsm6.F90','mp_wsm6_effectRad.F90','mp_radar.F90'):
        for m in re.finditer(r'^\s*(?:real\([^\n]+?\)|complex\([^\n]+?\))?\s*(?:subroutine|function)\s+(\w+)\s*(?:\(|$)',files[name].read_text(),re.M|re.I):
            routine=m[1].lower()
            if not re.search(r'// MPAS-A v8\.4\.2 '+re.escape(name)+r':\d+-\d+\n__device__ \w+ '+routine+r'\(',source):
                raise RuntimeError(f'missing cited full routine: {name}:{routine}')
            routines.append(dict(file=name,routine=routine))
    if len(routines)!=24:raise RuntimeError(f'expected 24 complete routines, got {len(routines)}')
    receipt.write_text(json.dumps(dict(scope='authenticated eight pinned v8.4.2 files; all cited windows and all 24 complete MMM routines',routines=routines,windows=rows),indent=2)+'\n')
    print(len(routines),'routines;',len(rows),'source windows; 0 failures')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();audit(a.reference,a.receipt)
