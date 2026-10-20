"""Export every unhit gcov compiler branch with its native source context."""
import argparse
import json
from pathlib import Path
import re


def export(root: Path, receipt: Path):
    rows=[]
    for name in ('mp_wsm6.F90','mp_wsm6_effectRad.F90','mp_radar.F90','module_mp_wsm6.F'):
        number=0;source='';routine=''
        for line in (root/(name+'.gcov')).read_text().splitlines():
            if line.startswith('function '):routine=line.split()[1]
            m=re.match(r'\s*[^:]+:\s*(\d+):(.*)',line)
            if m:number=int(m[1]);source=m[2].strip()
            if line.startswith('branch ') and ('never executed' in line or re.search(r'taken 0(?:\D|$)',line)):
                rows.append(dict(file=name,line=number,routine=routine,source=source,branch=line))
    receipt.write_text(json.dumps(dict(scope='strict synthetic native oracle; includes compiler-generated diagnostic/optional branches',branches=rows),indent=2)+'\n')
    print(len(rows),'unreached compiler branches')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();export(a.root,a.receipt)
