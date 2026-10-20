"""Record every unobserved gcov outcome without excluding it from the denominator."""
from pathlib import Path
import re
import json
import argparse
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('n1out'));p.add_argument('--output',type=Path,default=Path('receipts/uncovered-branches.json'));a=p.parse_args()
rows=[]
for filename in ('module_sf_noahlsm.F','module_sf_noahlsm_glacial_only.F','module_sf_noahdrv.F','private_helpers.F90'):
    path=a.directory/(filename+'.gcov')
    if not path.exists():continue
    routine='';line=0;statement=''
    for text in path.read_text().splitlines():
        match=re.match(r'\s*[^:]+:\s*(\d+):(.*)',text)
        if match:
            line=int(match[1]);statement=match[2].strip()
            if not statement.startswith('!'):
                sub=re.match(r'(?:REAL\s+)?(?:SUBROUTINE|FUNCTION)\s+(\w+)',statement,re.I)
                if sub:routine=sub[1]
        elif re.match(r'\s*branch\s+\d+ (?:never executed|taken 0(?:%|\s|$))',text):
            rows.append(dict(file=filename,routine=routine,line=line,statement=statement,branch=text.strip()))
a.output.write_text(json.dumps(rows,indent=2)+'\n')
for filename in sorted({r['file'] for r in rows}):
    print(filename, sorted({r['line'] for r in rows if r['file']==filename}))
