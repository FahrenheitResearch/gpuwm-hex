"""Record native branch edges, including every edge not reached by replay."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',default='synthetic standalone strict Fortran; branch counts include compiled native table reader');a=p.parse_args()
rows=[]
for file in sorted(a.input.glob('*.gcov')):
 if file.name in ('wrappers.f90.gcov','stubs.f90.gcov','drivers.F90.gcov','rucinit_slice.F90.gcov'):continue
 line=0;text='';branches=[];lines=[]
 for s in file.read_text().splitlines():
  m=re.match(r'\s*([^:]+):\s*(\d+):(.*)',s)
  if m:
   line=int(m[2]);text=m[3].strip()
   if m[1].strip()!='-':lines.append(dict(line=line,reached=m[1].strip() not in ('#####','=====')))
  m=re.match(r'branch\s+(\d+)\s+(.*)',s)
  if m:
   count=re.search(r'taken (\d+)',m[2]);taken=int(count[1]) if count else 0
   branches.append(dict(line=line,edge=int(m[1]),taken=taken,statement=text))
 if branches:
  rows.append(dict(source=file.name.removesuffix('.gcov'),branches=len(branches),reached=sum(b['taken']>0 for b in branches),lines=len(lines),lines_reached=sum(x['reached'] for x in lines),unreached=[b for b in branches if not b['taken']]))
a.output.write_text(json.dumps(dict(scope=a.scope,units=rows),indent=2)+'\n')
print([(r['source'],r['reached'],r['branches']) for r in rows])
