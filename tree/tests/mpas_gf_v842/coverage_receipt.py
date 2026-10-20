"""Retain every uncovered native branch with its concrete source condition."""
from pathlib import Path
import re,json,argparse,hashlib

def receipt(gcov,source,target):
    src=source.read_text();routines=[]
    for m in re.finditer(r'^[ \t]*(?:subroutine|real[ \t]+function)[ \t]+(\w+)[ \t]*\(',src,re.I|re.M):
        routines.append((src[:m.start()].count('\n')+1,m[1].lower()))
    current=0;condition='';routine='';unhit=[];branches=[];exec_lines=0;hit_lines=0;calls={}
    for line in gcov.read_text().splitlines():
      fn=re.match(r'function __module_cu_gf_MOD_(\w+) called (\d+) ',line)
      if fn:
        calls[fn[1]]=int(fn[2])
      m=re.match(r'\s*([^:]+):\s*(\d+):(.*)',line)
      if m:
        count,current,condition=m[1].strip(),int(m[2]),m[3].strip()
        before=[name for start,name in routines if start<=current]
        routine=before[-1] if before else ''
        if count!='-' and current:
          exec_lines+=1;hit_lines+=count.strip('*').isdigit() and int(count.strip('*'))>0
      b=re.match(r'branch\s+(\d+)\s+(.*)',line)
      if b:
        status=b[2];branches.append(status)
        if 'never executed' in status or re.search(r'\btaken 0\b',status):
          unhit.append({'routine':routine,'line':current,'branch':int(b[1]),'status':status,'source':condition})
    report={'authority_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'executable_lines':exec_lines,'executed_lines':hit_lines,'branches':len(branches),'branches_executed':sum('never executed' not in x for x in branches),'branches_taken':len(branches)-len(unhit),'routine_calls':calls,'uncovered_branches':unhit,'scope':'strict standalone GF driver columns and direct auxiliary tests; not campaign dump coverage'}
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='uncovered_branches'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gcov',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();receipt(a.gcov,a.source,a.receipt)
