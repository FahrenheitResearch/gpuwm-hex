"""Inventory the union of witnessed native branches in identical science files."""
from pathlib import Path
import argparse,hashlib,json,re

FILES=['bl_mynn.F90','bl_mynn_subroutines.F90','mynn_shared.F90','module_bl_mynn.F','bl_mynn_pre.F','bl_mynn_post.F']

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def parse(p):
 line=0;routine='module';lines={};branches={}
 for record in p.read_text().splitlines():
  m=re.match(r'\s*([^:]+):\s*(\d+):(.*)',record)
  if m:
   line=int(m[2]);source=m[3].strip();r=re.match(r'(?:subroutine|function)\s+(\w+)',source,re.I)
   if r:routine=r[1].lower()
   count=m[1].strip().rstrip('*')
   if line and count!='-':lines[line]=dict(source=source,executed=count.isdigit() and int(count)>0)
  m=re.match(r'branch\s+(\d+)\s+(.*)',record)
  if m:
   text=m[2];taken=re.match(r'taken (\d+)',text)
   branches[(line,int(m[1]))]=dict(line=line,branch=int(m[1]),routine=routine,source=source,taken=bool(taken and int(taken[1])>0),executed='never executed' not in text)
 return lines,branches

def main(a):
 a.output.mkdir(exist_ok=False);files={}
 for name in FILES:
  # Only identical unchanged native scientific files may share edge IDs.
  source_hashes={sha(root/name) for root in a.sources}
  if len(source_hashes)!=1:raise ValueError('Native scientific sources differ: '+name)
  merged_lines={};merged_branches={};receipts=[]
  for root in a.coverage:
   path=root/(name+'.gcov');lines,branches=parse(path);receipts.append(dict(path=str(path),sha256=sha(path)))
   for line,row in lines.items():
    previous=merged_lines.setdefault(line,dict(source=row['source'],executed=False))
    if previous['source']!=row['source']:raise ValueError('Native line identity differs')
    previous['executed']|=row['executed']
   if merged_branches and set(merged_branches)!=set(branches):raise ValueError('Native branch inventory differs')
   for key,row in branches.items():
    previous=merged_branches.setdefault(key,dict(row,taken=False,executed=False))
    if previous['source']!=row['source']:raise ValueError('Native edge source differs')
    previous['taken']|=row['taken'];previous['executed']|=row['executed']
  files[name]=dict(native_source_sha256=next(iter(source_hashes)),receipts=receipts,
   executable_lines=len(merged_lines),executed_lines=sum(r['executed'] for r in merged_lines.values()),
   branches=len(merged_branches),executed_branches=sum(r['executed'] for r in merged_branches.values()),taken_branches=sum(r['taken'] for r in merged_branches.values()),
   untaken_branches=[r for r in merged_branches.values() if not r['taken']])
 (a.output/'science-coverage-union.json').write_text(json.dumps(dict(scope=__doc__,process_count=1,diagnostic_only=True,files=files),indent=2)+'\n')
 print(json.dumps({k:{n:v[n] for n in ['executable_lines','executed_lines','branches','executed_branches','taken_branches']} for k,v in files.items()}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--coverage',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
 main(p.parse_args())
