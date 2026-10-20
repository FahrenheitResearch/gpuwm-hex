"""Summarize gcov's instrumented strict standalone replay, without hiding gaps."""
import argparse,gzip,json,re,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('gcov_json',type=Path);p.add_argument('source',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
r=json.loads(gzip.decompress(a.gcov_json.read_bytes()));source=a.source.read_text().splitlines();rows=[]
for f in r['files']:
 lines=f['lines'];branches=[b for line in lines for b in line.get('branches',[])]
 missing=[]
 for line in lines:
  bs=line.get('branches',[])
  if line['count']==0 or any(b['count']==0 for b in bs):
   i=line['line_number'];missing.append(dict(generated_line=i,count=line['count'],branch_counts=[b['count'] for b in bs],source=source[i-1].strip() if i<=len(source) else ''))
 # GNU descriptor/extent and allocation failures are not input-state branches.
 # These generated lines are pinned by source_sha256 below, never filtered from
 # the raw coverage receipt. Fortran RKIND SIGN(1,lat) always returns +/-1;
 # k=1 tropo candidates and k=nVertLevels strato candidates were seeded first.
 descriptor={12,31,110,111,153,169,179,194,233,365,366,367,368,488,491,575,949,1312}
 classified=[]
 for gap in missing:
  if not any(v==0 for v in gap['branch_counts']):continue
  i=gap['generated_line'];text=gap['source']
  if i in descriptor:category='compiler-array-descriptor/extent fallback outside contiguous valid replay contract'
  elif text.startswith(('allocate(', 'deallocate(')):category='compiler-allocation/deallocation failure'
  elif i in {316,728,843,915}:category='native unreachable SIGN(1,lat)==0; SIGN returns +/-1 including signed zero'
  elif i==762:category='native unreachable k==1 pending tropo candidate; bottom three levels already seeded'
  elif i==872:category='native unreachable k==nVertLevels pending strato candidate; top six levels already seeded'
  else:category='UNCLASSIFIED'
  classified.append(dict(generated_line=i,zero_count_branches=sum(v==0 for v in gap['branch_counts']),classification=category))
 rows.append(dict(file=f['file'],lines=len(lines),reached_lines=sum(line['count']>0 for line in lines),branches=len(branches),reached_branches=sum(b['count']>0 for b in branches),functions=f['functions'],unreached=missing,unreached_branch_classification=classified,unclassified_branches=sum(v['zero_count_branches'] for v in classified if v['classification']=='UNCLASSIFIED')))
a.receipt.write_text(json.dumps(dict(kind='strict-standalone-gcov-branch-coverage',compiler=r.get('gcc_version'),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),rows=rows,whole_forecast_proof=False),indent=2)+'\n')
print(json.dumps([dict(file=x['file'],lines=x['lines'],reached_lines=x['reached_lines'],branches=x['branches'],reached_branches=x['reached_branches']) for x in rows]))
