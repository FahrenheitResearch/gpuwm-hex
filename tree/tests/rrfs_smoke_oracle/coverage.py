"""Summarize actual GNU branch counts without assigning a proof grade."""
import json,gzip,subprocess
from pathlib import Path
root=Path('scratch/native').resolve();out=Path('scratch/gcov');out.mkdir(exist_ok=True)
rows=[];driver_rows=[]
for gcno in sorted(root.glob('*.gcno')):
 if gcno.stem in ('framework','constants','driver_bridge'):continue
 p=subprocess.run(['gcov','--json-format','--branch-probabilities',str(gcno)],cwd=out,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 for path in out.glob(gcno.stem+'*.gcov.json.gz'):
  data=json.loads(gzip.decompress(path.read_bytes()));path.unlink()
  for f in data['files']:
   branches=[dict(line_number=line['line_number'],**b) for line in f['lines'] for b in line['branches']]
   funcs=[dict(name=v['name'],start_line=v['start_line'],end_line=v['end_line'],execution_count=v['execution_count']) for v in f['functions']]
   target=driver_rows if gcno.stem=='driver_slice' else rows
   target.append(dict(file=f['file'],branch_total=len(branches),branch_taken=sum(v['count']>0 for v in branches),unreached_branches=[v for v in branches if not v['count']],functions=funcs))
result=dict(scope='standalone synthetic full wrapper; gcov instrumentation, not native forecast',files=rows,driver_files=driver_rows)
Path('receipts/coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(sum(v['branch_taken'] for v in rows),'of',sum(v['branch_total'] for v in rows),'branches')
