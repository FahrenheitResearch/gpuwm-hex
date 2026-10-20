"""Summarize native gcov branch counts and name every unreached branch."""
import gzip
import json
from pathlib import Path
import subprocess

out=Path('receipts');out.mkdir(exist_ok=True)
records=[]
for path in Path('oracle').glob('*.gcno'):
    if path.stem.startswith('module_sf_mynnsfc_'):
        p=subprocess.run(['gcov','-b','-c','-j',path.name],cwd='oracle',capture_output=True,text=True)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        data=json.loads(gzip.decompress(Path('oracle',path.stem+'.gcov.json.gz').read_bytes()))
        for file in data['files']:
            source=Path(file['file']);lines=Path('oracle',source.name).read_text().splitlines()
            branches=[]
            for line in file['lines']:
                for k,b in enumerate(line.get('branches',[])):
                    branches.append(dict(line=line['line_number'],branch=k,count=b['count'],
                                         source=lines[line['line_number']-1].strip()))
            records.append(dict(file=source.name,branches=len(branches),taken=sum(b['count']>0 for b in branches),
                instrumented_lines=len(file['lines']),executed_lines=sum(line['count']>0 for line in file['lines']),
                functions=[dict(name=f['name'],execution_count=f['execution_count']) for f in file['functions']],
                unreached=[b for b in branches if b['count']==0]))
result=dict(scope='standalone development, driver synthetic inputs; no forecast coverage',files=records,
            branches=sum(r['branches'] for r in records),taken=sum(r['taken'] for r in records))
(out/'coverage.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
