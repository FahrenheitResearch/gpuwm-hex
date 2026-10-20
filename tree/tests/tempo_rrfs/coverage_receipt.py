"""Summarize pristine TEMPO gcov, retaining every unreached branch location."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True)
args=parser.parse_args();rows=[]
for path in sorted(args.build.glob('module_mp_tempo_*.gcov.json.gz')):
    data=json.loads(gzip.decompress(path.read_bytes()))
    for file in data['files']:
        source=(args.build/file['file']).resolve()
        if not source.exists():raise ValueError(f'gcov source unavailable: {source}')
        lines=source.read_text().splitlines();executable=file['lines']
        branches=[b for line in executable for b in line.get('branches',[])]
        unreached=[]
        for line in executable:
            zero=[i for i,b in enumerate(line.get('branches',[])) if b['count']==0]
            if zero:
                index=line['line_number']
                unreached.append({'line':index,'branch_indices':zero,'native_statement':lines[index-1].strip(),'function':line.get('function_name',next((f['name'] for f in file.get('functions',[]) if f['start_line']<=index<=f['end_line']),None)),'line_executions':line['count']})
        rows.append({'fork_file':'src/core_atmosphere/physics/physics_noaa/TEMPO/src/'+source.name,
                     'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                     'executable_lines':len(executable),'lines_executed':sum(line['count']>0 for line in executable),
                     'branches':len(branches),'branches_taken':sum(b['count']>0 for b in branches),'unreached_conditions':unreached})
receipt={'schema':'tempo-pristine-gcov-v1','forecast_proven':False,
         'fork':'1d6e370e7b43a332074809f170532a216c2212fb','files':rows}
args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='unreached_conditions'} for row in rows]))
