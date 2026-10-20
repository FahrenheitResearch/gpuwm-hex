"""Report measured GNU gcov branches, including every unreached native branch."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def report(root,target,scope='standalone native synthetic wrapper/topology/spread execution, not forecast coverage'):
    rows=[];total=0;reached=0;functions=[]
    for path in sorted(root.glob('*.gcov.json.gz')):
        blob=json.loads(gzip.decompress(path.read_bytes()))
        for source in blob['files']:
            name=Path(source['file']).name
            if name not in ('module_cu_gfl_deep.F90','module_cu_gfl_sh.F90','module_cu_gfl_mpas.F90','driver_slice.F90','topology_slice.F90'):continue
            count=0;hit=0;miss=[]
            for line in source['lines']:
                for ix,b in enumerate(line.get('branches',())):
                    count+=1;hit+=int(b['count']>0)
                    if b['count']==0:miss.append({'line':line['line_number'],'branch':ix,'function':line.get('function_name'),'fallthrough':b['fallthrough'],'throw':b['throw']})
            functions.extend({'source':name,'name':fn['name'],'calls':fn['execution_count'],'start':fn['start_line'],'end':fn['end_line']} for fn in source['functions'])
            rows.append({'source':name,'branches':count,'reached':hit,'unreached':miss,'gcov_sha256':hashlib.sha256(path.read_bytes()).hexdigest()});total+=count;reached+=hit
    if not rows:raise ValueError('no measured gcov files')
    result={'scope':scope,'compiler':'GNU 15.2','pin':'1d6e370e/GFL bf6a4b0','branches':total,'reached':reached,'percent':100*reached/total if total else None,'sources':rows,'functions':functions}
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'branches':total,'reached':reached,'unreached_functions':[x for x in functions if not x['calls']]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--scope',default='standalone native synthetic wrapper/topology/spread execution, not forecast coverage');a=p.parse_args();report(a.root,a.receipt,a.scope)
