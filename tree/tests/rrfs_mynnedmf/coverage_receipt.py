"""Summarize pristine GNU branch counters without hiding unreached branches."""
import argparse,json
from pathlib import Path


def summarize(source,out,scope='standalone pristine GNU gcov branch coverage; synthetic columns, no real dumps'):
    text=source.read_text();decoder=json.JSONDecoder();position=0;documents=[]
    while position<len(text):
        while position<len(text) and text[position].isspace():position+=1
        if position==len(text):break
        document,position=decoder.raw_decode(text,position);documents.append(document)
    result={'scope':scope, 'files':{}}
    for doc in documents:
        for file in doc.get('files',[]):
            name=Path(file['file']).name
            row=result['files'].setdefault(name,{'lines':0,'executed_lines':0,'branches':0,'executed_branches':0,'untaken':[],'functions':[]})
            for line in file['lines']:
                row['lines']+=1;row['executed_lines']+=int(line['count']>0)
                for index,branch in enumerate(line.get('branches',[])):
                    row['branches']+=1;row['executed_branches']+=int(branch['count']>0)
                    if branch['count']==0:
                        row['untaken'].append({'line':line['line_number'],'index':index,'fallthrough':branch.get('fallthrough'), 'throw':branch.get('throw')})
            row['functions'] += [{k:f.get(k) for k in ('name','demangled_name','start_line','end_line','execution_count','blocks','blocks_executed')} for f in file.get('functions',[])]
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({n:{k:r[k] for k in ('lines','executed_lines','branches','executed_branches')} for n,r in result['files'].items()}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gcov',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--scope', default='standalone pristine GNU gcov branch coverage; synthetic columns, no real dumps')
    a=p.parse_args();summarize(a.gcov,a.receipt,a.scope)
