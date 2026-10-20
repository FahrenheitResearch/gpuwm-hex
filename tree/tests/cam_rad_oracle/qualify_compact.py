"""Qualify the whole standalone capture transport on every native fixture.

Every scalar/array is decoded and encoded by Rust, then compared there. This
is transport qualification, separate from solver or whole-forecast identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import numpy as np
from compact_codec import Codec


KINDS={'float':1,'double':2,'int':3,'bool':4}
DTYPES={'float':'float32','double':'float64','int':'int32','bool':'int32'}


def write_host(codec, stream, field, value):
    rank=len(field['dims'])
    if value is None:
        words=np.empty(0,dtype=DTYPES[field['typ']]);low=[1]*5;extent=[0]*rank+[1]*(5-rank)
    elif rank:
        words,lo,ex,_=value;low=list(lo)+[1]*(5-rank);extent=list(ex)+[1]*(5-rank)
    else:
        words=np.asarray([value],dtype=DTYPES[field['typ']]);low=extent=[1]*5
    codec.encode(stream,KINDS[field['typ']],rank,low,extent,words,field.get('optional',False))


def qualify(fixtures, library, output):
    codec=Codec(library)
    rows=json.loads(fixtures.read_text())
    output.mkdir(parents=True,exist_ok=False)
    summary=dict(scope='complete Rust standalone CAM capture decode/encode/word transport',
                 process_count=1,host=socket.gethostname(),excluded_words=0,
                 library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                 numerical_proof=False,whole_model_acceptance=False,fixtures=[],complete=False)
    try:
        for row in rows:
            directory=Path(row['directory']);schema=json.loads(Path(row['inventory']).read_text())['schema']
            files=[]
            for name,meta in row['files'].items():
                native=directory/name
                assert hashlib.sha256(native.read_bytes()).hexdigest()==meta['sha256'],native
                boundary=name in ('cam-before.bin','cam-after.bin')
                fields=schema['arguments' if boundary else 'globals']
                encoded=output/(row['label']+'-'+name)
                records=0
                with native.open('rb') as source,encoded.open('wb') as dest:
                    while source.tell()<native.stat().st_size:
                        if boundary:
                            field=dict(name='indirect',typ='bool',dims=[])
                            write_host(codec,dest,field,codec.read(source,field))
                        for field in fields:write_host(codec,dest,field,codec.read(source,field))
                        records+=1
                assert records==(row['expected_records'] if boundary else 1),(native,records)
                groups=[dict(name=name,fields=fields,records=records,indirect=boundary)]
                comparison=codec.compare(native,encoded,groups)
                digest=hashlib.sha256(encoded.read_bytes()).hexdigest()
                assert comparison['differing_words']==0 and digest==meta['sha256'],native
                files.append(dict(file=name,records=records,words=comparison['words'],
                                  differing_words=0,sha256=digest,bytes=meta['bytes']))
                encoded.unlink()
            summary['fixtures'].append(dict(label=row['label'],files=files))
            print(json.dumps(summary['fixtures'][-1]),flush=True)
        summary['complete']=True
        summary['words']=sum(f['words'] for row in summary['fixtures'] for f in row['files'])
        summary['differing_words']=0
        return 0
    finally:
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixtures','library','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(qualify(args.fixtures,args.library,args.output))
