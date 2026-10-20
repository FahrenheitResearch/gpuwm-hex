#!/usr/bin/env python3
"""Index explicit forecast jobs only after their native clocks are witnessed.

No model data processing or identity inference. A completed native reference is
an executed artifact, with manifest hashes and a separate Rust clock witness.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from publish_cache import sha
from run_matrix import validate_manifest


def entry(row, details):
    directory=Path(row['publish'])
    runtime=json.loads((Path(row['deck']).parent/'INDEX.json').read_text())['runtime']
    build=json.loads((Path(row['build'])/'MANIFEST.json').read_text())
    result={'key':{'pin_set_hash':row['pin_set_hash'],'recipe':build['recipe'],
            'mesh':runtime['mesh'],'date':runtime['start'],'deck':Path(row['deck']).name,
            'ranks':row['ranks'],'duration':runtime['duration']},
            'directory':str(directory),'runtime':runtime,'status':'pending'}
    if not (directory/'PUBLICATION.json').is_file():return result
    publication=json.loads((directory/'PUBLICATION.json').read_text())
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    validate_manifest(manifest,row)
    digest=sha(directory/'MANIFEST.json')
    if publication['files'].get('MANIFEST.json')!=digest:
        raise ValueError('forecast publication manifest hash differs')
    if manifest.get('runtime')!=runtime:raise ValueError('forecast catalog runtime differs from deck')
    retired=manifest.get('outputs_retired',False)
    if retired and publication['files'].get('CLOCKS.json')!=manifest['retired_clock_receipt_sha256']:
        raise ValueError('retired native clock receipt is not published')
    if not retired and any(publication['files'].get(name)!=value for name,value in manifest['outputs'].items()):
        raise ValueError('forecast publication output hashes differ from manifest')
    result.update(status='awaiting-clock-witness',manifest_sha256=digest,
                  publication_sha256=sha(directory/'PUBLICATION.json'),outputs=manifest['outputs'])
    key=hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    receipt=details/(key+'.json')
    if not receipt.is_file():return result
    witness=json.loads(receipt.read_text())
    if witness['directory']!=str(directory) or witness['manifest_sha256']!=digest:
        raise ValueError('forecast clock witness belongs to another manifest')
    if {file['file']:file['sha256'] for file in witness['files']}!=manifest['outputs']:
        raise ValueError('forecast clock witness output hashes differ')
    result.update(status=('native-digest-reference-complete' if retired else 'native-reference-complete') if witness['complete'] else 'clock-witness-failed',
                  clock_receipt=str(receipt.resolve()),clock_receipt_sha256=sha(receipt),
                  errors=witness['errors'])
    if retired:result['storage']={'retention':'digests-only; original native bytes retired','receipt':str(directory/'CLOCKS.json')}
    if (directory/'ARCHIVE.json').is_file():
        archive=json.loads((directory/'ARCHIVE.json').read_text())
        result['storage']={'host':archive['host'],'archive':archive['archive'],
                           'archive_sha256':archive['archive_sha256'],
                           'byte_restore_receipt':str(directory/'ARCHIVE.json')}
    return result


def catalog(jobs, details):
    if not jobs:raise ValueError('empty forecast catalog')
    entries=[entry(row,details) for row in jobs]
    keys=[json.dumps(row['key'],sort_keys=True) for row in entries]
    if len(set(keys))!=len(keys):raise ValueError('duplicate forecast catalog key')
    return {'schema':'mpas.native-forecast-catalog/v1',
            'complete':all(row['status'] in ('native-reference-complete','native-digest-reference-complete') for row in entries),
            'identity_claim':'native execution and output-clock provenance; no port equivalence implied',
            'entries':entries}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('index',type=Path);p.add_argument('--details',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    jobs=json.loads(a.index.read_text())['jobs'];previous=None
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists() and json.loads(a.out.read_text()).get('schema')!='mpas.native-forecast-catalog/v1':
        raise ValueError('refusing to overwrite an unrelated reference index')
    while True:
        result=catalog(jobs,a.details);payload=json.dumps(result,indent=2)+'\n'
        if payload!=previous:
            stage=a.out.with_name(a.out.name+'.partial.'+str(os.getpid()))
            stage.write_text(payload);stage.replace(a.out)
            receipt={'catalog':str(a.out.resolve()),'sha256':sha(a.out),'complete':result['complete'],
                'statuses':{status:sum(row['status']==status for row in result['entries'])
                            for status in sorted({row['status'] for row in result['entries']})}}
            a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
            print(json.dumps(receipt),flush=True);previous=payload
        if not a.watch or result['complete']:break
        time.sleep(20)


if __name__=='__main__':main()
