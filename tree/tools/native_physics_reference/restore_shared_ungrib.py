#!/usr/bin/env python3
"""Restore retired shared native inputs against their original encoded/output pins.

The WPS native executable alone decodes fields. This central recovery is for
previously retired inputs; ordinary decode still rejects missing sealed bytes.
"""
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from cache_cases import sha
from shared_ungrib import decode
from frozen_download import fetch


def restore(row,pins,ungrib,vtable,transport):
    directory=Path(row['directory'])
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    records=[]
    for name,digest in manifest['files'].items():
        target=directory/name
        if target.is_file():
            if sha(target)!=digest:raise ValueError('existing shared native input differs')
            records.append({'name':name,'sha256':digest,'restored':False});continue
        source=manifest['sources'][name]
        pin=next((p for p in pins['files'] if p['sha256']==source['sha256']),None)
        if pin is None:raise ValueError('missing frozen source pin for restoration')
        raw=directory/(pin['name']+'.restore-download')
        evidence=fetch(pin,raw,transport)
        time=source.get('time',name[4:]+':00:00')
        restored=decode(raw,time,directory,ungrib,vtable,restore_missing=True)
        if sha(restored)!=digest:raise ValueError('restored original native bytes differ')
        records.append({'name':name,'sha256':digest,'restored':True,'transport':evidence})
        raw.unlink()
        print(json.dumps({'restored_shared_native_input':str(restored),'sha256':digest}),flush=True)
    return {'directory':str(directory),'files':records,'complete':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['index','raw_manifest','ungrib','vtable','transport','receipt']:
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    a=p.parse_args();rows=json.loads(a.index.read_text())['directories']
    if len({r['directory'] for r in rows})!=len(rows):raise ValueError('duplicate shared directory')
    pins=json.loads(a.raw_manifest.read_text());results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(restore,row,pins,a.ungrib,a.vtable,a.transport) for row in rows]
        for future in as_completed(futures):
            results.append(future.result());a.receipt.write_text(json.dumps({'complete':len(results)==len(rows),'directories':results},indent=2)+'\n')
if __name__=='__main__':main()
