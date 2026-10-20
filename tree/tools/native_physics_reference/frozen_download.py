#!/usr/bin/env python3
"""Fetch exact pinned public bytes with bounded parallel HTTP ranges.

This copies encoded bytes through curl/cat. Native ungrib alone interprets GRIB.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shutil
import subprocess
from publish_cache import sha


def fetch(pin, destination, work, parts=4):
    if not pin['url'].startswith('https://') or pin['bytes'] < 1 or not re.fullmatch('[0-9a-f]{64}',pin['sha256']):
        raise ValueError('explicit HTTPS URL, byte size and SHA256 pin required')
    if not 1 <= parts <= 4:
        raise ValueError('one to four bounded transport workers required')
    destination=Path(destination);work=Path(work)
    if destination.is_symlink():raise ValueError('download destination is a symlink')
    if destination.exists() and destination.stat().st_size==pin['bytes'] and sha(destination)==pin['sha256']:
        return dict(pin,path=str(destination),existing=True)
    stage=work/pin['sha256'];stage.mkdir(parents=True,exist_ok=True)
    count=min(parts,pin['bytes']);width=(pin['bytes']+count-1)//count
    def download(start):
        stop=min(start+width,pin['bytes'])-1
        chunk=stage/f'{start}-{stop}.bytes';headers=stage/f'{start}-{stop}.headers'
        expected=f'bytes {start}-{stop}/{pin["bytes"]}'
        def valid():
            if not chunk.is_file() or chunk.stat().st_size!=stop-start+1 or not headers.is_file():return False
            values=re.findall(r'^Content-Range:\s*(.*?)\s*$',headers.read_text(),flags=re.I|re.M)
            return bool(values) and values[-1]==expected
        if not valid():
            subprocess.run(['curl','--silent','--show-error','--fail','--location','--retry','3',
                '--connect-timeout','30','--max-time','1800','--range',f'{start}-{stop}',
                '--max-filesize',str(stop-start+1),'--dump-header',str(headers),'--output',str(chunk),pin['url']],check=True)
        if not valid():raise ValueError('server did not return the requested exact byte range')
        return {'path':str(chunk),'range':expected,'bytes':chunk.stat().st_size,'sha256':sha(chunk)}
    with ThreadPoolExecutor(max_workers=count) as pool:
        chunks=list(pool.map(download,range(0,pin['bytes'],width)))
    assembled=stage/'assembled'
    with assembled.open('wb') as stream:
        subprocess.run(['cat',*(row['path'] for row in chunks)],stdout=stream,check=True)
    if assembled.stat().st_size!=pin['bytes'] or sha(assembled)!=pin['sha256']:
        raise ValueError('assembled download differs from frozen original pin')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Stage and destination may be on different filesystems. Copy first, then
    # atomically rename the verified destination-side temporary.
    temporary=destination.with_name(destination.name+'.verified')
    shutil.copyfile(assembled,temporary)
    if sha(temporary)!=pin['sha256']:raise ValueError('copied download differs')
    temporary.replace(destination)
    result=dict(pin,path=str(destination),parts=chunks,existing=False)
    shutil.rmtree(stage)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pin',type=Path);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();result=fetch(json.loads(a.pin.read_text()),a.destination,a.work)
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['url','bytes','sha256','existing']}))


if __name__=='__main__':main()
