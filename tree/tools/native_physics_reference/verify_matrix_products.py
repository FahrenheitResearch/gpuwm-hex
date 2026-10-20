#!/usr/bin/env python3
"""Witness clocks of explicitly listed published native products as they appear."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from verify_products import verify

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('index',type=Path)
    p.add_argument('--comparator',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--details',type=Path,required=True)
    a=p.parse_args();pending=json.loads(a.index.read_text())['products'];results=[]
    if not pending:raise ValueError('empty clock-witness product index')
    a.details.mkdir(parents=True,exist_ok=True)
    while pending:
        progress=False
        for row in list(pending):
            directory=Path(row['directory'])
            if not (directory/'PUBLICATION.json').is_file():continue
            result=verify(directory,a.comparator.resolve(),row.get('inactive_packages',[]))
            key=hashlib.sha256(str(directory).encode()).hexdigest()[:16]
            detail=a.details/(key+'.json');detail.write_text(json.dumps(result,indent=2)+'\n')
            results.append({'directory':str(directory),'receipt':str(detail),'complete':result['complete'],
                            'files':len(result['files']),'errors':result['errors']})
            pending.remove(row);progress=True
            a.receipt.write_text(json.dumps({'complete':not pending and all(r['complete'] for r in results),
                'products':results,'pending':pending},indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
            if not result['complete']:raise SystemExit(1)
        if not progress:time.sleep(20)

if __name__=='__main__':main()
