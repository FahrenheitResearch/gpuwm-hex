"""Audit the lane's v8.4.2 citations while f2 updates the shared checker."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re


def audit(reference, product, receipt):
    text=product.read_text()
    citations=Counter(re.findall(r'([\w]+\.F):(\d+)(?:-(\d+))?',text))
    pins=[]
    for (filename,start,end),occurrences in sorted(citations.items()):
        paths=list((reference/'src').rglob(filename))
        if len(paths)!=1:raise ValueError(f'ambiguous/missing MPAS citation {filename}')
        path=paths[0];lines=path.read_bytes().splitlines(keepends=True)
        first=int(start);last=int(end or start)
        if not 1<=first<=last<=len(lines):raise ValueError(f'invalid citation {filename}:{first}-{last}')
        pins.append(dict(source=str(path.relative_to(reference)),first=first,last=last,
                         occurrences=occurrences,sha256=hashlib.sha256(b''.join(lines[first-1:last])).hexdigest()))
    if not pins:raise ValueError('no MPAS citations found')
    receipt.write_text(json.dumps(dict(release='MPAS-A v8.4.2',
        scope='lane source windows; shared citation checker still targets v8.2.3',
        product_sha256=hashlib.sha256(product.read_bytes()).hexdigest(),
        occurrences=sum(citations.values()),pins=pins),indent=2)+'\n')
    print(f'CAM citation occurrences={sum(citations.values())}, distinct windows={len(pins)}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','product','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.reference,a.product,a.receipt)
