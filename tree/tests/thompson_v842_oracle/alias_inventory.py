"""Label Thompson captures for f1's --scheme thompson replay selector.

Only the manifest's semantic scheme label changes. Actual call keys, argument
schemas, instrumentation hashes and numerical source hashes remain unchanged.
The original inventory hash and explicit alias are retained in the derivative.
"""
import argparse
import hashlib
import json
from pathlib import Path


def alias(source,output):
    raw=source.read_bytes();inventory=json.loads(raw);changed=[]
    for row in inventory['sites']:
        if row['scheme']=='mp_gt_driver':
            row['scheme']='thompson';changed.append(row['site'])
    if not changed:raise ValueError('no actual MPAS Thompson call site in source inventory')
    inventory['scheme_alias_provenance']=dict(original_inventory_sha256=hashlib.sha256(raw).hexdigest(),
        original_scheme='mp_gt_driver',replay_scheme='thompson',sites=changed)
    with output.open('x') as f:json.dump(inventory,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();alias(a.source,a.output)
