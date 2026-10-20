"""Audit all routine bodies, including nested TAUGB1 and MT functions.

MPAS: module_ra_rrtmg_lw.F:1-12538. The complete unified diff remains the
authority for preprocessing, module declarations and non-routine DATA.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def routines(path):
    lines=path.read_text().splitlines(keepends=True);stack=[];out={}
    for number,line in enumerate(lines,1):
        code=line.split('!',1)[0].strip()
        if re.match(r'end\s+(?:subroutine|function)\b',code,re.I):
            name,first=stack.pop()
            body=''.join(lines[first-1:number])
            normalized=re.sub(r'\s+','', '\n'.join(
                row.split('!',1)[0] for row in body.splitlines())).lower()
            out[name]={'lines':[first,number],
                'window_sha256':hashlib.sha256(body.encode()).hexdigest(),
                'normalized_sha256':hashlib.sha256(normalized.encode()).hexdigest()}
        else:
            match=re.search(r'\b(?:subroutine|function)\s+(\w+)(?:\s*\(|\s*&|\s*$)',code,re.I)
            if match:stack.append((match[1].lower(),number))
    if stack:raise ValueError(f'unclosed routines: {stack}')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mpas',type=Path)
    p.add_argument('wrf',type=Path);p.add_argument('receipt',type=Path)
    a=p.parse_args();native=routines(a.mpas);wrf=routines(a.wrf)
    result={'mpas_sha256':hashlib.sha256(a.mpas.read_bytes()).hexdigest(),
            'wrf_lf_sha256':hashlib.sha256(a.wrf.read_bytes()).hexdigest(),
            'wrf_crlf_sha256':hashlib.sha256(a.wrf.read_bytes().replace(b'\n',b'\r\n')).hexdigest(),
            'routines':{}}
    for name,row in native.items():
        other=wrf.get(name)
        result['routines'][name]={'mpas':row,'wrf':other,
            'identical_without_comments_whitespace':bool(other and
                row['normalized_sha256']==other['normalized_sha256'])}
    a.receipt.write_text(json.dumps(result,indent=2)+'\n')
