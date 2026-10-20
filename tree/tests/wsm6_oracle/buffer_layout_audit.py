"""Authenticate CUDA automatic/global buffer species extents against native declarations.

A column has one horizontal point. Native arrays with a third species dimension
need three vertical vectors; km+1 interface arrays retain the existing padding.
This prevents cross-array writes that a host-only replay can fail to expose.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def declarations(path):
    routine=None;pending='';result={}
    for original in path.read_text().splitlines():
        line=original.split('!',1)[0].strip().lower()
        if not line:continue
        pending+=line.lstrip('&').rstrip('&')+' '
        if line.endswith('&'):continue
        line=pending;pending=''
        match=re.search(r'\bsubroutine\s+(\w+)\(',line)
        if match:routine=match[1]
        if re.search(r'\bend\s+subroutine\b',line):routine=None
        if routine:
            match=re.search(r'dimension\(([^)]*)\)[^:]*::\s*(.*)',line)
            if match:
                dimensions=[d.strip() for d in match[1].split(',')]
                for name in match[2].split(','):
                    name=name.strip()
                    if re.fullmatch(r'\w+',name):result[routine,name]=dimensions
    return result


def audit(oracle,receipt):
    from hexcore.mpasphys.wsm6_v842 import CUDA_MODULES
    native={}
    for name in ('mp_wsm6.F90','mp_wsm6_effectRad.F90','mp_radar.F90'):
        native.update(declarations(oracle/name))
    routine=None;rows=[]
    for line in CUDA_MODULES.splitlines():
        match=re.match(r'__device__ \w+ (\w+)\([^;]*\) \{',line)
        if match:routine=match[1]
        match=re.match(r'(float|double|bool) (\w+)\[WSM6_KMAX( \* 3)? \+ 3\];$',line)
        if match:
            kind,name,species=match.groups()
            dimensions=native[routine,name]
            expected=3 if any(d=='3' for d in dimensions) else 1
            actual=3 if species else 1
            if expected!=actual:
                raise AssertionError(f'{routine}/{name}: native {dimensions}, CUDA species extent {actual}')
            # Distinct workspaces must preserve the same native extent. A missing
            # mapping would leave a GPU call dereferencing unrelated storage even
            # when the host witness's automatic arrays passed its word comparison.
            field=routine+'_'+name
            extent='WSM6_KMAX'+(' * 3' if species else '')+' + 3'
            declaration=f"  {kind} {field}[{extent}];"
            binding=f"{kind} *{name}=S.workspace->{field};"
            if CUDA_MODULES.count(declaration)!=1 or CUDA_MODULES.count(binding)!=1:
                raise AssertionError(f'{routine}/{name}: missing or shared native workspace storage')
            rows.append(dict(routine=routine,array=name,native_dimensions=dimensions,
                             workspace_field=field,
                             cuda_species_extent=actual))
    if not rows:raise AssertionError('no native vertical buffers audited')
    value=dict(scope=__doc__,process_count=1,acceptance=False,
               production_source_sha256=hashlib.sha256(Path(__import__('hexcore.mpasphys.wsm6_v842',fromlist=['']).__file__).read_bytes()).hexdigest(),
               audited_buffers=len(rows),three_species_buffers=sum(r['cuda_species_extent']==3 for r in rows),
               rows=rows)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    return len(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('oracle',type=Path);parser.add_argument('receipt',type=Path)
    args=parser.parse_args();print(audit(args.oracle,args.receipt))
