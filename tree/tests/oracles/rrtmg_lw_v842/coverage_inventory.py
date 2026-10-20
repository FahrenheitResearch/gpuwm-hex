"""Summarize native gcov misses without conflating compiler scaffolding.

MPAS: module_ra_rrtmg_lw.F:1-12538; module_ra_rrtmg_vinterp.F:1-179.
"""
import argparse
import json
import re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('gcov',type=Path)
p.add_argument('receipt',type=Path)
a=p.parse_args()
stack=[];unreached={};branches={};scaffolding=[];line=0;code='';scope='module'
for record in a.gcov.read_text().splitlines():
    match=re.match(r'\s*([^:]+):\s*(\d+):(.*)',record)
    if match:
        count,line,code=match.groups();line=int(line);code=code.strip()
        if not code.startswith('!'):
            ending=re.match(r'end\s+(?:function|subroutine)\b',code,re.I)
            routine=re.search(r'\b(?:function|subroutine)\s+(\w+)(?:\s*\(|\s*&|\s*$)',code,re.I)
            if routine and not ending:stack.append(routine[1])
            scope=stack[-1] if stack else 'module'
            if count.strip() in ('#####','====='):
                if code.lower().startswith('end module'):
                    scaffolding.append({'line':line,'source':code})
                else:unreached.setdefault(scope,[]).append({'line':line,'source':code})
            if ending and stack:stack.pop()
    elif record.startswith('branch ') and ('never executed' in record or re.search(r'taken 0(?:\s|$)',record)):
        branches.setdefault(scope,[]).append({'line':line,'source':code,'gcov':record})
a.receipt.write_text(json.dumps({'source_gcov':a.gcov.name,'unreached_lines':unreached,
    'untaken_branches':branches,'compiler_end_module_records':scaffolding},indent=2)+'\n')
