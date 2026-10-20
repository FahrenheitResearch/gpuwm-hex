"""Record MPAS-vs-WRF procedure diffs before any scheme-kernel reuse."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re


def routines(path):
    text=path.read_text();found={}
    for m in re.finditer(r'^[ \t]*(?:subroutine|real[ \t]+function)[ \t]+(\w+)[ \t]*\(',text,re.M|re.I):
        end=re.search(r'^[ \t]*end[ \t]+(?:subroutine|function)\b[^\n]*',text[m.end():],re.M|re.I)
        if end:
            found.setdefault(m[1].lower(),(text[m.start():m.end()+end.end()],text[:m.start()].count('\n')+1))
    return found


def compare(mpas,wrf,out):
    files=[mpas]+[wrf/f'module_cu_gf_{n}.F' for n in ('deep','sh','wrfdrv')]
    authority=routines(mpas);others={}
    for path in files[1:]:
        for name,value in routines(path).items():others[name]=(path,value)
    rows=[];diff=[]
    norm=lambda s:''.join(re.sub(r'!.*','',s).lower().split())
    for name,(body,line) in authority.items():
        row={'routine':name,'mpas_line':line,'matching_wrf_routine':None,'identical_raw':False,'identical_without_comments_whitespace':False}
        if name in others:
            path,(other,wl)=others[name]
            row.update(matching_wrf_routine=str(path),wrf_line=wl,identical_raw=body==other,identical_without_comments_whitespace=norm(body)==norm(other))
            diff.extend(difflib.unified_diff(body.splitlines(True),other.splitlines(True),fromfile=f'MPAS:{name}:{line}',tofile=f'WRF:{name}:{wl}'))
        rows.append(row)
    out.mkdir(parents=True,exist_ok=True)
    (out/'source-comparison.json').write_text(json.dumps({'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'routines':rows,'reused_kernel_routines':[]},indent=2)+'\n')
    (out/'mpas-wrf-gf.diff').write_text(''.join(diff))
    print(f'{len(rows)} procedures; {sum(r["identical_raw"] for r in rows)} identical whole routines')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mpas',type=Path,required=True);p.add_argument('--wrf-phys',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();compare(a.mpas,a.wrf_phys,a.out)
