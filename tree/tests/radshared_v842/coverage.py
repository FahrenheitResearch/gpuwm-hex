"""Write compact gcov branch receipts for the copied pristine MPAS sources."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

FILES=('mpas_atmphys_driver_cloudiness.F','module_mp_thompson_cldfra3.F',
       'mpas_atmphys_o3climatology.F','module_ra_rrtmg_vinterp.F',
       'mpas_atmphys_functions.F','mpas_atmphys_date_time.F','cam_support.f90',
       'radius_slices.f90','controls_slice.f90')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('oracle',type=Path)
    parser.add_argument('receipt',type=Path)
    args=parser.parse_args()
    oracle=args.oracle.resolve();receipt=args.receipt.resolve()
    receipt.mkdir(parents=True,exist_ok=True)
    summaries=[];logs=[];artifacts={}
    for name in FILES:
        stem=f'liboracle.so-{Path(name).stem}'
        for suffix in ('.gcno','.gcda'):
            path=oracle/(stem+suffix)
            with path.open('rb') as stream:
                artifacts[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        r=subprocess.run(['gcov','-b','-c','-j',str(oracle/f'{stem}.gcno')],
                         cwd=oracle,capture_output=True,text=True,check=True)
        logs.append(r.stdout+r.stderr)
        with gzip.open(oracle/f'{stem}.gcov.json.gz','rt') as f:
            data=json.load(f)
        for source in data['files']:
            lines=source['lines']
            branches=[b for line in lines for b in line.get('branches',[])]
            summaries.append({'file':source['file'],'functions':source['functions'],
                'lines':len(lines),'lines_executed':sum(line['count']>0 for line in lines),
                'branches':len(branches),'branches_taken':sum(b['count']>0 for b in branches),
                'unreached_branch_lines':[line['line_number'] for line in lines
                    if any(b['count']==0 for b in line.get('branches',[]))]})
    (receipt/'gcov.log').write_text('\n'.join(logs))
    (receipt/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (receipt/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')


if __name__=='__main__':main()
