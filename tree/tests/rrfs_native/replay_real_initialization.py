"""Authenticate real initialization inputs; delegate decoding and words to Rust.

MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:373-614,1266-1309 and
src/core_init_atmosphere/mpas_atm_advection.F:743-939 at 1d6e370e.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lane',type=Path,required=True)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    contract=json.loads(args.contract.read_text())
    if contract['source_commit']!='1d6e370e7b43a332074809f170532a216c2212fb':
        raise ValueError('initialization contract is outside the pinned source')
    files=[]
    for source in contract['rust_sources']:
        files.append((args.stage/source['staged'],source['sha256']))
    for case in contract['cases']:
        files.append((args.lane/case['init'],case['init_sha256']))
        files.extend((args.lane/record['path'],record['sha256']) for record in case['records'])
    for path,expected in files:
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=expected:
            raise ValueError('real initialization proof dependency changed: '+str(path))
    if args.validate_only:return
    args.out.mkdir(parents=True,exist_ok=True)
    compiled=args.out.parent/'compiled'
    rows=[]
    for case in contract['cases']:
        name=case['case'];init=args.lane/case['init']
        if name=='global':
            target=args.out/(name+'-deformation.json')
            subprocess.run([str(compiled/'replay_real_initialization'),'deformation',str(init),
                str(compiled/'rw_mpas_rrfs_deformation'),str(target)],check=True)
            rows.append(json.loads(target.read_text()))
        # Atmosphere retains deformation from the initialized pool, especially
        # inherited regional border weights. Both constructors and the full
        # work replay exercise select_deformation_weights_rrfs instead.
        records={record['field']:args.lane/record['path'] for record in case['records']}
        target=args.out/(name+'-damping.json')
        subprocess.run([str(compiled/'replay_real_initialization'),'damping',str(init),
            str(records['zgrid']),str(records['dss']),str(case['xnutr']),str(case['zd']),
            str(compiled/'rw_mpas_rrfs_damping'),str(target)],check=True)
        rows.append(json.loads(target.read_text()))
    if not all(row['passed'] and row['different_words']==0 for row in rows):
        raise ValueError('real native initialization differs')
    words=sum(field['words'] for row in rows for field in row['comparisons'])
    if words!=contract['compared_words']:
        raise ValueError('real native initialization output inventory changed')
    summary=dict(schema=contract['schema'],source_commit=contract['source_commit'],
        rust_dependency_commit=contract['gpuwm_commit'],passed=True,different_words=0,
        compared_words=words,scope='global producer deformation and both real native damping work intervals; runtime retained regional weights are inside complete dynamics replay')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
