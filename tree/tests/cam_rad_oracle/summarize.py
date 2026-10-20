"""Write bounded native-only receipts; never label these CUDA/forecast proof."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

FIELDS = {
    'rthratenlw': lambda c, v: c*(v+1),
    'rthratensw': lambda c, v: c*(v+1),
    'flux': lambda c, v: c*16,
    **{name: lambda c, v: c for name in ('gsw','glw','swcf','lwcf','olr','coszr')},
    **{name: lambda c, v: c*(v+1) for name in ('cemiss','taucldc','taucldi')},
    'abstot_3d': lambda c, v: c*(v+1)*(v+1),
    'absnxt_3d': lambda c, v: c*(v+1)*4,
    'emstot_3d': lambda c, v: c*(v+1),
}


def summarize(directory: Path, receipts: Path):
    data = (directory / 'native-columns.bin').read_bytes()
    header = struct.unpack_from('>7i', data)
    magic, nc, nv, no, na, ns, nm = header
    if magic != 0x43414D38:
        raise ValueError('native CAM header mismatch')
    size = sum(count(nc,nv) for count in FIELDS.values()) * 4
    offset = 28
    records = []
    raw = {}
    while offset < len(data):
        index, variant, mode, doabs = struct.unpack_from('>4i', data, offset)
        offset += 16
        record = {'index': index, 'variant': variant, 'mode': mode, 'doabsems_after': bool(doabs), 'fields': {}}
        begin = offset
        for name, count in FIELDS.items():
            n = count(nc,nv)
            field = data[offset:offset+n*4]
            array = np.frombuffer(field, dtype='>f4')
            record['fields'][name] = {'words': n,
                'sha256': hashlib.sha256(field).hexdigest(),
                'nonfinite': int(np.count_nonzero(~np.isfinite(array)))}
            offset += n*4
        assert offset-begin == size
        raw[variant,mode] = data[begin:offset]
        records.append(record)
    assert offset == len(data)
    cache_checks = []
    for variant in sorted({v for v,m in raw}):
        for mode in (2,3):
            a = np.frombuffer(raw[variant,1],dtype='>u4')
            b = np.frombuffer(raw[variant,mode],dtype='>u4')
            cache_checks.append({'variant':variant,'mode':mode,
                                 'differing_words_from_fresh_lw':int(np.count_nonzero(a!=b))})
    receipt = {'scope':'standalone pristine MPAS native only',
        'cuda_comparison':False,'whole_model_acceptance':False,
        'artifact_sha256':hashlib.sha256(data).hexdigest(),
        'shape':dict(ncol=nc,nlev=nv,nOznLevels=no,nAerLevels=na,nAerosols=ns,nMonths=nm),
        'records':records,'cache_checks':cache_checks}
    (receipts/'native-columns.json').write_text(json.dumps(receipt,indent=2)+'\n')
    coverage = []
    for path in sorted(directory.glob('*.gcov.json.gz')):
        gcov = json.loads(gzip.decompress(path.read_bytes()))
        for file in gcov['files']:
            lines = file['lines']
            branches = [(line['line_number'], i, branch)
                        for line in lines for i,branch in enumerate(line.get('branches',[]))]
            coverage.append({'source':file['file'],
                'executable_lines':len(lines),
                'executed_lines':sum(line['count']>0 for line in lines),
                'branches':len(branches),
                'taken_branches':sum(branch['count']>0 for _,_,branch in branches),
                'unreached_branches':[{'line':n,'branch':i,'fallthrough':b['fallthrough'],
                    'throw':b['throw']} for n,i,b in branches if b['count']==0],
                'uncalled_routines':[f['demangled_name'] for f in file['functions']
                                     if f['execution_count']==0]})
    (receipts/'native-coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    print(json.dumps({'records':len(records),'cache_checks':cache_checks,
                      'coverage':[dict(source=c['source'],branches=c['branches'],
                                       taken=c['taken_branches']) for c in coverage]}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--receipts',type=Path,required=True)
    args=p.parse_args()
    summarize(args.directory,args.receipts)
