"""Whole LES RK slice on real profiles and an explicit synthetic stencil.

Rust prepares every weather field and native result. Geometry, deformation
coefficients and TKE are test seeds: this receipt does not prove a forecast.
Run the CUDA comparison on one GPU.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from hexcore.mixing_v842_les import LesDissipationV842, les_boundary_arguments


def cuda(inputs, receipt):
    import cupy as cp
    launcher=LesDissipationV842()
    spec=les_boundary_arguments()
    rows=[]
    for profile in sorted(Path(inputs).iterdir()):
        if not (profile/'profiles.json').is_file():
            continue
        provenance=json.loads((profile/'profiles.json').read_text())
        source=json.loads((profile/'source.json').read_text())
        for case in provenance['cases']:
            directory=profile/str(case['case'])
            frame=json.loads((directory/'frame.json').read_text())
            assert [f['name'] for f in frame]==list(spec), 'native ABI order changed'
            values={}
            for field in frame:
                name=field['name']
                array=np.fromfile(directory/f'input-{name}.bin',dtype='<i4' if field['integer'] else '<f4')
                shape=field['shape']
                assert array.size==(int(np.prod(shape)) if shape else 1), name
                if shape or spec[name]['type']=='float' and spec[name]['intent']!='in':
                    values[name]=cp.array(array.reshape(shape or (1,),order='F'),order='F')
                else:
                    values[name]=array.item()
            launcher.apply_rk_mixing(1,**values)
            cp.cuda.get_current_stream().synchronize()
            fields={};words=0;hashes={}
            for field in frame:
                name=field['name']
                expected=np.fromfile(directory/f'output-{name}.bin',dtype='<u4')
                value=values[name]
                if isinstance(value,cp.ndarray):
                    actual=cp.asnumpy(value).ravel(order='F').view('u4')
                else:
                    actual=np.array([value],dtype='i4' if field['integer'] else 'f4').view('u4')
                count=int(np.count_nonzero(actual!=expected))
                words+=expected.size
                if count:
                    fields[name]={'different_words':count,'examples':np.flatnonzero(actual!=expected)[:8].tolist()}
                hashes[name]=hashlib.sha256((directory/f'output-{name}.bin').read_bytes()).hexdigest()
            held={n:v.copy() for n,v in values.items() if isinstance(v,cp.ndarray)}
            launcher.apply_rk_mixing(2,**values)
            launcher.apply_rk_mixing(3,**values)
            for name,before in held.items():
                count=int(cp.count_nonzero(before.view('u4')!=values[name].view('u4')).item())
                if count:
                    fields[name+'/held-rk']={'different_words':count}
            rows.append({'profile':profile.name,**case,'compared_words':int(words),
                         'different_words':sum(f['different_words'] for f in fields.values()),
                         'fields':fields,'source':source,'provenance':{k:v for k,v in provenance.items() if k!='cases'},
                         'native_output_sha256':hashes})
    assert rows, 'no real-profile native cases'
    result={'kind':'five-routine LES RK slice on real profiles and synthetic stencil',
            'whole_forecast_proven':False,'process_count':1,'MPI_used':False,'compiler':'direct NVRTC; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true','compared_words':sum(r['compared_words'] for r in rows),
            'different_words':sum(r['different_words'] for r in rows),'cases':rows}
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    for row in rows:
        if row['different_words']:
            print(json.dumps({k:row[k] for k in ['profile','case','different_words','fields']}))
    return int(result['different_words']!=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs');parser.add_argument('receipt')
    args=parser.parse_args()
    raise SystemExit(cuda(args.inputs,args.receipt))
