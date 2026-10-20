"""Compare LES buoyancy CUDA with strict native on Rust-decoded real fields.

Run on one GPU. Input preparation and native output encoding live
in native_les_v842/real_columns.rs. No interpolation or Python NetCDF path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hexcore.mixing_v842_les import ROUTINES, LesDissipationV842


def frame(directory, qc):
    """Map the lossless Rust transport into the native Fortran argument frame."""
    directory=Path(directory)
    nc,nv=map(int,(directory/'dimensions.txt').read_text().split())
    assert nc>0 and nv>=3
    scalars={'ncells':nc,'nvertlevels':nv,'maxedges':1,'num_scalars':6,
             'cellstart':1,'cellend':nc,'index_qv':1,'index_qc':qc}
    fields=ROUTINES['calculate_n2']['fields'];values={}
    for name in ROUTINES['calculate_n2']['args']:
        if name in scalars:
            values[name]=scalars[name]
        else:
            shape=(6,nv,nc+1) if name=='scalars' else (nv+int(name=='zgrid'),nc+1)
            if name=='bn2':
                values[name]=np.full(shape,.913,dtype='f4',order='F')
            else:
                # Raw, little-endian RKIND words serialized by Rust.
                words=np.fromfile(directory/f'{name}.bin',dtype='<f4')
                assert words.size==np.prod(shape),f'{name}: wrong transport length'
                values[name]=words.reshape(shape,order='F')
            assert fields[name]['dims']
    return values


def cuda(inputs, receipt):
    import cupy as cp
    launcher=LesDissipationV842();rows=[];words=0;different=0
    for directory in sorted(Path(inputs).iterdir()):
        if not (directory/'provenance.json').is_file():continue
        provenance=json.loads((directory/'provenance.json').read_text())
        source=json.loads((directory/'source.json').read_text())
        for qc in [0,2]:
            values=frame(directory,qc)
            device={n:cp.asarray(v,order='F') if isinstance(v,np.ndarray) else v for n,v in values.items()}
            launcher.calculate_n2(**device)
            cp.cuda.get_current_stream().synchronize()
            expected=np.fromfile(directory/f'bn2-qc{qc}.bin',dtype='<u4')
            got=cp.asnumpy(device['bn2']).ravel(order='F').view('u4')
            bad=np.flatnonzero(got!=expected)
            # Verify every input word and the untouched sentinel too.
            input_bad={}
            for name,value in values.items():
                if name=='bn2' or not isinstance(value,np.ndarray):continue
                count=int(np.count_nonzero(cp.asnumpy(device[name]).ravel(order='F').view('u4')!=value.ravel(order='F').view('u4')))
                if count:input_bad[name]=count
                words+=value.size
            count=int(bad.size)+sum(input_bad.values());different+=count;words+=expected.size
            rows.append({'case':directory.name,'index_qc':qc,'different_words':count,
                'bn2_different_words':int(bad.size),'modified_inputs':input_bad,
                'examples':bad[:16].tolist(),'source':source,'inputs':provenance,
                'native_output_sha256':hashlib.sha256((directory/f'bn2-qc{qc}.bin').read_bytes()).hexdigest()})
    assert rows,'no real-column cases'
    result={'kind':'real native restart calculate_n2 standalone replay',
            'whole_forecast_proven':False,'process_count':1,'MPI_used':False,'compiler':'direct NVRTC; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true','other_four_routines_real_inputs_proven':False,
            'compared_words':int(words),'different_words':different,'cases':rows}
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    return int(different!=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('inputs');parser.add_argument('receipt')
    args=parser.parse_args();raise SystemExit(cuda(args.inputs,args.receipt))
