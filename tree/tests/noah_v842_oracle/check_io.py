"""Replay native MPAS field conversions with inactive-cell sentinels and option branches."""
from pathlib import Path
import argparse
import ctypes as C
import json
import numpy as np
from hexcore.mpasphys.noah_v842 import _MPAS_INPUT_FIELDS, _MPAS_OUTPUT_FIELDS, COLUMN_LAYOUTS, _module

p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');args=p.parse_args()
base=Path(__file__).resolve().parent
if args.gpu:
    import cupy as cp
    lib=None
else:
    lib=C.CDLL(str(base/'libnoah_host.so'))
native=None if args.gpu else C.CDLL(str(base/'libnoah_oracle.so'))
sizes={'soil':4,'atmosphere':2,'pressure':2}
offsets={};width=0
for key,shape in _MPAS_INPUT_FIELDS.items():
    if shape=='integer_surface':continue
    offsets[key]=width;width+=sizes.get(shape,1)
results=[];cells=11;start=2;count=7
for mode in range(4):
    path=Path('n1out')/f'mpas-io-{mode}.npz'
    if args.gpu:
        z=np.load(path);packed=z['input'];ivals=z['input_int'];options=z['options'];changed=z['changed']
        expected=z['native_from'];expected_int=z['native_from_int'];expected_to=z['native_to']
    else:
        rng=np.random.default_rng(642+mode)
        packed=rng.uniform(.1,10.,(cells,width)).astype(np.float32)
        packed[:,offsets['sfc_albedo']]=rng.uniform(.01,.9,cells)
        ivals=np.column_stack((np.arange(cells)+1,np.arange(cells)+2)).astype(np.int32)
        options=np.array([13,15,2,mode%2,mode%2,mode//2],np.int32)
        expected=np.empty((count,79),np.float32);expected_int=np.empty((count,7),np.int32)
        fn=native.io_from;fn.argtypes=[C.c_void_p]*4+[C.c_int]*3+[C.c_float]*2+[C.c_void_p]
        fn(packed.ctypes.data,ivals.ctypes.data,expected.ctypes.data,expected_int.ctypes.data,cells,start,count,60.,.55,options.ctypes.data)
        changed=expected.copy()
        # Every field written by the LSM and its residual receives a distinct new value.
        output_slots=[8,9,10,11,12,16,17,18,19,21,22,23,24,28,30,31,46,47,48,49,50,52,53,59,60,63,64,65,66,67,74]
        output_slots+=list(range(38,46))+list(range(55,59))+list(range(68,72))
        changed[:,output_slots]=rng.uniform(-100.,100.,(count,len(output_slots)))
        expected_to=packed.copy()
        fn=native.io_to;fn.argtypes=[C.c_void_p]*4+[C.c_int]*3+[C.c_float]*2+[C.c_void_p]
        fn(expected_to.ctypes.data,ivals.ctypes.data,changed.ctypes.data,expected_int.ctypes.data,cells,start,count,60.,.55,options.ctypes.data)
        np.savez(path,input=packed,input_int=ivals,options=options,changed=changed,native_from=expected,native_from_int=expected_int,native_to=expected_to)
    fields={}
    for key,shape in _MPAS_INPUT_FIELDS.items():
        if shape=='integer_surface':a=ivals[:,0 if key=='ivgtyp' else 1].copy()
        else:
            size=sizes.get(shape,1);a=packed[:,offsets[key]:offsets[key]+size].copy()
            if size==1:a=a[:,0]
        fields[key]=cp.asarray(a) if args.gpu else a
    x=cp.empty((count,79),cp.float32) if args.gpu else np.empty((count,79),np.float32)
    ix=cp.empty((count,7),cp.int32) if args.gpu else np.empty((count,7),np.int32)
    tail=(np.int32(2),np.int32(2),np.int32(start),np.int32(count),np.float32(60.),np.float32(.55),
          np.int32(options[4]),np.int32(options[5]),np.int32(13),np.int32(15),np.int32(2),np.int32(options[3]))
    if args.gpu:
        _module().get_function('noah_from_mpas')((1,),(64,),(x,ix,*fields.values(),*tail))
        got=cp.asnumpy(x);got_int=cp.asnumpy(ix)
    else:
        fn=lib.noah_from_mpas;fn.argtypes=[C.c_void_p]*(2+len(fields))+[C.c_int]*4+[C.c_float]*2+[C.c_int]*6
        fn(x.ctypes.data,ix.ctypes.data,*(a.ctypes.data for a in fields.values()),*tail)
        got=x;got_int=ix
    from_diff=int((got.view(np.uint32)!=expected.view(np.uint32)).sum())
    int_diff=int((got_int!=expected_int).sum())
    x=cp.asarray(changed) if args.gpu else changed
    status=cp.zeros(count,cp.int32) if args.gpu else np.zeros(count,np.int32)
    status[1]=2438
    expected_to=expected_to.copy()
    expected_to[start+1]=packed[start+1]
    if args.gpu:
        _module().get_function('noah_to_mpas')((1,),(64,),(x,status,*(fields[k] for k in _MPAS_OUTPUT_FIELDS),np.int32(start),np.int32(count)))
        fields={k:cp.asnumpy(v) for k,v in fields.items()}
    else:
        fn=lib.noah_to_mpas;fn.argtypes=[C.c_void_p]*(2+len(_MPAS_OUTPUT_FIELDS))+[C.c_int]*2
        fn(x.ctypes.data,status.ctypes.data,*(fields[k].ctypes.data for k in _MPAS_OUTPUT_FIELDS),start,count)
    by_field={}
    for key,offset in offsets.items():
        size=sizes.get(_MPAS_INPUT_FIELDS[key],1)
        want=expected_to[:,offset:offset+size].copy();got=fields[key].reshape(cells,size)
        d=int((got.view(np.uint32)!=want.view(np.uint32)).sum())
        if d:by_field[key]=d
    results.append({'options':mode,'from_differing_words':from_diff,'integer_differing_words':int_diff,
                    'to_differing_words':sum(by_field.values()),'to_by_field':by_field,
                    'failed_column_writeback_checked':True})
receipt=Path('receipts/io-cuda.json') if args.gpu else Path('n1out/io-host.json')
receipt.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if any(r['from_differing_words'] or r['integer_differing_words'] or r['to_differing_words'] for r in results):raise SystemExit(1)
