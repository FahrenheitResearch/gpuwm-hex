"""Word-exact driver/init receipts. No masks, tolerances or undefined arrays."""
from pathlib import Path
import json
import struct
import hashlib
import argparse
import numpy as np


def compare(native,port,groups,receipt,scope):
    results=[];different=0;words=0
    def read(stream,v):
        count=1;shape=[]
        if v['dims']:
            rank=struct.unpack('>i',stream.read(4))[0]
            assert rank==len(v['dims'])
            for _ in range(rank):
                low,high=struct.unpack('>ii',stream.read(8));shape.append((low,high));count*=high-low+1
        dtype='>u8' if v['typ']=='double' else '>u4'
        data=np.frombuffer(stream.read(count*np.dtype(dtype).itemsize),dtype=dtype)
        assert data.size==count
        return shape,data
    with native.open('rb') as a,port.open('rb') as b:
        for name,fields,records in groups:
            total=0;changes=0;first=[]
            for record in range(records):
                for v in fields:
                    sa,va=read(a,v);sb,vb=read(b,v);assert sa==sb,(name,v['name'],sa,sb)
                    diff=np.flatnonzero(va!=vb);total+=va.size;changes+=diff.size
                    if diff.size and len(first)<12:
                        j=int(diff[0]);first.append({'record':record,'field':v['name'],'element':j,
                                'native':hex(int(va[j])),'port':hex(int(vb[j])),'differing_words':int(diff.size)})
            results.append({'name':name,'records':records,'words':int(total),'differing_words':int(changes),'first':first})
            different+=int(changes);words+=int(total)
        assert not a.read(1) and not b.read(1),'unexpected trailing captures'
    receipt.write_text(json.dumps({'scope':scope,'words':words,'differing_words':different,'groups':results,
            'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
            'port_sha256':hashlib.sha256(port.read_bytes()).hexdigest()},indent=2)+'\n')
    return different


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('directory','inventory','receipt'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--kind',choices=['driver','init-state','init-pools'],required=True)
    a=p.parse_args();d=a.directory
    if a.kind=='driver':
        s=json.loads(a.inventory.read_text());groups=[(n,f,s['records'][n]) for n,f in s['schema'].items()]
        native=d/'driver-native-after.bin';port=d/'driver-host-after.bin'
    elif a.kind=='init-state':
        groups=[('CAM global state',json.loads(a.inventory.read_text())['schema']['globals'],1)]
        native=d/'cam-state.bin';port=d/'init-host-state.bin'
    else:
        groups=[('MPAS initialized pools',[{'name':n,'typ':'float','dims':[':']*r} for n,r in
                [('hybi',2),('ps',1),('aerp',3),('aern',3),('pin',1),('ozone',3)]],1)]
        native=d/'init-native-pools.bin';port=d/'init-host-pools.bin'
    raise SystemExit(bool(compare(native,port,groups,a.receipt,'strict MPAS '+a.kind+' against host glibc transcription')))
