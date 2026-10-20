"""Word comparison of compiled CAM native/host/CUDA test boundary captures."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def records(path,fields):
    data=path.read_bytes();offset=0
    while offset<len(data):
        values={}
        values['indirect']=np.frombuffer(data,dtype='>u4',count=1,offset=offset)
        offset+=4
        for field in fields:
            count=1;shape=[]
            if field['dims']:
                rank=struct.unpack_from('>i',data,offset)[0];offset+=4
                assert rank==len(field['dims']), (field['name'],rank)
                for d in range(rank):
                    lo,hi=struct.unpack_from('>ii',data,offset);offset+=8
                    shape.append(hi-lo+1);count*=shape[-1]
            dtype='>u8' if field['typ']=='double' else '>u4'
            values[field['name']]=np.frombuffer(data,dtype=dtype,count=count,offset=offset)
            offset+=count*(8 if field['typ']=='double' else 4)
        yield values
    assert offset==len(data)


def compare(reference,candidate,inventory,receipt):
    fields=json.loads(inventory.read_text())['schema']['arguments']
    rows=[];total=0;defined=0
    for record,(ref,got) in enumerate(zip(records(reference,fields),records(candidate,fields),strict=True),1):
        counts={};scratch={}
        for name in ref:
            different=np.flatnonzero(ref[name]!=got[name])
            n=int(different.size);total+=n
            # These are explicitly unassigned local diagnostics in LW-only.
            undefined=name in ('taucldc','taucldi') and not ref['dosw'][0]
            if not undefined: defined+=n
            target=scratch if undefined else counts
            if n:
                i=int(different[0])
                target[name]={'differing_words':n,'first_index':i,
                              'reference_hex':hex(int(ref[name][i])),
                              'candidate_hex':hex(int(got[name][i]))}
        rows.append({'record':record,'defined_differences':counts,'native_unwritten_scratch_differences':scratch})
    result={'scope':'standalone whole CAM boundary','reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'total_differing_words':total,'defined_differing_words':defined,
            'records':rows,'whole_model_acceptance':False,'process_count':1,
            'excluded_words':0,'comparison_result':'PASS' if total==0 else 'FAIL',
            'native_unwritten_classification':'diagnostic only; no approved exclusion'}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'records':len(rows),'total_differing_words':total,'defined_differing_words':defined}))
    return total


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','inventory','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(bool(compare(a.reference,a.candidate,a.inventory,a.receipt)))
