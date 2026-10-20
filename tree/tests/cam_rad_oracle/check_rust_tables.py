"""Compare the real Rust table artifact with native/raw initializer inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from replay_cuda import read_value
from hexcore.mpasphys import cam_rad_v842 as cam


def check(library,tables,native,inventory,receipt):
    fields=json.loads(inventory.read_text())['schema']['globals']
    expected={};sources={}
    with (native/'cam-state.bin').open('rb') as source:
        for field in fields:
            value=read_value(source,field)
            if field['dims']:expected[field['name']]=value;sources[field['name']]='pristine native initialized module state'
        assert not source.read(1)
    with (native/'init-arguments.bin').open('rb') as source:
        for stage in ('camradinit','aer_optics_initialize','aerosol_init','oznini'):
            for name,kind,rank in cam.STAGE_FIELDS[stage]:
                field=dict(name=name,typ=kind,dims=[':']*rank)
                value=read_value(source,field)
                if stage in ('aer_optics_initialize','oznini') and rank:
                    expected[name]=value;sources[name]='independent C++ raw initializer input'
        assert not source.read(1)
    rows=[];different=0
    with cam.RustTables(library,tables,ozone_levels=59,months=12) as decoded:
        for name,(data,lower,extent,stride) in decoded.host_views().items():
            reference,lo,ex,st=expected[name]
            assert (lower,extent,stride)==(lo,ex,st),(name,lower,extent,lo,ex)
            kind=np.uint64 if data.dtype==np.float64 else np.uint32
            a=reference.view(kind);b=data.view(kind)
            indices=np.flatnonzero(a!=b);different+=int(indices.size)
            row=dict(name=name,words=data.size,differing_words=int(indices.size),source=sources[name])
            if indices.size:
                i=int(indices[0]);row.update(first=i,native=hex(int(a[i])),rust=hex(int(b[i])))
            rows.append(row)
    receipt.write_text(json.dumps(dict(scope='real Rust raw table artifact versus native/independent initializer inputs',
        abi=cam.RustTables.__name__,artifact_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        words=sum(r['words'] for r in rows),differing_words=different,fields=rows,
        tables={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tables.iterdir()
                if p.name in ('CAM_ABS_DATA.DBL','CAM_AEROPT_DATA.DBL','OZONE_PLEV.TBL','OZONE_LAT.TBL','OZONE_DAT.TBL')}
        ),indent=2)+'\n')
    print(f'CAM Rust table words={sum(r["words"] for r in rows)}, differences={different}')
    return bool(different)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','tables','native','inventory','receipt'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raise SystemExit(check(a.library,a.tables,a.native,a.inventory,a.receipt))
