"""Actual native driver optional-radius and negative-vapor cleanup branches."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np
from hexcore.mpasphys import thompson_v842 as port
if __package__:
    from .replay import synthetic,tables,pointer,compare,engine_math_source
else:
    from replay import synthetic,tables,pointer,compare,engine_math_source


def replay(directory,receipt,backend='host',engine=None):
    directory=directory.resolve();os.chdir(directory)
    native=C.CDLL(str(directory/'liboracle.so'))
    host=C.CDLL(str(directory/'host_port.so')) if backend=='host' else None
    native.init_oracle();native.export_aux();data=tables(directory)
    native.run_oracle_flags.argtypes=[C.c_int,C.c_int,C.c_float,C.c_void_p,C.c_void_p,C.c_void_p]
    if host is not None:
        host.run_host_flags.argtypes=native.run_oracle_flags.argtypes+[C.c_void_p]*len(port.TABLE_ARGUMENTS)
    else:
        import cupy as cp
        maths=engine_math_source(engine.resolve(),backend=='engine-promoted-cuda',backend=='engine-native-log10-cuda')
        device_tables={n:cp.asarray(data[n].reshape(port.TABLE_SHAPES[n],order='F'),order='F') for n in port.TABLE_ARGUMENTS}
        device_tables['_file_sha256']=dict(port.TABLE_FILE_SHA256)
    f,s,_=synthetic();f=f[:,:,:8].copy();s=s[:,:8].copy();nz,ncol=f.shape[1:]
    f[0]=np.float32(0.001);f[1:8]=0
    for c in range(ncol):f[0,c if c<4 else nz-8+c,c]=np.float32(-1e-5)
    f[15:18]=np.float32(0.125)
    rows=[]
    for flags in ((0,0,0),(0,1,1),(1,0,1),(1,1,0),(1,1,1)):
        flags=np.array(flags,np.int32);a=f.copy();b=f.copy();u=s.copy();v=s.copy()
        native.run_oracle_flags(nz,ncol,1.,pointer(a),pointer(u),pointer(flags))
        if host is not None:
            host.run_host_flags(nz,ncol,1.,pointer(b),pointer(v),pointer(flags),*(pointer(data[n]) for n in port.TABLE_ARGUMENTS))
        else:
            fields={n:cp.asarray(b[i]) for i,n in enumerate(port.FIELD_ARGUMENTS)}
            surface={n:cp.asarray(v[i]) for i,n in enumerate(port.SURFACE_ARGUMENTS)}
            port.launch_driver(fields,surface,device_tables,dt=1.,device_math_source=maths,
                               **dict(zip(('has_reqc','has_reqi','has_reqs'),map(int,flags))))
            b=np.stack([cp.asnumpy(fields[n]) for n in port.FIELD_ARGUMENTS])
            v=np.stack([cp.asnumpy(surface[n]) for n in port.SURFACE_ARGUMENTS])
        groups=dict(fields=compare(a,b,port.FIELD_ARGUMENTS),surface=compare(u,v,port.SURFACE_ARGUMENTS))
        words=sum(r['words'] for g in groups.values() for r in g.values())
        rows.append(dict(flags=flags.tolist(),differing_words=words,**groups));print(flags,words,flush=True)
    receipt.write_text(json.dumps(dict(acceptance=False,backend=backend,cases=rows),indent=2)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','engine-cuda','engine-promoted-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    a=p.parse_args();replay(a.directory,a.receipt.resolve(),a.backend,a.engine)
