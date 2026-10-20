"""Pristine precipitation bucket and humidity driver-slice word comparison."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from hexcore.mpasphys import thompson_v842 as port
if __package__:
    from .replay import synthetic,compare,pointer,engine_math_source
else:
    from replay import synthetic,compare,pointer,engine_math_source


def replay(directory,receipt,backend='host',engine=None):
    native=C.CDLL(str(directory.resolve()/'diagnostics_oracle.so'))
    host=C.CDLL(str(directory.resolve()/'host_port.so')) if backend=='host' else None
    f,_,_=synthetic();nz,ncol=f.shape[1:]
    fields=np.stack((f[0],np.full((nz,ncol),0.7,np.float32),f[12],f[8],f[9],f[10]))
    if host is None:
        import cupy as cp
        maths=engine_math_source(engine.resolve(),native_log10=backend=='engine-native-log10-cuda')
        device_fields={n:cp.asarray(fields[i]) for i,n in enumerate(('qv','rho','dz','th','pii','p'))}
    rows=[]
    for flag in (0,1):
        for bucket in (-1.,0.,1.,500.):
            s=np.full((7,ncol),0.25,np.float32);u=s.copy()
            d=np.full((8,ncol),0.125,np.float32);e=d.copy()
            d[1]=np.resize(np.array((0.,1.,2.,4.),np.float32),ncol);e[:]=d
            counter=np.full(ncol,3,np.int32);other=counter.copy()
            if host is None:
                ds={n:cp.asarray(s[i]) for i,n in enumerate(port.SURFACE_ARGUMENTS[:7])}
                dd={n:cp.asarray(d[i]) for i,n in enumerate(port.PRECIP_ARGUMENTS)};dd['i_rainnc']=cp.asarray(counter)
            for phase in (0,1):
                if phase==1:
                    # Totals from multiple native substeps, including exact
                    # equality and totals exceeding two buckets.
                    s[0]=np.resize(np.array((0.,0.5,1.,3.),np.float32),ncol)
                    s[2]=np.float32(0.01);s[4]=np.float32(0.02);u[:]=s
                    if host is None:
                        for i,n in enumerate(port.SURFACE_ARGUMENTS[:7]):ds[n][...]=cp.asarray(s[i])
                signature=[C.c_int,C.c_int]+[C.c_void_p]*4+[C.c_int,C.c_int,C.c_float]
                native.precip_ref.argtypes=signature
                native.precip_ref(nz,ncol,pointer(fields),pointer(s),pointer(d),pointer(counter),phase,flag,bucket)
                if host is not None:
                    host.precip_ref.argtypes=signature
                    host.precip_ref(nz,ncol,pointer(fields),pointer(u),pointer(e),pointer(other),phase,flag,bucket)
                else:
                    port.launch_precip('from' if phase==0 else 'to',device_fields,ds,dd,device_math_source=maths,l_acrain=bool(flag),bucket=bucket)
                    u=np.stack([cp.asnumpy(ds[n]) for n in port.SURFACE_ARGUMENTS[:7]])
                    e=np.stack([cp.asnumpy(dd[n]) for n in port.PRECIP_ARGUMENTS]);other=cp.asnumpy(dd['i_rainnc'])
                groups=dict(surface=compare(s,u,port.SURFACE_ARGUMENTS[:7]),diagnostics=compare(d,e,port.PRECIP_ARGUMENTS),counter=compare(counter[None],other[None],('i_rainnc',)))
                words=sum(r['words'] for g in groups.values() for r in g.values())
                rows.append(dict(phase=phase,flag=flag,bucket=bucket,differing_words=words,**groups))
                print(f'precip phase={phase} flag={flag} bucket={bucket:g}: {words} differing words',flush=True)
    a=np.zeros((nz,ncol),np.float32);b=a.copy()
    native.relhum_ref.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p]
    native.relhum_ref(nz,ncol,pointer(fields),pointer(a))
    if host is not None:
        host.relhum_ref.argtypes=native.relhum_ref.argtypes;host.relhum_ref(nz,ncol,pointer(fields),pointer(b))
    else:
        out=cp.asarray(np.zeros((nz,ncol),np.float32))
        args=tuple(device_fields[n] for n in ('qv','th','pii','p'))+(out,np.int32(nz*ncol))
        port._module(out.device.id,maths,nz).get_function('mpas_thompson_relhum')(((nz*ncol+31)//32,),(32,),args)
        b=cp.asnumpy(out)
    humidity=compare(a[None],b[None],('relhum',));words=humidity['relhum']['words']
    rows.append(dict(differing_words=words,humidity=humidity));print(f'humidity: {words} differing words',flush=True)
    receipt.write_text(json.dumps(dict(acceptance=False,backend=backend,cases=rows),indent=2)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','engine-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    a=p.parse_args();replay(a.directory,a.receipt,a.backend,a.engine)
