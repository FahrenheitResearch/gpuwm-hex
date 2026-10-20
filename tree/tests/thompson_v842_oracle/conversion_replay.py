"""Word replay of extracted MPAS conversion statements against CUDA source."""
import argparse
import ctypes as C
import json
from pathlib import Path
import numpy as np
from hexcore.mpasphys import thompson_v842 as port
if __package__:
    from .replay import compare, pointer, engine_math_source
else:
    from replay import compare, pointer, engine_math_source


def inputs(nz=16,ncol=19):
    rng=np.random.default_rng(184)
    arrays={}
    for name in port._CONVERSION_ARRAYS:
        shape=(1,ncol) if name in ('d_surface_pressure','d_tend_sfc_pressure') else (nz+1,ncol) if name=='g_zgrid' else (nz,ncol)
        arrays[name]=rng.uniform(0.001,0.01,size=shape).astype(np.float32)
    arrays['g_zgrid'][:]=np.arange(nz+1,dtype=np.float32)[:,None]*100
    arrays['g_zz'][:]=1;arrays['m_rho_zz'][:]=1
    arrays['m_theta_m'][:]=310;arrays['d_rtheta_base'][:]=300
    arrays['d_exner'][:]=1;arrays['d_exner_base'][:]=1
    arrays['d_pressure_base'][:]=90000;arrays['d_pressure_p'][:]=1000
    arrays['d_surface_pressure'][:]=91000
    # Exercise the source MAX(0,qv), including negative transported vapor.
    arrays['m_qv'][::4,::3]=-1e-6
    constants=np.array((287.,461.6,100000.,0.2857143,1.608,9.81,2.49e-6,4.99e-6,9.99e-6),np.float32)
    return arrays,constants


def replay(directory,receipt,backend='host',engine=None):
    native=C.CDLL(str(directory.resolve()/'conversion_oracle.so'))
    host=C.CDLL(str(directory.resolve()/'host_port.so')) if backend=='host' else None
    if host is None:
        import cupy as cp
        maths=engine_math_source(engine.resolve(),native_log10=backend=='engine-native-log10-cuda')
    rows=[]
    for top in (1,7,16):
        a,constants=inputs();b={n:v.copy() for n,v in a.items()};nz,ncol=a['m_qv'].shape
        if host is None:b={n:cp.asarray(v.reshape(-1) if n in ('d_surface_pressure','d_tend_sfc_pressure') else v) for n,v in b.items()}
        for phase in ('from','to'):
            if phase=='to':
                # Supply a changed scheme state rather than a no-op conversion.
                a['f_qv'][:]=np.maximum(a['f_qv'],np.float32(1e-7));a['f_th'][:]+=np.float32(0.25)
                if host is not None:
                    b['f_qv'][:]=np.maximum(b['f_qv'],np.float32(1e-7));b['f_th'][:]+=np.float32(0.25)
                else:
                    b['f_qv'][...]=cp.asarray(a['f_qv']);b['f_th'][...]=cp.asarray(a['f_th'])
            ap=np.array([a[n].ctypes.data for n in port._CONVERSION_ARRAYS],np.uint64)
            function=getattr(native,'conversion_'+phase)
            function.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_float,C.c_void_p]
            function(pointer(ap),nz,ncol,top,10.,pointer(constants))
            if host is not None:
                bp=np.array([b[n].ctypes.data for n in port._CONVERSION_ARRAYS],np.uint64)
                fn=getattr(host,'conversion_'+phase);fn.argtypes=function.argtypes
                fn(pointer(bp),nz,ncol,top,10.,pointer(constants))
                actual=b
            else:
                model={n:b['m_'+n] for n in port.MODEL_ARGUMENTS};mesh={n:b['g_'+n] for n in port.MESH_ARGUMENTS}
                diag={n:b['d_'+n] for n in port.DIAG_ARGUMENTS+('surface_pressure','tend_sfc_pressure')}
                tendencies={n:b['t_'+n] for n in port.TENDENCY_ARGUMENTS}
                fields={n:b['f_'+n] for n in port.FIELD_ARGUMENTS+('rho','z')}
                port.launch_conversion(phase,model,mesh,diag,tendencies,fields,dict(zip(port.CONVERSION_CONSTANTS,constants)),device_math_source=maths,mp_top_level=top,dt_dyn=10.)
                actual={n:cp.asnumpy(v).reshape(a[n].shape) for n,v in b.items()}
            result={n:compare(a[n][None],actual[n][None],(n,))[n] for n in a}
            words=sum(r['words'] for r in result.values())
            rows.append(dict(phase=phase,top=top,arrays=result,differing_words=words))
            print(f'{phase} top={top}: {words} differing words',flush=True)
    receipt.write_text(json.dumps(dict(acceptance=False,backend=backend,cases=rows),indent=2)+'\n')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--backend',choices=('host','engine-cuda','engine-native-log10-cuda'),default='host');p.add_argument('--engine',type=Path)
    a=p.parse_args();replay(a.directory,a.receipt,a.backend,a.engine)
