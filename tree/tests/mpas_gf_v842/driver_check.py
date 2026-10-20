"""Driver conversion oracle: scalar geometry and CUDA pool layouts."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np


def surface_inputs():
    rng = np.random.default_rng(17492)
    n = 2051
    return np.array([
        rng.uniform(-200., 3500., n), rng.uniform(-300., 600., n),
        rng.uniform(-.001, .001, n), rng.choice([1., 2.], n),
        rng.uniform(0., 1000., n), np.exp(rng.uniform(-9., 9., n)),
        rng.uniform(1.e6, 2.e11, n),
    ], dtype='f4')


def invoke(lib, name, a, length, dt):
    out = np.empty((a.shape[1], 8), dtype='f4')
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*7 + [ctypes.c_float]*2 + [ctypes.c_void_p]
    fn(a.shape[1], *[ctypes.c_void_p(x.ctypes.data) for x in a],
       length, dt, ctypes.c_void_p(out.ctypes.data))
    return out


def check(root, receipt, fixture, gpu):
    if not gpu:
        a = surface_inputs();length=np.float32(194000.);dt=np.float32(120.)
        native = ctypes.CDLL(str(root/'gf_native.so'))
        host = ctypes.CDLL(str(root/'gf_host.so'))
        want = invoke(native, 'gf_native_surface', a, length, dt)
        have = invoke(host, 'gf_pack_surface_host', a, length, dt)
        np.savez_compressed(fixture, a=a, want=want, length=length, dt=dt)
        extras = {}
    else:
        import cupy as cp
        from hexcore.mpasphys.gf_mpas_v842 import X_from_MPAS, X_to_MPAS, LEVEL_INPUTS, LEVEL_OUTPUTS, SURFACE_OUTPUTS, INTEGER_OUTPUTS
        with np.load(fixture) as data:
            a,want,length,dt=(data[name] for name in ('a','want','length','dt'))
        n=a.shape[1];nz=55
        # Deliberately strided vectors and level arrays expose layout mistakes.
        fields = {}
        for index,name in enumerate(LEVEL_INPUTS):
            fields[name] = cp.arange(nz*n*2,dtype=cp.float32).reshape(nz,n*2)[:,::2]+index
        for source,target in (('rthften','rthdynten'),('rqvften','rqvdynten')):
            fields[target]=fields[source]
        fields['rthratenlw']=fields['rthraten']
        fields['rthratensw']=cp.full((nz,n),np.float32(.125),dtype=cp.float32)
        for index,name in enumerate(('ht','hfx','qfx','xland','gsw','meshDensity','areaCell')):
            vector=cp.empty(n*2,dtype=cp.float32);vector[::2]=cp.asarray(a[index]);fields[name]=vector[::2]
        fields['zgrid']=cp.stack((fields['ht'],fields['ht']+1.))
        fields['kpbl']=cp.full(n,8,dtype=cp.int32)
        levels,surface,kpbl=X_from_MPAS(fields,len_disp=length,dt=dt)
        have=cp.asnumpy(surface)
        packing_words=0
        for index,name in enumerate(LEVEL_INPUTS):
            expected=fields[name]+np.float32(.125) if name=='rthraten' else fields[name]
            packing_words+=int(cp.count_nonzero(levels[:,index,:].T.view(cp.uint32)!=expected.view(cp.uint32)).item())
        levels_out=cp.arange(n*6*nz,dtype=cp.float32).reshape(n,6,nz)
        surface_out=cp.arange(n*6,dtype=cp.float32).reshape(n,6)
        integer_out=cp.arange(n*3,dtype=cp.int32).reshape(n,3)
        output=X_to_MPAS({'levels':levels_out,'surface':surface_out,'integer':integer_out})
        aliases={'gdc':'qc_cu','gdc2':'qi_cu','htop':'cutop','hbot':'cubot','pratec':'cuprec'}
        for source,names,transpose in ((levels_out,LEVEL_OUTPUTS,True),(surface_out,SURFACE_OUTPUTS,False),(integer_out,INTEGER_OUTPUTS,False)):
            for index,name in enumerate(names):
                expected=source[:,index].T if transpose else source[:,index]
                packing_words+=int(cp.count_nonzero(output[aliases.get(name,name)].view(cp.uint32)!=expected.view(cp.uint32)).item())
        packing_words+=int(cp.count_nonzero(kpbl!=8).item())
        extras={'packing_different_words':packing_words}
    count=int(np.count_nonzero(want.view('u4')!=have.view('u4')))
    report={'backend':'hex-cuda' if gpu else 'host-glibc-witness','surface_columns':len(want),'geometry_different_words':count,'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),**extras,'scope':'standalone MPAS driver slice; no CLI or forecast qualification'}
    receipt.write_text(json.dumps(report,indent=2)+'\n');print(receipt.read_text())
    return bool(count+extras.get('packing_different_words',0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--gpu',action='store_true');args=parser.parse_args()
    raise SystemExit(check(args.root.resolve(),args.receipt,args.fixture,args.gpu))
