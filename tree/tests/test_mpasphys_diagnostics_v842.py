"""Full output/EOS conversion and shared trig vs strict native words."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('MPAS_PHYSICS_GPU_TESTS')!='1',
                            reason='explicit allocated GPU required')


def test_complete_output_and_cold_eos_against_native():
    import cupy as cp
    from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
    from hexcore.mpasphys.diagnostics_v842 import initialize_thermodynamics, output_diagnostics
    lib=ctypes.CDLL(str(Path(os.environ['MPAS_DIAGNOSTICS_REFERENCE'])/'libdiagnostics.so'))
    lib.cold.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*7
    lib.output.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9
    rng=np.random.default_rng(442)
    shape=(55,2562);count=np.prod(shape).item()
    def host(lo,hi):return rng.uniform(lo,hi,shape).astype(np.float32)
    zz,base,pert=host(.8,1.2),host(10,400),host(-2,2)
    theta_m,rho=host(230,350),host(.1,1.5)
    theta_m.ravel()[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)),-1.e-45]
    rho.ravel()[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)),-1.e-45]
    qv=host(-.005,.03)
    qv.ravel()[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)),-.001]
    expected=[np.empty(shape,np.float32) for _ in range(4)]
    lib.cold(count,*[a.ctypes.data for a in (zz,pert,base,*expected)])
    atmosphere=SimpleNamespace(state=SimpleNamespace(rho=cp.asarray(rho),scalars=cp.asarray(qv[None])),
        saved=SimpleNamespace(theta_m=cp.asarray(theta_m),rho_theta_perturbation=cp.asarray(pert),
            exner=cp.empty(shape,np.float32),pressure_perturbation=cp.empty(shape,np.float32)),
        reference=SimpleNamespace(rho_theta_base=cp.asarray(base),exner_base=cp.empty(shape,np.float32),
            pressure_base=cp.empty(shape,np.float32)),vertical=SimpleNamespace(zz=cp.asarray(zz)))
    cache=MpasPhysicsKernelCache()
    initialize_thermodynamics(atmosphere,cache)
    actual=(atmosphere.saved.exner,atmosphere.reference.exner_base,
            atmosphere.saved.pressure_perturbation,atmosphere.reference.pressure_base)
    for got,want in zip(actual,expected):
        np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),want.view(np.uint32))
    outputs=[np.empty(shape,np.float32) for _ in range(3)]
    lib.output(count,*[a.ctypes.data for a in (theta_m,qv,rho,zz,expected[3],expected[2],*outputs)])
    for got,want in zip(output_diagnostics(atmosphere,['qv'],cache,cp),outputs):
        np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),want.view(np.uint32))


def test_shared_trig_matches_native_gnu_functions():
    import cupy as cp
    from hexcore.mpasphys.math_v842 import bind_common_carrier_math
    from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
    values=np.linspace(-np.pi,np.pi,65537,dtype=np.float32)
    values[:4]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1)),np.float32(1.5707964)]
    lib=ctypes.CDLL(ctypes.util.find_library('m'))
    expected=[]
    for name in ('sinf','cosf'):
        f=getattr(lib,name);f.argtypes=[ctypes.c_float];f.restype=ctypes.c_float
        expected.append(np.asarray([f(float(x)) for x in values],np.float32))
    source=bind_common_carrier_math('''extern "C" __global__ void trig(const int n,const float *x,float *s,float *c) {
        const int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n){s[i]=sinf(x[i]);c[i]=cosf(x[i]);}
    }''')
    result=[cp.empty(values.shape,np.float32) for _ in range(2)]
    kernel=MpasPhysicsKernelCache().raw_kernel('trig',source,module_key='hexcore.mpasphys.trig_witness_v842')
    kernel(((values.size+127)//128,),(128,),(np.int32(values.size),cp.asarray(values),*result))
    for got,want in zip(result,expected):
        np.testing.assert_array_equal(cp.asnumpy(got).view(np.uint32),want.view(np.uint32))


def test_complete_output_on_published_initial_columns():
    import cupy as cp
    from hexcore.mpasphys.compile_v842 import MpasPhysicsKernelCache
    from hexcore.mpasphys.diagnostics_v842 import output_diagnostics
    work=Path(__file__).resolve().parents[3]
    folder=work/'scratch/real-columns/decoded'
    shape=json.loads((folder/'shape.json').read_text())
    nl,nc=shape['n_levels'],shape['n_cells']
    def read(name):
        return np.asfortranarray(np.frombuffer((folder/(name+'.bin')).read_bytes(),dtype='<f4').reshape(nc,nl).T)
    rho,zz,dry_theta,qv=(read(name) for name in ('rho','zz','theta','qv'))
    mass,theta_m=(np.empty((nl,nc),np.float32,order='F') for _ in range(2))
    init=ctypes.CDLL(str(work/'scratch/native-driver/libinitial_carriers.so'))
    init.initialize_carriers.argtypes=[ctypes.c_int]*2+[ctypes.c_void_p]*6
    init.initialize_carriers(nl,nc,*[a.ctypes.data for a in (rho,zz,dry_theta,qv,mass,theta_m)])
    # Pressure inputs remain synthetic: this is an output-boundary replay,
    # not captured full-model EOS or scheme arithmetic.
    base=np.full((nl,nc),1.e5,np.float32,order='F')
    perturbation=np.zeros_like(base)
    inputs=[np.ascontiguousarray(a) for a in (theta_m,qv,mass,zz,base,perturbation)]
    expected=[np.empty((nl,nc),np.float32) for _ in range(3)]
    lib=ctypes.CDLL(str(Path(os.environ['MPAS_DIAGNOSTICS_REFERENCE'])/'libdiagnostics.so'))
    lib.output.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9
    lib.output(nl*nc,*[a.ctypes.data for a in (*inputs,*expected)])
    atmosphere=SimpleNamespace(state=SimpleNamespace(rho=cp.asarray(inputs[2]),scalars=cp.asarray(inputs[1][None])),
        saved=SimpleNamespace(theta_m=cp.asarray(inputs[0]),pressure_perturbation=cp.asarray(inputs[5])),
        reference=SimpleNamespace(pressure_base=cp.asarray(inputs[4])),vertical=SimpleNamespace(zz=cp.asarray(inputs[3])))
    for actual,want in zip(output_diagnostics(atmosphere,['qv'],MpasPhysicsKernelCache(),cp),expected):
        np.testing.assert_array_equal(cp.asnumpy(actual).view(np.uint32),want.view(np.uint32))
