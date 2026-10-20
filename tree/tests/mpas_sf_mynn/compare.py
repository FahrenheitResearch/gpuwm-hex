"""Standalone synthetic full-module oracle and host-libm witness comparison."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np


def synthetic(fields, n=256):
    rng=np.random.default_rng(812)
    out=np.zeros((len(fields),n),np.float32)
    def setv(key,value): out[fields.index(key)]=value
    values={
        'u1d':rng.uniform(-25,25,n),'v1d':rng.uniform(-10,10,n),
        'u1d2':rng.uniform(-25,25,n),'v1d2':rng.uniform(-10,10,n),
        't1d':rng.uniform(235,315,n),'qv1d':rng.uniform(0.00001,.025,n),
        'p1d':rng.uniform(65000,103000,n),'psfcpa':rng.uniform(65000,103000,n),
        'dz8w1d':np.resize([.1,1,2,3,6,10,14,20,25.999,26,40,200],n),
        'dz2w1d':np.resize([.001,1,5,10,12,15,25,100],n),
        'rho1d':rng.uniform(.6,1.3,n),'pblh':rng.uniform(0,6000,n),
        'znt':rng.uniform(.00001,.8,n),'ust':rng.uniform(.001,2,n),
        'mavail':rng.uniform(0,1,n),'mol':rng.uniform(-3,3,n),
        'xland':np.resize([1,2,1.5,1.4999999,1.5000001],n),
        'hfx':rng.uniform(-250,600,n),'qfx':rng.uniform(-.0001,.0005,n),
        'tsk':rng.uniform(235,315,n),'qsfc':rng.uniform(0,.03,n),
        'snowh':np.resize([0,.09999999,.1,1],n),'dx':np.resize([3000,5000,10000,480000],n),
        'qstar':rng.uniform(-1,1,n),'ustm':rng.uniform(.001,2,n),
        'rstoch1d':rng.uniform(-.9,.9,n),
    }
    for k,v in values.items():setv(k,v)
    # Land neutral, weak stable, free convection and denominator edges.
    for i in range(0,n,16):
        out[fields.index('xland'),i]=1
        out[fields.index('tsk'),i]=out[fields.index('t1d'),i]
        out[fields.index('psfcpa'),i]=out[fields.index('p1d'),i]
        q=out[fields.index('qv1d'),i]
        out[fields.index('qsfc'),i]=q/(np.float32(1)+q)
    out[fields.index('qsfc'),1::17]=0
    # Small positive roughness keeps the sub-2m geometry logarithms defined.
    short=out[fields.index('dz8w1d')]<4
    out[fields.index('znt'),short]=np.float32(.00001)
    # Defined algebraic stress column: saturation exceeds surface pressure.
    # This reaches :359's second operand after :318 has recomputed qsfc.
    out[fields.index('xland'),2]=np.float32(1)
    out[fields.index('tsk'),2]=np.float32(450)
    out[fields.index('psfcpa'),2]=np.float32(100000)
    out[fields.index('qsfc'),2]=np.float32(0)
    return out


def constants():
    # mpas_atmphys_constants.F:29-59, supplied to the module by its driver.
    f=np.float32
    cp=f(1004.5);r=f(287.);rv=f(461.6)
    return np.asarray([cp,f(9.80616),f(r/cp),r,f(2.5e6),f(.6112),f(17.67),f(29.65),f(273.15),f(rv/r-f(1)),f(r/rv),f(.4)],np.float32)


def library(path,prefix):
    lib=ctypes.CDLL(str(path.resolve()))
    getattr(lib,prefix+'_init')()
    if prefix=='native' and hasattr(lib,'native_status'):
        lib.native_status()
    run=getattr(lib,prefix+'_columns')
    ptr=ctypes.POINTER(ctypes.c_float)
    run.argtypes=[ptr,ptr]+[ctypes.c_int]*7
    def call(data,c,step,flux,spp,water,land,present):
        result=data.copy()
        run(result.ctypes.data_as(ptr),c.ctypes.data_as(ptr),step,flux,spp,water,land,present,result.shape[1])
        return result
    return call


def grade(want,have,fields):
    diff=want.view(np.uint32)!=have.view(np.uint32)
    counts={k:int(diff[i].sum()) for i,k in enumerate(fields) if diff[i].any()}
    samples=[]
    for row,col in np.argwhere(diff)[:8]:
        samples.append({'field':fields[row],'column':int(col),'native':float(want[row,col]),'port':float(have[row,col]),'native_word':int(want.view(np.uint32)[row,col]),'port_word':int(have.view(np.uint32)[row,col])})
    return {'differing_words':int(diff.sum()),'per_array':counts,'samples':samples}


def cuda_artifact(root, capture):
    """Pin the actual assembled kernel source and saved native replay input."""
    from hexcore.mpasphys.sf_mynn_v842 import _cuda_source
    from importlib.resources import files
    kernels=files('gpuwm').joinpath('core/kernels')
    return {'cuda_source_sha256':hashlib.sha256(_cuda_source().encode()).hexdigest(),
            'native_capture_sha256':hashlib.sha256((root/capture).read_bytes()).hexdigest(),
            'shared_header_sha256':{name:hashlib.sha256(kernels.joinpath(name).read_bytes()).hexdigest()
                for name in ('mynn_libm.cuh','glibc_trig_flt32.cuh')}}


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--gpu',action='store_true');p.add_argument('--save',type=Path)
    p.add_argument('--native-library',default='native.so');p.add_argument('--host-library',default='witness.so')
    args=p.parse_args();root=args.root
    meta=json.loads((root/'metadata.json').read_text())
    fields=meta['fields']; data=synthetic(fields); c=constants()
    if args.gpu:
        from hexcore.mpasphys.sf_mynn_v842 import run_packed
        import cupy as cp
        saved=np.load(root/'oracle-replay.npz')
        data=saved['inputs'];c=saved['constants'];configs=saved['configs'];want=saved['native']
        def port(x,c,step,flux,spp,water,land,present):
            v=cp.asarray(x);run_packed(v,cp.asarray(c),itimestep=step,isfflx=flux,spp_pbl=spp,isftcflx=water,iz0tlnd=land,present=present)
            return cp.asnumpy(v)
    else:
        native=library(root/args.native_library,'native'); port=library(root/args.host_library,'witness')
        configs=np.asarray([(step,flux,spp,water,land,127) for step in (1,0,2) for flux in (0,1) for spp in (0,1) for water in range(4) for land in range(4)]+[(2,flux,0,0,0,mask) for flux in (0,1) for mask in range(127)]+[(2,1,0,0,0,255)],np.int32)
        native_rows=[]
        for config in configs:
            print('native', list(map(int,config)), flush=True)
            native_rows.append(native(data,c,*map(int,config)))
        want=np.stack(native_rows)
    rows=[]
    for ix,config in enumerate(configs):
        print('port', list(map(int,config)), flush=True)
        have=port(data,c,*map(int,config))
        result=grade(want[ix],have,fields);result['config']=list(map(int,config));rows.append(result)
    receipt={'scheme':'sf_mynn','reference':'pristine MPAS v8.4.2 sf_mynn_run GNU strict, single precision','backend':'hex-cuda' if args.gpu else 'generated C++ host glibc witness','columns':data.shape[1],'calls':len(rows),'compared_words':int(want.size),'differing_words':sum(r['differing_words'] for r in rows),'rows':rows}
    if args.gpu:receipt.update(cuda_artifact(root,'oracle-replay.npz'))
    args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}))
    if args.save:
        np.savez_compressed(args.save,inputs=data,constants=c,configs=configs,native=want)
    return int(receipt['differing_words']!=0)

if __name__=='__main__':raise SystemExit(main())
