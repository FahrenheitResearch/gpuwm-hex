"""Synthetic whole-module/wrapper replay; CPU witness or shared-math CUDA.

Use --backend host-witness for the strict host execution of the actual kernel
bodies. CUDA needs --device-math-header supplied by the campaign; this script
provides no approximate device math. f1 dump/forecast acceptance uses f1/f2's
real tools when their contract and cases are published.
"""
from pathlib import Path
import argparse
import ctypes as ct
import json
import numpy as np
from hexcore.mpasphys.wsm6_v842 import columns

CONSTANTS=np.array([9.80616,1004.5,1846.4,287,461.6,273.15,np.float32(461.6)/np.float32(287)-np.float32(1),np.float32(287)/np.float32(461.6),0,0,0,1e-15,2.85e6,2.5e6,3.5e5,1.28,1000,4190,2106,610.78,2.49e-6,4.99e-6,9.99e-6,50e-6,125e-6,999e-6],np.float32)


def replay(oracle: Path, receipt: Path, backend='host-witness', header=None, *, case_selection=None):
    # Explicit bounded GPU partitions; the default remains the whole matrix.
    # Generate all inputs in the original RNG order even for unselected cases.
    if case_selection is not None:
        case_selection = frozenset(case_selection)
        if not case_selection or any(type(w) is not bool or type(c) is not int
            or not 0 <= c < 1044 for w, c in case_selection):
            raise ValueError('partition must name native wrapper/case pairs')
    fp=ct.POINTER(ct.c_float);c=CONSTANTS;rows=[];words=0;rng=np.random.default_rng(52173)
    libs=[ct.CDLL(str(oracle.resolve()/name)) for name in ('liboracle.so','libwrapper.so','libhost.so')]
    for wrapper in (False,True):
        nf=libs[int(wrapper)].wsm6_wrapper_oracle if wrapper else libs[0].wsm6_oracle
        hf=libs[2].wsm6_wrapper_host if wrapper else libs[2].wsm6_host
        for f in (nf,hf):f.argtypes=[ct.c_int,ct.c_float,ct.c_int]+[fp]*(7 if wrapper else 6)+([ct.c_int] if wrapper else [])
        for case in range(1044):
            c=CONSTANTS.copy()
            nz=(2,3,8,55)[case%4]
            v=np.zeros((10,nz),np.float32)
            v[0]=rng.uniform(190,312,nz);v[1]=rng.uniform(1e-6,.035,nz)
            v[2:7]=rng.uniform(0,.008,(5,nz))
            category=(case//4)%5
            if category==0:v[2:7]=0
            elif category==1:v[3:7]=0
            elif category==2:v[2:7]*=np.float32(1e-7)
            elif category==3:v[2:7]*=np.float32(.001)
            v[8]=np.linspace(96000,22000,nz,dtype=np.float32)
            v[7]=v[8]/(c[3]*v[0]);v[9]=rng.uniform(80,900,nz)
            pi=np.linspace(.96,.6,nz,dtype=np.float32)
            if wrapper:v[0]/=pi
            hail=(0,1,2,-1)[(case//2)%4];dt=[1.5,30,120,300,60.01,180,240,600][(case//4)%8];radii_on=(case//8)%2
            initial=[v,np.full(7,1.25,np.float32),np.full((3,nz),1.7e-5,np.float32),np.full(nz,-35,np.float32),np.full((2,nz),.0027,np.float32)]
            if 1024 <= case < 1028:
                # Full native scheme with actual binary32 subnormal inputs.
                # A subnormal qmin argument lets native retain subnormal
                # water through arithmetic, instead of flooring it to the
                # normal production minimum before any calculation.
                c[11] = np.finfo(np.float32).smallest_subnormal
                v[1:7] = np.array([0x007fffff], np.uint32).view(np.float32)[0]
                initial[1].fill(np.finfo(np.float32).smallest_subnormal)
                if case >= 1026:
                    dt = float(np.array([0x007fffff], np.uint32).view(np.float32)[0])
            if case >= 1028:
                # Strict native continues for one-level columns and negative
                # timesteps. Cover every measured nonzero-dt boundary regime
                # in the complete module and wrapper replay, including outputs.
                boundary = case - 1028
                nz = (1, 2)[boundary % 2]
                regime = boundary // 4
                dt = (30., -30.)[(boundary // 2) % 2]
                hail, radii_on = 0, 1
                v = np.zeros((10, nz), np.float32)
                v[0] = 280 if regime < 2 else 250
                v[1] = .015
                if regime % 2:
                    v[2:7] = .001
                v[7], v[8], v[9] = 1.1, 90000, 500
                pi = np.full(nz, .97, np.float32)
                if wrapper:
                    v[0] /= pi
                initial = [v, np.zeros(7, np.float32),
                           np.zeros((3, nz), np.float32), np.zeros(nz, np.float32),
                           np.zeros((2, nz), np.float32)]
            if case_selection is not None and (wrapper, case) not in case_selection:
                continue
            outputs=[]
            for candidate in (False,True):
                vv,s,r,d,p=[a.copy() for a in initial]
                if candidate and backend=='hex-cuda':
                    import cupy as cp
                    arrays=[cp.asarray(a[...,None]) for a in (vv,s,r,d,p)]
                    try:
                        columns(*arrays,cp.asarray(c),dt=dt,hail_opt=hail,device_math_header=header,
                                exner=cp.asarray(pi[:,None]) if wrapper else None,
                                radii_on=bool(radii_on) if wrapper else True,production_on=not wrapper)
                    except Exception as error:
                        free, total = cp.cuda.runtime.memGetInfo()
                        raise RuntimeError(f'complete replay failed at wrapper={wrapper}, '
                            f'case={case}, nz={nz}, dt={dt}; GPU bytes free={free}, '
                            f'total={total}: {error}') from error
                    vv,s,r,d,p=[cp.asnumpy(a)[...,0] for a in arrays]
                else:
                    f=hf if candidate else nf
                    pointers=[a.ctypes.data_as(fp) for a in (c,vv,s,r,d,p)]
                    if wrapper:pointers+=[pi.ctypes.data_as(fp),radii_on]
                    f(nz,dt,hail,*pointers)
                outputs.append([vv,s,r,d,p])
            words+=sum(a.size for a in outputs[0])
            diff={name:int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for name,a,b in zip(('volume','surface','radii','dbz','production'),*outputs)}
            rows.append(dict(case=case,nz=nz,wrapper=wrapper,hail=hail,dt=dt,radii_on=radii_on if wrapper else True,subnormal_inputs=1024<=case<1028,native_continuing_boundary=case>=1028,qmin_word=int(c[11:12].view(np.uint32)[0]),differences=diff))
    total=sum(sum(r['differences'].values()) for r in rows)
    receipt.write_text(json.dumps(dict(scope='synthetic complete MMM and MPAS wrapper; not f1 dumps or forecast acceptance' if case_selection is None else 'explicit subset of the synthetic whole MMM/wrapper matrix; not whole matrix or forecast acceptance',backend=backend,process_count=1,whole_matrix=case_selection is None,selected_case_count=len(case_selection) if case_selection is not None else 2088,compared_words=words,rows=rows,total_differing_words=total),indent=2)+'\n')
    print(backend,len(rows),'columns; differing words',total)
    return total

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('oracle',type=Path);parser.add_argument('receipt',type=Path)
    parser.add_argument('--backend',choices=('host-witness','hex-cuda'),default='host-witness')
    parser.add_argument('--device-math-header',type=Path)
    args=parser.parse_args()
    if args.backend=='hex-cuda' and not args.device_math_header:parser.error('hex-cuda requires the shared device math header')
    raise SystemExit(bool(replay(args.oracle,args.receipt,args.backend,args.device_math_header.read_text() if args.device_math_header else None)))
