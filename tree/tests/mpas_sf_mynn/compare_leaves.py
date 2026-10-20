"""Native-private leaf coverage and CUDA/host witness replay, exact words."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
from compare import cuda_artifact, grade


def leaf_data(meta,n=64):
    f=np.float32
    x=np.zeros((11,n),np.float32)
    def vals(seq):return np.resize(seq,n).astype(np.float32)
    name=meta['name']
    unstable='unstable' in name
    v={
        'zolf':vals([0,.001,.25,.5,1,9.999,10,20])*(-1 if unstable else 1),
        'zl':vals([-15,-1,-.001,0,.001,.1,1,10]),
        'z_0':vals([.000001,.00001,.0001,.001,.01,.04,.5,1]),
        'z0':vals([.000001,.00001,.0001,.001,.01,.04,.5,1]),
        'zt':vals([1e-8,1e-7,1e-6,.00001,.0001,.001,.01,.5]),
        'za':vals([.1,1,3,10,20,50,100,200]),
        'ustar':vals([1e-8,.001,.05,.1,.2,.5,1,3]),
        'visc':vals([1e-5,1.3e-5,1.5e-5]),'bvisc':vals([1e-5,1.3e-5,1.5e-5]),
        'restar':vals([.001,.01,.1,.5,2,10,100,1000]),
        'ren':vals([.001,.01,.1,.5,2,10,100,1000]),
        'landsea':vals([2,1,2,1,2,1,2,1]),
        'karman':vals([.4]),'rstoch':vals([-.9,-.5,0,.5,.9]),
        'tstar':vals([-1,-.1,0,.1,1]),'qst':vals([-1,-.1,0,.1,1]),
        'wsp10':vals([0,1,5,10,15,20,40,60]),'zu':vals([2,10,20,50]),
        'rib':vals([-4,-1,-.01,0,.001,.1,.2,4]),
        'ri':vals([-4,-1,-.01,0,.001,.1,.2,4]),
        'ri2':vals([-4,-1,-.01,0,.001,.1,.2,4]),
        'zaz0':vals([1,100,1e3,1e4,1e5,1e6]),
        'z0zt':vals([.01,.5,1,10,100,1000]),
        'z0l':vals([0,.001,.1,1]),
        'zol1':vals([1,-1,.1,-.1,0,20,-20,.001]),
        'zol2':vals([1,-1,.1,-.1,0,20,-20,.001]),
        'logz0':vals([1,2,3,5,8,12]),'logzt':vals([2,4,6,8,10,16]),
    }
    if name=='psi_cb2005':v['zl']=vals([0,.001,.1,1,5,20])
    # sf_mynn.F90:1791-1844 assigns zolri only inside its iteration.
    # Opposite-sign initial bounds can both be clamped to zero by zolri2,
    # skipping that loop and returning uninitialized native storage.
    # sf_mynn.F90:1857-1886: at ri2=0 the wrong-sign clamp is skipped;
    # a negative zol2 would index the stable psi table below its lower bound.
    # Main sf_mynn_run calls zolrib, never this unused private alternative.
    if name=='zolri2':v['zol2'][v['ri2']==0]=0
    if name=='zolri':v['zol1']=np.where(v['ri']<0,-.1,.1).astype(np.float32)
    if name in ('zolri','zolrib'):
        for key,value in {'ri':[.01,.1,.2,.25,.5,1,2,4],
                          'za':[20], 'z0':[.01], 'zt':[.001],
                          'zol1':[.1], 'logz0':[np.log(np.float32(2001))],
                          'logzt':[np.log(np.float32(20010))]}.items():
            v[key][-8:]=np.resize(value,8)
    if name=='zolri':
        # Strict-native witnesses for the secant nmax fallback (:1836-1838).
        witnesses=(
            {'ri':7.534722805023193,'za':3.797328233718872,'z0':6.251169679671875e-7,
             'zt':2.4661672115325928,'zol1':5.301721096038818},
            {'ri':3.654179096221924,'za':5.522488594055176,'z0':1.247787440661341e-5,
             'zt':4.958445072174072,'zol1':.4415208101272583},
            {'ri':.348751425743103,'za':13.46549129486084,'z0':4.226258754730225,
             'zt':1.4605431033487548e-6,'zol1':509.8804931640625},
            {'ri':3.6947708129882812,'za':.01648181676864624,'z0':1.3761817463375792e-8,
             'zt':.01604284904897213,'zol1':21.625835418701172},
        )
        for col,witness in zip((-4,-3,-2,-1),witnesses):
            for key,value in witness.items():v[key][col]=value
    if name=='zolrib':
        # Strict-native gcov witness for its nmax fallback (:1952-1956).
        for key,value in {'ri':4.9457902908325195,'za':58.86027145385742,
                          'z0':.2151898294687271,'zt':1.1772545803978574e-5,
                          'logz0':5.615050315856934,'logzt':15.428556442260742,
                          'zol1':.1666915863752365}.items():v[key][-1]=value
    if name in ('psim_stable','psih_stable'):v['zolf']=abs(v['zolf'])
    if name in ('psim_unstable','psih_unstable'):v['zolf']=-abs(v['zolf'])
    for j,arg in enumerate(meta['float_args']):x[j]=v.get(arg,vals([0]))
    x[9]=vals([0,1])
    return x


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--save',type=Path);p.add_argument('--gpu',action='store_true');args=p.parse_args();root=args.root
    metas=json.loads((root/'leaf-meta.json').read_text())
    inputs=np.stack([leaf_data(m) for m in metas]);n=inputs.shape[2]
    configs=[(psi,k) for psi in (0,1) for k in range(len(metas))]
    if args.gpu:
        from hexcore.mpasphys.sf_mynn_v842 import _module
        import cupy as cp
        saved=np.load(root/'leaf-replay.npz');want=saved['native'];inputs=saved['inputs']
        module=_module()
        # psi_init is a private entry, so the test wrapper exposes it below.
        def init(psi):
            module.get_function('sf_mynn_leaf_tables')((1,),(1,),(np.int32(psi),))
        def port(data,k,psi):
            a=cp.asarray(data);module.get_function('sf_mynn_leaves')(((n+63)//64,),(64,),(a,np.int32(k),np.int32(psi),np.int32(n)))
            return cp.asnumpy(a)
    else:
        pointer=ctypes.POINTER(ctypes.c_float)
        native=ctypes.CDLL(str((root/'leaf-native.so').resolve()));host=ctypes.CDLL(str((root/'leaf-witness.so').resolve()))
        for lib,name in ((native,'native'),(host,'witness')):
            getattr(lib,name+'_leaves').argtypes=[pointer,ctypes.c_int,ctypes.c_int,ctypes.c_int]
            getattr(lib,name+'_leaf_init').argtypes=[ctypes.c_int]
        def call(lib,prefix,data,k,psi):
            a=data.copy();getattr(lib,prefix+'_leaves')(a.ctypes.data_as(pointer),k,psi,n);return a
        native_rows=[]
        for psi,k in configs:
            native.native_leaf_init(psi);native_rows.append(call(native,'native',inputs[k],k,psi))
        want=np.stack(native_rows)
        def init(psi):host.witness_leaf_init(psi)
        def port(data,k,psi):return call(host,'witness',data,k,psi)
    rows=[]
    for j,(psi,k) in enumerate(configs):
        init(psi);m=metas[k];have=port(inputs[k],k,psi)
        labels=m['float_args']+[f'unused_{i}' for i in range(len(m['float_args']),10)]+['result']
        row=grade(want[j],have,labels);row.update(routine=m['name'],psi_opt=psi);rows.append(row)
    init(0)
    receipt={'backend':'hex-cuda' if args.gpu else 'host glibc witness','calls':len(rows),'columns':n,'compared_words':int(want.size),'differing_words':sum(r['differing_words'] for r in rows),'rows':rows}
    if args.gpu:receipt.update(cuda_artifact(root,'leaf-replay.npz'))
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='rows'})
    if args.save:np.savez_compressed(args.save,native=want,inputs=inputs)
    return int(receipt['differing_words']!=0)

if __name__=='__main__':raise SystemExit(main())
