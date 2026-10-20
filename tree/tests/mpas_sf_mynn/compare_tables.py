"""Word oracle for every native psi_init table entry (sf_mynn.F90:1967-2004)."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
from compare import cuda_artifact, grade

TABLES=('psim_stab','psih_stab','psim_unstab','psih_unstab')
ROUTINES=('psim_stable_full','psih_stable_full','psim_unstable_full','psih_unstable_full')


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--save',type=Path)
    p.add_argument('--gpu',action='store_true');args=p.parse_args();root=args.root
    n=1001
    if args.gpu:
        import cupy as cp
        from hexcore.mpasphys.sf_mynn_v842 import _module
        want=np.load(root/'table-replay.npz')['native']
        module=_module();have=[]
        for mode in (0,1):
            module.get_function('sf_mynn_leaf_tables')((1,),(1,),(np.int32(mode),))
            have.append(np.stack([cp.asnumpy(cp.ndarray((n,),dtype=cp.float32,
                memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(
                    module.get_global_var(name), n * 4, module), 0))) for name in TABLES]))
        module.get_function('sf_mynn_leaf_tables')((1,),(1,),(np.int32(0),))
        have=np.stack(have)
    else:
        metas=json.loads((root/'leaf-meta.json').read_text())
        ptr=ctypes.POINTER(ctypes.c_float)
        native=ctypes.CDLL(str((root/'leaf-native.so').resolve()))
        host=ctypes.CDLL(str((root/'leaf-witness.so').resolve()))
        for lib,prefix in ((native,'native'),(host,'witness')):
            getattr(lib,prefix+'_leaf_init').argtypes=[ctypes.c_int]
            getattr(lib,prefix+'_leaves').argtypes=[ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        def capture(lib,prefix):
            result=[]
            for mode in (0,1):
                getattr(lib,prefix+'_leaf_init')(mode);rows=[]
                for j,routine in enumerate(ROUTINES):
                    name=routine+('_gfs' if mode else '')
                    k=next(k for k,m in enumerate(metas) if m['name']==name)
                    data=np.zeros((11,n),np.float32)
                    grid=np.arange(n,dtype=np.float32)
                    if j>=2:grid=-grid
                    data[0]=grid*np.float32(.01)
                    getattr(lib,prefix+'_leaves')(data.ctypes.data_as(ptr),k,mode,n)
                    rows.append(data[10])
                result.append(np.stack(rows))
            return np.stack(result)
        want=capture(native,'native');have=capture(host,'witness')
    rows=[]
    for mode in (0,1):
        row=grade(want[mode],have[mode],TABLES);row['psi_opt']=mode;rows.append(row)
    receipt={'backend':'hex-cuda' if args.gpu else 'host glibc witness',
        'compared_words':int(want.size),'differing_words':sum(r['differing_words'] for r in rows),
        'table_entries':n,'rows':rows}
    if args.gpu:receipt.update(cuda_artifact(root,'table-replay.npz'))
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='rows'})
    if args.save:np.savez_compressed(args.save,native=want)
    return int(receipt['differing_words']!=0)

if __name__=='__main__':raise SystemExit(main())
