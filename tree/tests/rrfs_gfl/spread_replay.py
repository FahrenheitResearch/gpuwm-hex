"""Native topology and subsidence replay including owned/halo and boundary words."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np


def invoke(fn,values,arrays):
    fn.argtypes=[ctypes.c_int]*len(values)+[ctypes.c_void_p]*len(arrays);fn.restype=None
    fn(*values,*[x.ctypes.data for x in arrays])


def capture(root,path):
    lib=ctypes.CDLL(str(root.resolve()/'gfl_native.so'));nc=37;nz=11;me=6;me2=36
    ne=np.full(nc,me,dtype='i4');coc=np.array([[(c+d)%nc+1 for d in (-3,-2,-1,1,2,3)] for c in range(nc)],dtype='i4')
    # Irregular receiving degrees catch the native receiving-cell denominator.
    ne[::7]=5
    mask=np.arange(nc,dtype='i4')%8
    sigma=np.linspace(0.,.9,nc,dtype='f4');sigma[::5]=0
    far=sigma*np.float32(.35);rng=np.random.default_rng(113)
    sub=rng.normal(0.,.001,(4,nc,nz)).astype('f4');sub[0,0,0]=np.float32(2e-40)
    tend=rng.normal(0.,.01,(4,nc,nz)).astype('f4');tend[0,0,0]=np.float32(1e-40)
    fixture=dict(ne=ne,coc=coc,mask=mask,sigma=sigma,far=far,sub=sub,tend=tend)
    for owned in (1,19,nc):
        n2=np.full(nc,-17,dtype='i4');coc2=np.full((nc,me2),-23,dtype='i4')
        invoke(lib.gfl_topology_native,(nc,owned,me,me2),(ne,coc,n2,coc2))
        fixture[f'n2_{owned}']=n2;fixture[f'coc2_{owned}']=coc2
        for option in (0,1):
            out=tend.copy()
            invoke(lib.gfl_spread_native,(nc,owned,nz,me,me2,option),
                   (mask,ne,coc,n2,coc2,sigma,far,*[sub[i] for i in range(4)],*[out[i] for i in range(4)]))
            fixture[f'out_{owned}_{option}']=out
    np.savez_compressed(path,**fixture)


def replay(path,receipt):
    import cupy as cp
    from hexcore.mpasphys.rrfs.cu_gfl_bf6a4b0 import cells_on_cells_on_cell, spread_subsidence
    z=np.load(path);nc,nz=z['sub'].shape[1:];me=z['coc'].shape[1];me2=z['coc2_1'].shape[1];checks=[]
    for owned in (1,19,nc):
        ne=cp.asarray(z['ne']);coc=cp.asarray(z['coc']);n2=cp.full(nc,-17,dtype=cp.int32);coc2=cp.full((nc,me2),-23,dtype=cp.int32)
        mesh={'nEdgesOnCell':ne,'cellsOnCell':coc,'nCellsOnCellsOnCell':n2,'cellsOnCellsOnCell':coc2,'bdyMaskCell':cp.asarray(z['mask'])}
        cells_on_cells_on_cell(mesh,nCellsSolve=owned)
        checks.append({'owned':owned,'kind':'topology','different_words':int(np.count_nonzero(cp.asnumpy(n2)!=z[f'n2_{owned}']))+int(np.count_nonzero(cp.asnumpy(coc2)!=z[f'coc2_{owned}']))})
        for option in (0,1):
            calls=[];out=cp.asarray(z['tend']);sub=cp.asarray(z['sub'])
            spread_subsidence(out,sub,sigma_deep=cp.asarray(z['sigma']),sigma_deep_far=cp.asarray(z['far']),
                mesh=mesh,nCellsSolve=owned,config_gfl_sub3d=option,exchange_halo_group=calls.append)
            diff=int(np.count_nonzero(cp.asnumpy(out).view('u4')!=z[f'out_{owned}_{option}'].view('u4')))
            assert calls==(['physics:sub3d_cuten'] if option else [])
            checks.append({'owned':owned,'sub3d':option,'kind':'spreading','different_words':diff,'halo_calls':calls})
    result={'scope':'standalone fork driver spreading and topology; partition ownership simulated, not MPI forecast',
            'checks':checks,'total_different_words':sum(x['different_words'] for x in checks)}
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return bool(result['total_different_words'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--receipt',type=Path);p.add_argument('--gpu',action='store_true');a=p.parse_args()
    if a.gpu:raise SystemExit(replay(a.fixture,a.receipt))
    capture(a.root,a.fixture)
