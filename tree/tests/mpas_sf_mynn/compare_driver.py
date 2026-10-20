"""Exact replay of pristine MPAS wrapper and surface-driver conversions."""
import argparse
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from compare import cuda_artifact, constants, grade, synthetic
from hexcore.mpasphys import sf_mynn_v842 as scheme


def inputs():
    fields=scheme.FIELDS; raw=synthetic(fields,64); n=raw.shape[1]
    packed=np.zeros_like(raw)
    for key in (*scheme._DIAG_INPUT,'pblh',*scheme._SURFACE):
        packed[fields.index(key)]=raw[fields.index(key)]
    profiles=np.empty((8,3,n),np.float32)
    for j,key in enumerate(('u1d','v1d','t1d','qv1d',None,'p1d','rho1d','dz8w1d')):
        profiles[j]=raw[fields.index(key)] if key else np.float32(.0001)
    profiles[0,1]=raw[fields.index('u1d2')]
    profiles[1,1]=raw[fields.index('v1d2')]
    profiles[7,1]=raw[fields.index('dz2w1d')]
    density=np.resize([.0625,.25,1,2,4,16],n).astype(np.float32)
    ice=np.resize([-.1,0,.009,.01,.1,.49999997,.5,.9,1,1.0001],n).astype(np.float32)
    sst=np.resize([250,271.39996,271.4,275,290],n).astype(np.float32)
    return packed,profiles,density,ice,sst


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--save',type=Path);p.add_argument('--gpu',action='store_true');p.add_argument('--host',action='store_true');p.add_argument('--slot',action='store_true',help='Replay the f4 callback with native driver parameter switches')
    args=p.parse_args();root=args.root
    data,profiles,density,ice,sst=inputs();c=constants();n=data.shape[1]
    # module_sf_mynn.F:251 reads a local uninitialized qstar on later calls.
    # Only Yang (land option 2) consumes it; test Yang at first-call seed.
    configs=np.asarray([(step,restart,flux,spp,water,land,fraction)
        for step,restart in ((1,0),(8,0),(1,1)) for flux in (0,1)
        for spp in (0,1) for water in range(4) for land in range(4)
        for fraction in (0,1) if land!=2 or (step==1 and restart==0)],np.int32)
    if args.slot:
        configs=configs[(configs[:,2]==1)&(configs[:,3]==0)&(configs[:,4]==0)&(configs[:,5]==0)]
    names=tuple(k for k in scheme.OUTPUTS if k not in ('qstar','wstar'))
    rows_idx=[scheme.FIELDS.index(k) for k in names]
    threshold=np.float32(.01);len_disp=np.float32(480000)
    if args.gpu:
        import cupy as cp
        saved=np.load(root/('driver-slot-replay.npz' if args.slot else 'driver-replay.npz'));want=saved['native'];configs=saved['configs']
        data=saved['inputs'];profiles=saved['profiles'];density=saved['density'];ice=saved['ice'];sst=saved['sst'];c=saved['constants']
        def port(config):
            step,restart,flux,spp,water,land,fraction=map(int,config)
            diagnostics={scheme._DIAG_NAMES.get(k,k):cp.asarray(data[scheme.FIELDS.index(k)]) for k in set(scheme._DIAG_INPUT+names)}
            diagnostics['hpbl']=cp.asarray(data[scheme.FIELDS.index('pblh')])
            surface={v:cp.asarray(data[scheme.FIELDS.index(k)]) for k,v in scheme._SURFACE.items()}
            surface.update(xice=cp.asarray(ice),sst=cp.asarray(sst))
            mesh={'meshDensity':cp.asarray(density)}
            atmosphere={k:cp.asarray(profiles[j]) for j,k in enumerate(scheme._PROFILE)}
            constant_map=dict(zip(scheme.CONSTANTS,c))
            if args.slot:
                prepared={'psfc_p':surface.pop('psfc')}
                if fraction:
                    prepared.update({key+'_sea':cp.empty(n,cp.float32)
                                     for key in scheme._SEA_HANDOFF})
                call=SimpleNamespace(slot='surface_layer',scheme='sf_mynn',backend='hex-cuda',
                    itimestep=step,options={'config_frac_seaice':bool(fraction)},
                    pools={'mesh':mesh,'diag_physics':diagnostics,'sfc_input':surface,
                           'atm_input':atmosphere,'constants':constant_map,'prepared':prepared,
                           'configs':{'config_len_disp':len_disp,'config_do_restart':bool(restart)},
                           'scalar_state':{'xice_threshold':threshold}})
                scheme.launch_slot(call)
            else:
                scheme.driver(mesh,diagnostics,surface,atmosphere,constant_map,
                    itimestep=step,config_do_restart=bool(restart),len_disp=len_disp,
                    xice_threshold=threshold,config_frac_seaice=bool(fraction),
                    isfflx=flux,spp_pbl=spp,isftcflx=water,iz0tlnd=land)
            result=np.stack([cp.asnumpy(diagnostics[scheme._DIAG_NAMES.get(k,k)]) for k in names])
            if args.slot and fraction:
                sea_result=np.stack([cp.asnumpy(prepared[key+'_sea']) for key in scheme._SEA_HANDOFF])
                sea_rows.append(dict(grade(saved['native_sea'][config_index[tuple(config)]],sea_result,scheme._SEA_HANDOFF),compared_words=int(sea_result.size)))
            return result
    else:
        ptr=ctypes.POINTER(ctypes.c_float)
        lib=ctypes.CDLL(str((root/'driver.so').resolve()));lib.native_init()
        lib.native_wrapper.argtypes=[ptr,ptr,ptr]+[ctypes.c_int]*7
        lib.native_from.argtypes=[ptr]*5+[ctypes.c_float,ctypes.c_float,ctypes.c_int,ctypes.c_int]
        lib.native_to.argtypes=[ptr]*4+[ctypes.c_float,ctypes.c_int]
        def pointer(x):return x.ctypes.data_as(ptr)
        native_sea=[]
        def native(config):
            step,restart,flux,spp,water,land,fraction=map(int,config)
            initflag=int(not restart and step<=1)
            a=data.copy();sea=np.empty_like(a);hold=a[scheme.FIELDS.index('regime')].copy()
            lib.native_from(pointer(a),pointer(sea),pointer(density),pointer(sst),pointer(ice),len_disp,threshold,fraction,n)
            lib.native_wrapper(pointer(a),pointer(c),pointer(profiles),initflag,flux,spp,water,land,n,profiles.shape[1])
            if fraction:
                lib.native_wrapper(pointer(sea),pointer(c),pointer(profiles),initflag,flux,spp,water,land,n,profiles.shape[1])
                native_sea.append(sea[[scheme.FIELDS.index(k) for k in scheme._SEA_HANDOFF]].copy())
                lib.native_to(pointer(a),pointer(sea),pointer(ice),pointer(hold),threshold,n)
            else:
                native_sea.append(np.zeros((len(scheme._SEA_HANDOFF),n),np.float32))
            return a[rows_idx]
        want=np.stack([native(config) for config in configs])
        port=None
        if args.host:
            witness=ctypes.CDLL(str((root/'driver-witness.so').resolve()));witness.witness_init()
            witness.witness_driver.argtypes=[ptr]*7+[ctypes.c_int]*8
            def port(config):
                step,restart,flux,spp,water,land,fraction=map(int,config)
                a=data.copy();sea=np.empty_like(a)
                witness.witness_driver(pointer(a),pointer(sea),pointer(profiles),pointer(density),pointer(sst),pointer(ice),pointer(c),int(not restart and step<=1),flux,spp,water,land,fraction,n,profiles.shape[1])
                if args.slot and fraction:
                    sea_result=sea[[scheme.FIELDS.index(k) for k in scheme._SEA_HANDOFF]]
                    sea_rows.append(dict(grade(native_sea[config_index[tuple(config)]],sea_result,scheme._SEA_HANDOFF),compared_words=int(sea_result.size)))
                return a[rows_idx]
    rows=[];sea_rows=[];config_index={tuple(config):j for j,config in enumerate(configs)}
    if port:
        for j,config in enumerate(configs):
            row=grade(want[j],port(config),names);row['config']=list(map(int,config));rows.append(row)
    receipt={'scheme':'sf_mynn','backend':'hex-cuda' if args.gpu else ('host glibc witness' if args.host else 'pristine MPAS wrapper capture'),
        'f4_callback':args.slot,'columns':n,'calls':len(configs),'compared_words':int(want.size) if port else 0,
        'differing_words':sum(r['differing_words'] for r in rows) if port else None,'rows':rows}
    if port and args.slot:
        receipt['sea_handoff']={'rows':sea_rows,'compared_words':sum(r['compared_words'] for r in sea_rows),
                               'differing_words':sum(r['differing_words'] for r in sea_rows)}
    if args.gpu:receipt.update(cuda_artifact(root,'driver-slot-replay.npz' if args.slot else 'driver-replay.npz'))
    args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k!='rows'})
    if args.save:np.savez_compressed(args.save,native=want,native_sea=np.stack(native_sea),inputs=data,profiles=profiles,density=density,ice=ice,sst=sst,constants=c,configs=configs)
    return int(port is not None and (receipt['differing_words']!=0 or receipt.get('sea_handoff',{}).get('differing_words',0)!=0))


if __name__=='__main__':raise SystemExit(main())
