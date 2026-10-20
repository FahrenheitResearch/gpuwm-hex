"""Coupled native LES RK mixing slice, diagnostic rather than forecast proof."""
import argparse,ctypes,hashlib,itertools,json
from pathlib import Path
import numpy as np
from hexcore.mixing_v842_les import ROUTINES,LesDissipationV842,les_boundary_arguments
from replay_les_v842 import fixtures


def cases():
    spec=les_boundary_arguments()
    prototypes={}
    for routine,index,args in fixtures():
        if index==0:
            for n,v in args.items():
                if n not in prototypes or ROUTINES[routine]['fields'][n]['intent']!='in':
                    prototypes[n]=v.copy(order='F')
    for case,(model,surface,substep,mix) in enumerate(itertools.product([1,2],[0,1,2],[1,2],[0,1])):
        values={n:v.copy(order='F') for n,v in prototypes.items()}
        values['les_model_opt']=np.array(model,dtype='i4');values['les_surface_opt']=np.array(surface,dtype='i4')
        values['dynamics_substep']=np.array(substep,dtype='i4');values['mix_scalars']=np.array(mix,dtype='i4')
        values['config_mix_full']=np.array(0,dtype='i4')
        values['h_mom_eddy_visc4']=np.array([.9],dtype='f4');values['h_theta_eddy_visc4']=np.array([.9],dtype='f4')
        values['v_mom_eddy_visc2']=np.array(.13,dtype='f4');values['v_theta_eddy_visc2']=np.array(.13,dtype='f4')
        yield case,{n:values[n] for n in spec}


def native(library,output):
    lib=ctypes.CDLL(str(Path(library).resolve()));fn=lib.replay_les_boundary;fn.restype=None
    fn.argtypes=[ctypes.c_void_p]*len(les_boundary_arguments());data={};rows=[]
    for case,values in cases():
        fn(*(ctypes.c_void_p(v.ctypes.data) for v in values.values()))
        data.update({f'{case}/{n}':v for n,v in values.items()})
        rows.append({'case':case,'model':int(values['les_model_opt']),
                     'surface':int(values['les_surface_opt']),
                     'substep':int(values['dynamics_substep']),
                     'mix_scalars':int(values['mix_scalars'])})
    np.savez(output,**data)
    Path(str(output)+'.json').write_text(json.dumps({
        'kind':'strict standalone native RK slice; no forecast proof','process_count':1,'MPI_used':False,
        'native_library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        'reference_sha256':hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        'cases':rows},indent=2))
    print('coupled cases: 24')


def cuda(reference,receipt):
    import cupy as cp
    expected=np.load(reference);spec=les_boundary_arguments();launcher=LesDissipationV842();rows=[]
    for case,values in cases():
        device={n:cp.array(v,order='F') if spec[n]['dims'] or spec[n]['type']=='float' and spec[n]['intent']!='in' else v.item() for n,v in values.items()}
        launcher.apply_rk_mixing(1,**device);cp.cuda.get_current_stream().synchronize()
        fields={};words=0
        for n,v in values.items():
            got=cp.asnumpy(device[n]) if isinstance(device[n],cp.ndarray) else v
            want=expected[f'{case}/{n}'];words+=want.size
            bad=int(np.count_nonzero(got.ravel(order='F').view('u4')!=want.ravel(order='F').view('u4')))
            if bad:fields[n]=bad
        # Later RK stages retain all device words; check the real adapter path.
        before={n:v.copy() for n,v in device.items() if isinstance(v,cp.ndarray)}
        launcher.apply_rk_mixing(2,**device);launcher.apply_rk_mixing(3,**device)
        for n,v in before.items():
            if not bool(cp.array_equal(v.view('u4'),device[n].view('u4'))):fields[n+'/held-rk']=1
        rows.append({'case':case,'compared_words':words,'different_words':sum(fields.values()),'fields':fields})
    result={'kind':'coupled native RK mixing slice diagnostic only','whole_forecast_proven':False,'process_count':1,'MPI_used':False,'compiler':'direct NVRTC; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true',
            'different_words':sum(r['different_words'] for r in rows),'cases':rows,
            'reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest()}
    Path(receipt).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return bool(result['different_words'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','cuda']);p.add_argument('input');p.add_argument('output');a=p.parse_args()
    if a.mode=='native':native(a.input,a.output)
    else:raise SystemExit(cuda(a.input,a.output))
