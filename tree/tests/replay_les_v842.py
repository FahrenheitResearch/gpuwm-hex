"""Strict pristine-MPAS replay; run native on a CPU host and CUDA on one GPU.

This is a diagnostic module test, not the six-hour forecast acceptance.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from hexcore.mixing_v842_les import ROUTINES, LesDissipationV842, _extent


def fixtures():
    for routine, spec in ROUTINES.items():
        if routine == "calculate_n2":
            choices = itertools.product([0, 2], [0, 1])
        elif routine == "les_models":
            choices = itertools.product([0, 1, 2], [1, 2], [0, 1])
        else:
            choices = itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1], [1, 2], [0, 1])
        for case, choice in enumerate(choices):
            rng = np.random.default_rng(981+case)
            a = {"nvertlevels":5,"maxedges":2,"num_scalars":3,"ncells":3,"nedges":3,"nvertices":3,"vertexdegree":2,
                 "cellstart":1,"cellend":3,"cellsolvestart":2,"cellsolveend":3,
                 "edgestart":1,"edgeend":3,"edgesolvestart":2,"edgesolveend":3,"vertexstart":1,"vertexend":3,
                 "index_tke":3,"index_qv":1,"index_qc":2,"num_scalars_dummy":3,
                 "c_s":.125,"config_len_disp":100.,"invdt":.1,"config_visc4_2dsmag":.05,
                 "prandtl_inv":1.,"h_mom_eddy_visc4":.07,"h_theta_eddy_visc4":.07,
                 "v_mom_eddy_visc2":.13,"v_theta_eddy_visc2":.13,"config_del4u_div_factor":10.,
                 "config_surface_drag_coefficient":.0013,"config_surface_heat_flux":.03,
                 "config_surface_moisture_flux":.0002,"dt":10.,"les_model_opt":2,"les_surface_opt":1,
                 "dynamics_substep":1,"config_mix_full":1,"mix_scalars":1}
            if routine == "calculate_n2":a['index_qc']=choice[0]
            elif routine == "les_models":a.update(les_model_opt=choice[0],dynamics_substep=choice[1])
            else:
                model,surface,full,fourth,substep,mix=choice
                a.update(les_model_opt=model,les_surface_opt=surface,config_mix_full=full,dynamics_substep=substep,mix_scalars=mix,
                         h_mom_eddy_visc4=.07 if fourth else 0.,h_theta_eddy_visc4=.07 if fourth else 0.,
                         v_mom_eddy_visc2=.13 if fourth else 0.,v_theta_eddy_visc2=.13 if fourth else 0.)
            for name in spec['args']:
                v=spec['fields'][name]
                if v['dims']:
                    dims=tuple(a['ncells']+1 if d==':' else _extent(d,a) for d in v['dims'])
                    if v['type']=='int':value=np.ones(dims,dtype=np.int32,order='F')
                    else:value=np.asfortranarray(rng.uniform(-1,1,dims).astype('f4'))
                    a[name]=value
                elif v['type']=='float' and v['intent']!='in':a[name]=np.array([.913],dtype='f4')
                else:a[name]=np.array(a[name],dtype='f4' if v['type']=='float' else 'i4')
            for name in spec['args']:
                v=a[name]
                if name=='nedgesoncell':v[:]=2
                if name in ('edgesoncell','edgesonvertex'):v[:]=[[1,1,2,1],[3,2,3,2]]
                if name=='cellsonedge':v[:]=[[1,2,1,1],[2,3,3,1]]
                if name=='verticesonedge':v[:]=[[1,2,1,1],[2,3,3,1]]
                if name.endswith('_sign'):v[:]=[[1.,-1.,-1.,1.],[-1.,1.,1.,-1.]]
                if name=='zgrid':v[:]=np.arange(6,dtype='f4')[:,None]*np.array([10.,23.,17.,21.],dtype='f4')[None,:]
                if name in ('rho_zz','rho_edge','zz','eddy_visc_horz','eddy_visc_vert','prandtl_3d_inv'):v[:]=np.abs(v)+np.float32(.2)
                if name in ('rdzu','rdzw'):v[:]=np.float32(.05)
                if name in ('fzm','fzp'):v[:]=np.float32(.5)
                if name in ('dcEdge'.lower(),'dvedge'):v[:]=np.float32(100)
                if name in ('invdcedge','invDvedge'.lower()):v[:]=np.float32(.01)
                if name in ('invareacell','invareatriangle'):v[:]=np.float32(.0001)
                if name in ('meshscalingdel2','meshscalingdel4'):v[:]=np.float32(.8)
                if name.startswith('deformation_coef'):v[:]*=np.float32(.02)
                if name=='theta_m':v[:]=np.float32(300)+v*np.float32(3)
                if name=='exner':v[:]=np.float32(.95)+v*np.float32(.01)
                if name=='pressure_b':v[:]=np.float32(90000)+v*np.float32(1000)
                if name=='scalars':
                    v[0]=np.float32(.01)+np.abs(v[0])*np.float32(.005)
                    v[1]=np.float32(.0001)*np.abs(v[1])
                    v[2]=np.array([-2.,0.,.003,20.,400.],dtype='f4')[:,None]
                if name=='bv_freq2':v[:]=np.array([-.01,0.,.000001,.000002,.02],dtype='f4')[:,None]
                if name=='qtot':v[:]=np.float32(.01)+np.abs(v)*np.float32(.005)
            if routine=='u_dissipation_3d' and a['les_surface_opt'].item()==2:
                a['u'][0,:]=np.float32(0);a['v'][0,:]=np.float32(0)
            # Separate buoyancy branches and pressure<esw correction.
            if routine=='calculate_n2':
                a['scalars'][1,:]=np.array([0.,.000009,.00001,.001,0.],dtype='f4')[:,None]
                if choice[1]:a['pressure_b'][:]=np.float32(1000)
            if routine=='les_models' and choice[2]:
                a['u'][:]*=np.float32(1000);a['v'][:]*=np.float32(1000)
            yield routine,case,{n:a[n] for n in spec['args']}


def subnormal_fixtures():
    """Native u tendency += zero must preserve both signed subnormal inputs.

    Actual routine: mpas_atm_dissipation_models.F:579-947. The zero stress
    operands remain runtime buffers, so this checks compiled arithmetic FTZ.
    """
    _,_,prototype=next(row for row in fixtures() if row[0]=='u_dissipation_3d')
    for case,bits in enumerate([0x100,0x80000100],start=144):
        values={n:v.copy(order='F') for n,v in prototype.items()}
        for name in ['u','v','ucell','vcell','divergence','vorticity',
                     'eddy_visc_horz','eddy_visc_vert','u_init','v_init']:
            if name in values:values[name].fill(0)
        for name in ['h_mom_eddy_visc4','v_mom_eddy_visc2']:
            values[name].fill(0)
        values['les_model_opt'].fill(0);values['les_surface_opt'].fill(0)
        values['config_mix_full'].fill(1)
        values['tend_u_euler'].view('u4').fill(bits)
        yield 'u_dissipation_3d',case,values


def all_fixtures():
    yield from fixtures()
    yield from subnormal_fixtures()


def native(library, output):
    lib=ctypes.CDLL(str(Path(library).resolve())); data={};rows=[]
    for routine,case,a in all_fixtures():
        fn=getattr(lib,'replay_'+routine);fn.restype=None
        fn.argtypes=[ctypes.c_void_p]*len(a)
        fn(*(ctypes.c_void_p(v.ctypes.data) for v in a.values()))
        for n,v in a.items():data[f'{routine}/{case}/{n}']=v.copy(order='F')
        rows.append({'routine':routine,'case':case})
    np.savez(output,**data)
    Path(str(output)+'.json').write_text(json.dumps({'kind':'standalone diagnostic only','process_count':1,'MPI_used':False,'subnormal_cases':2,'cases':rows,'native_library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest()},indent=2))
    print(f'native cases: {len(rows)}')


def cuda(reference, receipt):
    import cupy as cp
    expected=np.load(reference);launcher=LesDissipationV842();rows=[];total=0;failures=[]
    for routine,case,a in all_fixtures():
        spec=ROUTINES[routine];device={}
        for n,v in a.items():
            field=spec['fields'][n]
            if field['dims'] or field['type']=='float' and field['intent']!='in':device[n]=cp.array(v,order='F')
            else:device[n]=v.item()
        launcher.call(routine,**device)
        cp.cuda.get_current_stream().synchronize()
        fields={};words=0
        for n,v in a.items():
            got=cp.asnumpy(device[n]) if isinstance(device[n],cp.ndarray) else v
            want=expected[f'{routine}/{case}/{n}']
            bad=got.ravel(order='F').view('u4')!=want.ravel(order='F').view('u4')
            count=int(np.count_nonzero(bad));words+=count;total+=got.size
            if count:
                fields[n]=count
                if len(failures)<16:
                    inds=np.flatnonzero(bad)[:3]
                    failures.append({'routine':routine,'case':case,'field':n,'indices':inds.tolist(),
                                    'got':got.ravel(order='F')[inds].tolist(),'expected':want.ravel(order='F')[inds].tolist()})
        rows.append({'routine':routine,'case':case,'different_words':words,'fields':fields})
    result={'kind':'standalone diagnostic only','whole_forecast_proven':False,'process_count':1,'MPI_used':False,'subnormal_cases':2,'compiler':'direct NVRTC; --fmad=false --ftz=false --prec-div=true --prec-sqrt=true','compared_words':int(total),
            'different_words':sum(r['different_words'] for r in rows),'cases':rows,'examples':failures,
            'native_reference_sha256':hashlib.sha256(Path(reference).read_bytes()).hexdigest()}
    Path(receipt).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    return int(result['different_words']!=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','cuda']);p.add_argument('input');p.add_argument('output');args=p.parse_args()
    if args.mode=='native':native(args.input,args.output)
    else:raise SystemExit(cuda(args.input,args.output))
