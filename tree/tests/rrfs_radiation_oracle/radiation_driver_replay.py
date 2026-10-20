"""Complete stated real SW/LW public slot replay, with every driver record checked.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_driver_radiation_sw.F:374-1108
and mpas_atmphys_driver_radiation_lw.F:300-1029 @ 1d6e370e. Rust owns record I/O.
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from cloud_records import Records


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate(bank):
    manifest=json.loads((bank/'manifest.json').read_text())
    if manifest['model_commit']!='1d6e370e7b43a332074809f170532a216c2212fb' or manifest['native_binary_sha256']!='0b822cc7e2d76455a056f8914609cdc740b10ec5722fe0b7a22a3955f8087739' or manifest['process_count']!=1:
        raise ValueError('Pristine native fork/binary/rank authority changed')
    if digest(bank/'passivity.json')!=manifest['observer_passivity_sha256']: raise ValueError('Passivity receipt changed')
    passive=json.loads((bank/'passivity.json').read_text())
    if not passive['all_identical'] or passive['errors'] or not passive['files']: raise ValueError('Observer passivity not proven')
    root=bank/'dump'
    if {p.name for p in root.iterdir()}!=set(manifest['files']): raise ValueError('Capture inventory changed')
    for name,sha in manifest['files'].items():
        if Path(name).name!=name or digest(root/name)!=sha: raise ValueError('Capture member changed: '+name)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bank','coefficients','library','dependencies','output','receipt'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); checkout=Path(__file__).resolve().parents[3]
    root=authenticate(args.bank); coefficient_root=authenticate(args.coefficients)
    if args.output.resolve().is_relative_to(args.bank.resolve()) or args.output.resolve().is_relative_to(args.coefficients.resolve()): raise ValueError('Output overlaps immutable authority')
    from importlib.util import find_spec
    import hexcore.mpasphys
    for name,sha in [('radshared_v842','f6d6ee2b823aa102ccd6392a3250c37fd9dbd4a7860a0f5ba2355b0c83c4b902'),('rrtmg_sw_v842','2d73596d3d955880d5cb1cffabc16884b33dbe516a738ae20330f810c94f552b'),('rrtmg_lw_v842','7e9e004a7a4bc0b6d47aa0be004967d326ac5a8f57591cb50dd8cd46bfe3a5dc')]:
        if find_spec('hexcore.mpasphys.'+name) is None and digest(args.dependencies/(name+'.py'))!=sha: raise ValueError('Parent stand-in changed: '+name)
    hexcore.mpasphys.__path__.append(str(args.dependencies.resolve()))
    import cupy as cp
    import numpy as np
    from hexcore.mpasphys.rrfs import radiation_rrfs as radiation
    from hexcore.mpasphys import rrtmg_sw_v842 as sw, rrtmg_lw_v842 as lw
    from hexcore.mpasphys.driver_v842 import SlotCall, Backend, StepFlags
    if not Path(radiation.__file__).resolve().is_relative_to(checkout/'tree/src'): raise ValueError('Supplied production checkout was not loaded')
    records=Records(args.library); args.output.mkdir(parents=True,exist_ok=True)
    comparisons=[]; cases=[]; output_names=set()
    # Constants are observed at radconst's real invocation, not reconstructed in Python.
    sw_prefix=next(root.glob('boundary__driver_radiation_sw.*.scope.txt')).name.removesuffix('.scope.txt')
    constants={n:records.load(root/(sw_prefix+'.pre.constant.'+n+'.bin'))[0].item() for n in ('degrad','dpd')}
    for scope in sorted(root.glob('*.scope.txt')):
        prefix=scope.name.removesuffix('.scope.txt'); mode='sw' if prefix.startswith('boundary__driver_radiation_sw.') else 'lw'
        paths={p.name.split('.pre.',1)[1].removesuffix('.bin'):p for p in root.glob(prefix+'.pre.*.bin')}
        values={n:records.load(p)[0] for n,p in paths.items()}
        nc,nz=int(values['argument.ite']),int(values['global.kte'])
        if tuple(int(values[n]) for n in ('argument.its','global.ims','global.kms','global.kts','global.jms','global.jts','global.jte'))!=(1,1,1,1,1,1,1): raise ValueError('Stated real origin-1 tile differs')
        device={n:cp.ascontiguousarray(cp.asarray(v)) for n,v in values.items() if not isinstance(v,str)}
        public=lambda name: {n.removeprefix(name+'.').removesuffix('.t1'):v for n,v in device.items() if n.startswith(name+'.')}
        diag,sfc,mesh,tend=(public(n) for n in ('diag_physics','sfc_input','mesh','tend_physics'))
        options={n.removeprefix('configs.'):v.strip() if isinstance(v,str) else v.item() for n,v in values.items() if n.startswith('configs.')}
        prepared=lambda name,levels=nz: cp.ascontiguousarray(device['global.'+name+'_p'][:,:levels,0].T)
        flags={n:int(values['global.'+n]) for n in ('has_reqc','has_reqi','has_reqs')}
        coefficients={p.name.split('.pre.'+mode+'.table.',1)[1].removesuffix('.bin').replace('__','/'):cp.asarray(records.load(p)[0]) for p in coefficient_root.glob('boundary__rrtmg_*rad.*.pre.'+mode+'.table.*.bin')}
        changed=set()
        def carrier(name,source,profile=False,levels=nz):
            key='global.'+name; target=device[key]
            if profile: target[:nc,:levels,0]=source[:levels,:nc].T
            else: target[:nc,0]=source[:nc]
            changed.add(key)
        # Device copies implement native from/to carrier orchestration. No expected POST is read.
        for name,source in [('tsk_p','skintemp'),('snow_p','snow'),('xice_p','xice'),('xland_p','xland')]: carrier(name,sfc[source])
        carrier('cldfrac_p',diag['cldfrac'],True)
        for name in ('f_ice','f_rain'):
            device['global.'+name][:nc,:nz,0]=np.float32(0); changed.add('global.'+name)
        for native,public_name in [('recloud_p','re_cloud'),('reice_p','re_ice'),('resnow_p','re_snow')]:
            enabled=options['config_microp_re'] and options['config_microp_scheme'] in radiation.RADIUS_SCHEMES
            source=diag[public_name] if enabled else cp.zeros((nz,nc),cp.float32)
            carrier(native,source,True)
        if options['config_o3climatology']:
            device['global.pin_p'][:]=device['atm_input.pin.t1']; changed.add('global.pin_p')
            carrier('o3clim_p',diag['o3clim'],True,int(values['global.num_oznlevels']))
        else:
            device['global.pin_p'][:]=0; changed.add('global.pin_p')
            carrier('o3clim_p',cp.zeros((int(values['global.num_oznlevels']),nc),cp.float32),True,int(values['global.num_oznlevels']))
        if mode=='sw':
            class ObservedSW(radiation.shortwave_class()):
                def thompson_aerosols(self,*a,**kw):
                    result=super().thompson_aerosols(*a,**kw); self.replayed_aerosols=result; return result
            engine=ObservedSW(raw_coefficients={n:v for n,v in coefficients.items() if v.ndim})
            controls=engine.driver_controls(options,xtime_s=values['argument.xtime_s'].item(),dt_radtsw=values['global.dt_radtsw'].item(),curr_julday=values['calendar.curr_julday'].item(),**constants)
            controls.update(gmt=values['calendar.gmt'].item(),julday=int(values['calendar.julday']),icloud=1,**flags)
            aliases=dict(p3d='pres_hyd',pi3d='pi',t3d='t',dz8w='dz',qv3d='qvrad',qc3d='qcrad',qi3d='qirad',qs3d='qsrad')
            sounding={name:prepared(native) for name,native in aliases.items()}
            sounding.update(cldfra3d=diag['cldfrac'],p8w=prepared('pres2_hyd',nz+1),t8w=prepared('t2',nz+1),controls=controls)
            sounding.update({n:diag[n] if enabled else cp.zeros_like(sounding['p3d']) for n in ('re_cloud','re_ice','re_snow')})
            aware=options['config_microp_scheme']=='mp_thompson_aerosols' or options['config_microp_scheme']=='mp_tempo' and options['config_tempo_aerosolaware']
            if aware: sounding.update(nwfa=prepared('nwfa'),nifa=prepared('nifa'),aerosol_qv=prepared('qv'),ht=mesh['zgrid'][0])
            pools=dict(nCellsSolve=nc,rrtmg_sw_engine=engine,radiation_sw_input=sounding,diag_physics=diag,sfc_input=sfc,mesh=mesh,tend_physics=tend,atm_input={'pin':device['atm_input.pin.t1']})
            call=SlotCall('radiation_sw','rrtmg_sw',Backend.HEX_CUDA,int(values['argument.itimestep']),float(values['argument.xtime_s']),0.,StepFlags(),pools,options)
            out=radiation.shortwave_slot(call)
            for n in sw.SCALAR_OUTPUTS: carrier(n+'_p',out[n])
            carrier('rthratensw_p',out['rthratensw'],True)
            carrier('sfc_albedo_p',diag['sfc_albedo'])
            carrier('xlat_p',radiation.angles_to_degrees(mesh['latCell'],constants['degrad']))
            carrier('xlon_p',radiation.angles_to_degrees(mesh['lonCell'],constants['degrad']))
            for n in ('declin','solcon'): device['global.'+n][...]=controls[n]; changed.add('global.'+n)
            device['global.aer_opt'][...]=3 if aware else 0; changed.add('global.aer_opt')
            if aware:
                aer=engine.replayed_aerosols
                carrier('ht_p',mesh['zgrid'][0]); carrier('taer_type_p',aer['taer_type'])
                carrier('taod5502d_p',aer['taod5502d']); carrier('taod5503d_p',aer['taod5503d'].T,True)
                for n in ('tauaer','ssaaer','asyaer'):
                    device['global.'+n+'_p'][:nc,:nz,0,:]=aer[n][:,:,:nz].transpose(0,2,1); changed.add('global.'+n+'_p')
            else:
                for n,value in [('tauaer',0),('ssaaer',1),('asyaer',0)]:
                    device['global.'+n+'_p'][:nc,:nz,0,:]=value; changed.add('global.'+n+'_p')
            profile_names=('swupflx_p','swupflxc_p','swdnflx_p','swdnflxc_p')
        else:
            raw={}
            for band,_,source,_,_,_ in lw._REDUCTIONS:
                module=f'rrlw_kg{band:02d}';raw.setdefault(module,{})[source]=cp.ascontiguousarray(coefficients[module+'/'+source].ravel(order='F'))
            tables=lw.initialize_device(raw)
            prep={n:prepared(n,nz+1 if n in ('pres2_hyd','t2') else nz) for n in ('pres_hyd','pres2_hyd','t','t2','dz','pi')}
            prep['xlat']=radiation.angles_to_degrees(mesh['latCell'],constants['degrad'])
            pools=dict(nCellsSolve=nc,prepared=prep,diag_physics=diag,sfc_input=sfc,tend_physics=tend,atm_input={'pin':device['atm_input.pin.t1']},radshared={n:cp.ascontiguousarray(prepared(n).T) for n in ('qvrad','qcrad','qirad','qsrad')},scalar_state=flags,icloud=1,julday=int(values['calendar.julday']),rrtmg_lw_tables=tables)
            call=SlotCall('radiation_lw','rrtmg_lw',Backend.HEX_CUDA,1,float(values['argument.xtime_s']),0.,StepFlags(),pools,options)
            result=radiation.longwave_slot(call); out=lw.radiation_lw_to_mpas(result)
            carrier('sfc_emiss_p',diag['sfc_emiss'])
            for n in lw.FLUX_FIELDS: carrier(('olrtoa' if n=='olr' else n)+'_p',out[n])
            carrier('rthratenlw_p',out['rthratenlw'],True)
            for native,n in [('rrecloud_p','rre_cloud'),('rreice_p','rre_ice'),('rresnow_p','rre_snow')]: carrier(native,out[n] if enabled else cp.zeros((nz,nc),cp.float32),True)
            profile_names=('lwupflx_p','lwupflxc_p','lwdnflx_p','lwdnflxc_p')
        for name in profile_names:
            device['global.'+name][:nc,:nz+2,0]=np.float32(0); changed.add('global.'+name)
        cp.cuda.get_current_stream().synchronize()
        output_fields=changed|{n for n in paths if n.startswith(('diag_physics.','tend_physics.')) and n.removesuffix('.t1').split('.',1)[1] in set(sw.SCALAR_OUTPUTS)|set(lw.FLUX_FIELDS)|{'olrtoa','rthratensw','rthratenlw','taod5502d','taod5503d','o3vmr','rre_cloud','rre_ice','rre_snow'}}
        for name,path in paths.items():
            target=args.output/(prefix+'.post.'+name+'.bin')
            if target.exists(): target.unlink()
            replacement=cp.asnumpy(device[name]) if name in changed or name.startswith(('diag_physics.','tend_physics.')) else None
            records.post(path,target,replacement); row=records.compare(root/target.name,target)
            row.setdefault('comparison_complete',not row.get('schema_mismatch') and isinstance(row.get('different_words'),int));row.update(case=prefix,field=name,scientific_output=name in output_fields);comparisons.append(row)
        if {p.name for p in root.glob(prefix+'.post.*.bin')}!={prefix+'.post.'+name+'.bin' for name in paths}: raise ValueError('Native POST record was not compared')
        cases.append(dict(case=prefix,scheme='rrtmg_'+mode,nCellsSolve=nc,nVertLevels=nz,complete_record_inventory=True,aer_opt=3 if mode=='sw' and aware else None))
        del device,coefficients,pools;cp.get_default_memory_pool().free_all_blocks()
    complete={c['scheme'] for c in cases}=={'rrtmg_sw','rrtmg_lw'} and all(r['comparison_complete'] for r in comparisons);equal=complete and all(r['bitwise_equal'] for r in comparisons)
    result=dict(schema='r9.pristine-real-radiation-slots/v1',checkout_tip=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),production_sha256=digest(Path(radiation.__file__)),replay_source_sha256=digest(Path(__file__)),native_manifest_sha256=digest(args.bank/'manifest.json'),coefficient_manifest_sha256=digest(args.coefficients/'manifest.json'),native_process_count=1,cases=cases,comparisons=comparisons,arrays=len(comparisons),words=sum(r['words'] for r in comparisons),scientific_output_words=sum(r['words'] for r in comparisons if r['scientific_output']),preserved_words=sum(r['words'] for r in comparisons if not r['scientific_output']),different_words=sum(r['different_words'] for r in comparisons),comparison_complete=complete,bitwise_equal=equal,scope='all records of stated real full SW/LW public driver calls; unqueried framework state and forecast not claimed')
    result['loaded_modules']={m.__name__:dict(path=str(Path(m.__file__).resolve()),sha256=digest(Path(m.__file__))) for m in (radiation,sw,lw,sys.modules['hexcore.mpasphys.radshared_v842'])}
    args.receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('arrays','words','different_words','comparison_complete','bitwise_equal')}),flush=True)
    return 0 if equal else 1


if __name__=='__main__':raise SystemExit(main())
