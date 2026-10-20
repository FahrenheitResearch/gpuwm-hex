"""Replay every present argument of real pristine fork SW/LW calls.

MPAS-Model-rrfs/src/core_atmosphere/physics/physics_wrf/module_ra_rrtmg_sw.F:10026-10674
and physics_wrf/module_ra_rrtmg_lw.F:11574-12305 @ 1d6e370e. Rust owns record I/O.
"""
import argparse,hashlib,json,sys,subprocess
from pathlib import Path
from cloud_records import Records


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('bank','library','output','receipt','dependencies'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();checkout=Path(__file__).resolve().parents[3]
 manifest=json.loads((args.bank/'manifest.json').read_text())
 if manifest['model_commit']!='1d6e370e7b43a332074809f170532a216c2212fb' or manifest['native_binary_sha256']!='0b822cc7e2d76455a056f8914609cdc740b10ec5722fe0b7a22a3955f8087739' or manifest['process_count']!=1:raise ValueError('Native fork/binary/process authority differs')
 if digest(args.bank/'passivity.json')!=manifest['observer_passivity_sha256']:raise ValueError('Observer passivity receipt changed')
 passive=json.loads((args.bank/'passivity.json').read_text())
 if not passive['all_identical'] or passive['errors'] or not passive['files']:raise ValueError('Native observer changed emitted words')
 root=args.bank/'dump'
 if {p.name for p in root.iterdir()}!=set(manifest['files']):raise ValueError('Native inventory changed')
 for name,sha in manifest['files'].items():
  if Path(name).name!=name or digest(root/name)!=sha:raise ValueError('Native capture changed: '+name)
 if args.output.resolve().is_relative_to(args.bank.resolve()):raise ValueError('Output must be outside native bank')
 import hexcore.mpasphys
 for name,sha in [('radshared_v842','f6d6ee2b823aa102ccd6392a3250c37fd9dbd4a7860a0f5ba2355b0c83c4b902'),('rrtmg_sw_v842','2d73596d3d955880d5cb1cffabc16884b33dbe516a738ae20330f810c94f552b'),('rrtmg_lw_v842','7e9e004a7a4bc0b6d47aa0be004967d326ac5a8f57591cb50dd8cd46bfe3a5dc')]:
  from importlib.util import find_spec
  if find_spec('hexcore.mpasphys.'+name) is None:
   if digest(args.dependencies/(name+'.py'))!=sha:raise ValueError('Parent stand-in changed: '+name)
 hexcore.mpasphys.__path__.append(str(args.dependencies.resolve()))
 import cupy as cp
 import numpy as np
 from hexcore.mpasphys.rrfs import radiation_rrfs as radiation
 from hexcore.mpasphys import rrtmg_sw_v842 as sw,rrtmg_lw_v842 as lw
 if not Path(radiation.__file__).resolve().is_relative_to(checkout/'tree/src'):raise ValueError('Supplied production checkout not loaded')
 records=Records(args.library);args.output.mkdir(parents=True,exist_ok=True)
 comparisons=[];cases=[];coefficient_checks=[]
 for scope in sorted(root.glob('*.scope.txt')):
  prefix=scope.name.removesuffix('.scope.txt');mode='sw' if prefix.startswith('boundary__rrtmg_swrad.') else 'lw'
  paths={p.name.split('.pre.',1)[1].removesuffix('.bin'):p for p in root.glob(prefix+'.pre.*.bin')}
  values={n:records.load(p)[0] for n,p in paths.items()}
  arguments={n.removeprefix(mode+'.argument.'):v for n,v in values.items() if n.startswith(mode+'.argument.')}
  missing={n for n in ('p3d','p8w','pi3d','t3d','t8w','dz8w','qv3d','re_cloud','re_ice','re_snow','noznlevels','icloud','ims','its','ite','kts','kte') if n not in arguments}
  if missing:raise ValueError('Required real arguments missing: '+repr(missing))
  scalar=lambda n:int(arguments[n])
  if tuple(scalar(n) for n in ('ims','kms','jms','its','kts','jts','jte'))!=(1,1,1,1,1,1,1):raise ValueError('Stated case needs one origin-1 tile')
  nc,nz=scalar('ite'),scalar('kte')
  device={n:cp.ascontiguousarray(cp.asarray(v)) for n,v in arguments.items()}
  profile=lambda n:cp.ascontiguousarray(device[n][:nc,:nz,0])
  surface=lambda n:cp.ascontiguousarray(device[n][:nc,0])
  coefficients={n.removeprefix(mode+'.table.').replace('__','/'):cp.asarray(v) for n,v in values.items() if n.startswith(mode+'.table.')}
  if mode=='sw':
   # Source-embedded immutable constants are authenticated where the native symbol is present.
   for name,bits in sw.SCALAR_BITS.items():
    if name in coefficients:
     actual=int(cp.asnumpy(coefficients[name]).view(np.uint32))
     if actual!=bits:raise ValueError('Native SW scalar constant differs: '+name)
     coefficient_checks.append(dict(case=prefix,field=name,native_bits=actual,device_bits=bits))
   engine=radiation.shortwave_class()(raw_coefficients={n:v for n,v in coefficients.items() if v.ndim})
   options={n:arguments[n].item() for n in ('julian','xtime','gmt','julday','radt','degrad','declin','solcon','icloud','cldovrlp','idcor','o3input','has_reqc','has_reqi','has_reqs','aer_opt') if n in arguments}
   options.update(pin=cp.empty(0,cp.float32),o3clim=cp.empty((nc,0),cp.float32))
   if options.get('o3input',0)==2:
    options.update(pin=device['pin'],o3clim=cp.ascontiguousarray(device['o3clim'][:nc,:,0]))
   fields={n:profile(n) if n in device else cp.zeros((nc,nz),cp.float32) for n in sw.FIELD_NAMES}
   surfaces={n:surface(n) for n in sw.SURFACE_NAMES}
   if options.get('aer_opt',0)>0:
    options['aerosol']={name:cp.ascontiguousarray(device[native][:nc,:nz,0,:].transpose(0,2,1)) for name,native in [('tauaer','tauaer3d'),('ssaaer','ssaaer3d'),('asyaer','asyaer3d')]}
   # The public SW wrapper defaults to o3input=0; absence is preserved through that value.
   out=engine.rrtmg_swrad(fields,surfaces,cp.ascontiguousarray(device['p8w'][:nc,:nz+1,0]),cp.ascontiguousarray(device['t8w'][:nc,:nz+1,0]),**options)
   for n,v in out.items():
    if n not in device:continue
    if device[n].ndim==2:device[n][:nc,0]=v
    elif n=='rthratensw':device[n][:nc,:nz,0]=v
    elif n in sw.PROFILE_OUTPUTS:device[n][:nc,:nz+2,0]=v
  else:
   raw={}
   for band,_,source,_,_,_ in lw._REDUCTIONS:
    module=f'rrlw_kg{band:02d}';raw.setdefault(module,{})[source]=cp.ascontiguousarray(coefficients[module+'/'+source].ravel(order='F'))
   tables=lw.initialize_device(raw)
   state={n:cp.ascontiguousarray(profile(n).T) for n in sw.FIELD_NAMES}
   state.update(p8w=cp.ascontiguousarray(device['p8w'][:nc,:nz+1,0].T),t8w=cp.ascontiguousarray(device['t8w'][:nc,:nz+1,0].T))
   state.update({n:surface(n) for n in ('tsk','emiss','xland','xice','snow','xlat')})
   options={n:scalar(n) for n in ('icloud','cldovrlp','idcor','julday','has_reqc','has_reqi','has_reqs','o3input') if n in arguments}
   options.update(has_qc3d='qc3d' in arguments,has_qi3d='qi3d' in arguments,has_qs3d='qs3d' in arguments,has_cldfra3d='cldfra3d' in arguments)
   if options.get('o3input',0)==2:options.update(pin=device['pin'],ozone=cp.ascontiguousarray(device['o3clim'][:nc,:,0]))
   result=lw.rrtmg_lwrad_device(*lw.radiation_lw_from_mpas(state),tables,**options)
   for n in lw.FLUX_FIELDS:device[n][:nc,0]=result[n]
   for n in ('rthratenlw','rre_cloud','rre_ice','rre_snow'):
    if n in device:device[n][:nc,:nz,0]=result[n]
   if any(n in device for n in ('lwupflx','lwupflxc','lwdnflx','lwdnflxc')):raise ValueError('Captured LW full profiles require variable-top-layer replay; do not exclude them')
  cp.cuda.get_current_stream().synchronize()
  for name,path in paths.items():
   target=args.output/(prefix+'.post.'+name+'.bin')
   if target.exists():target.unlink()
   argument=name.removeprefix(mode+'.argument.')
   replacement=cp.asnumpy(device[argument]) if name.startswith(mode+'.argument.') else None
   records.post(path,target,replacement);entry=records.compare(root/target.name,target)
   entry.setdefault('comparison_complete',not entry.get('schema_mismatch') and isinstance(entry.get('different_words'),int))
   output_arguments=set(sw.SCALAR_OUTPUTS)|set(sw.PROFILE_OUTPUTS)|{'rthratensw'} if mode=='sw' else set(lw.FLUX_FIELDS)|{'rthratenlw','rre_cloud','rre_ice','rre_snow','lwupflx','lwupflxc','lwdnflx','lwdnflxc'}
   entry.update(case=prefix,field=name,scientific_output=name.startswith(mode+'.argument.') and argument in output_arguments);comparisons.append(entry)
  if {p.name for p in root.glob(prefix+'.post.*.bin')}!={prefix+'.post.'+n+'.bin' for n in paths}:raise ValueError('Uncompared native output record')
  cases.append(dict(case=prefix,scheme='rrtmg_'+mode,nCellsSolve=nc,nVertLevels=nz,present_arguments=sorted(arguments),absent_arguments=[line.removeprefix('absent=') for line in scope.read_text().splitlines() if line.startswith('absent=')]))
  del device,coefficients;cp.get_default_memory_pool().free_all_blocks()
 complete=bool(cases) and {c['scheme'] for c in cases}=={'rrtmg_sw','rrtmg_lw'} and all(r['comparison_complete'] for r in comparisons)
 equal=complete and all(r['bitwise_equal'] for r in comparisons)
 receipt=dict(schema='r9.pristine-real-radiation-replay/v1',checkout_tip=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip(),production_sha256=digest(Path(radiation.__file__)),replay_source_sha256=digest(Path(__file__)),native_manifest_sha256=digest(args.bank/'manifest.json'),native_process_count=1,cases=cases,coefficient_checks=coefficient_checks,comparisons=comparisons,arrays=len(comparisons),words=sum(r['words'] for r in comparisons),scientific_output_words=sum(r['words'] for r in comparisons if r['scientific_output']),preserved_words=sum(r['words'] for r in comparisons if not r['scientific_output']),different_words=sum(r['different_words'] for r in comparisons),comparison_complete=complete,bitwise_equal=equal,scope='all present scheme arguments and captured coefficient storage; full driver context and forecast not claimed',loaded_modules={m.__name__:dict(path=str(Path(m.__file__).resolve()),sha256=digest(Path(m.__file__))) for m in (radiation,sw,lw)})
 args.receipt.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ('arrays','words','different_words','comparison_complete','bitwise_equal')}),flush=True)
 return 0 if equal else 1
if __name__=='__main__':raise SystemExit(main())
