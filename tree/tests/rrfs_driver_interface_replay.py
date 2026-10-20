#!/usr/bin/env python3
"""Real complete MPAS_to_physics replay; fork mpas_atmphys_interface.F:284-652 at 1d6e370e."""
from pathlib import Path
import hashlib,json
import numpy as np


def replay(root,prefix,target,library,manifest_path):
    import cupy as cp
    from rrfs_driver_dump_replay import authenticate,Records
    from hexcore.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache,prepare_mpas_to_physics
    manifest_sha = authenticate(root,prefix,manifest_path,source='src/core_atmosphere/physics/mpas_atmphys_interface.F')
    manifest = json.loads(manifest_path.read_text())
    passive = json.loads((manifest_path.parent/'passivity.json').read_text())
    producer = json.loads((manifest_path.parent/'producer.json').read_text())
    if (not passive['bitwise_equal'] or passive['different_words'] != 0
            or hashlib.sha256((manifest_path.parent/'passivity.json').read_bytes()).hexdigest() != manifest['passivity_receipt_sha256']
            or hashlib.sha256((manifest_path.parent/'producer.json').read_bytes()).hexdigest() != manifest['producer_attempt_sha256']
            or hashlib.sha256((manifest_path.parent/'input-request.json').read_bytes()).hexdigest() != manifest['input_request_sha256']
            or producer['native_sha256'] != manifest['native_binary_sha256']
            or producer['input_request_sha256'] != manifest['input_request_sha256']
            or producer['process_count'] != 1 or any(r['exit'] != 0 for r in producer['runs'].values())):
        raise ValueError('Real complete interface producer/passivity/input binding changed')
    pre = {p.name[len(prefix+'.pre.'):-4]:p for p in root.glob(prefix+'.pre.*.bin')}
    post = {p.name[len(prefix+'.post.'):-4]:p for p in root.glob(prefix+'.post.*.bin')}
    if not pre or set(pre) != set(post):raise ValueError('Interface PRE/POST inventory differs')
    codec = Records(library);loaded = {k:codec.load(p)[0] for k,p in pre.items()}
    nc = int(loaded['argument.ite']);its = int(loaded['argument.its'])
    if its != 1:raise ValueError('This real interface packet covers the full native tile')
    def device(key):
        return cp.asarray(np.ascontiguousarray(loaded[key][...,:nc]))
    fields = {name:device('state.'+name+'.t1') for name in ('rho_zz','theta_m','w')}
    fields.update({name:device('mesh.'+name+'.t1') for name in ('zgrid','zz')})
    fields.update({name:device('diag.'+name+'.t1') for name in ('exner','pressure_base','pressure_p','uReconstructZonal','uReconstructMeridional')})
    fields.update({name:cp.asarray(np.ascontiguousarray(loaded['vars.'+name][:,:,0].T)) for name in ('fzm_p','fzp_p')})
    scalar = device('state.scalars.t1')
    names = ('qv','qc','qr','qi','qs','qg','nc','ni','nifa','nwfa')
    species = {name:cp.ascontiguousarray(scalar[int(loaded['state.index_'+name])-1]) for name in names if 'state.index_'+name in loaded}
    options = {name:loaded['block.configs.'+name] for name in ('config_microp_scheme','config_pbl_scheme','config_tempo_aerosolaware')}
    options['config_tempo_aerosolaware'] = bool(options['config_tempo_aerosolaware'])
    flags = {name:bool(loaded['vars.f_'+name]) for name in ('nc','ni','nifa','nwfa')}
    result = prepare_mpas_to_physics(fields=fields,species=species,options=options,scalar_flags=flags,kernel_cache=StrictPhysicsKernelCache())
    replacement = {}
    for key,value in result.items():
        name = 'diag_physics.plrad.t1' if key == 'plrad' else 'vars.'+key
        old = cp.asarray(np.ascontiguousarray(loaded[name]))
        if old.ndim == 3:old[:nc,:value.shape[0],0] = value.T
        elif old.ndim == 2:old[:nc,0] = value
        else:old[:nc] = value
        replacement[name] = cp.asnumpy(old)
    # Native writes diag.surface_pressure and psfc_p to identical real values.
    name = 'diag.surface_pressure.t1';old=cp.asarray(np.ascontiguousarray(loaded[name]));old[:nc]=result['psfc_p'];replacement[name]=cp.asnumpy(old)
    target.mkdir(parents=True,exist_ok=True);comparisons={}
    for key,path in pre.items():
        actual=target/post[key].name;codec.post(path,actual,replacement.get(key));comparisons[key]=codec.compare(post[key],actual)
    return {'schema':'r4.real-complete-interface-replay/v1','scope':'all allocated MPAS_to_physics scratch and all direct pool-getter records; uncaptured private/pool storage remains unknown',
        'bitwise_equal':all(c['bitwise_equal'] for c in comparisons.values()),'words':sum(c['words'] for c in comparisons.values()),
        'different_words':sum(c['different_words'] for c in comparisons.values()),'output_records':len(comparisons),
        'replaced_records':sorted(replacement),'manifest_sha256':manifest_sha,'comparisons':comparisons}
