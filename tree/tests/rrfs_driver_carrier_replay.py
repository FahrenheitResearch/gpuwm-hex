#!/usr/bin/env python3
"""Real number-carrier PRE to CUDA to all POST words, using Rust transport.

MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:357-505
at 1d6e370e. This packet contains the complete number-carrier slice, not the
remaining MPAS_to_physics temperature/geometry preparation.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np


def replay(root, prefix, target, library, manifest_path):
    import cupy as cp
    from rrfs_driver_dump_replay import Records, authenticate
    from hexcore.mpasphys.rrfs.driver_rrfs import StrictPhysicsKernelCache, prepare_number_carriers
    manifest_sha = authenticate(root,prefix,manifest_path,
        source='src/core_atmosphere/physics/mpas_atmphys_interface.F')
    manifest = json.loads(manifest_path.read_text())
    producer_path = manifest_path.parent/'producer.json'
    producer = json.loads(producer_path.read_text())
    if (hashlib.sha256(producer_path.read_bytes()).hexdigest() != manifest['producer_attempt_sha256']
            or producer['native_sha256'] != manifest['native_binary_sha256']
            or producer['input_request_sha256'] != manifest['input_request_sha256']
            or hashlib.sha256((manifest_path.parent/'input-request.json').read_bytes()).hexdigest() != manifest['input_request_sha256']
            or producer['process_count'] != manifest['process_count']
            or any(run['exit'] != 0 for run in producer['runs'].values())):
        raise ValueError('Native carrier producer did not complete the bound real case')
    passive = json.loads((manifest_path.parent/'passivity.json').read_text())
    if not (passive['bitwise_equal'] and passive['different_words'] == 0
            and hashlib.sha256((manifest_path.parent/'passivity.json').read_bytes()).hexdigest()
                == manifest['passivity_receipt_sha256']):
        raise ValueError('Native carrier observer passivity changed')
    pre = {p.name[len(prefix+'.pre.'):-4]:p for p in root.glob(prefix+'.pre.*.bin')}
    post = {p.name[len(prefix+'.post.'):-4]:p for p in root.glob(prefix+'.post.*.bin')}
    if not pre or set(pre) != set(post):
        raise ValueError('Real carrier PRE/POST inventory differs')
    codec = Records(library)
    loaded = {key:codec.load(path)[0] for key,path in pre.items()}
    its, ite = (int(loaded['argument.'+key]) for key in ('its','ite'))
    shapes = {loaded['vars.'+key+'_p'].shape for key in ('nc','ni','nifa','nwfa')}
    if len(shapes) != 1:
        raise ValueError('Real number carriers disagree on native allocation')
    nc, allocated_levels, nj = shapes.pop()
    if its != 1 or ite != nc or nj != 1:
        raise ValueError('This stated real carrier case must cover its entire native rectangular allocation')
    scalar = cp.asarray(np.ascontiguousarray(loaded['state.scalars.t1']))
    nl = int(scalar.shape[1])
    if allocated_levels != nl+1:
        raise ValueError('Native number allocation must include its persistent KME interface tail')
    fields = {'qv':cp.zeros((nl,nc),dtype=cp.float32)}
    flags = {key:bool(loaded['vars.f_'+key]) for key in ('nc','ni','nifa','nwfa')}
    for key in flags:
        index = int(loaded['state.index_'+key])-1
        # Native max(+0,q) preparation is performed on the device; no POST is read.
        fields[key] = cp.ascontiguousarray(cp.maximum(cp.float32(0),scalar[index,:,:nc]))
    options = {key:loaded['block.configs.'+key] for key in
        ('config_microp_scheme','config_pbl_scheme','config_tempo_aerosolaware')}
    options['config_tempo_aerosolaware'] = bool(options['config_tempo_aerosolaware'])
    result = prepare_number_carriers(SimpleNamespace(species_p=fields), options=options,
        scalar_flags=flags,n_cells_solve=nc,kernel_cache=StrictPhysicsKernelCache())
    if set(result) != set(flags):
        raise ValueError('This real MYNN packet requires all four native scratch carriers')
    target.mkdir(parents=True,exist_ok=True)
    replacements = {}
    for key,value in result.items():
        name = 'vars.'+key+'_p'
        # Native loops write KTS:KTE only; the KME interface tail survives from PRE.
        output = cp.asarray(np.ascontiguousarray(loaded[name]))
        output[:,:nl,0] = value.T
        replacements[name] = cp.asnumpy(output)
    comparisons = {}
    for key,path in pre.items():
        actual = target/post[key].name
        codec.post(path,actual,replacements.get(key))
        comparisons[key] = codec.compare(post[key],actual)
    equal = all(row['bitwise_equal'] for row in comparisons.values())
    return {'schema':'r4.real-driver-carrier-replay/v1','bitwise_equal':equal,
        'scope':'complete four number-carrier outputs plus unchanged captured context; remaining interface fields are outside packet',
        'output_records':len(comparisons),'words':sum(c['words'] for c in comparisons.values()),
        'evolved_output_words':len(replacements)*nc*nl,
        'preserved_interface_tail_words':len(replacements)*nc,
        'different_words':sum(c['different_words'] for c in comparisons.values()),
        'manifest_sha256':manifest_sha,'options':{k:(bool(v) if not isinstance(v,str) else v) for k,v in options.items()},
        'comparisons':comparisons,'native_passivity_words':passive['words']}
