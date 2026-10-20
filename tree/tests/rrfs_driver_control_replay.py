#!/usr/bin/env python3
"""Real full control check; fork mpas_atmphys_control.F:87-386 at 1d6e370e.

All twelve config pointers and moist_physics are observed and compared. Native
logging/error buffers are not represented as numerical module output.
"""
import hashlib,json
import numpy as np


def replay(root,prefix,target,library,manifest_path):
    from rrfs_driver_dump_replay import authenticate,Records
    from hexcore.mpasphys.rrfs.driver_rrfs import resolve_namelist,specification
    manifest_sha = authenticate(root,prefix,manifest_path,source='src/core_atmosphere/physics/mpas_atmphys_control.F')
    manifest = json.loads(manifest_path.read_text())
    passive = json.loads((manifest_path.parent/'passivity.json').read_text())
    producer = json.loads((manifest_path.parent/'producer.json').read_text())
    if (not manifest['observer_passivity_proven'] or not passive['bitwise_equal'] or passive['different_words'] != 0
            or hashlib.sha256((manifest_path.parent/'passivity.json').read_bytes()).hexdigest() != manifest['passivity_receipt_sha256']
            or hashlib.sha256((manifest_path.parent/'producer.json').read_bytes()).hexdigest() != manifest['producer_attempt_sha256']
            or hashlib.sha256((manifest_path.parent/'input-request.json').read_bytes()).hexdigest() != manifest['input_request_sha256']
            or producer['native_sha256'] != manifest['native_binary_sha256']
            or producer['input_request_sha256'] != manifest['input_request_sha256']
            or producer['observer_sha256'] != manifest['observer_sha256']
            or producer['source_sha256'] != manifest['observer_source_sha256']
            or producer['process_count'] != 1 or any(r['exit'] != 0 for r in producer['runs'].values())):
        raise ValueError('Real control producer/passivity/input binding changed')
    pre = {p.name[len(prefix+'.pre.'):-4]:p for p in root.glob(prefix+'.pre.*.bin')}
    post = {p.name[len(prefix+'.post.'):-4]:p for p in root.glob(prefix+'.post.*.bin')}
    expected = {'control.moist_physics', *('block.configs.config_'+n for n in ('physics_suite','microp_scheme',
        'convection_scheme','lsm_scheme','pbl_scheme','gwdo_scheme','radt_cld_scheme','radt_lw_scheme',
        'radt_sw_scheme','sfclayer_scheme','tempo_cldfra','mynn_cloudpdf'))}
    if set(pre) != expected or set(post) != expected:raise ValueError('Full control PRE/POST inventory differs')
    codec = Records(library);loaded = {k:codec.load(p) for k,p in pre.items()}
    grouped = {}
    spec = specification()
    for key,(value,header) in loaded.items():
        if key == 'control.moist_physics':continue  # overwritten by native, never an input
        name = key.removeprefix('block.configs.')
        kind = spec['registry_defaults'][name]['type']
        if header[2] != 0 or (header[0],header[1]) != {'character':(5,512),'logical':(4,4),'integer':(3,4)}[kind]:
            raise ValueError('Native control input type differs: '+name)
        if kind == 'character': literal = "'"+value.replace("'","''")+"'"
        elif kind == 'logical':
            if int(value) not in (0,1):raise ValueError('Native logical storage is not canonical')
            literal = '.true.' if int(value) else '.false.'
        else: literal = str(int(value))
        group = next(g for g,keys in spec['registry_groups'].items() if name in keys)
        grouped.setdefault(group,[]).append(name+' = '+literal)
    text = '\n'.join('&'+g+'\n '+',\n '.join(values)+'\n/' for g,values in grouped.items())
    resolved = resolve_namelist(text)
    target.mkdir(parents=True,exist_ok=True);comparisons={}
    for key,path in pre.items():
        if key == 'control.moist_physics': value = np.array(int(resolved.moist_physics),dtype='i4')
        else:
            name = key.removeprefix('block.configs.');result = resolved.options[name]
            if isinstance(result,str):
                encoded = result.encode('ascii')
                if len(encoded) > 512:raise ValueError('Native control string extent exceeded')
                value = np.frombuffer(encoded.ljust(512,b' '),dtype='u1')
            else:value = np.array(int(result),dtype='i4')
        actual = target/post[key].name;codec.post(path,actual,value);comparisons[key]=codec.compare(post[key],actual)
    return {'schema':'r4.real-complete-control-replay/v1','scope':'all twelve direct config outputs plus moist_physics; one stated successful native configuration; logging/error buffer and other control module routines not claimed',
        'bitwise_equal':all(c['bitwise_equal'] for c in comparisons.values()),'words':sum(c['words'] for c in comparisons.values()),
        'different_words':sum(c['different_words'] for c in comparisons.values()),'output_records':len(comparisons),
        'manifest_sha256':manifest_sha,'resolved_selectors':dict(resolved.selectors),'comparisons':comparisons}
