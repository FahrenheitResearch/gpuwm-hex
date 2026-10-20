#!/usr/bin/env python3
"""Link every native replay against an owned full MPAS build.

This uses the shared generated replay sources and the actual framework,
physics and ESMF libraries. Compile the emitted Makefile with make -j4 (through
HEX_BUILD_QUEUE on a shared host) after the full model build. No numerical framework stubs are used.
External library arguments must come from that model's recorded link command.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
from prepare_standalone import RECIPES
from generate_replay import generate
from fortran_source import routines


def caller_storage(site,build):
    """Keep the native caller's rank during legal Fortran sequence association.

    For example the fork's sfcdiags_ruclsm accepts explicit rank-two arrays,
    while its MPAS caller passes rank-one pool pointers (caller:136-144).
    Their contiguous storage is the same; the native capture records rank one.
    """
    site=copy.deepcopy(site)
    caller=next(r for r in routines(build/site['path'],build) if r.name==site['driver'])
    for argument in site['arguments']:
        # Earlier model inventories marked OUT components for entry transport
        # without enabling transport on their parent. Their native call hooks
        # therefore recorded only post-call shape and allocation descriptors.
        # Replay these audited captures using those descriptors, never post data.
        if argument.get('components') and argument['intent']=='out' and not argument.get('entry_storage'):
            for component in argument['components']:component['entry_storage']=False
        actual=argument['actual'].lower()
        variable=caller.variables.get(actual)
        if variable is None or not variable.dims or not argument['dims']:continue
        if len(variable.dims)==len(argument['dims']):continue
        # Only explicit-shape primitive dummy arrays admit this association.
        # Assumed-shape, pointer and derived interfaces must retain their rank.
        if argument['type'].startswith('type(') or any(d.strip()==':' for d in argument['dims']):
            raise ValueError('native caller rank differs for a non-sequence interface: '+argument['name'])
        if any(attribute in argument['attrs'].lower() for attribute in ('pointer','allocatable')):
            raise ValueError('native caller rank differs for a storage descriptor: '+argument['name'])
        argument['capture_rank']=len(variable.dims)
    return site


def framework_site(site,build):
    """Restore caller storage and initialize the full framework's native logger.

    MPAS pool/timer imports can re-export logging without an explicit physics
    USE mpas_log. Framework replay always links that real logging module.
    """
    site=caller_storage(site,build)
    site['runtime_modules']=sorted(set(site.get('runtime_modules',[]))|{'mpas_log'})
    return site


def prepare(build,out,recipe,compiler,external_args,sites=None):
    # The host runner preserves the logical path across hosts; resolving its
    # data-disk symlink here would make a Makefile that only one host can use.
    build=build.absolute();out=out.absolute()
    if out.exists():raise ValueError('framework replay output must be fresh')
    manifest_path=build/'oracle-inventory.json'
    manifest=json.loads(manifest_path.read_text())
    for relative,digest in manifest['generated_sha256'].items():
        if hashlib.sha256((build/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'generated native source changed: {relative}')
    selected=manifest['sites']
    if sites is not None:
        if not sites or len(sites)!=len(set(sites)):
            raise ValueError('selected native replay sites must be nonempty and unique')
        if not set(sites).issubset({s['site'] for s in selected}):
            raise ValueError('selected native replay site is absent from the inventory')
        selected=[s for s in selected if s['site'] in sites]
    if any(s['issues'] for s in selected):
        raise ValueError('native replay arguments unresolved')
    includes={build/'src/framework',build/'src/operators',build/'src/external/esmf_time_f90',build/'src/external/SMIOL',
              build/'src/core_atmosphere',build/'src/core_atmosphere/physics'}
    # Some NOAA wrappers put .mod files next to their source; others use the
    # submodule root. Include both, including ignored/generated MPAS wrappers.
    includes.update(p.parent for p in (build/'src/core_atmosphere/physics').rglob('*')
                    if p.suffix.lower() in {'.f','.f90'})
    includes.update(p.parent for p in build.rglob('*.mod') if p.is_file())
    libraries=[build/'src/libdycore.a',build/'src/libops.a',build/'src/libframework.a',
               build/'src/external/esmf_time_f90/libesmf_time.a']
    _,optimization=RECIPES[recipe]
    flags=['-ffree-form','-ffree-line-length-none','-fconvert=big-endian',
           '-fallow-argument-mismatch',*optimization]
    quote=shlex.quote
    if any('\n' in str(x) or '$' in str(x) or '#' in str(x)
           for x in [build,out,compiler,*external_args]):
        raise ValueError('unsafe Makefile command argument')
    # Replay orchestration can be fixed without modifying an audited native
    # model. Generate owned replay sources from its authenticated inventory
    # and bind this generator separately from the model instrumentation.
    out.mkdir(parents=True)
    (out/'sources').mkdir()
    commands=[];sources={};storage={}
    for site in selected:
        site=framework_site(site,build)
        ranks={a['name']:a['capture_rank'] for a in site['arguments'] if 'capture_rank' in a}
        if ranks:storage[site['site']]=ranks
        source=out/'sources'/('replay_'+site['site']+'.F90')
        source.write_text(generate(site))
        target='replay_'+site['site']
        sources[target]=hashlib.sha256(source.read_bytes()).hexdigest()
        commands.append(target+': '+str(source)+' '+' '.join(map(str,libraries))+'\n\t'+
            ' '.join(map(quote,[compiler,*flags,*['-I'+str(p) for p in sorted(includes)],str(source),
             '-Wl,--start-group',*map(str,libraries),'-Wl,--end-group',*external_args,'-o',target]))+'\n')
    (out/'Makefile').write_text('all: '+' '.join(sources)+'\n.PHONY: all\n'+''.join(commands))
    result={'schema':'mpas-native-framework-replays-v1','recipe':recipe,'compiler':compiler,
            'flags':flags,'external_link_args':external_args,'native_libraries':list(map(str,libraries)),
            'replay_source_sha256':sources,'inventory_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'replay_generator_sha256':hashlib.sha256((Path(__file__).parent/'generate_replay.py').read_bytes()).hexdigest(),
            'native_caller_capture_ranks':storage,
            'framework_runtime_modules':['mpas_log'],
            'framework_stubs':[],'forecast_proven':False,
            'selected_sites':sites,'scope':'selected-native-sites' if sites is not None else 'complete-native-inventory'}
    (out/'build-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--recipe',choices=['gnu-strict','gnu-o3','coverage'],required=True)
    p.add_argument('--compiler',required=True)
    p.add_argument('--sites',nargs='+',help='Optional complete native call-site subset; every selected output is replayed')
    p.add_argument('--external-link-args',type=Path,required=True,
                   help='JSON argument array from the native model link command')
    a=p.parse_args();args=json.loads(a.external_link_args.read_text())
    if not isinstance(args,list) or any(not isinstance(s,str) for s in args):
        p.error('external link arguments must be a JSON string array')
    result=prepare(a.build,a.out,a.recipe,a.compiler,args,a.sites)
    print(json.dumps({'replays':len(result['replay_source_sha256']),'recipe':a.recipe}))


if __name__=='__main__':main()
