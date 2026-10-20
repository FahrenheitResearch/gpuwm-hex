#!/usr/bin/env python3
"""Prepare complete native fixtures linked to an authenticated framework build.

The same source fixture and replay generators serve standalone schemes and
schemes that require the actual MPAS framework. No numerical stubs are added.
Compile the emitted Makefiles with make -j4 (through HEX_BUILD_QUEUE on a shared host).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex

from prepare_synthetic import prepare as prepare_profiles
from prepare_framework import framework_site
from generate_replay import generate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(build,framework,out,site_name,profiles,audit):
    build=build.resolve();framework=framework.resolve();out=out.resolve()
    if out.exists():raise ValueError('framework fixtures must be fresh')
    inventory=build/'oracle-inventory.json'
    native=json.loads(inventory.read_text())
    manifest=framework/'build-manifest.json';replays=json.loads(manifest.read_text())
    flags_audit=json.loads(audit.read_text())
    if not flags_audit['all_ok'] or flags_audit['recipe']!=replays['recipe']:
        raise ValueError('full framework object audit must pass the selected recipe')
    if replays['inventory_sha256']!=digest(inventory):
        raise ValueError('framework replay and native source inventories differ')
    site=next(s for s in native['sites'] if s['site']==site_name)
    target='replay_'+site_name
    source=build/'oracle_replay'/(target+'.F90')
    if digest(source)!=replays['replay_source_sha256'][target]:
        raise ValueError('framework replay source changed')
    # Framework modules can re-export mpas_log_write without the physics
    # module importing mpas_log explicitly (SMOKE imports pool routines).
    # Initialize the actual native logger for every full-framework fixture.
    # Keep the audited model inventory and existing replayers unchanged.
    fixture_site=framework_site(site,build)
    libraries=[Path(p) for p in replays['native_libraries']]
    library_hashes={str(p):digest(p) for p in libraries}
    includes={build/'src/framework',build/'src/operators',build/'src/external/esmf_time_f90',
              build/'src/external/SMIOL',build/'src/core_atmosphere',build/'src/core_atmosphere/physics'}
    includes.update(p.parent for p in (build/'src/core_atmosphere/physics').rglob('*')
                    if p.suffix.lower() in {'.f','.f90'})
    flags=[*replays['flags'],*['-I'+str(p) for p in sorted(includes)]]
    links=['-Wl,--start-group',*map(str,libraries),'-Wl,--end-group',*replays['external_link_args']]
    compiler=replays['compiler'];recipe=replays['recipe']
    if any('\n' in s or '$' in s or '#' in s for s in [compiler,*flags,*links,str(source)]):
        raise ValueError('unsafe framework fixture build argument')
    out.mkdir(parents=True);directory=out/recipe;directory.mkdir()
    framework_source=source
    source=out/(target+'.F90')
    source.write_text(generate(fixture_site))
    (directory/'Makefile').write_text('FC='+shlex.quote(compiler)+'\nFLAGS='+shlex.join(flags)+
        '\nall: replay\nreplay: '+str(source)+' '+' '.join(map(str,libraries))+
        '\n\t$(FC) $(FLAGS) $< '+shlex.join(links)+' -o $@\n')
    (out/'Makefile').write_text(f'all:\n\t$(MAKE) -C {recipe}\n')
    info={'schema':'mpas-native-standalone-recipes-v1','scheme':site_name,
          'inventory_sha256':digest(inventory),'framework_manifest_sha256':digest(manifest),
          'object_audit_sha256':digest(audit),
          'builder_sha256':digest(Path(__file__)),'replay_source_sha256':digest(source),
          'framework_replay_source_sha256':digest(framework_source),
          'runtime_modules':fixture_site['runtime_modules'],
          'native_library_sha256':library_hashes,'framework_stubs':[],
          'recipes':{recipe:{'compiler':compiler,'flags':flags,'source_order':[],
                            'objects':{},'external_link_args':links,
                            'external_link_dependencies':list(map(str,libraries))}},'forecast_proven':False}
    (out/'build-manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    prepare_profiles(inventory,out,profiles)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['build','framework','out','audit']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--site',required=True);p.add_argument('--profile',type=Path,action='append',required=True)
    a=p.parse_args();prepare(a.build,a.framework,a.out,a.site,a.profile,a.audit)


if __name__=='__main__':main()
