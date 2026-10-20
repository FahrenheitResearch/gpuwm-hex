#!/usr/bin/env python3
"""Prepare explicit native source fixtures against an owned standalone build.

Compile the emitted Makefiles with make -j4 (HEX_BUILD_QUEUE on a shared host). Profiles initialize
columns in Fortran; field decoding and exact comparisons remain native/Rust.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shlex
from synthetic_capture import generate


def prepare(inventory,standalone,profiles):
    manifest=json.loads(inventory.read_text())
    build=json.loads((standalone/'build-manifest.json').read_text())
    site=next(s for s in manifest['sites'] if s['site']==build['scheme'])
    site=copy.deepcopy(site)
    site['runtime_modules']=sorted(set(site.get('runtime_modules',[]))|
                                   set(build.get('runtime_modules',[])))
    if hashlib.sha256(inventory.read_bytes()).hexdigest()!=build['inventory_sha256']:
        raise ValueError('standalone build and source fixture inventories differ')
    fixtures={}
    for profile in profiles:
        name=profile.stem
        if not re.fullmatch(r'[a-z][a-z0-9_]*',name):raise ValueError('unsafe fixture target')
        if name in fixtures:raise ValueError('duplicate source fixture')
        data=json.loads(profile.read_text());source=generate(site,data)
        fixtures[name]={'profile':data,'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
                        'generated_source_sha256':hashlib.sha256(source.encode()).hexdigest()}
        for recipe,config in build['recipes'].items():
            directory=standalone/recipe
            target=directory/('capture_'+name+'.F90')
            if target.exists():raise ValueError('source fixture must be fresh')
            target.write_text(source)
            objects=' '.join(config['objects'][p] for p in config['source_order'])
            links=' '.join(shlex.quote(s) for s in config.get('external_link_args',[]))
            prerequisites=' '.join(config.get('external_link_dependencies',[]))
            if any('\n' in s or '$' in s or '#' in s for s in config.get('external_link_args',[])):
                raise ValueError('unsafe fixture link argument')
            makefile=directory/'synthetic.mk'
            if name==next(iter(fixtures)):
                if makefile.exists():raise ValueError('synthetic Makefile must be fresh')
                makefile.write_text('include Makefile\n')
            with makefile.open('a') as f:
                f.write(f'capture_{name}: capture_{name}.F90 Makefile synthetic.mk {objects} {prerequisites}\n'
                        f'\t$(FC) $(FLAGS) $< {objects} {links} -o $@\n')
    names=' '.join('capture_'+n for n in fixtures)
    recipes=list(build['recipes'])
    (standalone/'synthetic.mk').write_text('all: '+' '.join(recipes)+'\n.PHONY: all '+' '.join(recipes)+'\n'+
        ''.join(f'{r}:\n\t$(MAKE) -C {r} -f synthetic.mk {names}\n' for r in recipes))
    receipt={'schema':'mpas-native-synthetic-source-v1','site':site['site'],'profiles':fixtures,
             'fixture_generator_sha256':hashlib.sha256((Path(__file__).parent/'synthetic_capture.py').read_bytes()).hexdigest(),
             'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'inventory_sha256':build['inventory_sha256'],'build_manifest_sha256':hashlib.sha256((standalone/'build-manifest.json').read_bytes()).hexdigest(),
             'forecast_proven':False}
    (standalone/'synthetic-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--standalone',type=Path,required=True)
    p.add_argument('--profile',type=Path,action='append',required=True)
    a=p.parse_args();prepare(a.inventory,a.standalone,a.profile)


if __name__=='__main__':main()
